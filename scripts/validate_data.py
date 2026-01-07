#!/usr/bin/env python3
"""
Data Validation Script - Understand who matters and why.

Shows:
1. Important people (high reply rate, strong relationships)
2. Important services vs marketing noise
3. What you ignore vs what you engage with
4. Service breakdown by domain (e.g., chase.com transactions vs marketing)

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --top 20
"""
from __future__ import annotations

import argparse
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("DATABASE_URL")


def get_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(DB_URL)


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def important_people(conn: psycopg2.extensions.connection, limit: int = 15) -> None:
    """Show who you actually respond to."""
    print_section("IMPORTANT PEOPLE (by reply behavior)")

    cur = conn.cursor()
    cur.execute("""
        SELECT
            u.email,
            u.name,
            u.emails_from,
            u.reply_count,
            ROUND(u.reply_rate::numeric * 100, 1) as reply_rate_pct,
            u.is_important_sender,
            ROUND(AVG(ef.relationship_strength)::numeric, 2) as avg_rel_str
        FROM users u
        LEFT JOIN emails e ON LOWER(e.from_email) = LOWER(u.email)
        LEFT JOIN email_features ef ON ef.email_id = e.id
        WHERE u.emails_from > 0
          AND u.reply_count > 0
          AND NOT u.is_you
        GROUP BY u.email, u.name, u.emails_from, u.reply_count, u.reply_rate, u.is_important_sender
        ORDER BY u.reply_count DESC, u.reply_rate DESC
        LIMIT %s
    """, (limit,))

    results = cur.fetchall()

    print(f"\n{'Email':<40} {'Name':<20} {'Emails':<8} {'Replied':<8} {'Rate':<8} {'Rel Str':<8} {'Important'}")
    print("-"*110)
    for row in results:
        email, name, emails, replied, rate, important, rel_str = row
        name_short = (name or "")[:18]
        email_short = email[:38]
        imp = "✓" if important else ""
        print(f"{email_short:<40} {name_short:<20} {emails:<8} {replied:<8} {rate or 0:>5.1f}%   {rel_str or 0:>5.2f}    {imp}")

    # Summary
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN is_important_sender THEN 1 ELSE 0 END) as important
        FROM users WHERE emails_from > 0 AND NOT is_you
    """)
    result = cur.fetchone()
    total = result[0] if result else 0
    important = result[1] if result else 0
    print(f"\nTotal senders: {total}, Marked important: {important}")


def service_breakdown(conn: psycopg2.extensions.connection, limit: int = 20) -> None:
    """Show service emails broken down by what you engage with."""
    print_section("SERVICE EMAIL BREAKDOWN (by engagement)")

    cur = conn.cursor()

    # Group by domain and service type
    cur.execute("""
        WITH service_stats AS (
            SELECT
                CASE
                    WHEN e.from_email LIKE '%%@%%.%%'
                    THEN REGEXP_REPLACE(e.from_email, '^.*@', '')
                    ELSE e.from_email
                END as domain,
                ef.service_type,
                COUNT(*) as total,
                SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) as replied,
                SUM(CASE WHEN e.action = 'ARCHIVED' THEN 1 ELSE 0 END) as archived,
                SUM(CASE WHEN e.action = 'IGNORED' THEN 1 ELSE 0 END) as ignored,
                ROUND(AVG(ef.service_importance)::numeric, 2) as avg_importance
            FROM emails e
            JOIN email_features ef ON ef.email_id = e.id
            WHERE ef.is_service_email = true AND e.is_sent = false
            GROUP BY domain, ef.service_type
        )
        SELECT * FROM service_stats
        WHERE total >= 3
        ORDER BY replied DESC, total DESC
        LIMIT %s
    """, (limit,))

    results = cur.fetchall()

    print(f"\n{'Domain':<35} {'Type':<15} {'Total':<7} {'Replied':<8} {'Archived':<9} {'Ignored':<8} {'Importance'}")
    print("-"*100)
    for row in results:
        domain, svc_type, total, replied, archived, ignored, importance = row
        domain_short = domain[:33]
        svc_short = (svc_type or "unknown")[:13]
        print(f"{domain_short:<35} {svc_short:<15} {total:<7} {replied:<8} {archived:<9} {ignored:<8} {importance or 0:>5.2f}")


def what_you_ignore(conn: psycopg2.extensions.connection, limit: int = 15) -> None:
    """Show high-volume senders you never respond to."""
    print_section("NOISE: High-volume senders you NEVER respond to")

    cur = conn.cursor()
    cur.execute("""
        SELECT
            e.from_email,
            COUNT(*) as total_emails,
            ef.service_type,
            SUM(CASE WHEN e.action = 'ARCHIVED' THEN 1 ELSE 0 END) as archived,
            SUM(CASE WHEN e.action = 'IGNORED' THEN 1 ELSE 0 END) as ignored
        FROM emails e
        JOIN email_features ef ON ef.email_id = e.id
        WHERE e.is_sent = false
        GROUP BY e.from_email, ef.service_type
        HAVING SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) = 0
           AND COUNT(*) >= 5
        ORDER BY COUNT(*) DESC
        LIMIT %s
    """, (limit,))

    results = cur.fetchall()

    print(f"\n{'Sender':<50} {'Type':<15} {'Total':<7} {'Archived':<9} {'Ignored'}")
    print("-"*90)
    for row in results:
        sender, total, svc_type, archived, ignored = row
        sender_short = sender[:48]
        svc_short = (svc_type or "unknown")[:13]
        print(f"{sender_short:<50} {svc_short:<15} {total:<7} {archived:<9} {ignored}")


def domain_analysis(conn: psycopg2.extensions.connection, domain_filter: str | None = None) -> None:
    """Analyze a specific domain (e.g., chase.com) to see transaction vs marketing."""
    print_section(f"DOMAIN ANALYSIS: {domain_filter or 'all domains'}")

    cur = conn.cursor()

    if domain_filter:
        cur.execute("""
            SELECT
                e.from_email,
                ef.service_type,
                COUNT(*) as total,
                SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) as replied,
                SUM(CASE WHEN e.action = 'ARCHIVED' THEN 1 ELSE 0 END) as archived,
                SUM(CASE WHEN e.action = 'IGNORED' THEN 1 ELSE 0 END) as ignored,
                ROUND(AVG(ef.service_importance)::numeric, 2) as avg_importance
            FROM emails e
            JOIN email_features ef ON ef.email_id = e.id
            WHERE e.from_email LIKE %s AND e.is_sent = false
            GROUP BY e.from_email, ef.service_type
            ORDER BY total DESC
        """, (f"%{domain_filter}%",))
    else:
        # Show top domains by volume
        cur.execute("""
            SELECT
                REGEXP_REPLACE(e.from_email, '^.*@', '') as domain,
                ef.service_type,
                COUNT(*) as total,
                SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) as replied,
                SUM(CASE WHEN e.action = 'ARCHIVED' THEN 1 ELSE 0 END) as archived,
                SUM(CASE WHEN e.action = 'IGNORED' THEN 1 ELSE 0 END) as ignored,
                ROUND(AVG(ef.service_importance)::numeric, 2) as avg_importance
            FROM emails e
            JOIN email_features ef ON ef.email_id = e.id
            WHERE e.is_sent = false AND ef.is_service_email = true
            GROUP BY domain, ef.service_type
            HAVING COUNT(*) >= 5
            ORDER BY total DESC
            LIMIT 25
        """)

    results = cur.fetchall()

    print(f"\n{'Sender/Domain':<45} {'Type':<15} {'Total':<7} {'Replied':<8} {'Archived':<9} {'Ignored':<8} {'Imp'}")
    print("-"*105)
    for row in results:
        sender, svc_type, total, replied, archived, ignored, importance = row
        sender_short = sender[:43]
        svc_short = (svc_type or "unknown")[:13]
        print(f"{sender_short:<45} {svc_short:<15} {total:<7} {replied:<8} {archived:<9} {ignored:<8} {importance or 0:>4.2f}")


def relationship_validation(conn: psycopg2.extensions.connection) -> None:
    """Validate that relationship_strength predicts reply behavior."""
    print_section("RELATIONSHIP STRENGTH VALIDATION")

    cur = conn.cursor()
    cur.execute("""
        SELECT
            CASE
                WHEN ef.relationship_strength >= 0.7 THEN 'Strong (0.7+)'
                WHEN ef.relationship_strength >= 0.5 THEN 'Medium (0.5-0.7)'
                WHEN ef.relationship_strength >= 0.3 THEN 'Weak (0.3-0.5)'
                ELSE 'Minimal (<0.3)'
            END as tier,
            COUNT(*) as emails,
            SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) as replied,
            ROUND(100.0 * SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) / COUNT(*), 1) as reply_rate
        FROM email_features ef
        JOIN emails e ON e.id = ef.email_id
        WHERE e.is_sent = false
        GROUP BY 1
        ORDER BY MIN(ef.relationship_strength) DESC
    """)

    results = cur.fetchall()

    print(f"\n{'Relationship Tier':<20} {'Emails':<10} {'Replied':<10} {'Reply Rate'}")
    print("-"*55)
    for row in results:
        tier, emails, replied, rate = row
        print(f"{tier:<20} {emails:<10} {replied:<10} {rate:>6.1f}%")

    print("\n✓ If reply rate correlates with tier, relationship_strength is working correctly.")


def priority_validation(conn: psycopg2.extensions.connection) -> None:
    """Validate that priority_score identifies actionable emails."""
    print_section("PRIORITY SCORE VALIDATION")

    cur = conn.cursor()
    cur.execute("""
        SELECT
            CASE
                WHEN ep.priority_score >= 0.6 THEN 'High (0.6+)'
                WHEN ep.priority_score >= 0.4 THEN 'Medium (0.4-0.6)'
                WHEN ep.priority_score >= 0.2 THEN 'Low (0.2-0.4)'
                ELSE 'Very Low (<0.2)'
            END as tier,
            COUNT(*) as emails,
            SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) as replied,
            ROUND(100.0 * SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) / COUNT(*), 1) as reply_rate
        FROM email_priority ep
        JOIN emails e ON e.id = ep.email_id
        WHERE e.is_sent = false
        GROUP BY 1
        ORDER BY MIN(ep.priority_score) DESC
    """)

    results = cur.fetchall()

    print(f"\n{'Priority Tier':<20} {'Emails':<10} {'Replied':<10} {'Reply Rate'}")
    print("-"*55)
    for row in results:
        tier, emails, replied, rate = row
        print(f"{tier:<20} {emails:<10} {replied:<10} {rate:>6.1f}%")

    print("\n✓ Higher priority tiers should have higher reply rates.")


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate email data quality')
    parser.add_argument('--top', type=int, default=15, help='Number of results to show')
    parser.add_argument('--domain', type=str, help='Filter by domain (e.g., chase.com)')
    args = parser.parse_args()

    if not DB_URL:
        print("ERROR: DATABASE_URL not set")
        return

    conn = get_connection()

    try:
        important_people(conn, args.top)
        service_breakdown(conn, args.top)
        what_you_ignore(conn, args.top)

        if args.domain:
            domain_analysis(conn, args.domain)
        else:
            domain_analysis(conn)

        relationship_validation(conn)
        priority_validation(conn)

    finally:
        conn.close()

    print("\n" + "="*70)
    print(" Run with --domain chase.com to see breakdown for a specific domain")
    print("="*70)


if __name__ == '__main__':
    main()
