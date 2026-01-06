#!/usr/bin/env python3
"""Populate users table from email addresses in emails table.

Extracts unique users from from_email, to_emails[], cc_emails[] and computes:
- emails_from: count of emails sent by this user
- emails_to: count of emails received (to + cc)
- threads_with: distinct threads they're involved in
- reply_count: emails they sent that are replies
- avg_response_time_seconds: average response time for their replies
- reply_rate: fraction of received emails they replied to
- first_contact, last_contact: date bounds
- is_you: true for me@nik-patel.com variants
- name: extracted from from_name field
"""

import argparse
import os

import psycopg2
from psycopg2.extras import execute_values

# Default configuration (can be overridden by env vars or CLI args)
DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"


def get_connection(db_url: str):
    """Connect to PostgreSQL database."""
    return psycopg2.connect(db_url)


def populate_users(conn) -> int:
    """Extract unique users and populate the users table.

    Returns the number of users created.
    """
    cur = conn.cursor()

    # Clear existing users
    cur.execute("TRUNCATE users RESTART IDENTITY CASCADE")

    # Step 1: Extract all unique email addresses with their names
    # We get names from from_email -> from_name mapping
    print("Extracting unique email addresses...")

    cur.execute("""
        WITH all_emails AS (
            -- From addresses with names
            SELECT LOWER(TRIM(from_email)) as email, from_name as name
            FROM emails
            WHERE from_email IS NOT NULL AND from_email != ''

            UNION ALL

            -- To addresses (no names available)
            SELECT LOWER(TRIM(unnest(to_emails))) as email, NULL as name
            FROM emails
            WHERE to_emails IS NOT NULL

            UNION ALL

            -- CC addresses (no names available)
            SELECT LOWER(TRIM(unnest(cc_emails))) as email, NULL as name
            FROM emails
            WHERE cc_emails IS NOT NULL
        ),
        unique_emails AS (
            SELECT
                email,
                -- Pick the first non-null name we find for this email
                MAX(name) as name
            FROM all_emails
            WHERE email IS NOT NULL AND email != '' AND email LIKE '%@%'
            GROUP BY email
        )
        SELECT email, name FROM unique_emails ORDER BY email
    """)

    users = cur.fetchall()
    print(f"Found {len(users)} unique email addresses")

    # Step 2: Insert users with basic info
    print("Inserting users...")

    # Determine is_you based on email patterns
    you_patterns = ['me@nik-patel.com', 'nik@nik-patel.com', 'nikpatel']

    user_data = []
    for email, name in users:
        is_you = any(pattern in email.lower() for pattern in you_patterns)
        user_data.append((email, name, is_you))

    execute_values(
        cur,
        """
        INSERT INTO users (email, name, is_you)
        VALUES %s
        ON CONFLICT (email) DO NOTHING
        """,
        user_data
    )

    # Step 3: Compute emails_from (emails sent by this user)
    print("Computing emails_from counts...")
    cur.execute("""
        UPDATE users u
        SET emails_from = sub.cnt
        FROM (
            SELECT LOWER(TRIM(from_email)) as email, COUNT(*) as cnt
            FROM emails
            WHERE from_email IS NOT NULL
            GROUP BY LOWER(TRIM(from_email))
        ) sub
        WHERE u.email = sub.email
    """)

    # Step 4: Compute emails_to (emails received - to or cc)
    print("Computing emails_to counts...")
    cur.execute("""
        WITH received AS (
            SELECT LOWER(TRIM(unnest(to_emails))) as email FROM emails WHERE to_emails IS NOT NULL
            UNION ALL
            SELECT LOWER(TRIM(unnest(cc_emails))) as email FROM emails WHERE cc_emails IS NOT NULL
        )
        UPDATE users u
        SET emails_to = sub.cnt
        FROM (
            SELECT email, COUNT(*) as cnt
            FROM received
            WHERE email IS NOT NULL AND email != ''
            GROUP BY email
        ) sub
        WHERE u.email = sub.email
    """)

    # Step 5: Compute threads_with (distinct threads involving this user)
    print("Computing threads_with counts...")
    cur.execute("""
        WITH user_threads AS (
            -- Threads where user sent email
            SELECT LOWER(TRIM(from_email)) as email, thread_id
            FROM emails
            WHERE from_email IS NOT NULL AND thread_id IS NOT NULL

            UNION

            -- Threads where user received email
            SELECT LOWER(TRIM(unnest(to_emails))) as email, thread_id
            FROM emails
            WHERE to_emails IS NOT NULL AND thread_id IS NOT NULL

            UNION

            SELECT LOWER(TRIM(unnest(cc_emails))) as email, thread_id
            FROM emails
            WHERE cc_emails IS NOT NULL AND thread_id IS NOT NULL
        )
        UPDATE users u
        SET threads_with = sub.cnt
        FROM (
            SELECT email, COUNT(DISTINCT thread_id) as cnt
            FROM user_threads
            WHERE email IS NOT NULL AND email != ''
            GROUP BY email
        ) sub
        WHERE u.email = sub.email
    """)

    # Step 6: Compute reply_count (emails they sent that are replies)
    print("Computing reply_count...")
    cur.execute("""
        UPDATE users u
        SET reply_count = sub.cnt
        FROM (
            SELECT LOWER(TRIM(from_email)) as email, COUNT(*) as cnt
            FROM emails
            WHERE from_email IS NOT NULL
              AND in_reply_to IS NOT NULL
              AND in_reply_to != ''
            GROUP BY LOWER(TRIM(from_email))
        ) sub
        WHERE u.email = sub.email
    """)

    # Step 7: Compute avg_response_time_seconds
    print("Computing avg_response_time_seconds...")
    cur.execute("""
        UPDATE users u
        SET avg_response_time_seconds = sub.avg_time
        FROM (
            SELECT LOWER(TRIM(from_email)) as email,
                   AVG(response_time_seconds)::integer as avg_time
            FROM emails
            WHERE from_email IS NOT NULL
              AND response_time_seconds IS NOT NULL
              AND response_time_seconds > 0
            GROUP BY LOWER(TRIM(from_email))
        ) sub
        WHERE u.email = sub.email
    """)

    # Step 8: Compute reply_rate
    # reply_rate = replies sent / emails received (capped at 1.0)
    print("Computing reply_rate...")
    cur.execute("""
        UPDATE users
        SET reply_rate = LEAST(1.0,
            CASE
                WHEN emails_to > 0 THEN reply_count::float / emails_to
                ELSE 0
            END
        )
        WHERE emails_to > 0 OR reply_count > 0
    """)

    # Step 9: Compute first_contact and last_contact
    print("Computing contact date ranges...")
    cur.execute("""
        WITH contact_dates AS (
            -- Dates when user sent email
            SELECT LOWER(TRIM(from_email)) as email, date_parsed as contact_date
            FROM emails
            WHERE from_email IS NOT NULL AND date_parsed IS NOT NULL

            UNION ALL

            -- Dates when user received email
            SELECT LOWER(TRIM(unnest(to_emails))) as email, date_parsed as contact_date
            FROM emails
            WHERE to_emails IS NOT NULL AND date_parsed IS NOT NULL

            UNION ALL

            SELECT LOWER(TRIM(unnest(cc_emails))) as email, date_parsed as contact_date
            FROM emails
            WHERE cc_emails IS NOT NULL AND date_parsed IS NOT NULL
        )
        UPDATE users u
        SET first_contact = sub.first_dt,
            last_contact = sub.last_dt
        FROM (
            SELECT email, MIN(contact_date) as first_dt, MAX(contact_date) as last_dt
            FROM contact_dates
            WHERE email IS NOT NULL AND email != ''
            GROUP BY email
        ) sub
        WHERE u.email = sub.email
    """)

    conn.commit()

    # Get final count
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]

    cur.close()
    return count


def print_stats(conn):
    """Print statistics about populated users."""
    cur = conn.cursor()

    print("\n=== User Population Stats ===")

    cur.execute("SELECT COUNT(*) FROM users")
    print(f"Total users: {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM users WHERE is_you = true")
    print(f"Users marked as 'you': {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM users WHERE name IS NOT NULL AND name != ''")
    print(f"Users with names: {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM users WHERE emails_from > 0")
    print(f"Users who sent emails: {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM users WHERE emails_to > 0")
    print(f"Users who received emails: {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM users WHERE reply_count > 0")
    print(f"Users who replied to emails: {cur.fetchone()[0]}")

    cur.execute("SELECT COUNT(*) FROM users WHERE avg_response_time_seconds IS NOT NULL")
    print(f"Users with response time data: {cur.fetchone()[0]}")

    cur.execute("""
        SELECT email, name, emails_from, emails_to, reply_count,
               avg_response_time_seconds, reply_rate
        FROM users
        WHERE is_you = true
    """)
    you_users = cur.fetchall()
    if you_users:
        print("\n=== Your accounts ===")
        for u in you_users:
            print(f"  {u[0]}: sent={u[2]}, received={u[3]}, replies={u[4]}, "
                  f"avg_response={u[5]}s, reply_rate={u[6]:.2%}" if u[6] else f"  {u[0]}: sent={u[2]}, received={u[3]}")

    # Top senders
    cur.execute("""
        SELECT email, emails_from
        FROM users
        WHERE NOT is_you
        ORDER BY emails_from DESC
        LIMIT 5
    """)
    print("\n=== Top 5 senders (excluding you) ===")
    for email, count in cur.fetchall():
        print(f"  {email}: {count}")

    # Top receivers
    cur.execute("""
        SELECT email, emails_to
        FROM users
        WHERE NOT is_you
        ORDER BY emails_to DESC
        LIMIT 5
    """)
    print("\n=== Top 5 receivers (excluding you) ===")
    for email, count in cur.fetchall():
        print(f"  {email}: {count}")

    cur.close()


def main():
    parser = argparse.ArgumentParser(
        description='Populate users table from email addresses'
    )
    parser.add_argument(
        '--db-url',
        default=os.environ.get('DB_URL', DEFAULT_DB_URL),
        help='PostgreSQL connection URL (default: $DB_URL or built-in default)'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print stats, do not populate'
    )

    args = parser.parse_args()

    conn = get_connection(args.db_url)

    try:
        if not args.stats_only:
            count = populate_users(conn)
            print(f"\nSuccessfully populated {count} users")

        print_stats(conn)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
