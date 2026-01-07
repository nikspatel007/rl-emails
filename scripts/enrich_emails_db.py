#!/usr/bin/env python3
"""Enrich emails with action labels directly in PostgreSQL.

Computes action labels from behavioral signals:
- is_sent: TRUE if user sent this email
- action: REPLIED, FORWARDED, STARRED, ARCHIVED, IGNORED, COMPOSED
- timing: IMMEDIATE, SAME_DAY, NEXT_DAY, LATER, NEVER
- response_time_seconds: Time between sending/receiving and getting/sending reply

For RECEIVED emails (is_sent=FALSE):
  - Tracks YOUR behavior (did you reply, archive, ignore?)
  - response_time_seconds = how long YOU took to reply

For SENT emails (is_sent=TRUE):
  - Tracks OTHERS' behavior (did they reply to you?)
  - response_time_seconds = how long THEY took to reply

Usage:
    python scripts/enrich_emails_db.py
    python scripts/enrich_emails_db.py --dry-run
    python scripts/enrich_emails_db.py --additional-emails alias@example.com

Required .env variables:
    DB_URL - PostgreSQL connection URL
    YOUR_EMAIL - Your email address for identifying sent emails
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta

import asyncpg
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from .env (required)
DB_URL = os.environ.get('DB_URL')
YOUR_EMAIL = os.environ.get('YOUR_EMAIL')

# Response time thresholds in seconds
RESPONSE_TIME_THRESHOLDS = {
    'IMMEDIATE': 3600,        # < 1 hour
    'SAME_DAY': 86400,        # < 24 hours
    'NEXT_DAY': 172800,       # < 48 hours
    # LATER: >= 48 hours
}

# Age threshold for IGNORED classification (days)
IGNORED_AGE_DAYS = 7


def classify_response_time(seconds: float) -> str:
    """Classify response time into buckets."""
    if seconds < RESPONSE_TIME_THRESHOLDS['IMMEDIATE']:
        return 'IMMEDIATE'
    elif seconds < RESPONSE_TIME_THRESHOLDS['SAME_DAY']:
        return 'SAME_DAY'
    elif seconds < RESPONSE_TIME_THRESHOLDS['NEXT_DAY']:
        return 'NEXT_DAY'
    else:
        return 'LATER'


async def mark_sent_emails(conn: asyncpg.Connection, your_emails: list[str]) -> int:
    """Mark emails as is_sent=TRUE based on from_email matching user's addresses."""

    # Build WHERE clause for multiple email addresses
    placeholders = ', '.join(f'${i+1}' for i in range(len(your_emails)))

    result = await conn.execute(f"""
        UPDATE emails
        SET is_sent = TRUE
        WHERE LOWER(from_email) IN ({placeholders})
          AND (is_sent IS NULL OR is_sent = FALSE)
    """, *[e.lower() for e in your_emails])

    # Parse "UPDATE N" to get count
    count = int(result.split()[-1]) if result else 0
    return count


async def compute_replied_actions(conn: asyncpg.Connection, your_emails: list[str]) -> int:
    """Compute REPLIED action by matching sent emails to received via in_reply_to."""

    # Find received emails that were replied to
    # A received email was REPLIED if:
    # 1. There exists a sent email (is_sent=TRUE)
    # 2. The sent email's in_reply_to matches the received email's message_id
    # 3. OR the sent email is in the same thread and comes after the received email

    result = await conn.execute("""
        WITH replied_emails AS (
            -- Match via in_reply_to header
            SELECT DISTINCT received.id as email_id,
                   sent.date_parsed as reply_date,
                   received.date_parsed as received_date
            FROM emails received
            JOIN emails sent ON sent.in_reply_to = received.message_id
            WHERE received.is_sent = FALSE
              AND sent.is_sent = TRUE
              AND sent.date_parsed > received.date_parsed
        ),
        reply_times AS (
            SELECT email_id,
                   EXTRACT(EPOCH FROM (reply_date - received_date)) as response_seconds
            FROM replied_emails
            WHERE reply_date IS NOT NULL AND received_date IS NOT NULL
        )
        UPDATE emails e
        SET action = 'REPLIED',
            response_time_seconds = rt.response_seconds::INTEGER,
            timing = CASE
                WHEN rt.response_seconds < 3600 THEN 'IMMEDIATE'
                WHEN rt.response_seconds < 86400 THEN 'SAME_DAY'
                WHEN rt.response_seconds < 172800 THEN 'NEXT_DAY'
                ELSE 'LATER'
            END,
            enriched_at = NOW()
        FROM reply_times rt
        WHERE e.id = rt.email_id
          AND e.action IS NULL
    """)

    count = int(result.split()[-1]) if result else 0
    return count


async def compute_starred_actions(conn: asyncpg.Connection) -> int:
    """Mark STARRED emails based on labels."""

    result = await conn.execute("""
        UPDATE emails
        SET action = 'STARRED',
            enriched_at = NOW()
        WHERE is_sent = FALSE
          AND action IS NULL
          AND 'Starred' = ANY(labels)
    """)

    count = int(result.split()[-1]) if result else 0
    return count


async def compute_archived_actions(conn: asyncpg.Connection, reference_date: datetime) -> int:
    """Mark ARCHIVED emails - read but not replied, older than threshold.

    Includes:
    - Group/FYI emails (has CC or multiple TO recipients)
    - Emails from frequent senders (>20 emails total)
    - Read emails (not unread)
    """

    cutoff_date = reference_date - timedelta(days=IGNORED_AGE_DAYS)

    result = await conn.execute("""
        UPDATE emails e
        SET action = 'ARCHIVED',
            timing = 'NEVER',
            enriched_at = NOW()
        WHERE e.is_sent = FALSE
          AND e.action IS NULL
          AND e.date_parsed < $1
          AND (
              -- Explicit archive label
              'Archived' = ANY(e.labels)
              -- Read emails (not unread, not in inbox)
              OR ('Unread' != ALL(e.labels) AND 'Inbox' != ALL(e.labels))
              -- Group/FYI emails (has CC or multiple TO)
              OR (e.cc_emails IS NOT NULL AND array_length(e.cc_emails, 1) >= 1)
              OR (e.to_emails IS NOT NULL AND array_length(e.to_emails, 1) > 1)
              -- From frequent senders (>20 emails)
              OR EXISTS (
                  SELECT 1 FROM users u
                  WHERE LOWER(u.email) = LOWER(e.from_email)
                    AND u.emails_from > 20
              )
          )
    """, cutoff_date)

    count = int(result.split()[-1]) if result else 0
    return count


async def compute_ignored_actions(conn: asyncpg.Connection, reference_date: datetime) -> int:
    """Mark IGNORED emails - truly ignored 1:1 emails.

    Only marks as IGNORED if:
    - Unread and old
    - 1:1 communication (no CC, single TO recipient)
    - NOT from frequent senders (relationship exists but chose not to engage)

    This excludes group/FYI emails from close contacts.
    """

    cutoff_date = reference_date - timedelta(days=IGNORED_AGE_DAYS)

    result = await conn.execute("""
        UPDATE emails e
        SET action = 'IGNORED',
            timing = 'NEVER',
            enriched_at = NOW()
        WHERE e.is_sent = FALSE
          AND e.action IS NULL
          AND e.date_parsed < $1
          AND 'Unread' = ANY(e.labels)
          -- Must be 1:1 (no CC, single TO)
          AND (e.cc_emails IS NULL OR array_length(e.cc_emails, 1) = 0)
          AND (e.to_emails IS NULL OR array_length(e.to_emails, 1) = 1)
          -- NOT from frequent senders (those go to ARCHIVED)
          AND NOT EXISTS (
              SELECT 1 FROM users u
              WHERE LOWER(u.email) = LOWER(e.from_email)
                AND u.emails_from > 20
          )
    """, cutoff_date)

    count = int(result.split()[-1]) if result else 0
    return count


async def compute_pending_actions(conn: asyncpg.Connection) -> int:
    """Mark remaining recent emails as PENDING."""

    result = await conn.execute("""
        UPDATE emails
        SET action = 'PENDING',
            enriched_at = NOW()
        WHERE is_sent = FALSE
          AND action IS NULL
    """)

    count = int(result.split()[-1]) if result else 0
    return count


async def mark_sent_action(conn: asyncpg.Connection) -> int:
    """Mark sent emails with COMPOSED action."""

    result = await conn.execute("""
        UPDATE emails
        SET action = 'COMPOSED',
            enriched_at = NOW()
        WHERE is_sent = TRUE
          AND action IS NULL
    """)

    count = int(result.split()[-1]) if result else 0
    return count


async def compute_reverse_tracking(conn: asyncpg.Connection) -> int:
    """Track how quickly OTHERS respond to YOUR sent emails.

    For sent emails (is_sent=TRUE), compute:
    - response_time_seconds: Time until first reply received
    - timing: Bucket (IMMEDIATE, SAME_DAY, etc.)

    Excludes automated responses (bounces, auto-replies, etc.)
    """

    result = await conn.execute("""
        WITH sent_replies AS (
            -- Find replies TO your sent emails (excluding automated responses)
            SELECT DISTINCT sent.id as sent_email_id,
                   received.date_parsed as reply_date,
                   sent.date_parsed as sent_date
            FROM emails sent
            JOIN emails received ON received.in_reply_to = sent.message_id
            WHERE sent.is_sent = TRUE
              AND received.is_sent = FALSE
              AND received.date_parsed > sent.date_parsed
              -- Filter out automated responses
              AND LOWER(received.from_email) NOT LIKE '%mailer-daemon%'
              AND LOWER(received.from_email) NOT LIKE '%postmaster%'
              AND LOWER(received.from_email) NOT LIKE '%noreply%'
              AND LOWER(received.from_email) NOT LIKE '%no-reply%'
              AND LOWER(received.from_email) NOT LIKE '%donotreply%'
              AND LOWER(received.from_email) NOT LIKE '%autoreply%'
              AND LOWER(received.from_email) NOT LIKE '%auto-reply%'
              AND LOWER(received.subject) NOT LIKE '%delivery status notification%'
              AND LOWER(received.subject) NOT LIKE '%out of office%'
              AND LOWER(received.subject) NOT LIKE '%automatic reply%'
        ),
        first_replies AS (
            -- Get the FIRST reply for each sent email
            SELECT sent_email_id,
                   MIN(reply_date) as first_reply_date,
                   sent_date
            FROM sent_replies
            WHERE reply_date IS NOT NULL AND sent_date IS NOT NULL
            GROUP BY sent_email_id, sent_date
        ),
        reply_times AS (
            SELECT sent_email_id,
                   EXTRACT(EPOCH FROM (first_reply_date - sent_date)) as response_seconds
            FROM first_replies
        )
        UPDATE emails e
        SET response_time_seconds = rt.response_seconds::INTEGER,
            timing = CASE
                WHEN rt.response_seconds < 3600 THEN 'IMMEDIATE'
                WHEN rt.response_seconds < 86400 THEN 'SAME_DAY'
                WHEN rt.response_seconds < 172800 THEN 'NEXT_DAY'
                ELSE 'LATER'
            END,
            enriched_at = NOW()
        FROM reply_times rt
        WHERE e.id = rt.sent_email_id
          AND e.is_sent = TRUE
    """)

    count = int(result.split()[-1]) if result else 0
    return count


async def enrich_emails(conn: asyncpg.Connection, your_emails: list[str]) -> dict:
    """Run full enrichment pipeline on emails table."""

    stats = {
        'total_emails': 0,
        'sent_marked': 0,
        'replied': 0,
        'starred': 0,
        'archived': 0,
        'ignored': 0,
        'pending': 0,
        'composed': 0,
        'reverse_tracked': 0,
    }

    # Get total count
    stats['total_emails'] = await conn.fetchval("SELECT COUNT(*) FROM emails")
    print(f"Total emails: {stats['total_emails']:,}")

    # Get reference date (latest email)
    reference_date = await conn.fetchval("""
        SELECT MAX(date_parsed) FROM emails WHERE date_parsed IS NOT NULL
    """)
    if reference_date is None:
        reference_date = datetime.now()
    print(f"Reference date: {reference_date}")

    # Step 1: Mark sent emails
    print("\nStep 1: Marking sent emails...")
    stats['sent_marked'] = await mark_sent_emails(conn, your_emails)
    print(f"  Marked {stats['sent_marked']:,} emails as is_sent=TRUE")

    # Verify sent count
    sent_count = await conn.fetchval("SELECT COUNT(*) FROM emails WHERE is_sent = TRUE")
    print(f"  Total sent emails: {sent_count:,}")

    # Step 2: Compute REPLIED actions
    print("\nStep 2: Computing REPLIED actions...")
    stats['replied'] = await compute_replied_actions(conn, your_emails)
    print(f"  Found {stats['replied']:,} replied emails")

    # Step 3: Mark STARRED
    print("\nStep 3: Marking STARRED emails...")
    stats['starred'] = await compute_starred_actions(conn)
    print(f"  Found {stats['starred']:,} starred emails")

    # Step 4: Mark ARCHIVED
    print("\nStep 4: Marking ARCHIVED emails...")
    stats['archived'] = await compute_archived_actions(conn, reference_date)
    print(f"  Found {stats['archived']:,} archived emails")

    # Step 5: Mark IGNORED
    print("\nStep 5: Marking IGNORED emails...")
    stats['ignored'] = await compute_ignored_actions(conn, reference_date)
    print(f"  Found {stats['ignored']:,} ignored emails")

    # Step 6: Mark PENDING (recent, not yet acted on)
    print("\nStep 6: Marking PENDING emails...")
    stats['pending'] = await compute_pending_actions(conn)
    print(f"  Found {stats['pending']:,} pending emails")

    # Step 7: Mark COMPOSED for sent emails
    print("\nStep 7: Marking COMPOSED for sent emails...")
    stats['composed'] = await mark_sent_action(conn)
    print(f"  Marked {stats['composed']:,} composed emails")

    # Step 8: Reverse tracking - how quickly others respond to YOUR emails
    print("\nStep 8: Computing reverse tracking (others' responses to your emails)...")
    stats['reverse_tracked'] = await compute_reverse_tracking(conn)
    print(f"  Tracked {stats['reverse_tracked']:,} sent emails with replies")

    return stats


async def verify_results(conn: asyncpg.Connection):
    """Verify enrichment results."""
    print("\n" + "=" * 60)
    print("ENRICHMENT RESULTS")
    print("=" * 60)

    # Action distribution
    print("\nAction distribution:")
    actions = await conn.fetch("""
        SELECT action, COUNT(*) as cnt,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
        FROM emails
        GROUP BY action
        ORDER BY cnt DESC
    """)
    for row in actions:
        print(f"  {row['action'] or 'NULL':15s}: {row['cnt']:7,} ({row['pct']}%)")

    # Timing distribution for replied
    print("\nTiming distribution (REPLIED only):")
    timing = await conn.fetch("""
        SELECT timing, COUNT(*) as cnt
        FROM emails
        WHERE action = 'REPLIED'
        GROUP BY timing
        ORDER BY cnt DESC
    """)
    for row in timing:
        print(f"  {row['timing'] or 'NULL':15s}: {row['cnt']:7,}")

    # Response time stats
    print("\nResponse time statistics:")
    stats = await conn.fetchrow("""
        SELECT
            COUNT(*) as count,
            MIN(response_time_seconds) as min_secs,
            MAX(response_time_seconds) as max_secs,
            AVG(response_time_seconds) as avg_secs
        FROM emails
        WHERE response_time_seconds IS NOT NULL
    """)
    if stats['count'] > 0:
        print(f"  Emails with response time: {stats['count']:,}")
        print(f"  Min: {stats['min_secs']:,}s ({stats['min_secs']/3600:.1f}h)")
        print(f"  Max: {stats['max_secs']:,}s ({stats['max_secs']/3600:.1f}h)")
        print(f"  Avg: {stats['avg_secs']:,.0f}s ({stats['avg_secs']/3600:.1f}h)")

    # Sent vs received
    print("\nSent vs Received:")
    sent = await conn.fetchval("SELECT COUNT(*) FROM emails WHERE is_sent = TRUE")
    received = await conn.fetchval("SELECT COUNT(*) FROM emails WHERE is_sent = FALSE")
    print(f"  Sent: {sent:,}")
    print(f"  Received: {received:,}")

    # Reverse tracking stats
    print("\nReverse Tracking (Others' responses to YOUR sent emails):")
    sent_with_replies = await conn.fetchval("""
        SELECT COUNT(*) FROM emails
        WHERE is_sent = TRUE AND response_time_seconds IS NOT NULL
    """)
    print(f"  Sent emails that got replies: {sent_with_replies:,}")

    if sent_with_replies > 0:
        timing_dist = await conn.fetch("""
            SELECT timing, COUNT(*) as cnt
            FROM emails
            WHERE is_sent = TRUE AND timing IS NOT NULL
            GROUP BY timing
            ORDER BY cnt DESC
        """)
        print("  Response timing distribution:")
        for row in timing_dist:
            print(f"    {row['timing']:15s}: {row['cnt']:7,}")

        # Average response time received
        avg_response = await conn.fetchval("""
            SELECT AVG(response_time_seconds)
            FROM emails
            WHERE is_sent = TRUE AND response_time_seconds IS NOT NULL
        """)
        print(f"  Average time to get a reply: {avg_response/3600:.1f}h")

    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description='Enrich emails with action labels directly in PostgreSQL'
    )
    parser.add_argument(
        '--additional-emails',
        nargs='*',
        default=[],
        help='Additional email addresses that belong to you'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    # Validate required environment variables
    if not DB_URL:
        print("Error: DB_URL not set in .env file")
        sys.exit(1)
    if not YOUR_EMAIL:
        print("Error: YOUR_EMAIL not set in .env file")
        sys.exit(1)

    # Build list of user's email addresses
    your_emails = [YOUR_EMAIL] + args.additional_emails

    print("Email Enrichment (Database)")
    print("=" * 60)
    print(f"Database: {DB_URL}")
    print(f"Your emails: {', '.join(your_emails)}")

    if args.dry_run:
        print("\n[DRY RUN - no changes will be made]")

    print(f"\nConnecting to database...")
    conn = await asyncpg.connect(DB_URL)

    try:
        if args.dry_run:
            # Just show current state
            await verify_results(conn)
        else:
            # Run enrichment
            stats = await enrich_emails(conn, your_emails)

            # Verify results
            await verify_results(conn)

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
