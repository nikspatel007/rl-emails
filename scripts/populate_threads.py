#!/usr/bin/env python3
"""Populate threads table with aggregated data from emails.

This script computes thread-level statistics by aggregating email data:
- email_count: number of emails in thread
- participants: array of unique email addresses
- started_at: timestamp of first email
- last_activity: timestamp of last email
- your_email_count: emails sent by user
- your_reply_count: replies sent by user
- thread_duration_seconds: time between first and last email
- avg_response_time_seconds: average time between consecutive emails
- has_attachments: whether any email has attachments
- total_attachment_count: sum of attachments across all emails

Usage:
    # Start PostgreSQL first:
    # docker run -d --name rl-emails-pg -p 5433:5432 \
    #   -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rl_emails postgres:15

    # Run population:
    python scripts/populate_threads.py

    # With custom database URL:
    python scripts/populate_threads.py --db-url postgresql://user:pass@host:port/db
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

try:
    import asyncpg
except ImportError:
    print("Error: asyncpg required. Install with: pip install asyncpg")
    sys.exit(1)


async def populate_threads(
    conn: asyncpg.Connection,
    batch_size: int = 1000,
) -> dict:
    """Populate threads table with aggregated email data.

    Args:
        conn: Connected asyncpg connection
        batch_size: Number of threads to insert per batch

    Returns:
        Statistics dict with counts
    """
    stats = {
        'threads_inserted': 0,
        'threads_updated': 0,
        'threads_skipped': 0,
    }

    print("Querying thread statistics from emails...")

    # Compute thread aggregates using a single SQL query
    threads_data = await conn.fetch("""
        WITH thread_stats AS (
            SELECT
                thread_id,
                MIN(subject) as subject,
                array_agg(DISTINCT from_email) FILTER (WHERE from_email IS NOT NULL) as participants,
                COUNT(*) as email_count,
                COUNT(*) FILTER (WHERE is_sent = true) as your_email_count,
                COUNT(*) FILTER (WHERE is_sent = true AND in_reply_to IS NOT NULL) as your_reply_count,
                MIN(date_parsed) as started_at,
                MAX(date_parsed) as last_activity,
                EXTRACT(EPOCH FROM (MAX(date_parsed) - MIN(date_parsed))) as thread_duration_seconds,
                bool_or(has_attachments) as has_attachments,
                COALESCE(SUM(attachment_count), 0) as total_attachment_count
            FROM emails
            WHERE thread_id IS NOT NULL
            GROUP BY thread_id
        ),
        response_times AS (
            SELECT
                thread_id,
                AVG(EXTRACT(EPOCH FROM (date_parsed - lag_date))) as avg_response_time_seconds
            FROM (
                SELECT
                    thread_id,
                    date_parsed,
                    LAG(date_parsed) OVER (PARTITION BY thread_id ORDER BY date_parsed) as lag_date
                FROM emails
                WHERE thread_id IS NOT NULL
            ) sub
            WHERE lag_date IS NOT NULL
            GROUP BY thread_id
        )
        SELECT
            ts.*,
            rt.avg_response_time_seconds
        FROM thread_stats ts
        LEFT JOIN response_times rt ON ts.thread_id = rt.thread_id
    """)

    print(f"Found {len(threads_data)} threads to populate")

    if not threads_data:
        print("No threads found to populate")
        return stats

    # Check if threads table is empty or has data
    existing_count = await conn.fetchval("SELECT COUNT(*) FROM threads")
    print(f"Existing threads in table: {existing_count}")

    # Clear existing data if any (for idempotent re-runs)
    if existing_count > 0:
        print("Clearing existing thread data for fresh population...")
        await conn.execute("DELETE FROM threads")

    # Insert in batches
    print(f"Inserting threads in batches of {batch_size}...")

    for i in range(0, len(threads_data), batch_size):
        batch = threads_data[i:i + batch_size]

        await conn.executemany("""
            INSERT INTO threads (
                thread_id, subject, participants, email_count,
                your_email_count, your_reply_count, started_at, last_activity,
                thread_duration_seconds, avg_response_time_seconds,
                has_attachments, total_attachment_count
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """, [
            (
                row['thread_id'],
                row['subject'],
                row['participants'] or [],
                row['email_count'],
                row['your_email_count'],
                row['your_reply_count'],
                row['started_at'],
                row['last_activity'],
                row['thread_duration_seconds'],
                row['avg_response_time_seconds'],
                row['has_attachments'] or False,
                row['total_attachment_count'] or 0,
            )
            for row in batch
        ])

        stats['threads_inserted'] += len(batch)

        if (i + batch_size) % 5000 == 0:
            print(f"  Inserted {stats['threads_inserted']} threads...")

    return stats


async def verify_results(conn: asyncpg.Connection):
    """Verify the population results."""
    print("\n=== Verification ===")

    # Count threads
    thread_count = await conn.fetchval("SELECT COUNT(*) FROM threads")
    print(f"Total threads in table: {thread_count}")

    # Sample statistics
    sample = await conn.fetchrow("""
        SELECT
            AVG(email_count) as avg_emails_per_thread,
            MAX(email_count) as max_emails_per_thread,
            AVG(array_length(participants, 1)) as avg_participants,
            SUM(CASE WHEN has_attachments THEN 1 ELSE 0 END) as threads_with_attachments,
            AVG(thread_duration_seconds) / 3600 as avg_duration_hours
        FROM threads
    """)

    if sample:
        print(f"Avg emails per thread: {sample['avg_emails_per_thread']:.1f}")
        print(f"Max emails in thread: {sample['max_emails_per_thread']}")
        print(f"Avg participants: {sample['avg_participants']:.1f}")
        print(f"Threads with attachments: {sample['threads_with_attachments']}")
        if sample['avg_duration_hours'] is not None:
            print(f"Avg thread duration: {sample['avg_duration_hours']:.1f} hours")
        else:
            print("Avg thread duration: N/A")


async def main():
    """Main entry point."""
    # Required environment variable
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL environment variable is required")
        sys.exit(1)

    print("Populate Threads Table")
    print("=" * 40)
    print(f"\nConnecting to {db_url}...")

    try:
        conn = await asyncpg.connect(db_url)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    try:
        # Check if threads table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'threads'
            )
        """)

        if not table_exists:
            print("Error: threads table does not exist. Run create_schema.sql first.")
            sys.exit(1)

        # Populate threads
        stats = await populate_threads(conn, batch_size=1000)

        print(f"\nPopulation complete!")
        print(f"  Threads inserted: {stats['threads_inserted']}")

        # Verify
        await verify_results(conn)

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
