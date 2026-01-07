"""Stage 3: Populate threads table with aggregated email data.

Computes thread-level statistics from emails:
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
"""

from __future__ import annotations

import asyncio
import time

import asyncpg

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult


async def populate_threads_async(
    conn: asyncpg.Connection,
    batch_size: int = 1000,
) -> tuple[int, dict[str, int]]:
    """Populate threads table with aggregated email data.

    Args:
        conn: Connected asyncpg connection.
        batch_size: Number of threads to insert per batch.

    Returns:
        Tuple of (thread_count, stats_dict).
    """
    stats: dict[str, int] = {"threads_inserted": 0}

    # Compute thread aggregates using a single SQL query
    threads_data = await conn.fetch(
        """
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
    """
    )

    if not threads_data:
        return 0, stats

    # Clear existing data (idempotent)
    await conn.execute("DELETE FROM threads")

    # Insert in batches
    for i in range(0, len(threads_data), batch_size):
        batch = threads_data[i : i + batch_size]

        await conn.executemany(
            """
            INSERT INTO threads (
                thread_id, subject, participants, email_count,
                your_email_count, your_reply_count, started_at, last_activity,
                thread_duration_seconds, avg_response_time_seconds,
                has_attachments, total_attachment_count
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            [
                (
                    row["thread_id"],
                    row["subject"],
                    row["participants"] or [],
                    row["email_count"],
                    row["your_email_count"],
                    row["your_reply_count"],
                    row["started_at"],
                    row["last_activity"],
                    row["thread_duration_seconds"],
                    row["avg_response_time_seconds"],
                    row["has_attachments"] or False,
                    row["total_attachment_count"] or 0,
                )
                for row in batch
            ],
        )

        stats["threads_inserted"] += len(batch)

    return stats["threads_inserted"], stats


def convert_db_url(db_url: str) -> str:
    """Convert postgresql:// URL to postgres:// for asyncpg compatibility.

    Args:
        db_url: Database URL, possibly starting with postgresql://.

    Returns:
        URL with postgres:// prefix for asyncpg.
    """
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgres://", 1)
    return db_url


async def run_async(db_url: str, batch_size: int = 1000) -> tuple[int, dict[str, int]]:
    """Async implementation of thread population.

    Args:
        db_url: Database URL in asyncpg format.
        batch_size: Number of threads to insert per batch.

    Returns:
        Tuple of (thread_count, stats_dict).
    """
    conn = await asyncpg.connect(db_url)
    try:
        return await populate_threads_async(conn, batch_size)
    finally:
        await conn.close()


def run(config: Config, *, batch_size: int = 1000) -> StageResult:
    """Run Stage 3: Populate threads table from email data.

    Args:
        config: Application configuration with database_url.
        batch_size: Number of threads to insert per batch.

    Returns:
        StageResult with thread population counts.
    """
    start_time = time.time()

    db_url = convert_db_url(config.database_url)
    count, stats = asyncio.run(run_async(db_url, batch_size))
    duration = time.time() - start_time

    return StageResult(
        success=True,
        records_processed=count,
        duration_seconds=duration,
        message=f"Populated {count} threads",
        metadata=stats,
    )
