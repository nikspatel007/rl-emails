"""Stage 4: Enrich emails with action labels in PostgreSQL.

Computes action labels from behavioral signals:
- is_sent: TRUE if user sent this email
- action: REPLIED, FORWARDED, STARRED, ARCHIVED, IGNORED, COMPOSED
- timing: IMMEDIATE, SAME_DAY, NEXT_DAY, LATER, NEVER
- response_time_seconds: Time between sending/receiving and getting/sending reply
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any

import asyncpg

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult

# Response time thresholds in seconds
RESPONSE_TIME_THRESHOLDS = {
    "IMMEDIATE": 3600,  # < 1 hour
    "SAME_DAY": 86400,  # < 24 hours
    "NEXT_DAY": 172800,  # < 48 hours
    # LATER: >= 48 hours
}

# Age threshold for IGNORED classification (days)
IGNORED_AGE_DAYS = 7


def classify_response_time(seconds: float) -> str:
    """Classify response time into buckets.

    Args:
        seconds: Response time in seconds.

    Returns:
        Timing bucket string.
    """
    if seconds < RESPONSE_TIME_THRESHOLDS["IMMEDIATE"]:
        return "IMMEDIATE"
    elif seconds < RESPONSE_TIME_THRESHOLDS["SAME_DAY"]:
        return "SAME_DAY"
    elif seconds < RESPONSE_TIME_THRESHOLDS["NEXT_DAY"]:
        return "NEXT_DAY"
    else:
        return "LATER"


def _parse_update_count(result: str | None) -> int:
    """Parse count from PostgreSQL UPDATE result string.

    Args:
        result: Result string like "UPDATE N".

    Returns:
        Number of rows affected.
    """
    if not result:
        return 0
    parts = result.split()
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            return 0
    return 0


async def mark_sent_emails(conn: asyncpg.Connection, your_emails: list[str]) -> int:
    """Mark emails as is_sent=TRUE based on from_email matching user's addresses.

    Args:
        conn: Database connection.
        your_emails: List of user's email addresses.

    Returns:
        Number of emails marked as sent.
    """
    if not your_emails:
        return 0

    placeholders = ", ".join(f"${i + 1}" for i in range(len(your_emails)))
    result = await conn.execute(
        f"""
        UPDATE emails
        SET is_sent = TRUE
        WHERE LOWER(from_email) IN ({placeholders})
          AND (is_sent IS NULL OR is_sent = FALSE)
    """,
        *[e.lower() for e in your_emails],
    )
    return _parse_update_count(result)


async def compute_replied_actions(conn: asyncpg.Connection) -> int:
    """Compute REPLIED action by matching sent emails to received via in_reply_to.

    Args:
        conn: Database connection.

    Returns:
        Number of emails marked as replied.
    """
    result = await conn.execute(
        """
        WITH replied_emails AS (
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
    """
    )
    return _parse_update_count(result)


async def compute_starred_actions(conn: asyncpg.Connection) -> int:
    """Mark STARRED emails based on labels.

    Args:
        conn: Database connection.

    Returns:
        Number of emails marked as starred.
    """
    result = await conn.execute(
        """
        UPDATE emails
        SET action = 'STARRED',
            enriched_at = NOW()
        WHERE is_sent = FALSE
          AND action IS NULL
          AND 'Starred' = ANY(labels)
    """
    )
    return _parse_update_count(result)


async def compute_archived_actions(conn: asyncpg.Connection, reference_date: datetime) -> int:
    """Mark ARCHIVED emails - read but not replied, older than threshold.

    Args:
        conn: Database connection.
        reference_date: Reference date for age calculations.

    Returns:
        Number of emails marked as archived.
    """
    cutoff_date = reference_date - timedelta(days=IGNORED_AGE_DAYS)

    result = await conn.execute(
        """
        UPDATE emails e
        SET action = 'ARCHIVED',
            timing = 'NEVER',
            enriched_at = NOW()
        WHERE e.is_sent = FALSE
          AND e.action IS NULL
          AND e.date_parsed < $1
          AND (
              'Archived' = ANY(e.labels)
              OR ('Unread' != ALL(e.labels) AND 'Inbox' != ALL(e.labels))
              OR (e.cc_emails IS NOT NULL AND array_length(e.cc_emails, 1) >= 1)
              OR (e.to_emails IS NOT NULL AND array_length(e.to_emails, 1) > 1)
              OR EXISTS (
                  SELECT 1 FROM users u
                  WHERE LOWER(u.email) = LOWER(e.from_email)
                    AND u.emails_from > 20
              )
          )
    """,
        cutoff_date,
    )
    return _parse_update_count(result)


async def compute_ignored_actions(conn: asyncpg.Connection, reference_date: datetime) -> int:
    """Mark IGNORED emails - truly ignored 1:1 emails.

    Args:
        conn: Database connection.
        reference_date: Reference date for age calculations.

    Returns:
        Number of emails marked as ignored.
    """
    cutoff_date = reference_date - timedelta(days=IGNORED_AGE_DAYS)

    result = await conn.execute(
        """
        UPDATE emails e
        SET action = 'IGNORED',
            timing = 'NEVER',
            enriched_at = NOW()
        WHERE e.is_sent = FALSE
          AND e.action IS NULL
          AND e.date_parsed < $1
          AND 'Unread' = ANY(e.labels)
          AND (e.cc_emails IS NULL OR array_length(e.cc_emails, 1) = 0)
          AND (e.to_emails IS NULL OR array_length(e.to_emails, 1) = 1)
          AND NOT EXISTS (
              SELECT 1 FROM users u
              WHERE LOWER(u.email) = LOWER(e.from_email)
                AND u.emails_from > 20
          )
    """,
        cutoff_date,
    )
    return _parse_update_count(result)


async def compute_pending_actions(conn: asyncpg.Connection) -> int:
    """Mark remaining recent emails as PENDING.

    Args:
        conn: Database connection.

    Returns:
        Number of emails marked as pending.
    """
    result = await conn.execute(
        """
        UPDATE emails
        SET action = 'PENDING',
            enriched_at = NOW()
        WHERE is_sent = FALSE
          AND action IS NULL
    """
    )
    return _parse_update_count(result)


async def mark_sent_action(conn: asyncpg.Connection) -> int:
    """Mark sent emails with COMPOSED action.

    Args:
        conn: Database connection.

    Returns:
        Number of emails marked as composed.
    """
    result = await conn.execute(
        """
        UPDATE emails
        SET action = 'COMPOSED',
            enriched_at = NOW()
        WHERE is_sent = TRUE
          AND action IS NULL
    """
    )
    return _parse_update_count(result)


async def compute_reverse_tracking(conn: asyncpg.Connection) -> int:
    """Track how quickly OTHERS respond to YOUR sent emails.

    Args:
        conn: Database connection.

    Returns:
        Number of sent emails with replies tracked.
    """
    result = await conn.execute(
        """
        WITH sent_replies AS (
            SELECT DISTINCT sent.id as sent_email_id,
                   received.date_parsed as reply_date,
                   sent.date_parsed as sent_date
            FROM emails sent
            JOIN emails received ON received.in_reply_to = sent.message_id
            WHERE sent.is_sent = TRUE
              AND received.is_sent = FALSE
              AND received.date_parsed > sent.date_parsed
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
    """
    )
    return _parse_update_count(result)


async def enrich_emails_async(conn: asyncpg.Connection, your_emails: list[str]) -> dict[str, Any]:
    """Run full enrichment pipeline on emails table.

    Args:
        conn: Database connection.
        your_emails: List of user's email addresses.

    Returns:
        Statistics dictionary with counts for each step.
    """
    stats: dict[str, Any] = {
        "total_emails": 0,
        "sent_marked": 0,
        "replied": 0,
        "starred": 0,
        "archived": 0,
        "ignored": 0,
        "pending": 0,
        "composed": 0,
        "reverse_tracked": 0,
    }

    # Get total count
    total = await conn.fetchval("SELECT COUNT(*) FROM emails")
    stats["total_emails"] = total if total else 0

    # Get reference date (latest email)
    reference_date = await conn.fetchval(
        """
        SELECT MAX(date_parsed) FROM emails WHERE date_parsed IS NOT NULL
    """
    )
    if reference_date is None:
        reference_date = datetime.now()

    # Step 1: Mark sent emails
    stats["sent_marked"] = await mark_sent_emails(conn, your_emails)

    # Step 2: Compute REPLIED actions
    stats["replied"] = await compute_replied_actions(conn)

    # Step 3: Mark STARRED
    stats["starred"] = await compute_starred_actions(conn)

    # Step 4: Mark ARCHIVED
    stats["archived"] = await compute_archived_actions(conn, reference_date)

    # Step 5: Mark IGNORED
    stats["ignored"] = await compute_ignored_actions(conn, reference_date)

    # Step 6: Mark PENDING
    stats["pending"] = await compute_pending_actions(conn)

    # Step 7: Mark COMPOSED for sent emails
    stats["composed"] = await mark_sent_action(conn)

    # Step 8: Reverse tracking
    stats["reverse_tracked"] = await compute_reverse_tracking(conn)

    return stats


def _convert_db_url(db_url: str) -> str:
    """Convert postgresql:// URL to postgres:// for asyncpg compatibility.

    Args:
        db_url: Database URL.

    Returns:
        URL with postgres:// prefix.
    """
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgres://", 1)
    return db_url


async def run_async(db_url: str, your_emails: list[str]) -> dict[str, Any]:
    """Async implementation of email enrichment.

    Args:
        db_url: Database URL.
        your_emails: List of user's email addresses.

    Returns:
        Statistics dictionary.
    """
    conn = await asyncpg.connect(db_url)
    try:
        return await enrich_emails_async(conn, your_emails)
    finally:
        await conn.close()


def run(config: Config) -> StageResult:
    """Run Stage 4: Enrich emails with action labels.

    Args:
        config: Application configuration with database_url and your_email.

    Returns:
        StageResult with enrichment statistics.
    """
    start_time = time.time()

    if not config.your_email:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="YOUR_EMAIL not configured",
        )

    your_emails = [config.your_email]

    db_url = _convert_db_url(config.database_url)
    stats = asyncio.run(run_async(db_url, your_emails))
    duration = time.time() - start_time

    total_enriched = (
        stats["replied"]
        + stats["starred"]
        + stats["archived"]
        + stats["ignored"]
        + stats["pending"]
        + stats["composed"]
    )

    return StageResult(
        success=True,
        records_processed=total_enriched,
        duration_seconds=duration,
        message=f"Enriched {total_enriched} emails with action labels",
        metadata=stats,
    )
