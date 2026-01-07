"""Stage 8: Populate users table from email addresses.

Extracts unique users from from_email, to_emails[], cc_emails[] and computes:
- emails_from: count of emails sent by this user
- emails_to: count of emails received (to + cc)
- threads_with: distinct threads they're involved in
- reply_count: emails they sent that are replies
- avg_response_time_seconds: average response time for their replies
- reply_rate: fraction of received emails they replied to
- first_contact, last_contact: date bounds
- is_you: true for your email variants
- is_important_sender: high engagement users
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from psycopg2.extras import execute_values

from rl_emails.core.config import Config
from rl_emails.core.db import get_connection
from rl_emails.pipeline.stages.base import StageResult

if TYPE_CHECKING:  # pragma: no cover
    import psycopg2


def is_your_email(email: str, your_email: str) -> bool:
    """Check if email belongs to the user.

    Args:
        email: Email address to check.
        your_email: User's configured email address.

    Returns:
        True if email belongs to the user.
    """
    if not your_email:
        return False

    email_lower = email.lower()
    your_email_lower = your_email.lower()

    if email_lower == your_email_lower:
        return True

    # Check domain match with base name
    if "@" in your_email_lower:
        you_base = your_email_lower.split("@")[0]
        you_domain = your_email_lower.split("@")[1]
        if email_lower.endswith("@" + you_domain) and you_base in email_lower:
            return True

    return False


def populate_users_from_emails(
    conn: psycopg2.extensions.connection, your_email: str
) -> tuple[int, dict[str, int]]:
    """Extract unique users and populate the users table.

    Args:
        conn: Database connection.
        your_email: User's email address for is_you detection.

    Returns:
        Tuple of (user_count, stats_dict).
    """
    cur = conn.cursor()
    stats: dict[str, int] = {}

    # Clear existing users
    cur.execute("TRUNCATE users RESTART IDENTITY CASCADE")

    # Step 1: Extract all unique email addresses with their names
    cur.execute(
        """
        WITH all_emails AS (
            SELECT LOWER(TRIM(from_email)) as email, from_name as name
            FROM emails
            WHERE from_email IS NOT NULL AND from_email != ''
            UNION ALL
            SELECT LOWER(TRIM(unnest(to_emails))) as email, NULL as name
            FROM emails
            WHERE to_emails IS NOT NULL
            UNION ALL
            SELECT LOWER(TRIM(unnest(cc_emails))) as email, NULL as name
            FROM emails
            WHERE cc_emails IS NOT NULL
        ),
        unique_emails AS (
            SELECT email, MAX(name) as name
            FROM all_emails
            WHERE email IS NOT NULL AND email != '' AND email LIKE '%@%'
            GROUP BY email
        )
        SELECT email, name FROM unique_emails ORDER BY email
    """
    )

    users = cur.fetchall()
    stats["unique_emails"] = len(users)

    # Step 2: Insert users with basic info
    user_data = []
    for email, name in users:
        is_you = is_your_email(email, your_email)
        user_data.append((email, name, is_you))

    if user_data:
        execute_values(
            cur,
            """
            INSERT INTO users (email, name, is_you)
            VALUES %s
            ON CONFLICT (email) DO NOTHING
            """,
            user_data,
        )

    # Step 3: Compute emails_from (emails sent by this user)
    cur.execute(
        """
        UPDATE users u
        SET emails_from = sub.cnt
        FROM (
            SELECT LOWER(TRIM(from_email)) as email, COUNT(*) as cnt
            FROM emails
            WHERE from_email IS NOT NULL
            GROUP BY LOWER(TRIM(from_email))
        ) sub
        WHERE u.email = sub.email
    """
    )

    # Step 4: Compute emails_to (emails received - to or cc)
    cur.execute(
        """
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
    """
    )

    # Step 5: Compute threads_with (distinct threads involving this user)
    cur.execute(
        """
        WITH user_threads AS (
            SELECT LOWER(TRIM(from_email)) as email, thread_id
            FROM emails
            WHERE from_email IS NOT NULL AND thread_id IS NOT NULL
            UNION
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
    """
    )

    # Step 6: Compute reply_count (emails they sent that are replies)
    cur.execute(
        """
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
    """
    )

    # Step 7: Compute avg_response_time_seconds
    cur.execute(
        """
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
    """
    )

    # Step 8: Compute reply_rate
    cur.execute(
        """
        UPDATE users
        SET reply_rate = LEAST(1.0,
            CASE
                WHEN emails_to > 0 THEN reply_count::float / emails_to
                ELSE 0
            END
        )
        WHERE emails_to > 0 OR reply_count > 0
    """
    )

    # Step 9: Mark important senders
    cur.execute(
        """
        UPDATE users
        SET is_important_sender = true
        WHERE NOT is_you
          AND emails_from > 0
          AND (
              (reply_rate >= 0.5 AND emails_to >= 2)
              OR (emails_from >= 3 AND emails_to >= 3)
              OR reply_rate >= 0.8
          )
    """
    )

    # Step 10: Compute first_contact and last_contact
    cur.execute(
        """
        WITH contact_dates AS (
            SELECT LOWER(TRIM(from_email)) as email, date_parsed as contact_date
            FROM emails
            WHERE from_email IS NOT NULL AND date_parsed IS NOT NULL
            UNION ALL
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
    """
    )

    conn.commit()

    # Get final count and stats
    cur.execute("SELECT COUNT(*) FROM users")
    row = cur.fetchone()
    count: int = row[0] if row else 0

    cur.execute("SELECT COUNT(*) FROM users WHERE is_you = true")
    row = cur.fetchone()
    stats["is_you_count"] = row[0] if row else 0

    cur.execute("SELECT COUNT(*) FROM users WHERE is_important_sender = true")
    row = cur.fetchone()
    stats["important_sender_count"] = row[0] if row else 0

    cur.close()
    return count, stats


def run(config: Config) -> StageResult:
    """Run Stage 8: Populate users table from email addresses.

    Args:
        config: Application configuration with database_url and your_email.

    Returns:
        StageResult with user population counts.
    """
    start_time = time.time()

    your_email = config.your_email or ""

    with get_connection(config.database_url) as conn:
        count, stats = populate_users_from_emails(conn, your_email)

    duration = time.time() - start_time

    return StageResult(
        success=True,
        records_processed=count,
        duration_seconds=duration,
        message=f"Populated {count} users",
        metadata={
            "unique_emails": stats.get("unique_emails", 0),
            "is_you_count": stats.get("is_you_count", 0),
            "important_sender_count": stats.get("important_sender_count", 0),
        },
    )
