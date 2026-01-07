"""Stage 2: Import parsed email JSONL to PostgreSQL.

Reads JSONL from Stage 1 and imports to raw_emails + emails tables.
Handles email parsing, date normalization, and text sanitization.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult


def sanitize_text(text: str | None) -> str | None:
    """Remove null bytes and normalize line endings.

    Args:
        text: Raw text to sanitize.

    Returns:
        Sanitized text or None.
    """
    if text is None:
        return None
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def parse_email_address(raw: str | None) -> tuple[str | None, str | None]:
    """Extract email and name from raw address string.

    Handles formats like:
    - "Name" <email@example.com>
    - Name <email@example.com>
    - email@example.com

    Args:
        raw: Raw email address string.

    Returns:
        Tuple of (email, name).
    """
    if not raw:
        return None, None

    raw = sanitize_text(raw) or ""

    # Pattern: "Name" <email@example.com> or Name <email@example.com>
    match = re.search(r"([^<]*)<([^>]+)>", raw)
    if match:
        name = match.group(1).strip().strip("\"'")
        email = match.group(2).strip().lower()
        return email, name if name else None

    # Just an email address
    if "@" in raw:
        return raw.strip().lower(), None

    return None, None


def extract_all_emails(raw: str | None) -> list[str]:
    """Extract all email addresses from comma/semicolon separated string.

    Args:
        raw: Raw string with email addresses.

    Returns:
        List of extracted email addresses.
    """
    if not raw:
        return []

    raw = sanitize_text(raw) or ""
    emails = []

    parts = re.split(r"[,;]", raw)
    for part in parts:
        email, _ = parse_email_address(part.strip())
        if email:
            emails.append(email)

    return emails


def parse_date(date_str: str | None) -> datetime | None:
    """Parse date string to datetime, handling various formats.

    Args:
        date_str: Raw date string.

    Returns:
        Parsed datetime or None.
    """
    if not date_str:
        return None

    date_str = sanitize_text(date_str) or ""

    # Common date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822
        "%d %b %Y %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]

    # Clean up timezone notation
    date_str = re.sub(r"\s*\([^)]+\)\s*$", "", date_str)  # Remove (PST) etc
    date_str = re.sub(r"\s+", " ", date_str).strip()

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def generate_preview(body: str | None, max_length: int = 200) -> str | None:
    """Generate a preview from body text.

    Args:
        body: Full body text.
        max_length: Maximum preview length.

    Returns:
        Preview string or None.
    """
    if not body:
        return None

    preview = " ".join(body.split())
    if len(preview) > max_length:
        preview = preview[:max_length] + "..."
    return preview


def count_words(text: str | None) -> int:
    """Count words in text.

    Args:
        text: Text to count words in.

    Returns:
        Word count.
    """
    if not text:
        return 0
    return len(text.split())


def is_sent_email(labels: list[str] | None) -> bool:
    """Determine if email was sent by user based on labels.

    Args:
        labels: List of email labels.

    Returns:
        True if email was sent by user.
    """
    if not labels:
        return False
    return "SENT" in labels or "Sent" in [label.title() for label in labels]


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


async def import_emails_async(
    conn: asyncpg.Connection,
    jsonl_path: Path,
    batch_size: int = 500,
) -> tuple[int, dict[str, Any]]:
    """Import emails from JSONL to database.

    Args:
        conn: Database connection.
        jsonl_path: Path to JSONL file from Stage 1.
        batch_size: Number of emails per batch.

    Returns:
        Tuple of (imported_count, stats_dict).
    """
    stats: dict[str, Any] = {"raw_inserted": 0, "emails_inserted": 0, "skipped": 0}

    # Read all emails
    emails = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                emails.append(json.loads(line))

    total = len(emails)
    stats["total_in_file"] = total

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch = emails[batch_start : batch_start + batch_size]

        for email in batch:
            from_email, from_name = parse_email_address(email.get("from", ""))
            to_emails = extract_all_emails(email.get("to", ""))
            cc_emails = extract_all_emails(email.get("cc", ""))
            labels = email.get("labels", [])
            date_parsed = parse_date(email.get("date"))
            body = sanitize_text(email.get("body", ""))
            subject = sanitize_text(email.get("subject", ""))

            # Insert into raw_emails
            raw_id = await conn.fetchval(
                """
                INSERT INTO raw_emails (
                    message_id, thread_id, in_reply_to, references_raw,
                    date_raw, from_raw, to_raw, cc_raw, bcc_raw,
                    subject_raw, body_text, body_html, labels_raw,
                    mbox_offset, raw_size_bytes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (message_id) DO NOTHING
                RETURNING id
                """,
                email.get("message_id"),
                email.get("thread_id"),
                sanitize_text(email.get("in_reply_to")) or None,
                sanitize_text(email.get("references")) or None,
                email.get("date") or email.get("date_original"),
                email.get("from"),
                email.get("to"),
                email.get("cc"),
                email.get("bcc") or None,
                email.get("subject"),
                body,
                email.get("body_html") or None,
                ",".join(labels) if labels else None,
                email.get("mbox_offset"),
                email.get("raw_size_bytes"),
            )

            if raw_id is None:
                stats["skipped"] += 1
                continue

            stats["raw_inserted"] += 1

            # Insert into emails (enriched)
            await conn.execute(
                """
                INSERT INTO emails (
                    raw_email_id, message_id, thread_id, in_reply_to, date_parsed,
                    from_email, from_name, to_emails, cc_emails, subject,
                    body_text, body_preview, word_count, labels,
                    has_attachments, is_sent, action, timing,
                    response_time_seconds, enriched_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                    $15, $16, $17, $18, $19, NOW()
                )
                ON CONFLICT (message_id) DO NOTHING
                """,
                raw_id,
                email.get("message_id"),
                email.get("thread_id"),
                sanitize_text(email.get("in_reply_to")) or None,
                date_parsed,
                from_email,
                from_name,
                to_emails if to_emails else None,
                cc_emails if cc_emails else None,
                subject,
                body,
                generate_preview(body),
                count_words(body),
                labels if labels else None,
                email.get("has_attachments", False),
                is_sent_email(labels),
                email.get("action"),
                email.get("response_timing"),
                email.get("response_time_seconds"),
            )

            stats["emails_inserted"] += 1

    return stats["emails_inserted"], stats


async def run_async(
    db_url: str, jsonl_path: Path, batch_size: int = 500
) -> tuple[int, dict[str, Any]]:
    """Async implementation of email import.

    Args:
        db_url: Database URL in asyncpg format.
        jsonl_path: Path to JSONL file.
        batch_size: Number of emails per batch.

    Returns:
        Tuple of (imported_count, stats_dict).
    """
    conn = await asyncpg.connect(db_url)
    try:
        return await import_emails_async(conn, jsonl_path, batch_size)
    finally:
        await conn.close()


def run(config: Config, *, batch_size: int = 500) -> StageResult:
    """Run Stage 2: Import JSONL to PostgreSQL.

    Args:
        config: Application configuration with database_url and parsed_jsonl.
        batch_size: Number of emails per batch.

    Returns:
        StageResult with import statistics.
    """
    start_time = time.time()

    if not config.parsed_jsonl:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="PARSED_JSONL not configured",
        )

    if not config.parsed_jsonl.exists():
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message=f"JSONL file not found: {config.parsed_jsonl}",
        )

    db_url = convert_db_url(config.database_url)
    count, stats = asyncio.run(run_async(db_url, config.parsed_jsonl, batch_size))
    duration = time.time() - start_time

    return StageResult(
        success=True,
        records_processed=count,
        duration_seconds=duration,
        message=f"Imported {count} emails to database",
        metadata=stats,
    )
