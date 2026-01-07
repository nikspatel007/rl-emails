"""Import parsed email data into PostgreSQL database."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Batch size for processing
BATCH_SIZE = 500


def sanitize_text(text: str | None) -> str | None:
    """Remove null bytes and other problematic characters from text."""
    if text is None:
        return None
    # Remove null bytes
    text = text.replace('\x00', '')
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


def parse_email_address(raw: str) -> tuple[str | None, str | None]:
    """Extract email and name from a raw address string."""
    if not raw:
        return None, None

    raw = sanitize_text(raw) or ""

    # Pattern: "Name" <email@example.com> or Name <email@example.com>
    match = re.search(r'([^<]*)<([^>]+)>', raw)
    if match:
        name = match.group(1).strip().strip('"\'')
        email = match.group(2).strip().lower()
        return email, name if name else None

    # Just an email address
    if '@' in raw:
        return raw.strip().lower(), None

    return None, None


def extract_all_emails(raw: str) -> list[str]:
    """Extract all email addresses from a comma/semicolon separated string."""
    if not raw:
        return []

    raw = sanitize_text(raw) or ""
    emails = []

    # Split by common delimiters
    parts = re.split(r'[,;]', raw)
    for part in parts:
        email, _ = parse_email_address(part.strip())
        if email:
            emails.append(email)

    return emails


def parse_date(date_str: str | None) -> datetime | None:
    """Parse date string to datetime, handling various formats."""
    if not date_str:
        return None

    date_str = sanitize_text(date_str) or ""

    # Common date formats
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',  # RFC 2822
        '%d %b %Y %H:%M:%S %z',
        '%Y-%m-%d %H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
    ]

    # Clean up timezone notation
    date_str = re.sub(r'\s*\([^)]+\)\s*$', '', date_str)  # Remove (PST) etc
    date_str = re.sub(r'\s+', ' ', date_str).strip()

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def generate_preview(body: str | None, max_length: int = 200) -> str | None:
    """Generate a preview from body text."""
    if not body:
        return None

    # Remove excessive whitespace
    preview = ' '.join(body.split())
    if len(preview) > max_length:
        preview = preview[:max_length] + '...'
    return preview


def count_words(text: str | None) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def is_sent_email(from_email: str | None, labels: list[str]) -> bool:
    """Determine if email was sent by user."""
    if not labels:
        return False
    return 'SENT' in labels or 'Sent' in [l.title() for l in labels]


async def create_schema(conn: asyncpg.Connection, schema_path: Path) -> None:
    """Execute schema creation SQL."""
    schema_sql = schema_path.read_text()
    await conn.execute(schema_sql)
    print("Schema created successfully")


async def import_emails(conn: asyncpg.Connection, jsonl_path: Path) -> int:
    """Import emails from JSONL to database."""

    # Read all emails
    print(f"Reading {jsonl_path}...")
    emails = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                emails.append(json.loads(line))

    total = len(emails)
    print(f"Loaded {total} emails")

    # Process in batches (no transaction wrapper - autocommit each insert for reliability)
    imported = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch = emails[batch_start:batch_start + BATCH_SIZE]

        for email in batch:
            # Parse fields
            from_email, from_name = parse_email_address(email.get('from', ''))
            to_emails = extract_all_emails(email.get('to', ''))
            cc_emails = extract_all_emails(email.get('cc', ''))
            labels = email.get('labels', [])
            date_parsed = parse_date(email.get('date'))
            body = sanitize_text(email.get('body', ''))
            subject = sanitize_text(email.get('subject', ''))

            # Insert into raw_emails
            raw_id = await conn.fetchval("""
                INSERT INTO raw_emails (
                    message_id, thread_id, in_reply_to, references_raw,
                    date_raw, from_raw, to_raw, cc_raw, bcc_raw,
                    subject_raw, body_text, body_html, labels_raw,
                    mbox_offset, raw_size_bytes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (message_id) DO NOTHING
                RETURNING id
            """,
                email.get('message_id'),
                email.get('thread_id'),
                sanitize_text(email.get('in_reply_to')) or None,
                sanitize_text(email.get('references')) or None,
                email.get('date') or email.get('date_original'),
                email.get('from'),
                email.get('to'),
                email.get('cc'),
                email.get('bcc') or None,
                email.get('subject'),
                body,
                email.get('body_html') or None,
                ','.join(labels) if labels else None,
                email.get('mbox_offset'),
                email.get('raw_size_bytes'),
            )

            if raw_id is None:
                # Already exists, skip
                continue

            # Insert into emails (enriched)
            await conn.execute("""
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
                email.get('message_id'),
                email.get('thread_id'),
                sanitize_text(email.get('in_reply_to')) or None,
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
                email.get('has_attachments', False),
                is_sent_email(from_email, labels),
                email.get('action'),
                email.get('response_timing'),
                email.get('response_time_seconds')
            )

            imported += 1

        print(f"  Imported {min(batch_start + BATCH_SIZE, total)}/{total} emails...")

    print(f"\nImport complete: {imported} new emails")
    return imported


async def verify_import(conn: asyncpg.Connection) -> None:
    """Verify import by checking counts and sample data."""
    raw_count = await conn.fetchval("SELECT COUNT(*) FROM raw_emails")
    email_count = await conn.fetchval("SELECT COUNT(*) FROM emails")

    print(f"\nVerification:")
    print(f"  Raw emails: {raw_count}")
    print(f"  Enriched emails: {email_count}")

    # Sample some data
    sample = await conn.fetch("""
        SELECT message_id, subject, from_email, date_parsed
        FROM emails
        ORDER BY date_parsed DESC
        LIMIT 5
    """)

    print(f"\nRecent emails:")
    for row in sample:
        print(f"  {row['date_parsed']}: {row['from_email']} - {row['subject'][:50] if row['subject'] else '(no subject)'}...")


async def main() -> None:
    from urllib.parse import urlparse

    # Required environment variables
    database_url = os.environ.get('DATABASE_URL')
    parsed_jsonl = os.environ.get('PARSED_JSONL')

    if not database_url:
        print("ERROR: DATABASE_URL environment variable is required")
        sys.exit(1)
    if not parsed_jsonl:
        print("ERROR: PARSED_JSONL environment variable is required")
        sys.exit(1)

    # Parse DATABASE_URL
    parsed = urlparse(database_url)

    # Connect to database
    conn = await asyncpg.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        database=parsed.path.lstrip('/')
    )

    try:
        await import_emails(conn, Path(parsed_jsonl))
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
