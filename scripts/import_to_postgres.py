#!/usr/bin/env python3
"""
Stage 6: Import enriched emails to PostgreSQL.

Reads enriched_emails.jsonl and imports to:
- raw_emails (immutable source data)
- emails (enriched/derived data)

Usage:
    python scripts/import_to_postgres.py

Requirements:
    pip install asyncpg
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg

# Configuration - read from environment variables with fallbacks
DB_URL = os.environ.get("DB_URL", "postgresql://postgres:postgres@localhost:5433/rl_emails")
# Try PARSED_JSONL first (direct from parse), then ENRICHED_JSONL, then default
JSONL_PATH = Path(os.environ.get("PARSED_JSONL", os.environ.get("ENRICHED_JSONL", "/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/parsed_emails.jsonl")))
BATCH_SIZE = 1000

# Email parsing regex
EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
EMAIL_WITH_NAME_PATTERN = re.compile(r'"?([^"<]*)"?\s*<([^>]+)>|([^\s,<]+@[^\s,>]+)')


def parse_email_address(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Extract email and name from a raw address like 'Name <email>' or just 'email'."""
    if not raw:
        return None, None

    raw = raw.strip()

    # Try "Name <email>" format
    match = re.match(r'"?([^"<]*)"?\s*<([^>]+)>', raw)
    if match:
        name = match.group(1).strip() or None
        email = match.group(2).strip().lower()
        return email, name

    # Just an email address
    if '@' in raw:
        email = raw.strip().lower()
        # Remove quotes if present
        email = email.strip('"\'')
        return email, None

    return None, None


def extract_all_emails(raw: str) -> list[str]:
    """Extract all email addresses from a raw header (to, cc, etc.)."""
    if not raw:
        return []

    emails = []
    # Split by comma and process each
    for part in raw.split(','):
        email, _ = parse_email_address(part)
        if email:
            emails.append(email)

    return emails


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse ISO date string to datetime."""
    if not date_str:
        return None
    try:
        # Handle ISO format with Z suffix
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


def generate_preview(body: str, max_length: int = 500) -> str:
    """Generate a preview of the body text."""
    if not body:
        return ""

    # Strip whitespace and limit length
    preview = ' '.join(body.split())[:max_length]
    if len(body) > max_length:
        preview += "..."
    return preview


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def is_sent_email(from_email: Optional[str], labels: list[str]) -> bool:
    """Determine if this is a sent email."""
    if 'SENT' in labels:
        return True
    if from_email and any(addr in from_email for addr in ['me@nik-patel.com', 'nik@', 'nikpatel']):
        return True
    return False


async def create_schema(conn: asyncpg.Connection, schema_path: Path):
    """Execute schema creation SQL."""
    schema_sql = schema_path.read_text()
    await conn.execute(schema_sql)
    print("Schema created successfully")


async def import_emails(conn: asyncpg.Connection, jsonl_path: Path):
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

    # Process in batches
    imported = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch = emails[batch_start:batch_start + BATCH_SIZE]

        async with conn.transaction():
            for email in batch:
                # Parse fields
                from_email, from_name = parse_email_address(email.get('from', ''))
                to_emails = extract_all_emails(email.get('to', ''))
                cc_emails = extract_all_emails(email.get('cc', ''))
                labels = email.get('labels', [])
                date_parsed = parse_date(email.get('date'))
                body = email.get('body', '')

                # Insert into raw_emails
                raw_id = await conn.fetchval("""
                    INSERT INTO raw_emails (
                        message_id, thread_id, date_raw, from_raw, to_raw, cc_raw,
                        subject_raw, body_text, labels_raw
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (message_id) DO NOTHING
                    RETURNING id
                """,
                    email.get('message_id'),
                    email.get('thread_id'),
                    email.get('date_original'),
                    email.get('from'),
                    email.get('to'),
                    email.get('cc'),
                    email.get('subject'),
                    body,
                    ','.join(labels) if labels else None
                )

                if raw_id is None:
                    # Already exists, skip
                    continue

                # Insert into emails (enriched)
                await conn.execute("""
                    INSERT INTO emails (
                        raw_email_id, message_id, thread_id, date_parsed,
                        from_email, from_name, to_emails, cc_emails, subject,
                        body_text, body_preview, word_count, labels,
                        has_attachments, is_sent, action, timing,
                        response_time_seconds, enriched_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, NOW()
                    )
                    ON CONFLICT (message_id) DO NOTHING
                """,
                    raw_id,
                    email.get('message_id'),
                    email.get('thread_id'),
                    date_parsed,
                    from_email,
                    from_name,
                    to_emails if to_emails else None,
                    cc_emails if cc_emails else None,
                    email.get('subject'),
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


async def verify_import(conn: asyncpg.Connection):
    """Verify the import was successful."""
    print("\n=== Verification ===")

    # Count emails
    raw_count = await conn.fetchval("SELECT COUNT(*) FROM raw_emails")
    emails_count = await conn.fetchval("SELECT COUNT(*) FROM emails")
    print(f"raw_emails: {raw_count}")
    print(f"emails: {emails_count}")

    # Top labels
    print("\nTop labels:")
    labels = await conn.fetch("""
        SELECT unnest(labels) as label, COUNT(*) as cnt
        FROM emails
        WHERE labels IS NOT NULL
        GROUP BY label
        ORDER BY cnt DESC
        LIMIT 10
    """)
    for row in labels:
        print(f"  {row['label']}: {row['cnt']}")

    # Action distribution
    print("\nAction distribution:")
    actions = await conn.fetch("""
        SELECT action, COUNT(*) as cnt
        FROM emails
        GROUP BY action
        ORDER BY cnt DESC
    """)
    for row in actions:
        print(f"  {row['action'] or 'NULL'}: {row['cnt']}")

    # Date range
    date_range = await conn.fetchrow("""
        SELECT MIN(date_parsed) as earliest, MAX(date_parsed) as latest
        FROM emails
        WHERE date_parsed IS NOT NULL
    """)
    print(f"\nDate range: {date_range['earliest']} to {date_range['latest']}")


async def main():
    """Main entry point."""
    print("Stage 6: Import to PostgreSQL")
    print("=" * 40)

    # Get schema path
    script_dir = Path(__file__).parent
    schema_path = script_dir / "create_schema.sql"

    if not schema_path.exists():
        print(f"Error: Schema file not found: {schema_path}")
        sys.exit(1)

    if not JSONL_PATH.exists():
        print(f"Error: JSONL file not found: {JSONL_PATH}")
        sys.exit(1)

    # Connect to database
    print(f"\nConnecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Create schema
        print("\nCreating schema...")
        await create_schema(conn, schema_path)

        # Import emails
        print("\nImporting emails...")
        await import_emails(conn, JSONL_PATH)

        # Verify
        await verify_import(conn)

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
