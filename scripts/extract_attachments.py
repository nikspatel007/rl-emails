#!/usr/bin/env python3
"""
Extract attachments from Gmail MBOX and import to PostgreSQL.

This script:
1. Parses the MBOX file to extract attachments
2. Saves attachment files to data/nik_gmail/attachments/
3. Inserts attachment records into PostgreSQL
4. Updates emails table with attachment metadata

Usage:
    python scripts/extract_attachments.py

Requirements:
    pip install asyncpg
"""

import asyncio
import hashlib
import mailbox
import mimetypes
import sys
from collections import defaultdict
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"
MBOX_PATH = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/takeout/extracted/All mail Including Spam and Trash.mbox")
ATTACHMENTS_DIR = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/attachments")
BATCH_SIZE = 100


def get_message_id(message) -> Optional[str]:
    """Extract Message-ID from email."""
    msg_id = message.get('Message-ID', '')
    if msg_id:
        return ' '.join(str(msg_id).split())
    return None


def compute_hash(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def get_safe_filename(filename: str, content_hash: str) -> str:
    """Generate a safe filename using hash prefix."""
    if not filename:
        filename = "unnamed"

    # Clean filename
    filename = filename.replace('/', '_').replace('\\', '_').replace('\x00', '')

    # Truncate if too long
    name_part = Path(filename).stem[:50]
    ext = Path(filename).suffix[:10]

    # Use hash prefix to avoid collisions
    return f"{content_hash[:12]}_{name_part}{ext}"


def extract_attachments_from_message(message) -> list[dict]:
    """Extract all attachments from an email message.

    Returns list of dicts with: filename, content_type, data, content_disposition
    """
    attachments = []

    if not message.is_multipart():
        return attachments

    for part in message.walk():
        content_disposition = part.get('Content-Disposition', '')
        content_type = part.get_content_type()

        # Skip main body parts
        if content_type in ('text/plain', 'text/html') and 'attachment' not in content_disposition:
            continue

        # Check if it's an attachment or inline content (like embedded images)
        is_attachment = 'attachment' in content_disposition
        is_inline = 'inline' in content_disposition
        has_filename = part.get_filename() is not None

        # Skip multipart containers
        if content_type.startswith('multipart/'):
            continue

        # Only process if it's an attachment, inline with filename, or has a binary content type
        if not (is_attachment or (is_inline and has_filename) or
                (has_filename and content_type not in ('text/plain', 'text/html'))):
            continue

        try:
            # Get payload
            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            filename = part.get_filename() or ''

            # Decode filename if encoded
            if filename:
                try:
                    from email.header import decode_header
                    decoded = decode_header(filename)
                    filename = ''.join(
                        part.decode(enc or 'utf-8') if isinstance(part, bytes) else part
                        for part, enc in decoded
                    )
                except Exception:
                    pass

            # Determine disposition type
            disp_type = 'attachment' if is_attachment else ('inline' if is_inline else 'attachment')

            attachments.append({
                'filename': filename,
                'content_type': content_type,
                'data': payload,
                'content_disposition': disp_type,
            })

        except Exception as e:
            print(f"  Warning: Failed to extract attachment: {e}", file=sys.stderr)

    return attachments


async def process_mbox(conn: asyncpg.Connection, mbox_path: Path, attachments_dir: Path):
    """Process MBOX file and extract all attachments."""

    attachments_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening MBOX: {mbox_path}")
    mbox = mailbox.mbox(str(mbox_path))

    # Track stats
    total_messages = 0
    messages_with_attachments = 0
    total_attachments = 0
    unique_attachments = 0
    skipped_no_match = 0

    # Track seen hashes to avoid storing duplicates
    seen_hashes = set()

    # Batch for updates
    attachment_records = []
    email_updates = defaultdict(lambda: {'count': 0, 'types': set(), 'total_bytes': 0})

    print("Processing messages...")

    for i, message in enumerate(mbox):
        total_messages += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} messages, found {total_attachments} attachments...")

        message_id = get_message_id(message)
        if not message_id:
            continue

        attachments = extract_attachments_from_message(message)
        if not attachments:
            continue

        # Check if this message exists in the database
        email_row = await conn.fetchrow(
            "SELECT id, raw_email_id, thread_id FROM emails WHERE message_id = $1",
            message_id
        )

        if not email_row:
            skipped_no_match += 1
            continue

        messages_with_attachments += 1
        email_id = email_row['id']
        raw_email_id = email_row['raw_email_id']
        thread_id = email_row['thread_id']

        for attachment in attachments:
            total_attachments += 1

            content_hash = compute_hash(attachment['data'])
            size_bytes = len(attachment['data'])

            # Track for email update
            email_updates[email_id]['count'] += 1
            email_updates[email_id]['types'].add(attachment['content_type'])
            email_updates[email_id]['total_bytes'] += size_bytes

            # Store file if not seen before
            stored_path = None
            if content_hash not in seen_hashes:
                safe_filename = get_safe_filename(attachment['filename'], content_hash)
                file_path = attachments_dir / safe_filename

                try:
                    file_path.write_bytes(attachment['data'])
                    stored_path = str(file_path.relative_to(attachments_dir.parent.parent))
                    seen_hashes.add(content_hash)
                    unique_attachments += 1
                except Exception as e:
                    print(f"  Warning: Failed to write {safe_filename}: {e}", file=sys.stderr)
            else:
                # File already stored, find the path
                stored_path = f"attachments/{get_safe_filename(attachment['filename'], content_hash)}"

            attachment_records.append({
                'raw_email_id': raw_email_id,
                'email_id': email_id,
                'filename': attachment['filename'] or None,
                'content_type': attachment['content_type'],
                'size_bytes': size_bytes,
                'content_disposition': attachment['content_disposition'],
                'content_hash': content_hash,
                'stored_path': stored_path,
            })

        # Batch insert attachments every BATCH_SIZE emails
        if len(attachment_records) >= BATCH_SIZE * 5:
            await insert_attachments(conn, attachment_records)
            attachment_records = []

    # Insert remaining attachments
    if attachment_records:
        await insert_attachments(conn, attachment_records)

    # Update emails table
    print(f"\nUpdating {len(email_updates)} emails with attachment metadata...")
    await update_emails(conn, email_updates)

    print(f"\n=== Summary ===")
    print(f"Total messages processed: {total_messages}")
    print(f"Messages with attachments: {messages_with_attachments}")
    print(f"Skipped (not in DB): {skipped_no_match}")
    print(f"Total attachments found: {total_attachments}")
    print(f"Unique attachments stored: {unique_attachments}")


async def insert_attachments(conn: asyncpg.Connection, records: list[dict]):
    """Batch insert attachment records."""
    if not records:
        return

    await conn.executemany("""
        INSERT INTO attachments (
            raw_email_id, email_id, filename, content_type,
            size_bytes, content_disposition, content_hash, stored_path
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT DO NOTHING
    """, [
        (r['raw_email_id'], r['email_id'], r['filename'], r['content_type'],
         r['size_bytes'], r['content_disposition'], r['content_hash'], r['stored_path'])
        for r in records
    ])


async def update_emails(conn: asyncpg.Connection, email_updates: dict):
    """Update emails table with attachment metadata."""

    for email_id, data in email_updates.items():
        await conn.execute("""
            UPDATE emails
            SET has_attachments = TRUE,
                attachment_count = $2,
                attachment_types = $3,
                total_attachment_bytes = $4
            WHERE id = $1
        """,
            email_id,
            data['count'],
            list(data['types']),
            data['total_bytes']
        )


async def verify_results(conn: asyncpg.Connection):
    """Verify the extraction results."""
    print("\n=== Verification ===")

    # Count attachments
    attach_count = await conn.fetchval("SELECT COUNT(*) FROM attachments")
    print(f"Total attachments in DB: {attach_count}")

    # Emails with attachments
    emails_with = await conn.fetchval(
        "SELECT COUNT(*) FROM emails WHERE has_attachments = TRUE"
    )
    print(f"Emails with attachments: {emails_with}")

    # Top content types
    print("\nTop content types:")
    types = await conn.fetch("""
        SELECT content_type, COUNT(*) as cnt, SUM(size_bytes) as total_bytes
        FROM attachments
        GROUP BY content_type
        ORDER BY cnt DESC
        LIMIT 10
    """)
    for row in types:
        mb = row['total_bytes'] / (1024 * 1024) if row['total_bytes'] else 0
        print(f"  {row['content_type']}: {row['cnt']} ({mb:.1f} MB)")

    # Total storage used
    total_bytes = await conn.fetchval("SELECT SUM(size_bytes) FROM attachments")
    total_mb = total_bytes / (1024 * 1024) if total_bytes else 0
    print(f"\nTotal attachment storage: {total_mb:.1f} MB")


async def main():
    """Main entry point."""
    print("Extract Attachments from Gmail MBOX")
    print("=" * 40)

    if not MBOX_PATH.exists():
        print(f"Error: MBOX file not found: {MBOX_PATH}")
        sys.exit(1)

    print(f"\nConnecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Check if attachments table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'attachments'
            )
        """)

        if not table_exists:
            print("Error: attachments table does not exist. Run import_to_postgres.py first.")
            sys.exit(1)

        # Process MBOX
        await process_mbox(conn, MBOX_PATH, ATTACHMENTS_DIR)

        # Verify
        await verify_results(conn)

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
