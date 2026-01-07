#!/usr/bin/env python3
"""Parse Gmail MBOX export to normalized JSONL format.

Extracts all metadata and body content from Gmail Takeout MBOX files.
Handles multipart messages, label parsing, and attachment detection.
"""

from __future__ import annotations

import json
import mailbox
import os
import re
from collections import Counter
from datetime import datetime
from email.header import decode_header
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Required environment variables
_mbox_path = os.environ.get("MBOX_PATH")
_output_path = os.environ.get("PARSED_JSONL")

if not _mbox_path:
    print("ERROR: MBOX_PATH environment variable is required")
    exit(1)
if not _output_path:
    print("ERROR: PARSED_JSONL environment variable is required")
    exit(1)

MBOX_PATH = Path(_mbox_path)
OUTPUT_PATH = Path(_output_path)
REPORT_PATH = OUTPUT_PATH.parent / "parse_report.json"


def decode_header_value(value: str | None) -> str:
    """Decode RFC 2047 encoded header values."""
    if not value:
        return ""

    try:
        decoded_parts = decode_header(value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                try:
                    result.append(part.decode(charset or 'utf-8', errors='replace'))
                except (LookupError, UnicodeDecodeError):
                    result.append(part.decode('utf-8', errors='replace'))
            else:
                result.append(part)
        return ''.join(result)
    except Exception:
        return str(value)


def parse_labels(labels_str: str | None) -> list[str]:
    """Parse Gmail labels from X-Gmail-Labels header.

    Gmail labels can contain commas in the label name (e.g., [Superhuman]/AI/Marketing).
    Labels are comma-separated but quoted if they contain special chars.
    Handles RFC 2822 header folding (embedded \r\n).
    """
    if not labels_str:
        return []

    # Normalize header folding - remove \r\n and collapse whitespace
    labels_str = re.sub(r'\r\n\s*', ' ', labels_str)

    labels = []
    current = ""
    in_quotes = False

    for char in labels_str:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            label = current.strip().strip('"')
            if label:
                labels.append(label)
            current = ""
        else:
            current += char

    # Don't forget the last label
    label = current.strip().strip('"')
    if label:
        labels.append(label)

    return labels


def _decode_payload(payload: bytes, charset: str | None) -> str:
    """Decode payload bytes to string with charset fallback."""
    encoding = charset or "utf-8"
    try:
        return payload.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def get_body(msg: mailbox.mboxMessage) -> tuple[str, str]:
    """Extract body text and HTML from an email message.

    Handles multipart messages, extracting both plain text and HTML separately.

    Returns:
        Tuple of (body_text, body_html)
    """
    text_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        charset = part.get_content_charset()
                        text_parts.append(_decode_payload(payload, charset))
                except Exception:
                    pass
            elif content_type == "text/html":
                try:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        charset = part.get_content_charset()
                        html_parts.append(_decode_payload(payload, charset))
                except Exception:
                    pass
    else:
        content_type = msg.get_content_type()
        try:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                charset = msg.get_content_charset()
                decoded = _decode_payload(payload, charset)

                if content_type == "text/html":
                    html_parts.append(decoded)
                else:
                    text_parts.append(decoded)
            else:
                str_payload = msg.get_payload()
                if isinstance(str_payload, str):
                    if content_type == "text/html":
                        html_parts.append(str_payload)
                    else:
                        text_parts.append(str_payload)
        except Exception:
            pass

    body_text = "\n".join(text_parts)
    body_html = "\n".join(html_parts)

    # If no plain text but we have HTML, use HTML as fallback for text
    if not body_text and body_html:
        body_text = body_html

    return body_text, body_html


def has_attachments(msg: mailbox.mboxMessage) -> bool:
    """Check if message has any attachments."""
    if not msg.is_multipart():
        return False

    for part in msg.walk():
        content_disposition = str(part.get("Content-Disposition", ""))
        if "attachment" in content_disposition:
            return True

        # Also check for inline images and other non-text parts
        content_type = part.get_content_type()
        if content_type and not content_type.startswith(("text/", "multipart/")):
            if part.get_filename():
                return True

    return False


def parse_email(
    msg: mailbox.mboxMessage, mbox_offset: int = 0, raw_size_bytes: int = 0
) -> dict[str, Any]:
    """Parse a single email message to normalized dict.

    Args:
        msg: The mailbox message to parse
        mbox_offset: Byte offset in mbox file where message starts
        raw_size_bytes: Size of raw message in bytes
    """
    body_text, body_html = get_body(msg)

    return {
        'message_id': msg.get('Message-ID', ''),
        'thread_id': msg.get('X-GM-THRID', ''),
        'in_reply_to': msg.get('In-Reply-To', ''),
        'references': msg.get('References', ''),
        'labels': parse_labels(msg.get('X-Gmail-Labels', '')),
        'date': msg.get('Date', ''),
        'from': decode_header_value(msg.get('From', '')),
        'to': decode_header_value(msg.get('To', '')),
        'cc': decode_header_value(msg.get('Cc', '')),
        'bcc': decode_header_value(msg.get('Bcc', '')),
        'subject': decode_header_value(msg.get('Subject', '')),
        'body': body_text,
        'body_html': body_html if body_html else None,
        'has_attachments': has_attachments(msg),
        'mbox_offset': mbox_offset,
        'raw_size_bytes': raw_size_bytes,
    }


def parse_mbox() -> dict[str, Any]:
    """Parse the entire MBOX file and write to JSONL."""
    print(f"Opening MBOX: {MBOX_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    mbox = mailbox.mbox(str(MBOX_PATH))

    # Track stats during parsing
    total_emails = 0
    emails_with_body = 0
    emails_with_body_html = 0
    emails_with_attachments = 0
    emails_with_message_id = 0
    emails_with_thread_id = 0
    emails_with_in_reply_to = 0
    emails_with_references = 0
    emails_with_bcc = 0
    unique_labels: set[str] = set()
    label_counts: Counter[str] = Counter()
    errors = 0
    start_time = datetime.now().isoformat()

    # Track byte offset by reading mbox file positions
    current_offset = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, msg in enumerate(mbox):
            try:
                # Get raw message size
                try:
                    raw_bytes = msg.as_bytes()
                    raw_size = len(raw_bytes)
                except Exception:
                    raw_size = 0

                email_data = parse_email(msg, mbox_offset=current_offset, raw_size_bytes=raw_size)

                # Update offset for next message (approximate - mbox format adds "From " line)
                current_offset += raw_size + 1  # +1 for newline separator

                # Update stats
                total_emails += 1
                if email_data["body"]:
                    emails_with_body += 1
                if email_data.get("body_html"):
                    emails_with_body_html += 1
                if email_data["has_attachments"]:
                    emails_with_attachments += 1
                if email_data["message_id"]:
                    emails_with_message_id += 1
                if email_data["thread_id"]:
                    emails_with_thread_id += 1
                if email_data.get("in_reply_to"):
                    emails_with_in_reply_to += 1
                if email_data.get("references"):
                    emails_with_references += 1
                if email_data.get("bcc"):
                    emails_with_bcc += 1

                for label in email_data["labels"]:
                    unique_labels.add(label)
                    label_counts[label] += 1

                f.write(json.dumps(email_data, ensure_ascii=False) + "\n")

            except Exception as e:
                errors += 1
                print(f"Error processing email {i}: {e}")

            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1} emails...")

    end_time = datetime.now().isoformat()

    # Build final stats dict for JSON serialization
    stats: dict[str, Any] = {
        "total_emails": total_emails,
        "emails_with_body": emails_with_body,
        "emails_with_body_html": emails_with_body_html,
        "emails_with_attachments": emails_with_attachments,
        "emails_with_message_id": emails_with_message_id,
        "emails_with_thread_id": emails_with_thread_id,
        "emails_with_in_reply_to": emails_with_in_reply_to,
        "emails_with_references": emails_with_references,
        "emails_with_bcc": emails_with_bcc,
        "unique_labels": sorted(unique_labels),
        "label_counts": dict(label_counts.most_common(50)),
        "errors": errors,
        "start_time": start_time,
        "end_time": end_time,
    }

    # Write report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nCompleted!")
    print(f"Total emails: {total_emails}")
    print(f"Emails with body: {emails_with_body}")
    print(f"Emails with body_html: {emails_with_body_html}")
    print(f"Emails with message_id: {emails_with_message_id}")
    print(f"Emails with in_reply_to: {emails_with_in_reply_to}")
    print(f"Emails with references: {emails_with_references}")
    print(f"Emails with bcc: {emails_with_bcc}")
    print(f"Emails with attachments: {emails_with_attachments}")
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Errors: {errors}")
    print(f"\nReport written to: {REPORT_PATH}")

    return stats


if __name__ == '__main__':
    parse_mbox()
