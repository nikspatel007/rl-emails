"""Stage 1: Parse Gmail MBOX export to normalized JSONL format.

Extracts all metadata and body content from Gmail Takeout MBOX files.
Handles multipart messages, label parsing, and attachment detection.
"""

from __future__ import annotations

import json
import mailbox
import re
import time
from collections import Counter
from email.header import decode_header
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult

if TYPE_CHECKING:  # pragma: no cover
    pass


def decode_header_value(value: str | None) -> str:
    """Decode RFC 2047 encoded header values.

    Args:
        value: Header value to decode.

    Returns:
        Decoded string.
    """
    if not value:
        return ""

    try:
        decoded_parts = decode_header(value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                try:
                    result.append(part.decode(charset or "utf-8", errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    result.append(part.decode("utf-8", errors="replace"))
            else:
                result.append(part)
        return "".join(result)
    except Exception:
        return str(value)


def parse_labels(labels_str: str | None) -> list[str]:
    """Parse Gmail labels from X-Gmail-Labels header.

    Gmail labels can contain commas in the label name (e.g., [Superhuman]/AI/Marketing).
    Labels are comma-separated but quoted if they contain special chars.
    Handles RFC 2822 header folding (embedded \\r\\n).

    Args:
        labels_str: Raw labels string from header.

    Returns:
        List of parsed label strings.
    """
    if not labels_str:
        return []

    # Normalize header folding - remove \r\n and collapse whitespace
    labels_str = re.sub(r"\r\n\s*", " ", labels_str)

    labels = []
    current = ""
    in_quotes = False

    for char in labels_str:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
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

    Args:
        msg: The mailbox message to extract body from.

    Returns:
        Tuple of (body_text, body_html).
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


def _is_attachment_part(part: Any) -> bool:
    """Check if a message part is an attachment."""
    content_disposition = str(part.get("Content-Disposition", ""))
    if "attachment" in content_disposition:
        return True

    content_type = part.get_content_type()
    if not content_type or content_type.startswith(("text/", "multipart/")):
        return False

    return bool(part.get_filename())


def has_attachments(msg: mailbox.mboxMessage) -> bool:
    """Check if message has any attachments.

    Args:
        msg: The mailbox message to check.

    Returns:
        True if message has attachments.
    """
    if not msg.is_multipart():
        return False

    for part in msg.walk():
        if _is_attachment_part(part):
            return True
    return False


def parse_email(
    msg: mailbox.mboxMessage, mbox_offset: int = 0, raw_size_bytes: int = 0
) -> dict[str, Any]:
    """Parse a single email message to normalized dict.

    Args:
        msg: The mailbox message to parse.
        mbox_offset: Byte offset in mbox file where message starts.
        raw_size_bytes: Size of raw message in bytes.

    Returns:
        Normalized email data dictionary.
    """
    body_text, body_html = get_body(msg)

    return {
        "message_id": msg.get("Message-ID", ""),
        "thread_id": msg.get("X-GM-THRID", ""),
        "in_reply_to": msg.get("In-Reply-To", ""),
        "references": msg.get("References", ""),
        "labels": parse_labels(msg.get("X-Gmail-Labels", "")),
        "date": msg.get("Date", ""),
        "from": decode_header_value(msg.get("From", "")),
        "to": decode_header_value(msg.get("To", "")),
        "cc": decode_header_value(msg.get("Cc", "")),
        "bcc": decode_header_value(msg.get("Bcc", "")),
        "subject": decode_header_value(msg.get("Subject", "")),
        "body": body_text,
        "body_html": body_html if body_html else None,
        "has_attachments": has_attachments(msg),
        "mbox_offset": mbox_offset,
        "raw_size_bytes": raw_size_bytes,
    }


def parse_mbox_file(
    mbox_path: Path, output_path: Path, report_path: Path | None = None
) -> dict[str, Any]:
    """Parse MBOX file and write to JSONL.

    Args:
        mbox_path: Path to the MBOX file.
        output_path: Path to write JSONL output.
        report_path: Optional path to write JSON report.

    Returns:
        Statistics dictionary.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mbox = mailbox.mbox(str(mbox_path))

    # Track stats during parsing
    total_emails = 0
    emails_with_body = 0
    emails_with_attachments = 0
    unique_labels: set[str] = set()
    label_counts: Counter[str] = Counter()
    errors = 0

    # Track byte offset
    current_offset = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for msg in mbox:
            try:
                # Get raw message size
                try:
                    raw_bytes = msg.as_bytes()
                    raw_size = len(raw_bytes)
                except Exception:
                    raw_size = 0

                email_data = parse_email(msg, mbox_offset=current_offset, raw_size_bytes=raw_size)
                current_offset += raw_size + 1

                # Update stats
                total_emails += 1
                if email_data["body"]:
                    emails_with_body += 1
                if email_data["has_attachments"]:
                    emails_with_attachments += 1

                for label in email_data["labels"]:
                    unique_labels.add(label)
                    label_counts[label] += 1

                f.write(json.dumps(email_data, ensure_ascii=False) + "\n")

            except Exception:
                errors += 1

    # Build final stats dict
    stats: dict[str, Any] = {
        "total_emails": total_emails,
        "emails_with_body": emails_with_body,
        "emails_with_attachments": emails_with_attachments,
        "unique_labels": sorted(unique_labels),
        "label_counts": dict(label_counts.most_common(50)),
        "errors": errors,
    }

    # Write report if path provided
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def run(config: Config, *, output_path: Path | None = None) -> StageResult:
    """Run Stage 1: Parse MBOX file to JSONL format.

    Args:
        config: Application configuration with mbox_path.
        output_path: Optional override for output path.

    Returns:
        StageResult with parsing statistics.
    """
    start_time = time.time()

    if not config.mbox_path:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="MBOX_PATH not configured",
        )

    mbox_path = config.mbox_path
    if not mbox_path.exists():
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message=f"MBOX file not found: {mbox_path}",
        )

    # Determine output path
    if output_path is None:
        if config.parsed_jsonl:
            output_path = config.parsed_jsonl
        else:
            output_path = mbox_path.parent / "parsed_emails.jsonl"

    report_path = output_path.parent / "parse_report.json"

    stats = parse_mbox_file(mbox_path, output_path, report_path)
    duration = time.time() - start_time

    return StageResult(
        success=True,
        records_processed=stats["total_emails"],
        duration_seconds=duration,
        message=f"Parsed {stats['total_emails']} emails to {output_path}",
        metadata=stats,
    )
