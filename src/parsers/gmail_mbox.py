#!/usr/bin/env python3
"""Parse Gmail mbox export to structured JSON format.

Gmail mbox exports have useful headers:
- X-GM-THRID: Native thread ID (no subject matching needed!)
- X-Gmail-Labels: Explicit labels like Sent, Inbox, Archived, etc.

Usage:
    python -m src.parsers.gmail_mbox "path/to/All mail.mbox" -o emails.json

Get your mbox from Google Takeout: https://takeout.google.com/settings/takeout
"""

import argparse
import json
import mailbox
import sys
from email import policy
from pathlib import Path
from typing import Iterator, Optional


def get_body(message) -> str:
    """Extract plain text body from email message."""
    body = ''

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        # Try UTF-8 first, fall back to latin-1
                        try:
                            body = payload.decode('utf-8')
                        except UnicodeDecodeError:
                            body = payload.decode('latin-1', errors='replace')
                        break
                except Exception:
                    pass
    else:
        try:
            payload = message.get_payload(decode=True)
            if payload:
                try:
                    body = payload.decode('utf-8')
                except UnicodeDecodeError:
                    body = payload.decode('latin-1', errors='replace')
            else:
                body = message.get_payload() or ''
        except Exception:
            body = message.get_payload() or ''

    return body.strip()


def parse_labels(labels_header: str) -> list[str]:
    """Parse X-Gmail-Labels header into list of labels.

    Labels can be comma-separated and may contain quoted strings for
    labels with special characters.
    """
    if not labels_header:
        return []

    labels = []
    current = ''
    in_quotes = False

    for char in labels_header:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            label = current.strip().strip('"')
            if label:
                labels.append(label)
            current = ''
        else:
            current += char

    # Don't forget the last label
    label = current.strip().strip('"')
    if label:
        labels.append(label)

    return labels


def get_header(message, name: str) -> str:
    """Extract and normalize a header value."""
    value = message.get(name, '')
    if value:
        try:
            return ' '.join(str(value).split())
        except Exception:
            return ''
    return ''


def parse_gmail_message(message) -> dict:
    """Parse a single Gmail mbox message into structured data."""
    body = get_body(message)
    labels = parse_labels(get_header(message, 'X-Gmail-Labels'))

    return {
        'message_id': get_header(message, 'Message-ID'),
        'date': get_header(message, 'Date'),
        'from': get_header(message, 'From'),
        'to': get_header(message, 'To'),
        'cc': get_header(message, 'Cc'),
        'bcc': get_header(message, 'Bcc'),
        'subject': get_header(message, 'Subject'),
        'body': body,
        'in_reply_to': get_header(message, 'In-Reply-To'),
        'references': get_header(message, 'References'),
        # Gmail-specific fields
        'thread_id': get_header(message, 'X-GM-THRID'),
        'labels': labels,
    }


def parse_gmail_mbox(mbox_path: Path) -> Iterator[dict]:
    """Parse Gmail mbox file, yielding structured email dicts.

    Args:
        mbox_path: Path to the mbox file

    Yields:
        Dict for each email with standard fields plus thread_id and labels
    """
    mbox = mailbox.mbox(str(mbox_path))

    for message in mbox:
        yield parse_gmail_message(message)


def process_gmail_mbox(
    mbox_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    skip_sent: bool = False,
) -> int:
    """Process Gmail mbox file into JSON format.

    Args:
        mbox_path: Path to the mbox file
        output_path: Path for output JSON file
        limit: Optional limit on number of emails to process
        skip_sent: If True, skip emails with 'Sent' label

    Returns:
        Number of emails successfully processed
    """
    emails = []
    error_count = 0
    skipped_sent = 0

    mbox = mailbox.mbox(str(mbox_path))

    for i, message in enumerate(mbox):
        if limit and len(emails) >= limit:
            break

        try:
            email_data = parse_gmail_message(message)

            # Optionally skip sent emails
            if skip_sent and 'Sent' in email_data.get('labels', []):
                skipped_sent += 1
                continue

            emails.append(email_data)

        except Exception as e:
            print(f"Error parsing message {i}: {e}", file=sys.stderr)
            error_count += 1
            continue

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(emails)} emails ({error_count} errors)")
    if skip_sent:
        print(f"Skipped {skipped_sent} sent emails")
    print(f"Output written to: {output_path}")

    return len(emails)


def main():
    parser = argparse.ArgumentParser(
        description='Parse Gmail mbox export to JSON format'
    )
    parser.add_argument(
        'mbox',
        type=Path,
        help='Path to the Gmail mbox file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('gmail_emails.json'),
        help='Output JSON file path (default: gmail_emails.json)'
    )
    parser.add_argument(
        '-n', '--limit',
        type=int,
        help='Limit number of emails to process (for testing)'
    )
    parser.add_argument(
        '--skip-sent',
        action='store_true',
        help='Skip emails with Sent label'
    )

    args = parser.parse_args()

    if not args.mbox.is_file():
        print(f"Error: {args.mbox} is not a file", file=sys.stderr)
        sys.exit(1)

    process_gmail_mbox(args.mbox, args.output, args.limit, args.skip_sent)


if __name__ == '__main__':
    main()
