#!/usr/bin/env python3
"""Parse Enron maildir emails to structured JSON format."""

import argparse
import email
import json
import re
import sys
from email import policy
from pathlib import Path
from typing import Optional


def extract_attachments(body: str) -> tuple[str, list[str]]:
    """Extract attachment references from email body.

    Enron emails list attachments at the end like:
     - filename.ext
     - another_file.doc

    Returns (cleaned_body, list_of_attachments).
    """
    lines = body.rstrip().split('\n')
    attachments = []

    # Work backwards to find attachment lines at end
    i = len(lines) - 1
    while i >= 0:
        line = lines[i].strip()
        if line.startswith(' - ') or line.startswith('- '):
            # Extract filename
            filename = line.lstrip(' -').strip()
            if filename:
                attachments.insert(0, filename)
            i -= 1
        elif line == '':
            i -= 1
        else:
            break

    # Rebuild body without attachment lines
    if attachments:
        cleaned_body = '\n'.join(lines[:i+1]).rstrip()
    else:
        cleaned_body = body

    return cleaned_body, attachments


def parse_email_file(file_path: Path) -> Optional[dict]:
    """Parse a single email file into structured data."""
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            msg = email.message_from_file(f, policy=policy.default)
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None

    # Extract body
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                try:
                    body = part.get_content()
                    break
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode('latin-1', errors='replace')
            else:
                body = msg.get_payload() or ''
        except Exception:
            body = msg.get_payload() or ''

    # Extract attachments from body
    body, attachments = extract_attachments(body)

    # Parse headers
    def get_header(name: str) -> str:
        value = msg.get(name, '')
        if value:
            # Decode if needed and normalize whitespace
            try:
                return ' '.join(str(value).split())
            except Exception:
                return ''
        return ''

    return {
        'message_id': get_header('Message-ID'),
        'date': get_header('Date'),
        'from': get_header('From'),
        'to': get_header('To'),
        'cc': get_header('Cc'),
        'bcc': get_header('Bcc') or get_header('X-bcc'),
        'subject': get_header('Subject'),
        'body': body.strip(),
        'in_reply_to': get_header('In-Reply-To'),
        'references': get_header('References'),
        'x_from': get_header('X-From'),
        'x_to': get_header('X-To'),
        'x_folder': get_header('X-Folder'),
        'x_origin': get_header('X-Origin'),
        'attachments': attachments,
    }


def process_maildir(maildir_path: Path, output_path: Path, limit: Optional[int] = None) -> int:
    """Process entire maildir into JSON format.

    Args:
        maildir_path: Path to the maildir directory
        output_path: Path for output JSON file
        limit: Optional limit on number of emails to process

    Returns:
        Number of emails successfully processed
    """
    emails = []
    error_count = 0

    user_dirs = sorted([d for d in maildir_path.iterdir() if d.is_dir()])

    for user_dir in user_dirs:
        user = user_dir.name

        for email_file in sorted(user_dir.rglob('*')):
            if limit and len(emails) >= limit:
                break

            if not email_file.is_file():
                continue
            if email_file.name.startswith('.'):
                continue

            email_data = parse_email_file(email_file)
            if email_data is None:
                error_count += 1
                continue

            # Add metadata
            email_data['user'] = user
            email_data['folder'] = email_file.parent.name
            email_data['file_path'] = str(email_file.relative_to(maildir_path))

            emails.append(email_data)

        if limit and len(emails) >= limit:
            break

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(emails)} emails ({error_count} errors)")
    print(f"Output written to: {output_path}")

    return len(emails)


def main():
    parser = argparse.ArgumentParser(
        description='Parse Enron maildir emails to JSON format'
    )
    parser.add_argument(
        'maildir',
        type=Path,
        help='Path to the maildir directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('emails.json'),
        help='Output JSON file path (default: emails.json)'
    )
    parser.add_argument(
        '-n', '--limit',
        type=int,
        help='Limit number of emails to process (for testing)'
    )

    args = parser.parse_args()

    if not args.maildir.is_dir():
        print(f"Error: {args.maildir} is not a directory", file=sys.stderr)
        sys.exit(1)

    process_maildir(args.maildir, args.output, args.limit)


if __name__ == '__main__':
    main()
