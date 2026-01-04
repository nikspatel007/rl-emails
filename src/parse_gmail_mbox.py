#!/usr/bin/env python3
"""Parse Gmail mbox exports (from Google Takeout) to structured JSON format.

Gmail mbox files have:
- Standard mbox format (messages separated by "From " lines)
- X-Gmail-Labels header containing folder/label info
- Proper RFC 2822 threading: In-Reply-To and References headers

This addresses the key limitation of Enron data where only ~0.01% of emails
have threading headers. Gmail has ~25% with proper In-Reply-To/References.
"""

import argparse
import email
import json
import mailbox
import re
import sys
from email import policy
from pathlib import Path
from typing import Optional


def parse_gmail_labels(labels_header: str) -> list[str]:
    """Parse X-Gmail-Labels header into list of labels.

    Gmail labels can be comma-separated and may be quoted if they contain
    special characters.

    Args:
        labels_header: Value of X-Gmail-Labels header

    Returns:
        List of label strings
    """
    if not labels_header:
        return []

    labels = []
    # Handle quoted labels and comma separation
    current = ""
    in_quotes = False

    for char in labels_header:
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


def infer_folder_from_labels(labels: list[str]) -> str:
    """Infer a folder name from Gmail labels for compatibility with label_actions.py.

    Maps Gmail labels to folder names that label_actions.py understands:
    - Sent/Sent Mail -> sent
    - Trash -> deleted_items
    - Inbox -> inbox
    - Spam -> junk
    - Other labels -> archived (user-created folders)

    Args:
        labels: List of Gmail labels

    Returns:
        Folder name compatible with label_actions.py
    """
    labels_lower = [l.lower() for l in labels]

    # Priority order for folder classification
    if 'sent' in labels_lower or 'sent mail' in labels_lower:
        return 'sent'
    elif 'trash' in labels_lower:
        return 'deleted_items'
    elif 'spam' in labels_lower:
        return 'junk'
    elif 'inbox' in labels_lower:
        return 'inbox'
    elif 'drafts' in labels_lower:
        return 'drafts'
    elif 'important' in labels_lower:
        # Important but not in specific folder -> treat as inbox
        return 'inbox'
    else:
        # User-created labels = archived/project folders
        # Pick the first non-system label
        system_labels = {'opened', 'unread', 'starred', 'important',
                        'category personal', 'category social',
                        'category promotions', 'category updates',
                        'category forums', 'chats', 'archived'}
        for label in labels:
            if label.lower() not in system_labels:
                return label  # Use the label as folder name
        return 'archived'


def extract_body(msg: email.message.Message) -> str:
    """Extract text body from email message.

    Handles multipart messages by finding the first text/plain part.
    """
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        # Try UTF-8 first, fall back to latin-1
                        try:
                            return payload.decode('utf-8')
                        except UnicodeDecodeError:
                            return payload.decode('latin-1', errors='replace')
                except Exception:
                    pass
        # Fall back to first text part
        for part in msg.walk():
            if part.get_content_maintype() == 'text':
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            return payload.decode('utf-8')
                        except UnicodeDecodeError:
                            return payload.decode('latin-1', errors='replace')
                except Exception:
                    pass
        return ''
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    return payload.decode('utf-8')
                except UnicodeDecodeError:
                    return payload.decode('latin-1', errors='replace')
            else:
                return msg.get_payload() or ''
        except Exception:
            return msg.get_payload() or ''


def parse_email_message(msg: email.message.Message, idx: int) -> Optional[dict]:
    """Parse a single email message into structured data.

    Args:
        msg: Email message object
        idx: Index in mbox for reference

    Returns:
        Structured email dict or None on error
    """
    try:
        def get_header(name: str) -> str:
            value = msg.get(name, '')
            if value:
                try:
                    return ' '.join(str(value).split())
                except Exception:
                    return ''
            return ''

        # Extract Gmail-specific labels
        gmail_labels = parse_gmail_labels(get_header('X-Gmail-Labels'))
        folder = infer_folder_from_labels(gmail_labels)

        # Extract body
        body = extract_body(msg).strip()

        # Extract attachments from Content-Disposition headers
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                disposition = part.get('Content-Disposition', '')
                if 'attachment' in disposition.lower():
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)

        return {
            'message_id': get_header('Message-ID'),
            'date': get_header('Date'),
            'from': get_header('From'),
            'to': get_header('To'),
            'cc': get_header('Cc'),
            'bcc': get_header('Bcc'),
            'subject': get_header('Subject'),
            'body': body,
            'in_reply_to': get_header('In-Reply-To'),
            'references': get_header('References'),
            'x_gmail_labels': gmail_labels,
            'folder': folder,
            'attachments': attachments,
            'mbox_index': idx,
            # Compatibility with Enron format
            'user': 'gmail_user',  # Single user for personal export
        }
    except Exception as e:
        print(f"Error parsing message {idx}: {e}", file=sys.stderr)
        return None


def process_mbox(
    mbox_path: Path,
    output_path: Path,
    limit: Optional[int] = None
) -> dict:
    """Process Gmail mbox file into JSON format.

    Args:
        mbox_path: Path to the mbox file
        output_path: Path for output JSON file
        limit: Optional limit on number of emails to process

    Returns:
        Statistics dictionary
    """
    print(f"Loading mbox from {mbox_path}...")

    try:
        mbox = mailbox.mbox(str(mbox_path))
    except Exception as e:
        print(f"Error opening mbox: {e}", file=sys.stderr)
        return {'error': str(e)}

    emails = []
    error_count = 0

    # Count threading headers for statistics
    has_in_reply_to = 0
    has_references = 0

    for idx, msg in enumerate(mbox):
        if limit and len(emails) >= limit:
            break

        email_data = parse_email_message(msg, idx)

        if email_data is None:
            error_count += 1
            continue

        # Track threading statistics
        if email_data['in_reply_to']:
            has_in_reply_to += 1
        if email_data['references']:
            has_references += 1

        emails.append(email_data)

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} messages...")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    total = len(emails)
    stats = {
        'total_processed': total,
        'errors': error_count,
        'has_in_reply_to': has_in_reply_to,
        'has_references': has_references,
        'threading_rate': (has_in_reply_to / total * 100) if total > 0 else 0,
    }

    print(f"\nProcessed {total} emails ({error_count} errors)")
    print(f"Threading headers:")
    print(f"  In-Reply-To: {has_in_reply_to} ({stats['threading_rate']:.1f}%)")
    print(f"  References:  {has_references} ({has_references/total*100:.1f}%)" if total > 0 else "")
    print(f"Output written to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Parse Gmail mbox exports to JSON format',
        epilog='''
Example usage:
    # Parse a Google Takeout mbox export
    python parse_gmail_mbox.py ~/Takeout/Mail/All_mail.mbox

    # Limit for testing
    python parse_gmail_mbox.py mail.mbox -n 1000 -o test_emails.json

To get your Gmail data:
    1. Go to https://takeout.google.com
    2. Deselect all, then select only "Mail"
    3. Choose mbox format
    4. Download and extract the archive
    5. The mbox file will be in Takeout/Mail/
'''
    )
    parser.add_argument(
        'mbox',
        type=Path,
        help='Path to the Gmail mbox file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('data/gmail_emails.json'),
        help='Output JSON file path (default: data/gmail_emails.json)'
    )
    parser.add_argument(
        '-n', '--limit',
        type=int,
        help='Limit number of emails to process (for testing)'
    )

    args = parser.parse_args()

    if not args.mbox.exists():
        print(f"Error: {args.mbox} not found", file=sys.stderr)
        sys.exit(1)

    stats = process_mbox(args.mbox, args.output, args.limit)

    if 'error' in stats:
        sys.exit(1)


if __name__ == '__main__':
    main()
