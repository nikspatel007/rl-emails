#!/usr/bin/env python3
"""Download Apache mailing list archives for email RL training.

Apache mailing lists provide:
- Public emails with proper RFC 2822 threading (In-Reply-To/References)
- Gzipped mbox archives available for download
- Technical discussions with clear reply patterns

This is an alternative to personal Gmail exports when you need public data
with proper threading headers.

Note: Mailing list archives don't have folder structure (inbox/sent/trash),
so we create synthetic folder assignments based on email metadata:
- Emails with Re: prefix -> 'sent' (simulates a reply you sent)
- Emails with Fwd:/Fw: prefix -> 'sent' (simulates a forward you sent)
- Other emails -> 'inbox' (simulates incoming mail)

This allows using label_actions.py on mailing list data.
"""

import argparse
import gzip
import json
import mailbox
import os
import re
import sys
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


# Popular Apache mailing lists with high traffic
POPULAR_LISTS = {
    'hadoop-common': {
        'domain': 'hadoop.apache.org',
        'list': 'common-user',
        'description': 'Apache Hadoop common user list'
    },
    'spark-user': {
        'domain': 'spark.apache.org',
        'list': 'user',
        'description': 'Apache Spark user discussions'
    },
    'kafka-users': {
        'domain': 'kafka.apache.org',
        'list': 'users',
        'description': 'Apache Kafka user list'
    },
    'lucene-dev': {
        'domain': 'lucene.apache.org',
        'list': 'dev',
        'description': 'Apache Lucene development'
    },
    'httpd-users': {
        'domain': 'httpd.apache.org',
        'list': 'users',
        'description': 'Apache HTTP Server users'
    }
}


def get_mbox_url(domain: str, list_name: str, year: int, month: int) -> str:
    """Generate URL for downloading a monthly mbox archive.

    Args:
        domain: Apache project domain (e.g., 'spark.apache.org')
        list_name: Mailing list name (e.g., 'user', 'dev')
        year: Year (e.g., 2024)
        month: Month (1-12)

    Returns:
        URL to download the gzipped mbox file
    """
    date_str = f"{year}{month:02d}"
    return f"https://lists.apache.org/api/mbox.lua?list={list_name}&domain={domain}&d={date_str}"


def download_mbox(url: str) -> Optional[bytes]:
    """Download mbox content from URL.

    Args:
        url: URL to download from

    Returns:
        Mbox content as bytes or None on error
    """
    try:
        req = Request(url, headers={'User-Agent': 'email-rl-research/1.0'})
        with urlopen(req, timeout=60) as response:
            return response.read()
    except HTTPError as e:
        if e.code == 404:
            return None  # No archive for this month
        print(f"HTTP error {e.code}: {url}", file=sys.stderr)
        return None
    except URLError as e:
        print(f"URL error: {e.reason}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return None


def parse_mbox_content(
    content: bytes,
    assign_user: str = "ml_user"
) -> list[dict]:
    """Parse mbox content into list of email dicts.

    Creates synthetic folder assignments based on subject prefix:
    - Re:/RE: -> 'sent' (simulates reply you sent)
    - Fw:/Fwd:/FW: -> 'sent' (simulates forward you sent)
    - Otherwise -> 'inbox' (simulates incoming mail)

    This allows using label_actions.py on mailing list data.

    Args:
        content: Raw mbox bytes
        assign_user: User name to assign for data split purposes

    Returns:
        List of email dictionaries
    """
    emails = []

    # Write to temp file for mailbox module
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        mbox = mailbox.mbox(tmp_path)

        for idx, msg in enumerate(mbox):
            try:
                # Extract headers
                def get_header(name: str) -> str:
                    value = msg.get(name, '')
                    if value:
                        try:
                            return ' '.join(str(value).split())
                        except Exception:
                            return ''
                    return ''

                subject = get_header('Subject')

                # Synthetic folder assignment based on subject
                subject_lower = subject.lower().strip()
                if subject_lower.startswith('re:'):
                    folder = 'sent'  # Simulates a reply in sent folder
                elif subject_lower.startswith(('fw:', 'fwd:', 'forwarded:')):
                    folder = 'sent'  # Simulates a forward in sent folder
                else:
                    folder = 'inbox'  # Simulates incoming email

                # Extract body
                body = ''
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            try:
                                payload = part.get_payload(decode=True)
                                if payload:
                                    try:
                                        body = payload.decode('utf-8')
                                    except UnicodeDecodeError:
                                        body = payload.decode('latin-1', errors='replace')
                                    break
                            except Exception:
                                pass
                else:
                    try:
                        payload = msg.get_payload(decode=True)
                        if payload:
                            try:
                                body = payload.decode('utf-8')
                            except UnicodeDecodeError:
                                body = payload.decode('latin-1', errors='replace')
                        else:
                            body = msg.get_payload() or ''
                    except Exception:
                        body = msg.get_payload() or ''

                email_data = {
                    'message_id': get_header('Message-ID'),
                    'date': get_header('Date'),
                    'from': get_header('From'),
                    'to': get_header('To'),
                    'cc': get_header('Cc'),
                    'bcc': '',
                    'subject': subject,
                    'body': body.strip(),
                    'in_reply_to': get_header('In-Reply-To'),
                    'references': get_header('References'),
                    'folder': folder,
                    'user': assign_user,
                    'attachments': [],
                }

                emails.append(email_data)

            except Exception as e:
                print(f"Error parsing message {idx}: {e}", file=sys.stderr)
                continue

    finally:
        os.unlink(tmp_path)

    return emails


def download_list_archives(
    domain: str,
    list_name: str,
    start_year: int,
    start_month: int,
    num_months: int = 12,
    output_path: Optional[Path] = None,
) -> dict:
    """Download multiple months of mailing list archives.

    Args:
        domain: Apache project domain
        list_name: Mailing list name
        start_year: Starting year
        start_month: Starting month (1-12)
        num_months: Number of months to download
        output_path: Path to save combined JSON

    Returns:
        Statistics dictionary
    """
    all_emails = []
    months_downloaded = 0
    months_failed = 0

    year, month = start_year, start_month

    for i in range(num_months):
        url = get_mbox_url(domain, list_name, year, month)
        print(f"Downloading {year}-{month:02d}...")

        content = download_mbox(url)

        if content:
            emails = parse_mbox_content(content, assign_user=f"{list_name}_user")
            if emails:
                all_emails.extend(emails)
                months_downloaded += 1
                print(f"  Got {len(emails)} emails")
            else:
                print(f"  Empty or unparseable")
                months_failed += 1
        else:
            print(f"  Not found or error")
            months_failed += 1

        # Move to next month
        month += 1
        if month > 12:
            month = 1
            year += 1

    # Count threading statistics
    has_in_reply_to = sum(1 for e in all_emails if e.get('in_reply_to'))
    has_references = sum(1 for e in all_emails if e.get('references'))
    total = len(all_emails)

    stats = {
        'total_emails': total,
        'months_downloaded': months_downloaded,
        'months_failed': months_failed,
        'has_in_reply_to': has_in_reply_to,
        'has_references': has_references,
        'threading_rate': (has_in_reply_to / total * 100) if total > 0 else 0,
    }

    print(f"\nTotal: {total} emails from {months_downloaded} months")
    print(f"Threading headers:")
    print(f"  In-Reply-To: {has_in_reply_to} ({stats['threading_rate']:.1f}%)")
    print(f"  References:  {has_references} ({has_references/total*100:.1f}%)" if total > 0 else "")

    if output_path and all_emails:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_emails, f, indent=2, ensure_ascii=False)
        print(f"Output written to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Download Apache mailing list archives for email RL training',
        epilog='''
Example usage:
    # Download 6 months of Spark user list
    python download_mailing_list.py spark-user --months 6

    # Download specific list by domain
    python download_mailing_list.py --domain kafka.apache.org --list users

    # List available preset lists
    python download_mailing_list.py --list-presets
'''
    )
    parser.add_argument(
        'preset',
        nargs='?',
        choices=list(POPULAR_LISTS.keys()),
        help='Preset mailing list to download'
    )
    parser.add_argument(
        '--domain',
        help='Apache project domain (e.g., spark.apache.org)'
    )
    parser.add_argument(
        '--list',
        dest='list_name',
        help='Mailing list name (e.g., user, dev)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2024,
        help='Starting year (default: 2024)'
    )
    parser.add_argument(
        '--start-month',
        type=int,
        default=1,
        help='Starting month 1-12 (default: 1)'
    )
    parser.add_argument(
        '--months',
        type=int,
        default=6,
        help='Number of months to download (default: 6)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('data/mailing_list_emails.json'),
        help='Output JSON file path'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available preset mailing lists'
    )

    args = parser.parse_args()

    if args.list_presets:
        print("Available preset mailing lists:")
        for name, info in POPULAR_LISTS.items():
            print(f"  {name}: {info['description']}")
            print(f"    {info['domain']} / {info['list']}")
        return

    # Determine domain and list
    if args.preset:
        info = POPULAR_LISTS[args.preset]
        domain = info['domain']
        list_name = info['list']
    elif args.domain and args.list_name:
        domain = args.domain
        list_name = args.list_name
    else:
        parser.error("Either provide a preset or both --domain and --list")

    print(f"Downloading {domain}/{list_name}")
    print(f"From {args.start_year}-{args.start_month:02d} for {args.months} months")
    print()

    stats = download_list_archives(
        domain=domain,
        list_name=list_name,
        start_year=args.start_year,
        start_month=args.start_month,
        num_months=args.months,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
