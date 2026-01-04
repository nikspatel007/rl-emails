#!/usr/bin/env python3
"""Populate threads table with aggregated data from emails.

This script computes thread-level statistics by aggregating email data:
- email_count: number of emails in thread
- participants: array of unique email addresses
- started_at: timestamp of first email
- last_activity: timestamp of last email
- your_email_count: emails sent by user
- your_reply_count: replies sent by user
- thread_duration_seconds: time between first and last email
- avg_response_time_seconds: average time between consecutive emails
- has_attachments: whether any email has attachments
- total_attachment_count: sum of attachments across all emails

Usage:
    # Start SurrealDB first:
    surreal start file:data/enron.db --user root --pass root --bind 0.0.0.0:8000

    # Run population:
    python db/populate_threads.py --database gmail --user-email me@nik-patel.com
"""

import argparse
from collections import defaultdict
from datetime import datetime
from typing import Optional

from surrealdb import Surreal


def populate_threads(
    db: Surreal,
    user_email: str,
    batch_size: int = 100,
) -> dict:
    """Populate threads table with aggregated email data.

    Args:
        db: Connected SurrealDB instance
        user_email: User's email address for computing your_* fields
        batch_size: Number of threads to update per batch

    Returns:
        Statistics dict with counts
    """
    stats = {
        'threads_updated': 0,
        'threads_created': 0,
        'emails_processed': 0,
    }

    user_email = user_email.lower().strip()

    # Step 1: Get all emails with their thread associations
    print("Fetching emails...")
    result = db.query('''
        SELECT
            message_id,
            gmail_thread_id,
            in_reply_to,
            date,
            subject,
            from_email,
            to_emails,
            cc_emails,
            attachments
        FROM emails
    ''')

    if not result or not isinstance(result, list):
        print("No emails found")
        return stats

    emails = result
    print(f"Found {len(emails)} emails")

    # Step 2: Group emails by thread_id
    # For Gmail, use gmail_thread_id; for Enron, use in_reply_to chain or message_id
    thread_emails = defaultdict(list)

    for email in emails:
        # Determine thread_id
        thread_id = (
            email.get('gmail_thread_id') or
            email.get('in_reply_to') or
            email.get('message_id')
        )
        if thread_id:
            thread_emails[thread_id].append(email)
            stats['emails_processed'] += 1

    print(f"Grouped into {len(thread_emails)} threads")

    # Step 3: Compute aggregates and update threads
    print("Computing thread aggregates...")

    for thread_id, emails_in_thread in thread_emails.items():
        # Sort by date
        emails_sorted = sorted(
            emails_in_thread,
            key=lambda e: e.get('date') or datetime.min
        )

        # Compute participants (all unique email addresses)
        participants = set()
        for email in emails_sorted:
            from_email = email.get('from_email', '').lower()
            if from_email:
                participants.add(from_email)
            for to in email.get('to_emails', []):
                participants.add(to.lower())
            for cc in email.get('cc_emails', []):
                participants.add(cc.lower())

        # Compute timestamps
        dates = [e.get('date') for e in emails_sorted if e.get('date')]
        started_at = min(dates) if dates else None
        last_activity = max(dates) if dates else None

        # Compute duration
        thread_duration_seconds = None
        if started_at and last_activity and len(dates) > 1:
            if isinstance(started_at, datetime) and isinstance(last_activity, datetime):
                thread_duration_seconds = (last_activity - started_at).total_seconds()

        # Compute average response time
        avg_response_time_seconds = None
        if len(dates) > 1:
            response_times = []
            for i in range(1, len(dates)):
                prev_date = dates[i - 1]
                curr_date = dates[i]
                if isinstance(prev_date, datetime) and isinstance(curr_date, datetime):
                    delta = (curr_date - prev_date).total_seconds()
                    if delta > 0:
                        response_times.append(delta)
            if response_times:
                avg_response_time_seconds = sum(response_times) / len(response_times)

        # Compute user-specific counts
        your_email_count = 0
        your_reply_count = 0
        for email in emails_sorted:
            from_email = email.get('from_email', '').lower()
            if from_email == user_email:
                your_email_count += 1
                # If this email is a reply (has in_reply_to), count as reply
                if email.get('in_reply_to'):
                    your_reply_count += 1

        # Compute attachment stats
        total_attachment_count = 0
        has_attachments = False
        for email in emails_sorted:
            attachments = email.get('attachments', [])
            if attachments:
                has_attachments = True
                total_attachment_count += len(attachments)

        # Get subject from first email
        subject = emails_sorted[0].get('subject', '') if emails_sorted else ''

        # Update or create thread record
        thread_data = {
            'thread_id': thread_id,
            'subject': subject,
            'email_count': len(emails_in_thread),
            'participants': list(participants),
            'started_at': started_at,
            'last_activity': last_activity,
            'your_email_count': your_email_count,
            'your_reply_count': your_reply_count,
            'thread_duration_seconds': thread_duration_seconds,
            'avg_response_time_seconds': avg_response_time_seconds,
            'has_attachments': has_attachments,
            'total_attachment_count': total_attachment_count,
        }

        # Check if thread exists
        existing = db.query(
            'SELECT id FROM threads WHERE thread_id = $thread_id LIMIT 1',
            {'thread_id': thread_id}
        )

        if existing and len(existing) > 0 and existing[0].get('id'):
            # Update existing thread
            thread_rec_id = existing[0]['id']
            db.query(
                '''
                UPDATE $id SET
                    subject = $subject,
                    email_count = $email_count,
                    participants = $participants,
                    started_at = $started_at,
                    last_activity = $last_activity,
                    your_email_count = $your_email_count,
                    your_reply_count = $your_reply_count,
                    thread_duration_seconds = $thread_duration_seconds,
                    avg_response_time_seconds = $avg_response_time_seconds,
                    has_attachments = $has_attachments,
                    total_attachment_count = $total_attachment_count
                ''',
                {'id': thread_rec_id, **thread_data}
            )
            stats['threads_updated'] += 1
        else:
            # Create new thread
            try:
                db.create('threads', thread_data)
                stats['threads_created'] += 1
            except Exception as e:
                print(f"Warning: Could not create thread {thread_id}: {e}")

        # Progress reporting
        processed = stats['threads_updated'] + stats['threads_created']
        if processed % batch_size == 0:
            print(f"  Processed {processed}/{len(thread_emails)} threads...")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Populate threads table with aggregated email data'
    )
    parser.add_argument(
        '--url',
        default='ws://localhost:8000/rpc',
        help='SurrealDB connection URL'
    )
    parser.add_argument(
        '--user',
        default='root',
        help='SurrealDB username'
    )
    parser.add_argument(
        '--pass',
        dest='password',
        default='root',
        help='SurrealDB password'
    )
    parser.add_argument(
        '--namespace',
        default='rl_emails',
        help='SurrealDB namespace'
    )
    parser.add_argument(
        '--database',
        default='gmail',
        help='SurrealDB database name'
    )
    parser.add_argument(
        '--user-email',
        default='me@nik-patel.com',
        help='User email address for computing your_* fields'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for progress reporting'
    )

    args = parser.parse_args()

    print(f"Connecting to {args.url}...")

    with Surreal(args.url) as db:
        db.signin({'username': args.user, 'password': args.password})
        db.use(args.namespace, args.database)
        print(f"Connected to namespace={args.namespace}, database={args.database}")
        print(f"User email: {args.user_email}")

        stats = populate_threads(
            db,
            user_email=args.user_email,
            batch_size=args.batch_size,
        )

        print(f"\nPopulation complete!")
        print(f"  Emails processed: {stats['emails_processed']}")
        print(f"  Threads created: {stats['threads_created']}")
        print(f"  Threads updated: {stats['threads_updated']}")


if __name__ == '__main__':
    main()
