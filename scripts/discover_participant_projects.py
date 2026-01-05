#!/usr/bin/env python3
"""Discover projects by clustering emails based on recurring participant groups.

This script:
1. Extracts all unique participant groups from emails
2. Finds recurring groups (participants who email together frequently)
3. Creates projects for meaningful groups with source='participant'

Usage:
    uv run python scripts/discover_participant_projects.py
    uv run python scripts/discover_participant_projects.py --dry-run
    uv run python scripts/discover_participant_projects.py --min-emails 5
"""

import argparse
import asyncio
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import asyncpg
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection (from environment or default)
DB_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5433/rl_emails')

# Minimum emails for a participant group to be considered a project
DEFAULT_MIN_EMAILS = 3

# Minimum unique participants in a group (excluding self)
MIN_PARTICIPANTS = 2

# Owner's email addresses (to identify "your role" in projects)
YOUR_EMAILS = {'me@nik-patel.com', 'nik@nik-patel.com', 'nikpatel@gmail.com'}

# Patterns for automated/notification emails to filter out
AUTOMATED_PATTERNS = [
    'noreply', 'no-reply', 'no_reply', 'donotreply', 'do-not-reply',
    'notifications', 'notification', 'alerts', 'alert',
    'newsletter', 'newsletters', 'marketing', 'promo',
    'updates', 'update', 'info@', 'support@', 'help@',
    'feedback@', 'team@', 'hello@', 'contact@',
    'mailer', 'mail@', 'email@', 'news@',
    'auto-', 'automated', 'system@', 'admin@',
    'service@', 'services@', 'billing@', 'invoice',
    'receipt', 'order', 'shipping', 'delivery',
    'confirm', 'verify', 'password', 'security@',
    'members@', 'membership@', 'subscriptions@',
    'digest', 'daily', 'weekly', 'monthly',
    'crew@', 'jobs@', 'offers@', 'deals@',
    'notify@', 'editor', 'editors',
    '@email.', '@e.', '@em.', '@em1.',  # Common email marketing subdomains
    '@offers.', '@promo.', '@m.',  # More marketing subdomains
    'channing@', 'sidewalk@',  # Known newsletters
]

# Known automated domains to filter
AUTOMATED_DOMAINS = [
    'amazon.com', 'linkedin.com', 'github.com', 'google.com',
    'slack.com', 'stripe.com', 'uber.com', 'twitter.com',
    'facebook.com', 'instagram.com', 'youtube.com',
    'chase.com', 'bankofamerica.com', 'wellsfargo.com',
    'apple.com', 'microsoft.com', 'adobe.com',
    'dropbox.com', 'zoom.us', 'calendly.com',
    'mailchimp.com', 'sendgrid.com', 'constantcontact.com',
    'hubspot.com', 'salesforce.com', 'zendesk.com',
    'substack.com', 'medium.com', 'reddit.com',
    'expedia.com', 'booking.com', 'airbnb.com',
    'doordash.com', 'grubhub.com', 'postmates.com',
    'spotify.com', 'netflix.com', 'hulu.com',
    'beehiiv.com', 'convertkit.com',
]


def is_automated_email(email: str) -> bool:
    """Check if an email address appears to be automated/notification."""
    email_lower = email.lower()

    # Check for automated patterns in local part
    for pattern in AUTOMATED_PATTERNS:
        if pattern in email_lower:
            return True

    # Check for automated domains
    for domain in AUTOMATED_DOMAINS:
        if domain in email_lower:
            return True

    return False


@dataclass
class ParticipantGroup:
    """A group of participants who communicate together."""
    participants: frozenset[str]  # All participant emails
    email_count: int
    first_email: Optional[datetime]
    last_email: Optional[datetime]
    sample_subjects: list[str]

    def get_name(self) -> str:
        """Generate a project name from participants."""
        # Filter out owner's emails
        others = [p for p in self.participants if p not in YOUR_EMAILS]

        if not others:
            return "Self Communications"

        # Extract names from emails
        names = []
        for email in sorted(others)[:5]:  # Limit to 5 names
            # Try to extract name from email
            local_part = email.split('@')[0]
            # Clean up common patterns
            name = local_part.replace('.', ' ').replace('_', ' ').replace('-', ' ')
            # Capitalize words
            name = ' '.join(word.capitalize() for word in name.split())
            names.append(name)

        if len(names) == 1:
            return f"Thread with {names[0]}"
        elif len(names) <= 3:
            return f"Group: {', '.join(names)}"
        else:
            return f"Group: {', '.join(names[:3])} +{len(others) - 3} others"

    def get_key_people(self) -> list[str]:
        """Get key people in this group (excluding owner)."""
        return [p for p in sorted(self.participants) if p not in YOUR_EMAILS]

    def get_your_role(self) -> str:
        """Determine your role in this group."""
        has_you = bool(self.participants & YOUR_EMAILS)
        if not has_you:
            return 'observer'

        # You're involved - determine if sender or recipient
        # For now, assume participant means contributor
        return 'contributor'


async def get_all_emails(conn: asyncpg.Connection) -> list[dict]:
    """Fetch all emails with participant info."""
    print("Fetching emails...")
    emails = await conn.fetch("""
        SELECT
            id,
            from_email,
            to_emails,
            cc_emails,
            subject,
            date_parsed
        FROM emails
        WHERE from_email IS NOT NULL
        ORDER BY date_parsed
    """)
    print(f"Fetched {len(emails)} emails")
    return emails


def extract_participant_set(email: dict) -> frozenset[str]:
    """Extract all participants from an email as a frozen set.

    Filters out automated/notification emails to focus on human communication.
    """
    participants = set()

    # Add sender (if not automated)
    if email['from_email']:
        addr = email['from_email'].lower()
        if not is_automated_email(addr):
            participants.add(addr)

    # Add recipients (if not automated)
    if email['to_emails']:
        for addr in email['to_emails']:
            addr_lower = addr.lower()
            if not is_automated_email(addr_lower):
                participants.add(addr_lower)

    # Add CC recipients (if not automated)
    if email['cc_emails']:
        for addr in email['cc_emails']:
            addr_lower = addr.lower()
            if not is_automated_email(addr_lower):
                participants.add(addr_lower)

    return frozenset(participants)


def find_recurring_groups(
    emails: list[dict],
    min_emails: int = DEFAULT_MIN_EMAILS,
    min_participants: int = MIN_PARTICIPANTS,
) -> list[ParticipantGroup]:
    """Find participant groups that appear in multiple emails."""
    print(f"Finding recurring groups (min {min_emails} emails, min {min_participants} participants)...")

    # Group emails by participant set
    group_data = defaultdict(lambda: {
        'count': 0,
        'first_email': None,
        'last_email': None,
        'subjects': [],
    })

    skipped_automated = 0
    for email in emails:
        # Skip emails from automated sources entirely
        if email['from_email'] and is_automated_email(email['from_email'].lower()):
            skipped_automated += 1
            continue

        pset = extract_participant_set(email)

        # Skip groups with too few participants
        if len(pset) < min_participants:
            continue

        data = group_data[pset]
        data['count'] += 1

        email_date = email['date_parsed']
        if email_date:
            if data['first_email'] is None or email_date < data['first_email']:
                data['first_email'] = email_date
            if data['last_email'] is None or email_date > data['last_email']:
                data['last_email'] = email_date

        if email['subject'] and len(data['subjects']) < 5:
            data['subjects'].append(email['subject'])

    # Filter to groups that meet minimum email threshold
    groups = []
    for pset, data in group_data.items():
        if data['count'] >= min_emails:
            groups.append(ParticipantGroup(
                participants=pset,
                email_count=data['count'],
                first_email=data['first_email'],
                last_email=data['last_email'],
                sample_subjects=data['subjects'],
            ))

    # Sort by email count (most active first)
    groups.sort(key=lambda g: g.email_count, reverse=True)

    print(f"Skipped {skipped_automated} automated emails")
    print(f"Found {len(groups)} recurring groups")
    return groups


async def insert_projects(
    conn: asyncpg.Connection,
    groups: list[ParticipantGroup],
    dry_run: bool = False,
) -> int:
    """Insert discovered projects into database."""
    print(f"\n{'DRY RUN - ' if dry_run else ''}Inserting {len(groups)} projects...")

    inserted = 0
    for group in groups:
        name = group.get_name()
        key_people = group.get_key_people()
        your_role = group.get_your_role()

        # Generate description from sample subjects
        description = None
        if group.sample_subjects:
            subjects = ', '.join(group.sample_subjects[:3])
            description = f"Topics include: {subjects}"

        # Determine status based on recency
        status = 'active'
        if group.last_email:
            days_since = (datetime.now(group.last_email.tzinfo) - group.last_email).days
            if days_since > 365:
                status = 'completed'
            elif days_since > 90:
                status = 'paused'

        if dry_run:
            print(f"  Would create: {name}")
            print(f"    Emails: {group.email_count}, People: {len(key_people)}")
            print(f"    Key people: {', '.join(key_people[:5])}")
            if len(key_people) > 5:
                print(f"    (+{len(key_people) - 5} more)")
            inserted += 1
            continue

        # Check if similar project already exists
        existing = await conn.fetchval("""
            SELECT id FROM projects
            WHERE source = 'participant'
            AND key_people = $1
        """, key_people)

        if existing:
            print(f"  Skipping (exists): {name}")
            continue

        # Insert project
        await conn.execute("""
            INSERT INTO projects (
                name, description, source, source_detail,
                key_people, your_role, project_type, status,
                started_at, last_activity, email_count
            ) VALUES (
                $1, $2, 'participant', $3,
                $4, $5, 'team', $6,
                $7, $8, $9
            )
        """,
            name,
            description,
            f"Discovered from {group.email_count} emails",
            key_people,
            your_role,
            status,
            group.first_email,
            group.last_email,
            group.email_count,
        )

        print(f"  Created: {name} ({group.email_count} emails)")
        inserted += 1

    return inserted


async def main(args: argparse.Namespace):
    """Main entry point."""
    print("=" * 60)
    print("Participant Clustering for Project Discovery")
    print("=" * 60)

    conn = await asyncpg.connect(DB_URL)

    try:
        # Get all emails
        emails = await get_all_emails(conn)

        if not emails:
            print("No emails found in database!")
            return

        # Find recurring participant groups
        groups = find_recurring_groups(
            emails,
            min_emails=args.min_emails,
            min_participants=args.min_participants,
        )

        if not groups:
            print("No recurring groups found meeting criteria.")
            return

        # Show top groups
        print(f"\nTop {min(10, len(groups))} groups by email count:")
        for i, group in enumerate(groups[:10], 1):
            print(f"  {i}. {group.get_name()}")
            print(f"     {group.email_count} emails, {len(group.participants)} participants")

        if len(groups) > 10:
            print(f"  ... and {len(groups) - 10} more groups")

        # Insert projects
        if args.dry_run:
            print("\n[DRY RUN - no changes will be made]")

        inserted = await insert_projects(conn, groups, dry_run=args.dry_run)

        print(f"\n{'Would insert' if args.dry_run else 'Inserted'} {inserted} projects with source='participant'")

    finally:
        await conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discover projects by participant clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without inserting',
    )
    parser.add_argument(
        '--min-emails',
        type=int,
        default=DEFAULT_MIN_EMAILS,
        help=f'Minimum emails for a group (default: {DEFAULT_MIN_EMAILS})',
    )
    parser.add_argument(
        '--min-participants',
        type=int,
        default=MIN_PARTICIPANTS,
        help=f'Minimum participants per group (default: {MIN_PARTICIPANTS})',
    )

    args = parser.parse_args()
    asyncio.run(main(args))
