#!/usr/bin/env python3
"""
Detect priority contexts via participant bursts.

This script analyzes email patterns to find time periods with unusual
participant activity - e.g., many emails from recruiters in Q1 (job search),
or real estate agents during house buying.

Detection approach:
1. Group emails by (month, sender_domain)
2. Calculate baseline activity per domain
3. Find months where domain activity > 2x baseline (spike)
4. Group consecutive spike months into contexts
5. Write discovered contexts to priority_contexts table

Usage:
    python scripts/detect_participant_bursts.py [--dry-run]
"""

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Detection thresholds
MIN_EMAILS_FOR_DOMAIN = 5  # Minimum emails from domain to consider
SPIKE_THRESHOLD = 2.0  # Activity > 2x average = spike
MIN_SPIKE_MONTHS = 1  # Minimum months to form a context (1 = single month is ok)
MAX_GAP_MONTHS = 2  # Maximum gap between months to group together

# Domains to ignore (too generic or service-like)
IGNORE_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
    'googlemail.com', 'aol.com', 'icloud.com', 'me.com',
    'mail.com', 'protonmail.com', 'pm.me',
}

# Domain categories for labeling
DOMAIN_CATEGORIES = {
    'linkedin.com': 'Professional Networking',
    'indeed.com': 'Job Search',
    'glassdoor.com': 'Job Search',
    'lever.co': 'Job Search',
    'greenhouse.io': 'Job Search',
    'workday.com': 'Job Search',
    'amazon.com': 'Shopping',
    'ebay.com': 'Shopping',
    'zillow.com': 'Real Estate',
    'redfin.com': 'Real Estate',
    'realtor.com': 'Real Estate',
    'github.com': 'Development',
    'gitlab.com': 'Development',
    'stripe.com': 'Payments',
    'paypal.com': 'Payments',
}


async def get_monthly_domain_activity(conn: asyncpg.Connection) -> list[dict]:
    """Get email counts grouped by month and sender domain."""
    rows = await conn.fetch("""
        SELECT
            date_trunc('month', date_parsed) as month,
            LOWER(SPLIT_PART(from_email, '@', 2)) as domain,
            COUNT(*) as email_count,
            array_agg(DISTINCT subject) as sample_subjects
        FROM emails
        WHERE date_parsed IS NOT NULL
          AND from_email IS NOT NULL
          AND NOT is_sent
        GROUP BY month, domain
        HAVING COUNT(*) >= 1
        ORDER BY month, domain
    """)
    return [dict(row) for row in rows]


def calculate_domain_baselines(activity: list[dict]) -> dict[str, dict]:
    """Calculate baseline activity stats per domain."""
    domain_months = defaultdict(list)

    for row in activity:
        domain = row['domain']
        if domain not in IGNORE_DOMAINS:
            domain_months[domain].append(row['email_count'])

    baselines = {}
    for domain, counts in domain_months.items():
        if len(counts) >= 2:  # Need multiple months for baseline
            baselines[domain] = {
                'mean': mean(counts),
                'stdev': stdev(counts) if len(counts) > 1 else 0,
                'total_months': len(counts),
                'total_emails': sum(counts),
            }
        elif len(counts) == 1 and counts[0] >= MIN_EMAILS_FOR_DOMAIN:
            # Single month with significant activity
            baselines[domain] = {
                'mean': counts[0],
                'stdev': 0,
                'total_months': 1,
                'total_emails': counts[0],
            }

    return baselines


def detect_spikes(activity: list[dict], baselines: dict) -> list[dict]:
    """Find months where domain activity spikes above baseline."""
    spikes = []

    for row in activity:
        domain = row['domain']
        if domain in IGNORE_DOMAINS or domain not in baselines:
            continue

        baseline = baselines[domain]
        count = row['email_count']

        # Spike detection: count > baseline_mean * threshold
        if count >= baseline['mean'] * SPIKE_THRESHOLD and count >= MIN_EMAILS_FOR_DOMAIN:
            deviation = (count - baseline['mean']) / baseline['mean'] if baseline['mean'] > 0 else 0
            spikes.append({
                'month': row['month'],
                'domain': domain,
                'email_count': count,
                'baseline_mean': baseline['mean'],
                'deviation_factor': deviation,
                'sample_subjects': row['sample_subjects'][:5] if row['sample_subjects'] else [],
            })

    return sorted(spikes, key=lambda x: (x['month'], -x['deviation_factor']))


def group_into_contexts(spikes: list[dict]) -> list[dict]:
    """Group consecutive spike months by domain into contexts."""
    if not spikes:
        return []

    # Group spikes by domain
    domain_spikes = defaultdict(list)
    for spike in spikes:
        domain_spikes[spike['domain']].append(spike)

    contexts = []
    for domain, domain_spike_list in domain_spikes.items():
        # Sort by month
        sorted_spikes = sorted(domain_spike_list, key=lambda x: x['month'])

        # Group consecutive months
        groups = []
        current_group = [sorted_spikes[0]]

        for spike in sorted_spikes[1:]:
            prev_month = current_group[-1]['month']
            curr_month = spike['month']

            # Calculate gap in months
            gap_months = (curr_month.year - prev_month.year) * 12 + (curr_month.month - prev_month.month)

            if gap_months <= MAX_GAP_MONTHS:
                current_group.append(spike)
            else:
                if len(current_group) >= MIN_SPIKE_MONTHS:
                    groups.append(current_group)
                current_group = [spike]

        # Don't forget last group
        if len(current_group) >= MIN_SPIKE_MONTHS:
            groups.append(current_group)

        # Create context from each group
        for group in groups:
            ctx = create_context_from_group(domain, group)
            if ctx:
                contexts.append(ctx)

    return sorted(contexts, key=lambda x: x['week_start'])


def create_context_from_group(domain: str, group: list[dict]) -> dict:
    """Create a priority context from a group of spike months."""
    total_emails = sum(s['email_count'] for s in group)
    avg_deviation = mean(s['deviation_factor'] for s in group)
    avg_baseline = mean(s['baseline_mean'] for s in group)

    # Get sample subjects
    all_subjects = []
    for s in group:
        all_subjects.extend(s.get('sample_subjects', []))

    # Extract keywords from subjects
    keywords = extract_keywords(all_subjects)

    # Determine category
    category = DOMAIN_CATEGORIES.get(domain, 'General')

    start_month = min(s['month'] for s in group)
    end_month = max(s['month'] for s in group)

    return {
        'week_start': start_month.date(),  # Using week_start for month (per schema)
        'email_count': total_emails,
        'avg_response_hours': None,  # Not applicable for this detection method
        'baseline_hours': avg_baseline,  # Repurposing for baseline email count
        'deviation_factor': avg_deviation,
        'keywords': keywords + [domain],  # Include domain as keyword
        'key_participants': [domain],
        'domain': domain,
        'category': category,
        'duration_months': len(group),
        'start_month': start_month,
        'end_month': end_month,
    }


def extract_keywords(subjects: list[str], top_n: int = 5) -> list[str]:
    """Extract common keywords from email subjects."""
    if not subjects:
        return []

    import re
    from collections import Counter

    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'your', 'my', 'our', 'their', 'this', 'that', 'these', 'those',
        're', 'fwd', 'fw', 'you', 'i', 'we', 'they', 'it', 'he', 'she',
        'new', 'please', 'thanks', 'thank', 'hi', 'hello', 'hey',
    }

    word_counts = Counter()
    for subject in subjects:
        if not subject:
            continue
        words = re.findall(r'\b[a-zA-Z]{3,}\b', subject.lower())
        words = [w for w in words if w not in stop_words]
        word_counts.update(words)

    return [word for word, _ in word_counts.most_common(top_n)]


async def save_context(conn: asyncpg.Connection, context: dict) -> int:
    """Save a priority context to the database."""
    context_id = await conn.fetchval("""
        INSERT INTO priority_contexts (
            week_start, email_count, avg_response_hours, baseline_hours,
            deviation_factor, keywords, key_participants, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        RETURNING id
    """,
        context['week_start'],
        context['email_count'],
        context['avg_response_hours'],
        context['baseline_hours'],
        context['deviation_factor'],
        context['keywords'],
        context['key_participants'],
    )
    return context_id


async def main(dry_run: bool = False):
    """Main entry point."""
    print("Detect Priority Contexts via Participant Bursts")
    print("=" * 50)

    if dry_run:
        print("\n[DRY RUN - no changes will be made]\n")

    print(f"Connecting to database...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Get monthly domain activity
        print("\nAnalyzing monthly email activity by sender domain...")
        activity = await get_monthly_domain_activity(conn)
        print(f"Found {len(activity)} (month, domain) combinations")

        if not activity:
            print("No data to analyze")
            return

        # Calculate baselines
        print("\nCalculating domain baselines...")
        baselines = calculate_domain_baselines(activity)
        print(f"Computed baselines for {len(baselines)} domains")

        # Detect spikes
        print("\nDetecting activity spikes...")
        spikes = detect_spikes(activity, baselines)
        print(f"Found {len(spikes)} spike events")

        if not spikes:
            print("No spikes detected")
            return

        # Group into contexts
        print("\nGrouping into contexts...")
        contexts = group_into_contexts(spikes)
        print(f"Created {len(contexts)} priority contexts")

        if not contexts:
            print("No contexts formed")
            return

        # Display contexts
        print("\n" + "=" * 60)
        print("Discovered Priority Contexts (Participant Bursts):")
        print("=" * 60)

        for ctx in contexts:
            print(f"\n{ctx['domain']} ({ctx['category']})")
            print(f"  Period: {ctx['start_month'].strftime('%Y-%m')} to {ctx['end_month'].strftime('%Y-%m')}")
            print(f"  Emails: {ctx['email_count']} ({ctx['deviation_factor']:.1f}x baseline)")
            if ctx['keywords']:
                print(f"  Keywords: {', '.join(ctx['keywords'][:5])}")

        if dry_run:
            print("\n[DRY RUN - stopping here]")
            return

        # Save contexts
        print("\nSaving contexts to database...")
        for ctx in contexts:
            ctx_id = await save_context(conn, ctx)
            print(f"  Created: {ctx['domain'][:30]}... (id={ctx_id})")

        # Verification
        print("\n=== Verification ===")
        count = await conn.fetchval("SELECT COUNT(*) FROM priority_contexts")
        print(f"Total priority contexts: {count}")

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect priority contexts via participant bursts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without making changes",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
