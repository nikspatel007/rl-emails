#!/usr/bin/env python3
"""
Detect priority contexts via response time analysis.

This script analyzes email response times to find periods of heightened
engagement - weeks/months where response time was significantly faster
than average, indicating high-priority work periods.

Detection approach:
1. Calculate weekly average response times
2. Find weeks where response < 50% of overall average
3. Group consecutive fast weeks into contexts
4. Extract key participants and common keywords from those periods
5. Write discovered contexts to priority_contexts table

Usage:
    python scripts/detect_priority_contexts.py [--dry-run]
"""

import argparse
import asyncio
from collections import Counter
from datetime import timedelta
import re

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Detection thresholds
FAST_RESPONSE_THRESHOLD = 0.5  # Response < 50% of average = fast
MIN_EMAILS_PER_WEEK = 3  # Minimum emails to consider a week
MAX_RESPONSE_SECONDS = 86400 * 7  # Ignore responses > 1 week (outliers)
MIN_CONTEXT_WEEKS = 2  # Minimum weeks to form a context
MAX_GAP_WEEKS = 2  # Maximum gap between weeks to still group together
PRIORITY_BOOST_BASE = 1.5  # Base priority boost for fast-response contexts


async def get_weekly_response_times(conn: asyncpg.Connection) -> list[dict]:
    """Get weekly average response times."""
    rows = await conn.fetch("""
        WITH weekly_stats AS (
            SELECT
                date_trunc('week', date_parsed) as week,
                AVG(response_time_seconds) / 3600 as avg_hours,
                COUNT(*) as email_count,
                array_agg(DISTINCT from_email) as senders,
                array_agg(DISTINCT subject) as subjects
            FROM emails
            WHERE response_time_seconds IS NOT NULL
              AND response_time_seconds > 0
              AND response_time_seconds < $1
              AND NOT is_sent
              AND date_parsed IS NOT NULL
            GROUP BY week
            HAVING COUNT(*) >= $2
        )
        SELECT * FROM weekly_stats ORDER BY week
    """, MAX_RESPONSE_SECONDS, MIN_EMAILS_PER_WEEK)
    return [dict(row) for row in rows]


def extract_keywords(subjects: list[str], top_n: int = 5) -> list[str]:
    """Extract common keywords from email subjects."""
    if not subjects:
        return []

    # Common words to filter out
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
        # Extract words, lowercase, filter short and stop words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', subject.lower())
        words = [w for w in words if w not in stop_words and len(w) > 2]
        word_counts.update(words)

    return [word for word, _ in word_counts.most_common(top_n)]


def extract_key_participants(senders: list[str], top_n: int = 5) -> list[str]:
    """Extract most frequent senders, filtering generic domains."""
    if not senders:
        return []

    generic_domains = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'googlemail.com', 'aol.com', 'icloud.com', 'me.com',
    }

    # Filter out generic email domains and count
    filtered = []
    for sender in senders:
        if not sender or '@' not in sender:
            continue
        domain = sender.split('@')[1].lower()
        if domain not in generic_domains:
            filtered.append(sender)

    if not filtered:
        # Fall back to all senders if all are generic
        filtered = [s for s in senders if s]

    return Counter(filtered).most_common(top_n)


def group_consecutive_weeks(fast_weeks: list[dict]) -> list[list[dict]]:
    """Group consecutive fast weeks into contexts."""
    if not fast_weeks:
        return []

    groups = []
    current_group = [fast_weeks[0]]

    for week in fast_weeks[1:]:
        prev_week = current_group[-1]['week']
        curr_week = week['week']
        gap = (curr_week - prev_week).days // 7

        if gap <= MAX_GAP_WEEKS:
            current_group.append(week)
        else:
            if len(current_group) >= MIN_CONTEXT_WEEKS:
                groups.append(current_group)
            current_group = [week]

    # Don't forget the last group
    if len(current_group) >= MIN_CONTEXT_WEEKS:
        groups.append(current_group)

    return groups


def create_context_from_group(group: list[dict], overall_avg: float) -> dict:
    """Create a priority context from a group of fast weeks."""
    # Collect all data
    all_senders = []
    all_subjects = []
    total_emails = 0
    total_hours = 0

    for week in group:
        all_senders.extend(week['senders'] or [])
        all_subjects.extend(week['subjects'] or [])
        total_emails += week['email_count']
        total_hours += week['avg_hours'] * week['email_count']

    avg_response = total_hours / total_emails if total_emails > 0 else 0

    # Calculate priority boost based on how much faster than average
    speed_ratio = avg_response / overall_avg if overall_avg > 0 else 1
    priority_boost = PRIORITY_BOOST_BASE + (1 - speed_ratio)  # Faster = higher boost

    # Extract context details
    keywords = extract_keywords(all_subjects)
    key_participants = [p[0] for p in extract_key_participants(all_senders)]

    # Generate name
    start_date = group[0]['week']
    end_date = group[-1]['week'] + timedelta(days=6)

    if keywords:
        name = f"High Priority: {keywords[0].title()}"
        if len(keywords) > 1:
            name += f" / {keywords[1].title()}"
    else:
        name = f"Fast Response Period"

    name += f" ({start_date.strftime('%b %Y')})"

    return {
        'name': name[:100],  # Limit length
        'context_type': 'professional',
        'started_at': start_date,
        'ended_at': end_date,
        'priority_boost': round(priority_boost, 2),
        'keywords': keywords,
        'key_participants': key_participants[:5],
        'description': (
            f"Period of heightened email engagement. "
            f"Avg response: {avg_response:.1f}h (vs {overall_avg:.1f}h overall). "
            f"{total_emails} emails over {len(group)} weeks."
        ),
    }


async def save_context(conn: asyncpg.Connection, context: dict) -> int:
    """Save a priority context to the database."""
    context_id = await conn.fetchval("""
        INSERT INTO priority_contexts (
            name, context_type, started_at, ended_at,
            priority_boost, keywords, key_participants, description,
            created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        RETURNING id
    """,
        context['name'],
        context['context_type'],
        context['started_at'],
        context['ended_at'],
        context['priority_boost'],
        context['keywords'],
        context['key_participants'],
        context['description'],
    )
    return context_id


async def main(dry_run: bool = False):
    """Main entry point."""
    print("Detect Priority Contexts via Response Time Analysis")
    print("=" * 55)

    if dry_run:
        print("\n[DRY RUN - no changes will be made]\n")

    print(f"Connecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Get weekly response times
        print("\nAnalyzing weekly response times...")
        weeks = await get_weekly_response_times(conn)
        print(f"Found {len(weeks)} weeks with sufficient data")

        if not weeks:
            print("No data to analyze")
            return

        # Calculate overall average (convert Decimal to float)
        for w in weeks:
            w['avg_hours'] = float(w['avg_hours']) if w['avg_hours'] else 0.0
        total_hours = sum(w['avg_hours'] * w['email_count'] for w in weeks)
        total_emails = sum(w['email_count'] for w in weeks)
        overall_avg = total_hours / total_emails if total_emails > 0 else 0
        print(f"Overall average response time: {overall_avg:.1f} hours")

        # Find fast-response weeks
        threshold = overall_avg * FAST_RESPONSE_THRESHOLD
        fast_weeks = [w for w in weeks if w['avg_hours'] < threshold]
        print(f"Found {len(fast_weeks)} fast-response weeks (< {threshold:.1f}h)")

        if not fast_weeks:
            print("No fast-response periods detected")
            return

        # Group consecutive weeks
        groups = group_consecutive_weeks(fast_weeks)
        print(f"Grouped into {len(groups)} distinct contexts")

        if not groups:
            print("No contexts with sufficient duration")
            return

        # Create contexts
        contexts = [create_context_from_group(g, overall_avg) for g in groups]

        # Display contexts
        print("\n" + "=" * 60)
        print("Discovered Priority Contexts:")
        print("=" * 60)

        for ctx in contexts:
            print(f"\n{ctx['name']}")
            print(f"  Period: {ctx['started_at'].strftime('%Y-%m-%d')} to {ctx['ended_at'].strftime('%Y-%m-%d')}")
            print(f"  Priority Boost: {ctx['priority_boost']}x")
            if ctx['keywords']:
                print(f"  Keywords: {', '.join(ctx['keywords'])}")
            if ctx['key_participants']:
                print(f"  Key Participants: {len(ctx['key_participants'])} people")

        if dry_run:
            print("\n[DRY RUN - stopping here]")
            return

        # Save contexts
        print("\nSaving contexts to database...")
        for ctx in contexts:
            ctx_id = await save_context(conn, ctx)
            print(f"  Created: {ctx['name'][:50]}... (id={ctx_id})")

        # Verification
        print("\n=== Verification ===")
        count = await conn.fetchval("SELECT COUNT(*) FROM priority_contexts")
        print(f"Total priority contexts: {count}")

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect priority contexts via response time analysis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without making changes",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
