#!/usr/bin/env python3
"""Extract implicit preference pairs from Gmail SurrealDB.

Generates preference pairs from behavioral signals in Gmail data:
1. Response time: Quick reply (< 1 hour) beats slow reply
2. Labels: Starred/Important emails beat plain emails
3. Actions: reply > forward > archive > delete
4. Read latency: Immediate read beats delayed read (if available)

Usage:
    # Start Gmail database first:
    ./scripts/start_db.sh gmail

    # Extract preferences:
    python scripts/extract_implicit_preferences.py -o data/preferences_implicit.json

    # With limit for testing:
    python scripts/extract_implicit_preferences.py -n 1000 -o data/preferences_test.json
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from surrealdb import AsyncSurreal


# Priority signals - higher is better
LABEL_PRIORITY = {
    'STARRED': 3,
    'IMPORTANT': 2,
    'CATEGORY_PERSONAL': 1,
    'CATEGORY_UPDATES': 0,
    'CATEGORY_SOCIAL': 0,
    'CATEGORY_PROMOTIONS': -1,
    'CATEGORY_FORUMS': -1,
    'SPAM': -3,
    'TRASH': -3,
}

# Action ranking - higher index = higher priority action
ACTION_RANKING = {
    'DELETE': 0,
    'delete': 0,
    'ARCHIVE': 1,
    'archive': 1,
    'FORWARD': 2,
    'forward': 2,
    'REPLY_LATER': 3,
    'reply_later': 3,
    'REPLY_NOW': 4,
    'reply_now': 4,
}

# Response time thresholds in seconds
QUICK_RESPONSE_THRESHOLD = 3600  # 1 hour
VERY_QUICK_RESPONSE_THRESHOLD = 900  # 15 minutes


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse email date string to datetime."""
    if not date_str:
        return None

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S",
        "%a %b %d %H:%M:%S %Y",
    ]

    for tz in ['(PDT)', '(PST)', '(EDT)', '(EST)', '(CDT)', '(CST)', '(MDT)', '(MST)', '(UTC)']:
        date_str = date_str.replace(f' {tz}', '')

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


def compute_label_priority(labels: list[str]) -> int:
    """Compute priority score from labels."""
    if not labels:
        return 0
    return max(LABEL_PRIORITY.get(label.upper(), 0) for label in labels)


def compute_action_rank(action: str) -> int:
    """Get ranking for an action."""
    return ACTION_RANKING.get(action, 0)


class PreferenceExtractor:
    """Extract implicit preference pairs from Gmail SurrealDB."""

    def __init__(
        self,
        url: str = 'ws://localhost:8001/rpc',
        namespace: str = 'rl_emails',
        database: str = 'gmail',
    ):
        self.url = url
        self.namespace = namespace
        self.database = database
        self.db: AsyncSurreal = None
        self.emails = []
        self.threads = {}

    async def connect(self, username: str = 'root', password: str = 'root'):
        """Connect to SurrealDB."""
        self.db = AsyncSurreal(self.url)
        await self.db.connect()
        await self.db.signin({'username': username, 'password': password})
        await self.db.use(self.namespace, self.database)

    async def load_emails(self, limit: Optional[int] = None):
        """Load emails from SurrealDB."""
        query = '''
            SELECT
                id,
                message_id,
                subject,
                from_email,
                to_emails,
                date_str,
                labels,
                action,
                gmail_thread_id,
                in_reply_to
            FROM emails
            WHERE action IS NOT NONE
        '''
        if limit:
            query += f' LIMIT {limit}'

        result = await self.db.query(query)
        if result and isinstance(result, list):
            self.emails = result
            print(f"Loaded {len(self.emails)} emails with actions")
        else:
            self.emails = []
            print("No emails found")

    async def load_response_times(self) -> dict[str, float]:
        """Load response times by matching replies to originals.

        Returns:
            Dict mapping message_id -> response_time_seconds
        """
        # Build message index - index both with and without angle brackets
        msg_index = {}
        for email in self.emails:
            msg_id = email.get('message_id', '')
            if msg_id:
                # Store with both formats for easier matching
                msg_index[msg_id] = email
                msg_index[msg_id.strip('<>')] = email

        response_times = {}

        # Find replies and compute response times
        for email in self.emails:
            in_reply_to = email.get('in_reply_to')
            if not in_reply_to:
                continue

            # Try to find original with various ID formats
            in_reply_to_clean = in_reply_to.strip('<>')
            original = msg_index.get(in_reply_to_clean) or msg_index.get(in_reply_to) or msg_index.get(f"<{in_reply_to_clean}>")
            if not original:
                continue

            # Get dates - try both date and date_str fields
            reply_date = parse_date(email.get('date_str', '') or email.get('date', ''))
            original_date = parse_date(original.get('date_str', '') or original.get('date', ''))

            if reply_date and original_date:
                try:
                    # Make both timezone-naive for comparison
                    if reply_date.tzinfo:
                        reply_date = reply_date.replace(tzinfo=None)
                    if original_date.tzinfo:
                        original_date = original_date.replace(tzinfo=None)

                    delta = (reply_date - original_date).total_seconds()
                    if delta > 0:
                        response_times[email.get('message_id')] = delta
                except Exception:
                    pass

        print(f"Computed {len(response_times)} response times")
        return response_times

    async def build_thread_index(self):
        """Build index of emails by thread."""
        for email in self.emails:
            thread_id = email.get('gmail_thread_id')
            if not thread_id:
                continue

            if thread_id not in self.threads:
                self.threads[thread_id] = []
            self.threads[thread_id].append(email)

        print(f"Found {len(self.threads)} threads")

    def extract_response_time_pairs(
        self,
        response_times: dict[str, float],
        max_pairs: int = 3000
    ) -> list[dict]:
        """Extract preference pairs based on response time.

        Quick responses are preferred over slow responses.
        """
        pairs = []

        # Get emails with response times
        timed_emails = [
            (email, response_times[email.get('message_id')])
            for email in self.emails
            if email.get('message_id') in response_times
        ]

        if len(timed_emails) < 2:
            return pairs

        # Sort by response time (fastest first)
        timed_emails.sort(key=lambda x: x[1])

        # Sample pairs - quick vs slow
        quick_emails = [e for e, t in timed_emails if t < QUICK_RESPONSE_THRESHOLD]
        slow_emails = [e for e, t in timed_emails if t >= QUICK_RESPONSE_THRESHOLD]

        if quick_emails and slow_emails:
            for _ in range(min(max_pairs, len(quick_emails) * len(slow_emails))):
                quick = random.choice(quick_emails)
                slow = random.choice(slow_emails)
                quick_time = response_times.get(quick.get('message_id'), 0)
                slow_time = response_times.get(slow.get('message_id'), 0)

                pairs.append({
                    'chosen_id': quick.get('message_id'),
                    'rejected_id': slow.get('message_id'),
                    'signal_type': 'response_time',
                    'confidence': min(0.95, 0.5 + (slow_time - quick_time) / (24 * 3600)),
                    'metadata': {
                        'chosen_response_time_s': quick_time,
                        'rejected_response_time_s': slow_time,
                    }
                })

        # Also pair very quick vs somewhat quick
        very_quick = [e for e, t in timed_emails if t < VERY_QUICK_RESPONSE_THRESHOLD]
        somewhat_quick = [
            e for e, t in timed_emails
            if VERY_QUICK_RESPONSE_THRESHOLD <= t < QUICK_RESPONSE_THRESHOLD
        ]

        if very_quick and somewhat_quick:
            for _ in range(min(max_pairs // 3, len(very_quick) * len(somewhat_quick))):
                fast = random.choice(very_quick)
                medium = random.choice(somewhat_quick)
                pairs.append({
                    'chosen_id': fast.get('message_id'),
                    'rejected_id': medium.get('message_id'),
                    'signal_type': 'response_time_granular',
                    'confidence': 0.6,
                    'metadata': {
                        'chosen_response_time_s': response_times.get(fast.get('message_id'), 0),
                        'rejected_response_time_s': response_times.get(medium.get('message_id'), 0),
                    }
                })

        return pairs[:max_pairs]

    def extract_label_pairs(self, max_pairs: int = 3000) -> list[dict]:
        """Extract preference pairs based on labels.

        Starred/Important emails are preferred.
        """
        pairs = []

        # Group by label priority
        by_priority = {}
        for email in self.emails:
            labels = email.get('labels', [])
            priority = compute_label_priority(labels)
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(email)

        priorities = sorted(by_priority.keys(), reverse=True)

        # Create pairs between different priority levels
        for i, high_p in enumerate(priorities):
            for low_p in priorities[i + 1:]:
                high_emails = by_priority[high_p]
                low_emails = by_priority[low_p]

                # Confidence based on priority difference
                confidence = min(0.9, 0.5 + 0.15 * (high_p - low_p))

                for _ in range(min(max_pairs // len(priorities), len(high_emails) * len(low_emails))):
                    high = random.choice(high_emails)
                    low = random.choice(low_emails)

                    pairs.append({
                        'chosen_id': high.get('message_id'),
                        'rejected_id': low.get('message_id'),
                        'signal_type': 'label_priority',
                        'confidence': confidence,
                        'metadata': {
                            'chosen_labels': high.get('labels', []),
                            'rejected_labels': low.get('labels', []),
                            'chosen_priority': high_p,
                            'rejected_priority': low_p,
                        }
                    })

        return pairs[:max_pairs]

    def extract_action_pairs(self, max_pairs: int = 3000) -> list[dict]:
        """Extract preference pairs based on user actions.

        Reply > Forward > Archive > Delete
        """
        pairs = []

        # Group by action rank
        by_action = {}
        for email in self.emails:
            action = email.get('action')
            if not action:
                continue
            rank = compute_action_rank(action)
            if rank not in by_action:
                by_action[rank] = []
            by_action[rank].append(email)

        ranks = sorted(by_action.keys(), reverse=True)

        # Create pairs between different action ranks
        for i, high_rank in enumerate(ranks):
            for low_rank in ranks[i + 1:]:
                high_emails = by_action[high_rank]
                low_emails = by_action[low_rank]

                # Confidence based on rank difference
                confidence = min(0.95, 0.6 + 0.1 * (high_rank - low_rank))

                for _ in range(min(max_pairs // len(ranks), len(high_emails) * len(low_emails))):
                    high = random.choice(high_emails)
                    low = random.choice(low_emails)

                    pairs.append({
                        'chosen_id': high.get('message_id'),
                        'rejected_id': low.get('message_id'),
                        'signal_type': 'action_preference',
                        'confidence': confidence,
                        'metadata': {
                            'chosen_action': high.get('action'),
                            'rejected_action': low.get('action'),
                        }
                    })

        return pairs[:max_pairs]

    def extract_thread_position_pairs(self, max_pairs: int = 2000) -> list[dict]:
        """Extract preference pairs based on thread position.

        First response in a thread indicates higher priority.
        """
        pairs = []

        for thread_id, emails in self.threads.items():
            if len(emails) < 2:
                continue

            # Sort by date
            sorted_emails = sorted(
                emails,
                key=lambda e: parse_date(e.get('date_str', '')) or datetime.min
            )

            # First responder gets priority over later responders
            first = sorted_emails[0]
            for later in sorted_emails[1:]:
                pairs.append({
                    'chosen_id': first.get('message_id'),
                    'rejected_id': later.get('message_id'),
                    'signal_type': 'thread_position',
                    'confidence': 0.55,  # Lower confidence - position is weak signal
                    'metadata': {
                        'thread_id': thread_id,
                        'chosen_position': 0,
                        'rejected_position': sorted_emails.index(later),
                    }
                })

        return pairs[:max_pairs]

    async def extract_all_preferences(
        self,
        limit: Optional[int] = None,
        target_pairs: int = 10000
    ) -> list[dict]:
        """Extract all preference pairs from Gmail data.

        Args:
            limit: Limit emails to load (for testing)
            target_pairs: Target total number of pairs

        Returns:
            List of preference pair dicts
        """
        print("Loading emails...")
        await self.load_emails(limit)

        if not self.emails:
            print("No emails found. Make sure Gmail data is imported to SurrealDB.")
            return []

        print("Computing response times...")
        response_times = await self.load_response_times()

        print("Building thread index...")
        await self.build_thread_index()

        # Allocate pairs across signal types
        per_signal = target_pairs // 4

        print(f"\nExtracting preference pairs (target: {target_pairs})...")

        print("  Extracting response time pairs...")
        response_pairs = self.extract_response_time_pairs(response_times, per_signal)
        print(f"    -> {len(response_pairs)} pairs")

        print("  Extracting label pairs...")
        label_pairs = self.extract_label_pairs(per_signal)
        print(f"    -> {len(label_pairs)} pairs")

        print("  Extracting action pairs...")
        action_pairs = self.extract_action_pairs(per_signal)
        print(f"    -> {len(action_pairs)} pairs")

        print("  Extracting thread position pairs...")
        thread_pairs = self.extract_thread_position_pairs(per_signal)
        print(f"    -> {len(thread_pairs)} pairs")

        # Combine and deduplicate
        all_pairs = response_pairs + label_pairs + action_pairs + thread_pairs

        # Remove duplicates based on chosen/rejected pair
        seen = set()
        unique_pairs = []
        for pair in all_pairs:
            key = (pair['chosen_id'], pair['rejected_id'])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)

        print(f"\nTotal unique pairs: {len(unique_pairs)}")

        # Shuffle and trim to target
        random.shuffle(unique_pairs)
        final_pairs = unique_pairs[:target_pairs]

        # Print distribution
        signal_counts = {}
        for pair in final_pairs:
            sig = pair['signal_type']
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        print("\nSignal type distribution:")
        for sig, count in sorted(signal_counts.items()):
            pct = 100 * count / len(final_pairs) if final_pairs else 0
            print(f"  {sig}: {count} ({pct:.1f}%)")

        return final_pairs

    async def close(self):
        """Close database connection."""
        if self.db:
            await self.db.close()


async def main():
    parser = argparse.ArgumentParser(
        description='Extract implicit preference pairs from Gmail SurrealDB'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('data/preferences_implicit.json'),
        help='Output JSON file (default: data/preferences_implicit.json)'
    )
    parser.add_argument(
        '-n', '--limit',
        type=int,
        help='Limit emails to load (for testing)'
    )
    parser.add_argument(
        '--target-pairs',
        type=int,
        default=10000,
        help='Target number of preference pairs (default: 10000)'
    )
    parser.add_argument(
        '--url',
        default='ws://localhost:8001/rpc',
        help='SurrealDB connection URL'
    )
    parser.add_argument(
        '--user',
        default='root',
        help='SurrealDB username'
    )
    parser.add_argument(
        '--password',
        default='root',
        help='SurrealDB password'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    random.seed(args.seed)

    extractor = PreferenceExtractor(url=args.url)

    try:
        print(f"Connecting to SurrealDB at {args.url}...")
        await extractor.connect(args.user, args.password)

        pairs = await extractor.extract_all_preferences(
            limit=args.limit,
            target_pairs=args.target_pairs,
        )

        if pairs:
            args.output.parent.mkdir(parents=True, exist_ok=True)

            # Compute summary stats
            confidence_avg = sum(p['confidence'] for p in pairs) / len(pairs)

            output_data = {
                'metadata': {
                    'total_pairs': len(pairs),
                    'target_pairs': args.target_pairs,
                    'avg_confidence': round(confidence_avg, 3),
                    'seed': args.seed,
                    'generated_at': datetime.now().isoformat(),
                },
                'pairs': pairs
            }

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"\nWrote {len(pairs)} preference pairs to {args.output}")
        else:
            print("\nNo preference pairs extracted.")
            sys.exit(1)

    except ConnectionRefusedError:
        print(f"Error: Could not connect to SurrealDB at {args.url}")
        print("Start the Gmail database first: ./scripts/start_db.sh gmail")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await extractor.close()


if __name__ == '__main__':
    asyncio.run(main())
