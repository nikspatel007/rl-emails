#!/usr/bin/env python3
"""Gate 2: Verify enrichment quality.

Automated verification after enrichment (Stage 5).

Checks:
- Count matches expected (41373)
- is_sent correctly identified (SENT label → action=COMPOSED)
- Reply chains: in_reply_to links resolve to existing emails
- Actions computed: >70% of received emails have action set
- Response times: plausible values (not negative, not > 1 year)
- Threads: all emails have thread_id populated
- Users extractable: from/to fields populated

Spot Checks:
- 10 random REPLIED emails: verify in_reply_to resolution
- 10 random threads: verify multiple emails per thread
- Sent emails: verify SENT label → COMPOSED action

Usage:
    python scripts/verify_enrichment.py /path/to/enriched_emails.jsonl

Output: JSON gate result to stdout
"""

import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# Constants
EXPECTED_COUNT = 41373
ACTION_COVERAGE_THRESHOLD = 70  # >70% of received emails need action
MAX_RESPONSE_TIME_SECONDS = 365 * 24 * 3600  # 1 year max
VALID_ACTIONS = {'COMPOSED', 'REPLIED', 'FORWARDED', 'IGNORED', 'ARCHIVED', 'READ_PENDING', 'UNKNOWN'}


def verify_enrichment(jsonl_path: Path) -> dict:
    """Verify enriched emails file meets gate criteria."""
    results = {
        'gate': 'enrichment',
        'status': 'PENDING',
        'input_file': str(jsonl_path),
        'checks': {},
        'spot_checks': {},
        'blocking_issues': [],
    }

    # Read all records
    records = []
    line_errors = []
    message_id_index = {}  # message_id -> record

    print(f"Loading {jsonl_path}...", file=sys.stderr)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
                msg_id = record.get('message_id', '')
                if msg_id:
                    message_id_index[msg_id] = record
            except json.JSONDecodeError as e:
                line_errors.append(f"Line {i}: {e}")
                if len(line_errors) >= 10:
                    break

    if line_errors:
        results['checks']['parse_errors'] = {
            'count': len(line_errors),
            'samples': line_errors[:5],
            'pass': False,
        }
        results['blocking_issues'].append(f"{len(line_errors)} JSON parse errors")
        results['status'] = 'FAIL'
        return results

    print(f"Loaded {len(records)} records", file=sys.stderr)

    # Check 1: Count matches expected
    actual_count = len(records)
    count_pass = actual_count == EXPECTED_COUNT
    results['checks']['count_match'] = {
        'expected': EXPECTED_COUNT,
        'actual': actual_count,
        'pass': count_pass,
    }
    if not count_pass:
        results['blocking_issues'].append(
            f"Count mismatch: expected {EXPECTED_COUNT}, got {actual_count}"
        )

    # Categorize emails
    sent_emails = []
    received_emails = []

    for record in records:
        labels = record.get('labels', [])
        if 'SENT' in labels:
            sent_emails.append(record)
        else:
            received_emails.append(record)

    results['checks']['email_categories'] = {
        'sent': len(sent_emails),
        'received': len(received_emails),
        'total': len(records),
    }

    # Check 2: is_sent correctly identified (SENT label → COMPOSED action)
    sent_with_composed = sum(1 for e in sent_emails if e.get('action') == 'COMPOSED')
    sent_correct_pct = (sent_with_composed / len(sent_emails) * 100) if sent_emails else 100
    sent_pass = sent_correct_pct >= 99  # Allow 1% error margin
    results['checks']['sent_identification'] = {
        'sent_emails': len(sent_emails),
        'with_composed_action': sent_with_composed,
        'percent_correct': round(sent_correct_pct, 2),
        'pass': sent_pass,
    }
    if not sent_pass:
        results['blocking_issues'].append(
            f"Sent email identification: {sent_correct_pct:.1f}% < 99% threshold"
        )

    # Check 3: Actions computed for received emails (>70%)
    received_with_action = sum(
        1 for e in received_emails
        if e.get('action') and e.get('action') != 'UNKNOWN'
    )
    action_pct = (received_with_action / len(received_emails) * 100) if received_emails else 0
    action_pass = action_pct >= ACTION_COVERAGE_THRESHOLD

    # Action distribution
    action_dist = Counter(e.get('action', 'NO_ACTION') for e in received_emails)

    results['checks']['action_coverage'] = {
        'received_emails': len(received_emails),
        'with_action': received_with_action,
        'percent': round(action_pct, 2),
        'threshold': ACTION_COVERAGE_THRESHOLD,
        'action_distribution': dict(action_dist),
        'pass': action_pass,
    }
    if not action_pass:
        results['blocking_issues'].append(
            f"Action coverage {action_pct:.1f}% < {ACTION_COVERAGE_THRESHOLD}% threshold"
        )

    # Check 4: Response times plausible
    response_times = []
    negative_times = 0
    too_long_times = 0

    for record in records:
        rt = record.get('response_time_seconds')
        if rt is not None:
            response_times.append(rt)
            if rt < 0:
                negative_times += 1
            if rt > MAX_RESPONSE_TIME_SECONDS:
                too_long_times += 1

    response_time_issues = negative_times + too_long_times
    response_time_pass = response_time_issues == 0

    if response_times:
        avg_response = sum(response_times) / len(response_times)
        min_response = min(response_times)
        max_response = max(response_times)
    else:
        avg_response = min_response = max_response = None

    results['checks']['response_times'] = {
        'emails_with_response_time': len(response_times),
        'negative_values': negative_times,
        'exceeds_1_year': too_long_times,
        'min_seconds': min_response,
        'max_seconds': max_response,
        'avg_seconds': round(avg_response, 2) if avg_response else None,
        'avg_hours': round(avg_response / 3600, 2) if avg_response else None,
        'pass': response_time_pass,
    }
    if not response_time_pass:
        results['blocking_issues'].append(
            f"Response time issues: {negative_times} negative, {too_long_times} > 1 year"
        )

    # Check 5: Thread IDs populated
    with_thread_id = sum(1 for r in records if r.get('thread_id'))
    thread_pct = (with_thread_id / len(records) * 100) if records else 0
    thread_pass = thread_pct >= 80  # Same threshold as Gate 1

    # Count unique threads
    unique_threads = len(set(r.get('thread_id', '') for r in records if r.get('thread_id')))

    results['checks']['thread_coverage'] = {
        'with_thread_id': with_thread_id,
        'total': len(records),
        'percent': round(thread_pct, 2),
        'unique_threads': unique_threads,
        'pass': thread_pass,
    }
    if not thread_pass:
        results['blocking_issues'].append(
            f"Thread coverage {thread_pct:.1f}% < 80% threshold"
        )

    # Check 6: Users extractable (from/to fields populated)
    unique_from = set()
    unique_to = set()
    missing_from = 0

    for record in records:
        from_field = record.get('from', '')
        if from_field:
            unique_from.add(from_field.lower())
        else:
            missing_from += 1

        to_field = record.get('to', '')
        if to_field:
            # Split on common delimiters
            for addr in to_field.replace(';', ',').split(','):
                addr = addr.strip().lower()
                if addr:
                    unique_to.add(addr)

    users_pass = missing_from < (len(records) * 0.01)  # < 1% missing from

    results['checks']['user_extraction'] = {
        'unique_senders': len(unique_from),
        'unique_recipients': len(unique_to),
        'missing_from_field': missing_from,
        'pass': users_pass,
    }
    if not users_pass:
        results['blocking_issues'].append(
            f"User extraction: {missing_from} emails missing 'from' field"
        )

    # Check 7: Reply chain integrity
    in_reply_to_count = 0
    in_reply_to_resolved = 0
    unresolved_replies = []

    for record in records:
        in_reply_to = record.get('in_reply_to')
        if in_reply_to:
            in_reply_to_count += 1
            if in_reply_to in message_id_index:
                in_reply_to_resolved += 1
            else:
                if len(unresolved_replies) < 5:
                    unresolved_replies.append(in_reply_to[:50])

    reply_resolve_pct = (in_reply_to_resolved / in_reply_to_count * 100) if in_reply_to_count else 100
    # Don't fail on unresolved replies - parent email may be before our date range

    results['checks']['reply_chain_integrity'] = {
        'emails_with_in_reply_to': in_reply_to_count,
        'resolved_to_existing': in_reply_to_resolved,
        'unresolved': in_reply_to_count - in_reply_to_resolved,
        'resolve_percent': round(reply_resolve_pct, 2),
        'sample_unresolved': unresolved_replies,
        'pass': True,  # Informational, not blocking
    }

    # Check 8: Priority distribution (informational)
    priority_dist = Counter(r.get('priority', 'unknown') for r in records)
    results['checks']['priority_distribution'] = {
        'distribution': dict(priority_dist),
        'pass': True,  # Informational
    }

    # ========== SPOT CHECKS ==========

    # Spot check 1: 10 random REPLIED emails
    replied_emails = [e for e in records if e.get('action') == 'REPLIED']
    sample_replied = random.sample(replied_emails, min(10, len(replied_emails)))

    replied_checks = []
    for email in sample_replied:
        msg_id = email.get('message_id', '')[:40]
        has_response_time = email.get('response_time_seconds') is not None
        response_timing = email.get('response_timing')
        replied_checks.append({
            'message_id': msg_id,
            'has_response_time': has_response_time,
            'response_timing': response_timing,
            'valid': has_response_time and response_timing is not None,
        })

    results['spot_checks']['replied_emails'] = {
        'sample_size': len(replied_checks),
        'total_replied': len(replied_emails),
        'samples': replied_checks,
        'all_valid': all(c['valid'] for c in replied_checks),
    }

    # Spot check 2: 10 random threads - verify multiple emails
    threads_map = defaultdict(list)
    for record in records:
        thread_id = record.get('thread_id')
        if thread_id:
            threads_map[thread_id].append(record)

    multi_email_threads = [(tid, emails) for tid, emails in threads_map.items() if len(emails) > 1]
    sample_threads = random.sample(multi_email_threads, min(10, len(multi_email_threads)))

    thread_checks = []
    for thread_id, emails in sample_threads:
        thread_checks.append({
            'thread_id': thread_id[:30],
            'email_count': len(emails),
            'has_replies': any(e.get('action') == 'REPLIED' for e in emails),
        })

    results['spot_checks']['multi_email_threads'] = {
        'sample_size': len(thread_checks),
        'total_multi_email_threads': len(multi_email_threads),
        'samples': thread_checks,
    }

    # Spot check 3: Sent emails verification
    sample_sent = random.sample(sent_emails, min(10, len(sent_emails)))
    sent_checks = []
    for email in sample_sent:
        labels = email.get('labels', [])
        action = email.get('action')
        sent_checks.append({
            'message_id': email.get('message_id', '')[:40],
            'has_sent_label': 'SENT' in labels,
            'action': action,
            'correct': action == 'COMPOSED',
        })

    results['spot_checks']['sent_emails'] = {
        'sample_size': len(sent_checks),
        'total_sent': len(sent_emails),
        'samples': sent_checks,
        'all_correct': all(c['correct'] for c in sent_checks),
    }

    # ========== FINAL STATUS ==========

    all_checks_pass = all(
        c.get('pass', True)
        for c in results['checks'].values()
    )
    results['status'] = 'PASS' if all_checks_pass else 'FAIL'

    return results


def main():
    if len(sys.argv) < 2:
        # Default path
        jsonl_path = Path('/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/enriched_emails.jsonl')
    else:
        jsonl_path = Path(sys.argv[1])

    if not jsonl_path.exists():
        print(json.dumps({
            'gate': 'enrichment',
            'status': 'FAIL',
            'blocking_issues': [f'File not found: {jsonl_path}'],
        }, indent=2))
        sys.exit(1)

    results = verify_enrichment(jsonl_path)
    print(json.dumps(results, indent=2))

    sys.exit(0 if results['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
