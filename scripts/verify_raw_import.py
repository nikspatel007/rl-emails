#!/usr/bin/env python3
"""Gate 1: Verify raw_emails import.

Automated verification after MBOX parsing to JSONL.

Checks:
- Count matches expected (41377)
- No duplicate message_ids
- All required fields present (message_id, date, from)
- Labels coverage: >90% have labels populated
- Thread IDs: >80% have thread_id
- Date range sanity: dates within expected range

Usage:
    python scripts/verify_raw_import.py /path/to/parsed_emails.jsonl

Output: JSON gate result to stdout

Known Issues (2026-01-04):
    4 duplicate message_ids found (2 unique IDs appearing 3x each).
    Accepted as known issue (0.01% of dataset):
    - <0vkVzJ4VjBknZLvrNxqPAA@notifications.google.com>
    - <cohesionIB/feature_requests/copilot_for_business@github.com>
    These should be deduplicated during database import.
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_date(date_str: str) -> datetime | None:
    """Parse email date string to datetime."""
    if not date_str:
        return None

    import re

    # Clean up the date string
    date_str = date_str.strip()
    # Remove extra timezone info like "(UTC)" or "(PST)"
    date_str = re.sub(r'\s*\([A-Z]+\)\s*$', '', date_str)

    formats = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S %Z',
        '%d %b %Y %H:%M:%S %z',
        '%Y-%m-%d %H:%M:%S',
        '%a %b %d %H:%M:%S %Y',
        '%a, %d %b %Y %H:%M:%S',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def verify_jsonl(jsonl_path: Path, expected_count: int = 41377) -> dict:
    """Verify parsed JSONL file meets gate criteria."""
    results = {
        'gate': 'raw_import',
        'status': 'PENDING',
        'input_file': str(jsonl_path),
        'checks': {},
        'blocking_issues': [],
    }

    # Read all records
    records = []
    line_errors = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
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

    # Check 1: Count matches expected
    actual_count = len(records)
    count_pass = actual_count == expected_count
    results['checks']['count_match'] = {
        'expected': expected_count,
        'actual': actual_count,
        'pass': count_pass,
    }
    if not count_pass:
        results['blocking_issues'].append(
            f"Count mismatch: expected {expected_count}, got {actual_count}"
        )

    # Check 2: No duplicate message_ids
    message_ids = [r.get('message_id', '') for r in records]
    id_counts = Counter(message_ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1 and k}
    dup_count = sum(v - 1 for v in duplicates.values())
    empty_ids = message_ids.count('') + message_ids.count(None)
    results['checks']['duplicates'] = {
        'duplicate_count': dup_count,
        'empty_message_ids': empty_ids,
        'sample_duplicates': list(duplicates.keys())[:5] if duplicates else [],
        'pass': dup_count == 0,
    }
    if dup_count > 0:
        results['blocking_issues'].append(f"{dup_count} duplicate message_ids")

    # Check 3: Required fields present (message_id, date, from)
    missing_message_id = sum(1 for r in records if not r.get('message_id'))
    missing_date = sum(1 for r in records if not r.get('date'))
    missing_from = sum(1 for r in records if not r.get('from'))

    required_pass = missing_message_id == 0 and missing_date == 0 and missing_from == 0
    results['checks']['required_fields'] = {
        'missing_message_id': missing_message_id,
        'missing_date': missing_date,
        'missing_from': missing_from,
        'pass': required_pass,
    }
    if not required_pass:
        issues = []
        if missing_message_id:
            issues.append(f"{missing_message_id} missing message_id")
        if missing_date:
            issues.append(f"{missing_date} missing date")
        if missing_from:
            issues.append(f"{missing_from} missing from")
        results['blocking_issues'].extend(issues)

    # Check 4: Labels coverage (>90% have labels)
    has_labels = sum(1 for r in records if r.get('labels') and len(r['labels']) > 0)
    labels_percent = (has_labels / actual_count * 100) if actual_count > 0 else 0
    labels_pass = labels_percent >= 90
    results['checks']['labels_coverage'] = {
        'with_labels': has_labels,
        'total': actual_count,
        'percent': round(labels_percent, 2),
        'threshold': 90,
        'pass': labels_pass,
    }
    if not labels_pass:
        results['blocking_issues'].append(
            f"Labels coverage {labels_percent:.1f}% < 90% threshold"
        )

    # Check 5: Thread IDs (>80% have thread_id)
    has_thread_id = sum(1 for r in records if r.get('thread_id'))
    thread_percent = (has_thread_id / actual_count * 100) if actual_count > 0 else 0
    thread_pass = thread_percent >= 80
    results['checks']['thread_ids'] = {
        'with_thread_id': has_thread_id,
        'total': actual_count,
        'percent': round(thread_percent, 2),
        'threshold': 80,
        'pass': thread_pass,
    }
    if not thread_pass:
        results['blocking_issues'].append(
            f"Thread ID coverage {thread_percent:.1f}% < 80% threshold"
        )

    # Check 6: Date range sanity
    parsed_dates = []
    for r in records:
        dt = parse_date(r.get('date', ''))
        if dt:
            # Make naive for comparison if needed
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None)
            parsed_dates.append(dt)

    date_parse_rate = len(parsed_dates) / actual_count * 100 if actual_count > 0 else 0

    if parsed_dates:
        min_date = min(parsed_dates)
        max_date = max(parsed_dates)
        # Expect dates between 2000 and 2030
        min_expected = datetime(2000, 1, 1)
        max_expected = datetime(2030, 12, 31)
        date_range_pass = min_date >= min_expected and max_date <= max_expected
    else:
        min_date = max_date = None
        date_range_pass = False

    results['checks']['date_range'] = {
        'parseable_dates': len(parsed_dates),
        'parse_rate_percent': round(date_parse_rate, 2),
        'min_date': min_date.isoformat() if min_date else None,
        'max_date': max_date.isoformat() if max_date else None,
        'expected_range': '2000-01-01 to 2030-12-31',
        'pass': date_range_pass,
    }
    if not date_range_pass:
        results['blocking_issues'].append('Date range outside expected bounds')

    # Final status
    all_pass = all(c.get('pass', False) for c in results['checks'].values())
    results['status'] = 'PASS' if all_pass else 'FAIL'

    return results


def main():
    if len(sys.argv) < 2:
        # Default path
        jsonl_path = Path('/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/parsed_emails.jsonl')
    else:
        jsonl_path = Path(sys.argv[1])

    if not jsonl_path.exists():
        print(json.dumps({
            'gate': 'raw_import',
            'status': 'FAIL',
            'blocking_issues': [f'File not found: {jsonl_path}'],
        }, indent=2))
        sys.exit(1)

    results = verify_jsonl(jsonl_path)
    print(json.dumps(results, indent=2))

    sys.exit(0 if results['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
