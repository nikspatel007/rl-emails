#!/usr/bin/env python3
"""Stage 3: Analyze emails data quality.

Generates statistics and quality report for parsed email data.

Checks:
- Total count and basic stats
- Label distribution (frequency of each label)
- Thread length distribution
- Missing fields analysis
- Date parsing issues
- Encoding issues detection

Usage:
    python scripts/analyze_emails.py /path/to/parsed_emails.jsonl

Output: JSON analysis report to stdout
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_date(date_str: str) -> datetime | None:
    """Parse email date string to datetime."""
    if not date_str:
        return None

    date_str = date_str.strip()
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


def detect_encoding_issues(text: str) -> list[str]:
    """Detect common encoding issues in text."""
    issues = []
    if not text:
        return issues

    # Common encoding artifacts
    patterns = [
        (r'Ã©', 'UTF-8 decoded as Latin-1 (e-acute)'),
        (r'Ã¨', 'UTF-8 decoded as Latin-1 (e-grave)'),
        (r'Ã ', 'UTF-8 decoded as Latin-1 (a-grave)'),
        (r'â€™', 'UTF-8 decoded as Latin-1 (right-quote)'),
        (r'â€"', 'UTF-8 decoded as Latin-1 (em-dash)'),
        (r'â€œ', 'UTF-8 decoded as Latin-1 (left-dquote)'),
        (r'\\u[0-9a-fA-F]{4}', 'Escaped unicode'),
        (r'=\?[^?]+\?[BQ]\?', 'MIME encoded-word (may need decoding)'),
    ]

    for pattern, issue_type in patterns:
        if re.search(pattern, text):
            issues.append(issue_type)

    return list(set(issues))


def analyze_jsonl(jsonl_path: Path) -> dict:
    """Analyze parsed JSONL file and generate quality report."""
    report = {
        'stage': 'analyze',
        'status': 'PENDING',
        'input_file': str(jsonl_path),
        'statistics': {},
        'quality_checks': {},
        'issues': [],
    }

    # Read all records
    records = []
    parse_errors = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors.append(f"Line {i}: {e}")
                if len(parse_errors) >= 10:
                    break

    if parse_errors:
        report['quality_checks']['parse_errors'] = {
            'count': len(parse_errors),
            'samples': parse_errors[:5],
            'pass': False,
        }
        report['issues'].append(f"{len(parse_errors)} JSON parse errors")
        report['status'] = 'FAIL'
        return report

    total_count = len(records)
    report['statistics']['total_count'] = total_count

    # === LABEL DISTRIBUTION ===
    label_counter = Counter()
    emails_with_labels = 0
    for r in records:
        labels = r.get('labels', [])
        if labels:
            emails_with_labels += 1
            for label in labels:
                label_counter[label] += 1

    report['statistics']['labels'] = {
        'emails_with_labels': emails_with_labels,
        'unique_labels': len(label_counter),
        'top_20_labels': dict(label_counter.most_common(20)),
        'label_coverage_percent': round(emails_with_labels / total_count * 100, 2) if total_count > 0 else 0,
    }

    # === THREAD LENGTH DISTRIBUTION ===
    thread_emails = defaultdict(list)
    for r in records:
        thread_id = r.get('thread_id')
        if thread_id:
            thread_emails[thread_id].append(r.get('message_id'))

    thread_lengths = [len(emails) for emails in thread_emails.values()]
    if thread_lengths:
        avg_thread_length = sum(thread_lengths) / len(thread_lengths)
        max_thread_length = max(thread_lengths)
        single_email_threads = sum(1 for l in thread_lengths if l == 1)
        multi_email_threads = sum(1 for l in thread_lengths if l > 1)
    else:
        avg_thread_length = 0
        max_thread_length = 0
        single_email_threads = 0
        multi_email_threads = 0

    thread_length_buckets = {
        '1': sum(1 for l in thread_lengths if l == 1),
        '2-5': sum(1 for l in thread_lengths if 2 <= l <= 5),
        '6-10': sum(1 for l in thread_lengths if 6 <= l <= 10),
        '11-20': sum(1 for l in thread_lengths if 11 <= l <= 20),
        '21+': sum(1 for l in thread_lengths if l > 20),
    }

    report['statistics']['threads'] = {
        'total_threads': len(thread_emails),
        'average_length': round(avg_thread_length, 2),
        'max_length': max_thread_length,
        'single_email_threads': single_email_threads,
        'multi_email_threads': multi_email_threads,
        'length_distribution': thread_length_buckets,
    }

    # === DATE ANALYSIS ===
    parsed_dates = []
    unparseable_dates = []
    for r in records:
        date_str = r.get('date', '')
        if date_str:
            dt = parse_date(date_str)
            if dt:
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                parsed_dates.append(dt)
            else:
                unparseable_dates.append(date_str)

    if parsed_dates:
        min_date = min(parsed_dates)
        max_date = max(parsed_dates)

        # Year distribution
        year_counter = Counter(dt.year for dt in parsed_dates)
    else:
        min_date = max_date = None
        year_counter = Counter()

    report['statistics']['dates'] = {
        'parseable_count': len(parsed_dates),
        'unparseable_count': len(unparseable_dates),
        'parse_rate_percent': round(len(parsed_dates) / total_count * 100, 2) if total_count > 0 else 0,
        'min_date': min_date.isoformat() if min_date else None,
        'max_date': max_date.isoformat() if max_date else None,
        'year_distribution': dict(sorted(year_counter.items())),
        'sample_unparseable': unparseable_dates[:5] if unparseable_dates else [],
    }

    # === FIELD COVERAGE ===
    field_coverage = {}
    fields_to_check = ['message_id', 'date', 'from', 'to', 'subject', 'body', 'labels', 'thread_id']

    for field in fields_to_check:
        present = sum(1 for r in records if r.get(field))
        field_coverage[field] = {
            'present': present,
            'missing': total_count - present,
            'coverage_percent': round(present / total_count * 100, 2) if total_count > 0 else 0,
        }

    report['statistics']['field_coverage'] = field_coverage

    # === SENDER ANALYSIS ===
    sender_counter = Counter()
    for r in records:
        from_addr = r.get('from', '')
        if from_addr:
            # Extract domain
            match = re.search(r'@([\w.-]+)', from_addr)
            if match:
                sender_counter[match.group(1)] += 1

    report['statistics']['senders'] = {
        'unique_domains': len(sender_counter),
        'top_20_domains': dict(sender_counter.most_common(20)),
    }

    # === ENCODING ISSUES ===
    encoding_issues = defaultdict(int)
    emails_with_encoding_issues = 0

    for r in records:
        subject = r.get('subject', '')
        body = r.get('body', '')

        issues = detect_encoding_issues(subject) + detect_encoding_issues(body[:1000])
        if issues:
            emails_with_encoding_issues += 1
            for issue in issues:
                encoding_issues[issue] += 1

    report['quality_checks']['encoding'] = {
        'emails_with_issues': emails_with_encoding_issues,
        'issue_rate_percent': round(emails_with_encoding_issues / total_count * 100, 2) if total_count > 0 else 0,
        'issue_types': dict(encoding_issues),
        'pass': emails_with_encoding_issues < total_count * 0.05,  # <5% is acceptable
    }

    # === QUALITY CHECKS ===
    # Check required fields
    required_missing = field_coverage['message_id']['missing'] + field_coverage['date']['missing'] + field_coverage['from']['missing']
    report['quality_checks']['required_fields'] = {
        'missing_count': required_missing,
        'pass': required_missing == 0,
    }

    # Check date parsing rate
    date_parse_rate = len(parsed_dates) / total_count * 100 if total_count > 0 else 0
    report['quality_checks']['date_parsing'] = {
        'parse_rate_percent': round(date_parse_rate, 2),
        'unparseable_samples': unparseable_dates[:5],
        'pass': date_parse_rate >= 95,  # 95% threshold
    }

    # Check label coverage
    label_rate = emails_with_labels / total_count * 100 if total_count > 0 else 0
    report['quality_checks']['label_coverage'] = {
        'coverage_percent': round(label_rate, 2),
        'pass': label_rate >= 90,  # 90% threshold
    }

    # === FINAL STATUS ===
    all_checks_pass = all(
        check.get('pass', True)
        for check in report['quality_checks'].values()
    )

    if not all_checks_pass:
        failing_checks = [
            name for name, check in report['quality_checks'].items()
            if not check.get('pass', True)
        ]
        report['issues'].extend(f"Quality check failed: {name}" for name in failing_checks)

    report['status'] = 'PASS' if all_checks_pass else 'WARN'

    return report


def main():
    if len(sys.argv) < 2:
        jsonl_path = Path('/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/parsed_emails.jsonl')
    else:
        jsonl_path = Path(sys.argv[1])

    if not jsonl_path.exists():
        print(json.dumps({
            'stage': 'analyze',
            'status': 'FAIL',
            'issues': [f'File not found: {jsonl_path}'],
        }, indent=2))
        sys.exit(1)

    report = analyze_jsonl(jsonl_path)
    print(json.dumps(report, indent=2))

    sys.exit(0 if report['status'] in ('PASS', 'WARN') else 1)


if __name__ == '__main__':
    main()
