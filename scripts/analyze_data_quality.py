#!/usr/bin/env python3
"""Stage 3: Comprehensive data quality analysis for parsed Gmail export.

Generates data quality report with statistics and red flag checks
to determine if the pipeline can proceed.
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from email.utils import parsedate_to_datetime


# Input/Output paths
PARSED_EMAILS_PATH = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/parsed_emails.jsonl")
REPORT_PATH = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/data_quality_report.json")


def detect_mojibake(text: str) -> bool:
    """Detect common mojibake patterns indicating encoding issues.

    Checks for sequences like â€™ (UTF-8 interpreted as latin-1),
    Ã¡ patterns, and replacement characters.
    """
    if not text:
        return False

    # Common mojibake patterns
    patterns = [
        r'â€[™""˜¦]',  # Smart quotes/apostrophes mangled
        r'Ã[¡-ÿ]',      # UTF-8 → Latin-1 mangling
        r'\ufffd',       # Unicode replacement character
        r'Â[\xa0-\xff]', # Double-encoded
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def parse_date_safe(date_str: str) -> datetime | None:
    """Attempt to parse email date string, returning None on failure."""
    if not date_str:
        return None

    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass

    # Try common fallback formats
    fallback_formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in fallback_formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


def analyze_data_quality():
    """Run comprehensive data quality analysis."""
    print(f"Analyzing: {PARSED_EMAILS_PATH}")

    # Statistics
    total_count = 0
    dates = []
    date_failures = 0
    sender_counts = Counter()
    thread_sizes = defaultdict(int)  # thread_id -> count
    label_counts = Counter()

    # Quality checks
    missing_fields = defaultdict(int)
    fields_to_check = ['message_id', 'thread_id', 'date', 'from', 'to', 'subject', 'body', 'labels']
    message_ids = []
    mojibake_count = 0
    sent_count = 0

    # User's email for sent detection
    user_email_patterns = []  # Will detect from Sent emails

    with open(PARSED_EMAILS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            email_data = json.loads(line)
            total_count += 1

            # Check missing fields
            for field in fields_to_check:
                value = email_data.get(field)
                if value is None or value == '' or value == []:
                    missing_fields[field] += 1

            # Track message_ids for duplicate detection
            if email_data.get('message_id'):
                message_ids.append(email_data['message_id'])

            # Parse date
            date_str = email_data.get('date', '')
            parsed = parse_date_safe(date_str)
            if parsed:
                dates.append(parsed)
            else:
                date_failures += 1

            # Sender distribution
            sender = email_data.get('from', '')
            if sender:
                # Extract just email part for grouping
                match = re.search(r'[\w.+-]+@[\w.-]+', sender)
                sender_key = match.group(0).lower() if match else sender
                sender_counts[sender_key] += 1

            # Thread sizes
            thread_id = email_data.get('thread_id', '')
            if thread_id:
                thread_sizes[thread_id] += 1

            # Labels
            labels = email_data.get('labels', [])
            for label in labels:
                label_counts[label] += 1
                if label == 'Sent':
                    sent_count += 1
                    # Capture user email from Sent messages
                    from_addr = email_data.get('from', '')
                    match = re.search(r'[\w.+-]+@[\w.-]+', from_addr)
                    if match:
                        user_email_patterns.append(match.group(0).lower())

            # Mojibake detection
            body = email_data.get('body', '')
            subject = email_data.get('subject', '')
            if detect_mojibake(body) or detect_mojibake(subject):
                mojibake_count += 1

            if total_count % 10000 == 0:
                print(f"Processed {total_count} emails...")

    print(f"Total emails processed: {total_count}")

    # Calculate duplicate message_ids
    id_counts = Counter(message_ids)
    duplicate_ids = {k: v for k, v in id_counts.items() if v > 1}
    duplicate_count = sum(v - 1 for v in duplicate_ids.values())  # Extra occurrences

    # Thread statistics
    thread_lengths = list(thread_sizes.values())
    avg_thread_length = sum(thread_lengths) / len(thread_lengths) if thread_lengths else 0
    max_thread_length = max(thread_lengths) if thread_lengths else 0
    single_email_threads = sum(1 for t in thread_lengths if t == 1)

    # Date range - convert all to UTC for comparison
    from datetime import timezone
    normalized_dates = []
    for d in dates:
        if d.tzinfo is None:
            # Treat naive as UTC
            normalized_dates.append(d.replace(tzinfo=timezone.utc))
        else:
            normalized_dates.append(d)

    if normalized_dates:
        normalized_dates.sort()
        earliest = normalized_dates[0]
        latest = normalized_dates[-1]
    else:
        earliest = None
        latest = None

    # Identify user's primary email
    user_email = Counter(user_email_patterns).most_common(1)
    user_email_addr = user_email[0][0] if user_email else "unknown"

    # Red flag checks
    red_flags = []

    # Check: No SENT folder emails
    if sent_count == 0:
        red_flags.append("CRITICAL: No Sent folder emails - cannot compute 'replied' signal")

    # Check: No labels at all
    if not label_counts:
        red_flags.append("CRITICAL: No labels found - cannot compute priority signal")

    # Check: >5% date parse failures
    date_failure_rate = (date_failures / total_count * 100) if total_count else 0
    if date_failure_rate > 5:
        red_flags.append(f"CRITICAL: {date_failure_rate:.1f}% date parse failures (threshold: 5%)")

    # Check: >1% duplicate message_ids
    duplicate_rate = (duplicate_count / total_count * 100) if total_count else 0
    if duplicate_rate > 1:
        red_flags.append(f"CRITICAL: {duplicate_rate:.1f}% duplicate message_ids (threshold: 1%)")

    # Determine overall status
    pipeline_status = "FAIL" if red_flags else "PASS"

    # Build report
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "pipeline_status": pipeline_status,
        "red_flags": red_flags,

        "statistics": {
            "total_emails": total_count,
            "date_range": {
                "earliest": earliest.isoformat() if earliest else None,
                "latest": latest.isoformat() if latest else None,
            },
            "label_distribution": dict(label_counts.most_common(30)),
            "unique_labels": len(label_counts),
            "thread_statistics": {
                "total_threads": len(thread_sizes),
                "avg_thread_length": round(avg_thread_length, 2),
                "max_thread_length": max_thread_length,
                "single_email_threads": single_email_threads,
            },
            "top_20_senders": dict(sender_counts.most_common(20)),
            "user_email": user_email_addr,
            "sent_count": sent_count,
        },

        "quality_checks": {
            "missing_fields": dict(missing_fields),
            "date_parse_failures": date_failures,
            "date_failure_rate_pct": round(date_failure_rate, 2),
            "mojibake_count": mojibake_count,
            "mojibake_rate_pct": round((mojibake_count / total_count * 100) if total_count else 0, 2),
            "duplicate_message_ids": len(duplicate_ids),
            "duplicate_email_count": duplicate_count,
            "duplicate_rate_pct": round(duplicate_rate, 2),
        },

        "thresholds": {
            "date_failure_max_pct": 5.0,
            "duplicate_max_pct": 1.0,
            "sent_minimum": 1,
            "labels_minimum": 1,
        }
    }

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT")
    print(f"{'='*60}")
    print(f"Pipeline Status: {pipeline_status}")
    print(f"\nStatistics:")
    print(f"  Total emails: {total_count:,}")
    print(f"  Date range: {earliest.strftime('%Y-%m-%d') if earliest else 'N/A'} to {latest.strftime('%Y-%m-%d') if latest else 'N/A'}")
    print(f"  Sent emails: {sent_count:,}")
    print(f"  Unique labels: {len(label_counts)}")
    print(f"  Threads: {len(thread_sizes):,} (avg {avg_thread_length:.1f} emails, max {max_thread_length})")

    print(f"\nQuality Metrics:")
    print(f"  Date parse failures: {date_failures} ({date_failure_rate:.2f}%)")
    print(f"  Duplicate message_ids: {duplicate_count} ({duplicate_rate:.2f}%)")
    print(f"  Mojibake detected: {mojibake_count} ({mojibake_count/total_count*100:.2f}%)")

    if missing_fields:
        print(f"\nMissing Fields:")
        for field, count in sorted(missing_fields.items(), key=lambda x: -x[1]):
            print(f"  {field}: {count} ({count/total_count*100:.1f}%)")

    if red_flags:
        print(f"\n🚨 RED FLAGS:")
        for flag in red_flags:
            print(f"  - {flag}")
    else:
        print(f"\n✅ No red flags - pipeline can proceed")

    print(f"\nReport written to: {REPORT_PATH}")

    return report


if __name__ == '__main__':
    analyze_data_quality()
