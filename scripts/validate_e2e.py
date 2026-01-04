#!/usr/bin/env python3
"""Stage 7: End-to-End Validation.

Final validation before data is used for training. Validates:
1. Count consistency (source file == database)
2. Schema completeness (required fields, no nulls, correct types)
3. Content integrity (random sample comparison, hash validation)
4. Training readiness (sufficient data for SFT/DPO)

Usage:
    python scripts/validate_e2e.py

Output:
    validation_report.json with READY_FOR_TRAINING: true/false
"""

import asyncio
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"
JSONL_PATH = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/enriched_emails.jsonl")
SAMPLE_SIZE = 100
OUTPUT_PATH = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/validation_report.json")

# Training thresholds
MIN_TOTAL_EMAILS = 40000
MIN_REPLIED_FOR_SFT = 2000  # Need sufficient REPLIED for SFT
MIN_ACTIONS_FOR_DPO = 5000  # Need sufficient varied actions for DPO
MAX_LABEL_SKEW = 0.7  # No single action should be > 70% of total


def compute_body_hash(body: str) -> str:
    """Compute SHA256 hash of body text for integrity check."""
    if not body:
        return ""
    return hashlib.sha256(body.encode('utf-8', errors='replace')).hexdigest()[:16]


async def count_validation(conn: asyncpg.Connection, source_count: int) -> dict:
    """Validate counts match between source and database."""
    raw_count = await conn.fetchval("SELECT COUNT(*) FROM raw_emails")
    emails_count = await conn.fetchval("SELECT COUNT(*) FROM emails")

    counts_match = (raw_count == source_count) and (emails_count == source_count)

    return {
        'check': 'count_validation',
        'source_file_count': source_count,
        'raw_emails_count': raw_count,
        'emails_count': emails_count,
        'all_match': counts_match,
        'pass': counts_match,
        'issues': [] if counts_match else [
            f"Count mismatch: source={source_count}, raw_emails={raw_count}, emails={emails_count}"
        ]
    }


async def schema_validation(conn: asyncpg.Connection) -> dict:
    """Validate schema completeness and data types."""
    issues = []
    checks = {}

    # Required fields that must never be null
    required_fields = [
        ('raw_emails', 'message_id'),
        ('emails', 'message_id'),
        ('emails', 'raw_email_id'),
    ]

    for table, field in required_fields:
        null_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE {field} IS NULL"
        )
        key = f"{table}.{field}_nulls"
        checks[key] = null_count
        if null_count > 0:
            issues.append(f"{table}.{field} has {null_count} NULL values")

    # Check for orphaned emails (no raw_email_id)
    orphaned = await conn.fetchval("""
        SELECT COUNT(*) FROM emails
        WHERE raw_email_id IS NULL OR raw_email_id NOT IN (SELECT id FROM raw_emails)
    """)
    checks['orphaned_emails'] = orphaned
    if orphaned > 0:
        issues.append(f"{orphaned} emails have invalid raw_email_id reference")

    # Check action field values
    invalid_actions = await conn.fetchval("""
        SELECT COUNT(*) FROM emails
        WHERE action NOT IN ('COMPOSED', 'REPLIED', 'FORWARDED', 'IGNORED',
                             'ARCHIVED', 'READ_PENDING', 'UNKNOWN')
          AND action IS NOT NULL
    """)
    checks['invalid_action_values'] = invalid_actions
    if invalid_actions > 0:
        issues.append(f"{invalid_actions} emails have invalid action values")

    # Check date parsing coverage
    null_dates = await conn.fetchval(
        "SELECT COUNT(*) FROM emails WHERE date_parsed IS NULL"
    )
    total = await conn.fetchval("SELECT COUNT(*) FROM emails")
    date_coverage = ((total - null_dates) / total * 100) if total else 0
    checks['date_parsed_null_count'] = null_dates
    checks['date_parsed_coverage_pct'] = round(date_coverage, 2)
    if date_coverage < 90:
        issues.append(f"Only {date_coverage:.1f}% of emails have parsed dates")

    # Check from_email coverage
    null_from = await conn.fetchval(
        "SELECT COUNT(*) FROM emails WHERE from_email IS NULL"
    )
    from_coverage = ((total - null_from) / total * 100) if total else 0
    checks['from_email_null_count'] = null_from
    checks['from_email_coverage_pct'] = round(from_coverage, 2)
    if from_coverage < 99:
        issues.append(f"Only {from_coverage:.1f}% of emails have from_email")

    return {
        'check': 'schema_validation',
        'checks': checks,
        'pass': len(issues) == 0,
        'issues': issues
    }


async def content_validation(
    conn: asyncpg.Connection,
    source_records: dict[str, dict]
) -> dict:
    """Validate content integrity via random sampling."""
    issues = []
    sample_results = []

    # Get random sample of message_ids from database
    sample_ids = await conn.fetch(f"""
        SELECT message_id, body_text, subject, from_email, action
        FROM emails
        ORDER BY RANDOM()
        LIMIT {SAMPLE_SIZE}
    """)

    matches = 0
    mismatches = 0
    hash_matches = 0
    hash_mismatches = 0

    for row in sample_ids:
        msg_id = row['message_id']
        db_body = row['body_text'] or ''
        db_subject = row['subject'] or ''

        # Find in source
        source = source_records.get(msg_id)
        if not source:
            issues.append(f"message_id {msg_id[:40]} in DB but not in source")
            mismatches += 1
            continue

        source_body = source.get('body', '')
        source_subject = source.get('subject', '')

        # Compare bodies
        db_hash = compute_body_hash(db_body)
        source_hash = compute_body_hash(source_body)

        body_match = db_hash == source_hash
        subject_match = db_subject == source_subject

        if body_match:
            hash_matches += 1
        else:
            hash_mismatches += 1
            if len(issues) < 5:
                issues.append(
                    f"Body hash mismatch for {msg_id[:40]}: "
                    f"db={db_hash}, source={source_hash}"
                )

        if body_match and subject_match:
            matches += 1
        else:
            mismatches += 1

        sample_results.append({
            'message_id': msg_id[:40],
            'body_hash_match': body_match,
            'subject_match': subject_match,
            'db_action': row['action'],
            'source_action': source.get('action'),
        })

    match_rate = (matches / len(sample_ids) * 100) if sample_ids else 0
    hash_match_rate = (hash_matches / len(sample_ids) * 100) if sample_ids else 0

    return {
        'check': 'content_validation',
        'sample_size': len(sample_ids),
        'exact_matches': matches,
        'mismatches': mismatches,
        'match_rate_pct': round(match_rate, 2),
        'hash_matches': hash_matches,
        'hash_mismatches': hash_mismatches,
        'hash_match_rate_pct': round(hash_match_rate, 2),
        'sample_results': sample_results[:10],  # First 10 for review
        'pass': hash_match_rate >= 95,  # Allow 5% variance
        'issues': issues[:10]
    }


async def training_readiness(conn: asyncpg.Connection) -> dict:
    """Check if data is ready for ML training."""
    issues = []
    checks = {}

    # Total emails
    total = await conn.fetchval("SELECT COUNT(*) FROM emails")
    checks['total_emails'] = total
    if total < MIN_TOTAL_EMAILS:
        issues.append(f"Insufficient emails: {total} < {MIN_TOTAL_EMAILS}")

    # REPLIED count for SFT (supervised fine-tuning)
    replied = await conn.fetchval(
        "SELECT COUNT(*) FROM emails WHERE action = 'REPLIED'"
    )
    checks['replied_count'] = replied
    checks['min_replied_for_sft'] = MIN_REPLIED_FOR_SFT
    if replied < MIN_REPLIED_FOR_SFT:
        issues.append(f"Insufficient REPLIED for SFT: {replied} < {MIN_REPLIED_FOR_SFT}")

    # Action distribution for DPO (preference learning)
    action_dist = await conn.fetch("""
        SELECT action, COUNT(*) as cnt
        FROM emails
        WHERE action IS NOT NULL
        GROUP BY action
        ORDER BY cnt DESC
    """)
    action_counts = {row['action']: row['cnt'] for row in action_dist}
    checks['action_distribution'] = action_counts

    # Check for sufficient varied actions
    total_with_action = sum(action_counts.values())
    checks['total_with_action'] = total_with_action
    if total_with_action < MIN_ACTIONS_FOR_DPO:
        issues.append(f"Insufficient actions for DPO: {total_with_action} < {MIN_ACTIONS_FOR_DPO}")

    # Check label skew
    if total_with_action > 0:
        max_action_pct = max(action_counts.values()) / total_with_action
        checks['max_action_skew_pct'] = round(max_action_pct * 100, 2)
        if max_action_pct > MAX_LABEL_SKEW:
            dominant = max(action_counts, key=action_counts.get)
            issues.append(
                f"Label skew too high: {dominant} is {max_action_pct*100:.1f}% "
                f"(threshold: {MAX_LABEL_SKEW*100}%)"
            )

    # Check sent vs received distribution
    sent = await conn.fetchval("SELECT COUNT(*) FROM emails WHERE is_sent = true")
    received = total - sent
    checks['sent_count'] = sent
    checks['received_count'] = received

    # Check response time data availability
    with_response_time = await conn.fetchval(
        "SELECT COUNT(*) FROM emails WHERE response_time_seconds IS NOT NULL"
    )
    checks['with_response_time'] = with_response_time

    # Check timing distribution
    timing_dist = await conn.fetch("""
        SELECT timing, COUNT(*) as cnt
        FROM emails
        WHERE timing IS NOT NULL
        GROUP BY timing
    """)
    checks['timing_distribution'] = {row['timing']: row['cnt'] for row in timing_dist}

    # Preference pairs potential (emails with clear preference signal)
    # REPLIED > IGNORED shows clear preference
    preference_potential = min(
        action_counts.get('REPLIED', 0),
        action_counts.get('IGNORED', 0)
    )
    checks['preference_pairs_potential'] = preference_potential
    if preference_potential < 1000:
        issues.append(f"Limited preference pairs: only {preference_potential} potential pairs")

    return {
        'check': 'training_readiness',
        'checks': checks,
        'pass': len(issues) == 0,
        'issues': issues
    }


async def run_validation() -> dict:
    """Run all validation checks."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'source_file': str(JSONL_PATH),
        'database': DB_URL,
        'validations': [],
        'blocking_issues': [],
        'READY_FOR_TRAINING': False,
    }

    # Load source file
    print("Loading source file...", file=sys.stderr)
    source_records = {}
    source_count = 0
    with open(JSONL_PATH, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                msg_id = record.get('message_id')
                if msg_id:
                    source_records[msg_id] = record
                source_count += 1

    print(f"Loaded {source_count} source records", file=sys.stderr)
    report['source_count'] = source_count

    # Connect to database
    print("Connecting to database...", file=sys.stderr)
    conn = await asyncpg.connect(DB_URL)

    try:
        # Run validations
        print("Running count validation...", file=sys.stderr)
        count_result = await count_validation(conn, source_count)
        report['validations'].append(count_result)
        if not count_result['pass']:
            report['blocking_issues'].extend(count_result['issues'])

        print("Running schema validation...", file=sys.stderr)
        schema_result = await schema_validation(conn)
        report['validations'].append(schema_result)
        if not schema_result['pass']:
            report['blocking_issues'].extend(schema_result['issues'])

        print("Running content validation...", file=sys.stderr)
        content_result = await content_validation(conn, source_records)
        report['validations'].append(content_result)
        if not content_result['pass']:
            report['blocking_issues'].extend(content_result['issues'][:3])

        print("Running training readiness...", file=sys.stderr)
        training_result = await training_readiness(conn)
        report['validations'].append(training_result)
        if not training_result['pass']:
            report['blocking_issues'].extend(training_result['issues'])

    finally:
        await conn.close()

    # Determine final status
    all_pass = all(v['pass'] for v in report['validations'])
    report['READY_FOR_TRAINING'] = all_pass
    report['summary'] = {
        'total_checks': len(report['validations']),
        'passed': sum(1 for v in report['validations'] if v['pass']),
        'failed': sum(1 for v in report['validations'] if not v['pass']),
    }

    return report


def main():
    """Main entry point."""
    print("Stage 7: End-to-End Validation", file=sys.stderr)
    print("=" * 40, file=sys.stderr)

    if not JSONL_PATH.exists():
        print(json.dumps({
            'READY_FOR_TRAINING': False,
            'blocking_issues': [f'Source file not found: {JSONL_PATH}'],
        }, indent=2))
        sys.exit(1)

    report = asyncio.run(run_validation())

    # Write report
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {OUTPUT_PATH}", file=sys.stderr)

    # Print summary
    print(json.dumps(report, indent=2))

    sys.exit(0 if report['READY_FOR_TRAINING'] else 1)


if __name__ == '__main__':
    main()
