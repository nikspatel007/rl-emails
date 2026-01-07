#!/usr/bin/env python3
"""
Phase 0: Rule-based AI Handleability Classification

Classifies emails into:
- ai_full: AI can fully handle (file, skip)
- ai_partial: AI can help (draft, schedule)
- human_required: Needs human (decision, personal)
- needs_llm: Need LLM to understand

This runs BEFORE any LLM calls to reduce cost.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    print("ERROR: DATABASE_URL environment variable is required")
    sys.exit(1)


def classify_email(
    email: dict[str, Any], features: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    """
    Rule-based classification of what AI could do with this email.

    Returns: (handleability, reason, metadata)
    """
    body = (email.get('body_text') or '').lower()
    subject = (email.get('subject') or '').lower()

    relationship = features.get('relationship_strength', 0)
    is_service = features.get('is_service_email', False)
    service_type = features.get('service_type', '')

    # Detect patterns
    patterns = {
        'has_question': '?' in body or '?' in subject,
        'has_request': any(w in body for w in ['please', 'could you', 'can you', 'would you', 'need you']),
        'has_scheduling': any(w in body for w in ['meeting', 'calendar', 'schedule', 'call', 'zoom', 'teams']),
        'has_deadline': any(w in body for w in ['deadline', 'by friday', 'by monday', 'by eod', 'end of day', 'urgent', 'asap', 'by tomorrow']),
        'has_approval': any(w in body for w in ['approve', 'approval', 'sign off', 'authorize', 'permission']),
        'has_confirm': any(w in body for w in ['confirm', 'acknowledge', 'let me know', 'please reply']),
        'is_newsletter': any(w in body for w in ['unsubscribe', 'view in browser', 'email preferences', 'opt out']),
        'is_fyi': any(w in body for w in ['fyi', 'for your information', 'just wanted to let you know']),
        'is_calendar_response': any(w in subject for w in ['accepted:', 'declined:', 'tentative:']),
        'is_auto_reply': any(w in subject for w in ['out of office', 'automatic reply', 'auto-reply']),
        'has_attachment_ref': any(w in body for w in ['attached', 'attachment', 'see attached', 'please find']),
    }

    # ═══════════════════════════════════════════════════════════════
    # AI_FULL: Can be completely handled by AI without human
    # ═══════════════════════════════════════════════════════════════

    # Newsletters, marketing, notifications - just file them
    if is_service and service_type in ['marketing', 'newsletter', 'notification']:
        folder = 'Promotions' if service_type == 'marketing' else 'Updates'
        return 'ai_full', f'service_{service_type}', {
            'action': 'FILE_TO_FOLDER',
            'folder': folder,
            'patterns': patterns
        }

    # Calendar responses (accepted/declined) - just acknowledge
    if patterns['is_calendar_response']:
        return 'ai_full', 'calendar_response', {
            'action': 'FILE_TO_FOLDER',
            'folder': 'Calendar',
            'patterns': patterns
        }

    # Auto-replies - file and note
    if patterns['is_auto_reply']:
        return 'ai_full', 'auto_reply', {
            'action': 'FILE_TO_FOLDER',
            'folder': 'Auto-replies',
            'patterns': patterns
        }

    # Newsletter patterns even if not detected as service
    if patterns['is_newsletter'] and relationship < 0.2:
        return 'ai_full', 'newsletter_pattern', {
            'action': 'FILE_TO_FOLDER',
            'folder': 'Newsletters',
            'patterns': patterns
        }

    # Transactional service emails (orders, receipts)
    if is_service and service_type == 'transactional':
        if any(w in body for w in ['order', 'receipt', 'confirmation', 'shipped', 'delivered', 'tracking']):
            return 'ai_full', 'transactional', {
                'action': 'FILE_TO_FOLDER',
                'folder': 'Orders',
                'track_delivery': 'shipped' in body or 'tracking' in body,
                'patterns': patterns
            }

    # ═══════════════════════════════════════════════════════════════
    # HUMAN_REQUIRED: Only human can handle
    # ═══════════════════════════════════════════════════════════════

    # Approval/decision requests - always need human
    if patterns['has_approval']:
        return 'human_required', 'approval_request', {
            'action': 'PREPARE_DECISION_CONTEXT',
            'priority': 'high',
            'patterns': patterns
        }

    # High-relationship personal emails - human connection matters
    if relationship > 0.6 and not is_service:
        return 'human_required', 'important_relationship', {
            'action': 'PREPARE_DECISION_CONTEXT',
            'relationship': relationship,
            'patterns': patterns
        }

    # Explicit reply requests from real people
    if not is_service and patterns['has_confirm'] and relationship > 0.2:
        return 'human_required', 'explicit_reply_request', {
            'action': 'DRAFT_REPLY',
            'needs_personalization': True,
            'patterns': patterns
        }

    # Deadline + request from person = human attention
    if patterns['has_deadline'] and patterns['has_request'] and not is_service:
        return 'human_required', 'deadline_request', {
            'action': 'PREPARE_DECISION_CONTEXT',
            'priority': 'high',
            'patterns': patterns
        }

    # ═══════════════════════════════════════════════════════════════
    # AI_PARTIAL: AI can prepare, human finishes
    # ═══════════════════════════════════════════════════════════════

    # Scheduling requests from non-close relationships
    if patterns['has_scheduling'] and not is_service:
        if relationship < 0.5:
            return 'ai_partial', 'scheduling_request', {
                'action': 'SCHEDULE_MEETING',
                'needs_approval': True,
                'patterns': patterns
            }

    # FYI emails from known contacts - draft acknowledgment
    if patterns['is_fyi'] and relationship > 0.1 and not is_service:
        return 'ai_partial', 'fyi_from_contact', {
            'action': 'DRAFT_REPLY',
            'draft_type': 'acknowledgment',
            'patterns': patterns
        }

    # Confirmation requests from services/weak relationships
    if patterns['has_confirm'] and relationship < 0.3:
        return 'ai_partial', 'confirm_request', {
            'action': 'DRAFT_REPLY',
            'draft_type': 'confirmation',
            'patterns': patterns
        }

    # Has attachment reference - might need summary
    if patterns['has_attachment_ref']:
        return 'ai_partial', 'has_attachment', {
            'action': 'SUMMARIZE_ATTACHMENT',
            'patterns': patterns
        }

    # ═══════════════════════════════════════════════════════════════
    # NEEDS_LLM: Can't determine from rules alone
    # ═══════════════════════════════════════════════════════════════

    # Determine LLM priority based on signals
    llm_priority = 3  # default medium

    if relationship > 0.4:
        llm_priority = 1  # high priority - important person
    elif patterns['has_request']:
        llm_priority = 2  # medium-high - has request
    elif is_service:
        llm_priority = 4  # low priority - service email
    elif relationship < 0.1:
        llm_priority = 5  # lowest - unknown sender

    return 'needs_llm', 'ambiguous', {
        'llm_priority': llm_priority,
        'patterns': patterns
    }


def ensure_table(conn: psycopg2.extensions.connection) -> None:
    """Ensure the classification table exists (created by alembic) and is empty."""
    cur = conn.cursor()
    # Clear existing data for idempotent re-runs
    cur.execute("DELETE FROM email_ai_classification")
    conn.commit()
    print("Cleared email_ai_classification table for fresh classification")


def run_classification(conn: psycopg2.extensions.connection) -> int:
    """Run classification on all emails."""
    cur = conn.cursor()

    # Get all emails with features
    print("Loading emails and features...")
    cur.execute("""
        SELECT
            e.id,
            e.subject,
            e.body_text,
            e.from_email,
            ef.relationship_strength,
            ef.is_service_email,
            ef.service_type,
            ef.urgency_score
        FROM emails e
        JOIN email_features ef ON ef.email_id = e.id
        WHERE e.is_sent = FALSE
    """)

    rows = cur.fetchall()
    print(f"Processing {len(rows)} emails...")

    results = []
    for row in rows:
        email = {
            'id': row[0],
            'subject': row[1],
            'body_text': row[2],
            'from_email': row[3],
        }
        features = {
            'relationship_strength': row[4] or 0,
            'is_service_email': row[5] or False,
            'service_type': row[6] or '',
            'urgency_score': row[7] or 0,
        }

        handleability, reason, metadata = classify_email(email, features)
        patterns = metadata.get('patterns', {})

        results.append((
            email['id'],
            handleability,
            reason,
            json.dumps(metadata),
            patterns.get('has_question', False),
            patterns.get('has_request', False),
            patterns.get('has_scheduling', False),
            patterns.get('has_deadline', False),
            patterns.get('has_approval', False),
            patterns.get('has_confirm', False),
            patterns.get('is_newsletter', False),
            patterns.get('is_fyi', False),
            patterns.get('is_calendar_response', False),
            patterns.get('has_attachment_ref', False),
            handleability == 'needs_llm',
            metadata.get('llm_priority'),
        ))

    # Bulk insert
    print("Inserting results...")
    execute_values(cur, """
        INSERT INTO email_ai_classification (
            email_id, predicted_handleability, classification_reason,
            classification_metadata, has_question, has_request,
            has_scheduling, has_deadline, has_approval, has_confirm,
            is_newsletter, is_fyi, is_calendar_response, has_attachment_ref,
            needs_llm_classification, llm_priority
        ) VALUES %s
    """, results)

    conn.commit()
    print(f"Classified {len(results)} emails")

    return len(results)


def print_summary(conn: psycopg2.extensions.connection) -> None:
    """Print classification summary."""
    cur = conn.cursor()

    print("\n" + "="*70)
    print("PHASE 0: RULE-BASED CLASSIFICATION RESULTS")
    print("="*70)

    # Overall distribution
    print("\n📊 OVERALL DISTRIBUTION:")
    cur.execute("""
        SELECT
            predicted_handleability,
            COUNT(*) as count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
        FROM email_ai_classification
        GROUP BY predicted_handleability
        ORDER BY count DESC
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:20} {row[1]:>6} ({row[2]}%)")

    # By reason
    print("\n📋 BY CLASSIFICATION REASON:")
    cur.execute("""
        SELECT
            predicted_handleability,
            classification_reason,
            COUNT(*) as count
        FROM email_ai_classification
        GROUP BY predicted_handleability, classification_reason
        ORDER BY predicted_handleability, count DESC
    """)
    current_handle = None
    for row in cur.fetchall():
        if row[0] != current_handle:
            current_handle = row[0]
            print(f"\n  {current_handle}:")
        print(f"    {row[1]:30} {row[2]:>5}")

    # Validate against actual behavior
    print("\n✅ VALIDATION AGAINST ACTUAL USER BEHAVIOR:")
    cur.execute("""
        SELECT
            ac.predicted_handleability,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE e.action = 'REPLIED') as replied,
            ROUND(100.0 * COUNT(*) FILTER (WHERE e.action = 'REPLIED') / NULLIF(COUNT(*), 0), 1) as reply_rate
        FROM email_ai_classification ac
        JOIN emails e ON e.id = ac.email_id
        GROUP BY ac.predicted_handleability
        ORDER BY reply_rate DESC
    """)
    print(f"  {'Category':20} {'Total':>7} {'Replied':>8} {'Rate':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*8} {'-'*7}")
    for row in cur.fetchall():
        print(f"  {row[0]:20} {row[1]:>7} {row[2]:>8} {row[3]:>6}%")

    # Pattern frequency
    print("\n🔍 DETECTED PATTERNS:")
    cur.execute("""
        SELECT
            'has_question' as pattern, COUNT(*) FILTER (WHERE has_question) as count
        FROM email_ai_classification
        UNION ALL SELECT 'has_request', COUNT(*) FILTER (WHERE has_request) FROM email_ai_classification
        UNION ALL SELECT 'has_scheduling', COUNT(*) FILTER (WHERE has_scheduling) FROM email_ai_classification
        UNION ALL SELECT 'has_deadline', COUNT(*) FILTER (WHERE has_deadline) FROM email_ai_classification
        UNION ALL SELECT 'has_approval', COUNT(*) FILTER (WHERE has_approval) FROM email_ai_classification
        UNION ALL SELECT 'has_confirm', COUNT(*) FILTER (WHERE has_confirm) FROM email_ai_classification
        UNION ALL SELECT 'is_newsletter', COUNT(*) FILTER (WHERE is_newsletter) FROM email_ai_classification
        UNION ALL SELECT 'is_fyi', COUNT(*) FILTER (WHERE is_fyi) FROM email_ai_classification
        UNION ALL SELECT 'is_calendar_response', COUNT(*) FILTER (WHERE is_calendar_response) FROM email_ai_classification
        UNION ALL SELECT 'has_attachment_ref', COUNT(*) FILTER (WHERE has_attachment_ref) FROM email_ai_classification
        ORDER BY count DESC
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:25} {row[1]:>6}")

    # LLM priority distribution
    print("\n🎯 LLM PRIORITY DISTRIBUTION (for needs_llm emails):")
    cur.execute("""
        SELECT
            llm_priority,
            COUNT(*) as count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
        FROM email_ai_classification
        WHERE needs_llm_classification = TRUE
        GROUP BY llm_priority
        ORDER BY llm_priority
    """)
    priority_labels = {1: 'HIGH (important person)', 2: 'MED-HIGH (has request)', 3: 'MEDIUM (default)', 4: 'LOW (service)', 5: 'LOWEST (unknown)'}
    for row in cur.fetchall():
        label = priority_labels.get(row[0], 'unknown')
        print(f"  Priority {row[0]}: {label:30} {row[1]:>5} ({row[2]}%)")

    # Suggested actions
    print("\n🤖 SUGGESTED AI ACTIONS:")
    cur.execute("""
        SELECT
            classification_metadata->>'action' as action,
            COUNT(*) as count
        FROM email_ai_classification
        WHERE classification_metadata->>'action' IS NOT NULL
        GROUP BY classification_metadata->>'action'
        ORDER BY count DESC
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:30} {row[1]:>5}")

    print("\n" + "="*70)


def main() -> None:
    conn = psycopg2.connect(DB_URL)

    try:
        ensure_table(conn)
        count = run_classification(conn)
        print_summary(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
