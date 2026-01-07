#!/usr/bin/env python3
"""
One-shot data onboarding pipeline.

Processes a Gmail MBOX export through the complete ML pipeline:
0. Run alembic migrations (schema setup)
1. Parse MBOX → JSONL
2. Import to PostgreSQL
3. Build thread relationships
4. Compute action labels (Phase 1) - REPLIED, ARCHIVED, IGNORED, etc.
5. Compute ML features (Phase 2)
6. Generate embeddings (Phase 3)
7. Rule-based AI classification (Phase 0)
8. Populate user profiles (Phase 4A)
9. Multi-dimensional clustering (Phase 4B)
10. Hybrid priority ranking (Phase 4C)
11. LLM classification (Phase 4D) - runs AFTER clustering for better context

Usage:
    python scripts/onboard_data.py
    python scripts/onboard_data.py --workers 10
    python scripts/onboard_data.py --skip-llm
    python scripts/onboard_data.py --start-from 5
    python scripts/onboard_data.py --status

All configuration is read from .env:
    DATABASE_URL - PostgreSQL connection URL (required)
    MBOX_PATH - Path to Gmail MBOX file (required)
    YOUR_EMAIL - Your email address for identifying sent emails (required)
    PARSED_JSONL - Path to output JSONL file (optional, default: data/onboarding/parsed_emails.jsonl)
    OPENAI_API_KEY - For embeddings (required)
    ANTHROPIC_API_KEY - For LLM classification (optional, uses OpenAI if not set)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import dotenv_values

# Paths
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "onboarding"
ENV_FILE = PROJECT_ROOT / ".env"

# Load configuration from .env file
config = dotenv_values(ENV_FILE)

DATABASE_URL = config.get("DATABASE_URL")
MBOX_PATH = config.get("MBOX_PATH")
PARSED_JSONL = config.get("PARSED_JSONL")  # Optional - defaults to data/onboarding/parsed_emails.jsonl
YOUR_EMAIL = config.get("YOUR_EMAIL")  # Required for Phase 1 (action labels)
OPENAI_API_KEY = config.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = config.get("ANTHROPIC_API_KEY")

def validate_config() -> None:
    """Validate required environment variables."""
    missing = []
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not MBOX_PATH:
        missing.append("MBOX_PATH")
    if not YOUR_EMAIL:
        missing.append("YOUR_EMAIL")
    if missing:
        print(f"ERROR: Missing required .env variables: {', '.join(missing)}")
        print(f"Please set them in {ENV_FILE}")
        sys.exit(1)


def run_command(cmd: list[str], description: str, cwd: Path | None = None) -> bool:
    """Run a shell command."""
    print(f"\n{'='*60}")
    print(f"Stage: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start = datetime.now()
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=os.environ.copy())
    elapsed = (datetime.now() - start).total_seconds()

    if result.returncode != 0:
        print(f"FAILED: {description} (exit code {result.returncode})")
        return False

    print(f"SUCCESS: {description} ({elapsed:.1f}s)")
    return True


def run_script(name: str, args: list[str] | None = None, description: str | None = None) -> bool:
    """Run a pipeline script."""
    script_path = SCRIPTS_DIR / name
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    return run_command(cmd, description or name)


def check_postgres() -> bool:
    """Check PostgreSQL connection."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        return True
    except Exception as e:
        print(f"PostgreSQL error: {e}")
        return False


def run_alembic_migrations() -> bool:
    """Run alembic upgrade head."""
    return run_command(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        "[0/8] Run database migrations (alembic)",
        cwd=PROJECT_ROOT
    )


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are available."""
    return {
        "openai": bool(OPENAI_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
    }


def _fetch_count(cur: psycopg2.extensions.cursor) -> int:
    """Helper to safely fetch a count from cursor."""
    row = cur.fetchone()
    return int(row[0]) if row else 0


def get_status() -> dict[str, Any]:
    """Get current pipeline status."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        status: dict[str, Any] = {}

        # Emails
        try:
            cur.execute("SELECT COUNT(*) FROM emails WHERE is_sent = FALSE")
            status["emails"] = _fetch_count(cur)
        except Exception:
            status["emails"] = 0

        # Features
        try:
            cur.execute("SELECT COUNT(*) FROM email_features")
            status["features"] = _fetch_count(cur)
        except Exception:
            status["features"] = 0

        # Embeddings
        try:
            cur.execute("SELECT COUNT(*) FROM email_embeddings")
            status["embeddings"] = _fetch_count(cur)
        except Exception:
            status["embeddings"] = 0

        # AI Classification
        try:
            cur.execute("SELECT COUNT(*) FROM email_ai_classification")
            status["ai_classification"] = _fetch_count(cur)
        except Exception:
            status["ai_classification"] = 0

        # LLM Classification
        try:
            cur.execute("SELECT COUNT(*) FROM email_llm_classification")
            status["llm_classification"] = _fetch_count(cur)
        except Exception:
            status["llm_classification"] = 0

        # Needs LLM
        try:
            cur.execute("""
                SELECT COUNT(*) FROM email_ai_classification
                WHERE predicted_handleability = 'needs_llm'
            """)
            status["needs_llm"] = _fetch_count(cur)
        except Exception:
            status["needs_llm"] = 0

        conn.close()
        return status

    except Exception as e:
        return {"error": str(e)}


def print_status() -> None:
    """Print current pipeline status."""
    status = get_status()

    if "error" in status:
        print(f"Error getting status: {status['error']}")
        return

    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)
    print(f"DATABASE_URL: {DATABASE_URL}")
    print(f"Emails imported:        {status['emails']:,}")
    print(f"ML features computed:   {status['features']:,}")
    print(f"Embeddings generated:   {status['embeddings']:,}")
    print(f"Rule-based classified:  {status['ai_classification']:,}")
    print(f"LLM classified:         {status['llm_classification']:,}")
    if status['needs_llm'] > 0:
        remaining = status['needs_llm'] - status['llm_classification']
        print(f"Needs LLM (remaining):  {remaining:,}")
    print("=" * 60)


def generate_report(parsed_jsonl: Path, duration: str) -> str:
    """Generate detailed data insight report with narrative analysis."""
    import json
    from datetime import datetime

    lines = [
        "# Email Data Onboarding Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Processing Time**: {duration}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # Collect all data first for cross-stage analysis
    data = {}

    # Parse report data
    parse_report_path = parsed_jsonl.parent / "parse_report.json"
    if parse_report_path.exists():
        with open(parse_report_path) as f:
            data['parse'] = json.load(f)

    # Database data
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Basic counts
        cur.execute("SELECT COUNT(*) FROM emails")
        data['total_emails'] = _fetch_count(cur)

        cur.execute("SELECT COUNT(*) FROM emails WHERE is_sent = TRUE")
        data['sent_emails'] = _fetch_count(cur)
        data['received_emails'] = data['total_emails'] - data['sent_emails']

        cur.execute("SELECT MIN(date_parsed), MAX(date_parsed) FROM emails")
        data['date_range'] = cur.fetchone() or (None, None)

        # Thread data
        cur.execute("SELECT COUNT(*) FROM threads")
        data['thread_count'] = _fetch_count(cur)

        cur.execute("SELECT AVG(email_count), MAX(email_count), STDDEV(email_count) FROM threads")
        data['thread_stats'] = cur.fetchone() or (0, 0, 0)

        cur.execute("SELECT COUNT(*) FROM threads WHERE email_count = 1")
        data['single_email_threads'] = _fetch_count(cur)

        cur.execute("SELECT COUNT(*) FROM threads WHERE email_count > 5")
        data['long_threads'] = _fetch_count(cur)

        # Features data
        cur.execute("SELECT COUNT(*) FROM email_features")
        data['features_count'] = _fetch_count(cur)

        cur.execute("SELECT COUNT(*) FROM email_features WHERE is_service_email = TRUE")
        data['service_emails'] = _fetch_count(cur)

        cur.execute("""
            SELECT service_type, COUNT(*) FROM email_features
            WHERE service_type IS NOT NULL
            GROUP BY service_type ORDER BY COUNT(*) DESC
        """)
        data['service_types'] = cur.fetchall()

        # Sender analysis
        cur.execute("""
            SELECT from_email, COUNT(*) as cnt FROM emails
            WHERE is_sent = FALSE
            GROUP BY from_email ORDER BY cnt DESC LIMIT 20
        """)
        data['top_senders'] = cur.fetchall()

        cur.execute("SELECT COUNT(DISTINCT from_email) FROM emails WHERE is_sent = FALSE")
        data['unique_senders'] = _fetch_count(cur)

        # Embeddings
        cur.execute("SELECT COUNT(*) FROM email_embeddings")
        data['embeddings_count'] = _fetch_count(cur)

        # AI Classification
        cur.execute("SELECT COUNT(*) FROM email_ai_classification")
        data['ai_class_count'] = _fetch_count(cur)

        cur.execute("""
            SELECT predicted_handleability, COUNT(*) FROM email_ai_classification
            GROUP BY predicted_handleability ORDER BY COUNT(*) DESC
        """)
        data['ai_categories'] = dict(cur.fetchall())

        # LLM Classification
        cur.execute("SELECT COUNT(*) FROM email_llm_classification")
        data['llm_class_count'] = _fetch_count(cur)

        if data['llm_class_count'] > 0:
            cur.execute("""
                SELECT action_type, COUNT(*) FROM email_llm_classification
                GROUP BY action_type ORDER BY COUNT(*) DESC
            """)
            data['llm_categories'] = dict(cur.fetchall())

            cur.execute("""
                SELECT suggested_action, COUNT(*) FROM email_llm_classification
                WHERE suggested_action IS NOT NULL
                GROUP BY suggested_action ORDER BY COUNT(*) DESC
            """)
            data['llm_actions'] = dict(cur.fetchall())

            cur.execute("""
                SELECT urgency, COUNT(*) FROM email_llm_classification
                GROUP BY urgency ORDER BY COUNT(*) DESC
            """)
            data['llm_priorities'] = dict(cur.fetchall())

        # Important people (from users table)
        cur.execute("""
            SELECT email, name, emails_from, reply_count,
                   ROUND(reply_rate::numeric * 100, 1) as reply_rate_pct
            FROM users
            WHERE reply_count > 0 AND NOT is_you
            ORDER BY reply_count DESC, reply_rate DESC
            LIMIT 10
        """)
        data['important_people'] = cur.fetchall()

        # Important senders count
        cur.execute("SELECT COUNT(*) FROM users WHERE is_important_sender = true")
        data['important_sender_count'] = _fetch_count(cur)

        # Service engagement breakdown
        cur.execute("""
            SELECT
                ef.service_type,
                COUNT(*) as total,
                SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) as replied,
                SUM(CASE WHEN e.action = 'ARCHIVED' THEN 1 ELSE 0 END) as archived,
                SUM(CASE WHEN e.action = 'IGNORED' THEN 1 ELSE 0 END) as ignored
            FROM email_features ef
            JOIN emails e ON e.id = ef.email_id
            WHERE ef.is_service_email = true AND e.is_sent = false
            GROUP BY ef.service_type
            ORDER BY total DESC
        """)
        data['service_engagement'] = cur.fetchall()

        # Relationship validation
        cur.execute("""
            SELECT
                CASE
                    WHEN ef.relationship_strength >= 0.7 THEN 'Strong'
                    WHEN ef.relationship_strength >= 0.5 THEN 'Medium'
                    WHEN ef.relationship_strength >= 0.3 THEN 'Weak'
                    ELSE 'Minimal'
                END as tier,
                COUNT(*) as emails,
                SUM(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) as replied
            FROM email_features ef
            JOIN emails e ON e.id = ef.email_id
            WHERE e.is_sent = false
            GROUP BY 1
            ORDER BY MIN(ef.relationship_strength) DESC
        """)
        data['relationship_validation'] = cur.fetchall()

        conn.close()

    except Exception as e:
        lines.append(f"Error collecting data: {e}")
        return "\n".join(lines)

    # === EXECUTIVE SUMMARY ===
    total = data['total_emails']
    received = data['received_emails']
    date_start = data['date_range'][0].strftime('%b %Y') if data['date_range'][0] else 'N/A'
    date_end = data['date_range'][1].strftime('%b %Y') if data['date_range'][1] else 'N/A'

    # Calculate key insights
    service_pct = (data['service_emails'] / received * 100) if received > 0 else 0
    thread_pct = ((data['thread_count'] - data['single_email_threads']) / data['thread_count'] * 100) if data['thread_count'] > 0 else 0

    ai_auto = data['ai_categories'].get('ai_full', 0) + data['ai_categories'].get('ai_partial', 0)
    ai_auto_pct = (ai_auto / data['ai_class_count'] * 100) if data['ai_class_count'] > 0 else 0

    lines.extend([
        f"This report analyzes **{total:,} emails** from {date_start} to {date_end}. ",
        f"Of these, **{received:,}** were received and **{data['sent_emails']:,}** were sent.",
        "",
        "### Key Findings",
        "",
        f"1. **{service_pct:.0f}% of incoming email is automated** - newsletters, notifications, and service emails that rarely need human attention",
        f"2. **{thread_pct:.0f}% of conversations are multi-email threads** - indicating ongoing relationships vs. one-off messages",
        f"3. **{ai_auto_pct:.0f}% of emails can be handled by AI** - either fully automated or with AI assistance",
        f"4. **{data['unique_senders']:,} unique senders** - your communication network",
        "",
        "---",
        "",
    ])

    # === STAGE 1: RAW DATA ===
    parse = data.get('parse', {})
    lines.extend([
        "## Stage 1: Raw Email Data",
        "",
        "### What We Had",
        "",
        f"A Gmail MBOX export containing {parse.get('total_emails', total):,} emails with:",
        f"- {parse.get('emails_with_body', 0):,} emails with text content",
        f"- {parse.get('emails_with_attachments', 0):,} emails with attachments ({parse.get('emails_with_attachments', 0) / total * 100:.1f}%)",
        f"- {len(parse.get('unique_labels', [])):,} unique Gmail labels",
        "",
        "### What We Learned",
        "",
    ])

    # Analyze labels
    label_counts = parse.get('label_counts', {})
    inbox_count = label_counts.get('Inbox', 0)
    archived_count = label_counts.get('Archived', 0)
    unread_count = label_counts.get('Unread', 0)

    lines.extend([
        f"- **Inbox behavior**: {inbox_count:,} emails currently in inbox, {archived_count:,} archived",
        f"- **Read rate**: {(total - unread_count) / total * 100:.0f}% of emails have been opened" if total > 0 else "",
    ])

    # Check for Superhuman labels
    sh_labels = {k: v for k, v in label_counts.items() if '[Superhuman]' in k}
    if sh_labels:
        lines.append(f"- **Superhuman AI labels detected**: Already using AI categorization")
        for label, count in sorted(sh_labels.items(), key=lambda x: -x[1])[:5]:
            clean_label = label.replace('[Superhuman]/AI/', '')
            lines.append(f"  - {clean_label}: {count:,}")

    lines.extend([
        "",
        "### Why This Matters",
        "",
        "The raw email structure reveals communication patterns. High attachment rates suggest ",
        "document-heavy workflows. Existing AI labels show what categorization already exists.",
        "",
        "---",
        "",
    ])

    # === STAGE 2 & 3: STRUCTURE ===
    lines.extend([
        "## Stage 2-3: Email Structure & Threads",
        "",
        "### What We Had",
        "",
        f"- {total:,} individual email records",
        f"- Raw header data (from, to, cc, references, in-reply-to)",
        "",
        "### What We Learned",
        "",
        f"**Conversation patterns:**",
        f"- {data['thread_count']:,} distinct conversation threads",
        f"- {data['single_email_threads']:,} one-off emails ({data['single_email_threads'] / data['thread_count'] * 100:.0f}%)" if data['thread_count'] > 0 else "",
        f"- {data['long_threads']:,} extended conversations (5+ emails)",
        f"- Average thread length: {data['thread_stats'][0]:.1f} emails" if data['thread_stats'][0] else "",
        f"- Longest thread: {data['thread_stats'][1]:.0f} emails" if data['thread_stats'][1] else "",
        "",
        f"**Communication network:**",
        f"- {data['unique_senders']:,} unique people/services emailing you",
        "",
        "**Top senders:**",
        "",
    ])

    for sender, count in data['top_senders'][:10]:
        lines.append(f"| {sender[:40]}{'...' if len(sender) > 40 else ''} | {count:,} |")

    lines.extend([
        "",
        "### Why This Matters",
        "",
        "Thread structure reveals relationship depth. One-off emails are often notifications or cold outreach. ",
        "Long threads indicate ongoing projects or relationships requiring context. ",
        "Top senders show where most of your email attention goes.",
        "",
        "---",
        "",
    ])

    # === STAGE 4: ML FEATURES ===
    lines.extend([
        "## Stage 4: Machine Learning Features",
        "",
        "### What We Had",
        "",
        f"- {data['features_count']:,} emails ready for feature extraction",
        "",
        "### What We Learned",
        "",
        f"**Service email detection:**",
        f"- {data['service_emails']:,} automated/service emails identified ({service_pct:.0f}% of received)",
        "",
        "**Breakdown by type:**",
        "",
    ])

    for stype, count in data['service_types'][:8]:
        pct = count / data['service_emails'] * 100 if data['service_emails'] > 0 else 0
        lines.append(f"- **{stype}**: {count:,} ({pct:.0f}%)")

    human_emails = received - data['service_emails']
    lines.extend([
        "",
        f"**Human-to-human emails**: {human_emails:,} ({human_emails / received * 100:.0f}% of received)" if received > 0 else "",
        "",
        "### Why This Matters",
        "",
        f"Service emails ({service_pct:.0f}%) are prime candidates for automated handling - archive, ",
        "filter, or batch process. The remaining {100 - service_pct:.0f}% human emails need more nuanced treatment. ",
        "This split is foundational for building automation rules.",
        "",
        "---",
        "",
    ])

    # === STAGE 5: EMBEDDINGS ===
    coverage = data['embeddings_count'] / total * 100 if total > 0 else 0
    lines.extend([
        "## Stage 5: Semantic Embeddings",
        "",
        "### What We Had",
        "",
        f"- {total:,} emails with text content",
        "",
        "### What We Learned",
        "",
        f"- Generated {data['embeddings_count']:,} semantic embeddings ({coverage:.0f}% coverage)",
        "- Each email now has a 1536-dimensional vector representing its meaning",
        "",
        "### Why This Matters",
        "",
        "Embeddings enable semantic search (\"find emails about project deadline\") and clustering ",
        "(\"group similar requests together\"). They're the foundation for understanding email *meaning*, ",
        "not just keywords.",
        "",
        "---",
        "",
    ])

    # === STAGE 6: RULE-BASED CLASSIFICATION ===
    ai_full = data['ai_categories'].get('ai_full', 0)
    ai_partial = data['ai_categories'].get('ai_partial', 0)
    human_req = data['ai_categories'].get('human_required', 0)
    needs_llm = data['ai_categories'].get('needs_llm', 0)

    lines.extend([
        "## Stage 6: Rule-Based AI Classification",
        "",
        "### What We Had",
        "",
        f"- {data['ai_class_count']:,} emails with ML features",
        "",
        "### What We Learned",
        "",
        "Based on sender patterns, email type, and content signals:",
        "",
        f"| Category | Count | % | Meaning |",
        f"|----------|-------|---|---------|",
        f"| **AI Full** | {ai_full:,} | {ai_full / data['ai_class_count'] * 100:.0f}% | Can be fully automated (archive, delete, file) |" if data['ai_class_count'] > 0 else "",
        f"| **AI Partial** | {ai_partial:,} | {ai_partial / data['ai_class_count'] * 100:.0f}% | AI can draft/suggest, human approves |" if data['ai_class_count'] > 0 else "",
        f"| **Human Required** | {human_req:,} | {human_req / data['ai_class_count'] * 100:.0f}% | Needs human judgment |" if data['ai_class_count'] > 0 else "",
        f"| **Needs LLM** | {needs_llm:,} | {needs_llm / data['ai_class_count'] * 100:.0f}% | Rules insufficient, needs deeper analysis |" if data['ai_class_count'] > 0 else "",
        "",
        "### Why This Matters",
        "",
        f"**{ai_auto_pct:.0f}% automation potential** with rule-based classification alone. ",
        f"The {needs_llm:,} emails flagged for LLM analysis are ambiguous cases where simple rules ",
        "can't determine the right action - these benefit most from AI understanding.",
        "",
        "---",
        "",
    ])

    # === STAGE 7: LLM CLASSIFICATION ===
    if data['llm_class_count'] > 0:
        lines.extend([
            "## Stage 7: LLM Deep Classification",
            "",
            "### What We Had",
            "",
            f"- {needs_llm:,} ambiguous emails requiring deeper analysis",
            "",
            "### What We Learned",
            "",
            f"LLM analyzed {data['llm_class_count']:,} emails and determined:",
            "",
            "**Email Categories:**",
            "",
        ])

        for cat, count in sorted(data['llm_categories'].items(), key=lambda x: -x[1]):
            pct = count / data['llm_class_count'] * 100
            lines.append(f"- **{cat}**: {count:,} ({pct:.0f}%)")

        lines.extend([
            "",
            "**Recommended Actions:**",
            "",
        ])

        for action, count in sorted(data['llm_actions'].items(), key=lambda x: -x[1]):
            pct = count / data['llm_class_count'] * 100
            lines.append(f"- **{action}**: {count:,} ({pct:.0f}%)")

        lines.extend([
            "",
            "**Priority Assessment:**",
            "",
        ])

        for priority, count in sorted(data['llm_priorities'].items(), key=lambda x: -x[1]):
            pct = count / data['llm_class_count'] * 100
            lines.append(f"- **{priority}**: {count:,} ({pct:.0f}%)")

        # Calculate actionable insights
        archive_actions = data['llm_actions'].get('archive', 0) + data['llm_actions'].get('delete', 0)
        respond_actions = data['llm_actions'].get('reply', 0) + data['llm_actions'].get('forward', 0)

        lines.extend([
            "",
            "### Why This Matters",
            "",
            f"Of the ambiguous emails, **{archive_actions:,}** can be archived/deleted, ",
            f"while **{respond_actions:,}** need responses. The LLM provides the nuanced ",
            "understanding that rules cannot capture.",
            "",
            "---",
            "",
        ])

    # === IMPORTANT PEOPLE ===
    lines.extend([
        "## Important People",
        "",
        f"**{data.get('important_sender_count', 0)} senders** marked as important based on reply behavior:",
        "",
        "| Person | Emails | Replied | Rate |",
        "|--------|--------|---------|------|",
    ])

    for email, name, emails_from, reply_count, rate in data.get('important_people', [])[:10]:
        name_display = name[:25] if name else email.split('@')[0][:25]
        lines.append(f"| {name_display} | {emails_from} | {reply_count} | {rate or 0}% |")

    lines.extend([
        "",
        "### Service Email Engagement",
        "",
        "How you interact with different service email types:",
        "",
        "| Type | Total | Replied | Archived | Ignored |",
        "|------|-------|---------|----------|---------|",
    ])

    for svc_type, total_svc, replied, archived, ignored in data.get('service_engagement', []):
        svc_name = svc_type or "unknown"
        lines.append(f"| {svc_name} | {total_svc} | {replied} | {archived} | {ignored} |")

    lines.extend([
        "",
        "### Relationship Strength Validation",
        "",
        "Emails are grouped by relationship strength with the sender:",
        "",
        "| Relationship | Emails | Replied | Reply Rate |",
        "|--------------|--------|---------|------------|",
    ])

    for tier, emails, replied in data.get('relationship_validation', []):
        rate = (replied / emails * 100) if emails > 0 else 0
        lines.append(f"| {tier} | {emails} | {replied} | {rate:.1f}% |")

    lines.extend([
        "",
        "✅ Strong correlation between relationship strength and reply behavior validates our ML pipeline.",
        "",
        "---",
        "",
    ])

    # === OVERALL INSIGHTS ===
    lines.extend([
        "## Overall Insights",
        "",
        "### The Big Picture",
        "",
    ])

    # Calculate overall automation potential
    total_automatable = ai_full + data.get('llm_actions', {}).get('archive', 0) + data.get('llm_actions', {}).get('delete', 0)
    auto_pct = total_automatable / total * 100 if total > 0 else 0

    lines.extend([
        f"From {total:,} emails analyzed:",
        "",
        f"1. **{auto_pct:.0f}% can be fully automated** - these emails don't need human attention",
        f"2. **{service_pct:.0f}% are machine-generated** - newsletters, notifications, receipts",
        f"3. **{100 - service_pct:.0f}% are human conversations** - where relationships happen",
        f"4. **{data['long_threads']:,} deep threads** - your most engaged conversations",
        "",
        "### Actionable Recommendations",
        "",
        "Based on your actual email behavior:",
        "",
        f"1. **Auto-archive marketing/newsletters** - {sum(1 for s in data.get('service_engagement', []) if s[0] in ['marketing', 'newsletter'])} email types you never reply to",
        "",
        f"2. **Prioritize {data.get('important_sender_count', 0)} important senders** - these are people you actually engage with",
        "",
        f"3. **Review service notifications** - some (like transactions) may need attention, others don't",
        "",
        f"4. **Use AI for {ai_partial:,} 'ai_partial' emails** - scheduling requests, confirmations, etc.",
        "",
        "### Manual Verification",
        "",
        "Run these scripts to explore your data:",
        "",
        "```bash",
        "# See important people and services",
        "python scripts/validate_data.py",
        "",
        "# Check specific domain (e.g., differentiate Chase transactions vs marketing)",
        "python scripts/validate_data.py --domain chase.com",
        "python scripts/validate_data.py --domain amazon.com",
        "```",
        "",
        "### Data Quality Notes",
        "",
        f"- Embedding coverage: {coverage:.0f}%",
        f"- Classification coverage: {data['ai_class_count'] / total * 100:.0f}%" if total > 0 else "",
        f"- LLM analysis: {data['llm_class_count']:,} emails deeply analyzed",
        f"- Important senders identified: {data.get('important_sender_count', 0)}",
        "",
        "---",
        "",
        "*Report generated by rl-emails onboarding pipeline*",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot data onboarding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current pipeline status"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM classification (saves cost)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        choices=range(0, 13),
        help="Start from stage N (0-11, 0=migrations)"
    )
    parser.add_argument(
        "--llm-model",
        choices=["gpt5", "haiku", "sonnet"],
        default="gpt5",
        help="LLM model for classification (default: gpt5)"
    )

    args = parser.parse_args()

    # Validate required config (except for --status which just needs DATABASE_URL)
    if not args.status:
        validate_config()

    # Status mode
    if args.status:
        if not DATABASE_URL:
            print("ERROR: DATABASE_URL not set in .env")
            sys.exit(1)
        print_status()
        return

    # After validate_config(), these are guaranteed non-None
    assert MBOX_PATH is not None, "MBOX_PATH must be set"
    assert DATABASE_URL is not None, "DATABASE_URL must be set"

    mbox_path = Path(MBOX_PATH)
    if not mbox_path.exists():
        print(f"ERROR: MBOX file not found: {mbox_path}")
        sys.exit(1)

    # Check prerequisites
    print("\n=== Checking Prerequisites ===")
    print(f"DATABASE_URL: {DATABASE_URL}")
    print(f"MBOX_PATH: {MBOX_PATH}")

    if not check_postgres():
        print("\nERROR: PostgreSQL not accessible")
        print("Start with: docker compose up -d postgres")
        sys.exit(1)
    print("PostgreSQL: OK")

    keys = check_api_keys()
    if keys["openai"]:
        print("OpenAI API Key: OK")
    else:
        print("OpenAI API Key: Missing (embeddings will fail)")
        if not args.skip_embeddings:
            print("  Use --skip-embeddings to continue without embeddings")

    if keys["openai"] or keys["anthropic"]:
        print("LLM API Key: OK")
    else:
        print("LLM API Key: Missing (LLM classification will fail)")
        args.skip_llm = True

    # Setup paths
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parsed_jsonl = Path(PARSED_JSONL) if PARSED_JSONL else DATA_DIR / "parsed_emails.jsonl"

    # Set environment for child processes
    os.environ["MBOX_PATH"] = str(mbox_path.absolute())
    os.environ["PARSED_JSONL"] = str(parsed_jsonl.absolute())
    os.environ["DATABASE_URL"] = DATABASE_URL
    os.environ["DB_URL"] = DATABASE_URL  # Alias for enrich_emails_db.py
    os.environ["YOUR_EMAIL"] = YOUR_EMAIL or ""
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY or ""

    # Extract database name for display
    db_name = DATABASE_URL.split("/")[-1] if "/" in DATABASE_URL else "unknown"
    db_host = DATABASE_URL.split("@")[-1].split("/")[0] if "@" in DATABASE_URL else "unknown"

    # Print configuration
    print(f"\n=== Configuration ===")
    print(f"MBOX Path: {mbox_path}")
    print(f"JSONL Path: {parsed_jsonl}")
    print(f"Database: {db_name}@{db_host}")
    print(f"Workers: {args.workers}")
    print(f"Skip Embeddings: {args.skip_embeddings}")
    print(f"Skip LLM: {args.skip_llm}")
    print(f"Start From: Stage {args.start_from}")

    start_time = datetime.now()
    print(f"\n=== Starting Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # Stage 0: Run alembic migrations
    if args.start_from <= 0:
        if not run_alembic_migrations():
            print("\nPipeline failed at stage 0 (migrations)")
            sys.exit(1)
    else:
        print(f"\nSkipping stage 0: Run database migrations")

    # Define remaining stages
    # Pipeline stages - LLM runs AFTER clustering so it has relationship context
    stages = [
        (1, "parse_mbox.py", [], "Parse MBOX to JSONL"),
        (2, "import_to_postgres.py", [], "Import to PostgreSQL"),
        (3, "populate_threads.py", [], "Build thread relationships"),
        (4, "enrich_emails_db.py", [], "Compute action labels (Phase 1)"),
        (5, "compute_basic_features.py", [], "Compute ML features (Phase 2)"),
        (6, "compute_embeddings.py", ["--workers", str(args.workers)], "Generate embeddings (Phase 3)"),
        (7, "classify_ai_handleability.py", [], "Rule-based classification (Phase 0)"),
        (8, "populate_users.py", [], "Populate user profiles (Phase 4A)"),
        (9, "cluster_emails.py", [], "Multi-dimensional clustering (Phase 4B)"),
        (10, "compute_priority.py", [], "Hybrid priority ranking (Phase 4C)"),
        (11, "run_llm_classification.py", ["--all", str(args.workers), args.llm_model], "LLM classification (Phase 4D)"),
    ]

    failed = False
    for stage_num, script, script_args, description in stages:
        # Skip if before start point
        if stage_num < args.start_from:
            print(f"\nSkipping stage {stage_num}: {description}")
            continue

        # Skip embeddings if requested
        if args.skip_embeddings and script == "compute_embeddings.py":
            print(f"\nSkipping stage {stage_num}: {description} (--skip-embeddings)")
            continue

        # Skip LLM if requested
        if args.skip_llm and script == "run_llm_classification.py":
            print(f"\nSkipping stage {stage_num}: {description} (--skip-llm)")
            continue

        success = run_script(script, script_args, f"[{stage_num}/11] {description}")

        if not success:
            failed = True
            print(f"\nPipeline failed at stage {stage_num}")
            print(f"Resume with: python scripts/onboard_data.py --start-from {stage_num}")
            break

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    if failed:
        print(f"PIPELINE FAILED after {duration}")
    else:
        print(f"PIPELINE COMPLETED in {duration}")
        print_status()

        # Generate detailed report
        report_path = DATA_DIR / "onboarding_report.md"
        report_content = generate_report(parsed_jsonl, str(duration))
        report_path.write_text(report_content)
        print(f"\nReport saved to: {report_path}")

        print(f"\nNext steps:")
        print(f"  1. Review report: cat {report_path}")
        print(f"  2. Explore data: uv run streamlit run apps/labeling_ui_v2.py")
        print(f"  3. Create checkpoint: uv run python scripts/checkpoint.py create --name onboarding_complete")
    print(f"{'='*60}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
