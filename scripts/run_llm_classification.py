#!/usr/bin/env python3
"""
Run LLM classification on emails with parallel workers.
Saves raw prompts and responses for ML training.

Usage:
    python run_llm_classification.py <limit> <workers> [model]
    python run_llm_classification.py --all <workers> [model]

    limit: number of emails to process (or --all for all remaining)
    workers: number of parallel workers (default: 10)
    model: gpt5, haiku, sonnet (default: gpt5 if OPENAI_API_KEY set)

Examples:
    python run_llm_classification.py 100 10 gpt5    # Process 100 emails
    python run_llm_classification.py --all 10       # Process all remaining
    python run_llm_classification.py --status       # Show current status
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import psycopg2
from dotenv import load_dotenv
from litellm import completion

# Load .env file
load_dotenv()

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    print("ERROR: DATABASE_URL environment variable is required")
    sys.exit(1)

# Model mapping (LiteLLM format)
MODELS = {
    "gpt5": "gpt-5-mini",
    "haiku": "anthropic/claude-haiku-4-5",
    "sonnet": "anthropic/claude-sonnet-4-5",
}

MODEL = None  # Set in main()

PROMPT_TEMPLATE = """Classify this email for an AI assistant. Be concise. Respond with JSON only.

## Email
From: {from_email}
Subject: {subject}
Date: {date}

---
{body_preview}
---

## Known Context (from ML pipeline)
- Sender relationship: {relationship_level} ({reply_rate}% reply rate to this sender)
- Important sender: {is_important}
- Is service/automated email: {is_service}
- Computed priority score: {priority_score} (rank #{priority_rank})
- LLM Priority tier: {llm_priority}

## Classify (respond with valid JSON only, no markdown):
{{
  "action_type": "reply|task|decision|approval|fyi|none",
  "urgency": "immediate|today|this_week|whenever|none",
  "ai_can_handle": "fully|partially|not_at_all",
  "next_step": "skip|quick_action|needs_deep_analysis",
  "suggested_action": "FILE_TO_FOLDER|DRAFT_REPLY|SCHEDULE_MEETING|SUMMARIZE_ATTACHMENT|PREPARE_CONTEXT|null",
  "one_liner": "10 words max summary of what this email is about"
}}"""


def get_model(model_arg: str | None = None) -> str | None:
    """Get model based on argument or available API keys."""
    if model_arg and model_arg in MODELS:
        return MODELS[model_arg]

    # Default based on available keys
    if os.environ.get("OPENAI_API_KEY"):
        return MODELS["gpt5"]
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return MODELS["haiku"]
    return None


def get_status(conn: psycopg2.extensions.connection) -> dict[str, int]:
    """Get current processing status."""
    cur = conn.cursor()

    # Total needs_llm emails
    cur.execute("""
        SELECT COUNT(*) FROM emails e
        JOIN email_ai_classification ac ON ac.email_id = e.id
        WHERE ac.predicted_handleability = 'needs_llm'
        AND e.is_sent = FALSE
        AND LENGTH(e.body_text) > 50
    """)
    row = cur.fetchone()
    total = row[0] if row else 0

    # Already processed
    cur.execute("SELECT COUNT(*) FROM email_llm_classification")
    row = cur.fetchone()
    processed = row[0] if row else 0

    # Remaining
    remaining = total - processed

    return {"total": total, "processed": processed, "remaining": remaining}


def get_emails_to_process(
    conn: psycopg2.extensions.connection, limit: int = 50
) -> list[tuple[Any, ...]]:
    """Get emails that need LLM classification with user/priority context."""
    cur = conn.cursor()

    cur.execute("""
        SELECT
            e.id,
            e.from_email,
            e.subject,
            e.date_parsed,
            LEFT(e.body_text, 800) as body_preview,
            ef.relationship_strength,
            ef.user_replied_to_sender_rate,
            ef.is_service_email,
            ac.llm_priority,
            e.action as actual_action,
            e.thread_id,
            COALESCE(u.is_important_sender, FALSE) as is_important_sender,
            COALESCE(ep.priority_score, 0) as priority_score,
            COALESCE(ep.priority_rank, 9999) as priority_rank
        FROM emails e
        JOIN email_features ef ON ef.email_id = e.id
        JOIN email_ai_classification ac ON ac.email_id = e.id
        LEFT JOIN users u ON LOWER(u.email) = LOWER(e.from_email)
        LEFT JOIN email_priority ep ON ep.email_id = e.id
        WHERE ac.predicted_handleability = 'needs_llm'
        AND e.is_sent = FALSE
        AND LENGTH(e.body_text) > 50
        AND e.id NOT IN (SELECT email_id FROM email_llm_classification)
        ORDER BY ac.llm_priority, RANDOM()
        LIMIT %s
    """, (limit,))

    return cur.fetchall()


def build_prompt(email_data: tuple[Any, ...]) -> str:
    """Build the LLM prompt for an email with ML context."""
    email_id, from_email, subject, date_parsed, body_preview, \
        relationship, reply_rate, is_service, llm_priority, actual_action, thread_id, \
        is_important_sender, priority_score, priority_rank = email_data

    # Clean body preview
    body_clean = body_preview or ""
    if "<html" in body_clean.lower() or "<head" in body_clean.lower():
        body_clean = re.sub(r'<[^>]+>', ' ', body_clean)
        body_clean = re.sub(r'\s+', ' ', body_clean).strip()[:500]

    # Determine relationship level
    if relationship > 0.6:
        rel_level = "STRONG"
    elif relationship > 0.3:
        rel_level = "MODERATE"
    elif relationship > 0.1:
        rel_level = "WEAK"
    else:
        rel_level = "MINIMAL"

    prompt = PROMPT_TEMPLATE.format(
        from_email=from_email,
        subject=subject,
        date=str(date_parsed)[:10] if date_parsed else "unknown",
        body_preview=body_clean[:500] if body_clean else "(empty body)",
        relationship_level=rel_level,
        reply_rate=round((reply_rate or 0) * 100, 1),
        is_important="Yes (high engagement history)" if is_important_sender else "No",
        is_service="Yes" if is_service else "No",
        priority_score=round(priority_score, 2) if priority_score else 0,
        priority_rank=priority_rank if priority_rank < 9999 else "N/A",
        llm_priority=llm_priority
    )

    return prompt


def classify_email(email_data: tuple[Any, ...]) -> dict[str, Any]:
    """Run LLM classification on a single email."""
    email_id = email_data[0]
    thread_id = email_data[10]  # thread_id is still at index 10

    prompt = build_prompt(email_data)

    try:
        # Build completion kwargs - MODEL is set by main() before this runs
        assert MODEL is not None, "MODEL must be set before calling classify_email"
        completion_kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
        }
        if MODEL.startswith("gpt-5"):
            completion_kwargs["reasoning_effort"] = "minimal"
        else:
            completion_kwargs["temperature"] = 0

        response = completion(**completion_kwargs)
        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            if "```" in result_text:
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result_text)
                if json_match:
                    result_text = json_match.group(1)
            result = json.loads(result_text)
        except json.JSONDecodeError:
            result = {"raw_response": result_text, "parse_error": True}

        return {
            "email_id": email_id,
            "thread_id": thread_id,
            "raw_prompt": prompt,
            "raw_response": result_text,
            "result": result,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            },
            "error": None
        }

    except Exception as e:
        return {
            "email_id": email_id,
            "thread_id": thread_id,
            "raw_prompt": prompt,
            "raw_response": None,
            "result": None,
            "tokens": None,
            "error": str(e)
        }


def save_result(cur: psycopg2.extensions.cursor, result: dict[str, Any]) -> bool:
    """Save a classification result to the database."""
    if result["error"] or not result["result"]:
        return False

    r = result["result"]

    try:
        cur.execute("""
            INSERT INTO email_llm_classification (
                email_id, raw_response, raw_prompt,
                action_type, urgency, ai_can_handle, next_step,
                suggested_action, one_liner,
                model, prompt_tokens, completion_tokens, total_tokens,
                thread_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (email_id) DO UPDATE SET
                raw_response = EXCLUDED.raw_response,
                raw_prompt = EXCLUDED.raw_prompt,
                action_type = EXCLUDED.action_type,
                urgency = EXCLUDED.urgency,
                ai_can_handle = EXCLUDED.ai_can_handle,
                next_step = EXCLUDED.next_step,
                suggested_action = EXCLUDED.suggested_action,
                one_liner = EXCLUDED.one_liner,
                model = EXCLUDED.model,
                prompt_tokens = EXCLUDED.prompt_tokens,
                completion_tokens = EXCLUDED.completion_tokens,
                total_tokens = EXCLUDED.total_tokens,
                thread_id = EXCLUDED.thread_id,
                created_at = NOW()
        """, (
            result["email_id"],
            result["raw_response"],
            result["raw_prompt"],
            r.get("action_type"),
            r.get("urgency"),
            r.get("ai_can_handle"),
            r.get("next_step"),
            r.get("suggested_action"),
            r.get("one_liner"),
            MODEL,
            result["tokens"]["prompt"],
            result["tokens"]["completion"],
            result["tokens"]["total"],
            result["thread_id"]
        ))
        return True
    except Exception as e:
        print(f"  DB ERROR for email {result['email_id']}: {e}")
        return False


def process_batch(
    emails: list[tuple[Any, ...]], workers: int, conn: psycopg2.extensions.connection
) -> dict[str, Any]:
    """Process a batch of emails in parallel and save results."""
    results = []
    total_tokens = 0
    errors = 0
    saved = 0

    cur = conn.cursor()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_email, email): email for email in emails}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            if result["error"]:
                errors += 1
                print(f"  [{i}/{len(emails)}] Email {result['email_id']}: ERROR - {result['error'][:50]}")
            else:
                total_tokens += result["tokens"]["total"]
                r = result["result"]
                # Save immediately
                if save_result(cur, result):
                    saved += 1
                print(f"  [{i}/{len(emails)}] Email {result['email_id']}: {r.get('action_type', '?')} / {r.get('one_liner', '?')[:35]}")

    conn.commit()
    return {"results": results, "tokens": total_tokens, "errors": errors, "saved": saved}


def main() -> None:
    global MODEL

    # Handle --status flag
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        conn = psycopg2.connect(DB_URL)
        status = get_status(conn)
        conn.close()
        print(f"Total needs_llm emails: {status['total']}")
        print(f"Already processed: {status['processed']}")
        print(f"Remaining: {status['remaining']}")
        return

    # Parse arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        limit = None  # Will be set after checking status
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        model_arg = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        model_arg = sys.argv[3] if len(sys.argv) > 3 else None

    MODEL = get_model(model_arg)

    if not MODEL:
        print("ERROR: No API key found")
        print("Set either ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        return

    # Get status
    conn = psycopg2.connect(DB_URL)
    status = get_status(conn)

    if limit is None:
        limit = status["remaining"]

    print("=" * 60)
    print("LLM EMAIL CLASSIFICATION")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Workers: {workers}")
    print(f"Status: {status['processed']}/{status['total']} processed, {status['remaining']} remaining")
    print(f"This run: {min(limit, status['remaining'])} emails")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if status["remaining"] == 0:
        print("\nAll emails already processed!")
        conn.close()
        return

    # Process in batches of 100 for better progress tracking
    batch_size = 100
    total_processed = 0
    total_tokens = 0
    total_errors = 0
    total_saved = 0
    start_time = datetime.now()

    emails_to_process = min(limit, status["remaining"])
    batches = (emails_to_process + batch_size - 1) // batch_size

    for batch_num in range(batches):
        batch_limit = min(batch_size, emails_to_process - total_processed)
        emails = get_emails_to_process(conn, limit=batch_limit)

        if not emails:
            break

        print(f"\n--- Batch {batch_num + 1}/{batches} ({len(emails)} emails) ---")

        batch_result = process_batch(emails, workers, conn)
        total_processed += len(batch_result["results"])
        total_tokens += batch_result["tokens"]
        total_errors += batch_result["errors"]
        total_saved += batch_result["saved"]

        # Progress update
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining = emails_to_process - total_processed
        eta = remaining / rate if rate > 0 else 0

        print(f"  Progress: {total_processed}/{emails_to_process} | Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}min")

    conn.close()

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Emails processed: {total_processed}")
    print(f"Saved to DB: {total_saved}")
    print(f"Errors: {total_errors}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Average speed: {total_processed/elapsed:.1f} emails/sec")
    print(f"Estimated cost: ${total_tokens * 0.15 / 1_000_000:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
