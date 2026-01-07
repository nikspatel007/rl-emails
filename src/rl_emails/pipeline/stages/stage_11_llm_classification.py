"""Stage 11: LLM classification for emails.

Uses LiteLLM to classify emails that need deeper analysis.
Stores prompts and responses for ML training.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import psycopg2

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult

# Model mapping (LiteLLM format)
MODELS = {
    "gpt5": "gpt-5-mini",
    "haiku": "anthropic/claude-haiku-4-5",
    "sonnet": "anthropic/claude-sonnet-4-5",
}

DEFAULT_MODEL = "gpt5"
DEFAULT_WORKERS = 10
DEFAULT_BATCH_SIZE = 100

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


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create LLM classification table if it doesn't exist.

    Args:
        conn: Database connection.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_llm_classification (
                id SERIAL PRIMARY KEY,
                email_id INTEGER REFERENCES emails(id) UNIQUE,
                raw_response TEXT,
                raw_prompt TEXT,
                action_type TEXT,
                urgency TEXT,
                ai_can_handle TEXT,
                next_step TEXT,
                suggested_action TEXT,
                one_liner TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                thread_id INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_email_llm_classification_email_id
            ON email_llm_classification(email_id)
        """
        )

        conn.commit()


def get_model(
    model_name: str | None, openai_key: str | None, anthropic_key: str | None
) -> str | None:
    """Get model based on name or available API keys.

    Args:
        model_name: Optional model name (gpt5, haiku, sonnet).
        openai_key: OpenAI API key.
        anthropic_key: Anthropic API key.

    Returns:
        LiteLLM model string or None.
    """
    if model_name and model_name in MODELS:
        return MODELS[model_name]

    if openai_key:
        return MODELS["gpt5"]
    if anthropic_key:
        return MODELS["haiku"]
    return None


def get_status(conn: psycopg2.extensions.connection) -> dict[str, int]:
    """Get current processing status.

    Args:
        conn: Database connection.

    Returns:
        Dictionary with total, processed, remaining counts.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM emails e
            JOIN email_ai_classification ac ON ac.email_id = e.id
            WHERE ac.predicted_handleability = 'needs_llm'
            AND e.is_sent = FALSE
            AND LENGTH(e.body_text) > 50
        """
        )
        row = cur.fetchone()
        total = row[0] if row else 0

        cur.execute("SELECT COUNT(*) FROM email_llm_classification")
        row = cur.fetchone()
        processed = row[0] if row else 0

        return {"total": total, "processed": processed, "remaining": total - processed}


def get_emails_to_process(
    conn: psycopg2.extensions.connection, limit: int = 50
) -> list[tuple[Any, ...]]:
    """Get emails that need LLM classification.

    Args:
        conn: Database connection.
        limit: Maximum number of emails to return.

    Returns:
        List of email tuples with context.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
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
        """,
            (limit,),
        )

        return cur.fetchall()


def build_prompt(email_data: tuple[Any, ...]) -> str:
    """Build the LLM prompt for an email.

    Args:
        email_data: Email tuple from get_emails_to_process.

    Returns:
        Formatted prompt string.
    """
    (
        email_id,
        from_email,
        subject,
        date_parsed,
        body_preview,
        relationship,
        reply_rate,
        is_service,
        llm_priority,
        actual_action,
        thread_id,
        is_important_sender,
        priority_score,
        priority_rank,
    ) = email_data

    body_clean = body_preview or ""
    if "<html" in body_clean.lower() or "<head" in body_clean.lower():
        body_clean = re.sub(r"<[^>]+>", " ", body_clean)
        body_clean = re.sub(r"\s+", " ", body_clean).strip()[:500]

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
        llm_priority=llm_priority,
    )

    return prompt


def classify_email(
    email_data: tuple[Any, ...],
    model: str,
    completion_func: Any,
) -> dict[str, Any]:
    """Run LLM classification on a single email.

    Args:
        email_data: Email tuple.
        model: LiteLLM model string.
        completion_func: LiteLLM completion function.

    Returns:
        Classification result dictionary.
    """
    email_id = email_data[0]
    thread_id = email_data[10]

    prompt = build_prompt(email_data)

    try:
        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
        }
        if model.startswith("gpt-5"):
            completion_kwargs["reasoning_effort"] = "minimal"
        else:
            completion_kwargs["temperature"] = 0

        response = completion_func(**completion_kwargs)
        result_text = response.choices[0].message.content.strip()

        try:
            if "```" in result_text:
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result_text)
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
                "total": response.usage.total_tokens,
            },
            "error": None,
        }

    except Exception as e:
        return {
            "email_id": email_id,
            "thread_id": thread_id,
            "raw_prompt": prompt,
            "raw_response": None,
            "result": None,
            "tokens": None,
            "error": str(e),
        }


def save_result(
    cur: psycopg2.extensions.cursor,
    result: dict[str, Any],
    model: str,
) -> bool:
    """Save a classification result to the database.

    Args:
        cur: Database cursor.
        result: Classification result.
        model: Model used.

    Returns:
        True if saved successfully.
    """
    if result["error"] or not result["result"]:
        return False

    r = result["result"]

    try:
        cur.execute(
            """
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
        """,
            (
                result["email_id"],
                result["raw_response"],
                result["raw_prompt"],
                r.get("action_type"),
                r.get("urgency"),
                r.get("ai_can_handle"),
                r.get("next_step"),
                r.get("suggested_action"),
                r.get("one_liner"),
                model,
                result["tokens"]["prompt"],
                result["tokens"]["completion"],
                result["tokens"]["total"],
                result["thread_id"],
            ),
        )
        return True
    except Exception:
        return False


def process_batch(
    emails: list[tuple[Any, ...]],
    workers: int,
    model: str,
    conn: psycopg2.extensions.connection,
    completion_func: Any,
) -> dict[str, Any]:
    """Process a batch of emails in parallel.

    Args:
        emails: List of email tuples.
        workers: Number of parallel workers.
        model: LiteLLM model string.
        conn: Database connection.
        completion_func: LiteLLM completion function.

    Returns:
        Batch results dictionary.
    """
    results = []
    total_tokens = 0
    errors = 0
    saved = 0

    cur = conn.cursor()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(classify_email, email, model, completion_func): email
            for email in emails
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result["error"]:
                errors += 1
            else:
                total_tokens += result["tokens"]["total"]
                if save_result(cur, result, model):
                    saved += 1

    conn.commit()
    cur.close()

    return {"results": results, "tokens": total_tokens, "errors": errors, "saved": saved}


def run(
    config: Config,
    model_name: str | None = None,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> StageResult:
    """Run Stage 11: LLM classification.

    Args:
        config: Application configuration.
        model_name: Optional model name.
        workers: Number of parallel workers.
        batch_size: Emails per batch.
        limit: Maximum emails to process.

    Returns:
        StageResult with classification statistics.
    """
    start_time = time.time()

    model = get_model(model_name, config.openai_api_key, config.anthropic_api_key)
    if not model:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="No API key configured (need OPENAI_API_KEY or ANTHROPIC_API_KEY)",
        )

    try:
        from litellm import completion as litellm_completion
    except ImportError:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="litellm package not installed",
        )

    conn = psycopg2.connect(config.database_url)
    try:
        create_tables(conn)

        status = get_status(conn)
        remaining = status["remaining"]

        if limit:
            remaining = min(remaining, limit)

        if remaining == 0:
            duration = time.time() - start_time
            return StageResult(
                success=True,
                records_processed=0,
                duration_seconds=duration,
                message="All emails already classified",
                metadata=status,
            )

        total_processed = 0
        total_tokens = 0
        total_errors = 0
        total_saved = 0

        while total_processed < remaining:
            batch_limit = min(batch_size, remaining - total_processed)
            emails = get_emails_to_process(conn, limit=batch_limit)

            if not emails:
                break

            batch_result = process_batch(emails, workers, model, conn, litellm_completion)
            total_processed += len(batch_result["results"])
            total_tokens += batch_result["tokens"]
            total_errors += batch_result["errors"]
            total_saved += batch_result["saved"]

        duration = time.time() - start_time

        return StageResult(
            success=True,
            records_processed=total_saved,
            duration_seconds=duration,
            message=f"Classified {total_saved} emails ({total_errors} errors)",
            metadata={
                "model": model,
                "total_tokens": total_tokens,
                "errors": total_errors,
                "processed": total_processed,
            },
        )
    finally:
        conn.close()
