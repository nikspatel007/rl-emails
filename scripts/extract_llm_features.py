#!/usr/bin/env python3
"""Extract LLM features from emails using Claude Haiku.

Stage 5 of the pipeline: Uses Claude Haiku to extract tasks, urgency,
and topic classification from emails. Results are stored in email_llm_features table.

Requires: ANTHROPIC_API_KEY environment variable
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import psycopg2

try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Claude Haiku configuration
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# PostgreSQL configuration
PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", "5433")),
    "database": os.getenv("PG_DATABASE", "rl_emails"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
}


@dataclass
class ExtractedFeatures:
    email_id: str
    is_service_email: bool
    service_type: Optional[str]
    tasks: list[dict]
    overall_urgency: float
    requires_response: bool
    topic_category: str
    summary: str
    extraction_time_ms: int
    parse_success: bool


SYSTEM_PROMPT = """You are an email analysis expert. Extract features from the email and respond with ONLY valid JSON.

Output this exact JSON structure (no markdown, no explanation):
{
  "is_service_email": true or false,
  "service_type": "newsletter" | "notification" | "billing" | "social" | "marketing" | "system" | "calendar" | null,
  "tasks": [{"description": "...", "deadline": "..." or null, "assignee": "user" | "other" | null, "task_type": "review" | "send" | "schedule" | "decision" | "create" | "followup", "urgency": 0.5}],
  "overall_urgency": 0.0 to 1.0,
  "requires_response": true or false,
  "topic_category": "project" | "meeting" | "request" | "fyi" | "social" | "admin",
  "summary": "one line summary"
}

Rules:
- is_service_email: true for automated/system emails (newsletters, notifications, receipts)
- service_type: only set if is_service_email is true
- tasks: extract ALL actionable items, empty list [] if none
- overall_urgency: 0.0-0.3 normal, 0.3-0.6 moderate, 0.6-1.0 urgent
- requires_response: true if sender expects a reply
- topic_category: main purpose of the email
- summary: max 100 chars

Output ONLY the JSON object, nothing else."""


def get_unprocessed_emails(limit: int = 500) -> list[dict]:
    """Get emails not yet processed with this model."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.message_id, e.subject, e.body_text, e.from_email, e.date_parsed
                FROM emails e
                LEFT JOIN email_llm_features f ON e.message_id = f.email_id AND f.model = %s
                WHERE e.body_text IS NOT NULL AND e.body_text != ''
                  AND e.subject IS NOT NULL AND length(e.body_text) > 50
                  AND f.id IS NULL
                ORDER BY e.date_parsed DESC
                LIMIT %s
            """, (HAIKU_MODEL, limit))
            rows = cur.fetchall()
            return [{
                "id": row[0],
                "message_id": row[1],
                "subject": row[2] or "",
                "body_text": row[3] or "",
                "from_email": row[4] or "",
                "date_parsed": str(row[5]) if row[5] else "",
            } for row in rows]
    finally:
        conn.close()


def get_total_counts() -> tuple[int, int]:
    """Get total eligible and processed email counts."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM emails
                WHERE body_text IS NOT NULL AND body_text != ''
                  AND subject IS NOT NULL AND length(body_text) > 50
            """)
            total = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM email_llm_features WHERE model = %s
            """, (HAIKU_MODEL,))
            processed = cur.fetchone()[0]

            return total, processed
    finally:
        conn.close()


def parse_llm_response(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return None


async def extract_features_async(
    client: "AsyncAnthropic",
    email: dict,
    semaphore: asyncio.Semaphore,
) -> ExtractedFeatures:
    email_id = email["message_id"]
    subject = email["subject"]
    body = email["body_text"][:2000]
    sender = email["from_email"]

    user_message = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
    start_time = time.time()

    async with semaphore:
        try:
            response = await client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            response_text = response.content[0].text
            data = parse_llm_response(response_text)

            if data:
                return ExtractedFeatures(
                    email_id=email_id,
                    is_service_email=data.get("is_service_email", False),
                    service_type=data.get("service_type"),
                    tasks=data.get("tasks", []),
                    overall_urgency=float(data.get("overall_urgency", 0.0)),
                    requires_response=data.get("requires_response", False),
                    topic_category=data.get("topic_category", "fyi"),
                    summary=str(data.get("summary", ""))[:100],
                    extraction_time_ms=elapsed_ms,
                    parse_success=True,
                )
            else:
                return ExtractedFeatures(
                    email_id=email_id,
                    is_service_email=False,
                    service_type=None,
                    tasks=[],
                    overall_urgency=0.0,
                    requires_response=False,
                    topic_category="fyi",
                    summary="JSON parse failed",
                    extraction_time_ms=elapsed_ms,
                    parse_success=False,
                )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExtractedFeatures(
                email_id=email_id,
                is_service_email=False,
                service_type=None,
                tasks=[],
                overall_urgency=0.0,
                requires_response=False,
                topic_category="fyi",
                summary=f"Error: {str(e)[:50]}",
                extraction_time_ms=elapsed_ms,
                parse_success=False,
            )


async def process_batch(emails: list[dict], workers: int) -> list[ExtractedFeatures]:
    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(workers)
    tasks = [extract_features_async(client, email, semaphore) for email in emails]
    return list(await asyncio.gather(*tasks))


def save_batch(features_list: list[ExtractedFeatures]) -> int:
    """Save a batch of results to database."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS email_llm_features (
                    id SERIAL PRIMARY KEY,
                    email_id TEXT NOT NULL,
                    is_service_email BOOLEAN,
                    service_type TEXT,
                    tasks JSONB,
                    overall_urgency FLOAT,
                    requires_response BOOLEAN,
                    topic_category TEXT,
                    summary TEXT,
                    extraction_time_ms INTEGER,
                    parse_success BOOLEAN,
                    model TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(email_id, model)
                )
            """)

            inserted = 0
            for f in features_list:
                try:
                    cur.execute("""
                        INSERT INTO email_llm_features
                        (email_id, is_service_email, service_type, tasks,
                         overall_urgency, requires_response, topic_category,
                         summary, extraction_time_ms, parse_success, model)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (email_id, model) DO UPDATE SET
                            is_service_email = EXCLUDED.is_service_email,
                            service_type = EXCLUDED.service_type,
                            tasks = EXCLUDED.tasks,
                            overall_urgency = EXCLUDED.overall_urgency,
                            requires_response = EXCLUDED.requires_response,
                            topic_category = EXCLUDED.topic_category,
                            summary = EXCLUDED.summary,
                            extraction_time_ms = EXCLUDED.extraction_time_ms,
                            parse_success = EXCLUDED.parse_success,
                            created_at = NOW()
                    """, (
                        f.email_id, f.is_service_email, f.service_type,
                        json.dumps(f.tasks), f.overall_urgency, f.requires_response,
                        f.topic_category, f.summary, f.extraction_time_ms,
                        f.parse_success, HAIKU_MODEL,
                    ))
                    inserted += 1
                except Exception as e:
                    print(f"  Warning: Insert failed: {e}", file=sys.stderr)

            conn.commit()
            return inserted
    finally:
        conn.close()


async def main_async(args):
    workers = args.workers
    batch_size = args.batch_size

    print("=" * 60)
    print("Claude Haiku LLM Feature Extraction")
    print("=" * 60)
    print(f"  Model: {HAIKU_MODEL}")
    print(f"  Workers: {workers}")
    print(f"  Batch size: {batch_size}")
    print()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Set with: export ANTHROPIC_API_KEY='sk-ant-...'")
        return 1

    try:
        total_emails, already_processed = get_total_counts()
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}")
        print("Ensure PostgreSQL is running and accessible")
        return 1

    remaining = total_emails - already_processed

    print(f"Database status:")
    print(f"  Total eligible emails: {total_emails:,}")
    print(f"  Already processed:     {already_processed:,}")
    print(f"  Remaining:             {remaining:,}")
    print()

    if remaining == 0:
        print("All emails already processed!")
        return 0

    total_processed = 0
    total_successes = 0
    total_tasks_found = 0
    all_times = []
    start_time = time.time()
    batch_num = 0

    while True:
        batch_num += 1
        emails = get_unprocessed_emails(limit=batch_size)

        if not emails:
            print("\nNo more unprocessed emails.")
            break

        batch_start = time.time()
        print(f"\nBatch {batch_num}: Processing {len(emails)} emails...")
        sys.stdout.flush()

        results = await process_batch(emails, workers)
        batch_time = time.time() - batch_start

        # Save immediately
        saved = save_batch(results)

        # Track stats
        successes = sum(1 for r in results if r.parse_success)
        tasks_found = sum(len(r.tasks) for r in results)
        times = [r.extraction_time_ms for r in results]

        total_processed += len(results)
        total_successes += successes
        total_tasks_found += tasks_found
        all_times.extend(times)

        # Progress report
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining_now = remaining - total_processed
        eta_seconds = remaining_now / rate if rate > 0 else 0
        eta_minutes = eta_seconds / 60

        print(f"  Saved {saved} records in {batch_time:.1f}s")
        print(f"  Success rate: {successes}/{len(results)} ({successes/len(results)*100:.0f}%)")
        print(f"  Progress: {total_processed:,}/{remaining:,} ({total_processed/remaining*100:.1f}%)")
        print(f"  Rate: {rate:.2f} emails/s | ETA: {eta_minutes:.0f} min")
        sys.stdout.flush()

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print()
    print(f"Total processed: {total_processed:,}")
    print(f"Total time:      {total_time/60:.1f} minutes")
    if total_time > 0:
        print(f"Avg rate:        {total_processed/total_time:.2f} emails/s")
    print()
    if total_processed > 0:
        print(f"Parse success:   {total_successes}/{total_processed} ({total_successes/total_processed*100:.1f}%)")
    print(f"Tasks found:     {total_tasks_found:,}")
    print()
    if all_times:
        print(f"API timing:")
        print(f"  Avg: {sum(all_times)/len(all_times):.0f}ms")
        print(f"  Min: {min(all_times):.0f}ms")
        print(f"  Max: {max(all_times):.0f}ms")
    print("=" * 60)

    return 0


def main():
    if not HAS_ANTHROPIC:
        print("ERROR: anthropic package not installed")
        print("Install with: uv pip install anthropic")
        return 1

    parser = argparse.ArgumentParser(
        description="Extract LLM features from emails using Claude Haiku",
        epilog="Requires ANTHROPIC_API_KEY environment variable"
    )
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--batch-size", "-b", type=int, default=500,
                        help="Emails per batch (default: 500)")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
