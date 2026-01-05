#!/usr/bin/env python3
"""Async parallel email feature extraction using Claude Haiku 4.5.

Uses anthropic SDK with async parallel requests for fast extraction.
Target: ~0.3-0.5s per email.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import psycopg2

try:
    import anthropic
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Claude Haiku configuration
HAIKU_MODEL = "claude-haiku-4-5-20250120"

# PostgreSQL configuration
PG_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "rl_emails",
    "user": "postgres",
    "password": "postgres",
}


@dataclass
class ExtractedFeatures:
    """All features extracted from an email."""
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


# One-shot extraction prompt
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


def get_emails_from_db(limit: int = 20) -> list[dict]:
    """Query sample emails from PostgreSQL."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, message_id, subject, body_text, from_email, date_parsed
                FROM emails
                WHERE body_text IS NOT NULL AND body_text != ''
                  AND subject IS NOT NULL AND length(body_text) > 50
                ORDER BY date_parsed DESC
                LIMIT %s
            """, (limit,))

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


def parse_llm_response(text: str) -> Optional[dict]:
    """Parse LLM response to extract JSON."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object
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
    """Extract features from a single email using Claude Haiku."""

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
                messages=[
                    {"role": "user", "content": user_message}
                ],
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


async def process_batch(
    emails: list[dict],
    workers: int,
) -> tuple[list[ExtractedFeatures], float]:
    """Process all emails with parallel workers."""

    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(workers)

    start_time = time.time()

    tasks = [
        extract_features_async(client, email, semaphore)
        for email in emails
    ]

    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    return list(results), total_time


def save_to_database(features_list: list[ExtractedFeatures], model: str) -> int:
    """Save extracted features to database."""
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
                        f.parse_success, model,
                    ))
                    inserted += 1
                except Exception as e:
                    print(f"  Warning: Insert failed: {e}", file=sys.stderr)

            conn.commit()
            return inserted
    finally:
        conn.close()


async def main_async(args):
    """Async main entry point."""

    workers = args.workers
    count = args.count

    print("=" * 60)
    print("Claude Haiku 4.5 Parallel Email Feature Extraction")
    print("=" * 60)
    print(f"  Model: {HAIKU_MODEL}")
    print(f"  Workers: {workers}")
    print(f"  Emails: {count}")
    print(f"  Target: ~0.3-0.5s/email effective throughput")
    print()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        return 1

    print("1. API key found")
    print()

    # Query emails
    print("2. Querying emails...")
    try:
        emails = get_emails_from_db(limit=count)
        print(f"   Retrieved {len(emails)} emails")
    except Exception as e:
        print(f"   ERROR: {e}")
        return 1
    print()

    # Process in parallel
    print(f"3. Processing with {workers} parallel workers...")
    results, total_time = await process_batch(emails, workers)

    # Calculate metrics
    successes = sum(1 for r in results if r.parse_success)
    times_ms = [r.extraction_time_ms for r in results]
    avg_time_ms = sum(times_ms) / len(times_ms) if times_ms else 0
    throughput = len(emails) / total_time if total_time > 0 else 0
    effective_time_per_email = total_time / len(emails) if emails else 0

    print(f"   Completed in {total_time:.2f}s")
    print()

    # Save to database if requested
    if args.save_db:
        print("4. Saving to database...")
        inserted = save_to_database(results, HAIKU_MODEL)
        print(f"   Saved {inserted} records")
        print()

    # Report
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print("Timing:")
    print(f"  Total wall time:     {total_time:.2f}s")
    print(f"  Effective per email: {effective_time_per_email:.3f}s")
    print(f"  Throughput:          {throughput:.2f} emails/s")
    print(f"  Target:              0.3-0.5s/email")
    print(f"  Status:              {'PASS' if effective_time_per_email <= 0.5 else 'FAIL'}")
    print()
    print("Per-request timing:")
    print(f"  Avg API time:        {avg_time_ms:.0f}ms")
    print(f"  Min:                 {min(times_ms):.0f}ms")
    print(f"  Max:                 {max(times_ms):.0f}ms")
    print()
    print("Quality:")
    print(f"  Parse success:       {successes}/{len(results)} ({successes/len(results)*100:.0f}%)")
    print(f"  Tasks found:         {sum(len(r.tasks) for r in results)}")
    print(f"  Service emails:      {sum(1 for r in results if r.is_service_email)}/{len(results)}")
    print()

    # Comparison
    print("Comparison to local models:")
    print(f"  phi-4 (LM Studio):   5.38s/email")
    print(f"  gpt-oss:20b (Ollama): 4.76s/email")
    print(f"  Haiku 4.5 (API):     {effective_time_per_email:.3f}s/email")
    if effective_time_per_email > 0:
        print(f"  Speedup vs phi-4:    {5.38 / effective_time_per_email:.1f}x")
    print()

    # Save JSON output
    output_file = "haiku_features.json"
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": HAIKU_MODEL,
        "workers": workers,
        "email_count": len(emails),
        "timing": {
            "total_seconds": round(total_time, 3),
            "effective_per_email": round(effective_time_per_email, 4),
            "throughput": round(throughput, 2),
            "avg_api_ms": round(avg_time_ms, 1),
            "target_met": effective_time_per_email <= 0.5,
        },
        "quality": {
            "parse_success_rate": round(successes / len(results) * 100, 1) if results else 0,
            "total_tasks": sum(len(r.tasks) for r in results),
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to: {output_file}")
    print("=" * 60)

    return 0


def main():
    if not HAS_ANTHROPIC:
        print("ERROR: anthropic package not installed")
        print("Install with: pip install anthropic")
        return 1

    parser = argparse.ArgumentParser(
        description="Parallel email feature extraction using Claude Haiku 4.5"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=20,
        help="Number of emails to process (default: 20)"
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Save results to database"
    )
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
