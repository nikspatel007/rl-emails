#!/usr/bin/env python3
"""One-shot email feature extraction using phi-4.

Extracts all email features in a single LLM call for speed.
Target: <1s per email.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import psycopg2
from openai import OpenAI


# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "phi-4"

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
    """All features extracted from an email in one shot."""
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
SYSTEM_PROMPT = """You are an email analysis expert. Extract all features from the email in ONE JSON response.

Output ONLY valid JSON with this exact structure:
{
  "is_service_email": true/false,
  "service_type": "newsletter" | "notification" | "billing" | "social" | "marketing" | "system" | "calendar" | null,
  "tasks": [
    {
      "description": "what needs to be done",
      "deadline": "original text like 'by Friday' or 'EOD'" or null,
      "deadline_date": "YYYY-MM-DD" or null,
      "assignee": "user" | "other" | null,
      "task_type": "review" | "send" | "schedule" | "decision" | "research" | "create" | "followup",
      "urgency": 0.0 to 1.0
    }
  ],
  "overall_urgency": 0.0 to 1.0,
  "requires_response": true/false,
  "topic_category": "project" | "meeting" | "request" | "fyi" | "social" | "admin",
  "summary": "one line summary of the email"
}

Rules:
- is_service_email: true for automated/system emails (newsletters, notifications, receipts)
- service_type: only if is_service_email is true
- tasks: extract ALL actionable items, empty list if none
- overall_urgency: 0.0-0.3 normal, 0.3-0.6 moderate, 0.6-1.0 urgent
- requires_response: true if sender expects a reply
- topic_category: main purpose of the email
- summary: max 100 chars

Return ONLY the JSON object, no markdown, no explanation."""


def get_emails_from_db(limit: int = 20) -> list[dict]:
    """Query sample emails from PostgreSQL."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    message_id,
                    subject,
                    body_text,
                    from_email,
                    date_parsed
                FROM emails
                WHERE body_text IS NOT NULL
                  AND body_text != ''
                  AND subject IS NOT NULL
                  AND length(body_text) > 50
                ORDER BY date_parsed DESC
                LIMIT %s
            """, (limit,))

            rows = cur.fetchall()
            emails = []
            for row in rows:
                emails.append({
                    "id": row[0],
                    "message_id": row[1],
                    "subject": row[2] or "",
                    "body_text": row[3] or "",
                    "from_email": row[4] or "",
                    "date_parsed": str(row[5]) if row[5] else "",
                })
            return emails
    finally:
        conn.close()


def extract_features_oneshot(
    subject: str,
    body: str,
    email_id: str,
    sender: str,
    client: OpenAI,
) -> ExtractedFeatures:
    """Extract all features from an email in one LLM call."""

    # Build compact email context
    body_truncated = body[:2000]  # Limit for speed
    email_context = f"Subject: {subject}\nFrom: {sender}\n\n{body_truncated}"

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": email_context}
            ],
            max_tokens=512,
            temperature=0.1,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        response_text = response.choices[0].message.content.strip()

        # Clean up response
        if response_text.startswith("```"):
            # Remove markdown code blocks
            lines = response_text.split("\n")
            response_text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        # Parse JSON
        data = json.loads(response_text)

        return ExtractedFeatures(
            email_id=email_id,
            is_service_email=data.get("is_service_email", False),
            service_type=data.get("service_type"),
            tasks=data.get("tasks", []),
            overall_urgency=float(data.get("overall_urgency", 0.0)),
            requires_response=data.get("requires_response", False),
            topic_category=data.get("topic_category", "fyi"),
            summary=data.get("summary", "")[:100],
            extraction_time_ms=elapsed_ms,
            parse_success=True,
        )

    except json.JSONDecodeError as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ExtractedFeatures(
            email_id=email_id,
            is_service_email=False,
            service_type=None,
            tasks=[],
            overall_urgency=0.0,
            requires_response=False,
            topic_category="fyi",
            summary=f"Parse error: {str(e)[:50]}",
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


def save_to_database(features_list: list[ExtractedFeatures]) -> int:
    """Save extracted features to email_llm_features table."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            # Create table if not exists
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

            # Insert features
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
                        f.email_id,
                        f.is_service_email,
                        f.service_type,
                        json.dumps(f.tasks),
                        f.overall_urgency,
                        f.requires_response,
                        f.topic_category,
                        f.summary,
                        f.extraction_time_ms,
                        f.parse_success,
                        LM_STUDIO_MODEL,
                    ))
                    inserted += 1
                except Exception as e:
                    print(f"  Warning: Failed to insert {f.email_id}: {e}", file=sys.stderr)

            conn.commit()
            return inserted
    finally:
        conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="One-shot email feature extraction using phi-4"
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

    print("=" * 60)
    print("phi-4 One-Shot Email Feature Extraction")
    print("=" * 60)
    print(f"  Target: <1s per email")
    print(f"  Emails: {args.count}")
    print()

    # Connect to LM Studio
    print("1. Connecting to LM Studio...")
    try:
        client = OpenAI(
            base_url=LM_STUDIO_URL,
            api_key="lm-studio",
        )
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if LM_STUDIO_MODEL not in model_ids:
            print(f"   ERROR: {LM_STUDIO_MODEL} not found!")
            print(f"   Available: {model_ids}")
            return 1
        print(f"   Using model: {LM_STUDIO_MODEL}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return 1
    print()

    # Query emails
    print("2. Querying emails...")
    try:
        emails = get_emails_from_db(limit=args.count)
        print(f"   Retrieved {len(emails)} emails")
    except Exception as e:
        print(f"   FAILED: {e}")
        return 1
    print()

    # Extract features
    print("3. Extracting features...")
    all_features = []
    times_ms = []
    parse_successes = 0

    for i, email in enumerate(emails, 1):
        subject = email["subject"][:40]
        print(f"   [{i:2d}/{len(emails)}] {subject}...", end=" ", flush=True)

        features = extract_features_oneshot(
            subject=email["subject"],
            body=email["body_text"],
            email_id=email["message_id"],
            sender=email["from_email"],
            client=client,
        )

        all_features.append(features)
        times_ms.append(features.extraction_time_ms)
        if features.parse_success:
            parse_successes += 1

        status = "OK" if features.parse_success else "FAIL"
        print(f"{features.extraction_time_ms}ms [{status}]")

    print()

    # Calculate metrics
    avg_time_ms = sum(times_ms) / len(times_ms) if times_ms else 0
    min_time_ms = min(times_ms) if times_ms else 0
    max_time_ms = max(times_ms) if times_ms else 0
    parse_rate = parse_successes / len(all_features) * 100 if all_features else 0

    # Count extractions
    total_tasks = sum(len(f.tasks) for f in all_features)
    service_emails = sum(1 for f in all_features if f.is_service_email)
    response_needed = sum(1 for f in all_features if f.requires_response)

    # Save to database if requested
    if args.save_db:
        print("4. Saving to database...")
        inserted = save_to_database(all_features)
        print(f"   Saved {inserted} records to email_llm_features")
        print()

    # Report
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print("Timing:")
    print(f"  Average:  {avg_time_ms:.0f}ms ({avg_time_ms/1000:.2f}s)")
    print(f"  Min:      {min_time_ms}ms")
    print(f"  Max:      {max_time_ms}ms")
    print(f"  Target:   1000ms (<1s)")
    print(f"  Status:   {'PASS' if avg_time_ms < 1000 else 'FAIL'}")
    print()
    print("Quality:")
    print(f"  Parse success: {parse_successes}/{len(all_features)} ({parse_rate:.0f}%)")
    print(f"  Tasks found:   {total_tasks}")
    print(f"  Service emails: {service_emails}/{len(all_features)}")
    print(f"  Need response:  {response_needed}/{len(all_features)}")
    print()

    # Topic breakdown
    topics = {}
    for f in all_features:
        if f.parse_success:
            topics[f.topic_category] = topics.get(f.topic_category, 0) + 1

    if topics:
        print("Topics:")
        for t, c in sorted(topics.items(), key=lambda x: -x[1]):
            print(f"  - {t}: {c}")
    print()

    # Save JSON output
    output_file = "phi4_features.json"
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": LM_STUDIO_MODEL,
        "email_count": len(emails),
        "timing": {
            "avg_ms": round(avg_time_ms, 1),
            "min_ms": min_time_ms,
            "max_ms": max_time_ms,
            "target_met": avg_time_ms < 1000,
        },
        "quality": {
            "parse_success_rate": round(parse_rate, 1),
            "total_tasks": total_tasks,
            "service_emails": service_emails,
            "requires_response": response_needed,
        },
        "results": [asdict(f) for f in all_features],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to: {output_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
