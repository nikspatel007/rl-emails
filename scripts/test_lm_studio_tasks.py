#!/usr/bin/env python3
"""Test script for LM Studio task extraction from emails.

Connects to LM Studio (OpenAI-compatible API) to extract tasks from
emails and saves results to test_tasks.json.

Usage:
    python test_lm_studio_tasks.py --count 100 --batch-size 10
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

import psycopg2
from openai import OpenAI


# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "openai/gpt-oss-120b"

# PostgreSQL configuration
PG_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "rl_emails",
    "user": "postgres",
    "password": "postgres",
}


class TaskType(str, Enum):
    """Types of tasks that can be extracted from emails."""
    ACTION = "action"
    DELIVERABLE = "deliverable"
    REVIEW = "review"
    MEETING = "meeting"
    FOLLOWUP = "followup"
    DECISION = "decision"
    INFORMATION = "information"
    OTHER = "other"


@dataclass
class ExtractedTask:
    """A discrete task extracted from an email."""
    task_id: str
    description: str
    deadline: Optional[str] = None
    assignee_hint: Optional[str] = None
    task_type: str = "action"
    urgency_score: float = 0.0
    source_text: str = ""
    email_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# System prompt for task extraction
SYSTEM_PROMPT = """You are an expert at analyzing emails and extracting actionable tasks.

Given an email, identify ALL discrete tasks that need to be completed. For each task, extract:
1. description: A clear, actionable description of what needs to be done
2. deadline: Any mentioned deadline (exact text like "by Friday", "EOD tomorrow", dates, etc.) or null if none
3. assignee_hint: Who should do this task ("you" if directed at recipient, specific name if mentioned, null if unclear)
4. task_type: One of: action, deliverable, review, meeting, followup, decision, information, other
5. urgency_score: A float from 0.0 (not urgent) to 1.0 (extremely urgent) based on language and context
6. source_text: The exact phrase/sentence from the email that indicates this task (max 100 chars)

Rules:
- Extract ALL distinct tasks, even if subtle
- Be specific in descriptions - include context from the email
- If a bullet list contains tasks, extract each as separate
- "Please review X" and "Please approve X" are different tasks
- FYI items are NOT tasks unless they require a response
- Greetings and sign-offs are not tasks
- Return an empty list if no tasks are found
- IMPORTANT: urgency_score should be 0.0-0.3 for normal, 0.3-0.6 for moderate, 0.6-1.0 for urgent

Respond with ONLY valid JSON in this format:
{"tasks": [{"description": "...", "deadline": "..." or null, "assignee_hint": "..." or null, "task_type": "...", "urgency_score": 0.5, "source_text": "..."}]}"""


def get_emails_from_db(limit: int = 10) -> list[dict]:
    """Query sample emails from PostgreSQL.

    Selects emails with non-empty body_text that are likely to contain tasks
    (received emails, not sent).
    """
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
                  AND is_sent = false
                  AND subject IS NOT NULL
                  AND length(body_text) > 100
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


def extract_tasks_lm_studio(
    subject: str,
    body: str,
    email_id: str,
    sender: str,
    client: OpenAI,
) -> list[ExtractedTask]:
    """Extract tasks from a single email using LM Studio."""

    # Build the user message
    email_context = f"Subject: {subject}\nFrom: {sender}\n\n{body[:3000]}"  # Limit body length
    user_message = f"Extract all tasks from this email:\n\n{email_context}"

    try:
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=2048,
            temperature=0.1,
        )

        response_text = response.choices[0].message.content

        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            parts = response_text.split("```")
            if len(parts) >= 2:
                response_text = parts[1]

        # Parse JSON
        data = json.loads(response_text.strip())
        raw_tasks = data.get("tasks", [])

        tasks = []
        for i, raw_task in enumerate(raw_tasks):
            task = ExtractedTask(
                task_id=f"{email_id[:8]}_{i:02d}",
                description=raw_task.get("description", ""),
                deadline=raw_task.get("deadline"),
                assignee_hint=raw_task.get("assignee_hint"),
                task_type=raw_task.get("task_type", "action"),
                urgency_score=float(raw_task.get("urgency_score", 0.0)),
                source_text=raw_task.get("source_text", "")[:100],
                email_id=email_id,
            )
            tasks.append(task)

        return tasks

    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON response: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  Warning: LLM call failed: {e}", file=sys.stderr)
        return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract tasks from emails using LM Studio"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=10,
        help="Number of emails to process (default: 10)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Batch size for progress reporting (default: 10)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file (default: test_tasks.json)"
    )
    args = parser.parse_args()

    email_count = args.count
    batch_size = args.batch_size
    output_file = args.output or os.path.join(
        os.path.dirname(__file__), "..", "test_tasks.json"
    )

    print("=" * 60)
    print("LM Studio Task Extraction Test")
    print("=" * 60)
    print(f"  Emails: {email_count}, Batch size: {batch_size}")
    print()

    # Step 1: Connect to LM Studio
    print("1. Connecting to LM Studio...")
    try:
        client = OpenAI(
            base_url=LM_STUDIO_URL,
            api_key="lm-studio",  # LM Studio doesn't need a real key
        )
        # Test connection
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"   Connected! Available models: {len(model_ids)}")
        if LM_STUDIO_MODEL in model_ids:
            print(f"   Using model: {LM_STUDIO_MODEL}")
        else:
            print(f"   Warning: {LM_STUDIO_MODEL} not found, available: {model_ids[:3]}")
    except Exception as e:
        print(f"   FAILED: Could not connect to LM Studio: {e}")
        return 1
    print()

    # Step 2: Query emails from PostgreSQL
    print("2. Querying emails from PostgreSQL...")
    try:
        emails = get_emails_from_db(limit=email_count)
        print(f"   Retrieved {len(emails)} emails")
    except Exception as e:
        print(f"   FAILED: Could not query database: {e}")
        return 1
    print()

    # Step 3: Extract tasks from each email (with timing)
    print("3. Extracting tasks from emails...")
    all_results = []
    total_tasks = 0
    batch_times = []
    errors = 0

    start_time = time.time()
    batch_start = time.time()

    for i, email in enumerate(emails, 1):
        email_id = email["message_id"]
        subject = email["subject"][:50]

        tasks = extract_tasks_lm_studio(
            subject=email["subject"],
            body=email["body_text"],
            email_id=email["message_id"],
            sender=email["from_email"],
            client=client,
        )

        result = {
            "email_id": email["message_id"],
            "subject": email["subject"],
            "from": email["from_email"],
            "date": email["date_parsed"],
            "task_count": len(tasks),
            "tasks": [t.to_dict() for t in tasks],
        }
        all_results.append(result)
        total_tasks += len(tasks)

        if len(tasks) == 0 and "Warning" in str(sys.stderr):
            errors += 1

        # Report progress every batch_size emails
        if i % batch_size == 0 or i == len(emails):
            batch_elapsed = time.time() - batch_start
            batch_times.append(batch_elapsed)
            emails_per_sec = batch_size / batch_elapsed if batch_elapsed > 0 else 0
            print(f"   [{i:3d}/{len(emails)}] Batch: {batch_elapsed:.1f}s ({emails_per_sec:.2f} emails/s)")
            batch_start = time.time()

    total_time = time.time() - start_time
    print()
    print(f"   Total: {total_tasks} tasks from {len(emails)} emails in {total_time:.1f}s")
    print()

    # Step 4: Save results to JSON
    print("4. Saving results...")
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": LM_STUDIO_MODEL,
        "email_count": len(emails),
        "total_tasks": total_tasks,
        "timing": {
            "total_seconds": round(total_time, 2),
            "emails_per_second": round(len(emails) / total_time, 3) if total_time > 0 else 0,
            "avg_batch_time": round(sum(batch_times) / len(batch_times), 2) if batch_times else 0,
        },
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"   Saved to: {output_file}")
    print()

    # Step 5: Report summary
    print("5. Summary:")
    print("-" * 40)
    print(f"   Emails processed:      {len(emails)}")
    print(f"   Total tasks extracted: {total_tasks}")
    print(f"   Average tasks/email:   {total_tasks / len(emails):.2f}")
    print()
    print("   Timing:")
    print(f"     Total time:          {total_time:.1f}s")
    print(f"     Throughput:          {len(emails) / total_time:.2f} emails/s")
    print(f"     Avg time per email:  {total_time / len(emails):.2f}s")
    if batch_times:
        print(f"     Avg batch time:      {sum(batch_times) / len(batch_times):.2f}s")

    # Task type breakdown
    type_counts = {}
    urgency_sum = 0.0
    for result in all_results:
        for task in result["tasks"]:
            task_type = task.get("task_type", "other")
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
            urgency_sum += task.get("urgency_score", 0.0)

    if type_counts:
        print()
        print("   Task types:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"     - {t}: {c}")

        if total_tasks > 0:
            print(f"   Average urgency: {urgency_sum / total_tasks:.2f}")

    print()
    print("=" * 60)
    print("SUCCESS: Test completed")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
