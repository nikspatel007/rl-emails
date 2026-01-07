#!/usr/bin/env python3
"""
Test Phase 1 LLM classification on a few sample emails.
Uses Claude Haiku via LiteLLM.
"""

import json
import os
import psycopg2
from dotenv import load_dotenv
from litellm import completion

# Load .env file
load_dotenv()

DB_URL = "postgresql://postgres:postgres@localhost:5433/gmail_twoyrs"

# Model selection - GPT-5-mini is fast/cheap, fallback to Haiku
if os.environ.get("OPENAI_API_KEY"):
    MODEL = "gpt-5-mini"  # Fast/cheap OpenAI model (Jan 2026)
elif os.environ.get("ANTHROPIC_API_KEY"):
    MODEL = "claude-3-5-haiku-latest"
else:
    MODEL = None

PHASE1_PROMPT = """Classify this email for an AI assistant. Be concise. Respond with JSON only.

## Email
From: {from_email}
Subject: {subject}
Date: {date}

---
{body_preview}
---

## Known Context
- Sender relationship: {relationship_level} ({reply_rate}% reply rate to this sender)
- Is service/automated email: {is_service}
- LLM Priority: {llm_priority}

## Classify (respond with valid JSON only, no markdown):
{{
  "action_type": "reply|task|decision|approval|fyi|none",
  "urgency": "immediate|today|this_week|whenever|none",
  "ai_can_handle": "fully|partially|not_at_all",
  "next_step": "skip|quick_action|needs_deep_analysis",
  "suggested_action": "FILE_TO_FOLDER|DRAFT_REPLY|SCHEDULE_MEETING|SUMMARIZE_ATTACHMENT|PREPARE_CONTEXT|null",
  "one_liner": "10 words max summary of what this email is about"
}}"""


def get_sample_emails(conn, email_ids=None, limit=3, exclude_processed=True):
    """Get sample emails for testing."""
    cur = conn.cursor()

    if email_ids:
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
                e.thread_id
            FROM emails e
            JOIN email_features ef ON ef.email_id = e.id
            JOIN email_ai_classification ac ON ac.email_id = e.id
            WHERE e.id = ANY(%s)
            ORDER BY ac.llm_priority
        """, (email_ids,))
    else:
        # Get diverse samples, excluding already processed
        exclude_clause = ""
        if exclude_processed:
            exclude_clause = "AND e.id NOT IN (SELECT email_id FROM email_llm_phase1)"

        cur.execute(f"""
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
                e.thread_id
            FROM emails e
            JOIN email_features ef ON ef.email_id = e.id
            JOIN email_ai_classification ac ON ac.email_id = e.id
            WHERE ac.predicted_handleability = 'needs_llm'
            AND e.is_sent = FALSE
            AND LENGTH(e.body_text) > 50
            {exclude_clause}
            ORDER BY ac.llm_priority, RANDOM()
            LIMIT %s
        """, (limit,))

    return cur.fetchall()


def classify_email(email_data):
    """Run Phase 1 classification on an email."""
    email_id, from_email, subject, date_parsed, body_preview, \
        relationship, reply_rate, is_service, llm_priority, actual_action, thread_id = email_data

    # Clean body preview (remove HTML if present)
    body_clean = body_preview or ""
    if "<html" in body_clean.lower() or "<head" in body_clean.lower():
        # Simple HTML stripping for preview
        import re
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

    prompt = PHASE1_PROMPT.format(
        from_email=from_email,
        subject=subject,
        date=str(date_parsed)[:10] if date_parsed else "unknown",
        body_preview=body_clean[:500] if body_clean else "(empty body)",
        relationship_level=rel_level,
        reply_rate=round((reply_rate or 0) * 100, 1),
        is_service="Yes" if is_service else "No",
        llm_priority=llm_priority
    )

    print(f"\n{'='*70}")
    print(f"EMAIL ID: {email_id}")
    print(f"FROM: {from_email}")
    print(f"SUBJECT: {subject}")
    print(f"RELATIONSHIP: {rel_level} ({round(relationship*100, 1)}%)")
    print(f"IS SERVICE: {is_service}")
    print(f"ACTUAL ACTION: {actual_action}")
    print(f"{'='*70}")
    print(f"\nBODY PREVIEW:\n{body_clean[:300]}...")
    print(f"\n{'-'*70}")
    print("SENDING TO LLM...")

    try:
        # GPT-5 models don't support temperature=0, and need reasoning_effort
        completion_kwargs = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
        }
        # GPT-5 models: add reasoning_effort='minimal' for direct output
        if MODEL.startswith("gpt-5"):
            completion_kwargs["reasoning_effort"] = "minimal"
        else:
            # Non-GPT-5 models support temperature=0
            completion_kwargs["temperature"] = 0

        response = completion(**completion_kwargs)

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Try to extract JSON if wrapped in markdown
            if "```" in result_text:
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result_text)
                if json_match:
                    result_text = json_match.group(1)

            result = json.loads(result_text)
        except json.JSONDecodeError:
            result = {"raw_response": result_text, "parse_error": True}

        # Add token usage and raw data for storage
        result["_tokens"] = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
        result["_raw_response"] = result_text
        result["_raw_prompt"] = prompt
        result["_thread_id"] = thread_id

        print(f"\nLLM RESPONSE:")
        print(json.dumps({k: v for k, v in result.items() if not k.startswith("_raw")}, indent=2))

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e), "_raw_prompt": prompt, "_thread_id": thread_id}


def main():
    # Check for API key
    if not MODEL:
        print("ERROR: No API key found")
        print("Set either ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        return

    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print(f"Using model: {MODEL}")
    print(f"Running Phase 1 test on {limit} sample emails...")

    conn = psycopg2.connect(DB_URL)

    try:
        # Get samples - one from each priority level
        emails = get_sample_emails(conn, limit=limit)

        if not emails:
            print("No emails found in needs_llm category")
            return

        results = []
        total_tokens = 0
        cur = conn.cursor()

        for email_data in emails:
            result = classify_email(email_data)
            results.append({
                "email_id": email_data[0],
                "subject": email_data[2],
                "actual_action": email_data[9],
                "llm_result": result
            })
            if "_tokens" in result:
                total_tokens += result["_tokens"]["total"]

            # Save to database
            if "error" not in result:
                try:
                    cur.execute("""
                        INSERT INTO email_llm_phase1 (
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
                        email_data[0],  # email_id
                        result.get("_raw_response"),
                        result.get("_raw_prompt"),
                        result.get("action_type"),
                        result.get("urgency"),
                        result.get("ai_can_handle"),
                        result.get("next_step"),
                        result.get("suggested_action"),
                        result.get("one_liner"),
                        MODEL,
                        result.get("_tokens", {}).get("prompt"),
                        result.get("_tokens", {}).get("completion"),
                        result.get("_tokens", {}).get("total"),
                        result.get("_thread_id")
                    ))
                    conn.commit()
                except Exception as e:
                    print(f"  DB ERROR: {e}")
                    conn.rollback()

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Emails processed: {len(results)}")
        print(f"Total tokens used: {total_tokens}")
        print(f"Estimated cost: ${total_tokens * 0.25 / 1_000_000:.6f} (Haiku input pricing)")

        print(f"\n{'='*70}")
        print("RESULTS COMPARISON")
        print(f"{'='*70}")
        for r in results:
            llm = r["llm_result"]
            print(f"\nEmail {r['email_id']}: {r['subject'][:50]}")
            print(f"  Actual: {r['actual_action']}")
            if "action_type" in llm:
                print(f"  LLM says: {llm['action_type']} / {llm['urgency']} / AI can handle: {llm['ai_can_handle']}")
                print(f"  Next step: {llm['next_step']} → {llm.get('suggested_action', 'N/A')}")
                print(f"  One-liner: {llm.get('one_liner', 'N/A')}")
            else:
                print(f"  LLM error: {llm}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
