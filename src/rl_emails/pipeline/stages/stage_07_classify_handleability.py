"""Stage 7: Rule-based AI Handleability Classification.

Classifies emails into categories based on patterns:
- ai_full: AI can fully handle (file, skip)
- ai_partial: AI can help (draft, schedule)
- human_required: Needs human (decision, personal)
- needs_llm: Need LLM to understand

This runs BEFORE any LLM calls to reduce cost.
"""

from __future__ import annotations

import json
import time
from typing import Any

from psycopg2.extras import execute_values

from rl_emails.core.config import Config
from rl_emails.core.db import get_connection
from rl_emails.pipeline.stages.base import StageResult


def detect_patterns(body: str, subject: str) -> dict[str, bool]:
    """Detect content patterns in email body and subject.

    Args:
        body: Email body text (lowercase).
        subject: Email subject (lowercase).

    Returns:
        Dictionary of pattern names to boolean values.
    """
    return {
        "has_question": "?" in body or "?" in subject,
        "has_request": any(
            w in body for w in ["please", "could you", "can you", "would you", "need you"]
        ),
        "has_scheduling": any(
            w in body for w in ["meeting", "calendar", "schedule", "call", "zoom", "teams"]
        ),
        "has_deadline": any(
            w in body
            for w in [
                "deadline",
                "by friday",
                "by monday",
                "by eod",
                "end of day",
                "urgent",
                "asap",
                "by tomorrow",
            ]
        ),
        "has_approval": any(
            w in body for w in ["approve", "approval", "sign off", "authorize", "permission"]
        ),
        "has_confirm": any(
            w in body for w in ["confirm", "acknowledge", "let me know", "please reply"]
        ),
        "is_newsletter": any(
            w in body for w in ["unsubscribe", "view in browser", "email preferences", "opt out"]
        ),
        "is_fyi": any(
            w in body for w in ["fyi", "for your information", "just wanted to let you know"]
        ),
        "is_calendar_response": any(w in subject for w in ["accepted:", "declined:", "tentative:"]),
        "is_auto_reply": any(
            w in subject for w in ["out of office", "automatic reply", "auto-reply"]
        ),
        "has_attachment_ref": any(
            w in body for w in ["attached", "attachment", "see attached", "please find"]
        ),
    }


def classify_email(
    email: dict[str, Any], features: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    """Classify email based on rules and patterns.

    Args:
        email: Email data with 'body_text' and 'subject' keys.
        features: Email features with 'relationship_strength', 'is_service_email',
                 'service_type' keys.

    Returns:
        Tuple of (handleability, reason, metadata).
    """
    body = (email.get("body_text") or "").lower()
    subject = (email.get("subject") or "").lower()

    relationship = features.get("relationship_strength", 0) or 0
    is_service = features.get("is_service_email", False) or False
    service_type = features.get("service_type", "") or ""

    patterns = detect_patterns(body, subject)

    # AI_FULL: Can be completely handled by AI without human
    if is_service and service_type in ["marketing", "newsletter", "notification"]:
        folder = "Promotions" if service_type == "marketing" else "Updates"
        return (
            "ai_full",
            f"service_{service_type}",
            {"action": "FILE_TO_FOLDER", "folder": folder, "patterns": patterns},
        )

    if patterns["is_calendar_response"]:
        return (
            "ai_full",
            "calendar_response",
            {"action": "FILE_TO_FOLDER", "folder": "Calendar", "patterns": patterns},
        )

    if patterns["is_auto_reply"]:
        return (
            "ai_full",
            "auto_reply",
            {"action": "FILE_TO_FOLDER", "folder": "Auto-replies", "patterns": patterns},
        )

    if patterns["is_newsletter"] and relationship < 0.2:
        return (
            "ai_full",
            "newsletter_pattern",
            {"action": "FILE_TO_FOLDER", "folder": "Newsletters", "patterns": patterns},
        )

    if is_service and service_type == "transactional":
        if any(
            w in body
            for w in ["order", "receipt", "confirmation", "shipped", "delivered", "tracking"]
        ):
            return (
                "ai_full",
                "transactional",
                {
                    "action": "FILE_TO_FOLDER",
                    "folder": "Orders",
                    "track_delivery": "shipped" in body or "tracking" in body,
                    "patterns": patterns,
                },
            )

    # HUMAN_REQUIRED: Only human can handle
    if patterns["has_approval"]:
        return (
            "human_required",
            "approval_request",
            {"action": "PREPARE_DECISION_CONTEXT", "priority": "high", "patterns": patterns},
        )

    if relationship > 0.6 and not is_service:
        return (
            "human_required",
            "important_relationship",
            {
                "action": "PREPARE_DECISION_CONTEXT",
                "relationship": relationship,
                "patterns": patterns,
            },
        )

    if not is_service and patterns["has_confirm"] and relationship > 0.2:
        return (
            "human_required",
            "explicit_reply_request",
            {"action": "DRAFT_REPLY", "needs_personalization": True, "patterns": patterns},
        )

    if patterns["has_deadline"] and patterns["has_request"] and not is_service:
        return (
            "human_required",
            "deadline_request",
            {"action": "PREPARE_DECISION_CONTEXT", "priority": "high", "patterns": patterns},
        )

    # AI_PARTIAL: AI can prepare, human finishes
    if patterns["has_scheduling"] and not is_service:
        if relationship < 0.5:
            return (
                "ai_partial",
                "scheduling_request",
                {"action": "SCHEDULE_MEETING", "needs_approval": True, "patterns": patterns},
            )

    if patterns["is_fyi"] and relationship > 0.1 and not is_service:
        return (
            "ai_partial",
            "fyi_from_contact",
            {"action": "DRAFT_REPLY", "draft_type": "acknowledgment", "patterns": patterns},
        )

    if patterns["has_confirm"] and relationship < 0.3:
        return (
            "ai_partial",
            "confirm_request",
            {"action": "DRAFT_REPLY", "draft_type": "confirmation", "patterns": patterns},
        )

    if patterns["has_attachment_ref"]:
        return (
            "ai_partial",
            "has_attachment",
            {"action": "SUMMARIZE_ATTACHMENT", "patterns": patterns},
        )

    # NEEDS_LLM: Can't determine from rules alone
    llm_priority = 3  # default medium

    if relationship > 0.4:
        llm_priority = 1  # high priority - important person
    elif patterns["has_request"]:
        llm_priority = 2  # medium-high - has request
    elif is_service:
        llm_priority = 4  # low priority - service email
    elif relationship < 0.1:
        llm_priority = 5  # lowest - unknown sender

    return "needs_llm", "ambiguous", {"llm_priority": llm_priority, "patterns": patterns}


def run(config: Config) -> StageResult:
    """Run Stage 7: Rule-based AI handleability classification.

    Args:
        config: Application configuration with database_url.

    Returns:
        StageResult with classification counts.
    """
    start_time = time.time()

    with get_connection(config.database_url) as conn:
        with conn.cursor() as cur:
            # Clear existing data for idempotent re-runs
            cur.execute("DELETE FROM email_ai_classification")
            conn.commit()

            # Get all emails with features
            cur.execute(
                """
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
            """
            )

            rows = cur.fetchall()

            results = []
            category_counts: dict[str, int] = {}

            for row in rows:
                email = {
                    "id": row[0],
                    "subject": row[1],
                    "body_text": row[2],
                    "from_email": row[3],
                }
                features = {
                    "relationship_strength": row[4] or 0,
                    "is_service_email": row[5] or False,
                    "service_type": row[6] or "",
                    "urgency_score": row[7] or 0,
                }

                handleability, reason, metadata = classify_email(email, features)
                patterns = metadata.get("patterns", {})

                # Track counts
                category_counts[handleability] = category_counts.get(handleability, 0) + 1

                results.append(
                    (
                        email["id"],
                        handleability,
                        reason,
                        json.dumps(metadata),
                        patterns.get("has_question", False),
                        patterns.get("has_request", False),
                        patterns.get("has_scheduling", False),
                        patterns.get("has_deadline", False),
                        patterns.get("has_approval", False),
                        patterns.get("has_confirm", False),
                        patterns.get("is_newsletter", False),
                        patterns.get("is_fyi", False),
                        patterns.get("is_calendar_response", False),
                        patterns.get("has_attachment_ref", False),
                        handleability == "needs_llm",
                        metadata.get("llm_priority"),
                    )
                )

            # Bulk insert
            if results:
                execute_values(
                    cur,
                    """
                    INSERT INTO email_ai_classification (
                        email_id, predicted_handleability, classification_reason,
                        classification_metadata, has_question, has_request,
                        has_scheduling, has_deadline, has_approval, has_confirm,
                        is_newsletter, is_fyi, is_calendar_response, has_attachment_ref,
                        needs_llm_classification, llm_priority
                    ) VALUES %s
                    """,
                    results,
                )
                conn.commit()

    duration = time.time() - start_time

    return StageResult(
        success=True,
        records_processed=len(results),
        duration_seconds=duration,
        message=f"Classified {len(results)} emails",
        metadata={"category_counts": category_counts},
    )
