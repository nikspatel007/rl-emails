#!/usr/bin/env python3
"""LLM-based multi-task extraction from email content.

Extracts multiple discrete tasks from a single email using an LLM.
Each task includes: task_id, description, deadline, assignee_hint, task_type.

This module is LLM-primary, with rule-based fallback when LLM is unavailable.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore
    HAS_ANTHROPIC = False


class TaskType(str, Enum):
    """Types of tasks that can be extracted from emails."""
    ACTION = "action"           # Direct action required (do X)
    DELIVERABLE = "deliverable" # Produce a specific output
    REVIEW = "review"           # Review/approve something
    MEETING = "meeting"         # Schedule or attend a meeting
    FOLLOWUP = "followup"       # Follow up on something
    DECISION = "decision"       # Make a decision
    INFORMATION = "information" # Provide information/response
    OTHER = "other"


@dataclass
class ExtractedTask:
    """A discrete task extracted from an email.

    Attributes:
        task_id: Unique identifier for this task (hash of email_id + index)
        description: Clear description of what needs to be done
        deadline: Optional deadline text (e.g., "by Friday", "EOD", "2024-01-15")
        assignee_hint: Optional hint about who should do this (e.g., "you", "team", "John")
        task_type: Type classification of the task
        confidence: Extraction confidence score (0-1)
        source_text: Original text snippet that triggered this extraction
        email_id: ID of the source email
    """
    task_id: str
    description: str
    deadline: Optional[str] = None
    assignee_hint: Optional[str] = None
    task_type: TaskType = TaskType.ACTION
    confidence: float = 1.0
    source_text: str = ""
    email_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['task_type'] = self.task_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedTask":
        """Create from dictionary."""
        data = data.copy()
        if 'task_type' in data and isinstance(data['task_type'], str):
            data['task_type'] = TaskType(data['task_type'])
        return cls(**data)


def generate_task_id(email_id: str, index: int, description: str) -> str:
    """Generate a unique task ID."""
    content = f"{email_id}:{index}:{description[:50]}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


# System prompt for LLM task extraction
TASK_EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing emails and extracting actionable tasks.

Given an email, identify ALL discrete tasks that need to be completed. For each task, extract:
1. description: A clear, actionable description of what needs to be done
2. deadline: Any mentioned deadline (exact text like "by Friday", "EOD tomorrow", dates, etc.) or null if none
3. assignee_hint: Who should do this task ("you" if directed at recipient, specific name if mentioned, "team" if group, null if unclear)
4. task_type: One of: action, deliverable, review, meeting, followup, decision, information, other
5. source_text: The exact phrase/sentence from the email that indicates this task

Rules:
- Extract ALL distinct tasks, even if subtle
- Be specific in descriptions - include context from the email
- If a bullet list contains tasks, extract each as separate
- "Please review X" and "Please approve X" are different tasks
- FYI items are NOT tasks unless they require a response
- Greetings and sign-offs are not tasks
- Return an empty list if no tasks are found

Respond with ONLY valid JSON in this format:
{"tasks": [{"description": "...", "deadline": "..." or null, "assignee_hint": "..." or null, "task_type": "...", "source_text": "..."}]}"""


def extract_tasks_llm(
    subject: str,
    body: str,
    email_id: str = "",
    sender: str = "",
    model: str = "claude-sonnet-4-20250514",
) -> list[ExtractedTask]:
    """Extract tasks from email using LLM.

    Args:
        subject: Email subject line
        body: Email body text
        email_id: Optional email ID for task ID generation
        sender: Optional sender for context
        model: Anthropic model to use

    Returns:
        List of ExtractedTask objects

    Raises:
        RuntimeError: If anthropic package not installed
        anthropic.APIError: If API call fails
    """
    if not HAS_ANTHROPIC:
        raise RuntimeError(
            "anthropic package required for LLM extraction. "
            "Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable not set"
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Build the user message with email content
    email_context = f"Subject: {subject}\n"
    if sender:
        email_context += f"From: {sender}\n"
    email_context += f"\n{body}"

    user_message = f"Extract all tasks from this email:\n\n{email_context}"

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=TASK_EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    # Parse response
    response_text = response.content[0].text

    # Handle potential markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    try:
        data = json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        # Try to extract JSON from response
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    tasks = []
    raw_tasks = data.get("tasks", [])

    for i, raw_task in enumerate(raw_tasks):
        task_type_str = raw_task.get("task_type", "action")
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.OTHER

        task = ExtractedTask(
            task_id=generate_task_id(email_id, i, raw_task.get("description", "")),
            description=raw_task.get("description", ""),
            deadline=raw_task.get("deadline"),
            assignee_hint=raw_task.get("assignee_hint"),
            task_type=task_type,
            confidence=0.9,  # LLM extraction has high confidence
            source_text=raw_task.get("source_text", ""),
            email_id=email_id,
        )
        tasks.append(task)

    return tasks


def extract_tasks_rule_based(
    subject: str,
    body: str,
    email_id: str = "",
) -> list[ExtractedTask]:
    """Extract tasks using rule-based patterns (fallback).

    This is a simpler extraction method that doesn't require an LLM.
    Used when ANTHROPIC_API_KEY is not available.

    Args:
        subject: Email subject line
        body: Email body text
        email_id: Optional email ID for task ID generation

    Returns:
        List of ExtractedTask objects
    """
    tasks = []
    combined_text = f"{subject}\n\n{body}"

    # Request patterns with capture groups
    request_patterns = [
        (r'(?:please|pls)\s+([^.?!\n]{10,100})', TaskType.ACTION),
        (r'(?:can you|could you|would you)\s+([^.?!\n]{10,100})', TaskType.ACTION),
        (r'(?:need you to|need to)\s+([^.?!\n]{10,100})', TaskType.ACTION),
        (r'(?:kindly)\s+([^.?!\n]{10,100})', TaskType.ACTION),
        (r'(?:review|approve)\s+(?:the\s+)?([^.?!\n]{5,80})', TaskType.REVIEW),
        (r'(?:schedule|set up)\s+(?:a\s+)?(?:meeting|call)\s*(?:with|for)?\s*([^.?!\n]{5,80})', TaskType.MEETING),
        (r'(?:send|provide|share)\s+(?:me\s+)?(?:the\s+)?([^.?!\n]{5,80})', TaskType.DELIVERABLE),
        (r'(?:follow up)\s+(?:on|with)\s+([^.?!\n]{5,80})', TaskType.FOLLOWUP),
        (r'(?:decide|make a decision)\s+(?:on|about)\s+([^.?!\n]{5,80})', TaskType.DECISION),
    ]

    # Deadline patterns
    deadline_patterns = [
        r'\bby\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\bby\s+(tomorrow|today|eod|cob|eow)\b',
        r'\bby\s+(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)\b',
        r'\b(asap|urgent|immediately)\b',
    ]

    seen_descriptions = set()

    for pattern, task_type in request_patterns:
        for match in re.finditer(pattern, combined_text, re.IGNORECASE):
            description = match.group(1).strip()

            # Skip if too short or already seen
            if len(description) < 10 or description.lower() in seen_descriptions:
                continue

            seen_descriptions.add(description.lower())

            # Look for deadline near this match
            deadline = None
            context_start = max(0, match.start() - 50)
            context_end = min(len(combined_text), match.end() + 100)
            context = combined_text[context_start:context_end]

            for dl_pattern in deadline_patterns:
                dl_match = re.search(dl_pattern, context, re.IGNORECASE)
                if dl_match:
                    deadline = dl_match.group(1)
                    break

            task = ExtractedTask(
                task_id=generate_task_id(email_id, len(tasks), description),
                description=description,
                deadline=deadline,
                assignee_hint="you",  # Rule-based assumes directed at recipient
                task_type=task_type,
                confidence=0.6,  # Lower confidence for rule-based
                source_text=match.group(0)[:100],
                email_id=email_id,
            )
            tasks.append(task)

    # Also extract bullet points that look like action items
    bullet_pattern = r'(?:^|\n)\s*[-•*]\s*((?:review|send|prepare|complete|update|submit|create|schedule|follow|check|confirm)[^.\n]{5,80})'

    for match in re.finditer(bullet_pattern, combined_text, re.IGNORECASE | re.MULTILINE):
        description = match.group(1).strip()

        if description.lower() in seen_descriptions:
            continue

        seen_descriptions.add(description.lower())

        # Determine task type from verb
        task_type = TaskType.ACTION
        desc_lower = description.lower()
        if desc_lower.startswith("review"):
            task_type = TaskType.REVIEW
        elif desc_lower.startswith(("send", "prepare", "create", "submit")):
            task_type = TaskType.DELIVERABLE
        elif desc_lower.startswith("schedule"):
            task_type = TaskType.MEETING
        elif desc_lower.startswith("follow"):
            task_type = TaskType.FOLLOWUP

        task = ExtractedTask(
            task_id=generate_task_id(email_id, len(tasks), description),
            description=description,
            deadline=None,
            assignee_hint="you",
            task_type=task_type,
            confidence=0.5,
            source_text=match.group(0)[:100],
            email_id=email_id,
        )
        tasks.append(task)

    return tasks[:10]  # Limit to 10 tasks


def extract_tasks(
    subject: str,
    body: str,
    email_id: str = "",
    sender: str = "",
    use_llm: bool = True,
    model: str = "claude-sonnet-4-20250514",
) -> list[ExtractedTask]:
    """Extract multiple tasks from an email.

    Primary entry point for task extraction. Uses LLM by default,
    falls back to rule-based extraction if LLM is unavailable.

    Args:
        subject: Email subject line
        body: Email body text
        email_id: Optional email ID for task ID generation
        sender: Optional sender for context (LLM only)
        use_llm: Whether to use LLM extraction (default True)
        model: Anthropic model to use for LLM extraction

    Returns:
        List of ExtractedTask objects

    Example:
        >>> tasks = extract_tasks(
        ...     subject="Project update needed",
        ...     body="Please review the proposal by Friday and send me your feedback.",
        ...     email_id="msg-123"
        ... )
        >>> for task in tasks:
        ...     print(f"{task.task_type.value}: {task.description}")
        review: review the proposal
        deliverable: send me your feedback
    """
    if use_llm:
        try:
            return extract_tasks_llm(
                subject=subject,
                body=body,
                email_id=email_id,
                sender=sender,
                model=model,
            )
        except (RuntimeError, Exception) as e:
            # Fall back to rule-based if LLM fails
            import sys
            print(f"LLM extraction failed, using rule-based fallback: {e}", file=sys.stderr)

    return extract_tasks_rule_based(
        subject=subject,
        body=body,
        email_id=email_id,
    )


def extract_tasks_batch(
    emails: list[dict],
    use_llm: bool = True,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, list[ExtractedTask]]:
    """Extract tasks from multiple emails.

    Args:
        emails: List of email dictionaries with 'subject', 'body', 'message_id', 'from' keys
        use_llm: Whether to use LLM extraction
        model: Anthropic model to use

    Returns:
        Dictionary mapping email_id to list of ExtractedTask
    """
    results = {}

    for email in emails:
        email_id = email.get("message_id", "")
        subject = email.get("subject", "")
        body = email.get("body", "")
        sender = email.get("from", "")

        tasks = extract_tasks(
            subject=subject,
            body=body,
            email_id=email_id,
            sender=sender,
            use_llm=use_llm,
            model=model,
        )

        results[email_id] = tasks

    return results


if __name__ == "__main__":
    # Example usage
    sample_subject = "Action Required: Q2 Review and Budget Approval"
    sample_body = """
    Hi Team,

    A few things need your attention this week:

    1. Please review the Q2 performance report by Friday EOD
    2. I need you to approve the revised budget estimates before our Monday meeting
    3. Can you schedule a follow-up call with the client for next week?
    4. Send me the updated project timeline when you have a chance

    Also, FYI - the new office hours start next month.

    Let me know if you have any questions. This is blocking the board presentation.

    Thanks,
    Sarah
    """

    print("Extracting tasks (rule-based fallback)...")
    tasks = extract_tasks(
        subject=sample_subject,
        body=sample_body,
        email_id="sample-001",
        use_llm=False,  # Use rule-based for demo
    )

    print(f"\nFound {len(tasks)} tasks:\n")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. [{task.task_type.value.upper()}] {task.description}")
        if task.deadline:
            print(f"   Deadline: {task.deadline}")
        if task.assignee_hint:
            print(f"   Assignee: {task.assignee_hint}")
        print(f"   Confidence: {task.confidence:.0%}")
        print()
