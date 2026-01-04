"""Task extraction from email content.

This module extracts actionable tasks and deadlines from emails, providing
features that help the RL agent decide whether to create tracked tasks.

Features extracted:
- Deadline detection with date parsing
- Action item extraction
- Assignment detection
- Effort estimation
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class TaskFeatures:
    """Features indicating whether an email contains actionable tasks.

    Attributes:
        has_deadline: Whether email mentions a deadline
        deadline_date: Parsed deadline datetime if detected
        deadline_text: Raw text that triggered deadline detection
        has_deliverable: Whether email mentions a concrete deliverable
        deliverable_description: Extracted deliverable text
        is_assigned_to_user: Whether task seems assigned to the recipient
        assigned_by: Sender if assignment detected
        assignment_confidence: Confidence score for assignment (0-1)
        estimated_effort: Estimated effort level (quick/medium/substantial)
        requires_others: Whether task requires coordination with others
        is_blocker_for_others: Whether others are waiting on this
        action_items: Extracted action item text snippets
    """
    has_deadline: bool = False
    deadline_date: Optional[datetime] = None
    deadline_text: str = ""
    has_deliverable: bool = False
    deliverable_description: str = ""
    is_assigned_to_user: bool = False
    assigned_by: str = ""
    assignment_confidence: float = 0.0
    estimated_effort: str = "medium"  # quick, medium, substantial
    requires_others: bool = False
    is_blocker_for_others: bool = False
    action_items: list[str] = field(default_factory=list)


# Day name to weekday number mapping
DAY_NAMES = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6,
    'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6,
}


def parse_relative_day(day_name: str, reference_date: Optional[datetime] = None) -> Optional[datetime]:
    """Parse a day name to the next occurrence of that day.

    Args:
        day_name: Day name like 'monday', 'friday', etc.
        reference_date: Reference date (defaults to now)

    Returns:
        DateTime of next occurrence of that day
    """
    day_name = day_name.lower().strip()
    if day_name not in DAY_NAMES:
        return None

    ref = reference_date or datetime.now()
    target_weekday = DAY_NAMES[day_name]
    days_ahead = target_weekday - ref.weekday()

    # If target day is today or past, go to next week
    if days_ahead <= 0:
        days_ahead += 7

    return ref + timedelta(days=days_ahead)


def parse_date_reference(date_str: str, reference_date: Optional[datetime] = None) -> Optional[datetime]:
    """Parse various date references to datetime.

    Handles:
    - Day names: "by Friday", "on Monday"
    - Relative: "tomorrow", "today", "next week"
    - EOD/COB/EOW: End of day, close of business, end of week
    - Numeric: "12/25", "1-15"

    Args:
        date_str: Date string to parse
        reference_date: Reference date for relative calculations

    Returns:
        Parsed datetime or None if unparseable
    """
    ref = reference_date or datetime.now()
    date_str = date_str.lower().strip()

    # End of day / close of business (same day, 5pm)
    if date_str in ('eod', 'end of day', 'cob', 'close of business', 'today'):
        return ref.replace(hour=17, minute=0, second=0, microsecond=0)

    # Tomorrow
    if date_str == 'tomorrow':
        tomorrow = ref + timedelta(days=1)
        return tomorrow.replace(hour=17, minute=0, second=0, microsecond=0)

    # End of week (Friday 5pm)
    if date_str in ('eow', 'end of week', 'this week'):
        days_until_friday = (4 - ref.weekday()) % 7
        if days_until_friday == 0 and ref.hour >= 17:
            days_until_friday = 7
        friday = ref + timedelta(days=days_until_friday)
        return friday.replace(hour=17, minute=0, second=0, microsecond=0)

    # Next week (next Monday 9am)
    if date_str == 'next week':
        days_until_monday = (7 - ref.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        monday = ref + timedelta(days=days_until_monday)
        return monday.replace(hour=9, minute=0, second=0, microsecond=0)

    # Day names
    for day in DAY_NAMES:
        if day in date_str:
            return parse_relative_day(day, ref)

    # Numeric date patterns: MM/DD, M/D, MM-DD, M-D
    numeric_match = re.search(r'(\d{1,2})[/\-](\d{1,2})', date_str)
    if numeric_match:
        month = int(numeric_match.group(1))
        day = int(numeric_match.group(2))
        try:
            # Assume current year, or next year if date has passed
            target = ref.replace(month=month, day=day, hour=17, minute=0, second=0, microsecond=0)
            if target < ref:
                target = target.replace(year=target.year + 1)
            return target
        except ValueError:
            pass

    return None


def extract_deadlines(text: str, reference_date: Optional[datetime] = None) -> tuple[bool, Optional[datetime], str]:
    """Extract deadline information from text.

    Args:
        text: Email text to analyze
        reference_date: Reference date for relative calculations

    Returns:
        Tuple of (has_deadline, deadline_date, deadline_text)
    """
    deadline_patterns = [
        # "by <day/date>" patterns
        (r'by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 1),
        (r'by\s+(tomorrow|today|eod|cob|eow|end of day|end of week|close of business)', 1),
        (r'by\s+(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)', 1),
        (r'by\s+(next week)', 1),

        # "due <day/date>" patterns
        (r'due\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 1),
        (r'due\s+(tomorrow|today)', 1),
        (r'due\s+(?:date[:\s]+)?(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)', 1),

        # "before <day>" patterns
        (r'before\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 1),

        # "deadline: <text>" pattern
        (r'deadline[:\s]+([^\n.]+)', 1),

        # Urgency markers that imply immediate deadline
        (r'\b(asap|urgent|immediately)\b', 1),
    ]

    text_lower = text.lower()

    for pattern, group in deadline_patterns:
        match = re.search(pattern, text_lower, re.I)
        if match:
            deadline_text = match.group(group).strip()

            # Handle ASAP/urgent as today
            if deadline_text in ('asap', 'urgent', 'immediately'):
                deadline_date = parse_date_reference('eod', reference_date)
            else:
                deadline_date = parse_date_reference(deadline_text, reference_date)

            return True, deadline_date, deadline_text

    return False, None, ""


def extract_action_items(text: str) -> list[str]:
    """Extract potential action items from email text.

    Looks for:
    - Explicit requests: "Please...", "Can you...", "Need you to..."
    - Action required markers
    - Bullet points with action verbs

    Args:
        text: Email text to analyze

    Returns:
        List of extracted action item strings (max 5)
    """
    action_items = []

    # Request patterns - capture the action description
    request_patterns = [
        r'(?:please|pls)\s+([^.?!\n]{10,100})',
        r'(?:can you|could you|would you)\s+([^.?!\n]{10,100})',
        r'(?:need you to|need to)\s+([^.?!\n]{10,100})',
        r'(?:kindly)\s+([^.?!\n]{10,100})',
        r'i\'d like you to\s+([^.?!\n]{10,100})',
    ]

    for pattern in request_patterns:
        matches = re.findall(pattern, text, re.I)
        for match in matches:
            cleaned = match.strip()
            # Filter out very short or very long matches
            if 10 <= len(cleaned) <= 200:
                action_items.append(cleaned)

    # Action required markers
    action_marker_patterns = [
        r'action\s*(?:item|required|needed)?[:\s]+([^.\n]{10,150})',
        r'todo[:\s]+([^.\n]{10,150})',
        r'task[:\s]+([^.\n]{10,150})',
    ]

    for pattern in action_marker_patterns:
        matches = re.findall(pattern, text, re.I)
        for match in matches:
            cleaned = match.strip()
            if 10 <= len(cleaned) <= 200:
                action_items.append(cleaned)

    # Bullet points with action verbs at start
    action_verbs = r'(send|review|prepare|complete|finish|submit|update|create|write|call|email|schedule|confirm|check|follow up|investigate|resolve|fix|test)'
    bullet_pattern = rf'(?:^|\n)\s*[-•*]\s*{action_verbs}\s+([^.\n]{{5,100}})'

    matches = re.findall(bullet_pattern, text, re.I | re.M)
    for verb, rest in matches:
        action_items.append(f"{verb} {rest}".strip())

    # Deduplicate and limit
    seen = set()
    unique_items = []
    for item in action_items:
        normalized = item.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_items.append(item)

    return unique_items[:5]


def extract_deliverables(text: str) -> tuple[bool, str]:
    """Extract deliverable mentions from text.

    Looks for patterns indicating something needs to be produced/delivered.

    Args:
        text: Email text to analyze

    Returns:
        Tuple of (has_deliverable, deliverable_description)
    """
    deliverable_patterns = [
        r'send\s+(?:me\s+)?(?:the\s+)?([^.?!\n]{5,80})',
        r'prepare\s+(?:a\s+|the\s+)?([^.?!\n]{5,80})',
        r'draft\s+(?:a\s+|the\s+)?([^.?!\n]{5,80})',
        r'create\s+(?:a\s+|the\s+)?([^.?!\n]{5,80})',
        r'provide\s+(?:a\s+|the\s+)?([^.?!\n]{5,80})',
        r'submit\s+(?:the\s+)?([^.?!\n]{5,80})',
        r'deliver\s+(?:the\s+)?([^.?!\n]{5,80})',
    ]

    for pattern in deliverable_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            deliverable = match.group(1).strip()
            if len(deliverable) >= 5:
                return True, deliverable

    return False, ""


def detect_assignment(text: str, sender: str) -> tuple[bool, float]:
    """Detect if email assigns a task to the recipient.

    Args:
        text: Email text to analyze
        sender: Email sender address

    Returns:
        Tuple of (is_assigned, confidence)
    """
    # Strong assignment indicators
    strong_patterns = [
        r'\b(?:i need you to|you need to|your task is to|assigned to you)\b',
        r'\b(?:can you please|could you please|would you please)\b',
        r'\b(?:please take care of|please handle|please address)\b',
    ]

    # Moderate assignment indicators
    moderate_patterns = [
        r'\b(?:can you|could you|would you)\b',
        r'\b(?:please)\b',
        r'\b(?:let me know|get back to me)\b',
    ]

    text_lower = text.lower()

    # Check for strong patterns
    for pattern in strong_patterns:
        if re.search(pattern, text_lower):
            return True, 0.9

    # Check for moderate patterns
    moderate_count = sum(1 for p in moderate_patterns if re.search(p, text_lower))
    if moderate_count >= 2:
        return True, 0.7
    elif moderate_count == 1:
        return True, 0.5

    return False, 0.0


def estimate_effort(text: str, action_items: list[str]) -> str:
    """Estimate the effort required for tasks in email.

    Args:
        text: Email text
        action_items: Extracted action items

    Returns:
        Effort level: 'quick', 'medium', or 'substantial'
    """
    # Indicators of substantial effort
    substantial_patterns = [
        r'\b(?:comprehensive|detailed|thorough|in-depth|full)\b',
        r'\b(?:research|analysis|investigation|study)\b',
        r'\b(?:project|initiative|proposal|plan)\b',
        r'\b(?:multiple|several|all|entire)\b',
    ]

    # Indicators of quick effort
    quick_patterns = [
        r'\b(?:quick|brief|short|simple|just)\b',
        r'\b(?:confirm|acknowledge|let me know if)\b',
        r'\b(?:fyi|for your information)\b',
    ]

    text_lower = text.lower()

    substantial_count = sum(1 for p in substantial_patterns if re.search(p, text_lower))
    quick_count = sum(1 for p in quick_patterns if re.search(p, text_lower))

    # Also consider number of action items
    if len(action_items) >= 3:
        substantial_count += 1
    elif len(action_items) == 0:
        quick_count += 1

    if substantial_count >= 2:
        return 'substantial'
    elif quick_count >= 2 and substantial_count == 0:
        return 'quick'
    else:
        return 'medium'


def detect_blocking(text: str) -> tuple[bool, bool]:
    """Detect if task involves waiting on or blocking others.

    Args:
        text: Email text

    Returns:
        Tuple of (requires_others, is_blocker_for_others)
    """
    requires_others_patterns = [
        r'\b(?:coordinate with|work with|collaborate with)\b',
        r'\b(?:need.*from|waiting.*for|depends.*on)\b',
        r'\b(?:get.*input|get.*feedback|get.*approval)\b',
    ]

    blocker_patterns = [
        r'\b(?:blocking|waiting on you|need this to)\b',
        r'\b(?:can\'t proceed|cannot proceed|blocked until)\b',
        r'\b(?:depends on your|waiting for your)\b',
        r'\b(?:urgent|asap|critical|time.sensitive)\b',
    ]

    text_lower = text.lower()

    requires_others = any(re.search(p, text_lower) for p in requires_others_patterns)
    is_blocker = any(re.search(p, text_lower) for p in blocker_patterns)

    return requires_others, is_blocker


def extract_tasks(
    email: dict,
    reference_date: Optional[datetime] = None,
) -> TaskFeatures:
    """Extract task features from an email.

    Args:
        email: Email dictionary with 'subject', 'body', 'from' fields
        reference_date: Reference date for deadline calculations

    Returns:
        TaskFeatures with all extracted information
    """
    subject = email.get('subject', '')
    body = email.get('body', '')
    sender = email.get('from', '')
    full_text = f"{subject} {body}"

    # Extract deadlines
    has_deadline, deadline_date, deadline_text = extract_deadlines(full_text, reference_date)

    # Extract action items
    action_items = extract_action_items(full_text)

    # Extract deliverables
    has_deliverable, deliverable_description = extract_deliverables(full_text)

    # Detect assignment
    is_assigned, assignment_confidence = detect_assignment(full_text, sender)

    # Estimate effort
    effort = estimate_effort(full_text, action_items)

    # Detect blocking relationships
    requires_others, is_blocker = detect_blocking(full_text)

    return TaskFeatures(
        has_deadline=has_deadline,
        deadline_date=deadline_date,
        deadline_text=deadline_text,
        has_deliverable=has_deliverable,
        deliverable_description=deliverable_description,
        is_assigned_to_user=is_assigned,
        assigned_by=sender if is_assigned else "",
        assignment_confidence=assignment_confidence,
        estimated_effort=effort,
        requires_others=requires_others,
        is_blocker_for_others=is_blocker,
        action_items=action_items,
    )


def compute_task_score(features: TaskFeatures) -> float:
    """Compute likelihood that an email should become a tracked task.

    Score ranges from 0-1, with higher scores indicating the email
    should likely be converted to a task.

    Args:
        features: Extracted TaskFeatures

    Returns:
        Task score between 0 and 1
    """
    # Base case: no indicators
    if not features.action_items and not features.has_deadline and not features.is_assigned_to_user:
        return 0.1

    score = 0.2  # Base score for having any task indicators

    # Deadline boost
    if features.has_deadline:
        score += 0.25

        # Additional boost for imminent deadlines
        if features.deadline_date:
            days_until = (features.deadline_date - datetime.now()).days
            if days_until < 1:
                score += 0.25  # Due today
            elif days_until < 3:
                score += 0.15  # Due in 2-3 days
            elif days_until < 7:
                score += 0.10  # Due this week

    # Assignment boost
    if features.is_assigned_to_user:
        score += 0.15 * features.assignment_confidence

    # Deliverable boost
    if features.has_deliverable:
        score += 0.15

    # Action items boost
    if features.action_items:
        # More action items = more likely to be a task
        item_score = min(len(features.action_items) * 0.1, 0.2)
        score += item_score

    # Blocker boost - others waiting on this
    if features.is_blocker_for_others:
        score += 0.15

    # Effort penalty for trivial tasks
    if features.estimated_effort == 'quick' and score < 0.5:
        score *= 0.8  # Reduce score for quick tasks

    return min(score, 1.0)
