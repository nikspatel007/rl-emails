#!/usr/bin/env python3
"""Task extraction from email content.

Extracts potential tasks and action items:
- Deadlines (explicit dates, day names, urgency markers)
- Action items (requests, questions, todos)
- Deliverables (attachments to send, reports to prepare)
- Assignment detection (who is being asked to do what)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# Day name patterns
DAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
DAY_PATTERN = r'\b(?:' + '|'.join(DAY_NAMES) + r')\b'

# Deadline patterns
DEADLINE_PATTERNS = [
    # Explicit day references
    (r'\bby\s+(' + DAY_PATTERN + r')\b', 'day'),
    (r'\bby\s+(today|tomorrow|tonight)\b', 'relative'),
    (r'\b(today|tomorrow|tonight)\b', 'relative'),

    # Time-bound expressions
    (r'\bby\s+(?:the\s+)?end\s+of\s+(?:the\s+)?(day|week|month|quarter|year)\b', 'period'),
    (r'\b(EOD|EOB|COB|EOW|EOM)\b', 'acronym'),

    # Specific dates
    (r'\bby\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b', 'date'),
    (r'\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b', 'date'),
    (r'\bdue\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b', 'date'),

    # Month day patterns
    (r'\bby\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?)\b', 'date'),
    (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?)\b', 'date'),

    # Time expressions
    (r'\bby\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM))\b', 'time'),
]

# Action item patterns (imperative/request language)
ACTION_ITEM_PATTERNS = [
    (r'\b(please\s+(?:review|send|update|confirm|check|call|respond|reply|forward|prepare|complete|provide|schedule|set up|follow up|look into)[^.?!]{0,50})', 'request'),
    (r'\b(can you\s+[^.?!]{5,50})', 'request'),
    (r'\b(could you\s+[^.?!]{5,50})', 'request'),
    (r'\b(would you\s+[^.?!]{5,50})', 'request'),
    (r'\b(need you to\s+[^.?!]{5,50})', 'assignment'),
    (r'\b(you need to\s+[^.?!]{5,50})', 'assignment'),
    (r'\b(action\s*(?:item|required)[:\s]+[^.?!\n]{5,80})', 'explicit'),
    (r'\b(TODO[:\s]+[^.?!\n]{5,80})', 'explicit'),
    (r'(?:^|\n)\s*[-•*]\s*([^.?!\n]{10,80}(?:you|your)[^.?!\n]{0,40})', 'bullet'),
]

# Deliverable patterns
DELIVERABLE_PATTERNS = [
    (r'\b(send\s+(?:me|us|them)\s+[^.?!]{5,50})', 'send'),
    (r'\b(prepare\s+[^.?!]{5,50})', 'prepare'),
    (r'\b(draft\s+[^.?!]{5,50})', 'draft'),
    (r'\b(create\s+[^.?!]{5,50})', 'create'),
    (r'\b(update\s+(?:the\s+)?[^.?!]{5,50})', 'update'),
    (r'\b(provide\s+[^.?!]{5,50})', 'provide'),
    (r'\b(submit\s+[^.?!]{5,50})', 'submit'),
]

# Assignment confidence patterns
STRONG_ASSIGNMENT_PATTERNS = [
    r'\byou\s+(?:are|need|should|must|will)\b',
    r'\b(?:your|you)\s+(?:responsibility|task|assignment)\b',
    r'\bassigned\s+to\s+you\b',
    r'\bI(?:\'m|am)\s+asking\s+you\b',
]

WEAK_ASSIGNMENT_PATTERNS = [
    r'\bcan\s+you\b',
    r'\bcould\s+you\b',
    r'\bwould\s+you\b',
    r'\bplease\b',
]

# Effort estimation patterns
QUICK_EFFORT_PATTERNS = [
    r'\bquick\b',
    r'\bbrief\b',
    r'\bshort\b',
    r'\bsimple\b',
    r'\bjust\s+(?:a|one)\b',
    r'\bfew\s+minutes\b',
]

SUBSTANTIAL_EFFORT_PATTERNS = [
    r'\bdetailed\b',
    r'\bcomprehensive\b',
    r'\bthorough\b',
    r'\bextensive\b',
    r'\banalysis\b',
    r'\breport\b',
    r'\bpresentation\b',
]

# Blocking patterns
BLOCKING_PATTERNS = [
    r'\bblocking\b',
    r'\bblocked\s+on\b',
    r'\bwaiting\s+(?:for|on)\b',
    r'\bdepends\s+on\b',
    r'\bbefore\s+(?:we|I)\s+can\b',
    r'\bneeded\s+(?:before|for)\b',
]


@dataclass
class ExtractedDeadline:
    """A detected deadline in email text."""
    text: str
    deadline_type: str  # 'day', 'relative', 'period', 'acronym', 'date', 'time'
    start: int
    end: int


@dataclass
class ExtractedActionItem:
    """A detected action item in email text."""
    text: str
    action_type: str  # 'request', 'assignment', 'explicit', 'bullet'
    start: int
    end: int


@dataclass
class TaskFeatures:
    """Extracted task features from an email."""
    # Deadline info
    has_deadline: bool
    deadline_text: Optional[str]
    deadline_urgency: float  # 0-1, higher = more urgent

    # Deliverable info
    has_deliverable: bool
    deliverable_description: Optional[str]

    # Assignment info
    is_assigned_to_user: bool
    assignment_confidence: float  # 0-1

    # Task complexity
    estimated_effort: str  # 'quick', 'medium', 'substantial'
    requires_others: bool
    is_blocker_for_others: bool

    # Extracted items
    action_items: list[str] = field(default_factory=list)
    deadlines: list[ExtractedDeadline] = field(default_factory=list)

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array for ML pipeline.

        Returns 12-dimensional vector.
        """
        effort_map = {'quick': 0.2, 'medium': 0.5, 'substantial': 0.9}
        values = [
            1.0 if self.has_deadline else 0.0,
            self.deadline_urgency,
            1.0 if self.has_deliverable else 0.0,
            1.0 if self.is_assigned_to_user else 0.0,
            self.assignment_confidence,
            effort_map.get(self.estimated_effort, 0.5),
            1.0 if self.requires_others else 0.0,
            1.0 if self.is_blocker_for_others else 0.0,
            min(len(self.action_items), 5) / 5.0,  # Normalized count
            min(len(self.deadlines), 3) / 3.0,
            # Derived scores
            self._compute_task_likelihood(),
            self._compute_urgency_likelihood(),
        ]
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values

    def _compute_task_likelihood(self) -> float:
        """Compute likelihood this email represents a task."""
        score = 0.0
        if self.has_deadline:
            score += 0.3
        if self.has_deliverable:
            score += 0.2
        if self.is_assigned_to_user:
            score += 0.3 * self.assignment_confidence
        if self.action_items:
            score += min(0.2, len(self.action_items) * 0.05)
        return min(1.0, score)

    def _compute_urgency_likelihood(self) -> float:
        """Compute urgency of the task."""
        score = self.deadline_urgency * 0.5
        if self.is_blocker_for_others:
            score += 0.3
        if self.estimated_effort == 'quick':
            score += 0.1  # Quick tasks often need quick response
        return min(1.0, score)


def _extract_pattern_matches(text: str, patterns: list[tuple[str, str]]) -> list[tuple[str, str, int, int]]:
    """Extract matches for a list of (pattern, type) tuples."""
    matches = []
    for pattern, match_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            # Get the captured group if exists, else full match
            captured = match.group(1) if match.lastindex else match.group(0)
            matches.append((captured.strip(), match_type, match.start(), match.end()))
    return matches


def _deduplicate_items(items: list, key_fn=lambda x: (x[2], x[3])) -> list:
    """Remove overlapping items, keeping first occurrence."""
    if not items:
        return items

    sorted_items = sorted(items, key=lambda x: (x[2], -(x[3] - x[2])))
    result = []
    last_end = -1

    for item in sorted_items:
        start, end = item[2], item[3]
        if start >= last_end:
            result.append(item)
            last_end = end

    return result


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count matches for a list of regex patterns."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def _compute_deadline_urgency(deadlines: list[ExtractedDeadline]) -> float:
    """Compute urgency score based on detected deadlines."""
    if not deadlines:
        return 0.0

    max_urgency = 0.0
    for deadline in deadlines:
        # Acronyms (EOD, ASAP) are most urgent
        if deadline.deadline_type == 'acronym':
            max_urgency = max(max_urgency, 0.9)
        # Relative (today, tomorrow)
        elif deadline.deadline_type == 'relative':
            if 'today' in deadline.text.lower() or 'tonight' in deadline.text.lower():
                max_urgency = max(max_urgency, 0.9)
            elif 'tomorrow' in deadline.text.lower():
                max_urgency = max(max_urgency, 0.7)
        # Day names
        elif deadline.deadline_type == 'day':
            max_urgency = max(max_urgency, 0.5)
        # Specific dates/times
        elif deadline.deadline_type in ('date', 'time'):
            max_urgency = max(max_urgency, 0.4)
        # Period (end of week, month)
        elif deadline.deadline_type == 'period':
            if 'day' in deadline.text.lower():
                max_urgency = max(max_urgency, 0.8)
            elif 'week' in deadline.text.lower():
                max_urgency = max(max_urgency, 0.5)
            else:
                max_urgency = max(max_urgency, 0.3)

    return max_urgency


def _estimate_effort(text: str) -> str:
    """Estimate effort level based on text patterns."""
    quick_count = _count_pattern_matches(text, QUICK_EFFORT_PATTERNS)
    substantial_count = _count_pattern_matches(text, SUBSTANTIAL_EFFORT_PATTERNS)

    if substantial_count > quick_count:
        return 'substantial'
    elif quick_count > substantial_count:
        return 'quick'
    else:
        return 'medium'


def extract_tasks(
    subject: str,
    body: str,
    *,
    subject_weight: float = 2.0,
) -> TaskFeatures:
    """Extract task-related features from an email.

    Args:
        subject: Email subject line
        body: Email body text
        subject_weight: Weight for subject line (not used for extraction, kept for API consistency)

    Returns:
        TaskFeatures with extracted information
    """
    combined_text = f"{subject}\n\n{body}"
    text_lower = combined_text.lower()

    # Extract deadlines
    deadline_matches = _extract_pattern_matches(combined_text, DEADLINE_PATTERNS)
    deadline_matches = _deduplicate_items(deadline_matches)
    deadlines = [
        ExtractedDeadline(text=m[0], deadline_type=m[1], start=m[2], end=m[3])
        for m in deadline_matches
    ]

    # Extract action items
    action_matches = _extract_pattern_matches(combined_text, ACTION_ITEM_PATTERNS)
    action_matches = _deduplicate_items(action_matches)
    action_items = [m[0] for m in action_matches[:5]]  # Top 5

    # Extract deliverables
    deliverable_matches = _extract_pattern_matches(combined_text, DELIVERABLE_PATTERNS)
    has_deliverable = len(deliverable_matches) > 0
    deliverable_description = deliverable_matches[0][0] if deliverable_matches else None

    # Compute assignment confidence
    strong_matches = _count_pattern_matches(text_lower, STRONG_ASSIGNMENT_PATTERNS)
    weak_matches = _count_pattern_matches(text_lower, WEAK_ASSIGNMENT_PATTERNS)
    assignment_confidence = min(1.0, strong_matches * 0.3 + weak_matches * 0.1)
    is_assigned_to_user = assignment_confidence > 0.3

    # Check for blocking relationships
    blocking_matches = _count_pattern_matches(text_lower, BLOCKING_PATTERNS)
    requires_others = blocking_matches > 0
    # Check if this task blocks others
    blocks_others_patterns = [r'\bblocking\b', r'\bneeded\s+(?:before|for)\b', r'\bbefore\s+(?:we|I)\s+can\b']
    is_blocker = _count_pattern_matches(text_lower, blocks_others_patterns) > 0

    # Compute deadline urgency
    deadline_urgency = _compute_deadline_urgency(deadlines)

    # Estimate effort
    estimated_effort = _estimate_effort(combined_text)

    return TaskFeatures(
        has_deadline=len(deadlines) > 0,
        deadline_text=deadlines[0].text if deadlines else None,
        deadline_urgency=deadline_urgency,
        has_deliverable=has_deliverable,
        deliverable_description=deliverable_description,
        is_assigned_to_user=is_assigned_to_user,
        assignment_confidence=assignment_confidence,
        estimated_effort=estimated_effort,
        requires_others=requires_others,
        is_blocker_for_others=is_blocker,
        action_items=action_items,
        deadlines=deadlines,
    )


def compute_task_score(features: TaskFeatures) -> float:
    """Compute overall task score for RL pipeline.

    Score 0-1 indicating likelihood this should become a tracked task.

    Args:
        features: TaskFeatures from extract_tasks()

    Returns:
        Score from 0 to 1
    """
    if not features.action_items and not features.has_deadline:
        return 0.1

    score = 0.3  # Base for having some task indicators

    if features.has_deadline:
        score += 0.3
        # Add urgency component
        score += features.deadline_urgency * 0.2

    if features.is_assigned_to_user:
        score += 0.2 * features.assignment_confidence

    if features.has_deliverable:
        score += 0.15

    if features.is_blocker_for_others:
        score += 0.2

    return min(1.0, score)


if __name__ == '__main__':
    # Example usage
    sample_subject = "Action Required: Review proposal by Friday"
    sample_body = """
    Hi Team,

    Please review the attached proposal and provide your feedback by EOD Friday.

    Key items that need your attention:
    - Budget estimates for Q2
    - Timeline review
    - Resource allocation

    Can you also send me the updated spreadsheet? I need it before we can
    proceed with the client meeting.

    This is blocking the final presentation to the board.

    Thanks,
    Mike
    """

    features = extract_tasks(sample_subject, sample_body)

    print("Task Features Extracted:")
    print(f"  Has deadline: {features.has_deadline}")
    if features.deadline_text:
        print(f"    Deadline: {features.deadline_text}")
    print(f"    Urgency: {features.deadline_urgency:.2f}")
    print()
    print(f"  Has deliverable: {features.has_deliverable}")
    if features.deliverable_description:
        print(f"    Deliverable: {features.deliverable_description}")
    print()
    print(f"  Assigned to user: {features.is_assigned_to_user}")
    print(f"    Confidence: {features.assignment_confidence:.2f}")
    print()
    print(f"  Effort estimate: {features.estimated_effort}")
    print(f"  Requires others: {features.requires_others}")
    print(f"  Blocks others: {features.is_blocker_for_others}")
    print()
    print(f"  Action items ({len(features.action_items)}):")
    for item in features.action_items:
        print(f"    - {item[:60]}...")
    print()
    print(f"  Task score: {compute_task_score(features):.2f}")
    print(f"  Feature vector ({len(features.to_feature_vector())} dims)")
