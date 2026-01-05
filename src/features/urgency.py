#!/usr/bin/env python3
"""Email-level urgency scoring.

URG-001: Unified urgency scoring combining:
- Explicit keywords (URGENT, ASAP, etc.)
- Sender hierarchy (executives get higher urgency weight)
- Deadline proximity
- Temporal factors
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from .topic import classify_topic, TopicFeatures
from .task import extract_tasks, TaskFeatures
from .people import extract_people_features, PeopleFeatures
from .temporal import extract_temporal_features, TemporalFeatures


# Explicit urgency keyword patterns with weights
URGENCY_KEYWORDS = {
    'critical': 1.0,
    'emergency': 1.0,
    'asap': 0.9,
    'urgent': 0.9,
    'immediately': 0.9,
    'time-sensitive': 0.8,
    'priority': 0.7,
    'important': 0.6,
    'deadline': 0.6,
    'due today': 0.8,
    'due tomorrow': 0.7,
    'eod': 0.8,
    'eob': 0.8,
    'cob': 0.8,
}

# Sender hierarchy urgency multipliers
SENDER_HIERARCHY_MULTIPLIERS = {
    0: 1.0,   # External
    1: 1.1,   # Peer
    2: 1.3,   # Manager
    3: 1.5,   # Executive
}


@dataclass
class UrgencyFeatures:
    """Email-level urgency features."""
    # Component scores
    keyword_urgency: float      # From explicit keywords (0-1)
    sender_urgency: float       # Adjusted for sender hierarchy (0-1)
    deadline_urgency: float     # From task deadline proximity (0-1)
    temporal_urgency: float     # From timing factors (0-1)

    # Combined score
    overall_urgency: float      # Weighted combination (0-1)

    # Metadata
    detected_keywords: list[str]
    sender_org_level: int
    has_explicit_deadline: bool
    hours_since_receipt: float

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array for ML pipeline.

        Returns 8-dimensional vector.
        """
        values = [
            self.keyword_urgency,
            self.sender_urgency,
            self.deadline_urgency,
            self.temporal_urgency,
            self.overall_urgency,
            float(self.has_explicit_deadline),
            self.sender_org_level / 3.0,  # Normalized
            min(self.hours_since_receipt / 168.0, 1.0),  # Normalized to 1 week
        ]
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


def _detect_urgency_keywords(text: str) -> tuple[float, list[str]]:
    """Detect urgency keywords and compute keyword-based urgency score.

    Args:
        text: Combined subject and body text

    Returns:
        Tuple of (urgency_score, list of detected keywords)
    """
    text_lower = text.lower()
    detected = []
    max_weight = 0.0

    for keyword, weight in URGENCY_KEYWORDS.items():
        pattern = r'\b' + re.escape(keyword).replace(r'\ ', r'\s+') + r'\b'
        if re.search(pattern, text_lower):
            detected.append(keyword)
            max_weight = max(max_weight, weight)

    # Use max weight rather than sum to avoid over-counting
    return max_weight, detected


def _compute_sender_adjusted_urgency(
    base_urgency: float,
    sender_org_level: int,
) -> float:
    """Adjust urgency based on sender hierarchy.

    Args:
        base_urgency: Base urgency score (0-1)
        sender_org_level: 0=external, 1=peer, 2=manager, 3=executive

    Returns:
        Adjusted urgency score (0-1)
    """
    multiplier = SENDER_HIERARCHY_MULTIPLIERS.get(sender_org_level, 1.0)
    return min(1.0, base_urgency * multiplier)


def compute_email_urgency(
    email: dict,
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
    reference_time: Optional[datetime] = None,
    weights: Optional[dict[str, float]] = None,
) -> UrgencyFeatures:
    """Compute unified email-level urgency score.

    Args:
        email: Email dictionary with subject, body, from, to, date, etc.
        user_email: User's email for context
        user_context: Historical interaction data
        reference_time: Time for temporal calculations (default: now)
        weights: Optional custom weights for combining scores

    Returns:
        UrgencyFeatures with component and combined urgency scores
    """
    if weights is None:
        weights = {
            'keyword': 0.30,      # Explicit keywords
            'sender': 0.15,       # Sender hierarchy
            'deadline': 0.35,     # Deadline proximity
            'temporal': 0.20,     # Time factors
        }

    subject = email.get('subject', '')
    body = email.get('body', '')
    combined_text = f"{subject}\n\n{body}"

    # 1. Extract keyword urgency
    keyword_urgency, detected_keywords = _detect_urgency_keywords(combined_text)

    # 2. Extract task features for deadline urgency
    task_features = extract_tasks(subject, body)
    deadline_urgency = task_features.deadline_urgency

    # 3. Extract people features for sender hierarchy
    people_features = extract_people_features(
        email,
        user_email=user_email,
        user_context=user_context,
    )
    sender_org_level = people_features.sender_org_level

    # 4. Extract temporal features
    temporal_features = extract_temporal_features(
        email,
        reference_time=reference_time,
    )
    temporal_urgency = temporal_features.temporal_urgency
    hours_since_receipt = temporal_features.time_since_receipt_hours

    # Compute sender-adjusted urgency
    sender_urgency = _compute_sender_adjusted_urgency(
        keyword_urgency, sender_org_level
    )

    # Combine all signals with weights
    overall_urgency = (
        weights.get('keyword', 0.30) * keyword_urgency +
        weights.get('sender', 0.15) * (sender_org_level / 3.0) +
        weights.get('deadline', 0.35) * deadline_urgency +
        weights.get('temporal', 0.20) * temporal_urgency
    )

    # Apply sender multiplier to final score
    if sender_org_level >= 2:  # Manager or higher
        overall_urgency = min(1.0, overall_urgency * 1.2)

    return UrgencyFeatures(
        keyword_urgency=keyword_urgency,
        sender_urgency=sender_urgency,
        deadline_urgency=deadline_urgency,
        temporal_urgency=temporal_urgency,
        overall_urgency=min(1.0, max(0.0, overall_urgency)),
        detected_keywords=detected_keywords,
        sender_org_level=sender_org_level,
        has_explicit_deadline=task_features.has_deadline,
        hours_since_receipt=hours_since_receipt,
    )


def batch_compute_urgency(
    emails: list[dict],
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
    reference_time: Optional[datetime] = None,
    weights: Optional[dict[str, float]] = None,
) -> list[UrgencyFeatures]:
    """Compute urgency for a batch of emails.

    Args:
        emails: List of email dictionaries
        user_email: User's email for context
        user_context: Historical interaction data
        reference_time: Time for temporal calculations
        weights: Custom weights for combining scores

    Returns:
        List of UrgencyFeatures, one per email
    """
    return [
        compute_email_urgency(
            email,
            user_email=user_email,
            user_context=user_context,
            reference_time=reference_time,
            weights=weights,
        )
        for email in emails
    ]


def urgency_to_priority_bucket(urgency: float) -> str:
    """Convert urgency score to priority bucket.

    Args:
        urgency: Urgency score 0-1

    Returns:
        Priority bucket: 'critical', 'high', 'medium', 'low'
    """
    if urgency >= 0.8:
        return 'critical'
    elif urgency >= 0.6:
        return 'high'
    elif urgency >= 0.4:
        return 'medium'
    else:
        return 'low'


if __name__ == '__main__':
    # Example usage
    sample_email = {
        'from': 'ceo@enron.com',
        'to': 'user@enron.com',
        'cc': '',
        'x_from': 'John Smith, CEO',
        'subject': 'URGENT: Decision needed by EOD',
        'body': '''
        Hi,

        This is critical - we need your decision on the proposal ASAP.
        The deadline is today and we cannot proceed without your approval.

        Please review and respond immediately.

        Thanks,
        John
        ''',
        'date': datetime.now().isoformat(),
    }

    urgency = compute_email_urgency(sample_email, user_email='user@enron.com')

    print("=" * 60)
    print("EMAIL URGENCY SCORING")
    print("=" * 60)
    print()
    print("Component Scores:")
    print(f"  Keyword urgency:  {urgency.keyword_urgency:.2f}")
    print(f"  Sender urgency:   {urgency.sender_urgency:.2f}")
    print(f"  Deadline urgency: {urgency.deadline_urgency:.2f}")
    print(f"  Temporal urgency: {urgency.temporal_urgency:.2f}")
    print()
    print(f"Overall Urgency: {urgency.overall_urgency:.2f}")
    print(f"Priority Bucket: {urgency_to_priority_bucket(urgency.overall_urgency)}")
    print()
    print("Detected Keywords:", urgency.detected_keywords)
    print(f"Sender Org Level: {urgency.sender_org_level}")
    print(f"Has Deadline: {urgency.has_explicit_deadline}")
    print(f"Hours Since Receipt: {urgency.hours_since_receipt:.1f}")
    print()
    print(f"Feature Vector ({len(urgency.to_feature_vector())} dims):")
    print(f"  {list(urgency.to_feature_vector())}")
    print("=" * 60)
