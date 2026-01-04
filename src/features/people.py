#!/usr/bin/env python3
"""People score extraction from email metadata.

Scores emails based on:
- Sender importance (org level, historical patterns)
- Relationship strength (reply rate, interaction frequency)
- Recipient analysis (direct vs CC, includes executives)
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# Domain classification patterns
INTERNAL_DOMAIN_PATTERNS = [
    r'@enron\.com$',
    r'@enron\.net$',
]

EXTERNAL_DOMAIN_PATTERNS = [
    r'@(?!enron\.)[a-z0-9.-]+\.[a-z]{2,}$',
]

# Executive title patterns (in X-From or display names)
EXECUTIVE_PATTERNS = [
    r'\b(?:CEO|CFO|COO|CTO|CIO|CHRO)\b',
    r'\b(?:President|Vice\s+President|VP)\b',
    r'\b(?:Director|Senior\s+Director)\b',
    r'\b(?:Managing\s+Director|MD)\b',
    r'\b(?:Partner|Principal)\b',
    r'\b(?:Chief\s+[A-Z][a-z]+\s+Officer)\b',
]

MANAGER_PATTERNS = [
    r'\b(?:Manager|Senior\s+Manager)\b',
    r'\b(?:Team\s+Lead|Lead)\b',
    r'\b(?:Supervisor|Head)\b',
]

# Automated sender patterns
AUTOMATED_SENDER_PATTERNS = [
    r'\b(?:noreply|no-reply|donotreply|do-not-reply)\b',
    r'\b(?:automated|notification|alert|system)\b',
    r'@.*\.(?:notifications?|alerts?|system)\.com$',
]


@dataclass
class PeopleFeatures:
    """People-related features for email scoring."""
    # Sender analysis
    sender_email: str
    sender_domain: str
    sender_is_internal: bool
    sender_org_level: int  # 0=external, 1=peer, 2=manager, 3=executive
    sender_is_automated: bool

    # Recipient analysis
    recipient_count: int
    cc_count: int
    is_direct_to: bool  # User in To: vs CC:
    includes_executives: bool
    includes_managers: bool

    # Historical metrics (if available, default to neutral)
    reply_rate_to_sender: float  # 0-1
    avg_response_time_hours: float  # Average response time
    emails_from_sender_30d: int  # Volume from this sender
    last_interaction_days: int  # Days since last interaction

    # Derived scores
    sender_importance: float  # 0-1
    relationship_strength: float  # 0-1

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array for ML pipeline.

        Returns 15-dimensional vector.
        """
        values = [
            # Sender features (5)
            1.0 if self.sender_is_internal else 0.0,
            self.sender_org_level / 3.0,  # Normalized 0-1
            1.0 if self.sender_is_automated else 0.0,
            self.sender_importance,
            self.relationship_strength,

            # Recipient features (5)
            min(self.recipient_count, 20) / 20.0,
            min(self.cc_count, 50) / 50.0,
            1.0 if self.is_direct_to else 0.0,
            1.0 if self.includes_executives else 0.0,
            1.0 if self.includes_managers else 0.0,

            # Historical metrics (5)
            self.reply_rate_to_sender,
            min(self.avg_response_time_hours, 168) / 168.0,  # Cap at 1 week
            min(self.emails_from_sender_30d, 100) / 100.0,
            min(self.last_interaction_days, 90) / 90.0,
            self._compute_overall_score(),
        ]
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values

    def _compute_overall_score(self) -> float:
        """Compute overall people score."""
        return compute_people_score(self)


def _extract_email_parts(email_addr: str) -> tuple[str, str]:
    """Extract local part and domain from email address."""
    email_addr = email_addr.strip().lower()

    # Handle display name format: "John Smith" <john.smith@example.com>
    match = re.search(r'<([^>]+)>', email_addr)
    if match:
        email_addr = match.group(1)

    if '@' in email_addr:
        local, domain = email_addr.rsplit('@', 1)
        return local, domain
    return email_addr, ''


def _is_internal_email(email_addr: str) -> bool:
    """Check if email is from internal domain."""
    _, domain = _extract_email_parts(email_addr)
    for pattern in INTERNAL_DOMAIN_PATTERNS:
        if re.search(pattern, f"@{domain}", re.IGNORECASE):
            return True
    return False


def _is_automated_sender(email_addr: str, x_from: str = '') -> bool:
    """Check if sender is automated."""
    combined = f"{email_addr} {x_from}".lower()
    for pattern in AUTOMATED_SENDER_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return True
    return False


def _detect_org_level(x_from: str, sender_email: str, is_internal: bool) -> int:
    """Detect organizational level of sender.

    Returns:
        0 = external
        1 = peer/individual contributor
        2 = manager
        3 = executive
    """
    if not is_internal:
        return 0

    combined = f"{x_from} {sender_email}".upper()

    # Check executive patterns
    for pattern in EXECUTIVE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return 3

    # Check manager patterns
    for pattern in MANAGER_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return 2

    # Default to peer for internal
    return 1


def _count_recipients(to_field: str, cc_field: str) -> tuple[int, int]:
    """Count recipients in To and CC fields."""
    def count_emails(field: str) -> int:
        if not field:
            return 0
        # Count email addresses (look for @ signs or comma-separated items)
        emails = re.findall(r'[^,;\s]+@[^,;\s]+|[^,;]+(?=,|;|$)', field)
        return len([e for e in emails if e.strip()])

    to_count = count_emails(to_field)
    cc_count = count_emails(cc_field)
    return to_count, cc_count


def _check_executive_recipients(to_field: str, cc_field: str, x_to: str = '') -> tuple[bool, bool]:
    """Check if recipients include executives or managers."""
    combined = f"{to_field} {cc_field} {x_to}".upper()

    has_executives = any(
        re.search(pattern, combined, re.IGNORECASE)
        for pattern in EXECUTIVE_PATTERNS
    )

    has_managers = any(
        re.search(pattern, combined, re.IGNORECASE)
        for pattern in MANAGER_PATTERNS
    )

    return has_executives, has_managers


def _is_direct_recipient(to_field: str, user_email: str) -> bool:
    """Check if user is in To: field (not CC)."""
    if not user_email or not to_field:
        return True  # Assume direct if we don't know

    to_lower = to_field.lower()
    user_lower = user_email.lower()

    # Check if user email or local part appears in To field
    local, domain = _extract_email_parts(user_email)
    return user_lower in to_lower or local in to_lower


def extract_people_features(
    email: dict,
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
) -> PeopleFeatures:
    """Extract people-related features from an email.

    Args:
        email: Email dictionary with from, to, cc, x_from, x_to fields
        user_email: The user's email address (for context)
        user_context: Optional dict with historical interaction data

    Returns:
        PeopleFeatures with extracted information
    """
    # Extract sender info
    sender_email = email.get('from', '')
    x_from = email.get('x_from', '')
    local, domain = _extract_email_parts(sender_email)
    is_internal = _is_internal_email(sender_email)
    is_automated = _is_automated_sender(sender_email, x_from)
    org_level = _detect_org_level(x_from, sender_email, is_internal)

    # Extract recipient info
    to_field = email.get('to', '')
    cc_field = email.get('cc', '')
    x_to = email.get('x_to', '')
    recipient_count, cc_count = _count_recipients(to_field, cc_field)
    is_direct = _is_direct_recipient(to_field, user_email)
    has_execs, has_managers = _check_executive_recipients(to_field, cc_field, x_to)

    # Historical metrics (use context if available, else defaults)
    if user_context:
        reply_rate = user_context.get('reply_rate_to_sender', 0.5)
        avg_response = user_context.get('avg_response_time_hours', 24.0)
        email_volume = user_context.get('emails_from_sender_30d', 10)
        last_days = user_context.get('last_interaction_days', 7)
    else:
        # Neutral defaults for when we don't have history
        reply_rate = 0.5
        avg_response = 24.0
        email_volume = 10
        last_days = 7

    # Compute derived scores
    sender_importance = _compute_sender_importance(
        org_level, is_internal, is_automated, email_volume
    )
    relationship_strength = _compute_relationship_strength(
        reply_rate, avg_response, last_days, email_volume
    )

    return PeopleFeatures(
        sender_email=sender_email,
        sender_domain=domain,
        sender_is_internal=is_internal,
        sender_org_level=org_level,
        sender_is_automated=is_automated,
        recipient_count=recipient_count,
        cc_count=cc_count,
        is_direct_to=is_direct,
        includes_executives=has_execs,
        includes_managers=has_managers,
        reply_rate_to_sender=reply_rate,
        avg_response_time_hours=avg_response,
        emails_from_sender_30d=email_volume,
        last_interaction_days=last_days,
        sender_importance=sender_importance,
        relationship_strength=relationship_strength,
    )


def _compute_sender_importance(
    org_level: int,
    is_internal: bool,
    is_automated: bool,
    email_volume: int,
) -> float:
    """Compute sender importance score."""
    if is_automated:
        return 0.1

    # Base score from org level
    level_scores = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.9}
    score = level_scores.get(org_level, 0.5)

    # Internal boost
    if is_internal:
        score += 0.1

    # Frequent sender penalty (too many emails = less important per email)
    if email_volume > 50:
        score *= 0.8
    elif email_volume > 20:
        score *= 0.9

    return min(1.0, score)


def _compute_relationship_strength(
    reply_rate: float,
    avg_response: float,
    last_days: int,
    email_volume: int,
) -> float:
    """Compute relationship strength score."""
    score = 0.0

    # Reply rate component
    if reply_rate > 0.7:
        score += 0.4
    elif reply_rate > 0.4:
        score += 0.25
    elif reply_rate > 0.2:
        score += 0.1

    # Response time component (faster = stronger relationship)
    if avg_response < 1:
        score += 0.3
    elif avg_response < 4:
        score += 0.2
    elif avg_response < 24:
        score += 0.1

    # Recency component
    if last_days < 7:
        score += 0.2
    elif last_days < 30:
        score += 0.1

    # Volume component (some interaction history)
    if email_volume > 10:
        score += 0.1
    elif email_volume > 5:
        score += 0.05

    return min(1.0, score)


def compute_people_score(features: PeopleFeatures) -> float:
    """Compute overall people score for RL pipeline.

    Score 0-1 based on who sent the email and who's involved.

    Args:
        features: PeopleFeatures from extract_people_features()

    Returns:
        Score from 0 to 1
    """
    if features.sender_is_automated:
        return 0.1

    score = 0.0

    # Organizational hierarchy weight (30%)
    hierarchy_weights = {0: 0.3, 1: 0.5, 2: 0.8, 3: 1.0}
    score += hierarchy_weights.get(features.sender_org_level, 0.3) * 0.3

    # Relationship strength (20%)
    if features.reply_rate_to_sender > 0.7:
        score += 0.2
    elif features.reply_rate_to_sender > 0.4:
        score += 0.1

    # Direct recipient bonus (20%)
    if features.is_direct_to:
        score += 0.2

    # Recency penalty for cold contacts
    if features.last_interaction_days > 90:
        score *= 0.8
    elif features.last_interaction_days > 30:
        score *= 0.9

    # Executive involvement (15%)
    if features.includes_executives:
        score += 0.15

    # Manager involvement (10%)
    if features.includes_managers:
        score += 0.1

    # Large CC list penalty (mass emails less important)
    if features.cc_count > 20:
        score *= 0.7
    elif features.cc_count > 10:
        score *= 0.85

    return max(0.0, min(1.0, score))


if __name__ == '__main__':
    # Example usage
    sample_email = {
        'from': 'john.smith@enron.com',
        'to': 'jane.doe@enron.com',
        'cc': 'bob.jones@enron.com, alice.wang@enron.com',
        'x_from': 'Smith, John (Vice President, Trading)',
        'x_to': 'Doe, Jane',
    }

    features = extract_people_features(
        sample_email,
        user_email='jane.doe@enron.com',
        user_context={
            'reply_rate_to_sender': 0.8,
            'avg_response_time_hours': 2.5,
            'emails_from_sender_30d': 15,
            'last_interaction_days': 2,
        }
    )

    print("People Features Extracted:")
    print(f"  Sender: {features.sender_email}")
    print(f"  Domain: {features.sender_domain}")
    print(f"  Internal: {features.sender_is_internal}")
    print(f"  Org level: {features.sender_org_level}")
    print(f"  Automated: {features.sender_is_automated}")
    print()
    print(f"  Recipients: {features.recipient_count}")
    print(f"  CC count: {features.cc_count}")
    print(f"  Direct recipient: {features.is_direct_to}")
    print(f"  Includes executives: {features.includes_executives}")
    print(f"  Includes managers: {features.includes_managers}")
    print()
    print(f"  Reply rate: {features.reply_rate_to_sender:.2f}")
    print(f"  Avg response: {features.avg_response_time_hours:.1f}h")
    print(f"  Volume (30d): {features.emails_from_sender_30d}")
    print(f"  Last interaction: {features.last_interaction_days}d")
    print()
    print(f"  Sender importance: {features.sender_importance:.2f}")
    print(f"  Relationship strength: {features.relationship_strength:.2f}")
    print(f"  People score: {compute_people_score(features):.2f}")
    print(f"  Feature vector ({len(features.to_feature_vector())} dims)")
