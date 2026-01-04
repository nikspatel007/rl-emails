#!/usr/bin/env python3
"""Temporal feature extraction for email RL system.

Extracts time-based features that capture:
- Hour of day and day of week (cyclical encoding)
- Time since email was received
- Business hours / weekend detection
- Thread timing patterns
- Relationship decay over time

User email behavior varies significantly by time - people respond faster
during business hours, may defer weekend emails, etc.

Relationship decay model:
- Relationships decay without contact (half-life ~30 days)
- Formula: relationship_strength * decay_factor^days_since_contact
- recent_emails provide a baseline that decays over time
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# Business hours configuration (can be customized per user)
DEFAULT_BUSINESS_START = 9   # 9 AM
DEFAULT_BUSINESS_END = 18    # 6 PM
DEFAULT_BUSINESS_DAYS = {0, 1, 2, 3, 4}  # Monday-Friday

# Relationship decay configuration
DEFAULT_DECAY_HALF_LIFE_DAYS = 30.0  # Relationships decay by half every 30 days
# decay_factor = 0.5^(1/half_life) for daily decay
# Or use exponential: exp(-ln(2) * days / half_life)
DECAY_LAMBDA = math.log(2) / DEFAULT_DECAY_HALF_LIFE_DAYS  # ~0.0231


@dataclass
class TemporalFeatures:
    """Time-based features for email scoring."""
    # Raw temporal values
    hour_of_day: int          # 0-23
    day_of_week: int          # 0=Monday, 6=Sunday
    email_timestamp: Optional[datetime]

    # Cyclical encodings (for continuous representation)
    hour_sin: float           # sin(2π * hour/24)
    hour_cos: float           # cos(2π * hour/24)
    day_sin: float            # sin(2π * day/7)
    day_cos: float            # cos(2π * day/7)

    # Derived features
    time_since_receipt_hours: float  # Hours since email arrived
    is_business_hours: bool          # Within business hours
    is_weekend: bool                 # Saturday or Sunday

    # Thread timing
    time_since_last_in_thread_hours: float  # Hours since last email in thread

    # Derived score
    temporal_urgency: float  # 0-1 score based on temporal factors

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array for ML pipeline.

        Returns 8-dimensional vector.
        """
        values = [
            # Cyclical hour encoding (2)
            self.hour_sin,
            self.hour_cos,

            # Cyclical day encoding (2)
            self.day_sin,
            self.day_cos,

            # Time since receipt - normalized (1)
            # Cap at 7 days (168 hours), normalize to 0-1
            min(self.time_since_receipt_hours, 168.0) / 168.0,

            # Binary features (2)
            1.0 if self.is_business_hours else 0.0,
            1.0 if self.is_weekend else 0.0,

            # Thread timing - normalized (1)
            # Cap at 7 days, normalize to 0-1
            min(self.time_since_last_in_thread_hours, 168.0) / 168.0,
        ]
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


@dataclass
class RelationshipDecayFeatures:
    """Features modeling relationship strength decay over time.

    Key formula: decayed_strength = recent_emails * decay_factor^days
    With half-life of 30 days, relationships lose half their "freshness"
    each month without contact.
    """
    # Input values
    days_since_last_contact: float  # Days since last email from sender
    recent_email_count: int  # Emails in the observation window (e.g., 90 days)
    half_life_days: float  # Decay half-life (default 30)

    # Computed decay values
    decay_factor: float  # exp(-lambda * days), range 0-1
    decayed_strength: float  # recent_emails * decay_factor
    relationship_freshness: float  # Normalized 0-1 score

    # Derived signals
    is_dormant: bool  # No contact > 2x half-life (60+ days default)
    is_active: bool  # Contact within half-life (30 days default)
    urgency_boost: float  # Boost for re-engaging dormant relationships

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array for ML pipeline.

        Returns 6-dimensional vector.
        """
        values = [
            # Core decay features (3)
            self.decay_factor,
            min(self.decayed_strength, 10.0) / 10.0,  # Normalized
            self.relationship_freshness,

            # Binary signals (2)
            1.0 if self.is_dormant else 0.0,
            1.0 if self.is_active else 0.0,

            # Urgency (1)
            self.urgency_boost,
        ]
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


def _parse_email_date(date_str: str) -> Optional[datetime]:
    """Parse email date string into datetime.

    Handles common email date formats from RFC 2822 and variations.
    """
    if not date_str:
        return None

    # Common patterns in email headers
    patterns = [
        # RFC 2822: "Wed, 1 Jan 2020 12:00:00 -0500"
        r'(\d{1,2})\s+(\w{3})\s+(\d{4})\s+(\d{1,2}):(\d{2}):?(\d{2})?\s*([+-]\d{4})?',
        # ISO-like: "2020-01-01 12:00:00"
        r'(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2}):?(\d{2})?',
        # Enron format: "12/1/2000 10:30 AM" or "12/1/2000 10:30:00 AM"
        r'(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?',
    ]

    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    # Try RFC 2822 format first
    match = re.search(patterns[0], date_str, re.IGNORECASE)
    if match:
        day = int(match.group(1))
        month_str = match.group(2).lower()
        year = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))
        second = int(match.group(6)) if match.group(6) else 0

        month = month_map.get(month_str, 1)

        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass

    # Try ISO-like format
    match = re.search(patterns[1], date_str)
    if match:
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4))
            minute = int(match.group(5))
            second = int(match.group(6)) if match.group(6) else 0
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass

    # Try Enron format
    match = re.search(patterns[2], date_str, re.IGNORECASE)
    if match:
        try:
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            hour = int(match.group(4))
            minute = int(match.group(5))
            second = int(match.group(6)) if match.group(6) else 0
            am_pm = match.group(7)

            # Handle AM/PM
            if am_pm:
                if am_pm.upper() == 'PM' and hour != 12:
                    hour += 12
                elif am_pm.upper() == 'AM' and hour == 12:
                    hour = 0

            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass

    return None


def _cyclical_encode(value: float, max_value: float) -> tuple[float, float]:
    """Encode a cyclical value using sin/cos transformation.

    This preserves the cyclical nature (e.g., hour 23 is close to hour 0).

    Args:
        value: The value to encode
        max_value: The period of the cycle

    Returns:
        Tuple of (sin_encoding, cos_encoding)
    """
    angle = 2 * math.pi * value / max_value
    return math.sin(angle), math.cos(angle)


def _is_business_hours(
    hour: int,
    day: int,
    start_hour: int = DEFAULT_BUSINESS_START,
    end_hour: int = DEFAULT_BUSINESS_END,
    business_days: set[int] = DEFAULT_BUSINESS_DAYS,
) -> bool:
    """Check if the given time is within business hours."""
    is_weekday = day in business_days
    is_working_hour = start_hour <= hour < end_hour
    return is_weekday and is_working_hour


def compute_relationship_decay(
    days_since_last_contact: float,
    recent_email_count: int,
    half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
) -> RelationshipDecayFeatures:
    """Compute relationship decay features based on time since last contact.

    Models relationship strength decay using exponential decay with configurable
    half-life. Default half-life is 30 days, meaning relationships lose half
    their "freshness" each month without contact.

    Formula: decayed_strength = recent_emails * exp(-lambda * days)
    where lambda = ln(2) / half_life

    Args:
        days_since_last_contact: Days since last email from this sender
        recent_email_count: Number of emails from sender in observation window
        half_life_days: Decay half-life in days (default: 30)

    Returns:
        RelationshipDecayFeatures with computed decay values

    Example:
        >>> features = compute_relationship_decay(days_since=45, recent_emails=10)
        >>> features.decay_factor  # ~0.35 (more than half-life elapsed)
        >>> features.decayed_strength  # 10 * 0.35 = 3.5
    """
    # Ensure non-negative days
    days = max(0.0, days_since_last_contact)

    # Compute decay factor using exponential decay
    # decay_factor = exp(-lambda * days) where lambda = ln(2) / half_life
    decay_lambda = math.log(2) / half_life_days
    decay_factor = math.exp(-decay_lambda * days)

    # Compute decayed strength
    decayed_strength = recent_email_count * decay_factor

    # Normalize to 0-1 relationship freshness
    # Use sigmoid-like transform: fresh within half-life, stale beyond 2x
    # freshness = 1 / (1 + exp((days - half_life) / (half_life / 3)))
    freshness_midpoint = half_life_days
    freshness_slope = half_life_days / 3.0
    relationship_freshness = 1.0 / (1.0 + math.exp((days - freshness_midpoint) / freshness_slope))

    # Binary signals for relationship state
    is_dormant = days > (2 * half_life_days)  # No contact > 60 days (default)
    is_active = days <= half_life_days  # Contact within 30 days (default)

    # Urgency boost for re-engaging dormant relationships
    # High-volume senders who've gone dormant deserve attention
    if is_dormant and recent_email_count >= 5:
        # Boost proportional to historical volume, capped at 0.5
        urgency_boost = min(0.5, recent_email_count / 20.0)
    elif is_active and recent_email_count >= 10:
        # Active high-volume sender - slight boost
        urgency_boost = min(0.3, recent_email_count / 50.0)
    else:
        urgency_boost = 0.0

    return RelationshipDecayFeatures(
        days_since_last_contact=days,
        recent_email_count=recent_email_count,
        half_life_days=half_life_days,
        decay_factor=decay_factor,
        decayed_strength=decayed_strength,
        relationship_freshness=relationship_freshness,
        is_dormant=is_dormant,
        is_active=is_active,
        urgency_boost=urgency_boost,
    )


def compute_sender_decay_score(
    days_since_last_contact: float,
    emails_7d: int = 0,
    emails_30d: int = 0,
    emails_90d: int = 0,
    half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
) -> float:
    """Compute a single decay-weighted score for a sender relationship.

    Combines time-windowed email counts with decay to produce a 0-1 score.
    This is the primary interface for the ML pipeline.

    Args:
        days_since_last_contact: Days since last email from sender
        emails_7d: Emails from sender in last 7 days
        emails_30d: Emails from sender in last 30 days
        emails_90d: Emails from sender in last 90 days
        half_life_days: Decay half-life (default 30)

    Returns:
        Score from 0 to 1 where higher = stronger/fresher relationship
    """
    # Weighted combination of time windows
    # Recent emails count more
    weighted_count = (
        emails_7d * 3.0 +  # 7-day emails weighted 3x
        emails_30d * 1.0 +  # 30-day emails weighted 1x
        emails_90d * 0.3    # 90-day emails weighted 0.3x
    )

    # Get decay features
    features = compute_relationship_decay(
        days_since_last_contact=days_since_last_contact,
        recent_email_count=int(weighted_count),
        half_life_days=half_life_days,
    )

    # Combine decay factor with freshness for final score
    # decay_factor handles the pure time decay
    # relationship_freshness handles the sigmoid normalization
    raw_score = (
        features.decay_factor * 0.4 +
        features.relationship_freshness * 0.4 +
        min(weighted_count / 30.0, 1.0) * 0.2  # Volume component
    )

    return min(1.0, max(0.0, raw_score))


def extract_temporal_features(
    email: dict,
    *,
    reference_time: Optional[datetime] = None,
    thread_context: Optional[dict] = None,
    business_start: int = DEFAULT_BUSINESS_START,
    business_end: int = DEFAULT_BUSINESS_END,
) -> TemporalFeatures:
    """Extract temporal features from an email.

    Args:
        email: Email dictionary with 'date' field
        reference_time: Time to compute "since receipt" from (default: now)
        thread_context: Optional dict with thread timing info:
            - last_email_timestamp: datetime of previous email in thread
        business_start: Start of business hours (default: 9)
        business_end: End of business hours (default: 18)

    Returns:
        TemporalFeatures with extracted information
    """
    # Parse email timestamp
    date_str = email.get('date', '')
    email_dt = _parse_email_date(date_str)

    # Use current time as fallback
    if email_dt is None:
        email_dt = datetime.now()

    # Reference time for computing "time since"
    if reference_time is None:
        reference_time = datetime.now()

    # Extract hour and day
    hour = email_dt.hour
    day = email_dt.weekday()  # 0=Monday, 6=Sunday

    # Cyclical encodings
    hour_sin, hour_cos = _cyclical_encode(hour, 24.0)
    day_sin, day_cos = _cyclical_encode(day, 7.0)

    # Time since receipt
    time_delta = reference_time - email_dt
    time_since_receipt = max(0.0, time_delta.total_seconds() / 3600.0)

    # Business hours check
    is_business = _is_business_hours(hour, day, business_start, business_end)

    # Weekend check
    is_weekend = day >= 5  # Saturday=5, Sunday=6

    # Thread timing
    time_since_last_in_thread = 0.0
    if thread_context:
        last_in_thread = thread_context.get('last_email_timestamp')
        if last_in_thread and isinstance(last_in_thread, datetime):
            thread_delta = email_dt - last_in_thread
            time_since_last_in_thread = max(0.0, thread_delta.total_seconds() / 3600.0)

    # Compute temporal urgency score
    temporal_urgency = compute_temporal_urgency(
        time_since_receipt,
        is_business,
        is_weekend,
        time_since_last_in_thread,
    )

    return TemporalFeatures(
        hour_of_day=hour,
        day_of_week=day,
        email_timestamp=email_dt,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        day_sin=day_sin,
        day_cos=day_cos,
        time_since_receipt_hours=time_since_receipt,
        is_business_hours=is_business,
        is_weekend=is_weekend,
        time_since_last_in_thread_hours=time_since_last_in_thread,
        temporal_urgency=temporal_urgency,
    )


def compute_temporal_urgency(
    time_since_receipt: float,
    is_business: bool,
    is_weekend: bool,
    time_since_last_in_thread: float,
) -> float:
    """Compute temporal urgency score.

    Factors:
    - Fresh emails are more urgent
    - Business hours emails may need faster response
    - Active threads (recent prior email) are more urgent

    Args:
        time_since_receipt: Hours since email arrived
        is_business: Whether email arrived during business hours
        is_weekend: Whether email arrived on weekend
        time_since_last_in_thread: Hours since last email in thread

    Returns:
        Urgency score from 0 to 1
    """
    score = 0.0

    # Freshness component (0.4 max)
    # Exponential decay - email loses urgency over time
    freshness = math.exp(-time_since_receipt / 24.0)  # Half-life ~24 hours
    score += freshness * 0.4

    # Business hours bonus (0.2 max)
    if is_business:
        score += 0.2
    elif not is_weekend:
        # Non-business hours on weekday - moderate
        score += 0.1

    # Weekend penalty is implicit (no business bonus)

    # Thread activity bonus (0.3 max)
    # Recent thread activity increases urgency
    if time_since_last_in_thread > 0:
        # Thread exists
        thread_freshness = math.exp(-time_since_last_in_thread / 12.0)  # Half-life ~12 hours
        score += thread_freshness * 0.3
    else:
        # New thread or no context - baseline
        score += 0.1

    # Base score (0.1)
    score += 0.1

    return min(1.0, max(0.0, score))


def compute_temporal_score(features: TemporalFeatures) -> float:
    """Compute overall temporal score for RL pipeline.

    This is a simplified score that can be used in priority computation.

    Args:
        features: TemporalFeatures from extract_temporal_features()

    Returns:
        Score from 0 to 1
    """
    return features.temporal_urgency


if __name__ == '__main__':
    # Example usage
    from datetime import timedelta

    print("=" * 60)
    print("TEMPORAL FEATURE EXTRACTION TEST")
    print("=" * 60)

    # Test with various email timestamps
    now = datetime.now()

    test_cases = [
        {
            'name': 'Fresh business hours email',
            'email': {'date': now.strftime('%m/%d/%Y %I:%M %p')},
            'reference_time': now,
        },
        {
            'name': 'Day old email',
            'email': {'date': (now - timedelta(days=1)).strftime('%m/%d/%Y %I:%M %p')},
            'reference_time': now,
        },
        {
            'name': 'Week old email',
            'email': {'date': (now - timedelta(days=7)).strftime('%m/%d/%Y %I:%M %p')},
            'reference_time': now,
        },
        {
            'name': 'Active thread (1 hour gap)',
            'email': {'date': now.strftime('%m/%d/%Y %I:%M %p')},
            'reference_time': now,
            'thread_context': {'last_email_timestamp': now - timedelta(hours=1)},
        },
        {
            'name': 'RFC 2822 format',
            'email': {'date': 'Wed, 15 Jan 2020 14:30:00 -0500'},
            'reference_time': datetime(2020, 1, 15, 15, 0, 0),
        },
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")
        features = extract_temporal_features(
            case['email'],
            reference_time=case.get('reference_time'),
            thread_context=case.get('thread_context'),
        )
        print(f"  Hour: {features.hour_of_day}, Day: {features.day_of_week}")
        print(f"  Cyclical hour: sin={features.hour_sin:.3f}, cos={features.hour_cos:.3f}")
        print(f"  Cyclical day: sin={features.day_sin:.3f}, cos={features.day_cos:.3f}")
        print(f"  Time since receipt: {features.time_since_receipt_hours:.1f}h")
        print(f"  Business hours: {features.is_business_hours}")
        print(f"  Weekend: {features.is_weekend}")
        print(f"  Thread gap: {features.time_since_last_in_thread_hours:.1f}h")
        print(f"  Urgency score: {features.temporal_urgency:.3f}")
        print(f"  Vector dims: {len(features.to_feature_vector())}")

    # Test relationship decay modeling
    print("\n" + "=" * 60)
    print("RELATIONSHIP DECAY MODELING TEST")
    print("=" * 60)

    decay_test_cases = [
        {'name': 'Active relationship (0 days)', 'days': 0, 'emails': 20},
        {'name': 'Recent contact (7 days)', 'days': 7, 'emails': 15},
        {'name': 'At half-life (30 days)', 'days': 30, 'emails': 10},
        {'name': 'Stale relationship (60 days)', 'days': 60, 'emails': 10},
        {'name': 'Dormant high-volume (90 days)', 'days': 90, 'emails': 25},
        {'name': 'New contact (7 days, low vol)', 'days': 7, 'emails': 2},
    ]

    print(f"\nUsing half-life of {DEFAULT_DECAY_HALF_LIFE_DAYS} days")
    print("-" * 60)

    for case in decay_test_cases:
        features = compute_relationship_decay(
            days_since_last_contact=case['days'],
            recent_email_count=case['emails'],
        )
        print(f"\n{case['name']}:")
        print(f"  Days since contact: {features.days_since_last_contact}")
        print(f"  Recent emails: {features.recent_email_count}")
        print(f"  Decay factor: {features.decay_factor:.3f}")
        print(f"  Decayed strength: {features.decayed_strength:.2f}")
        print(f"  Freshness: {features.relationship_freshness:.3f}")
        print(f"  Active: {features.is_active}, Dormant: {features.is_dormant}")
        print(f"  Urgency boost: {features.urgency_boost:.3f}")
        print(f"  Vector dims: {len(features.to_feature_vector())}")

    # Test sender decay score
    print("\n" + "-" * 60)
    print("SENDER DECAY SCORE (combined metric)")
    print("-" * 60)

    score_test_cases = [
        {'name': 'Very active sender', 'days': 1, 'e7': 5, 'e30': 15, 'e90': 30},
        {'name': 'Moderate sender', 'days': 10, 'e7': 1, 'e30': 5, 'e90': 10},
        {'name': 'Infrequent sender', 'days': 45, 'e7': 0, 'e30': 1, 'e90': 3},
        {'name': 'Gone cold', 'days': 90, 'e7': 0, 'e30': 0, 'e90': 5},
    ]

    for case in score_test_cases:
        score = compute_sender_decay_score(
            days_since_last_contact=case['days'],
            emails_7d=case['e7'],
            emails_30d=case['e30'],
            emails_90d=case['e90'],
        )
        print(f"  {case['name']}: score={score:.3f}")

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
