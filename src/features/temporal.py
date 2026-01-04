#!/usr/bin/env python3
"""Temporal feature extraction for email RL system.

Extracts time-based features that capture:
- Hour of day and day of week (cyclical encoding)
- Time since email was received
- Business hours / weekend detection
- Thread timing patterns

User email behavior varies significantly by time - people respond faster
during business hours, may defer weekend emails, etc.
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

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
