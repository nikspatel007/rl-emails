#!/usr/bin/env python3
"""Topic and project lifecycle tracking.

Tracks topic phases through their lifecycle:
- EMERGING: New topic with increasing activity
- ACTIVE: Sustained high activity
- DECLINING: Decreasing activity
- DORMANT: Very low or no recent activity

Computes:
- topic_momentum: Rate of change in activity (positive = growing, negative = shrinking)
- lifecycle_phase: Current phase based on activity patterns
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


class LifecyclePhase(Enum):
    """Lifecycle phase for a topic or project."""
    EMERGING = "emerging"       # New topic with increasing activity
    ACTIVE = "active"           # Sustained high activity
    DECLINING = "declining"     # Decreasing activity
    DORMANT = "dormant"         # Very low or no recent activity


# Phase transition thresholds
DEFAULT_ACTIVITY_THRESHOLD = 5      # Min emails in window to be "active"
DEFAULT_MOMENTUM_THRESHOLD = 0.2    # Momentum threshold for phase changes
DEFAULT_DORMANT_THRESHOLD = 1       # Max emails in window to be "dormant"
DEFAULT_WINDOW_DAYS = 7             # Days per activity window


@dataclass
class TopicActivityWindow:
    """Activity data for a single time window."""
    start_date: datetime
    end_date: datetime
    email_count: int
    unique_senders: int = 0
    unique_threads: int = 0

    @property
    def window_days(self) -> float:
        """Number of days in this window."""
        delta = self.end_date - self.start_date
        return max(delta.total_seconds() / 86400, 0.001)  # Avoid division by zero

    @property
    def daily_rate(self) -> float:
        """Emails per day in this window."""
        return self.email_count / self.window_days


@dataclass
class TopicHistory:
    """Historical activity data for a topic."""
    topic_id: str
    topic_name: str
    windows: list[TopicActivityWindow] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_emails: int = 0

    def add_window(self, window: TopicActivityWindow) -> None:
        """Add an activity window to the history.

        Note: first_seen and last_seen should be set externally to the actual
        email timestamps, not inferred from window boundaries which may extend
        beyond the actual email dates.
        """
        self.windows.append(window)
        self.total_emails += window.email_count


@dataclass
class LifecycleFeatures:
    """Lifecycle features for a topic or project."""
    # Core metrics
    topic_id: str
    topic_name: str
    lifecycle_phase: LifecyclePhase
    topic_momentum: float           # -1 to +1 rate of change

    # Activity metrics
    current_activity: float         # Emails in most recent window
    average_activity: float         # Average emails per window
    peak_activity: float            # Maximum emails in any window
    activity_variance: float        # Variance in activity across windows

    # Timing metrics
    days_since_first_seen: float    # Age of topic
    days_since_last_activity: float # Recency of activity
    windows_analyzed: int           # Number of time windows in history

    # Phase confidence
    phase_confidence: float         # 0-1 confidence in phase classification

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to feature vector for ML pipeline.

        Returns 10-dimensional vector:
        - Phase encoding (4 dims - one-hot)
        - Momentum (1 dim)
        - Activity metrics (3 dims - current, average, peak normalized)
        - Timing metrics (2 dims - age, recency normalized)
        """
        # One-hot encoding of phase
        phase_encoding = [0.0, 0.0, 0.0, 0.0]
        phase_idx = {
            LifecyclePhase.EMERGING: 0,
            LifecyclePhase.ACTIVE: 1,
            LifecyclePhase.DECLINING: 2,
            LifecyclePhase.DORMANT: 3,
        }
        phase_encoding[phase_idx[self.lifecycle_phase]] = 1.0

        # Normalize activity metrics (cap at 50 emails/window)
        max_activity = 50.0
        current_norm = min(self.current_activity, max_activity) / max_activity
        average_norm = min(self.average_activity, max_activity) / max_activity
        peak_norm = min(self.peak_activity, max_activity) / max_activity

        # Normalize timing (cap at 365 days)
        max_days = 365.0
        age_norm = min(self.days_since_first_seen, max_days) / max_days
        recency_norm = min(self.days_since_last_activity, max_days) / max_days

        values = [
            *phase_encoding,        # 4 dims
            self.topic_momentum,    # 1 dim (already -1 to +1)
            current_norm,           # 1 dim
            average_norm,           # 1 dim
            peak_norm,              # 1 dim
            age_norm,               # 1 dim
            recency_norm,           # 1 dim
        ]

        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


def _compute_momentum(
    windows: list[TopicActivityWindow],
    *,
    recent_weight: float = 2.0,
) -> float:
    """Compute topic momentum from activity windows.

    Uses weighted linear regression on window activity to compute slope.
    More recent windows are weighted higher.

    Args:
        windows: List of activity windows (oldest to newest)
        recent_weight: Weight multiplier for most recent windows

    Returns:
        Momentum score from -1 to +1
    """
    if len(windows) < 2:
        return 0.0

    # Get daily rates for each window
    rates = [w.daily_rate for w in windows]
    n = len(rates)

    # Compute weights (more recent = higher weight)
    weights = []
    for i in range(n):
        # Linear weight from 1 to recent_weight
        w = 1.0 + (recent_weight - 1.0) * (i / (n - 1))
        weights.append(w)

    # Weighted linear regression for slope
    # y = mx + b, we want m (slope)
    x_vals = list(range(n))
    sum_w = sum(weights)
    sum_wx = sum(w * x for w, x in zip(weights, x_vals))
    sum_wy = sum(w * y for w, y in zip(weights, rates))
    sum_wxx = sum(w * x * x for w, x in zip(weights, x_vals))
    sum_wxy = sum(w * x * y for w, x, y in zip(weights, x_vals, rates))

    # Calculate slope
    denom = sum_w * sum_wxx - sum_wx * sum_wx
    if abs(denom) < 1e-10:
        return 0.0

    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom

    # Normalize slope to -1 to +1 range
    # Use average rate as baseline for normalization
    avg_rate = sum(rates) / n if n > 0 else 1.0
    if avg_rate < 0.001:
        avg_rate = 0.001  # Avoid division by very small numbers

    # Slope relative to average activity
    normalized_momentum = slope / avg_rate

    # Clamp to -1 to +1
    return max(-1.0, min(1.0, normalized_momentum))


def _classify_phase(
    current_activity: float,
    average_activity: float,
    momentum: float,
    days_since_first_seen: float,
    days_since_last_activity: float,
    *,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    momentum_threshold: float = DEFAULT_MOMENTUM_THRESHOLD,
    dormant_threshold: float = DEFAULT_DORMANT_THRESHOLD,
) -> tuple[LifecyclePhase, float]:
    """Classify lifecycle phase based on activity and momentum.

    Returns:
        Tuple of (phase, confidence)
    """
    confidence = 0.0

    # Check for dormant first - no recent activity at all
    # A topic is dormant if there's been no activity for 2+ weeks
    if days_since_last_activity > 14:
        confidence = min(1.0, days_since_last_activity / 30.0)
        return LifecyclePhase.DORMANT, confidence

    # Emerging: new topic with positive momentum
    is_new = days_since_first_seen < 30
    is_growing = momentum > momentum_threshold
    if is_new and is_growing:
        confidence = min(1.0, momentum + 0.5)
        return LifecyclePhase.EMERGING, confidence

    # Also emerging if low average but strong growth
    if average_activity < activity_threshold and is_growing:
        confidence = min(1.0, momentum)
        return LifecyclePhase.EMERGING, confidence

    # Active: high activity with stable or positive momentum
    is_active = current_activity >= activity_threshold
    is_stable = momentum >= -momentum_threshold
    if is_active and is_stable:
        # Higher confidence with higher activity and momentum
        activity_confidence = min(1.0, current_activity / (activity_threshold * 2))
        momentum_confidence = min(1.0, (momentum + 1.0) / 2.0)
        confidence = (activity_confidence + momentum_confidence) / 2.0
        return LifecyclePhase.ACTIVE, confidence

    # Declining: negative momentum
    is_declining = momentum < -momentum_threshold
    if is_declining:
        confidence = min(1.0, abs(momentum))
        return LifecyclePhase.DECLINING, confidence

    # Default to dormant if low activity
    if current_activity < dormant_threshold:
        confidence = 0.5
        return LifecyclePhase.DORMANT, confidence

    # Edge case: moderate activity, slight negative momentum
    if current_activity >= dormant_threshold:
        confidence = 0.5
        return LifecyclePhase.ACTIVE, confidence

    return LifecyclePhase.DORMANT, 0.3


def compute_lifecycle_features(
    history: TopicHistory,
    *,
    reference_time: Optional[datetime] = None,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    momentum_threshold: float = DEFAULT_MOMENTUM_THRESHOLD,
    dormant_threshold: float = DEFAULT_DORMANT_THRESHOLD,
) -> LifecycleFeatures:
    """Compute lifecycle features from topic history.

    Args:
        history: TopicHistory with activity windows
        reference_time: Time to compute recency from (default: now)
        activity_threshold: Min emails per window for "active" phase
        momentum_threshold: Momentum threshold for phase transitions
        dormant_threshold: Max emails per window for "dormant" phase

    Returns:
        LifecycleFeatures with computed metrics
    """
    if reference_time is None:
        reference_time = datetime.now()

    windows = history.windows
    n_windows = len(windows)

    # Handle empty history
    if n_windows == 0:
        return LifecycleFeatures(
            topic_id=history.topic_id,
            topic_name=history.topic_name,
            lifecycle_phase=LifecyclePhase.DORMANT,
            topic_momentum=0.0,
            current_activity=0.0,
            average_activity=0.0,
            peak_activity=0.0,
            activity_variance=0.0,
            days_since_first_seen=0.0,
            days_since_last_activity=float('inf'),
            windows_analyzed=0,
            phase_confidence=1.0,
        )

    # Sort windows by date
    sorted_windows = sorted(windows, key=lambda w: w.start_date)

    # Activity metrics
    activities = [w.email_count for w in sorted_windows]
    current_activity = float(activities[-1]) if activities else 0.0
    average_activity = sum(activities) / len(activities) if activities else 0.0
    peak_activity = float(max(activities)) if activities else 0.0

    # Variance
    if len(activities) > 1:
        mean = average_activity
        variance = sum((a - mean) ** 2 for a in activities) / len(activities)
    else:
        variance = 0.0

    # Timing metrics
    if history.first_seen:
        days_since_first = (reference_time - history.first_seen).total_seconds() / 86400
    else:
        days_since_first = 0.0

    if history.last_seen:
        days_since_last = (reference_time - history.last_seen).total_seconds() / 86400
    else:
        days_since_last = float('inf')

    # Compute momentum
    momentum = _compute_momentum(sorted_windows)

    # Classify phase
    phase, confidence = _classify_phase(
        current_activity=current_activity,
        average_activity=average_activity,
        momentum=momentum,
        days_since_first_seen=days_since_first,
        days_since_last_activity=days_since_last,
        activity_threshold=activity_threshold,
        momentum_threshold=momentum_threshold,
        dormant_threshold=dormant_threshold,
    )

    return LifecycleFeatures(
        topic_id=history.topic_id,
        topic_name=history.topic_name,
        lifecycle_phase=phase,
        topic_momentum=momentum,
        current_activity=current_activity,
        average_activity=average_activity,
        peak_activity=peak_activity,
        activity_variance=variance,
        days_since_first_seen=days_since_first,
        days_since_last_activity=days_since_last,
        windows_analyzed=n_windows,
        phase_confidence=confidence,
    )


def build_topic_history(
    emails: list[dict],
    topic_id: str,
    topic_name: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    topic_matcher: Optional[callable] = None,
) -> TopicHistory:
    """Build topic history from a list of emails.

    Args:
        emails: List of email dicts with 'date' and optionally 'topics' fields
        topic_id: ID of the topic to track
        topic_name: Human-readable topic name
        window_days: Number of days per activity window
        topic_matcher: Optional function(email, topic_id) -> bool to match emails

    Returns:
        TopicHistory with computed activity windows
    """
    history = TopicHistory(topic_id=topic_id, topic_name=topic_name)

    # Default matcher checks for topic in email's topics list
    def default_matcher(email: dict, tid: str) -> bool:
        topics = email.get('topics', [])
        if isinstance(topics, str):
            return tid.lower() in topics.lower()
        return tid in topics or tid.lower() in [t.lower() for t in topics]

    matcher = topic_matcher or default_matcher

    # Filter emails matching this topic
    matching_emails = [e for e in emails if matcher(e, topic_id)]

    if not matching_emails:
        return history

    # Parse dates and sort
    dated_emails = []
    for email in matching_emails:
        date_str = email.get('date', '')
        if isinstance(date_str, datetime):
            dated_emails.append((date_str, email))
        elif date_str:
            # Try parsing common formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    dt = datetime.strptime(date_str[:19], fmt)
                    dated_emails.append((dt, email))
                    break
                except ValueError:
                    continue

    if not dated_emails:
        return history

    # Sort by date
    dated_emails.sort(key=lambda x: x[0])

    # Set first/last seen (total_emails is accumulated via add_window)
    history.first_seen = dated_emails[0][0]
    history.last_seen = dated_emails[-1][0]

    # Build windows
    window_delta = timedelta(days=window_days)
    window_start = dated_emails[0][0]
    window_end = window_start + window_delta

    window_emails = []
    for dt, email in dated_emails:
        if dt < window_end:
            window_emails.append(email)
        else:
            # Close current window
            if window_emails:
                senders = set(e.get('from', '') for e in window_emails)
                threads = set(e.get('thread_id', e.get('message_id', '')) for e in window_emails)
                history.add_window(TopicActivityWindow(
                    start_date=window_start,
                    end_date=window_end,
                    email_count=len(window_emails),
                    unique_senders=len(senders),
                    unique_threads=len(threads),
                ))

            # Start new window(s)
            while dt >= window_end:
                window_start = window_end
                window_end = window_start + window_delta

            window_emails = [email]

    # Close final window
    if window_emails:
        senders = set(e.get('from', '') for e in window_emails)
        threads = set(e.get('thread_id', e.get('message_id', '')) for e in window_emails)
        history.add_window(TopicActivityWindow(
            start_date=window_start,
            end_date=window_end,
            email_count=len(window_emails),
            unique_senders=len(senders),
            unique_threads=len(threads),
        ))

    return history


def analyze_topic_lifecycle(
    emails: list[dict],
    topic_id: str,
    topic_name: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    reference_time: Optional[datetime] = None,
    topic_matcher: Optional[callable] = None,
) -> LifecycleFeatures:
    """Analyze lifecycle of a topic from email history.

    Convenience function that builds history and computes features.

    Args:
        emails: List of email dicts
        topic_id: Topic identifier
        topic_name: Human-readable topic name
        window_days: Days per activity window
        reference_time: Time to compute recency from
        topic_matcher: Optional custom email matcher function

    Returns:
        LifecycleFeatures for the topic
    """
    history = build_topic_history(
        emails,
        topic_id,
        topic_name,
        window_days=window_days,
        topic_matcher=topic_matcher,
    )
    return compute_lifecycle_features(history, reference_time=reference_time)


def compute_lifecycle_score(features: LifecycleFeatures) -> float:
    """Compute overall lifecycle importance score.

    Higher scores for active/emerging topics that need attention.

    Args:
        features: LifecycleFeatures from compute_lifecycle_features

    Returns:
        Score from 0 to 1
    """
    # Base score by phase
    phase_scores = {
        LifecyclePhase.EMERGING: 0.8,    # New topics need attention
        LifecyclePhase.ACTIVE: 0.7,      # Active topics are important
        LifecyclePhase.DECLINING: 0.4,   # Declining topics less urgent
        LifecyclePhase.DORMANT: 0.1,     # Dormant topics low priority
    }

    score = phase_scores.get(features.lifecycle_phase, 0.5)

    # Boost for positive momentum
    if features.topic_momentum > 0:
        score += features.topic_momentum * 0.2

    # Boost for high activity
    if features.current_activity > DEFAULT_ACTIVITY_THRESHOLD * 2:
        score += 0.1

    # Penalty for very old dormant topics
    if features.lifecycle_phase == LifecyclePhase.DORMANT:
        if features.days_since_last_activity > 30:
            score -= 0.1

    # Weight by confidence
    score = score * features.phase_confidence + (1 - features.phase_confidence) * 0.5

    return max(0.0, min(1.0, score))


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("TOPIC LIFECYCLE TRACKING TEST")
    print("=" * 60)

    # Create sample email data
    now = datetime.now()

    # Simulate different lifecycle patterns
    test_cases = [
        {
            'name': 'Emerging Topic (Project Alpha)',
            'description': 'New topic, started 21 days ago with increasing activity',
            'emails': [
                # Week 1: light activity (1 email)
                {'date': now - timedelta(days=21), 'topics': ['alpha'], 'from': 'a@test.com'},
                # Week 2: growing (2 emails)
                {'date': now - timedelta(days=14), 'topics': ['alpha'], 'from': 'b@test.com'},
                {'date': now - timedelta(days=13), 'topics': ['alpha'], 'from': 'c@test.com'},
                # Week 3: more growth (4 emails)
                {'date': now - timedelta(days=6), 'topics': ['alpha'], 'from': 'd@test.com'},
                {'date': now - timedelta(days=5), 'topics': ['alpha'], 'from': 'e@test.com'},
                {'date': now - timedelta(days=4), 'topics': ['alpha'], 'from': 'f@test.com'},
                {'date': now - timedelta(days=3), 'topics': ['alpha'], 'from': 'g@test.com'},
            ],
            'topic_id': 'alpha',
        },
        {
            'name': 'Active Topic (Project Beta)',
            'description': 'Steady high activity over 8 weeks',
            'emails': [
                *[{'date': now - timedelta(days=56-i*7+j), 'topics': ['beta'], 'from': f'{chr(97+j)}@test.com'}
                  for i in range(8) for j in range(5)],  # 5 emails per week for 8 weeks
            ],
            'topic_id': 'beta',
        },
        {
            'name': 'Declining Topic (Project Gamma)',
            'description': 'Heavy early activity, tapering off recently',
            'emails': [
                # First 2 weeks: heavy activity (10+ emails)
                *[{'date': now - timedelta(days=60-i), 'topics': ['gamma'], 'from': f'{chr(97+i%5)}@test.com'}
                  for i in range(14)],
                # Week 4-6: moderate (3-4 emails)
                {'date': now - timedelta(days=35), 'topics': ['gamma'], 'from': 'a@test.com'},
                {'date': now - timedelta(days=28), 'topics': ['gamma'], 'from': 'b@test.com'},
                {'date': now - timedelta(days=21), 'topics': ['gamma'], 'from': 'c@test.com'},
                # Recent: very light (1 email)
                {'date': now - timedelta(days=7), 'topics': ['gamma'], 'from': 'a@test.com'},
            ],
            'topic_id': 'gamma',
        },
        {
            'name': 'Dormant Topic (Project Delta)',
            'description': 'No activity in over 30 days',
            'emails': [
                {'date': now - timedelta(days=45), 'topics': ['delta'], 'from': 'a@test.com'},
                {'date': now - timedelta(days=44), 'topics': ['delta'], 'from': 'b@test.com'},
                {'date': now - timedelta(days=43), 'topics': ['delta'], 'from': 'a@test.com'},
            ],
            'topic_id': 'delta',
        },
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  ({case['description']})")
        features = analyze_topic_lifecycle(
            case['emails'],
            case['topic_id'],
            case['name'],
            reference_time=now,
        )

        print(f"  Phase: {features.lifecycle_phase.value}")
        print(f"  Momentum: {features.topic_momentum:.3f}")
        print(f"  Current activity: {features.current_activity:.1f} emails/window")
        print(f"  Average activity: {features.average_activity:.1f} emails/window")
        print(f"  Peak activity: {features.peak_activity:.1f} emails/window")
        print(f"  Days since first seen: {features.days_since_first_seen:.1f}")
        print(f"  Days since last activity: {features.days_since_last_activity:.1f}")
        print(f"  Windows analyzed: {features.windows_analyzed}")
        print(f"  Phase confidence: {features.phase_confidence:.2f}")
        print(f"  Lifecycle score: {compute_lifecycle_score(features):.2f}")
        print(f"  Feature vector dims: {len(features.to_feature_vector())}")

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
