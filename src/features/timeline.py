#!/usr/bin/env python3
"""Timeline reconstruction API for email analysis.

Enables point-in-time queries to reconstruct the state of relationships
and topics at any date in the email timeline. Essential for:
- Understanding how relationships evolved over time
- Analyzing topic/project lifecycles
- Building temporal training data for RL models

Key functions:
- get_relationship_at(sender, date): Relationship state at a specific date
- get_topic_state_at(topic, date): Topic/project activity at a specific date
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Union

from .relationship import (
    CommunicationGraph,
    CommunicationStats,
    RelationshipFeatures,
    normalize_email,
    parse_date,
    parse_recipients,
    extract_thread_id,
)
from .temporal import (
    compute_relationship_decay,
    RelationshipDecayFeatures,
    DEFAULT_DECAY_HALF_LIFE_DAYS,
)
from .topic import classify_topic, TopicFeatures

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


@dataclass
class RelationshipSnapshot:
    """Point-in-time snapshot of a relationship with a sender.

    Captures the state of a relationship at a specific date, including
    communication history up to that point and decay-adjusted strength.
    """
    # Target query parameters
    sender: str
    user: str
    target_date: datetime

    # Communication stats at target date
    emails_received: int  # Emails from sender up to target_date
    emails_sent: int  # Emails to sender up to target_date
    total_responses: int  # Response events detected

    # Temporal features
    days_since_last_contact: Optional[float]  # Days from last email to target_date
    last_contact_date: Optional[datetime]
    first_contact_date: Optional[datetime]

    # Decay-adjusted scores
    decay_factor: float  # exp(-lambda * days_since_last_contact)
    relationship_strength: float  # 0-1 score based on volume and recency
    relationship_freshness: float  # 0-1 sigmoid-based freshness score

    # Relationship state
    is_active: bool  # Contact within half-life
    is_dormant: bool  # No contact > 2x half-life

    # Extended features
    avg_response_time_hours: Optional[float]
    inferred_hierarchy: float  # 0.3=report, 0.5=peer, 0.9=manager
    communication_asymmetry: float  # >0 if user initiates more

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numerical vector for ML pipeline.

        Returns 12-dimensional vector.
        """
        values = [
            min(self.emails_received, 100) / 100.0,
            min(self.emails_sent, 100) / 100.0,
            min(self.total_responses, 50) / 50.0,
            min(self.days_since_last_contact or 365, 365) / 365.0,
            self.decay_factor,
            self.relationship_strength,
            self.relationship_freshness,
            1.0 if self.is_active else 0.0,
            1.0 if self.is_dormant else 0.0,
            (self.avg_response_time_hours or 24.0) / 168.0,  # Normalize by week
            self.inferred_hierarchy,
            (self.communication_asymmetry + 1.0) / 2.0,  # Normalize to 0-1
        ]
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


@dataclass
class TopicSnapshot:
    """Point-in-time snapshot of a topic/project's state.

    Captures the activity level and lifecycle phase of a topic at
    a specific date.
    """
    # Target query parameters
    topic: str
    target_date: datetime

    # Activity metrics at target date
    email_count: int  # Emails mentioning topic up to target_date
    email_count_7d: int  # Emails in 7 days before target_date
    email_count_30d: int  # Emails in 30 days before target_date
    email_count_90d: int  # Emails in 90 days before target_date

    # Temporal features
    days_since_first_mention: Optional[float]
    days_since_last_mention: Optional[float]
    first_mention_date: Optional[datetime]
    last_mention_date: Optional[datetime]

    # Activity scores
    activity_level: float  # 0-1 based on recent activity
    decay_factor: float  # Decay based on days since last mention

    # Lifecycle phase
    lifecycle_phase: str  # 'emerging', 'active', 'declining', 'dormant', 'unknown'
    lifecycle_confidence: float  # Confidence in phase classification

    # Participants
    unique_senders: int  # Unique people discussing topic
    unique_threads: int  # Unique email threads

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numerical vector for ML pipeline.

        Returns 10-dimensional vector.
        """
        phase_encoding = {
            'emerging': 0.0,
            'active': 0.25,
            'declining': 0.5,
            'dormant': 0.75,
            'unknown': 1.0,
        }
        values = [
            min(self.email_count, 200) / 200.0,
            min(self.email_count_7d, 20) / 20.0,
            min(self.email_count_30d, 50) / 50.0,
            min(self.email_count_90d, 100) / 100.0,
            min(self.days_since_last_mention or 365, 365) / 365.0,
            self.activity_level,
            self.decay_factor,
            phase_encoding.get(self.lifecycle_phase, 1.0),
            min(self.unique_senders, 20) / 20.0,
            min(self.unique_threads, 50) / 50.0,
        ]
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


@dataclass
class TimelineEmail:
    """Parsed email with extracted temporal metadata."""
    raw: dict
    timestamp: Optional[datetime]
    sender: str
    recipients: list[str]
    subject: str
    thread_id: str
    message_id: str
    in_reply_to: Optional[str]
    body: str


class TimelineReconstructor:
    """Reconstructs email state at any point in time.

    Provides point-in-time queries for relationship and topic state,
    enabling temporal analysis and "what if" scenarios.

    Usage:
        reconstructor = TimelineReconstructor(emails)

        # Get relationship state on Jan 1, 2024
        rel = reconstructor.get_relationship_at(
            sender="alice@example.com",
            user="bob@example.com",
            target_date=datetime(2024, 1, 1)
        )

        # Get topic state
        topic = reconstructor.get_topic_state_at(
            topic="Project Eagle",
            target_date=datetime(2024, 1, 1)
        )
    """

    def __init__(
        self,
        emails: list[dict],
        *,
        half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
    ):
        """Initialize with email dataset.

        Args:
            emails: List of email dictionaries with fields:
                - from: sender email
                - to: recipients
                - cc: CC recipients (optional)
                - date: email date string
                - subject: email subject
                - body/content: email body (optional)
                - message_id: unique message ID (optional)
                - in_reply_to: parent message ID (optional)
            half_life_days: Decay half-life for relationship freshness
        """
        self.half_life_days = half_life_days
        self._decay_lambda = math.log(2) / half_life_days

        # Parse and sort emails by timestamp
        self._emails: list[TimelineEmail] = []
        for email in emails:
            parsed = self._parse_email(email)
            if parsed.timestamp:
                self._emails.append(parsed)

        self._emails.sort(key=lambda e: e.timestamp)

        # Build indexes for efficient querying
        self._emails_by_sender: dict[str, list[TimelineEmail]] = defaultdict(list)
        self._emails_by_recipient: dict[str, list[TimelineEmail]] = defaultdict(list)
        self._emails_by_thread: dict[str, list[TimelineEmail]] = defaultdict(list)

        for email in self._emails:
            self._emails_by_sender[email.sender].append(email)
            for recipient in email.recipients:
                self._emails_by_recipient[recipient].append(email)
            if email.thread_id:
                self._emails_by_thread[email.thread_id].append(email)

    def _parse_email(self, email: dict) -> TimelineEmail:
        """Parse email dictionary into TimelineEmail."""
        sender = normalize_email(email.get('from', ''))
        recipients = parse_recipients(email.get('to', ''))
        cc = parse_recipients(email.get('cc', ''))
        all_recipients = recipients + cc

        timestamp = parse_date(email.get('date', ''))
        subject = email.get('subject', '')
        thread_id = extract_thread_id(subject)
        message_id = email.get('message_id', '')
        in_reply_to = email.get('in_reply_to', '')
        body = email.get('body', email.get('content', ''))

        return TimelineEmail(
            raw=email,
            timestamp=timestamp,
            sender=sender,
            recipients=all_recipients,
            subject=subject,
            thread_id=thread_id,
            message_id=message_id,
            in_reply_to=in_reply_to,
            body=body,
        )

    def _filter_before_date(
        self,
        emails: list[TimelineEmail],
        target_date: datetime,
    ) -> list[TimelineEmail]:
        """Filter emails to only those before target_date."""
        return [e for e in emails if e.timestamp and e.timestamp <= target_date]

    def get_relationship_at(
        self,
        sender: str,
        user: str,
        target_date: datetime,
    ) -> RelationshipSnapshot:
        """Get relationship state with sender at a specific date.

        Reconstructs the relationship by:
        1. Filtering all communications before target_date
        2. Computing communication stats
        3. Applying decay from last contact to target_date

        Args:
            sender: Sender email address to query
            user: User email address (the recipient perspective)
            target_date: Point in time to reconstruct

        Returns:
            RelationshipSnapshot with relationship state at target_date
        """
        sender = normalize_email(sender)
        user = normalize_email(user)

        # Get emails from sender to user (received by user)
        sender_to_user = self._filter_before_date(
            [e for e in self._emails_by_sender.get(sender, [])
             if user in e.recipients],
            target_date
        )

        # Get emails from user to sender (sent by user)
        user_to_sender = self._filter_before_date(
            [e for e in self._emails_by_sender.get(user, [])
             if sender in e.recipients],
            target_date
        )

        emails_received = len(sender_to_user)
        emails_sent = len(user_to_sender)

        # Find first and last contact dates
        all_contacts = sender_to_user + user_to_sender
        if all_contacts:
            all_contacts.sort(key=lambda e: e.timestamp)
            first_contact_date = all_contacts[0].timestamp
            last_contact_date = all_contacts[-1].timestamp
            days_since_last = (target_date - last_contact_date).total_seconds() / 86400.0
            days_since_last = max(0.0, days_since_last)
        else:
            first_contact_date = None
            last_contact_date = None
            days_since_last = None

        # Compute decay factor
        if days_since_last is not None:
            decay_factor = math.exp(-self._decay_lambda * days_since_last)
        else:
            decay_factor = 0.0

        # Detect responses (simplified - count emails from user after sender's email)
        total_responses = 0
        response_times = []
        for recv_email in sender_to_user:
            # Look for a response within 7 days
            for sent_email in user_to_sender:
                if sent_email.timestamp > recv_email.timestamp:
                    delta = (sent_email.timestamp - recv_email.timestamp).total_seconds() / 3600.0
                    if 0 < delta < 168:  # Within a week
                        total_responses += 1
                        response_times.append(delta)
                        break

        avg_response_time = None
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)

        # Compute relationship strength (volume + recency)
        if days_since_last is not None:
            volume_score = min(1.0, (emails_received + emails_sent) ** 0.5 / 10.0)
            relationship_strength = volume_score * decay_factor * 0.6 + decay_factor * 0.4
        else:
            relationship_strength = 0.0

        # Compute freshness using sigmoid
        if days_since_last is not None:
            freshness_midpoint = self.half_life_days
            freshness_slope = self.half_life_days / 3.0
            relationship_freshness = 1.0 / (1.0 + math.exp(
                (days_since_last - freshness_midpoint) / freshness_slope
            ))
        else:
            relationship_freshness = 0.0

        # Determine active/dormant state
        is_active = days_since_last is not None and days_since_last <= self.half_life_days
        is_dormant = days_since_last is not None and days_since_last > (2 * self.half_life_days)

        # Infer hierarchy (simplified)
        inferred_hierarchy = 0.5  # Default to peer
        if avg_response_time is not None:
            # If user responds very fast, sender might be higher priority
            if avg_response_time < 4:  # < 4 hours
                inferred_hierarchy = 0.7
            elif avg_response_time > 24:  # > 24 hours
                inferred_hierarchy = 0.4

        # Communication asymmetry
        if emails_received + emails_sent > 0:
            communication_asymmetry = (emails_sent - emails_received) / (emails_received + emails_sent)
        else:
            communication_asymmetry = 0.0

        return RelationshipSnapshot(
            sender=sender,
            user=user,
            target_date=target_date,
            emails_received=emails_received,
            emails_sent=emails_sent,
            total_responses=total_responses,
            days_since_last_contact=days_since_last,
            last_contact_date=last_contact_date,
            first_contact_date=first_contact_date,
            decay_factor=decay_factor,
            relationship_strength=relationship_strength,
            relationship_freshness=relationship_freshness,
            is_active=is_active,
            is_dormant=is_dormant,
            avg_response_time_hours=avg_response_time,
            inferred_hierarchy=inferred_hierarchy,
            communication_asymmetry=communication_asymmetry,
        )

    def get_topic_state_at(
        self,
        topic: str,
        target_date: datetime,
        *,
        case_sensitive: bool = False,
    ) -> TopicSnapshot:
        """Get topic/project state at a specific date.

        Reconstructs topic activity by:
        1. Finding all emails mentioning the topic before target_date
        2. Computing activity metrics across time windows
        3. Determining lifecycle phase

        Args:
            topic: Topic string to search for (matches subject and body)
            target_date: Point in time to reconstruct
            case_sensitive: Whether topic matching is case-sensitive

        Returns:
            TopicSnapshot with topic state at target_date
        """
        # Filter emails before target date
        filtered_emails = self._filter_before_date(self._emails, target_date)

        # Find emails mentioning the topic
        topic_pattern = re.escape(topic)
        if not case_sensitive:
            pattern = re.compile(topic_pattern, re.IGNORECASE)
        else:
            pattern = re.compile(topic_pattern)

        matching_emails: list[TimelineEmail] = []
        for email in filtered_emails:
            text = f"{email.subject} {email.body}"
            if pattern.search(text):
                matching_emails.append(email)

        email_count = len(matching_emails)

        # Compute time-windowed counts
        window_7d = target_date - timedelta(days=7)
        window_30d = target_date - timedelta(days=30)
        window_90d = target_date - timedelta(days=90)

        email_count_7d = sum(1 for e in matching_emails if e.timestamp >= window_7d)
        email_count_30d = sum(1 for e in matching_emails if e.timestamp >= window_30d)
        email_count_90d = sum(1 for e in matching_emails if e.timestamp >= window_90d)

        # Find first and last mention
        if matching_emails:
            matching_emails.sort(key=lambda e: e.timestamp)
            first_mention = matching_emails[0].timestamp
            last_mention = matching_emails[-1].timestamp
            days_since_first = (target_date - first_mention).total_seconds() / 86400.0
            days_since_last = (target_date - last_mention).total_seconds() / 86400.0
            days_since_last = max(0.0, days_since_last)
        else:
            first_mention = None
            last_mention = None
            days_since_first = None
            days_since_last = None

        # Compute activity level (weighted by recency)
        if email_count > 0:
            activity_level = min(1.0, (
                email_count_7d * 3.0 +
                email_count_30d * 1.0 +
                email_count_90d * 0.3
            ) / 30.0)
        else:
            activity_level = 0.0

        # Compute decay factor
        if days_since_last is not None:
            decay_factor = math.exp(-self._decay_lambda * days_since_last)
        else:
            decay_factor = 0.0

        # Determine lifecycle phase
        lifecycle_phase, lifecycle_confidence = self._determine_lifecycle_phase(
            email_count=email_count,
            email_count_7d=email_count_7d,
            email_count_30d=email_count_30d,
            email_count_90d=email_count_90d,
            days_since_first=days_since_first,
            days_since_last=days_since_last,
        )

        # Count unique participants and threads
        unique_senders = len(set(e.sender for e in matching_emails))
        unique_threads = len(set(e.thread_id for e in matching_emails if e.thread_id))

        return TopicSnapshot(
            topic=topic,
            target_date=target_date,
            email_count=email_count,
            email_count_7d=email_count_7d,
            email_count_30d=email_count_30d,
            email_count_90d=email_count_90d,
            days_since_first_mention=days_since_first,
            days_since_last_mention=days_since_last,
            first_mention_date=first_mention,
            last_mention_date=last_mention,
            activity_level=activity_level,
            decay_factor=decay_factor,
            lifecycle_phase=lifecycle_phase,
            lifecycle_confidence=lifecycle_confidence,
            unique_senders=unique_senders,
            unique_threads=unique_threads,
        )

    def _determine_lifecycle_phase(
        self,
        email_count: int,
        email_count_7d: int,
        email_count_30d: int,
        email_count_90d: int,
        days_since_first: Optional[float],
        days_since_last: Optional[float],
    ) -> tuple[str, float]:
        """Determine topic lifecycle phase based on activity patterns.

        Phases:
        - emerging: New topic, recently started activity
        - active: Consistent ongoing activity
        - declining: Activity decreasing over time
        - dormant: No recent activity
        - unknown: Not enough data

        Returns:
            Tuple of (phase, confidence)
        """
        if email_count == 0:
            return 'unknown', 0.0

        if days_since_first is None or days_since_last is None:
            return 'unknown', 0.0

        # Emerging: topic is new (< 30 days old) and has activity
        if days_since_first < 30 and email_count >= 2:
            return 'emerging', min(1.0, email_count / 5.0)

        # Dormant: no activity in 60+ days
        if days_since_last > 60:
            return 'dormant', min(1.0, days_since_last / 90.0)

        # Compare activity windows to detect trend
        if email_count_90d > 0:
            recent_ratio = email_count_30d / email_count_90d
            very_recent_ratio = email_count_7d / max(email_count_30d, 1)

            # Active: consistent or increasing activity
            if recent_ratio >= 0.4 and email_count_30d >= 3:
                confidence = min(1.0, email_count_30d / 10.0)
                return 'active', confidence

            # Declining: activity tapering off
            if recent_ratio < 0.3 and email_count_90d >= 5:
                confidence = min(1.0, (1.0 - recent_ratio))
                return 'declining', confidence

        # Default to active if there's recent activity
        if email_count_30d >= 1:
            return 'active', 0.5

        return 'declining', 0.5

    def get_all_senders_at(
        self,
        user: str,
        target_date: datetime,
        min_emails: int = 1,
    ) -> list[RelationshipSnapshot]:
        """Get all sender relationships for a user at a specific date.

        Args:
            user: User email address
            target_date: Point in time to reconstruct
            min_emails: Minimum emails to include sender

        Returns:
            List of RelationshipSnapshot for all relevant senders
        """
        user = normalize_email(user)

        # Get all emails received by user before target_date
        all_received = self._filter_before_date(
            self._emails_by_recipient.get(user, []),
            target_date
        )

        # Count emails per sender
        sender_counts: dict[str, int] = defaultdict(int)
        for email in all_received:
            sender_counts[email.sender] += 1

        # Build relationship snapshots for qualifying senders
        results = []
        for sender, count in sender_counts.items():
            if count >= min_emails:
                snapshot = self.get_relationship_at(sender, user, target_date)
                results.append(snapshot)

        # Sort by relationship strength (descending)
        results.sort(key=lambda s: s.relationship_strength, reverse=True)

        return results

    def get_topic_history(
        self,
        topic: str,
        start_date: datetime,
        end_date: datetime,
        interval_days: int = 7,
    ) -> list[TopicSnapshot]:
        """Get topic state evolution over a time period.

        Args:
            topic: Topic string to track
            start_date: Start of analysis period
            end_date: End of analysis period
            interval_days: Days between snapshots

        Returns:
            List of TopicSnapshot at regular intervals
        """
        snapshots = []
        current_date = start_date

        while current_date <= end_date:
            snapshot = self.get_topic_state_at(topic, current_date)
            snapshots.append(snapshot)
            current_date += timedelta(days=interval_days)

        return snapshots

    def get_relationship_history(
        self,
        sender: str,
        user: str,
        start_date: datetime,
        end_date: datetime,
        interval_days: int = 7,
    ) -> list[RelationshipSnapshot]:
        """Get relationship evolution over a time period.

        Args:
            sender: Sender email address
            user: User email address
            start_date: Start of analysis period
            end_date: End of analysis period
            interval_days: Days between snapshots

        Returns:
            List of RelationshipSnapshot at regular intervals
        """
        snapshots = []
        current_date = start_date

        while current_date <= end_date:
            snapshot = self.get_relationship_at(sender, user, current_date)
            snapshots.append(snapshot)
            current_date += timedelta(days=interval_days)

        return snapshots

    @property
    def date_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range of emails in the dataset."""
        if not self._emails:
            return None, None
        return self._emails[0].timestamp, self._emails[-1].timestamp

    @property
    def email_count(self) -> int:
        """Total number of parsed emails."""
        return len(self._emails)


# Convenience functions for simple queries

def get_relationship_at(
    emails: list[dict],
    sender: str,
    user: str,
    target_date: datetime,
    *,
    half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
) -> RelationshipSnapshot:
    """Get relationship state at a specific date.

    Convenience function that creates a TimelineReconstructor internally.
    For multiple queries, use TimelineReconstructor directly.

    Args:
        emails: List of email dictionaries
        sender: Sender email address
        user: User email address
        target_date: Point in time to reconstruct
        half_life_days: Decay half-life

    Returns:
        RelationshipSnapshot with relationship state
    """
    reconstructor = TimelineReconstructor(emails, half_life_days=half_life_days)
    return reconstructor.get_relationship_at(sender, user, target_date)


def get_topic_state_at(
    emails: list[dict],
    topic: str,
    target_date: datetime,
    *,
    half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
    case_sensitive: bool = False,
) -> TopicSnapshot:
    """Get topic state at a specific date.

    Convenience function that creates a TimelineReconstructor internally.
    For multiple queries, use TimelineReconstructor directly.

    Args:
        emails: List of email dictionaries
        topic: Topic string to search for
        target_date: Point in time to reconstruct
        half_life_days: Decay half-life
        case_sensitive: Whether topic matching is case-sensitive

    Returns:
        TopicSnapshot with topic state
    """
    reconstructor = TimelineReconstructor(emails, half_life_days=half_life_days)
    return reconstructor.get_topic_state_at(topic, target_date, case_sensitive=case_sensitive)


if __name__ == '__main__':
    # Example usage and tests
    from datetime import datetime, timedelta

    print("=" * 60)
    print("TIMELINE RECONSTRUCTION API TEST")
    print("=" * 60)

    # Create sample email dataset
    base_date = datetime(2024, 1, 1)

    sample_emails = [
        # Project Eagle emails
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Project Eagle kickoff',
            'body': 'Starting Project Eagle today. Excited!',
            'date': (base_date - timedelta(days=90)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'bob@example.com',
            'to': 'alice@example.com',
            'subject': 'Re: Project Eagle kickoff',
            'body': 'Great! Looking forward to it.',
            'date': (base_date - timedelta(days=89)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Project Eagle update',
            'body': 'Phase 1 complete. Moving to phase 2.',
            'date': (base_date - timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Project Eagle phase 2',
            'body': 'Need your review on the design doc.',
            'date': (base_date - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'bob@example.com',
            'to': 'alice@example.com',
            'subject': 'Re: Project Eagle phase 2',
            'body': 'Reviewed. Looks good!',
            'date': (base_date - timedelta(days=29)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        # Carol emails (less frequent)
        {
            'from': 'carol@example.com',
            'to': 'bob@example.com',
            'subject': 'Quick question',
            'body': 'Can you help with budget review?',
            'date': (base_date - timedelta(days=45)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        # Recent email
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Project Eagle final review',
            'body': 'Final presentation next week.',
            'date': (base_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S'),
        },
    ]

    # Create reconstructor
    reconstructor = TimelineReconstructor(sample_emails)

    print(f"\nLoaded {reconstructor.email_count} emails")
    print(f"Date range: {reconstructor.date_range}")

    # Test relationship queries
    print("\n" + "-" * 60)
    print("RELATIONSHIP QUERIES")
    print("-" * 60)

    # Query at different points in time
    test_dates = [
        base_date - timedelta(days=60),  # After phase 1
        base_date - timedelta(days=30),  # During phase 2
        base_date,  # Now
    ]

    for target in test_dates:
        rel = reconstructor.get_relationship_at('alice@example.com', 'bob@example.com', target)
        print(f"\nAlice <-> Bob relationship at {target.date()}:")
        print(f"  Emails received: {rel.emails_received}")
        print(f"  Emails sent: {rel.emails_sent}")
        print(f"  Days since last contact: {rel.days_since_last_contact:.1f}" if rel.days_since_last_contact else "  Days since last contact: N/A")
        print(f"  Decay factor: {rel.decay_factor:.3f}")
        print(f"  Relationship strength: {rel.relationship_strength:.3f}")
        print(f"  Is active: {rel.is_active}")

    # Test topic queries
    print("\n" + "-" * 60)
    print("TOPIC QUERIES")
    print("-" * 60)

    for target in test_dates:
        topic = reconstructor.get_topic_state_at('Project Eagle', target)
        print(f"\n'Project Eagle' state at {target.date()}:")
        print(f"  Total emails: {topic.email_count}")
        print(f"  Last 30 days: {topic.email_count_30d}")
        print(f"  Activity level: {topic.activity_level:.3f}")
        print(f"  Lifecycle phase: {topic.lifecycle_phase}")
        print(f"  Unique senders: {topic.unique_senders}")

    # Test history tracking
    print("\n" + "-" * 60)
    print("TOPIC HISTORY")
    print("-" * 60)

    history = reconstructor.get_topic_history(
        'Project Eagle',
        start_date=base_date - timedelta(days=90),
        end_date=base_date,
        interval_days=30,
    )

    for snapshot in history:
        print(f"  {snapshot.target_date.date()}: "
              f"emails={snapshot.email_count}, "
              f"phase={snapshot.lifecycle_phase}, "
              f"activity={snapshot.activity_level:.2f}")

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
