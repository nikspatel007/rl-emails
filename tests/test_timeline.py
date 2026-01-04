#!/usr/bin/env python3
"""Tests for timeline reconstruction API."""

import pytest
from datetime import datetime, timedelta

from src.features.timeline import (
    TimelineReconstructor,
    RelationshipSnapshot,
    TopicSnapshot,
    get_relationship_at,
    get_topic_state_at,
)


@pytest.fixture
def sample_emails():
    """Sample email dataset for testing."""
    base_date = datetime(2024, 1, 1)
    return [
        # Regular communication between alice and bob
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Project Eagle kickoff',
            'body': 'Starting Project Eagle today.',
            'date': (base_date - timedelta(days=90)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'bob@example.com',
            'to': 'alice@example.com',
            'subject': 'Re: Project Eagle kickoff',
            'body': 'Looking forward to it.',
            'date': (base_date - timedelta(days=89, hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Project Eagle update',
            'body': 'Phase 1 complete.',
            'date': (base_date - timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Project Eagle review',
            'body': 'Need your review.',
            'date': (base_date - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'bob@example.com',
            'to': 'alice@example.com',
            'subject': 'Re: Project Eagle review',
            'body': 'Reviewed.',
            'date': (base_date - timedelta(days=29)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Final Project Eagle',
            'body': 'Final presentation.',
            'date': (base_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S'),
        },
        # Different topic
        {
            'from': 'carol@example.com',
            'to': 'bob@example.com',
            'subject': 'Budget review',
            'body': 'Please review the budget.',
            'date': (base_date - timedelta(days=45)).strftime('%Y-%m-%d %H:%M:%S'),
        },
    ]


@pytest.fixture
def base_date():
    return datetime(2024, 1, 1)


class TestTimelineReconstructor:
    """Tests for TimelineReconstructor class."""

    def test_init_parses_emails(self, sample_emails):
        """Test that emails are parsed and sorted correctly."""
        reconstructor = TimelineReconstructor(sample_emails)
        assert reconstructor.email_count == len(sample_emails)

    def test_date_range(self, sample_emails, base_date):
        """Test date range property."""
        reconstructor = TimelineReconstructor(sample_emails)
        start, end = reconstructor.date_range
        assert start is not None
        assert end is not None
        assert start < end

    def test_empty_emails(self):
        """Test with empty email list."""
        reconstructor = TimelineReconstructor([])
        assert reconstructor.email_count == 0
        assert reconstructor.date_range == (None, None)


class TestRelationshipAt:
    """Tests for get_relationship_at function."""

    def test_relationship_exists(self, sample_emails, base_date):
        """Test querying an existing relationship."""
        reconstructor = TimelineReconstructor(sample_emails)
        rel = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            base_date
        )

        assert isinstance(rel, RelationshipSnapshot)
        assert rel.sender == 'alice@example.com'
        assert rel.user == 'bob@example.com'
        assert rel.emails_received > 0
        assert rel.relationship_strength > 0

    def test_relationship_at_earlier_date(self, sample_emails, base_date):
        """Test that earlier dates have fewer emails."""
        reconstructor = TimelineReconstructor(sample_emails)

        # Query at day 60 (should see fewer emails)
        early = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            base_date - timedelta(days=60)
        )

        # Query at current date (should see more)
        late = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            base_date
        )

        assert early.emails_received < late.emails_received

    def test_decay_factor(self, sample_emails, base_date):
        """Test that decay factor decreases with time since contact."""
        reconstructor = TimelineReconstructor(sample_emails)

        # Query right after last contact
        fresh = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            base_date - timedelta(days=6)  # 1 day after last email
        )

        # Query 30 days after last contact
        stale = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            base_date + timedelta(days=23)  # 30 days after last email
        )

        assert fresh.decay_factor > stale.decay_factor

    def test_no_relationship(self, sample_emails, base_date):
        """Test querying a non-existent relationship."""
        reconstructor = TimelineReconstructor(sample_emails)
        rel = reconstructor.get_relationship_at(
            'unknown@example.com',
            'bob@example.com',
            base_date
        )

        assert rel.emails_received == 0
        assert rel.emails_sent == 0
        assert rel.relationship_strength == 0.0

    def test_active_relationship(self, sample_emails, base_date):
        """Test is_active flag for recent contact."""
        reconstructor = TimelineReconstructor(sample_emails, half_life_days=30)

        # Query within half-life of last contact
        rel = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            base_date - timedelta(days=5)  # 2 days after last email
        )

        assert rel.is_active is True
        assert rel.is_dormant is False

    def test_dormant_relationship(self, sample_emails, base_date):
        """Test is_dormant flag for old contact."""
        reconstructor = TimelineReconstructor(sample_emails, half_life_days=30)

        # Query 65 days after last contact (> 2x half-life)
        rel = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            base_date + timedelta(days=58)
        )

        assert rel.is_active is False
        assert rel.is_dormant is True

    def test_feature_vector(self, sample_emails, base_date):
        """Test to_feature_vector method."""
        reconstructor = TimelineReconstructor(sample_emails)
        rel = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            base_date
        )

        vec = rel.to_feature_vector()
        assert len(vec) == 12  # Expected dimension
        assert all(isinstance(v, float) for v in vec)


class TestTopicStateAt:
    """Tests for get_topic_state_at function."""

    def test_topic_exists(self, sample_emails, base_date):
        """Test querying an existing topic."""
        reconstructor = TimelineReconstructor(sample_emails)
        topic = reconstructor.get_topic_state_at('Project Eagle', base_date)

        assert isinstance(topic, TopicSnapshot)
        assert topic.topic == 'Project Eagle'
        assert topic.email_count > 0

    def test_topic_at_earlier_date(self, sample_emails, base_date):
        """Test that earlier dates have fewer topic emails."""
        reconstructor = TimelineReconstructor(sample_emails)

        # Query at day 60
        early = reconstructor.get_topic_state_at(
            'Project Eagle',
            base_date - timedelta(days=60)
        )

        # Query at current date
        late = reconstructor.get_topic_state_at(
            'Project Eagle',
            base_date
        )

        assert early.email_count <= late.email_count

    def test_topic_time_windows(self, sample_emails, base_date):
        """Test email counts for different time windows."""
        reconstructor = TimelineReconstructor(sample_emails)
        topic = reconstructor.get_topic_state_at('Project Eagle', base_date)

        # 7-day count should be <= 30-day count <= 90-day count <= total
        assert topic.email_count_7d <= topic.email_count_30d
        assert topic.email_count_30d <= topic.email_count_90d
        assert topic.email_count_90d <= topic.email_count

    def test_topic_not_found(self, sample_emails, base_date):
        """Test querying a non-existent topic."""
        reconstructor = TimelineReconstructor(sample_emails)
        topic = reconstructor.get_topic_state_at('Unknown Topic', base_date)

        assert topic.email_count == 0
        assert topic.lifecycle_phase == 'unknown'

    def test_case_insensitive_topic(self, sample_emails, base_date):
        """Test case-insensitive topic matching."""
        reconstructor = TimelineReconstructor(sample_emails)

        lower = reconstructor.get_topic_state_at('project eagle', base_date)
        upper = reconstructor.get_topic_state_at('PROJECT EAGLE', base_date)

        assert lower.email_count == upper.email_count

    def test_case_sensitive_topic(self, sample_emails, base_date):
        """Test case-sensitive topic matching."""
        reconstructor = TimelineReconstructor(sample_emails)

        exact = reconstructor.get_topic_state_at(
            'Project Eagle',
            base_date,
            case_sensitive=True
        )
        wrong_case = reconstructor.get_topic_state_at(
            'project eagle',
            base_date,
            case_sensitive=True
        )

        # Should find exact case
        assert exact.email_count > 0
        # Might not find wrong case (depending on actual data)
        assert wrong_case.email_count <= exact.email_count

    def test_lifecycle_phase_emerging(self, base_date):
        """Test emerging lifecycle phase detection."""
        # New topic with recent emails only
        emails = [
            {
                'from': 'alice@example.com',
                'to': 'bob@example.com',
                'subject': 'New Topic discussion',
                'body': 'Starting new initiative.',
                'date': (base_date - timedelta(days=15)).strftime('%Y-%m-%d %H:%M:%S'),
            },
            {
                'from': 'bob@example.com',
                'to': 'alice@example.com',
                'subject': 'Re: New Topic discussion',
                'body': 'Sounds good.',
                'date': (base_date - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S'),
            },
        ]
        reconstructor = TimelineReconstructor(emails)
        topic = reconstructor.get_topic_state_at('New Topic', base_date)

        assert topic.lifecycle_phase == 'emerging'

    def test_lifecycle_phase_dormant(self, base_date):
        """Test dormant lifecycle phase detection."""
        # Old topic with no recent activity
        emails = [
            {
                'from': 'alice@example.com',
                'to': 'bob@example.com',
                'subject': 'Old Project discussion',
                'body': 'Working on old project.',
                'date': (base_date - timedelta(days=120)).strftime('%Y-%m-%d %H:%M:%S'),
            },
            {
                'from': 'bob@example.com',
                'to': 'alice@example.com',
                'subject': 'Re: Old Project discussion',
                'body': 'Done.',
                'date': (base_date - timedelta(days=100)).strftime('%Y-%m-%d %H:%M:%S'),
            },
        ]
        reconstructor = TimelineReconstructor(emails)
        topic = reconstructor.get_topic_state_at('Old Project', base_date)

        assert topic.lifecycle_phase == 'dormant'

    def test_feature_vector(self, sample_emails, base_date):
        """Test to_feature_vector method."""
        reconstructor = TimelineReconstructor(sample_emails)
        topic = reconstructor.get_topic_state_at('Project Eagle', base_date)

        vec = topic.to_feature_vector()
        assert len(vec) == 10  # Expected dimension
        assert all(isinstance(v, float) for v in vec)


class TestGetAllSendersAt:
    """Tests for get_all_senders_at method."""

    def test_gets_all_senders(self, sample_emails, base_date):
        """Test retrieving all sender relationships."""
        reconstructor = TimelineReconstructor(sample_emails)
        senders = reconstructor.get_all_senders_at('bob@example.com', base_date)

        assert len(senders) > 0
        assert all(isinstance(s, RelationshipSnapshot) for s in senders)

    def test_min_emails_filter(self, sample_emails, base_date):
        """Test min_emails filter."""
        reconstructor = TimelineReconstructor(sample_emails)

        # With min_emails=1, should get alice and carol
        all_senders = reconstructor.get_all_senders_at(
            'bob@example.com',
            base_date,
            min_emails=1
        )

        # With higher threshold, should get fewer
        frequent = reconstructor.get_all_senders_at(
            'bob@example.com',
            base_date,
            min_emails=3
        )

        assert len(frequent) <= len(all_senders)


class TestHistoryMethods:
    """Tests for history tracking methods."""

    def test_topic_history(self, sample_emails, base_date):
        """Test get_topic_history method."""
        reconstructor = TimelineReconstructor(sample_emails)
        history = reconstructor.get_topic_history(
            'Project Eagle',
            start_date=base_date - timedelta(days=90),
            end_date=base_date,
            interval_days=30,
        )

        assert len(history) > 0
        assert all(isinstance(s, TopicSnapshot) for s in history)

        # Email count should increase over time
        for i in range(1, len(history)):
            assert history[i].email_count >= history[i - 1].email_count

    def test_relationship_history(self, sample_emails, base_date):
        """Test get_relationship_history method."""
        reconstructor = TimelineReconstructor(sample_emails)
        history = reconstructor.get_relationship_history(
            'alice@example.com',
            'bob@example.com',
            start_date=base_date - timedelta(days=90),
            end_date=base_date,
            interval_days=30,
        )

        assert len(history) > 0
        assert all(isinstance(s, RelationshipSnapshot) for s in history)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_relationship_at_function(self, sample_emails, base_date):
        """Test standalone get_relationship_at function."""
        rel = get_relationship_at(
            sample_emails,
            'alice@example.com',
            'bob@example.com',
            base_date
        )

        assert isinstance(rel, RelationshipSnapshot)
        assert rel.emails_received > 0

    def test_get_topic_state_at_function(self, sample_emails, base_date):
        """Test standalone get_topic_state_at function."""
        topic = get_topic_state_at(
            sample_emails,
            'Project Eagle',
            base_date
        )

        assert isinstance(topic, TopicSnapshot)
        assert topic.email_count > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_email_without_timestamp(self, base_date):
        """Test handling emails without valid timestamps."""
        emails = [
            {
                'from': 'alice@example.com',
                'to': 'bob@example.com',
                'subject': 'Test',
                'date': 'invalid date',
            },
            {
                'from': 'alice@example.com',
                'to': 'bob@example.com',
                'subject': 'Test 2',
                'date': base_date.strftime('%Y-%m-%d %H:%M:%S'),
            },
        ]
        reconstructor = TimelineReconstructor(emails)

        # Should only count email with valid timestamp
        assert reconstructor.email_count == 1

    def test_email_without_body(self, base_date):
        """Test handling emails without body."""
        emails = [
            {
                'from': 'alice@example.com',
                'to': 'bob@example.com',
                'subject': 'Test Topic',
                'date': base_date.strftime('%Y-%m-%d %H:%M:%S'),
            },
        ]
        reconstructor = TimelineReconstructor(emails)
        topic = reconstructor.get_topic_state_at('Test Topic', base_date)

        assert topic.email_count == 1

    def test_target_date_before_all_emails(self, sample_emails, base_date):
        """Test querying before any emails exist."""
        reconstructor = TimelineReconstructor(sample_emails)

        # Query before any emails
        early = base_date - timedelta(days=365)
        rel = reconstructor.get_relationship_at(
            'alice@example.com',
            'bob@example.com',
            early
        )

        assert rel.emails_received == 0
        assert rel.relationship_strength == 0.0

    def test_normalized_email_addresses(self, base_date):
        """Test that email addresses are normalized."""
        emails = [
            {
                'from': 'Alice Smith <ALICE@example.com>',
                'to': 'Bob <bob@example.com>',
                'subject': 'Test',
                'date': base_date.strftime('%Y-%m-%d %H:%M:%S'),
            },
        ]
        reconstructor = TimelineReconstructor(emails)

        # Should match normalized addresses
        rel = reconstructor.get_relationship_at(
            'alice@example.com',  # lowercase, no name
            'bob@example.com',
            base_date
        )

        assert rel.emails_received == 1
