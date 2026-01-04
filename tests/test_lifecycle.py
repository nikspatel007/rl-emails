#!/usr/bin/env python3
"""Tests for topic lifecycle tracking module."""

import pytest
from datetime import datetime, timedelta

from src.features.lifecycle import (
    LifecyclePhase,
    TopicActivityWindow,
    TopicHistory,
    LifecycleFeatures,
    compute_lifecycle_features,
    build_topic_history,
    analyze_topic_lifecycle,
    compute_lifecycle_score,
    _compute_momentum,
    _classify_phase,
    DEFAULT_ACTIVITY_THRESHOLD,
    DEFAULT_WINDOW_DAYS,
)


class TestLifecyclePhase:
    """Tests for LifecyclePhase enum."""

    def test_phase_values(self):
        """Test that all phases have correct string values."""
        assert LifecyclePhase.EMERGING.value == "emerging"
        assert LifecyclePhase.ACTIVE.value == "active"
        assert LifecyclePhase.DECLINING.value == "declining"
        assert LifecyclePhase.DORMANT.value == "dormant"


class TestTopicActivityWindow:
    """Tests for TopicActivityWindow dataclass."""

    def test_daily_rate_calculation(self):
        """Test daily rate is computed correctly."""
        window = TopicActivityWindow(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 8),  # 7 days
            email_count=14,
        )
        assert abs(window.daily_rate - 2.0) < 0.01

    def test_window_days_calculation(self):
        """Test window days property."""
        window = TopicActivityWindow(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 8),
            email_count=10,
        )
        assert abs(window.window_days - 7.0) < 0.01


class TestTopicHistory:
    """Tests for TopicHistory dataclass."""

    def test_add_window_updates_totals(self):
        """Test that adding windows accumulates total_emails correctly."""
        history = TopicHistory(topic_id="test", topic_name="Test Topic")
        # first_seen and last_seen are set externally, not by add_window
        history.first_seen = datetime(2024, 1, 1)
        history.last_seen = datetime(2024, 1, 14)

        window1 = TopicActivityWindow(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 8),
            email_count=5,
        )
        history.add_window(window1)

        assert history.total_emails == 5
        assert len(history.windows) == 1

        window2 = TopicActivityWindow(
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 15),
            email_count=7,
        )
        history.add_window(window2)

        assert history.total_emails == 12
        assert len(history.windows) == 2


class TestMomentumComputation:
    """Tests for _compute_momentum function."""

    def test_increasing_activity_positive_momentum(self):
        """Increasing activity should give positive momentum."""
        windows = [
            TopicActivityWindow(datetime(2024, 1, 1), datetime(2024, 1, 8), 2),
            TopicActivityWindow(datetime(2024, 1, 8), datetime(2024, 1, 15), 4),
            TopicActivityWindow(datetime(2024, 1, 15), datetime(2024, 1, 22), 6),
            TopicActivityWindow(datetime(2024, 1, 22), datetime(2024, 1, 29), 8),
        ]
        momentum = _compute_momentum(windows)
        assert momentum > 0

    def test_decreasing_activity_negative_momentum(self):
        """Decreasing activity should give negative momentum."""
        windows = [
            TopicActivityWindow(datetime(2024, 1, 1), datetime(2024, 1, 8), 10),
            TopicActivityWindow(datetime(2024, 1, 8), datetime(2024, 1, 15), 7),
            TopicActivityWindow(datetime(2024, 1, 15), datetime(2024, 1, 22), 4),
            TopicActivityWindow(datetime(2024, 1, 22), datetime(2024, 1, 29), 2),
        ]
        momentum = _compute_momentum(windows)
        assert momentum < 0

    def test_stable_activity_near_zero_momentum(self):
        """Stable activity should give momentum near zero."""
        windows = [
            TopicActivityWindow(datetime(2024, 1, 1), datetime(2024, 1, 8), 5),
            TopicActivityWindow(datetime(2024, 1, 8), datetime(2024, 1, 15), 5),
            TopicActivityWindow(datetime(2024, 1, 15), datetime(2024, 1, 22), 5),
            TopicActivityWindow(datetime(2024, 1, 22), datetime(2024, 1, 29), 5),
        ]
        momentum = _compute_momentum(windows)
        assert abs(momentum) < 0.1

    def test_single_window_returns_zero(self):
        """Single window should return zero momentum."""
        windows = [
            TopicActivityWindow(datetime(2024, 1, 1), datetime(2024, 1, 8), 5),
        ]
        momentum = _compute_momentum(windows)
        assert momentum == 0.0

    def test_empty_windows_returns_zero(self):
        """Empty window list should return zero momentum."""
        momentum = _compute_momentum([])
        assert momentum == 0.0


class TestPhaseClassification:
    """Tests for _classify_phase function."""

    def test_dormant_with_no_recent_activity(self):
        """Low activity + old last activity = DORMANT."""
        phase, confidence = _classify_phase(
            current_activity=0.0,
            average_activity=2.0,
            momentum=-0.1,
            days_since_first_seen=60.0,
            days_since_last_activity=30.0,
        )
        assert phase == LifecyclePhase.DORMANT

    def test_emerging_new_topic_with_growth(self):
        """New topic with positive momentum = EMERGING."""
        phase, confidence = _classify_phase(
            current_activity=3.0,
            average_activity=2.0,
            momentum=0.5,
            days_since_first_seen=10.0,
            days_since_last_activity=1.0,
        )
        assert phase == LifecyclePhase.EMERGING

    def test_active_high_activity_stable_momentum(self):
        """High activity with stable momentum = ACTIVE."""
        phase, confidence = _classify_phase(
            current_activity=10.0,
            average_activity=8.0,
            momentum=0.1,
            days_since_first_seen=60.0,
            days_since_last_activity=1.0,
        )
        assert phase == LifecyclePhase.ACTIVE

    def test_declining_negative_momentum(self):
        """Negative momentum = DECLINING."""
        phase, confidence = _classify_phase(
            current_activity=3.0,
            average_activity=8.0,
            momentum=-0.5,
            days_since_first_seen=60.0,
            days_since_last_activity=5.0,
        )
        assert phase == LifecyclePhase.DECLINING


class TestBuildTopicHistory:
    """Tests for build_topic_history function."""

    def test_builds_history_from_emails(self):
        """Test building history from email list."""
        base_date = datetime(2024, 1, 1)
        emails = [
            {'date': base_date, 'topics': ['test'], 'from': 'a@test.com'},
            {'date': base_date + timedelta(days=1), 'topics': ['test'], 'from': 'b@test.com'},
            {'date': base_date + timedelta(days=8), 'topics': ['test'], 'from': 'a@test.com'},
        ]

        history = build_topic_history(emails, 'test', 'Test Topic')

        assert history.topic_id == 'test'
        assert history.topic_name == 'Test Topic'
        assert history.total_emails == 3
        assert len(history.windows) >= 1

    def test_filters_by_topic(self):
        """Test that only matching topic emails are included."""
        base_date = datetime(2024, 1, 1)
        emails = [
            {'date': base_date, 'topics': ['test'], 'from': 'a@test.com'},
            {'date': base_date + timedelta(days=1), 'topics': ['other'], 'from': 'b@test.com'},
            {'date': base_date + timedelta(days=2), 'topics': ['test'], 'from': 'c@test.com'},
        ]

        history = build_topic_history(emails, 'test', 'Test Topic')

        assert history.total_emails == 2

    def test_empty_emails_returns_empty_history(self):
        """Empty email list returns empty history."""
        history = build_topic_history([], 'test', 'Test Topic')

        assert history.total_emails == 0
        assert len(history.windows) == 0

    def test_custom_matcher(self):
        """Test using custom topic matcher function."""
        emails = [
            {'date': datetime(2024, 1, 1), 'subject': 'About Alpha project', 'from': 'a@test.com'},
            {'date': datetime(2024, 1, 2), 'subject': 'Beta update', 'from': 'b@test.com'},
            {'date': datetime(2024, 1, 3), 'subject': 'Alpha phase 2', 'from': 'c@test.com'},
        ]

        def subject_matcher(email, topic_id):
            return topic_id.lower() in email.get('subject', '').lower()

        history = build_topic_history(
            emails, 'alpha', 'Alpha Project',
            topic_matcher=subject_matcher
        )

        assert history.total_emails == 2


class TestComputeLifecycleFeatures:
    """Tests for compute_lifecycle_features function."""

    def test_computes_features_from_history(self):
        """Test computing features from populated history."""
        history = TopicHistory(topic_id='test', topic_name='Test')
        history.add_window(TopicActivityWindow(
            datetime(2024, 1, 1), datetime(2024, 1, 8), 5, 3, 2
        ))
        history.add_window(TopicActivityWindow(
            datetime(2024, 1, 8), datetime(2024, 1, 15), 8, 4, 3
        ))

        ref_time = datetime(2024, 1, 16)
        features = compute_lifecycle_features(history, reference_time=ref_time)

        assert features.topic_id == 'test'
        assert features.topic_name == 'Test'
        assert features.current_activity == 8.0
        assert features.average_activity == 6.5
        assert features.peak_activity == 8.0
        assert features.windows_analyzed == 2

    def test_empty_history_returns_dormant(self):
        """Empty history should return dormant phase."""
        history = TopicHistory(topic_id='test', topic_name='Test')

        features = compute_lifecycle_features(history)

        assert features.lifecycle_phase == LifecyclePhase.DORMANT
        assert features.topic_momentum == 0.0
        assert features.current_activity == 0.0


class TestAnalyzeTopicLifecycle:
    """Tests for analyze_topic_lifecycle convenience function."""

    def test_end_to_end_analysis(self):
        """Test full analysis from emails to features."""
        now = datetime.now()
        base_date = now - timedelta(days=30)

        emails = [
            {'date': base_date + timedelta(days=i), 'topics': ['project-x'], 'from': f'{i}@test.com'}
            for i in range(0, 30, 2)  # Every other day for 30 days
        ]

        features = analyze_topic_lifecycle(
            emails,
            'project-x',
            'Project X',
            reference_time=now,
        )

        assert features.topic_id == 'project-x'
        assert features.lifecycle_phase in LifecyclePhase
        assert -1.0 <= features.topic_momentum <= 1.0


class TestLifecycleFeatures:
    """Tests for LifecycleFeatures dataclass."""

    def test_to_feature_vector_dimensions(self):
        """Test feature vector has correct dimensions (10)."""
        features = LifecycleFeatures(
            topic_id='test',
            topic_name='Test',
            lifecycle_phase=LifecyclePhase.ACTIVE,
            topic_momentum=0.5,
            current_activity=10.0,
            average_activity=8.0,
            peak_activity=15.0,
            activity_variance=5.0,
            days_since_first_seen=30.0,
            days_since_last_activity=1.0,
            windows_analyzed=4,
            phase_confidence=0.9,
        )

        vec = features.to_feature_vector()
        assert len(vec) == 10

    def test_feature_vector_phase_encoding(self):
        """Test one-hot encoding of phase in feature vector."""
        for phase in LifecyclePhase:
            features = LifecycleFeatures(
                topic_id='test',
                topic_name='Test',
                lifecycle_phase=phase,
                topic_momentum=0.0,
                current_activity=5.0,
                average_activity=5.0,
                peak_activity=5.0,
                activity_variance=0.0,
                days_since_first_seen=10.0,
                days_since_last_activity=1.0,
                windows_analyzed=2,
                phase_confidence=0.8,
            )

            vec = features.to_feature_vector()
            # First 4 elements are one-hot encoding
            phase_encoding = vec[:4]
            assert sum(phase_encoding) == 1.0  # Exactly one 1.0


class TestComputeLifecycleScore:
    """Tests for compute_lifecycle_score function."""

    def test_emerging_has_high_score(self):
        """Emerging topics should have high scores."""
        features = LifecycleFeatures(
            topic_id='test',
            topic_name='Test',
            lifecycle_phase=LifecyclePhase.EMERGING,
            topic_momentum=0.5,
            current_activity=3.0,
            average_activity=2.0,
            peak_activity=3.0,
            activity_variance=0.5,
            days_since_first_seen=10.0,
            days_since_last_activity=1.0,
            windows_analyzed=2,
            phase_confidence=0.9,
        )

        score = compute_lifecycle_score(features)
        assert score >= 0.7

    def test_dormant_has_low_score(self):
        """Dormant topics should have low scores."""
        features = LifecycleFeatures(
            topic_id='test',
            topic_name='Test',
            lifecycle_phase=LifecyclePhase.DORMANT,
            topic_momentum=-0.1,
            current_activity=0.0,
            average_activity=1.0,
            peak_activity=5.0,
            activity_variance=2.0,
            days_since_first_seen=90.0,
            days_since_last_activity=45.0,
            windows_analyzed=8,
            phase_confidence=0.95,
        )

        score = compute_lifecycle_score(features)
        assert score < 0.3

    def test_active_with_growth_has_good_score(self):
        """Active topic with positive momentum should score well."""
        features = LifecycleFeatures(
            topic_id='test',
            topic_name='Test',
            lifecycle_phase=LifecyclePhase.ACTIVE,
            topic_momentum=0.3,
            current_activity=12.0,
            average_activity=10.0,
            peak_activity=15.0,
            activity_variance=3.0,
            days_since_first_seen=60.0,
            days_since_last_activity=1.0,
            windows_analyzed=8,
            phase_confidence=0.85,
        )

        score = compute_lifecycle_score(features)
        assert 0.7 <= score <= 0.95

    def test_score_bounded_zero_to_one(self):
        """Score should always be between 0 and 1."""
        # Test various edge cases
        test_cases = [
            (LifecyclePhase.EMERGING, 1.0, 100.0),   # Max momentum, high activity
            (LifecyclePhase.DORMANT, -1.0, 0.0),     # Min momentum, no activity
            (LifecyclePhase.ACTIVE, 0.0, 5.0),       # Neutral momentum
        ]

        for phase, momentum, activity in test_cases:
            features = LifecycleFeatures(
                topic_id='test',
                topic_name='Test',
                lifecycle_phase=phase,
                topic_momentum=momentum,
                current_activity=activity,
                average_activity=activity,
                peak_activity=activity,
                activity_variance=0.0,
                days_since_first_seen=30.0,
                days_since_last_activity=1.0,
                windows_analyzed=4,
                phase_confidence=1.0,
            )

            score = compute_lifecycle_score(features)
            assert 0.0 <= score <= 1.0


class TestIntegration:
    """Integration tests for lifecycle tracking."""

    def test_emerging_topic_detection(self):
        """Test detection of emerging topic pattern."""
        now = datetime.now()
        base_date = now - timedelta(days=10)  # New topic, only 10 days old

        # Increasing activity over recent days - starts low, grows
        emails = []
        for i in range(10):
            day = base_date + timedelta(days=i)
            # Starts with 1 email, grows slowly - stays below activity threshold initially
            count = 1 if i < 7 else (i - 5)  # 1,1,1,1,1,1,1,2,3,4
            for j in range(count):
                emails.append({
                    'date': day,
                    'topics': ['new-project'],
                    'from': f'user{j}@test.com'
                })

        features = analyze_topic_lifecycle(
            emails, 'new-project', 'New Project',
            reference_time=now,
            window_days=3,  # Smaller windows for test
        )

        # Topic should be emerging (new with positive momentum) or active if momentum is strong
        assert features.lifecycle_phase in [LifecyclePhase.EMERGING, LifecyclePhase.ACTIVE]
        assert features.topic_momentum > 0

    def test_declining_topic_detection(self):
        """Test detection of declining topic pattern."""
        now = datetime.now()
        base_date = now - timedelta(days=60)

        # Heavy early activity, tapering off
        emails = []
        # First week: heavy activity
        for i in range(7):
            for _ in range(10):
                emails.append({
                    'date': base_date + timedelta(days=i),
                    'topics': ['old-project'],
                    'from': f'user{_}@test.com'
                })

        # Week 3-4: light activity
        for i in range(21, 28):
            emails.append({
                'date': base_date + timedelta(days=i),
                'topics': ['old-project'],
                'from': 'user@test.com'
            })

        # Week 7-8: minimal activity
        emails.append({
            'date': base_date + timedelta(days=50),
            'topics': ['old-project'],
            'from': 'user@test.com'
        })

        features = analyze_topic_lifecycle(
            emails, 'old-project', 'Old Project',
            reference_time=now,
        )

        assert features.topic_momentum < 0
        # Could be DECLINING or DORMANT depending on thresholds
        assert features.lifecycle_phase in [LifecyclePhase.DECLINING, LifecyclePhase.DORMANT]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
