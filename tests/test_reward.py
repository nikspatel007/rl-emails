"""Tests for the reward function module."""

import pytest
from datetime import datetime

from src.reward import (
    RewardConfig,
    RewardComponents,
    RewardShaper,
    compute_accuracy_reward,
    compute_satisfaction_reward,
    compute_timeliness_reward,
    compute_confidence_bonus,
    compute_reward,
    compute_batch_rewards,
    sparse_reward,
    hierarchical_reward,
    ACTION_REWARD_MATRIX,
)
from src.email_action import EmailAction
from src.email_state import (
    EmailState,
    EmailMetadata,
    SenderFeatures,
    ThreadFeatures,
    ContentFeatures,
    TemporalFeatures,
    UserContext,
)


def make_test_state(
    urgency: float = 0.0,
    has_question: bool = False,
    has_deadline: bool = False,
    is_action_request: bool = False,
    priority_score: float = 0.5,
) -> EmailState:
    """Create a test EmailState with customizable features."""
    metadata = EmailMetadata(
        message_id='test-123',
        date='Mon, 01 Jan 2024 10:00:00 -0500',
        sender='sender@example.com',
        to=['recipient@example.com'],
        cc=[],
        bcc=[],
        subject='Test Subject',
        body='Test body content',
        in_reply_to=None,
        references=[],
        folder='inbox',
        user='testuser',
        attachments=[],
        file_path='test/path',
    )

    sender = SenderFeatures(
        email='sender@example.com',
        domain='example.com',
        frequency=0.5,
        importance=0.5,
        reply_rate=0.5,
        org_level=1,
        last_interaction_days=1,
    )

    thread = ThreadFeatures(
        is_reply=False,
        thread_length=1,
        thread_participants=2,
        user_already_replied=False,
        thread_age_hours=0.0,
    )

    content = ContentFeatures(
        has_question=has_question,
        has_deadline=has_deadline,
        has_attachment=False,
        email_length=100,
        urgency_signals=urgency,
        is_automated=False,
        is_meeting_request=False,
        is_action_request=is_action_request,
    )

    temporal = TemporalFeatures(
        hour_of_day=10,
        day_of_week=0,
        time_since_last_email=1.0,
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
    )

    user = UserContext(
        user_email='testuser@example.com',
        user_department='Engineering',
        user_role='Developer',
        user_manager=None,
        frequent_contacts={},
        typical_daily_volume=50,
        current_inbox_size=10,
    )

    state = EmailState(
        email=metadata,
        sender=sender,
        thread=thread,
        content=content,
        temporal=temporal,
        user_context=user,
        people_score=priority_score,
        project_score=priority_score,
        topic_score=priority_score,
        task_score=priority_score,
    )

    return state


class TestRewardConfig:
    """Tests for RewardConfig."""

    def test_default_config(self):
        """Default config weights sum to 1."""
        config = RewardConfig()
        total = config.accuracy_weight + config.satisfaction_weight + config.timeliness_weight
        assert abs(total - 1.0) < 0.01

    def test_invalid_weights_raise(self):
        """Config raises error if weights don't sum to 1."""
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            RewardConfig(accuracy_weight=0.5, satisfaction_weight=0.5, timeliness_weight=0.5)

    def test_custom_weights(self):
        """Custom weights are accepted if they sum to 1."""
        config = RewardConfig(
            accuracy_weight=0.6,
            satisfaction_weight=0.3,
            timeliness_weight=0.1,
        )
        assert config.accuracy_weight == 0.6
        assert config.satisfaction_weight == 0.3
        assert config.timeliness_weight == 0.1


class TestAccuracyReward:
    """Tests for compute_accuracy_reward."""

    def test_exact_match_high_reward(self):
        """Exact action type match gives reward of 1.0."""
        action = EmailAction(action_type='archive')
        reward, match_score = compute_accuracy_reward(action, 'ARCHIVED')
        assert reward == 1.0
        assert match_score == 1.0

    def test_reply_to_archive_negative(self):
        """Predicting reply when archive is correct gives negative reward."""
        action = EmailAction(action_type='reply_now')
        reward, match_score = compute_accuracy_reward(action, 'ARCHIVED')
        assert reward < 0

    def test_delete_important_very_negative(self):
        """Deleting an important email (that should be replied to) is heavily penalized."""
        action = EmailAction(action_type='delete')
        reward, match_score = compute_accuracy_reward(action, 'REPLIED')
        assert reward <= -0.5

    def test_similar_actions_partial_credit(self):
        """Similar actions get partial credit."""
        action = EmailAction(action_type='archive')
        reward, match_score = compute_accuracy_reward(action, 'DELETED')
        assert reward > 0  # archive and delete are similar (passive actions)

    def test_all_action_pairs_defined(self):
        """Verify all action pairs have defined rewards."""
        for pred_type in ACTION_REWARD_MATRIX:
            for gt_type in ACTION_REWARD_MATRIX[pred_type]:
                assert -1.0 <= ACTION_REWARD_MATRIX[pred_type][gt_type] <= 1.0


class TestSatisfactionReward:
    """Tests for compute_satisfaction_reward."""

    def test_priority_alignment_high(self):
        """High priority alignment when action and state priorities match."""
        state = make_test_state(priority_score=0.8)
        action = EmailAction(action_type='reply_now', priority=0.8)
        reward, alignment = compute_satisfaction_reward(action, state, 'REPLIED')
        assert alignment > 0.9  # Close to 1.0

    def test_priority_misalignment_low(self):
        """Low priority alignment when priorities differ significantly."""
        state = make_test_state(priority_score=0.9)
        action = EmailAction(action_type='reply_now', priority=0.1)
        reward, alignment = compute_satisfaction_reward(action, state, 'REPLIED')
        assert alignment < 0.3

    def test_question_reply_bonus(self):
        """Bonus for replying to questions."""
        state = make_test_state(has_question=True)
        action = EmailAction(action_type='reply_now')
        reward1, _ = compute_satisfaction_reward(action, state, 'REPLIED')

        action2 = EmailAction(action_type='archive')
        reward2, _ = compute_satisfaction_reward(action2, state, 'REPLIED')

        assert reward1 > reward2

    def test_deadline_urgency_bonus(self):
        """Bonus for treating deadline emails as urgent."""
        state = make_test_state(has_deadline=True)
        action = EmailAction(action_type='reply_now')
        reward, _ = compute_satisfaction_reward(action, state, 'REPLIED')
        assert reward > 0

    def test_action_request_delete_penalty(self):
        """Penalty for deleting action requests."""
        state = make_test_state(is_action_request=True)
        action = EmailAction(action_type='delete')
        reward, _ = compute_satisfaction_reward(action, state, 'DELETED')
        assert reward < 0


class TestTimelinessReward:
    """Tests for compute_timeliness_reward."""

    def test_urgent_fast_response_positive(self):
        """Fast response to urgent email is rewarded."""
        state = make_test_state(urgency=0.8, has_deadline=True)
        action = EmailAction(action_type='reply_now', suggested_response_time='immediate')
        reward = compute_timeliness_reward(action, state)
        assert reward >= 0

    def test_urgent_slow_response_negative(self):
        """Slow response to urgent email is penalized."""
        state = make_test_state(urgency=0.8, has_deadline=True)
        action = EmailAction(action_type='reply_later', suggested_response_time='this_week')
        reward = compute_timeliness_reward(action, state)
        assert reward < 0

    def test_low_urgency_slow_ok(self):
        """Slow response to non-urgent email is acceptable."""
        state = make_test_state(urgency=0.1)
        action = EmailAction(action_type='reply_later', suggested_response_time='when_possible')
        reward = compute_timeliness_reward(action, state)
        assert reward >= -0.1  # Should not be heavily penalized


class TestConfidenceBonus:
    """Tests for compute_confidence_bonus."""

    def test_high_confidence_correct_bonus(self):
        """High confidence when correct gives bonus."""
        action = EmailAction(action_type='archive', confidence=0.95)
        bonus = compute_confidence_bonus(action, 'ARCHIVED')
        assert bonus > 0

    def test_high_confidence_wrong_penalty(self):
        """High confidence when wrong gives penalty."""
        action = EmailAction(action_type='delete', confidence=0.95)
        bonus = compute_confidence_bonus(action, 'REPLIED')
        assert bonus < 0

    def test_low_confidence_small_effect(self):
        """Low confidence has small effect either way."""
        action = EmailAction(action_type='archive', confidence=0.1)
        bonus_correct = compute_confidence_bonus(action, 'ARCHIVED')
        bonus_wrong = compute_confidence_bonus(action, 'REPLIED')
        assert abs(bonus_correct) < 0.05
        assert abs(bonus_wrong) < 0.05


class TestComputeReward:
    """Tests for the main compute_reward function."""

    def test_perfect_prediction_high_reward(self):
        """Perfect prediction should give high total reward."""
        state = make_test_state(priority_score=0.8, urgency=0.5)
        action = EmailAction(
            action_type='reply_now',
            priority=0.8,
            suggested_response_time='same_day',
            confidence=0.9,
        )
        result = compute_reward(action, state, 'REPLIED')

        assert result.total_reward > 0.5
        assert result.accuracy_reward == 1.0
        assert result.action_match_score == 1.0

    def test_wrong_prediction_low_reward(self):
        """Wrong prediction should give low or negative reward."""
        state = make_test_state()
        action = EmailAction(action_type='delete', confidence=0.9)
        result = compute_reward(action, state, 'REPLIED')

        assert result.total_reward < 0
        assert result.accuracy_reward < 0

    def test_missed_important_penalty(self):
        """Extra penalty for missing important emails."""
        state = make_test_state()
        action = EmailAction(action_type='archive')
        result = compute_reward(action, state, 'REPLIED')

        # Should have missed_important_penalty applied
        assert result.total_reward < result.accuracy_reward  # Penalty reduces total

    def test_false_urgent_penalty(self):
        """Penalty for marking non-urgent as urgent."""
        state = make_test_state()
        action = EmailAction(action_type='reply_now')
        result = compute_reward(action, state, 'ARCHIVED')

        assert result.total_reward < 0

    def test_components_to_dict(self):
        """RewardComponents can be converted to dict."""
        state = make_test_state()
        action = EmailAction(action_type='archive')
        result = compute_reward(action, state, 'ARCHIVED')
        d = result.to_dict()

        assert 'total_reward' in d
        assert 'accuracy_reward' in d
        assert 'satisfaction_reward' in d
        assert 'timeliness_reward' in d


class TestBatchRewards:
    """Tests for compute_batch_rewards."""

    def test_batch_computation(self):
        """Batch computation returns correct length and stats."""
        states = [make_test_state() for _ in range(5)]
        actions = [EmailAction(action_type='archive') for _ in range(5)]
        labels = ['ARCHIVED', 'ARCHIVED', 'REPLIED', 'DELETED', 'ARCHIVED']

        rewards, stats = compute_batch_rewards(actions, states, labels)

        assert len(rewards) == 5
        assert 'mean_reward' in stats
        assert 'action_accuracy' in stats
        assert 0 <= stats['action_accuracy'] <= 1

    def test_batch_mismatched_lengths_raise(self):
        """Batch with mismatched lengths raises error."""
        states = [make_test_state() for _ in range(3)]
        actions = [EmailAction(action_type='archive') for _ in range(2)]
        labels = ['ARCHIVED', 'ARCHIVED', 'ARCHIVED']

        with pytest.raises(ValueError):
            compute_batch_rewards(actions, states, labels)

    def test_empty_batch(self):
        """Empty batch returns zeros."""
        rewards, stats = compute_batch_rewards([], [], [])
        assert rewards == []
        assert stats['mean_reward'] == 0.0


class TestRewardShaper:
    """Tests for RewardShaper curriculum learning."""

    def test_warmup_phase_lenient(self):
        """Warmup phase has higher accuracy weight."""
        shaper = RewardShaper(warmup_steps=1000, curriculum_steps=10000)
        config = shaper.get_config()
        assert config.accuracy_weight > 0.5  # More lenient

    def test_curriculum_progression(self):
        """Shaper progresses through curriculum."""
        shaper = RewardShaper(warmup_steps=100, curriculum_steps=1000)

        config_start = shaper.get_config()
        shaper.step_forward(50)
        config_mid = shaper.get_config()
        shaper.step_forward(500)
        config_late = shaper.get_config()

        # Accuracy weight should decrease over warmup
        assert config_start.accuracy_weight >= config_mid.accuracy_weight

    def test_full_difficulty_after_curriculum(self):
        """After curriculum, uses initial config."""
        initial = RewardConfig()
        shaper = RewardShaper(
            initial_config=initial,
            warmup_steps=10,
            curriculum_steps=10,
        )
        shaper.step_forward(100)  # Past all phases

        config = shaper.get_config()
        assert config.accuracy_weight == initial.accuracy_weight

    def test_shaped_reward_computation(self):
        """Shaped reward uses current config."""
        shaper = RewardShaper(warmup_steps=1000)
        state = make_test_state()
        action = EmailAction(action_type='archive')

        result = shaper.compute_shaped_reward(action, state, 'ARCHIVED')
        assert isinstance(result, RewardComponents)


class TestAlternativeRewards:
    """Tests for alternative reward functions."""

    def test_sparse_reward_exact_match(self):
        """Sparse reward gives 1 for exact match."""
        action = EmailAction(action_type='archive')
        assert sparse_reward(action, 'ARCHIVED') == 1.0

    def test_sparse_reward_mismatch(self):
        """Sparse reward gives 0 for mismatch."""
        action = EmailAction(action_type='delete')
        assert sparse_reward(action, 'REPLIED') == 0.0

    def test_hierarchical_reward_exact(self):
        """Hierarchical reward gives 1 for exact match."""
        action = EmailAction(action_type='archive')
        assert hierarchical_reward(action, 'ARCHIVED') == 1.0

    def test_hierarchical_reward_same_category(self):
        """Hierarchical reward gives 0.5 for same category."""
        action = EmailAction(action_type='archive')
        assert hierarchical_reward(action, 'DELETED') == 0.5  # Both passive

        action2 = EmailAction(action_type='reply_now')
        assert hierarchical_reward(action2, 'KEPT') == 0.5  # Both response

    def test_hierarchical_reward_different_category(self):
        """Hierarchical reward gives 0 for different category."""
        action = EmailAction(action_type='archive')
        assert hierarchical_reward(action, 'REPLIED') == 0.0
