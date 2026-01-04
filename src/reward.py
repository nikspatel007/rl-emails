"""Multi-signal reward function for email RL agent.

This module implements a reward function that combines multiple signals:
- Accuracy: How well the predicted action matches the ground truth
- User satisfaction: Priority alignment and response appropriateness
- Response time: Timeliness of response relative to urgency

The reward function is designed for training an RL agent to handle emails
effectively while balancing multiple objectives.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .email_action import (
    EmailAction,
    ActionType,
    ResponseTime,
    compute_action_similarity,
    response_time_to_hours,
    LABEL_TO_ACTION_TYPE,
)
from .email_state import EmailState

if TYPE_CHECKING:
    import numpy as np

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


# Action type reward matrix: reward[predicted][ground_truth]
# Rows are predicted, columns are ground truth
# Higher values for correct or acceptable predictions
ACTION_REWARD_MATRIX: dict[ActionType, dict[ActionType, float]] = {
    'reply_now': {
        'reply_now': 1.0,
        'reply_later': 0.6,  # Urgency mismatch but right intent
        'forward': 0.1,
        'archive': -0.3,  # Missed that no action needed
        'delete': -0.5,  # Should have deleted
        'create_task': 0.4,  # Both involve work
    },
    'reply_later': {
        'reply_now': 0.4,  # Missed urgency
        'reply_later': 1.0,
        'forward': 0.2,
        'archive': -0.1,  # Minor miss
        'delete': -0.3,
        'create_task': 0.5,  # Similar deferral
    },
    'forward': {
        'reply_now': -0.2,
        'reply_later': 0.0,
        'forward': 1.0,
        'archive': 0.2,  # Both non-reply
        'delete': 0.1,
        'create_task': 0.3,
    },
    'archive': {
        'reply_now': -0.5,  # Missed important email
        'reply_later': -0.3,
        'forward': 0.2,
        'archive': 1.0,
        'delete': 0.7,  # Both are "dismiss" actions
        'create_task': -0.1,
    },
    'delete': {
        'reply_now': -0.8,  # Deleted important email!
        'reply_later': -0.6,
        'forward': -0.2,
        'archive': 0.5,  # Both are "dismiss" actions
        'delete': 1.0,
        'create_task': -0.4,
    },
    'create_task': {
        'reply_now': 0.2,
        'reply_later': 0.5,
        'forward': 0.3,
        'archive': -0.2,
        'delete': -0.4,
        'create_task': 1.0,
    },
}


@dataclass
class RewardComponents:
    """Breakdown of reward into component signals.

    Useful for debugging and understanding agent behavior.
    """
    accuracy_reward: float
    satisfaction_reward: float
    timeliness_reward: float
    total_reward: float

    # Detailed breakdown
    action_match_score: float
    priority_alignment: float
    response_time_score: float
    confidence_bonus: float

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'accuracy_reward': self.accuracy_reward,
            'satisfaction_reward': self.satisfaction_reward,
            'timeliness_reward': self.timeliness_reward,
            'total_reward': self.total_reward,
            'action_match_score': self.action_match_score,
            'priority_alignment': self.priority_alignment,
            'response_time_score': self.response_time_score,
            'confidence_bonus': self.confidence_bonus,
        }


@dataclass
class RewardConfig:
    """Configuration for reward function weights and parameters.

    Attributes:
        accuracy_weight: Weight for accuracy component (0-1)
        satisfaction_weight: Weight for satisfaction component (0-1)
        timeliness_weight: Weight for timeliness component (0-1)
        confidence_bonus_scale: Scale factor for confidence bonus
        urgency_penalty_scale: Scale factor for urgency timing penalty
        missed_important_penalty: Extra penalty for missing important emails
        false_urgent_penalty: Penalty for marking non-urgent as urgent
    """
    accuracy_weight: float = 0.5
    satisfaction_weight: float = 0.3
    timeliness_weight: float = 0.2
    confidence_bonus_scale: float = 0.1
    urgency_penalty_scale: float = 0.5
    missed_important_penalty: float = 0.3
    false_urgent_penalty: float = 0.2

    def __post_init__(self) -> None:
        """Validate weights sum to 1."""
        total = self.accuracy_weight + self.satisfaction_weight + self.timeliness_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total}"
            )


def compute_accuracy_reward(
    predicted: EmailAction,
    ground_truth_label: str,
) -> tuple[float, float]:
    """Compute accuracy reward based on action match.

    Args:
        predicted: Predicted EmailAction from the agent
        ground_truth_label: Ground truth action label (e.g., 'REPLIED', 'ARCHIVED')

    Returns:
        Tuple of (accuracy_reward, action_match_score)
        - accuracy_reward: Weighted accuracy score (-1 to 1)
        - action_match_score: Raw match score (0 to 1)
    """
    # Convert ground truth label to action type
    gt_action_type = LABEL_TO_ACTION_TYPE.get(ground_truth_label, 'archive')

    # Get reward from matrix
    pred_type = predicted.action_type
    if pred_type in ACTION_REWARD_MATRIX and gt_action_type in ACTION_REWARD_MATRIX[pred_type]:
        reward = ACTION_REWARD_MATRIX[pred_type][gt_action_type]
    else:
        # Fallback to similarity-based reward
        gt_action = EmailAction(action_type=gt_action_type)
        reward = compute_action_similarity(predicted, gt_action)

    # Compute match score (0-1 for metrics)
    match_score = 1.0 if pred_type == gt_action_type else max(0.0, reward)

    return reward, match_score


def compute_satisfaction_reward(
    predicted: EmailAction,
    state: EmailState,
    ground_truth_label: str,
) -> tuple[float, float]:
    """Compute user satisfaction reward.

    User satisfaction is based on:
    - Priority alignment: Does the action priority match email importance?
    - Response appropriateness: Is the action appropriate for the email type?
    - Confidence calibration: Is the agent confident when it should be?

    Args:
        predicted: Predicted EmailAction
        state: EmailState with context features
        ground_truth_label: Ground truth action label

    Returns:
        Tuple of (satisfaction_reward, priority_alignment)
    """
    reward = 0.0

    # Priority alignment: Does action priority match state priority?
    state_priority = state.priority_score
    action_priority = predicted.priority
    priority_diff = abs(state_priority - action_priority)
    priority_alignment = 1.0 - priority_diff
    reward += priority_alignment * 0.5

    # Response appropriateness based on content features
    content = state.content

    # Reward for handling questions appropriately
    if content.has_question:
        if predicted.action_type in ('reply_now', 'reply_later'):
            reward += 0.2
        elif predicted.action_type in ('archive', 'delete'):
            reward -= 0.2

    # Reward for handling deadlines appropriately
    if content.has_deadline:
        if predicted.requires_immediate_attention:
            reward += 0.2
        elif predicted.action_type in ('archive', 'delete'):
            reward -= 0.3

    # Penalty for deleting action requests
    if content.is_action_request and predicted.action_type == 'delete':
        reward -= 0.4

    # Bonus for correctly identifying automatable actions
    if predicted.can_be_automated and ground_truth_label in ('DELETED', 'ARCHIVED', 'JUNK'):
        reward += 0.1

    return reward, priority_alignment


def compute_timeliness_reward(
    predicted: EmailAction,
    state: EmailState,
) -> float:
    """Compute timeliness reward based on response time appropriateness.

    Compares suggested response time against urgency signals in the email.

    Args:
        predicted: Predicted EmailAction with response time
        state: EmailState with temporal and content features

    Returns:
        Timeliness reward (-1 to 1)
    """
    # Get urgency from content features
    urgency = state.content.urgency_signals
    has_deadline = state.content.has_deadline
    is_action_request = state.content.is_action_request

    # Compute expected response time based on urgency
    if has_deadline or urgency > 0.7:
        expected_hours = 4.0  # same_day
    elif is_action_request or urgency > 0.4:
        expected_hours = 16.0  # next_day
    elif urgency > 0.2:
        expected_hours = 72.0  # this_week
    else:
        expected_hours = 168.0  # when_possible

    # Get actual response time in hours
    actual_hours = response_time_to_hours(predicted.suggested_response_time)

    # Compute reward based on difference
    if actual_hours <= expected_hours:
        # Faster than expected - good, but diminishing returns
        ratio = expected_hours / max(actual_hours, 0.5)
        reward = min(0.2 * (ratio - 1.0), 0.5)  # Cap bonus
    else:
        # Slower than expected - penalty
        ratio = actual_hours / expected_hours
        reward = -min(0.3 * (ratio - 1.0), 1.0)  # Cap penalty

    # Extra penalty for very slow response to urgent emails
    if urgency > 0.7 and actual_hours > 24.0:
        reward -= 0.3

    return reward


def compute_confidence_bonus(
    predicted: EmailAction,
    ground_truth_label: str,
) -> float:
    """Compute bonus/penalty based on confidence calibration.

    Rewards high confidence when correct, penalizes when wrong.

    Args:
        predicted: Predicted EmailAction with confidence
        ground_truth_label: Ground truth action label

    Returns:
        Confidence bonus (-0.2 to 0.2)
    """
    gt_action_type = LABEL_TO_ACTION_TYPE.get(ground_truth_label, 'archive')
    is_correct = predicted.action_type == gt_action_type

    if is_correct:
        # Bonus for high confidence when correct
        return 0.2 * predicted.confidence
    else:
        # Penalty for high confidence when wrong
        return -0.2 * predicted.confidence


def compute_reward(
    predicted: EmailAction,
    state: EmailState,
    ground_truth_label: str,
    config: Optional[RewardConfig] = None,
) -> RewardComponents:
    """Compute complete reward for an agent action.

    This is the main reward function that combines all signals.

    Args:
        predicted: Predicted EmailAction from the agent
        state: EmailState providing context
        ground_truth_label: Ground truth action label (e.g., 'REPLIED')
        config: Optional RewardConfig for weight customization

    Returns:
        RewardComponents with detailed breakdown
    """
    if config is None:
        config = RewardConfig()

    # Compute individual components
    accuracy_reward, action_match_score = compute_accuracy_reward(
        predicted, ground_truth_label
    )

    satisfaction_reward, priority_alignment = compute_satisfaction_reward(
        predicted, state, ground_truth_label
    )

    timeliness_reward = compute_timeliness_reward(predicted, state)

    confidence_bonus = compute_confidence_bonus(predicted, ground_truth_label)

    # Compute weighted total
    total_reward = (
        config.accuracy_weight * accuracy_reward +
        config.satisfaction_weight * satisfaction_reward +
        config.timeliness_weight * timeliness_reward +
        config.confidence_bonus_scale * confidence_bonus
    )

    # Apply missed important penalty
    if ground_truth_label in ('REPLIED', 'FORWARDED') and \
       predicted.action_type in ('archive', 'delete'):
        total_reward -= config.missed_important_penalty

    # Apply false urgent penalty
    if ground_truth_label in ('ARCHIVED', 'DELETED', 'JUNK') and \
       predicted.action_type == 'reply_now':
        total_reward -= config.false_urgent_penalty

    return RewardComponents(
        accuracy_reward=accuracy_reward,
        satisfaction_reward=satisfaction_reward,
        timeliness_reward=timeliness_reward,
        total_reward=total_reward,
        action_match_score=action_match_score,
        priority_alignment=priority_alignment,
        response_time_score=timeliness_reward,  # Same as timeliness
        confidence_bonus=confidence_bonus,
    )


def compute_batch_rewards(
    predictions: list[EmailAction],
    states: list[EmailState],
    ground_truth_labels: list[str],
    config: Optional[RewardConfig] = None,
) -> tuple[list[float], dict]:
    """Compute rewards for a batch of predictions.

    Args:
        predictions: List of predicted EmailActions
        states: List of corresponding EmailStates
        ground_truth_labels: List of ground truth labels
        config: Optional RewardConfig

    Returns:
        Tuple of (rewards_list, aggregate_stats)
    """
    if len(predictions) != len(states) != len(ground_truth_labels):
        raise ValueError("All input lists must have the same length")

    rewards = []
    components_list = []

    for pred, state, label in zip(predictions, states, ground_truth_labels):
        components = compute_reward(pred, state, label, config)
        rewards.append(components.total_reward)
        components_list.append(components)

    # Compute aggregate statistics
    n = len(rewards)
    if n > 0:
        stats = {
            'mean_reward': sum(rewards) / n,
            'mean_accuracy': sum(c.accuracy_reward for c in components_list) / n,
            'mean_satisfaction': sum(c.satisfaction_reward for c in components_list) / n,
            'mean_timeliness': sum(c.timeliness_reward for c in components_list) / n,
            'action_accuracy': sum(c.action_match_score for c in components_list) / n,
            'mean_priority_alignment': sum(c.priority_alignment for c in components_list) / n,
        }
    else:
        stats = {
            'mean_reward': 0.0,
            'mean_accuracy': 0.0,
            'mean_satisfaction': 0.0,
            'mean_timeliness': 0.0,
            'action_accuracy': 0.0,
            'mean_priority_alignment': 0.0,
        }

    return rewards, stats


class RewardShaper:
    """Reward shaping for curriculum learning.

    Provides progressive difficulty by adjusting reward sensitivity
    based on training progress.
    """

    def __init__(
        self,
        initial_config: Optional[RewardConfig] = None,
        warmup_steps: int = 1000,
        curriculum_steps: int = 10000,
    ):
        """Initialize reward shaper.

        Args:
            initial_config: Starting reward configuration
            warmup_steps: Steps with lenient rewards
            curriculum_steps: Steps to full difficulty
        """
        self.initial_config = initial_config or RewardConfig()
        self.warmup_steps = warmup_steps
        self.curriculum_steps = curriculum_steps
        self.step = 0

    def get_config(self) -> RewardConfig:
        """Get current reward config based on training progress."""
        if self.step < self.warmup_steps:
            # Lenient phase: focus on accuracy, less on timing
            progress = self.step / self.warmup_steps
            return RewardConfig(
                accuracy_weight=0.7 - 0.2 * progress,
                satisfaction_weight=0.2 + 0.1 * progress,
                timeliness_weight=0.1 + 0.1 * progress,
                missed_important_penalty=0.1 + 0.2 * progress,
                false_urgent_penalty=0.1 + 0.1 * progress,
            )
        elif self.step < self.warmup_steps + self.curriculum_steps:
            # Curriculum phase: gradually increase penalties
            progress = (self.step - self.warmup_steps) / self.curriculum_steps
            return RewardConfig(
                accuracy_weight=0.5,
                satisfaction_weight=0.3,
                timeliness_weight=0.2,
                missed_important_penalty=0.3 * (1 + progress),
                false_urgent_penalty=0.2 * (1 + progress),
            )
        else:
            # Full difficulty
            return self.initial_config

    def step_forward(self, n: int = 1) -> None:
        """Advance training step counter."""
        self.step += n

    def compute_shaped_reward(
        self,
        predicted: EmailAction,
        state: EmailState,
        ground_truth_label: str,
    ) -> RewardComponents:
        """Compute reward with current shaping."""
        config = self.get_config()
        return compute_reward(predicted, state, ground_truth_label, config)


def sparse_reward(
    predicted: EmailAction,
    ground_truth_label: str,
) -> float:
    """Simple sparse reward for evaluation.

    Returns 1.0 for exact match, 0.0 otherwise.

    Args:
        predicted: Predicted EmailAction
        ground_truth_label: Ground truth label

    Returns:
        1.0 if action types match, 0.0 otherwise
    """
    gt_action_type = LABEL_TO_ACTION_TYPE.get(ground_truth_label, 'archive')
    return 1.0 if predicted.action_type == gt_action_type else 0.0


def hierarchical_reward(
    predicted: EmailAction,
    ground_truth_label: str,
) -> float:
    """Hierarchical reward with partial credit.

    Provides partial credit for similar actions:
    - 1.0 for exact match
    - 0.5 for same category (response, passive, task)
    - 0.0 otherwise

    Args:
        predicted: Predicted EmailAction
        ground_truth_label: Ground truth label

    Returns:
        Hierarchical reward (0.0, 0.5, or 1.0)
    """
    gt_action_type = LABEL_TO_ACTION_TYPE.get(ground_truth_label, 'archive')

    if predicted.action_type == gt_action_type:
        return 1.0

    # Define action categories
    response_actions = {'reply_now', 'reply_later'}
    passive_actions = {'archive', 'delete'}
    active_actions = {'forward', 'create_task'}

    pred_type = predicted.action_type
    for category in [response_actions, passive_actions, active_actions]:
        if pred_type in category and gt_action_type in category:
            return 0.5

    return 0.0
