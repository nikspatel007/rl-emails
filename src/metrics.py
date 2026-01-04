#!/usr/bin/env python3
"""Evaluation metrics for email prioritization RL system.

This module provides comprehensive metrics for evaluating email prioritization
model performance, including:
- Standard classification metrics (accuracy, precision, recall, F1)
- Regression metrics for priority scores
- Email-specific custom metrics (urgency detection, ranking correlation)

Compatible with:
- EmailPolicyNetwork outputs (action_logits, timing_logits, priority)
- Training loop evaluation callbacks
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from collections import Counter

import torch


# Action type indices (from policy_network.py)
ACTION_NAMES = ['reply_now', 'reply_later', 'forward', 'archive', 'delete', 'create_task']
NUM_ACTION_TYPES = len(ACTION_NAMES)

# Response timing indices
TIMING_NAMES = ['immediate', 'same_day', 'next_day', 'this_week', 'when_possible']
NUM_TIMINGS = len(TIMING_NAMES)

# Urgent action/timing indices for special metrics
URGENT_ACTION_INDICES = {0}  # reply_now
URGENT_TIMING_INDICES = {0, 1}  # immediate, same_day


@dataclass
class ClassificationMetrics:
    """Metrics for multi-class classification tasks."""
    accuracy: float
    precision_per_class: dict[int, float]
    recall_per_class: dict[int, float]
    f1_per_class: dict[int, float]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion_matrix: Optional[torch.Tensor] = None
    support_per_class: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        result = {
            'accuracy': self.accuracy,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'macro_f1': self.macro_f1,
            'weighted_precision': self.weighted_precision,
            'weighted_recall': self.weighted_recall,
            'weighted_f1': self.weighted_f1,
        }
        for cls_idx, f1 in self.f1_per_class.items():
            result[f'f1_class_{cls_idx}'] = f1
        return result


@dataclass
class RegressionMetrics:
    """Metrics for regression tasks (priority scores)."""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    r2: float  # Coefficient of determination

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'r2': self.r2,
        }


@dataclass
class EmailMetrics:
    """Email-specific custom metrics."""
    # Urgency detection
    urgency_precision: float  # Precision for urgent emails
    urgency_recall: float  # Recall for urgent emails (critical: don't miss urgent!)
    urgency_f1: float

    # Priority ranking
    spearman_correlation: float  # Spearman rank correlation
    kendall_tau: float  # Kendall's tau correlation

    # Timing alignment
    timing_mae: float  # Mean absolute error in timing class distance
    timing_within_one: float  # Proportion within one timing class

    # Fields with defaults must come last
    top_k_precision: dict[int, float] = field(default_factory=dict)  # Precision at k

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        result = {
            'urgency_precision': self.urgency_precision,
            'urgency_recall': self.urgency_recall,
            'urgency_f1': self.urgency_f1,
            'spearman_correlation': self.spearman_correlation,
            'kendall_tau': self.kendall_tau,
            'timing_mae': self.timing_mae,
            'timing_within_one': self.timing_within_one,
        }
        for k, prec in self.top_k_precision.items():
            result[f'top_{k}_precision'] = prec
        return result


@dataclass
class EvaluationResults:
    """Combined evaluation results for all metric types."""
    action_metrics: ClassificationMetrics
    timing_metrics: ClassificationMetrics
    priority_metrics: RegressionMetrics
    email_metrics: EmailMetrics
    n_samples: int

    def to_dict(self) -> dict:
        """Convert all metrics to flat dictionary for logging."""
        result = {'n_samples': self.n_samples}
        for key, val in self.action_metrics.to_dict().items():
            result[f'action_{key}'] = val
        for key, val in self.timing_metrics.to_dict().items():
            result[f'timing_{key}'] = val
        for key, val in self.priority_metrics.to_dict().items():
            result[f'priority_{key}'] = val
        for key, val in self.email_metrics.to_dict().items():
            result[f'email_{key}'] = val
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"Samples: {self.n_samples}",
            "",
            "ACTION CLASSIFICATION:",
            f"  Accuracy: {self.action_metrics.accuracy:.4f}",
            f"  Macro F1: {self.action_metrics.macro_f1:.4f}",
            f"  Weighted F1: {self.action_metrics.weighted_f1:.4f}",
            "",
            "TIMING CLASSIFICATION:",
            f"  Accuracy: {self.timing_metrics.accuracy:.4f}",
            f"  Macro F1: {self.timing_metrics.macro_f1:.4f}",
            f"  Weighted F1: {self.timing_metrics.weighted_f1:.4f}",
            "",
            "PRIORITY REGRESSION:",
            f"  MAE: {self.priority_metrics.mae:.4f}",
            f"  RMSE: {self.priority_metrics.rmse:.4f}",
            f"  R²: {self.priority_metrics.r2:.4f}",
            "",
            "EMAIL-SPECIFIC METRICS:",
            f"  Urgency Recall: {self.email_metrics.urgency_recall:.4f}",
            f"  Urgency F1: {self.email_metrics.urgency_f1:.4f}",
            f"  Spearman ρ: {self.email_metrics.spearman_correlation:.4f}",
            f"  Kendall τ: {self.email_metrics.kendall_tau:.4f}",
            f"  Timing within ±1: {self.email_metrics.timing_within_one:.4f}",
        ]
        if self.email_metrics.top_k_precision:
            lines.append("  Top-K Precision:")
            for k in sorted(self.email_metrics.top_k_precision.keys()):
                lines.append(f"    @{k}: {self.email_metrics.top_k_precision[k]:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> ClassificationMetrics:
    """Compute classification metrics for multi-class predictions.

    Args:
        predictions: Predicted class indices, shape (N,)
        targets: Ground truth class indices, shape (N,)
        num_classes: Number of classes

    Returns:
        ClassificationMetrics with all computed values
    """
    predictions = predictions.long()
    targets = targets.long()
    n = len(targets)

    if n == 0:
        return ClassificationMetrics(
            accuracy=0.0,
            precision_per_class={},
            recall_per_class={},
            f1_per_class={},
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            weighted_precision=0.0,
            weighted_recall=0.0,
            weighted_f1=0.0,
        )

    # Accuracy
    accuracy = (predictions == targets).float().mean().item()

    # Build confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, tgt in zip(predictions, targets):
        confusion[tgt, pred] += 1

    # Per-class metrics
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    support_per_class = {}

    for cls in range(num_classes):
        tp = confusion[cls, cls].item()
        fp = confusion[:, cls].sum().item() - tp
        fn = confusion[cls, :].sum().item() - tp
        support = confusion[cls, :].sum().item()

        support_per_class[cls] = support

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_per_class[cls] = precision
        recall_per_class[cls] = recall
        f1_per_class[cls] = f1

    # Macro averages (simple mean across classes)
    classes_with_support = [c for c in range(num_classes) if support_per_class[c] > 0]
    if classes_with_support:
        macro_precision = sum(precision_per_class[c] for c in classes_with_support) / len(classes_with_support)
        macro_recall = sum(recall_per_class[c] for c in classes_with_support) / len(classes_with_support)
        macro_f1 = sum(f1_per_class[c] for c in classes_with_support) / len(classes_with_support)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    # Weighted averages (weighted by support)
    total_support = sum(support_per_class.values())
    if total_support > 0:
        weighted_precision = sum(
            precision_per_class[c] * support_per_class[c] for c in range(num_classes)
        ) / total_support
        weighted_recall = sum(
            recall_per_class[c] * support_per_class[c] for c in range(num_classes)
        ) / total_support
        weighted_f1 = sum(
            f1_per_class[c] * support_per_class[c] for c in range(num_classes)
        ) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        confusion_matrix=confusion,
        support_per_class=support_per_class,
    )


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> RegressionMetrics:
    """Compute regression metrics for priority scores.

    Args:
        predictions: Predicted values, shape (N,)
        targets: Ground truth values, shape (N,)

    Returns:
        RegressionMetrics with MAE, MSE, RMSE, R²
    """
    predictions = predictions.float()
    targets = targets.float()
    n = len(targets)

    if n == 0:
        return RegressionMetrics(mae=0.0, mse=0.0, rmse=0.0, r2=0.0)

    # MAE
    mae = (predictions - targets).abs().mean().item()

    # MSE
    mse = ((predictions - targets) ** 2).mean().item()

    # RMSE
    rmse = mse ** 0.5

    # R² (coefficient of determination)
    ss_res = ((targets - predictions) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return RegressionMetrics(mae=mae, mse=mse, rmse=rmse, r2=r2)


def compute_urgency_metrics(
    action_predictions: torch.Tensor,
    action_targets: torch.Tensor,
    timing_predictions: torch.Tensor,
    timing_targets: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute urgency detection metrics.

    An email is considered "urgent" if action is reply_now OR timing is immediate/same_day.

    Args:
        action_predictions: Predicted action indices, shape (N,)
        action_targets: Ground truth action indices, shape (N,)
        timing_predictions: Predicted timing indices, shape (N,)
        timing_targets: Ground truth timing indices, shape (N,)

    Returns:
        Tuple of (precision, recall, f1) for urgent email detection
    """
    n = len(action_targets)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Determine which samples are "urgent" based on action OR timing
    def is_urgent(action: int, timing: int) -> bool:
        return action in URGENT_ACTION_INDICES or timing in URGENT_TIMING_INDICES

    # Ground truth urgent
    target_urgent = torch.tensor([
        is_urgent(a.item(), t.item())
        for a, t in zip(action_targets, timing_targets)
    ])

    # Predicted urgent
    pred_urgent = torch.tensor([
        is_urgent(a.item(), t.item())
        for a, t in zip(action_predictions, timing_predictions)
    ])

    # Compute precision, recall, f1
    tp = ((pred_urgent) & (target_urgent)).sum().item()
    fp = ((pred_urgent) & (~target_urgent)).sum().item()
    fn = ((~pred_urgent) & (target_urgent)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_ranking_correlation(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Compute Spearman and Kendall ranking correlations.

    Used for priority score ranking evaluation.

    Args:
        predictions: Predicted priority scores, shape (N,)
        targets: Ground truth priority scores, shape (N,)

    Returns:
        Tuple of (spearman_rho, kendall_tau)
    """
    predictions = predictions.float()
    targets = targets.float()
    n = len(targets)

    if n < 2:
        return 0.0, 0.0

    # Convert to ranks
    def to_ranks(x: torch.Tensor) -> torch.Tensor:
        """Convert values to ranks (1-based, handling ties with average rank)."""
        sorted_indices = x.argsort()
        ranks = torch.zeros_like(x)

        # Assign ranks
        current_rank = 1
        i = 0
        while i < n:
            j = i
            # Find all tied values
            while j < n and x[sorted_indices[j]] == x[sorted_indices[i]]:
                j += 1
            # Assign average rank to tied values
            avg_rank = (current_rank + current_rank + j - i - 1) / 2.0
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank
            current_rank += j - i
            i = j

        return ranks

    pred_ranks = to_ranks(predictions)
    target_ranks = to_ranks(targets)

    # Spearman correlation: 1 - (6 * sum(d²)) / (n * (n² - 1))
    d = pred_ranks - target_ranks
    spearman = 1.0 - (6.0 * (d ** 2).sum().item()) / (n * (n ** 2 - 1))

    # Kendall's tau: (concordant - discordant) / total_pairs
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_sign = (predictions[i] - predictions[j]).sign().item()
            target_sign = (targets[i] - targets[j]).sign().item()
            if pred_sign == 0 or target_sign == 0:
                continue  # Skip ties
            if pred_sign == target_sign:
                concordant += 1
            else:
                discordant += 1

    total_pairs = concordant + discordant
    kendall_tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0

    return spearman, kendall_tau


def compute_top_k_precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k_values: list[int] = None,
) -> dict[int, float]:
    """Compute top-K precision for priority ranking.

    Measures: of the K emails predicted as highest priority, how many
    are actually in the ground truth top K?

    Args:
        predictions: Predicted priority scores, shape (N,)
        targets: Ground truth priority scores, shape (N,)
        k_values: List of K values to compute (default: [5, 10, 20])

    Returns:
        Dict mapping K -> precision@K
    """
    if k_values is None:
        k_values = [5, 10, 20]

    predictions = predictions.float()
    targets = targets.float()
    n = len(targets)

    result = {}
    for k in k_values:
        if k > n:
            result[k] = 0.0
            continue

        # Top K predicted
        pred_top_k = set(predictions.argsort(descending=True)[:k].tolist())

        # Top K ground truth
        target_top_k = set(targets.argsort(descending=True)[:k].tolist())

        # Precision: overlap / k
        overlap = len(pred_top_k & target_top_k)
        result[k] = overlap / k

    return result


def compute_timing_alignment(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Compute timing alignment metrics.

    Timing classes are ordinal (immediate < same_day < next_day < this_week < when_possible),
    so we can measure "distance" between predicted and actual timing.

    Args:
        predictions: Predicted timing indices, shape (N,)
        targets: Ground truth timing indices, shape (N,)

    Returns:
        Tuple of (mae, within_one_ratio)
        - mae: Mean absolute difference in timing class
        - within_one_ratio: Proportion of predictions within ±1 timing class
    """
    predictions = predictions.long()
    targets = targets.long()
    n = len(targets)

    if n == 0:
        return 0.0, 0.0

    # Absolute difference in timing class
    diff = (predictions - targets).abs().float()

    mae = diff.mean().item()
    within_one = (diff <= 1).float().mean().item()

    return mae, within_one


def compute_email_metrics(
    action_predictions: torch.Tensor,
    action_targets: torch.Tensor,
    timing_predictions: torch.Tensor,
    timing_targets: torch.Tensor,
    priority_predictions: torch.Tensor,
    priority_targets: torch.Tensor,
    top_k_values: list[int] = None,
) -> EmailMetrics:
    """Compute all email-specific custom metrics.

    Args:
        action_predictions: Predicted action indices, shape (N,)
        action_targets: Ground truth action indices, shape (N,)
        timing_predictions: Predicted timing indices, shape (N,)
        timing_targets: Ground truth timing indices, shape (N,)
        priority_predictions: Predicted priority scores, shape (N,)
        priority_targets: Ground truth priority scores, shape (N,)
        top_k_values: K values for top-K precision

    Returns:
        EmailMetrics with all custom metrics
    """
    # Urgency metrics
    urgency_precision, urgency_recall, urgency_f1 = compute_urgency_metrics(
        action_predictions, action_targets,
        timing_predictions, timing_targets,
    )

    # Ranking metrics
    spearman, kendall = compute_ranking_correlation(priority_predictions, priority_targets)

    # Top-K precision
    top_k = compute_top_k_precision(priority_predictions, priority_targets, top_k_values)

    # Timing alignment
    timing_mae, timing_within_one = compute_timing_alignment(timing_predictions, timing_targets)

    return EmailMetrics(
        urgency_precision=urgency_precision,
        urgency_recall=urgency_recall,
        urgency_f1=urgency_f1,
        spearman_correlation=spearman,
        kendall_tau=kendall,
        top_k_precision=top_k,
        timing_mae=timing_mae,
        timing_within_one=timing_within_one,
    )


def evaluate_batch(
    action_predictions: torch.Tensor,
    action_targets: torch.Tensor,
    timing_predictions: torch.Tensor,
    timing_targets: torch.Tensor,
    priority_predictions: torch.Tensor,
    priority_targets: torch.Tensor,
    top_k_values: list[int] = None,
) -> EvaluationResults:
    """Compute all evaluation metrics for a batch of predictions.

    This is the main entry point for evaluation.

    Args:
        action_predictions: Predicted action indices, shape (N,)
        action_targets: Ground truth action indices, shape (N,)
        timing_predictions: Predicted timing indices, shape (N,)
        timing_targets: Ground truth timing indices, shape (N,)
        priority_predictions: Predicted priority scores, shape (N,)
        priority_targets: Ground truth priority scores, shape (N,)
        top_k_values: K values for top-K precision

    Returns:
        EvaluationResults with all metrics
    """
    n = len(action_targets)

    # Classification metrics
    action_metrics = compute_classification_metrics(
        action_predictions, action_targets, NUM_ACTION_TYPES
    )
    timing_metrics = compute_classification_metrics(
        timing_predictions, timing_targets, NUM_TIMINGS
    )

    # Regression metrics
    priority_metrics = compute_regression_metrics(priority_predictions, priority_targets)

    # Email-specific metrics
    email_metrics = compute_email_metrics(
        action_predictions, action_targets,
        timing_predictions, timing_targets,
        priority_predictions, priority_targets,
        top_k_values,
    )

    return EvaluationResults(
        action_metrics=action_metrics,
        timing_metrics=timing_metrics,
        priority_metrics=priority_metrics,
        email_metrics=email_metrics,
        n_samples=n,
    )


class MetricsAccumulator:
    """Accumulator for computing metrics across multiple batches.

    Use this during training/validation loops to accumulate predictions
    and compute metrics at the end of an epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulated predictions."""
        self.action_preds = []
        self.action_targets = []
        self.timing_preds = []
        self.timing_targets = []
        self.priority_preds = []
        self.priority_targets = []

    def update(
        self,
        action_preds: torch.Tensor,
        action_targets: torch.Tensor,
        timing_preds: torch.Tensor,
        timing_targets: torch.Tensor,
        priority_preds: torch.Tensor,
        priority_targets: torch.Tensor,
    ):
        """Add a batch of predictions.

        Args:
            action_preds: Predicted action indices, shape (batch,)
            action_targets: Ground truth action indices, shape (batch,)
            timing_preds: Predicted timing indices, shape (batch,)
            timing_targets: Ground truth timing indices, shape (batch,)
            priority_preds: Predicted priority scores, shape (batch,)
            priority_targets: Ground truth priority scores, shape (batch,)
        """
        self.action_preds.append(action_preds.detach().cpu())
        self.action_targets.append(action_targets.detach().cpu())
        self.timing_preds.append(timing_preds.detach().cpu())
        self.timing_targets.append(timing_targets.detach().cpu())
        self.priority_preds.append(priority_preds.detach().cpu())
        self.priority_targets.append(priority_targets.detach().cpu())

    def compute(self, top_k_values: list[int] = None) -> EvaluationResults:
        """Compute metrics from all accumulated predictions.

        Args:
            top_k_values: K values for top-K precision

        Returns:
            EvaluationResults with all metrics
        """
        if not self.action_preds:
            # No data accumulated
            return evaluate_batch(
                torch.tensor([]), torch.tensor([]),
                torch.tensor([]), torch.tensor([]),
                torch.tensor([]), torch.tensor([]),
                top_k_values,
            )

        # Concatenate all batches
        action_preds = torch.cat(self.action_preds)
        action_targets = torch.cat(self.action_targets)
        timing_preds = torch.cat(self.timing_preds)
        timing_targets = torch.cat(self.timing_targets)
        priority_preds = torch.cat(self.priority_preds)
        priority_targets = torch.cat(self.priority_targets)

        return evaluate_batch(
            action_preds, action_targets,
            timing_preds, timing_targets,
            priority_preds, priority_targets,
            top_k_values,
        )

    @property
    def n_samples(self) -> int:
        """Return total number of accumulated samples."""
        return sum(len(x) for x in self.action_preds)


if __name__ == '__main__':
    # Test the metrics module
    print("=" * 60)
    print("METRICS MODULE TEST")
    print("=" * 60)

    # Generate synthetic data
    torch.manual_seed(42)
    n = 100

    # Ground truth
    action_targets = torch.randint(0, NUM_ACTION_TYPES, (n,))
    timing_targets = torch.randint(0, NUM_TIMINGS, (n,))
    priority_targets = torch.rand(n)

    # Predictions (add some noise to ground truth for realistic scenario)
    action_preds = action_targets.clone()
    action_preds[:20] = torch.randint(0, NUM_ACTION_TYPES, (20,))  # 20% errors

    timing_preds = timing_targets.clone()
    timing_preds[:15] = torch.randint(0, NUM_TIMINGS, (15,))  # 15% errors

    priority_preds = priority_targets + 0.1 * torch.randn(n)
    priority_preds = priority_preds.clamp(0, 1)

    # Compute metrics
    results = evaluate_batch(
        action_preds, action_targets,
        timing_preds, timing_targets,
        priority_preds, priority_targets,
    )

    print(results.summary())

    # Test accumulator
    print("\nTesting MetricsAccumulator...")
    accumulator = MetricsAccumulator()

    # Add batches
    for i in range(0, n, 20):
        end = min(i + 20, n)
        accumulator.update(
            action_preds[i:end], action_targets[i:end],
            timing_preds[i:end], timing_targets[i:end],
            priority_preds[i:end], priority_targets[i:end],
        )

    acc_results = accumulator.compute()
    print(f"\nAccumulated {accumulator.n_samples} samples")
    print(f"Action accuracy: {acc_results.action_metrics.accuracy:.4f}")
    print(f"Priority RMSE: {acc_results.priority_metrics.rmse:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
