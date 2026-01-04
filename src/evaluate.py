#!/usr/bin/env python3
"""Evaluate email policy network on test data.

Loads test.json, extracts features, runs model predictions, and computes metrics.
Designed to be re-run after each training stage to track improvement.

Usage:
    python src/evaluate.py --data data/test.json --output eval_results/baseline.json
    python src/evaluate.py --data data/test.json --checkpoint checkpoints/stage_1.pt --output eval_results/stage_1.json
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Running in metrics-only mode.")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import config classes for checkpoint loading (if available)
try:
    from sft_training import SFTConfig
except ImportError:
    # Define minimal SFTConfig for unpickling checkpoints
    @dataclass
    class SFTConfig:
        learning_rate: float = 1e-3
        weight_decay: float = 0.01
        batch_size: int = 64
        epochs: int = 10
        warmup_epochs: int = 1
        lr_decay: float = 0.1
        lr_decay_epochs: tuple = (7, 9)
        dropout: float = 0.1
        label_smoothing: float = 0.1
        log_every: int = 10
        save_every: int = 1
        checkpoint_dir: str = "checkpoints"

try:
    from dpo_training import DPOConfig
except ImportError:
    # Define minimal DPOConfig for unpickling checkpoints
    @dataclass
    class DPOConfig:
        beta: float = 0.1
        reference_free: bool = False
        learning_rate: float = 5e-5
        weight_decay: float = 0.01
        batch_size: int = 32
        epochs: int = 5
        min_priority_gap: int = 1
        max_pairs_per_email: int = 5
        use_margin_weighting: bool = True
        margin_scale: float = 0.5
        sft_weight: float = 0.0
        log_every: int = 10
        save_every: int = 1
        checkpoint_dir: str = "checkpoints"
        val_split: float = 0.1

from features.combined import extract_combined_features, CombinedFeatureExtractor
from policy_network import EmailPolicyNetwork, PolicyConfig, create_policy_network


# Action mapping: dataset labels -> model indices (5-class action space)
# Model outputs: 0=reply_now, 1=reply_later, 2=forward, 3=archive, 4=delete
ACTION_NAMES = ['reply_now', 'reply_later', 'forward', 'archive', 'delete']

# Map dataset labels to model action indices
LABEL_TO_ACTION = {
    'REPLY_NOW': 0,    # reply_now (responded < 1 hour)
    'REPLY_LATER': 1,  # reply_later (responded >= 1 hour)
    'FORWARD': 2,      # forward
    'ARCHIVE': 3,      # archive
    'DELETE': 4,       # delete
}

# Timing labels: 0=immediate, 1=same_day, 2=next_day, 3=this_week, 4=when_possible
TIMING_NAMES = ['immediate', 'same_day', 'next_day', 'this_week', 'when_possible']


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Overall accuracy
    action_accuracy: float
    timing_accuracy: float

    # Per-action metrics
    action_precision: dict[str, float]
    action_recall: dict[str, float]
    action_f1: dict[str, float]

    # Priority metrics
    priority_mae: float  # Mean absolute error
    priority_correlation: float  # Spearman correlation with ground truth priority

    # Additional stats
    total_samples: int
    action_distribution_true: dict[str, int]
    action_distribution_pred: dict[str, int]
    confusion_matrix: dict[str, dict[str, int]]

    # Timing
    inference_time_seconds: float
    samples_per_second: float


def compute_precision_recall_f1(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute per-class precision, recall, and F1 scores."""
    # Count true positives, false positives, false negatives per class
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    precision = {}
    recall = {}
    f1 = {}

    for i in range(num_classes):
        name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f'class_{i}'

        # Precision = TP / (TP + FP)
        if tp[i] + fp[i] > 0:
            precision[name] = tp[i] / (tp[i] + fp[i])
        else:
            precision[name] = 0.0

        # Recall = TP / (TP + FN)
        if tp[i] + fn[i] > 0:
            recall[name] = tp[i] / (tp[i] + fn[i])
        else:
            recall[name] = 0.0

        # F1 = 2 * P * R / (P + R)
        if precision[name] + recall[name] > 0:
            f1[name] = 2 * precision[name] * recall[name] / (precision[name] + recall[name])
        else:
            f1[name] = 0.0

    return precision, recall, f1


def compute_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> dict[str, dict[str, int]]:
    """Compute confusion matrix as nested dict."""
    matrix = {ACTION_NAMES[i]: {ACTION_NAMES[j]: 0 for j in range(num_classes)}
              for i in range(num_classes)}

    for t, p in zip(y_true, y_pred):
        if t < num_classes and p < num_classes:
            matrix[ACTION_NAMES[t]][ACTION_NAMES[p]] += 1

    return matrix


def compute_spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    n = len(x)

    # Compute ranks
    def rank(arr):
        sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = rank_val + 1
        return ranks

    x_ranks = rank(x)
    y_ranks = rank(y)

    # Compute correlation
    d_squared_sum = sum((xr - yr) ** 2 for xr, yr in zip(x_ranks, y_ranks))
    correlation = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1)) if n > 1 else 0.0

    return correlation


def load_emails(data_path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load emails from JSON file."""
    print(f"Loading emails from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    if limit:
        emails = emails[:limit]

    print(f"Loaded {len(emails)} emails")
    return emails


def extract_features_batch(
    emails: list[dict],
    batch_size: int = 1000,
) -> list:
    """Extract features from emails in batches."""
    print(f"Extracting features from {len(emails)} emails...")

    extractor = CombinedFeatureExtractor()
    features_list = []

    for i in range(0, len(emails), batch_size):
        batch = emails[i:i + batch_size]
        batch_features = [extractor.to_vector(email) for email in batch]
        features_list.extend(batch_features)

        if (i + batch_size) % 10000 == 0 or i + batch_size >= len(emails):
            print(f"  Processed {min(i + batch_size, len(emails))}/{len(emails)} emails")

    return features_list


def get_ground_truth(emails: list[dict]) -> tuple[list[int], list[float]]:
    """Extract ground truth action labels and priorities from emails.

    Returns:
        Tuple of (action_indices, priorities)
    """
    action_indices = []
    priorities = []

    for email in emails:
        label = email.get('action', 'ARCHIVE')
        action_idx = LABEL_TO_ACTION.get(label, 3)  # Default to archive
        action_indices.append(action_idx)

        # Infer priority from action (higher priority for immediate replies)
        if label == 'REPLY_NOW':
            priority = 0.9
        elif label == 'REPLY_LATER':
            priority = 0.7
        elif label == 'FORWARD':
            priority = 0.6
        elif label == 'DELETE':
            priority = 0.2
        else:  # ARCHIVE
            priority = 0.4
        priorities.append(priority)

    return action_indices, priorities


def evaluate_model(
    model: "EmailPolicyNetwork",
    features: list,
    y_true_actions: list[int],
    y_true_priorities: list[float],
    batch_size: int = 512,
) -> EvaluationMetrics:
    """Evaluate model on test data."""
    import torch

    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')

    y_pred_actions = []
    y_pred_timings = []
    y_pred_priorities = []

    print(f"Running inference on {len(features)} samples...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]

            # Stack features into tensor
            if isinstance(batch_features[0], np.ndarray):
                x = torch.tensor(np.stack(batch_features), dtype=torch.float32).to(device)
            else:
                x = torch.tensor(batch_features, dtype=torch.float32).to(device)

            # Get predictions
            output = model(x)

            # Get argmax predictions
            actions = output.action_logits.argmax(dim=-1).cpu().tolist()
            timings = output.timing_logits.argmax(dim=-1).cpu().tolist()
            priorities = output.priority.squeeze(-1).cpu().tolist()

            y_pred_actions.extend(actions)
            y_pred_timings.extend(timings)
            y_pred_priorities.extend(priorities)

    inference_time = time.time() - start_time
    samples_per_second = len(features) / inference_time

    print(f"Inference complete: {inference_time:.2f}s ({samples_per_second:.0f} samples/sec)")

    # Compute metrics
    num_classes = 5

    # Action accuracy
    correct_actions = sum(1 for t, p in zip(y_true_actions, y_pred_actions) if t == p)
    action_accuracy = correct_actions / len(y_true_actions)

    # For timing, we don't have ground truth, so we'll use a placeholder
    timing_accuracy = 0.0  # No ground truth for timing

    # Per-action precision/recall/F1
    precision, recall, f1 = compute_precision_recall_f1(y_true_actions, y_pred_actions, num_classes)

    # Confusion matrix
    confusion = compute_confusion_matrix(y_true_actions, y_pred_actions, num_classes)

    # Priority metrics
    priority_errors = [abs(t - p) for t, p in zip(y_true_priorities, y_pred_priorities)]
    priority_mae = sum(priority_errors) / len(priority_errors)
    priority_correlation = compute_spearman_correlation(y_true_priorities, y_pred_priorities)

    # Distribution stats
    action_dist_true = Counter(y_true_actions)
    action_dist_pred = Counter(y_pred_actions)

    return EvaluationMetrics(
        action_accuracy=action_accuracy,
        timing_accuracy=timing_accuracy,
        action_precision=precision,
        action_recall=recall,
        action_f1=f1,
        priority_mae=priority_mae,
        priority_correlation=priority_correlation,
        total_samples=len(features),
        action_distribution_true={ACTION_NAMES[k]: v for k, v in action_dist_true.items()},
        action_distribution_pred={ACTION_NAMES[k]: v for k, v in action_dist_pred.items()},
        confusion_matrix=confusion,
        inference_time_seconds=inference_time,
        samples_per_second=samples_per_second,
    )


def print_metrics(metrics: EvaluationMetrics) -> None:
    """Print evaluation metrics in a readable format."""
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()
    print(f"Total samples: {metrics.total_samples}")
    print(f"Inference time: {metrics.inference_time_seconds:.2f}s ({metrics.samples_per_second:.0f} samples/sec)")
    print()
    print("ACCURACY:")
    print(f"  Action prediction: {metrics.action_accuracy:.4f} ({metrics.action_accuracy * 100:.2f}%)")
    print(f"  Priority MAE: {metrics.priority_mae:.4f}")
    print(f"  Priority correlation: {metrics.priority_correlation:.4f}")
    print()
    print("PER-ACTION METRICS:")
    print(f"  {'Action':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 45)
    for action in ACTION_NAMES:
        p = metrics.action_precision.get(action, 0)
        r = metrics.action_recall.get(action, 0)
        f = metrics.action_f1.get(action, 0)
        print(f"  {action:<15} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    print()
    print("ACTION DISTRIBUTION (True vs Predicted):")
    for action in ACTION_NAMES:
        true_count = metrics.action_distribution_true.get(action, 0)
        pred_count = metrics.action_distribution_pred.get(action, 0)
        print(f"  {action:<15}: {true_count:>8} true, {pred_count:>8} predicted")
    print()
    print("=" * 60)


def save_results(metrics: EvaluationMetrics, output_path: Path) -> None:
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = asdict(metrics)
    results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate email policy network')
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/test.json'),
        help='Path to test data JSON (default: data/test.json)',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Path to model checkpoint (default: untrained baseline)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('eval_results/baseline.json'),
        help='Output path for results (default: eval_results/baseline.json)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of samples to evaluate',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for inference (default: 512)',
    )

    args = parser.parse_args()

    if not HAS_TORCH:
        print("Error: PyTorch is required for evaluation")
        sys.exit(1)

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    # Load data
    emails = load_emails(args.data, args.limit)

    # Extract features
    features = extract_features_batch(emails)

    # Get ground truth
    y_true_actions, y_true_priorities = get_ground_truth(emails)

    # Load or create model
    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model = create_policy_network()
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded")
    else:
        print("Using untrained baseline model")
        model = create_policy_network()

    # Evaluate
    metrics = evaluate_model(
        model,
        features,
        y_true_actions,
        y_true_priorities,
        batch_size=args.batch_size,
    )

    # Print and save results
    print_metrics(metrics)
    save_results(metrics, args.output)


if __name__ == '__main__':
    main()
