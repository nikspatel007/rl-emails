#!/usr/bin/env python3
"""Inference pipeline for email action prediction.

Runs trained model on email data and outputs predictions vs actual behavior.

Two modes:
- Offline: Historical validation on emails older than N months
- Online: Simulated live prediction on recent emails

Usage:
    python inference.py --model checkpoints/stage3.pt \
        --data data/emails_labeled.json \
        --mode offline \
        --cutoff-months 2
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

# Lazy imports for modules that require torch
EmailPolicyNetwork = None
PolicyConfig = None
CombinedFeatureExtractor = None
infer_action = None


def _import_ml_modules():
    """Lazily import ML modules that require torch."""
    global EmailPolicyNetwork, PolicyConfig, CombinedFeatureExtractor, infer_action

    if EmailPolicyNetwork is None:
        from src.policy_network import EmailPolicyNetwork as _EPN, PolicyConfig as _PC
        EmailPolicyNetwork = _EPN
        PolicyConfig = _PC

    if CombinedFeatureExtractor is None:
        from src.features.combined import CombinedFeatureExtractor as _CFE
        CombinedFeatureExtractor = _CFE

    if infer_action is None:
        from src.label_actions import infer_action as _ia
        infer_action = _ia


# Model action indices to names
ACTION_NAMES = ['reply_now', 'reply_later', 'forward', 'archive', 'delete', 'create_task']

# Map model action indices to comparable ground truth labels
# Model predictions are mapped to these canonical actions
MODEL_TO_CANONICAL = {
    0: 'REPLY',      # reply_now
    1: 'REPLY',      # reply_later
    2: 'FORWARD',    # forward
    3: 'ARCHIVE',    # archive
    4: 'DELETE',     # delete
    5: 'TASK',       # create_task
}

# Map ground truth labels to canonical actions
GROUND_TRUTH_TO_CANONICAL = {
    'REPLIED': 'REPLY',
    'FORWARDED': 'FORWARD',
    'DELETED': 'DELETE',
    'ARCHIVED': 'ARCHIVE',
    'AUTO_FILED': 'ARCHIVE',  # Treat auto-filed as archive
    'KEPT': 'KEEP',           # Left in inbox
    'COMPOSED': 'COMPOSE',    # New email (not a response)
    'JUNK': 'DELETE',         # Treat junk as delete
}

# Canonical action set for evaluation
CANONICAL_ACTIONS = ['REPLY', 'FORWARD', 'ARCHIVE', 'DELETE', 'KEEP', 'COMPOSE', 'TASK']


@dataclass
class InferenceResult:
    """Result of inference on a single email."""
    email_id: str
    date: Optional[datetime]
    ground_truth: str
    ground_truth_canonical: str
    predicted_action: int
    predicted_action_name: str
    predicted_canonical: str
    predicted_timing: int
    predicted_priority: float
    correct: bool


@dataclass
class InferenceStats:
    """Aggregated statistics from inference run."""
    mode: str
    total_emails: int
    correct: int
    accuracy: float
    confusion_matrix: dict[str, dict[str, int]]
    per_action_metrics: dict[str, dict[str, float]]
    timing_distribution: dict[str, int]
    priority_stats: dict[str, float]


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse email date string to datetime."""
    if not date_str:
        return None

    # Common email date formats
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S %Z',
        '%d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
    ]

    # Clean up timezone info that Python might not handle
    date_str = date_str.replace(' (PST)', '').replace(' (PDT)', '')
    date_str = date_str.replace(' (EST)', '').replace(' (EDT)', '')
    date_str = date_str.replace(' (CST)', '').replace(' (CDT)', '')

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


def filter_emails_by_date(
    emails: list[dict],
    cutoff_months: int,
    mode: str,
    reference_date: Optional[datetime] = None,
) -> list[dict]:
    """Filter emails based on mode and cutoff.

    Args:
        emails: List of email dictionaries
        cutoff_months: Number of months for cutoff
        mode: 'offline' (older than cutoff) or 'online' (newer than cutoff)
        reference_date: Reference date for cutoff (default: latest email date)

    Returns:
        Filtered list of emails
    """
    # Parse all dates first
    emails_with_dates = []
    for email in emails:
        date = parse_date(email.get('date', ''))
        if date:
            emails_with_dates.append((email, date))

    if not emails_with_dates:
        print("Warning: No emails with parseable dates found", file=sys.stderr)
        return emails  # Return all if no dates

    # Find reference date (latest email)
    if reference_date is None:
        reference_date = max(d for _, d in emails_with_dates)

    cutoff_date = reference_date - timedelta(days=cutoff_months * 30)

    if mode == 'offline':
        # Emails older than cutoff
        filtered = [e for e, d in emails_with_dates if d < cutoff_date]
    else:  # online
        # Emails newer than cutoff
        filtered = [e for e, d in emails_with_dates if d >= cutoff_date]

    return filtered


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(results: list[InferenceResult]) -> InferenceStats:
    """Compute comprehensive metrics from inference results."""
    if not results:
        return InferenceStats(
            mode='unknown',
            total_emails=0,
            correct=0,
            accuracy=0.0,
            confusion_matrix={},
            per_action_metrics={},
            timing_distribution={},
            priority_stats={},
        )

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total > 0 else 0.0

    # Build confusion matrix
    # confusion_matrix[ground_truth][predicted] = count
    confusion: dict[str, dict[str, int]] = {}
    for action in CANONICAL_ACTIONS:
        confusion[action] = {a: 0 for a in CANONICAL_ACTIONS}

    for r in results:
        gt = r.ground_truth_canonical
        pred = r.predicted_canonical
        if gt in confusion and pred in confusion[gt]:
            confusion[gt][pred] += 1

    # Per-action metrics
    per_action: dict[str, dict[str, float]] = {}

    for action in CANONICAL_ACTIONS:
        # True positives: predicted action and it was correct
        tp = confusion[action][action]

        # False positives: predicted action but ground truth was different
        fp = sum(confusion[gt][action] for gt in CANONICAL_ACTIONS if gt != action)

        # False negatives: ground truth was action but predicted different
        fn = sum(confusion[action][pred] for pred in CANONICAL_ACTIONS if pred != action)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = compute_f1(precision, recall)

        support = sum(confusion[action].values())

        per_action[action] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': float(support),
        }

    # Timing distribution
    timing_dist: dict[str, int] = Counter()
    timing_names = ['immediate', 'same_day', 'next_day', 'this_week', 'when_possible']
    for r in results:
        timing_name = timing_names[r.predicted_timing] if r.predicted_timing < len(timing_names) else 'unknown'
        timing_dist[timing_name] += 1

    # Priority statistics
    priorities = [r.predicted_priority for r in results]
    priority_stats = {
        'mean': sum(priorities) / len(priorities),
        'min': min(priorities),
        'max': max(priorities),
        'std': (sum((p - sum(priorities)/len(priorities))**2 for p in priorities) / len(priorities)) ** 0.5,
    }

    return InferenceStats(
        mode='unknown',  # Set by caller
        total_emails=total,
        correct=correct,
        accuracy=accuracy,
        confusion_matrix=confusion,
        per_action_metrics=per_action,
        timing_distribution=dict(timing_dist),
        priority_stats=priority_stats,
    )


def load_model(model_path: Path, device: str = 'cpu'):
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded EmailPolicyNetwork
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for inference. Install with: pip install torch")

    _import_ml_modules()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Extract config if available
        config_dict = checkpoint.get('config', {})
        config = PolicyConfig(**config_dict) if config_dict else PolicyConfig()
    else:
        # Assume it's just the state dict
        state_dict = checkpoint
        config = PolicyConfig()

    model = EmailPolicyNetwork(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def run_inference(
    model,
    emails: list[dict],
    user_email: str = '',
    device: str = 'cpu',
) -> list[InferenceResult]:
    """Run inference on a list of emails.

    Args:
        model: Trained EmailPolicyNetwork
        emails: List of email dictionaries (must have 'action' label)
        user_email: User's email for feature extraction
        device: Device for inference

    Returns:
        List of InferenceResult for each email
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for inference")

    _import_ml_modules()

    results = []
    extractor = CombinedFeatureExtractor(user_email=user_email)

    for email in emails:
        # Get ground truth
        ground_truth = email.get('action') or infer_action(email)
        ground_truth_canonical = GROUND_TRUTH_TO_CANONICAL.get(ground_truth, 'UNKNOWN')

        # Extract features
        try:
            features = extractor.to_vector(email)
            if HAS_NUMPY:
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Warning: Failed to extract features for email: {e}", file=sys.stderr)
            continue

        # Run inference
        with torch.no_grad():
            action_idx, timing_idx, priority = model.predict_greedy(features_tensor)

        predicted_canonical = MODEL_TO_CANONICAL.get(action_idx, 'UNKNOWN')
        correct = (predicted_canonical == ground_truth_canonical)

        # Parse date
        email_date = parse_date(email.get('date', ''))

        results.append(InferenceResult(
            email_id=email.get('message_id', ''),
            date=email_date,
            ground_truth=ground_truth,
            ground_truth_canonical=ground_truth_canonical,
            predicted_action=action_idx,
            predicted_action_name=ACTION_NAMES[action_idx],
            predicted_canonical=predicted_canonical,
            predicted_timing=timing_idx,
            predicted_priority=priority,
            correct=correct,
        ))

    return results


def print_results(stats: InferenceStats, verbose: bool = False) -> None:
    """Print inference results in a formatted way."""
    print()
    print("=" * 60)
    print(f"{stats.mode.upper()} RESULTS")
    print("=" * 60)
    print()
    print(f"Total: {stats.total_emails:,} emails")
    print(f"Accuracy: {stats.accuracy * 100:.1f}%")
    print()

    # Top-k accuracy (if we tracked probabilities, we'd compute this)
    # For now, just show per-action breakdown

    print("Per-action F1 scores:")
    print("-" * 40)
    for action in CANONICAL_ACTIONS:
        metrics = stats.per_action_metrics.get(action, {})
        f1 = metrics.get('f1', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        support = int(metrics.get('support', 0))

        if support > 0:
            print(f"  {action:12s}: F1={f1:.2f}  P={precision:.2f}  R={recall:.2f}  (n={support})")
    print()

    print("Timing distribution:")
    print("-" * 40)
    for timing, count in sorted(stats.timing_distribution.items(), key=lambda x: -x[1]):
        pct = 100 * count / stats.total_emails if stats.total_emails > 0 else 0
        print(f"  {timing:15s}: {count:6d} ({pct:5.1f}%)")
    print()

    print("Priority statistics:")
    print("-" * 40)
    print(f"  Mean: {stats.priority_stats.get('mean', 0):.3f}")
    print(f"  Std:  {stats.priority_stats.get('std', 0):.3f}")
    print(f"  Min:  {stats.priority_stats.get('min', 0):.3f}")
    print(f"  Max:  {stats.priority_stats.get('max', 0):.3f}")
    print()

    if verbose:
        print("Confusion Matrix:")
        print("-" * 40)
        # Print header
        header = "              " + " ".join(f"{a[:6]:>6s}" for a in CANONICAL_ACTIONS)
        print(header)

        for gt in CANONICAL_ACTIONS:
            row = f"{gt:12s}: "
            row += " ".join(f"{stats.confusion_matrix.get(gt, {}).get(pred, 0):6d}"
                          for pred in CANONICAL_ACTIONS)
            print(row)
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on email data with trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Offline validation on historical data
    python inference.py --model checkpoints/stage3.pt \\
        --data data/emails_labeled.json \\
        --mode offline --cutoff-months 2

    # Simulated online inference on recent data
    python inference.py --model checkpoints/stage3.pt \\
        --data data/emails_labeled.json \\
        --mode online --recent-months 2

    # Quick test run
    python inference.py --model checkpoints/stage3.pt \\
        --data data/emails_labeled.json \\
        --limit 100 --verbose
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=Path,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--data', '-d',
        type=Path,
        required=True,
        help='Path to labeled email data (JSON)'
    )
    parser.add_argument(
        '--mode',
        choices=['offline', 'online', 'all'],
        default='all',
        help='Inference mode: offline (historical), online (recent), or all (default: all)'
    )
    parser.add_argument(
        '--cutoff-months',
        type=int,
        default=2,
        help='Months cutoff for offline/online split (default: 2)'
    )
    parser.add_argument(
        '--user-email',
        type=str,
        default='',
        help='User email address for feature extraction context'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for inference: cpu or cuda (default: cpu)'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        help='Limit number of emails to process'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output path for detailed results (JSON)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output including confusion matrix'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.model.exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    print(f"Loaded {len(emails):,} emails")

    if args.limit:
        emails = emails[:args.limit]
        print(f"Limited to {args.limit} emails")

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device=args.device)
    print(f"Model loaded successfully")

    all_results = []

    if args.mode in ['offline', 'all']:
        print(f"\n--- Offline Mode (emails > {args.cutoff_months} months old) ---")
        offline_emails = filter_emails_by_date(emails, args.cutoff_months, 'offline')
        print(f"Found {len(offline_emails):,} offline emails")

        if offline_emails:
            results = run_inference(model, offline_emails, args.user_email, args.device)
            stats = compute_metrics(results)
            stats.mode = f"Offline (emails > {args.cutoff_months} months old)"
            print_results(stats, verbose=args.verbose)
            all_results.extend(results)

    if args.mode in ['online', 'all']:
        print(f"\n--- Online Mode (last {args.cutoff_months} months) ---")
        online_emails = filter_emails_by_date(emails, args.cutoff_months, 'online')
        print(f"Found {len(online_emails):,} online emails")

        if online_emails:
            results = run_inference(model, online_emails, args.user_email, args.device)
            stats = compute_metrics(results)
            stats.mode = f"Online (last {args.cutoff_months} months)"
            print_results(stats, verbose=args.verbose)
            all_results.extend(results)

    # Save detailed results if requested
    if args.output and all_results:
        output_data = [
            {
                'email_id': r.email_id,
                'date': r.date.isoformat() if r.date else None,
                'ground_truth': r.ground_truth,
                'ground_truth_canonical': r.ground_truth_canonical,
                'predicted_action': r.predicted_action,
                'predicted_action_name': r.predicted_action_name,
                'predicted_canonical': r.predicted_canonical,
                'predicted_timing': r.predicted_timing,
                'predicted_priority': r.predicted_priority,
                'correct': r.correct,
            }
            for r in all_results
        ]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
