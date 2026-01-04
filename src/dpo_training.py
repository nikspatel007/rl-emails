#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) Training for Email Policy.

Implements DPO for training email prioritization policies using preference pairs.
This is Stage 4 of the RL training pipeline, targeting 88-90% accuracy.

DPO Key Insight:
- No separate reward model needed during training
- Directly optimize policy on preference pairs
- Loss = -log(sigmoid(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))))
- Simpler than RLHF/GRPO while achieving similar results

Reference: Rafailov et al. "Direct Preference Optimization" (2023)

Usage:
    python src/dpo_training.py --data data/train.json --val-data data/val.json
    python src/dpo_training.py --data data/train.json --policy-checkpoint checkpoints/best_sft.pt
"""

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .features.combined import CombinedFeatureExtractor
    from .policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        create_policy_network,
        NUM_ACTION_TYPES,
    )
    from .sft_training import SFTConfig
except ImportError:
    from features.combined import CombinedFeatureExtractor
    from policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        create_policy_network,
        NUM_ACTION_TYPES,
    )
    from sft_training import SFTConfig


# Action label mapping matching sft_training.py
ACTION_TO_IDX = {
    'REPLIED': 0,
    'COMPOSED': 0,
    'REPLY': 0,
    'REPLY_NOW': 0,
    'REPLY_LATER': 1,
    'FORWARDED': 2,
    'FORWARD': 2,
    'ARCHIVED': 3,
    'ARCHIVE': 3,
    'AUTO_FILED': 3,
    'KEPT': 3,
    'DELETED': 4,
    'DELETE': 4,
    'JUNK': 4,
}

IDX_TO_ACTION = {
    0: 'reply_now',
    1: 'reply_later',
    2: 'forward',
    3: 'archive',
    4: 'delete',
}

# Action priority for preference ordering (higher = more preferred)
ACTION_PRIORITY = {
    0: 4,  # reply_now - highest priority
    1: 3,  # reply_later
    2: 2,  # forward
    3: 1,  # archive
    4: 0,  # delete - lowest priority
}


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    # DPO parameters
    beta: float = 0.1  # Temperature for DPO loss (lower = more conservative)
    reference_free: bool = False  # If True, skip reference policy (simpler but less stable)
    label_smoothing: float = 0.0  # Label smoothing for preferences

    # Optimization
    learning_rate: float = 5e-5  # Lower than SFT for fine-tuning
    weight_decay: float = 0.01
    batch_size: int = 32  # Smaller batch for preference pairs
    epochs: int = 5
    warmup_epochs: int = 1
    max_grad_norm: float = 1.0

    # Preference pair generation
    min_priority_gap: int = 1  # Minimum action priority gap to form pair
    max_pairs_per_email: int = 5  # Maximum preference pairs per email
    use_margin_weighting: bool = True  # Weight loss by preference margin

    # Regularization
    sft_weight: float = 0.0  # Optional SFT loss to prevent forgetting

    # Logging and checkpointing
    log_every: int = 10
    save_every: int = 1
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "auto"


class PreferencePairDataset(Dataset):
    """Dataset of preference pairs for DPO training.

    Each sample is (features, chosen_action, rejected_action, margin).
    """

    def __init__(
        self,
        emails: list[dict],
        extractor: Optional[CombinedFeatureExtractor] = None,
        min_priority_gap: int = 1,
        max_pairs_per_email: int = 5,
        seed: int = 42,
    ):
        """Initialize preference pair dataset.

        Args:
            emails: List of email dicts with 'action' field
            extractor: Feature extractor (created if not provided)
            min_priority_gap: Minimum priority gap to form pair
            max_pairs_per_email: Maximum pairs per email
            seed: Random seed for pair sampling
        """
        self.emails = emails
        self.extractor = extractor or CombinedFeatureExtractor()
        self.min_priority_gap = min_priority_gap
        self.max_pairs_per_email = max_pairs_per_email

        random.seed(seed)

        # Extract features and actions
        print(f"Extracting features from {len(emails)} emails...")
        self.features = []
        self.actions = []

        for i, email in enumerate(emails):
            feature_vec = self.extractor.to_vector(email)
            self.features.append(torch.tensor(feature_vec, dtype=torch.float32))

            action = email.get('action', 'KEPT')
            action_idx = ACTION_TO_IDX.get(action.upper(), 3)
            self.actions.append(action_idx)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(emails)} emails")

        print(f"Feature extraction complete. {len(self.features)} samples ready.")

        # Generate preference pairs
        self.pairs = self._generate_pairs()
        print(f"Generated {len(self.pairs)} preference pairs")

        self._report_distribution()

    def _generate_pairs(self) -> list[tuple[int, int, int, int]]:
        """Generate preference pairs from labeled emails.

        Returns:
            List of (email_idx, chosen_action, rejected_action, margin)
        """
        pairs = []

        # Group emails by action for efficient pairing
        action_groups = defaultdict(list)
        for idx, action in enumerate(self.actions):
            action_groups[action].append(idx)

        # For each email, generate pairs comparing its action to lower-priority actions
        for email_idx, true_action in enumerate(self.actions):
            true_priority = ACTION_PRIORITY[true_action]

            # The true action is "chosen", sample rejected actions from lower priority
            rejected_candidates = []
            for action_idx in range(NUM_ACTION_TYPES):
                action_priority = ACTION_PRIORITY[action_idx]
                gap = true_priority - action_priority

                if gap >= self.min_priority_gap and action_idx != true_action:
                    rejected_candidates.append((action_idx, gap))

            # Sample up to max_pairs_per_email rejected actions
            if rejected_candidates:
                sample_size = min(self.max_pairs_per_email, len(rejected_candidates))
                sampled = random.sample(rejected_candidates, sample_size)

                for rejected_action, margin in sampled:
                    pairs.append((email_idx, true_action, rejected_action, margin))

        return pairs

    def _report_distribution(self) -> None:
        """Print pair distribution for debugging."""
        chosen_counts = defaultdict(int)
        rejected_counts = defaultdict(int)
        margin_counts = defaultdict(int)

        for _, chosen, rejected, margin in self.pairs:
            chosen_counts[chosen] += 1
            rejected_counts[rejected] += 1
            margin_counts[margin] += 1

        print("  Chosen action distribution:")
        for idx in sorted(chosen_counts.keys()):
            pct = chosen_counts[idx] / len(self.pairs) * 100
            print(f"    {IDX_TO_ACTION[idx]}: {chosen_counts[idx]} ({pct:.1f}%)")

        print("  Rejected action distribution:")
        for idx in sorted(rejected_counts.keys()):
            pct = rejected_counts[idx] / len(self.pairs) * 100
            print(f"    {IDX_TO_ACTION[idx]}: {rejected_counts[idx]} ({pct:.1f}%)")

        print("  Margin distribution:")
        for margin in sorted(margin_counts.keys()):
            pct = margin_counts[margin] / len(self.pairs) * 100
            print(f"    gap={margin}: {margin_counts[margin]} ({pct:.1f}%)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int, int]:
        """Get a preference pair.

        Returns:
            Tuple of (features, chosen_action, rejected_action, margin, true_action)
        """
        email_idx, chosen, rejected, margin = self.pairs[idx]
        return self.features[email_idx], chosen, rejected, margin, self.actions[email_idx]


def get_device(device_str: str) -> torch.device:
    """Get the appropriate torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def collate_fn(batch: list) -> tuple:
    """Collate batch of preference pairs."""
    features = torch.stack([item[0] for item in batch])
    chosen = torch.tensor([item[1] for item in batch], dtype=torch.long)
    rejected = torch.tensor([item[2] for item in batch], dtype=torch.long)
    margins = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    true_actions = torch.tensor([item[4] for item in batch], dtype=torch.long)
    return features, chosen, rejected, margins, true_actions


class DPOTrainer:
    """DPO Trainer for email policy optimization.

    Implements Direct Preference Optimization:
    1. Compute log probabilities under policy and reference policy
    2. DPO loss = -log(sigmoid(β * (log_ratio_chosen - log_ratio_rejected)))
    3. Where log_ratio = log π(y|x) - log π_ref(y|x)
    """

    def __init__(
        self,
        policy: EmailPolicyNetwork,
        config: Optional[DPOConfig] = None,
    ):
        """Initialize DPO trainer.

        Args:
            policy: Policy network to train (will be modified in-place)
            config: DPO training configuration
        """
        self.policy = policy
        self.config = config or DPOConfig()

        self.device = get_device(self.config.device)
        self.policy.to(self.device)

        # Create reference policy (frozen copy)
        self.ref_policy = create_policy_network()
        self.ref_policy.load_state_dict(policy.state_dict())
        self.ref_policy.to(self.device)
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Metrics tracking
        self.history = []

    def _get_log_probs(
        self,
        policy: EmailPolicyNetwork,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probabilities for actions under a policy.

        Args:
            policy: Policy network
            features: Input features (batch, feature_dim)
            actions: Action indices (batch,)

        Returns:
            Log probabilities (batch,)
        """
        output = policy(features)
        log_probs = F.log_softmax(output.action_logits, dim=-1)
        return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    def compute_dpo_loss(
        self,
        features: torch.Tensor,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
        margins: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss.

        Args:
            features: Input features (batch, feature_dim)
            chosen: Chosen action indices (batch,)
            rejected: Rejected action indices (batch,)
            margins: Optional preference margins for weighting

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get log probs under current policy
        pi_logprob_chosen = self._get_log_probs(self.policy, features, chosen)
        pi_logprob_rejected = self._get_log_probs(self.policy, features, rejected)

        if self.config.reference_free:
            # Simplified DPO without reference policy
            log_ratio_chosen = pi_logprob_chosen
            log_ratio_rejected = pi_logprob_rejected
        else:
            # Get log probs under reference policy
            with torch.no_grad():
                ref_logprob_chosen = self._get_log_probs(self.ref_policy, features, chosen)
                ref_logprob_rejected = self._get_log_probs(self.ref_policy, features, rejected)

            # Compute log ratios
            log_ratio_chosen = pi_logprob_chosen - ref_logprob_chosen
            log_ratio_rejected = pi_logprob_rejected - ref_logprob_rejected

        # DPO loss: -log(sigmoid(β * (log_ratio_chosen - log_ratio_rejected)))
        logits = self.config.beta * (log_ratio_chosen - log_ratio_rejected)

        if self.config.label_smoothing > 0:
            # Label smoothing for preferences
            loss = (
                (1 - self.config.label_smoothing) * F.logsigmoid(logits) +
                self.config.label_smoothing * F.logsigmoid(-logits)
            )
            loss = -loss
        else:
            loss = -F.logsigmoid(logits)

        # Optional margin weighting
        if self.config.use_margin_weighting and margins is not None:
            # Higher margin = higher weight (more confident preference)
            weights = margins / margins.mean()
            loss = loss * weights

        loss = loss.mean()

        # Compute accuracy (how often we prefer chosen over rejected)
        with torch.no_grad():
            accuracy = (logits > 0).float().mean().item()
            chosen_reward = log_ratio_chosen.mean().item()
            rejected_reward = log_ratio_rejected.mean().item()

        metrics = {
            'dpo_loss': loss.item(),
            'accuracy': accuracy,
            'chosen_reward': chosen_reward,
            'rejected_reward': rejected_reward,
            'reward_margin': chosen_reward - rejected_reward,
        }

        return loss, metrics

    def compute_sft_loss(
        self,
        features: torch.Tensor,
        true_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SFT loss for regularization.

        Args:
            features: Input features (batch, feature_dim)
            true_actions: Ground truth action indices (batch,)

        Returns:
            SFT loss scalar
        """
        output = self.policy(features)
        return F.cross_entropy(output.action_logits, true_actions)

    def _adjust_learning_rate(self, epoch: int) -> float:
        """Adjust learning rate with warmup."""
        lr = self.config.learning_rate

        if epoch < self.config.warmup_epochs:
            lr = lr * (epoch + 1) / self.config.warmup_epochs

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average metrics
        """
        self.policy.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0

        for batch_idx, (features, chosen, rejected, margins, true_actions) in enumerate(dataloader):
            features = features.to(self.device)
            chosen = chosen.to(self.device)
            rejected = rejected.to(self.device)
            margins = margins.to(self.device)
            true_actions = true_actions.to(self.device)

            self.optimizer.zero_grad()

            # DPO loss
            loss, metrics = self.compute_dpo_loss(features, chosen, rejected, margins)

            # Optional SFT regularization
            if self.config.sft_weight > 0:
                sft_loss = self.compute_sft_loss(features, true_actions)
                loss = loss + self.config.sft_weight * sft_loss
                metrics['sft_loss'] = sft_loss.item()

            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )

            self.optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1

            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = epoch_metrics['dpo_loss'] / num_batches
                avg_acc = epoch_metrics['accuracy'] / num_batches
                print(f"  Batch {batch_idx + 1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

        return {k: v / num_batches for k, v in epoch_metrics.items()}

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate on validation/test set.

        Args:
            dataloader: Evaluation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()
        total_correct = 0
        total_samples = 0
        total_dpo_loss = 0.0
        num_batches = 0

        # Per-class accuracy tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for features, chosen, rejected, margins, true_actions in dataloader:
                features = features.to(self.device)
                chosen = chosen.to(self.device)
                rejected = rejected.to(self.device)
                margins = margins.to(self.device)
                true_actions = true_actions.to(self.device)

                # DPO metrics
                loss, _ = self.compute_dpo_loss(features, chosen, rejected, margins)
                total_dpo_loss += loss.item()
                num_batches += 1

                # Action prediction accuracy
                output = self.policy(features)
                preds = output.action_logits.argmax(dim=-1)

                correct = (preds == true_actions).sum().item()
                total_correct += correct
                total_samples += features.size(0)

                # Per-class metrics
                for i in range(len(true_actions)):
                    label = true_actions[i].item()
                    pred = preds[i].item()
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1

        # Compute per-class accuracy
        per_class_acc = {}
        for cls in range(NUM_ACTION_TYPES):
            if class_total[cls] > 0:
                per_class_acc[f'acc_{IDX_TO_ACTION[cls]}'] = class_correct[cls] / class_total[cls]
            else:
                per_class_acc[f'acc_{IDX_TO_ACTION[cls]}'] = 0.0

        return {
            'dpo_loss': total_dpo_loss / num_batches if num_batches > 0 else 0.0,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
            'total_samples': total_samples,
            **per_class_acc,
        }

    def train(
        self,
        train_dataset: PreferencePairDataset,
        val_dataset: Optional[PreferencePairDataset] = None,
        num_epochs: Optional[int] = None,
    ) -> list[dict]:
        """Full training loop.

        Args:
            train_dataset: Training dataset of preference pairs
            val_dataset: Optional validation dataset
            num_epochs: Override config epochs

        Returns:
            List of per-epoch metrics
        """
        epochs = num_epochs or self.config.epochs
        history = []

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )

        print(f"\nStarting DPO training for {epochs} epochs")
        print(f"  Train pairs: {len(train_dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Beta: {self.config.beta}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Device: {self.device}")
        print()

        best_val_acc = 0.0
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            start_time = time.time()

            # Adjust learning rate
            lr = self._adjust_learning_rate(epoch)

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - start_time

            metrics = {
                'epoch': epoch + 1,
                'time': epoch_time,
                'lr': lr,
                **{f'train_{k}': v for k, v in train_metrics.items()},
            }

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint(checkpoint_dir / 'best_dpo.pt')

            history.append(metrics)

            # Print epoch summary
            msg = f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s, lr={lr:.2e})"
            msg += f" | Train: loss={train_metrics['dpo_loss']:.4f}"
            msg += f", acc={train_metrics['accuracy']:.1%}"
            if val_loader is not None:
                msg += f" | Val: acc={val_metrics['accuracy']:.1%}"
            print(msg)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(checkpoint_dir / f'dpo_epoch_{epoch + 1}.pt')

        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1%}")
        return history

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'ref_model_state_dict': self.ref_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"  Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")


def load_emails(data_path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load emails from JSON file."""
    print(f"Loading emails from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    if limit:
        emails = emails[:limit]

    print(f"Loaded {len(emails)} emails")
    return emails


def create_synthetic_dataset(
    num_samples: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Create synthetic email dataset for testing."""
    random.seed(seed)

    # Match Enron distribution with 5-class actions
    actions = ['REPLY_NOW', 'REPLY_LATER', 'FORWARD', 'ARCHIVE', 'DELETE']
    action_weights = [0.01, 0.30, 0.05, 0.50, 0.14]

    subjects = [
        "URGENT: Need your approval",
        "FYI: Weekly report",
        "RE: Project status update",
        "Meeting tomorrow at 3pm",
        "Question about the budget",
        "Action required: Review document",
        "Quick question",
        "Following up on our discussion",
        "Please review and respond",
        "FW: Customer inquiry",
    ]

    emails = []
    for i in range(num_samples):
        action = random.choices(actions, weights=action_weights)[0]
        subject = random.choice(subjects)

        email = {
            'from': f'user{random.randint(1, 100)}@example.com',
            'to': 'you@example.com',
            'subject': subject,
            'body': f"Email body content {i}. " * random.randint(1, 5),
            'action': action,
        }
        emails.append(email)

    return emails


def main():
    parser = argparse.ArgumentParser(
        description='DPO Training for Email Policy (Stage 4)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on labeled data with SFT checkpoint
    python src/dpo_training.py --data data/train.json --val-data data/val.json \\
        --policy-checkpoint checkpoints/best_sft.pt

    # Train with custom parameters
    python src/dpo_training.py --data data/train.json --epochs 10 --beta 0.1

    # Test with synthetic data
    python src/dpo_training.py --synthetic --epochs 3
        """,
    )
    parser.add_argument(
        '--data',
        type=Path,
        help='Path to training data JSON',
    )
    parser.add_argument(
        '--val-data',
        type=Path,
        help='Path to validation data JSON',
    )
    parser.add_argument(
        '--policy-checkpoint',
        type=Path,
        help='Path to initial policy checkpoint (from SFT)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('checkpoints/stage_4.pt'),
        help='Output checkpoint path (default: checkpoints/stage_4.pt)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate (default: 5e-5)',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.1,
        help='DPO beta (temperature) parameter (default: 0.1)',
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of training samples',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use (default: auto)',
    )
    parser.add_argument(
        '--min-gap',
        type=int,
        default=1,
        help='Minimum priority gap for preference pairs (default: 1)',
    )
    parser.add_argument(
        '--sft-weight',
        type=float,
        default=0.0,
        help='Weight for SFT regularization loss (default: 0.0)',
    )

    args = parser.parse_args()

    # Load or create data
    if args.synthetic or args.data is None:
        print("Using synthetic dataset for testing...")
        train_emails = create_synthetic_dataset(num_samples=args.limit or 1000)
        val_emails = create_synthetic_dataset(num_samples=200, seed=123)
    else:
        train_emails = load_emails(args.data, args.limit)
        val_emails = load_emails(args.val_data) if args.val_data else None

    # Create datasets
    extractor = CombinedFeatureExtractor()
    train_dataset = PreferencePairDataset(
        train_emails,
        extractor,
        min_priority_gap=args.min_gap,
    )
    val_dataset = PreferencePairDataset(
        val_emails,
        extractor,
        min_priority_gap=args.min_gap,
    ) if val_emails else None

    # Create policy network
    policy = create_policy_network()

    # Load SFT checkpoint if provided
    if args.policy_checkpoint and args.policy_checkpoint.exists():
        print(f"Loading policy from {args.policy_checkpoint}")
        checkpoint = torch.load(args.policy_checkpoint, map_location='cpu', weights_only=False)
        policy.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nPolicy network: {sum(p.numel() for p in policy.parameters()):,} parameters")

    # Configure trainer
    config = DPOConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        device=args.device,
        checkpoint_dir=str(args.output.parent),
        min_priority_gap=args.min_gap,
        sft_weight=args.sft_weight,
    )

    # Create trainer and train
    trainer = DPOTrainer(policy, config)

    print("\n" + "=" * 60)
    print("STAGE 4: DIRECT PREFERENCE OPTIMIZATION")
    print("=" * 60)

    history = trainer.train(train_dataset, val_dataset)

    # Save final checkpoint
    trainer.save_checkpoint(args.output)

    # Final evaluation
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
        )
        final_metrics = trainer.evaluate(val_loader)

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        print(f"Validation accuracy: {final_metrics['accuracy']:.1%}")
        print("\nPer-class accuracy:")
        for cls in range(NUM_ACTION_TYPES):
            key = f'acc_{IDX_TO_ACTION[cls]}'
            print(f"  {IDX_TO_ACTION[cls]}: {final_metrics.get(key, 0):.1%}")

    print("\n" + "=" * 60)
    print(f"Training complete! Checkpoint saved to {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
