#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) for Email Policy Network.

Stage 1 of the RL training pipeline: Train the policy network on labeled
email-action pairs using cross-entropy loss.

This establishes a strong baseline before RL fine-tuning with GRPO (Stage 3).
Target accuracy: 65-70% on action prediction.

Usage:
    python src/sft_training.py --data data/train.json --val-data data/val.json \
        --epochs 10 --batch-size 64 --output checkpoints/stage_1.pt

    # Test with synthetic data
    python src/sft_training.py --synthetic --epochs 5
"""

import argparse
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .features.combined import CombinedFeatureExtractor, FEATURE_DIMS
    from .policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        create_policy_network,
        NUM_ACTION_TYPES,
    )
except ImportError:
    from features.combined import CombinedFeatureExtractor, FEATURE_DIMS
    from policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        create_policy_network,
        NUM_ACTION_TYPES,
    )


# Action label mapping for 5-class action space
ACTION_TO_IDX = {
    # reply_now (0) - immediate replies
    'REPLIED': 0,
    'COMPOSED': 0,
    'REPLY': 0,
    'REPLY_NOW': 0,
    # reply_later (1) - deferred replies
    'REPLY_LATER': 1,
    # forward (2)
    'FORWARDED': 2,
    'FORWARD': 2,
    # archive (3)
    'ARCHIVED': 3,
    'ARCHIVE': 3,
    'AUTO_FILED': 3,
    'KEPT': 3,
    # delete (4)
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


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 10

    # Learning rate schedule
    warmup_epochs: int = 1
    lr_decay: float = 0.1
    lr_decay_epochs: tuple[int, ...] = (7, 9)

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Class imbalance handling
    use_class_weights: bool = True          # Inverse frequency weighting
    use_focal_loss: bool = False            # Focal loss for hard examples
    focal_gamma: float = 2.0                # Focal loss gamma parameter
    focal_alpha: Optional[list[float]] = None  # Per-class alpha weights
    use_balanced_sampling: bool = False     # Oversample minority classes

    # Logging and checkpointing
    log_every: int = 10
    save_every: int = 1
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "auto"


class SFTDataset(Dataset):
    """Dataset of emails with ground truth action labels."""

    def __init__(
        self,
        emails: list[dict],
        extractor: Optional[CombinedFeatureExtractor] = None,
    ):
        """Initialize dataset with email list.

        Args:
            emails: List of email dicts with 'action' field
            extractor: Feature extractor (created if not provided)
        """
        self.emails = emails
        self.extractor = extractor or CombinedFeatureExtractor()

        # Pre-extract features and labels
        print(f"Extracting features from {len(emails)} emails...")
        self.features = []
        self.labels = []
        self.skipped = 0

        for i, email in enumerate(emails):
            # Extract features
            feature_vec = self.extractor.to_vector(email)
            self.features.append(torch.tensor(feature_vec, dtype=torch.float32))

            # Get ground truth action label
            action = email.get('action', 'KEPT')
            action_idx = ACTION_TO_IDX.get(action.upper(), 3)  # Default: archive
            self.labels.append(action_idx)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(emails)} emails")

        print(f"Feature extraction complete. {len(self.features)} samples ready.")

        # Report label distribution
        self._report_distribution()

    def _report_distribution(self) -> None:
        """Print label distribution for debugging."""
        counts = defaultdict(int)
        for label in self.labels:
            counts[label] += 1

        print("  Label distribution:")
        for idx in sorted(counts.keys()):
            pct = counts[idx] / len(self.labels) * 100
            print(f"    {idx} ({IDX_TO_ACTION[idx]}): {counts[idx]} ({pct:.1f}%)")

    def get_class_weights(self, num_classes: int = NUM_ACTION_TYPES) -> torch.Tensor:
        """Compute inverse frequency class weights for balancing.

        Returns:
            Tensor of shape (num_classes,) with weights inversely proportional to class frequency
        """
        counts = torch.zeros(num_classes)
        for label in self.labels:
            counts[label] += 1

        # Avoid division by zero - use max count for missing classes
        # This effectively gives 0 weight to classes not in training data
        total_samples = len(self.labels)

        # Inverse frequency weighting
        # For classes with 0 samples, use weight of 0 (can't train on them anyway)
        weights = torch.zeros(num_classes)
        for i in range(num_classes):
            if counts[i] > 0:
                # Standard inverse frequency: n_total / (n_classes * n_class_i)
                weights[i] = total_samples / (num_classes * counts[i])
            else:
                # No samples for this class - weight doesn't matter
                weights[i] = 0.0

        # Normalize so average weight is 1.0 (preserves gradient scale)
        non_zero_weights = weights[weights > 0]
        if len(non_zero_weights) > 0:
            weights = weights / non_zero_weights.mean()

        print(f"  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Compute per-sample weights for balanced sampling.

        Returns:
            Tensor of shape (num_samples,) with weight for each sample
        """
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor([class_weights[label].item() for label in self.labels])
        return sample_weights

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.features[idx], self.labels[idx]


def get_device(device_str: str) -> torch.device:
    """Get the appropriate torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate batch of (features, label) tuples."""
    features = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return features, labels


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Focal Loss = -alpha * (1 - p)^gamma * log(p)

    Down-weights easy examples and focuses training on hard negatives.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Per-class weights (tensor of shape num_classes)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Scalar loss if reduction is 'mean' or 'sum', else (N,) tensor
        """
        # Apply label smoothing if needed
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)

            log_probs = F.log_softmax(inputs, dim=-1)
            probs = torch.exp(log_probs)

            # Focal weight: (1 - p)^gamma
            focal_weights = (1 - probs) ** self.gamma

            # Apply alpha weights if provided
            if self.alpha is not None:
                alpha = self.alpha.to(inputs.device)
                alpha_weights = alpha[targets].unsqueeze(1)
                focal_weights = focal_weights * alpha_weights

            loss = -focal_weights * smooth_targets * log_probs
            loss = loss.sum(dim=-1)
        else:
            # Standard focal loss without label smoothing
            log_probs = F.log_softmax(inputs, dim=-1)
            probs = torch.exp(log_probs)

            # Get probability of correct class
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # Focal weight
            focal_weight = (1 - p_t) ** self.gamma

            # Apply alpha weights if provided
            if self.alpha is not None:
                alpha = self.alpha.to(inputs.device)
                alpha_t = alpha.gather(0, targets)
                focal_weight = focal_weight * alpha_t

            loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SFTTrainer:
    """Supervised Fine-Tuning trainer for email policy."""

    def __init__(
        self,
        policy: EmailPolicyNetwork,
        config: Optional[SFTConfig] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """Initialize trainer.

        Args:
            policy: Policy network to train
            config: Training configuration
            class_weights: Optional pre-computed class weights
        """
        self.policy = policy
        self.config = config or SFTConfig()
        self.class_weights = class_weights

        self.device = get_device(self.config.device)
        self.policy.to(self.device)

        # Loss function - will be set up when class weights are available
        self.criterion = None
        self._setup_criterion(class_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = None

        # Metrics tracking
        self.history = []

    def _setup_criterion(self, class_weights: Optional[torch.Tensor] = None) -> None:
        """Set up loss function based on config and class weights.

        Args:
            class_weights: Optional class weights for balancing
        """
        self.class_weights = class_weights

        if self.config.use_focal_loss:
            # Use Focal Loss
            alpha = class_weights if self.config.use_class_weights else None
            if self.config.focal_alpha is not None:
                alpha = torch.tensor(self.config.focal_alpha)

            self.criterion = FocalLoss(
                alpha=alpha,
                gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing,
            )
            print(f"Using Focal Loss (gamma={self.config.focal_gamma})")
        else:
            # Use CrossEntropyLoss with optional class weights
            weight = None
            if self.config.use_class_weights and class_weights is not None:
                weight = class_weights.to(self.device)
                print(f"Using CrossEntropyLoss with class weights")

            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=self.config.label_smoothing,
            )

    def _adjust_learning_rate(self, epoch: int) -> float:
        """Adjust learning rate based on epoch."""
        lr = self.config.learning_rate

        # Warmup
        if epoch < self.config.warmup_epochs:
            lr = lr * (epoch + 1) / self.config.warmup_epochs
        # Step decay
        else:
            for decay_epoch in self.config.lr_decay_epochs:
                if epoch >= decay_epoch:
                    lr = lr * self.config.lr_decay

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
            Dictionary of training metrics
        """
        self.policy.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.policy(features)

            # Compute loss on action logits
            loss = self.criterion(output.action_logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

            self.optimizer.step()

            # Compute accuracy
            preds = output.action_logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()

            epoch_loss += loss.item() * features.size(0)
            epoch_correct += correct
            epoch_samples += features.size(0)

            if (batch_idx + 1) % self.config.log_every == 0:
                batch_acc = correct / features.size(0) * 100
                print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}, acc={batch_acc:.1f}%")

        return {
            'loss': epoch_loss / epoch_samples,
            'accuracy': epoch_correct / epoch_samples,
        }

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
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Per-class accuracy tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                output = self.policy(features)
                loss = self.criterion(output.action_logits, labels)

                preds = output.action_logits.argmax(dim=-1)
                correct = (preds == labels).sum().item()

                total_loss += loss.item() * features.size(0)
                total_correct += correct
                total_samples += features.size(0)

                # Per-class metrics
                for i in range(len(labels)):
                    label = labels[i].item()
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
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'total_samples': total_samples,
            **per_class_acc,
        }

    def train(
        self,
        train_dataset: SFTDataset,
        val_dataset: Optional[SFTDataset] = None,
        num_epochs: Optional[int] = None,
    ) -> list[dict]:
        """Full training loop.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            num_epochs: Override config epochs

        Returns:
            List of per-epoch metrics
        """
        epochs = num_epochs or self.config.epochs
        history = []

        # Set up class weights if needed and not already set
        if self.config.use_class_weights and self.class_weights is None:
            class_weights = train_dataset.get_class_weights()
            self._setup_criterion(class_weights)

        # Set up sampler for balanced sampling
        sampler = None
        shuffle = True
        if self.config.use_balanced_sampling:
            sample_weights = train_dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True,
            )
            shuffle = False  # Sampler handles shuffling
            print("Using balanced sampling (WeightedRandomSampler)")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
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

        print(f"\nStarting SFT training for {epochs} epochs")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
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
                    self.save_checkpoint(checkpoint_dir / 'best_sft.pt')

            history.append(metrics)

            # Print epoch summary
            msg = f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s, lr={lr:.2e})"
            msg += f" | Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}"
            if val_loader is not None:
                msg += f" | Val: acc={val_metrics['accuracy']:.1%}"
            print(msg)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(checkpoint_dir / f'sft_epoch_{epoch + 1}.pt')

        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1%}")
        return history

    def save_checkpoint(self, path: Path, input_dim: Optional[int] = None) -> None:
        """Save model checkpoint.

        Args:
            path: Checkpoint file path
            input_dim: Optional input dimension (for model reconstruction)
        """
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        if input_dim is not None:
            checkpoint['input_dim'] = input_dim
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")


def create_synthetic_dataset(
    num_samples: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Create synthetic email dataset for testing.

    Args:
        num_samples: Number of emails to generate
        seed: Random seed

    Returns:
        List of email dictionaries
    """
    random.seed(seed)

    # Roughly matching Enron distribution
    actions = ['REPLIED', 'FORWARDED', 'ARCHIVED', 'DELETED', 'KEPT']
    action_weights = [0.25, 0.03, 0.45, 0.09, 0.18]

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


def load_emails(data_path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load emails from JSON file.

    Args:
        data_path: Path to JSON file
        limit: Optional limit on number of emails

    Returns:
        List of email dictionaries
    """
    print(f"Loading emails from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    if limit:
        emails = emails[:limit]

    print(f"Loaded {len(emails)} emails")
    return emails


def main():
    parser = argparse.ArgumentParser(
        description='SFT Training for Email Policy (Stage 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on labeled data
    python src/sft_training.py --data data/train.json --val-data data/val.json

    # Train with custom parameters
    python src/sft_training.py --data data/train.json --epochs 20 --lr 5e-4

    # Test with synthetic data
    python src/sft_training.py --synthetic --epochs 5
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
        '--output',
        type=Path,
        default=Path('checkpoints/stage_1.pt'),
        help='Output checkpoint path (default: checkpoints/stage_1.pt)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)',
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

    # Class imbalance options
    parser.add_argument(
        '--no-class-weights',
        action='store_true',
        help='Disable class weighting (enabled by default)',
    )
    parser.add_argument(
        '--focal-loss',
        action='store_true',
        help='Use focal loss instead of cross-entropy',
    )
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Focal loss gamma parameter (default: 2.0)',
    )
    parser.add_argument(
        '--balanced-sampling',
        action='store_true',
        help='Use balanced sampling (oversample minority classes)',
    )
    parser.add_argument(
        '--include-content',
        action='store_true',
        help='Include content embeddings (768-dim sentence transformer)',
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

    # Determine feature dimension
    include_content = args.include_content
    input_dim = FEATURE_DIMS['total_with_content'] if include_content else FEATURE_DIMS['total_base']
    print(f"\nFeature dimension: {input_dim} (content embeddings: {'enabled' if include_content else 'disabled'})")

    # Create datasets with content features if requested
    extractor = CombinedFeatureExtractor(include_content=include_content)
    train_dataset = SFTDataset(train_emails, extractor)
    val_dataset = SFTDataset(val_emails, extractor) if val_emails else None

    # Create policy network with correct input dimension
    policy = create_policy_network(input_dim=input_dim)
    print(f"Policy network: {sum(p.numel() for p in policy.parameters()):,} parameters")

    # Configure trainer with class imbalance options
    config = SFTConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=str(args.output.parent),
        # Class imbalance handling
        use_class_weights=not args.no_class_weights,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        use_balanced_sampling=args.balanced_sampling,
    )

    # Create trainer and train
    trainer = SFTTrainer(policy, config)

    print("\n" + "=" * 60)
    print("STAGE 1: SUPERVISED FINE-TUNING")
    print("=" * 60)

    history = trainer.train(train_dataset, val_dataset)

    # Save final checkpoint with input_dim for model reconstruction
    trainer.save_checkpoint(args.output, input_dim=input_dim)

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
