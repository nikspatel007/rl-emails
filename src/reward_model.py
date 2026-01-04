#!/usr/bin/env python3
"""Reward Model for Email Prioritization.

This module implements a reward model trained on preference pairs to predict
email importance. The reward model is used in Stage 2 of the RL training
pipeline, before GRPO/DPO fine-tuning.

Training approach:
- Generate preference pairs from user actions (REPLY_NOW > DELETE, etc.)
- Train using Bradley-Terry loss: L = -log(sigmoid(r_preferred - r_rejected))
- Output scalar reward values for emails

Compatible with:
- CombinedFeatures (60-dim input from features/combined.py)
- EmailPolicyNetwork (shares encoder architecture)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from .policy_network import FeatureEncoder, PolicyConfig, DEFAULT_FEATURE_DIM
except ImportError:
    from policy_network import FeatureEncoder, PolicyConfig, DEFAULT_FEATURE_DIM


class ActionPriority(IntEnum):
    """Priority ranking based on true user behavior distribution.

    Higher values indicate more common actions in real user behavior.
    This avoids artificial bias toward rare actions (like REPLY_NOW)
    and instead reflects what users actually do with their emails.

    Distribution from Gmail data: DELETE ~85%, REPLY ~14%, FORWARD <1%
    """
    FORWARD = 0     # Least common action
    REPLY_NOW = 1   # Urgent response (rare)
    REPLY_LATER = 2 # Deferred response
    ARCHIVE = 3     # User filed away
    DELETE = 4      # Most common action (~85% of emails)


# Map action strings to priority
ACTION_TO_PRIORITY = {
    'FORWARD': ActionPriority.FORWARD,
    'REPLY_NOW': ActionPriority.REPLY_NOW,
    'REPLY_LATER': ActionPriority.REPLY_LATER,
    'ARCHIVE': ActionPriority.ARCHIVE,
    'DELETE': ActionPriority.DELETE,
}


class PreferencePair(NamedTuple):
    """A preference pair for reward model training."""
    preferred_features: torch.Tensor  # Shape: (feature_dim,)
    rejected_features: torch.Tensor   # Shape: (feature_dim,)
    margin: float  # Priority difference (can be used for weighted loss)


@dataclass
class RewardConfig:
    """Configuration for EmailRewardModel."""
    input_dim: int = DEFAULT_FEATURE_DIM
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    use_layer_norm: bool = True
    activation: str = 'relu'
    use_margin_loss: bool = False  # Whether to use margin in Bradley-Terry loss


class EmailRewardModel(nn.Module):
    """Reward model for email prioritization.

    Takes email features and outputs a scalar reward value representing
    how important/valuable the email is. Trained on preference pairs
    derived from user actions.

    Example usage:
        >>> config = RewardConfig(input_dim=60)
        >>> reward_model = EmailRewardModel(config)
        >>> features = torch.randn(32, 60)  # batch of 32 emails
        >>> rewards = reward_model(features)
        >>> print(rewards.shape)  # (32, 1)

        # Compute preference loss
        >>> preferred = torch.randn(16, 60)
        >>> rejected = torch.randn(16, 60)
        >>> loss = reward_model.preference_loss(preferred, rejected)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__()
        self.config = config or RewardConfig()

        # Convert RewardConfig to PolicyConfig for encoder
        encoder_config = PolicyConfig(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            use_layer_norm=self.config.use_layer_norm,
            activation=self.config.activation,
        )

        # Shared encoder with policy network architecture
        self.encoder = FeatureEncoder(encoder_config)
        hidden_dim = self.encoder.output_dim

        # Reward head - outputs scalar reward value
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reward for email features.

        Args:
            x: Input features of shape (batch, input_dim)

        Returns:
            Reward values of shape (batch, 1)
        """
        hidden = self.encoder(x)
        return self.reward_head(hidden)

    def get_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Get reward values (convenience method).

        Args:
            x: Input features of shape (batch, input_dim) or (input_dim,)

        Returns:
            Reward values of shape (batch,) or scalar
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.forward(x).squeeze()
        return self.forward(x).squeeze(-1)

    def preference_loss(
        self,
        preferred: torch.Tensor,
        rejected: torch.Tensor,
        margin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Bradley-Terry preference loss.

        Loss = -log(sigmoid(r_preferred - r_rejected))

        If margin is provided and use_margin_loss is True:
        Loss = -log(sigmoid((r_preferred - r_rejected) / margin))

        Args:
            preferred: Features of preferred emails, shape (batch, input_dim)
            rejected: Features of rejected emails, shape (batch, input_dim)
            margin: Optional priority difference, shape (batch,)

        Returns:
            Scalar loss value
        """
        r_preferred = self.forward(preferred)  # (batch, 1)
        r_rejected = self.forward(rejected)    # (batch, 1)

        diff = r_preferred - r_rejected  # (batch, 1)

        if self.config.use_margin_loss and margin is not None:
            # Scale by margin (larger margin = more confident preference)
            margin = margin.unsqueeze(-1).clamp(min=1.0)
            diff = diff / margin

        # Bradley-Terry loss: -log(sigmoid(diff))
        loss = -F.logsigmoid(diff)

        return loss.mean()

    def preference_accuracy(
        self,
        preferred: torch.Tensor,
        rejected: torch.Tensor,
    ) -> float:
        """Compute preference prediction accuracy.

        Args:
            preferred: Features of preferred emails
            rejected: Features of rejected emails

        Returns:
            Accuracy as float in [0, 1]
        """
        with torch.no_grad():
            r_preferred = self.forward(preferred)
            r_rejected = self.forward(rejected)
            correct = (r_preferred > r_rejected).float()
            return correct.mean().item()


class PreferencePairDataset(Dataset):
    """Dataset of preference pairs for reward model training.

    Generates preference pairs from labeled emails based on action priority.
    """

    def __init__(
        self,
        features: torch.Tensor,
        actions: list[str],
        min_priority_gap: int = 1,
    ):
        """Initialize preference pair dataset.

        Args:
            features: Email feature vectors, shape (n_emails, feature_dim)
            actions: List of action labels for each email
            min_priority_gap: Minimum priority difference to form a pair
        """
        self.features = features
        self.actions = actions
        self.min_priority_gap = min_priority_gap

        # Precompute all valid pairs
        self.pairs = self._generate_pairs()

    def _generate_pairs(self) -> list[tuple[int, int, int]]:
        """Generate all valid preference pairs.

        Returns:
            List of (preferred_idx, rejected_idx, priority_gap)
        """
        pairs = []
        n = len(self.actions)

        priorities = [
            ACTION_TO_PRIORITY.get(action, ActionPriority.ARCHIVE)
            for action in self.actions
        ]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                gap = priorities[i] - priorities[j]
                if gap >= self.min_priority_gap:
                    pairs.append((i, j, gap))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Get a preference pair.

        Returns:
            Tuple of (preferred_features, rejected_features, margin)
        """
        preferred_idx, rejected_idx, margin = self.pairs[idx]
        return (
            self.features[preferred_idx],
            self.features[rejected_idx],
            float(margin),
        )


def generate_preference_pairs(
    features: torch.Tensor,
    actions: list[str],
    min_priority_gap: int = 1,
    max_pairs_per_email: int = 10,
    seed: Optional[int] = None,
) -> list[PreferencePair]:
    """Generate preference pairs from labeled emails.

    Args:
        features: Email feature vectors, shape (n_emails, feature_dim)
        actions: List of action labels for each email
        min_priority_gap: Minimum priority difference to form a pair
        max_pairs_per_email: Maximum pairs to generate per email
        seed: Random seed for reproducibility

    Returns:
        List of PreferencePair objects
    """
    if seed is not None:
        torch.manual_seed(seed)

    pairs = []
    n = len(actions)

    priorities = [
        ACTION_TO_PRIORITY.get(action, ActionPriority.ARCHIVE)
        for action in actions
    ]

    # Group emails by priority
    priority_groups: dict[int, list[int]] = {}
    for idx, priority in enumerate(priorities):
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(idx)

    # Generate pairs across priority groups
    sorted_priorities = sorted(priority_groups.keys(), reverse=True)

    for i, high_priority in enumerate(sorted_priorities):
        high_indices = priority_groups[high_priority]

        for low_priority in sorted_priorities[i + 1:]:
            gap = high_priority - low_priority
            if gap < min_priority_gap:
                continue

            low_indices = priority_groups[low_priority]

            # Sample pairs
            for high_idx in high_indices:
                # Limit pairs per high-priority email
                sample_size = min(max_pairs_per_email, len(low_indices))
                sampled_low = torch.randperm(len(low_indices))[:sample_size]

                for j in sampled_low:
                    low_idx = low_indices[j.item()]
                    pairs.append(PreferencePair(
                        preferred_features=features[high_idx],
                        rejected_features=features[low_idx],
                        margin=float(gap),
                    ))

    return pairs


class RewardModelTrainer:
    """Trainer for EmailRewardModel.

    Handles training loop, validation, and checkpointing.
    """

    def __init__(
        self,
        model: EmailRewardModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.

        Args:
            model: The reward model to train
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            device: Device to train on (auto-detected if None)
        """
        self.model = model
        self.device = device or self._get_device()
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _get_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding (preferred, rejected, margin)

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for preferred, rejected, margin in dataloader:
            preferred = preferred.to(self.device)
            rejected = rejected.to(self.device)
            margin = margin.to(self.device)

            self.optimizer.zero_grad()

            loss = self.model.preference_loss(preferred, rejected, margin)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_accuracy += self.model.preference_accuracy(preferred, rejected)
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
        }

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate on validation set.

        Args:
            dataloader: DataLoader yielding (preferred, rejected, margin)

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for preferred, rejected, margin in dataloader:
                preferred = preferred.to(self.device)
                rejected = rejected.to(self.device)
                margin = margin.to(self.device)

                loss = self.model.preference_loss(preferred, rejected, margin)

                total_loss += loss.item()
                total_accuracy += self.model.preference_accuracy(preferred, rejected)
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        early_stopping_patience: int = 3,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Full training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            num_epochs: Number of training epochs
            early_stopping_patience: Epochs to wait for improvement
            verbose: Whether to print progress

        Returns:
            List of per-epoch metrics
        """
        history = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_dataloader)

            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
            }

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                metrics['val_loss'] = val_metrics['loss']
                metrics['val_accuracy'] = val_metrics['accuracy']

                # Early stopping check
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            history.append(metrics)

            if verbose:
                msg = f"Epoch {epoch + 1}/{num_epochs}"
                msg += f" | Train Loss: {metrics['train_loss']:.4f}"
                msg += f" | Train Acc: {metrics['train_accuracy']:.4f}"
                if val_dataloader is not None:
                    msg += f" | Val Loss: {metrics['val_loss']:.4f}"
                    msg += f" | Val Acc: {metrics['val_accuracy']:.4f}"
                print(msg)

        return history


def create_reward_model(
    input_dim: int = DEFAULT_FEATURE_DIM,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    **kwargs,
) -> EmailRewardModel:
    """Factory function to create reward model.

    Args:
        input_dim: Dimension of input features
        hidden_dims: Tuple of hidden layer sizes
        **kwargs: Additional RewardConfig parameters

    Returns:
        Configured EmailRewardModel
    """
    config = RewardConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        **kwargs,
    )
    return EmailRewardModel(config)


if __name__ == '__main__':
    # Example usage and testing
    print("=" * 60)
    print("EMAIL REWARD MODEL TEST")
    print("=" * 60)

    # Create model
    config = RewardConfig(input_dim=60, hidden_dims=(256, 128, 64))
    reward_model = EmailRewardModel(config)

    print(f"\nNetwork architecture:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Parameters: {sum(p.numel() for p in reward_model.parameters()):,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 60)

    print(f"\nForward pass (batch_size={batch_size}):")
    rewards = reward_model(x)
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Reward range: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")

    # Test preference loss
    print("\nPreference loss:")
    preferred = torch.randn(16, 60)
    rejected = torch.randn(16, 60)
    loss = reward_model.preference_loss(preferred, rejected)
    print(f"  Loss: {loss.item():.4f}")

    accuracy = reward_model.preference_accuracy(preferred, rejected)
    print(f"  Accuracy: {accuracy:.4f}")

    # Test preference pair generation
    print("\nPreference pair generation:")
    features = torch.randn(100, 60)
    actions = ['REPLY_NOW'] * 20 + ['REPLY_LATER'] * 20 + ['ARCHIVE'] * 30 + ['DELETE'] * 30

    pairs = generate_preference_pairs(features, actions, min_priority_gap=2, seed=42)
    print(f"  Generated {len(pairs)} preference pairs")

    # Test dataset
    print("\nPreferencePairDataset:")
    dataset = PreferencePairDataset(features, actions, min_priority_gap=1)
    print(f"  Dataset size: {len(dataset)}")

    preferred, rejected, margin = dataset[0]
    print(f"  Sample shapes: preferred={preferred.shape}, rejected={rejected.shape}")
    print(f"  Sample margin: {margin}")

    # Test training
    print("\nTraining test (2 epochs):")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    trainer = RewardModelTrainer(reward_model, learning_rate=1e-4)
    history = trainer.train(dataloader, num_epochs=2, verbose=True)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
