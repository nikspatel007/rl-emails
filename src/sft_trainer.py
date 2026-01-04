#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) trainer for email policy network.

Stage 1 of the training pipeline. Trains the policy network using
labeled email actions with cross-entropy loss. Target: 65-70% accuracy.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .policy_network import EmailPolicyNetwork, PolicyConfig


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    input_dim: int = 60
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1

    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Loss weights
    action_loss_weight: float = 1.0
    timing_loss_weight: float = 0.5
    priority_loss_weight: float = 0.3

    # Early stopping
    patience: int = 5
    min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path('checkpoints'))
    save_every: int = 5

    # Device
    device: str = 'auto'


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""
    epoch: int
    train_loss: float
    train_action_acc: float
    train_timing_acc: float
    val_loss: float
    val_action_acc: float
    val_timing_acc: float
    learning_rate: float
    duration_seconds: float


class SFTTrainer:
    """Supervised Fine-Tuning trainer for EmailPolicyNetwork.

    Trains the policy network to predict:
    - Action type (reply_now, reply_later, forward, archive, delete, create_task)
    - Response timing (immediate, same_day, next_day, this_week, when_possible)
    - Priority score (0-1)
    """

    def __init__(
        self,
        config: Optional[SFTConfig] = None,
        model: Optional[EmailPolicyNetwork] = None,
    ):
        self.config = config or SFTConfig()

        # Resolve device
        if self.config.device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.config.device)

        # Initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            policy_config = PolicyConfig(
                input_dim=self.config.input_dim,
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
            )
            self.model = EmailPolicyNetwork(policy_config).to(self.device)

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler: Optional[CosineAnnealingLR] = None

        # Loss functions
        self.action_criterion = nn.CrossEntropyLoss()
        self.timing_criterion = nn.CrossEntropyLoss()
        self.priority_criterion = nn.MSELoss()

        # Training state
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history: list[TrainingMetrics] = []

        # Callbacks
        self.callbacks: list[Callable[[TrainingMetrics], None]] = []

    def compute_loss(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss for a batch.

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        features = batch['features'].to(self.device)
        action_labels = batch['action_label'].to(self.device)
        timing_labels = batch['timing_label'].to(self.device)
        priority_targets = batch['priority_target'].to(self.device)

        # Forward pass
        output = self.model(features)

        # Compute individual losses
        action_loss = self.action_criterion(output.action_logits, action_labels)
        timing_loss = self.timing_criterion(output.timing_logits, timing_labels)
        priority_loss = self.priority_criterion(output.priority.squeeze(), priority_targets)

        # Weighted total
        total_loss = (
            self.config.action_loss_weight * action_loss +
            self.config.timing_loss_weight * timing_loss +
            self.config.priority_loss_weight * priority_loss
        )

        loss_dict = {
            'action_loss': action_loss.item(),
            'timing_loss': timing_loss.item(),
            'priority_loss': priority_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict

    def compute_accuracy(self, batch: dict) -> dict[str, float]:
        """Compute accuracy metrics for a batch."""
        features = batch['features'].to(self.device)
        action_labels = batch['action_label'].to(self.device)
        timing_labels = batch['timing_label'].to(self.device)

        with torch.no_grad():
            output = self.model(features)

        action_preds = output.action_logits.argmax(dim=-1)
        timing_preds = output.timing_logits.argmax(dim=-1)

        action_correct = (action_preds == action_labels).sum().item()
        timing_correct = (timing_preds == timing_labels).sum().item()
        total = len(action_labels)

        return {
            'action_acc': action_correct / total,
            'timing_acc': timing_correct / total,
            'action_correct': action_correct,
            'timing_correct': timing_correct,
            'total': total,
        }

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        action_correct = 0
        timing_correct = 0
        total_samples = 0
        loss_components: dict[str, float] = {}

        for batch in train_loader:
            # Forward and loss
            loss, losses = self.compute_loss(batch)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * len(batch['action_label'])

            acc = self.compute_accuracy(batch)
            action_correct += acc['action_correct']
            timing_correct += acc['timing_correct']
            total_samples += acc['total']

            for k, v in losses.items():
                loss_components[k] = loss_components.get(k, 0) + v

        # Average
        n_batches = len(train_loader)
        return {
            'loss': total_loss / total_samples,
            'action_acc': action_correct / total_samples,
            'timing_acc': timing_correct / total_samples,
            **{k: v / n_batches for k, v in loss_components.items()},
        }

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        action_correct = 0
        timing_correct = 0
        total_samples = 0

        for batch in val_loader:
            loss, _ = self.compute_loss(batch)
            total_loss += loss.item() * len(batch['action_label'])

            acc = self.compute_accuracy(batch)
            action_correct += acc['action_correct']
            timing_correct += acc['timing_correct']
            total_samples += acc['total']

        return {
            'loss': total_loss / total_samples,
            'action_acc': action_correct / total_samples,
            'timing_acc': timing_correct / total_samples,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> EmailPolicyNetwork:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Whether to print progress

        Returns:
            Trained model
        """
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6,
        )

        # Ensure checkpoint dir exists
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Training on {self.device}")
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Val samples: {len(val_loader.dataset)}")
            print(f"Epochs: {self.config.epochs}")
            print("-" * 60)

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Step scheduler
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            # Record metrics
            duration = time.time() - start_time
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_metrics['loss'],
                train_action_acc=train_metrics['action_acc'],
                train_timing_acc=train_metrics['timing_acc'],
                val_loss=val_metrics['loss'],
                val_action_acc=val_metrics['action_acc'],
                val_timing_acc=val_metrics['timing_acc'],
                learning_rate=current_lr,
                duration_seconds=duration,
            )
            self.history.append(metrics)

            # Callbacks
            for callback in self.callbacks:
                callback(metrics)

            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['action_acc']:.1%} | "
                    f"Val Acc: {val_metrics['action_acc']:.1%} | "
                    f"Time: {duration:.1f}s"
                )

            # Checkpointing
            if val_metrics['action_acc'] > self.best_val_acc + self.config.min_delta:
                self.best_val_acc = val_metrics['action_acc']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                if verbose:
                    print(f"  -> New best: {self.best_val_acc:.1%}")
            else:
                self.patience_counter += 1

            # Periodic save
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')

            # Early stopping
            if self.patience_counter >= self.config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        best_path = self.config.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            self.load_checkpoint('best_model.pt')

        if verbose:
            print("-" * 60)
            print(f"Training complete. Best val accuracy: {self.best_val_acc:.1%}")

        return self.model

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.config.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'epoch': len(self.history),
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.config.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)


def train_sft(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[SFTConfig] = None,
    verbose: bool = True,
) -> EmailPolicyNetwork:
    """Convenience function for SFT training.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        verbose: Whether to print progress

    Returns:
        Trained EmailPolicyNetwork
    """
    trainer = SFTTrainer(config)
    return trainer.train(train_loader, val_loader, verbose=verbose)


if __name__ == '__main__':
    # Quick test with random data
    print("=" * 60)
    print("SFT TRAINER TEST")
    print("=" * 60)

    from torch.utils.data import TensorDataset

    # Generate random training data
    n_train, n_val = 1000, 200
    feature_dim = 60

    train_features = torch.randn(n_train, feature_dim)
    train_actions = torch.randint(0, 6, (n_train,))
    train_timings = torch.randint(0, 5, (n_train,))
    train_priorities = torch.rand(n_train)

    val_features = torch.randn(n_val, feature_dim)
    val_actions = torch.randint(0, 6, (n_val,))
    val_timings = torch.randint(0, 5, (n_val,))
    val_priorities = torch.rand(n_val)

    # Create dataloaders with dict format
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, features, actions, timings, priorities):
            self.features = features
            self.actions = actions
            self.timings = timings
            self.priorities = priorities

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return {
                'features': self.features[idx],
                'action_label': self.actions[idx],
                'timing_label': self.timings[idx],
                'priority_target': self.priorities[idx],
            }

    train_dataset = DictDataset(train_features, train_actions, train_timings, train_priorities)
    val_dataset = DictDataset(val_features, val_actions, val_timings, val_priorities)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Train
    config = SFTConfig(
        epochs=3,
        patience=2,
        checkpoint_dir=Path('/tmp/sft_test'),
    )

    trainer = SFTTrainer(config)
    model = trainer.train(train_loader, val_loader)

    print(f"\nFinal model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training history: {len(trainer.history)} epochs")

    print("\n" + "=" * 60)
    print("SFT Trainer tests passed!")
    print("=" * 60)
