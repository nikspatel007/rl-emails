#!/usr/bin/env python3
"""PPO Training loop for email policy network.

Implements a training loop with:
- PPO-style policy gradient with clipped objective
- Behavioral cloning mode for supervised pre-training
- Checkpointing (save/resume)
- Logging (file-based and optional tensorboard)
- Apple Silicon (MPS) support
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .policy_network import (
    EmailPolicyNetwork,
    PolicyConfig,
    create_policy_network,
)
from .dataset import (
    EmailDataset,
    create_dataloaders,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    NUM_ACTIONS,
)


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    # Data
    data_dir: str = 'data'
    batch_size: int = 64
    precompute_features: bool = True

    # Model
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Training mode
    mode: str = 'ppo'  # 'ppo' or 'supervised'

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5

    # PPO specific
    ppo_epochs: int = 4  # Number of PPO update passes per batch
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99

    # Training schedule
    num_epochs: int = 100
    warmup_epochs: int = 5
    early_stop_patience: int = 10

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3

    # Logging
    log_dir: str = 'logs'
    log_every_n_steps: int = 50
    use_tensorboard: bool = False

    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'

    # Reproducibility
    seed: int = 42


@dataclass
class TrainingState:
    """Mutable training state for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    best_val_accuracy: float = 0.0
    epochs_without_improvement: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)


class MetricsLogger:
    """Simple file-based metrics logger."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f'train_{datetime.now():%Y%m%d_%H%M%S}.jsonl'
        self.tensorboard_writer = None

    def enable_tensorboard(self) -> None:
        """Enable tensorboard logging if available."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(self.log_dir / 'tensorboard')
        except ImportError:
            logger.warning("tensorboard not available, using file logging only")

    def log(self, metrics: dict, step: int, prefix: str = '') -> None:
        """Log metrics to file and optionally tensorboard."""
        record = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'prefix': prefix,
            **metrics,
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        if self.tensorboard_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tag = f'{prefix}/{key}' if prefix else key
                    self.tensorboard_writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """Close tensorboard writer if open."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()


class EmailRLTrainer:
    """PPO Trainer for email policy network.

    Supports two training modes:
    - 'supervised': Standard cross-entropy loss for behavioral cloning
    - 'ppo': PPO-style policy gradient with reward based on action matching

    Example:
        >>> config = TrainingConfig(data_dir='data', num_epochs=50)
        >>> trainer = EmailRLTrainer(config)
        >>> trainer.train()
        >>> trainer.save_checkpoint('final.pt')
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()

        # Setup device
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")

        # Setup reproducibility
        self._set_seed(config.seed)

        # Create model
        self.policy = self._create_model()
        self.policy.to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.policy.parameters()):,}")

        # Create optimizer and scheduler
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = None  # Set after dataloaders are created

        # Setup logging
        self.logger = MetricsLogger(config.log_dir)
        if config.use_tensorboard:
            self.logger.enable_tensorboard()

        # Data loaders (lazy initialization)
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        # Class weights for imbalanced data
        self.class_weights: Optional[torch.Tensor] = None

    def _setup_device(self) -> torch.device:
        """Setup compute device with Apple Silicon support."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.config.device)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_model(self) -> EmailPolicyNetwork:
        """Create policy network from config.

        Note: Uses NUM_ACTIONS from dataset (7 classes) rather than the
        architecture doc's 6 action types. The dataset reflects actual
        user behavior labels from the Enron corpus.
        """
        from .policy_network import ActionHead

        policy_config = PolicyConfig(
            input_dim=69,  # CombinedFeatures dimension (with content features)
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
            use_layer_norm=self.config.use_layer_norm,
        )
        model = EmailPolicyNetwork(policy_config)

        # Replace action head to match dataset's 7 classes
        hidden_dim = model.encoder.output_dim
        model.action_head = ActionHead(hidden_dim, num_actions=NUM_ACTIONS)

        return model

    def setup_data(self) -> None:
        """Initialize data loaders."""
        logger.info(f"Loading data from {self.config.data_dir}")

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.config.data_dir,
            batch_size=self.config.batch_size,
            precompute=self.config.precompute_features,
        )

        # Get class weights for imbalanced data
        self.class_weights = self.train_loader.dataset.get_class_weights().to(self.device)

        # Setup scheduler based on training steps
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.01,
        )

        logger.info(f"Train: {len(self.train_loader.dataset)} samples, {len(self.train_loader)} batches")
        logger.info(f"Val: {len(self.val_loader.dataset)} samples")
        logger.info(f"Test: {len(self.test_loader.dataset)} samples")

    def compute_reward(
        self,
        predicted_actions: torch.Tensor,
        true_actions: torch.Tensor,
        priorities: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward signal for PPO training.

        Reward is based on matching ground truth actions from the dataset.

        Args:
            predicted_actions: Predicted action indices (batch,)
            true_actions: Ground truth action indices (batch,)
            priorities: Predicted priority values (batch,)

        Returns:
            Reward tensor (batch,)
        """
        # Base reward: 1.0 for correct action, 0.0 for incorrect
        correct = (predicted_actions == true_actions).float()
        reward = correct

        # Partial credit for "similar" actions
        # REPLIED (0) and KEPT (4) are similar (both require attention)
        # DELETED (2) and ARCHIVED (3) are similar (both are dismissive)
        similar_pairs = [
            (0, 4),  # REPLIED <-> KEPT
            (2, 3),  # DELETED <-> ARCHIVED
            (2, 6),  # DELETED <-> JUNK
            (3, 6),  # ARCHIVED <-> JUNK
        ]
        for a, b in similar_pairs:
            partial_match = (
                ((predicted_actions == a) & (true_actions == b)) |
                ((predicted_actions == b) & (true_actions == a))
            )
            reward = reward + partial_match.float() * 0.5

        # Penalty for missing important actions (false negatives)
        # If true action is REPLIED but we predicted ARCHIVED/DELETED
        important_miss = (
            (true_actions == ACTION_TO_IDX['REPLIED']) &
            ((predicted_actions == ACTION_TO_IDX['ARCHIVED']) |
             (predicted_actions == ACTION_TO_IDX['DELETED']))
        )
        reward = reward - important_miss.float() * 1.0

        return reward

    def ppo_step(
        self,
        features: torch.Tensor,
        true_actions: torch.Tensor,
    ) -> dict:
        """Perform one PPO training step.

        Args:
            features: Input features (batch, 60)
            true_actions: Ground truth actions (batch,)

        Returns:
            Dictionary of loss components and metrics
        """
        batch_size = features.size(0)
        metrics = {}

        # Collect rollout: sample actions from current policy
        with torch.no_grad():
            sample = self.policy.sample_action(features)
            old_action_log_probs = sample.action_log_prob.detach()
            old_timing_log_probs = sample.timing_log_prob.detach()
            sampled_actions = sample.action_idx.detach()
            sampled_timing = sample.timing_idx.detach()

            # Compute rewards
            rewards = self.compute_reward(
                sampled_actions,
                true_actions,
                sample.priority.detach(),
            )

            # Get value estimates
            old_values = self.policy.get_value(features).detach()

        # PPO update: multiple epochs over the same batch
        for _ in range(self.config.ppo_epochs):
            # Evaluate current policy on collected actions
            new_action_log_probs, new_timing_log_probs, entropy, values = \
                self.policy.evaluate_actions(features, sampled_actions, sampled_timing)

            # Compute advantages (simple: reward - value baseline)
            advantages = rewards - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss (PPO clipped objective)
            ratio = torch.exp(new_action_log_probs - old_action_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon,
            )
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages,
            ).mean()

            # Value loss
            value_loss = F.mse_loss(values, rewards)

            # Entropy bonus for exploration
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss +
                self.config.value_loss_coef * value_loss +
                self.config.entropy_coef * entropy_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        # Compute accuracy for logging
        with torch.no_grad():
            greedy_actions = self.policy(features).action_logits.argmax(dim=-1)
            accuracy = (greedy_actions == true_actions).float().mean()

        metrics = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'reward_mean': rewards.mean().item(),
            'accuracy': accuracy.item(),
            'advantage_mean': advantages.mean().item(),
        }

        return metrics

    def supervised_step(
        self,
        features: torch.Tensor,
        true_actions: torch.Tensor,
    ) -> dict:
        """Perform supervised learning step (behavioral cloning).

        Args:
            features: Input features (batch, 60)
            true_actions: Ground truth actions (batch,)

        Returns:
            Dictionary of loss components and metrics
        """
        # Forward pass
        output = self.policy(features)

        # Cross-entropy loss for action prediction with class weights
        action_loss = F.cross_entropy(
            output.action_logits,
            true_actions,
            weight=self.class_weights,
        )

        # Total loss (could add auxiliary losses here)
        loss = action_loss

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            predictions = output.action_logits.argmax(dim=-1)
            accuracy = (predictions == true_actions).float().mean()

        return {
            'loss': loss.item(),
            'action_loss': action_loss.item(),
            'accuracy': accuracy.item(),
        }

    def train_epoch(self) -> dict:
        """Train for one epoch.

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.policy.train()
        epoch_metrics = []

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Training step based on mode
            if self.config.mode == 'ppo':
                metrics = self.ppo_step(features, labels)
            else:
                metrics = self.supervised_step(features, labels)

            epoch_metrics.append(metrics)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            self.state.global_step += 1
            if self.state.global_step % self.config.log_every_n_steps == 0:
                self.logger.log(metrics, self.state.global_step, prefix='train')

        # Aggregate epoch metrics
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)

        return avg_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate model on a data loader.

        Args:
            loader: DataLoader to evaluate on

        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            output = self.policy(features)

            # Loss
            loss = F.cross_entropy(output.action_logits, labels)
            total_loss += loss.item() * features.size(0)

            # Predictions
            predictions = output.action_logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += features.size(0)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        # Compute per-class accuracy
        class_correct = {action: 0 for action in ACTION_TO_IDX}
        class_total = {action: 0 for action in ACTION_TO_IDX}

        for pred, label in zip(all_predictions, all_labels):
            action_name = IDX_TO_ACTION[label]
            class_total[action_name] += 1
            if pred == label:
                class_correct[action_name] += 1

        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
        }

        # Add per-class accuracy
        for action in ACTION_TO_IDX:
            if class_total[action] > 0:
                metrics[f'acc_{action.lower()}'] = class_correct[action] / class_total[action]

        return metrics

    def train(self) -> None:
        """Run full training loop."""
        if self.train_loader is None:
            self.setup_data()

        logger.info(f"Starting training: {self.config.num_epochs} epochs, mode={self.config.mode}")
        start_time = time.time()

        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()
            self.state.train_losses.append(train_metrics['loss'])

            # Validate
            val_metrics = self.evaluate(self.val_loader)
            self.state.val_losses.append(val_metrics['loss'])
            self.state.val_accuracies.append(val_metrics['accuracy'])

            epoch_time = time.time() - epoch_start

            # Log epoch summary
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} "
                f"| train_loss: {train_metrics['loss']:.4f} "
                f"| val_loss: {val_metrics['loss']:.4f} "
                f"| val_acc: {val_metrics['accuracy']:.4f} "
                f"| time: {epoch_time:.1f}s"
            )

            self.logger.log(
                {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}},
                step=epoch,
                prefix='epoch',
            )

            # Check for improvement
            if val_metrics['loss'] < self.state.best_val_loss:
                self.state.best_val_loss = val_metrics['loss']
                self.state.best_val_accuracy = val_metrics['accuracy']
                self.state.epochs_without_improvement = 0
                self.save_checkpoint('best.pt')
                logger.info(f"  New best model! val_loss={val_metrics['loss']:.4f}")
            else:
                self.state.epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')

            # Early stopping
            if self.state.epochs_without_improvement >= self.config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time/60:.1f} minutes")
        logger.info(f"Best val_loss: {self.state.best_val_loss:.4f}, val_acc: {self.state.best_val_accuracy:.4f}")

        # Final evaluation on test set
        self.load_checkpoint('best.pt')
        test_metrics = self.evaluate(self.test_loader)
        logger.info(f"Test metrics: {test_metrics}")
        self.logger.log(test_metrics, step=self.state.epoch, prefix='test')

        self.logger.close()

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'config': self.config,
            'state': self.state,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }

        path = checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = Path(self.config.checkpoint_dir) / filename
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.state = checkpoint['state']

        logger.info(f"Loaded checkpoint from {path} (epoch {self.state.epoch})")

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the N most recent."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return

        # Find epoch checkpoints (not 'best.pt')
        epoch_checkpoints = sorted(
            checkpoint_dir.glob('epoch_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Keep only N most recent
        for old_checkpoint in epoch_checkpoints[self.config.keep_n_checkpoints:]:
            old_checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {old_checkpoint}")


def train_model(
    data_dir: str,
    output_dir: str = 'output',
    mode: str = 'supervised',
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device: str = 'auto',
) -> EmailRLTrainer:
    """Convenience function to train a model.

    Args:
        data_dir: Directory containing train.json, val.json, test.json
        output_dir: Directory for checkpoints and logs
        mode: Training mode ('supervised' or 'ppo')
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use

    Returns:
        Trained EmailRLTrainer instance
    """
    config = TrainingConfig(
        data_dir=data_dir,
        checkpoint_dir=os.path.join(output_dir, 'checkpoints'),
        log_dir=os.path.join(output_dir, 'logs'),
        mode=mode,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    trainer = EmailRLTrainer(config)
    trainer.train()

    return trainer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train email policy network')
    parser.add_argument('data_dir', type=str, help='Directory with train/val/test splits')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--mode', choices=['supervised', 'ppo'], default='supervised',
                        help='Training mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda/mps)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    config = TrainingConfig(
        data_dir=args.data_dir,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
        log_dir=os.path.join(args.output_dir, 'logs'),
        mode=args.mode,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    trainer = EmailRLTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
