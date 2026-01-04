#!/usr/bin/env python3
"""Tests for GRPO (Group Relative Policy Optimization) training module."""

import sys
from pathlib import Path

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytest.skip("PyTorch not available", allow_module_level=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from grpo_training import (
    GRPOConfig,
    GRPOTrainer,
    EmailDataset,
    compute_action_reward,
    create_synthetic_dataset,
    get_device,
)
from policy_network import create_policy_network, EmailPolicyNetwork
from reward_model import create_reward_model, EmailRewardModel


class TestGRPOConfig:
    """Tests for GRPOConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GRPOConfig()
        assert config.num_samples == 8
        assert config.clip_epsilon == 0.2
        assert config.entropy_coef == 0.01
        assert config.kl_coef == 0.1
        assert config.learning_rate == 1e-4
        assert config.batch_size == 64
        assert config.epochs == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = GRPOConfig(
            num_samples=4,
            clip_epsilon=0.1,
            learning_rate=1e-5,
            epochs=5,
        )
        assert config.num_samples == 4
        assert config.clip_epsilon == 0.1
        assert config.learning_rate == 1e-5
        assert config.epochs == 5


class TestEmailDataset:
    """Tests for EmailDataset class."""

    def test_create_dataset(self):
        """Test dataset creation from emails."""
        emails = create_synthetic_dataset(num_samples=10, seed=42)
        dataset = EmailDataset(emails)

        assert len(dataset) == 10

        # Test __getitem__
        features, action = dataset[0]
        assert features.shape == (69,)  # Combined feature dimension (with content features)
        assert isinstance(action, str)
        assert action in ['REPLIED', 'FORWARDED', 'ARCHIVED', 'DELETED', 'KEPT']

    def test_empty_dataset(self):
        """Test empty dataset."""
        dataset = EmailDataset([])
        assert len(dataset) == 0


class TestGRPOTrainer:
    """Tests for GRPOTrainer class."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer for testing."""
        policy = create_policy_network()
        reward_model = create_reward_model()
        config = GRPOConfig(
            num_samples=4,
            batch_size=8,
            epochs=1,
            device='cpu',
        )
        return GRPOTrainer(policy, reward_model, config)

    def test_trainer_creation(self, trainer):
        """Test trainer initialization."""
        assert isinstance(trainer.policy, EmailPolicyNetwork)
        assert isinstance(trainer.reward_model, EmailRewardModel)
        assert isinstance(trainer.ref_policy, EmailPolicyNetwork)
        assert trainer.device == torch.device('cpu')

    def test_sample_actions(self, trainer):
        """Test action sampling."""
        features = torch.randn(4, 69)
        actions, timings, log_probs = trainer.sample_actions(features, num_samples=3)

        assert actions.shape == (4, 3)
        assert timings.shape == (4, 3)
        assert log_probs.shape == (4, 3)

        # Actions should be valid indices
        assert (actions >= 0).all() and (actions < 6).all()
        assert (timings >= 0).all() and (timings < 5).all()

        # Log probs should be negative
        assert (log_probs <= 0).all()

    def test_compute_advantages(self, trainer):
        """Test group-relative advantage computation."""
        # Create rewards with known structure
        rewards = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],  # mean = 2.5
            [0.0, 0.0, 0.0, 0.0],  # mean = 0.0
        ])

        # Without normalization
        advantages = trainer.compute_advantages(rewards, normalize=False)

        # Advantages should be reward - group_mean
        expected = torch.tensor([
            [-1.5, -0.5, 0.5, 1.5],
            [0.0, 0.0, 0.0, 0.0],
        ])
        assert torch.allclose(advantages, expected)

    def test_compute_advantages_normalized(self, trainer):
        """Test normalized advantage computation."""
        rewards = torch.randn(8, 4)
        advantages = trainer.compute_advantages(rewards, normalize=True)

        # Normalized advantages should have mean ~0 and std ~1
        assert abs(advantages.mean().item()) < 0.1
        assert abs(advantages.std().item() - 1.0) < 0.1

    def test_train_step(self, trainer):
        """Test single training step."""
        features = torch.randn(8, 69)
        actions = ['REPLIED', 'ARCHIVED', 'DELETED', 'KEPT',
                   'FORWARDED', 'REPLIED', 'ARCHIVED', 'DELETED']

        metrics = trainer.train_step(features, actions)

        assert 'policy_loss' in metrics
        assert 'entropy' in metrics
        assert 'kl' in metrics
        assert 'reward_mean' in metrics
        assert 'advantage_mean' in metrics

        # Entropy should be positive
        assert metrics['entropy'] > 0
        # KL should be small (we just started)
        assert metrics['kl'] >= 0

    def test_policy_loss_computation(self, trainer):
        """Test policy loss with clipping."""
        features = torch.randn(4, 69)
        actions = torch.randint(0, 6, (4, 3))
        timings = torch.randint(0, 5, (4, 3))
        old_log_probs = torch.randn(4, 3) - 2  # Negative log probs
        advantages = torch.randn(4, 3)

        loss, metrics = trainer.compute_policy_loss(
            features, actions, timings, old_log_probs, advantages
        )

        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad
        assert 'policy_loss' in metrics
        assert 'entropy_loss' in metrics
        assert 'kl_loss' in metrics

    def test_evaluate(self, trainer):
        """Test model evaluation."""
        emails = create_synthetic_dataset(num_samples=20, seed=42)
        dataset = EmailDataset(emails)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        metrics = trainer.evaluate(dataloader)

        assert 'accuracy' in metrics
        assert 'avg_reward' in metrics
        assert 'total_samples' in metrics

        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['total_samples'] == 20


class TestComputeActionReward:
    """Tests for action reward computation."""

    def test_reward_with_ground_truth(self):
        """Test reward computation with ground truth matching."""
        reward_model = create_reward_model()
        features = torch.randn(4, 69)

        # Actions that match ground truth should get bonus
        actions = torch.tensor([0, 2, 4, 3])  # reply_now, forward, delete, archive
        timings = torch.zeros(4, dtype=torch.long)
        ground_truth = ['REPLIED', 'FORWARDED', 'DELETED', 'ARCHIVED']

        rewards = compute_action_reward(
            reward_model, features, actions, timings, ground_truth
        )

        assert rewards.shape == (4,)
        # All should have bonus since they match
        # Base reward + 1.0 bonus

    def test_reward_without_ground_truth(self):
        """Test reward computation without ground truth."""
        reward_model = create_reward_model()
        features = torch.randn(4, 69)
        actions = torch.randint(0, 6, (4,))
        timings = torch.zeros(4, dtype=torch.long)

        rewards = compute_action_reward(
            reward_model, features, actions, timings, None
        )

        assert rewards.shape == (4,)

    def test_reward_multi_sample(self):
        """Test reward with multiple samples per state."""
        reward_model = create_reward_model()
        features = torch.randn(4, 69)
        actions = torch.randint(0, 6, (4, 3))  # 3 samples per state
        timings = torch.zeros(4, 3, dtype=torch.long)

        rewards = compute_action_reward(
            reward_model, features, actions, timings, None
        )

        assert rewards.shape == (4, 3)


class TestSyntheticDataset:
    """Tests for synthetic dataset creation."""

    def test_create_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        emails = create_synthetic_dataset(num_samples=100, seed=42)

        assert len(emails) == 100

        # Check structure
        for email in emails:
            assert 'from' in email
            assert 'to' in email
            assert 'subject' in email
            assert 'body' in email
            assert 'action' in email
            assert email['action'] in ['REPLIED', 'FORWARDED', 'ARCHIVED', 'DELETED', 'KEPT']

    def test_synthetic_reproducibility(self):
        """Test that same seed produces same dataset."""
        emails1 = create_synthetic_dataset(num_samples=10, seed=42)
        emails2 = create_synthetic_dataset(num_samples=10, seed=42)

        for e1, e2 in zip(emails1, emails2):
            assert e1['action'] == e2['action']
            assert e1['subject'] == e2['subject']


class TestIntegration:
    """Integration tests for full training pipeline."""

    def test_full_training_loop(self):
        """Test complete training for one epoch."""
        # Create models
        policy = create_policy_network()
        reward_model = create_reward_model()
        config = GRPOConfig(
            num_samples=4,
            batch_size=8,
            epochs=1,
            device='cpu',
            checkpoint_dir='/tmp/grpo_test',
            log_every=100,  # Reduce logging
        )

        trainer = GRPOTrainer(policy, reward_model, config)

        # Create datasets
        train_emails = create_synthetic_dataset(num_samples=32, seed=42)
        val_emails = create_synthetic_dataset(num_samples=16, seed=123)
        train_dataset = EmailDataset(train_emails)
        val_dataset = EmailDataset(val_emails)

        # Train
        history = trainer.train(train_dataset, val_dataset, num_epochs=1)

        assert len(history) == 1
        assert 'train_total_loss' in history[0]
        assert 'val_accuracy' in history[0]

    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        # Create and train
        policy = create_policy_network()
        reward_model = create_reward_model()
        config = GRPOConfig(device='cpu', checkpoint_dir=str(tmp_path))
        trainer = GRPOTrainer(policy, reward_model, config)

        # Save checkpoint
        checkpoint_path = tmp_path / 'test_checkpoint.pt'
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load into new trainer
        new_policy = create_policy_network()
        new_trainer = GRPOTrainer(new_policy, reward_model, config)
        new_trainer.load_checkpoint(checkpoint_path)

        # Verify weights match
        for p1, p2 in zip(policy.parameters(), new_policy.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
