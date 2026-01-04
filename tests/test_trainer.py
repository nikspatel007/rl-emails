#!/usr/bin/env python3
"""Tests for EmailRLTrainer training loop."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from src.trainer import (
    EmailRLTrainer,
    TrainingConfig,
    TrainingState,
    MetricsLogger,
    train_model,
)
from src.policy_network import EmailPolicyNetwork, PolicyConfig
from src.dataset import ACTION_TO_IDX


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with synthetic training data."""
    tmpdir = tempfile.mkdtemp()

    # Create synthetic emails
    def make_emails(n: int, seed: int = 42) -> list:
        torch.manual_seed(seed)
        emails = []
        actions = list(ACTION_TO_IDX.keys())

        for i in range(n):
            action = actions[i % len(actions)]
            emails.append({
                'from': f'sender{i}@example.com',
                'to': 'user@example.com',
                'subject': f'Test email {i}',
                'body': f'This is test email body {i}. ' * 5,
                'action': action,
            })
        return emails

    # Create train/val/test splits
    with open(os.path.join(tmpdir, 'train.json'), 'w') as f:
        json.dump(make_emails(100, seed=42), f)
    with open(os.path.join(tmpdir, 'val.json'), 'w') as f:
        json.dump(make_emails(20, seed=43), f)
    with open(os.path.join(tmpdir, 'test.json'), 'w') as f:
        json.dump(make_emails(20, seed=44), f)

    yield tmpdir

    # Cleanup
    shutil.rmtree(tmpdir)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.batch_size == 64
        assert config.learning_rate == 3e-4
        assert config.num_epochs == 100
        assert config.mode == 'ppo'

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=1e-4,
            num_epochs=10,
            mode='supervised',
        )
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 10
        assert config.mode == 'supervised'


class TestTrainingState:
    """Tests for TrainingState."""

    def test_initial_state(self):
        """Test initial state values."""
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_val_loss == float('inf')
        assert len(state.train_losses) == 0


class TestMetricsLogger:
    """Tests for MetricsLogger."""

    def test_log_creation(self, temp_output_dir):
        """Test that logger creates log file."""
        logger = MetricsLogger(temp_output_dir)
        logger.log({'loss': 0.5, 'accuracy': 0.8}, step=1)

        # Check log file exists
        log_files = list(Path(temp_output_dir).glob('train_*.jsonl'))
        assert len(log_files) == 1

        # Check log content
        with open(log_files[0]) as f:
            record = json.loads(f.readline())
            assert record['loss'] == 0.5
            assert record['accuracy'] == 0.8
            assert record['step'] == 1


class TestEmailRLTrainer:
    """Tests for EmailRLTrainer."""

    def test_trainer_initialization(self, temp_data_dir, temp_output_dir):
        """Test trainer initialization."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=1,
            batch_size=16,
        )
        trainer = EmailRLTrainer(config)

        assert trainer.policy is not None
        assert trainer.optimizer is not None
        assert trainer.device is not None

    def test_setup_data(self, temp_data_dir, temp_output_dir):
        """Test data loading."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=1,
            batch_size=16,
        )
        trainer = EmailRLTrainer(config)
        trainer.setup_data()

        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.test_loader is not None
        assert len(trainer.train_loader.dataset) == 100

    def test_supervised_step(self, temp_data_dir, temp_output_dir):
        """Test single supervised training step."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=1,
            batch_size=16,
            mode='supervised',
        )
        trainer = EmailRLTrainer(config)
        trainer.setup_data()

        # Get a batch
        features, labels = next(iter(trainer.train_loader))
        features = features.to(trainer.device)
        labels = labels.to(trainer.device)

        # Run supervised step
        metrics = trainer.supervised_step(features, labels)

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_ppo_step(self, temp_data_dir, temp_output_dir):
        """Test single PPO training step."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=1,
            batch_size=16,
            mode='ppo',
            ppo_epochs=2,  # Reduce for faster test
        )
        trainer = EmailRLTrainer(config)
        trainer.setup_data()

        # Get a batch
        features, labels = next(iter(trainer.train_loader))
        features = features.to(trainer.device)
        labels = labels.to(trainer.device)

        # Run PPO step
        metrics = trainer.ppo_step(features, labels)

        assert 'loss' in metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics
        assert 'reward_mean' in metrics

    def test_compute_reward(self, temp_data_dir, temp_output_dir):
        """Test reward computation."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
        )
        trainer = EmailRLTrainer(config)

        # Test exact match
        predicted = torch.tensor([0, 1, 2])
        true = torch.tensor([0, 1, 2])
        priorities = torch.tensor([0.5, 0.5, 0.5])
        rewards = trainer.compute_reward(predicted, true, priorities)
        assert torch.all(rewards == 1.0)

        # Test mismatch
        predicted = torch.tensor([0, 1, 2])
        true = torch.tensor([1, 2, 3])
        rewards = trainer.compute_reward(predicted, true, priorities)
        assert torch.all(rewards <= 0.5)

    def test_train_epoch(self, temp_data_dir, temp_output_dir):
        """Test full training epoch."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=1,
            batch_size=32,
            mode='supervised',
            log_every_n_steps=1,
        )
        trainer = EmailRLTrainer(config)
        trainer.setup_data()

        metrics = trainer.train_epoch()

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert trainer.state.global_step > 0

    def test_evaluate(self, temp_data_dir, temp_output_dir):
        """Test evaluation."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=1,
            batch_size=16,
        )
        trainer = EmailRLTrainer(config)
        trainer.setup_data()

        metrics = trainer.evaluate(trainer.val_loader)

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_save_load_checkpoint(self, temp_data_dir, temp_output_dir):
        """Test checkpoint save/load."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=1,
        )
        trainer = EmailRLTrainer(config)
        trainer.setup_data()

        # Modify state
        trainer.state.epoch = 5
        trainer.state.best_val_loss = 0.123

        # Save
        trainer.save_checkpoint('test.pt')
        checkpoint_path = Path(config.checkpoint_dir) / 'test.pt'
        assert checkpoint_path.exists()

        # Create new trainer and load
        trainer2 = EmailRLTrainer(config)
        trainer2.setup_data()
        trainer2.load_checkpoint('test.pt')

        assert trainer2.state.epoch == 5
        assert trainer2.state.best_val_loss == 0.123

    def test_full_training_supervised(self, temp_data_dir, temp_output_dir):
        """Test full supervised training run."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=2,
            batch_size=32,
            mode='supervised',
            early_stop_patience=100,  # Disable early stopping
            log_every_n_steps=10,
        )
        trainer = EmailRLTrainer(config)
        trainer.train()

        # Check training completed (epoch count = number of epochs run)
        assert len(trainer.state.train_losses) >= 1
        assert len(trainer.state.val_losses) >= 1
        assert trainer.state.global_step > 0

        # Check checkpoint exists
        assert Path(config.checkpoint_dir, 'best.pt').exists()

    def test_full_training_ppo(self, temp_data_dir, temp_output_dir):
        """Test full PPO training run."""
        config = TrainingConfig(
            data_dir=temp_data_dir,
            checkpoint_dir=os.path.join(temp_output_dir, 'checkpoints'),
            log_dir=os.path.join(temp_output_dir, 'logs'),
            num_epochs=2,
            batch_size=32,
            mode='ppo',
            ppo_epochs=2,
            early_stop_patience=100,
            log_every_n_steps=10,
        )
        trainer = EmailRLTrainer(config)
        trainer.train()

        # Check training completed
        assert len(trainer.state.train_losses) >= 1
        assert trainer.state.global_step > 0
        assert Path(config.checkpoint_dir, 'best.pt').exists()


class TestTrainModelFunction:
    """Tests for train_model convenience function."""

    def test_train_model(self, temp_data_dir, temp_output_dir):
        """Test train_model function."""
        trainer = train_model(
            data_dir=temp_data_dir,
            output_dir=temp_output_dir,
            mode='supervised',
            num_epochs=1,
            batch_size=32,
        )

        assert trainer is not None
        assert trainer.state.epoch >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
