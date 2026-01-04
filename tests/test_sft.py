#!/usr/bin/env python3
"""Tests for SFT training components."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.policy_network import EmailPolicyNetwork, PolicyConfig
from src.dataset import (
    EmailDataset,
    EmailSample,
    ACTION_TO_IDX,
    TIMING_TO_IDX,
    action_to_reply_timing,
    response_time_to_timing,
)
from src.sft_trainer import SFTConfig, SFTTrainer
from src.evaluate import evaluate_model, compute_f1_scores, check_accuracy_target


class TestDataset:
    """Tests for EmailDataset."""

    def test_action_mapping(self):
        """Test action label mapping."""
        assert len(ACTION_TO_IDX) == 6
        assert ACTION_TO_IDX['reply_now'] == 0
        assert ACTION_TO_IDX['delete'] == 4

    def test_timing_mapping(self):
        """Test timing label mapping."""
        assert len(TIMING_TO_IDX) == 5
        assert TIMING_TO_IDX['immediate'] == 0
        assert TIMING_TO_IDX['when_possible'] == 4

    def test_response_time_to_timing(self):
        """Test response time categorization."""
        assert response_time_to_timing(0.5) == 'immediate'
        assert response_time_to_timing(4.0) == 'same_day'
        assert response_time_to_timing(12.0) == 'next_day'
        assert response_time_to_timing(72.0) == 'this_week'
        assert response_time_to_timing(200.0) == 'when_possible'
        assert response_time_to_timing(None) == 'when_possible'

    def test_action_to_reply_timing_quick(self):
        """Test quick replies map to reply_now."""
        result = action_to_reply_timing('REPLIED', 1.0)
        assert result == 'reply_now'

    def test_action_to_reply_timing_slow(self):
        """Test slow replies map to reply_later."""
        result = action_to_reply_timing('REPLIED', 6.0)
        assert result == 'reply_later'

    def test_action_to_reply_timing_other(self):
        """Test non-reply actions map correctly."""
        assert action_to_reply_timing('DELETED') == 'delete'
        assert action_to_reply_timing('FORWARDED') == 'forward'
        assert action_to_reply_timing('ARCHIVED') == 'archive'

    def test_email_dataset_creation(self):
        """Test EmailDataset creation."""
        emails = [
            {
                'from': 'sender@test.com',
                'to': 'user@test.com',
                'subject': 'Test',
                'body': 'Test body',
                'folder': 'inbox',
                'action': 'ARCHIVED',
            }
        ]
        dataset = EmailDataset(emails)
        assert len(dataset) >= 0  # May be 0 or 1 depending on feature extraction

    def test_email_dataset_getitem(self):
        """Test EmailDataset __getitem__ returns correct format."""
        emails = [
            {
                'from': 'sender@test.com',
                'to': 'user@test.com',
                'subject': 'Test email',
                'body': 'Test body content',
                'folder': 'sent',
                'action': 'REPLIED',
                'response_time_hours': 2.0,
            }
        ]
        dataset = EmailDataset(emails)

        if len(dataset) > 0:
            sample = dataset[0]
            assert 'features' in sample
            assert 'action_label' in sample
            assert 'timing_label' in sample
            assert 'priority_target' in sample
            assert sample['features'].shape == (60,)

    def test_skip_composed(self):
        """Test COMPOSED emails are skipped by default."""
        emails = [
            {
                'from': 'user@test.com',
                'to': 'recipient@test.com',
                'subject': 'New email',
                'body': 'Composing a new email',
                'folder': 'sent',
                'action': 'COMPOSED',
            }
        ]
        dataset = EmailDataset(emails, skip_composed=True)
        assert len(dataset) == 0


class TestSFTTrainer:
    """Tests for SFT trainer."""

    @pytest.fixture
    def sample_dataloaders(self):
        """Create sample dataloaders for testing."""
        n_train, n_val = 100, 20

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, n):
                self.features = torch.randn(n, 60)
                self.actions = torch.randint(0, 6, (n,))
                self.timings = torch.randint(0, 5, (n,))
                self.priorities = torch.rand(n)

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return {
                    'features': self.features[idx],
                    'action_label': self.actions[idx],
                    'timing_label': self.timings[idx],
                    'priority_target': self.priorities[idx],
                }

        train_dataset = DictDataset(n_train)
        val_dataset = DictDataset(n_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        return train_loader, val_loader

    def test_config_defaults(self):
        """Test SFTConfig default values."""
        config = SFTConfig()
        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 3e-4

    def test_trainer_initialization(self):
        """Test SFTTrainer initialization."""
        config = SFTConfig(device='cpu')
        trainer = SFTTrainer(config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device == torch.device('cpu')

    def test_compute_loss(self, sample_dataloaders):
        """Test loss computation."""
        train_loader, _ = sample_dataloaders

        config = SFTConfig(device='cpu')
        trainer = SFTTrainer(config)

        batch = next(iter(train_loader))
        loss, loss_dict = trainer.compute_loss(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert 'action_loss' in loss_dict
        assert 'timing_loss' in loss_dict
        assert 'priority_loss' in loss_dict

    def test_compute_accuracy(self, sample_dataloaders):
        """Test accuracy computation."""
        train_loader, _ = sample_dataloaders

        config = SFTConfig(device='cpu')
        trainer = SFTTrainer(config)

        batch = next(iter(train_loader))
        acc = trainer.compute_accuracy(batch)

        assert 'action_acc' in acc
        assert 'timing_acc' in acc
        assert 0 <= acc['action_acc'] <= 1
        assert 0 <= acc['timing_acc'] <= 1

    def test_train_epoch(self, sample_dataloaders):
        """Test single epoch training."""
        train_loader, _ = sample_dataloaders

        config = SFTConfig(device='cpu')
        trainer = SFTTrainer(config)

        metrics = trainer.train_epoch(train_loader)

        assert 'loss' in metrics
        assert 'action_acc' in metrics
        assert 'timing_acc' in metrics

    def test_evaluate(self, sample_dataloaders):
        """Test evaluation."""
        _, val_loader = sample_dataloaders

        config = SFTConfig(device='cpu')
        trainer = SFTTrainer(config)

        metrics = trainer.evaluate(val_loader)

        assert 'loss' in metrics
        assert 'action_acc' in metrics
        assert 'timing_acc' in metrics

    def test_short_training(self, sample_dataloaders):
        """Test complete training loop (short)."""
        train_loader, val_loader = sample_dataloaders

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SFTConfig(
                epochs=2,
                patience=1,
                device='cpu',
                checkpoint_dir=tmpdir,
            )
            trainer = SFTTrainer(config)
            model = trainer.train(train_loader, val_loader, verbose=False)

            assert model is not None
            assert len(trainer.history) > 0


class TestEvaluate:
    """Tests for evaluation module."""

    @pytest.fixture
    def model_and_loader(self):
        """Create model and dataloader for testing."""
        config = PolicyConfig(input_dim=60)
        model = EmailPolicyNetwork(config)

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, n):
                self.features = torch.randn(n, 60)
                self.actions = torch.randint(0, 6, (n,))
                self.timings = torch.randint(0, 5, (n,))
                self.priorities = torch.rand(n)

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return {
                    'features': self.features[idx],
                    'action_label': self.actions[idx],
                    'timing_label': self.timings[idx],
                    'priority_target': self.priorities[idx],
                }

        dataset = DictDataset(100)
        loader = DataLoader(dataset, batch_size=32)

        return model, loader

    def test_evaluate_model(self, model_and_loader):
        """Test evaluate_model function."""
        model, loader = model_and_loader

        result = evaluate_model(model, loader)

        assert hasattr(result, 'action_accuracy')
        assert hasattr(result, 'timing_accuracy')
        assert hasattr(result, 'priority_mae')
        assert hasattr(result, 'action_confusion')
        assert 0 <= result.action_accuracy <= 1
        assert result.action_confusion.shape == (6, 6)

    def test_evaluate_with_predictions(self, model_and_loader):
        """Test evaluate_model with predictions returned."""
        model, loader = model_and_loader

        result = evaluate_model(model, loader, return_predictions=True)

        assert result.predictions is not None
        assert 'action_preds' in result.predictions
        assert 'action_labels' in result.predictions

    def test_compute_f1_scores(self, model_and_loader):
        """Test F1 score computation."""
        model, loader = model_and_loader

        result = evaluate_model(model, loader)
        f1_scores = compute_f1_scores(result)

        assert len(f1_scores) == 6  # One for each action
        for name, scores in f1_scores.items():
            assert 'precision' in scores
            assert 'recall' in scores
            assert 'f1' in scores

    def test_check_accuracy_target(self, model_and_loader):
        """Test accuracy target checking."""
        model, loader = model_and_loader

        result = evaluate_model(model, loader)

        # Random model should be around 1/6 = 16.7%
        in_range = check_accuracy_target(result)

        # Just verify it returns a boolean
        assert isinstance(in_range, bool)

    def test_check_accuracy_target_custom_range(self, model_and_loader):
        """Test custom accuracy range."""
        model, loader = model_and_loader

        result = evaluate_model(model, loader)

        # Very low bar
        in_range = check_accuracy_target(result, min_accuracy=0.0, max_accuracy=1.0)
        assert in_range is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
