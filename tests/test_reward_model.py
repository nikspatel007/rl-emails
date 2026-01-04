#!/usr/bin/env python3
"""Tests for EmailRewardModel.

These tests verify the reward model architecture, preference pair generation,
and training functionality.
"""

import unittest

# Skip tests if torch is not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    import sys
    sys.path.insert(0, str(__file__).replace('/tests/test_reward_model.py', ''))
    from src.reward_model import (
        EmailRewardModel,
        RewardConfig,
        PreferencePairDataset,
        RewardModelTrainer,
        create_reward_model,
        generate_preference_pairs,
        ActionPriority,
        ACTION_TO_PRIORITY,
        PreferencePair,
    )
    from torch.utils.data import DataLoader


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestRewardConfig(unittest.TestCase):
    """Tests for RewardConfig."""

    def test_default_config(self):
        config = RewardConfig()
        self.assertEqual(config.input_dim, 69)  # Updated for content features
        self.assertEqual(config.hidden_dims, (256, 128, 64))
        self.assertEqual(config.dropout, 0.1)
        self.assertTrue(config.use_layer_norm)

    def test_custom_config(self):
        config = RewardConfig(
            input_dim=100,
            hidden_dims=(512, 256),
            dropout=0.2,
        )
        self.assertEqual(config.input_dim, 100)
        self.assertEqual(config.hidden_dims, (512, 256))
        self.assertEqual(config.dropout, 0.2)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestActionPriority(unittest.TestCase):
    """Tests for ActionPriority ordering."""

    def test_priority_ordering(self):
        """Verify priority hierarchy is correct."""
        self.assertGreater(ActionPriority.REPLIED, ActionPriority.FORWARDED)
        self.assertGreater(ActionPriority.FORWARDED, ActionPriority.COMPOSED)
        self.assertGreater(ActionPriority.COMPOSED, ActionPriority.ARCHIVED)
        self.assertGreater(ActionPriority.ARCHIVED, ActionPriority.KEPT)
        self.assertGreater(ActionPriority.KEPT, ActionPriority.AUTO_FILED)
        self.assertGreater(ActionPriority.AUTO_FILED, ActionPriority.DELETED)
        self.assertGreater(ActionPriority.DELETED, ActionPriority.JUNK)

    def test_action_to_priority_mapping(self):
        """Verify string to priority mapping."""
        self.assertEqual(ACTION_TO_PRIORITY['REPLIED'], ActionPriority.REPLIED)
        self.assertEqual(ACTION_TO_PRIORITY['DELETED'], ActionPriority.DELETED)
        self.assertEqual(ACTION_TO_PRIORITY['JUNK'], ActionPriority.JUNK)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestEmailRewardModel(unittest.TestCase):
    """Tests for EmailRewardModel."""

    def setUp(self):
        self.config = RewardConfig(input_dim=69, hidden_dims=(128, 64))
        self.model = EmailRewardModel(self.config)
        self.batch_size = 32

    def test_forward_output_shape(self):
        x = torch.randn(self.batch_size, 69)
        output = self.model(x)

        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_forward_single_sample(self):
        x = torch.randn(1, 69)
        output = self.model(x)

        self.assertEqual(output.shape, (1, 1))

    def test_get_reward_batch(self):
        x = torch.randn(self.batch_size, 69)
        rewards = self.model.get_reward(x)

        self.assertEqual(rewards.shape, (self.batch_size,))

    def test_get_reward_single(self):
        x = torch.randn(69)
        reward = self.model.get_reward(x)

        self.assertEqual(reward.shape, ())

    def test_preference_loss_computation(self):
        preferred = torch.randn(16, 69)
        rejected = torch.randn(16, 69)

        loss = self.model.preference_loss(preferred, rejected)

        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)  # Loss should be non-negative

    def test_preference_loss_with_margin(self):
        preferred = torch.randn(16, 69)
        rejected = torch.randn(16, 69)
        margin = torch.ones(16) * 2.0

        # Enable margin loss
        self.model.config.use_margin_loss = True
        loss = self.model.preference_loss(preferred, rejected, margin)

        self.assertEqual(loss.shape, ())

    def test_preference_accuracy(self):
        preferred = torch.randn(16, 69)
        rejected = torch.randn(16, 69)

        accuracy = self.model.preference_accuracy(preferred, rejected)

        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_gradient_flow(self):
        x = torch.randn(self.batch_size, 60, requires_grad=True)
        output = self.model(x)

        loss = output.mean()
        loss.backward()

        self.assertIsNotNone(x.grad)

    def test_preference_loss_gradients(self):
        preferred = torch.randn(16, 60, requires_grad=True)
        rejected = torch.randn(16, 60, requires_grad=True)

        loss = self.model.preference_loss(preferred, rejected)
        loss.backward()

        self.assertIsNotNone(preferred.grad)
        self.assertIsNotNone(rejected.grad)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestPreferencePairGeneration(unittest.TestCase):
    """Tests for preference pair generation."""

    def setUp(self):
        self.features = torch.randn(100, 69)
        self.actions = (
            ['REPLIED'] * 20 +
            ['ARCHIVED'] * 30 +
            ['KEPT'] * 30 +
            ['DELETED'] * 20
        )

    def test_generate_pairs_basic(self):
        pairs = generate_preference_pairs(
            self.features,
            self.actions,
            min_priority_gap=1,
            seed=42,
        )

        self.assertGreater(len(pairs), 0)
        self.assertIsInstance(pairs[0], PreferencePair)

    def test_generate_pairs_min_gap(self):
        pairs_gap_1 = generate_preference_pairs(
            self.features,
            self.actions,
            min_priority_gap=1,
            seed=42,
        )
        pairs_gap_3 = generate_preference_pairs(
            self.features,
            self.actions,
            min_priority_gap=3,
            seed=42,
        )

        # Larger gap should produce fewer pairs
        self.assertGreater(len(pairs_gap_1), len(pairs_gap_3))

    def test_generate_pairs_correct_ordering(self):
        """Verify preferred has higher priority than rejected."""
        pairs = generate_preference_pairs(
            self.features,
            self.actions,
            min_priority_gap=1,
            seed=42,
        )

        # All margins should be positive (preferred > rejected)
        for pair in pairs:
            self.assertGreater(pair.margin, 0)

    def test_generate_pairs_reproducible(self):
        pairs1 = generate_preference_pairs(
            self.features,
            self.actions,
            seed=42,
        )
        pairs2 = generate_preference_pairs(
            self.features,
            self.actions,
            seed=42,
        )

        self.assertEqual(len(pairs1), len(pairs2))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestPreferencePairDataset(unittest.TestCase):
    """Tests for PreferencePairDataset."""

    def setUp(self):
        self.features = torch.randn(50, 69)
        self.actions = (
            ['REPLIED'] * 10 +
            ['ARCHIVED'] * 15 +
            ['KEPT'] * 15 +
            ['DELETED'] * 10
        )

    def test_dataset_creation(self):
        dataset = PreferencePairDataset(self.features, self.actions)

        self.assertGreater(len(dataset), 0)

    def test_dataset_getitem(self):
        dataset = PreferencePairDataset(self.features, self.actions)

        preferred, rejected, margin = dataset[0]

        self.assertEqual(preferred.shape, (60,))
        self.assertEqual(rejected.shape, (60,))
        self.assertIsInstance(margin, float)

    def test_dataset_min_gap_filter(self):
        dataset_gap_1 = PreferencePairDataset(
            self.features,
            self.actions,
            min_priority_gap=1,
        )
        dataset_gap_3 = PreferencePairDataset(
            self.features,
            self.actions,
            min_priority_gap=3,
        )

        self.assertGreater(len(dataset_gap_1), len(dataset_gap_3))

    def test_dataloader_compatibility(self):
        dataset = PreferencePairDataset(self.features, self.actions)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        batch = next(iter(dataloader))
        preferred, rejected, margin = batch

        self.assertEqual(preferred.shape, (8, 60))
        self.assertEqual(rejected.shape, (8, 60))
        self.assertEqual(margin.shape, (8,))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestRewardModelTrainer(unittest.TestCase):
    """Tests for RewardModelTrainer."""

    def setUp(self):
        self.config = RewardConfig(input_dim=69, hidden_dims=(64, 32))
        self.model = EmailRewardModel(self.config)
        self.features = torch.randn(50, 69)
        self.actions = (
            ['REPLIED'] * 10 +
            ['ARCHIVED'] * 15 +
            ['KEPT'] * 15 +
            ['DELETED'] * 10
        )

    def test_trainer_creation(self):
        trainer = RewardModelTrainer(self.model)

        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.device)

    def test_train_epoch(self):
        dataset = PreferencePairDataset(self.features, self.actions)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        trainer = RewardModelTrainer(self.model)

        metrics = trainer.train_epoch(dataloader)

        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertTrue(metrics['loss'] >= 0)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)

    def test_evaluate(self):
        dataset = PreferencePairDataset(self.features, self.actions)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        trainer = RewardModelTrainer(self.model)

        metrics = trainer.evaluate(dataloader)

        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)

    def test_train_full_loop(self):
        dataset = PreferencePairDataset(self.features, self.actions)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        trainer = RewardModelTrainer(self.model)

        history = trainer.train(
            dataloader,
            num_epochs=2,
            verbose=False,
        )

        self.assertEqual(len(history), 2)
        self.assertIn('train_loss', history[0])
        self.assertIn('train_accuracy', history[0])

    def test_train_with_validation(self):
        dataset = PreferencePairDataset(self.features, self.actions)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        trainer = RewardModelTrainer(self.model)

        history = trainer.train(
            train_loader,
            val_dataloader=val_loader,
            num_epochs=2,
            verbose=False,
        )

        self.assertIn('val_loss', history[0])
        self.assertIn('val_accuracy', history[0])


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestCreateRewardModel(unittest.TestCase):
    """Tests for factory function."""

    def test_create_default(self):
        model = create_reward_model()
        self.assertIsInstance(model, EmailRewardModel)
        self.assertEqual(model.config.input_dim, 69)

    def test_create_custom_dims(self):
        model = create_reward_model(input_dim=100, hidden_dims=(512, 256))

        x = torch.randn(32, 100)
        output = model(x)

        self.assertEqual(output.shape, (32, 1))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestRewardModelLearning(unittest.TestCase):
    """Tests to verify the model actually learns preferences."""

    def test_model_learns_clear_preferences(self):
        """Model should learn to prefer high-value features over low-value."""
        # Create synthetic data where "good" emails have positive features
        n_samples = 100
        good_features = torch.randn(n_samples, 69) + 1.0  # Shifted positive
        bad_features = torch.randn(n_samples, 69) - 1.0   # Shifted negative

        features = torch.cat([good_features, bad_features])
        actions = ['REPLIED'] * n_samples + ['DELETED'] * n_samples

        # Train model
        model = create_reward_model(hidden_dims=(64, 32))
        dataset = PreferencePairDataset(features, actions)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        trainer = RewardModelTrainer(model, learning_rate=1e-3)

        initial_accuracy = model.preference_accuracy(good_features[:16], bad_features[:16])

        history = trainer.train(dataloader, num_epochs=20, verbose=False)

        final_accuracy = model.preference_accuracy(good_features[:16], bad_features[:16])

        # Accuracy should improve
        self.assertGreater(final_accuracy, initial_accuracy)
        # Should achieve reasonable accuracy on this synthetic task
        self.assertGreater(final_accuracy, 0.6)


if __name__ == '__main__':
    unittest.main()
