#!/usr/bin/env python3
"""Tests for EmailPolicyNetwork.

These tests verify the policy network architecture and behavior.
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
    sys.path.insert(0, str(__file__).replace('/tests/test_policy_network.py', ''))
    from src.policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        PolicyOutput,
        ActionSample,
        create_policy_network,
        DuelingPolicyNetwork,
        FeatureEncoder,
        ActionHead,
        TimingHead,
        PriorityHead,
        ValueHead,
        NUM_ACTION_TYPES,
        NUM_RESPONSE_TIMES,
        DEFAULT_FEATURE_DIM,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestPolicyConfig(unittest.TestCase):
    """Tests for PolicyConfig."""

    def test_default_config(self):
        config = PolicyConfig()
        self.assertEqual(config.input_dim, 69)  # Updated for content features
        self.assertEqual(config.hidden_dims, (256, 128, 64))
        self.assertEqual(config.dropout, 0.1)
        self.assertTrue(config.use_layer_norm)

    def test_custom_config(self):
        config = PolicyConfig(
            input_dim=100,
            hidden_dims=(512, 256),
            dropout=0.2,
        )
        self.assertEqual(config.input_dim, 100)
        self.assertEqual(config.hidden_dims, (512, 256))
        self.assertEqual(config.dropout, 0.2)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestFeatureEncoder(unittest.TestCase):
    """Tests for FeatureEncoder."""

    def test_encoder_output_shape(self):
        config = PolicyConfig(input_dim=69, hidden_dims=(128, 64))
        encoder = FeatureEncoder(config)

        x = torch.randn(32, 69)
        output = encoder(x)

        self.assertEqual(output.shape, (32, 64))

    def test_encoder_single_sample(self):
        config = PolicyConfig(input_dim=69, hidden_dims=(128,))
        encoder = FeatureEncoder(config)

        x = torch.randn(1, 69)
        output = encoder(x)

        self.assertEqual(output.shape, (1, 128))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestHeads(unittest.TestCase):
    """Tests for individual output heads."""

    def test_action_head(self):
        head = ActionHead(64, NUM_ACTION_TYPES)
        x = torch.randn(32, 64)
        output = head(x)
        self.assertEqual(output.shape, (32, NUM_ACTION_TYPES))

    def test_timing_head(self):
        head = TimingHead(64, NUM_RESPONSE_TIMES)
        x = torch.randn(32, 64)
        output = head(x)
        self.assertEqual(output.shape, (32, NUM_RESPONSE_TIMES))

    def test_priority_head(self):
        head = PriorityHead(64)
        x = torch.randn(32, 64)
        output = head(x)
        self.assertEqual(output.shape, (32, 1))
        # Priority should be in [0, 1]
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())

    def test_value_head(self):
        head = ValueHead(64)
        x = torch.randn(32, 64)
        output = head(x)
        self.assertEqual(output.shape, (32, 1))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestEmailPolicyNetwork(unittest.TestCase):
    """Tests for EmailPolicyNetwork."""

    def setUp(self):
        self.config = PolicyConfig(input_dim=69, hidden_dims=(128, 64))
        self.policy = EmailPolicyNetwork(self.config)
        self.batch_size = 32

    def test_forward_output_types(self):
        x = torch.randn(self.batch_size, 60)
        output = self.policy(x)

        self.assertIsInstance(output, PolicyOutput)
        self.assertIsInstance(output.action_logits, torch.Tensor)
        self.assertIsInstance(output.timing_logits, torch.Tensor)
        self.assertIsInstance(output.priority, torch.Tensor)
        self.assertIsInstance(output.value, torch.Tensor)

    def test_forward_output_shapes(self):
        x = torch.randn(self.batch_size, 60)
        output = self.policy(x)

        self.assertEqual(output.action_logits.shape, (self.batch_size, NUM_ACTION_TYPES))
        self.assertEqual(output.timing_logits.shape, (self.batch_size, NUM_RESPONSE_TIMES))
        self.assertEqual(output.priority.shape, (self.batch_size, 1))
        self.assertEqual(output.value.shape, (self.batch_size, 1))

    def test_action_probs_sum_to_one(self):
        x = torch.randn(self.batch_size, 60)
        action_probs, timing_probs = self.policy.get_action_probs(x)

        # Probabilities should sum to 1
        action_sums = action_probs.sum(dim=-1)
        timing_sums = timing_probs.sum(dim=-1)

        self.assertTrue(torch.allclose(action_sums, torch.ones(self.batch_size), atol=1e-5))
        self.assertTrue(torch.allclose(timing_sums, torch.ones(self.batch_size), atol=1e-5))

    def test_sample_action_output(self):
        x = torch.randn(self.batch_size, 60)
        sample = self.policy.sample_action(x)

        self.assertIsInstance(sample, ActionSample)
        self.assertEqual(sample.action_idx.shape, (self.batch_size,))
        self.assertEqual(sample.timing_idx.shape, (self.batch_size,))
        self.assertEqual(sample.priority.shape, (self.batch_size,))
        self.assertEqual(sample.action_log_prob.shape, (self.batch_size,))
        self.assertEqual(sample.timing_log_prob.shape, (self.batch_size,))
        self.assertEqual(sample.entropy.shape, (self.batch_size,))

    def test_sample_action_valid_indices(self):
        x = torch.randn(self.batch_size, 60)
        sample = self.policy.sample_action(x)

        # Action indices should be valid
        self.assertTrue((sample.action_idx >= 0).all())
        self.assertTrue((sample.action_idx < NUM_ACTION_TYPES).all())

        # Timing indices should be valid
        self.assertTrue((sample.timing_idx >= 0).all())
        self.assertTrue((sample.timing_idx < NUM_RESPONSE_TIMES).all())

    def test_sample_action_deterministic(self):
        x = torch.randn(self.batch_size, 60)

        # Deterministic samples should be consistent
        sample1 = self.policy.sample_action(x, deterministic=True)
        sample2 = self.policy.sample_action(x, deterministic=True)

        self.assertTrue((sample1.action_idx == sample2.action_idx).all())
        self.assertTrue((sample1.timing_idx == sample2.timing_idx).all())

    def test_evaluate_actions(self):
        x = torch.randn(self.batch_size, 60)
        sample = self.policy.sample_action(x)

        action_log_prob, timing_log_prob, entropy, value = self.policy.evaluate_actions(
            x, sample.action_idx, sample.timing_idx
        )

        self.assertEqual(action_log_prob.shape, (self.batch_size,))
        self.assertEqual(timing_log_prob.shape, (self.batch_size,))
        self.assertEqual(entropy.shape, (self.batch_size,))
        self.assertEqual(value.shape, (self.batch_size,))

    def test_get_value(self):
        x = torch.randn(self.batch_size, 60)
        value = self.policy.get_value(x)

        self.assertEqual(value.shape, (self.batch_size,))

    def test_predict_greedy(self):
        x = torch.randn(1, 69)
        action_idx, timing_idx, priority = self.policy.predict_greedy(x)

        self.assertIsInstance(action_idx, int)
        self.assertIsInstance(timing_idx, int)
        self.assertIsInstance(priority, float)

        self.assertTrue(0 <= action_idx < NUM_ACTION_TYPES)
        self.assertTrue(0 <= timing_idx < NUM_RESPONSE_TIMES)
        self.assertTrue(0.0 <= priority <= 1.0)

    def test_predict_greedy_1d_input(self):
        # Should handle 1D input
        x = torch.randn(60)
        action_idx, timing_idx, priority = self.policy.predict_greedy(x)

        self.assertTrue(0 <= action_idx < NUM_ACTION_TYPES)

    def test_gradient_flow(self):
        x = torch.randn(self.batch_size, 60, requires_grad=True)
        output = self.policy(x)

        # Should be able to compute gradients
        loss = output.value.mean()
        loss.backward()

        self.assertIsNotNone(x.grad)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestDuelingPolicyNetwork(unittest.TestCase):
    """Tests for DuelingPolicyNetwork."""

    def setUp(self):
        self.config = PolicyConfig(input_dim=69, hidden_dims=(128, 64))
        self.policy = DuelingPolicyNetwork(self.config)

    def test_forward_output_shapes(self):
        x = torch.randn(32, 69)
        output = self.policy(x)

        self.assertEqual(output.action_logits.shape, (32, NUM_ACTION_TYPES))
        self.assertEqual(output.timing_logits.shape, (32, NUM_RESPONSE_TIMES))
        self.assertEqual(output.priority.shape, (32, 1))
        self.assertEqual(output.value.shape, (32, 1))

    def test_sample_action(self):
        x = torch.randn(32, 69)
        sample = self.policy.sample_action(x)

        self.assertEqual(sample.action_idx.shape, (32,))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestCreatePolicyNetwork(unittest.TestCase):
    """Tests for factory function."""

    def test_create_default(self):
        policy = create_policy_network()
        self.assertIsInstance(policy, EmailPolicyNetwork)
        self.assertNotIsInstance(policy, DuelingPolicyNetwork)

    def test_create_dueling(self):
        policy = create_policy_network(use_dueling=True)
        self.assertIsInstance(policy, DuelingPolicyNetwork)

    def test_create_custom_dims(self):
        policy = create_policy_network(input_dim=100, hidden_dims=(512, 256))
        x = torch.randn(32, 100)
        output = policy(x)

        self.assertEqual(output.action_logits.shape[1], NUM_ACTION_TYPES)


if __name__ == '__main__':
    unittest.main()
