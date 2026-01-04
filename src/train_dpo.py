#!/usr/bin/env python3
"""Direct Preference Optimization (DPO) training for email policy network.

Stage 2 of the training pipeline: Learn from preference pairs derived from
user action priorities.

DPO directly optimizes the policy using preference pairs without training
a separate reward model. The loss is:
    L_DPO = -E[log sigma(beta * (log pi_theta(y_w|x) - log pi_ref(y_w|x)
                                 - (log pi_theta(y_l|x) - log pi_ref(y_l|x))))]

Where:
- pi_theta is the policy being trained
- pi_ref is the frozen reference policy (SFT model)
- y_w is the preferred action
- y_l is the rejected action
- beta is a temperature parameter

Usage:
    python src/train_dpo.py --train data/gmail_splits/train.json \
                            --val data/gmail_splits/val.json \
                            --ref-model checkpoints/sft_gmail.pt
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Optional
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from features.combined import extract_combined_features, CombinedFeatureExtractor
from policy_network import EmailPolicyNetwork, PolicyConfig


# Action priority based on true user behavior distribution.
# Higher values = more common actions in real user behavior.
# This avoids artificial bias toward rare actions like REPLY_NOW.
# Distribution from Gmail data: DELETE ~85%, REPLY ~14%, FORWARD <1%, ARCHIVE ~0%
ACTION_PRIORITY = {
    'DELETE': 4,      # Most common action (~85% of emails)
    'ARCHIVE': 3,     # Filing away
    'REPLY_LATER': 2, # Deferred response
    'REPLY_NOW': 1,   # Urgent response
    'FORWARD': 0,     # Least common action
}

ACTION_NAMES = ['reply_now', 'reply_later', 'forward', 'archive', 'delete']
LABEL_TO_ACTION = {
    'REPLY_NOW': 0,
    'REPLY_LATER': 1,
    'FORWARD': 2,
    'ARCHIVE': 3,
    'DELETE': 4,
}


@dataclass
class PreferencePair:
    """A preference pair for DPO training."""
    features: torch.Tensor       # Email features
    preferred_idx: int           # Index of preferred action
    rejected_idx: int            # Index of rejected action
    margin: float                # Priority difference


class DPODataset(Dataset):
    """Dataset of preference pairs for DPO training.

    Generates preference pairs by pairing each email's actual action (preferred)
    with lower-priority actions (rejected).
    """

    def __init__(
        self,
        emails: list[dict],
        feature_extractor: CombinedFeatureExtractor,
        min_margin: int = 1,
        pairs_per_sample: int = 2,
        seed: int = 42,
        balanced: bool = True,
    ):
        """Initialize DPO dataset.

        Args:
            emails: List of email dicts with 'action' field
            feature_extractor: Feature extractor
            min_margin: Minimum priority gap for a valid pair
            pairs_per_sample: Number of negative samples per email
            seed: Random seed for reproducibility
            balanced: Whether to balance pairs across action classes
        """
        self.min_margin = min_margin
        self.pairs_per_sample = pairs_per_sample

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Group emails by action for balanced sampling
        emails_by_action = {action: [] for action in ACTION_PRIORITY.keys()}

        print(f"Generating preference pairs from {len(emails)} emails...")

        # First pass: extract features and group by action
        for email in emails:
            try:
                combined = extract_combined_features(email)
                feat_vec = combined.to_feature_vector()
                if isinstance(feat_vec, np.ndarray):
                    feat_vec = torch.tensor(feat_vec, dtype=torch.float32)
                else:
                    feat_vec = torch.tensor(feat_vec, dtype=torch.float32)

                action_label = email.get('action', 'ARCHIVE')
                if action_label in emails_by_action:
                    emails_by_action[action_label].append((feat_vec, action_label))
            except Exception:
                continue

        # Print class distribution
        print("  Class distribution:")
        for action, samples in emails_by_action.items():
            print(f"    {action}: {len(samples)}")

        # Generate pairs - for balanced training, generate contrastive pairs
        # where we pair each action with its immediate lower-priority neighbor
        self.pairs = []

        if balanced:
            # For each email, only create one pair with the next-lower priority action
            # This prevents over-weighting high-priority actions
            priority_order = sorted(ACTION_PRIORITY.keys(), key=lambda x: ACTION_PRIORITY[x], reverse=True)

            for feat_vec, action_label in [item for items in emails_by_action.values() for item in items]:
                preferred_idx = LABEL_TO_ACTION.get(action_label, 3)
                preferred_priority = ACTION_PRIORITY.get(action_label, 1)

                # Find the next-lower priority action
                rejected_action = None
                for other_action in priority_order:
                    other_priority = ACTION_PRIORITY[other_action]
                    if other_priority < preferred_priority:
                        rejected_action = other_action
                        break

                if rejected_action is None:
                    continue  # No lower-priority action exists

                rejected_idx = LABEL_TO_ACTION[rejected_action]
                margin = preferred_priority - ACTION_PRIORITY[rejected_action]

                if margin >= min_margin:
                    self.pairs.append(PreferencePair(
                        features=feat_vec,
                        preferred_idx=preferred_idx,
                        rejected_idx=rejected_idx,
                        margin=float(margin),
                    ))
        else:
            # Original unbalanced approach
            for feat_vec, action_label in [item for items in emails_by_action.values() for item in items]:
                preferred_idx = LABEL_TO_ACTION.get(action_label, 3)
                preferred_priority = ACTION_PRIORITY.get(action_label, 1)

                for other_action, other_priority in ACTION_PRIORITY.items():
                    if other_priority >= preferred_priority:
                        continue

                    margin = preferred_priority - other_priority
                    if margin < min_margin:
                        continue

                    rejected_idx = LABEL_TO_ACTION[other_action]
                    self.pairs.append(PreferencePair(
                        features=feat_vec,
                        preferred_idx=preferred_idx,
                        rejected_idx=rejected_idx,
                        margin=float(margin),
                    ))

        print(f"Generated {len(self.pairs)} preference pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return (
            pair.features,
            torch.tensor(pair.preferred_idx, dtype=torch.long),
            torch.tensor(pair.rejected_idx, dtype=torch.long),
            torch.tensor(pair.margin, dtype=torch.float32),
        )


def dpo_loss(
    policy: EmailPolicyNetwork,
    ref_policy: EmailPolicyNetwork,
    features: torch.Tensor,
    preferred_idx: torch.Tensor,
    rejected_idx: torch.Tensor,
    beta: float = 0.1,
    kl_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute DPO loss with optional KL regularization.

    Args:
        policy: Policy being trained
        ref_policy: Frozen reference policy
        features: Input features (batch, dim)
        preferred_idx: Preferred action indices (batch,)
        rejected_idx: Rejected action indices (batch,)
        beta: Temperature parameter (higher = less aggressive updates)
        kl_weight: Weight for KL divergence regularization

    Returns:
        Tuple of (loss, metrics dict)
    """
    # Get logits from both policies
    policy_output = policy(features)
    with torch.no_grad():
        ref_output = ref_policy(features)

    policy_logits = policy_output.action_logits
    ref_logits = ref_output.action_logits

    # Convert to log probabilities
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Get log probs for preferred and rejected actions
    batch_indices = torch.arange(features.size(0), device=features.device)

    pi_log_preferred = policy_log_probs[batch_indices, preferred_idx]
    pi_log_rejected = policy_log_probs[batch_indices, rejected_idx]

    ref_log_preferred = ref_log_probs[batch_indices, preferred_idx]
    ref_log_rejected = ref_log_probs[batch_indices, rejected_idx]

    # DPO objective: log sigma(beta * (log_ratio_preferred - log_ratio_rejected))
    # where log_ratio = log(pi_theta / pi_ref)
    log_ratio_preferred = pi_log_preferred - ref_log_preferred
    log_ratio_rejected = pi_log_rejected - ref_log_rejected

    logits = beta * (log_ratio_preferred - log_ratio_rejected)

    # Negative log sigmoid = log(1 + exp(-x))
    dpo_loss = -F.logsigmoid(logits).mean()

    # Optional KL divergence regularization to stay close to reference
    kl_div = 0.0
    if kl_weight > 0:
        # KL(policy || ref) = sum(policy * log(policy / ref))
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        kl_div = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1).mean()

    total_loss = dpo_loss + kl_weight * kl_div

    # Compute accuracy (preference accuracy)
    with torch.no_grad():
        accuracy = (logits > 0).float().mean().item()

        # Reward margin (how much we prefer the preferred action)
        reward_margin = (log_ratio_preferred - log_ratio_rejected).mean().item()

    metrics = {
        'loss': total_loss.item(),
        'dpo_loss': dpo_loss.item(),
        'kl_div': kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
        'accuracy': accuracy,
        'reward_margin': reward_margin,
    }

    return total_loss, metrics


def train_epoch(
    policy: EmailPolicyNetwork,
    ref_policy: EmailPolicyNetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float = 0.1,
) -> dict:
    """Train for one epoch."""
    policy.train()
    ref_policy.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    num_batches = 0

    for features, preferred_idx, rejected_idx, margin in dataloader:
        features = features.to(device)
        preferred_idx = preferred_idx.to(device)
        rejected_idx = rejected_idx.to(device)

        optimizer.zero_grad()

        loss, metrics = dpo_loss(
            policy, ref_policy, features, preferred_idx, rejected_idx, beta
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

        optimizer.step()

        total_loss += metrics['loss']
        total_accuracy += metrics['accuracy']
        total_margin += metrics['reward_margin']
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'reward_margin': total_margin / num_batches,
    }


def evaluate(
    policy: EmailPolicyNetwork,
    ref_policy: EmailPolicyNetwork,
    dataloader: DataLoader,
    device: torch.device,
    beta: float = 0.1,
) -> dict:
    """Evaluate on validation set."""
    policy.eval()
    ref_policy.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    num_batches = 0

    with torch.no_grad():
        for features, preferred_idx, rejected_idx, margin in dataloader:
            features = features.to(device)
            preferred_idx = preferred_idx.to(device)
            rejected_idx = rejected_idx.to(device)

            loss, metrics = dpo_loss(
                policy, ref_policy, features, preferred_idx, rejected_idx, beta
            )

            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            total_margin += metrics['reward_margin']
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'reward_margin': total_margin / num_batches,
    }


def evaluate_action_accuracy(
    policy: EmailPolicyNetwork,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate action prediction accuracy."""
    policy.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, preferred_idx, rejected_idx, margin in dataloader:
            features = features.to(device)
            preferred_idx = preferred_idx.to(device)

            output = policy(features)
            preds = output.action_logits.argmax(dim=1)

            correct += (preds == preferred_idx).sum().item()
            total += features.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(preferred_idx.cpu().tolist())

    # Per-class accuracy
    class_correct = Counter()
    class_total = Counter()
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    per_class_acc = {}
    for i, name in enumerate(ACTION_NAMES):
        if class_total[i] > 0:
            per_class_acc[name] = class_correct[i] / class_total[i]
        else:
            per_class_acc[name] = 0.0

    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'per_class_accuracy': per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='DPO training for email policy')
    parser.add_argument('--train', type=Path, required=True, help='Training data JSON')
    parser.add_argument('--val', type=Path, help='Validation data JSON')
    parser.add_argument('--ref-model', type=Path, default=Path('checkpoints/sft_gmail.pt'),
                       help='Reference model (SFT checkpoint)')
    parser.add_argument('--output', type=Path, default=Path('checkpoints/dpo_gmail.pt'),
                       help='Output checkpoint path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1, help='DPO temperature')
    parser.add_argument('--min-margin', type=int, default=1,
                       help='Minimum priority margin for pairs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda, mps)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load reference model
    print(f"\nLoading reference model from {args.ref_model}...")
    checkpoint = torch.load(args.ref_model, map_location=device, weights_only=False)

    config = checkpoint['config']
    ref_policy = EmailPolicyNetwork(config).to(device)
    ref_policy.load_state_dict(checkpoint['model_state_dict'])
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad = False
    print(f"Reference model loaded (val_acc: {checkpoint.get('val_accuracy', 'N/A')})")

    # Create trainable policy (copy of reference)
    policy = EmailPolicyNetwork(config).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    print(f"Policy initialized from reference model")

    # Load data
    print(f"\nLoading training data from {args.train}...")
    with open(args.train) as f:
        train_emails = json.load(f)
    print(f"Loaded {len(train_emails)} training emails")

    val_emails = None
    if args.val and args.val.exists():
        print(f"Loading validation data from {args.val}...")
        with open(args.val) as f:
            val_emails = json.load(f)
        print(f"Loaded {len(val_emails)} validation emails")

    # Create datasets
    feature_extractor = CombinedFeatureExtractor()
    train_dataset = DPODataset(
        train_emails, feature_extractor, min_margin=args.min_margin
    )

    val_dataset = None
    if val_emails:
        val_dataset = DPODataset(
            val_emails, feature_extractor, min_margin=args.min_margin
        )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Print statistics
    print(f"\nTraining pairs: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation pairs: {len(val_dataset)}")

    # Optimizer
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    print(f"\nDPO Training (beta={args.beta})")
    print("-" * 70)

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            policy, ref_policy, train_loader, optimizer, device, args.beta
        )

        log = f"Epoch {epoch+1:3d}: "
        log += f"loss={train_metrics['loss']:.4f}, "
        log += f"pref_acc={train_metrics['accuracy']:.3f}, "
        log += f"margin={train_metrics['reward_margin']:.3f}"

        if val_loader:
            val_metrics = evaluate(
                policy, ref_policy, val_loader, device, args.beta
            )
            action_metrics = evaluate_action_accuracy(policy, val_loader, device)

            log += f" | val_loss={val_metrics['loss']:.4f}"
            log += f", val_pref_acc={val_metrics['accuracy']:.3f}"
            log += f", val_action_acc={action_metrics['accuracy']:.3f}"

            # Save best model by action accuracy
            if action_metrics['accuracy'] > best_val_acc:
                best_val_acc = action_metrics['accuracy']
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_accuracy': best_val_acc,
                    'dpo_beta': args.beta,
                    'training_type': 'dpo',
                }, args.output)
                log += " *"
        else:
            # Save periodically
            if (epoch + 1) % 5 == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'config': config,
                    'dpo_beta': args.beta,
                    'training_type': 'dpo',
                }, args.output)

        print(log)
        scheduler.step()

    print("-" * 70)
    print(f"DPO training complete. Best validation accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {args.output}")

    # Final evaluation
    if val_loader:
        print("\nFinal validation metrics (per-class accuracy):")
        final_metrics = evaluate_action_accuracy(policy, val_loader, device)
        for action, acc in final_metrics['per_class_accuracy'].items():
            print(f"  {action}: {acc:.3f}")


if __name__ == '__main__':
    main()
