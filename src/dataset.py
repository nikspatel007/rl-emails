#!/usr/bin/env python3
"""PyTorch Dataset for email RL training.

Provides EmailDataset class for batched email loading with proper shuffling,
designed for use with PyTorch DataLoader.
"""

import json
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from .features import CombinedFeatureExtractor, FEATURE_DIMS


# Action label mapping
ACTION_TO_IDX = {
    'REPLIED': 0,
    'FORWARDED': 1,
    'DELETED': 2,
    'ARCHIVED': 3,
    'KEPT': 4,
    'COMPOSED': 5,
    'JUNK': 6,
}

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

NUM_ACTIONS = len(ACTION_TO_IDX)


class EmailDataset(Dataset):
    """PyTorch Dataset for email feature vectors and action labels.

    Loads emails from JSON split files, extracts features using
    CombinedFeatureExtractor, and returns (features, label) tensors.

    Features are extracted lazily by default but can be precomputed
    for faster training with precompute=True.

    Example:
        >>> dataset = EmailDataset('data/train.json')
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for features, labels in loader:
        ...     # features: (batch_size, 60)
        ...     # labels: (batch_size,)
        ...     output = model(features)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        *,
        user_email: str = '',
        user_context: Optional[dict] = None,
        precompute: bool = False,
        transform: Optional[callable] = None,
    ):
        """Initialize EmailDataset.

        Args:
            data_path: Path to JSON file containing email list
            user_email: Optional user email for feature extraction context
            user_context: Optional historical context for feature extraction
            precompute: If True, extract all features upfront (uses more memory
                but faster training). If False, extract features on demand.
            transform: Optional transform to apply to features
        """
        self.data_path = Path(data_path)
        self.transform = transform

        # Load emails
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.emails = json.load(f)

        # Initialize feature extractor
        self.extractor = CombinedFeatureExtractor(
            user_email=user_email,
            user_context=user_context,
        )

        # Precompute features if requested
        self._precomputed_features: Optional[torch.Tensor] = None
        self._precomputed_labels: Optional[torch.Tensor] = None

        if precompute:
            self._precompute_all()

    def _precompute_all(self) -> None:
        """Precompute all features and labels."""
        features_list = []
        labels_list = []

        for email in self.emails:
            features = self.extractor.to_vector(email)
            label = ACTION_TO_IDX.get(email.get('action', 'KEPT'), 4)

            features_list.append(features)
            labels_list.append(label)

        self._precomputed_features = torch.tensor(
            features_list, dtype=torch.float32
        )
        self._precomputed_labels = torch.tensor(
            labels_list, dtype=torch.long
        )

    def __len__(self) -> int:
        """Return number of emails in dataset."""
        return len(self.emails)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get features and label for a single email.

        Args:
            idx: Index of email to retrieve

        Returns:
            Tuple of (features, label) where:
                - features: Float tensor of shape (60,)
                - label: Long tensor scalar (action index)
        """
        if self._precomputed_features is not None:
            features = self._precomputed_features[idx]
            label = self._precomputed_labels[idx]
        else:
            email = self.emails[idx]
            features_vec = self.extractor.to_vector(email)
            features = torch.tensor(features_vec, dtype=torch.float32)
            label = torch.tensor(
                ACTION_TO_IDX.get(email.get('action', 'KEPT'), 4),
                dtype=torch.long
            )

        if self.transform is not None:
            features = self.transform(features)

        return features, label

    @property
    def feature_dim(self) -> int:
        """Return dimensionality of feature vectors."""
        return FEATURE_DIMS['total']

    @property
    def num_classes(self) -> int:
        """Return number of action classes."""
        return NUM_ACTIONS

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data.

        Returns:
            Tensor of shape (num_classes,) with class weights
        """
        counts = torch.zeros(NUM_ACTIONS, dtype=torch.float32)

        if self._precomputed_labels is not None:
            for label in self._precomputed_labels:
                counts[label] += 1
        else:
            for email in self.emails:
                label = ACTION_TO_IDX.get(email.get('action', 'KEPT'), 4)
                counts[label] += 1

        # Inverse frequency weighting with smoothing
        weights = len(self.emails) / (NUM_ACTIONS * counts + 1e-6)
        return weights

    def get_label_distribution(self) -> dict[str, int]:
        """Get distribution of action labels.

        Returns:
            Dictionary mapping action names to counts
        """
        counts = {action: 0 for action in ACTION_TO_IDX}

        for email in self.emails:
            action = email.get('action', 'KEPT')
            if action in counts:
                counts[action] += 1

        return counts


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
    precompute: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple:
    """Create train, validation, and test DataLoaders.

    Args:
        data_dir: Directory containing train.json, val.json, test.json
        batch_size: Batch size for training
        user_email: Optional user email for feature extraction
        user_context: Optional historical context
        precompute: Whether to precompute features (recommended)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU training

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    data_dir = Path(data_dir)

    # Create datasets
    train_dataset = EmailDataset(
        data_dir / 'train.json',
        user_email=user_email,
        user_context=user_context,
        precompute=precompute,
    )
    val_dataset = EmailDataset(
        data_dir / 'val.json',
        user_email=user_email,
        user_context=user_context,
        precompute=precompute,
    )
    test_dataset = EmailDataset(
        data_dir / 'test.json',
        user_email=user_email,
        user_context=user_context,
        precompute=precompute,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test EmailDataset')
    parser.add_argument('data_dir', type=Path, help='Directory with split files')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    print(f"Loading datasets from {args.data_dir}...")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        precompute=True,
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} emails")
    print(f"  Val:   {len(val_loader.dataset)} emails")
    print(f"  Test:  {len(test_loader.dataset)} emails")

    print(f"\nBatch sizes (batch_size={args.batch_size}):")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    print(f"\nFeature dimensions: {train_loader.dataset.feature_dim}")
    print(f"Number of classes: {train_loader.dataset.num_classes}")

    print("\nLabel distribution (train):")
    dist = train_loader.dataset.get_label_distribution()
    for action, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(train_loader.dataset)
        print(f"  {action:12s}: {count:6d} ({pct:5.1f}%)")

    print("\nClass weights (train):")
    weights = train_loader.dataset.get_class_weights()
    for i, (action, _) in enumerate(sorted(ACTION_TO_IDX.items(), key=lambda x: x[1])):
        print(f"  {action:12s}: {weights[i]:.4f}")

    print("\nSample batch:")
    for features, labels in train_loader:
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape:   {labels.shape}")
        print(f"  Feature sample: {features[0, :5].tolist()}")
        print(f"  Labels sample:  {[IDX_TO_ACTION[l.item()] for l in labels[:5]]}")
        break

    print("\n✓ Dataset test complete")
