"""Email RL source modules.

This package provides the core components for email prioritization RL:
- features/: Feature extraction (project, topic, task, people, combined)
- policy_network: PyTorch policy network for action selection
- dataset: PyTorch Dataset for email training data
- trainer: PPO training loop with checkpointing and logging

Optional modules (require additional dependencies):
- email_action: Action representation (requires numpy)
- email_state: State representation (requires numpy)
- reward: Multi-signal reward function (requires numpy)
"""

# Feature extraction is always available
from .features import (
    CombinedFeatures,
    CombinedFeatureExtractor,
    extract_combined_features,
    FEATURE_DIMS,
)

# Policy network and training require torch
try:
    from .policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        PolicyOutput,
        ActionSample,
        create_policy_network,
        DuelingPolicyNetwork,
    )
    from .dataset import (
        EmailDataset,
        create_dataloaders,
        ACTION_TO_IDX,
        IDX_TO_ACTION,
        NUM_ACTIONS,
    )
    from .trainer import (
        EmailRLTrainer,
        TrainingConfig,
        TrainingState,
        train_model,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    # Features
    'CombinedFeatures',
    'CombinedFeatureExtractor',
    'extract_combined_features',
    'FEATURE_DIMS',
]

if HAS_TORCH:
    __all__.extend([
        # Policy Network
        'EmailPolicyNetwork',
        'PolicyConfig',
        'PolicyOutput',
        'ActionSample',
        'create_policy_network',
        'DuelingPolicyNetwork',
        # Dataset
        'EmailDataset',
        'create_dataloaders',
        'ACTION_TO_IDX',
        'IDX_TO_ACTION',
        'NUM_ACTIONS',
        # Training
        'EmailRLTrainer',
        'TrainingConfig',
        'TrainingState',
        'train_model',
    ])
