#!/usr/bin/env python3
"""Inference CLI for human testing of the email prioritization model.

Load a trained model checkpoint and predict actions for emails.

Usage:
    # From subject/body arguments
    python scripts/infer.py --subject "Meeting tomorrow" --body "Can we meet at 3pm?"

    # From JSON file
    python scripts/infer.py --file email.json

    # From stdin (JSON format)
    cat email.json | python scripts/infer.py

    # Use DPO model instead of SFT
    python scripts/infer.py --checkpoint checkpoints/dpo_gmail.pt --subject "Hi"
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.combined import extract_combined_features, FEATURE_DIMS
from policy_network import create_policy_network


# Action and timing names (from evaluate.py)
ACTION_NAMES = ['reply_now', 'reply_later', 'forward', 'archive', 'delete']
TIMING_NAMES = ['immediate', 'same_day', 'next_day', 'this_week', 'when_possible']

# Feature dimension names for interpretability
FEATURE_SECTIONS = [
    ('Project Features', 8, [
        'project_mention_count', 'deadline_count', 'action_item_count',
        'contract_ref', 'code_ref', 'money_ref', 'urgency_score', 'project_score'
    ]),
    ('Topic Features', 20, [
        'topic_meeting', 'topic_scheduling', 'topic_hr', 'topic_legal',
        'topic_financial', 'topic_technical', 'topic_admin', 'topic_personal',
        'topic_urgent', 'topic_info_sharing', 'topic_decision', 'topic_follow_up',
        'topic_introduction', 'topic_other', 'topic_confidence', 'is_question',
        'is_action_request', 'is_decision_needed', 'is_fyi', 'urgency_score'
    ]),
    ('Task Features', 12, [
        'has_deadline', 'deadline_urgency', 'deadline_days', 'has_deliverable',
        'is_assigned_to_user', 'is_assigned_to_other', 'estimated_effort',
        'task_count', 'question_count', 'request_count', 'deliverable_count',
        'task_score'
    ]),
    ('People Features', 15, [
        'sender_org_level', 'sender_is_executive', 'sender_is_manager',
        'sender_is_internal', 'sender_is_external', 'is_direct_to', 'is_cc',
        'is_bcc', 'recipient_count', 'cc_count', 'includes_executives',
        'includes_managers', 'sender_importance', 'interaction_frequency',
        'response_expectation'
    ]),
    ('Temporal Features', 8, [
        'hour_of_day_norm', 'day_of_week_norm', 'is_business_hours', 'is_weekend',
        'time_since_receipt_hours', 'is_same_day', 'is_overdue', 'temporal_urgency'
    ]),
    ('Computed Scores', 6, [
        'project_score', 'topic_score', 'task_score',
        'people_score', 'temporal_score', 'overall_priority'
    ]),
]


def load_model(checkpoint_path: Path) -> tuple:
    """Load model from checkpoint.

    Returns:
        Tuple of (model, input_dim, include_content)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Auto-detect input_dim from checkpoint config
    input_dim = None
    if 'config' in checkpoint and hasattr(checkpoint['config'], 'input_dim'):
        input_dim = checkpoint['config'].input_dim
    elif 'input_dim' in checkpoint:
        input_dim = checkpoint['input_dim']

    # Default to base feature dim if not found
    if input_dim is None:
        input_dim = FEATURE_DIMS['total_base']

    # Check if content features are included
    include_content = input_dim > FEATURE_DIMS['total_base']

    # Create and load model
    model = create_policy_network(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, input_dim, include_content


def parse_email_input(args) -> dict:
    """Parse email from various input sources.

    Returns:
        Email dict with subject, body, and optional from/to fields
    """
    # Priority: file > stdin > subject/body args
    if args.file:
        with open(args.file) as f:
            return json.load(f)

    # Check for stdin (non-tty means piped input)
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return json.loads(data)

    # Use command line args
    if args.subject or args.body:
        return {
            'subject': args.subject or '',
            'body': args.body or '',
            'from': args.sender or 'unknown@example.com',
            'to': args.recipient or 'user@example.com',
        }

    raise ValueError(
        "No email input provided. Use --subject/--body, --file, or pipe JSON to stdin"
    )


def extract_features(email: dict, include_content: bool = False) -> np.ndarray:
    """Extract feature vector from email."""
    combined = extract_combined_features(email, include_content=include_content)
    feat_vec = combined.to_feature_vector(include_content=include_content)
    if isinstance(feat_vec, np.ndarray):
        return feat_vec
    return np.array(feat_vec, dtype=np.float32)


def get_top_features(features: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
    """Get top contributing features by absolute value.

    Returns:
        List of (feature_name, value) tuples
    """
    # Build feature name mapping
    feature_names = []
    for section_name, dim, names in FEATURE_SECTIONS:
        for name in names:
            feature_names.append(f"{section_name.split()[0].lower()}/{name}")

    # Handle dimension mismatch gracefully
    if len(features) < len(feature_names):
        feature_names = feature_names[:len(features)]
    elif len(features) > len(feature_names):
        # Content features not in our list
        for i in range(len(features) - len(feature_names)):
            feature_names.append(f"content/embed_{i}")

    # Get top features by absolute value
    abs_values = np.abs(features)
    top_indices = np.argsort(abs_values)[-top_k:][::-1]

    return [(feature_names[i], float(features[i])) for i in top_indices]


def format_prediction(
    action_probs: np.ndarray,
    timing_probs: np.ndarray,
    priority: float,
    top_features: list[tuple[str, float]],
    verbose: bool = False,
) -> str:
    """Format prediction output for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("EMAIL PREDICTION")
    lines.append("=" * 60)

    # Primary prediction
    action_idx = action_probs.argmax()
    timing_idx = timing_probs.argmax()

    lines.append("")
    lines.append("PREDICTION:")
    lines.append(f"  Action:   {ACTION_NAMES[action_idx].upper()} ({action_probs[action_idx]:.1%})")
    lines.append(f"  Timing:   {TIMING_NAMES[timing_idx]} ({timing_probs[timing_idx]:.1%})")
    lines.append(f"  Priority: {priority:.2f}")

    # Action probabilities
    lines.append("")
    lines.append("ACTION PROBABILITIES:")
    sorted_actions = sorted(enumerate(action_probs), key=lambda x: x[1], reverse=True)
    for idx, prob in sorted_actions:
        bar = "#" * int(prob * 40)
        lines.append(f"  {ACTION_NAMES[idx]:12s} {prob:5.1%} {bar}")

    # Timing probabilities
    lines.append("")
    lines.append("TIMING PROBABILITIES:")
    sorted_timings = sorted(enumerate(timing_probs), key=lambda x: x[1], reverse=True)
    for idx, prob in sorted_timings:
        bar = "#" * int(prob * 40)
        lines.append(f"  {TIMING_NAMES[idx]:15s} {prob:5.1%} {bar}")

    # Feature influence
    if verbose and top_features:
        lines.append("")
        lines.append("TOP CONTRIBUTING FEATURES:")
        for name, value in top_features:
            sign = "+" if value > 0 else ""
            lines.append(f"  {name:35s} {sign}{value:.3f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_json_output(
    email: dict,
    action_probs: np.ndarray,
    timing_probs: np.ndarray,
    priority: float,
    top_features: list[tuple[str, float]],
) -> str:
    """Format prediction as JSON for programmatic use."""
    action_idx = action_probs.argmax()
    timing_idx = timing_probs.argmax()

    result = {
        'input': {
            'subject': email.get('subject', ''),
            'body': email.get('body', '')[:200] + '...' if len(email.get('body', '')) > 200 else email.get('body', ''),
        },
        'prediction': {
            'action': ACTION_NAMES[action_idx],
            'action_confidence': float(action_probs[action_idx]),
            'timing': TIMING_NAMES[timing_idx],
            'timing_confidence': float(timing_probs[timing_idx]),
            'priority': float(priority),
        },
        'action_probabilities': {
            name: float(prob) for name, prob in zip(ACTION_NAMES, action_probs)
        },
        'timing_probabilities': {
            name: float(prob) for name, prob in zip(TIMING_NAMES, timing_probs)
        },
        'top_features': [
            {'name': name, 'value': value} for name, value in top_features
        ],
    }

    return json.dumps(result, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Inference CLI for email prioritization model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_argument_group('input options')
    input_group.add_argument(
        '--subject', '-s',
        help='Email subject line'
    )
    input_group.add_argument(
        '--body', '-b',
        help='Email body text'
    )
    input_group.add_argument(
        '--file', '-f', type=Path,
        help='Path to JSON file with email data'
    )
    input_group.add_argument(
        '--sender',
        help='Sender email address (optional, for people features)'
    )
    input_group.add_argument(
        '--recipient',
        help='Recipient email address (optional, for people features)'
    )

    # Model options
    model_group = parser.add_argument_group('model options')
    model_group.add_argument(
        '--checkpoint', '-c', type=Path,
        default=Path('checkpoints/sft_gmail.pt'),
        help='Path to model checkpoint (default: checkpoints/sft_gmail.pt)'
    )

    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--json', action='store_true',
        help='Output as JSON instead of formatted text'
    )
    output_group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed feature contributions'
    )
    output_group.add_argument(
        '--top-features', type=int, default=10,
        help='Number of top features to show (default: 10)'
    )

    args = parser.parse_args()

    # Load model
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    model, input_dim, include_content = load_model(args.checkpoint)

    if not args.json:
        print(f"Loaded model from {args.checkpoint}")
        print(f"  Input dim: {input_dim}")
        print(f"  Content features: {'enabled' if include_content else 'disabled'}")
        print()

    # Parse email input
    try:
        email = parse_email_input(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print(f"Email:")
        print(f"  Subject: {email.get('subject', '(empty)')}")
        body_preview = email.get('body', '')[:100].replace('\n', ' ')
        if len(email.get('body', '')) > 100:
            body_preview += '...'
        print(f"  Body: {body_preview or '(empty)'}")
        print()

    # Extract features
    features = extract_features(email, include_content=include_content)

    # Run inference
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        output = model(x)

        action_probs = F.softmax(output.action_logits, dim=-1).squeeze().numpy()
        timing_probs = F.softmax(output.timing_logits, dim=-1).squeeze().numpy()
        priority = output.priority.squeeze().item()

    # Get top features
    top_features = get_top_features(features, args.top_features)

    # Format output
    if args.json:
        print(format_json_output(email, action_probs, timing_probs, priority, top_features))
    else:
        print(format_prediction(
            action_probs, timing_probs, priority, top_features, verbose=args.verbose
        ))


if __name__ == '__main__':
    main()
