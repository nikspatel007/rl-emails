"""CLI entry point for rl-emails pipeline.

Usage:
    rl-emails                    # Run full pipeline
    rl-emails --status           # Check pipeline status
    rl-emails --user UUID        # Run for specific user (multi-tenant)
    rl-emails --help             # Show all options
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rl_emails.core.config import Config
from rl_emails.pipeline import (
    PipelineOptions,
    PipelineOrchestrator,
    format_status,
    get_status,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Email ML pipeline for analyzing Gmail exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current pipeline status",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM classification (saves cost)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        choices=range(0, 12),
        metavar="STAGE",
        help="Start from stage N (0-11, 0=migrations)",
    )
    parser.add_argument(
        "--llm-model",
        choices=["gpt5", "haiku", "sonnet"],
        default="gpt5",
        help="LLM model for classification (default: gpt5)",
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=None,
        help="Maximum emails to classify with LLM",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: .env in project root)",
    )
    # Multi-tenant options
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        metavar="UUID",
        help="User ID for multi-tenant mode (filters data by user)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default=None,
        metavar="UUID",
        help="Organization ID for multi-tenant mode",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for the email onboarding pipeline."""
    args = parse_args()

    # Find .env file
    env_file = args.env_file
    if env_file is None:
        # Try project root
        cli_module = Path(__file__).resolve()
        project_root = cli_module.parent.parent.parent
        env_file = project_root / ".env"

    # Load configuration
    try:
        config = Config.from_env(env_file if env_file.exists() else None)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply multi-tenant context if provided
    if args.user:
        try:
            from uuid import UUID

            user_id = UUID(args.user)
            org_id = UUID(args.org) if args.org else None
            config = config.with_user(user_id, org_id)
            print(f"Multi-tenant mode: user_id={user_id}")
        except ValueError:
            print(f"Invalid UUID format: {args.user}", file=sys.stderr)
            sys.exit(1)

    # Status mode
    if args.status:
        status = get_status(config)
        print(format_status(status))
        return

    # Create pipeline options
    options = PipelineOptions(
        workers=args.workers,
        batch_size=args.batch_size,
        skip_embeddings=args.skip_embeddings,
        skip_llm=args.skip_llm,
        start_from=args.start_from,
        llm_model=args.llm_model,
        llm_limit=args.llm_limit,
    )

    # Create and validate orchestrator
    orchestrator = PipelineOrchestrator(config, options)
    errors = orchestrator.validate()

    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    print(f"Starting pipeline (workers={options.workers}, batch_size={options.batch_size})")

    def on_stage_event(stage: int, event: str, result: object) -> None:
        """Handle stage events for progress output."""
        if event == "start":
            info = orchestrator.get_stage_info()
            stage_info = next((s for s in info if s["number"] == stage), None)
            desc = stage_info["description"] if stage_info else f"Stage {stage}"
            print(f"\n[{stage}/11] {desc}...")
        elif event == "complete" and result:
            from rl_emails.pipeline.stages.base import StageResult

            if isinstance(result, StageResult):
                print(f"  ✓ {result.message} ({result.duration_seconds:.1f}s)")
        elif event == "skip":
            print(f"  - Stage {stage} skipped")
        elif event == "fail" and result:
            from rl_emails.pipeline.stages.base import StageResult

            if isinstance(result, StageResult):
                print(f"  ✗ {result.message}")

    orchestrator.add_callback(on_stage_event)
    result = orchestrator.run()

    # Print summary
    print("\n" + "=" * 60)
    if result.success:
        print(f"PIPELINE COMPLETED in {result.duration_seconds:.1f}s")
        if result.final_status:
            print(format_status(result.final_status))
    else:
        print(f"PIPELINE FAILED: {result.error}")
        if result.stages_failed:
            print(f"Resume with: rl-emails --start-from {result.stages_failed[0]}")

    print("=" * 60)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
