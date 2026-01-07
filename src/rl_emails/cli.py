"""CLI entry point for rl-emails pipeline.

Usage:
    rl-emails                    # Run full pipeline
    rl-emails --status           # Check pipeline status
    rl-emails --user UUID        # Run for specific user (multi-tenant)
    rl-emails auth connect       # Connect Gmail via OAuth
    rl-emails auth status        # Check OAuth connection status
    rl-emails auth disconnect    # Disconnect Gmail
    rl-emails --help             # Show all options
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser
from pathlib import Path
from uuid import UUID

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

    # Common arguments
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: .env in project root)",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Auth subcommand
    auth_parser = subparsers.add_parser("auth", help="Manage Gmail authentication")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_action", help="Auth actions")

    # Auth connect
    auth_connect = auth_subparsers.add_parser("connect", help="Connect Gmail via OAuth")
    auth_connect.add_argument(
        "--user",
        type=str,
        required=True,
        metavar="UUID",
        help="User ID to connect",
    )
    auth_connect.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    # Auth status
    auth_status = auth_subparsers.add_parser("status", help="Check OAuth connection status")
    auth_status.add_argument(
        "--user",
        type=str,
        required=True,
        metavar="UUID",
        help="User ID to check status for",
    )

    # Auth disconnect
    auth_disconnect = auth_subparsers.add_parser("disconnect", help="Disconnect Gmail")
    auth_disconnect.add_argument(
        "--user",
        type=str,
        required=True,
        metavar="UUID",
        help="User ID to disconnect",
    )

    # Pipeline arguments (for default command)
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


def get_env_file(args: argparse.Namespace) -> Path | None:
    """Get the .env file path from args or default location."""
    env_file = args.env_file
    if env_file is None:
        # Try project root
        cli_module = Path(__file__).resolve()
        project_root = cli_module.parent.parent.parent
        env_file = project_root / ".env"
    return env_file if env_file.exists() else None


def auth_connect(args: argparse.Namespace, config: Config) -> None:
    """Handle auth connect command."""
    # Validate user UUID
    try:
        user_id = UUID(args.user)
    except ValueError:
        print(f"Invalid UUID format: {args.user}", file=sys.stderr)
        sys.exit(1)

    # Check if Google OAuth is configured
    if not config.has_google_oauth():
        print(
            "Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import here to avoid circular imports and reduce startup time
    from rl_emails.auth.google import GoogleOAuth

    oauth = GoogleOAuth(
        client_id=config.google_client_id or "",
        client_secret=config.google_client_secret or "",
        redirect_uri=config.google_redirect_uri,
    )

    # Generate authorization URL with user_id as state
    auth_url = oauth.get_authorization_url(state=str(user_id))

    print("Gmail OAuth Authorization")
    print("=" * 60)
    print(f"\nUser ID: {user_id}")
    print(f"\nAuthorization URL:\n{auth_url}")

    if not args.no_browser:
        print("\nOpening browser...")
        webbrowser.open(auth_url)
    else:
        print("\nOpen this URL in your browser to authorize.")

    print("\n" + "=" * 60)
    print("After authorizing, you'll be redirected to your callback URL.")
    print("Use 'rl-emails auth callback --code CODE --user UUID' to complete setup.")


def auth_status(args: argparse.Namespace, config: Config) -> None:
    """Handle auth status command."""
    # Validate user UUID
    try:
        user_id = UUID(args.user)
    except ValueError:
        print(f"Invalid UUID format: {args.user}", file=sys.stderr)
        sys.exit(1)

    async def _check_status() -> dict[str, bool | str | None]:  # pragma: no cover
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

        from rl_emails.repositories.oauth_token import OAuthTokenRepository

        # Convert database URL for async
        db_url = config.database_url
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        engine = create_async_engine(db_url)
        async with AsyncSession(engine) as session:
            repo = OAuthTokenRepository(session)
            token = await repo.get_by_user(user_id)

            if token is None:
                return {
                    "connected": False,
                    "provider": None,
                    "expires_at": None,
                    "is_expired": False,
                }

            return {
                "connected": True,
                "provider": token.provider,
                "expires_at": str(token.expires_at),
                "is_expired": token.is_expired,
            }

    status = asyncio.run(_check_status())

    print("Gmail OAuth Status")
    print("=" * 60)
    print(f"\nUser ID: {user_id}")

    if status["connected"]:
        print("Status: Connected")
        print(f"Provider: {status['provider']}")
        print(f"Expires: {status['expires_at']}")
        if status["is_expired"]:
            print("WARNING: Token is expired. Run 'auth connect' to re-authorize.")
    else:
        print("Status: Not connected")
        print("\nRun 'rl-emails auth connect --user UUID' to connect Gmail.")

    print("=" * 60)


def auth_disconnect(args: argparse.Namespace, config: Config) -> None:
    """Handle auth disconnect command."""
    # Validate user UUID
    try:
        user_id = UUID(args.user)
    except ValueError:
        print(f"Invalid UUID format: {args.user}", file=sys.stderr)
        sys.exit(1)

    async def _disconnect() -> bool:  # pragma: no cover
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

        from rl_emails.auth.google import GoogleOAuth
        from rl_emails.repositories.oauth_token import OAuthTokenRepository
        from rl_emails.services.auth_service import AuthService

        # Convert database URL for async
        db_url = config.database_url
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        engine = create_async_engine(db_url)
        async with AsyncSession(engine) as session:
            oauth = GoogleOAuth(
                client_id=config.google_client_id or "",
                client_secret=config.google_client_secret or "",
                redirect_uri=config.google_redirect_uri,
            )
            repo = OAuthTokenRepository(session)
            service = AuthService(oauth=oauth, token_repo=repo)

            return await service.revoke_token(user_id)

    disconnected = asyncio.run(_disconnect())

    print("Gmail OAuth Disconnect")
    print("=" * 60)
    print(f"\nUser ID: {user_id}")

    if disconnected:
        print("Status: Successfully disconnected")
        print("\nYour Gmail access has been revoked.")
    else:
        print("Status: No connection found")
        print("\nNo Gmail connection exists for this user.")

    print("=" * 60)


def run_pipeline(args: argparse.Namespace, config: Config) -> None:
    """Run the main pipeline."""
    # Apply multi-tenant context if provided
    if args.user:
        try:
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
                print(f"  OK: {result.message} ({result.duration_seconds:.1f}s)")
        elif event == "skip":
            print(f"  - Stage {stage} skipped")
        elif event == "fail" and result:
            from rl_emails.pipeline.stages.base import StageResult

            if isinstance(result, StageResult):
                print(f"  FAIL: {result.message}")

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


def main() -> None:
    """CLI entry point for the email onboarding pipeline."""
    args = parse_args()

    # Load configuration
    env_file = get_env_file(args)
    try:
        config = Config.from_env(env_file)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Route to appropriate handler
    if args.command == "auth":
        if args.auth_action == "connect":
            auth_connect(args, config)
        elif args.auth_action == "status":
            auth_status(args, config)
        elif args.auth_action == "disconnect":
            auth_disconnect(args, config)
        else:
            print("Usage: rl-emails auth {connect|status|disconnect} --user UUID")
            sys.exit(1)
    else:
        run_pipeline(args, config)


if __name__ == "__main__":
    main()
