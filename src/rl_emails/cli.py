"""CLI entry point for rl-emails pipeline.

Usage:
    rl-emails                    # Run full pipeline
    rl-emails --status           # Check pipeline status
    rl-emails --user UUID        # Run for specific user (multi-tenant)
    rl-emails auth connect       # Connect Gmail via OAuth
    rl-emails auth status        # Check OAuth connection status
    rl-emails auth disconnect    # Disconnect Gmail
    rl-emails sync --user UUID   # Sync emails from Gmail
    rl-emails sync --status      # Check sync status
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

    # Auth callback (to complete OAuth flow manually)
    auth_callback = auth_subparsers.add_parser(
        "callback", help="Complete OAuth flow with authorization code"
    )
    auth_callback.add_argument(
        "--user",
        type=str,
        required=True,
        metavar="UUID",
        help="User ID to authenticate",
    )
    auth_callback.add_argument(
        "--code",
        type=str,
        required=True,
        help="Authorization code from OAuth redirect URL",
    )

    # Sync subcommand
    sync_parser = subparsers.add_parser("sync", help="Sync emails from Gmail")
    sync_parser.add_argument(
        "--user",
        type=str,
        required=True,
        metavar="UUID",
        help="User ID to sync emails for",
    )
    sync_parser.add_argument(
        "--days",
        type=int,
        default=15,
        help="Number of days to sync (default: 15)",
    )
    sync_parser.add_argument(
        "--max-messages",
        type=int,
        default=1000,
        help="Maximum messages to sync (default: 1000)",
    )
    sync_parser.add_argument(
        "--status",
        action="store_true",
        help="Show sync status instead of syncing",
    )

    # Onboard subcommand - progressive sync with full pipeline
    onboard_parser = subparsers.add_parser(
        "onboard",
        help="Onboard user with progressive sync (7d quick, then background)",
    )
    onboard_parser.add_argument(
        "--user",
        type=str,
        required=True,
        metavar="UUID",
        help="User ID to onboard",
    )
    onboard_parser.add_argument(
        "--quick-only",
        action="store_true",
        help="Only run quick phase (7 days), skip background sync",
    )
    onboard_parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation",
    )
    onboard_parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM classification",
    )
    onboard_parser.add_argument(
        "--llm-limit",
        type=int,
        default=50,
        help="Max emails to classify with LLM per phase (default: 50)",
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
        default=35,
        help="Number of parallel workers (default: 35)",
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


def auth_callback(args: argparse.Namespace, config: Config) -> None:
    """Handle auth callback command to complete OAuth flow."""
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

    async def _exchange_code() -> tuple[bool, str]:  # pragma: no cover
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

            try:
                await service.complete_auth_flow(
                    user_id=user_id,
                    code=args.code,
                )
                return True, "Token saved successfully"
            except Exception as e:
                return False, str(e)

    success, message = asyncio.run(_exchange_code())

    print("Gmail OAuth Callback")
    print("=" * 60)
    print(f"\nUser ID: {user_id}")

    if success:
        print("Status: Successfully authenticated!")
        print("\nYou can now sync your emails:")
        print(f"  uv run rl-emails sync --user {user_id}")
    else:
        print("Status: Failed")
        print(f"Error: {message}")

    print("=" * 60)


def sync_emails(args: argparse.Namespace, config: Config) -> None:
    """Handle sync command - sync emails from Gmail."""
    # Validate user UUID
    try:
        user_id = UUID(args.user)
    except ValueError:
        print(f"Invalid UUID format: {args.user}", file=sys.stderr)
        sys.exit(1)

    # Apply user to config
    config = config.with_user(user_id)

    if args.status:
        # Show sync status
        _show_sync_status(user_id, config)
    else:
        # Run sync
        _run_sync(user_id, config, args.days, args.max_messages)


def _show_sync_status(user_id: UUID, config: Config) -> None:
    """Show sync status for a user."""

    async def _get_status() -> dict[str, object]:  # pragma: no cover
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

        from rl_emails.integrations.gmail.client import GmailClient
        from rl_emails.repositories.oauth_token import OAuthTokenRepository
        from rl_emails.repositories.sync_state import SyncStateRepository
        from rl_emails.services.sync_service import SyncService

        # Convert database URL for async
        db_url = config.database_url
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        engine = create_async_engine(db_url)
        async with AsyncSession(engine) as session:
            # Check OAuth token
            token_repo = OAuthTokenRepository(session)
            token = await token_repo.get_by_user(user_id)

            if token is None:
                return {
                    "connected": False,
                    "sync_status": "not_connected",
                    "message": "No Gmail connection. Run 'rl-emails auth connect' first.",
                }

            # Get sync status
            async with GmailClient(access_token=token.access_token) as gmail_client:
                sync_repo = SyncStateRepository(session)
                service = SyncService(gmail_client=gmail_client, sync_repo=sync_repo)
                return await service.get_sync_status(user_id)

    status = asyncio.run(_get_status())

    print("Gmail Sync Status")
    print("=" * 60)
    print(f"\nUser ID: {user_id}")
    print(f"Status: {status.get('status', 'unknown')}")

    if status.get("emails_synced"):
        print(f"Emails Synced: {status['emails_synced']}")

    if status.get("last_sync_at"):
        print(f"Last Sync: {status['last_sync_at']}")

    if status.get("last_history_id"):
        print(f"History ID: {status['last_history_id']}")

    if status.get("error"):
        print(f"Error: {status['error']}")

    if status.get("message"):
        print(f"\n{status['message']}")

    print("=" * 60)


def _run_sync(user_id: UUID, config: Config, days: int, max_messages: int | None) -> None:
    """Run Gmail sync for a user."""
    from rl_emails.pipeline.stages import stage_00_gmail_sync

    print("Gmail Sync")
    print("=" * 60)
    print(f"\nUser ID: {user_id}")
    print(f"Days: {days}")
    if max_messages:
        print(f"Max Messages: {max_messages}")
    print("\nSyncing emails...")

    result = stage_00_gmail_sync.run(config, days=days, max_messages=max_messages)

    if result.success:
        print("\nSync completed successfully!")
        print(f"Emails stored: {result.records_processed}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        if result.metadata:
            if result.metadata.get("synced"):
                print(f"Total synced from Gmail: {result.metadata['synced']}")
    else:
        print(f"\nSync failed: {result.message}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)


def onboard_user(args: argparse.Namespace, config: Config) -> None:
    """Onboard a user with progressive sync and full pipeline processing."""
    # Validate user UUID
    try:
        user_id = UUID(args.user)
    except ValueError:
        print(f"Invalid UUID format: {args.user}", file=sys.stderr)
        sys.exit(1)

    # Apply multi-tenant context
    config = config.with_user(user_id)

    print("Progressive User Onboarding")
    print("=" * 60)
    print(f"\nUser ID: {user_id}")
    print(f"Quick Only: {args.quick_only}")
    print(f"Skip Embeddings: {args.skip_embeddings}")
    print(f"Skip LLM: {args.skip_llm}")
    print(f"LLM Limit: {args.llm_limit}")
    print("\nStarting progressive sync...")

    async def _run_onboard() -> None:  # pragma: no cover
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

        from rl_emails.auth.google import GoogleOAuth
        from rl_emails.integrations.gmail.client import GmailClient
        from rl_emails.repositories.oauth_token import OAuthTokenRepository
        from rl_emails.repositories.sync_state import SyncStateRepository
        from rl_emails.services.auth_service import AuthService
        from rl_emails.services.progressive_sync import (
            PhaseConfig,
            ProgressiveSyncService,
            SyncPhase,
        )

        # Convert database URL for async
        db_url = config.database_url
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        engine = create_async_engine(db_url)
        async with AsyncSession(engine) as session:
            # Use AuthService to get a valid (refreshed if needed) token
            token_repo = OAuthTokenRepository(session)
            oauth = GoogleOAuth(
                client_id=config.google_client_id or "",
                client_secret=config.google_client_secret or "",
                redirect_uri=config.google_redirect_uri,
            )
            auth_service = AuthService(oauth=oauth, token_repo=token_repo)

            try:
                access_token = await auth_service.get_valid_token(user_id)
            except ValueError as e:
                print(f"\nNo Gmail connection found: {e}", file=sys.stderr)
                print("Run 'rl-emails auth connect' first.", file=sys.stderr)
                sys.exit(1)

            # Create service with valid token
            async with GmailClient(access_token=access_token) as gmail_client:
                sync_repo = SyncStateRepository(session)
                service = ProgressiveSyncService(
                    gmail_client=gmail_client,
                    sync_repo=sync_repo,
                    session=session,
                )

                # Build phases
                phases = [
                    PhaseConfig(
                        phase=SyncPhase.QUICK,
                        days_start=0,
                        days_end=7,
                        batch_size=100,
                        run_embeddings=not args.skip_embeddings,
                        run_llm=not args.skip_llm,
                        llm_limit=args.llm_limit,
                    ),
                ]

                if not args.quick_only:
                    phases.append(
                        PhaseConfig(
                            phase=SyncPhase.STANDARD,
                            days_start=7,
                            days_end=30,
                            batch_size=200,
                            run_embeddings=not args.skip_embeddings,
                            run_llm=not args.skip_llm,
                            llm_limit=args.llm_limit,
                        )
                    )

                # Run progressive sync
                import time

                start_time = time.time()

                async for progress in service.sync_progressive(
                    user_id=user_id,
                    config=config,
                    phases=phases,
                ):
                    phase_name = progress.phase.value.upper()
                    if progress.phase_complete:
                        print(f"\n[{phase_name}] Phase Complete!")
                        print(f"  Emails fetched: {progress.emails_fetched}")
                        print(f"  Emails processed: {progress.emails_processed}")
                        print(f"  Features computed: {progress.features_computed}")
                        print(f"  Embeddings: {progress.embeddings_generated}")
                        print(f"  LLM classified: {progress.llm_classified}")
                        print(f"  Clusters updated: {progress.clusters_updated}")
                        print(f"  Priority computed: {progress.priority_computed}")
                    elif progress.error:
                        print(f"\n[{phase_name}] Error: {progress.error}", file=sys.stderr)
                    else:
                        # Progress update
                        total = progress.total_in_phase or "?"
                        print(
                            f"\r[{phase_name}] Fetched: {progress.emails_fetched}/{total}, "
                            f"Processed: {progress.emails_processed}",
                            end="",
                            flush=True,
                        )

                duration = time.time() - start_time
                print(f"\n\nOnboarding completed in {duration:.1f}s")

    asyncio.run(_run_onboard())
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

    # Set up logging with rolling file handler
    from rl_emails.core.logging import setup_logging

    setup_logging(console=True, file_logging=True)

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
        elif args.auth_action == "callback":
            auth_callback(args, config)
        else:
            print("Usage: rl-emails auth {connect|status|disconnect|callback} --user UUID")
            sys.exit(1)
    elif args.command == "sync":
        sync_emails(args, config)
    elif args.command == "onboard":
        onboard_user(args, config)
    else:
        run_pipeline(args, config)


if __name__ == "__main__":
    main()
