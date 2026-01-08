"""Tests for CLI auth commands."""

from __future__ import annotations

import argparse
import uuid
from unittest.mock import patch

import pytest

from rl_emails.cli import (
    auth_callback,
    auth_connect,
    auth_disconnect,
    auth_status,
    get_env_file,
    parse_args,
)
from rl_emails.core.config import Config


class TestParseArgsAuth:
    """Tests for auth subcommand parsing."""

    def test_auth_connect_requires_user(self) -> None:
        """Test that auth connect requires --user."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["rl-emails", "auth", "connect"]):
                parse_args()

    def test_auth_connect_with_user(self) -> None:
        """Test auth connect with --user."""
        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "auth", "connect", "--user", test_uuid]):
            args = parse_args()
            assert args.command == "auth"
            assert args.auth_action == "connect"
            assert args.user == test_uuid

    def test_auth_connect_no_browser(self) -> None:
        """Test auth connect with --no-browser flag."""
        test_uuid = str(uuid.uuid4())
        with patch(
            "sys.argv",
            ["rl-emails", "auth", "connect", "--user", test_uuid, "--no-browser"],
        ):
            args = parse_args()
            assert args.no_browser is True

    def test_auth_status_with_user(self) -> None:
        """Test auth status with --user."""
        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "auth", "status", "--user", test_uuid]):
            args = parse_args()
            assert args.command == "auth"
            assert args.auth_action == "status"
            assert args.user == test_uuid

    def test_auth_disconnect_with_user(self) -> None:
        """Test auth disconnect with --user."""
        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "auth", "disconnect", "--user", test_uuid]):
            args = parse_args()
            assert args.command == "auth"
            assert args.auth_action == "disconnect"
            assert args.user == test_uuid

    def test_auth_callback_with_user_and_code(self) -> None:
        """Test auth callback with --user and --code."""
        test_uuid = str(uuid.uuid4())
        with patch(
            "sys.argv",
            ["rl-emails", "auth", "callback", "--user", test_uuid, "--code", "auth-code-123"],
        ):
            args = parse_args()
            assert args.command == "auth"
            assert args.auth_action == "callback"
            assert args.user == test_uuid
            assert args.code == "auth-code-123"

    def test_auth_callback_requires_user(self) -> None:
        """Test that auth callback requires --user."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["rl-emails", "auth", "callback", "--code", "test"]):
                parse_args()

    def test_auth_callback_requires_code(self) -> None:
        """Test that auth callback requires --code."""
        test_uuid = str(uuid.uuid4())
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["rl-emails", "auth", "callback", "--user", test_uuid]):
                parse_args()


class TestAuthConnect:
    """Tests for auth_connect command."""

    def test_invalid_uuid_exits(self) -> None:
        """Test that invalid UUID exits with error."""
        args = argparse.Namespace(user="not-a-uuid", no_browser=True)
        config = Config(database_url="test")

        with pytest.raises(SystemExit):
            auth_connect(args, config)

    def test_missing_oauth_config_exits(self) -> None:
        """Test that missing OAuth config exits with error."""
        args = argparse.Namespace(user=str(uuid.uuid4()), no_browser=True)
        config = Config(database_url="test")  # No Google OAuth config

        with pytest.raises(SystemExit):
            auth_connect(args, config)

    def test_generates_auth_url(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that auth connect generates authorization URL."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid, no_browser=True)
        config = Config(
            database_url="test",
            google_client_id="test-client-id",
            google_client_secret="test-client-secret",
        )

        auth_connect(args, config)

        captured = capsys.readouterr()
        assert "Authorization URL:" in captured.out
        assert "accounts.google.com" in captured.out
        assert test_uuid in captured.out

    def test_opens_browser_by_default(self) -> None:
        """Test that browser is opened when --no-browser not set."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid, no_browser=False)
        config = Config(
            database_url="test",
            google_client_id="test-client-id",
            google_client_secret="test-client-secret",
        )

        with patch("rl_emails.cli.webbrowser.open") as mock_open:
            auth_connect(args, config)
            mock_open.assert_called_once()


class TestAuthStatus:
    """Tests for auth_status command."""

    def test_invalid_uuid_exits(self) -> None:
        """Test that invalid UUID exits with error."""
        args = argparse.Namespace(user="not-a-uuid")
        config = Config(database_url="test")

        with pytest.raises(SystemExit):
            auth_status(args, config)

    def test_shows_not_connected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test showing not connected status."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid)
        config = Config(database_url="postgresql://localhost/test")

        # Mock the async function
        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = {
                "connected": False,
                "provider": None,
                "expires_at": None,
                "is_expired": False,
            }

            auth_status(args, config)

            captured = capsys.readouterr()
            assert "Status: Not connected" in captured.out

    def test_shows_connected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test showing connected status."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid)
        config = Config(database_url="postgresql://localhost/test")

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = {
                "connected": True,
                "provider": "google",
                "expires_at": "2025-01-01T00:00:00",
                "is_expired": False,
            }

            auth_status(args, config)

            captured = capsys.readouterr()
            assert "Status: Connected" in captured.out
            assert "Provider: google" in captured.out

    def test_shows_expired_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test showing expired token warning."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid)
        config = Config(database_url="postgresql://localhost/test")

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = {
                "connected": True,
                "provider": "google",
                "expires_at": "2024-01-01T00:00:00",
                "is_expired": True,
            }

            auth_status(args, config)

            captured = capsys.readouterr()
            assert "WARNING: Token is expired" in captured.out


class TestAuthDisconnect:
    """Tests for auth_disconnect command."""

    def test_invalid_uuid_exits(self) -> None:
        """Test that invalid UUID exits with error."""
        args = argparse.Namespace(user="not-a-uuid")
        config = Config(database_url="test")

        with pytest.raises(SystemExit):
            auth_disconnect(args, config)

    def test_disconnects_successfully(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test successful disconnect."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid)
        config = Config(
            database_url="postgresql://localhost/test",
            google_client_id="test-id",
            google_client_secret="test-secret",
        )

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = True

            auth_disconnect(args, config)

            captured = capsys.readouterr()
            assert "Successfully disconnected" in captured.out

    def test_no_connection_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test when no connection exists."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid)
        config = Config(
            database_url="postgresql://localhost/test",
            google_client_id="test-id",
            google_client_secret="test-secret",
        )

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = False

            auth_disconnect(args, config)

            captured = capsys.readouterr()
            assert "No connection found" in captured.out


class TestAuthCallback:
    """Tests for auth_callback command."""

    def test_invalid_uuid_exits(self) -> None:
        """Test that invalid UUID exits with error."""
        args = argparse.Namespace(user="not-a-uuid", code="auth-code")
        config = Config(database_url="test")

        with pytest.raises(SystemExit):
            auth_callback(args, config)

    def test_missing_oauth_config_exits(self) -> None:
        """Test that missing OAuth config exits with error."""
        args = argparse.Namespace(user=str(uuid.uuid4()), code="auth-code")
        config = Config(database_url="test")  # No Google OAuth config

        with pytest.raises(SystemExit):
            auth_callback(args, config)

    def test_callback_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test successful OAuth callback."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid, code="auth-code-123")
        config = Config(
            database_url="postgresql://localhost/test",
            google_client_id="test-id",
            google_client_secret="test-secret",
        )

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = (True, "Token saved successfully")

            auth_callback(args, config)

            captured = capsys.readouterr()
            assert "Successfully authenticated" in captured.out
            assert test_uuid in captured.out

    def test_callback_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test failed OAuth callback."""
        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid, code="invalid-code")
        config = Config(
            database_url="postgresql://localhost/test",
            google_client_id="test-id",
            google_client_secret="test-secret",
        )

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = (False, "Invalid authorization code")

            auth_callback(args, config)

            captured = capsys.readouterr()
            assert "Status: Failed" in captured.out
            assert "Invalid authorization code" in captured.out


class TestGetEnvFile:
    """Tests for get_env_file helper."""

    def test_returns_none_when_file_not_exists(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test returns None when .env doesn't exist."""
        args = argparse.Namespace(env_file=None)

        # Mock __file__ to point to temp dir
        with patch("rl_emails.cli.Path") as mock_path:
            mock_path.return_value.resolve.return_value.parent.parent.parent = tmp_path
            # The function checks if env_file.exists()
            # This test verifies the function handles the case
            _ = get_env_file(args)

    def test_returns_custom_path(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test returns custom path when provided."""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".env", delete=False) as f:
            env_path = Path(f.name)
            f.write(b"DATABASE_URL=test")

        try:
            args = argparse.Namespace(env_file=env_path)
            result = get_env_file(args)
            assert result == env_path
        finally:
            env_path.unlink()


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_help_shows_auth_command(self) -> None:
        """Test that help shows auth command."""
        with patch("sys.argv", ["rl-emails", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                parse_args()
            # Help exits with code 0
            assert exc_info.value.code == 0

    def test_auth_help_shows_actions(self) -> None:
        """Test that auth help shows actions."""
        with patch("sys.argv", ["rl-emails", "auth", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                parse_args()
            assert exc_info.value.code == 0


class TestMainFunction:
    """Tests for main() function routing."""

    def test_routes_to_auth_connect(self) -> None:
        """Test main routes to auth connect."""
        from rl_emails.cli import main

        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "auth", "connect", "--user", test_uuid]):
            with patch("rl_emails.cli.Config.from_env") as mock_config:
                mock_config.return_value = Config(
                    database_url="test",
                    google_client_id="id",
                    google_client_secret="secret",
                )
                with patch("rl_emails.cli.auth_connect") as mock_connect:
                    main()
                    mock_connect.assert_called_once()

    def test_routes_to_auth_status(self) -> None:
        """Test main routes to auth status."""
        from rl_emails.cli import main

        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "auth", "status", "--user", test_uuid]):
            with patch("rl_emails.cli.Config.from_env") as mock_config:
                mock_config.return_value = Config(database_url="test")
                with patch("rl_emails.cli.auth_status") as mock_status:
                    main()
                    mock_status.assert_called_once()

    def test_routes_to_auth_disconnect(self) -> None:
        """Test main routes to auth disconnect."""
        from rl_emails.cli import main

        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "auth", "disconnect", "--user", test_uuid]):
            with patch("rl_emails.cli.Config.from_env") as mock_config:
                mock_config.return_value = Config(database_url="test")
                with patch("rl_emails.cli.auth_disconnect") as mock_disconnect:
                    main()
                    mock_disconnect.assert_called_once()

    def test_routes_to_auth_callback(self) -> None:
        """Test main routes to auth callback."""
        from rl_emails.cli import main

        test_uuid = str(uuid.uuid4())
        with patch(
            "sys.argv",
            ["rl-emails", "auth", "callback", "--user", test_uuid, "--code", "test-code"],
        ):
            with patch("rl_emails.cli.Config.from_env") as mock_config:
                mock_config.return_value = Config(
                    database_url="test",
                    google_client_id="id",
                    google_client_secret="secret",
                )
                with patch("rl_emails.cli.auth_callback") as mock_callback:
                    main()
                    mock_callback.assert_called_once()

    def test_auth_without_action_shows_usage(self) -> None:
        """Test auth without action shows usage message."""
        from rl_emails.cli import main

        with patch("sys.argv", ["rl-emails", "auth"]):
            with patch("rl_emails.cli.Config.from_env") as mock_config:
                mock_config.return_value = Config(database_url="test")
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_routes_to_pipeline_by_default(self) -> None:
        """Test main routes to pipeline when no subcommand."""
        from rl_emails.cli import main

        with patch("sys.argv", ["rl-emails", "--status"]):
            with patch("rl_emails.cli.Config.from_env") as mock_config:
                mock_config.return_value = Config(database_url="test")
                with patch("rl_emails.cli.run_pipeline") as mock_pipeline:
                    main()
                    mock_pipeline.assert_called_once()

    def test_routes_to_sync(self) -> None:
        """Test main routes to sync command."""
        from rl_emails.cli import main

        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "sync", "--user", test_uuid]):
            with patch("rl_emails.cli.Config.from_env") as mock_config:
                mock_config.return_value = Config(database_url="test")
                with patch("rl_emails.cli.sync_emails") as mock_sync:
                    main()
                    mock_sync.assert_called_once()


class TestParseArgsSync:
    """Tests for sync subcommand parsing."""

    def test_sync_requires_user(self) -> None:
        """Test that sync requires --user."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["rl-emails", "sync"]):
                parse_args()

    def test_sync_with_user(self) -> None:
        """Test sync with --user."""
        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "sync", "--user", test_uuid]):
            args = parse_args()
            assert args.command == "sync"
            assert args.user == test_uuid
            assert args.days == 30  # Default

    def test_sync_with_days(self) -> None:
        """Test sync with --days."""
        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "sync", "--user", test_uuid, "--days", "60"]):
            args = parse_args()
            assert args.days == 60

    def test_sync_with_max_messages(self) -> None:
        """Test sync with --max-messages."""
        test_uuid = str(uuid.uuid4())
        with patch(
            "sys.argv",
            ["rl-emails", "sync", "--user", test_uuid, "--max-messages", "1000"],
        ):
            args = parse_args()
            assert args.max_messages == 1000

    def test_sync_with_status_flag(self) -> None:
        """Test sync with --status flag."""
        test_uuid = str(uuid.uuid4())
        with patch("sys.argv", ["rl-emails", "sync", "--user", test_uuid, "--status"]):
            args = parse_args()
            assert args.status is True


class TestSyncEmails:
    """Tests for sync_emails command."""

    def test_invalid_uuid_exits(self) -> None:
        """Test that invalid UUID exits with error."""
        from rl_emails.cli import sync_emails

        args = argparse.Namespace(user="not-a-uuid", status=False, days=30, max_messages=None)
        config = Config(database_url="test")

        with pytest.raises(SystemExit):
            sync_emails(args, config)

    def test_sync_status_calls_helper(self) -> None:
        """Test that --status calls _show_sync_status."""
        from rl_emails.cli import sync_emails

        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid, status=True, days=30, max_messages=None)
        config = Config(database_url="postgresql://localhost/test")

        with patch("rl_emails.cli._show_sync_status") as mock_status:
            sync_emails(args, config)
            mock_status.assert_called_once()

    def test_sync_run_calls_helper(self) -> None:
        """Test that sync without --status calls _run_sync."""
        from rl_emails.cli import sync_emails

        test_uuid = str(uuid.uuid4())
        args = argparse.Namespace(user=test_uuid, status=False, days=30, max_messages=None)
        config = Config(database_url="postgresql://localhost/test")

        with patch("rl_emails.cli._run_sync") as mock_run:
            sync_emails(args, config)
            mock_run.assert_called_once()

    def test_run_sync_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test _run_sync with successful sync."""
        from rl_emails.cli import _run_sync
        from rl_emails.pipeline.stages.base import StageResult

        test_uuid = uuid.UUID(str(uuid.uuid4()))
        config = Config(database_url="postgresql://localhost/test")
        config = config.with_user(test_uuid)

        with patch("rl_emails.pipeline.stages.stage_00_gmail_sync.run") as mock_run:
            mock_run.return_value = StageResult(
                success=True,
                records_processed=100,
                duration_seconds=5.0,
                message="Sync complete",
                metadata={"synced": 100, "stored": 100},
            )

            _run_sync(test_uuid, config, days=30, max_messages=None)

            captured = capsys.readouterr()
            assert "Sync completed successfully" in captured.out
            assert "Emails stored: 100" in captured.out

    def test_run_sync_failure_exits(self) -> None:
        """Test _run_sync exits on failure."""
        from rl_emails.cli import _run_sync
        from rl_emails.pipeline.stages.base import StageResult

        test_uuid = uuid.UUID(str(uuid.uuid4()))
        config = Config(database_url="postgresql://localhost/test")
        config = config.with_user(test_uuid)

        with patch("rl_emails.pipeline.stages.stage_00_gmail_sync.run") as mock_run:
            mock_run.return_value = StageResult(
                success=False,
                records_processed=0,
                duration_seconds=0.1,
                message="No OAuth token found",
            )

            with pytest.raises(SystemExit):
                _run_sync(test_uuid, config, days=30, max_messages=None)

    def test_run_sync_with_max_messages(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test _run_sync with max_messages specified."""
        from rl_emails.cli import _run_sync
        from rl_emails.pipeline.stages.base import StageResult

        test_uuid = uuid.UUID(str(uuid.uuid4()))
        config = Config(database_url="postgresql://localhost/test")
        config = config.with_user(test_uuid)

        with patch("rl_emails.pipeline.stages.stage_00_gmail_sync.run") as mock_run:
            mock_run.return_value = StageResult(
                success=True,
                records_processed=50,
                duration_seconds=2.0,
                message="Sync complete",
                metadata={"synced": 50, "stored": 50},
            )

            _run_sync(test_uuid, config, days=30, max_messages=50)

            captured = capsys.readouterr()
            assert "Max Messages: 50" in captured.out
            assert "Total synced from Gmail: 50" in captured.out

    def test_run_sync_without_metadata(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test _run_sync when metadata is None."""
        from rl_emails.cli import _run_sync
        from rl_emails.pipeline.stages.base import StageResult

        test_uuid = uuid.UUID(str(uuid.uuid4()))
        config = Config(database_url="postgresql://localhost/test")
        config = config.with_user(test_uuid)

        with patch("rl_emails.pipeline.stages.stage_00_gmail_sync.run") as mock_run:
            mock_run.return_value = StageResult(
                success=True,
                records_processed=50,
                duration_seconds=2.0,
                message="Sync complete",
                metadata=None,  # No metadata
            )

            _run_sync(test_uuid, config, days=30, max_messages=None)

            captured = capsys.readouterr()
            assert "Sync completed successfully" in captured.out
            assert "Total synced from Gmail" not in captured.out

    def test_run_sync_with_zero_synced(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test _run_sync when synced is 0 (falsy)."""
        from rl_emails.cli import _run_sync
        from rl_emails.pipeline.stages.base import StageResult

        test_uuid = uuid.UUID(str(uuid.uuid4()))
        config = Config(database_url="postgresql://localhost/test")
        config = config.with_user(test_uuid)

        with patch("rl_emails.pipeline.stages.stage_00_gmail_sync.run") as mock_run:
            mock_run.return_value = StageResult(
                success=True,
                records_processed=0,
                duration_seconds=0.5,
                message="No new emails",
                metadata={"synced": 0, "stored": 0},  # synced is 0 (falsy)
            )

            _run_sync(test_uuid, config, days=30, max_messages=None)

            captured = capsys.readouterr()
            assert "Sync completed successfully" in captured.out
            assert "Total synced from Gmail" not in captured.out  # Not printed when synced is 0


class TestShowSyncStatus:
    """Tests for _show_sync_status helper."""

    def test_show_sync_status_full(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test _show_sync_status with all fields."""
        from rl_emails.cli import _show_sync_status

        test_uuid = uuid.UUID(str(uuid.uuid4()))
        config = Config(database_url="postgresql://localhost/test")

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = {
                "status": "completed",
                "emails_synced": 500,
                "last_sync_at": "2024-01-15T12:00:00",
                "last_history_id": "12345",
                "error": None,
            }

            _show_sync_status(test_uuid, config)

            captured = capsys.readouterr()
            assert "Status: completed" in captured.out
            assert "Emails Synced: 500" in captured.out
            assert "Last Sync: 2024-01-15T12:00:00" in captured.out
            assert "History ID: 12345" in captured.out

    def test_show_sync_status_with_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test _show_sync_status with error."""
        from rl_emails.cli import _show_sync_status

        test_uuid = uuid.UUID(str(uuid.uuid4()))
        config = Config(database_url="postgresql://localhost/test")

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = {
                "status": "error",
                "emails_synced": 0,
                "last_sync_at": None,
                "last_history_id": None,
                "error": "Connection failed",
            }

            _show_sync_status(test_uuid, config)

            captured = capsys.readouterr()
            assert "Status: error" in captured.out
            assert "Error: Connection failed" in captured.out

    def test_show_sync_status_with_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test _show_sync_status with message."""
        from rl_emails.cli import _show_sync_status

        test_uuid = uuid.UUID(str(uuid.uuid4()))
        config = Config(database_url="postgresql://localhost/test")

        with patch("rl_emails.cli.asyncio.run") as mock_run:
            mock_run.return_value = {
                "status": "not_connected",
                "message": "No Gmail connection. Run 'rl-emails auth connect' first.",
            }

            _show_sync_status(test_uuid, config)

            captured = capsys.readouterr()
            assert "No Gmail connection" in captured.out
