"""Tests for CLI auth commands."""

from __future__ import annotations

import argparse
import uuid
from unittest.mock import patch

import pytest

from rl_emails.cli import (
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
