"""Tests for rl_emails.cli."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rl_emails import cli
from rl_emails.pipeline import PipelineResult
from rl_emails.pipeline.status import PipelineStatus


class TestParseArgs:
    """Tests for parse_args function."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default argument values."""
        monkeypatch.setattr(sys, "argv", ["rl-emails"])
        args = cli.parse_args()

        assert args.status is False
        assert args.workers == 10
        assert args.batch_size == 100
        assert args.skip_embeddings is False
        assert args.skip_llm is False
        assert args.start_from == 0
        assert args.llm_model == "gpt5"
        assert args.llm_limit is None
        assert args.env_file is None

    def test_status_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --status flag."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--status"])
        args = cli.parse_args()
        assert args.status is True

    def test_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --workers option."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--workers", "5"])
        args = cli.parse_args()
        assert args.workers == 5

    def test_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --batch-size option."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--batch-size", "50"])
        args = cli.parse_args()
        assert args.batch_size == 50

    def test_skip_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --skip-embeddings and --skip-llm flags."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--skip-embeddings", "--skip-llm"])
        args = cli.parse_args()
        assert args.skip_embeddings is True
        assert args.skip_llm is True

    def test_start_from(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --start-from option."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--start-from", "5"])
        args = cli.parse_args()
        assert args.start_from == 5

    def test_llm_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --llm-model option."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--llm-model", "haiku"])
        args = cli.parse_args()
        assert args.llm_model == "haiku"

    def test_llm_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --llm-limit option."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--llm-limit", "100"])
        args = cli.parse_args()
        assert args.llm_limit == 100

    def test_env_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --env-file option."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--env-file", "/path/.env"])
        args = cli.parse_args()
        assert args.env_file == Path("/path/.env")

    def test_user_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --user option."""
        monkeypatch.setattr(
            sys, "argv", ["rl-emails", "--user", "123e4567-e89b-12d3-a456-426614174000"]
        )
        args = cli.parse_args()
        assert args.user == "123e4567-e89b-12d3-a456-426614174000"

    def test_org_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --org option."""
        monkeypatch.setattr(
            sys, "argv", ["rl-emails", "--org", "987e6543-e21b-12d3-a456-426614174000"]
        )
        args = cli.parse_args()
        assert args.org == "987e6543-e21b-12d3-a456-426614174000"

    def test_user_and_org_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --user and --org options together."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rl-emails",
                "--user",
                "123e4567-e89b-12d3-a456-426614174000",
                "--org",
                "987e6543-e21b-12d3-a456-426614174000",
            ],
        )
        args = cli.parse_args()
        assert args.user == "123e4567-e89b-12d3-a456-426614174000"
        assert args.org == "987e6543-e21b-12d3-a456-426614174000"


class TestCli:
    """Tests for CLI main function."""

    @patch("rl_emails.cli.get_status")
    @patch("rl_emails.cli.Config")
    def test_status_mode(
        self,
        mock_config_cls: MagicMock,
        mock_get_status: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test --status mode."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--status"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        mock_get_status.return_value = PipelineStatus(emails=100, features=80)

        cli.main()

        captured = capsys.readouterr()
        assert "PIPELINE STATUS" in captured.out
        assert "100" in captured.out

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_config_error(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test configuration error handling."""
        monkeypatch.setattr(sys, "argv", ["rl-emails"])

        mock_config_cls.from_env.side_effect = ValueError("DATABASE_URL is required")

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Configuration error" in captured.err

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_validation_errors(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test validation error handling."""
        monkeypatch.setattr(sys, "argv", ["rl-emails"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        mock_orchestrator = MagicMock()
        mock_orchestrator.validate.return_value = ["Error 1", "Error 2"]
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Configuration errors" in captured.err
        assert "Error 1" in captured.err

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_successful_run(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test successful pipeline run."""
        monkeypatch.setattr(sys, "argv", ["rl-emails"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        mock_orchestrator = MagicMock()
        mock_orchestrator.validate.return_value = []
        mock_orchestrator.run.return_value = PipelineResult(
            success=True,
            stages_completed=[1, 2, 3],
            duration_seconds=10.5,
            final_status=PipelineStatus(emails=100),
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "PIPELINE COMPLETED" in captured.out

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_failed_run(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test failed pipeline run."""
        monkeypatch.setattr(sys, "argv", ["rl-emails"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        mock_orchestrator = MagicMock()
        mock_orchestrator.validate.return_value = []
        mock_orchestrator.run.return_value = PipelineResult(
            success=False,
            stages_failed=[3],
            error="Stage 3 failed",
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "PIPELINE FAILED" in captured.out
        assert "Stage 3" in captured.out

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_failed_run_no_stages_failed(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test failed pipeline run without specific stage failure."""
        monkeypatch.setattr(sys, "argv", ["rl-emails"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        mock_orchestrator = MagicMock()
        mock_orchestrator.validate.return_value = []
        mock_orchestrator.run.return_value = PipelineResult(
            success=False,
            stages_failed=[],  # No specific stage
            error="Database migrations failed",
        )
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "PIPELINE FAILED" in captured.out
        assert "Resume with" not in captured.out  # No resume message

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_passes_options(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that CLI options are passed to orchestrator."""
        monkeypatch.setattr(
            sys, "argv", ["rl-emails", "--workers", "5", "--batch-size", "50", "--skip-llm"]
        )

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        mock_orchestrator = MagicMock()
        mock_orchestrator.validate.return_value = []
        mock_orchestrator.run.return_value = PipelineResult(success=True)
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit):
            cli.main()

        # Check options passed to orchestrator
        call_args = mock_orchestrator_cls.call_args
        options = call_args[0][1]
        assert options.workers == 5
        assert options.batch_size == 50
        assert options.skip_llm is True

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_callback_events(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that stage events are printed."""
        from rl_emails.pipeline.stages.base import StageResult

        monkeypatch.setattr(sys, "argv", ["rl-emails"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        mock_orchestrator = MagicMock()
        mock_orchestrator.validate.return_value = []
        mock_orchestrator.get_stage_info.return_value = [
            {"number": 1, "name": "test", "description": "Test Stage"}
        ]
        mock_orchestrator.run.return_value = PipelineResult(
            success=True,
            final_status=None,  # No final status
        )

        # Capture callback for testing
        callbacks: list[object] = []

        def capture_callback(cb: object) -> None:
            callbacks.append(cb)

        mock_orchestrator.add_callback.side_effect = capture_callback
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit):
            cli.main()

        # Verify callback was added
        assert len(callbacks) == 1
        callback = callbacks[0]

        # Test callback events
        callback(1, "start", None)
        callback(1, "complete", StageResult(True, 10, 1.5, "Done"))
        callback(1, "skip", None)
        callback(1, "fail", StageResult(False, 0, 0.1, "Failed"))

        captured = capsys.readouterr()
        assert "Test Stage" in captured.out
        assert "Done" in captured.out
        assert "Stage 1 skipped" in captured.out
        assert "Failed" in captured.out

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_with_custom_env_file(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test using custom env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("DATABASE_URL=test")

        monkeypatch.setattr(sys, "argv", ["rl-emails", "--env-file", str(env_file), "--status"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        with patch("rl_emails.cli.get_status") as mock_status:
            mock_status.return_value = PipelineStatus(emails=50)
            cli.main()

        # Verify from_env was called with the env file path
        mock_config_cls.from_env.assert_called_once_with(env_file)

    @patch("rl_emails.cli.PipelineOrchestrator")
    @patch("rl_emails.cli.Config")
    def test_callback_with_non_stage_result(
        self,
        mock_config_cls: MagicMock,
        mock_orchestrator_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test callback with non-StageResult object."""
        monkeypatch.setattr(sys, "argv", ["rl-emails"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        mock_orchestrator = MagicMock()
        mock_orchestrator.validate.return_value = []
        mock_orchestrator.get_stage_info.return_value = []
        mock_orchestrator.run.return_value = PipelineResult(success=True)

        callbacks: list[object] = []
        mock_orchestrator.add_callback.side_effect = lambda cb: callbacks.append(cb)
        mock_orchestrator_cls.return_value = mock_orchestrator

        with pytest.raises(SystemExit):
            cli.main()

        callback = callbacks[0]

        # Test with non-StageResult objects - should not print anything
        callback(1, "complete", "not a stage result")
        callback(1, "fail", {"error": "not a stage result"})

        # Test unknown event - should do nothing
        callback(1, "unknown", None)

        captured = capsys.readouterr()
        # Should not have printed success/fail messages for non-StageResult
        assert "✓" not in captured.out
        assert "✗" not in captured.out


class TestCliMultiTenant:
    """Tests for CLI multi-tenant mode."""

    @patch("rl_emails.cli.get_status")
    @patch("rl_emails.cli.Config")
    def test_user_flag_sets_context(
        self,
        mock_config_cls: MagicMock,
        mock_get_status: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test --user flag sets multi-tenant context."""
        user_uuid = "123e4567-e89b-12d3-a456-426614174000"
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--user", user_uuid, "--status"])

        mock_config = MagicMock()
        mock_config_with_user = MagicMock()
        mock_config.with_user.return_value = mock_config_with_user
        mock_config_cls.from_env.return_value = mock_config

        mock_get_status.return_value = PipelineStatus(emails=100)

        cli.main()

        # Verify with_user was called with correct UUID
        from uuid import UUID

        mock_config.with_user.assert_called_once()
        call_args = mock_config.with_user.call_args
        assert call_args[0][0] == UUID(user_uuid)
        assert call_args[0][1] is None  # No org_id

        # Verify status was called with the user-scoped config
        mock_get_status.assert_called_once_with(mock_config_with_user)

        captured = capsys.readouterr()
        assert "Multi-tenant mode" in captured.out
        assert user_uuid in captured.out

    @patch("rl_emails.cli.get_status")
    @patch("rl_emails.cli.Config")
    def test_user_and_org_flags_set_context(
        self,
        mock_config_cls: MagicMock,
        mock_get_status: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test --user and --org flags set multi-tenant context."""
        user_uuid = "123e4567-e89b-12d3-a456-426614174000"
        org_uuid = "987e6543-e21b-12d3-a456-426614174000"
        monkeypatch.setattr(
            sys, "argv", ["rl-emails", "--user", user_uuid, "--org", org_uuid, "--status"]
        )

        mock_config = MagicMock()
        mock_config_with_user = MagicMock()
        mock_config.with_user.return_value = mock_config_with_user
        mock_config_cls.from_env.return_value = mock_config

        mock_get_status.return_value = PipelineStatus(emails=100)

        cli.main()

        # Verify with_user was called with both UUIDs
        from uuid import UUID

        mock_config.with_user.assert_called_once()
        call_args = mock_config.with_user.call_args
        assert call_args[0][0] == UUID(user_uuid)
        assert call_args[0][1] == UUID(org_uuid)

    @patch("rl_emails.cli.Config")
    def test_invalid_user_uuid_exits(
        self,
        mock_config_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test invalid --user UUID causes exit."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--user", "not-a-valid-uuid"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Invalid UUID format" in captured.err


class TestCliModuleExecution:
    """Tests for CLI module __main__ execution."""

    def test_module_name_main(self) -> None:
        """Test that module can be executed directly."""
        import ast

        cli_file = cli.__file__
        assert cli_file is not None

        with open(cli_file) as f:
            tree = ast.parse(f.read())

        # Find the if __name__ == "__main__" block
        main_block_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name):
                        if node.test.left.id == "__name__":
                            main_block_found = True
                            break

        assert main_block_found, "Module should have if __name__ == '__main__' block"


class TestParseArgsOnboard:
    """Tests for onboard subcommand argument parsing."""

    def test_onboard_subcommand(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test onboard subcommand is recognized."""
        monkeypatch.setattr(
            sys, "argv", ["rl-emails", "onboard", "--user", "123e4567-e89b-12d3-a456-426614174000"]
        )
        args = cli.parse_args()
        assert args.command == "onboard"
        assert args.user == "123e4567-e89b-12d3-a456-426614174000"

    def test_onboard_quick_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test onboard --quick-only flag."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rl-emails",
                "onboard",
                "--user",
                "123e4567-e89b-12d3-a456-426614174000",
                "--quick-only",
            ],
        )
        args = cli.parse_args()
        assert args.quick_only is True

    def test_onboard_skip_embeddings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test onboard --skip-embeddings flag."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rl-emails",
                "onboard",
                "--user",
                "123e4567-e89b-12d3-a456-426614174000",
                "--skip-embeddings",
            ],
        )
        args = cli.parse_args()
        assert args.skip_embeddings is True

    def test_onboard_skip_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test onboard --skip-llm flag."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rl-emails",
                "onboard",
                "--user",
                "123e4567-e89b-12d3-a456-426614174000",
                "--skip-llm",
            ],
        )
        args = cli.parse_args()
        assert args.skip_llm is True

    def test_onboard_llm_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test onboard --llm-limit flag."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rl-emails",
                "onboard",
                "--user",
                "123e4567-e89b-12d3-a456-426614174000",
                "--llm-limit",
                "50",
            ],
        )
        args = cli.parse_args()
        assert args.llm_limit == 50


class TestOnboardUser:
    """Tests for onboard_user function."""

    @patch("rl_emails.cli.Config")
    def test_onboard_invalid_uuid_exits(
        self,
        mock_config_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test onboard with invalid UUID causes exit."""
        monkeypatch.setattr(sys, "argv", ["rl-emails", "onboard", "--user", "not-a-valid-uuid"])

        mock_config = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Invalid UUID format" in captured.err

    @patch("rl_emails.cli.asyncio.run")
    @patch("rl_emails.cli.Config")
    def test_onboard_prints_header(
        self,
        mock_config_cls: MagicMock,
        mock_asyncio_run: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test onboard prints header information."""
        user_uuid = "123e4567-e89b-12d3-a456-426614174000"
        monkeypatch.setattr(sys, "argv", ["rl-emails", "onboard", "--user", user_uuid])

        mock_config = MagicMock()
        mock_config_with_user = MagicMock()
        mock_config.with_user.return_value = mock_config_with_user
        mock_config_cls.from_env.return_value = mock_config

        # asyncio.run will just complete without doing anything
        mock_asyncio_run.return_value = None

        cli.main()

        captured = capsys.readouterr()
        assert "Progressive User Onboarding" in captured.out
        assert user_uuid in captured.out
        assert "Quick Only: False" in captured.out

    @patch("rl_emails.cli.asyncio.run")
    @patch("rl_emails.cli.Config")
    def test_onboard_with_quick_only(
        self,
        mock_config_cls: MagicMock,
        mock_asyncio_run: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test onboard with --quick-only flag."""
        user_uuid = "123e4567-e89b-12d3-a456-426614174000"
        monkeypatch.setattr(
            sys, "argv", ["rl-emails", "onboard", "--user", user_uuid, "--quick-only"]
        )

        mock_config = MagicMock()
        mock_config_with_user = MagicMock()
        mock_config.with_user.return_value = mock_config_with_user
        mock_config_cls.from_env.return_value = mock_config

        mock_asyncio_run.return_value = None

        cli.main()

        captured = capsys.readouterr()
        assert "Quick Only: True" in captured.out

    @patch("rl_emails.cli.asyncio.run")
    @patch("rl_emails.cli.Config")
    def test_onboard_applies_multi_tenant_context(
        self,
        mock_config_cls: MagicMock,
        mock_asyncio_run: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test onboard applies multi-tenant context correctly."""
        from uuid import UUID

        user_uuid = "123e4567-e89b-12d3-a456-426614174000"
        monkeypatch.setattr(sys, "argv", ["rl-emails", "onboard", "--user", user_uuid])

        mock_config = MagicMock()
        mock_config_with_user = MagicMock()
        mock_config.with_user.return_value = mock_config_with_user
        mock_config_cls.from_env.return_value = mock_config

        mock_asyncio_run.return_value = None

        cli.main()

        # Verify with_user was called with the user UUID
        mock_config.with_user.assert_called_once_with(UUID(user_uuid))
