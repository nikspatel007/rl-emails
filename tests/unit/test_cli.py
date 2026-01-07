"""Tests for rl_emails.cli."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rl_emails import cli


class TestCli:
    """Tests for CLI entry point."""

    def test_main_runs_script(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that main runs the onboard_data.py script."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch.object(subprocess, "run", return_value=mock_result) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                cli.main()

            assert exc_info.value.code == 0
            mock_run.assert_called_once()

            # Verify the script path is correct
            call_args = mock_run.call_args
            script_path = call_args[0][0][1]  # Second element in first arg list
            assert "onboard_data.py" in script_path

    def test_main_passes_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that main passes CLI arguments through."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        # Set sys.argv to simulate CLI arguments
        monkeypatch.setattr(sys, "argv", ["rl-emails", "--status", "--workers", "5"])

        with patch.object(subprocess, "run", return_value=mock_result) as mock_run:
            with pytest.raises(SystemExit):
                cli.main()

            call_args = mock_run.call_args[0][0]
            assert "--status" in call_args
            assert "--workers" in call_args
            assert "5" in call_args

    def test_main_exits_with_script_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that main exits with the script's return code."""
        mock_result = MagicMock()
        mock_result.returncode = 42

        with patch.object(subprocess, "run", return_value=mock_result):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()

            assert exc_info.value.code == 42

    def test_main_script_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that main exits with error if script not found."""
        # Mock Path.exists to return False for the script path
        original_exists = Path.exists

        def fake_exists(self: Path) -> bool:
            if "onboard_data.py" in str(self):
                return False
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1

    def test_main_runs_from_correct_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that main runs from the project root directory."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch.object(subprocess, "run", return_value=mock_result) as mock_run:
            with pytest.raises(SystemExit):
                cli.main()

            # Check that cwd is set to project root
            call_kwargs = mock_run.call_args[1]
            cwd = call_kwargs.get("cwd")
            assert cwd is not None
            assert Path(cwd).is_dir()


class TestCliModuleExecution:
    """Tests for CLI module __main__ execution."""

    def test_module_name_main(self) -> None:
        """Test that module can be executed directly."""
        # Verify the if __name__ == "__main__" block exists in cli module
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
