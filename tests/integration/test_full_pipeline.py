"""Integration tests for the full pipeline.

These tests verify that the pipeline modules exist and can be imported.
They also test the configuration loading with real files.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rl_emails.core.config import Config
from rl_emails.pipeline import (
    PipelineOptions,
    PipelineOrchestrator,
    PipelineStatus,
    get_status,
)
from rl_emails.pipeline.stages import (
    StageResult,
    stage_01_parse_mbox,
    stage_02_import_postgres,
    stage_03_populate_threads,
    stage_04_enrich_emails,
    stage_05_compute_features,
    stage_06_compute_embeddings,
    stage_07_classify_handleability,
    stage_08_populate_users,
    stage_09_cluster_emails,
    stage_10_compute_priority,
    stage_11_llm_classification,
)


class TestPipelineStagesExist:
    """Test that all pipeline stage modules exist and have run function."""

    STAGE_MODULES = [
        stage_01_parse_mbox,
        stage_02_import_postgres,
        stage_03_populate_threads,
        stage_04_enrich_emails,
        stage_05_compute_features,
        stage_06_compute_embeddings,
        stage_07_classify_handleability,
        stage_08_populate_users,
        stage_09_cluster_emails,
        stage_10_compute_priority,
        stage_11_llm_classification,
    ]

    @pytest.mark.parametrize(
        "module",
        STAGE_MODULES,
        ids=[m.__name__.split(".")[-1] for m in STAGE_MODULES],
    )
    def test_stage_has_run_function(self, module: object) -> None:
        """Test that each stage module has a run function."""
        assert hasattr(module, "run"), f"Stage {module.__name__} missing run function"
        assert callable(module.run)


class TestPipelineImports:
    """Test that pipeline modules can be imported."""

    def test_orchestrator_imports(self) -> None:
        """Test PipelineOrchestrator can be imported."""
        assert PipelineOrchestrator is not None

    def test_pipeline_options_imports(self) -> None:
        """Test PipelineOptions can be imported."""
        assert PipelineOptions is not None

    def test_pipeline_status_imports(self) -> None:
        """Test PipelineStatus can be imported."""
        assert PipelineStatus is not None

    def test_stage_result_imports(self) -> None:
        """Test StageResult can be imported."""
        assert StageResult is not None

    def test_get_status_imports(self) -> None:
        """Test get_status can be imported."""
        assert get_status is not None


class TestConfigurationLoading:
    """Test configuration loading integration."""

    def test_config_loads_from_real_env_file(self, tmp_path: Path) -> None:
        """Test that Config can load from a real .env file."""
        # Create a real .env file
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
DATABASE_URL=postgresql://test:test@localhost:5432/testdb
MBOX_PATH=/tmp/test.mbox
YOUR_EMAIL=test@example.com
OPENAI_API_KEY=sk-test-key
ANTHROPIC_API_KEY=ant-test-key
"""
        )

        # Clear environment variables that might interfere
        old_env = {}
        for key in [
            "DATABASE_URL",
            "MBOX_PATH",
            "YOUR_EMAIL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]:
            if key in os.environ:
                old_env[key] = os.environ.pop(key)

        try:
            config = Config.from_env(env_file=env_file)

            assert config.database_url == "postgresql://test:test@localhost:5432/testdb"
            assert config.mbox_path == Path("/tmp/test.mbox")
            assert config.your_email == "test@example.com"
            assert config.openai_api_key == "sk-test-key"
            assert config.anthropic_api_key == "ant-test-key"
        finally:
            # Restore environment
            os.environ.update(old_env)


class TestCLIStatus:
    """Test CLI --status command."""

    @pytest.mark.skipif(
        "DATABASE_URL" not in os.environ,
        reason="DATABASE_URL not set - skip status check",
    )
    def test_cli_status_runs(self) -> None:
        """Test that CLI status command runs without error."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "rl_emails.cli", "--status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Status command should work if DATABASE_URL is set
        # It may exit with error if database doesn't exist, but shouldn't crash
        assert "Traceback" not in result.stderr or "DATABASE_URL" in result.stderr
