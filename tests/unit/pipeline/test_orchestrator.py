"""Tests for pipeline orchestrator module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rl_emails.core.config import Config
from rl_emails.pipeline.orchestrator import (
    PipelineOptions,
    PipelineOrchestrator,
    PipelineResult,
)
from rl_emails.pipeline.stages.base import StageResult


class TestPipelineOptions:
    """Tests for PipelineOptions dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        options = PipelineOptions()
        assert options.workers == 10
        assert options.batch_size == 100
        assert options.skip_embeddings is False
        assert options.skip_llm is False
        assert options.start_from == 0
        assert options.llm_model == "gpt5"
        assert options.llm_limit is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        options = PipelineOptions(
            workers=5,
            batch_size=50,
            skip_embeddings=True,
            skip_llm=True,
            start_from=3,
            llm_model="haiku",
            llm_limit=100,
        )
        assert options.workers == 5
        assert options.batch_size == 50
        assert options.skip_embeddings is True
        assert options.skip_llm is True
        assert options.start_from == 3
        assert options.llm_model == "haiku"
        assert options.llm_limit == 100


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        result = PipelineResult(success=True)
        assert result.success is True
        assert result.stages_completed == []
        assert result.stages_skipped == []
        assert result.stages_failed == []
        assert result.duration_seconds == 0.0
        assert result.final_status is None
        assert result.error is None

    def test_message_success(self) -> None:
        """Test message property for success."""
        result = PipelineResult(
            success=True,
            stages_completed=[1, 2, 3],
            duration_seconds=10.5,
        )
        assert "3 stages" in result.message
        assert "10.5s" in result.message

    def test_message_failure(self) -> None:
        """Test message property for failure."""
        result = PipelineResult(
            success=False,
            stages_failed=[5],
        )
        assert "stage 5" in result.message

    def test_message_error(self) -> None:
        """Test message property with error."""
        result = PipelineResult(
            success=False,
            error="Something broke",
        )
        assert "Something broke" in result.message


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator class."""

    def _make_config(self, **kwargs: str | Path | None) -> Config:
        """Create a test config."""
        defaults: dict[str, str | Path | None] = {
            "database_url": "postgresql://localhost/test",
            "mbox_path": Path("/tmp/test.mbox"),
            "your_email": "test@example.com",
            "openai_api_key": "sk-test",
            "anthropic_api_key": "ant-test",
        }
        defaults.update(kwargs)  # type: ignore[arg-type]
        return Config(**defaults)  # type: ignore[arg-type]

    def test_init_with_defaults(self) -> None:
        """Test initialization with default options."""
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        assert orchestrator.config == config
        assert orchestrator.options.workers == 10

    def test_init_with_options(self) -> None:
        """Test initialization with custom options."""
        config = self._make_config()
        options = PipelineOptions(workers=5)
        orchestrator = PipelineOrchestrator(config, options)
        assert orchestrator.options.workers == 5

    def test_get_stage_info(self) -> None:
        """Test getting stage info."""
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        info = orchestrator.get_stage_info()

        assert len(info) == 13
        assert info[0]["number"] == 1
        assert info[0]["name"] == "parse_mbox"
        assert info[5]["requires_openai"] is True
        assert info[10]["requires_llm"] is True
        assert info[11]["name"] == "entity_extraction"
        assert info[12]["name"] == "enhance_clusters"

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_missing_database_url(self, mock_check: MagicMock) -> None:
        """Test validation with missing database_url."""
        config = Config(database_url="")
        orchestrator = PipelineOrchestrator(config)
        errors = orchestrator.validate()
        assert "DATABASE_URL not configured" in errors

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_missing_mbox_path(self, mock_check: MagicMock) -> None:
        """Test validation with missing mbox_path."""
        config = self._make_config(mbox_path=None)
        orchestrator = PipelineOrchestrator(config)
        errors = orchestrator.validate()
        assert "MBOX_PATH not configured" in errors

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_mbox_not_found(self, mock_check: MagicMock) -> None:
        """Test validation with non-existent mbox file."""
        config = self._make_config(mbox_path=Path("/nonexistent/file.mbox"))
        orchestrator = PipelineOrchestrator(config)
        errors = orchestrator.validate()
        assert any("not found" in e for e in errors)

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_missing_your_email(self, mock_check: MagicMock) -> None:
        """Test validation with missing your_email."""
        config = self._make_config(your_email=None)
        orchestrator = PipelineOrchestrator(config)
        errors = orchestrator.validate()
        assert "YOUR_EMAIL not configured" in errors

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_missing_openai_key(self, mock_check: MagicMock) -> None:
        """Test validation with missing OpenAI key."""
        config = self._make_config(openai_api_key=None)
        orchestrator = PipelineOrchestrator(config)
        errors = orchestrator.validate()
        assert any("OPENAI_API_KEY" in e for e in errors)

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_missing_llm_key(self, mock_check: MagicMock) -> None:
        """Test validation with missing LLM keys."""
        config = self._make_config(openai_api_key=None, anthropic_api_key=None)
        orchestrator = PipelineOrchestrator(config)
        errors = orchestrator.validate()
        assert any("LLM API key" in e for e in errors)

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_skip_embeddings(self, mock_check: MagicMock) -> None:
        """Test validation with skip_embeddings allows missing OpenAI key."""
        mock_check.return_value = True
        config = self._make_config(openai_api_key=None)
        options = PipelineOptions(skip_embeddings=True, skip_llm=True)
        orchestrator = PipelineOrchestrator(config, options)
        errors = orchestrator.validate()
        assert not any("OPENAI_API_KEY" in e for e in errors)

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_postgres_failure(self, mock_check: MagicMock) -> None:
        """Test validation with postgres connection failure."""
        mock_check.return_value = False
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        errors = orchestrator.validate()
        assert any("PostgreSQL" in e for e in errors)

    def test_add_callback(self) -> None:
        """Test adding callbacks."""
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)

        events: list[tuple[int, str, StageResult | None]] = []

        def callback(stage: int, event: str, result: StageResult | None) -> None:
            events.append((stage, event, result))

        orchestrator.add_callback(callback)
        orchestrator._notify(1, "start", None)

        assert len(events) == 1
        assert events[0] == (1, "start", None)

    @patch("rl_emails.pipeline.orchestrator.subprocess.run")
    def test_run_migrations_success(self, mock_run: MagicMock) -> None:
        """Test successful migration run."""
        mock_run.return_value = MagicMock(returncode=0)
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)

        result = orchestrator.run_migrations()

        assert result is True
        mock_run.assert_called_once()

    @patch("rl_emails.pipeline.orchestrator.subprocess.run")
    def test_run_migrations_failure(self, mock_run: MagicMock) -> None:
        """Test failed migration run."""
        mock_run.return_value = MagicMock(returncode=1)
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)

        result = orchestrator.run_migrations()

        assert result is False

    @patch("rl_emails.pipeline.orchestrator.subprocess.run")
    def test_run_migrations_exception(self, mock_run: MagicMock) -> None:
        """Test migration run with exception."""
        mock_run.side_effect = Exception("Subprocess failed")
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)

        result = orchestrator.run_migrations()

        assert result is False

    @patch("rl_emails.pipeline.orchestrator.get_status")
    @patch("rl_emails.pipeline.orchestrator.stage_01_parse_mbox")
    def test_run_single_stage(self, mock_stage: MagicMock, mock_status: MagicMock) -> None:
        """Test running a single stage."""
        mock_stage.run.return_value = StageResult(
            success=True,
            records_processed=100,
            duration_seconds=1.0,
            message="Success",
        )
        mock_status.return_value = MagicMock()

        config = self._make_config()
        options = PipelineOptions(start_from=1, skip_embeddings=True, skip_llm=True)
        orchestrator = PipelineOrchestrator(config, options)

        result = orchestrator.run(run_migrations=False)

        # Stage 1 should be completed, others skipped (embeddings, llm)
        assert 1 in result.stages_completed
        mock_stage.run.assert_called_once()

    @patch("rl_emails.pipeline.orchestrator.get_status")
    @patch("rl_emails.pipeline.orchestrator.stage_01_parse_mbox")
    def test_run_stage_failure(self, mock_stage: MagicMock, mock_status: MagicMock) -> None:
        """Test handling stage failure."""
        mock_stage.run.return_value = StageResult(
            success=False,
            records_processed=0,
            duration_seconds=1.0,
            message="Stage failed",
        )
        mock_status.return_value = MagicMock()

        config = self._make_config()
        options = PipelineOptions(start_from=1)
        orchestrator = PipelineOrchestrator(config, options)

        result = orchestrator.run(run_migrations=False)

        assert result.success is False
        assert 1 in result.stages_failed
        assert "Stage 1" in str(result.error)

    @patch("rl_emails.pipeline.orchestrator.get_status")
    @patch("rl_emails.pipeline.orchestrator.stage_01_parse_mbox")
    def test_run_stage_exception(self, mock_stage: MagicMock, mock_status: MagicMock) -> None:
        """Test handling stage exception."""
        mock_stage.run.side_effect = Exception("Stage crashed")
        mock_status.return_value = MagicMock()

        config = self._make_config()
        options = PipelineOptions(start_from=1)
        orchestrator = PipelineOrchestrator(config, options)

        result = orchestrator.run(run_migrations=False)

        assert result.success is False
        assert 1 in result.stages_failed

    @patch("rl_emails.pipeline.orchestrator.get_status")
    def test_run_skips_earlier_stages(self, mock_status: MagicMock) -> None:
        """Test skipping stages before start_from."""
        mock_status.return_value = MagicMock()

        config = self._make_config()
        options = PipelineOptions(
            start_from=14, skip_embeddings=True, skip_llm=True
        )  # After all 13 stages
        orchestrator = PipelineOrchestrator(config, options)

        result = orchestrator.run(run_migrations=False)

        assert result.success is True
        assert len(result.stages_skipped) == 13

    @patch("rl_emails.pipeline.orchestrator.get_status")
    @patch("rl_emails.pipeline.orchestrator.subprocess.run")
    def test_run_migrations_failure_stops_pipeline(
        self, mock_run: MagicMock, mock_status: MagicMock
    ) -> None:
        """Test that migration failure stops the pipeline."""
        mock_run.return_value = MagicMock(returncode=1)

        config = self._make_config()
        options = PipelineOptions(start_from=0)
        orchestrator = PipelineOrchestrator(config, options)

        result = orchestrator.run(run_migrations=True)

        assert result.success is False
        assert "migrations failed" in str(result.error)

    @patch("rl_emails.pipeline.orchestrator.stage_01_parse_mbox")
    def test_run_stage_by_number(self, mock_stage: MagicMock) -> None:
        """Test running a specific stage by number."""
        mock_stage.run.return_value = StageResult(
            success=True,
            records_processed=100,
            duration_seconds=1.0,
            message="Success",
        )

        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(1)

        assert result.success is True
        mock_stage.run.assert_called_once()

    @patch("rl_emails.pipeline.orchestrator.stage_02_import_postgres")
    def test_run_stage_02(self, mock_stage: MagicMock) -> None:
        """Test running stage 2."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(2)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_03_populate_threads")
    def test_run_stage_03(self, mock_stage: MagicMock) -> None:
        """Test running stage 3."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(3)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_04_enrich_emails")
    def test_run_stage_04(self, mock_stage: MagicMock) -> None:
        """Test running stage 4."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(4)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_05_compute_features")
    def test_run_stage_05(self, mock_stage: MagicMock) -> None:
        """Test running stage 5."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(5)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_06_compute_embeddings")
    def test_run_stage_06(self, mock_stage: MagicMock) -> None:
        """Test running stage 6."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(6)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_07_classify_handleability")
    def test_run_stage_07(self, mock_stage: MagicMock) -> None:
        """Test running stage 7."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(7)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_08_populate_users")
    def test_run_stage_08(self, mock_stage: MagicMock) -> None:
        """Test running stage 8."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(8)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_09_cluster_emails")
    def test_run_stage_09(self, mock_stage: MagicMock) -> None:
        """Test running stage 9."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(9)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_10_compute_priority")
    def test_run_stage_10(self, mock_stage: MagicMock) -> None:
        """Test running stage 10."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(10)
        assert result.success is True

    @patch("rl_emails.pipeline.orchestrator.stage_11_llm_classification")
    def test_run_stage_11(self, mock_stage: MagicMock) -> None:
        """Test running stage 11."""
        mock_stage.run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(11)
        assert result.success is True

    @patch("rl_emails.pipeline.stages.stage_12_entity_extraction.run")
    def test_run_stage_12(self, mock_run: MagicMock) -> None:
        """Test running stage 12."""
        mock_run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(12)
        assert result.success is True

    @patch("rl_emails.pipeline.stages.stage_13_enhance_clusters.run")
    def test_run_stage_13(self, mock_run: MagicMock) -> None:
        """Test running stage 13."""
        mock_run.return_value = StageResult(
            success=True, records_processed=100, duration_seconds=1.0, message="OK"
        )
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_stage(13)
        assert result.success is True

    def test_run_stage_invalid_number(self) -> None:
        """Test running invalid stage number raises error."""
        config = self._make_config()
        orchestrator = PipelineOrchestrator(config)

        import pytest

        with pytest.raises(ValueError, match="Invalid stage number"):
            orchestrator.run_stage(99)

    @patch("rl_emails.pipeline.orchestrator.get_status")
    def test_run_notifies_callbacks(self, mock_status: MagicMock) -> None:
        """Test that callbacks are notified for stage events."""
        mock_status.return_value = MagicMock()

        config = self._make_config()
        options = PipelineOptions(start_from=14)  # Skip all 13 stages
        orchestrator = PipelineOrchestrator(config, options)

        events: list[tuple[int, str]] = []

        def callback(stage: int, event: str, result: StageResult | None) -> None:
            events.append((stage, event))

        orchestrator.add_callback(callback)
        orchestrator.run(run_migrations=False)

        # All 13 stages should have skip events
        assert len(events) == 13
        assert all(e[1] == "skip" for e in events)

    @patch("rl_emails.pipeline.orchestrator.get_status")
    @patch("rl_emails.pipeline.orchestrator.stage_06_compute_embeddings")
    @patch("rl_emails.pipeline.orchestrator.stage_05_compute_features")
    @patch("rl_emails.pipeline.orchestrator.stage_04_enrich_emails")
    @patch("rl_emails.pipeline.orchestrator.stage_03_populate_threads")
    @patch("rl_emails.pipeline.orchestrator.stage_02_import_postgres")
    @patch("rl_emails.pipeline.orchestrator.stage_01_parse_mbox")
    def test_run_skips_embeddings_stage(
        self,
        mock_s1: MagicMock,
        mock_s2: MagicMock,
        mock_s3: MagicMock,
        mock_s4: MagicMock,
        mock_s5: MagicMock,
        mock_s6: MagicMock,
        mock_status: MagicMock,
    ) -> None:
        """Test that embeddings stage is skipped when skip_embeddings=True."""
        for mock in [mock_s1, mock_s2, mock_s3, mock_s4, mock_s5, mock_s6]:
            mock.run.return_value = StageResult(
                success=True, records_processed=1, duration_seconds=0.1, message="OK"
            )
        mock_status.return_value = MagicMock()

        config = self._make_config()
        options = PipelineOptions(start_from=1, skip_embeddings=True, skip_llm=True)
        orchestrator = PipelineOrchestrator(config, options)

        # Stop after stage 6 by making stage 7 fail (since we can't mock all stages)
        result = orchestrator.run(run_migrations=False)

        # Stage 6 should be skipped
        assert 6 in result.stages_skipped
        mock_s6.run.assert_not_called()

    @patch("rl_emails.pipeline.orchestrator.get_status")
    @patch("rl_emails.pipeline.orchestrator.stage_11_llm_classification")
    @patch("rl_emails.pipeline.orchestrator.stage_10_compute_priority")
    def test_run_skips_llm_stage(
        self,
        mock_s10: MagicMock,
        mock_s11: MagicMock,
        mock_status: MagicMock,
    ) -> None:
        """Test that LLM stage is skipped when skip_llm=True."""
        mock_s10.run.return_value = StageResult(
            success=True, records_processed=1, duration_seconds=0.1, message="OK"
        )
        mock_s11.run.return_value = StageResult(
            success=True, records_processed=1, duration_seconds=0.1, message="OK"
        )
        mock_status.return_value = MagicMock()

        config = self._make_config()
        options = PipelineOptions(
            start_from=10, skip_embeddings=True, skip_llm=True
        )  # Start from stage 10
        orchestrator = PipelineOrchestrator(config, options)

        result = orchestrator.run(run_migrations=False)

        # Stage 11 should be skipped
        assert 11 in result.stages_skipped
        mock_s11.run.assert_not_called()

    @patch("rl_emails.pipeline.orchestrator.get_status")
    @patch("rl_emails.pipeline.orchestrator.subprocess.run")
    @patch("rl_emails.pipeline.orchestrator.stage_01_parse_mbox")
    def test_run_with_successful_migrations(
        self,
        mock_s1: MagicMock,
        mock_subprocess: MagicMock,
        mock_status: MagicMock,
    ) -> None:
        """Test pipeline run with successful migrations."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_s1.run.return_value = StageResult(
            success=True, records_processed=1, duration_seconds=0.1, message="OK"
        )
        mock_status.return_value = MagicMock()

        config = self._make_config()
        options = PipelineOptions(start_from=0, skip_embeddings=True, skip_llm=True)
        orchestrator = PipelineOrchestrator(config, options)

        # Only run first stage, skip others to avoid mocking all
        orchestrator.run(run_migrations=True)

        # Migrations should have been called
        mock_subprocess.assert_called_once()
        # Stage 1 should be attempted
        mock_s1.run.assert_called_once()

    @patch("rl_emails.pipeline.orchestrator.check_postgres")
    def test_validate_with_existing_mbox(self, mock_check: MagicMock, tmp_path: Path) -> None:
        """Test validation when mbox file exists."""
        mock_check.return_value = True

        # Create a temporary mbox file
        mbox_file = tmp_path / "test.mbox"
        mbox_file.touch()

        config = self._make_config(mbox_path=mbox_file)
        orchestrator = PipelineOrchestrator(config)
        errors = orchestrator.validate()

        # Should have no mbox-related errors
        assert not any("MBOX" in e for e in errors)
        assert not any("not found" in e for e in errors)
