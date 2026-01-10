"""Tests for stage 13: enhanced cluster analysis."""

from __future__ import annotations

from unittest import mock

import pytest

from rl_emails.pipeline.stages.stage_13_enhance_clusters import _run_async, run
from rl_emails.services.project_detector import ProjectDetectionSummary


@pytest.fixture
def mock_config() -> mock.MagicMock:
    """Create a mock config."""
    config = mock.MagicMock()
    config.database_url = "postgresql://localhost/test"
    config.is_multi_tenant = False
    config.user_id = None
    return config


@pytest.fixture
def mock_config_multi_tenant() -> mock.MagicMock:
    """Create a mock config for multi-tenant mode."""
    import uuid

    config = mock.MagicMock()
    config.database_url = "postgresql://localhost/test"
    config.is_multi_tenant = True
    config.user_id = uuid.uuid4()
    return config


class TestRun:
    """Tests for the run function."""

    def test_run_calls_async(self, mock_config: mock.MagicMock) -> None:
        """Test that run calls the async implementation."""
        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=10,
            projects_detected=3,
            active_projects=2,
            stale_projects=1,
            completed_projects=0,
            detection_results=[],
        )

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=10)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=10)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            result = run(mock_config)

            assert result.success is True
            assert result.records_processed == 10


class TestRunAsync:
    """Tests for the _run_async function."""

    @pytest.mark.asyncio
    async def test_run_async_without_llm(self, mock_config: mock.MagicMock) -> None:
        """Test running without LLM API keys."""
        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=10,
            projects_detected=3,
            active_projects=2,
            stale_projects=1,
            completed_projects=0,
            detection_results=[],
        )

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=10)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=10)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            result = await _run_async(mock_config)

            assert result.success is True
            assert result.records_processed == 10
            assert "3 projects detected" in result.message
            assert "0 labeled" in result.message

    @pytest.mark.asyncio
    async def test_run_async_with_openai_key(self, mock_config: mock.MagicMock) -> None:
        """Test running with OpenAI API key."""
        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=10,
            projects_detected=3,
            active_projects=2,
            stale_projects=1,
            completed_projects=0,
            detection_results=[],
        )

        from rl_emails.services.cluster_labeler import LabelResult

        mock_label_results = [
            LabelResult(
                cluster_id=1, dimension="content", label="Label 1", confidence=0.9, success=True
            ),
            LabelResult(
                cluster_id=2, dimension="content", label="Label 2", confidence=0.85, success=True
            ),
            LabelResult(
                cluster_id=3, dimension="content", label=None, confidence=0.0, success=False
            ),
        ]

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch("rl_emails.services.cluster_labeler.ClusterLabelerService") as MockLabeler,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=10)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=10)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_labeler = MockLabeler.return_value
            mock_labeler.label_unlabeled_clusters = mock.AsyncMock(return_value=mock_label_results)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            result = await _run_async(mock_config)

            assert result.success is True
            assert "2 labeled" in result.message

    @pytest.mark.asyncio
    async def test_run_async_with_anthropic_key(self, mock_config: mock.MagicMock) -> None:
        """Test running with Anthropic API key."""
        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=5,
            projects_detected=2,
            active_projects=1,
            stale_projects=1,
            completed_projects=0,
            detection_results=[],
        )

        from rl_emails.services.cluster_labeler import LabelResult

        mock_label_results = [
            LabelResult(
                cluster_id=1, dimension="content", label="Label 1", confidence=0.9, success=True
            ),
        ]

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch("rl_emails.services.cluster_labeler.ClusterLabelerService") as MockLabeler,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "test-key"}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=5)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=5)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_labeler = MockLabeler.return_value
            mock_labeler.label_unlabeled_clusters = mock.AsyncMock(return_value=mock_label_results)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            result = await _run_async(mock_config)

            assert result.success is True
            assert "1 labeled" in result.message

    @pytest.mark.asyncio
    async def test_run_async_multi_tenant(self, mock_config_multi_tenant: mock.MagicMock) -> None:
        """Test running in multi-tenant mode."""
        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=8,
            projects_detected=2,
            active_projects=1,
            stale_projects=1,
            completed_projects=0,
            detection_results=[],
        )

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=8)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=8)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            result = await _run_async(mock_config_multi_tenant)

            assert result.success is True
            # Verify user_id was passed
            mock_detector.compute_temporal_strength.assert_called_once_with(
                user_id=mock_config_multi_tenant.user_id
            )
            mock_detector.enrich_cluster_metadata.assert_called_once_with(
                user_id=mock_config_multi_tenant.user_id
            )
            mock_detector.detect_projects.assert_called_once_with(
                user_id=mock_config_multi_tenant.user_id, min_score=0.3
            )

    @pytest.mark.asyncio
    async def test_run_async_url_conversion(self, mock_config: mock.MagicMock) -> None:
        """Test database URL conversion to async."""
        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=0,
            projects_detected=0,
            active_projects=0,
            stale_projects=0,
            completed_projects=0,
            detection_results=[],
        )

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=0)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=0)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            await _run_async(mock_config)

            # Verify URL was converted to async
            mock_engine.assert_called_once()
            call_args = mock_engine.call_args[0][0]
            assert "postgresql+asyncpg://" in call_args

    @pytest.mark.asyncio
    async def test_run_async_already_asyncpg_url(self) -> None:
        """Test that URL already containing asyncpg is not modified."""
        config = mock.MagicMock()
        config.database_url = "postgresql+asyncpg://localhost/test"  # Already has asyncpg
        config.is_multi_tenant = False
        config.user_id = None

        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=0,
            projects_detected=0,
            active_projects=0,
            stale_projects=0,
            completed_projects=0,
            detection_results=[],
        )

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=0)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=0)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            await _run_async(config)

            # Verify URL was NOT modified (still postgresql+asyncpg)
            mock_engine.assert_called_once()
            call_args = mock_engine.call_args[0][0]
            assert call_args == "postgresql+asyncpg://localhost/test"
            # Make sure it's not doubled (no postgresql+asyncpg+asyncpg)
            assert "asyncpg+asyncpg" not in call_args

    @pytest.mark.asyncio
    async def test_run_async_engine_disposed(self, mock_config: mock.MagicMock) -> None:
        """Test that engine is disposed after run."""
        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=0,
            projects_detected=0,
            active_projects=0,
            stale_projects=0,
            completed_projects=0,
            detection_results=[],
        )

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=0)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=0)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            await _run_async(mock_config)

            # Verify engine was disposed
            mock_engine_instance.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_result_stats(self, mock_config: mock.MagicMock) -> None:
        """Test that result contains correct stats."""
        mock_summary = ProjectDetectionSummary(
            clusters_analyzed=20,
            projects_detected=5,
            active_projects=3,
            stale_projects=1,
            completed_projects=1,
            detection_results=[],
        )

        with (
            mock.patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_engine,
            mock.patch("sqlalchemy.ext.asyncio.async_sessionmaker") as mock_sessionmaker,
            mock.patch(
                "rl_emails.services.project_detector.ProjectDetectorService"
            ) as MockDetector,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}),
        ):
            # Setup mocks
            mock_session = mock.MagicMock()
            mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = mock.AsyncMock(return_value=None)
            mock_sessionmaker.return_value = mock.MagicMock(return_value=mock_session)

            mock_detector = MockDetector.return_value
            mock_detector.compute_temporal_strength = mock.AsyncMock(return_value=20)
            mock_detector.enrich_cluster_metadata = mock.AsyncMock(return_value=20)
            mock_detector.detect_projects = mock.AsyncMock(return_value=mock_summary)

            mock_engine_instance = mock.MagicMock()
            mock_engine_instance.dispose = mock.AsyncMock()
            mock_engine.return_value = mock_engine_instance

            result = await _run_async(mock_config)

            assert result.success is True
            assert result.records_processed == 20
            assert result.duration_seconds > 0
            assert "Enhanced 20 clusters" in result.message
            assert "5 projects detected" in result.message
