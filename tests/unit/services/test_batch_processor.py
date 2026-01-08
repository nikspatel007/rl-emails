"""Tests for BatchProcessor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rl_emails.services.batch_processor import BatchProcessor, BatchResult, StageResult


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful batch result."""
        result = BatchResult(
            success=True,
            emails_stored=10,
            features_computed=10,
            embeddings_generated=10,
            llm_classified=5,
        )

        assert result.success is True
        assert result.emails_stored == 10
        assert result.features_computed == 10
        assert result.embeddings_generated == 10
        assert result.llm_classified == 5
        assert result.error is None

    def test_create_error_result(self) -> None:
        """Test creating an error batch result."""
        result = BatchResult(
            success=False,
            emails_stored=0,
            features_computed=0,
            embeddings_generated=0,
            llm_classified=0,
            error="Database connection failed",
        )

        assert result.success is False
        assert result.error == "Database connection failed"


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful stage result."""
        result = StageResult(
            success=True,
            records_processed=50,
        )

        assert result.success is True
        assert result.records_processed == 50
        assert result.error is None

    def test_create_error_result(self) -> None:
        """Test creating an error stage result."""
        result = StageResult(
            success=False,
            records_processed=0,
            error="Clustering failed",
        )

        assert result.success is False
        assert result.error == "Clustering failed"


class TestBatchProcessorInit:
    """Tests for BatchProcessor initialization."""

    def test_init_stores_dependencies(self) -> None:
        """Test that init stores session and config."""
        session = MagicMock()
        config = MagicMock()

        processor = BatchProcessor(session=session, config=config)

        assert processor.session == session
        assert processor.config == config


class TestBatchProcessorProcessBatch:
    """Tests for BatchProcessor.process_batch method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        config = MagicMock()
        config.openai_api_key = "test-key"
        config.anthropic_api_key = None
        return config

    @pytest.mark.asyncio
    async def test_empty_emails_returns_success(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test processing empty email list returns success."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        result = await processor.process_batch(emails=[])

        assert result.success is True
        assert result.emails_stored == 0
        assert result.features_computed == 0
        assert result.embeddings_generated == 0
        assert result.llm_classified == 0

    @pytest.mark.asyncio
    async def test_process_batch_stores_emails(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch stores emails in database."""
        # Mock raw_emails insert returning row
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)  # raw_id = 1
        mock_session.execute.return_value = raw_result

        # Mock stage runners
        with (
            patch.object(BatchProcessor, "_compute_features", return_value=1) as mock_features,
            patch.object(BatchProcessor, "_classify_rule_based", return_value=1) as mock_rules,
            patch.object(BatchProcessor, "_generate_embeddings", return_value=1) as mock_embeddings,
            patch.object(BatchProcessor, "_classify_llm", return_value=1) as mock_llm,
        ):
            processor = BatchProcessor(session=mock_session, config=mock_config)

            emails = [
                {
                    "message_id": "<test@example.com>",
                    "subject": "Test Subject",
                    "from_email": "sender@example.com",
                    "from_name": "Sender Name",
                    "date_str": "2025-01-07T12:00:00Z",
                    "body_text": "Test body",
                    "labels": ["INBOX"],
                    "to_emails": ["recipient@example.com"],
                }
            ]

            result = await processor.process_batch(emails=emails)

            assert result.success is True
            assert result.emails_stored == 1
            mock_features.assert_called_once()
            mock_rules.assert_called_once()
            mock_embeddings.assert_called_once()
            mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_skips_embeddings_when_disabled(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch skips embeddings when disabled."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        with (
            patch.object(BatchProcessor, "_compute_features", return_value=1),
            patch.object(BatchProcessor, "_classify_rule_based", return_value=1),
            patch.object(BatchProcessor, "_generate_embeddings", return_value=0) as mock_embeddings,
            patch.object(BatchProcessor, "_classify_llm", return_value=0),
        ):
            processor = BatchProcessor(session=mock_session, config=mock_config)

            emails = [{"message_id": "<test@example.com>", "body_text": "Test"}]

            result = await processor.process_batch(
                emails=emails, run_embeddings=False, run_llm=False
            )

            assert result.success is True
            assert result.embeddings_generated == 0
            mock_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_batch_skips_email_without_message_id(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch skips emails without message_id."""
        with (
            patch.object(BatchProcessor, "_compute_features", return_value=0),
            patch.object(BatchProcessor, "_classify_rule_based", return_value=0),
            patch.object(BatchProcessor, "_generate_embeddings", return_value=0),
            patch.object(BatchProcessor, "_classify_llm", return_value=0),
        ):
            processor = BatchProcessor(session=mock_session, config=mock_config)

            emails = [
                {"subject": "No message ID", "body_text": "Test"},
                {"message_id": "", "body_text": "Empty message ID"},
            ]

            result = await processor.process_batch(emails=emails)

            assert result.success is True
            assert result.emails_stored == 0

    @pytest.mark.asyncio
    async def test_process_batch_handles_duplicate_emails(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch handles duplicate emails gracefully."""
        # First call returns row, second returns None (duplicate)
        raw_result1 = MagicMock()
        raw_result1.fetchone.return_value = (1,)
        raw_result2 = MagicMock()
        raw_result2.fetchone.return_value = None

        mock_session.execute.side_effect = [
            raw_result1,
            MagicMock(),  # emails insert
            raw_result2,  # duplicate raw_emails
        ]

        with (
            patch.object(BatchProcessor, "_compute_features", return_value=1),
            patch.object(BatchProcessor, "_classify_rule_based", return_value=1),
            patch.object(BatchProcessor, "_generate_embeddings", return_value=0),
            patch.object(BatchProcessor, "_classify_llm", return_value=0),
        ):
            processor = BatchProcessor(session=mock_session, config=mock_config)

            emails = [
                {"message_id": "<test1@example.com>", "body_text": "First"},
                {"message_id": "<test2@example.com>", "body_text": "Duplicate"},
            ]

            result = await processor.process_batch(
                emails=emails, run_embeddings=False, run_llm=False
            )

            assert result.success is True
            # Only first email stored
            assert result.emails_stored == 1

    @pytest.mark.asyncio
    async def test_process_batch_handles_exception(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch handles exceptions and rolls back."""
        mock_session.execute.side_effect = Exception("Database error")

        processor = BatchProcessor(session=mock_session, config=mock_config)

        emails = [{"message_id": "<test@example.com>", "body_text": "Test"}]

        result = await processor.process_batch(emails=emails)

        assert result.success is False
        assert result.error == "Database error"
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_parses_date(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch correctly parses date strings."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        with (
            patch.object(BatchProcessor, "_compute_features", return_value=1),
            patch.object(BatchProcessor, "_classify_rule_based", return_value=1),
            patch.object(BatchProcessor, "_generate_embeddings", return_value=0),
            patch.object(BatchProcessor, "_classify_llm", return_value=0),
        ):
            processor = BatchProcessor(session=mock_session, config=mock_config)

            emails = [
                {
                    "message_id": "<test@example.com>",
                    "date_str": "2025-01-07T12:00:00Z",
                    "body_text": "Test",
                }
            ]

            result = await processor.process_batch(
                emails=emails, run_embeddings=False, run_llm=False
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_process_batch_handles_invalid_date(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch handles invalid date strings."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        with (
            patch.object(BatchProcessor, "_compute_features", return_value=1),
            patch.object(BatchProcessor, "_classify_rule_based", return_value=1),
            patch.object(BatchProcessor, "_generate_embeddings", return_value=0),
            patch.object(BatchProcessor, "_classify_llm", return_value=0),
        ):
            processor = BatchProcessor(session=mock_session, config=mock_config)

            emails = [
                {
                    "message_id": "<test@example.com>",
                    "date_str": "not-a-valid-date",
                    "body_text": "Test",
                }
            ]

            result = await processor.process_batch(
                emails=emails, run_embeddings=False, run_llm=False
            )

            # Should still succeed, just with None date
            assert result.success is True

    @pytest.mark.asyncio
    async def test_process_batch_handles_sent_emails(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch correctly handles SENT label."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        with (
            patch.object(BatchProcessor, "_compute_features", return_value=1),
            patch.object(BatchProcessor, "_classify_rule_based", return_value=1),
            patch.object(BatchProcessor, "_generate_embeddings", return_value=0),
            patch.object(BatchProcessor, "_classify_llm", return_value=0),
        ):
            processor = BatchProcessor(session=mock_session, config=mock_config)

            emails = [
                {
                    "message_id": "<test@example.com>",
                    "labels": ["SENT", "IMPORTANT"],
                    "body_text": "Test",
                }
            ]

            result = await processor.process_batch(
                emails=emails, run_embeddings=False, run_llm=False
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_process_batch_with_llm_limit(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test process_batch passes LLM limit correctly."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        with (
            patch.object(BatchProcessor, "_compute_features", return_value=1),
            patch.object(BatchProcessor, "_classify_rule_based", return_value=1),
            patch.object(BatchProcessor, "_generate_embeddings", return_value=0),
            patch.object(BatchProcessor, "_classify_llm", return_value=5) as mock_llm,
        ):
            processor = BatchProcessor(session=mock_session, config=mock_config)

            emails = [{"message_id": "<test@example.com>", "body_text": "Test"}]

            result = await processor.process_batch(
                emails=emails, run_embeddings=False, run_llm=True, llm_limit=5
            )

            assert result.success is True
            mock_llm.assert_called_once_with(limit=5)


class TestBatchProcessorComputeFeatures:
    """Tests for BatchProcessor._compute_features method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_compute_features_calls_stage(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test that _compute_features calls the stage runner."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        mock_result = MagicMock()
        mock_result.records_processed = 10

        with patch("rl_emails.pipeline.stages.stage_05_compute_features.run") as mock_run:
            mock_run.return_value = mock_result

            count = await processor._compute_features()

            assert count == 10


class TestBatchProcessorClassifyRuleBased:
    """Tests for BatchProcessor._classify_rule_based method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_classify_rule_based_calls_stage(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test that _classify_rule_based calls the stage runner."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        mock_result = MagicMock()
        mock_result.records_processed = 5

        with patch("rl_emails.pipeline.stages.stage_07_classify_handleability.run") as mock_run:
            mock_run.return_value = mock_result

            count = await processor._classify_rule_based()

            assert count == 5


class TestBatchProcessorGenerateEmbeddings:
    """Tests for BatchProcessor._generate_embeddings method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_generate_embeddings_skips_without_api_key(self, mock_session: AsyncMock) -> None:
        """Test that _generate_embeddings skips when no API key."""
        config = MagicMock()
        config.openai_api_key = None

        processor = BatchProcessor(session=mock_session, config=config)

        count = await processor._generate_embeddings()

        assert count == 0

    @pytest.mark.asyncio
    async def test_generate_embeddings_calls_stage(self, mock_session: AsyncMock) -> None:
        """Test that _generate_embeddings calls the stage runner."""
        config = MagicMock()
        config.openai_api_key = "test-key"

        processor = BatchProcessor(session=mock_session, config=config)

        mock_result = MagicMock()
        mock_result.records_processed = 15

        with patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.run") as mock_run:
            mock_run.return_value = mock_result

            count = await processor._generate_embeddings()

            assert count == 15


class TestBatchProcessorClassifyLLM:
    """Tests for BatchProcessor._classify_llm method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_classify_llm_skips_without_api_keys(self, mock_session: AsyncMock) -> None:
        """Test that _classify_llm skips when no API keys."""
        config = MagicMock()
        config.openai_api_key = None
        config.anthropic_api_key = None

        processor = BatchProcessor(session=mock_session, config=config)

        count = await processor._classify_llm()

        assert count == 0

    @pytest.mark.asyncio
    async def test_classify_llm_with_openai_key(self, mock_session: AsyncMock) -> None:
        """Test that _classify_llm works with OpenAI key."""
        config = MagicMock()
        config.openai_api_key = "test-openai-key"
        config.anthropic_api_key = None

        processor = BatchProcessor(session=mock_session, config=config)

        mock_result = MagicMock()
        mock_result.records_processed = 3

        with patch("rl_emails.pipeline.stages.stage_11_llm_classification.run") as mock_run:
            mock_run.return_value = mock_result

            count = await processor._classify_llm(limit=10)

            assert count == 3

    @pytest.mark.asyncio
    async def test_classify_llm_with_anthropic_key(self, mock_session: AsyncMock) -> None:
        """Test that _classify_llm works with Anthropic key."""
        config = MagicMock()
        config.openai_api_key = None
        config.anthropic_api_key = "test-anthropic-key"

        processor = BatchProcessor(session=mock_session, config=config)

        mock_result = MagicMock()
        mock_result.records_processed = 7

        with patch("rl_emails.pipeline.stages.stage_11_llm_classification.run") as mock_run:
            mock_run.return_value = mock_result

            count = await processor._classify_llm()

            assert count == 7


class TestBatchProcessorUpdateClusters:
    """Tests for BatchProcessor.update_clusters method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_update_clusters_success(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test successful cluster update."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.records_processed = 100
        mock_result.message = "Success"

        with patch("rl_emails.pipeline.stages.stage_09_cluster_emails.run") as mock_run:
            mock_run.return_value = mock_result

            result = await processor.update_clusters()

            assert result.success is True
            assert result.records_processed == 100
            assert result.error is None

    @pytest.mark.asyncio
    async def test_update_clusters_failure(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test cluster update failure."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.records_processed = 0
        mock_result.message = "Not enough data for clustering"

        with patch("rl_emails.pipeline.stages.stage_09_cluster_emails.run") as mock_run:
            mock_run.return_value = mock_result

            result = await processor.update_clusters()

            assert result.success is False
            assert result.error == "Not enough data for clustering"

    @pytest.mark.asyncio
    async def test_update_clusters_handles_exception(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test cluster update handles exceptions."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        with patch("rl_emails.pipeline.stages.stage_09_cluster_emails.run") as mock_run:
            mock_run.side_effect = Exception("Clustering error")

            result = await processor.update_clusters()

            assert result.success is False
            assert "Clustering error" in result.error


class TestBatchProcessorComputePriority:
    """Tests for BatchProcessor.compute_priority method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_compute_priority_success(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test successful priority computation."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.records_processed = 50
        mock_result.message = "Success"

        with patch("rl_emails.pipeline.stages.stage_10_compute_priority.run") as mock_run:
            mock_run.return_value = mock_result

            result = await processor.compute_priority()

            assert result.success is True
            assert result.records_processed == 50

    @pytest.mark.asyncio
    async def test_compute_priority_handles_exception(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test priority computation handles exceptions."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        with patch("rl_emails.pipeline.stages.stage_10_compute_priority.run") as mock_run:
            mock_run.side_effect = Exception("Priority error")

            result = await processor.compute_priority()

            assert result.success is False
            assert "Priority error" in result.error


class TestBatchProcessorPopulateUsers:
    """Tests for BatchProcessor.populate_users method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_populate_users_success(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test successful user population."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.records_processed = 25
        mock_result.message = "Success"

        with patch("rl_emails.pipeline.stages.stage_08_populate_users.run") as mock_run:
            mock_run.return_value = mock_result

            result = await processor.populate_users()

            assert result.success is True
            assert result.records_processed == 25

    @pytest.mark.asyncio
    async def test_populate_users_handles_exception(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test user population handles exceptions."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        with patch("rl_emails.pipeline.stages.stage_08_populate_users.run") as mock_run:
            mock_run.side_effect = Exception("User population error")

            result = await processor.populate_users()

            assert result.success is False
            assert "User population error" in result.error


class TestBatchProcessorBuildThreads:
    """Tests for BatchProcessor.build_threads method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_build_threads_success(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test successful thread building."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.records_processed = 200
        mock_result.message = "Success"

        with patch("rl_emails.pipeline.stages.stage_03_populate_threads.run") as mock_run:
            mock_run.return_value = mock_result

            result = await processor.build_threads()

            assert result.success is True
            assert result.records_processed == 200

    @pytest.mark.asyncio
    async def test_build_threads_handles_exception(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test thread building handles exceptions."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        with patch("rl_emails.pipeline.stages.stage_03_populate_threads.run") as mock_run:
            mock_run.side_effect = Exception("Thread building error")

            result = await processor.build_threads()

            assert result.success is False
            assert "Thread building error" in result.error


class TestBatchProcessorEnrichEmails:
    """Tests for BatchProcessor.enrich_emails method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_enrich_emails_success(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test successful email enrichment."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.records_processed = 150
        mock_result.message = "Success"

        with patch("rl_emails.pipeline.stages.stage_04_enrich_emails.run") as mock_run:
            mock_run.return_value = mock_result

            result = await processor.enrich_emails()

            assert result.success is True
            assert result.records_processed == 150

    @pytest.mark.asyncio
    async def test_enrich_emails_handles_exception(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test email enrichment handles exceptions."""
        processor = BatchProcessor(session=mock_session, config=mock_config)

        with patch("rl_emails.pipeline.stages.stage_04_enrich_emails.run") as mock_run:
            mock_run.side_effect = Exception("Enrichment error")

            result = await processor.enrich_emails()

            assert result.success is False
            assert "Enrichment error" in result.error


class TestBatchProcessorStoreEmails:
    """Tests for BatchProcessor._store_emails method."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_store_emails_handles_none_labels(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test _store_emails handles None labels gracefully."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        processor = BatchProcessor(session=mock_session, config=mock_config)

        emails = [
            {
                "message_id": "<test@example.com>",
                "labels": None,  # None instead of list
                "body_text": "Test",
            }
        ]

        count = await processor._store_emails(emails)

        assert count == 1

    @pytest.mark.asyncio
    async def test_store_emails_handles_non_list_fields(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test _store_emails handles non-list fields gracefully."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        processor = BatchProcessor(session=mock_session, config=mock_config)

        emails = [
            {
                "message_id": "<test@example.com>",
                "labels": "not-a-list",  # String instead of list
                "to_emails": "also-not-a-list",
                "cc_emails": 123,  # Number instead of list
                "references": {"key": "value"},  # Dict instead of list
                "body_text": "Test",
            }
        ]

        count = await processor._store_emails(emails)

        assert count == 1

    @pytest.mark.asyncio
    async def test_store_emails_handles_empty_from_name(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test _store_emails handles empty from_name."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        processor = BatchProcessor(session=mock_session, config=mock_config)

        emails = [
            {
                "message_id": "<test@example.com>",
                "from_email": "sender@example.com",
                "from_name": None,
                "body_text": "Test",
            }
        ]

        count = await processor._store_emails(emails)

        assert count == 1

    @pytest.mark.asyncio
    async def test_store_emails_handles_body_html(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test _store_emails handles body_html field."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        processor = BatchProcessor(session=mock_session, config=mock_config)

        emails = [
            {
                "message_id": "<test@example.com>",
                "body_text": "Plain text",
                "body_html": "<p>HTML content</p>",
            }
        ]

        count = await processor._store_emails(emails)

        assert count == 1

    @pytest.mark.asyncio
    async def test_store_emails_handles_in_reply_to(
        self, mock_session: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Test _store_emails handles in_reply_to field."""
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        processor = BatchProcessor(session=mock_session, config=mock_config)

        emails = [
            {
                "message_id": "<reply@example.com>",
                "in_reply_to": "<original@example.com>",
                "body_text": "This is a reply",
            }
        ]

        count = await processor._store_emails(emails)

        assert count == 1
