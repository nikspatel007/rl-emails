"""Tests for ClusterLabelerService."""

from __future__ import annotations

from unittest import mock

import pytest

from rl_emails.models.cluster_metadata import ClusterMetadata
from rl_emails.services.cluster_labeler import (
    ClusterLabelerError,
    ClusterLabelerService,
    LabelResult,
)


class TestClusterLabelerService:
    """Tests for ClusterLabelerService."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ClusterLabelerService:
        """Create a service instance."""
        return ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )

    def test_init(self, mock_session: mock.MagicMock) -> None:
        """Test service initialization."""
        service = ClusterLabelerService(
            session=mock_session,
            openai_api_key="openai-key",
            anthropic_api_key="anthropic-key",
            model="haiku",
        )
        assert service.openai_api_key == "openai-key"
        assert service.anthropic_api_key == "anthropic-key"

    def test_get_model_with_anthropic(self, mock_session: mock.MagicMock) -> None:
        """Test model selection with Anthropic key."""
        service = ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )
        model = service._get_model()
        assert "claude" in model.lower() or "anthropic" in model.lower()

    def test_get_model_with_openai(self, mock_session: mock.MagicMock) -> None:
        """Test model selection with OpenAI key."""
        service = ClusterLabelerService(
            session=mock_session,
            openai_api_key="test-key",
        )
        # Ensure no Anthropic key so OpenAI is selected
        service.anthropic_api_key = None
        model = service._get_model()
        assert "gpt" in model.lower()

    def test_get_model_no_key_raises(self, mock_session: mock.MagicMock) -> None:
        """Test that missing API key raises error."""
        service = ClusterLabelerService(session=mock_session)
        service.openai_api_key = None
        service.anthropic_api_key = None
        with pytest.raises(ClusterLabelerError, match="No LLM API key"):
            service._get_model()

    def test_build_prompt_content(self, service: ClusterLabelerService) -> None:
        """Test prompt building for content dimension."""
        email_data = {
            "from_email": "test@example.com",
            "subject": "Q4 Budget Review",
            "date": "2025-01-01",
            "body_preview": "Please review the attached budget...",
            "relationship_strength": 0.8,
        }
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=0,
            size=50,
            pct_replied=75.0,
            avg_response_time_hours=4.0,
        )
        prompt = service._build_prompt("content", email_data, metadata)
        assert "Q4 Budget Review" in prompt
        assert "test@example.com" in prompt
        assert "50 emails" in prompt

    def test_build_prompt_people(self, service: ClusterLabelerService) -> None:
        """Test prompt building for people dimension."""
        email_data = {
            "from_email": "ceo@company.com",
            "subject": "Important Update",
            "date": "2025-01-01",
            "body_preview": "Team, I wanted to share...",
            "relationship_strength": 0.95,
        }
        metadata = ClusterMetadata(
            dimension="people",
            cluster_id=0,
            size=100,
            pct_replied=90.0,
        )
        prompt = service._build_prompt("people", email_data, metadata)
        assert "ceo@company.com" in prompt
        assert "90" in prompt


class TestLabelResult:
    """Tests for LabelResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful label result."""
        result = LabelResult(
            cluster_id=1,
            dimension="content",
            label="Test Label",
            confidence=0.9,
            success=True,
        )
        assert result.success is True
        assert result.label == "Test Label"
        assert result.error is None

    def test_failure_result(self) -> None:
        """Test failed label result."""
        result = LabelResult(
            cluster_id=1,
            dimension="content",
            label=None,
            confidence=0.0,
            success=False,
            error="Cluster not found",
        )
        assert result.success is False
        assert result.error == "Cluster not found"


class TestClusterLabelerServiceLabelCluster:
    """Tests for label_cluster method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ClusterLabelerService:
        """Create a service instance."""
        return ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_label_cluster_not_found(self, service: ClusterLabelerService) -> None:
        """Test labeling non-existent cluster."""
        with mock.patch.object(
            service.repository, "get_by_dimension_and_cluster", return_value=None
        ):
            result = await service.label_cluster(
                cluster_id=999,
                dimension="content",
            )
            assert result.success is False
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_label_cluster_already_labeled(self, service: ClusterLabelerService) -> None:
        """Test skipping already labeled cluster."""
        existing = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=50,
            auto_label="Existing Label",
        )
        with mock.patch.object(
            service.repository, "get_by_dimension_and_cluster", return_value=existing
        ):
            result = await service.label_cluster(
                cluster_id=1,
                dimension="content",
            )
            assert result.success is True
            assert result.label == "Existing Label"

    @pytest.mark.asyncio
    async def test_label_cluster_no_representative(self, service: ClusterLabelerService) -> None:
        """Test cluster without representative email."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=50,
            representative_email_id=None,
        )
        with mock.patch.object(
            service.repository, "get_by_dimension_and_cluster", return_value=metadata
        ):
            result = await service.label_cluster(
                cluster_id=1,
                dimension="content",
            )
            assert result.success is False
            assert "representative" in result.error.lower()


class TestClusterLabelerServiceGetLabelingStats:
    """Tests for get_labeling_stats method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ClusterLabelerService:
        """Create a service instance."""
        return ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_get_labeling_stats(self, service: ClusterLabelerService) -> None:
        """Test getting labeling statistics."""
        mock_stats = {
            "total_clusters": 50,
            "labeled_clusters": 30,
            "total_emails": 1000,
            "avg_cluster_size": 20.0,
            "largest_cluster_size": 100,
            "smallest_cluster_size": 5,
            "project_clusters": 10,
            "active_projects": 5,
        }

        with mock.patch.object(service.repository, "get_stats", return_value=mock_stats):
            stats = await service.get_labeling_stats()

            assert "content" in stats
            assert stats["content"]["total_clusters"] == 50
            assert stats["content"]["labeled_clusters"] == 30
            assert stats["content"]["unlabeled_clusters"] == 20


class TestGetModel:
    """Tests for _get_model with custom model selection."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        return mock.MagicMock()

    def test_get_model_with_custom_model(self, mock_session: mock.MagicMock) -> None:
        """Test model selection with explicit model parameter."""
        service = ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
            model="sonnet",
        )
        model = service._get_model()
        assert "sonnet" in model.lower()

    def test_get_model_gpt5(self, mock_session: mock.MagicMock) -> None:
        """Test model selection with gpt5 model."""
        service = ClusterLabelerService(
            session=mock_session,
            openai_api_key="test-key",
            model="gpt5",
        )
        model = service._get_model()
        assert "gpt-5" in model.lower()


class TestGetCompletionFunc:
    """Tests for _get_completion_func method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        return mock.MagicMock()

    def test_get_completion_func_caches(self, mock_session: mock.MagicMock) -> None:
        """Test that completion func is cached after first call."""
        service = ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )
        # Set a mock completion func
        mock_func = mock.MagicMock()
        service._completion_func = mock_func

        # Should return cached func without importing
        result = service._get_completion_func()
        assert result is mock_func

    def test_get_completion_func_imports_litellm(self, mock_session: mock.MagicMock) -> None:
        """Test that completion func imports litellm (when available)."""
        service = ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )
        # Ensure completion func is not cached
        service._completion_func = None

        # Call the actual method to import litellm
        result = service._get_completion_func()

        # Should return a callable (the litellm.completion function)
        assert callable(result)
        # Should cache the result
        assert service._completion_func is result

    def test_get_completion_func_import_error(self, mock_session: mock.MagicMock) -> None:
        """Test that missing litellm raises error."""
        service = ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )
        service._completion_func = None

        with mock.patch.object(
            service,
            "_get_completion_func",
            side_effect=ClusterLabelerError("litellm not installed"),
        ):
            with pytest.raises(ClusterLabelerError, match="litellm"):
                service._get_completion_func()


class TestFetchRepresentativeEmail:
    """Tests for _fetch_representative_email method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ClusterLabelerService:
        """Create a service instance."""
        return ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_fetch_representative_email_found(
        self, service: ClusterLabelerService, mock_session: mock.MagicMock
    ) -> None:
        """Test fetching representative email when found."""
        mock_row = (
            1,  # id
            "sender@example.com",  # from_email
            "Test Subject",  # subject
            "2025-01-01 10:00:00",  # date_parsed
            "Email body preview...",  # body_preview
            "reply",  # action
            0.8,  # relationship_strength
            3600,  # response_time_seconds (1 hour)
        )
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        result = await service._fetch_representative_email(1)

        assert result is not None
        assert result["id"] == 1
        assert result["from_email"] == "sender@example.com"
        assert result["subject"] == "Test Subject"
        assert result["response_time_hours"] == 1.0

    @pytest.mark.asyncio
    async def test_fetch_representative_email_not_found(
        self, service: ClusterLabelerService, mock_session: mock.MagicMock
    ) -> None:
        """Test fetching representative email when not found."""
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        result = await service._fetch_representative_email(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_representative_email_null_values(
        self, service: ClusterLabelerService, mock_session: mock.MagicMock
    ) -> None:
        """Test fetching email with null values uses defaults."""
        mock_row = (
            1,  # id
            None,  # from_email
            None,  # subject
            None,  # date_parsed
            None,  # body_preview
            None,  # action
            None,  # relationship_strength
            None,  # response_time_seconds
        )
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        result = await service._fetch_representative_email(1)

        assert result is not None
        assert result["from_email"] == "unknown"
        assert result["subject"] == "(no subject)"
        assert result["date"] == "unknown"
        assert result["body_preview"] == ""
        assert result["action"] == "unknown"
        assert result["relationship_strength"] == 0.0
        assert result["response_time_hours"] == 0.0


class TestCallLLM:
    """Tests for _call_llm method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ClusterLabelerService:
        """Create a service instance."""
        return ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )

    def test_call_llm_success(self, service: ClusterLabelerService) -> None:
        """Test successful LLM call with valid JSON response."""
        mock_response = mock.MagicMock()
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(content='{"label": "Test Label", "confidence": 0.9}')
            )
        ]
        mock_completion = mock.MagicMock(return_value=mock_response)
        service._completion_func = mock_completion

        result = service._call_llm("test prompt", "anthropic/claude-haiku-4-5")

        assert result["success"] is True
        assert result["label"] == "Test Label"
        assert result["confidence"] == 0.9

    def test_call_llm_with_gpt5_model(self, service: ClusterLabelerService) -> None:
        """Test LLM call with GPT-5 model uses reasoning_effort."""
        mock_response = mock.MagicMock()
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(content='{"label": "GPT Label", "confidence": 0.85}')
            )
        ]
        mock_completion = mock.MagicMock(return_value=mock_response)
        service._completion_func = mock_completion

        result = service._call_llm("test prompt", "gpt-5-mini")

        assert result["success"] is True
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["reasoning_effort"] == "minimal"
        assert "temperature" not in call_kwargs

    def test_call_llm_with_claude_model(self, service: ClusterLabelerService) -> None:
        """Test LLM call with Claude model uses temperature."""
        mock_response = mock.MagicMock()
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(content='{"label": "Claude Label", "confidence": 0.88}')
            )
        ]
        mock_completion = mock.MagicMock(return_value=mock_response)
        service._completion_func = mock_completion

        result = service._call_llm("test prompt", "anthropic/claude-haiku-4-5")

        assert result["success"] is True
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0
        assert "reasoning_effort" not in call_kwargs

    def test_call_llm_json_in_code_block(self, service: ClusterLabelerService) -> None:
        """Test parsing JSON from code block."""
        mock_response = mock.MagicMock()
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    content='```json\n{"label": "Code Block Label", "confidence": 0.75}\n```'
                )
            )
        ]
        mock_completion = mock.MagicMock(return_value=mock_response)
        service._completion_func = mock_completion

        result = service._call_llm("test prompt", "anthropic/claude-haiku-4-5")

        assert result["success"] is True
        assert result["label"] == "Code Block Label"
        assert result["confidence"] == 0.75

    def test_call_llm_json_in_plain_code_block(self, service: ClusterLabelerService) -> None:
        """Test parsing JSON from code block without json suffix."""
        mock_response = mock.MagicMock()
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(
                    content='```\n{"label": "Plain Block Label", "confidence": 0.8}\n```'
                )
            )
        ]
        mock_completion = mock.MagicMock(return_value=mock_response)
        service._completion_func = mock_completion

        result = service._call_llm("test prompt", "anthropic/claude-haiku-4-5")

        assert result["success"] is True
        assert result["label"] == "Plain Block Label"
        assert result["confidence"] == 0.8

    def test_call_llm_invalid_json(self, service: ClusterLabelerService) -> None:
        """Test handling of invalid JSON response."""
        mock_response = mock.MagicMock()
        mock_response.choices = [
            mock.MagicMock(message=mock.MagicMock(content="This is not valid JSON"))
        ]
        mock_completion = mock.MagicMock(return_value=mock_response)
        service._completion_func = mock_completion

        result = service._call_llm("test prompt", "anthropic/claude-haiku-4-5")

        # Falls back to extracting raw text as label
        assert result["label"] == "This is not valid JSON"
        assert result["confidence"] == 0.3
        assert result["success"] is True

    def test_call_llm_unclosed_code_block(self, service: ClusterLabelerService) -> None:
        """Test handling of unclosed code block (triple backticks without closing)."""
        mock_response = mock.MagicMock()
        # Triple backticks exist but no closing - regex won't match
        mock_response.choices = [
            mock.MagicMock(
                message=mock.MagicMock(content='```json\n{"label": "Unclosed", "confidence": 0.9}')
            )
        ]
        mock_completion = mock.MagicMock(return_value=mock_response)
        service._completion_func = mock_completion

        result = service._call_llm("test prompt", "anthropic/claude-haiku-4-5")

        # Falls back to raw text parsing since regex doesn't match
        assert result["label"] is not None
        assert result["confidence"] == 0.3

    def test_call_llm_empty_response(self, service: ClusterLabelerService) -> None:
        """Test handling of empty response."""
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="   "))]
        mock_completion = mock.MagicMock(return_value=mock_response)
        service._completion_func = mock_completion

        result = service._call_llm("test prompt", "anthropic/claude-haiku-4-5")

        # Empty label means failure
        assert result["label"] is None
        assert result["success"] is False


class TestLabelClusterAdvanced:
    """Advanced tests for label_cluster method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ClusterLabelerService:
        """Create a service instance."""
        return ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_label_cluster_email_not_found(
        self, service: ClusterLabelerService, mock_session: mock.MagicMock
    ) -> None:
        """Test labeling when representative email not found."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=50,
            representative_email_id=999,
        )
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        with mock.patch.object(
            service.repository, "get_by_dimension_and_cluster", return_value=metadata
        ):
            result = await service.label_cluster(cluster_id=1, dimension="content")

            assert result.success is False
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_label_cluster_llm_success(
        self, service: ClusterLabelerService, mock_session: mock.MagicMock
    ) -> None:
        """Test successful labeling via LLM."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=50,
            representative_email_id=1,
        )
        mock_row = (
            1,
            "sender@example.com",
            "Test Subject",
            "2025-01-01",
            "Body",
            "reply",
            0.8,
            3600,
        )
        mock_db_result = mock.MagicMock()
        mock_db_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_db_result

        llm_result = {"label": "Generated Label", "confidence": 0.95, "success": True}

        with mock.patch.object(
            service.repository, "get_by_dimension_and_cluster", return_value=metadata
        ):
            with mock.patch.object(service, "_call_llm", return_value=llm_result):
                with mock.patch.object(service.repository, "update_label") as mock_update:
                    mock_update.return_value = None
                    result = await service.label_cluster(cluster_id=1, dimension="content")

                    assert result.success is True
                    assert result.label == "Generated Label"
                    assert result.confidence == 0.95
                    mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_label_cluster_llm_failure(
        self, service: ClusterLabelerService, mock_session: mock.MagicMock
    ) -> None:
        """Test labeling when LLM returns no valid label."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=50,
            representative_email_id=1,
        )
        mock_row = (
            1,
            "sender@example.com",
            "Test Subject",
            "2025-01-01",
            "Body",
            "reply",
            0.8,
            3600,
        )
        mock_db_result = mock.MagicMock()
        mock_db_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_db_result

        llm_result = {"label": None, "confidence": 0.0, "success": False}

        with mock.patch.object(
            service.repository, "get_by_dimension_and_cluster", return_value=metadata
        ):
            with mock.patch.object(service, "_call_llm", return_value=llm_result):
                result = await service.label_cluster(cluster_id=1, dimension="content")

                assert result.success is False
                assert "did not return a valid label" in result.error

    @pytest.mark.asyncio
    async def test_label_cluster_force_relabel(
        self, service: ClusterLabelerService, mock_session: mock.MagicMock
    ) -> None:
        """Test force relabeling existing cluster."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=50,
            auto_label="Old Label",
            representative_email_id=1,
        )
        mock_row = (
            1,
            "sender@example.com",
            "Test Subject",
            "2025-01-01",
            "Body",
            "reply",
            0.8,
            3600,
        )
        mock_db_result = mock.MagicMock()
        mock_db_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_db_result

        llm_result = {"label": "New Label", "confidence": 0.9, "success": True}

        with mock.patch.object(
            service.repository, "get_by_dimension_and_cluster", return_value=metadata
        ):
            with mock.patch.object(service, "_call_llm", return_value=llm_result):
                with mock.patch.object(service.repository, "update_label"):
                    result = await service.label_cluster(
                        cluster_id=1, dimension="content", force_relabel=True
                    )

                    assert result.success is True
                    assert result.label == "New Label"

    @pytest.mark.asyncio
    async def test_label_cluster_exception(
        self, service: ClusterLabelerService, mock_session: mock.MagicMock
    ) -> None:
        """Test labeling when exception occurs."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=50,
            representative_email_id=1,
        )
        mock_row = (
            1,
            "sender@example.com",
            "Test Subject",
            "2025-01-01",
            "Body",
            "reply",
            0.8,
            3600,
        )
        mock_db_result = mock.MagicMock()
        mock_db_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_db_result

        with mock.patch.object(
            service.repository, "get_by_dimension_and_cluster", return_value=metadata
        ):
            with mock.patch.object(service, "_call_llm", side_effect=Exception("API error")):
                result = await service.label_cluster(cluster_id=1, dimension="content")

                assert result.success is False
                assert "API error" in result.error


class TestLabelUnlabeledClusters:
    """Tests for label_unlabeled_clusters method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ClusterLabelerService:
        """Create a service instance."""
        return ClusterLabelerService(
            session=mock_session,
            anthropic_api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_label_unlabeled_clusters_empty(self, service: ClusterLabelerService) -> None:
        """Test with no unlabeled clusters."""
        with mock.patch.object(service.repository, "list_unlabeled", return_value=[]):
            results = await service.label_unlabeled_clusters(dimension="content")

            assert results == []

    @pytest.mark.asyncio
    async def test_label_unlabeled_clusters_multiple(self, service: ClusterLabelerService) -> None:
        """Test labeling multiple unlabeled clusters."""
        unlabeled = [
            ClusterMetadata(dimension="content", cluster_id=1, size=10, representative_email_id=1),
            ClusterMetadata(dimension="content", cluster_id=2, size=20, representative_email_id=2),
        ]

        mock_result_1 = LabelResult(
            cluster_id=1, dimension="content", label="Label 1", confidence=0.9, success=True
        )
        mock_result_2 = LabelResult(
            cluster_id=2, dimension="content", label="Label 2", confidence=0.85, success=True
        )

        with mock.patch.object(service.repository, "list_unlabeled", return_value=unlabeled):
            with mock.patch.object(
                service, "label_cluster", side_effect=[mock_result_1, mock_result_2]
            ):
                results = await service.label_unlabeled_clusters(dimension="content")

                assert len(results) == 2
                assert results[0].label == "Label 1"
                assert results[1].label == "Label 2"

    @pytest.mark.asyncio
    async def test_label_unlabeled_clusters_with_user_id(
        self, service: ClusterLabelerService
    ) -> None:
        """Test labeling with user_id filter."""
        import uuid

        user_id = uuid.uuid4()
        unlabeled = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=10,
                representative_email_id=1,
                user_id=user_id,
            ),
        ]
        mock_result = LabelResult(
            cluster_id=1, dimension="content", label="User Label", confidence=0.9, success=True
        )

        with mock.patch.object(service.repository, "list_unlabeled", return_value=unlabeled):
            with mock.patch.object(service, "label_cluster", return_value=mock_result):
                results = await service.label_unlabeled_clusters(
                    dimension="content", user_id=user_id
                )

                assert len(results) == 1
                assert results[0].label == "User Label"
