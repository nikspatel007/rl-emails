"""Tests for pipeline status module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rl_emails.core.config import Config
from rl_emails.pipeline import status


class TestPipelineStatus:
    """Tests for PipelineStatus dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        s = status.PipelineStatus()
        assert s.emails == 0
        assert s.sent_emails == 0
        assert s.error is None

    def test_received_emails(self) -> None:
        """Test received_emails property."""
        s = status.PipelineStatus(emails=100, sent_emails=30)
        assert s.received_emails == 70

    def test_llm_remaining(self) -> None:
        """Test llm_remaining property."""
        s = status.PipelineStatus(needs_llm=50, llm_classification=20)
        assert s.llm_remaining == 30

    def test_llm_remaining_negative_clamped(self) -> None:
        """Test llm_remaining clamps to zero."""
        s = status.PipelineStatus(needs_llm=20, llm_classification=50)
        assert s.llm_remaining == 0

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        s = status.PipelineStatus(
            emails=100,
            sent_emails=30,
            threads=50,
            features=80,
            embeddings=70,
            ai_classification=90,
            llm_classification=20,
            needs_llm=40,
            clusters=60,
            priority=75,
            users=10,
        )
        d = s.to_dict()
        assert d["emails"] == 100
        assert d["sent_emails"] == 30
        assert d["received_emails"] == 70
        assert d["threads"] == 50
        assert d["features"] == 80
        assert d["embeddings"] == 70
        assert d["ai_classification"] == 90
        assert d["llm_classification"] == 20
        assert d["needs_llm"] == 40
        assert d["llm_remaining"] == 20
        assert d["clusters"] == 60
        assert d["priority"] == 75
        assert d["users"] == 10
        assert d["error"] is None


class TestSafeCount:
    """Tests for _safe_count function."""

    def test_returns_count(self) -> None:
        """Test returning count from query."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)
        result = status._safe_count(mock_cursor, "SELECT COUNT(*) FROM test")
        assert result == 42
        mock_cursor.execute.assert_called_once_with("SELECT COUNT(*) FROM test")

    def test_returns_zero_on_none(self) -> None:
        """Test returning zero when fetchone returns None."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        result = status._safe_count(mock_cursor, "SELECT COUNT(*) FROM test")
        assert result == 0

    def test_returns_zero_on_exception(self) -> None:
        """Test returning zero on exception."""
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Database error")
        result = status._safe_count(mock_cursor, "SELECT COUNT(*) FROM test")
        assert result == 0


class TestGetStatus:
    """Tests for get_status function."""

    @patch("rl_emails.pipeline.status.psycopg2")
    def test_returns_status(self, mock_psycopg2: MagicMock) -> None:
        """Test getting status from database."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Mock different counts for each query
        counts = iter([100, 30, 50, 80, 70, 90, 40, 20, 60, 75, 10])
        mock_cursor.fetchone.side_effect = lambda: (next(counts),)

        mock_psycopg2.connect.return_value = mock_conn

        config = Config(database_url="postgresql://localhost/test")
        result = status.get_status(config)

        assert result.emails == 100
        assert result.sent_emails == 30
        assert result.threads == 50
        assert result.features == 80
        assert result.embeddings == 70
        assert result.ai_classification == 90
        assert result.needs_llm == 40
        assert result.llm_classification == 20
        assert result.clusters == 60
        assert result.priority == 75
        assert result.users == 10
        assert result.error is None

        mock_conn.close.assert_called_once()

    @patch("rl_emails.pipeline.status.psycopg2")
    def test_returns_error_on_connection_failure(self, mock_psycopg2: MagicMock) -> None:
        """Test returning error when connection fails."""
        mock_psycopg2.connect.side_effect = Exception("Connection failed")

        config = Config(database_url="postgresql://localhost/test")
        result = status.get_status(config)

        assert result.error == "Connection failed"
        assert result.emails == 0


class TestCheckPostgres:
    """Tests for check_postgres function."""

    @patch("rl_emails.pipeline.status.psycopg2")
    def test_returns_true_on_success(self, mock_psycopg2: MagicMock) -> None:
        """Test returning True when connection succeeds."""
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn

        config = Config(database_url="postgresql://localhost/test")
        result = status.check_postgres(config)

        assert result is True
        mock_conn.close.assert_called_once()

    @patch("rl_emails.pipeline.status.psycopg2")
    def test_returns_false_on_failure(self, mock_psycopg2: MagicMock) -> None:
        """Test returning False when connection fails."""
        mock_psycopg2.connect.side_effect = Exception("Connection failed")

        config = Config(database_url="postgresql://localhost/test")
        result = status.check_postgres(config)

        assert result is False


class TestFormatStatus:
    """Tests for format_status function."""

    def test_formats_status(self) -> None:
        """Test formatting status."""
        s = status.PipelineStatus(
            emails=100,
            sent_emails=30,
            threads=50,
            features=80,
            embeddings=70,
            ai_classification=90,
            llm_classification=20,
            needs_llm=40,
            clusters=60,
            priority=75,
            users=10,
        )
        result = status.format_status(s)

        assert "PIPELINE STATUS" in result
        assert "100" in result  # emails
        assert "70" in result  # received
        assert "30" in result  # sent
        assert "50" in result  # threads
        assert "20" in result  # llm_remaining

    def test_formats_error(self) -> None:
        """Test formatting error status."""
        s = status.PipelineStatus(error="Test error")
        result = status.format_status(s)

        assert "Error getting status: Test error" in result

    def test_hides_llm_remaining_when_zero(self) -> None:
        """Test not showing LLM remaining when needs_llm is 0."""
        s = status.PipelineStatus(needs_llm=0, llm_classification=0)
        result = status.format_status(s)

        assert "Needs LLM" not in result
