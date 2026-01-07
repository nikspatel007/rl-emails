"""Tests for Stage 6: Compute email embeddings."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_06_compute_embeddings
from rl_emails.pipeline.stages.base import StageResult


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_counts_tokens(self) -> None:
        """Test token counting."""
        text = "Hello world"  # 11 chars -> ~3 tokens
        result = stage_06_compute_embeddings.count_tokens(text)
        assert result == 3

    def test_empty_string(self) -> None:
        """Test empty string."""
        assert stage_06_compute_embeddings.count_tokens("") == 0


class TestTruncateToTokens:
    """Tests for truncate_to_tokens function."""

    def test_no_truncation_needed(self) -> None:
        """Test when text fits within limit."""
        text = "Hello"
        result = stage_06_compute_embeddings.truncate_to_tokens(text, 100)
        assert result == "Hello"

    def test_truncates_long_text(self) -> None:
        """Test truncation of long text."""
        text = "a" * 1000
        result = stage_06_compute_embeddings.truncate_to_tokens(text, 10)  # 30 chars
        assert len(result) == 33  # 30 + "..."
        assert result.endswith("...")

    def test_empty_text(self) -> None:
        """Test empty text."""
        assert stage_06_compute_embeddings.truncate_to_tokens("", 100) == ""


class TestStripHtml:
    """Tests for strip_html function."""

    def test_strips_html_tags(self) -> None:
        """Test HTML tag removal."""
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        result = stage_06_compute_embeddings.strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_removes_script_style(self) -> None:
        """Test script and style removal."""
        html = "<html><script>alert('x')</script><style>.foo{}</style><p>Content</p></html>"
        result = stage_06_compute_embeddings.strip_html(html)
        assert "Content" in result
        assert "alert" not in result
        assert ".foo" not in result

    def test_empty_input(self) -> None:
        """Test empty input."""
        assert stage_06_compute_embeddings.strip_html("") == ""

    def test_returns_original_on_error(self) -> None:
        """Test fallback on parsing error."""
        # BeautifulSoup is very tolerant, so just test None-like input
        assert stage_06_compute_embeddings.strip_html(None) == ""  # type: ignore[arg-type]

    @patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.BeautifulSoup")
    def test_returns_html_on_exception(self, mock_soup: MagicMock) -> None:
        """Test returns original HTML on exception."""
        mock_soup.side_effect = Exception("Parse error")
        result = stage_06_compute_embeddings.strip_html("<p>Hello</p>")
        assert result == "<p>Hello</p>"


class TestStripTemplateSyntax:
    """Tests for strip_template_syntax function."""

    def test_removes_liquid_tags(self) -> None:
        """Test Liquid/Jinja tag removal."""
        text = "Hello {% if true %}World{% endif %}"
        result = stage_06_compute_embeddings.strip_template_syntax(text)
        assert "{%" not in result
        assert "Hello" in result

    def test_removes_variable_tags(self) -> None:
        """Test variable tag removal."""
        text = "Hello {{ name }}"
        result = stage_06_compute_embeddings.strip_template_syntax(text)
        assert "{{" not in result

    def test_removes_comments(self) -> None:
        """Test comment removal."""
        text = "Hello {# comment #} World"
        result = stage_06_compute_embeddings.strip_template_syntax(text)
        assert "{#" not in result
        assert "Hello" in result
        assert "World" in result

    def test_empty_input(self) -> None:
        """Test empty input."""
        assert stage_06_compute_embeddings.strip_template_syntax("") == ""


class TestStripQuotedReplies:
    """Tests for strip_quoted_replies function."""

    def test_removes_quoted_lines(self) -> None:
        """Test quoted line removal."""
        text = "My reply\n> Previous message\n> More quoted"
        result = stage_06_compute_embeddings.strip_quoted_replies(text)
        assert "My reply" in result
        assert "Previous message" not in result

    def test_removes_on_wrote_lines(self) -> None:
        """Test 'On ... wrote:' removal."""
        text = "My reply\nOn Monday, John wrote:\nQuoted text"
        result = stage_06_compute_embeddings.strip_quoted_replies(text)
        assert "My reply" in result

    def test_stops_at_signature(self) -> None:
        """Test signature detection."""
        text = "My message\n--\nJohn Doe"
        result = stage_06_compute_embeddings.strip_quoted_replies(text)
        assert "My message" in result
        assert "John Doe" not in result

    def test_stops_at_sent_from_iphone(self) -> None:
        """Test Sent from iPhone detection."""
        text = "Quick reply\nSent from my iPhone"
        result = stage_06_compute_embeddings.strip_quoted_replies(text)
        assert "Quick reply" in result
        assert "Sent from my iPhone" not in result

    def test_removes_from_header(self) -> None:
        """Test From: header removal."""
        text = "My reply\nFrom: John Doe\nTo: Jane"
        result = stage_06_compute_embeddings.strip_quoted_replies(text)
        assert "My reply" in result
        assert "From:" not in result

    def test_stops_at_signature_with_space(self) -> None:
        """Test signature marker with trailing space."""
        text = "Message here\n-- \nSignature"
        result = stage_06_compute_embeddings.strip_quoted_replies(text)
        assert "Message here" in result
        assert "Signature" not in result

    def test_empty_input(self) -> None:
        """Test empty input."""
        assert stage_06_compute_embeddings.strip_quoted_replies("") == ""


class TestBuildEmbeddingText:
    """Tests for build_embedding_text function."""

    def test_person_high_priority(self) -> None:
        """Test person email with high relationship."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="Meeting",
            body="Let's meet tomorrow",
            is_service=False,
            service_importance=0.5,
            relationship_strength=0.8,
        )
        assert "[TYPE: PERSON]" in result
        assert "[PRIORITY: HIGH]" in result
        assert "[SUBJECT] Meeting" in result
        assert "[BODY]" in result

    def test_person_medium_priority(self) -> None:
        """Test person email with medium relationship."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="Hello",
            body="Hi there",
            is_service=False,
            service_importance=0.5,
            relationship_strength=0.3,
        )
        assert "[TYPE: PERSON]" in result
        assert "[PRIORITY: MEDIUM]" in result

    def test_person_low_priority(self) -> None:
        """Test person email with low relationship."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="Hello",
            body="Hi there",
            is_service=False,
            service_importance=0.5,
            relationship_strength=0.1,
        )
        assert "[TYPE: PERSON]" in result
        assert "[PRIORITY: LOW]" in result

    def test_service_high_importance(self) -> None:
        """Test service email with high importance."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="Order Shipped",
            body="Your order has shipped",
            is_service=True,
            service_importance=0.8,
            relationship_strength=0.0,
        )
        assert "[TYPE: SERVICE]" in result
        assert "[PRIORITY: HIGH]" in result

    def test_service_medium_importance(self) -> None:
        """Test service email with medium importance."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="Newsletter",
            body="This week's news",
            is_service=True,
            service_importance=0.5,
            relationship_strength=0.0,
        )
        assert "[TYPE: SERVICE]" in result
        assert "[PRIORITY: MEDIUM]" in result

    def test_service_low_importance(self) -> None:
        """Test service email with low importance."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="Sale",
            body="Big sale today",
            is_service=True,
            service_importance=0.2,
            relationship_strength=0.0,
        )
        assert "[TYPE: SERVICE]" in result
        assert "[PRIORITY: LOW]" in result

    def test_no_subject(self) -> None:
        """Test email without subject."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="",
            body="Message body",
            is_service=False,
            service_importance=0.5,
            relationship_strength=0.5,
        )
        assert "[SUBJECT]" not in result
        assert "[BODY]" in result

    def test_no_body(self) -> None:
        """Test email without body."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="Subject only",
            body="",
            is_service=False,
            service_importance=0.5,
            relationship_strength=0.5,
        )
        assert "[SUBJECT] Subject only" in result
        assert "[BODY]" not in result

    def test_small_body_budget(self) -> None:
        """Test when body token budget is too small."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="x" * 30000,  # Very long subject
            body="Body text",
            is_service=False,
            service_importance=0.5,
            relationship_strength=0.5,
        )
        # Body might be skipped due to budget
        assert "[TYPE: PERSON]" in result

    def test_body_empty_after_cleaning(self) -> None:
        """Test when body becomes empty after HTML/template stripping."""
        result = stage_06_compute_embeddings.build_embedding_text(
            subject="Test Subject",
            body="<script>alert('x')</script><style>.foo{}</style>",
            is_service=False,
            service_importance=0.5,
            relationship_strength=0.5,
        )
        # Body should be skipped because it's only script/style
        assert "[SUBJECT] Test Subject" in result
        assert "[BODY]" not in result


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_computes_hash(self) -> None:
        """Test hash computation."""
        result = stage_06_compute_embeddings.compute_content_hash("Hello world")
        assert len(result) == 16
        assert result.isalnum()

    def test_deterministic(self) -> None:
        """Test hash is deterministic."""
        hash1 = stage_06_compute_embeddings.compute_content_hash("Test")
        hash2 = stage_06_compute_embeddings.compute_content_hash("Test")
        assert hash1 == hash2


class TestCreateTables:
    """Tests for create_tables function."""

    def test_creates_tables(self) -> None:
        """Test table creation."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        stage_06_compute_embeddings.create_tables(conn)

        assert mock_cursor.execute.call_count == 2
        conn.commit.assert_called_once()


class TestGetUnprocessedEmails:
    """Tests for get_unprocessed_emails function."""

    def test_returns_emails(self) -> None:
        """Test getting unprocessed emails."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, "Subject", "Body", False, 0.5, 0.5),
        ]

        result = stage_06_compute_embeddings.get_unprocessed_emails(conn, limit=10)

        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["subject"] == "Subject"

    def test_empty_results(self) -> None:
        """Test with no results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        result = stage_06_compute_embeddings.get_unprocessed_emails(conn)

        assert result == []


class TestGetEmbeddingCounts:
    """Tests for get_embedding_counts function."""

    def test_returns_counts(self) -> None:
        """Test getting embedding counts."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100,), (50,)]

        total, embedded = stage_06_compute_embeddings.get_embedding_counts(conn)

        assert total == 100
        assert embedded == 50

    def test_handles_none(self) -> None:
        """Test handling None results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [None, None]

        total, embedded = stage_06_compute_embeddings.get_embedding_counts(conn)

        assert total == 0
        assert embedded == 0


class TestSaveEmbeddingsToDb:
    """Tests for save_embeddings_to_db function."""

    @patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.execute_values")
    def test_saves_embeddings(self, mock_execute_values: MagicMock) -> None:
        """Test saving embeddings."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        data: list[tuple[int, list[float], int, str]] = [
            (1, [0.1, 0.2, 0.3], 100, "abc123"),
        ]

        result = stage_06_compute_embeddings.save_embeddings_to_db(conn, data)

        assert result == 1
        mock_execute_values.assert_called_once()
        conn.commit.assert_called_once()

    def test_empty_data(self) -> None:
        """Test with empty data."""
        conn = MagicMock()
        result = stage_06_compute_embeddings.save_embeddings_to_db(conn, [])
        assert result == 0


class TestGenerateSingleEmbedding:
    """Tests for generate_single_embedding function."""

    @pytest.mark.asyncio
    async def test_generates_embedding(self) -> None:
        """Test embedding generation."""
        import asyncio

        semaphore = asyncio.Semaphore(1)
        mock_embedding_func = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_embedding_func.return_value = mock_response

        result = await stage_06_compute_embeddings.generate_single_embedding(
            "Test text", semaphore, mock_embedding_func
        )

        assert result == [0.1, 0.2, 0.3]
        mock_embedding_func.assert_called_once()


class TestProcessEmailsParallel:
    """Tests for process_emails_parallel function."""

    @pytest.mark.asyncio
    async def test_processes_emails(self) -> None:
        """Test parallel email processing."""
        mock_embedding_func = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 1536}]
        mock_embedding_func.return_value = mock_response

        emails: list[dict[str, Any]] = [
            {
                "id": 1,
                "subject": "Test",
                "body": "Body text here",
                "is_service": False,
                "service_importance": 0.5,
                "relationship_strength": 0.5,
            }
        ]

        result = await stage_06_compute_embeddings.process_emails_parallel(
            emails, workers=1, embedding_func=mock_embedding_func
        )

        assert len(result) == 1
        assert result[0][0] == 1  # email_id

    @pytest.mark.asyncio
    @patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.build_embedding_text")
    async def test_skips_short_text(self, mock_build: MagicMock) -> None:
        """Test skipping emails with short text."""
        mock_build.return_value = "short"  # Less than 10 chars
        mock_embedding_func = MagicMock()

        emails: list[dict[str, Any]] = [
            {
                "id": 1,
                "subject": "",
                "body": "",
                "is_service": False,
                "service_importance": 0.5,
                "relationship_strength": 0.5,
            }
        ]

        result = await stage_06_compute_embeddings.process_emails_parallel(
            emails, workers=1, embedding_func=mock_embedding_func
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_handles_errors(self) -> None:
        """Test error handling in parallel processing."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.side_effect = Exception("API error")

        emails: list[dict[str, Any]] = [
            {
                "id": 1,
                "subject": "Test",
                "body": "Body text here",
                "is_service": False,
                "service_importance": 0.5,
                "relationship_strength": 0.5,
            }
        ]

        result = await stage_06_compute_embeddings.process_emails_parallel(
            emails, workers=1, embedding_func=mock_embedding_func
        )

        assert len(result) == 0  # Error results in None, filtered out


class TestComputeEmbeddingsAsync:
    """Tests for compute_embeddings_async function."""

    @pytest.mark.asyncio
    async def test_processes_all(self) -> None:
        """Test processing all emails."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        # get_embedding_counts
        mock_cursor.fetchone.side_effect = [(10,), (5,)]  # 10 total, 5 embedded

        # get_unprocessed_emails
        mock_cursor.fetchall.side_effect = [
            [(1, "Test", "Body", False, 0.5, 0.5)],  # First batch
            [],  # No more emails
        ]

        mock_embedding_func = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 1536}]
        mock_embedding_func.return_value = mock_response

        with patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.execute_values"):
            stats = await stage_06_compute_embeddings.compute_embeddings_async(
                conn, mock_embedding_func, workers=1, batch_size=10
            )

        assert stats["total_emails"] == 10
        assert stats["already_embedded"] == 5
        assert stats["processed"] == 1

    @pytest.mark.asyncio
    async def test_all_embedded(self) -> None:
        """Test when all emails are already embedded."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100,), (100,)]  # All embedded

        mock_embedding_func = MagicMock()

        stats = await stage_06_compute_embeddings.compute_embeddings_async(
            conn, mock_embedding_func
        )

        assert stats["processed"] == 0

    @pytest.mark.asyncio
    async def test_respects_limit(self) -> None:
        """Test limit parameter."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        # 100 total, 0 embedded, but limit to 10
        mock_cursor.fetchone.side_effect = [(100,), (0,)]
        mock_cursor.fetchall.side_effect = [
            [(i, "Test", "Body text", False, 0.5, 0.5) for i in range(10)],
            [],  # No more
        ]

        mock_embedding_func = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1] * 1536}]
        mock_embedding_func.return_value = mock_response

        with patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.execute_values"):
            stats = await stage_06_compute_embeddings.compute_embeddings_async(
                conn, mock_embedding_func, workers=1, batch_size=100, limit=10
            )

        assert stats["processed"] == 10


class TestRun:
    """Tests for run function."""

    def test_run_without_api_key(self) -> None:
        """Test run fails without API key."""
        config = Config(database_url="postgresql://test")

        result = stage_06_compute_embeddings.run(config)

        assert result.success is False
        assert "OPENAI_API_KEY" in result.message

    @patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.psycopg2.connect")
    @patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.asyncio.run")
    def test_run_success(self, mock_asyncio_run: MagicMock, mock_connect: MagicMock) -> None:
        """Test successful run."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        mock_asyncio_run.return_value = {
            "total_emails": 100,
            "already_embedded": 50,
            "processed": 50,
            "batches": 1,
        }

        config = Config(database_url="postgresql://test", openai_api_key="sk-test")
        result = stage_06_compute_embeddings.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 50
        mock_conn.close.assert_called_once()

    def test_run_without_litellm(self) -> None:
        """Test run fails without litellm package."""
        config = Config(database_url="postgresql://test", openai_api_key="sk-test")

        with patch.dict("sys.modules", {"litellm": None}):
            with patch("rl_emails.pipeline.stages.stage_06_compute_embeddings.psycopg2.connect"):
                # Simulate import error
                with patch("builtins.__import__", side_effect=ImportError("No module")):
                    result = stage_06_compute_embeddings.run(config)

        assert result.success is False
        assert "litellm" in result.message
