"""Tests for Stage 11: LLM classification."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_11_llm_classification
from rl_emails.pipeline.stages.base import StageResult


class TestCreateTables:
    """Tests for create_tables function."""

    def test_creates_tables(self) -> None:
        """Test table creation."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        stage_11_llm_classification.create_tables(conn)

        assert mock_cursor.execute.call_count == 2
        conn.commit.assert_called_once()


class TestGetModel:
    """Tests for get_model function."""

    def test_explicit_model(self) -> None:
        """Test explicit model selection."""
        result = stage_11_llm_classification.get_model("haiku", None, None)
        assert result == "anthropic/claude-haiku-4-5"

    def test_openai_default(self) -> None:
        """Test OpenAI key defaults to gpt5."""
        result = stage_11_llm_classification.get_model(None, "sk-test", None)
        assert result == "gpt-5-mini"

    def test_anthropic_default(self) -> None:
        """Test Anthropic key defaults to haiku."""
        result = stage_11_llm_classification.get_model(None, None, "ant-test")
        assert result == "anthropic/claude-haiku-4-5"

    def test_no_key(self) -> None:
        """Test no key returns None."""
        result = stage_11_llm_classification.get_model(None, None, None)
        assert result is None

    def test_invalid_model(self) -> None:
        """Test invalid model name falls back to keys."""
        result = stage_11_llm_classification.get_model("invalid", "sk-test", None)
        assert result == "gpt-5-mini"


class TestGetStatus:
    """Tests for get_status function."""

    def test_returns_status(self) -> None:
        """Test getting status."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100,), (50,)]

        result = stage_11_llm_classification.get_status(conn)

        assert result["total"] == 100
        assert result["processed"] == 50
        assert result["remaining"] == 50

    def test_handles_none(self) -> None:
        """Test handling None results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [None, None]

        result = stage_11_llm_classification.get_status(conn)

        assert result["total"] == 0
        assert result["processed"] == 0


class TestGetEmailsToProcess:
    """Tests for get_emails_to_process function."""

    def test_returns_emails(self) -> None:
        """Test getting emails to process."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (
                1,
                "test@example.com",
                "Subject",
                datetime.now(),
                "Body",
                0.5,
                0.3,
                False,
                "high",
                "PENDING",
                100,
                False,
                0.7,
                5,
            ),
        ]

        result = stage_11_llm_classification.get_emails_to_process(conn, limit=10)

        assert len(result) == 1


class TestBuildPrompt:
    """Tests for build_prompt function."""

    def test_builds_prompt_strong_relationship(self) -> None:
        """Test prompt building with strong relationship."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.8,
            0.7,
            False,
            "high",
            "PENDING",
            100,
            True,
            0.9,
            1,
        )

        result = stage_11_llm_classification.build_prompt(email_data)

        assert "STRONG" in result
        assert "Subject" in result
        assert "Yes (high engagement history)" in result

    def test_builds_prompt_moderate_relationship(self) -> None:
        """Test prompt building with moderate relationship."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            None,
            "Body text",
            0.4,
            0.3,
            True,
            "medium",
            "PENDING",
            100,
            False,
            0.5,
            50,
        )

        result = stage_11_llm_classification.build_prompt(email_data)

        assert "MODERATE" in result
        assert "unknown" in result  # No date
        assert "Yes" in result  # is_service

    def test_builds_prompt_weak_relationship(self) -> None:
        """Test prompt building with weak relationship."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.2,
            0.1,
            False,
            "low",
            "PENDING",
            100,
            False,
            0.3,
            100,
        )

        result = stage_11_llm_classification.build_prompt(email_data)

        assert "WEAK" in result

    def test_builds_prompt_minimal_relationship(self) -> None:
        """Test prompt building with minimal relationship."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.05,
            0.0,
            False,
            "low",
            "PENDING",
            100,
            False,
            0.1,
            200,
        )

        result = stage_11_llm_classification.build_prompt(email_data)

        assert "MINIMAL" in result

    def test_builds_prompt_html_body(self) -> None:
        """Test prompt building with HTML body."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "<html><body>Clean text</body></html>",
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            10,
        )

        result = stage_11_llm_classification.build_prompt(email_data)

        assert "Clean text" in result
        assert "<html>" not in result

    def test_builds_prompt_empty_body(self) -> None:
        """Test prompt building with empty body."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            None,
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            10,
        )

        result = stage_11_llm_classification.build_prompt(email_data)

        assert "(empty body)" in result

    def test_builds_prompt_high_rank(self) -> None:
        """Test prompt building with high rank (N/A)."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body",
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            9999,
        )

        result = stage_11_llm_classification.build_prompt(email_data)

        assert "N/A" in result


class TestClassifyEmail:
    """Tests for classify_email function."""

    def test_classifies_email_gpt5(self) -> None:
        """Test email classification with GPT-5."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            10,
        )

        mock_completion = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action_type": "reply", "urgency": "today"}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 120
        mock_completion.return_value = mock_response

        result = stage_11_llm_classification.classify_email(
            email_data, "gpt-5-mini", mock_completion
        )

        assert result["email_id"] == 1
        assert result["error"] is None
        assert result["result"]["action_type"] == "reply"
        # Check reasoning_effort was used
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["reasoning_effort"] == "minimal"

    def test_classifies_email_anthropic(self) -> None:
        """Test email classification with Anthropic."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            10,
        )

        mock_completion = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action_type": "fyi"}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 120
        mock_completion.return_value = mock_response

        result = stage_11_llm_classification.classify_email(
            email_data, "anthropic/claude-haiku-4-5", mock_completion
        )

        assert result["error"] is None
        # Check temperature was used
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0

    def test_parses_markdown_json(self) -> None:
        """Test parsing JSON from markdown code block."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            10,
        )

        mock_completion = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"action_type": "task"}\n```'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 120
        mock_completion.return_value = mock_response

        result = stage_11_llm_classification.classify_email(
            email_data, "gpt-5-mini", mock_completion
        )

        assert result["result"]["action_type"] == "task"

    def test_handles_malformed_markdown(self) -> None:
        """Test handling malformed markdown code block."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            10,
        )

        mock_completion = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Has ``` but no closing ``` so regex won't match
        mock_response.choices[0].message.content = '```json\n{"action_type": "task"}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 120
        mock_completion.return_value = mock_response

        result = stage_11_llm_classification.classify_email(
            email_data, "gpt-5-mini", mock_completion
        )

        # Should fall back to parsing the original text, which will fail
        assert result["result"]["parse_error"] is True

    def test_handles_invalid_json(self) -> None:
        """Test handling invalid JSON response."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            10,
        )

        mock_completion = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 120
        mock_completion.return_value = mock_response

        result = stage_11_llm_classification.classify_email(
            email_data, "gpt-5-mini", mock_completion
        )

        assert result["result"]["parse_error"] is True

    def test_handles_api_error(self) -> None:
        """Test handling API error."""
        email_data = (
            1,
            "test@example.com",
            "Subject",
            datetime.now(),
            "Body text",
            0.5,
            0.3,
            False,
            "high",
            "PENDING",
            100,
            False,
            0.5,
            10,
        )

        mock_completion = MagicMock()
        mock_completion.side_effect = Exception("API Error")

        result = stage_11_llm_classification.classify_email(
            email_data, "gpt-5-mini", mock_completion
        )

        assert result["error"] == "API Error"
        assert result["result"] is None


class TestSaveResult:
    """Tests for save_result function."""

    def test_saves_result(self) -> None:
        """Test saving a result."""
        cur = MagicMock()
        result: dict[str, Any] = {
            "email_id": 1,
            "thread_id": 100,
            "raw_prompt": "prompt",
            "raw_response": "response",
            "result": {
                "action_type": "reply",
                "urgency": "today",
                "ai_can_handle": "fully",
                "next_step": "quick_action",
                "suggested_action": "DRAFT_REPLY",
                "one_liner": "Summary",
            },
            "tokens": {"prompt": 100, "completion": 20, "total": 120},
            "error": None,
        }

        saved = stage_11_llm_classification.save_result(cur, result, "gpt-5-mini")

        assert saved is True
        cur.execute.assert_called_once()

    def test_skips_error_result(self) -> None:
        """Test skipping result with error."""
        cur = MagicMock()
        result: dict[str, Any] = {
            "email_id": 1,
            "error": "API Error",
            "result": None,
        }

        saved = stage_11_llm_classification.save_result(cur, result, "gpt-5-mini")

        assert saved is False
        cur.execute.assert_not_called()

    def test_handles_db_error(self) -> None:
        """Test handling database error."""
        cur = MagicMock()
        cur.execute.side_effect = Exception("DB Error")
        result: dict[str, Any] = {
            "email_id": 1,
            "thread_id": 100,
            "raw_prompt": "prompt",
            "raw_response": "response",
            "result": {"action_type": "reply"},
            "tokens": {"prompt": 100, "completion": 20, "total": 120},
            "error": None,
        }

        saved = stage_11_llm_classification.save_result(cur, result, "gpt-5-mini")

        assert saved is False


class TestProcessBatch:
    """Tests for process_batch function."""

    def test_processes_batch(self) -> None:
        """Test processing a batch of emails."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value = mock_cursor

        emails = [
            (
                1,
                "test@example.com",
                "Subject",
                datetime.now(),
                "Body",
                0.5,
                0.3,
                False,
                "high",
                "PENDING",
                100,
                False,
                0.5,
                10,
            ),
        ]

        mock_completion = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action_type": "reply"}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 120
        mock_completion.return_value = mock_response

        result = stage_11_llm_classification.process_batch(
            emails, workers=1, model="gpt-5-mini", conn=conn, completion_func=mock_completion
        )

        assert len(result["results"]) == 1
        assert result["saved"] == 1
        conn.commit.assert_called_once()

    def test_handles_error_in_batch(self) -> None:
        """Test processing batch with error."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value = mock_cursor

        emails = [
            (
                1,
                "test@example.com",
                "Subject",
                datetime.now(),
                "Body",
                0.5,
                0.3,
                False,
                "high",
                "PENDING",
                100,
                False,
                0.5,
                10,
            ),
        ]

        mock_completion = MagicMock()
        mock_completion.side_effect = Exception("API Error")

        result = stage_11_llm_classification.process_batch(
            emails, workers=1, model="gpt-5-mini", conn=conn, completion_func=mock_completion
        )

        assert len(result["results"]) == 1
        assert result["errors"] == 1
        assert result["saved"] == 0

    @patch("rl_emails.pipeline.stages.stage_11_llm_classification.save_result")
    def test_handles_save_failure(self, mock_save: MagicMock) -> None:
        """Test processing batch when save fails."""
        mock_save.return_value = False

        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value = mock_cursor

        emails = [
            (
                1,
                "test@example.com",
                "Subject",
                datetime.now(),
                "Body",
                0.5,
                0.3,
                False,
                "high",
                "PENDING",
                100,
                False,
                0.5,
                10,
            ),
        ]

        mock_completion = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action_type": "reply"}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 120
        mock_completion.return_value = mock_response

        result = stage_11_llm_classification.process_batch(
            emails, workers=1, model="gpt-5-mini", conn=conn, completion_func=mock_completion
        )

        assert len(result["results"]) == 1
        assert result["saved"] == 0  # Save failed


class TestRun:
    """Tests for run function."""

    def test_run_without_api_key(self) -> None:
        """Test run fails without API key."""
        config = Config(database_url="postgresql://test")

        result = stage_11_llm_classification.run(config)

        assert result.success is False
        assert "API key" in result.message

    @patch("rl_emails.pipeline.stages.stage_11_llm_classification.psycopg2.connect")
    def test_run_all_classified(self, mock_connect: MagicMock) -> None:
        """Test run when all emails already classified."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100,), (100,)]  # All processed
        mock_connect.return_value = mock_conn

        config = Config(database_url="postgresql://test", openai_api_key="sk-test")
        result = stage_11_llm_classification.run(config)

        assert result.success is True
        assert result.records_processed == 0
        assert "already classified" in result.message

    @patch("rl_emails.pipeline.stages.stage_11_llm_classification.psycopg2.connect")
    @patch("rl_emails.pipeline.stages.stage_11_llm_classification.process_batch")
    def test_run_success(self, mock_batch: MagicMock, mock_connect: MagicMock) -> None:
        """Test successful run."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100,), (50,)]  # 50 remaining
        mock_cursor.fetchall.side_effect = [
            [
                (
                    1,
                    "test@example.com",
                    "Subject",
                    None,
                    "Body",
                    0.5,
                    0.3,
                    False,
                    "high",
                    "PENDING",
                    100,
                    False,
                    0.5,
                    10,
                )
            ],
            [],  # No more
        ]
        mock_connect.return_value = mock_conn

        mock_batch.return_value = {
            "results": [{"email_id": 1}],
            "tokens": 120,
            "errors": 0,
            "saved": 1,
        }

        config = Config(database_url="postgresql://test", openai_api_key="sk-test")
        result = stage_11_llm_classification.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 1
        mock_conn.close.assert_called_once()

    @patch("rl_emails.pipeline.stages.stage_11_llm_classification.psycopg2.connect")
    @patch("rl_emails.pipeline.stages.stage_11_llm_classification.process_batch")
    def test_run_with_limit(self, mock_batch: MagicMock, mock_connect: MagicMock) -> None:
        """Test run with limit parameter."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(1000,), (0,)]  # 1000 remaining, limit to 10
        mock_cursor.fetchall.side_effect = [
            [
                (
                    i,
                    f"test{i}@example.com",
                    "Subject",
                    None,
                    "Body",
                    0.5,
                    0.3,
                    False,
                    "high",
                    "PENDING",
                    100,
                    False,
                    0.5,
                    10,
                )
                for i in range(10)
            ],
            [],  # No more
        ]
        mock_connect.return_value = mock_conn

        mock_batch.return_value = {
            "results": [{"email_id": i} for i in range(10)],
            "tokens": 1200,
            "errors": 0,
            "saved": 10,
        }

        config = Config(database_url="postgresql://test", openai_api_key="sk-test")
        result = stage_11_llm_classification.run(config, limit=10)

        assert result.success is True
        assert result.records_processed == 10

    def test_run_without_litellm(self) -> None:
        """Test run fails without litellm package."""
        config = Config(database_url="postgresql://test", openai_api_key="sk-test")

        with patch.dict("sys.modules", {"litellm": None}):
            with patch("rl_emails.pipeline.stages.stage_11_llm_classification.psycopg2.connect"):
                with patch("builtins.__import__", side_effect=ImportError("No module")):
                    result = stage_11_llm_classification.run(config)

        assert result.success is False
        assert "litellm" in result.message
