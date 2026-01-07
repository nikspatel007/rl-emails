"""Tests for Stage 2: Import JSONL to PostgreSQL."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_02_import_postgres
from rl_emails.pipeline.stages.base import StageResult


class TestSanitizeText:
    """Tests for sanitize_text function."""

    def test_none_input(self) -> None:
        """Test None input returns None."""
        assert stage_02_import_postgres.sanitize_text(None) is None

    def test_removes_null_bytes(self) -> None:
        """Test null bytes are removed."""
        result = stage_02_import_postgres.sanitize_text("Hello\x00World")
        assert result == "HelloWorld"

    def test_normalizes_crlf(self) -> None:
        """Test CRLF is normalized to LF."""
        result = stage_02_import_postgres.sanitize_text("Line1\r\nLine2")
        assert result == "Line1\nLine2"

    def test_normalizes_cr(self) -> None:
        """Test CR is normalized to LF."""
        result = stage_02_import_postgres.sanitize_text("Line1\rLine2")
        assert result == "Line1\nLine2"

    def test_clean_text_unchanged(self) -> None:
        """Test clean text is unchanged."""
        result = stage_02_import_postgres.sanitize_text("Hello World")
        assert result == "Hello World"


class TestParseEmailAddress:
    """Tests for parse_email_address function."""

    def test_none_input(self) -> None:
        """Test None input returns None, None."""
        email, name = stage_02_import_postgres.parse_email_address(None)
        assert email is None
        assert name is None

    def test_empty_input(self) -> None:
        """Test empty input returns None, None."""
        email, name = stage_02_import_postgres.parse_email_address("")
        assert email is None
        assert name is None

    def test_name_with_angle_brackets(self) -> None:
        """Test format: Name <email@example.com>."""
        email, name = stage_02_import_postgres.parse_email_address("John Doe <john@example.com>")
        assert email == "john@example.com"
        assert name == "John Doe"

    def test_quoted_name_with_angle_brackets(self) -> None:
        """Test format: "Name" <email@example.com>."""
        email, name = stage_02_import_postgres.parse_email_address('"Jane Doe" <jane@example.com>')
        assert email == "jane@example.com"
        assert name == "Jane Doe"

    def test_email_only_in_brackets(self) -> None:
        """Test format: <email@example.com> with no name."""
        email, name = stage_02_import_postgres.parse_email_address("<test@example.com>")
        assert email == "test@example.com"
        assert name is None

    def test_bare_email(self) -> None:
        """Test format: email@example.com."""
        email, name = stage_02_import_postgres.parse_email_address("test@example.com")
        assert email == "test@example.com"
        assert name is None

    def test_no_at_sign(self) -> None:
        """Test input without @ returns None."""
        email, name = stage_02_import_postgres.parse_email_address("not an email")
        assert email is None
        assert name is None

    def test_email_lowercased(self) -> None:
        """Test email is lowercased."""
        email, name = stage_02_import_postgres.parse_email_address("Test@EXAMPLE.COM")
        assert email == "test@example.com"


class TestExtractAllEmails:
    """Tests for extract_all_emails function."""

    def test_none_input(self) -> None:
        """Test None input returns empty list."""
        assert stage_02_import_postgres.extract_all_emails(None) == []

    def test_empty_input(self) -> None:
        """Test empty input returns empty list."""
        assert stage_02_import_postgres.extract_all_emails("") == []

    def test_comma_separated(self) -> None:
        """Test comma-separated emails."""
        result = stage_02_import_postgres.extract_all_emails("a@example.com, b@example.com")
        assert result == ["a@example.com", "b@example.com"]

    def test_semicolon_separated(self) -> None:
        """Test semicolon-separated emails."""
        result = stage_02_import_postgres.extract_all_emails("a@example.com; b@example.com")
        assert result == ["a@example.com", "b@example.com"]

    def test_with_names(self) -> None:
        """Test emails with names."""
        result = stage_02_import_postgres.extract_all_emails(
            "Alice <a@example.com>, Bob <b@example.com>"
        )
        assert result == ["a@example.com", "b@example.com"]

    def test_invalid_entries_filtered(self) -> None:
        """Test invalid entries are filtered out."""
        result = stage_02_import_postgres.extract_all_emails("a@example.com, invalid")
        assert result == ["a@example.com"]


class TestParseDate:
    """Tests for parse_date function."""

    def test_none_input(self) -> None:
        """Test None input returns None."""
        assert stage_02_import_postgres.parse_date(None) is None

    def test_empty_input(self) -> None:
        """Test empty input returns None."""
        assert stage_02_import_postgres.parse_date("") is None

    def test_rfc_2822_format(self) -> None:
        """Test RFC 2822 date format."""
        result = stage_02_import_postgres.parse_date("Mon, 01 Jan 2024 12:00:00 +0000")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_iso_format(self) -> None:
        """Test ISO date format."""
        result = stage_02_import_postgres.parse_date("2024-01-15 10:30:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_iso_format_with_timezone(self) -> None:
        """Test ISO date format with timezone."""
        result = stage_02_import_postgres.parse_date("2024-01-15 10:30:00+0500")
        assert result is not None
        assert result.year == 2024

    def test_iso_t_format(self) -> None:
        """Test ISO date with T separator."""
        result = stage_02_import_postgres.parse_date("2024-01-15T10:30:00")
        assert result is not None
        assert result.year == 2024

    def test_iso_t_format_with_timezone(self) -> None:
        """Test ISO date with T separator and timezone."""
        result = stage_02_import_postgres.parse_date("2024-01-15T10:30:00+0000")
        assert result is not None
        assert result.year == 2024

    def test_removes_timezone_comment(self) -> None:
        """Test timezone comment (PST) is stripped."""
        result = stage_02_import_postgres.parse_date("Mon, 01 Jan 2024 12:00:00 +0000 (PST)")
        assert result is not None

    def test_invalid_date_returns_none(self) -> None:
        """Test invalid date returns None."""
        result = stage_02_import_postgres.parse_date("not a date")
        assert result is None

    def test_format_without_day_name(self) -> None:
        """Test date format without day name."""
        result = stage_02_import_postgres.parse_date("15 Jan 2024 10:30:00 +0000")
        assert result is not None
        assert result.day == 15


class TestGeneratePreview:
    """Tests for generate_preview function."""

    def test_none_input(self) -> None:
        """Test None input returns None."""
        assert stage_02_import_postgres.generate_preview(None) is None

    def test_empty_input(self) -> None:
        """Test empty input returns None."""
        assert stage_02_import_postgres.generate_preview("") is None

    def test_short_text(self) -> None:
        """Test short text is returned as-is."""
        result = stage_02_import_postgres.generate_preview("Hello World")
        assert result == "Hello World"

    def test_long_text_truncated(self) -> None:
        """Test long text is truncated with ellipsis."""
        long_text = "word " * 100
        result = stage_02_import_postgres.generate_preview(long_text, max_length=50)
        assert result is not None
        assert len(result) == 53  # 50 chars + "..."
        assert result.endswith("...")

    def test_whitespace_normalized(self) -> None:
        """Test whitespace is normalized."""
        result = stage_02_import_postgres.generate_preview("Hello  \n  World")
        assert result == "Hello World"


class TestCountWords:
    """Tests for count_words function."""

    def test_none_input(self) -> None:
        """Test None input returns 0."""
        assert stage_02_import_postgres.count_words(None) == 0

    def test_empty_input(self) -> None:
        """Test empty input returns 0."""
        assert stage_02_import_postgres.count_words("") == 0

    def test_counts_words(self) -> None:
        """Test word counting."""
        assert stage_02_import_postgres.count_words("Hello World") == 2

    def test_multiple_whitespace(self) -> None:
        """Test multiple whitespace handled."""
        assert stage_02_import_postgres.count_words("Hello   World") == 2


class TestIsSentEmail:
    """Tests for is_sent_email function."""

    def test_none_input(self) -> None:
        """Test None input returns False."""
        assert stage_02_import_postgres.is_sent_email(None) is False

    def test_empty_list(self) -> None:
        """Test empty list returns False."""
        assert stage_02_import_postgres.is_sent_email([]) is False

    def test_sent_label(self) -> None:
        """Test SENT label detected."""
        assert stage_02_import_postgres.is_sent_email(["SENT"]) is True

    def test_sent_label_lowercase(self) -> None:
        """Test sent label in various cases."""
        assert stage_02_import_postgres.is_sent_email(["sent"]) is True
        assert stage_02_import_postgres.is_sent_email(["Sent"]) is True

    def test_no_sent_label(self) -> None:
        """Test no SENT label returns False."""
        assert stage_02_import_postgres.is_sent_email(["Inbox", "Important"]) is False


class TestConvertDbUrl:
    """Tests for convert_db_url function."""

    def test_postgresql_converted(self) -> None:
        """Test postgresql:// is converted to postgres://."""
        result = stage_02_import_postgres.convert_db_url("postgresql://user:pass@host:5432/db")
        assert result == "postgres://user:pass@host:5432/db"

    def test_postgres_unchanged(self) -> None:
        """Test postgres:// is unchanged."""
        result = stage_02_import_postgres.convert_db_url("postgres://user:pass@host:5432/db")
        assert result == "postgres://user:pass@host:5432/db"

    def test_other_scheme_unchanged(self) -> None:
        """Test other schemes are unchanged."""
        result = stage_02_import_postgres.convert_db_url("mysql://user:pass@host:3306/db")
        assert result == "mysql://user:pass@host:3306/db"


class TestImportEmailsAsync:
    """Tests for import_emails_async function."""

    @pytest.mark.asyncio
    async def test_imports_emails(self, tmp_path: Path) -> None:
        """Test importing emails from JSONL."""
        # Create test JSONL file
        jsonl_path = tmp_path / "emails.jsonl"
        emails = [
            {
                "message_id": "<test1@example.com>",
                "thread_id": "123",
                "in_reply_to": "",
                "references": "",
                "date": "Mon, 01 Jan 2024 12:00:00 +0000",
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "cc": "",
                "bcc": "",
                "subject": "Test Subject",
                "body": "Test body",
                "body_html": None,
                "labels": ["Inbox"],
                "has_attachments": False,
                "mbox_offset": 0,
                "raw_size_bytes": 100,
            }
        ]
        with open(jsonl_path, "w") as f:
            for email in emails:
                f.write(json.dumps(email) + "\n")

        # Mock connection
        conn = AsyncMock()
        conn.fetchval.return_value = 1  # Return raw_id

        count, stats = await stage_02_import_postgres.import_emails_async(
            conn, jsonl_path, batch_size=500
        )

        assert count == 1
        assert stats["raw_inserted"] == 1
        assert stats["emails_inserted"] == 1
        assert stats["skipped"] == 0
        assert stats["total_in_file"] == 1

    @pytest.mark.asyncio
    async def test_skips_duplicate(self, tmp_path: Path) -> None:
        """Test duplicate emails are skipped."""
        jsonl_path = tmp_path / "emails.jsonl"
        emails = [
            {
                "message_id": "<test1@example.com>",
                "thread_id": "123",
                "from": "sender@example.com",
                "to": "recipient@example.com",
                "subject": "Test",
                "body": "Body",
                "labels": [],
            }
        ]
        with open(jsonl_path, "w") as f:
            for email in emails:
                f.write(json.dumps(email) + "\n")

        conn = AsyncMock()
        conn.fetchval.return_value = None  # Duplicate, returns None

        count, stats = await stage_02_import_postgres.import_emails_async(
            conn, jsonl_path, batch_size=500
        )

        assert count == 0
        assert stats["skipped"] == 1
        assert stats["raw_inserted"] == 0

    @pytest.mark.asyncio
    async def test_handles_empty_file(self, tmp_path: Path) -> None:
        """Test handling empty JSONL file."""
        jsonl_path = tmp_path / "emails.jsonl"
        jsonl_path.write_text("")

        conn = AsyncMock()

        count, stats = await stage_02_import_postgres.import_emails_async(
            conn, jsonl_path, batch_size=500
        )

        assert count == 0
        assert stats["total_in_file"] == 0

    @pytest.mark.asyncio
    async def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Test blank lines in JSONL file are skipped."""
        jsonl_path = tmp_path / "emails.jsonl"
        email = {
            "message_id": "<test@example.com>",
            "thread_id": "123",
            "from": "sender@example.com",
            "to": "recipient@example.com",
            "subject": "Test",
            "body": "Body",
            "labels": [],
        }
        # File with blank lines
        content = "\n" + json.dumps(email) + "\n\n  \n"
        jsonl_path.write_text(content)

        conn = AsyncMock()
        conn.fetchval.return_value = 1

        count, stats = await stage_02_import_postgres.import_emails_async(
            conn, jsonl_path, batch_size=500
        )

        assert count == 1
        assert stats["total_in_file"] == 1  # Only 1 actual email

    @pytest.mark.asyncio
    async def test_handles_email_with_all_fields(self, tmp_path: Path) -> None:
        """Test email with all optional fields."""
        jsonl_path = tmp_path / "emails.jsonl"
        emails = [
            {
                "message_id": "<test@example.com>",
                "thread_id": "123",
                "in_reply_to": "<reply@example.com>",
                "references": "<ref@example.com>",
                "date": "2024-01-15 10:30:00",
                "date_original": "2024-01-15",
                "from": '"Sender Name" <sender@example.com>',
                "to": "recipient1@example.com, recipient2@example.com",
                "cc": "cc@example.com",
                "bcc": "bcc@example.com",
                "subject": "Full Test",
                "body": "Full body text",
                "body_html": "<html>Body</html>",
                "labels": ["Inbox", "Important"],
                "has_attachments": True,
                "mbox_offset": 100,
                "raw_size_bytes": 500,
                "action": "reply",
                "response_timing": "fast",
                "response_time_seconds": 3600,
            }
        ]
        with open(jsonl_path, "w") as f:
            for email in emails:
                f.write(json.dumps(email) + "\n")

        conn = AsyncMock()
        conn.fetchval.return_value = 1

        count, stats = await stage_02_import_postgres.import_emails_async(
            conn, jsonl_path, batch_size=500
        )

        assert count == 1
        assert stats["emails_inserted"] == 1


class TestRunAsync:
    """Tests for run_async function."""

    @pytest.mark.asyncio
    @patch("rl_emails.pipeline.stages.stage_02_import_postgres.asyncpg.connect")
    async def test_connects_and_imports(self, mock_connect: MagicMock, tmp_path: Path) -> None:
        """Test run_async connects and imports."""
        jsonl_path = tmp_path / "emails.jsonl"
        jsonl_path.write_text("")

        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn

        count, stats = await stage_02_import_postgres.run_async(
            "postgres://test", jsonl_path, batch_size=500
        )

        mock_connect.assert_called_once_with("postgres://test")
        mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("rl_emails.pipeline.stages.stage_02_import_postgres.asyncpg.connect")
    async def test_closes_on_error(self, mock_connect: MagicMock, tmp_path: Path) -> None:
        """Test connection is closed on error."""
        jsonl_path = tmp_path / "nonexistent.jsonl"

        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn

        with pytest.raises(FileNotFoundError):
            await stage_02_import_postgres.run_async("postgres://test", jsonl_path, batch_size=500)

        mock_conn.close.assert_called_once()


class TestRun:
    """Tests for run function."""

    def test_run_without_parsed_jsonl(self) -> None:
        """Test run fails without parsed_jsonl configured."""
        config = Config(database_url="postgresql://test")

        result = stage_02_import_postgres.run(config)

        assert result.success is False
        assert "PARSED_JSONL not configured" in result.message

    def test_run_with_missing_file(self) -> None:
        """Test run fails when JSONL file doesn't exist."""
        config = Config(
            database_url="postgresql://test",
            parsed_jsonl=Path("/nonexistent/emails.jsonl"),
        )

        result = stage_02_import_postgres.run(config)

        assert result.success is False
        assert "not found" in result.message

    @patch("rl_emails.pipeline.stages.stage_02_import_postgres.asyncio.run")
    def test_run_success(self, mock_asyncio_run: MagicMock, tmp_path: Path) -> None:
        """Test successful run."""
        jsonl_path = tmp_path / "emails.jsonl"
        jsonl_path.touch()

        mock_asyncio_run.return_value = (
            100,
            {"raw_inserted": 100, "emails_inserted": 100, "skipped": 0},
        )

        config = Config(database_url="postgresql://test", parsed_jsonl=jsonl_path)
        result = stage_02_import_postgres.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 100
        assert result.metadata is not None
        assert "Imported 100 emails" in result.message

    @patch("rl_emails.pipeline.stages.stage_02_import_postgres.asyncio.run")
    def test_run_with_custom_batch_size(self, mock_asyncio_run: MagicMock, tmp_path: Path) -> None:
        """Test run with custom batch size."""
        jsonl_path = tmp_path / "emails.jsonl"
        jsonl_path.touch()

        mock_asyncio_run.return_value = (50, {"raw_inserted": 50})

        config = Config(database_url="postgresql://test", parsed_jsonl=jsonl_path)
        result = stage_02_import_postgres.run(config, batch_size=100)

        assert result.success is True
        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()
