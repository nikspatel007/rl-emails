"""Tests for Stage 4: Enrich emails with action labels."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_04_enrich_emails
from rl_emails.pipeline.stages.base import StageResult


class TestClassifyResponseTime:
    """Tests for classify_response_time function."""

    def test_immediate(self) -> None:
        """Test immediate response (< 1 hour)."""
        assert stage_04_enrich_emails.classify_response_time(1800) == "IMMEDIATE"

    def test_same_day(self) -> None:
        """Test same day response (1-24 hours)."""
        assert stage_04_enrich_emails.classify_response_time(7200) == "SAME_DAY"

    def test_next_day(self) -> None:
        """Test next day response (24-48 hours)."""
        assert stage_04_enrich_emails.classify_response_time(100000) == "NEXT_DAY"

    def test_later(self) -> None:
        """Test later response (>= 48 hours)."""
        assert stage_04_enrich_emails.classify_response_time(200000) == "LATER"


class TestParseUpdateCount:
    """Tests for _parse_update_count function."""

    def test_none_input(self) -> None:
        """Test None input returns 0."""
        assert stage_04_enrich_emails._parse_update_count(None) == 0

    def test_empty_string(self) -> None:
        """Test empty string returns 0."""
        assert stage_04_enrich_emails._parse_update_count("") == 0

    def test_update_format(self) -> None:
        """Test standard UPDATE N format."""
        assert stage_04_enrich_emails._parse_update_count("UPDATE 100") == 100

    def test_invalid_number(self) -> None:
        """Test invalid number returns 0."""
        assert stage_04_enrich_emails._parse_update_count("UPDATE abc") == 0

    def test_single_word(self) -> None:
        """Test single word returns 0."""
        assert stage_04_enrich_emails._parse_update_count("UPDATE") == 0


class TestMarkSentEmails:
    """Tests for mark_sent_emails function."""

    @pytest.mark.asyncio
    async def test_empty_emails_list(self) -> None:
        """Test empty emails list returns 0."""
        conn = AsyncMock()
        result = await stage_04_enrich_emails.mark_sent_emails(conn, [])
        assert result == 0
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_marks_sent(self) -> None:
        """Test marking sent emails."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 50"

        result = await stage_04_enrich_emails.mark_sent_emails(
            conn, ["user@example.com", "alias@example.com"]
        )

        assert result == 50
        conn.execute.assert_called_once()


class TestComputeRepliedActions:
    """Tests for compute_replied_actions function."""

    @pytest.mark.asyncio
    async def test_computes_replied(self) -> None:
        """Test computing replied actions."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 25"

        result = await stage_04_enrich_emails.compute_replied_actions(conn)

        assert result == 25
        conn.execute.assert_called_once()


class TestComputeStarredActions:
    """Tests for compute_starred_actions function."""

    @pytest.mark.asyncio
    async def test_computes_starred(self) -> None:
        """Test computing starred actions."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 10"

        result = await stage_04_enrich_emails.compute_starred_actions(conn)

        assert result == 10


class TestComputeArchivedActions:
    """Tests for compute_archived_actions function."""

    @pytest.mark.asyncio
    async def test_computes_archived(self) -> None:
        """Test computing archived actions."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 100"
        reference_date = datetime(2024, 1, 15, 12, 0, 0)

        result = await stage_04_enrich_emails.compute_archived_actions(conn, reference_date)

        assert result == 100
        conn.execute.assert_called_once()


class TestComputeIgnoredActions:
    """Tests for compute_ignored_actions function."""

    @pytest.mark.asyncio
    async def test_computes_ignored(self) -> None:
        """Test computing ignored actions."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 30"
        reference_date = datetime(2024, 1, 15, 12, 0, 0)

        result = await stage_04_enrich_emails.compute_ignored_actions(conn, reference_date)

        assert result == 30


class TestComputePendingActions:
    """Tests for compute_pending_actions function."""

    @pytest.mark.asyncio
    async def test_computes_pending(self) -> None:
        """Test computing pending actions."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 15"

        result = await stage_04_enrich_emails.compute_pending_actions(conn)

        assert result == 15


class TestMarkSentAction:
    """Tests for mark_sent_action function."""

    @pytest.mark.asyncio
    async def test_marks_composed(self) -> None:
        """Test marking composed actions."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 40"

        result = await stage_04_enrich_emails.mark_sent_action(conn)

        assert result == 40


class TestComputeReverseTracking:
    """Tests for compute_reverse_tracking function."""

    @pytest.mark.asyncio
    async def test_computes_reverse_tracking(self) -> None:
        """Test computing reverse tracking."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 20"

        result = await stage_04_enrich_emails.compute_reverse_tracking(conn)

        assert result == 20


class TestEnrichEmailsAsync:
    """Tests for enrich_emails_async function."""

    @pytest.mark.asyncio
    async def test_full_enrichment(self) -> None:
        """Test full enrichment pipeline."""
        conn = AsyncMock()
        conn.fetchval.side_effect = [
            1000,  # total emails
            datetime(2024, 1, 15, 12, 0, 0),  # reference date
        ]
        conn.execute.return_value = "UPDATE 50"

        stats = await stage_04_enrich_emails.enrich_emails_async(conn, ["user@example.com"])

        assert stats["total_emails"] == 1000
        assert stats["sent_marked"] == 50
        assert stats["replied"] == 50

    @pytest.mark.asyncio
    async def test_enrichment_with_null_date(self) -> None:
        """Test enrichment uses current date if no emails have dates."""
        conn = AsyncMock()
        conn.fetchval.side_effect = [
            500,  # total emails
            None,  # no reference date (NULL)
        ]
        conn.execute.return_value = "UPDATE 10"

        stats = await stage_04_enrich_emails.enrich_emails_async(conn, ["user@example.com"])

        assert stats["total_emails"] == 500

    @pytest.mark.asyncio
    async def test_enrichment_with_null_total(self) -> None:
        """Test enrichment handles NULL total count."""
        conn = AsyncMock()
        conn.fetchval.side_effect = [
            None,  # NULL total
            datetime(2024, 1, 15),
        ]
        conn.execute.return_value = "UPDATE 0"

        stats = await stage_04_enrich_emails.enrich_emails_async(conn, [])

        assert stats["total_emails"] == 0


class TestConvertDbUrl:
    """Tests for _convert_db_url function."""

    def test_postgresql_converted(self) -> None:
        """Test postgresql:// is converted."""
        result = stage_04_enrich_emails._convert_db_url("postgresql://user:pass@host/db")
        assert result == "postgres://user:pass@host/db"

    def test_postgres_unchanged(self) -> None:
        """Test postgres:// is unchanged."""
        result = stage_04_enrich_emails._convert_db_url("postgres://user:pass@host/db")
        assert result == "postgres://user:pass@host/db"


class TestRunAsync:
    """Tests for run_async function."""

    @pytest.mark.asyncio
    @patch("rl_emails.pipeline.stages.stage_04_enrich_emails.asyncpg.connect")
    async def test_connects_and_enriches(self, mock_connect: MagicMock) -> None:
        """Test run_async connects and enriches."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = [100, datetime(2024, 1, 15)]
        mock_conn.execute.return_value = "UPDATE 10"
        mock_connect.return_value = mock_conn

        stats = await stage_04_enrich_emails.run_async("postgres://test", ["user@example.com"])

        mock_connect.assert_called_once_with("postgres://test")
        mock_conn.close.assert_called_once()
        assert stats["total_emails"] == 100

    @pytest.mark.asyncio
    @patch("rl_emails.pipeline.stages.stage_04_enrich_emails.asyncpg.connect")
    async def test_closes_on_error(self, mock_connect: MagicMock) -> None:
        """Test connection is closed on error."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = Exception("DB error")
        mock_connect.return_value = mock_conn

        with pytest.raises(Exception, match="DB error"):
            await stage_04_enrich_emails.run_async("postgres://test", ["user@example.com"])

        mock_conn.close.assert_called_once()


class TestRun:
    """Tests for run function."""

    def test_run_without_your_email(self) -> None:
        """Test run fails without your_email configured."""
        config = Config(database_url="postgresql://test")

        result = stage_04_enrich_emails.run(config)

        assert result.success is False
        assert "YOUR_EMAIL not configured" in result.message

    @patch("rl_emails.pipeline.stages.stage_04_enrich_emails.asyncio.run")
    def test_run_success(self, mock_asyncio_run: MagicMock) -> None:
        """Test successful run."""
        mock_asyncio_run.return_value = {
            "total_emails": 1000,
            "sent_marked": 100,
            "replied": 50,
            "starred": 10,
            "archived": 200,
            "ignored": 30,
            "pending": 500,
            "composed": 100,
            "reverse_tracked": 20,
        }

        config = Config(database_url="postgresql://test", your_email="user@example.com")
        result = stage_04_enrich_emails.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        # Total enriched = replied + starred + archived + ignored + pending + composed
        assert result.records_processed == 890
        assert result.metadata is not None
        assert result.metadata["total_emails"] == 1000
