"""Tests for Stage 3: Populate threads table."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_03_populate_threads
from rl_emails.pipeline.stages.base import StageResult


class TestPopulateThreadsAsync:
    """Tests for populate_threads_async function."""

    @pytest.mark.asyncio
    async def test_populates_threads(self) -> None:
        """Test basic thread population."""
        mock_conn = AsyncMock()

        # Mock thread data
        mock_conn.fetch.return_value = [
            {
                "thread_id": "thread1",
                "subject": "Test Thread",
                "participants": ["user1@example.com", "user2@example.com"],
                "email_count": 5,
                "your_email_count": 2,
                "your_reply_count": 1,
                "started_at": None,
                "last_activity": None,
                "thread_duration_seconds": 3600,
                "has_attachments": False,
                "total_attachment_count": 0,
                "avg_response_time_seconds": 1800,
            }
        ]

        count, stats = await stage_03_populate_threads.populate_threads_async(mock_conn)

        assert count == 1
        assert stats["threads_inserted"] == 1
        mock_conn.execute.assert_called()  # DELETE was called
        mock_conn.executemany.assert_called()

    @pytest.mark.asyncio
    async def test_handles_empty_data(self) -> None:
        """Test handling when no threads exist."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        count, stats = await stage_03_populate_threads.populate_threads_async(mock_conn)

        assert count == 0
        assert stats["threads_inserted"] == 0
        mock_conn.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_null_values(self) -> None:
        """Test handling of null values in thread data."""
        mock_conn = AsyncMock()

        mock_conn.fetch.return_value = [
            {
                "thread_id": "thread1",
                "subject": None,
                "participants": None,
                "email_count": 1,
                "your_email_count": 0,
                "your_reply_count": 0,
                "started_at": None,
                "last_activity": None,
                "thread_duration_seconds": None,
                "has_attachments": None,
                "total_attachment_count": None,
                "avg_response_time_seconds": None,
            }
        ]

        count, stats = await stage_03_populate_threads.populate_threads_async(mock_conn)

        assert count == 1
        # Verify executemany was called with proper defaults
        call_args = mock_conn.executemany.call_args
        insert_data = call_args[0][1]
        row = insert_data[0]
        assert row[2] == []  # participants defaults to empty list
        assert row[10] is False  # has_attachments defaults to False
        assert row[11] == 0  # total_attachment_count defaults to 0

    @pytest.mark.asyncio
    async def test_batches_inserts(self) -> None:
        """Test that inserts are batched correctly."""
        mock_conn = AsyncMock()

        # Create 5 threads
        mock_conn.fetch.return_value = [
            {
                "thread_id": f"thread{i}",
                "subject": f"Thread {i}",
                "participants": [],
                "email_count": 1,
                "your_email_count": 0,
                "your_reply_count": 0,
                "started_at": None,
                "last_activity": None,
                "thread_duration_seconds": None,
                "has_attachments": False,
                "total_attachment_count": 0,
                "avg_response_time_seconds": None,
            }
            for i in range(5)
        ]

        # Use batch_size=2 to force multiple batches
        count, stats = await stage_03_populate_threads.populate_threads_async(
            mock_conn, batch_size=2
        )

        assert count == 5
        # Should have 3 batches: 2 + 2 + 1
        assert mock_conn.executemany.call_count == 3

    @pytest.mark.asyncio
    async def test_clears_existing_data(self) -> None:
        """Test that existing thread data is cleared."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "thread_id": "thread1",
                "subject": "Test",
                "participants": [],
                "email_count": 1,
                "your_email_count": 0,
                "your_reply_count": 0,
                "started_at": None,
                "last_activity": None,
                "thread_duration_seconds": None,
                "has_attachments": False,
                "total_attachment_count": 0,
                "avg_response_time_seconds": None,
            }
        ]

        await stage_03_populate_threads.populate_threads_async(mock_conn)

        # Check DELETE was called
        delete_calls = [c for c in mock_conn.execute.call_args_list if "DELETE" in str(c)]
        assert len(delete_calls) == 1


class TestConvertDbUrl:
    """Tests for convert_db_url function."""

    def test_converts_postgresql_to_postgres(self) -> None:
        """Test postgresql:// is converted to postgres://."""
        result = stage_03_populate_threads.convert_db_url("postgresql://user:pass@host/db")
        assert result == "postgres://user:pass@host/db"

    def test_preserves_postgres_url(self) -> None:
        """Test postgres:// URL is preserved."""
        result = stage_03_populate_threads.convert_db_url("postgres://user:pass@host/db")
        assert result == "postgres://user:pass@host/db"

    def test_preserves_other_url(self) -> None:
        """Test other URLs are preserved."""
        result = stage_03_populate_threads.convert_db_url("sqlite:///path/to/db")
        assert result == "sqlite:///path/to/db"


class TestRunAsync:
    """Tests for run_async function."""

    @pytest.mark.asyncio
    async def test_connects_and_populates(self) -> None:
        """Test run_async connects to DB and populates threads."""
        with patch(
            "rl_emails.pipeline.stages.stage_03_populate_threads.asyncpg.connect"
        ) as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn
            mock_conn.fetch.return_value = []

            count, stats = await stage_03_populate_threads.run_async("postgres://test")

            mock_connect.assert_called_once_with("postgres://test")
            assert count == 0
            mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_connection_on_success(self) -> None:
        """Test connection is closed on success."""
        with patch(
            "rl_emails.pipeline.stages.stage_03_populate_threads.asyncpg.connect"
        ) as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn
            mock_conn.fetch.return_value = []

            await stage_03_populate_threads.run_async("postgres://test")

            mock_conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_connection_on_error(self) -> None:
        """Test connection is closed even on error."""
        with patch(
            "rl_emails.pipeline.stages.stage_03_populate_threads.asyncpg.connect"
        ) as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn
            mock_conn.fetch.side_effect = Exception("DB error")

            with pytest.raises(Exception, match="DB error"):
                await stage_03_populate_threads.run_async("postgres://test")

            mock_conn.close.assert_called_once()


class TestRun:
    """Tests for run function."""

    @patch("rl_emails.pipeline.stages.stage_03_populate_threads.asyncio.run")
    def test_run_returns_stage_result(self, mock_asyncio_run: MagicMock) -> None:
        """Test run returns proper StageResult."""
        mock_asyncio_run.return_value = (100, {"threads_inserted": 100})

        config = Config(database_url="postgresql://test")
        result = stage_03_populate_threads.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 100
        assert result.metadata is not None
        assert result.metadata["threads_inserted"] == 100

    @patch("rl_emails.pipeline.stages.stage_03_populate_threads.asyncio.run")
    def test_run_with_custom_batch_size(self, mock_asyncio_run: MagicMock) -> None:
        """Test run with custom batch size."""
        mock_asyncio_run.return_value = (50, {"threads_inserted": 50})

        config = Config(database_url="postgresql://test")
        result = stage_03_populate_threads.run(config, batch_size=500)

        assert result.success is True
        assert result.records_processed == 50

    @patch("rl_emails.pipeline.stages.stage_03_populate_threads.asyncio.run")
    def test_run_calculates_duration(self, mock_asyncio_run: MagicMock) -> None:
        """Test run calculates duration."""
        mock_asyncio_run.return_value = (10, {"threads_inserted": 10})

        config = Config(database_url="postgresql://test")
        result = stage_03_populate_threads.run(config)

        assert result.duration_seconds >= 0
