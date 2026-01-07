"""Tests for Stage 8: Populate users table."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_08_populate_users
from rl_emails.pipeline.stages.base import StageResult


class TestIsYourEmail:
    """Tests for is_your_email function."""

    def test_exact_match(self) -> None:
        """Test exact email match."""
        assert stage_08_populate_users.is_your_email("user@example.com", "user@example.com") is True

    def test_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        assert stage_08_populate_users.is_your_email("USER@Example.COM", "user@example.com") is True

    def test_no_match(self) -> None:
        """Test no match."""
        assert (
            stage_08_populate_users.is_your_email("other@example.com", "user@example.com") is False
        )

    def test_empty_your_email(self) -> None:
        """Test empty your_email returns False."""
        assert stage_08_populate_users.is_your_email("user@example.com", "") is False

    def test_domain_variant_match(self) -> None:
        """Test domain variant matching (same domain, base name in email)."""
        assert (
            stage_08_populate_users.is_your_email("user.name@example.com", "user@example.com")
            is True
        )

    def test_different_domain(self) -> None:
        """Test different domain doesn't match."""
        assert stage_08_populate_users.is_your_email("user@other.com", "user@example.com") is False

    def test_your_email_no_at_symbol(self) -> None:
        """Test your_email without @ symbol."""
        assert stage_08_populate_users.is_your_email("user@example.com", "invalid") is False


class TestPopulateUsersFromEmails:
    """Tests for populate_users_from_emails function."""

    @patch("rl_emails.pipeline.stages.stage_08_populate_users.execute_values")
    def test_populates_users(self, mock_execute_values: MagicMock) -> None:
        """Test basic user population."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Mock: unique emails query returns 2 users
        mock_cursor.fetchall.side_effect = [
            [("user1@example.com", "User One"), ("user2@example.com", None)],  # unique emails
        ]
        # Mock: final counts
        mock_cursor.fetchone.side_effect = [
            (2,),  # total count
            (1,),  # is_you count
            (0,),  # important sender count
        ]

        count, stats = stage_08_populate_users.populate_users_from_emails(
            mock_conn, "user1@example.com"
        )

        assert count == 2
        assert stats["unique_emails"] == 2
        assert mock_execute_values.called

    @patch("rl_emails.pipeline.stages.stage_08_populate_users.execute_values")
    def test_handles_empty_emails(self, mock_execute_values: MagicMock) -> None:
        """Test handling when no emails exist."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # No users found
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.side_effect = [(0,), (0,), (0,)]

        count, stats = stage_08_populate_users.populate_users_from_emails(
            mock_conn, "me@example.com"
        )

        assert count == 0
        assert stats["unique_emails"] == 0
        # execute_values should not be called with empty data
        mock_execute_values.assert_not_called()

    @patch("rl_emails.pipeline.stages.stage_08_populate_users.execute_values")
    def test_truncates_table_first(self, mock_execute_values: MagicMock) -> None:
        """Test that TRUNCATE is called before populating."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.side_effect = [(0,), (0,), (0,)]

        stage_08_populate_users.populate_users_from_emails(mock_conn, "me@example.com")

        # Check that TRUNCATE was called
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("TRUNCATE" in str(c) for c in calls)

    @patch("rl_emails.pipeline.stages.stage_08_populate_users.execute_values")
    def test_marks_your_emails(self, mock_execute_values: MagicMock) -> None:
        """Test that your emails are marked correctly."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # One email is yours
        mock_cursor.fetchall.return_value = [
            ("me@example.com", "Me"),
            ("other@example.com", "Other"),
        ]
        mock_cursor.fetchone.side_effect = [(2,), (1,), (0,)]

        stage_08_populate_users.populate_users_from_emails(mock_conn, "me@example.com")

        # Check execute_values was called with correct is_you flags
        call_args = mock_execute_values.call_args
        user_data = call_args[0][2]  # Third positional argument is the data

        # Find the row for me@example.com
        me_row = [row for row in user_data if row[0] == "me@example.com"][0]
        other_row = [row for row in user_data if row[0] == "other@example.com"][0]

        assert me_row[2] is True  # is_you
        assert other_row[2] is False


class TestRun:
    """Tests for run function."""

    @patch("rl_emails.pipeline.stages.stage_08_populate_users.get_connection")
    @patch("rl_emails.pipeline.stages.stage_08_populate_users.populate_users_from_emails")
    def test_run_returns_stage_result(
        self, mock_populate: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        """Test run returns proper StageResult."""
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_populate.return_value = (
            100,
            {"unique_emails": 100, "is_you_count": 1, "important_sender_count": 10},
        )

        config = Config(database_url="postgresql://test", your_email="test@example.com")
        result = stage_08_populate_users.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 100
        assert result.metadata is not None
        assert result.metadata["unique_emails"] == 100
        assert result.metadata["is_you_count"] == 1
        assert result.metadata["important_sender_count"] == 10

    @patch("rl_emails.pipeline.stages.stage_08_populate_users.get_connection")
    @patch("rl_emails.pipeline.stages.stage_08_populate_users.populate_users_from_emails")
    def test_run_with_no_your_email(
        self, mock_populate: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        """Test run works without your_email configured."""
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_populate.return_value = (
            50,
            {"unique_emails": 50, "is_you_count": 0, "important_sender_count": 5},
        )

        config = Config(database_url="postgresql://test")  # No your_email
        result = stage_08_populate_users.run(config)

        assert result.success is True
        assert result.records_processed == 50
        # Verify populate was called with empty string for your_email
        mock_populate.assert_called_once_with(mock_conn, "")

    @patch("rl_emails.pipeline.stages.stage_08_populate_users.get_connection")
    @patch("rl_emails.pipeline.stages.stage_08_populate_users.populate_users_from_emails")
    def test_run_calculates_duration(
        self, mock_populate: MagicMock, mock_get_connection: MagicMock
    ) -> None:
        """Test run calculates duration."""
        mock_conn = MagicMock()
        mock_get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_populate.return_value = (
            10,
            {"unique_emails": 10, "is_you_count": 0, "important_sender_count": 0},
        )

        config = Config(database_url="postgresql://test")
        result = stage_08_populate_users.run(config)

        assert result.duration_seconds >= 0
