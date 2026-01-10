"""Tests for inbox service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from rl_emails.services.inbox_service import InboxService


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    return AsyncMock()


@pytest.fixture
def service(mock_session: AsyncMock) -> InboxService:
    """Create service with mock session."""
    return InboxService(mock_session)


class TestInboxService:
    """Tests for InboxService."""

    @pytest.mark.asyncio
    async def test_init(self, mock_session: AsyncMock) -> None:
        """Test service initialization."""
        service = InboxService(mock_session)
        assert service._session == mock_session

    @pytest.mark.asyncio
    async def test_get_priority_inbox_empty(
        self, service: InboxService, mock_session: AsyncMock
    ) -> None:
        """Test getting priority inbox with no emails."""
        # Mock main query
        mock_main_result = MagicMock()
        mock_main_result.fetchall.return_value = []

        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        # Mock stats queries
        mock_stats_results = [
            MagicMock(scalar=MagicMock(return_value=0)),  # total
            MagicMock(scalar=MagicMock(return_value=0)),  # pending
            MagicMock(scalar=MagicMock(return_value=0)),  # urgent
            MagicMock(scalar=MagicMock(return_value=0)),  # real people
            MagicMock(scalar=MagicMock(return_value=None)),  # avg
            MagicMock(scalar=MagicMock(return_value=None)),  # oldest
        ]

        mock_session.execute.side_effect = [
            mock_main_result,
            mock_count_result,
            *mock_stats_results,
        ]

        result = await service.get_priority_inbox()

        assert result.emails == []
        assert result.total == 0
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_get_priority_inbox_with_emails(
        self, service: InboxService, mock_session: AsyncMock
    ) -> None:
        """Test getting priority inbox with emails."""
        # Create mock email row
        email_row = (
            1,  # id
            "msg-123",  # message_id
            "thread-123",  # thread_id
            "Test Subject",  # subject
            "test@example.com",  # from_email
            "Test User",  # from_name
            datetime.now(UTC),  # date_parsed
            "Preview text...",  # body_preview
            False,  # is_sent
            "received",  # action
            False,  # has_attachments
            ["INBOX"],  # labels
            1,  # priority_rank
            0.8,  # priority_score
            "test@example.com",  # sender_email
            0.9,  # sender_importance
            0.7,  # sender_reply_rate
            3,  # thread_length
            True,  # is_business_hours
            2.5,  # age_hours
            0.85,  # people_score
            0.75,  # temporal_score
            0.8,  # relationship_score
            0.8,  # overall_priority
            2,  # task_count
            "Test Project",  # project_name
        )

        mock_main_result = MagicMock()
        mock_main_result.fetchall.return_value = [email_row]

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_stats_results = [
            MagicMock(scalar=MagicMock(return_value=1)),  # total
            MagicMock(scalar=MagicMock(return_value=2)),  # pending
            MagicMock(scalar=MagicMock(return_value=1)),  # urgent
            MagicMock(scalar=MagicMock(return_value=1)),  # real people
            MagicMock(scalar=MagicMock(return_value=0.5)),  # avg
            MagicMock(scalar=MagicMock(return_value=24.0)),  # oldest
        ]

        mock_session.execute.side_effect = [
            mock_main_result,
            mock_count_result,
            *mock_stats_results,
        ]

        result = await service.get_priority_inbox()

        assert len(result.emails) == 1
        assert result.total == 1
        assert result.emails[0].email.subject == "Test Subject"
        assert result.emails[0].priority_rank == 1
        assert result.emails[0].context is not None
        assert result.emails[0].context.sender_email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_priority_inbox_with_pagination(
        self, service: InboxService, mock_session: AsyncMock
    ) -> None:
        """Test getting priority inbox with pagination."""
        mock_main_result = MagicMock()
        mock_main_result.fetchall.return_value = []

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 100

        mock_stats_results = [
            MagicMock(scalar=MagicMock(return_value=100)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=None)),
            MagicMock(scalar=MagicMock(return_value=None)),
        ]

        mock_session.execute.side_effect = [
            mock_main_result,
            mock_count_result,
            *mock_stats_results,
        ]

        result = await service.get_priority_inbox(limit=10, offset=20)

        assert result.limit == 10
        assert result.offset == 20
        assert result.has_more is True

    @pytest.mark.asyncio
    async def test_get_priority_inbox_email_without_context(
        self, service: InboxService, mock_session: AsyncMock
    ) -> None:
        """Test getting priority inbox email without priority context."""
        email_row = (
            1,  # id
            "msg-123",  # message_id
            None,  # thread_id
            "Test Subject",  # subject
            "test@example.com",  # from_email
            None,  # from_name
            datetime.now(UTC),  # date_parsed
            None,  # body_preview
            False,  # is_sent
            None,  # action
            None,  # has_attachments
            None,  # labels
            None,  # priority_rank
            None,  # priority_score
            None,  # sender_email (no context)
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,  # task_count
            None,  # project_name
        )

        mock_main_result = MagicMock()
        mock_main_result.fetchall.return_value = [email_row]

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_stats_results = [
            MagicMock(scalar=MagicMock(return_value=1)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=None)),
            MagicMock(scalar=MagicMock(return_value=None)),
        ]

        mock_session.execute.side_effect = [
            mock_main_result,
            mock_count_result,
            *mock_stats_results,
        ]

        result = await service.get_priority_inbox()

        assert len(result.emails) == 1
        assert result.emails[0].context is None
        assert result.emails[0].priority_rank == 0

    @pytest.mark.asyncio
    async def test_get_inbox_stats(self, service: InboxService, mock_session: AsyncMock) -> None:
        """Test getting inbox stats."""
        mock_session.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=100)),  # total
            MagicMock(scalar=MagicMock(return_value=5)),  # pending
            MagicMock(scalar=MagicMock(return_value=3)),  # urgent
            MagicMock(scalar=MagicMock(return_value=50)),  # real people
            MagicMock(scalar=MagicMock(return_value=0.65)),  # avg
            MagicMock(scalar=MagicMock(return_value=48.0)),  # oldest
        ]

        result = await service.get_inbox_stats()

        assert result.total_emails == 100
        assert result.pending_tasks == 5
        assert result.urgent_emails == 3
        assert result.from_real_people == 50
        assert result.avg_priority_score == 0.65
        assert result.oldest_unanswered_hours == 48.0

    @pytest.mark.asyncio
    async def test_get_inbox_stats_with_nulls(
        self, service: InboxService, mock_session: AsyncMock
    ) -> None:
        """Test getting inbox stats with null values."""
        mock_session.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=None)),  # total
            MagicMock(scalar=MagicMock(return_value=None)),  # pending
            MagicMock(scalar=MagicMock(return_value=None)),  # urgent
            MagicMock(scalar=MagicMock(return_value=None)),  # real people
            MagicMock(scalar=MagicMock(return_value=None)),  # avg
            MagicMock(scalar=MagicMock(return_value=None)),  # oldest
        ]

        result = await service.get_inbox_stats()

        assert result.total_emails == 0
        assert result.pending_tasks == 0
        assert result.urgent_emails == 0
        assert result.from_real_people == 0
        assert result.avg_priority_score is None
        assert result.oldest_unanswered_hours is None

    @pytest.mark.asyncio
    async def test_get_inbox_stats_with_user_id(
        self, service: InboxService, mock_session: AsyncMock
    ) -> None:
        """Test getting inbox stats with user_id."""
        user_id = uuid.uuid4()

        mock_session.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=10)),
            MagicMock(scalar=MagicMock(return_value=1)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=5)),
            MagicMock(scalar=MagicMock(return_value=0.4)),
            MagicMock(scalar=MagicMock(return_value=12.0)),
        ]

        result = await service.get_inbox_stats(user_id)

        assert result.total_emails == 10
        assert result.unread_count == 0  # Always 0 without label tracking
