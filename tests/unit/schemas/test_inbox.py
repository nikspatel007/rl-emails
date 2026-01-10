"""Tests for inbox schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from rl_emails.schemas.inbox import (
    EmailSummary,
    InboxStats,
    PriorityContext,
    PriorityEmail,
    PriorityInboxResponse,
)


class TestEmailSummary:
    """Tests for EmailSummary schema."""

    def test_minimal_fields(self) -> None:
        """Test creating with minimal fields."""
        summary = EmailSummary(
            id=1,
            message_id="<msg123@example.com>",
        )
        assert summary.id == 1
        assert summary.message_id == "<msg123@example.com>"
        assert summary.is_sent is False
        assert summary.has_attachments is False

    def test_all_fields(self) -> None:
        """Test creating with all fields."""
        now = datetime.now(UTC)
        summary = EmailSummary(
            id=1,
            message_id="<msg123@example.com>",
            thread_id="thread_456",
            subject="Re: Important Update",
            from_email="sender@example.com",
            from_name="John Doe",
            date_parsed=now,
            body_preview="Hi, I wanted to follow up...",
            is_sent=False,
            action="received",
            has_attachments=True,
            labels=["INBOX", "IMPORTANT"],
        )
        assert summary.subject == "Re: Important Update"
        assert summary.has_attachments is True
        assert summary.labels == ["INBOX", "IMPORTANT"]


class TestPriorityContext:
    """Tests for PriorityContext schema."""

    def test_minimal_fields(self) -> None:
        """Test creating with minimal fields."""
        context = PriorityContext(email_id=1)
        assert context.email_id == 1
        assert context.overall_priority is None

    def test_all_fields(self) -> None:
        """Test creating with all fields."""
        context = PriorityContext(
            email_id=1,
            sender_email="sender@example.com",
            sender_importance=0.9,
            sender_reply_rate=0.8,
            thread_length=5,
            is_business_hours=True,
            age_hours=2.5,
            people_score=0.85,
            temporal_score=0.7,
            relationship_score=0.9,
            overall_priority=0.82,
        )
        assert context.sender_importance == 0.9
        assert context.overall_priority == 0.82


class TestPriorityEmail:
    """Tests for PriorityEmail schema."""

    def test_minimal_fields(self) -> None:
        """Test creating with minimal fields."""
        email = EmailSummary(id=1, message_id="<msg123@example.com>")
        priority_email = PriorityEmail(
            email=email,
            priority_rank=1,
            priority_score=0.9,
        )
        assert priority_email.priority_rank == 1
        assert priority_email.task_count == 0
        assert priority_email.project_name is None

    def test_with_context(self) -> None:
        """Test creating with context and associations."""
        email = EmailSummary(id=1, message_id="<msg123@example.com>")
        context = PriorityContext(email_id=1, overall_priority=0.8)
        priority_email = PriorityEmail(
            email=email,
            priority_rank=1,
            priority_score=0.8,
            context=context,
            task_count=2,
            project_name="Q4 Planning",
        )
        assert priority_email.context is not None
        assert priority_email.task_count == 2
        assert priority_email.project_name == "Q4 Planning"


class TestPriorityInboxResponse:
    """Tests for PriorityInboxResponse schema."""

    def test_pagination_fields(self) -> None:
        """Test pagination fields."""
        email = EmailSummary(id=1, message_id="<msg123@example.com>")
        priority_emails = [
            PriorityEmail(email=email, priority_rank=1, priority_score=0.9),
        ]
        response = PriorityInboxResponse(
            emails=priority_emails,
            total=100,
            limit=20,
            offset=0,
            has_more=True,
        )
        assert len(response.emails) == 1
        assert response.total == 100
        assert response.has_more is True

    def test_stats_fields(self) -> None:
        """Test stats fields."""
        response = PriorityInboxResponse(
            emails=[],
            total=0,
            limit=20,
            offset=0,
            has_more=False,
            pending_tasks=15,
            urgent_count=5,
            from_real_people_count=80,
        )
        assert response.pending_tasks == 15
        assert response.urgent_count == 5
        assert response.from_real_people_count == 80


class TestInboxStats:
    """Tests for InboxStats schema."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = InboxStats()
        assert stats.total_emails == 0
        assert stats.pending_tasks == 0
        assert stats.avg_priority_score is None

    def test_all_fields(self) -> None:
        """Test all fields."""
        stats = InboxStats(
            total_emails=1000,
            unread_count=50,
            pending_tasks=25,
            urgent_emails=10,
            from_real_people=700,
            avg_priority_score=0.65,
            oldest_unanswered_hours=48.5,
        )
        assert stats.total_emails == 1000
        assert stats.from_real_people == 700
        assert stats.oldest_unanswered_hours == 48.5
