"""Tests for Task model."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from rl_emails.models.task import Task


class TestTask:
    """Tests for Task model."""

    def test_create_minimal(self) -> None:
        """Test creating task with minimal fields."""
        task = Task(task_id="test_123", description="Complete the report", status="pending")
        assert task.task_id == "test_123"
        assert task.description == "Complete the report"
        assert task.status == "pending"

    def test_create_full(self) -> None:
        """Test creating task with all fields."""
        user_id = uuid4()
        now = datetime.now(UTC)
        task = Task(
            task_id="llm_123_task",
            email_id=100,
            project_id=5,
            description="Complete the analysis",
            task_type="research",
            complexity="substantial",
            deadline=now,
            deadline_text="by Friday",
            urgency_score=0.9,
            status="in_progress",
            assigned_to="user@example.com",
            assigned_by="manager@example.com",
            is_assigned_to_user=True,
            extraction_method="llm_real_person",
            user_id=user_id,
        )
        assert task.task_type == "research"
        assert task.urgency_score == 0.9
        assert task.is_assigned_to_user is True

    def test_is_pending_property(self) -> None:
        """Test is_pending property."""
        task = Task(task_id="test", description="Test", status="pending")
        assert task.is_pending is True

        task.status = "completed"
        assert task.is_pending is False

    def test_is_completed_property(self) -> None:
        """Test is_completed property."""
        task = Task(task_id="test", description="Test", status="pending")
        assert task.is_completed is False

        task.status = "completed"
        assert task.is_completed is True

    def test_is_dismissed_property(self) -> None:
        """Test is_dismissed property."""
        task = Task(task_id="test", description="Test", status="pending")
        assert task.is_dismissed is False

        task.status = "dismissed"
        assert task.is_dismissed is True

    def test_is_urgent_property(self) -> None:
        """Test is_urgent property."""
        task = Task(task_id="test", description="Test", status="pending")
        assert task.is_urgent is False  # No urgency score

        task.urgency_score = 0.5
        assert task.is_urgent is False  # Below threshold

        task.urgency_score = 0.8
        assert task.is_urgent is True  # Above threshold

    def test_repr(self) -> None:
        """Test string representation."""
        task = Task(task_id="test_123", description="Test", status="pending", urgency_score=0.8)
        repr_str = repr(task)
        assert "pending" in repr_str
        assert "0.8" in repr_str
