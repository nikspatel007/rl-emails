"""Tests for task schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from rl_emails.schemas.task import (
    TaskCreate,
    TaskDetailResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdate,
)


class TestTaskResponse:
    """Tests for TaskResponse schema."""

    def test_minimal_fields(self) -> None:
        """Test creating with minimal fields."""
        response = TaskResponse(
            id=1,
            task_id="llm_123_task",
            description="Complete the report",
            created_at=datetime.now(UTC),
        )
        assert response.id == 1
        assert response.task_id == "llm_123_task"
        assert response.status == "pending"
        assert response.is_assigned_to_user is False

    def test_all_fields(self) -> None:
        """Test creating with all fields."""
        user_id = uuid4()
        now = datetime.now(UTC)
        response = TaskResponse(
            id=1,
            task_id="llm_123_task",
            email_id=100,
            project_id=5,
            description="Complete the report",
            task_type="send",
            complexity="medium",
            deadline=now,
            deadline_text="by Friday",
            urgency_score=0.8,
            status="in_progress",
            assigned_to="user@example.com",
            assigned_by="manager@example.com",
            is_assigned_to_user=True,
            extraction_method="llm_real_person",
            completed_at=None,
            created_at=now,
            user_id=user_id,
        )
        assert response.task_type == "send"
        assert response.urgency_score == 0.8
        assert response.is_assigned_to_user is True


class TestTaskDetailResponse:
    """Tests for TaskDetailResponse schema."""

    def test_includes_additional_fields(self) -> None:
        """Test that detail response includes additional fields."""
        now = datetime.now(UTC)
        response = TaskDetailResponse(
            id=1,
            task_id="llm_123_task",
            description="Complete the report",
            created_at=now,
            email_subject="Re: Project Update",
            email_from="sender@example.com",
            email_date=now,
            source_text="Please complete the report by Friday",
            project_name="Q4 Planning",
        )
        assert response.email_subject == "Re: Project Update"
        assert response.project_name == "Q4 Planning"


class TestTaskListResponse:
    """Tests for TaskListResponse schema."""

    def test_pagination_fields(self) -> None:
        """Test pagination fields."""
        now = datetime.now(UTC)
        tasks = [
            TaskResponse(id=1, task_id="task_1", description="Task 1", created_at=now),
            TaskResponse(id=2, task_id="task_2", description="Task 2", created_at=now),
        ]
        response = TaskListResponse(
            tasks=tasks,
            total=50,
            limit=20,
            offset=0,
            has_more=True,
        )
        assert len(response.tasks) == 2
        assert response.total == 50
        assert response.has_more is True


class TestTaskCreate:
    """Tests for TaskCreate schema."""

    def test_minimal_create(self) -> None:
        """Test creating with minimal fields."""
        data = TaskCreate(description="New task")
        assert data.description == "New task"
        assert data.task_type == "other"  # Default
        assert data.urgency_score == 0.5  # Default

    def test_full_create(self) -> None:
        """Test creating with all fields."""
        now = datetime.now(UTC)
        data = TaskCreate(
            email_id=100,
            project_id=5,
            description="Complete the analysis",
            task_type="research",
            complexity="substantial",
            deadline=now,
            urgency_score=0.9,
        )
        assert data.task_type == "research"
        assert data.complexity == "substantial"
        assert data.urgency_score == 0.9

    def test_description_validation(self) -> None:
        """Test description length validation."""
        with pytest.raises(ValueError):
            TaskCreate(description="")  # Too short

    def test_urgency_score_validation(self) -> None:
        """Test urgency score bounds validation."""
        with pytest.raises(ValueError):
            TaskCreate(description="Test", urgency_score=-0.1)  # Below minimum

        with pytest.raises(ValueError):
            TaskCreate(description="Test", urgency_score=1.1)  # Above maximum


class TestTaskUpdate:
    """Tests for TaskUpdate schema."""

    def test_partial_update(self) -> None:
        """Test partial update with some fields."""
        data = TaskUpdate(description="Updated description")
        assert data.description == "Updated description"
        assert data.status is None  # Not set

    def test_status_update(self) -> None:
        """Test status update."""
        data = TaskUpdate(status="completed")
        assert data.status == "completed"


class TestTaskStatusUpdate:
    """Tests for TaskStatusUpdate schema."""

    def test_valid_statuses(self) -> None:
        """Test all valid status values."""
        for status in ["pending", "in_progress", "completed", "dismissed"]:
            update = TaskStatusUpdate(status=status)  # type: ignore[arg-type]
            assert update.status == status

    def test_invalid_status(self) -> None:
        """Test invalid status value."""
        with pytest.raises(ValueError):
            TaskStatusUpdate(status="invalid")  # type: ignore[arg-type]
