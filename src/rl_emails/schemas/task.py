"""Task Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

TaskStatus = Literal["pending", "in_progress", "completed", "dismissed"]
TaskType = Literal[
    "review", "send", "schedule", "decision", "research", "create", "follow_up", "other"
]
TaskComplexity = Literal["trivial", "quick", "medium", "substantial", "unknown"]


class TaskResponse(BaseModel):
    """Schema for task response."""

    id: int
    task_id: str
    email_id: int | None = None
    project_id: int | None = None
    description: str
    task_type: str | None = None
    complexity: str | None = None
    deadline: datetime | None = None
    deadline_text: str | None = None
    urgency_score: float | None = None
    status: str = "pending"
    assigned_to: str | None = None
    assigned_by: str | None = None
    is_assigned_to_user: bool = False
    extraction_method: str | None = None
    completed_at: datetime | None = None
    created_at: datetime
    user_id: UUID | None = None

    model_config = ConfigDict(from_attributes=True)


class TaskDetailResponse(TaskResponse):
    """Schema for task detail response with email context."""

    email_subject: str | None = None
    email_from: str | None = None
    email_date: datetime | None = None
    source_text: str | None = None
    project_name: str | None = None


class TaskListResponse(BaseModel):
    """Schema for paginated task list response."""

    tasks: list[TaskResponse]
    total: int
    limit: int
    offset: int
    has_more: bool

    model_config = ConfigDict(from_attributes=True)


class TaskStatusUpdate(BaseModel):
    """Schema for updating task status."""

    status: TaskStatus


class TaskCreate(BaseModel):
    """Schema for creating a task."""

    email_id: int | None = None
    project_id: int | None = None
    description: str = Field(..., min_length=1, max_length=1000)
    task_type: TaskType = "other"
    complexity: TaskComplexity = "medium"
    deadline: datetime | None = None
    urgency_score: float = Field(default=0.5, ge=0.0, le=1.0)


class TaskUpdate(BaseModel):
    """Schema for updating a task."""

    description: str | None = Field(None, min_length=1, max_length=1000)
    task_type: TaskType | None = None
    complexity: TaskComplexity | None = None
    deadline: datetime | None = None
    urgency_score: float | None = Field(None, ge=0.0, le=1.0)
    status: TaskStatus | None = None
