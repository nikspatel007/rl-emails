"""Project Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ProjectResponse(BaseModel):
    """Schema for project response."""

    id: int
    name: str
    project_type: str | None = None
    owner_email: str | None = None
    participants: list[str] | None = None
    is_active: bool = True
    priority: int | None = None
    email_count: int = 0
    last_activity: datetime | None = None
    detected_from: str | None = None
    confidence: float | None = None
    user_id: UUID | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProjectDetailResponse(ProjectResponse):
    """Schema for project detail response with additional context."""

    description: str | None = None
    keywords: list[str] | None = None
    start_date: datetime | None = None
    due_date: datetime | None = None
    completed_at: datetime | None = None
    cluster_id: int | None = None
    related_email_count: int = 0


class ProjectListResponse(BaseModel):
    """Schema for paginated project list response."""

    projects: list[ProjectResponse]
    total: int
    limit: int
    offset: int
    has_more: bool

    model_config = ConfigDict(from_attributes=True)


class ProjectCreate(BaseModel):
    """Schema for creating a project."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    project_type: str | None = None
    priority: int = Field(default=3, ge=0, le=5)
    keywords: list[str] | None = None


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""

    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    project_type: str | None = None
    is_active: bool | None = None
    priority: int | None = Field(None, ge=0, le=5)
    keywords: list[str] | None = None
