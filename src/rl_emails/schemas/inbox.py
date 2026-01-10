"""Inbox Pydantic schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EmailSummary(BaseModel):
    """Schema for email summary in inbox."""

    id: int
    message_id: str
    thread_id: str | None = None
    subject: str | None = None
    from_email: str | None = None
    from_name: str | None = None
    date_parsed: datetime | None = None
    body_preview: str | None = None
    is_sent: bool = False
    action: str | None = None
    has_attachments: bool = False
    labels: list[str] | None = None

    model_config = ConfigDict(from_attributes=True)


class PriorityContext(BaseModel):
    """Schema for priority context information."""

    email_id: int
    sender_email: str | None = None
    sender_importance: float | None = None
    sender_reply_rate: float | None = None
    thread_length: int | None = None
    is_business_hours: bool | None = None
    age_hours: float | None = None
    people_score: float | None = None
    temporal_score: float | None = None
    relationship_score: float | None = None
    overall_priority: float | None = None

    model_config = ConfigDict(from_attributes=True)


class PriorityEmail(BaseModel):
    """Schema for email with priority context."""

    email: EmailSummary
    priority_rank: int
    priority_score: float
    context: PriorityContext | None = None
    # Associated tasks for this email
    task_count: int = 0
    # Associated project
    project_name: str | None = None

    model_config = ConfigDict(from_attributes=True)


class PriorityInboxResponse(BaseModel):
    """Schema for priority inbox response."""

    emails: list[PriorityEmail]
    total: int
    limit: int
    offset: int
    has_more: bool
    # Summary stats
    pending_tasks: int = 0
    urgent_count: int = 0
    from_real_people_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class InboxStats(BaseModel):
    """Schema for inbox statistics."""

    total_emails: int = 0
    unread_count: int = 0
    pending_tasks: int = 0
    urgent_emails: int = 0
    from_real_people: int = 0
    avg_priority_score: float | None = None
    oldest_unanswered_hours: float | None = None

    model_config = ConfigDict(from_attributes=True)
