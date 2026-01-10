"""Task model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_emails.models.base import Base

if TYPE_CHECKING:
    from rl_emails.models.project import Project


class Task(Base):
    """Task model for action items extracted from emails."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    email_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("emails.id", ondelete="CASCADE"),
        nullable=True,
    )
    description: Mapped[str] = mapped_column(String, nullable=False)
    deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deadline_text: Mapped[str | None] = mapped_column(String, nullable=True)
    assignee_hint: Mapped[str | None] = mapped_column(String, nullable=True)
    complexity: Mapped[str | None] = mapped_column(String, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String, nullable=True)
    urgency_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_text: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Added in entity extraction migration
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("org_users.id", ondelete="CASCADE"),
        nullable=True,
    )
    project_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
    )
    deadline_type: Mapped[str | None] = mapped_column(String, nullable=True)
    assigned_to: Mapped[str | None] = mapped_column(String, nullable=True)
    assigned_by: Mapped[str | None] = mapped_column(String, nullable=True)
    is_assigned_to_user: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )
    assignment_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_method: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending", server_default="pending")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    project: Mapped[Project | None] = relationship(
        "Project",
        back_populates="tasks",
    )

    @property
    def is_pending(self) -> bool:
        """Check if task is pending."""
        return self.status == "pending"

    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == "completed"

    @property
    def is_dismissed(self) -> bool:
        """Check if task is dismissed."""
        return self.status == "dismissed"

    @property
    def is_urgent(self) -> bool:
        """Check if task is urgent (score > 0.7)."""
        return self.urgency_score is not None and self.urgency_score > 0.7

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Task(id={self.id}, status={self.status!r}, urgency={self.urgency_score})"
