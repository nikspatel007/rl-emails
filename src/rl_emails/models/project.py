"""Project model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_emails.models.base import Base

if TYPE_CHECKING:
    from rl_emails.models.task import Task


class Project(Base):
    """Project model for email-based project tracking."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    keywords: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    related_domains: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    priority: Mapped[int | None] = mapped_column(Integer, default=3, server_default="3")
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
    project_type: Mapped[str | None] = mapped_column(String, nullable=True)
    owner_email: Mapped[str | None] = mapped_column(String, nullable=True)
    participants: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    due_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    email_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_activity: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    detected_from: Mapped[str | None] = mapped_column(String, nullable=True)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    tasks: Mapped[list[Task]] = relationship(
        "Task",
        back_populates="project",
        lazy="selectin",
    )

    @property
    def is_completed(self) -> bool:
        """Check if project is completed."""
        return self.completed_at is not None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Project(id={self.id}, name={self.name!r}, type={self.project_type!r})"
