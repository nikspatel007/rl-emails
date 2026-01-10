"""ClusterMetadata model for cluster auto-labeling and project detection."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from rl_emails.models.base import Base

# Type aliases for cluster dimensions and project status
ClusterDimension = Literal["people", "content", "behavior", "service", "temporal"]
ProjectStatus = Literal["active", "stale", "completed"]


class ClusterMetadata(Base):
    """Cluster metadata for auto-labeling and project detection.

    Stores computed statistics and labels for each cluster across
    all 5 clustering dimensions (people, content, behavior, service, temporal).
    Content clusters can be flagged as projects for priority boosting.
    """

    __tablename__ = "cluster_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Multi-tenant fields (nullable for single-tenant mode)
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # Cluster identification
    dimension: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Clustering dimension: people, content, behavior, service, temporal",
    )
    cluster_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Cluster ID within the dimension",
    )

    # Basic statistics
    size: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Number of emails in this cluster",
    )
    representative_email_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="ID of a representative email for this cluster",
    )

    # Auto-labeling
    auto_label: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        comment="LLM-generated or extracted label for this cluster",
    )

    # Engagement statistics
    pct_replied: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Percentage of emails in cluster that were replied to",
    )
    avg_response_time_hours: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Average response time in hours for emails in cluster",
    )
    avg_relationship_strength: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Average relationship strength for people clusters",
    )

    # Cluster quality metrics
    coherence_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Cluster coherence score (0-1, higher = more cohesive)",
    )
    participant_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of unique participants (senders/recipients) in cluster",
    )

    # Project detection (primarily for content clusters)
    is_project: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="Whether this cluster represents an active project/topic",
    )
    project_status: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        comment="Project status: active, stale, completed",
    )
    last_activity_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of most recent email in cluster",
    )

    # Temporal strength metrics
    temporal_strength: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Overall temporal strength score (0-1), combining recency, velocity, trend",
    )
    activity_velocity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Emails per day in the last 7 days",
    )
    activity_trend: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Trend direction: positive=increasing, negative=decreasing activity",
    )
    first_activity_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of first email in cluster",
    )
    emails_last_7d: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of emails in last 7 days",
    )
    emails_last_30d: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of emails in last 30 days",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    @property
    def is_content_cluster(self) -> bool:
        """Check if this is a content cluster (eligible for project detection)."""
        return self.dimension == "content"

    @property
    def is_active_project(self) -> bool:
        """Check if this is an active project cluster."""
        return self.is_project and self.project_status == "active"

    @property
    def engagement_rate(self) -> float:
        """Get engagement rate (alias for pct_replied / 100)."""
        if self.pct_replied is None:
            return 0.0
        return self.pct_replied / 100.0

    def __repr__(self) -> str:
        """Return string representation."""
        label = self.auto_label or "unlabeled"
        project_indicator = " [PROJECT]" if self.is_project else ""
        return (
            f"ClusterMetadata(id={self.id}, dimension={self.dimension!r}, "
            f"cluster_id={self.cluster_id}, size={self.size}, "
            f"label={label!r}{project_indicator})"
        )
