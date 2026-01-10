"""Pydantic schemas for cluster metadata API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Type aliases
ClusterDimension = Literal["people", "content", "behavior", "service", "temporal"]
ProjectStatus = Literal["active", "stale", "completed"]


class ClusterMetadataBase(BaseModel):
    """Base schema for cluster metadata."""

    dimension: ClusterDimension = Field(
        ...,
        description="Clustering dimension",
    )
    cluster_id: int = Field(
        ...,
        description="Cluster ID within the dimension",
    )


class ClusterMetadataCreate(ClusterMetadataBase):
    """Schema for creating cluster metadata."""

    user_id: UUID | None = Field(
        default=None,
        description="User ID for multi-tenant mode",
    )
    org_id: UUID | None = Field(
        default=None,
        description="Organization ID for multi-tenant mode",
    )
    size: int = Field(
        default=0,
        ge=0,
        description="Number of emails in cluster",
    )
    representative_email_id: int | None = Field(
        default=None,
        description="ID of representative email",
    )
    auto_label: str | None = Field(
        default=None,
        max_length=500,
        description="Auto-generated label for cluster",
    )
    pct_replied: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of emails replied to",
    )
    avg_response_time_hours: float | None = Field(
        default=None,
        ge=0,
        description="Average response time in hours",
    )
    avg_relationship_strength: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Average relationship strength (0-1)",
    )
    coherence_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Cluster coherence score (0-1)",
    )
    participant_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of unique participants",
    )
    is_project: bool = Field(
        default=False,
        description="Whether cluster represents a project",
    )
    project_status: ProjectStatus | None = Field(
        default=None,
        description="Project status if is_project is True",
    )
    last_activity_at: datetime | None = Field(
        default=None,
        description="Timestamp of most recent email",
    )


class ClusterMetadataUpdate(BaseModel):
    """Schema for updating cluster metadata."""

    auto_label: str | None = Field(
        default=None,
        max_length=500,
        description="Auto-generated label for cluster",
    )
    is_project: bool | None = Field(
        default=None,
        description="Whether cluster represents a project",
    )
    project_status: ProjectStatus | None = Field(
        default=None,
        description="Project status",
    )
    coherence_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Cluster coherence score",
    )


class ClusterMetadataResponse(ClusterMetadataBase):
    """Schema for cluster metadata API response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: UUID | None = None
    org_id: UUID | None = None
    size: int
    representative_email_id: int | None = None
    auto_label: str | None = None
    pct_replied: float | None = None
    avg_response_time_hours: float | None = None
    avg_relationship_strength: float | None = None
    coherence_score: float | None = None
    participant_count: int | None = None
    is_project: bool = False
    project_status: ProjectStatus | None = None
    last_activity_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ClusterMetadataListResponse(BaseModel):
    """Schema for list of cluster metadata."""

    clusters: list[ClusterMetadataResponse]
    total: int = Field(..., description="Total number of clusters")
    dimension: ClusterDimension | None = Field(
        default=None,
        description="Dimension filter applied",
    )


class ProjectClusterResponse(BaseModel):
    """Schema for project cluster summary."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    cluster_id: int
    auto_label: str | None = None
    size: int
    pct_replied: float | None = None
    participant_count: int | None = None
    project_status: ProjectStatus | None = None
    last_activity_at: datetime | None = None

    @property
    def display_name(self) -> str:
        """Get display name for the project."""
        return self.auto_label or f"Project #{self.cluster_id}"


class ProjectListResponse(BaseModel):
    """Schema for list of detected projects."""

    projects: list[ProjectClusterResponse]
    total: int = Field(..., description="Total number of projects")
    active_count: int = Field(..., description="Number of active projects")
    stale_count: int = Field(..., description="Number of stale projects")


class ClusterLabelRequest(BaseModel):
    """Request schema for labeling a cluster."""

    cluster_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of cluster IDs to label",
    )
    dimension: ClusterDimension = Field(
        ...,
        description="Dimension of clusters to label",
    )
    force_relabel: bool = Field(
        default=False,
        description="Whether to relabel already-labeled clusters",
    )


class ClusterLabelResult(BaseModel):
    """Result of cluster labeling operation."""

    cluster_id: int
    dimension: ClusterDimension
    auto_label: str | None
    success: bool
    error: str | None = None


class ClusterLabelResponse(BaseModel):
    """Response schema for cluster labeling."""

    results: list[ClusterLabelResult]
    total_labeled: int
    total_failed: int


class ProjectDetectionRequest(BaseModel):
    """Request schema for project detection."""

    dimension: ClusterDimension = Field(
        default="content",
        description="Dimension to analyze for projects (usually content)",
    )
    min_size: int = Field(
        default=5,
        ge=1,
        description="Minimum cluster size to consider as project",
    )
    min_engagement: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Minimum engagement rate (pct_replied/100) for projects",
    )
    stale_days: int = Field(
        default=14,
        ge=1,
        description="Days without activity before project is marked stale",
    )


class ProjectDetectionResponse(BaseModel):
    """Response schema for project detection."""

    projects_detected: int
    active_projects: int
    stale_projects: int
    clusters_analyzed: int


class ClusterStatsResponse(BaseModel):
    """Response schema for cluster statistics."""

    dimension: ClusterDimension
    total_clusters: int
    total_emails: int
    avg_cluster_size: float
    largest_cluster_size: int
    smallest_cluster_size: int
    labeled_clusters: int
    project_clusters: int
    active_projects: int
