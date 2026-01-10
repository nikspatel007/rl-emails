"""Tests for cluster metadata Pydantic schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rl_emails.schemas.cluster_metadata import (
    ClusterLabelRequest,
    ClusterLabelResult,
    ClusterMetadataCreate,
    ClusterMetadataListResponse,
    ClusterMetadataResponse,
    ClusterMetadataUpdate,
    ClusterStatsResponse,
    ProjectClusterResponse,
    ProjectDetectionRequest,
    ProjectDetectionResponse,
)


class TestClusterMetadataCreate:
    """Tests for ClusterMetadataCreate schema."""

    def test_minimal_create(self) -> None:
        """Test creating with minimal fields."""
        schema = ClusterMetadataCreate(
            dimension="content",
            cluster_id=0,
        )
        assert schema.dimension == "content"
        assert schema.cluster_id == 0
        assert schema.size == 0
        assert schema.is_project is False

    def test_full_create(self) -> None:
        """Test creating with all fields."""
        user_id = uuid4()
        now = datetime.now(UTC)

        schema = ClusterMetadataCreate(
            dimension="content",
            cluster_id=5,
            user_id=user_id,
            size=100,
            auto_label="Test Label",
            pct_replied=75.5,
            avg_response_time_hours=4.2,
            coherence_score=0.85,
            is_project=True,
            project_status="active",
            last_activity_at=now,
        )

        assert schema.user_id == user_id
        assert schema.size == 100
        assert schema.auto_label == "Test Label"
        assert schema.is_project is True
        assert schema.project_status == "active"

    def test_invalid_dimension(self) -> None:
        """Test invalid dimension value."""
        with pytest.raises(ValidationError):
            ClusterMetadataCreate(
                dimension="invalid",  # type: ignore[arg-type]
                cluster_id=0,
            )

    def test_pct_replied_validation(self) -> None:
        """Test pct_replied must be 0-100."""
        with pytest.raises(ValidationError):
            ClusterMetadataCreate(
                dimension="content",
                cluster_id=0,
                pct_replied=150.0,
            )

    def test_coherence_score_validation(self) -> None:
        """Test coherence_score must be 0-1."""
        with pytest.raises(ValidationError):
            ClusterMetadataCreate(
                dimension="content",
                cluster_id=0,
                coherence_score=1.5,
            )


class TestClusterMetadataUpdate:
    """Tests for ClusterMetadataUpdate schema."""

    def test_partial_update(self) -> None:
        """Test partial update with some fields."""
        schema = ClusterMetadataUpdate(
            auto_label="New Label",
            is_project=True,
        )
        assert schema.auto_label == "New Label"
        assert schema.is_project is True
        assert schema.project_status is None

    def test_empty_update(self) -> None:
        """Test empty update is valid."""
        schema = ClusterMetadataUpdate()
        assert schema.auto_label is None
        assert schema.is_project is None


class TestClusterMetadataResponse:
    """Tests for ClusterMetadataResponse schema."""

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        now = datetime.now(UTC)
        data = {
            "id": 1,
            "dimension": "content",
            "cluster_id": 5,
            "size": 100,
            "auto_label": "Test",
            "is_project": True,
            "created_at": now,
            "updated_at": now,
        }
        schema = ClusterMetadataResponse(**data)
        assert schema.id == 1
        assert schema.dimension == "content"
        assert schema.auto_label == "Test"


class TestClusterMetadataListResponse:
    """Tests for ClusterMetadataListResponse schema."""

    def test_empty_list(self) -> None:
        """Test with empty list."""
        schema = ClusterMetadataListResponse(
            clusters=[],
            total=0,
        )
        assert len(schema.clusters) == 0
        assert schema.total == 0
        assert schema.dimension is None

    def test_with_dimension_filter(self) -> None:
        """Test with dimension filter."""
        schema = ClusterMetadataListResponse(
            clusters=[],
            total=0,
            dimension="content",
        )
        assert schema.dimension == "content"


class TestProjectClusterResponse:
    """Tests for ProjectClusterResponse schema."""

    def test_display_name_with_label(self) -> None:
        """Test display_name with auto_label."""
        now = datetime.now(UTC)
        schema = ProjectClusterResponse(
            id=1,
            cluster_id=5,
            auto_label="Q4 Review",
            size=50,
            last_activity_at=now,
        )
        assert schema.display_name == "Q4 Review"

    def test_display_name_without_label(self) -> None:
        """Test display_name without auto_label."""
        schema = ProjectClusterResponse(
            id=1,
            cluster_id=5,
            size=50,
        )
        assert schema.display_name == "Project #5"


class TestClusterLabelRequest:
    """Tests for ClusterLabelRequest schema."""

    def test_valid_request(self) -> None:
        """Test valid label request."""
        schema = ClusterLabelRequest(
            cluster_ids=[1, 2, 3],
            dimension="content",
        )
        assert len(schema.cluster_ids) == 3
        assert schema.force_relabel is False

    def test_empty_cluster_ids(self) -> None:
        """Test empty cluster_ids is invalid."""
        with pytest.raises(ValidationError):
            ClusterLabelRequest(
                cluster_ids=[],
                dimension="content",
            )

    def test_max_cluster_ids(self) -> None:
        """Test max 100 cluster_ids."""
        with pytest.raises(ValidationError):
            ClusterLabelRequest(
                cluster_ids=list(range(101)),
                dimension="content",
            )


class TestClusterLabelResult:
    """Tests for ClusterLabelResult schema."""

    def test_success_result(self) -> None:
        """Test successful label result."""
        schema = ClusterLabelResult(
            cluster_id=1,
            dimension="content",
            auto_label="Test Label",
            success=True,
        )
        assert schema.success is True
        assert schema.error is None

    def test_failure_result(self) -> None:
        """Test failed label result."""
        schema = ClusterLabelResult(
            cluster_id=1,
            dimension="content",
            auto_label=None,
            success=False,
            error="LLM unavailable",
        )
        assert schema.success is False
        assert schema.error == "LLM unavailable"


class TestProjectDetectionRequest:
    """Tests for ProjectDetectionRequest schema."""

    def test_defaults(self) -> None:
        """Test default values."""
        schema = ProjectDetectionRequest()
        assert schema.dimension == "content"
        assert schema.min_size == 5
        assert schema.min_engagement == 0.2
        assert schema.stale_days == 14

    def test_custom_values(self) -> None:
        """Test custom values."""
        schema = ProjectDetectionRequest(
            min_size=10,
            min_engagement=0.5,
            stale_days=7,
        )
        assert schema.min_size == 10
        assert schema.min_engagement == 0.5
        assert schema.stale_days == 7


class TestProjectDetectionResponse:
    """Tests for ProjectDetectionResponse schema."""

    def test_response(self) -> None:
        """Test detection response."""
        schema = ProjectDetectionResponse(
            projects_detected=10,
            active_projects=6,
            stale_projects=3,
            clusters_analyzed=50,
        )
        assert schema.projects_detected == 10
        assert schema.active_projects == 6


class TestClusterStatsResponse:
    """Tests for ClusterStatsResponse schema."""

    def test_stats_response(self) -> None:
        """Test stats response."""
        schema = ClusterStatsResponse(
            dimension="content",
            total_clusters=50,
            total_emails=1000,
            avg_cluster_size=20.0,
            largest_cluster_size=100,
            smallest_cluster_size=5,
            labeled_clusters=30,
            project_clusters=10,
            active_projects=6,
        )
        assert schema.dimension == "content"
        assert schema.total_clusters == 50
        assert schema.avg_cluster_size == 20.0
