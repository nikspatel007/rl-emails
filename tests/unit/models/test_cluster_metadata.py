"""Tests for ClusterMetadata model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from rl_emails.models.cluster_metadata import ClusterMetadata


class TestClusterMetadata:
    """Tests for ClusterMetadata model."""

    def test_create_minimal(self) -> None:
        """Test creating cluster metadata with minimal fields."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=0,
            size=10,
            is_project=False,  # Explicit since no DB to apply default
        )
        assert metadata.dimension == "content"
        assert metadata.cluster_id == 0
        assert metadata.size == 10
        assert metadata.is_project is False
        assert metadata.auto_label is None

    def test_create_full(self) -> None:
        """Test creating cluster metadata with all fields."""
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        now = datetime.now(UTC)

        metadata = ClusterMetadata(
            org_id=org_id,
            user_id=user_id,
            dimension="content",
            cluster_id=5,
            size=100,
            representative_email_id=42,
            auto_label="Q4 Budget Review",
            pct_replied=75.5,
            avg_response_time_hours=4.2,
            avg_relationship_strength=0.85,
            coherence_score=0.92,
            participant_count=8,
            is_project=True,
            project_status="active",
            last_activity_at=now,
        )

        assert metadata.org_id == org_id
        assert metadata.user_id == user_id
        assert metadata.dimension == "content"
        assert metadata.cluster_id == 5
        assert metadata.size == 100
        assert metadata.representative_email_id == 42
        assert metadata.auto_label == "Q4 Budget Review"
        assert metadata.pct_replied == 75.5
        assert metadata.avg_response_time_hours == 4.2
        assert metadata.avg_relationship_strength == 0.85
        assert metadata.coherence_score == 0.92
        assert metadata.participant_count == 8
        assert metadata.is_project is True
        assert metadata.project_status == "active"
        assert metadata.last_activity_at == now

    def test_is_content_cluster_property(self) -> None:
        """Test is_content_cluster property."""
        content = ClusterMetadata(dimension="content", cluster_id=0, size=0)
        people = ClusterMetadata(dimension="people", cluster_id=0, size=0)

        assert content.is_content_cluster is True
        assert people.is_content_cluster is False

    def test_is_active_project_property(self) -> None:
        """Test is_active_project property."""
        # Active project
        active = ClusterMetadata(
            dimension="content",
            cluster_id=0,
            size=10,
            is_project=True,
            project_status="active",
        )
        assert active.is_active_project is True

        # Stale project
        stale = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=10,
            is_project=True,
            project_status="stale",
        )
        assert stale.is_active_project is False

        # Not a project
        not_project = ClusterMetadata(
            dimension="content",
            cluster_id=2,
            size=10,
            is_project=False,
        )
        assert not_project.is_active_project is False

    def test_engagement_rate_property(self) -> None:
        """Test engagement_rate property."""
        # With pct_replied
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=0,
            size=10,
            pct_replied=50.0,
        )
        assert metadata.engagement_rate == 0.5

        # Without pct_replied
        metadata_no_reply = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=10,
        )
        assert metadata_no_reply.engagement_rate == 0.0

    def test_repr(self) -> None:
        """Test string representation."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=5,
            size=100,
            auto_label="Test Project",
            is_project=True,
        )
        repr_str = repr(metadata)
        assert "content" in repr_str
        assert "cluster_id=5" in repr_str
        assert "size=100" in repr_str
        assert "Test Project" in repr_str
        assert "[PROJECT]" in repr_str

    def test_repr_unlabeled(self) -> None:
        """Test string representation for unlabeled cluster."""
        metadata = ClusterMetadata(
            dimension="people",
            cluster_id=3,
            size=50,
        )
        repr_str = repr(metadata)
        assert "unlabeled" in repr_str
        assert "[PROJECT]" not in repr_str


class TestClusterMetadataTablename:
    """Tests for ClusterMetadata table configuration."""

    def test_tablename(self) -> None:
        """Test that tablename is correct."""
        assert ClusterMetadata.__tablename__ == "cluster_metadata"
