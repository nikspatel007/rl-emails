"""Tests for project schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from rl_emails.schemas.project import (
    ProjectCreate,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)


class TestProjectResponse:
    """Tests for ProjectResponse schema."""

    def test_minimal_fields(self) -> None:
        """Test creating with minimal fields."""
        response = ProjectResponse(
            id=1,
            name="Test Project",
            created_at=datetime.now(UTC),
        )
        assert response.id == 1
        assert response.name == "Test Project"
        assert response.project_type is None
        assert response.is_active is True
        assert response.email_count == 0

    def test_all_fields(self) -> None:
        """Test creating with all fields."""
        user_id = uuid4()
        now = datetime.now(UTC)
        response = ProjectResponse(
            id=1,
            name="Test Project",
            project_type="conversation",
            owner_email="test@example.com",
            participants=["user1@example.com", "user2@example.com"],
            is_active=True,
            priority=2,
            email_count=10,
            last_activity=now,
            detected_from="real_person_thread",
            confidence=0.95,
            user_id=user_id,
            created_at=now,
        )
        assert response.project_type == "conversation"
        assert response.confidence == 0.95
        assert response.user_id == user_id


class TestProjectDetailResponse:
    """Tests for ProjectDetailResponse schema."""

    def test_includes_additional_fields(self) -> None:
        """Test that detail response includes additional fields."""
        now = datetime.now(UTC)
        response = ProjectDetailResponse(
            id=1,
            name="Test Project",
            created_at=now,
            description="A detailed description",
            keywords=["keyword1", "keyword2"],
            start_date=now,
            cluster_id=42,
            related_email_count=15,
        )
        assert response.description == "A detailed description"
        assert response.keywords == ["keyword1", "keyword2"]
        assert response.cluster_id == 42
        assert response.related_email_count == 15


class TestProjectListResponse:
    """Tests for ProjectListResponse schema."""

    def test_pagination_fields(self) -> None:
        """Test pagination fields."""
        now = datetime.now(UTC)
        projects = [
            ProjectResponse(id=1, name="Project 1", created_at=now),
            ProjectResponse(id=2, name="Project 2", created_at=now),
        ]
        response = ProjectListResponse(
            projects=projects,
            total=100,
            limit=20,
            offset=0,
            has_more=True,
        )
        assert len(response.projects) == 2
        assert response.total == 100
        assert response.has_more is True


class TestProjectCreate:
    """Tests for ProjectCreate schema."""

    def test_minimal_create(self) -> None:
        """Test creating with minimal fields."""
        data = ProjectCreate(name="New Project")
        assert data.name == "New Project"
        assert data.priority == 3  # Default

    def test_full_create(self) -> None:
        """Test creating with all fields."""
        data = ProjectCreate(
            name="Full Project",
            description="A full description",
            project_type="topic_cluster",
            priority=1,
            keywords=["ai", "ml"],
        )
        assert data.description == "A full description"
        assert data.priority == 1

    def test_name_validation(self) -> None:
        """Test name length validation."""
        with pytest.raises(ValueError):
            ProjectCreate(name="")  # Too short

    def test_priority_validation(self) -> None:
        """Test priority bounds validation."""
        with pytest.raises(ValueError):
            ProjectCreate(name="Test", priority=-1)  # Below minimum

        with pytest.raises(ValueError):
            ProjectCreate(name="Test", priority=6)  # Above maximum


class TestProjectUpdate:
    """Tests for ProjectUpdate schema."""

    def test_partial_update(self) -> None:
        """Test partial update with some fields."""
        data = ProjectUpdate(name="Updated Name")
        assert data.name == "Updated Name"
        assert data.priority is None  # Not set

    def test_full_update(self) -> None:
        """Test full update with all fields."""
        data = ProjectUpdate(
            name="Updated Name",
            description="Updated description",
            is_active=False,
            priority=5,
        )
        assert data.is_active is False
        assert data.priority == 5
