"""Tests for Project model."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from rl_emails.models.project import Project


class TestProject:
    """Tests for Project model."""

    def test_create_minimal(self) -> None:
        """Test creating project with minimal fields."""
        project = Project(name="Test Project", is_active=True, email_count=0)
        assert project.name == "Test Project"
        assert project.is_active is True
        assert project.email_count == 0

    def test_create_full(self) -> None:
        """Test creating project with all fields."""
        user_id = uuid4()
        now = datetime.now(UTC)
        project = Project(
            name="Full Project",
            description="A complete project",
            project_type="conversation",
            owner_email="owner@example.com",
            participants=["user1@example.com"],
            is_active=True,
            priority=1,
            email_count=10,
            last_activity=now,
            detected_from="real_person_thread",
            confidence=0.95,
            user_id=user_id,
        )
        assert project.project_type == "conversation"
        assert project.confidence == 0.95
        assert project.user_id == user_id

    def test_is_completed_property(self) -> None:
        """Test is_completed property."""
        project = Project(name="Test")
        assert project.is_completed is False

        project.completed_at = datetime.now(UTC)
        assert project.is_completed is True

    def test_repr(self) -> None:
        """Test string representation."""
        project = Project(name="Test Project", project_type="conversation")
        repr_str = repr(project)
        assert "Test Project" in repr_str
        assert "conversation" in repr_str
