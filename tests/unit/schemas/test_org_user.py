"""Tests for organization user schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rl_emails.schemas.org_user import (
    OrgUserCreate,
    OrgUserResponse,
    OrgUserUpdate,
)


class TestOrgUserCreate:
    """Tests for OrgUserCreate schema."""

    def test_valid_user_create(self) -> None:
        """Test creating a valid user."""
        data = OrgUserCreate(email="user@example.com")
        assert data.email == "user@example.com"
        assert data.name is None
        assert data.role == "member"

    def test_user_create_with_name(self) -> None:
        """Test creating user with name."""
        data = OrgUserCreate(email="user@example.com", name="John Doe")
        assert data.name == "John Doe"

    def test_user_create_admin_role(self) -> None:
        """Test creating user with admin role."""
        data = OrgUserCreate(email="user@example.com", role="admin")
        assert data.role == "admin"

    def test_user_create_validates_email(self) -> None:
        """Test that email is validated."""
        with pytest.raises(ValidationError):
            OrgUserCreate(email="not-an-email")

    def test_user_create_validates_role(self) -> None:
        """Test that role is validated."""
        with pytest.raises(ValidationError):
            OrgUserCreate(email="user@example.com", role="invalid")  # type: ignore[arg-type]


class TestOrgUserUpdate:
    """Tests for OrgUserUpdate schema."""

    def test_user_update_empty(self) -> None:
        """Test creating an empty update."""
        data = OrgUserUpdate()
        assert data.name is None
        assert data.role is None

    def test_user_update_name_only(self) -> None:
        """Test updating name only."""
        data = OrgUserUpdate(name="New Name")
        assert data.name == "New Name"
        assert data.role is None

    def test_user_update_role_only(self) -> None:
        """Test updating role only."""
        data = OrgUserUpdate(role="admin")
        assert data.name is None
        assert data.role == "admin"

    def test_user_update_validates_role(self) -> None:
        """Test that role is validated."""
        with pytest.raises(ValidationError):
            OrgUserUpdate(role="invalid")  # type: ignore[arg-type]


class TestOrgUserResponse:
    """Tests for OrgUserResponse schema."""

    def test_user_response_from_dict(self) -> None:
        """Test creating response from dict."""
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        now = datetime.now(UTC)
        data = OrgUserResponse(
            id=user_id,
            org_id=org_id,
            email="user@example.com",
            name="John Doe",
            role="admin",
            gmail_connected=True,
            created_at=now,
            updated_at=now,
        )
        assert data.id == user_id
        assert data.org_id == org_id
        assert data.email == "user@example.com"
        assert data.gmail_connected is True

    def test_user_response_from_attributes(self) -> None:
        """Test that from_attributes is enabled."""
        assert OrgUserResponse.model_config.get("from_attributes") is True
