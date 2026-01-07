"""Tests for organization schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rl_emails.schemas.organization import (
    OrganizationCreate,
    OrganizationResponse,
    OrganizationUpdate,
)


class TestOrganizationCreate:
    """Tests for OrganizationCreate schema."""

    def test_valid_organization_create(self) -> None:
        """Test creating a valid organization."""
        data = OrganizationCreate(name="Acme Corp", slug="acme")
        assert data.name == "Acme Corp"
        assert data.slug == "acme"
        assert data.settings == {}

    def test_organization_create_with_settings(self) -> None:
        """Test creating organization with settings."""
        data = OrganizationCreate(
            name="Acme Corp",
            slug="acme",
            settings={"feature_x": True},
        )
        assert data.settings == {"feature_x": True}

    def test_organization_create_validates_name_required(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError) as exc_info:
            OrganizationCreate(slug="acme")  # type: ignore[call-arg]
        assert "name" in str(exc_info.value)

    def test_organization_create_validates_name_min_length(self) -> None:
        """Test that name must have at least 1 character."""
        with pytest.raises(ValidationError):
            OrganizationCreate(name="", slug="acme")

    def test_organization_create_validates_slug_required(self) -> None:
        """Test that slug is required."""
        with pytest.raises(ValidationError) as exc_info:
            OrganizationCreate(name="Acme")  # type: ignore[call-arg]
        assert "slug" in str(exc_info.value)

    def test_organization_create_validates_slug_format(self) -> None:
        """Test slug validation pattern."""
        # Valid slugs
        assert OrganizationCreate(name="Test", slug="a").slug == "a"
        assert OrganizationCreate(name="Test", slug="abc").slug == "abc"
        assert OrganizationCreate(name="Test", slug="abc-123").slug == "abc-123"
        assert OrganizationCreate(name="Test", slug="a-b-c").slug == "a-b-c"

        # Invalid slugs
        with pytest.raises(ValidationError):
            OrganizationCreate(name="Test", slug="")

        with pytest.raises(ValidationError):
            OrganizationCreate(name="Test", slug="-abc")

        with pytest.raises(ValidationError):
            OrganizationCreate(name="Test", slug="abc-")

        with pytest.raises(ValidationError):
            OrganizationCreate(name="Test", slug="ABC")

        with pytest.raises(ValidationError):
            OrganizationCreate(name="Test", slug="a_b")


class TestOrganizationUpdate:
    """Tests for OrganizationUpdate schema."""

    def test_organization_update_empty(self) -> None:
        """Test creating an empty update."""
        data = OrganizationUpdate()
        assert data.name is None
        assert data.settings is None

    def test_organization_update_name_only(self) -> None:
        """Test updating name only."""
        data = OrganizationUpdate(name="New Name")
        assert data.name == "New Name"
        assert data.settings is None

    def test_organization_update_settings_only(self) -> None:
        """Test updating settings only."""
        data = OrganizationUpdate(settings={"new": "value"})
        assert data.name is None
        assert data.settings == {"new": "value"}


class TestOrganizationResponse:
    """Tests for OrganizationResponse schema."""

    def test_organization_response_from_dict(self) -> None:
        """Test creating response from dict."""
        org_id = uuid.uuid4()
        now = datetime.now(UTC)
        data = OrganizationResponse(
            id=org_id,
            name="Acme Corp",
            slug="acme",
            settings={},
            created_at=now,
            updated_at=now,
        )
        assert data.id == org_id
        assert data.name == "Acme Corp"
        assert data.slug == "acme"

    def test_organization_response_from_attributes(self) -> None:
        """Test that from_attributes is enabled."""
        assert OrganizationResponse.model_config.get("from_attributes") is True
