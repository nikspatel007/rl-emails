"""Tests for OAuth token schemas."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from rl_emails.schemas.oauth_token import (
    OAuthTokenCreate,
    OAuthTokenResponse,
    OAuthTokenStatus,
    OAuthTokenUpdate,
)


class TestOAuthTokenCreate:
    """Tests for OAuthTokenCreate schema."""

    def test_create_with_required_fields(self) -> None:
        """Test creating schema with required fields only."""
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        data = OAuthTokenCreate(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
        )
        assert data.access_token == "access-123"
        assert data.refresh_token == "refresh-456"
        assert data.expires_at == expires_at
        assert data.scopes is None
        assert data.provider == "google"

    def test_create_with_all_fields(self) -> None:
        """Test creating schema with all fields."""
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        data = OAuthTokenCreate(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=["gmail.readonly", "gmail.send"],
            provider="microsoft",
        )
        assert data.access_token == "access-123"
        assert data.scopes == ["gmail.readonly", "gmail.send"]
        assert data.provider == "microsoft"


class TestOAuthTokenUpdate:
    """Tests for OAuthTokenUpdate schema."""

    def test_update_empty(self) -> None:
        """Test creating empty update schema."""
        data = OAuthTokenUpdate()
        assert data.access_token is None
        assert data.refresh_token is None
        assert data.expires_at is None
        assert data.scopes is None

    def test_update_partial(self) -> None:
        """Test creating partial update schema."""
        data = OAuthTokenUpdate(access_token="new-access")
        assert data.access_token == "new-access"
        assert data.refresh_token is None

    def test_update_all_fields(self) -> None:
        """Test creating update schema with all fields."""
        expires_at = datetime.now(UTC) + timedelta(hours=2)
        data = OAuthTokenUpdate(
            access_token="new-access",
            refresh_token="new-refresh",
            expires_at=expires_at,
            scopes=["new.scope"],
        )
        assert data.access_token == "new-access"
        assert data.refresh_token == "new-refresh"
        assert data.expires_at == expires_at
        assert data.scopes == ["new.scope"]


class TestOAuthTokenResponse:
    """Tests for OAuthTokenResponse schema."""

    def test_response_from_dict(self) -> None:
        """Test creating response from dictionary."""
        token_id = uuid.uuid4()
        user_id = uuid.uuid4()
        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=1)

        data = OAuthTokenResponse(
            id=token_id,
            user_id=user_id,
            provider="google",
            expires_at=expires_at,
            scopes=["gmail.readonly"],
            created_at=now,
            updated_at=now,
        )
        assert data.id == token_id
        assert data.user_id == user_id
        assert data.provider == "google"
        assert data.expires_at == expires_at
        assert data.scopes == ["gmail.readonly"]

    def test_response_with_none_scopes(self) -> None:
        """Test creating response with None scopes."""
        now = datetime.now(UTC)
        data = OAuthTokenResponse(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="google",
            expires_at=now + timedelta(hours=1),
            scopes=None,
            created_at=now,
            updated_at=now,
        )
        assert data.scopes is None


class TestOAuthTokenStatus:
    """Tests for OAuthTokenStatus schema."""

    def test_status_connected(self) -> None:
        """Test creating connected status."""
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        status = OAuthTokenStatus(
            connected=True,
            provider="google",
            expires_at=expires_at,
            is_expired=False,
        )
        assert status.connected is True
        assert status.provider == "google"
        assert status.expires_at == expires_at
        assert status.is_expired is False

    def test_status_disconnected(self) -> None:
        """Test creating disconnected status."""
        status = OAuthTokenStatus(connected=False)
        assert status.connected is False
        assert status.provider is None
        assert status.expires_at is None
        assert status.is_expired is False

    def test_status_expired(self) -> None:
        """Test creating expired status."""
        expires_at = datetime.now(UTC) - timedelta(hours=1)
        status = OAuthTokenStatus(
            connected=True,
            provider="google",
            expires_at=expires_at,
            is_expired=True,
        )
        assert status.connected is True
        assert status.is_expired is True
