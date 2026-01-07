"""Tests for OAuth base types."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rl_emails.auth.oauth import GoogleTokens, OAuthError


class TestOAuthError:
    """Tests for OAuthError exception."""

    def test_error_with_description(self) -> None:
        """OAuthError includes both error and description in message."""
        error = OAuthError("invalid_grant", "Token has expired")
        assert error.error == "invalid_grant"
        assert error.description == "Token has expired"
        assert str(error) == "invalid_grant: Token has expired"

    def test_error_without_description(self) -> None:
        """OAuthError works without description."""
        error = OAuthError("access_denied")
        assert error.error == "access_denied"
        assert error.description is None
        assert str(error) == "access_denied"

    def test_error_is_exception(self) -> None:
        """OAuthError can be raised and caught."""
        with pytest.raises(OAuthError) as exc_info:
            raise OAuthError("invalid_client", "Client not found")
        assert exc_info.value.error == "invalid_client"


class TestGoogleTokens:
    """Tests for GoogleTokens dataclass."""

    def test_create_tokens(self) -> None:
        """GoogleTokens can be created with all fields."""
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        tokens = GoogleTokens(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        assert tokens.access_token == "access-123"
        assert tokens.refresh_token == "refresh-456"
        assert tokens.expires_at == expires_at
        assert len(tokens.scopes) == 1
        assert "gmail.readonly" in tokens.scopes[0]

    def test_is_expired_false_for_future(self) -> None:
        """is_expired returns False for future expiration."""
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        tokens = GoogleTokens(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=[],
        )
        assert tokens.is_expired() is False

    def test_is_expired_true_for_past(self) -> None:
        """is_expired returns True for past expiration."""
        expires_at = datetime.now(UTC) - timedelta(hours=1)
        tokens = GoogleTokens(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=[],
        )
        assert tokens.is_expired() is True

    def test_is_expired_handles_naive_datetime(self) -> None:
        """is_expired handles naive datetime by assuming UTC."""
        # Naive datetime in the past
        expires_at = datetime.now() - timedelta(hours=1)
        tokens = GoogleTokens(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=[],
        )
        # Should still work - treats naive as UTC
        assert tokens.is_expired() is True

    def test_expires_in_seconds_positive(self) -> None:
        """expires_in_seconds returns positive for future expiration."""
        expires_at = datetime.now(UTC) + timedelta(minutes=30)
        tokens = GoogleTokens(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=[],
        )
        seconds = tokens.expires_in_seconds()
        # Should be around 1800 seconds (30 minutes)
        assert 1790 <= seconds <= 1810

    def test_expires_in_seconds_negative(self) -> None:
        """expires_in_seconds returns negative for past expiration."""
        expires_at = datetime.now(UTC) - timedelta(minutes=10)
        tokens = GoogleTokens(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=[],
        )
        seconds = tokens.expires_in_seconds()
        # Should be around -600 seconds (10 minutes ago)
        assert -610 <= seconds <= -590

    def test_expires_in_seconds_handles_naive_datetime(self) -> None:
        """expires_in_seconds handles naive datetime by assuming UTC."""
        # Naive datetime (no tzinfo) in the future relative to UTC
        # Use utcnow() to get naive UTC datetime
        expires_at = datetime.utcnow() + timedelta(minutes=15)  # noqa: DTZ003
        tokens = GoogleTokens(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=[],
        )
        seconds = tokens.expires_in_seconds()
        # Should be around 900 seconds (15 minutes)
        assert 890 <= seconds <= 910
