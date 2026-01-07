"""OAuth2 base types and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


class OAuthError(Exception):
    """Base exception for OAuth errors."""

    def __init__(self, error: str, description: str | None = None) -> None:
        """Initialize OAuth error.

        Args:
            error: Error code from OAuth provider.
            description: Human-readable error description.
        """
        self.error = error
        self.description = description
        message = f"{error}: {description}" if description else error
        super().__init__(message)


@dataclass
class GoogleTokens:
    """OAuth tokens from Google.

    Attributes:
        access_token: Short-lived token for API access.
        refresh_token: Long-lived token for obtaining new access tokens.
        expires_at: When the access token expires.
        scopes: List of OAuth scopes granted.
    """

    access_token: str
    refresh_token: str
    expires_at: datetime
    scopes: list[str]

    def is_expired(self) -> bool:
        """Check if access token is expired.

        Returns:
            True if the access token has expired, False otherwise.
        """
        now = datetime.now(UTC)
        # Ensure expires_at has timezone info
        expires_at = self.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return now >= expires_at

    def expires_in_seconds(self) -> int:
        """Get seconds until token expires.

        Returns:
            Seconds until expiration (negative if already expired).
        """
        now = datetime.now(UTC)
        expires_at = self.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        delta = expires_at - now
        return int(delta.total_seconds())
