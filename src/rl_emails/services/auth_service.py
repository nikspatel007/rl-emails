"""Authentication service for OAuth flow orchestration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from rl_emails.auth.google import GoogleOAuth
from rl_emails.auth.oauth import OAuthError
from rl_emails.models.oauth_token import OAuthToken
from rl_emails.repositories.oauth_token import OAuthTokenRepository
from rl_emails.schemas.oauth_token import OAuthTokenCreate, OAuthTokenUpdate


class AuthServiceError(Exception):
    """Base exception for auth service errors."""


class TokenNotFoundError(AuthServiceError):
    """Raised when no token exists for user."""


class AuthService:
    """Service for managing authentication flows.

    This service orchestrates OAuth operations by combining the GoogleOAuth
    client (handles HTTP communication with Google) with the OAuthTokenRepository
    (handles database storage).

    Typical flow:
    1. start_auth_flow() - Get URL to redirect user to Google
    2. User authorizes on Google, gets redirected back with code
    3. complete_auth_flow() - Exchange code for tokens, store in DB
    4. get_valid_token() - Get token for API calls (auto-refreshes if expired)
    5. revoke_token() - Disconnect user's Gmail access
    """

    # Refresh token 5 minutes before expiry to avoid race conditions
    REFRESH_BUFFER_SECONDS = 300

    def __init__(
        self,
        oauth: GoogleOAuth,
        token_repo: OAuthTokenRepository,
    ) -> None:
        """Initialize auth service.

        Args:
            oauth: Google OAuth client for API communication.
            token_repo: Repository for token database operations.
        """
        self.oauth = oauth
        self.token_repo = token_repo

    def start_auth_flow(self, state: str | None = None) -> str:
        """Start OAuth flow and return authorization URL.

        The returned URL should be opened in a browser for the user to
        grant access to their Gmail account.

        Args:
            state: Optional state parameter for CSRF protection.
                   Typically includes user_id or session identifier.

        Returns:
            Authorization URL to redirect user to.
        """
        return self.oauth.get_authorization_url(state=state)

    async def complete_auth_flow(self, user_id: UUID, code: str) -> OAuthToken:
        """Complete OAuth flow with authorization code.

        After the user authorizes on Google, they're redirected back with
        an authorization code. This method:
        1. Exchanges the code for access/refresh tokens
        2. Stores the tokens in the database

        If a token already exists for the user, it is updated.

        Args:
            user_id: User UUID to associate tokens with.
            code: Authorization code from OAuth callback.

        Returns:
            The created or updated OAuthToken.

        Raises:
            OAuthError: If the code exchange fails.
        """
        # Exchange authorization code for tokens
        tokens = await self.oauth.exchange_code(code)

        # Check if token already exists for user
        existing = await self.token_repo.get_by_user(user_id)

        if existing:
            # Update existing token
            update_data = OAuthTokenUpdate(
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                expires_at=tokens.expires_at,
                scopes=tokens.scopes,
            )
            result = await self.token_repo.update(existing.id, update_data)
            if result is None:
                # Should not happen since we just fetched the token
                raise AuthServiceError("Failed to update existing token")
            return result
        else:
            # Create new token
            create_data = OAuthTokenCreate(
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                expires_at=tokens.expires_at,
                scopes=tokens.scopes,
            )
            return await self.token_repo.create(user_id, create_data)

    async def get_valid_token(self, user_id: UUID, provider: str = "google") -> str:
        """Get valid access token, refreshing if needed.

        This method ensures the returned access token is valid:
        1. Fetches the stored token from the database
        2. If expired or near expiry, refreshes it
        3. Updates the database with new token
        4. Returns the valid access token

        Args:
            user_id: User UUID.
            provider: OAuth provider (default: "google").

        Returns:
            Valid access token string.

        Raises:
            TokenNotFoundError: If no token exists for user.
            OAuthError: If token refresh fails.
        """
        token = await self.token_repo.get_by_user(user_id, provider)
        if token is None:
            raise TokenNotFoundError(f"No token found for user {user_id}")

        # Check if token needs refresh (expired or will expire soon)
        if self._needs_refresh(token):
            # Refresh the token
            new_tokens = await self.oauth.refresh_token(token.refresh_token)

            # Update in database
            updated = await self.token_repo.update_tokens(
                user_id=user_id,
                access_token=new_tokens.access_token,
                refresh_token=new_tokens.refresh_token,
                expires_at=new_tokens.expires_at,
                provider=provider,
            )
            if updated is None:
                raise AuthServiceError("Failed to update refreshed token")

            return new_tokens.access_token

        return token.access_token

    async def revoke_token(self, user_id: UUID, provider: str = "google") -> bool:
        """Revoke and delete user's tokens.

        This completely disconnects the user's Gmail access:
        1. Revokes the token with Google (invalidates consent)
        2. Deletes the token from the database

        Args:
            user_id: User UUID.
            provider: OAuth provider (default: "google").

        Returns:
            True if token was revoked and deleted, False if no token existed.

        Raises:
            OAuthError: If revocation fails (token still deleted locally).
        """
        token = await self.token_repo.get_by_user(user_id, provider)
        if token is None:
            return False

        try:
            # Revoke with Google - prefer refresh token as it revokes both
            await self.oauth.revoke_token(token.refresh_token)
        except OAuthError:
            # Token may already be revoked; continue to delete locally
            pass

        # Always delete from database
        await self.token_repo.delete(user_id, provider)
        return True

    async def get_token_status(
        self, user_id: UUID, provider: str = "google"
    ) -> dict[str, bool | str | datetime | None]:
        """Get status of user's OAuth token.

        Args:
            user_id: User UUID.
            provider: OAuth provider (default: "google").

        Returns:
            Dict with connection status information.
        """
        token = await self.token_repo.get_by_user(user_id, provider)

        if token is None:
            return {
                "connected": False,
                "provider": None,
                "expires_at": None,
                "is_expired": False,
            }

        return {
            "connected": True,
            "provider": token.provider,
            "expires_at": token.expires_at,
            "is_expired": token.is_expired,
        }

    def _needs_refresh(self, token: OAuthToken) -> bool:
        """Check if token needs to be refreshed.

        Refresh if:
        - Token is already expired
        - Token will expire within REFRESH_BUFFER_SECONDS

        Args:
            token: OAuth token to check.

        Returns:
            True if token should be refreshed.
        """
        now = datetime.now(UTC)
        # Get timezone-aware expiration
        expires_at = token.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)

        # Refresh if expired or will expire soon
        buffer = timedelta(seconds=self.REFRESH_BUFFER_SECONDS)
        return now >= (expires_at - buffer)

    async def has_valid_token(self, user_id: UUID, provider: str = "google") -> bool:
        """Check if user has a valid (non-expired) token.

        Args:
            user_id: User UUID.
            provider: OAuth provider (default: "google").

        Returns:
            True if user has a valid token.
        """
        token = await self.token_repo.get_by_user(user_id, provider)
        if token is None:
            return False
        return not self._needs_refresh(token)
