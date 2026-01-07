"""Tests for AuthService."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from rl_emails.auth.google import GoogleOAuth
from rl_emails.auth.oauth import GoogleTokens, OAuthError
from rl_emails.models.oauth_token import OAuthToken
from rl_emails.repositories.oauth_token import OAuthTokenRepository
from rl_emails.services.auth_service import (
    AuthService,
    AuthServiceError,
    TokenNotFoundError,
)


@pytest.fixture
def mock_oauth() -> MagicMock:
    """Create a mock GoogleOAuth client."""
    oauth = MagicMock(spec=GoogleOAuth)
    oauth.get_authorization_url = MagicMock(
        return_value="https://accounts.google.com/o/oauth2/v2/auth?client_id=test"
    )
    return oauth


@pytest.fixture
def mock_token_repo() -> AsyncMock:
    """Create a mock OAuthTokenRepository."""
    return AsyncMock(spec=OAuthTokenRepository)


@pytest.fixture
def auth_service(mock_oauth: MagicMock, mock_token_repo: AsyncMock) -> AuthService:
    """Create AuthService with mocked dependencies."""
    return AuthService(oauth=mock_oauth, token_repo=mock_token_repo)


@pytest.fixture
def sample_tokens() -> GoogleTokens:
    """Create sample GoogleTokens."""
    return GoogleTokens(
        access_token="access-123",
        refresh_token="refresh-456",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )


@pytest.fixture
def sample_db_token() -> OAuthToken:
    """Create sample OAuthToken from database."""
    token = OAuthToken(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        provider="google",
        access_token="access-789",
        refresh_token="refresh-012",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )
    return token


class TestStartAuthFlow:
    """Tests for start_auth_flow method."""

    def test_returns_authorization_url(
        self, auth_service: AuthService, mock_oauth: MagicMock
    ) -> None:
        """Test that start_auth_flow returns authorization URL."""
        url = auth_service.start_auth_flow()

        mock_oauth.get_authorization_url.assert_called_once_with(state=None)
        assert "accounts.google.com" in url

    def test_passes_state_parameter(self, auth_service: AuthService, mock_oauth: MagicMock) -> None:
        """Test that state parameter is passed through."""
        auth_service.start_auth_flow(state="csrf-token-123")

        mock_oauth.get_authorization_url.assert_called_once_with(state="csrf-token-123")


class TestCompleteAuthFlow:
    """Tests for complete_auth_flow method."""

    @pytest.mark.asyncio
    async def test_creates_new_token(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
        sample_tokens: GoogleTokens,
    ) -> None:
        """Test creating new token when none exists."""
        user_id = uuid.uuid4()
        code = "auth-code-123"

        mock_oauth.exchange_code = AsyncMock(return_value=sample_tokens)
        mock_token_repo.get_by_user.return_value = None
        mock_token_repo.create.return_value = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token=sample_tokens.access_token,
            refresh_token=sample_tokens.refresh_token,
            expires_at=sample_tokens.expires_at,
            scopes=sample_tokens.scopes,
        )

        result = await auth_service.complete_auth_flow(user_id, code)

        mock_oauth.exchange_code.assert_called_once_with(code)
        mock_token_repo.get_by_user.assert_called_once_with(user_id)
        mock_token_repo.create.assert_called_once()
        assert result.access_token == sample_tokens.access_token

    @pytest.mark.asyncio
    async def test_updates_existing_token(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
        sample_tokens: GoogleTokens,
        sample_db_token: OAuthToken,
    ) -> None:
        """Test updating existing token."""
        user_id = sample_db_token.user_id
        code = "auth-code-123"

        mock_oauth.exchange_code = AsyncMock(return_value=sample_tokens)
        mock_token_repo.get_by_user.return_value = sample_db_token
        mock_token_repo.update.return_value = sample_db_token

        result = await auth_service.complete_auth_flow(user_id, code)

        mock_oauth.exchange_code.assert_called_once_with(code)
        mock_token_repo.update.assert_called_once()
        assert result == sample_db_token

    @pytest.mark.asyncio
    async def test_raises_on_exchange_error(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
    ) -> None:
        """Test that OAuthError is raised on exchange failure."""
        mock_oauth.exchange_code = AsyncMock(
            side_effect=OAuthError("invalid_grant", "Code expired")
        )

        with pytest.raises(OAuthError) as exc_info:
            await auth_service.complete_auth_flow(uuid.uuid4(), "bad-code")

        assert exc_info.value.error == "invalid_grant"

    @pytest.mark.asyncio
    async def test_raises_on_update_failure(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
        sample_tokens: GoogleTokens,
        sample_db_token: OAuthToken,
    ) -> None:
        """Test that AuthServiceError is raised when update fails."""
        user_id = sample_db_token.user_id

        mock_oauth.exchange_code = AsyncMock(return_value=sample_tokens)
        mock_token_repo.get_by_user.return_value = sample_db_token
        mock_token_repo.update.return_value = None  # Update failed

        with pytest.raises(AuthServiceError) as exc_info:
            await auth_service.complete_auth_flow(user_id, "code")

        assert "Failed to update" in str(exc_info.value)


class TestGetValidToken:
    """Tests for get_valid_token method."""

    @pytest.mark.asyncio
    async def test_returns_token_when_valid(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
        sample_db_token: OAuthToken,
    ) -> None:
        """Test returning valid token without refresh."""
        mock_token_repo.get_by_user.return_value = sample_db_token

        result = await auth_service.get_valid_token(sample_db_token.user_id)

        assert result == sample_db_token.access_token
        mock_token_repo.update_tokens.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_no_token(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test raising TokenNotFoundError when no token exists."""
        mock_token_repo.get_by_user.return_value = None

        with pytest.raises(TokenNotFoundError):
            await auth_service.get_valid_token(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_refreshes_expired_token(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test refreshing expired token."""
        user_id = uuid.uuid4()
        expired_token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="old-access",
            refresh_token="refresh-token",
            expires_at=datetime.now(UTC) - timedelta(hours=1),  # Expired
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )

        new_tokens = GoogleTokens(
            access_token="new-access",
            refresh_token="new-refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )

        mock_token_repo.get_by_user.return_value = expired_token
        mock_oauth.refresh_token = AsyncMock(return_value=new_tokens)
        mock_token_repo.update_tokens.return_value = MagicMock(access_token=new_tokens.access_token)

        result = await auth_service.get_valid_token(user_id)

        mock_oauth.refresh_token.assert_called_once_with(expired_token.refresh_token)
        mock_token_repo.update_tokens.assert_called_once()
        assert result == new_tokens.access_token

    @pytest.mark.asyncio
    async def test_refreshes_token_near_expiry(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test refreshing token that will expire soon."""
        user_id = uuid.uuid4()
        # Token expires in 2 minutes (less than 5 minute buffer)
        near_expiry_token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="old-access",
            refresh_token="refresh-token",
            expires_at=datetime.now(UTC) + timedelta(minutes=2),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )

        new_tokens = GoogleTokens(
            access_token="new-access",
            refresh_token="new-refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )

        mock_token_repo.get_by_user.return_value = near_expiry_token
        mock_oauth.refresh_token = AsyncMock(return_value=new_tokens)
        mock_token_repo.update_tokens.return_value = MagicMock(access_token=new_tokens.access_token)

        result = await auth_service.get_valid_token(user_id)

        mock_oauth.refresh_token.assert_called_once()
        assert result == new_tokens.access_token

    @pytest.mark.asyncio
    async def test_raises_on_refresh_failure(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test raising OAuthError when refresh fails."""
        user_id = uuid.uuid4()
        expired_token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="old-access",
            refresh_token="invalid-refresh",
            expires_at=datetime.now(UTC) - timedelta(hours=1),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )

        mock_token_repo.get_by_user.return_value = expired_token
        mock_oauth.refresh_token = AsyncMock(
            side_effect=OAuthError("invalid_grant", "Token revoked")
        )

        with pytest.raises(OAuthError) as exc_info:
            await auth_service.get_valid_token(user_id)

        assert exc_info.value.error == "invalid_grant"

    @pytest.mark.asyncio
    async def test_raises_on_update_failure(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test raising AuthServiceError when update fails after refresh."""
        user_id = uuid.uuid4()
        expired_token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="old-access",
            refresh_token="refresh-token",
            expires_at=datetime.now(UTC) - timedelta(hours=1),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )

        new_tokens = GoogleTokens(
            access_token="new-access",
            refresh_token="new-refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )

        mock_token_repo.get_by_user.return_value = expired_token
        mock_oauth.refresh_token = AsyncMock(return_value=new_tokens)
        mock_token_repo.update_tokens.return_value = None  # Update failed

        with pytest.raises(AuthServiceError) as exc_info:
            await auth_service.get_valid_token(user_id)

        assert "Failed to update" in str(exc_info.value)


class TestRevokeToken:
    """Tests for revoke_token method."""

    @pytest.mark.asyncio
    async def test_revokes_and_deletes_token(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
        sample_db_token: OAuthToken,
    ) -> None:
        """Test successful token revocation and deletion."""
        mock_token_repo.get_by_user.return_value = sample_db_token
        mock_oauth.revoke_token = AsyncMock()
        mock_token_repo.delete.return_value = True

        result = await auth_service.revoke_token(sample_db_token.user_id)

        assert result is True
        mock_oauth.revoke_token.assert_called_once_with(sample_db_token.refresh_token)
        mock_token_repo.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_no_token(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test returning False when no token exists."""
        mock_token_repo.get_by_user.return_value = None

        result = await auth_service.revoke_token(uuid.uuid4())

        assert result is False
        mock_token_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_deletes_even_on_revoke_error(
        self,
        auth_service: AuthService,
        mock_oauth: MagicMock,
        mock_token_repo: AsyncMock,
        sample_db_token: OAuthToken,
    ) -> None:
        """Test that token is deleted even if revocation fails."""
        mock_token_repo.get_by_user.return_value = sample_db_token
        mock_oauth.revoke_token = AsyncMock(
            side_effect=OAuthError("revocation_failed", "Already revoked")
        )
        mock_token_repo.delete.return_value = True

        result = await auth_service.revoke_token(sample_db_token.user_id)

        assert result is True
        mock_token_repo.delete.assert_called_once()


class TestGetTokenStatus:
    """Tests for get_token_status method."""

    @pytest.mark.asyncio
    async def test_returns_connected_status(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
        sample_db_token: OAuthToken,
    ) -> None:
        """Test returning connected status for existing token."""
        mock_token_repo.get_by_user.return_value = sample_db_token

        status = await auth_service.get_token_status(sample_db_token.user_id)

        assert status["connected"] is True
        assert status["provider"] == "google"
        assert status["expires_at"] == sample_db_token.expires_at
        assert status["is_expired"] is False

    @pytest.mark.asyncio
    async def test_returns_disconnected_status(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test returning disconnected status when no token."""
        mock_token_repo.get_by_user.return_value = None

        status = await auth_service.get_token_status(uuid.uuid4())

        assert status["connected"] is False
        assert status["provider"] is None
        assert status["expires_at"] is None
        assert status["is_expired"] is False


class TestHasValidToken:
    """Tests for has_valid_token method."""

    @pytest.mark.asyncio
    async def test_returns_true_for_valid_token(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
        sample_db_token: OAuthToken,
    ) -> None:
        """Test returning True when user has valid token."""
        mock_token_repo.get_by_user.return_value = sample_db_token

        result = await auth_service.has_valid_token(sample_db_token.user_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_token(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test returning False when no token exists."""
        mock_token_repo.get_by_user.return_value = None

        result = await auth_service.has_valid_token(uuid.uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_expired_token(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test returning False when token is expired."""
        user_id = uuid.uuid4()
        expired_token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="old-access",
            refresh_token="refresh-token",
            expires_at=datetime.now(UTC) - timedelta(hours=1),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        mock_token_repo.get_by_user.return_value = expired_token

        result = await auth_service.has_valid_token(user_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_token_near_expiry(
        self,
        auth_service: AuthService,
        mock_token_repo: AsyncMock,
    ) -> None:
        """Test returning False when token will expire within buffer."""
        user_id = uuid.uuid4()
        # Token expires in 2 minutes (less than 5 minute buffer)
        near_expiry_token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="access-token",
            refresh_token="refresh-token",
            expires_at=datetime.now(UTC) + timedelta(minutes=2),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        mock_token_repo.get_by_user.return_value = near_expiry_token

        result = await auth_service.has_valid_token(user_id)

        assert result is False


class TestNeedsRefresh:
    """Tests for _needs_refresh internal method."""

    def test_needs_refresh_for_expired_token(self, auth_service: AuthService) -> None:
        """Test that expired token needs refresh."""
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="google",
            access_token="access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) - timedelta(hours=1),
            scopes=None,
        )

        assert auth_service._needs_refresh(token) is True

    def test_needs_refresh_for_token_within_buffer(self, auth_service: AuthService) -> None:
        """Test that token within refresh buffer needs refresh."""
        # 2 minutes is less than 5 minute buffer
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="google",
            access_token="access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(minutes=2),
            scopes=None,
        )

        assert auth_service._needs_refresh(token) is True

    def test_does_not_need_refresh_for_valid_token(self, auth_service: AuthService) -> None:
        """Test that valid token with time remaining doesn't need refresh."""
        # 1 hour is more than 5 minute buffer
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="google",
            access_token="access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            scopes=None,
        )

        assert auth_service._needs_refresh(token) is False

    def test_handles_naive_datetime(self, auth_service: AuthService) -> None:
        """Test handling of naive datetime (no timezone)."""
        from datetime import datetime as dt_mod

        # Create token with naive datetime
        naive_expires = dt_mod.utcnow() + timedelta(hours=1)  # noqa: DTZ003
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            provider="google",
            access_token="access",
            refresh_token="refresh",
            expires_at=naive_expires,
            scopes=None,
        )

        # Should handle without error
        assert auth_service._needs_refresh(token) is False
