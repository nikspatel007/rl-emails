"""Tests for OAuth token repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from rl_emails.models.oauth_token import OAuthToken
from rl_emails.repositories.oauth_token import OAuthTokenRepository
from rl_emails.schemas.oauth_token import OAuthTokenCreate, OAuthTokenUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> OAuthTokenRepository:
    """Create repository with mock session."""
    return OAuthTokenRepository(mock_session)


class TestOAuthTokenRepository:
    """Tests for OAuthTokenRepository."""

    @pytest.mark.asyncio
    async def test_create_token(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating a token."""
        # Setup
        user_id = uuid.uuid4()
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        data = OAuthTokenCreate(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )

        # Execute
        result = await repository.create(user_id, data)

        # Verify
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        assert isinstance(result, OAuthToken)
        assert result.access_token == "access-123"
        assert result.user_id == user_id

    @pytest.mark.asyncio
    async def test_create_token_with_default_provider(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating a token uses default provider."""
        # Setup
        user_id = uuid.uuid4()
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        data = OAuthTokenCreate(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at=expires_at,
        )

        # Execute
        result = await repository.create(user_id, data)

        # Verify
        assert result.provider == "google"

    @pytest.mark.asyncio
    async def test_get_by_user_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting token by user ID when found."""
        # Setup
        user_id = uuid.uuid4()
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = token
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_user(user_id)

        # Verify
        assert result == token

    @pytest.mark.asyncio
    async def test_get_by_user_not_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting token by user ID when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_user(uuid.uuid4())

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_user_with_provider(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting token by user ID with specific provider."""
        # Setup
        user_id = uuid.uuid4()
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="microsoft",
            access_token="access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = token
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_user(user_id, provider="microsoft")

        # Verify
        assert result == token

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting token by ID when found."""
        # Setup
        token_id = uuid.uuid4()
        token = OAuthToken(
            id=token_id,
            user_id=uuid.uuid4(),
            provider="google",
            access_token="access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = token
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_id(token_id)

        # Verify
        assert result == token

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting token by ID when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_id(uuid.uuid4())

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_update_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating token when found."""
        # Setup
        token_id = uuid.uuid4()
        token = OAuthToken(
            id=token_id,
            user_id=uuid.uuid4(),
            provider="google",
            access_token="old-access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        data = OAuthTokenUpdate(access_token="new-access")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = token
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update(token_id, data)

        # Verify
        assert result is not None
        assert result.access_token == "new-access"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating token when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update(uuid.uuid4(), OAuthTokenUpdate(access_token="new"))

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_update_tokens_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating tokens when found."""
        # Setup
        user_id = uuid.uuid4()
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="old-access",
            refresh_token="old-refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = token
        mock_session.execute.return_value = mock_result

        new_expires = datetime.now(UTC) + timedelta(hours=2)

        # Execute
        result = await repository.update_tokens(
            user_id=user_id,
            access_token="new-access",
            refresh_token="new-refresh",
            expires_at=new_expires,
        )

        # Verify
        assert result is not None
        assert result.access_token == "new-access"
        assert result.refresh_token == "new-refresh"
        assert result.expires_at == new_expires
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_tokens_not_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating tokens when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update_tokens(
            user_id=uuid.uuid4(),
            access_token="new-access",
            refresh_token="new-refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting token when found."""
        # Setup
        user_id = uuid.uuid4()
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = token
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.delete(user_id)

        # Verify
        assert result is True
        mock_session.delete.assert_called_once_with(token)

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting token when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.delete(uuid.uuid4())

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test exists returns True when token found."""
        # Setup
        user_id = uuid.uuid4()
        token = OAuthToken(
            id=uuid.uuid4(),
            user_id=user_id,
            provider="google",
            access_token="access",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = token
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.exists(user_id)

        # Verify
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(
        self, repository: OAuthTokenRepository, mock_session: AsyncMock
    ) -> None:
        """Test exists returns False when token not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.exists(uuid.uuid4())

        # Verify
        assert result is False
