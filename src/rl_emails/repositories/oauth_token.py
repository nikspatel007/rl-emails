"""OAuth token repository for database operations."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.oauth_token import OAuthToken
from rl_emails.schemas.oauth_token import OAuthTokenCreate, OAuthTokenUpdate


class OAuthTokenRepository:
    """Repository for OAuth token database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def create(self, user_id: UUID, data: OAuthTokenCreate) -> OAuthToken:
        """Create a new OAuth token for a user.

        Args:
            user_id: User UUID.
            data: Token creation data.

        Returns:
            Created token.
        """
        token = OAuthToken(user_id=user_id, **data.model_dump())
        self.session.add(token)
        await self.session.commit()
        await self.session.refresh(token)
        return token

    async def get_by_user(self, user_id: UUID, provider: str = "google") -> OAuthToken | None:
        """Get token by user ID and provider.

        Args:
            user_id: User UUID.
            provider: OAuth provider (default: "google").

        Returns:
            Token if found, None otherwise.
        """
        result = await self.session.execute(
            select(OAuthToken).where(
                and_(OAuthToken.user_id == user_id, OAuthToken.provider == provider)
            )
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, token_id: UUID) -> OAuthToken | None:
        """Get token by ID.

        Args:
            token_id: Token UUID.

        Returns:
            Token if found, None otherwise.
        """
        result = await self.session.execute(select(OAuthToken).where(OAuthToken.id == token_id))
        return result.scalar_one_or_none()

    async def update(self, token_id: UUID, data: OAuthTokenUpdate) -> OAuthToken | None:
        """Update a token.

        Args:
            token_id: Token UUID.
            data: Update data.

        Returns:
            Updated token if found, None otherwise.
        """
        token = await self.get_by_id(token_id)
        if token is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(token, key, value)

        await self.session.commit()
        await self.session.refresh(token)
        return token

    async def update_tokens(
        self,
        user_id: UUID,
        access_token: str,
        refresh_token: str,
        expires_at: datetime,
        provider: str = "google",
    ) -> OAuthToken | None:
        """Update access and refresh tokens for a user.

        This is a convenience method for token refresh operations.

        Args:
            user_id: User UUID.
            access_token: New access token.
            refresh_token: New refresh token.
            expires_at: New expiration time.
            provider: OAuth provider (default: "google").

        Returns:
            Updated token if found, None otherwise.
        """
        token = await self.get_by_user(user_id, provider)
        if token is None:
            return None

        token.access_token = access_token
        token.refresh_token = refresh_token
        token.expires_at = expires_at

        await self.session.commit()
        await self.session.refresh(token)
        return token

    async def delete(self, user_id: UUID, provider: str = "google") -> bool:
        """Delete token for a user and provider.

        Args:
            user_id: User UUID.
            provider: OAuth provider (default: "google").

        Returns:
            True if deleted, False if not found.
        """
        token = await self.get_by_user(user_id, provider)
        if token is None:
            return False

        await self.session.delete(token)
        await self.session.commit()
        return True

    async def exists(self, user_id: UUID, provider: str = "google") -> bool:
        """Check if a token exists for user and provider.

        Args:
            user_id: User UUID.
            provider: OAuth provider (default: "google").

        Returns:
            True if token exists, False otherwise.
        """
        token = await self.get_by_user(user_id, provider)
        return token is not None
