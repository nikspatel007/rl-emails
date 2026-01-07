"""Organization user repository for database operations."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.org_user import OrgUser
from rl_emails.schemas.org_user import OrgUserCreate, OrgUserUpdate


class OrgUserRepository:
    """Repository for organization user database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def create(self, org_id: UUID, data: OrgUserCreate) -> OrgUser:
        """Create a new organization user.

        Args:
            org_id: Organization UUID.
            data: User creation data.

        Returns:
            Created user.
        """
        user = OrgUser(org_id=org_id, **data.model_dump())
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def get_by_id(self, user_id: UUID) -> OrgUser | None:
        """Get user by ID.

        Args:
            user_id: User UUID.

        Returns:
            User if found, None otherwise.
        """
        result = await self.session.execute(select(OrgUser).where(OrgUser.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, org_id: UUID, email: str) -> OrgUser | None:
        """Get user by email within an organization.

        Args:
            org_id: Organization UUID.
            email: User email.

        Returns:
            User if found, None otherwise.
        """
        result = await self.session.execute(
            select(OrgUser).where(and_(OrgUser.org_id == org_id, OrgUser.email == email))
        )
        return result.scalar_one_or_none()

    async def list_by_org(self, org_id: UUID, limit: int = 100, offset: int = 0) -> list[OrgUser]:
        """List all users in an organization.

        Args:
            org_id: Organization UUID.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of users.
        """
        result = await self.session.execute(
            select(OrgUser)
            .where(OrgUser.org_id == org_id)
            .order_by(OrgUser.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update(self, user_id: UUID, data: OrgUserUpdate) -> OrgUser | None:
        """Update a user.

        Args:
            user_id: User UUID.
            data: Update data.

        Returns:
            Updated user if found, None otherwise.
        """
        user = await self.get_by_id(user_id)
        if user is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(user, key, value)

        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def set_gmail_connected(self, user_id: UUID, connected: bool) -> OrgUser | None:
        """Set Gmail connection status for a user.

        Args:
            user_id: User UUID.
            connected: Whether Gmail is connected.

        Returns:
            Updated user if found, None otherwise.
        """
        user = await self.get_by_id(user_id)
        if user is None:
            return None

        user.gmail_connected = connected
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def delete(self, user_id: UUID) -> bool:
        """Delete a user.

        Args:
            user_id: User UUID.

        Returns:
            True if deleted, False if not found.
        """
        user = await self.get_by_id(user_id)
        if user is None:
            return False

        await self.session.delete(user)
        await self.session.commit()
        return True
