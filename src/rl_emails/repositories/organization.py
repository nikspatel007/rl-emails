"""Organization repository for database operations."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.organization import Organization
from rl_emails.schemas.organization import OrganizationCreate, OrganizationUpdate


class OrganizationRepository:
    """Repository for organization database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def create(self, data: OrganizationCreate) -> Organization:
        """Create a new organization.

        Args:
            data: Organization creation data.

        Returns:
            Created organization.
        """
        org = Organization(**data.model_dump())
        self.session.add(org)
        await self.session.commit()
        await self.session.refresh(org)
        return org

    async def get_by_id(self, org_id: UUID) -> Organization | None:
        """Get organization by ID.

        Args:
            org_id: Organization UUID.

        Returns:
            Organization if found, None otherwise.
        """
        result = await self.session.execute(select(Organization).where(Organization.id == org_id))
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Organization | None:
        """Get organization by slug.

        Args:
            slug: Organization slug.

        Returns:
            Organization if found, None otherwise.
        """
        result = await self.session.execute(select(Organization).where(Organization.slug == slug))
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Organization]:
        """List all organizations.

        Args:
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of organizations.
        """
        result = await self.session.execute(
            select(Organization)
            .order_by(Organization.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update(self, org_id: UUID, data: OrganizationUpdate) -> Organization | None:
        """Update an organization.

        Args:
            org_id: Organization UUID.
            data: Update data.

        Returns:
            Updated organization if found, None otherwise.
        """
        org = await self.get_by_id(org_id)
        if org is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(org, key, value)

        await self.session.commit()
        await self.session.refresh(org)
        return org

    async def delete(self, org_id: UUID) -> bool:
        """Delete an organization.

        Args:
            org_id: Organization UUID.

        Returns:
            True if deleted, False if not found.
        """
        org = await self.get_by_id(org_id)
        if org is None:
            return False

        await self.session.delete(org)
        await self.session.commit()
        return True
