"""Project repository for database operations."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.project import Project
from rl_emails.schemas.project import ProjectCreate, ProjectUpdate


class ProjectRepository:
    """Repository for project database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def create(self, user_id: UUID, data: ProjectCreate) -> Project:
        """Create a new project.

        Args:
            user_id: User UUID.
            data: Project creation data.

        Returns:
            Created project.
        """
        project = Project(user_id=user_id, **data.model_dump())
        self.session.add(project)
        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def get_by_id(self, project_id: int, user_id: UUID | None = None) -> Project | None:
        """Get project by ID.

        Args:
            project_id: Project ID.
            user_id: Optional user UUID to scope query.

        Returns:
            Project if found, None otherwise.
        """
        conditions = [Project.id == project_id]
        if user_id is not None:
            conditions.append(Project.user_id == user_id)

        result = await self.session.execute(select(Project).where(and_(*conditions)))
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: UUID,
        *,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Project], int]:
        """List projects for a user.

        Args:
            user_id: User UUID.
            is_active: Filter by active status.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (projects list, total count).
        """
        conditions = [Project.user_id == user_id]
        if is_active is not None:
            conditions.append(Project.is_active == is_active)

        # Get total count
        count_query = select(func.count()).select_from(Project).where(and_(*conditions))
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        # Get projects
        query = (
            select(Project)
            .where(and_(*conditions))
            .order_by(Project.last_activity.desc().nulls_last(), Project.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        projects = list(result.scalars().all())

        return projects, total

    async def list_all(
        self,
        *,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Project], int]:
        """List all projects (for non-multi-tenant mode).

        Args:
            is_active: Filter by active status.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (projects list, total count).
        """
        conditions = []
        if is_active is not None:
            conditions.append(Project.is_active == is_active)

        # Get total count
        count_query = select(func.count()).select_from(Project)
        if conditions:
            count_query = count_query.where(and_(*conditions))
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        # Get projects
        query = select(Project).order_by(
            Project.last_activity.desc().nulls_last(), Project.created_at.desc()
        )
        if conditions:
            query = query.where(and_(*conditions))
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        projects = list(result.scalars().all())

        return projects, total

    async def update(
        self, project_id: int, data: ProjectUpdate, user_id: UUID | None = None
    ) -> Project | None:
        """Update a project.

        Args:
            project_id: Project ID.
            data: Update data.
            user_id: Optional user UUID to scope query.

        Returns:
            Updated project if found, None otherwise.
        """
        project = await self.get_by_id(project_id, user_id)
        if project is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(project, key, value)

        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def delete(self, project_id: int, user_id: UUID | None = None) -> bool:
        """Delete a project.

        Args:
            project_id: Project ID.
            user_id: Optional user UUID to scope query.

        Returns:
            True if deleted, False if not found.
        """
        project = await self.get_by_id(project_id, user_id)
        if project is None:
            return False

        await self.session.delete(project)
        await self.session.commit()
        return True

    async def count_by_user(self, user_id: UUID, *, is_active: bool | None = None) -> int:
        """Count projects for a user.

        Args:
            user_id: User UUID.
            is_active: Filter by active status.

        Returns:
            Count of projects.
        """
        conditions = [Project.user_id == user_id]
        if is_active is not None:
            conditions.append(Project.is_active == is_active)

        query = select(func.count()).select_from(Project).where(and_(*conditions))
        result = await self.session.execute(query)
        return result.scalar() or 0
