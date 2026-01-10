"""Project service for business logic."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.project import Project
from rl_emails.repositories.project import ProjectRepository
from rl_emails.schemas.project import (
    ProjectCreate,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)


class ProjectNotFoundError(Exception):
    """Raised when a project is not found."""

    def __init__(self, project_id: int) -> None:
        """Initialize error.

        Args:
            project_id: The project ID that was not found.
        """
        self.project_id = project_id
        super().__init__(f"Project {project_id} not found")


class ProjectService:
    """Service for project operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize service with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self._repo = ProjectRepository(session)

    async def list_projects(
        self,
        user_id: UUID | None = None,
        *,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ProjectListResponse:
        """List projects with pagination.

        Args:
            user_id: Optional user UUID to scope query.
            is_active: Filter by active status.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Paginated project list response.
        """
        if user_id is not None:
            projects, total = await self._repo.list_by_user(
                user_id, is_active=is_active, limit=limit, offset=offset
            )
        else:
            projects, total = await self._repo.list_all(
                is_active=is_active, limit=limit, offset=offset
            )

        return ProjectListResponse(
            projects=[self._to_response(p) for p in projects],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(projects) < total,
        )

    async def get_project(
        self, project_id: int, user_id: UUID | None = None
    ) -> ProjectDetailResponse:
        """Get project details.

        Args:
            project_id: Project ID.
            user_id: Optional user UUID to scope query.

        Returns:
            Project detail response.

        Raises:
            ProjectNotFoundError: If project not found.
        """
        project = await self._repo.get_by_id(project_id, user_id)
        if project is None:
            raise ProjectNotFoundError(project_id)

        return self._to_detail_response(project)

    async def create_project(self, user_id: UUID, data: ProjectCreate) -> ProjectResponse:
        """Create a new project.

        Args:
            user_id: User UUID.
            data: Project creation data.

        Returns:
            Created project response.
        """
        project = await self._repo.create(user_id, data)
        return self._to_response(project)

    async def update_project(
        self, project_id: int, data: ProjectUpdate, user_id: UUID | None = None
    ) -> ProjectResponse:
        """Update a project.

        Args:
            project_id: Project ID.
            data: Update data.
            user_id: Optional user UUID to scope query.

        Returns:
            Updated project response.

        Raises:
            ProjectNotFoundError: If project not found.
        """
        project = await self._repo.update(project_id, data, user_id)
        if project is None:
            raise ProjectNotFoundError(project_id)

        return self._to_response(project)

    async def delete_project(self, project_id: int, user_id: UUID | None = None) -> bool:
        """Delete a project.

        Args:
            project_id: Project ID.
            user_id: Optional user UUID to scope query.

        Returns:
            True if deleted.

        Raises:
            ProjectNotFoundError: If project not found.
        """
        deleted = await self._repo.delete(project_id, user_id)
        if not deleted:
            raise ProjectNotFoundError(project_id)
        return True

    def _to_response(self, project: Project) -> ProjectResponse:
        """Convert project model to response schema."""
        return ProjectResponse(
            id=project.id,
            name=project.name,
            project_type=project.project_type,
            owner_email=project.owner_email,
            participants=project.participants,
            is_active=project.is_active,
            priority=project.priority,
            email_count=project.email_count,
            last_activity=project.last_activity,
            detected_from=project.detected_from,
            confidence=project.confidence,
            user_id=project.user_id,
            created_at=project.created_at,
        )

    def _to_detail_response(self, project: Project) -> ProjectDetailResponse:
        """Convert project model to detail response schema."""
        return ProjectDetailResponse(
            id=project.id,
            name=project.name,
            project_type=project.project_type,
            owner_email=project.owner_email,
            participants=project.participants,
            is_active=project.is_active,
            priority=project.priority,
            email_count=project.email_count,
            last_activity=project.last_activity,
            detected_from=project.detected_from,
            confidence=project.confidence,
            user_id=project.user_id,
            created_at=project.created_at,
            description=project.description,
            keywords=project.keywords,
            start_date=project.start_date,
            due_date=project.due_date,
            completed_at=project.completed_at,
            cluster_id=project.cluster_id,
            related_email_count=project.email_count,
        )
