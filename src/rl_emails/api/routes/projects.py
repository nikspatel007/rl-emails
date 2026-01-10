"""Project API endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.api.auth.dependencies import CurrentUserOrApiKey
from rl_emails.schemas.project import (
    ProjectCreate,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)
from rl_emails.services.project_service import ProjectNotFoundError, ProjectService

if TYPE_CHECKING:
    from rl_emails.api.auth.clerk import ClerkUser

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

router = APIRouter(prefix="/projects", tags=["projects"])
logger = structlog.get_logger(__name__)

# Database session dependency - will be overridden in app setup
_session_factory: SessionFactory | None = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session.

    Yields:
        AsyncSession instance.

    Raises:
        HTTPException: If database not configured.
    """
    if _session_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not configured",
        )
    async with _session_factory() as session:
        yield session


def set_session_factory(factory: SessionFactory | None) -> None:
    """Set the session factory.

    Args:
        factory: AsyncSession factory to use.
    """
    global _session_factory
    _session_factory = factory


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _get_user_id(user: ClerkUser) -> UUID:
    """Get UUID from ClerkUser.id string.

    Args:
        user: Authenticated user.

    Returns:
        User ID as UUID.
    """
    # ClerkUser.id is a string like "user_xxx" or a UUID
    try:
        return UUID(user.id)
    except ValueError:
        # For Clerk user IDs, we need a different approach
        # Using a deterministic UUID from the user ID string
        import hashlib

        hash_bytes = hashlib.md5(user.id.encode()).digest()
        return UUID(bytes=hash_bytes)


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    user: CurrentUserOrApiKey,
    session: SessionDep,
    is_active: bool | None = Query(None, description="Filter by active status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> ProjectListResponse:
    """List projects for the authenticated user.

    Returns paginated list of projects sorted by last activity.
    """
    service = ProjectService(session)
    user_id = _get_user_id(user)
    return await service.list_projects(user_id, is_active=is_active, limit=limit, offset=offset)


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: int,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> ProjectDetailResponse:
    """Get project details by ID."""
    service = ProjectService(session)
    user_id = _get_user_id(user)
    try:
        return await service.get_project(project_id, user_id)
    except ProjectNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> ProjectResponse:
    """Create a new project."""
    service = ProjectService(session)
    user_id = _get_user_id(user)
    return await service.create_project(user_id, data)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    data: ProjectUpdate,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> ProjectResponse:
    """Update a project."""
    service = ProjectService(session)
    user_id = _get_user_id(user)
    try:
        return await service.update_project(project_id, data, user_id)
    except ProjectNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> None:
    """Delete a project."""
    service = ProjectService(session)
    user_id = _get_user_id(user)
    try:
        await service.delete_project(project_id, user_id)
    except ProjectNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
