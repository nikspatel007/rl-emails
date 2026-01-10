"""Task API endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.api.auth.dependencies import CurrentUserOrApiKey
from rl_emails.schemas.task import (
    TaskCreate,
    TaskDetailResponse,
    TaskListResponse,
    TaskResponse,
    TaskUpdate,
)
from rl_emails.services.task_service import TaskNotFoundError, TaskService

if TYPE_CHECKING:
    from rl_emails.api.auth.clerk import ClerkUser

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

router = APIRouter(prefix="/tasks", tags=["tasks"])
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
    try:
        return UUID(user.id)
    except ValueError:
        import hashlib

        hash_bytes = hashlib.md5(user.id.encode()).digest()
        return UUID(bytes=hash_bytes)


class TaskStatsResponse(BaseModel):
    """Task statistics response."""

    pending: int = Field(description="Number of pending tasks")
    urgent: int = Field(description="Number of urgent pending tasks")


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    user: CurrentUserOrApiKey,
    session: SessionDep,
    status_filter: str | None = Query(None, alias="status", description="Filter by status"),
    project_id: int | None = Query(None, description="Filter by project"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> TaskListResponse:
    """List tasks for the authenticated user.

    Returns paginated list of tasks sorted by urgency.
    """
    service = TaskService(session)
    user_id = _get_user_id(user)
    return await service.list_tasks(
        user_id, status=status_filter, project_id=project_id, limit=limit, offset=offset
    )


@router.get("/stats", response_model=TaskStatsResponse)
async def get_task_stats(
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> TaskStatsResponse:
    """Get task statistics for the authenticated user."""
    service = TaskService(session)
    user_id = _get_user_id(user)
    pending = await service.count_pending(user_id)
    urgent = await service.count_urgent(user_id)
    return TaskStatsResponse(pending=pending, urgent=urgent)


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task(
    task_id: int,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> TaskDetailResponse:
    """Get task details by ID."""
    service = TaskService(session)
    user_id = _get_user_id(user)
    try:
        return await service.get_task(task_id, user_id)
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    data: TaskCreate,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> TaskResponse:
    """Create a new task manually."""
    service = TaskService(session)
    user_id = _get_user_id(user)
    return await service.create_task(user_id, data)


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    data: TaskUpdate,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> TaskResponse:
    """Update a task."""
    service = TaskService(session)
    user_id = _get_user_id(user)
    try:
        return await service.update_task(task_id, data, user_id)
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("/{task_id}/complete", response_model=TaskResponse)
async def complete_task(
    task_id: int,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> TaskResponse:
    """Mark a task as completed."""
    service = TaskService(session)
    user_id = _get_user_id(user)
    try:
        return await service.complete_task(task_id, user_id)
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("/{task_id}/dismiss", response_model=TaskResponse)
async def dismiss_task(
    task_id: int,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> TaskResponse:
    """Dismiss a task (mark as not applicable)."""
    service = TaskService(session)
    user_id = _get_user_id(user)
    try:
        return await service.dismiss_task(task_id, user_id)
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: int,
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> None:
    """Delete a task."""
    service = TaskService(session)
    user_id = _get_user_id(user)
    try:
        await service.delete_task(task_id, user_id)
    except TaskNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
