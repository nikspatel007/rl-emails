"""Priority inbox API endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.api.auth.dependencies import CurrentUserOrApiKey
from rl_emails.schemas.inbox import InboxStats, PriorityInboxResponse
from rl_emails.services.inbox_service import InboxService

if TYPE_CHECKING:
    from rl_emails.api.auth.clerk import ClerkUser

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

router = APIRouter(prefix="/inbox", tags=["inbox"])
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


@router.get("", response_model=PriorityInboxResponse)
async def get_priority_inbox(
    user: CurrentUserOrApiKey,
    session: SessionDep,
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> PriorityInboxResponse:
    """Get priority-sorted inbox.

    Returns emails sorted by priority score with context information.
    Emails from real people are weighted higher than automated/marketing emails.
    """
    service = InboxService(session)
    user_id = _get_user_id(user)
    return await service.get_priority_inbox(user_id, limit=limit, offset=offset)


@router.get("/stats", response_model=InboxStats)
async def get_inbox_stats(
    user: CurrentUserOrApiKey,
    session: SessionDep,
) -> InboxStats:
    """Get inbox statistics.

    Returns aggregate statistics about the inbox including:
    - Total email count
    - Pending task count
    - Urgent email count
    - Emails from real people count
    - Average priority score
    """
    service = InboxService(session)
    user_id = _get_user_id(user)
    return await service.get_inbox_stats(user_id)
