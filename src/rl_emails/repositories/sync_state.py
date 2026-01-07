"""Sync state repository for database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.sync_state import SyncState
from rl_emails.schemas.sync import SyncStateUpdate


class SyncStateRepository:
    """Repository for sync state database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def get_or_create(self, user_id: UUID) -> SyncState:
        """Get or create sync state for a user.

        Args:
            user_id: User UUID.

        Returns:
            Sync state (existing or newly created).
        """
        state = await self.get_by_user_id(user_id)
        if state is None:
            state = SyncState(user_id=user_id)
            self.session.add(state)
            await self.session.commit()
            await self.session.refresh(state)
        return state

    async def get_by_user_id(self, user_id: UUID) -> SyncState | None:
        """Get sync state by user ID.

        Args:
            user_id: User UUID.

        Returns:
            Sync state if found, None otherwise.
        """
        result = await self.session.execute(select(SyncState).where(SyncState.user_id == user_id))
        return result.scalar_one_or_none()

    async def update(self, user_id: UUID, data: SyncStateUpdate) -> SyncState | None:
        """Update sync state.

        Args:
            user_id: User UUID.
            data: Update data.

        Returns:
            Updated sync state if found, None otherwise.
        """
        state = await self.get_by_user_id(user_id)
        if state is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(state, key, value)

        await self.session.commit()
        await self.session.refresh(state)
        return state

    async def start_sync(self, user_id: UUID) -> SyncState:
        """Mark sync as started.

        Args:
            user_id: User UUID.

        Returns:
            Updated sync state.
        """
        state = await self.get_or_create(user_id)
        state.sync_status = "syncing"
        state.error_message = None
        await self.session.commit()
        await self.session.refresh(state)
        return state

    async def complete_sync(
        self, user_id: UUID, history_id: str | None, emails_synced: int
    ) -> SyncState:
        """Mark sync as completed.

        Args:
            user_id: User UUID.
            history_id: Gmail history ID from sync.
            emails_synced: Number of emails synced.

        Returns:
            Updated sync state.
        """
        state = await self.get_or_create(user_id)
        state.sync_status = "idle"
        state.last_sync_at = datetime.now(UTC)
        state.emails_synced = emails_synced
        if history_id:
            state.last_history_id = history_id
        state.error_message = None
        await self.session.commit()
        await self.session.refresh(state)
        return state

    async def fail_sync(self, user_id: UUID, error_message: str) -> SyncState:
        """Mark sync as failed.

        Args:
            user_id: User UUID.
            error_message: Error description.

        Returns:
            Updated sync state.
        """
        state = await self.get_or_create(user_id)
        state.sync_status = "error"
        state.error_message = error_message
        await self.session.commit()
        await self.session.refresh(state)
        return state
