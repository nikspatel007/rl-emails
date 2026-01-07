"""Tests for sync state repository."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from rl_emails.models.sync_state import SyncState
from rl_emails.repositories.sync_state import SyncStateRepository
from rl_emails.schemas.sync import SyncStateUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> SyncStateRepository:
    """Create repository with mock session."""
    return SyncStateRepository(mock_session)


class TestSyncStateRepository:
    """Tests for SyncStateRepository."""

    @pytest.mark.asyncio
    async def test_get_or_create_existing(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test get_or_create when state exists."""
        # Setup
        user_id = uuid.uuid4()
        state = SyncState(id=uuid.uuid4(), user_id=user_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = state
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_or_create(user_id)

        # Verify
        assert result == state
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_new(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test get_or_create when state doesn't exist."""
        # Setup
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_or_create(user_id)

        # Verify
        assert isinstance(result, SyncState)
        assert result.user_id == user_id
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_user_id_found(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting sync state by user ID when found."""
        # Setup
        user_id = uuid.uuid4()
        state = SyncState(id=uuid.uuid4(), user_id=user_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = state
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_user_id(user_id)

        # Verify
        assert result == state

    @pytest.mark.asyncio
    async def test_get_by_user_id_not_found(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting sync state by user ID when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_user_id(uuid.uuid4())

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_update_found(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating sync state when found."""
        # Setup
        user_id = uuid.uuid4()
        state = SyncState(id=uuid.uuid4(), user_id=user_id, sync_status="idle")
        data = SyncStateUpdate(sync_status="syncing")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = state
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update(user_id, data)

        # Verify
        assert result is not None
        assert result.sync_status == "syncing"

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating sync state when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update(uuid.uuid4(), SyncStateUpdate(sync_status="idle"))

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_start_sync(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test starting a sync."""
        # Setup
        user_id = uuid.uuid4()
        state = SyncState(id=uuid.uuid4(), user_id=user_id, sync_status="idle")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = state
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.start_sync(user_id)

        # Verify
        assert result.sync_status == "syncing"
        assert result.error_message is None
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_complete_sync(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test completing a sync."""
        # Setup
        user_id = uuid.uuid4()
        state = SyncState(id=uuid.uuid4(), user_id=user_id, sync_status="syncing")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = state
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.complete_sync(user_id, "history123", 500)

        # Verify
        assert result.sync_status == "idle"
        assert result.last_history_id == "history123"
        assert result.emails_synced == 500
        assert result.last_sync_at is not None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_complete_sync_no_history_id(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test completing a sync without history ID."""
        # Setup
        user_id = uuid.uuid4()
        state = SyncState(
            id=uuid.uuid4(),
            user_id=user_id,
            sync_status="syncing",
            last_history_id="old123",
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = state
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.complete_sync(user_id, None, 100)

        # Verify
        assert result.sync_status == "idle"
        assert result.last_history_id == "old123"  # Unchanged

    @pytest.mark.asyncio
    async def test_fail_sync(
        self, repository: SyncStateRepository, mock_session: AsyncMock
    ) -> None:
        """Test failing a sync."""
        # Setup
        user_id = uuid.uuid4()
        state = SyncState(id=uuid.uuid4(), user_id=user_id, sync_status="syncing")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = state
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.fail_sync(user_id, "Connection timeout")

        # Verify
        assert result.sync_status == "error"
        assert result.error_message == "Connection timeout"
