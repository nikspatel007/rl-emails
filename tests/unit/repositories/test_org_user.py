"""Tests for organization user repository."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from rl_emails.models.org_user import OrgUser
from rl_emails.repositories.org_user import OrgUserRepository
from rl_emails.schemas.org_user import OrgUserCreate, OrgUserUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> OrgUserRepository:
    """Create repository with mock session."""
    return OrgUserRepository(mock_session)


class TestOrgUserRepository:
    """Tests for OrgUserRepository."""

    @pytest.mark.asyncio
    async def test_create_user(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating a user."""
        # Setup
        org_id = uuid.uuid4()
        data = OrgUserCreate(email="user@example.com", name="Test User")

        # Execute
        result = await repository.create(org_id, data)

        # Verify
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        assert isinstance(result, OrgUser)
        assert result.email == "user@example.com"
        assert result.org_id == org_id

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting user by ID when found."""
        # Setup
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        user = OrgUser(id=user_id, org_id=org_id, email="test@example.com")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_id(user_id)

        # Verify
        assert result == user

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting user by ID when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_id(uuid.uuid4())

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_email_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting user by email when found."""
        # Setup
        org_id = uuid.uuid4()
        user = OrgUser(id=uuid.uuid4(), org_id=org_id, email="test@example.com")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_email(org_id, "test@example.com")

        # Verify
        assert result == user

    @pytest.mark.asyncio
    async def test_get_by_email_not_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting user by email when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_email(uuid.uuid4(), "nonexistent@example.com")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_org(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing users by organization."""
        # Setup
        org_id = uuid.uuid4()
        users = [
            OrgUser(id=uuid.uuid4(), org_id=org_id, email="user1@example.com"),
            OrgUser(id=uuid.uuid4(), org_id=org_id, email="user2@example.com"),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = users
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.list_by_org(org_id)

        # Verify
        assert len(result) == 2
        assert result == users

    @pytest.mark.asyncio
    async def test_list_by_org_with_pagination(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing users with pagination."""
        # Setup
        org_id = uuid.uuid4()
        users = [OrgUser(id=uuid.uuid4(), org_id=org_id, email="user@example.com")]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = users
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.list_by_org(org_id, limit=10, offset=5)

        # Verify
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_update_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating user when found."""
        # Setup
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        user = OrgUser(id=user_id, org_id=org_id, email="test@example.com", name="Old")
        data = OrgUserUpdate(name="New Name")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update(user_id, data)

        # Verify
        assert result is not None
        assert result.name == "New Name"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating user when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update(uuid.uuid4(), OrgUserUpdate(name="New"))

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_set_gmail_connected_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test setting Gmail connection status when found."""
        # Setup
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        user = OrgUser(id=user_id, org_id=org_id, email="test@example.com", gmail_connected=False)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.set_gmail_connected(user_id, True)

        # Verify
        assert result is not None
        assert result.gmail_connected is True

    @pytest.mark.asyncio
    async def test_set_gmail_connected_not_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test setting Gmail connection status when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.set_gmail_connected(uuid.uuid4(), True)

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting user when found."""
        # Setup
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()
        user = OrgUser(id=user_id, org_id=org_id, email="test@example.com")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.delete(user_id)

        # Verify
        assert result is True
        mock_session.delete.assert_called_once_with(user)

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, repository: OrgUserRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting user when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.delete(uuid.uuid4())

        # Verify
        assert result is False
