"""Tests for organization repository."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from rl_emails.models.organization import Organization
from rl_emails.repositories.organization import OrganizationRepository
from rl_emails.schemas.organization import OrganizationCreate, OrganizationUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> OrganizationRepository:
    """Create repository with mock session."""
    return OrganizationRepository(mock_session)


class TestOrganizationRepository:
    """Tests for OrganizationRepository."""

    @pytest.mark.asyncio
    async def test_create_organization(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating an organization."""
        # Setup
        data = OrganizationCreate(name="Acme Corp", slug="acme")

        # Execute
        result = await repository.create(data)

        # Verify
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        assert isinstance(result, Organization)
        assert result.name == "Acme Corp"
        assert result.slug == "acme"

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting organization by ID when found."""
        # Setup
        org_id = uuid.uuid4()
        org = Organization(id=org_id, name="Test", slug="test")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = org
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_id(org_id)

        # Verify
        assert result == org
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting organization by ID when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_id(uuid.uuid4())

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_slug_found(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting organization by slug when found."""
        # Setup
        org = Organization(id=uuid.uuid4(), name="Test", slug="test")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = org
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_slug("test")

        # Verify
        assert result == org

    @pytest.mark.asyncio
    async def test_get_by_slug_not_found(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting organization by slug when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.get_by_slug("nonexistent")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing all organizations."""
        # Setup
        orgs = [
            Organization(id=uuid.uuid4(), name="Org 1", slug="org1"),
            Organization(id=uuid.uuid4(), name="Org 2", slug="org2"),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = orgs
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.list_all()

        # Verify
        assert len(result) == 2
        assert result == orgs

    @pytest.mark.asyncio
    async def test_list_all_with_pagination(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing organizations with pagination."""
        # Setup
        orgs = [Organization(id=uuid.uuid4(), name="Org", slug="org")]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = orgs
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.list_all(limit=10, offset=5)

        # Verify
        assert len(result) == 1
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_found(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating organization when found."""
        # Setup
        org_id = uuid.uuid4()
        org = Organization(id=org_id, name="Old Name", slug="test")
        data = OrganizationUpdate(name="New Name")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = org
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update(org_id, data)

        # Verify
        assert result is not None
        assert result.name == "New Name"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating organization when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.update(uuid.uuid4(), OrganizationUpdate(name="New"))

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_found(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting organization when found."""
        # Setup
        org_id = uuid.uuid4()
        org = Organization(id=org_id, name="Test", slug="test")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = org
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.delete(org_id)

        # Verify
        assert result is True
        mock_session.delete.assert_called_once_with(org)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, repository: OrganizationRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting organization when not found."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Execute
        result = await repository.delete(uuid.uuid4())

        # Verify
        assert result is False
        mock_session.delete.assert_not_called()
