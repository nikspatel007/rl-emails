"""Tests for project repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from rl_emails.models.project import Project
from rl_emails.repositories.project import ProjectRepository
from rl_emails.schemas.project import ProjectCreate, ProjectUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> ProjectRepository:
    """Create repository with mock session."""
    return ProjectRepository(mock_session)


@pytest.fixture
def sample_project() -> Project:
    """Create a sample project."""
    return Project(
        id=1,
        name="Test Project",
        project_type="work",
        user_id=uuid.uuid4(),
        is_active=True,
        priority=3,
        email_count=10,
        created_at=datetime.now(UTC),
    )


class TestProjectRepository:
    """Tests for ProjectRepository."""

    @pytest.mark.asyncio
    async def test_create_project(
        self, repository: ProjectRepository, mock_session: AsyncMock
    ) -> None:
        """Test creating a project."""
        user_id = uuid.uuid4()
        data = ProjectCreate(name="New Project", project_type="work")

        result = await repository.create(user_id, data)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        assert isinstance(result, Project)
        assert result.name == "New Project"
        assert result.project_type == "work"
        assert result.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test getting project by ID when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_project
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(1)

        assert result == sample_project
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_with_user_id(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test getting project by ID with user_id scope."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_project
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(1, sample_project.user_id)

        assert result == sample_project

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, repository: ProjectRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting project by ID when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_user(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test listing projects for a user."""
        projects = [sample_project]

        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        # Mock list query
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = projects
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_projects, total = await repository.list_by_user(sample_project.user_id)

        assert len(result_projects) == 1
        assert total == 1
        assert result_projects[0] == sample_project

    @pytest.mark.asyncio
    async def test_list_by_user_with_active_filter(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test listing projects with active status filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_project]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_projects, total = await repository.list_by_user(
            sample_project.user_id, is_active=True
        )

        assert len(result_projects) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_by_user_with_pagination(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test listing projects with pagination."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 50

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_project]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_projects, total = await repository.list_by_user(
            sample_project.user_id, limit=10, offset=20
        )

        assert total == 50

    @pytest.mark.asyncio
    async def test_list_all(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test listing all projects."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_project]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_projects, total = await repository.list_all()

        assert len(result_projects) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_all_with_active_filter(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test listing all projects with active filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_project]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_projects, total = await repository.list_all(is_active=True)

        assert len(result_projects) == 1

    @pytest.mark.asyncio
    async def test_list_all_with_pagination(
        self, repository: ProjectRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing all projects with pagination."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_projects, total = await repository.list_all(limit=10, offset=5)

        assert len(result_projects) == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_update_found(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test updating project when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_project
        mock_session.execute.return_value = mock_result

        data = ProjectUpdate(name="Updated Name")
        result = await repository.update(1, data)

        assert result is not None
        assert result.name == "Updated Name"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_with_user_id(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test updating project with user_id scope."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_project
        mock_session.execute.return_value = mock_result

        data = ProjectUpdate(is_active=False)
        result = await repository.update(1, data, sample_project.user_id)

        assert result is not None
        assert result.is_active is False

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: ProjectRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating project when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.update(999, ProjectUpdate(name="New"))

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_found(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test deleting project when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_project
        mock_session.execute.return_value = mock_result

        result = await repository.delete(1)

        assert result is True
        mock_session.delete.assert_called_once_with(sample_project)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_with_user_id(
        self,
        repository: ProjectRepository,
        mock_session: AsyncMock,
        sample_project: Project,
    ) -> None:
        """Test deleting project with user_id scope."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_project
        mock_session.execute.return_value = mock_result

        result = await repository.delete(1, sample_project.user_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, repository: ProjectRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting project when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.delete(999)

        assert result is False
        mock_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_count_by_user(
        self, repository: ProjectRepository, mock_session: AsyncMock
    ) -> None:
        """Test counting projects for a user."""
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        count = await repository.count_by_user(user_id)

        assert count == 5

    @pytest.mark.asyncio
    async def test_count_by_user_with_active_filter(
        self, repository: ProjectRepository, mock_session: AsyncMock
    ) -> None:
        """Test counting projects with active filter."""
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_session.execute.return_value = mock_result

        count = await repository.count_by_user(user_id, is_active=True)

        assert count == 3

    @pytest.mark.asyncio
    async def test_count_by_user_returns_zero_on_none(
        self, repository: ProjectRepository, mock_session: AsyncMock
    ) -> None:
        """Test counting returns 0 when scalar returns None."""
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        count = await repository.count_by_user(user_id)

        assert count == 0
