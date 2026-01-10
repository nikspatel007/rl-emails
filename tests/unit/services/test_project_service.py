"""Tests for project service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from rl_emails.models.project import Project
from rl_emails.schemas.project import ProjectCreate, ProjectUpdate
from rl_emails.services.project_service import ProjectNotFoundError, ProjectService


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    return AsyncMock()


@pytest.fixture
def service(mock_session: AsyncMock) -> ProjectService:
    """Create service with mock session."""
    return ProjectService(mock_session)


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
        description="A test project",
        keywords=["test", "project"],
    )


class TestProjectNotFoundError:
    """Tests for ProjectNotFoundError."""

    def test_error_message(self) -> None:
        """Test error message format."""
        error = ProjectNotFoundError(123)
        assert error.project_id == 123
        assert str(error) == "Project 123 not found"


class TestProjectService:
    """Tests for ProjectService."""

    def test_init(self, mock_session: AsyncMock) -> None:
        """Test service initialization."""
        service = ProjectService(mock_session)
        assert service._repo is not None

    @pytest.mark.asyncio
    async def test_list_projects_with_user_id(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test listing projects with user_id."""
        with patch.object(service._repo, "list_by_user") as mock_list:
            mock_list.return_value = ([sample_project], 1)

            result = await service.list_projects(sample_project.user_id)

            assert len(result.projects) == 1
            assert result.total == 1
            assert result.limit == 100
            assert result.offset == 0
            assert result.has_more is False
            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_projects_without_user_id(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test listing all projects without user_id."""
        with patch.object(service._repo, "list_all") as mock_list:
            mock_list.return_value = ([sample_project], 1)

            result = await service.list_projects()

            assert len(result.projects) == 1
            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_projects_with_active_filter(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test listing projects with active filter."""
        with patch.object(service._repo, "list_by_user") as mock_list:
            mock_list.return_value = ([sample_project], 1)

            result = await service.list_projects(sample_project.user_id, is_active=True)

            assert len(result.projects) == 1
            mock_list.assert_called_once_with(
                sample_project.user_id, is_active=True, limit=100, offset=0
            )

    @pytest.mark.asyncio
    async def test_list_projects_has_more(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test listing projects with pagination has_more."""
        with patch.object(service._repo, "list_by_user") as mock_list:
            mock_list.return_value = ([sample_project], 50)

            result = await service.list_projects(sample_project.user_id, limit=10, offset=0)

            assert result.has_more is True

    @pytest.mark.asyncio
    async def test_get_project_found(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test getting project when found."""
        with patch.object(service._repo, "get_by_id") as mock_get:
            mock_get.return_value = sample_project

            result = await service.get_project(1)

            assert result.id == 1
            assert result.name == "Test Project"
            assert result.description == "A test project"
            mock_get.assert_called_once_with(1, None)

    @pytest.mark.asyncio
    async def test_get_project_with_user_id(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test getting project with user_id scope."""
        with patch.object(service._repo, "get_by_id") as mock_get:
            mock_get.return_value = sample_project

            await service.get_project(1, sample_project.user_id)

            mock_get.assert_called_once_with(1, sample_project.user_id)

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, service: ProjectService) -> None:
        """Test getting project when not found."""
        with patch.object(service._repo, "get_by_id") as mock_get:
            mock_get.return_value = None

            with pytest.raises(ProjectNotFoundError) as exc_info:
                await service.get_project(999)

            assert exc_info.value.project_id == 999

    @pytest.mark.asyncio
    async def test_create_project(self, service: ProjectService, sample_project: Project) -> None:
        """Test creating a project."""
        with patch.object(service._repo, "create") as mock_create:
            mock_create.return_value = sample_project
            data = ProjectCreate(name="Test Project", project_type="work")

            result = await service.create_project(sample_project.user_id, data)

            assert result.name == "Test Project"
            mock_create.assert_called_once_with(sample_project.user_id, data)

    @pytest.mark.asyncio
    async def test_update_project_found(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test updating project when found."""
        with patch.object(service._repo, "update") as mock_update:
            sample_project.name = "Updated Name"
            mock_update.return_value = sample_project
            data = ProjectUpdate(name="Updated Name")

            result = await service.update_project(1, data)

            assert result.name == "Updated Name"
            mock_update.assert_called_once_with(1, data, None)

    @pytest.mark.asyncio
    async def test_update_project_with_user_id(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test updating project with user_id scope."""
        with patch.object(service._repo, "update") as mock_update:
            mock_update.return_value = sample_project
            data = ProjectUpdate(is_active=False)

            await service.update_project(1, data, sample_project.user_id)

            mock_update.assert_called_once_with(1, data, sample_project.user_id)

    @pytest.mark.asyncio
    async def test_update_project_not_found(self, service: ProjectService) -> None:
        """Test updating project when not found."""
        with patch.object(service._repo, "update") as mock_update:
            mock_update.return_value = None
            data = ProjectUpdate(name="New Name")

            with pytest.raises(ProjectNotFoundError) as exc_info:
                await service.update_project(999, data)

            assert exc_info.value.project_id == 999

    @pytest.mark.asyncio
    async def test_delete_project_found(self, service: ProjectService) -> None:
        """Test deleting project when found."""
        with patch.object(service._repo, "delete") as mock_delete:
            mock_delete.return_value = True

            result = await service.delete_project(1)

            assert result is True
            mock_delete.assert_called_once_with(1, None)

    @pytest.mark.asyncio
    async def test_delete_project_with_user_id(
        self, service: ProjectService, sample_project: Project
    ) -> None:
        """Test deleting project with user_id scope."""
        with patch.object(service._repo, "delete") as mock_delete:
            mock_delete.return_value = True

            await service.delete_project(1, sample_project.user_id)

            mock_delete.assert_called_once_with(1, sample_project.user_id)

    @pytest.mark.asyncio
    async def test_delete_project_not_found(self, service: ProjectService) -> None:
        """Test deleting project when not found."""
        with patch.object(service._repo, "delete") as mock_delete:
            mock_delete.return_value = False

            with pytest.raises(ProjectNotFoundError) as exc_info:
                await service.delete_project(999)

            assert exc_info.value.project_id == 999

    def test_to_response(self, service: ProjectService, sample_project: Project) -> None:
        """Test converting project to response."""
        result = service._to_response(sample_project)

        assert result.id == sample_project.id
        assert result.name == sample_project.name
        assert result.project_type == sample_project.project_type
        assert result.user_id == sample_project.user_id
        assert result.is_active == sample_project.is_active

    def test_to_detail_response(self, service: ProjectService, sample_project: Project) -> None:
        """Test converting project to detail response."""
        result = service._to_detail_response(sample_project)

        assert result.id == sample_project.id
        assert result.name == sample_project.name
        assert result.description == sample_project.description
        assert result.keywords == sample_project.keywords
