"""Tests for task service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from rl_emails.models.project import Project
from rl_emails.models.task import Task
from rl_emails.schemas.task import TaskCreate, TaskUpdate
from rl_emails.services.task_service import TaskNotFoundError, TaskService


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    return AsyncMock()


@pytest.fixture
def service(mock_session: AsyncMock) -> TaskService:
    """Create service with mock session."""
    return TaskService(mock_session)


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task."""
    return Task(
        id=1,
        task_id="task_123",
        user_id=uuid.uuid4(),
        description="Test task",
        task_type="action",
        status="pending",
        urgency_score=0.5,
        is_assigned_to_user=False,
        created_at=datetime.now(UTC),
        source_text="Original source text",
    )


@pytest.fixture
def sample_task_with_project() -> Task:
    """Create a sample task with project."""
    project = Project(
        id=1,
        name="Test Project",
        project_type="work",
        user_id=uuid.uuid4(),
        created_at=datetime.now(UTC),
    )
    task = Task(
        id=1,
        task_id="task_123",
        user_id=project.user_id,
        description="Test task",
        task_type="action",
        status="pending",
        is_assigned_to_user=False,
        project_id=1,
        project=project,
        created_at=datetime.now(UTC),
    )
    return task


class TestTaskNotFoundError:
    """Tests for TaskNotFoundError."""

    def test_error_message(self) -> None:
        """Test error message format."""
        error = TaskNotFoundError(123)
        assert error.task_id == 123
        assert str(error) == "Task 123 not found"


class TestTaskService:
    """Tests for TaskService."""

    def test_init(self, mock_session: AsyncMock) -> None:
        """Test service initialization."""
        service = TaskService(mock_session)
        assert service._repo is not None

    @pytest.mark.asyncio
    async def test_list_tasks_with_user_id(self, service: TaskService, sample_task: Task) -> None:
        """Test listing tasks with user_id."""
        with patch.object(service._repo, "list_by_user") as mock_list:
            mock_list.return_value = ([sample_task], 1)

            result = await service.list_tasks(sample_task.user_id)

            assert len(result.tasks) == 1
            assert result.total == 1
            assert result.limit == 100
            assert result.offset == 0
            assert result.has_more is False
            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tasks_without_user_id(
        self, service: TaskService, sample_task: Task
    ) -> None:
        """Test listing all tasks without user_id."""
        with patch.object(service._repo, "list_all") as mock_list:
            mock_list.return_value = ([sample_task], 1)

            result = await service.list_tasks()

            assert len(result.tasks) == 1
            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tasks_with_status_filter(
        self, service: TaskService, sample_task: Task
    ) -> None:
        """Test listing tasks with status filter."""
        with patch.object(service._repo, "list_by_user") as mock_list:
            mock_list.return_value = ([sample_task], 1)

            await service.list_tasks(sample_task.user_id, status="pending")

            mock_list.assert_called_once_with(
                sample_task.user_id, status="pending", project_id=None, limit=100, offset=0
            )

    @pytest.mark.asyncio
    async def test_list_tasks_with_project_filter(
        self, service: TaskService, sample_task: Task
    ) -> None:
        """Test listing tasks with project filter."""
        with patch.object(service._repo, "list_by_user") as mock_list:
            mock_list.return_value = ([sample_task], 1)

            await service.list_tasks(sample_task.user_id, project_id=1)

            mock_list.assert_called_once_with(
                sample_task.user_id, status=None, project_id=1, limit=100, offset=0
            )

    @pytest.mark.asyncio
    async def test_list_tasks_has_more(self, service: TaskService, sample_task: Task) -> None:
        """Test listing tasks with pagination has_more."""
        with patch.object(service._repo, "list_by_user") as mock_list:
            mock_list.return_value = ([sample_task], 50)

            result = await service.list_tasks(sample_task.user_id, limit=10, offset=0)

            assert result.has_more is True

    @pytest.mark.asyncio
    async def test_get_task_found(self, service: TaskService, sample_task: Task) -> None:
        """Test getting task when found."""
        with patch.object(service._repo, "get_by_id") as mock_get:
            mock_get.return_value = sample_task

            result = await service.get_task(1)

            assert result.id == 1
            assert result.description == "Test task"
            assert result.source_text == "Original source text"
            mock_get.assert_called_once_with(1, None)

    @pytest.mark.asyncio
    async def test_get_task_with_user_id(self, service: TaskService, sample_task: Task) -> None:
        """Test getting task with user_id scope."""
        with patch.object(service._repo, "get_by_id") as mock_get:
            mock_get.return_value = sample_task

            await service.get_task(1, sample_task.user_id)

            mock_get.assert_called_once_with(1, sample_task.user_id)

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, service: TaskService) -> None:
        """Test getting task when not found."""
        with patch.object(service._repo, "get_by_id") as mock_get:
            mock_get.return_value = None

            with pytest.raises(TaskNotFoundError) as exc_info:
                await service.get_task(999)

            assert exc_info.value.task_id == 999

    @pytest.mark.asyncio
    async def test_get_task_with_project(
        self, service: TaskService, sample_task_with_project: Task
    ) -> None:
        """Test getting task detail response with project name."""
        with patch.object(service._repo, "get_by_id") as mock_get:
            mock_get.return_value = sample_task_with_project

            result = await service.get_task(1)

            assert result.project_name == "Test Project"

    @pytest.mark.asyncio
    async def test_create_task(self, service: TaskService, sample_task: Task) -> None:
        """Test creating a task."""
        with patch.object(service._repo, "create") as mock_create:
            mock_create.return_value = sample_task
            data = TaskCreate(description="Test task", task_type="other")

            result = await service.create_task(sample_task.user_id, data)

            assert result.description == "Test task"
            mock_create.assert_called_once_with(sample_task.user_id, data)

    @pytest.mark.asyncio
    async def test_update_task_found(self, service: TaskService, sample_task: Task) -> None:
        """Test updating task when found."""
        with patch.object(service._repo, "update") as mock_update:
            sample_task.description = "Updated description"
            mock_update.return_value = sample_task
            data = TaskUpdate(description="Updated description")

            result = await service.update_task(1, data)

            assert result.description == "Updated description"
            mock_update.assert_called_once_with(1, data, None)

    @pytest.mark.asyncio
    async def test_update_task_with_user_id(self, service: TaskService, sample_task: Task) -> None:
        """Test updating task with user_id scope."""
        with patch.object(service._repo, "update") as mock_update:
            mock_update.return_value = sample_task
            data = TaskUpdate(status="completed")

            await service.update_task(1, data, sample_task.user_id)

            mock_update.assert_called_once_with(1, data, sample_task.user_id)

    @pytest.mark.asyncio
    async def test_update_task_not_found(self, service: TaskService) -> None:
        """Test updating task when not found."""
        with patch.object(service._repo, "update") as mock_update:
            mock_update.return_value = None
            data = TaskUpdate(description="New description")

            with pytest.raises(TaskNotFoundError) as exc_info:
                await service.update_task(999, data)

            assert exc_info.value.task_id == 999

    @pytest.mark.asyncio
    async def test_complete_task_found(self, service: TaskService, sample_task: Task) -> None:
        """Test completing task when found."""
        with patch.object(service._repo, "complete") as mock_complete:
            sample_task.status = "completed"
            mock_complete.return_value = sample_task

            result = await service.complete_task(1)

            assert result.status == "completed"
            mock_complete.assert_called_once_with(1, None)

    @pytest.mark.asyncio
    async def test_complete_task_with_user_id(
        self, service: TaskService, sample_task: Task
    ) -> None:
        """Test completing task with user_id scope."""
        with patch.object(service._repo, "complete") as mock_complete:
            sample_task.status = "completed"
            mock_complete.return_value = sample_task

            await service.complete_task(1, sample_task.user_id)

            mock_complete.assert_called_once_with(1, sample_task.user_id)

    @pytest.mark.asyncio
    async def test_complete_task_not_found(self, service: TaskService) -> None:
        """Test completing task when not found."""
        with patch.object(service._repo, "complete") as mock_complete:
            mock_complete.return_value = None

            with pytest.raises(TaskNotFoundError) as exc_info:
                await service.complete_task(999)

            assert exc_info.value.task_id == 999

    @pytest.mark.asyncio
    async def test_dismiss_task_found(self, service: TaskService, sample_task: Task) -> None:
        """Test dismissing task when found."""
        with patch.object(service._repo, "dismiss") as mock_dismiss:
            sample_task.status = "dismissed"
            mock_dismiss.return_value = sample_task

            result = await service.dismiss_task(1)

            assert result.status == "dismissed"
            mock_dismiss.assert_called_once_with(1, None)

    @pytest.mark.asyncio
    async def test_dismiss_task_with_user_id(self, service: TaskService, sample_task: Task) -> None:
        """Test dismissing task with user_id scope."""
        with patch.object(service._repo, "dismiss") as mock_dismiss:
            sample_task.status = "dismissed"
            mock_dismiss.return_value = sample_task

            await service.dismiss_task(1, sample_task.user_id)

            mock_dismiss.assert_called_once_with(1, sample_task.user_id)

    @pytest.mark.asyncio
    async def test_dismiss_task_not_found(self, service: TaskService) -> None:
        """Test dismissing task when not found."""
        with patch.object(service._repo, "dismiss") as mock_dismiss:
            mock_dismiss.return_value = None

            with pytest.raises(TaskNotFoundError) as exc_info:
                await service.dismiss_task(999)

            assert exc_info.value.task_id == 999

    @pytest.mark.asyncio
    async def test_delete_task_found(self, service: TaskService) -> None:
        """Test deleting task when found."""
        with patch.object(service._repo, "delete") as mock_delete:
            mock_delete.return_value = True

            result = await service.delete_task(1)

            assert result is True
            mock_delete.assert_called_once_with(1, None)

    @pytest.mark.asyncio
    async def test_delete_task_with_user_id(self, service: TaskService, sample_task: Task) -> None:
        """Test deleting task with user_id scope."""
        with patch.object(service._repo, "delete") as mock_delete:
            mock_delete.return_value = True

            await service.delete_task(1, sample_task.user_id)

            mock_delete.assert_called_once_with(1, sample_task.user_id)

    @pytest.mark.asyncio
    async def test_delete_task_not_found(self, service: TaskService) -> None:
        """Test deleting task when not found."""
        with patch.object(service._repo, "delete") as mock_delete:
            mock_delete.return_value = False

            with pytest.raises(TaskNotFoundError) as exc_info:
                await service.delete_task(999)

            assert exc_info.value.task_id == 999

    @pytest.mark.asyncio
    async def test_count_pending(self, service: TaskService, sample_task: Task) -> None:
        """Test counting pending tasks."""
        with patch.object(service._repo, "count_by_user") as mock_count:
            mock_count.return_value = 5

            result = await service.count_pending(sample_task.user_id)

            assert result == 5
            mock_count.assert_called_once_with(sample_task.user_id, status="pending")

    @pytest.mark.asyncio
    async def test_count_urgent(self, service: TaskService, sample_task: Task) -> None:
        """Test counting urgent tasks."""
        with patch.object(service._repo, "count_urgent") as mock_count:
            mock_count.return_value = 2

            result = await service.count_urgent(sample_task.user_id)

            assert result == 2
            mock_count.assert_called_once_with(sample_task.user_id)

    def test_to_response(self, service: TaskService, sample_task: Task) -> None:
        """Test converting task to response."""
        result = service._to_response(sample_task)

        assert result.id == sample_task.id
        assert result.task_id == sample_task.task_id
        assert result.description == sample_task.description
        assert result.task_type == sample_task.task_type
        assert result.status == sample_task.status
        assert result.user_id == sample_task.user_id

    def test_to_detail_response(self, service: TaskService, sample_task: Task) -> None:
        """Test converting task to detail response."""
        result = service._to_detail_response(sample_task)

        assert result.id == sample_task.id
        assert result.source_text == sample_task.source_text
        assert result.project_name is None

    def test_to_detail_response_with_project(
        self, service: TaskService, sample_task_with_project: Task
    ) -> None:
        """Test converting task with project to detail response."""
        result = service._to_detail_response(sample_task_with_project)

        assert result.project_name == "Test Project"
