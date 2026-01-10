"""Tests for task repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from rl_emails.models.task import Task
from rl_emails.repositories.task import TaskRepository
from rl_emails.schemas.task import TaskCreate, TaskUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> TaskRepository:
    """Create repository with mock session."""
    return TaskRepository(mock_session)


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
    )


class TestTaskRepository:
    """Tests for TaskRepository."""

    @pytest.mark.asyncio
    async def test_create_task(self, repository: TaskRepository, mock_session: AsyncMock) -> None:
        """Test creating a task."""
        user_id = uuid.uuid4()
        data = TaskCreate(description="New task", task_type="other")

        result = await repository.create(user_id, data)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        assert isinstance(result, Task)
        assert result.description == "New task"
        assert result.task_type == "other"
        assert result.user_id == user_id
        assert result.task_id.startswith("manual_")

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test getting task by ID when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(1)

        assert result == sample_task
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_with_user_id(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test getting task by ID with user_id scope."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(1, sample_task.user_id)

        assert result == sample_task

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting task by ID when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_user(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test listing tasks for a user."""
        tasks = [sample_task]

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = tasks
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_tasks, total = await repository.list_by_user(sample_task.user_id)

        assert len(result_tasks) == 1
        assert total == 1
        assert result_tasks[0] == sample_task

    @pytest.mark.asyncio
    async def test_list_by_user_with_status_filter(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test listing tasks with status filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_task]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_tasks, total = await repository.list_by_user(sample_task.user_id, status="pending")

        assert len(result_tasks) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_by_user_with_project_filter(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test listing tasks with project filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_task]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_tasks, total = await repository.list_by_user(sample_task.user_id, project_id=1)

        assert len(result_tasks) == 1

    @pytest.mark.asyncio
    async def test_list_by_user_with_pagination(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test listing tasks with pagination."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 50

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_task]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_tasks, total = await repository.list_by_user(
            sample_task.user_id, limit=10, offset=20
        )

        assert total == 50

    @pytest.mark.asyncio
    async def test_list_all(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test listing all tasks."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_task]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_tasks, total = await repository.list_all()

        assert len(result_tasks) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_all_with_status_filter(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test listing all tasks with status filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_task]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_tasks, total = await repository.list_all(status="pending")

        assert len(result_tasks) == 1

    @pytest.mark.asyncio
    async def test_list_all_with_project_filter(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test listing all tasks with project filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_task]
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_tasks, total = await repository.list_all(project_id=1)

        assert len(result_tasks) == 1

    @pytest.mark.asyncio
    async def test_list_all_with_pagination(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test listing all tasks with pagination."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_list_result]

        result_tasks, total = await repository.list_all(limit=10, offset=5)

        assert len(result_tasks) == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_update_found(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test updating task when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        data = TaskUpdate(description="Updated description")
        result = await repository.update(1, data)

        assert result is not None
        assert result.description == "Updated description"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_with_user_id(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test updating task with user_id scope."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        data = TaskUpdate(status="completed")
        result = await repository.update(1, data, sample_task.user_id)

        assert result is not None
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating task when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.update(999, TaskUpdate(description="New"))

        assert result is None

    @pytest.mark.asyncio
    async def test_complete_found(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test completing task when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        result = await repository.complete(1)

        assert result is not None
        assert result.status == "completed"
        assert result.completed_at is not None
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_user_id(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test completing task with user_id scope."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        result = await repository.complete(1, sample_task.user_id)

        assert result is not None
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_complete_not_found(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test completing task when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.complete(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_dismiss_found(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test dismissing task when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        result = await repository.dismiss(1)

        assert result is not None
        assert result.status == "dismissed"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_dismiss_with_user_id(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test dismissing task with user_id scope."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        result = await repository.dismiss(1, sample_task.user_id)

        assert result is not None
        assert result.status == "dismissed"

    @pytest.mark.asyncio
    async def test_dismiss_not_found(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test dismissing task when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.dismiss(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_found(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test deleting task when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        result = await repository.delete(1)

        assert result is True
        mock_session.delete.assert_called_once_with(sample_task)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_with_user_id(
        self,
        repository: TaskRepository,
        mock_session: AsyncMock,
        sample_task: Task,
    ) -> None:
        """Test deleting task with user_id scope."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_task
        mock_session.execute.return_value = mock_result

        result = await repository.delete(1, sample_task.user_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting task when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.delete(999)

        assert result is False
        mock_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_count_by_user(self, repository: TaskRepository, mock_session: AsyncMock) -> None:
        """Test counting tasks for a user."""
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        count = await repository.count_by_user(user_id)

        assert count == 5

    @pytest.mark.asyncio
    async def test_count_by_user_with_status_filter(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test counting tasks with status filter."""
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_session.execute.return_value = mock_result

        count = await repository.count_by_user(user_id, status="pending")

        assert count == 3

    @pytest.mark.asyncio
    async def test_count_by_user_returns_zero_on_none(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test counting returns 0 when scalar returns None."""
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        count = await repository.count_by_user(user_id)

        assert count == 0

    @pytest.mark.asyncio
    async def test_count_urgent(self, repository: TaskRepository, mock_session: AsyncMock) -> None:
        """Test counting urgent tasks."""
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar.return_value = 2
        mock_session.execute.return_value = mock_result

        count = await repository.count_urgent(user_id)

        assert count == 2

    @pytest.mark.asyncio
    async def test_count_urgent_returns_zero_on_none(
        self, repository: TaskRepository, mock_session: AsyncMock
    ) -> None:
        """Test count_urgent returns 0 when scalar returns None."""
        user_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        count = await repository.count_urgent(user_id)

        assert count == 0
