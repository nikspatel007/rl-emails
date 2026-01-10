"""Task service for business logic."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.task import Task
from rl_emails.repositories.task import TaskRepository
from rl_emails.schemas.task import (
    TaskCreate,
    TaskDetailResponse,
    TaskListResponse,
    TaskResponse,
    TaskUpdate,
)


class TaskNotFoundError(Exception):
    """Raised when a task is not found."""

    def __init__(self, task_id: int) -> None:
        """Initialize error.

        Args:
            task_id: The task ID that was not found.
        """
        self.task_id = task_id
        super().__init__(f"Task {task_id} not found")


class TaskService:
    """Service for task operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize service with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self._repo = TaskRepository(session)

    async def list_tasks(
        self,
        user_id: UUID | None = None,
        *,
        status: str | None = None,
        project_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> TaskListResponse:
        """List tasks with pagination.

        Args:
            user_id: Optional user UUID to scope query.
            status: Filter by status.
            project_id: Filter by project.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Paginated task list response.
        """
        if user_id is not None:
            tasks, total = await self._repo.list_by_user(
                user_id, status=status, project_id=project_id, limit=limit, offset=offset
            )
        else:
            tasks, total = await self._repo.list_all(
                status=status, project_id=project_id, limit=limit, offset=offset
            )

        return TaskListResponse(
            tasks=[self._to_response(t) for t in tasks],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(tasks) < total,
        )

    async def get_task(self, task_id: int, user_id: UUID | None = None) -> TaskDetailResponse:
        """Get task details.

        Args:
            task_id: Task ID.
            user_id: Optional user UUID to scope query.

        Returns:
            Task detail response.

        Raises:
            TaskNotFoundError: If task not found.
        """
        task = await self._repo.get_by_id(task_id, user_id)
        if task is None:
            raise TaskNotFoundError(task_id)

        return self._to_detail_response(task)

    async def create_task(self, user_id: UUID, data: TaskCreate) -> TaskResponse:
        """Create a new task.

        Args:
            user_id: User UUID.
            data: Task creation data.

        Returns:
            Created task response.
        """
        task = await self._repo.create(user_id, data)
        return self._to_response(task)

    async def update_task(
        self, task_id: int, data: TaskUpdate, user_id: UUID | None = None
    ) -> TaskResponse:
        """Update a task.

        Args:
            task_id: Task ID.
            data: Update data.
            user_id: Optional user UUID to scope query.

        Returns:
            Updated task response.

        Raises:
            TaskNotFoundError: If task not found.
        """
        task = await self._repo.update(task_id, data, user_id)
        if task is None:
            raise TaskNotFoundError(task_id)

        return self._to_response(task)

    async def complete_task(self, task_id: int, user_id: UUID | None = None) -> TaskResponse:
        """Mark a task as completed.

        Args:
            task_id: Task ID.
            user_id: Optional user UUID to scope query.

        Returns:
            Updated task response.

        Raises:
            TaskNotFoundError: If task not found.
        """
        task = await self._repo.complete(task_id, user_id)
        if task is None:
            raise TaskNotFoundError(task_id)

        return self._to_response(task)

    async def dismiss_task(self, task_id: int, user_id: UUID | None = None) -> TaskResponse:
        """Dismiss a task (mark as not applicable).

        Args:
            task_id: Task ID.
            user_id: Optional user UUID to scope query.

        Returns:
            Updated task response.

        Raises:
            TaskNotFoundError: If task not found.
        """
        task = await self._repo.dismiss(task_id, user_id)
        if task is None:
            raise TaskNotFoundError(task_id)

        return self._to_response(task)

    async def delete_task(self, task_id: int, user_id: UUID | None = None) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID.
            user_id: Optional user UUID to scope query.

        Returns:
            True if deleted.

        Raises:
            TaskNotFoundError: If task not found.
        """
        deleted = await self._repo.delete(task_id, user_id)
        if not deleted:
            raise TaskNotFoundError(task_id)
        return True

    async def count_pending(self, user_id: UUID) -> int:
        """Count pending tasks for a user.

        Args:
            user_id: User UUID.

        Returns:
            Count of pending tasks.
        """
        return await self._repo.count_by_user(user_id, status="pending")

    async def count_urgent(self, user_id: UUID) -> int:
        """Count urgent pending tasks for a user.

        Args:
            user_id: User UUID.

        Returns:
            Count of urgent pending tasks.
        """
        return await self._repo.count_urgent(user_id)

    def _to_response(self, task: Task) -> TaskResponse:
        """Convert task model to response schema."""
        return TaskResponse(
            id=task.id,
            task_id=task.task_id,
            email_id=task.email_id,
            project_id=task.project_id,
            description=task.description,
            task_type=task.task_type,
            complexity=task.complexity,
            deadline=task.deadline,
            deadline_text=task.deadline_text,
            urgency_score=task.urgency_score,
            status=task.status,
            assigned_to=task.assigned_to,
            assigned_by=task.assigned_by,
            is_assigned_to_user=task.is_assigned_to_user,
            extraction_method=task.extraction_method,
            completed_at=task.completed_at,
            created_at=task.created_at,
            user_id=task.user_id,
        )

    def _to_detail_response(self, task: Task) -> TaskDetailResponse:
        """Convert task model to detail response schema."""
        # Get project name if associated
        project_name = None
        if task.project is not None:
            project_name = task.project.name

        return TaskDetailResponse(
            id=task.id,
            task_id=task.task_id,
            email_id=task.email_id,
            project_id=task.project_id,
            description=task.description,
            task_type=task.task_type,
            complexity=task.complexity,
            deadline=task.deadline,
            deadline_text=task.deadline_text,
            urgency_score=task.urgency_score,
            status=task.status,
            assigned_to=task.assigned_to,
            assigned_by=task.assigned_by,
            is_assigned_to_user=task.is_assigned_to_user,
            extraction_method=task.extraction_method,
            completed_at=task.completed_at,
            created_at=task.created_at,
            user_id=task.user_id,
            source_text=task.source_text,
            project_name=project_name,
        )
