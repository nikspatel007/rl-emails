"""Task repository for database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.task import Task
from rl_emails.schemas.task import TaskCreate, TaskUpdate


class TaskRepository:
    """Repository for task database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def create(self, user_id: UUID, data: TaskCreate) -> Task:
        """Create a new task.

        Args:
            user_id: User UUID.
            data: Task creation data.

        Returns:
            Created task.
        """
        # Generate task_id if not provided
        task_id = f"manual_{user_id}_{datetime.now(UTC).timestamp()}"
        task = Task(user_id=user_id, task_id=task_id, **data.model_dump())
        self.session.add(task)
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def get_by_id(self, task_id: int, user_id: UUID | None = None) -> Task | None:
        """Get task by ID.

        Args:
            task_id: Task ID.
            user_id: Optional user UUID to scope query.

        Returns:
            Task if found, None otherwise.
        """
        conditions = [Task.id == task_id]
        if user_id is not None:
            conditions.append(Task.user_id == user_id)

        result = await self.session.execute(select(Task).where(and_(*conditions)))
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: UUID,
        *,
        status: str | None = None,
        project_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Task], int]:
        """List tasks for a user.

        Args:
            user_id: User UUID.
            status: Filter by status.
            project_id: Filter by project.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (tasks list, total count).
        """
        conditions = [Task.user_id == user_id]
        if status is not None:
            conditions.append(Task.status == status)
        if project_id is not None:
            conditions.append(Task.project_id == project_id)

        # Get total count
        count_query = select(func.count()).select_from(Task).where(and_(*conditions))
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        # Get tasks
        query = (
            select(Task)
            .where(and_(*conditions))
            .order_by(Task.urgency_score.desc().nulls_last(), Task.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        tasks = list(result.scalars().all())

        return tasks, total

    async def list_all(
        self,
        *,
        status: str | None = None,
        project_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Task], int]:
        """List all tasks (for non-multi-tenant mode).

        Args:
            status: Filter by status.
            project_id: Filter by project.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (tasks list, total count).
        """
        conditions = []
        if status is not None:
            conditions.append(Task.status == status)
        if project_id is not None:
            conditions.append(Task.project_id == project_id)

        # Get total count
        count_query = select(func.count()).select_from(Task)
        if conditions:
            count_query = count_query.where(and_(*conditions))
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        # Get tasks
        query = select(Task).order_by(
            Task.urgency_score.desc().nulls_last(), Task.created_at.desc()
        )
        if conditions:
            query = query.where(and_(*conditions))
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        tasks = list(result.scalars().all())

        return tasks, total

    async def update(
        self, task_id: int, data: TaskUpdate, user_id: UUID | None = None
    ) -> Task | None:
        """Update a task.

        Args:
            task_id: Task ID.
            data: Update data.
            user_id: Optional user UUID to scope query.

        Returns:
            Updated task if found, None otherwise.
        """
        task = await self.get_by_id(task_id, user_id)
        if task is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(task, key, value)

        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def complete(self, task_id: int, user_id: UUID | None = None) -> Task | None:
        """Mark a task as completed.

        Args:
            task_id: Task ID.
            user_id: Optional user UUID to scope query.

        Returns:
            Updated task if found, None otherwise.
        """
        task = await self.get_by_id(task_id, user_id)
        if task is None:
            return None

        task.status = "completed"
        task.completed_at = datetime.now(UTC)
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def dismiss(self, task_id: int, user_id: UUID | None = None) -> Task | None:
        """Mark a task as dismissed.

        Args:
            task_id: Task ID.
            user_id: Optional user UUID to scope query.

        Returns:
            Updated task if found, None otherwise.
        """
        task = await self.get_by_id(task_id, user_id)
        if task is None:
            return None

        task.status = "dismissed"
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def delete(self, task_id: int, user_id: UUID | None = None) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID.
            user_id: Optional user UUID to scope query.

        Returns:
            True if deleted, False if not found.
        """
        task = await self.get_by_id(task_id, user_id)
        if task is None:
            return False

        await self.session.delete(task)
        await self.session.commit()
        return True

    async def count_by_user(self, user_id: UUID, *, status: str | None = None) -> int:
        """Count tasks for a user.

        Args:
            user_id: User UUID.
            status: Filter by status.

        Returns:
            Count of tasks.
        """
        conditions = [Task.user_id == user_id]
        if status is not None:
            conditions.append(Task.status == status)

        query = select(func.count()).select_from(Task).where(and_(*conditions))
        result = await self.session.execute(query)
        return result.scalar() or 0

    async def count_urgent(self, user_id: UUID) -> int:
        """Count urgent pending tasks for a user.

        Args:
            user_id: User UUID.

        Returns:
            Count of urgent pending tasks.
        """
        conditions = [
            Task.user_id == user_id,
            Task.status == "pending",
            Task.urgency_score > 0.7,
        ]

        query = select(func.count()).select_from(Task).where(and_(*conditions))
        result = await self.session.execute(query)
        return result.scalar() or 0
