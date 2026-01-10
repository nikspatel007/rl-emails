"""Tests for task API endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rl_emails.api.auth.clerk import ClerkUser
from rl_emails.api.auth.dependencies import get_current_user_or_api_key
from rl_emails.api.routes.tasks import router, set_session_factory
from rl_emails.schemas.task import (
    TaskDetailResponse,
    TaskListResponse,
    TaskResponse,
)


@pytest.fixture
def mock_user() -> ClerkUser:
    """Create a mock authenticated user."""
    return ClerkUser(
        id="550e8400-e29b-41d4-a716-446655440000",
        email="test@example.com",
        first_name="Test",
        last_name="User",
    )


@pytest.fixture
def mock_session() -> mock.MagicMock:
    """Create a mock database session."""
    return mock.MagicMock()


@pytest.fixture
def app(mock_user: ClerkUser, mock_session: mock.MagicMock) -> FastAPI:
    """Create a test FastAPI app."""
    from contextlib import asynccontextmanager

    app = FastAPI()
    app.include_router(router)

    async def override_get_current_user() -> ClerkUser:
        return mock_user

    app.dependency_overrides[get_current_user_or_api_key] = override_get_current_user

    @asynccontextmanager
    async def mock_session_cm() -> AsyncGenerator[mock.MagicMock, None]:
        yield mock_session

    set_session_factory(mock_session_cm)  # type: ignore[arg-type]

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestListTasks:
    """Tests for list_tasks endpoint."""

    def test_list_tasks(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test listing tasks."""
        now = datetime.now(UTC)
        mock_response = TaskListResponse(
            tasks=[
                TaskResponse(id=1, task_id="task_1", description="Task 1", created_at=now),
                TaskResponse(id=2, task_id="task_2", description="Task 2", created_at=now),
            ],
            total=2,
            limit=100,
            offset=0,
            has_more=False,
        )

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.list_tasks = mock.AsyncMock(return_value=mock_response)

            response = client.get("/tasks")

            assert response.status_code == 200
            data = response.json()
            assert len(data["tasks"]) == 2
            assert data["total"] == 2

    def test_list_tasks_with_filters(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test listing tasks with filters."""
        mock_response = TaskListResponse(
            tasks=[],
            total=0,
            limit=50,
            offset=10,
            has_more=False,
        )

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.list_tasks = mock.AsyncMock(return_value=mock_response)

            response = client.get("/tasks?status=pending&project_id=1&limit=50&offset=10")

            assert response.status_code == 200
            mock_service.list_tasks.assert_called_once()


class TestGetTaskStats:
    """Tests for get_task_stats endpoint."""

    def test_get_task_stats(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting task stats."""
        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.count_pending = mock.AsyncMock(return_value=5)
            mock_service.count_urgent = mock.AsyncMock(return_value=2)

            response = client.get("/tasks/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["pending"] == 5
            assert data["urgent"] == 2


class TestGetTask:
    """Tests for get_task endpoint."""

    def test_get_task_success(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting a task."""
        now = datetime.now(UTC)
        mock_response = TaskDetailResponse(
            id=1,
            task_id="task_1",
            description="Test Task",
            created_at=now,
            source_text="Original text",
        )

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_task = mock.AsyncMock(return_value=mock_response)

            response = client.get("/tasks/1")

            assert response.status_code == 200
            data = response.json()
            assert data["description"] == "Test Task"

    def test_get_task_not_found(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting a non-existent task."""
        from rl_emails.services.task_service import TaskNotFoundError

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_task = mock.AsyncMock(side_effect=TaskNotFoundError(999))

            response = client.get("/tasks/999")

            assert response.status_code == 404
            assert "999" in response.json()["detail"]


class TestCreateTask:
    """Tests for create_task endpoint."""

    def test_create_task(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test creating a task."""
        now = datetime.now(UTC)
        mock_response = TaskResponse(
            id=1,
            task_id="manual_123",
            description="New Task",
            created_at=now,
        )

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.create_task = mock.AsyncMock(return_value=mock_response)

            response = client.post(
                "/tasks",
                json={"description": "New Task", "task_type": "other"},
            )

            assert response.status_code == 201
            data = response.json()
            assert data["description"] == "New Task"


class TestUpdateTask:
    """Tests for update_task endpoint."""

    def test_update_task(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test updating a task."""
        now = datetime.now(UTC)
        mock_response = TaskResponse(
            id=1,
            task_id="task_1",
            description="Updated Task",
            created_at=now,
        )

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.update_task = mock.AsyncMock(return_value=mock_response)

            response = client.patch(
                "/tasks/1",
                json={"description": "Updated Task"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["description"] == "Updated Task"

    def test_update_task_not_found(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test updating a non-existent task."""
        from rl_emails.services.task_service import TaskNotFoundError

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.update_task = mock.AsyncMock(side_effect=TaskNotFoundError(999))

            response = client.patch(
                "/tasks/999",
                json={"description": "Updated Task"},
            )

            assert response.status_code == 404


class TestCompleteTask:
    """Tests for complete_task endpoint."""

    def test_complete_task(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test completing a task."""
        now = datetime.now(UTC)
        mock_response = TaskResponse(
            id=1,
            task_id="task_1",
            description="Task",
            status="completed",
            created_at=now,
        )

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.complete_task = mock.AsyncMock(return_value=mock_response)

            response = client.post("/tasks/1/complete")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"

    def test_complete_task_not_found(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test completing a non-existent task."""
        from rl_emails.services.task_service import TaskNotFoundError

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.complete_task = mock.AsyncMock(side_effect=TaskNotFoundError(999))

            response = client.post("/tasks/999/complete")

            assert response.status_code == 404


class TestDismissTask:
    """Tests for dismiss_task endpoint."""

    def test_dismiss_task(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test dismissing a task."""
        now = datetime.now(UTC)
        mock_response = TaskResponse(
            id=1,
            task_id="task_1",
            description="Task",
            status="dismissed",
            created_at=now,
        )

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.dismiss_task = mock.AsyncMock(return_value=mock_response)

            response = client.post("/tasks/1/dismiss")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "dismissed"

    def test_dismiss_task_not_found(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test dismissing a non-existent task."""
        from rl_emails.services.task_service import TaskNotFoundError

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.dismiss_task = mock.AsyncMock(side_effect=TaskNotFoundError(999))

            response = client.post("/tasks/999/dismiss")

            assert response.status_code == 404


class TestDeleteTask:
    """Tests for delete_task endpoint."""

    def test_delete_task(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test deleting a task."""
        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.delete_task = mock.AsyncMock(return_value=True)

            response = client.delete("/tasks/1")

            assert response.status_code == 204

    def test_delete_task_not_found(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test deleting a non-existent task."""
        from rl_emails.services.task_service import TaskNotFoundError

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.delete_task = mock.AsyncMock(side_effect=TaskNotFoundError(999))

            response = client.delete("/tasks/999")

            assert response.status_code == 404


class TestClerkUserIdHandling:
    """Tests for Clerk-style non-UUID user ID handling."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock database session."""
        return mock.MagicMock()

    @pytest.fixture
    def mock_clerk_user(self) -> ClerkUser:
        """Create a mock Clerk user with non-UUID ID."""
        return ClerkUser(
            id="user_2abc123xyz456",  # Clerk-style user ID (not a UUID)
            email="clerk@example.com",
            first_name="Clerk",
            last_name="User",
        )

    @pytest.fixture
    def app(self, mock_clerk_user: ClerkUser, mock_session: mock.MagicMock) -> FastAPI:
        """Create a test FastAPI app with Clerk user."""
        from contextlib import asynccontextmanager

        app = FastAPI()
        app.include_router(router)

        async def override_get_current_user() -> ClerkUser:
            return mock_clerk_user

        app.dependency_overrides[get_current_user_or_api_key] = override_get_current_user

        @asynccontextmanager
        async def mock_session_cm() -> AsyncGenerator[mock.MagicMock, None]:
            yield mock_session

        set_session_factory(mock_session_cm)  # type: ignore[arg-type]

        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create a test client."""
        return TestClient(app)

    def test_list_tasks_with_clerk_user_id(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test listing tasks with Clerk-style user ID."""
        now = datetime.now(UTC)
        mock_response = TaskListResponse(
            tasks=[
                TaskResponse(id=1, task_id="task_1", description="Task 1", created_at=now),
            ],
            total=1,
            limit=100,
            offset=0,
            has_more=False,
        )

        with mock.patch("rl_emails.api.routes.tasks.TaskService") as MockService:
            mock_service = MockService.return_value
            mock_service.list_tasks = mock.AsyncMock(return_value=mock_response)

            response = client.get("/tasks")

            assert response.status_code == 200
            # Verify service was called (user ID was converted)
            mock_service.list_tasks.assert_called_once()


class TestDatabaseNotConfigured:
    """Tests for database not configured scenario."""

    def test_service_not_configured(self) -> None:
        """Test error when database not configured."""
        app = FastAPI()
        app.include_router(router)

        async def override_get_current_user() -> ClerkUser:
            return ClerkUser(id="550e8400-e29b-41d4-a716-446655440000")

        app.dependency_overrides[get_current_user_or_api_key] = override_get_current_user

        set_session_factory(None)  # type: ignore[arg-type]

        client = TestClient(app)
        response = client.get("/tasks")

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
