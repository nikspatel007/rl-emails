"""Tests for project API endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rl_emails.api.auth.clerk import ClerkUser
from rl_emails.api.auth.dependencies import get_current_user_or_api_key
from rl_emails.api.routes.projects import router, set_session_factory
from rl_emails.schemas.project import (
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
)


@pytest.fixture
def mock_user() -> ClerkUser:
    """Create a mock authenticated user."""
    return ClerkUser(
        id="550e8400-e29b-41d4-a716-446655440000",  # Valid UUID format
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

    # Override authentication
    async def override_get_current_user() -> ClerkUser:
        return mock_user

    app.dependency_overrides[get_current_user_or_api_key] = override_get_current_user

    # Create a mock async context manager
    @asynccontextmanager
    async def mock_session_cm() -> AsyncGenerator[mock.MagicMock, None]:
        yield mock_session

    set_session_factory(mock_session_cm)  # type: ignore[arg-type]

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestListProjects:
    """Tests for list_projects endpoint."""

    def test_list_projects(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test listing projects."""
        # Mock the service
        now = datetime.now(UTC)
        mock_response = ProjectListResponse(
            projects=[
                ProjectResponse(id=1, name="Project 1", created_at=now),
                ProjectResponse(id=2, name="Project 2", created_at=now),
            ],
            total=2,
            limit=100,
            offset=0,
            has_more=False,
        )

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.list_projects = mock.AsyncMock(return_value=mock_response)

            response = client.get("/projects")

            assert response.status_code == 200
            data = response.json()
            assert len(data["projects"]) == 2
            assert data["total"] == 2

    def test_list_projects_with_filters(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test listing projects with filters."""
        mock_response = ProjectListResponse(
            projects=[],
            total=0,
            limit=50,
            offset=10,
            has_more=False,
        )

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.list_projects = mock.AsyncMock(return_value=mock_response)

            response = client.get("/projects?is_active=true&limit=50&offset=10")

            assert response.status_code == 200
            mock_service.list_projects.assert_called_once()


class TestGetProject:
    """Tests for get_project endpoint."""

    def test_get_project_success(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting a project."""
        now = datetime.now(UTC)
        mock_response = ProjectDetailResponse(
            id=1,
            name="Test Project",
            created_at=now,
            description="A test project",
        )

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_project = mock.AsyncMock(return_value=mock_response)

            response = client.get("/projects/1")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Project"

    def test_get_project_not_found(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting a non-existent project."""
        from rl_emails.services.project_service import ProjectNotFoundError

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_project = mock.AsyncMock(side_effect=ProjectNotFoundError(999))

            response = client.get("/projects/999")

            assert response.status_code == 404
            assert "999" in response.json()["detail"]


class TestCreateProject:
    """Tests for create_project endpoint."""

    def test_create_project(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test creating a project."""
        now = datetime.now(UTC)
        mock_response = ProjectResponse(id=1, name="New Project", created_at=now)

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.create_project = mock.AsyncMock(return_value=mock_response)

            response = client.post(
                "/projects",
                json={"name": "New Project", "description": "A new project"},
            )

            assert response.status_code == 201
            data = response.json()
            assert data["name"] == "New Project"


class TestUpdateProject:
    """Tests for update_project endpoint."""

    def test_update_project(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test updating a project."""
        now = datetime.now(UTC)
        mock_response = ProjectResponse(id=1, name="Updated Project", created_at=now)

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.update_project = mock.AsyncMock(return_value=mock_response)

            response = client.patch(
                "/projects/1",
                json={"name": "Updated Project"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated Project"


class TestDeleteProject:
    """Tests for delete_project endpoint."""

    def test_delete_project(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test deleting a project."""
        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.delete_project = mock.AsyncMock(return_value=True)

            response = client.delete("/projects/1")

            assert response.status_code == 204

    def test_delete_project_not_found(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test deleting a non-existent project."""
        from rl_emails.services.project_service import ProjectNotFoundError

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.delete_project = mock.AsyncMock(side_effect=ProjectNotFoundError(999))

            response = client.delete("/projects/999")

            assert response.status_code == 404


class TestUpdateProjectNotFound:
    """Tests for update_project not found."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock database session."""
        return mock.MagicMock()

    @pytest.fixture
    def mock_user(self) -> ClerkUser:
        """Create a mock authenticated user."""
        return ClerkUser(id="550e8400-e29b-41d4-a716-446655440000")

    @pytest.fixture
    def app(self, mock_user: ClerkUser, mock_session: mock.MagicMock) -> FastAPI:
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
    def client(self, app: FastAPI) -> TestClient:
        """Create a test client."""
        return TestClient(app)

    def test_update_project_not_found(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test updating a non-existent project."""
        from rl_emails.services.project_service import ProjectNotFoundError

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.update_project = mock.AsyncMock(side_effect=ProjectNotFoundError(999))

            response = client.patch("/projects/999", json={"name": "Updated"})

            assert response.status_code == 404
            assert "999" in response.json()["detail"]


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

    def test_list_projects_with_clerk_user_id(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test listing projects with Clerk-style user ID."""
        now = datetime.now(UTC)
        mock_response = ProjectListResponse(
            projects=[
                ProjectResponse(id=1, name="Project 1", created_at=now),
            ],
            total=1,
            limit=100,
            offset=0,
            has_more=False,
        )

        with mock.patch("rl_emails.api.routes.projects.ProjectService") as MockService:
            mock_service = MockService.return_value
            mock_service.list_projects = mock.AsyncMock(return_value=mock_response)

            response = client.get("/projects")

            assert response.status_code == 200
            # Verify service was called (user ID was converted)
            mock_service.list_projects.assert_called_once()


class TestDatabaseNotConfigured:
    """Tests for database not configured scenario."""

    def test_service_not_configured(self) -> None:
        """Test error when database not configured."""
        app = FastAPI()
        app.include_router(router)

        # Override auth but not session factory
        async def override_get_current_user() -> ClerkUser:
            return ClerkUser(id="550e8400-e29b-41d4-a716-446655440000")

        app.dependency_overrides[get_current_user_or_api_key] = override_get_current_user

        # Reset the session factory
        set_session_factory(None)  # type: ignore[arg-type]

        client = TestClient(app)
        response = client.get("/projects")

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
