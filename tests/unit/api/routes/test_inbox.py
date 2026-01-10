"""Tests for inbox API endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rl_emails.api.auth.clerk import ClerkUser
from rl_emails.api.auth.dependencies import get_current_user_or_api_key
from rl_emails.api.routes.inbox import router, set_session_factory
from rl_emails.schemas.inbox import (
    EmailSummary,
    InboxStats,
    PriorityEmail,
    PriorityInboxResponse,
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


class TestGetPriorityInbox:
    """Tests for get_priority_inbox endpoint."""

    def test_get_priority_inbox(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting priority inbox."""
        now = datetime.now(UTC)
        mock_response = PriorityInboxResponse(
            emails=[
                PriorityEmail(
                    email=EmailSummary(
                        id=1,
                        message_id="msg-1",
                        subject="Test Email",
                        from_email="sender@example.com",
                        date_parsed=now,
                    ),
                    priority_rank=1,
                    priority_score=0.9,
                    task_count=2,
                ),
            ],
            total=1,
            limit=20,
            offset=0,
            has_more=False,
            pending_tasks=5,
            urgent_count=2,
            from_real_people_count=10,
        )

        with mock.patch("rl_emails.api.routes.inbox.InboxService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_priority_inbox = mock.AsyncMock(return_value=mock_response)

            response = client.get("/inbox")

            assert response.status_code == 200
            data = response.json()
            assert len(data["emails"]) == 1
            assert data["total"] == 1
            assert data["pending_tasks"] == 5

    def test_get_priority_inbox_with_pagination(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test getting priority inbox with pagination."""
        mock_response = PriorityInboxResponse(
            emails=[],
            total=100,
            limit=10,
            offset=20,
            has_more=True,
            pending_tasks=0,
            urgent_count=0,
            from_real_people_count=0,
        )

        with mock.patch("rl_emails.api.routes.inbox.InboxService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_priority_inbox = mock.AsyncMock(return_value=mock_response)

            response = client.get("/inbox?limit=10&offset=20")

            assert response.status_code == 200
            data = response.json()
            assert data["limit"] == 10
            assert data["offset"] == 20
            assert data["has_more"] is True


class TestGetInboxStats:
    """Tests for get_inbox_stats endpoint."""

    def test_get_inbox_stats(self, client: TestClient, mock_session: mock.MagicMock) -> None:
        """Test getting inbox stats."""
        mock_response = InboxStats(
            total_emails=100,
            unread_count=10,
            pending_tasks=5,
            urgent_emails=3,
            from_real_people=50,
            avg_priority_score=0.6,
            oldest_unanswered_hours=48.0,
        )

        with mock.patch("rl_emails.api.routes.inbox.InboxService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_inbox_stats = mock.AsyncMock(return_value=mock_response)

            response = client.get("/inbox/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["total_emails"] == 100
            assert data["pending_tasks"] == 5
            assert data["urgent_emails"] == 3
            assert data["from_real_people"] == 50


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

    def test_get_priority_inbox_with_clerk_user_id(
        self, client: TestClient, mock_session: mock.MagicMock
    ) -> None:
        """Test getting priority inbox with Clerk-style user ID."""
        mock_response = PriorityInboxResponse(
            emails=[],
            total=0,
            limit=20,
            offset=0,
            has_more=False,
            pending_tasks=0,
            urgent_count=0,
            from_real_people_count=0,
        )

        with mock.patch("rl_emails.api.routes.inbox.InboxService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_priority_inbox = mock.AsyncMock(return_value=mock_response)

            response = client.get("/inbox")

            assert response.status_code == 200
            # Verify service was called (user ID was converted)
            mock_service.get_priority_inbox.assert_called_once()


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
        response = client.get("/inbox")

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
