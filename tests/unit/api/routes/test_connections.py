"""Tests for connection endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest import mock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rl_emails.api.auth.clerk import ClerkUser
from rl_emails.api.auth.dependencies import get_current_user
from rl_emails.api.routes.connections import (
    router,
    set_connection_service,
)
from rl_emails.providers import (
    ConnectionService,
    ConnectionState,
    ConnectionStatus,
    EmailProvider,
    ProviderRegistry,
    ProviderType,
    SyncProgress,
)

if TYPE_CHECKING:
    from rl_emails.core.types import EmailData


class MockProvider(EmailProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        provider_type: ProviderType = ProviderType.GMAIL,
        *,
        connected: bool = True,
    ) -> None:
        self._provider_type = provider_type
        self._connected = connected

    @property
    def provider_type(self) -> ProviderType:
        return self._provider_type

    async def get_auth_url(self, state: str | None = None) -> str:
        return f"https://auth.example.com?state={state or ''}"

    async def complete_auth(self, user_id: UUID, code: str) -> ConnectionStatus:
        return ConnectionStatus(
            provider=self._provider_type,
            state=ConnectionState.CONNECTED,
            connected_at=datetime.now(UTC),
        )

    async def disconnect(self, user_id: UUID) -> bool:
        if self._connected:
            self._connected = False
            return True
        return False

    async def get_status(self, user_id: UUID) -> ConnectionStatus:
        if self._connected:
            return ConnectionStatus(
                provider=self._provider_type,
                state=ConnectionState.CONNECTED,
                email="test@example.com",
                connected_at=datetime.now(UTC),
            )
        return ConnectionStatus(
            provider=self._provider_type,
            state=ConnectionState.DISCONNECTED,
        )

    async def sync_messages(
        self,
        user_id: UUID,
        days: int | None = None,
        max_messages: int | None = None,
    ) -> AsyncIterator[EmailData]:
        return
        yield  # type: ignore[misc]

    async def get_sync_progress(self, user_id: UUID) -> SyncProgress | None:
        return None


@pytest.fixture
def mock_user() -> ClerkUser:
    """Create a mock authenticated user."""
    return ClerkUser(
        id="user_test123",
        email="test@example.com",
        first_name="Test",
        last_name="User",
    )


@pytest.fixture
def mock_service() -> ConnectionService:
    """Create a mock connection service."""
    registry = ProviderRegistry()
    registry.register(MockProvider())
    return ConnectionService(registry)


@pytest.fixture
def app(mock_user: ClerkUser, mock_service: ConnectionService) -> FastAPI:
    """Create a test FastAPI app."""
    app = FastAPI()
    app.include_router(router)

    # Override authentication
    async def override_get_current_user() -> ClerkUser:
        return mock_user

    app.dependency_overrides[get_current_user] = override_get_current_user

    # Set the connection service
    set_connection_service(mock_service)

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestListConnections:
    """Tests for list_connections endpoint."""

    def test_list_connections(self, client: TestClient) -> None:
        """Test listing all connections."""
        response = client.get("/connections")

        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "connected_count" in data
        assert len(data["providers"]) == 1
        assert data["providers"][0]["provider"] == "gmail"
        assert data["connected_count"] == 1

    def test_list_connections_empty(self, mock_user: ClerkUser, client: TestClient) -> None:
        """Test listing connections with no providers."""
        registry = ProviderRegistry()
        service = ConnectionService(registry)
        set_connection_service(service)

        response = client.get("/connections")

        assert response.status_code == 200
        data = response.json()
        assert data["providers"] == []
        assert data["connected_count"] == 0


class TestGetConnectionStatus:
    """Tests for get_connection_status endpoint."""

    def test_get_status(self, client: TestClient) -> None:
        """Test getting provider status."""
        response = client.get("/connections/gmail")

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "gmail"
        assert data["state"] == "connected"
        assert data["is_connected"] is True

    def test_get_status_unknown_provider(self, client: TestClient) -> None:
        """Test getting status of unknown provider."""
        response = client.get("/connections/unknown")

        assert response.status_code == 404
        assert "Unknown provider" in response.json()["detail"]

    def test_get_status_not_registered(self, client: TestClient) -> None:
        """Test getting status when provider not registered."""
        # Use an empty registry
        registry = ProviderRegistry()
        service = ConnectionService(registry)
        set_connection_service(service)

        response = client.get("/connections/gmail")

        assert response.status_code == 404
        assert "not available" in response.json()["detail"]


class TestStartConnection:
    """Tests for start_connection endpoint."""

    def test_start_connection(self, client: TestClient) -> None:
        """Test starting OAuth flow."""
        response = client.post("/connections/gmail/connect")

        assert response.status_code == 200
        data = response.json()
        assert "auth_url" in data
        assert data["provider"] == "gmail"
        assert "auth.example.com" in data["auth_url"]

    def test_start_connection_with_state(self, client: TestClient) -> None:
        """Test starting OAuth flow with state."""
        response = client.post("/connections/gmail/connect?state=csrf123")

        assert response.status_code == 200
        data = response.json()
        assert "csrf123" in data["auth_url"]

    def test_start_connection_unknown_provider(self, client: TestClient) -> None:
        """Test starting connection for unknown provider."""
        response = client.post("/connections/unknown/connect")

        assert response.status_code == 404


class TestCompleteConnection:
    """Tests for complete_connection endpoint."""

    def test_complete_connection(self, client: TestClient) -> None:
        """Test completing OAuth flow."""
        response = client.post("/connections/gmail/callback?code=auth_code_123")

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "gmail"
        assert data["state"] == "connected"
        assert "Successfully" in data["message"]

    def test_complete_connection_unknown_provider(self, client: TestClient) -> None:
        """Test completing connection for unknown provider."""
        response = client.post("/connections/unknown/callback?code=auth_code")

        assert response.status_code == 404

    def test_complete_connection_auth_failure(
        self, mock_user: ClerkUser, client: TestClient
    ) -> None:
        """Test completing connection when auth fails."""
        # Create a provider that fails auth
        registry = ProviderRegistry()
        provider = mock.MagicMock(spec=EmailProvider)
        provider.provider_type = ProviderType.GMAIL
        provider.complete_auth = mock.AsyncMock(side_effect=Exception("Authorization failed"))
        registry.register(provider)
        service = ConnectionService(registry)
        set_connection_service(service)

        response = client.post("/connections/gmail/callback?code=bad_code")

        assert response.status_code == 400
        assert "failed" in response.json()["detail"].lower()


class TestDisconnectProvider:
    """Tests for disconnect_provider endpoint."""

    def test_disconnect(self, client: TestClient) -> None:
        """Test disconnecting provider."""
        response = client.delete("/connections/gmail")

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "gmail"
        assert data["disconnected"] is True
        assert "Disconnected" in data["message"]

    def test_disconnect_not_connected(self, mock_user: ClerkUser, client: TestClient) -> None:
        """Test disconnecting when not connected."""
        registry = ProviderRegistry()
        registry.register(MockProvider(connected=False))
        service = ConnectionService(registry)
        set_connection_service(service)

        response = client.delete("/connections/gmail")

        assert response.status_code == 200
        data = response.json()
        assert data["disconnected"] is False
        assert "not connected" in data["message"].lower()

    def test_disconnect_unknown_provider(self, client: TestClient) -> None:
        """Test disconnecting unknown provider."""
        response = client.delete("/connections/unknown")

        assert response.status_code == 404


class TestGetSyncProgress:
    """Tests for get_sync_progress endpoint."""

    def test_get_progress_not_syncing(self, client: TestClient) -> None:
        """Test getting progress when not syncing."""
        response = client.get("/connections/gmail/sync/progress")

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "gmail"
        assert data["in_progress"] is False
        assert data["processed"] is None
        assert data["total"] is None

    def test_get_progress_syncing(self, mock_user: ClerkUser, client: TestClient) -> None:
        """Test getting progress during sync."""
        registry = ProviderRegistry()
        provider = mock.MagicMock(spec=EmailProvider)
        provider.provider_type = ProviderType.GMAIL
        provider.get_sync_progress = mock.AsyncMock(
            return_value=SyncProgress(processed=50, total=100, current_phase="processing")
        )
        registry.register(provider)
        service = ConnectionService(registry)
        set_connection_service(service)

        response = client.get("/connections/gmail/sync/progress")

        assert response.status_code == 200
        data = response.json()
        assert data["in_progress"] is True
        assert data["processed"] == 50
        assert data["total"] == 100
        assert data["phase"] == "processing"

    def test_get_progress_unknown_provider(self, client: TestClient) -> None:
        """Test getting progress for unknown provider."""
        response = client.get("/connections/unknown/sync/progress")

        assert response.status_code == 404


class TestListAvailableProviders:
    """Tests for list_available_providers endpoint."""

    def test_list_available(self, client: TestClient) -> None:
        """Test listing available providers."""
        response = client.get("/connections/available")

        assert response.status_code == 200
        data = response.json()
        assert data == ["gmail"]

    def test_list_available_empty(self, mock_user: ClerkUser, client: TestClient) -> None:
        """Test listing providers when none registered."""
        registry = ProviderRegistry()
        service = ConnectionService(registry)
        set_connection_service(service)

        response = client.get("/connections/available")

        assert response.status_code == 200
        assert response.json() == []


class TestConnectionServiceDependency:
    """Tests for connection service dependency."""

    def test_service_not_configured(self) -> None:
        """Test error when service not configured."""
        app = FastAPI()
        app.include_router(router)

        # Override auth but not service
        async def override_get_current_user() -> ClerkUser:
            return ClerkUser(id="user_123")

        app.dependency_overrides[get_current_user] = override_get_current_user

        # Reset the global service
        set_connection_service(None)  # type: ignore[arg-type]

        client = TestClient(app)
        response = client.get("/connections")

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
