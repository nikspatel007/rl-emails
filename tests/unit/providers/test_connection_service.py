"""Tests for ConnectionService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest import mock
from uuid import UUID

import pytest

from rl_emails.providers.base import (
    ConnectionState,
    ConnectionStatus,
    EmailProvider,
    ProviderType,
    SyncProgress,
)
from rl_emails.providers.connection_service import ConnectionService
from rl_emails.providers.registry import ProviderNotFoundError, ProviderRegistry

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
        self._sync_progress: SyncProgress | None = None

    @property
    def provider_type(self) -> ProviderType:
        return self._provider_type

    async def get_auth_url(self, state: str | None = None) -> str:
        return f"https://auth.example.com?state={state or ''}"

    async def complete_auth(self, user_id: UUID, code: str) -> ConnectionStatus:
        self._connected = True
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
        # Empty generator for testing
        return
        yield  # type: ignore[misc]

    async def get_sync_progress(self, user_id: UUID) -> SyncProgress | None:
        return self._sync_progress

    def set_sync_progress(self, progress: SyncProgress | None) -> None:
        """Helper to set sync progress for testing."""
        self._sync_progress = progress


@pytest.fixture
def registry() -> ProviderRegistry:
    """Create a fresh registry."""
    return ProviderRegistry()


@pytest.fixture
def mock_provider() -> MockProvider:
    """Create a mock provider."""
    return MockProvider()


@pytest.fixture
def service(registry: ProviderRegistry, mock_provider: MockProvider) -> ConnectionService:
    """Create a connection service with registered mock provider."""
    registry.register(mock_provider)
    return ConnectionService(registry)


@pytest.fixture
def user_id() -> UUID:
    """Create a test user ID."""
    return UUID("12345678-1234-5678-1234-567812345678")


class TestConnectionServiceInit:
    """Tests for ConnectionService initialization."""

    def test_init_with_registry(self) -> None:
        """Test initializing with registry."""
        registry = ProviderRegistry()
        service = ConnectionService(registry)

        assert service._registry == registry


class TestGetAuthUrl:
    """Tests for get_auth_url method."""

    @pytest.mark.asyncio
    async def test_get_auth_url(self, service: ConnectionService) -> None:
        """Test getting auth URL."""
        url = await service.get_auth_url(ProviderType.GMAIL)

        assert "auth.example.com" in url

    @pytest.mark.asyncio
    async def test_get_auth_url_with_state(self, service: ConnectionService) -> None:
        """Test getting auth URL with state."""
        url = await service.get_auth_url(ProviderType.GMAIL, state="csrf123")

        assert "csrf123" in url

    @pytest.mark.asyncio
    async def test_get_auth_url_unknown_provider(self, registry: ProviderRegistry) -> None:
        """Test get_auth_url with unregistered provider."""
        service = ConnectionService(registry)

        with pytest.raises(ProviderNotFoundError):
            await service.get_auth_url(ProviderType.GMAIL)


class TestCompleteAuth:
    """Tests for complete_auth method."""

    @pytest.mark.asyncio
    async def test_complete_auth(self, service: ConnectionService, user_id: UUID) -> None:
        """Test completing authorization."""
        status = await service.complete_auth(user_id, ProviderType.GMAIL, "auth_code")

        assert status.state == ConnectionState.CONNECTED
        assert status.connected_at is not None

    @pytest.mark.asyncio
    async def test_complete_auth_unknown_provider(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test complete_auth with unregistered provider."""
        service = ConnectionService(registry)

        with pytest.raises(ProviderNotFoundError):
            await service.complete_auth(user_id, ProviderType.GMAIL, "code")


class TestDisconnect:
    """Tests for disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect_connected(self, service: ConnectionService, user_id: UUID) -> None:
        """Test disconnecting when connected."""
        result = await service.disconnect(user_id, ProviderType.GMAIL)

        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test disconnecting when not connected."""
        provider = MockProvider(connected=False)
        registry.register(provider)
        service = ConnectionService(registry)

        result = await service.disconnect(user_id, ProviderType.GMAIL)

        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_unknown_provider(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test disconnect with unregistered provider."""
        service = ConnectionService(registry)

        with pytest.raises(ProviderNotFoundError):
            await service.disconnect(user_id, ProviderType.GMAIL)


class TestGetStatus:
    """Tests for get_status method."""

    @pytest.mark.asyncio
    async def test_get_status_connected(self, service: ConnectionService, user_id: UUID) -> None:
        """Test getting status when connected."""
        status = await service.get_status(user_id, ProviderType.GMAIL)

        assert status.state == ConnectionState.CONNECTED
        assert status.is_connected is True

    @pytest.mark.asyncio
    async def test_get_status_disconnected(self, registry: ProviderRegistry, user_id: UUID) -> None:
        """Test getting status when disconnected."""
        provider = MockProvider(connected=False)
        registry.register(provider)
        service = ConnectionService(registry)

        status = await service.get_status(user_id, ProviderType.GMAIL)

        assert status.state == ConnectionState.DISCONNECTED
        assert status.is_connected is False

    @pytest.mark.asyncio
    async def test_get_status_unknown_provider(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test get_status with unregistered provider."""
        service = ConnectionService(registry)

        with pytest.raises(ProviderNotFoundError):
            await service.get_status(user_id, ProviderType.GMAIL)


class TestGetAllStatuses:
    """Tests for get_all_statuses method."""

    @pytest.mark.asyncio
    async def test_get_all_statuses(self, service: ConnectionService, user_id: UUID) -> None:
        """Test getting all statuses."""
        statuses = await service.get_all_statuses(user_id)

        assert ProviderType.GMAIL in statuses
        assert statuses[ProviderType.GMAIL].is_connected

    @pytest.mark.asyncio
    async def test_get_all_statuses_empty(self, user_id: UUID) -> None:
        """Test getting all statuses with no providers."""
        registry = ProviderRegistry()
        service = ConnectionService(registry)

        statuses = await service.get_all_statuses(user_id)

        assert statuses == {}

    @pytest.mark.asyncio
    async def test_get_all_statuses_handles_errors(self, user_id: UUID) -> None:
        """Test get_all_statuses handles provider errors gracefully."""
        registry = ProviderRegistry()
        provider = mock.MagicMock(spec=EmailProvider)
        provider.provider_type = ProviderType.GMAIL
        provider.get_status = mock.AsyncMock(side_effect=Exception("Test error"))

        registry.register(provider)
        service = ConnectionService(registry)

        statuses = await service.get_all_statuses(user_id)

        assert ProviderType.GMAIL in statuses
        assert statuses[ProviderType.GMAIL].state == ConnectionState.ERROR
        assert "Test error" in str(statuses[ProviderType.GMAIL].error)


class TestSyncMessages:
    """Tests for sync_messages method."""

    @pytest.mark.asyncio
    async def test_sync_messages(self, service: ConnectionService, user_id: UUID) -> None:
        """Test syncing messages."""
        messages = []
        async for msg in service.sync_messages(user_id, ProviderType.GMAIL):
            messages.append(msg)

        assert messages == []

    @pytest.mark.asyncio
    async def test_sync_messages_with_params(
        self, service: ConnectionService, user_id: UUID
    ) -> None:
        """Test syncing messages with parameters."""
        messages = []
        async for msg in service.sync_messages(
            user_id, ProviderType.GMAIL, days=7, max_messages=100
        ):
            messages.append(msg)

        assert messages == []

    @pytest.mark.asyncio
    async def test_sync_messages_unknown_provider(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test sync_messages with unregistered provider."""
        service = ConnectionService(registry)

        with pytest.raises(ProviderNotFoundError):
            async for _ in service.sync_messages(user_id, ProviderType.GMAIL):
                pass


class TestGetSyncProgress:
    """Tests for get_sync_progress method."""

    @pytest.mark.asyncio
    async def test_get_sync_progress_none(self, service: ConnectionService, user_id: UUID) -> None:
        """Test getting sync progress when not syncing."""
        progress = await service.get_sync_progress(user_id, ProviderType.GMAIL)

        assert progress is None

    @pytest.mark.asyncio
    async def test_get_sync_progress_active(
        self,
        registry: ProviderRegistry,
        user_id: UUID,
    ) -> None:
        """Test getting sync progress during sync."""
        provider = MockProvider()
        provider.set_sync_progress(SyncProgress(processed=50, total=100, current_phase="fetching"))
        registry.register(provider)
        service = ConnectionService(registry)

        progress = await service.get_sync_progress(user_id, ProviderType.GMAIL)

        assert progress is not None
        assert progress.processed == 50
        assert progress.total == 100
        assert progress.current_phase == "fetching"

    @pytest.mark.asyncio
    async def test_get_sync_progress_unknown_provider(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test get_sync_progress with unregistered provider."""
        service = ConnectionService(registry)

        with pytest.raises(ProviderNotFoundError):
            await service.get_sync_progress(user_id, ProviderType.GMAIL)


class TestListAvailableProviders:
    """Tests for list_available_providers method."""

    def test_list_available_providers(self, service: ConnectionService) -> None:
        """Test listing available providers."""
        providers = service.list_available_providers()

        assert providers == [ProviderType.GMAIL]

    def test_list_available_providers_empty(self) -> None:
        """Test listing available providers when empty."""
        registry = ProviderRegistry()
        service = ConnectionService(registry)

        providers = service.list_available_providers()

        assert providers == []


class TestIsConnected:
    """Tests for is_connected method."""

    @pytest.mark.asyncio
    async def test_is_connected_true(self, service: ConnectionService, user_id: UUID) -> None:
        """Test is_connected returns True when connected."""
        result = await service.is_connected(user_id, ProviderType.GMAIL)

        assert result is True

    @pytest.mark.asyncio
    async def test_is_connected_false_disconnected(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test is_connected returns False when disconnected."""
        provider = MockProvider(connected=False)
        registry.register(provider)
        service = ConnectionService(registry)

        result = await service.is_connected(user_id, ProviderType.GMAIL)

        assert result is False

    @pytest.mark.asyncio
    async def test_is_connected_false_not_registered(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test is_connected returns False when provider not registered."""
        service = ConnectionService(registry)

        result = await service.is_connected(user_id, ProviderType.GMAIL)

        assert result is False

    @pytest.mark.asyncio
    async def test_is_connected_handles_provider_error(self, user_id: UUID) -> None:
        """Test is_connected handles ProviderError gracefully."""
        from rl_emails.providers.base import ProviderError

        registry = ProviderRegistry()
        provider = mock.MagicMock(spec=EmailProvider)
        provider.provider_type = ProviderType.GMAIL
        provider.get_status = mock.AsyncMock(side_effect=ProviderError("Error"))

        registry.register(provider)
        service = ConnectionService(registry)

        result = await service.is_connected(user_id, ProviderType.GMAIL)

        assert result is False


class TestGetConnectedProviders:
    """Tests for get_connected_providers method."""

    @pytest.mark.asyncio
    async def test_get_connected_providers(self, service: ConnectionService, user_id: UUID) -> None:
        """Test getting connected providers."""
        providers = await service.get_connected_providers(user_id)

        assert providers == [ProviderType.GMAIL]

    @pytest.mark.asyncio
    async def test_get_connected_providers_none_connected(
        self, registry: ProviderRegistry, user_id: UUID
    ) -> None:
        """Test getting connected providers when none connected."""
        provider = MockProvider(connected=False)
        registry.register(provider)
        service = ConnectionService(registry)

        providers = await service.get_connected_providers(user_id)

        assert providers == []

    @pytest.mark.asyncio
    async def test_get_connected_providers_empty_registry(self, user_id: UUID) -> None:
        """Test getting connected providers with empty registry."""
        registry = ProviderRegistry()
        service = ConnectionService(registry)

        providers = await service.get_connected_providers(user_id)

        assert providers == []
