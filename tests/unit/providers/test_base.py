"""Tests for provider base types and abstract classes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from rl_emails.providers.base import (
    AuthorizationError,
    ConnectionError,
    ConnectionState,
    ConnectionStatus,
    EmailProvider,
    ProviderError,
    ProviderType,
    SyncError,
    SyncProgress,
)

if TYPE_CHECKING:
    from rl_emails.core.types import EmailData


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_gmail_value(self) -> None:
        """Test Gmail provider type value."""
        assert ProviderType.GMAIL.value == "gmail"

    def test_string_enum(self) -> None:
        """Test ProviderType is a string enum."""
        assert isinstance(ProviderType.GMAIL, str)
        assert ProviderType.GMAIL == "gmail"

    def test_from_string(self) -> None:
        """Test creating from string value."""
        provider = ProviderType("gmail")
        assert provider == ProviderType.GMAIL

    def test_invalid_provider(self) -> None:
        """Test invalid provider value raises error."""
        with pytest.raises(ValueError):
            ProviderType("invalid")


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_all_states(self) -> None:
        """Test all connection states exist."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.ERROR.value == "error"

    def test_string_enum(self) -> None:
        """Test ConnectionState is a string enum."""
        assert isinstance(ConnectionState.CONNECTED, str)
        assert ConnectionState.CONNECTED == "connected"


class TestConnectionStatus:
    """Tests for ConnectionStatus dataclass."""

    def test_minimal_status(self) -> None:
        """Test creating status with required fields only."""
        status = ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.DISCONNECTED,
        )

        assert status.provider == ProviderType.GMAIL
        assert status.state == ConnectionState.DISCONNECTED
        assert status.email is None
        assert status.connected_at is None
        assert status.last_sync is None
        assert status.error is None
        assert status.metadata == {}

    def test_full_status(self) -> None:
        """Test creating status with all fields."""
        now = datetime.now(UTC)
        metadata = {"scopes": ["gmail.readonly"]}

        status = ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTED,
            email="test@example.com",
            connected_at=now,
            last_sync=now,
            error=None,
            metadata=metadata,
        )

        assert status.provider == ProviderType.GMAIL
        assert status.state == ConnectionState.CONNECTED
        assert status.email == "test@example.com"
        assert status.connected_at == now
        assert status.last_sync == now
        assert status.error is None
        assert status.metadata == metadata

    def test_is_connected_true(self) -> None:
        """Test is_connected returns True when connected."""
        status = ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTED,
        )

        assert status.is_connected is True

    def test_is_connected_false_disconnected(self) -> None:
        """Test is_connected returns False when disconnected."""
        status = ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.DISCONNECTED,
        )

        assert status.is_connected is False

    def test_is_connected_false_error(self) -> None:
        """Test is_connected returns False when in error state."""
        status = ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.ERROR,
            error="Token expired",
        )

        assert status.is_connected is False

    def test_is_connected_false_connecting(self) -> None:
        """Test is_connected returns False when connecting."""
        status = ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTING,
        )

        assert status.is_connected is False

    def test_frozen(self) -> None:
        """Test ConnectionStatus is immutable."""
        status = ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTED,
        )

        with pytest.raises(FrozenInstanceError):
            status.state = ConnectionState.DISCONNECTED  # type: ignore[misc]

    def test_slots(self) -> None:
        """Test ConnectionStatus uses slots."""
        status = ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTED,
        )

        assert hasattr(status, "__slots__") or not hasattr(status, "__dict__")


class TestSyncProgress:
    """Tests for SyncProgress dataclass."""

    def test_basic_progress(self) -> None:
        """Test creating basic progress."""
        progress = SyncProgress(processed=50, total=100)

        assert progress.processed == 50
        assert progress.total == 100
        assert progress.current_phase == "syncing"

    def test_with_phase(self) -> None:
        """Test creating progress with custom phase."""
        progress = SyncProgress(
            processed=10,
            total=50,
            current_phase="fetching",
        )

        assert progress.current_phase == "fetching"

    def test_mutable(self) -> None:
        """Test SyncProgress is mutable (not frozen)."""
        progress = SyncProgress(processed=0, total=100)

        progress.processed = 50
        assert progress.processed == 50


class TestProviderError:
    """Tests for ProviderError exception."""

    def test_basic_error(self) -> None:
        """Test creating basic error."""
        error = ProviderError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.provider is None

    def test_with_provider(self) -> None:
        """Test creating error with provider."""
        error = ProviderError(
            "Gmail error",
            provider=ProviderType.GMAIL,
        )

        assert str(error) == "Gmail error"
        assert error.provider == ProviderType.GMAIL

    def test_inheritance(self) -> None:
        """Test ProviderError inherits from Exception."""
        assert issubclass(ProviderError, Exception)


class TestConnectionError:
    """Tests for ConnectionError exception."""

    def test_inheritance(self) -> None:
        """Test ConnectionError inherits from ProviderError."""
        assert issubclass(ConnectionError, ProviderError)

    def test_with_provider(self) -> None:
        """Test creating connection error with provider."""
        error = ConnectionError(
            "Connection failed",
            provider=ProviderType.GMAIL,
        )

        assert str(error) == "Connection failed"
        assert error.provider == ProviderType.GMAIL


class TestSyncError:
    """Tests for SyncError exception."""

    def test_inheritance(self) -> None:
        """Test SyncError inherits from ProviderError."""
        assert issubclass(SyncError, ProviderError)

    def test_with_provider(self) -> None:
        """Test creating sync error with provider."""
        error = SyncError(
            "Sync failed",
            provider=ProviderType.GMAIL,
        )

        assert str(error) == "Sync failed"
        assert error.provider == ProviderType.GMAIL


class TestAuthorizationError:
    """Tests for AuthorizationError exception."""

    def test_inheritance(self) -> None:
        """Test AuthorizationError inherits from ProviderError."""
        assert issubclass(AuthorizationError, ProviderError)

    def test_with_provider(self) -> None:
        """Test creating authorization error with provider."""
        error = AuthorizationError(
            "Auth failed",
            provider=ProviderType.GMAIL,
        )

        assert str(error) == "Auth failed"
        assert error.provider == ProviderType.GMAIL


class ConcreteProvider(EmailProvider):
    """Concrete implementation for testing abstract base class."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GMAIL

    async def get_auth_url(self, state: str | None = None) -> str:
        return f"https://auth.example.com?state={state}"

    async def complete_auth(self, user_id: UUID, code: str) -> ConnectionStatus:
        return ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTED,
        )

    async def disconnect(self, user_id: UUID) -> bool:
        return True

    async def get_status(self, user_id: UUID) -> ConnectionStatus:
        return ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTED,
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
        return None


class TestEmailProvider:
    """Tests for EmailProvider abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Test cannot instantiate abstract class directly."""
        with pytest.raises(TypeError):
            EmailProvider()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_concrete_implementation(self) -> None:
        """Test concrete implementation works."""
        provider = ConcreteProvider()

        assert provider.provider_type == ProviderType.GMAIL

        url = await provider.get_auth_url(state="test")
        assert "test" in url

        user_id = UUID("12345678-1234-5678-1234-567812345678")

        status = await provider.complete_auth(user_id, "code123")
        assert status.state == ConnectionState.CONNECTED

        disconnected = await provider.disconnect(user_id)
        assert disconnected is True

        current_status = await provider.get_status(user_id)
        assert current_status.is_connected

        progress = await provider.get_sync_progress(user_id)
        assert progress is None

    @pytest.mark.asyncio
    async def test_sync_messages_generator(self) -> None:
        """Test sync_messages returns async iterator."""
        provider = ConcreteProvider()
        user_id = UUID("12345678-1234-5678-1234-567812345678")

        messages = []
        async for msg in provider.sync_messages(user_id):
            messages.append(msg)

        assert messages == []
