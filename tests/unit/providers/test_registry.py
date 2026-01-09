"""Tests for provider registry."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from rl_emails.providers.base import (
    ConnectionState,
    ConnectionStatus,
    EmailProvider,
    ProviderType,
    SyncProgress,
)
from rl_emails.providers.registry import (
    ProviderNotFoundError,
    ProviderRegistry,
    get_provider_registry,
    reset_provider_registry,
)

if TYPE_CHECKING:
    from rl_emails.core.types import EmailData


class MockProvider(EmailProvider):
    """Mock provider for testing."""

    def __init__(self, provider_type: ProviderType = ProviderType.GMAIL) -> None:
        self._provider_type = provider_type

    @property
    def provider_type(self) -> ProviderType:
        return self._provider_type

    async def get_auth_url(self, state: str | None = None) -> str:
        return f"https://auth.example.com?state={state}"

    async def complete_auth(self, user_id: UUID, code: str) -> ConnectionStatus:
        return ConnectionStatus(
            provider=self._provider_type,
            state=ConnectionState.CONNECTED,
        )

    async def disconnect(self, user_id: UUID) -> bool:
        return True

    async def get_status(self, user_id: UUID) -> ConnectionStatus:
        return ConnectionStatus(
            provider=self._provider_type,
            state=ConnectionState.CONNECTED,
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


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def test_empty_registry(self) -> None:
        """Test newly created registry is empty."""
        registry = ProviderRegistry()

        assert registry.list_providers() == []
        assert registry.list_all() == {}

    def test_register_provider(self) -> None:
        """Test registering a provider."""
        registry = ProviderRegistry()
        provider = MockProvider()

        registry.register(provider)

        assert registry.has(ProviderType.GMAIL)
        assert registry.get(ProviderType.GMAIL) == provider
        assert ProviderType.GMAIL in registry.list_providers()

    def test_register_replaces_existing(self) -> None:
        """Test registering replaces existing provider."""
        registry = ProviderRegistry()
        provider1 = MockProvider()
        provider2 = MockProvider()

        registry.register(provider1)
        registry.register(provider2)

        assert registry.get(ProviderType.GMAIL) == provider2

    def test_unregister_provider(self) -> None:
        """Test unregistering a provider."""
        registry = ProviderRegistry()
        provider = MockProvider()

        registry.register(provider)
        result = registry.unregister(ProviderType.GMAIL)

        assert result is True
        assert not registry.has(ProviderType.GMAIL)
        assert ProviderType.GMAIL not in registry.list_providers()

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering non-existent provider returns False."""
        registry = ProviderRegistry()

        result = registry.unregister(ProviderType.GMAIL)

        assert result is False

    def test_get_nonexistent(self) -> None:
        """Test getting non-existent provider raises error."""
        registry = ProviderRegistry()

        with pytest.raises(ProviderNotFoundError) as exc_info:
            registry.get(ProviderType.GMAIL)

        assert exc_info.value.provider == ProviderType.GMAIL
        assert "gmail" in str(exc_info.value)

    def test_get_optional_exists(self) -> None:
        """Test get_optional returns provider when exists."""
        registry = ProviderRegistry()
        provider = MockProvider()

        registry.register(provider)

        result = registry.get_optional(ProviderType.GMAIL)
        assert result == provider

    def test_get_optional_missing(self) -> None:
        """Test get_optional returns None when missing."""
        registry = ProviderRegistry()

        result = registry.get_optional(ProviderType.GMAIL)
        assert result is None

    def test_has_provider(self) -> None:
        """Test has checks provider existence."""
        registry = ProviderRegistry()
        provider = MockProvider()

        assert not registry.has(ProviderType.GMAIL)

        registry.register(provider)
        assert registry.has(ProviderType.GMAIL)

    def test_list_providers(self) -> None:
        """Test listing provider types."""
        registry = ProviderRegistry()
        provider = MockProvider()

        registry.register(provider)

        providers = registry.list_providers()
        assert providers == [ProviderType.GMAIL]

    def test_list_all(self) -> None:
        """Test listing all providers."""
        registry = ProviderRegistry()
        provider = MockProvider()

        registry.register(provider)

        all_providers = registry.list_all()
        assert all_providers == {ProviderType.GMAIL: provider}

    def test_list_all_returns_copy(self) -> None:
        """Test list_all returns a copy."""
        registry = ProviderRegistry()
        provider = MockProvider()

        registry.register(provider)

        all_providers = registry.list_all()
        all_providers.clear()

        # Original should be unaffected
        assert registry.has(ProviderType.GMAIL)

    def test_clear(self) -> None:
        """Test clearing all providers."""
        registry = ProviderRegistry()
        provider = MockProvider()

        registry.register(provider)
        registry.clear()

        assert registry.list_providers() == []
        assert not registry.has(ProviderType.GMAIL)


class TestProviderNotFoundError:
    """Tests for ProviderNotFoundError exception."""

    def test_inherits_provider_error(self) -> None:
        """Test ProviderNotFoundError is a ProviderError."""
        from rl_emails.providers.base import ProviderError

        assert issubclass(ProviderNotFoundError, ProviderError)

    def test_with_provider(self) -> None:
        """Test error includes provider info."""
        error = ProviderNotFoundError(
            "Provider 'gmail' is not registered",
            provider=ProviderType.GMAIL,
        )

        assert error.provider == ProviderType.GMAIL
        assert "gmail" in str(error)


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_returns_singleton(self) -> None:
        """Test get_provider_registry returns same instance."""
        reset_provider_registry()

        registry1 = get_provider_registry()
        registry2 = get_provider_registry()

        assert registry1 is registry2

    def test_reset_clears_singleton(self) -> None:
        """Test reset_provider_registry clears singleton."""
        reset_provider_registry()

        registry1 = get_provider_registry()
        reset_provider_registry()
        registry2 = get_provider_registry()

        assert registry1 is not registry2

    def test_reset_provider_registry_multiple_times(self) -> None:
        """Test resetting multiple times works."""
        reset_provider_registry()
        reset_provider_registry()
        reset_provider_registry()

        registry = get_provider_registry()
        assert registry is not None
