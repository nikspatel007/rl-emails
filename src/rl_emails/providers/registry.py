"""Provider registry for managing email providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from rl_emails.providers.base import ProviderError, ProviderType

if TYPE_CHECKING:
    from rl_emails.providers.base import EmailProvider

logger = structlog.get_logger(__name__)


class ProviderNotFoundError(ProviderError):
    """Raised when requested provider is not registered."""


class ProviderRegistry:
    """Registry for email providers.

    Provides a central location to register and retrieve email providers.
    Supports dynamic registration for extensibility.

    Example:
        registry = ProviderRegistry()
        registry.register(gmail_provider)

        # Get provider by type
        provider = registry.get(ProviderType.GMAIL)

        # List all available providers
        available = registry.list_providers()
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._providers: dict[ProviderType, EmailProvider] = {}

    def register(self, provider: EmailProvider) -> None:
        """Register an email provider.

        Args:
            provider: Provider instance to register.

        Note:
            If a provider of the same type is already registered,
            it will be replaced.
        """
        provider_type = provider.provider_type
        self._providers[provider_type] = provider

        logger.info(
            "provider_registered",
            provider_type=provider_type.value,
        )

    def unregister(self, provider_type: ProviderType) -> bool:
        """Unregister a provider.

        Args:
            provider_type: Type of provider to unregister.

        Returns:
            True if provider was unregistered, False if not found.
        """
        if provider_type in self._providers:
            del self._providers[provider_type]
            logger.info(
                "provider_unregistered",
                provider_type=provider_type.value,
            )
            return True
        return False

    def get(self, provider_type: ProviderType) -> EmailProvider:
        """Get a registered provider by type.

        Args:
            provider_type: Type of provider to retrieve.

        Returns:
            The registered provider.

        Raises:
            ProviderNotFoundError: If provider is not registered.
        """
        provider = self._providers.get(provider_type)
        if provider is None:
            raise ProviderNotFoundError(
                f"Provider '{provider_type.value}' is not registered",
                provider=provider_type,
            )
        return provider

    def get_optional(self, provider_type: ProviderType) -> EmailProvider | None:
        """Get a provider if registered, None otherwise.

        Args:
            provider_type: Type of provider to retrieve.

        Returns:
            The provider if registered, None otherwise.
        """
        return self._providers.get(provider_type)

    def has(self, provider_type: ProviderType) -> bool:
        """Check if a provider is registered.

        Args:
            provider_type: Type of provider to check.

        Returns:
            True if provider is registered.
        """
        return provider_type in self._providers

    def list_providers(self) -> list[ProviderType]:
        """List all registered provider types.

        Returns:
            List of registered provider types.
        """
        return list(self._providers.keys())

    def list_all(self) -> dict[ProviderType, EmailProvider]:
        """Get all registered providers.

        Returns:
            Dictionary mapping provider types to providers.
        """
        return dict(self._providers)

    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        logger.info("provider_registry_cleared")


# Global singleton registry
_registry: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry singleton.

    Returns:
        The global ProviderRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def reset_provider_registry() -> None:
    """Reset the global provider registry.

    Primarily useful for testing.
    """
    global _registry
    _registry = None
