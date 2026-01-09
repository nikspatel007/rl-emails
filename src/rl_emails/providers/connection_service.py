"""Connection service for managing multi-provider connections."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from rl_emails.providers.base import (
    ConnectionState,
    ConnectionStatus,
    ProviderError,
    ProviderType,
)
from rl_emails.providers.registry import ProviderNotFoundError, ProviderRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rl_emails.core.types import EmailData
    from rl_emails.providers.base import SyncProgress

logger = structlog.get_logger(__name__)


class ConnectionService:
    """Service for managing email provider connections.

    Provides a unified interface for:
    - Starting OAuth flows for any provider
    - Completing authorization
    - Managing connections (connect/disconnect)
    - Syncing messages across providers
    - Checking connection status

    Example:
        service = ConnectionService(registry)

        # Connect Gmail
        auth_url = await service.get_auth_url(ProviderType.GMAIL)
        # redirect user...
        status = await service.complete_auth(user_id, ProviderType.GMAIL, code)

        # Sync messages
        async for email in service.sync_messages(user_id, ProviderType.GMAIL):
            process(email)

        # Get status across all providers
        statuses = await service.get_all_statuses(user_id)
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        """Initialize connection service.

        Args:
            registry: Provider registry with registered providers.
        """
        self._registry = registry

    async def get_auth_url(
        self,
        provider_type: ProviderType,
        state: str | None = None,
    ) -> str:
        """Get authorization URL for a provider.

        Args:
            provider_type: Type of provider to authorize.
            state: Optional state for CSRF protection.

        Returns:
            Authorization URL to redirect user to.

        Raises:
            ProviderNotFoundError: If provider is not registered.
        """
        provider = self._registry.get(provider_type)
        return await provider.get_auth_url(state=state)

    async def complete_auth(
        self,
        user_id: UUID,
        provider_type: ProviderType,
        code: str,
    ) -> ConnectionStatus:
        """Complete OAuth authorization for a provider.

        Args:
            user_id: User to connect.
            provider_type: Type of provider.
            code: Authorization code from callback.

        Returns:
            Connection status after authorization.

        Raises:
            ProviderNotFoundError: If provider is not registered.
            AuthorizationError: If authorization fails.
        """
        provider = self._registry.get(provider_type)
        status = await provider.complete_auth(user_id, code)

        await logger.ainfo(
            "connection_established",
            user_id=str(user_id),
            provider=provider_type.value,
        )

        return status

    async def disconnect(
        self,
        user_id: UUID,
        provider_type: ProviderType,
    ) -> bool:
        """Disconnect user from a provider.

        Args:
            user_id: User to disconnect.
            provider_type: Type of provider.

        Returns:
            True if disconnected, False if wasn't connected.

        Raises:
            ProviderNotFoundError: If provider is not registered.
        """
        provider = self._registry.get(provider_type)
        result = await provider.disconnect(user_id)

        if result:
            await logger.ainfo(
                "connection_removed",
                user_id=str(user_id),
                provider=provider_type.value,
            )

        return result

    async def get_status(
        self,
        user_id: UUID,
        provider_type: ProviderType,
    ) -> ConnectionStatus:
        """Get connection status for a specific provider.

        Args:
            user_id: User to check.
            provider_type: Type of provider.

        Returns:
            Connection status.

        Raises:
            ProviderNotFoundError: If provider is not registered.
        """
        provider = self._registry.get(provider_type)
        return await provider.get_status(user_id)

    async def get_all_statuses(
        self,
        user_id: UUID,
    ) -> dict[ProviderType, ConnectionStatus]:
        """Get connection status for all registered providers.

        Args:
            user_id: User to check.

        Returns:
            Dictionary mapping provider types to their status.
        """
        statuses: dict[ProviderType, ConnectionStatus] = {}

        for provider_type, provider in self._registry.list_all().items():
            try:
                statuses[provider_type] = await provider.get_status(user_id)
            except Exception as e:
                await logger.awarning(
                    "status_check_failed",
                    user_id=str(user_id),
                    provider=provider_type.value,
                    error=str(e),
                )
                statuses[provider_type] = ConnectionStatus(
                    provider=provider_type,
                    state=ConnectionState.ERROR,
                    error=str(e),
                )

        return statuses

    async def sync_messages(
        self,
        user_id: UUID,
        provider_type: ProviderType,
        days: int | None = None,
        max_messages: int | None = None,
    ) -> AsyncIterator[EmailData]:
        """Sync messages from a provider.

        Args:
            user_id: User to sync for.
            provider_type: Type of provider.
            days: Number of days to sync.
            max_messages: Maximum messages to sync.

        Yields:
            EmailData for each synced message.

        Raises:
            ProviderNotFoundError: If provider is not registered.
            ConnectionError: If not connected.
            SyncError: If sync fails.
        """
        provider = self._registry.get(provider_type)

        async for email in provider.sync_messages(
            user_id=user_id,
            days=days,
            max_messages=max_messages,
        ):
            yield email

    async def get_sync_progress(
        self,
        user_id: UUID,
        provider_type: ProviderType,
    ) -> SyncProgress | None:
        """Get sync progress for a provider.

        Args:
            user_id: User to check.
            provider_type: Type of provider.

        Returns:
            Sync progress if active, None otherwise.

        Raises:
            ProviderNotFoundError: If provider is not registered.
        """
        provider = self._registry.get(provider_type)
        return await provider.get_sync_progress(user_id)

    def list_available_providers(self) -> list[ProviderType]:
        """List all available (registered) providers.

        Returns:
            List of available provider types.
        """
        return self._registry.list_providers()

    async def is_connected(
        self,
        user_id: UUID,
        provider_type: ProviderType,
    ) -> bool:
        """Check if user is connected to a provider.

        Args:
            user_id: User to check.
            provider_type: Type of provider.

        Returns:
            True if connected, False otherwise.
        """
        try:
            provider = self._registry.get(provider_type)
            status = await provider.get_status(user_id)
            return status.is_connected
        except ProviderNotFoundError:
            return False
        except ProviderError:
            return False

    async def get_connected_providers(
        self,
        user_id: UUID,
    ) -> list[ProviderType]:
        """Get list of providers user is connected to.

        Args:
            user_id: User to check.

        Returns:
            List of connected provider types.
        """
        connected: list[ProviderType] = []

        for provider_type in self._registry.list_providers():
            if await self.is_connected(user_id, provider_type):
                connected.append(provider_type)

        return connected
