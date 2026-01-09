"""Email provider interface and implementations.

This module provides:
- Abstract EmailProvider interface for multi-provider support
- GmailProvider wrapping existing Gmail integration
- ProviderRegistry for provider management
- ConnectionService for unified connection management

Example:
    from rl_emails.providers import (
        ConnectionService,
        GmailProvider,
        ProviderRegistry,
        ProviderType,
    )

    # Setup
    registry = ProviderRegistry()
    registry.register(gmail_provider)

    service = ConnectionService(registry)

    # Connect user
    auth_url = await service.get_auth_url(ProviderType.GMAIL)
    status = await service.complete_auth(user_id, ProviderType.GMAIL, code)

    # Sync messages
    async for email in service.sync_messages(user_id, ProviderType.GMAIL):
        process(email)
"""

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
from rl_emails.providers.connection_service import ConnectionService
from rl_emails.providers.gmail import GmailProvider
from rl_emails.providers.registry import (
    ProviderNotFoundError,
    ProviderRegistry,
    get_provider_registry,
    reset_provider_registry,
)

__all__ = [
    # Base types
    "ProviderType",
    "ConnectionState",
    "ConnectionStatus",
    "SyncProgress",
    # Exceptions
    "ProviderError",
    "ConnectionError",
    "SyncError",
    "AuthorizationError",
    "ProviderNotFoundError",
    # Abstract interface
    "EmailProvider",
    # Implementations
    "GmailProvider",
    # Registry
    "ProviderRegistry",
    "get_provider_registry",
    "reset_provider_registry",
    # Service
    "ConnectionService",
]
