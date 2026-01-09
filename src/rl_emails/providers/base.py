"""Base email provider interface and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uuid import UUID

    from rl_emails.core.types import EmailData


class ProviderType(str, Enum):
    """Supported email provider types."""

    GMAIL = "gmail"
    # Future providers:
    # OUTLOOK = "outlook"
    # IMAP = "imap"


class ConnectionState(str, Enum):
    """Connection state for email providers."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ConnectionStatus:
    """Status of a provider connection.

    Attributes:
        provider: Provider type.
        state: Current connection state.
        email: Connected email address.
        connected_at: When connection was established.
        last_sync: When last sync occurred.
        error: Error message if in error state.
        metadata: Additional provider-specific metadata.
    """

    provider: ProviderType
    state: ConnectionState
    email: str | None = None
    connected_at: datetime | None = None
    last_sync: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self.state == ConnectionState.CONNECTED


@dataclass
class SyncProgress:
    """Progress information during sync.

    Attributes:
        processed: Number of messages processed.
        total: Total messages to process.
        current_phase: Current sync phase description.
    """

    processed: int
    total: int
    current_phase: str = "syncing"


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, provider: ProviderType | None = None) -> None:
        """Initialize provider error.

        Args:
            message: Error description.
            provider: Provider that caused the error.
        """
        super().__init__(message)
        self.provider = provider


class ConnectionError(ProviderError):
    """Error during connection establishment."""


class SyncError(ProviderError):
    """Error during message sync."""


class AuthorizationError(ProviderError):
    """Error with provider authorization/OAuth."""


class EmailProvider(ABC):
    """Abstract base class for email providers.

    Email providers handle:
    - OAuth authorization flow
    - Connection management
    - Message synchronization
    - Status reporting

    Implementations should wrap provider-specific APIs (Gmail, Outlook, IMAP)
    and expose a unified interface.
    """

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Get the provider type identifier."""

    @abstractmethod
    async def get_auth_url(self, state: str | None = None) -> str:
        """Get OAuth authorization URL.

        Args:
            state: Optional state parameter for CSRF protection.

        Returns:
            URL to redirect user for authorization.
        """

    @abstractmethod
    async def complete_auth(self, user_id: UUID, code: str) -> ConnectionStatus:
        """Complete OAuth flow with authorization code.

        Args:
            user_id: User to connect for.
            code: Authorization code from OAuth callback.

        Returns:
            Connection status after authorization.

        Raises:
            AuthorizationError: If authorization fails.
        """

    @abstractmethod
    async def disconnect(self, user_id: UUID) -> bool:
        """Disconnect user from provider.

        Revokes OAuth tokens and cleans up connection.

        Args:
            user_id: User to disconnect.

        Returns:
            True if disconnected, False if wasn't connected.
        """

    @abstractmethod
    async def get_status(self, user_id: UUID) -> ConnectionStatus:
        """Get current connection status for user.

        Args:
            user_id: User to check status for.

        Returns:
            Current connection status.
        """

    @abstractmethod
    async def sync_messages(
        self,
        user_id: UUID,
        days: int | None = None,
        max_messages: int | None = None,
    ) -> AsyncIterator[EmailData]:
        """Sync messages from provider.

        Yields messages as they are fetched and parsed.

        Args:
            user_id: User to sync for.
            days: Number of days to sync (None for all).
            max_messages: Maximum messages to sync (None for all).

        Yields:
            EmailData for each synced message.

        Raises:
            SyncError: If sync fails.
            ConnectionError: If not connected.
        """
        # This yield is here to make mypy happy about the return type
        # Subclasses will implement the actual logic
        yield  # type: ignore[misc]

    @abstractmethod
    async def get_sync_progress(self, user_id: UUID) -> SyncProgress | None:
        """Get current sync progress if sync is in progress.

        Args:
            user_id: User to check progress for.

        Returns:
            Sync progress if sync is active, None otherwise.
        """
