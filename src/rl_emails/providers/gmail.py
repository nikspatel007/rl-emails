"""Gmail email provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from rl_emails.integrations.gmail.client import GmailApiError, GmailClient
from rl_emails.integrations.gmail.models import GmailMessageRef
from rl_emails.integrations.gmail.parser import gmail_to_email_data, parse_raw_message
from rl_emails.providers.base import (
    AuthorizationError,
    ConnectionError,
    ConnectionState,
    ConnectionStatus,
    EmailProvider,
    ProviderType,
    SyncError,
    SyncProgress,
)

if TYPE_CHECKING:
    from rl_emails.core.types import EmailData
    from rl_emails.repositories.oauth_token import OAuthTokenRepository
    from rl_emails.repositories.sync_state import SyncStateRepository
    from rl_emails.services.auth_service import AuthService

logger = structlog.get_logger(__name__)


class GmailProvider(EmailProvider):
    """Gmail email provider implementation.

    Wraps the existing Gmail integration to provide a unified interface.
    Uses AuthService for OAuth and SyncStateRepository for sync tracking.

    Example:
        provider = GmailProvider(auth_service, token_repo, sync_repo)
        status = await provider.get_status(user_id)

        if not status.is_connected:
            auth_url = await provider.get_auth_url()
            # redirect user to auth_url
    """

    def __init__(
        self,
        auth_service: AuthService,
        token_repo: OAuthTokenRepository,
        sync_repo: SyncStateRepository,
    ) -> None:
        """Initialize Gmail provider.

        Args:
            auth_service: Service for OAuth flow orchestration.
            token_repo: Repository for token storage.
            sync_repo: Repository for sync state tracking.
        """
        self._auth_service = auth_service
        self._token_repo = token_repo
        self._sync_repo = sync_repo
        self._active_syncs: dict[UUID, SyncProgress] = {}

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type identifier."""
        return ProviderType.GMAIL

    async def get_auth_url(self, state: str | None = None) -> str:
        """Get OAuth authorization URL for Gmail.

        Args:
            state: Optional state parameter for CSRF protection.

        Returns:
            URL to redirect user for Gmail authorization.
        """
        return self._auth_service.start_auth_flow(state=state)

    async def complete_auth(self, user_id: UUID, code: str) -> ConnectionStatus:
        """Complete Gmail OAuth flow.

        Args:
            user_id: User to connect for.
            code: Authorization code from OAuth callback.

        Returns:
            Connection status after authorization.

        Raises:
            AuthorizationError: If authorization fails.
        """
        try:
            token = await self._auth_service.complete_auth_flow(user_id, code)

            await logger.ainfo(
                "gmail_connected",
                user_id=str(user_id),
                expires_at=str(token.expires_at),
            )

            return ConnectionStatus(
                provider=ProviderType.GMAIL,
                state=ConnectionState.CONNECTED,
                connected_at=datetime.now(UTC),
                metadata={"scopes": token.scopes},
            )

        except Exception as e:
            await logger.aerror(
                "gmail_auth_failed",
                user_id=str(user_id),
                error=str(e),
            )
            raise AuthorizationError(
                f"Gmail authorization failed: {e}",
                provider=ProviderType.GMAIL,
            ) from e

    async def disconnect(self, user_id: UUID) -> bool:
        """Disconnect user from Gmail.

        Args:
            user_id: User to disconnect.

        Returns:
            True if disconnected, False if wasn't connected.
        """
        result = await self._auth_service.revoke_token(user_id, provider="google")

        if result:
            await logger.ainfo("gmail_disconnected", user_id=str(user_id))

        return result

    async def get_status(self, user_id: UUID) -> ConnectionStatus:
        """Get Gmail connection status for user.

        Args:
            user_id: User to check status for.

        Returns:
            Current connection status.
        """
        token = await self._token_repo.get_by_user(user_id, provider="google")

        if token is None:
            return ConnectionStatus(
                provider=ProviderType.GMAIL,
                state=ConnectionState.DISCONNECTED,
            )

        # Check if token is expired
        if token.is_expired:
            return ConnectionStatus(
                provider=ProviderType.GMAIL,
                state=ConnectionState.ERROR,
                error="Token expired, reconnection needed",
                connected_at=token.created_at,
                metadata={"scopes": token.scopes},
            )

        # Get last sync time if available
        sync_state = await self._sync_repo.get_by_user_id(user_id)
        last_sync = sync_state.last_sync_at if sync_state else None

        return ConnectionStatus(
            provider=ProviderType.GMAIL,
            state=ConnectionState.CONNECTED,
            connected_at=token.created_at,
            last_sync=last_sync,
            metadata={"scopes": token.scopes},
        )

    async def sync_messages(
        self,
        user_id: UUID,
        days: int | None = None,
        max_messages: int | None = None,
    ) -> AsyncIterator[EmailData]:
        """Sync messages from Gmail.

        Args:
            user_id: User to sync for.
            days: Number of days to sync (default 30).
            max_messages: Maximum messages to sync (None for all).

        Yields:
            EmailData for each synced message.

        Raises:
            SyncError: If sync fails.
            ConnectionError: If not connected.
        """
        # Get valid token
        try:
            access_token = await self._auth_service.get_valid_token(user_id, "google")
        except Exception as e:
            raise ConnectionError(
                f"Not connected to Gmail: {e}",
                provider=ProviderType.GMAIL,
            ) from e

        # Build query
        sync_days = days or 30
        query = f"newer_than:{sync_days}d"

        # Mark sync started
        await self._sync_repo.start_sync(user_id)

        try:
            async with GmailClient(access_token) as client:
                # List messages - collect all matching messages
                message_refs: list[GmailMessageRef] = []
                try:
                    async for ref in client.list_all_messages(
                        query=query,
                        max_messages=max_messages,
                    ):
                        message_refs.append(ref)
                except GmailApiError as e:
                    raise SyncError(
                        f"Failed to list messages: {e}",
                        provider=ProviderType.GMAIL,
                    ) from e

                total = len(message_refs)
                self._active_syncs[user_id] = SyncProgress(
                    processed=0,
                    total=total,
                    current_phase="fetching",
                )

                await logger.ainfo(
                    "gmail_sync_started",
                    user_id=str(user_id),
                    total_messages=total,
                    days=sync_days,
                )

                # Fetch and yield messages
                for i, ref in enumerate(message_refs):
                    try:
                        raw_message = await client.get_message(ref.id)
                        gmail_message = parse_raw_message(raw_message)
                        email_data = gmail_to_email_data(gmail_message)

                        self._active_syncs[user_id] = SyncProgress(
                            processed=i + 1,
                            total=total,
                            current_phase="processing",
                        )

                        yield email_data

                    except Exception as e:
                        await logger.awarning(
                            "gmail_message_parse_error",
                            message_id=ref.id,
                            error=str(e),
                        )
                        # Continue with next message

                # Get history ID for incremental sync
                profile = await client.get_profile()
                history_id = profile.get("historyId")
                history_id_str = str(history_id) if history_id else None

                # Mark sync complete
                await self._sync_repo.complete_sync(
                    user_id=user_id,
                    history_id=history_id_str,
                    emails_synced=total,
                )

                await logger.ainfo(
                    "gmail_sync_completed",
                    user_id=str(user_id),
                    messages_synced=total,
                    history_id=history_id,
                )

        except (ConnectionError, SyncError):
            raise
        except Exception as e:
            await self._sync_repo.fail_sync(user_id, str(e))
            raise SyncError(
                f"Sync failed: {e}",
                provider=ProviderType.GMAIL,
            ) from e
        finally:
            # Clean up progress tracking
            self._active_syncs.pop(user_id, None)

    async def get_sync_progress(self, user_id: UUID) -> SyncProgress | None:
        """Get current sync progress if sync is in progress.

        Args:
            user_id: User to check progress for.

        Returns:
            Sync progress if sync is active, None otherwise.
        """
        return self._active_syncs.get(user_id)
