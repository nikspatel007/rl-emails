"""Tests for GmailProvider."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest import mock
from uuid import UUID

import pytest

from rl_emails.providers.base import (
    AuthorizationError,
    ConnectionError,
    ConnectionState,
    ProviderType,
)
from rl_emails.providers.gmail import GmailProvider

if TYPE_CHECKING:
    pass


@pytest.fixture
def user_id() -> UUID:
    """Create a test user ID."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def mock_auth_service() -> mock.MagicMock:
    """Create a mock auth service."""
    service = mock.MagicMock()
    service.start_auth_flow.return_value = "https://accounts.google.com/authorize"
    return service


@pytest.fixture
def mock_token_repo() -> mock.MagicMock:
    """Create a mock token repository."""
    return mock.MagicMock()


@pytest.fixture
def mock_sync_repo() -> mock.MagicMock:
    """Create a mock sync repository."""
    repo = mock.MagicMock()
    repo.get_by_user_id = mock.AsyncMock(return_value=None)
    repo.start_sync = mock.AsyncMock()
    repo.complete_sync = mock.AsyncMock()
    repo.fail_sync = mock.AsyncMock()
    return repo


@pytest.fixture
def provider(
    mock_auth_service: mock.MagicMock,
    mock_token_repo: mock.MagicMock,
    mock_sync_repo: mock.MagicMock,
) -> GmailProvider:
    """Create a GmailProvider instance."""
    return GmailProvider(mock_auth_service, mock_token_repo, mock_sync_repo)


class TestGmailProviderInit:
    """Tests for GmailProvider initialization."""

    def test_init(
        self,
        mock_auth_service: mock.MagicMock,
        mock_token_repo: mock.MagicMock,
        mock_sync_repo: mock.MagicMock,
    ) -> None:
        """Test provider initialization."""
        provider = GmailProvider(mock_auth_service, mock_token_repo, mock_sync_repo)

        assert provider._auth_service == mock_auth_service
        assert provider._token_repo == mock_token_repo
        assert provider._sync_repo == mock_sync_repo
        assert provider._active_syncs == {}


class TestProviderType:
    """Tests for provider_type property."""

    def test_provider_type(self, provider: GmailProvider) -> None:
        """Test provider type is GMAIL."""
        assert provider.provider_type == ProviderType.GMAIL


class TestGetAuthUrl:
    """Tests for get_auth_url method."""

    @pytest.mark.asyncio
    async def test_get_auth_url(self, provider: GmailProvider) -> None:
        """Test getting auth URL."""
        url = await provider.get_auth_url()

        assert url == "https://accounts.google.com/authorize"
        provider._auth_service.start_auth_flow.assert_called_once_with(state=None)

    @pytest.mark.asyncio
    async def test_get_auth_url_with_state(self, provider: GmailProvider) -> None:
        """Test getting auth URL with state."""
        await provider.get_auth_url(state="csrf123")

        provider._auth_service.start_auth_flow.assert_called_once_with(state="csrf123")


class TestCompleteAuth:
    """Tests for complete_auth method."""

    @pytest.mark.asyncio
    async def test_complete_auth_success(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test successful auth completion."""
        mock_token = mock.MagicMock()
        mock_token.expires_at = datetime.now(UTC)
        mock_token.scopes = ["gmail.readonly"]
        provider._auth_service.complete_auth_flow = mock.AsyncMock(return_value=mock_token)

        status = await provider.complete_auth(user_id, "auth_code")

        assert status.state == ConnectionState.CONNECTED
        assert status.provider == ProviderType.GMAIL
        assert status.metadata.get("scopes") == ["gmail.readonly"]

    @pytest.mark.asyncio
    async def test_complete_auth_failure(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test auth completion failure."""
        provider._auth_service.complete_auth_flow = mock.AsyncMock(
            side_effect=Exception("Auth failed")
        )

        with pytest.raises(AuthorizationError) as exc_info:
            await provider.complete_auth(user_id, "bad_code")

        assert exc_info.value.provider == ProviderType.GMAIL
        assert "failed" in str(exc_info.value).lower()


class TestDisconnect:
    """Tests for disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test successful disconnect."""
        provider._auth_service.revoke_token = mock.AsyncMock(return_value=True)

        result = await provider.disconnect(user_id)

        assert result is True
        provider._auth_service.revoke_token.assert_called_once_with(user_id, provider="google")

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test disconnect when not connected."""
        provider._auth_service.revoke_token = mock.AsyncMock(return_value=False)

        result = await provider.disconnect(user_id)

        assert result is False


class TestGetStatus:
    """Tests for get_status method."""

    @pytest.mark.asyncio
    async def test_get_status_disconnected(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test status when no token exists."""
        provider._token_repo.get_by_user = mock.AsyncMock(return_value=None)

        status = await provider.get_status(user_id)

        assert status.state == ConnectionState.DISCONNECTED
        assert status.provider == ProviderType.GMAIL

    @pytest.mark.asyncio
    async def test_get_status_token_expired(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test status when token is expired."""
        mock_token = mock.MagicMock()
        mock_token.is_expired = True
        mock_token.created_at = datetime.now(UTC)
        mock_token.scopes = ["gmail.readonly"]
        provider._token_repo.get_by_user = mock.AsyncMock(return_value=mock_token)

        status = await provider.get_status(user_id)

        assert status.state == ConnectionState.ERROR
        assert "expired" in str(status.error).lower()

    @pytest.mark.asyncio
    async def test_get_status_connected(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test status when connected."""
        mock_token = mock.MagicMock()
        mock_token.is_expired = False
        mock_token.created_at = datetime.now(UTC)
        mock_token.scopes = ["gmail.readonly"]
        provider._token_repo.get_by_user = mock.AsyncMock(return_value=mock_token)

        status = await provider.get_status(user_id)

        assert status.state == ConnectionState.CONNECTED
        assert status.connected_at is not None
        assert status.metadata.get("scopes") == ["gmail.readonly"]

    @pytest.mark.asyncio
    async def test_get_status_with_last_sync(
        self,
        provider: GmailProvider,
        user_id: UUID,
        mock_sync_repo: mock.MagicMock,
    ) -> None:
        """Test status includes last sync time."""
        mock_token = mock.MagicMock()
        mock_token.is_expired = False
        mock_token.created_at = datetime.now(UTC)
        mock_token.scopes = ["gmail.readonly"]
        provider._token_repo.get_by_user = mock.AsyncMock(return_value=mock_token)

        mock_sync_state = mock.MagicMock()
        mock_sync_state.last_sync_at = datetime.now(UTC)
        mock_sync_repo.get_by_user_id.return_value = mock_sync_state

        status = await provider.get_status(user_id)

        assert status.last_sync is not None


class TestSyncMessages:
    """Tests for sync_messages method."""

    @pytest.mark.asyncio
    async def test_sync_messages_not_connected(
        self, provider: GmailProvider, user_id: UUID
    ) -> None:
        """Test sync fails when not connected."""
        provider._auth_service.get_valid_token = mock.AsyncMock(side_effect=Exception("No token"))

        with pytest.raises(ConnectionError) as exc_info:
            async for _ in provider.sync_messages(user_id):
                pass

        assert exc_info.value.provider == ProviderType.GMAIL

    @pytest.mark.asyncio
    async def test_sync_with_default_days(
        self,
        provider: GmailProvider,
        user_id: UUID,
        mock_sync_repo: mock.MagicMock,
    ) -> None:
        """Test sync uses default 30 days when not specified."""
        provider._auth_service.get_valid_token = mock.AsyncMock(return_value="access_token")

        with mock.patch("rl_emails.providers.gmail.GmailClient") as MockClient:
            mock_client = mock.AsyncMock()
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock()

            # Track the query passed to list_all_messages
            captured_query = None

            async def capture_messages(query=None, **kwargs):
                nonlocal captured_query
                captured_query = query
                return
                yield  # type: ignore[misc]

            mock_client.list_all_messages = capture_messages
            mock_client.get_profile = mock.AsyncMock(return_value={"historyId": "123"})
            MockClient.return_value = mock_client

            async for _ in provider.sync_messages(user_id):
                pass

            # Verify default 30 days was used
            assert captured_query is not None
            assert "newer_than:30d" in captured_query

    @pytest.mark.asyncio
    async def test_sync_messages_success(
        self,
        provider: GmailProvider,
        user_id: UUID,
        mock_sync_repo: mock.MagicMock,
    ) -> None:
        """Test successful message sync."""

        provider._auth_service.get_valid_token = mock.AsyncMock(return_value="access_token")

        with mock.patch("rl_emails.providers.gmail.GmailClient") as MockClient:
            mock_client = mock.AsyncMock()
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock()

            # Return empty list of messages
            async def empty_messages(*args, **kwargs):
                return
                yield  # type: ignore[misc]

            mock_client.list_all_messages = empty_messages
            mock_client.get_profile = mock.AsyncMock(return_value={"historyId": "123"})
            MockClient.return_value = mock_client

            messages = []
            async for msg in provider.sync_messages(user_id):
                messages.append(msg)

            assert messages == []
            mock_sync_repo.start_sync.assert_called_once_with(user_id)
            mock_sync_repo.complete_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_messages_with_messages(
        self,
        provider: GmailProvider,
        user_id: UUID,
        mock_sync_repo: mock.MagicMock,
    ) -> None:
        """Test sync with actual messages."""
        from rl_emails.integrations.gmail.models import GmailMessageRef

        provider._auth_service.get_valid_token = mock.AsyncMock(return_value="access_token")

        with (
            mock.patch("rl_emails.providers.gmail.GmailClient") as MockClient,
            mock.patch("rl_emails.providers.gmail.parse_raw_message") as mock_parse,
            mock.patch("rl_emails.providers.gmail.gmail_to_email_data") as mock_convert,
        ):
            mock_client = mock.AsyncMock()
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock()

            # Return one message
            async def one_message(*args, **kwargs):
                yield GmailMessageRef(id="msg1", thread_id="thread1")

            mock_client.list_all_messages = one_message
            mock_client.get_message = mock.AsyncMock(return_value={"id": "msg1"})
            mock_client.get_profile = mock.AsyncMock(return_value={"historyId": "123"})
            MockClient.return_value = mock_client

            mock_gmail_message = mock.MagicMock()
            mock_parse.return_value = mock_gmail_message

            mock_email_data = {"message_id": "msg1"}
            mock_convert.return_value = mock_email_data

            messages = []
            async for msg in provider.sync_messages(user_id):
                messages.append(msg)

            assert len(messages) == 1
            assert messages[0] == mock_email_data

    @pytest.mark.asyncio
    async def test_sync_messages_handles_parse_error(
        self,
        provider: GmailProvider,
        user_id: UUID,
        mock_sync_repo: mock.MagicMock,
    ) -> None:
        """Test sync continues after parse error."""
        from rl_emails.integrations.gmail.models import GmailMessageRef

        provider._auth_service.get_valid_token = mock.AsyncMock(return_value="access_token")

        with (
            mock.patch("rl_emails.providers.gmail.GmailClient") as MockClient,
            mock.patch("rl_emails.providers.gmail.parse_raw_message") as mock_parse,
        ):
            mock_client = mock.AsyncMock()
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock()

            # Return one message
            async def one_message(*args, **kwargs):
                yield GmailMessageRef(id="msg1", thread_id="thread1")

            mock_client.list_all_messages = one_message
            mock_client.get_message = mock.AsyncMock(return_value={"id": "msg1"})
            mock_client.get_profile = mock.AsyncMock(return_value={"historyId": "123"})
            MockClient.return_value = mock_client

            # Make parsing fail
            mock_parse.side_effect = Exception("Parse error")

            messages = []
            async for msg in provider.sync_messages(user_id):
                messages.append(msg)

            # Should continue and not raise
            assert messages == []

    @pytest.mark.asyncio
    async def test_sync_messages_cleans_up_progress(
        self,
        provider: GmailProvider,
        user_id: UUID,
        mock_sync_repo: mock.MagicMock,
    ) -> None:
        """Test sync cleans up progress tracking after completion."""
        provider._auth_service.get_valid_token = mock.AsyncMock(return_value="access_token")

        with mock.patch("rl_emails.providers.gmail.GmailClient") as MockClient:
            mock_client = mock.AsyncMock()
            mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = mock.AsyncMock()

            # Return no messages
            async def empty_messages(*args, **kwargs):
                return
                yield  # type: ignore[misc]

            mock_client.list_all_messages = empty_messages
            mock_client.get_profile = mock.AsyncMock(return_value={"historyId": "456"})
            MockClient.return_value = mock_client

            # Consume all messages
            messages = []
            async for msg in provider.sync_messages(user_id):
                messages.append(msg)

            # Verify progress was cleaned up
            assert user_id not in provider._active_syncs
            assert messages == []


class TestGetSyncProgress:
    """Tests for get_sync_progress method."""

    @pytest.mark.asyncio
    async def test_get_sync_progress_none(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test getting sync progress when not syncing."""
        progress = await provider.get_sync_progress(user_id)

        assert progress is None

    @pytest.mark.asyncio
    async def test_get_sync_progress_active(self, provider: GmailProvider, user_id: UUID) -> None:
        """Test getting sync progress during sync."""
        from rl_emails.providers.base import SyncProgress

        # Manually set progress
        provider._active_syncs[user_id] = SyncProgress(
            processed=50,
            total=100,
            current_phase="processing",
        )

        progress = await provider.get_sync_progress(user_id)

        assert progress is not None
        assert progress.processed == 50
        assert progress.total == 100
        assert progress.current_phase == "processing"
