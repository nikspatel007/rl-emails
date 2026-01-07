"""Tests for Gmail API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rl_emails.integrations.gmail.client import GmailApiError, GmailClient
from rl_emails.integrations.gmail.models import GmailMessageRef
from rl_emails.integrations.gmail.rate_limiter import RateLimiter


class TestGmailApiError:
    """Tests for GmailApiError exception."""

    def test_create_error(self) -> None:
        """Test creating an API error."""
        error = GmailApiError(
            message="Bad request",
            status_code=400,
            error_code="400",
        )

        assert error.message == "Bad request"
        assert error.status_code == 400
        assert error.error_code == "400"
        assert str(error) == "Bad request"

    def test_create_error_minimal(self) -> None:
        """Test creating error with message only."""
        error = GmailApiError(message="Something went wrong")

        assert error.message == "Something went wrong"
        assert error.status_code is None
        assert error.error_code is None


class TestGmailClientInit:
    """Tests for GmailClient initialization."""

    def test_creates_with_token(self) -> None:
        """Test creating client with access token."""
        client = GmailClient(access_token="test-token")

        assert client.access_token == "test-token"
        assert client.user_id == "me"
        assert client.rate_limiter is not None

    def test_creates_with_custom_rate_limiter(self) -> None:
        """Test creating client with custom rate limiter."""
        limiter = RateLimiter(requests_per_second=5)
        client = GmailClient(access_token="test-token", rate_limiter=limiter)

        assert client.rate_limiter is limiter

    def test_creates_with_custom_user_id(self) -> None:
        """Test creating client with custom user ID."""
        client = GmailClient(access_token="test-token", user_id="user@example.com")

        assert client.user_id == "user@example.com"


class TestGmailClientContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self) -> None:
        """Test that context manager closes HTTP client."""
        async with GmailClient(access_token="test-token") as client:
            assert client is not None

        # Client should be closed after exiting context


class TestGmailClientListMessages:
    """Tests for list_messages method."""

    @pytest.mark.asyncio
    async def test_list_messages_success(self) -> None:
        """Test listing messages successfully."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [
                {"id": "msg1", "threadId": "t1"},
                {"id": "msg2", "threadId": "t2"},
            ],
            "nextPageToken": "next123",
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            refs, next_token = await client.list_messages()

        assert len(refs) == 2
        assert refs[0].id == "msg1"
        assert refs[0].thread_id == "t1"
        assert next_token == "next123"

        await client.close()

    @pytest.mark.asyncio
    async def test_list_messages_empty(self) -> None:
        """Test listing messages when no messages exist."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            refs, next_token = await client.list_messages()

        assert len(refs) == 0
        assert next_token is None

        await client.close()

    @pytest.mark.asyncio
    async def test_list_messages_with_query(self) -> None:
        """Test listing messages with search query."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": []}

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.list_messages(query="is:unread")

            # Verify query was passed
            call_args = mock_request.call_args
            assert "q" in call_args.kwargs.get("params", {})

        await client.close()

    @pytest.mark.asyncio
    async def test_list_messages_api_error(self) -> None:
        """Test handling API errors."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.content = b'{"error": {"message": "Unauthorized"}}'
        mock_response.text = "Unauthorized"
        mock_response.json.return_value = {"error": {"message": "Unauthorized", "code": "401"}}

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(GmailApiError) as exc_info:
                await client.list_messages()

            assert exc_info.value.status_code == 401

        await client.close()

    @pytest.mark.asyncio
    async def test_list_messages_api_error_non_dict(self) -> None:
        """Test handling API errors when error is not a dict."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b'{"error": "string error"}'
        mock_response.text = "Server error"
        mock_response.json.return_value = {"error": "string error"}

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(GmailApiError) as exc_info:
                await client.list_messages()

            assert exc_info.value.status_code == 500
            assert exc_info.value.message == "Server error"

        await client.close()

    @pytest.mark.asyncio
    async def test_list_messages_with_label_ids(self) -> None:
        """Test listing messages with label filter."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": []}

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.list_messages(label_ids=["INBOX", "UNREAD"])

            # Verify label IDs were passed
            call_args = mock_request.call_args
            assert "labelIds" in call_args.kwargs.get("params", {})

        await client.close()

    @pytest.mark.asyncio
    async def test_list_messages_non_list_response(self) -> None:
        """Test handling non-list messages response."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": "not a list"}

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            refs, next_token = await client.list_messages()

        assert len(refs) == 0
        assert next_token is None

        await client.close()

    @pytest.mark.asyncio
    async def test_list_messages_invalid_message_format(self) -> None:
        """Test handling messages that aren't dicts."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [
                {"id": "msg1", "threadId": "t1"},
                "not a dict",
                {"id": "msg2", "threadId": "t2"},
            ]
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            refs, next_token = await client.list_messages()

        # Should only get 2 valid messages
        assert len(refs) == 2

        await client.close()

    @pytest.mark.asyncio
    async def test_list_messages_missing_id(self) -> None:
        """Test handling messages missing id field."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [
                {"id": "msg1", "threadId": "t1"},
                {"threadId": "t2"},  # Missing id
                {"id": "msg3", "threadId": "t3"},
            ]
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            refs, next_token = await client.list_messages()

        # Should only get 2 valid messages
        assert len(refs) == 2

        await client.close()


class TestGmailClientGetMessage:
    """Tests for get_message method."""

    @pytest.mark.asyncio
    async def test_get_message_success(self) -> None:
        """Test getting a single message."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg123",
            "threadId": "t123",
            "payload": {"headers": []},
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await client.get_message("msg123")

        assert result["id"] == "msg123"

        await client.close()

    @pytest.mark.asyncio
    async def test_get_message_not_found(self) -> None:
        """Test getting a message that doesn't exist."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b'{"error": {"message": "Not found"}}'
        mock_response.text = "Not found"
        mock_response.json.return_value = {"error": {"message": "Not found", "code": "404"}}

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(GmailApiError) as exc_info:
                await client.get_message("nonexistent")

            assert exc_info.value.status_code == 404

        await client.close()


class TestGmailClientBatchGetMessages:
    """Tests for batch_get_messages method."""

    @pytest.mark.asyncio
    async def test_batch_get_messages_success(self) -> None:
        """Test batch getting multiple messages."""
        client = GmailClient(access_token="test-token")

        refs = [
            GmailMessageRef(id="msg1", thread_id="t1"),
            GmailMessageRef(id="msg2", thread_id="t2"),
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [
            {"id": "msg1", "threadId": "t1"},
            {"id": "msg2", "threadId": "t2"},
        ]

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            results = await client.batch_get_messages(refs)

        assert len(results) == 2

        await client.close()

    @pytest.mark.asyncio
    async def test_batch_get_messages_partial_failure(self) -> None:
        """Test batch get with some failures."""
        client = GmailClient(access_token="test-token")

        refs = [
            GmailMessageRef(id="msg1", thread_id="t1"),
            GmailMessageRef(id="msg2", thread_id="t2"),
        ]

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"id": "msg1"}

        error_response = MagicMock()
        error_response.status_code = 404
        error_response.content = b'{"error": {"message": "Not found"}}'
        error_response.text = "Not found"
        error_response.json.return_value = {"error": {"message": "Not found"}}

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [success_response, error_response]
            results = await client.batch_get_messages(refs)

        assert len(results) == 2
        assert isinstance(results[0], dict)
        assert isinstance(results[1], GmailApiError)

        await client.close()


class TestGmailClientGetProfile:
    """Tests for get_profile method."""

    @pytest.mark.asyncio
    async def test_get_profile_success(self) -> None:
        """Test getting user profile."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emailAddress": "user@example.com",
            "messagesTotal": 1000,
            "threadsTotal": 500,
            "historyId": "12345",
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            result = await client.get_profile()

        assert result["emailAddress"] == "user@example.com"
        assert result["historyId"] == "12345"

        await client.close()


class TestGmailClientGetHistory:
    """Tests for get_history method."""

    @pytest.mark.asyncio
    async def test_get_history_success(self) -> None:
        """Test getting history changes."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "history": [
                {
                    "id": "100",
                    "messagesAdded": [{"message": {"id": "msg1"}}],
                },
                {
                    "id": "101",
                    "messagesDeleted": [{"message": {"id": "msg2"}}],
                },
            ],
            "nextPageToken": "next123",
            "historyId": "102",
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            records, next_token, history_id = await client.get_history("99")

        assert len(records) == 2
        assert next_token == "next123"
        assert history_id == "102"

        await client.close()

    @pytest.mark.asyncio
    async def test_get_history_empty(self) -> None:
        """Test getting history when no changes."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "historyId": "100",
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            records, next_token, history_id = await client.get_history("100")

        assert len(records) == 0
        assert next_token is None
        assert history_id == "100"

        await client.close()

    @pytest.mark.asyncio
    async def test_get_history_with_page_token_and_labels(self) -> None:
        """Test getting history with pagination and label filter."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "history": [],
            "historyId": "200",
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            await client.get_history(
                start_history_id="100",
                page_token="page123",
                label_ids=["INBOX"],
            )

            # Verify params
            call_args = mock_request.call_args
            params = call_args.kwargs.get("params", {})
            assert "pageToken" in params
            assert "labelIds" in params

        await client.close()

    @pytest.mark.asyncio
    async def test_get_history_non_list_response(self) -> None:
        """Test handling non-list history response."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "history": "not a list",
            "historyId": "100",
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            records, next_token, history_id = await client.get_history("99")

        assert len(records) == 0
        assert history_id == "100"

        await client.close()

    @pytest.mark.asyncio
    async def test_get_history_invalid_record_format(self) -> None:
        """Test handling history records that aren't dicts."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "history": [
                {"id": "100", "messagesAdded": []},
                "not a dict",
                {"id": "101", "messagesDeleted": []},
            ],
            "historyId": "102",
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            records, next_token, history_id = await client.get_history("99")

        # Should only get 2 valid records
        assert len(records) == 2

        await client.close()


class TestGmailClientListAllMessages:
    """Tests for list_all_messages async iterator."""

    @pytest.mark.asyncio
    async def test_list_all_messages_single_page(self) -> None:
        """Test iterating over messages from single page."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [
                {"id": "msg1", "threadId": "t1"},
                {"id": "msg2", "threadId": "t2"},
            ],
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            refs = [ref async for ref in client.list_all_messages()]

        assert len(refs) == 2

        await client.close()

    @pytest.mark.asyncio
    async def test_list_all_messages_with_limit(self) -> None:
        """Test limiting total messages returned."""
        client = GmailClient(access_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [{"id": f"msg{i}", "threadId": f"t{i}"} for i in range(10)],
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            refs = [ref async for ref in client.list_all_messages(max_messages=5)]

        assert len(refs) == 5

        await client.close()

    @pytest.mark.asyncio
    async def test_list_all_messages_multiple_pages(self) -> None:
        """Test iterating over multiple pages."""
        client = GmailClient(access_token="test-token")

        page1_response = MagicMock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "messages": [{"id": "msg1", "threadId": "t1"}],
            "nextPageToken": "page2",
        }

        page2_response = MagicMock()
        page2_response.status_code = 200
        page2_response.json.return_value = {
            "messages": [{"id": "msg2", "threadId": "t2"}],
        }

        with patch.object(client._client, "request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [page1_response, page2_response]
            refs = [ref async for ref in client.list_all_messages()]

        assert len(refs) == 2
        assert refs[0].id == "msg1"
        assert refs[1].id == "msg2"

        await client.close()
