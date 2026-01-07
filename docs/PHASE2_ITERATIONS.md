# Phase 2: Gmail API Integration

## Status: Pending

**Prerequisite**: Phase 1 Complete (multi-tenant foundation)
**Iterations**: 4 (Iterations 4-7 in overall plan)
**Goal**: Enable Gmail API as alternative data source alongside existing MBOX pipeline

---

## Overview

Phase 2 adds Gmail API integration as an alternative to MBOX file ingestion. Both paths remain fully supported:

```
                    ┌─────────────────┐
                    │  Data Sources   │
                    └────────┬────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │                                   │
    ┌──────▼──────┐                    ┌───────▼──────┐
    │  MBOX File  │                    │  Gmail API   │
    │  (Stage 1-2)│                    │  (New Stage) │
    └──────┬──────┘                    └───────┬──────┘
           │                                   │
           └─────────────────┬─────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Shared Pipeline │
                    │  (Stages 3-11)  │
                    └─────────────────┘
```

**Key Principle**: MBOX pipeline remains unchanged. Gmail API is additive.

---

## Iteration 4: OAuth2 Flow

### Story
As a user, I need to connect my Gmail account via OAuth so that rl-emails can access my emails securely.

### Deliverables
1. OAuth2 authentication service
2. Token storage and refresh logic
3. CLI commands for auth flow
4. Configuration for Google credentials

### Architecture

```
src/rl_emails/
├── auth/                      # NEW: Authentication module
│   ├── __init__.py
│   ├── oauth.py              # OAuth base classes
│   └── google.py             # Google-specific OAuth
├── services/                  # NEW: Business logic services
│   ├── __init__.py
│   └── auth_service.py       # Auth orchestration
└── core/
    └── config.py             # MODIFY: Add Google credentials
```

### Files to Create

| File | Description |
|------|-------------|
| `src/rl_emails/auth/__init__.py` | Barrel exports |
| `src/rl_emails/auth/oauth.py` | OAuth2 base types and utilities |
| `src/rl_emails/auth/google.py` | Google OAuth implementation |
| `src/rl_emails/services/__init__.py` | Barrel exports |
| `src/rl_emails/services/auth_service.py` | Auth flow orchestration |
| `tests/unit/auth/test_oauth.py` | OAuth base tests |
| `tests/unit/auth/test_google.py` | Google OAuth tests |
| `tests/unit/services/test_auth_service.py` | Service tests |

### Files to Modify

| File | Changes |
|------|---------|
| `src/rl_emails/core/config.py` | Add Google OAuth credentials |
| `src/rl_emails/cli.py` | Add `auth` subcommands |
| `pyproject.toml` | Add `google-auth`, `google-auth-oauthlib` dependencies |

### Implementation Design

```python
# src/rl_emails/auth/google.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class GoogleTokens:
    """OAuth tokens from Google."""
    access_token: str
    refresh_token: str
    expires_at: datetime
    scopes: list[str]

class GoogleOAuth:
    """Google OAuth2 implementation."""

    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
    ]

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorization_url(self, state: str | None = None) -> str:
        """Generate OAuth authorization URL."""
        ...

    async def exchange_code(self, code: str) -> GoogleTokens:
        """Exchange authorization code for tokens."""
        ...

    async def refresh_token(self, refresh_token: str) -> GoogleTokens:
        """Refresh expired access token."""
        ...
```

```python
# src/rl_emails/services/auth_service.py
from uuid import UUID

from rl_emails.auth.google import GoogleOAuth, GoogleTokens
from rl_emails.repositories.org_user import OrgUserRepository
from rl_emails.models.oauth_token import OAuthToken

class AuthService:
    """Service for managing authentication."""

    def __init__(
        self,
        oauth: GoogleOAuth,
        user_repo: OrgUserRepository,
        session: AsyncSession,
    ) -> None:
        self.oauth = oauth
        self.user_repo = user_repo
        self.session = session

    async def start_auth_flow(self, user_id: UUID) -> str:
        """Start OAuth flow, return authorization URL."""
        ...

    async def complete_auth_flow(self, user_id: UUID, code: str) -> OAuthToken:
        """Complete OAuth flow with authorization code."""
        ...

    async def get_valid_token(self, user_id: UUID) -> str:
        """Get valid access token, refreshing if needed."""
        ...

    async def revoke_token(self, user_id: UUID) -> None:
        """Revoke and delete user's tokens."""
        ...
```

### CLI Commands

```bash
# Connect Gmail account (opens browser)
rl-emails auth connect --email user@gmail.com

# Check connection status
rl-emails auth status --email user@gmail.com

# Disconnect Gmail
rl-emails auth disconnect --email user@gmail.com
```

### Config Updates

```python
# src/rl_emails/core/config.py additions
@dataclass
class Config:
    # Existing fields...

    # Google OAuth (new)
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    def has_google_oauth(self) -> bool:
        """Check if Google OAuth is configured."""
        return bool(self.google_client_id and self.google_client_secret)
```

### Acceptance Criteria

- [ ] `GoogleOAuth` class implements authorization URL generation
- [ ] `GoogleOAuth` class implements code exchange
- [ ] `GoogleOAuth` class implements token refresh
- [ ] `AuthService` stores tokens in `oauth_tokens` table
- [ ] `AuthService` refreshes expired tokens automatically
- [ ] CLI `auth connect` opens browser and completes flow
- [ ] CLI `auth status` shows token status and expiry
- [ ] CLI `auth disconnect` revokes and removes tokens
- [ ] Config loads Google credentials from environment
- [ ] All error cases handled (invalid code, network errors, etc.)
- [ ] 100% test coverage on new code
- [ ] mypy --strict passes

### Test Plan

```python
# tests/unit/auth/test_google.py

class TestGoogleOAuth:
    """Tests for Google OAuth implementation."""

    def test_get_authorization_url_includes_scopes(self):
        """Authorization URL includes required scopes."""
        oauth = GoogleOAuth(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost/callback",
        )
        url = oauth.get_authorization_url()
        assert "gmail.readonly" in url
        assert "test-id" in url

    def test_get_authorization_url_includes_state(self):
        """Authorization URL includes state parameter."""
        oauth = GoogleOAuth(...)
        url = oauth.get_authorization_url(state="random-state")
        assert "state=random-state" in url

    @patch("httpx.AsyncClient.post")
    async def test_exchange_code_returns_tokens(self, mock_post):
        """Code exchange returns GoogleTokens."""
        mock_post.return_value.json.return_value = {
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_in": 3600,
            "scope": "https://www.googleapis.com/auth/gmail.readonly",
        }

        oauth = GoogleOAuth(...)
        tokens = await oauth.exchange_code("auth-code")

        assert tokens.access_token == "access-123"
        assert tokens.refresh_token == "refresh-456"

    @patch("httpx.AsyncClient.post")
    async def test_exchange_code_handles_error(self, mock_post):
        """Code exchange raises on error response."""
        mock_post.return_value.json.return_value = {
            "error": "invalid_grant",
            "error_description": "Code expired",
        }
        mock_post.return_value.status_code = 400

        oauth = GoogleOAuth(...)
        with pytest.raises(OAuthError, match="invalid_grant"):
            await oauth.exchange_code("expired-code")

    @patch("httpx.AsyncClient.post")
    async def test_refresh_token_returns_new_access_token(self, mock_post):
        """Token refresh returns new access token."""
        mock_post.return_value.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
        }

        oauth = GoogleOAuth(...)
        tokens = await oauth.refresh_token("refresh-456")

        assert tokens.access_token == "new-access-token"


class TestAuthService:
    """Tests for AuthService."""

    async def test_start_auth_flow_returns_url(self, mock_session):
        """start_auth_flow returns authorization URL."""
        service = AuthService(...)
        url = await service.start_auth_flow(user_id)

        assert url.startswith("https://accounts.google.com")

    async def test_complete_auth_flow_stores_tokens(self, mock_session):
        """complete_auth_flow stores tokens in database."""
        service = AuthService(...)
        token = await service.complete_auth_flow(user_id, "auth-code")

        assert token.user_id == user_id
        assert token.access_token is not None

    async def test_get_valid_token_refreshes_expired(self, mock_session):
        """get_valid_token refreshes when expired."""
        # Create expired token in DB
        expired_token = OAuthToken(
            user_id=user_id,
            access_token="old-token",
            refresh_token="refresh-token",
            expires_at=datetime.now() - timedelta(hours=1),
        )

        service = AuthService(...)
        token = await service.get_valid_token(user_id)

        assert token != "old-token"  # Should be refreshed
```

### Verification Steps

1. **Configure credentials**:
   ```bash
   # Verify .env has Google credentials
   cat .env | grep GOOGLE
   # Should show:
   # GOOGLE_CLIENT_ID=...
   # GOOGLE_CLIENT_SECRET=...
   ```

2. **Test auth flow manually**:
   ```bash
   # Start auth (opens browser)
   rl-emails auth connect --email your@gmail.com

   # Follow browser prompts, approve access
   # Should see: "Successfully connected your@gmail.com"
   ```

3. **Verify token storage**:
   ```sql
   SELECT user_id, provider, expires_at
   FROM oauth_tokens
   WHERE user_id = '<user-uuid>';
   ```

4. **Check status**:
   ```bash
   rl-emails auth status --email your@gmail.com
   # Should show: Connected, expires at <datetime>
   ```

5. **Run tests**:
   ```bash
   make check  # All tests pass, 100% coverage
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| OAuth flow | User can connect Gmail via browser |
| Token storage | Tokens stored securely in database |
| Token refresh | Expired tokens refresh automatically |
| CLI commands | `connect`, `status`, `disconnect` work |
| Error handling | Clear messages for all failure cases |
| Type safety | mypy --strict passes |
| Test coverage | 100% on new code |

---

## Iteration 5: Gmail API Client

### Story
As a developer, I need a Gmail API client wrapper so that I can fetch emails efficiently with proper rate limiting and error handling.

### Deliverables
1. Gmail API client wrapper
2. Message listing with filters
3. Batch message fetching
4. Rate limiting and retry logic
5. Email parsing from Gmail format

### Architecture

```
src/rl_emails/
├── integrations/              # NEW: External service integrations
│   ├── __init__.py
│   └── gmail/
│       ├── __init__.py
│       ├── client.py         # Gmail API client
│       ├── models.py         # Gmail-specific data models
│       ├── parser.py         # Parse Gmail format to internal format
│       └── rate_limiter.py   # Rate limiting utilities
```

### Files to Create

| File | Description |
|------|-------------|
| `src/rl_emails/integrations/__init__.py` | Barrel exports |
| `src/rl_emails/integrations/gmail/__init__.py` | Gmail module exports |
| `src/rl_emails/integrations/gmail/client.py` | Gmail API wrapper |
| `src/rl_emails/integrations/gmail/models.py` | Gmail data models |
| `src/rl_emails/integrations/gmail/parser.py` | Email format parser |
| `src/rl_emails/integrations/gmail/rate_limiter.py` | Rate limiting |
| `tests/unit/integrations/gmail/test_client.py` | Client tests |
| `tests/unit/integrations/gmail/test_parser.py` | Parser tests |
| `tests/unit/integrations/gmail/test_rate_limiter.py` | Rate limiter tests |

### Implementation Design

```python
# src/rl_emails/integrations/gmail/models.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class GmailMessageRef:
    """Reference to a Gmail message (from list)."""
    id: str
    thread_id: str

@dataclass
class GmailMessage:
    """Full Gmail message with parsed data."""
    id: str
    thread_id: str
    history_id: str
    label_ids: list[str]

    # Parsed headers
    message_id: str
    subject: str
    from_address: str
    to_addresses: list[str]
    cc_addresses: list[str]
    date_sent: datetime
    in_reply_to: str | None
    references: list[str]

    # Content
    snippet: str
    body_plain: str | None
    body_html: str | None

    # Metadata
    size_bytes: int
    has_attachments: bool
```

```python
# src/rl_emails/integrations/gmail/client.py
from typing import AsyncIterator
from datetime import datetime, timedelta

class GmailClient:
    """Gmail API client with rate limiting and batching."""

    DEFAULT_BATCH_SIZE = 100
    DEFAULT_REQUESTS_PER_SECOND = 10

    def __init__(
        self,
        access_token: str,
        requests_per_second: int = DEFAULT_REQUESTS_PER_SECOND,
    ) -> None:
        self.access_token = access_token
        self.rate_limiter = RateLimiter(requests_per_second)

    async def list_messages(
        self,
        days: int | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        labels: list[str] | None = None,
        max_results: int | None = None,
    ) -> AsyncIterator[GmailMessageRef]:
        """List messages with optional filters.

        Args:
            days: Fetch last N days of messages
            after: Fetch messages after this date
            before: Fetch messages before this date
            labels: Filter by label IDs (e.g., ["INBOX"])
            max_results: Maximum number of messages to return

        Yields:
            GmailMessageRef objects
        """
        ...

    async def get_message(self, message_id: str) -> GmailMessage:
        """Get full message by ID."""
        ...

    async def batch_get_messages(
        self,
        message_ids: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> AsyncIterator[GmailMessage]:
        """Fetch multiple messages in batches.

        Args:
            message_ids: List of message IDs to fetch
            batch_size: Number of messages per batch (max 100)

        Yields:
            GmailMessage objects
        """
        ...

    async def get_history(
        self,
        start_history_id: str,
        history_types: list[str] | None = None,
    ) -> AsyncIterator[HistoryRecord]:
        """Get message history changes since history ID.

        Used for incremental sync.
        """
        ...
```

```python
# src/rl_emails/integrations/gmail/rate_limiter.py
import asyncio
from collections import deque
from time import monotonic

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_second: int) -> None:
        self.requests_per_second = requests_per_second
        self.window = deque()

    async def acquire(self) -> None:
        """Wait until a request is allowed."""
        now = monotonic()

        # Remove old timestamps outside window
        while self.window and self.window[0] < now - 1.0:
            self.window.popleft()

        # If at limit, wait
        if len(self.window) >= self.requests_per_second:
            sleep_time = 1.0 - (now - self.window[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.window.append(monotonic())
```

```python
# src/rl_emails/integrations/gmail/parser.py
from rl_emails.integrations.gmail.models import GmailMessage
from rl_emails.core.types import EmailData

def gmail_to_email_data(gmail_msg: GmailMessage, user_id: UUID) -> EmailData:
    """Convert Gmail message to internal EmailData format.

    This bridges the Gmail API format to the format expected by
    pipeline stages 3-11.
    """
    return EmailData(
        message_id=gmail_msg.message_id,
        gmail_id=gmail_msg.id,
        thread_id=gmail_msg.thread_id,
        user_id=str(user_id),
        subject=gmail_msg.subject,
        from_address=gmail_msg.from_address,
        to_addresses=gmail_msg.to_addresses,
        cc_addresses=gmail_msg.cc_addresses,
        date_sent=gmail_msg.date_sent.isoformat(),
        body=gmail_msg.body_plain or "",
        body_html=gmail_msg.body_html,
        in_reply_to=gmail_msg.in_reply_to,
        labels=gmail_msg.label_ids,
    )
```

### Acceptance Criteria

- [ ] `GmailClient.list_messages()` fetches message IDs with pagination
- [ ] `GmailClient.list_messages()` supports date range filtering
- [ ] `GmailClient.list_messages()` supports label filtering
- [ ] `GmailClient.get_message()` fetches full message with headers/body
- [ ] `GmailClient.batch_get_messages()` fetches in batches of 100
- [ ] `RateLimiter` enforces requests per second limit
- [ ] `RateLimiter` handles burst requests gracefully
- [ ] Retry logic with exponential backoff for transient errors
- [ ] Parser converts Gmail format to internal EmailData
- [ ] Parser handles missing fields gracefully
- [ ] All Gmail API errors wrapped in custom exceptions
- [ ] 100% test coverage on new code
- [ ] mypy --strict passes

### Test Plan

```python
# tests/unit/integrations/gmail/test_client.py

class TestGmailClient:
    """Tests for Gmail API client."""

    @patch("httpx.AsyncClient.get")
    async def test_list_messages_returns_refs(self, mock_get):
        """list_messages returns message references."""
        mock_get.return_value.json.return_value = {
            "messages": [
                {"id": "msg1", "threadId": "thread1"},
                {"id": "msg2", "threadId": "thread1"},
            ]
        }

        client = GmailClient(access_token="test")
        refs = [ref async for ref in client.list_messages()]

        assert len(refs) == 2
        assert refs[0].id == "msg1"

    @patch("httpx.AsyncClient.get")
    async def test_list_messages_handles_pagination(self, mock_get):
        """list_messages handles nextPageToken."""
        mock_get.side_effect = [
            Mock(json=lambda: {
                "messages": [{"id": "msg1", "threadId": "t1"}],
                "nextPageToken": "token123",
            }),
            Mock(json=lambda: {
                "messages": [{"id": "msg2", "threadId": "t2"}],
            }),
        ]

        client = GmailClient(access_token="test")
        refs = [ref async for ref in client.list_messages()]

        assert len(refs) == 2

    @patch("httpx.AsyncClient.get")
    async def test_list_messages_with_days_filter(self, mock_get):
        """list_messages builds correct query for days."""
        mock_get.return_value.json.return_value = {"messages": []}

        client = GmailClient(access_token="test")
        _ = [ref async for ref in client.list_messages(days=30)]

        call_args = mock_get.call_args
        assert "after:" in call_args.kwargs["params"]["q"]

    @patch("httpx.AsyncClient.get")
    async def test_get_message_parses_full_message(self, mock_get):
        """get_message parses full message format."""
        mock_get.return_value.json.return_value = {
            "id": "msg1",
            "threadId": "thread1",
            "historyId": "12345",
            "labelIds": ["INBOX", "UNREAD"],
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    {"name": "Message-ID", "value": "<msg@example.com>"},
                ],
                "body": {"data": "SGVsbG8gV29ybGQ="},  # Base64 "Hello World"
            },
            "sizeEstimate": 1234,
        }

        client = GmailClient(access_token="test")
        msg = await client.get_message("msg1")

        assert msg.subject == "Test Subject"
        assert msg.from_address == "sender@example.com"
        assert "Hello World" in msg.body_plain

    @patch("httpx.AsyncClient.post")
    async def test_batch_get_messages_batches_correctly(self, mock_post):
        """batch_get_messages splits into correct batch sizes."""
        # Test that 250 messages are fetched in 3 batches
        message_ids = [f"msg{i}" for i in range(250)]

        mock_post.return_value.json.return_value = {
            "responses": [{"id": f"msg{i}"} for i in range(100)]
        }

        client = GmailClient(access_token="test")
        _ = [msg async for msg in client.batch_get_messages(message_ids)]

        assert mock_post.call_count == 3


class TestRateLimiter:
    """Tests for rate limiter."""

    async def test_allows_requests_within_limit(self):
        """Allows requests within rate limit."""
        limiter = RateLimiter(requests_per_second=10)

        start = monotonic()
        for _ in range(10):
            await limiter.acquire()
        elapsed = monotonic() - start

        assert elapsed < 0.5  # Should be fast

    async def test_throttles_requests_over_limit(self):
        """Throttles when over rate limit."""
        limiter = RateLimiter(requests_per_second=5)

        start = monotonic()
        for _ in range(10):
            await limiter.acquire()
        elapsed = monotonic() - start

        assert elapsed >= 1.0  # Should wait


class TestGmailParser:
    """Tests for Gmail to EmailData parser."""

    def test_converts_gmail_to_email_data(self):
        """Parser converts Gmail format correctly."""
        gmail_msg = GmailMessage(
            id="gmail-id-123",
            thread_id="thread-456",
            message_id="<msg@example.com>",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            ...
        )

        email_data = gmail_to_email_data(gmail_msg, user_id=uuid4())

        assert email_data["gmail_id"] == "gmail-id-123"
        assert email_data["message_id"] == "<msg@example.com>"

    def test_handles_missing_body(self):
        """Parser handles missing body gracefully."""
        gmail_msg = GmailMessage(
            ...,
            body_plain=None,
            body_html=None,
        )

        email_data = gmail_to_email_data(gmail_msg, user_id=uuid4())

        assert email_data["body"] == ""
```

### Verification Steps

1. **Unit tests pass**:
   ```bash
   make test
   ```

2. **Manual API test** (after auth connected):
   ```python
   # In Python REPL
   from rl_emails.integrations.gmail.client import GmailClient
   from rl_emails.services.auth_service import AuthService

   # Get valid token
   token = await auth_service.get_valid_token(user_id)

   # Create client
   client = GmailClient(access_token=token)

   # List recent messages
   async for ref in client.list_messages(days=7, max_results=10):
       print(ref.id, ref.thread_id)

   # Get full message
   msg = await client.get_message(ref.id)
   print(msg.subject, msg.from_address)
   ```

3. **Rate limiting test**:
   ```python
   # Should take ~10 seconds for 100 requests at 10/sec
   start = time.time()
   async for ref in client.list_messages(max_results=100):
       pass
   print(f"Elapsed: {time.time() - start}s")
   ```

4. **Run full check**:
   ```bash
   make check
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| Message listing | Paginated list with filters |
| Batch fetching | 100 messages per batch |
| Rate limiting | Configurable requests/second |
| Retry logic | Exponential backoff on errors |
| Parsing | Gmail to EmailData conversion |
| Type safety | mypy --strict passes |
| Test coverage | 100% on new code |

---

## Iteration 6: Initial Sync Implementation

### Story
As a user, I need to sync my Gmail emails so that I can run the analysis pipeline on my real email data.

### Deliverables
1. Gmail sync service
2. New pipeline stage for Gmail ingestion
3. CLI sync commands
4. Sync state tracking

### Architecture

```
src/rl_emails/
├── pipeline/
│   └── stages/
│       └── stage_00_gmail_sync.py  # NEW: Gmail sync stage (before stage 1)
├── services/
│   └── sync_service.py             # NEW: Sync orchestration
```

**Key Design Decision**: Gmail sync is a **new stage 0** that feeds into the existing pipeline. MBOX stages 1-2 remain unchanged.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Option A: MBOX (unchanged)          Option B: Gmail API (new)      │
│  ┌───────────────────────┐           ┌───────────────────────┐      │
│  │ Stage 1: Parse MBOX   │           │ Stage 0: Gmail Sync   │      │
│  │ Stage 2: Import PG    │           │ (replaces 1-2)        │      │
│  └───────────┬───────────┘           └───────────┬───────────┘      │
│              │                                   │                   │
│              └─────────────┬─────────────────────┘                   │
│                            │                                         │
│                            ▼                                         │
│              ┌─────────────────────────┐                            │
│              │ Stages 3-11 (shared)    │                            │
│              │ Threads, Features,      │                            │
│              │ Embeddings, Clustering  │                            │
│              └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Files to Create

| File | Description |
|------|-------------|
| `src/rl_emails/pipeline/stages/stage_00_gmail_sync.py` | Gmail sync stage |
| `src/rl_emails/services/sync_service.py` | Sync orchestration |
| `tests/unit/pipeline/stages/test_stage_00_gmail_sync.py` | Sync stage tests |
| `tests/unit/services/test_sync_service.py` | Sync service tests |

### Files to Modify

| File | Changes |
|------|---------|
| `src/rl_emails/cli.py` | Add `sync` subcommand |
| `src/rl_emails/pipeline/orchestrator.py` | Support Gmail mode |
| `src/rl_emails/repositories/sync_state.py` | Add sync state methods |

### Implementation Design

```python
# src/rl_emails/services/sync_service.py
from uuid import UUID
from datetime import datetime

class SyncService:
    """Orchestrates Gmail sync operations."""

    def __init__(
        self,
        gmail_client: GmailClient,
        sync_repo: SyncStateRepository,
        config: Config,
    ) -> None:
        self.gmail_client = gmail_client
        self.sync_repo = sync_repo
        self.config = config

    async def initial_sync(
        self,
        user_id: UUID,
        days: int = 30,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SyncResult:
        """Perform initial full sync.

        Args:
            user_id: User to sync for
            days: Number of days to sync
            progress_callback: Called with (processed, total)

        Returns:
            SyncResult with counts and any errors
        """
        # 1. Mark sync as in progress
        await self.sync_repo.start_sync(user_id)

        try:
            # 2. List all messages in date range
            message_refs = []
            async for ref in self.gmail_client.list_messages(days=days):
                message_refs.append(ref)

            total = len(message_refs)
            processed = 0

            # 3. Batch fetch and store
            async for msg in self.gmail_client.batch_get_messages(
                [r.id for r in message_refs]
            ):
                await self._store_message(user_id, msg)
                processed += 1

                if progress_callback:
                    progress_callback(processed, total)

            # 4. Record sync state
            last_history_id = message_refs[0].history_id if message_refs else None
            await self.sync_repo.complete_sync(
                user_id,
                history_id=last_history_id,
                emails_synced=processed,
            )

            return SyncResult(
                success=True,
                emails_synced=processed,
                history_id=last_history_id,
            )

        except Exception as e:
            await self.sync_repo.fail_sync(user_id, str(e))
            raise

    async def _store_message(self, user_id: UUID, msg: GmailMessage) -> None:
        """Store Gmail message in database."""
        email_data = gmail_to_email_data(msg, user_id)
        # Insert into emails table
        ...
```

```python
# src/rl_emails/pipeline/stages/stage_00_gmail_sync.py
from rl_emails.pipeline.stages.base import StageResult
from rl_emails.core.config import Config
from rl_emails.services.sync_service import SyncService

def run(config: Config, days: int = 30) -> StageResult:
    """Run Gmail sync stage.

    This stage syncs emails from Gmail API and stores them in the database.
    It replaces stages 1-2 (parse_mbox, import_postgres) for Gmail users.

    Args:
        config: Config with user_id and database URL
        days: Number of days to sync (default 30)

    Returns:
        StageResult with sync statistics
    """
    if not config.user_id:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="Gmail sync requires --user flag",
        )

    async def _run() -> StageResult:
        start = time.time()

        # Get valid token
        auth_service = AuthService(...)
        token = await auth_service.get_valid_token(config.user_id)

        # Create sync service
        gmail_client = GmailClient(access_token=token)
        sync_service = SyncService(gmail_client, ...)

        # Run sync
        result = await sync_service.initial_sync(
            user_id=config.user_id,
            days=days,
        )

        duration = time.time() - start
        return StageResult(
            success=result.success,
            records_processed=result.emails_synced,
            duration_seconds=duration,
            message=f"Synced {result.emails_synced} emails from last {days} days",
        )

    return asyncio.run(_run())
```

### CLI Commands

```bash
# Initial sync (last 30 days)
rl-emails sync --user <uuid>

# Sync specific date range
rl-emails sync --user <uuid> --days 90

# Sync since specific date
rl-emails sync --user <uuid> --since 2024-01-01

# Check sync status
rl-emails sync --user <uuid> --status

# Full pipeline with Gmail source
rl-emails --user <uuid> --source gmail --days 30
```

### Database Updates

```sql
-- Update sync_state when sync completes
UPDATE sync_state SET
    last_history_id = 'history-id-from-gmail',
    last_sync_at = now(),
    sync_status = 'idle',
    emails_synced = 1234
WHERE user_id = '<user-uuid>';

-- Query sync state
SELECT
    sync_status,
    last_sync_at,
    emails_synced,
    error_message
FROM sync_state
WHERE user_id = '<user-uuid>';
```

### Acceptance Criteria

- [ ] `SyncService.initial_sync()` fetches and stores emails
- [ ] Sync respects `--days` parameter
- [ ] Sync tracks progress with callback
- [ ] Sync state stored in database (`sync_state` table)
- [ ] Sync handles errors and stores error message
- [ ] Gmail emails stored in `emails` table with `gmail_id`
- [ ] `stage_00_gmail_sync` integrates with orchestrator
- [ ] CLI `sync` command works end-to-end
- [ ] CLI `--source gmail` runs Gmail path instead of MBOX
- [ ] MBOX pipeline still works unchanged
- [ ] 100% test coverage on new code
- [ ] mypy --strict passes

### Test Plan

```python
# tests/unit/services/test_sync_service.py

class TestSyncService:
    """Tests for sync service."""

    @patch("rl_emails.integrations.gmail.client.GmailClient")
    async def test_initial_sync_fetches_messages(self, mock_client):
        """Initial sync fetches and stores messages."""
        mock_client.list_messages.return_value = AsyncIterator([
            GmailMessageRef(id="msg1", thread_id="t1"),
            GmailMessageRef(id="msg2", thread_id="t2"),
        ])
        mock_client.batch_get_messages.return_value = AsyncIterator([
            GmailMessage(id="msg1", ...),
            GmailMessage(id="msg2", ...),
        ])

        service = SyncService(mock_client, ...)
        result = await service.initial_sync(user_id, days=30)

        assert result.success
        assert result.emails_synced == 2

    async def test_initial_sync_updates_sync_state(self, mock_client, db_session):
        """Sync updates sync_state table."""
        service = SyncService(...)
        await service.initial_sync(user_id, days=30)

        state = await sync_repo.get_by_user(user_id)
        assert state.sync_status == "idle"
        assert state.last_sync_at is not None

    async def test_initial_sync_handles_error(self, mock_client):
        """Sync handles and records errors."""
        mock_client.list_messages.side_effect = GmailApiError("Rate limited")

        service = SyncService(...)
        with pytest.raises(GmailApiError):
            await service.initial_sync(user_id, days=30)

        state = await sync_repo.get_by_user(user_id)
        assert state.sync_status == "error"
        assert "Rate limited" in state.error_message


# tests/unit/pipeline/stages/test_stage_00_gmail_sync.py

class TestGmailSyncStage:
    """Tests for Gmail sync stage."""

    def test_requires_user_id(self):
        """Stage fails without user_id."""
        config = Config(database_url="...", user_id=None)
        result = stage_00_gmail_sync.run(config)

        assert not result.success
        assert "requires --user" in result.message

    @patch("rl_emails.services.sync_service.SyncService")
    def test_runs_sync_with_days(self, mock_service):
        """Stage passes days parameter."""
        config = Config(database_url="...", user_id=uuid4())
        mock_service.initial_sync.return_value = SyncResult(
            success=True, emails_synced=100
        )

        result = stage_00_gmail_sync.run(config, days=60)

        mock_service.initial_sync.assert_called_with(
            user_id=config.user_id,
            days=60,
        )
```

### Verification Steps

1. **Connect Gmail first**:
   ```bash
   rl-emails auth connect --email your@gmail.com
   ```

2. **Run initial sync**:
   ```bash
   rl-emails sync --user <uuid> --days 30

   # Expected output:
   # Syncing emails from last 30 days...
   # [============================] 100% (1234/1234)
   # Synced 1234 emails in 45.2s
   ```

3. **Verify data in database**:
   ```sql
   SELECT COUNT(*) FROM emails WHERE user_id = '<uuid>';
   -- Should match synced count

   SELECT * FROM sync_state WHERE user_id = '<uuid>';
   -- Should show idle status, last_sync_at, etc.
   ```

4. **Run full pipeline**:
   ```bash
   # With Gmail source (skips stages 1-2)
   rl-emails --user <uuid> --source gmail

   # Stages 3-11 run on synced data
   ```

5. **Verify MBOX still works**:
   ```bash
   # MBOX path unchanged
   rl-emails  # Without --user, uses MBOX
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| Initial sync | Fetches all emails in date range |
| Data storage | Emails stored with gmail_id |
| State tracking | Sync state in database |
| Progress | Real-time progress feedback |
| Error handling | Errors recorded, clear messages |
| Integration | Works with orchestrator |
| Backward compat | MBOX pipeline unaffected |
| Type safety | mypy --strict passes |
| Test coverage | 100% on new code |

---

## Iteration 7: Incremental Sync

### Story
As a user, I need incremental sync so that only new emails are fetched, making subsequent syncs fast and efficient.

### Deliverables
1. Gmail History API integration
2. Delta sync processing (added/modified/deleted)
3. Efficient re-sync workflow
4. CLI incremental sync command

### Architecture

```
src/rl_emails/
├── integrations/
│   └── gmail/
│       └── history.py            # NEW: History API wrapper
├── services/
│   └── delta_processor.py        # NEW: Process sync deltas
```

### Files to Create

| File | Description |
|------|-------------|
| `src/rl_emails/integrations/gmail/history.py` | History API wrapper |
| `src/rl_emails/services/delta_processor.py` | Delta processing logic |
| `tests/unit/integrations/gmail/test_history.py` | History API tests |
| `tests/unit/services/test_delta_processor.py` | Delta processor tests |

### Files to Modify

| File | Changes |
|------|---------|
| `src/rl_emails/services/sync_service.py` | Add incremental_sync method |
| `src/rl_emails/cli.py` | Add `--incremental` flag |

### Implementation Design

```python
# src/rl_emails/integrations/gmail/history.py
from dataclasses import dataclass
from enum import Enum

class HistoryType(Enum):
    MESSAGE_ADDED = "messageAdded"
    MESSAGE_DELETED = "messageDeleted"
    LABEL_ADDED = "labelAdded"
    LABEL_REMOVED = "labelRemoved"

@dataclass
class HistoryRecord:
    """A single history change record."""
    id: str
    history_type: HistoryType
    message_id: str
    label_ids: list[str] | None = None

async def get_history_changes(
    client: GmailClient,
    start_history_id: str,
) -> tuple[list[HistoryRecord], str]:
    """Get all history changes since history ID.

    Returns:
        Tuple of (history records, new history ID)
    """
    records = []
    new_history_id = start_history_id

    async for record in client.get_history(start_history_id):
        records.append(record)
        new_history_id = record.id

    return records, new_history_id
```

```python
# src/rl_emails/services/delta_processor.py
from uuid import UUID
from rl_emails.integrations.gmail.history import HistoryRecord, HistoryType

@dataclass
class DeltaResult:
    """Result of processing deltas."""
    messages_added: int
    messages_deleted: int
    labels_updated: int

class DeltaProcessor:
    """Processes sync deltas from Gmail History API."""

    async def process_deltas(
        self,
        user_id: UUID,
        records: list[HistoryRecord],
        gmail_client: GmailClient,
    ) -> DeltaResult:
        """Process history records.

        Args:
            user_id: User ID for database operations
            records: History records to process
            gmail_client: Client to fetch new messages

        Returns:
            DeltaResult with counts
        """
        added = []
        deleted = []
        label_updates = []

        for record in records:
            if record.history_type == HistoryType.MESSAGE_ADDED:
                added.append(record.message_id)
            elif record.history_type == HistoryType.MESSAGE_DELETED:
                deleted.append(record.message_id)
            elif record.history_type in (HistoryType.LABEL_ADDED, HistoryType.LABEL_REMOVED):
                label_updates.append(record)

        # Fetch and store new messages
        if added:
            async for msg in gmail_client.batch_get_messages(added):
                await self._store_message(user_id, msg)

        # Mark deleted messages
        if deleted:
            await self._mark_deleted(user_id, deleted)

        # Update labels
        for update in label_updates:
            await self._update_labels(user_id, update)

        return DeltaResult(
            messages_added=len(added),
            messages_deleted=len(deleted),
            labels_updated=len(label_updates),
        )
```

```python
# src/rl_emails/services/sync_service.py (additions)

class SyncService:
    # ... existing methods ...

    async def incremental_sync(
        self,
        user_id: UUID,
        progress_callback: Callable[[str], None] | None = None,
    ) -> SyncResult:
        """Perform incremental sync using History API.

        Only fetches changes since last sync.

        Args:
            user_id: User to sync for
            progress_callback: Called with status messages

        Returns:
            SyncResult with delta counts
        """
        # 1. Get last history ID
        sync_state = await self.sync_repo.get_by_user(user_id)
        if not sync_state or not sync_state.last_history_id:
            # No previous sync, need initial sync
            return SyncResult(
                success=False,
                message="No previous sync found. Run initial sync first.",
            )

        await self.sync_repo.start_sync(user_id)

        try:
            # 2. Get history changes
            records, new_history_id = await get_history_changes(
                self.gmail_client,
                sync_state.last_history_id,
            )

            if not records:
                await self.sync_repo.complete_sync(user_id, new_history_id, 0)
                return SyncResult(
                    success=True,
                    message="No new changes",
                    emails_synced=0,
                )

            # 3. Process deltas
            delta_processor = DeltaProcessor(...)
            result = await delta_processor.process_deltas(
                user_id, records, self.gmail_client
            )

            # 4. Update sync state
            await self.sync_repo.complete_sync(
                user_id,
                history_id=new_history_id,
                emails_synced=result.messages_added,
            )

            return SyncResult(
                success=True,
                emails_synced=result.messages_added,
                emails_deleted=result.messages_deleted,
                labels_updated=result.labels_updated,
                history_id=new_history_id,
            )

        except HistoryIdExpired:
            # History ID too old, need full re-sync
            await self.sync_repo.fail_sync(
                user_id,
                "History expired. Run full sync with --days."
            )
            return SyncResult(
                success=False,
                message="History expired. Run: rl-emails sync --user <id> --days 30",
            )
```

### CLI Commands

```bash
# Incremental sync (uses last history ID)
rl-emails sync --user <uuid> --incremental

# Force full re-sync
rl-emails sync --user <uuid> --days 30 --force

# Auto mode: incremental if possible, full if needed
rl-emails sync --user <uuid>

# Show sync history
rl-emails sync --user <uuid> --history
```

### Acceptance Criteria

- [ ] `get_history_changes()` fetches history since ID
- [ ] `DeltaProcessor` handles MESSAGE_ADDED
- [ ] `DeltaProcessor` handles MESSAGE_DELETED
- [ ] `DeltaProcessor` handles LABEL_ADDED/REMOVED
- [ ] `incremental_sync()` uses stored history ID
- [ ] Incremental sync updates history ID after completion
- [ ] Handles expired history ID gracefully
- [ ] CLI `--incremental` flag works
- [ ] Auto-detection of sync mode
- [ ] 100% test coverage on new code
- [ ] mypy --strict passes

### Test Plan

```python
# tests/unit/integrations/gmail/test_history.py

class TestGmailHistory:
    """Tests for Gmail History API."""

    @patch("httpx.AsyncClient.get")
    async def test_get_history_returns_records(self, mock_get):
        """get_history returns history records."""
        mock_get.return_value.json.return_value = {
            "history": [
                {
                    "id": "12346",
                    "messagesAdded": [{"message": {"id": "msg1"}}],
                },
                {
                    "id": "12347",
                    "messagesDeleted": [{"message": {"id": "msg2"}}],
                },
            ],
            "historyId": "12347",
        }

        records, new_id = await get_history_changes(client, "12345")

        assert len(records) == 2
        assert records[0].history_type == HistoryType.MESSAGE_ADDED
        assert records[1].history_type == HistoryType.MESSAGE_DELETED
        assert new_id == "12347"

    @patch("httpx.AsyncClient.get")
    async def test_handles_expired_history_id(self, mock_get):
        """Raises HistoryIdExpired for 404."""
        mock_get.return_value.status_code = 404
        mock_get.return_value.json.return_value = {
            "error": {"code": 404, "message": "historyId not found"}
        }

        with pytest.raises(HistoryIdExpired):
            await get_history_changes(client, "old-id")


# tests/unit/services/test_delta_processor.py

class TestDeltaProcessor:
    """Tests for delta processor."""

    async def test_processes_added_messages(self, mock_client):
        """Fetches and stores added messages."""
        records = [
            HistoryRecord(
                id="1",
                history_type=HistoryType.MESSAGE_ADDED,
                message_id="msg1",
            ),
        ]
        mock_client.batch_get_messages.return_value = AsyncIterator([
            GmailMessage(id="msg1", ...)
        ])

        processor = DeltaProcessor(...)
        result = await processor.process_deltas(user_id, records, mock_client)

        assert result.messages_added == 1

    async def test_marks_deleted_messages(self):
        """Marks deleted messages in database."""
        records = [
            HistoryRecord(
                id="1",
                history_type=HistoryType.MESSAGE_DELETED,
                message_id="msg1",
            ),
        ]

        processor = DeltaProcessor(...)
        result = await processor.process_deltas(user_id, records, mock_client)

        assert result.messages_deleted == 1
        # Verify message marked as deleted in DB
```

### Verification Steps

1. **Initial sync first**:
   ```bash
   rl-emails sync --user <uuid> --days 30
   ```

2. **Wait for new emails** (or send yourself a test email)

3. **Run incremental sync**:
   ```bash
   rl-emails sync --user <uuid> --incremental

   # Expected output:
   # Checking for changes since last sync...
   # Found 3 new messages, 0 deleted
   # Synced 3 emails in 2.1s
   ```

4. **Verify sync state updated**:
   ```sql
   SELECT last_history_id, last_sync_at, emails_synced
   FROM sync_state
   WHERE user_id = '<uuid>';
   ```

5. **Test expired history**:
   ```sql
   -- Set old history ID
   UPDATE sync_state
   SET last_history_id = '1'
   WHERE user_id = '<uuid>';
   ```
   ```bash
   rl-emails sync --user <uuid> --incremental
   # Should suggest full re-sync
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| History API | Fetches changes since history ID |
| Delta processing | Handles add/delete/label changes |
| Incremental sync | Fast sync using history ID |
| Expired history | Graceful fallback to full sync |
| CLI integration | --incremental flag works |
| Type safety | mypy --strict passes |
| Test coverage | 100% on new code |

---

## Phase 2 Completion Checklist

### Pre-Implementation
- [ ] Gmail API credentials configured in .env
- [ ] Phase 1 complete (multi-tenant foundation)
- [ ] Test Gmail account available

### Iteration 4: OAuth2 Flow
- [ ] Create auth module structure
- [ ] Implement GoogleOAuth class
- [ ] Implement AuthService
- [ ] Add CLI auth commands
- [ ] Update Config for Google credentials
- [ ] Write tests (100% coverage)
- [ ] Verify end-to-end OAuth flow

### Iteration 5: Gmail API Client
- [ ] Create integrations/gmail module
- [ ] Implement GmailClient
- [ ] Implement rate limiter
- [ ] Implement Gmail parser
- [ ] Write tests (100% coverage)
- [ ] Manual API verification

### Iteration 6: Initial Sync
- [ ] Implement SyncService
- [ ] Create stage_00_gmail_sync
- [ ] Add CLI sync commands
- [ ] Integrate with orchestrator
- [ ] Write tests (100% coverage)
- [ ] End-to-end sync test

### Iteration 7: Incremental Sync
- [ ] Implement History API wrapper
- [ ] Implement DeltaProcessor
- [ ] Add incremental_sync method
- [ ] Handle expired history IDs
- [ ] Write tests (100% coverage)
- [ ] Verify incremental sync works

### Post-Implementation
- [ ] Run `make check` (all pass)
- [ ] Update CLAUDE.md with progress
- [ ] Manual testing with real Gmail account
- [ ] Document any rate limiting observations

---

## Technical Notes

### Rate Limiting Strategy

Gmail API has per-user quotas:
- 25,000 queries per 100 seconds per user
- Batch requests count as 1 quota unit

Our implementation:
```python
DEFAULT_REQUESTS_PER_SECOND = 10  # Conservative
BATCH_SIZE = 100                   # Max allowed
RETRY_DELAYS = [1, 2, 4, 8, 16]   # Exponential backoff
```

### Error Handling

| Error | Handling |
|-------|----------|
| 401 Unauthorized | Refresh token, retry |
| 403 Rate Limited | Exponential backoff |
| 404 History Expired | Prompt full re-sync |
| 5xx Server Error | Retry with backoff |

### Security Considerations

1. **Token Storage**: Tokens encrypted at rest (consider adding column encryption)
2. **Scope Minimization**: Only `gmail.readonly` requested
3. **Token Refresh**: Automatic refresh before expiry
4. **Revocation**: Clean disconnect removes all tokens

### Testing with Real Gmail

For integration testing:
1. Use a dedicated test Gmail account
2. Send test emails with known content
3. Verify sync captures them correctly
4. Test incremental sync with new emails

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing ...

    # Gmail API (new)
    "google-auth>=2.0.0",
    "google-auth-oauthlib>=1.0.0",
    "google-api-python-client>=2.0.0",

    # HTTP client (if not using google client)
    "httpx>=0.25.0",
]
```

---

## Success Criteria Summary

| Iteration | Key Deliverable | Verification |
|-----------|-----------------|--------------|
| 4 | OAuth flow | `rl-emails auth connect` works |
| 5 | Gmail client | Can list and fetch messages |
| 6 | Initial sync | Full sync populates database |
| 7 | Incremental | Delta sync is fast and correct |

**Phase 2 Complete When**:
- User can connect Gmail via OAuth
- Initial sync fetches all emails in date range
- Incremental sync fetches only new changes
- Full pipeline works with Gmail data
- MBOX pipeline still works unchanged
- 100% test coverage maintained
