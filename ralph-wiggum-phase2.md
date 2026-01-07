# Ralph Wiggum Phase 2: Gmail API Integration

**Goal**: Add Gmail API as an alternative data source alongside existing MBOX pipeline
- OAuth2 authentication flow
- Gmail API client with rate limiting
- Initial and incremental sync
- Full integration with existing pipeline stages 3-11

**Status**: PENDING

**Prerequisite**: Phase 1 Complete (multi-tenant foundation)

---

## Overview

Phase 2 adds Gmail API integration as an alternative to MBOX file ingestion:

```
                    +------------------+
                    |   Data Sources   |
                    +--------+---------+
                             |
           +-----------------+------------------+
           |                                    |
    +------v------+                    +--------v-------+
    |  MBOX File  |                    |   Gmail API    |
    |  (Stage 1-2)|                    |  (New Stage 0) |
    +------+------+                    +--------+-------+
           |                                    |
           +----------------+-------------------+
                            |
                   +--------v---------+
                   |  Shared Pipeline |
                   |   (Stages 3-11)  |
                   +------------------+
```

**Key Principle**: MBOX pipeline remains unchanged. Gmail API is additive.

---

## Iteration 1: Auth Module Setup + OAuth Types

**Goal**: Create authentication module structure and OAuth base types

**Success Criteria**:
- [ ] `src/rl_emails/auth/` directory created with proper structure
- [ ] Dependencies added to pyproject.toml (google-auth, google-auth-oauthlib, httpx)
- [ ] `oauth.py` with base types and `OAuthError` exception
- [ ] `GoogleTokens` dataclass created
- [ ] Tests pass with 100% coverage on new code
- [ ] `make check` passes

**Tasks**:
1. Create directory structure:
   ```bash
   mkdir -p src/rl_emails/auth
   touch src/rl_emails/auth/__init__.py
   ```

2. Add dependencies to `pyproject.toml`:
   ```toml
   dependencies = [
       # ... existing ...
       "google-auth>=2.0.0",
       "google-auth-oauthlib>=1.0.0",
       "httpx>=0.25.0",
   ]
   ```

3. Create `src/rl_emails/auth/oauth.py`:
   ```python
   """OAuth2 base types and utilities."""
   from dataclasses import dataclass
   from datetime import datetime

   class OAuthError(Exception):
       """Base exception for OAuth errors."""
       def __init__(self, error: str, description: str | None = None) -> None:
           self.error = error
           self.description = description
           super().__init__(f"{error}: {description}" if description else error)

   @dataclass
   class GoogleTokens:
       """OAuth tokens from Google."""
       access_token: str
       refresh_token: str
       expires_at: datetime
       scopes: list[str]

       def is_expired(self) -> bool:
           """Check if access token is expired."""
           return datetime.now() >= self.expires_at
   ```

4. Create `src/rl_emails/auth/__init__.py` with exports

5. Create `tests/unit/auth/__init__.py` and `tests/unit/auth/test_oauth.py`

6. Run `uv sync` to install new dependencies

7. Commit: "Iteration 1: Auth module setup and OAuth types"

**Verification Contract**:
```bash
# MUST have auth module structure
test -d src/rl_emails/auth && echo "PASS: auth/ exists"
test -f src/rl_emails/auth/oauth.py && echo "PASS: oauth.py exists"
test -f src/rl_emails/auth/__init__.py && echo "PASS: __init__.py exists"

# MUST import successfully
uv run python -c "from rl_emails.auth.oauth import GoogleTokens, OAuthError; print('PASS: Import works')"

# Dependencies MUST be installed
uv run python -c "import google.auth; import httpx; print('PASS: Dependencies installed')"

# Tests MUST pass
uv run pytest tests/unit/auth/ -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: All verification commands exit with code 0
- FAIL: Any command fails or import error

---

## Iteration 2: GoogleOAuth Implementation

**Goal**: Implement Google OAuth2 class with authorization URL, code exchange, and token refresh

**Success Criteria**:
- [ ] `GoogleOAuth` class in `src/rl_emails/auth/google.py`
- [ ] `get_authorization_url()` generates proper OAuth URL with scopes
- [ ] `exchange_code()` exchanges authorization code for tokens
- [ ] `refresh_token()` refreshes expired access tokens
- [ ] All error cases handled with `OAuthError`
- [ ] Tests with mocking for HTTP calls
- [ ] `make check` passes

**Tasks**:
1. Create `src/rl_emails/auth/google.py`:
   ```python
   """Google OAuth2 implementation."""
   from urllib.parse import urlencode
   import httpx
   from datetime import datetime, timedelta
   from rl_emails.auth.oauth import GoogleTokens, OAuthError

   class GoogleOAuth:
       """Google OAuth2 implementation."""

       AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
       TOKEN_URL = "https://oauth2.googleapis.com/token"
       SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

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

2. Implement all three methods with proper error handling

3. Create `tests/unit/auth/test_google.py` with mocked HTTP responses

4. Update `src/rl_emails/auth/__init__.py` exports

5. Commit: "Iteration 2: GoogleOAuth implementation"

**Verification Contract**:
```bash
# MUST have google.py
test -f src/rl_emails/auth/google.py && echo "PASS: google.py exists"

# MUST import successfully
uv run python -c "from rl_emails.auth.google import GoogleOAuth; print('PASS: GoogleOAuth imports')"

# MUST have authorization URL with scopes
uv run python -c "
from rl_emails.auth.google import GoogleOAuth
oauth = GoogleOAuth('test-id', 'test-secret', 'http://localhost/callback')
url = oauth.get_authorization_url(state='test-state')
assert 'gmail.readonly' in url, 'Missing scope'
assert 'test-id' in url, 'Missing client_id'
assert 'test-state' in url, 'Missing state'
print('PASS: Authorization URL correct')
"

# Tests MUST pass
uv run pytest tests/unit/auth/test_google.py -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: Authorization URL contains required parameters, tests pass
- FAIL: Missing parameters or test failures

---

## Iteration 3: OAuthToken Repository

**Goal**: Create repository for OAuth token CRUD operations

**Success Criteria**:
- [ ] `OAuthTokenRepository` class with async CRUD operations
- [ ] `create()`, `get_by_user()`, `update()`, `delete()` methods
- [ ] Encryption consideration documented (future enhancement)
- [ ] Tests with database mocking
- [ ] `make check` passes

**Tasks**:
1. Create `src/rl_emails/repositories/oauth_token.py`:
   ```python
   """Repository for OAuth token operations."""
   from uuid import UUID
   from sqlalchemy.ext.asyncio import AsyncSession
   from rl_emails.models.oauth_token import OAuthToken

   class OAuthTokenRepository:
       """Repository for OAuth token CRUD operations."""

       def __init__(self, session: AsyncSession) -> None:
           self.session = session

       async def create(self, user_id: UUID, provider: str, ...) -> OAuthToken:
           """Create new OAuth token."""
           ...

       async def get_by_user(self, user_id: UUID, provider: str = "google") -> OAuthToken | None:
           """Get token by user ID and provider."""
           ...

       async def update(self, token: OAuthToken) -> OAuthToken:
           """Update existing token."""
           ...

       async def delete(self, user_id: UUID, provider: str = "google") -> bool:
           """Delete token by user and provider."""
           ...
   ```

2. Update `src/rl_emails/repositories/__init__.py` exports

3. Create `tests/unit/repositories/test_oauth_token.py`

4. Commit: "Iteration 3: OAuthToken repository"

**Verification Contract**:
```bash
# MUST have repository file
test -f src/rl_emails/repositories/oauth_token.py && echo "PASS: oauth_token.py exists"

# MUST import successfully
uv run python -c "from rl_emails.repositories.oauth_token import OAuthTokenRepository; print('PASS: Repository imports')"

# Tests MUST pass
uv run pytest tests/unit/repositories/test_oauth_token.py -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: Repository methods work, tests pass
- FAIL: Any method fails or test failures

---

## Iteration 4: AuthService Implementation

**Goal**: Create authentication service for orchestrating OAuth flows

**Success Criteria**:
- [ ] `src/rl_emails/services/` directory created
- [ ] `AuthService` class with full OAuth flow orchestration
- [ ] `start_auth_flow()` returns authorization URL
- [ ] `complete_auth_flow()` exchanges code and stores tokens
- [ ] `get_valid_token()` returns valid token, refreshing if needed
- [ ] `revoke_token()` deletes user's tokens
- [ ] Tests with mocking
- [ ] `make check` passes

**Tasks**:
1. Create directory structure:
   ```bash
   mkdir -p src/rl_emails/services
   touch src/rl_emails/services/__init__.py
   ```

2. Create `src/rl_emails/services/auth_service.py`:
   ```python
   """Authentication service for OAuth flow orchestration."""
   from uuid import UUID
   from rl_emails.auth.google import GoogleOAuth
   from rl_emails.repositories.oauth_token import OAuthTokenRepository

   class AuthService:
       """Service for managing authentication."""

       def __init__(
           self,
           oauth: GoogleOAuth,
           token_repo: OAuthTokenRepository,
       ) -> None:
           self.oauth = oauth
           self.token_repo = token_repo

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

3. Create `tests/unit/services/__init__.py` and `tests/unit/services/test_auth_service.py`

4. Commit: "Iteration 4: AuthService implementation"

**Verification Contract**:
```bash
# MUST have services directory
test -d src/rl_emails/services && echo "PASS: services/ exists"
test -f src/rl_emails/services/auth_service.py && echo "PASS: auth_service.py exists"

# MUST import successfully
uv run python -c "from rl_emails.services.auth_service import AuthService; print('PASS: AuthService imports')"

# Tests MUST pass
uv run pytest tests/unit/services/test_auth_service.py -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: All AuthService methods work, tests pass
- FAIL: Any method fails or test failures

---

## Iteration 5: CLI Auth Commands + Config

**Goal**: Add auth CLI commands and Google OAuth config

**Success Criteria**:
- [ ] `Config` updated with `google_client_id`, `google_client_secret`, `google_redirect_uri`
- [ ] `Config.has_google_oauth()` method
- [ ] CLI `auth connect --email` command (opens browser)
- [ ] CLI `auth status --email` command
- [ ] CLI `auth disconnect --email` command
- [ ] Tests for CLI and config
- [ ] `make check` passes

**Tasks**:
1. Update `src/rl_emails/core/config.py`:
   ```python
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

2. Add auth subcommand to `src/rl_emails/cli.py`:
   ```python
   @app.command()
   def auth(
       action: str = typer.Argument(..., help="connect|status|disconnect"),
       email: str = typer.Option(..., "--email", help="Gmail address"),
   ) -> None:
       """Manage Gmail authentication."""
       ...
   ```

3. Update `tests/unit/core/test_config.py` for new fields

4. Create `tests/unit/test_cli_auth.py`

5. Commit: "Iteration 5: CLI auth commands and config"

**Verification Contract**:
```bash
# MUST have updated config
uv run python -c "
from rl_emails.core.config import Config
c = Config(database_url='test', google_client_id='id', google_client_secret='secret')
assert c.has_google_oauth() == True
c2 = Config(database_url='test')
assert c2.has_google_oauth() == False
print('PASS: Config updated')
"

# CLI help MUST show auth command
uv run python -m rl_emails.cli --help | grep -q "auth" && echo "PASS: auth command exists"

# Tests MUST pass
uv run pytest tests/unit/core/test_config.py tests/unit/test_cli_auth.py -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: Config has Google fields, CLI shows auth command, tests pass
- FAIL: Missing fields or test failures

---

## Iteration 6: Gmail Integration Module + Models

**Goal**: Create Gmail integration module with data models and rate limiter

**Success Criteria**:
- [ ] `src/rl_emails/integrations/gmail/` directory created
- [ ] `models.py` with `GmailMessageRef` and `GmailMessage` dataclasses
- [ ] `rate_limiter.py` with token bucket `RateLimiter` class
- [ ] Rate limiter tests verify throttling behavior
- [ ] `make check` passes

**Tasks**:
1. Create directory structure:
   ```bash
   mkdir -p src/rl_emails/integrations/gmail
   touch src/rl_emails/integrations/__init__.py
   touch src/rl_emails/integrations/gmail/__init__.py
   ```

2. Create `src/rl_emails/integrations/gmail/models.py`:
   ```python
   """Gmail API data models."""
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
       message_id: str
       subject: str
       from_address: str
       to_addresses: list[str]
       cc_addresses: list[str]
       date_sent: datetime
       in_reply_to: str | None
       references: list[str]
       snippet: str
       body_plain: str | None
       body_html: str | None
       size_bytes: int
       has_attachments: bool
   ```

3. Create `src/rl_emails/integrations/gmail/rate_limiter.py`:
   ```python
   """Rate limiting utilities."""
   import asyncio
   from collections import deque
   from time import monotonic

   class RateLimiter:
       """Token bucket rate limiter."""
       def __init__(self, requests_per_second: int) -> None:
           ...

       async def acquire(self) -> None:
           """Wait until a request is allowed."""
           ...
   ```

4. Create `tests/unit/integrations/gmail/test_models.py` and `test_rate_limiter.py`

5. Commit: "Iteration 6: Gmail integration module and models"

**Verification Contract**:
```bash
# MUST have gmail integration structure
test -d src/rl_emails/integrations/gmail && echo "PASS: gmail/ exists"
test -f src/rl_emails/integrations/gmail/models.py && echo "PASS: models.py exists"
test -f src/rl_emails/integrations/gmail/rate_limiter.py && echo "PASS: rate_limiter.py exists"

# MUST import successfully
uv run python -c "
from rl_emails.integrations.gmail.models import GmailMessageRef, GmailMessage
from rl_emails.integrations.gmail.rate_limiter import RateLimiter
print('PASS: Gmail models import')
"

# Rate limiter MUST throttle
uv run python -c "
import asyncio
from time import monotonic
from rl_emails.integrations.gmail.rate_limiter import RateLimiter

async def test():
    limiter = RateLimiter(requests_per_second=5)
    start = monotonic()
    for _ in range(10):
        await limiter.acquire()
    elapsed = monotonic() - start
    assert elapsed >= 1.0, f'Should throttle but elapsed={elapsed}'
    print('PASS: Rate limiter throttles')

asyncio.run(test())
"

# Tests MUST pass
uv run pytest tests/unit/integrations/gmail/ -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: Models import, rate limiter throttles, tests pass
- FAIL: Import errors or rate limiter doesn't throttle

---

## Iteration 7: GmailClient Implementation

**Goal**: Implement Gmail API client with list, get, and batch operations

**Success Criteria**:
- [ ] `GmailClient` class in `src/rl_emails/integrations/gmail/client.py`
- [ ] `list_messages()` with pagination and filtering (days, labels)
- [ ] `get_message()` fetches full message with headers/body
- [ ] `batch_get_messages()` fetches in batches of 100
- [ ] Rate limiting integrated
- [ ] Tests with mocked HTTP responses
- [ ] `make check` passes

**Tasks**:
1. Create `src/rl_emails/integrations/gmail/client.py`:
   ```python
   """Gmail API client with rate limiting and batching."""
   from typing import AsyncIterator
   import httpx
   from rl_emails.integrations.gmail.models import GmailMessageRef, GmailMessage
   from rl_emails.integrations.gmail.rate_limiter import RateLimiter

   class GmailClient:
       """Gmail API client with rate limiting and batching."""

       BASE_URL = "https://gmail.googleapis.com/gmail/v1"
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
           """List messages with optional filters."""
           ...

       async def get_message(self, message_id: str) -> GmailMessage:
           """Get full message by ID."""
           ...

       async def batch_get_messages(
           self,
           message_ids: list[str],
           batch_size: int = DEFAULT_BATCH_SIZE,
       ) -> AsyncIterator[GmailMessage]:
           """Fetch multiple messages in batches."""
           ...
   ```

2. Create `tests/unit/integrations/gmail/test_client.py`

3. Update `src/rl_emails/integrations/gmail/__init__.py` exports

4. Commit: "Iteration 7: GmailClient implementation"

**Verification Contract**:
```bash
# MUST have client.py
test -f src/rl_emails/integrations/gmail/client.py && echo "PASS: client.py exists"

# MUST import successfully
uv run python -c "from rl_emails.integrations.gmail.client import GmailClient; print('PASS: GmailClient imports')"

# Tests MUST pass
uv run pytest tests/unit/integrations/gmail/test_client.py -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: Client methods work with mocked responses, tests pass
- FAIL: Any method fails or test failures

---

## Iteration 8: Gmail Parser + Error Handling

**Goal**: Create Gmail to EmailData parser and custom exceptions

**Success Criteria**:
- [ ] `parser.py` with `gmail_to_email_data()` function
- [ ] Converts `GmailMessage` to internal `EmailData` format
- [ ] Handles missing fields gracefully (empty body, missing headers)
- [ ] `exceptions.py` with `GmailApiError`, `HistoryIdExpired`
- [ ] Retry logic helpers with exponential backoff
- [ ] Tests for parser and error handling
- [ ] `make check` passes

**Tasks**:
1. Create `src/rl_emails/integrations/gmail/parser.py`:
   ```python
   """Parse Gmail format to internal EmailData format."""
   from uuid import UUID
   from rl_emails.integrations.gmail.models import GmailMessage
   from rl_emails.core.types import EmailData

   def gmail_to_email_data(gmail_msg: GmailMessage, user_id: UUID) -> EmailData:
       """Convert Gmail message to internal EmailData format."""
       return {
           "message_id": gmail_msg.message_id,
           "gmail_id": gmail_msg.id,
           "thread_id": gmail_msg.thread_id,
           "user_id": str(user_id),
           "subject": gmail_msg.subject,
           "from_address": gmail_msg.from_address,
           "to_addresses": gmail_msg.to_addresses,
           "cc_addresses": gmail_msg.cc_addresses,
           "date_sent": gmail_msg.date_sent.isoformat(),
           "body": gmail_msg.body_plain or "",
           "body_html": gmail_msg.body_html,
           "in_reply_to": gmail_msg.in_reply_to,
           "labels": gmail_msg.label_ids,
       }
   ```

2. Create `src/rl_emails/integrations/gmail/exceptions.py`:
   ```python
   """Gmail API exceptions."""

   class GmailApiError(Exception):
       """Base exception for Gmail API errors."""
       def __init__(self, message: str, status_code: int | None = None) -> None:
           self.status_code = status_code
           super().__init__(message)

   class HistoryIdExpired(GmailApiError):
       """Raised when history ID is too old."""
       pass

   class RateLimitExceeded(GmailApiError):
       """Raised when rate limit is exceeded."""
       pass
   ```

3. Update `GmailClient` to use these exceptions and add retry logic

4. Create `tests/unit/integrations/gmail/test_parser.py` and `test_exceptions.py`

5. Commit: "Iteration 8: Gmail parser and error handling"

**Verification Contract**:
```bash
# MUST have parser.py and exceptions.py
test -f src/rl_emails/integrations/gmail/parser.py && echo "PASS: parser.py exists"
test -f src/rl_emails/integrations/gmail/exceptions.py && echo "PASS: exceptions.py exists"

# MUST import successfully
uv run python -c "
from rl_emails.integrations.gmail.parser import gmail_to_email_data
from rl_emails.integrations.gmail.exceptions import GmailApiError, HistoryIdExpired
print('PASS: Parser and exceptions import')
"

# Tests MUST pass
uv run pytest tests/unit/integrations/gmail/test_parser.py tests/unit/integrations/gmail/test_exceptions.py -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: Parser converts correctly, exceptions defined, tests pass
- FAIL: Conversion errors or test failures

---

## Iteration 9: SyncService + Stage 00

**Goal**: Implement sync service and Gmail sync pipeline stage

**Success Criteria**:
- [ ] `SyncService` class with `initial_sync()` method
- [ ] `stage_00_gmail_sync.py` in pipeline stages
- [ ] CLI `sync` command with `--user`, `--days`, `--status` flags
- [ ] Orchestrator updated to support `--source gmail` mode
- [ ] Sync state tracking in database
- [ ] Tests for service and stage
- [ ] `make check` passes

**Tasks**:
1. Create `src/rl_emails/services/sync_service.py`:
   ```python
   """Gmail sync service."""
   from uuid import UUID
   from dataclasses import dataclass
   from typing import Callable

   @dataclass
   class SyncResult:
       success: bool
       emails_synced: int = 0
       history_id: str | None = None
       message: str = ""

   class SyncService:
       """Orchestrates Gmail sync operations."""

       async def initial_sync(
           self,
           user_id: UUID,
           days: int = 30,
           progress_callback: Callable[[int, int], None] | None = None,
       ) -> SyncResult:
           """Perform initial full sync."""
           ...
   ```

2. Create `src/rl_emails/pipeline/stages/stage_00_gmail_sync.py`:
   ```python
   """Gmail sync stage (Stage 0)."""
   from rl_emails.pipeline.stages.base import StageResult
   from rl_emails.core.config import Config

   def run(config: Config, days: int = 30) -> StageResult:
       """Run Gmail sync stage."""
       ...
   ```

3. Add sync CLI command to `src/rl_emails/cli.py`

4. Update `src/rl_emails/pipeline/orchestrator.py` for Gmail mode

5. Update `src/rl_emails/repositories/sync_state.py` with sync tracking methods

6. Create tests for sync service and stage

7. Commit: "Iteration 9: SyncService and stage_00_gmail_sync"

**Verification Contract**:
```bash
# MUST have sync service and stage
test -f src/rl_emails/services/sync_service.py && echo "PASS: sync_service.py exists"
test -f src/rl_emails/pipeline/stages/stage_00_gmail_sync.py && echo "PASS: stage_00_gmail_sync.py exists"

# MUST import successfully
uv run python -c "
from rl_emails.services.sync_service import SyncService, SyncResult
from rl_emails.pipeline.stages import stage_00_gmail_sync
print('PASS: Sync modules import')
"

# CLI help MUST show sync command
uv run python -m rl_emails.cli --help | grep -q "sync" && echo "PASS: sync command exists"

# Tests MUST pass
uv run pytest tests/unit/services/test_sync_service.py tests/unit/pipeline/stages/test_stage_00_gmail_sync.py -v

# Full check MUST pass
make check
```

**Pass/Fail Criteria**:
- PASS: Sync service works, CLI has sync command, tests pass
- FAIL: Any method fails or test failures

---

## Iteration 10: Incremental Sync + Final Testing

**Goal**: Implement incremental sync with History API and complete Phase 2

**Success Criteria**:
- [ ] `history.py` with `get_history_changes()` function
- [ ] `delta_processor.py` for processing MESSAGE_ADDED/DELETED/LABEL changes
- [ ] `incremental_sync()` method in SyncService
- [ ] CLI `sync --incremental` flag
- [ ] Handles expired history ID gracefully (suggests full re-sync)
- [ ] MBOX pipeline still works unchanged
- [ ] All tests pass with 100% coverage on new code
- [ ] `make check` passes
- [ ] Update CLAUDE.md with Phase 2 completion status

**Tasks**:
1. Create `src/rl_emails/integrations/gmail/history.py`:
   ```python
   """Gmail History API wrapper."""
   from dataclasses import dataclass
   from enum import Enum
   from rl_emails.integrations.gmail.client import GmailClient

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
       """Get all history changes since history ID."""
       ...
   ```

2. Create `src/rl_emails/services/delta_processor.py`:
   ```python
   """Process sync deltas from Gmail History API."""
   from dataclasses import dataclass
   from uuid import UUID
   from rl_emails.integrations.gmail.history import HistoryRecord

   @dataclass
   class DeltaResult:
       messages_added: int
       messages_deleted: int
       labels_updated: int

   class DeltaProcessor:
       """Processes sync deltas."""

       async def process_deltas(
           self,
           user_id: UUID,
           records: list[HistoryRecord],
           gmail_client: GmailClient,
       ) -> DeltaResult:
           """Process history records."""
           ...
   ```

3. Add `incremental_sync()` method to SyncService

4. Add `--incremental` flag to CLI sync command

5. Create comprehensive tests

6. Update CLAUDE.md with Phase 2 completion

7. Commit: "Iteration 10: Incremental sync - Phase 2 complete"

**Verification Contract**:
```bash
# MUST have history and delta processor
test -f src/rl_emails/integrations/gmail/history.py && echo "PASS: history.py exists"
test -f src/rl_emails/services/delta_processor.py && echo "PASS: delta_processor.py exists"

# MUST import successfully
uv run python -c "
from rl_emails.integrations.gmail.history import get_history_changes, HistoryType, HistoryRecord
from rl_emails.services.delta_processor import DeltaProcessor, DeltaResult
print('PASS: History and delta modules import')
"

# MBOX pipeline MUST still work
uv run python -m rl_emails.cli --help | grep -q "run" && echo "PASS: run command exists"

# Tests MUST pass
uv run pytest tests/ -v

# Full check MUST pass with 100% coverage
make check

# CLAUDE.md MUST mention Phase 2
grep -q "Phase 2" CLAUDE.md && echo "PASS: CLAUDE.md updated"
```

**Pass/Fail Criteria**:
- PASS: Incremental sync works, MBOX unchanged, tests pass, CLAUDE.md updated
- FAIL: Any method fails or test failures

---

## Commands Reference

```bash
# Auth commands (new in Phase 2)
rl-emails auth connect --email user@gmail.com    # Connect Gmail via OAuth
rl-emails auth status --email user@gmail.com     # Check connection status
rl-emails auth disconnect --email user@gmail.com # Disconnect Gmail

# Sync commands (new in Phase 2)
rl-emails sync --user <uuid>                     # Initial sync (last 30 days)
rl-emails sync --user <uuid> --days 90           # Sync specific date range
rl-emails sync --user <uuid> --incremental       # Incremental sync
rl-emails sync --user <uuid> --status            # Check sync status

# Pipeline with Gmail source (new in Phase 2)
rl-emails --user <uuid> --source gmail           # Run pipeline with Gmail data

# Existing commands (unchanged)
make check                                        # Run all quality checks
make run                                          # Run MBOX pipeline
make status                                       # Check pipeline status
```

---

## Progress Log

| Iteration | Status | Date | Commit | Notes |
|-----------|--------|------|--------|-------|
| 1 | PENDING | - | - | Auth module setup |
| 2 | PENDING | - | - | GoogleOAuth implementation |
| 3 | PENDING | - | - | OAuthToken repository |
| 4 | PENDING | - | - | AuthService implementation |
| 5 | PENDING | - | - | CLI auth commands + config |
| 6 | PENDING | - | - | Gmail integration module |
| 7 | PENDING | - | - | GmailClient implementation |
| 8 | PENDING | - | - | Gmail parser + exceptions |
| 9 | PENDING | - | - | SyncService + stage 00 |
| 10 | PENDING | - | - | Incremental sync + final |

---

## Definition of Done

Each iteration is complete when:
1. All success criteria checkboxes are checked
2. **Verification Contract passes** (all commands exit 0)
3. Changes are committed with descriptive message
4. Progress log is updated with date and commit hash
5. CLAUDE.md is updated with iteration status

**Critical Rule**: An iteration is NOT complete until the Verification Contract passes. No exceptions.

Phase 2 is complete when:
- All 10 iterations pass verification
- `make check` passes (lint + types + tests + 100% coverage)
- User can connect Gmail via OAuth
- Initial sync fetches all emails in date range
- Incremental sync fetches only new changes
- Full pipeline works with Gmail data
- MBOX pipeline still works unchanged

---

## Technical Notes

### Rate Limiting Strategy
Gmail API has per-user quotas:
- 25,000 queries per 100 seconds per user
- Batch requests count as 1 quota unit

Implementation:
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
1. Token Storage: Consider column encryption (future enhancement)
2. Scope Minimization: Only `gmail.readonly` requested
3. Token Refresh: Automatic refresh before expiry
4. Revocation: Clean disconnect removes all tokens

---

## Dependencies (to add to pyproject.toml)

```toml
[project]
dependencies = [
    # ... existing ...

    # Gmail API (new)
    "google-auth>=2.0.0",
    "google-auth-oauthlib>=1.0.0",
    "httpx>=0.25.0",
]
```
