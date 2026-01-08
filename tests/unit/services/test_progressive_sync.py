"""Tests for ProgressiveSyncService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from rl_emails.integrations.gmail.client import GmailApiError
from rl_emails.integrations.gmail.models import GmailMessageRef
from rl_emails.services.progressive_sync import (
    DEFAULT_PHASES,
    PhaseConfig,
    ProgressiveSyncResult,
    ProgressiveSyncService,
    SyncPhase,
    SyncProgress,
)


class TestSyncPhase:
    """Tests for SyncPhase enum."""

    def test_quick_phase_value(self) -> None:
        """Test QUICK phase has correct value."""
        assert SyncPhase.QUICK.value == "quick"

    def test_standard_phase_value(self) -> None:
        """Test STANDARD phase has correct value."""
        assert SyncPhase.STANDARD.value == "standard"

    def test_deep_phase_value(self) -> None:
        """Test DEEP phase has correct value."""
        assert SyncPhase.DEEP.value == "deep"


class TestPhaseConfig:
    """Tests for PhaseConfig dataclass."""

    def test_create_phase_config(self) -> None:
        """Test creating a phase config."""
        config = PhaseConfig(
            phase=SyncPhase.QUICK,
            days_start=0,
            days_end=7,
            batch_size=100,
            run_embeddings=True,
            run_llm=True,
            llm_limit=50,
        )

        assert config.phase == SyncPhase.QUICK
        assert config.days_start == 0
        assert config.days_end == 7
        assert config.batch_size == 100
        assert config.run_embeddings is True
        assert config.run_llm is True
        assert config.llm_limit == 50

    def test_default_values(self) -> None:
        """Test phase config default values."""
        config = PhaseConfig(
            phase=SyncPhase.STANDARD,
            days_start=7,
            days_end=30,
        )

        assert config.batch_size == 100
        assert config.run_embeddings is True
        assert config.run_llm is True
        assert config.llm_limit is None


class TestSyncProgress:
    """Tests for SyncProgress dataclass."""

    def test_create_progress(self) -> None:
        """Test creating a sync progress."""
        progress = SyncProgress(
            phase=SyncPhase.QUICK,
            phase_complete=False,
            emails_fetched=50,
            emails_processed=45,
            total_in_phase=100,
            features_computed=45,
            embeddings_generated=40,
            llm_classified=30,
            clusters_updated=False,
            priority_computed=False,
        )

        assert progress.phase == SyncPhase.QUICK
        assert progress.phase_complete is False
        assert progress.emails_fetched == 50
        assert progress.emails_processed == 45
        assert progress.error is None

    def test_error_progress(self) -> None:
        """Test progress with error."""
        progress = SyncProgress(
            phase=SyncPhase.STANDARD,
            phase_complete=False,
            emails_fetched=0,
            emails_processed=0,
            total_in_phase=None,
            features_computed=0,
            embeddings_generated=0,
            llm_classified=0,
            clusters_updated=False,
            priority_computed=False,
            error="API rate limit exceeded",
        )

        assert progress.error == "API rate limit exceeded"

    def test_started_at_default(self) -> None:
        """Test started_at has default value."""
        before = datetime.now(UTC)
        progress = SyncProgress(
            phase=SyncPhase.QUICK,
            phase_complete=False,
            emails_fetched=0,
            emails_processed=0,
            total_in_phase=None,
            features_computed=0,
            embeddings_generated=0,
            llm_classified=0,
            clusters_updated=False,
            priority_computed=False,
        )
        after = datetime.now(UTC)

        assert before <= progress.started_at <= after


class TestProgressiveSyncResult:
    """Tests for ProgressiveSyncResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful result."""
        result = ProgressiveSyncResult(
            success=True,
            phases_completed=[SyncPhase.QUICK, SyncPhase.STANDARD],
            total_emails_synced=500,
            total_emails_processed=480,
            duration_seconds=120.5,
        )

        assert result.success is True
        assert len(result.phases_completed) == 2
        assert result.total_emails_synced == 500
        assert result.error is None

    def test_create_error_result(self) -> None:
        """Test creating an error result."""
        result = ProgressiveSyncResult(
            success=False,
            phases_completed=[SyncPhase.QUICK],
            total_emails_synced=100,
            total_emails_processed=100,
            duration_seconds=30.0,
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"


class TestDefaultPhases:
    """Tests for DEFAULT_PHASES configuration."""

    def test_default_phases_count(self) -> None:
        """Test default phases has 3 phases."""
        assert len(DEFAULT_PHASES) == 3

    def test_quick_phase_config(self) -> None:
        """Test quick phase configuration."""
        quick = DEFAULT_PHASES[0]
        assert quick.phase == SyncPhase.QUICK
        assert quick.days_start == 0
        assert quick.days_end == 7
        assert quick.run_embeddings is True
        assert quick.run_llm is True

    def test_standard_phase_config(self) -> None:
        """Test standard phase configuration."""
        standard = DEFAULT_PHASES[1]
        assert standard.phase == SyncPhase.STANDARD
        assert standard.days_start == 7
        assert standard.days_end == 30

    def test_deep_phase_config(self) -> None:
        """Test deep phase configuration."""
        deep = DEFAULT_PHASES[2]
        assert deep.phase == SyncPhase.DEEP
        assert deep.days_start == 30
        assert deep.days_end == 90
        assert deep.run_embeddings is False
        assert deep.run_llm is False


class TestProgressiveSyncServiceInit:
    """Tests for ProgressiveSyncService initialization."""

    def test_init_stores_dependencies(self) -> None:
        """Test that init stores dependencies."""
        gmail_client = MagicMock()
        sync_repo = MagicMock()
        session = MagicMock()

        service = ProgressiveSyncService(
            gmail_client=gmail_client,
            sync_repo=sync_repo,
            session=session,
        )

        assert service.gmail_client == gmail_client
        assert service.sync_repo == sync_repo
        assert service.session == session


class TestProgressiveSyncServiceSyncProgressive:
    """Tests for ProgressiveSyncService.sync_progressive method."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        client = MagicMock()
        client.batch_get_messages = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        return MagicMock()

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        config = MagicMock()
        config.openai_api_key = "test-key"
        config.anthropic_api_key = None
        return config

    @pytest.mark.asyncio
    async def test_sync_progressive_empty_messages(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test sync_progressive with no messages."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        # Only run quick phase
        phases = [
            PhaseConfig(
                phase=SyncPhase.QUICK,
                days_start=0,
                days_end=7,
            )
        ]

        progress_updates = []
        async for progress in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
        ):
            progress_updates.append(progress)

        # Should have progress updates: initial + phase complete
        assert len(progress_updates) >= 1
        final = progress_updates[-1]
        assert final.phase_complete is True
        assert final.total_in_phase == 0

    @pytest.mark.asyncio
    async def test_sync_progressive_with_messages(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test sync_progressive with messages."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")
            yield GmailMessageRef(id="msg2", thread_id="t2")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1@ex.com>"}]},
            },
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "101",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2@ex.com>"}]},
            },
        ]

        # Mock session execute for email storage
        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases = [
            PhaseConfig(
                phase=SyncPhase.QUICK,
                days_start=0,
                days_end=7,
                run_embeddings=False,
                run_llm=False,
            )
        ]

        progress_updates = []
        async for progress in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
        ):
            progress_updates.append(progress)

        # Should have multiple progress updates
        assert len(progress_updates) >= 2
        final = progress_updates[-1]
        assert final.phase_complete is True
        assert final.total_in_phase == 2

    @pytest.mark.asyncio
    async def test_sync_progressive_handles_api_error(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test sync_progressive handles API errors."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise GmailApiError("Rate limit exceeded", status_code=429)
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases = [
            PhaseConfig(
                phase=SyncPhase.QUICK,
                days_start=0,
                days_end=7,
            )
        ]

        progress_updates = []
        async for progress in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
        ):
            progress_updates.append(progress)

        # Should have error in final progress
        final = progress_updates[-1]
        assert final.error is not None
        assert "Rate limit" in final.error

    @pytest.mark.asyncio
    async def test_sync_progressive_with_batch_callback(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test sync_progressive calls batch callback."""
        user_id = uuid4()
        callback_calls: list[int] = []

        def on_batch(emails: list) -> None:  # type: ignore[type-arg]
            callback_calls.append(len(emails))

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1@ex.com>"}]},
            },
        ]

        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases = [
            PhaseConfig(
                phase=SyncPhase.QUICK,
                days_start=0,
                days_end=7,
                run_embeddings=False,
                run_llm=False,
            )
        ]

        async for _ in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
            on_batch_complete=on_batch,
        ):
            pass

        # Callback should have been called
        assert len(callback_calls) == 1
        assert callback_calls[0] == 1

    @pytest.mark.asyncio
    async def test_sync_progressive_uses_default_phases(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test sync_progressive uses default phases when none provided."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases_seen: set[SyncPhase] = set()
        async for progress in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=None,  # Use default
        ):
            phases_seen.add(progress.phase)

        # Should have seen all three default phases
        assert SyncPhase.QUICK in phases_seen
        assert SyncPhase.STANDARD in phases_seen
        assert SyncPhase.DEEP in phases_seen

    @pytest.mark.asyncio
    async def test_sync_progressive_stops_on_error(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test sync_progressive stops when phase errors."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("Test error")
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases = [
            PhaseConfig(phase=SyncPhase.QUICK, days_start=0, days_end=7),
            PhaseConfig(phase=SyncPhase.STANDARD, days_start=7, days_end=30),
        ]

        phases_seen: set[SyncPhase] = set()
        async for progress in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
        ):
            phases_seen.add(progress.phase)

        # Should only have seen QUICK phase (stopped due to error)
        assert phases_seen == {SyncPhase.QUICK}


class TestProgressiveSyncServiceSyncPhase:
    """Tests for ProgressiveSyncService._sync_phase method."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        client = MagicMock()
        client.batch_get_messages = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        return MagicMock()

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        config = MagicMock()
        config.openai_api_key = "test-key"
        config.anthropic_api_key = None
        return config

    @pytest.mark.asyncio
    async def test_sync_phase_builds_correct_query_for_quick(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test _sync_phase builds correct query for quick phase."""
        user_id = uuid4()
        captured_query: list[str] = []

        async def mock_list(query: str):  # type: ignore[no-untyped-def]
            captured_query.append(query)
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases = [
            PhaseConfig(
                phase=SyncPhase.QUICK,
                days_start=0,
                days_end=7,
            )
        ]

        async for _ in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
        ):
            pass

        # Query for days_start=0 should be "newer_than:7d"
        assert len(captured_query) == 1
        assert captured_query[0] == "newer_than:7d"

    @pytest.mark.asyncio
    async def test_sync_phase_builds_correct_query_for_standard(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test _sync_phase builds correct query for standard phase."""
        user_id = uuid4()
        captured_query: list[str] = []

        async def mock_list(query: str):  # type: ignore[no-untyped-def]
            captured_query.append(query)
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases = [
            PhaseConfig(
                phase=SyncPhase.STANDARD,
                days_start=7,
                days_end=30,
            )
        ]

        async for _ in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
        ):
            pass

        # Query should be "older_than:7d newer_than:30d"
        assert len(captured_query) == 1
        assert "older_than:7d" in captured_query[0]
        assert "newer_than:30d" in captured_query[0]

    @pytest.mark.asyncio
    async def test_sync_phase_handles_failed_fetch(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test _sync_phase handles failed message fetches."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")
            yield GmailMessageRef(id="msg2", thread_id="t2")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            GmailApiError("Not found", status_code=404),  # Failed
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2@ex.com>"}]},
            },
        ]

        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases = [
            PhaseConfig(
                phase=SyncPhase.QUICK,
                days_start=0,
                days_end=7,
                run_embeddings=False,
                run_llm=False,
            )
        ]

        progress_updates = []
        async for progress in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
        ):
            progress_updates.append(progress)

        # Should complete successfully despite one failed fetch
        final = progress_updates[-1]
        assert final.phase_complete is True
        assert final.emails_fetched == 1  # Only one succeeded

    @pytest.mark.asyncio
    async def test_sync_phase_handles_unparseable_messages(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test _sync_phase handles unparseable messages."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")
            yield GmailMessageRef(id="msg2", thread_id="t2")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            {},  # Missing id, will fail parse
            {
                "id": "msg2",
                "threadId": "t2",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m2@ex.com>"}]},
            },
        ]

        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases = [
            PhaseConfig(
                phase=SyncPhase.QUICK,
                days_start=0,
                days_end=7,
                run_embeddings=False,
                run_llm=False,
            )
        ]

        progress_updates = []
        async for progress in service.sync_progressive(
            user_id=user_id,
            config=mock_config,
            phases=phases,
        ):
            progress_updates.append(progress)

        # Should complete successfully
        final = progress_updates[-1]
        assert final.phase_complete is True


class TestProgressiveSyncServiceSyncQuick:
    """Tests for ProgressiveSyncService.sync_quick method."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        client = MagicMock()
        client.batch_get_messages = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        return MagicMock()

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        config = MagicMock()
        config.openai_api_key = "test-key"
        config.anthropic_api_key = None
        return config

    @pytest.mark.asyncio
    async def test_sync_quick_returns_final_progress(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test sync_quick returns final progress."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield GmailMessageRef(id="msg1", thread_id="t1")

        mock_gmail_client.list_all_messages = mock_list
        mock_gmail_client.batch_get_messages.return_value = [
            {
                "id": "msg1",
                "threadId": "t1",
                "historyId": "100",
                "payload": {"headers": [{"name": "Message-ID", "value": "<m1@ex.com>"}]},
            },
        ]

        raw_result = MagicMock()
        raw_result.fetchone.return_value = (1,)
        mock_session.execute.return_value = raw_result

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        result = await service.sync_quick(user_id=user_id, config=mock_config)

        assert result.phase == SyncPhase.QUICK
        assert result.phase_complete is True

    @pytest.mark.asyncio
    async def test_sync_quick_handles_empty_generator(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test sync_quick handles case where no progress is returned."""
        user_id = uuid4()

        # Create a service that yields nothing (edge case)
        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        # Patch sync_progressive to return no items
        async def mock_sync(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # type: ignore[misc]

        service.sync_progressive = mock_sync  # type: ignore[method-assign]

        result = await service.sync_quick(user_id=user_id, config=mock_config)

        # Should return error progress
        assert result.phase == SyncPhase.QUICK
        assert result.phase_complete is False
        assert result.error == "No progress returned"


class TestProgressiveSyncServiceContinueBackgroundSync:
    """Tests for ProgressiveSyncService.continue_background_sync method."""

    @pytest.fixture
    def mock_gmail_client(self) -> MagicMock:
        """Create mock Gmail client."""
        client = MagicMock()
        client.batch_get_messages = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_repo(self) -> MagicMock:
        """Create mock sync state repository."""
        return MagicMock()

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Create mock async session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create mock config."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_continue_background_sync_runs_standard_and_deep(
        self,
        mock_gmail_client: MagicMock,
        mock_sync_repo: MagicMock,
        mock_session: AsyncMock,
        mock_config: MagicMock,
    ) -> None:
        """Test continue_background_sync runs standard and deep phases."""
        user_id = uuid4()

        async def mock_list(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # type: ignore[misc]

        mock_gmail_client.list_all_messages = mock_list

        service = ProgressiveSyncService(
            gmail_client=mock_gmail_client,
            sync_repo=mock_sync_repo,
            session=mock_session,
        )

        phases_seen: set[SyncPhase] = set()
        async for progress in service.continue_background_sync(
            user_id=user_id,
            config=mock_config,
        ):
            phases_seen.add(progress.phase)

        # Should only run standard and deep, not quick
        assert SyncPhase.QUICK not in phases_seen
        assert SyncPhase.STANDARD in phases_seen
        assert SyncPhase.DEEP in phases_seen
