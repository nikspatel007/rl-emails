"""Progressive sync service for streaming email ingestion and processing.

This service implements a multi-phase sync strategy:
1. Phase 1 (Quick): Sync last 7 days, process with full pipeline
2. Phase 2 (Background): Sync days 8-30, update incrementally
3. Phase 3 (Deep): Sync 30+ days for complete history

Each phase processes emails through the full pipeline including:
- Feature computation
- Embeddings generation
- LLM classification
- Clustering
- Priority ranking
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rl_emails.core.config import Config
    from rl_emails.core.types import EmailData
    from rl_emails.integrations.gmail.client import GmailClient
    from rl_emails.repositories.sync_state import SyncStateRepository
    from rl_emails.services.batch_processor import BatchProcessor


class SyncPhase(Enum):
    """Phases of progressive sync."""

    QUICK = "quick"  # 7 days - fast initial results
    STANDARD = "standard"  # 30 days - full context
    DEEP = "deep"  # 90+ days - complete history


@dataclass
class PhaseConfig:
    """Configuration for a sync phase."""

    phase: SyncPhase
    days_start: int  # Start of range (0 = today)
    days_end: int  # End of range
    batch_size: int = 100
    run_embeddings: bool = True
    run_llm: bool = True
    llm_limit: int | None = None  # Limit LLM calls per phase


@dataclass
class SyncProgress:
    """Progress update from sync operation."""

    phase: SyncPhase
    phase_complete: bool
    emails_fetched: int
    emails_processed: int
    total_in_phase: int | None
    features_computed: int
    embeddings_generated: int
    llm_classified: int
    clusters_updated: bool
    priority_computed: bool
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ProgressiveSyncResult:
    """Final result of progressive sync."""

    success: bool
    phases_completed: list[SyncPhase]
    total_emails_synced: int
    total_emails_processed: int
    duration_seconds: float
    error: str | None = None


# Default phase configurations
DEFAULT_PHASES = [
    PhaseConfig(
        phase=SyncPhase.QUICK,
        days_start=0,
        days_end=7,
        batch_size=100,
        run_embeddings=True,
        run_llm=True,
        llm_limit=50,  # Limit LLM for speed
    ),
    PhaseConfig(
        phase=SyncPhase.STANDARD,
        days_start=7,
        days_end=30,
        batch_size=200,
        run_embeddings=True,
        run_llm=True,
        llm_limit=100,
    ),
    PhaseConfig(
        phase=SyncPhase.DEEP,
        days_start=30,
        days_end=90,
        batch_size=500,
        run_embeddings=False,  # Skip for speed
        run_llm=False,  # Skip for speed
    ),
]


class ProgressiveSyncService:
    """Service for progressive email sync with streaming processing.

    This service orchestrates the multi-phase sync:
    1. Fetches emails in date-range batches (newest first)
    2. Processes each batch through the full pipeline
    3. Yields progress updates for UI feedback
    4. Continues in background for older emails

    Example:
        async for progress in service.sync_progressive(user_id, config):
            print(f"Phase {progress.phase}: {progress.emails_processed} processed")
            if progress.phase == SyncPhase.QUICK and progress.phase_complete:
                # User can start using the app
                show_results()
    """

    def __init__(
        self,
        gmail_client: GmailClient,
        sync_repo: SyncStateRepository,
        session: AsyncSession,
    ) -> None:
        """Initialize progressive sync service.

        Args:
            gmail_client: Gmail API client.
            sync_repo: Repository for sync state.
            session: Database session for pipeline operations.
        """
        self.gmail_client = gmail_client
        self.sync_repo = sync_repo
        self.session = session

    async def sync_progressive(
        self,
        user_id: UUID,
        config: Config,
        phases: list[PhaseConfig] | None = None,
        on_batch_complete: Callable[[list[EmailData]], None] | None = None,
    ) -> AsyncGenerator[SyncProgress, None]:
        """Execute progressive sync with streaming updates.

        Syncs emails in phases, processing each batch through the full
        pipeline before moving to older emails.

        Args:
            user_id: User to sync for.
            config: Pipeline configuration.
            phases: Custom phase configs (default: 7d, 30d, 90d).
            on_batch_complete: Callback when a batch is processed.

        Yields:
            SyncProgress updates as sync progresses.
        """
        if phases is None:
            phases = DEFAULT_PHASES

        # Import here to avoid circular imports
        from rl_emails.services.batch_processor import BatchProcessor

        processor = BatchProcessor(session=self.session, config=config)

        for phase_config in phases:
            async for progress in self._sync_phase(
                user_id=user_id,
                phase_config=phase_config,
                processor=processor,
                on_batch_complete=on_batch_complete,
            ):
                yield progress

                # If phase failed, stop
                if progress.error:
                    return

    async def _sync_phase(
        self,
        user_id: UUID,
        phase_config: PhaseConfig,
        processor: BatchProcessor,
        on_batch_complete: Callable[[list[EmailData]], None] | None = None,
    ) -> AsyncGenerator[SyncProgress, None]:
        """Execute a single sync phase.

        Args:
            user_id: User to sync for.
            phase_config: Configuration for this phase.
            processor: Batch processor for pipeline.
            on_batch_complete: Callback for completed batches.

        Yields:
            SyncProgress updates.
        """
        from rl_emails.integrations.gmail.client import GmailApiError
        from rl_emails.integrations.gmail.parser import gmail_to_email_data, parse_raw_message

        phase = phase_config.phase
        progress = SyncProgress(
            phase=phase,
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

        try:
            # Build query for this phase's date range
            if phase_config.days_start == 0:
                query = f"newer_than:{phase_config.days_end}d"
            else:
                query = f"older_than:{phase_config.days_start}d newer_than:{phase_config.days_end}d"

            # List messages for this phase
            message_refs = []
            async for ref in self.gmail_client.list_all_messages(query=query):
                message_refs.append(ref)

            progress.total_in_phase = len(message_refs)
            yield progress

            if not message_refs:
                progress.phase_complete = True
                yield progress
                return

            # Process in batches
            batch_size = phase_config.batch_size
            for i in range(0, len(message_refs), batch_size):
                batch_refs = message_refs[i : i + batch_size]

                # Fetch batch
                results = await self.gmail_client.batch_get_messages(batch_refs)

                # Parse emails
                emails: list[EmailData] = []
                for result in results:
                    if isinstance(result, GmailApiError):
                        continue
                    try:
                        gmail_msg = parse_raw_message(result)
                        email_data = gmail_to_email_data(gmail_msg)
                        emails.append(email_data)
                    except Exception:
                        continue

                progress.emails_fetched += len(emails)
                yield progress

                # Process batch through pipeline
                batch_result = await processor.process_batch(
                    emails=emails,
                    run_embeddings=phase_config.run_embeddings,
                    run_llm=phase_config.run_llm,
                    llm_limit=phase_config.llm_limit,
                )

                progress.emails_processed += batch_result.emails_stored
                progress.features_computed += batch_result.features_computed
                progress.embeddings_generated += batch_result.embeddings_generated
                progress.llm_classified += batch_result.llm_classified

                yield progress

                if on_batch_complete:
                    on_batch_complete(emails)

            # Run clustering and priority for this phase
            cluster_result = await processor.update_clusters()
            progress.clusters_updated = cluster_result.success

            priority_result = await processor.compute_priority()
            progress.priority_computed = priority_result.success

            progress.phase_complete = True
            yield progress

        except Exception as e:
            progress.error = str(e)
            yield progress

    async def sync_quick(
        self,
        user_id: UUID,
        config: Config,
    ) -> SyncProgress:
        """Execute just the quick phase (7 days) synchronously.

        Convenience method for getting initial results quickly.

        Args:
            user_id: User to sync for.
            config: Pipeline configuration.

        Returns:
            Final progress of quick phase.
        """
        quick_phase = PhaseConfig(
            phase=SyncPhase.QUICK,
            days_start=0,
            days_end=7,
            batch_size=100,
            run_embeddings=True,
            run_llm=True,
            llm_limit=50,
        )

        final_progress = None
        async for progress in self.sync_progressive(
            user_id=user_id,
            config=config,
            phases=[quick_phase],
        ):
            final_progress = progress

        if final_progress is None:
            return SyncProgress(
                phase=SyncPhase.QUICK,
                phase_complete=False,
                emails_fetched=0,
                emails_processed=0,
                total_in_phase=0,
                features_computed=0,
                embeddings_generated=0,
                llm_classified=0,
                clusters_updated=False,
                priority_computed=False,
                error="No progress returned",
            )

        return final_progress

    async def continue_background_sync(
        self,
        user_id: UUID,
        config: Config,
    ) -> AsyncGenerator[SyncProgress, None]:
        """Continue syncing in background after quick phase.

        Call this after sync_quick() to continue with standard and deep phases.

        Args:
            user_id: User to sync for.
            config: Pipeline configuration.

        Yields:
            Progress updates for background phases.
        """
        background_phases = [
            PhaseConfig(
                phase=SyncPhase.STANDARD,
                days_start=7,
                days_end=30,
                batch_size=200,
                run_embeddings=True,
                run_llm=True,
                llm_limit=100,
            ),
            PhaseConfig(
                phase=SyncPhase.DEEP,
                days_start=30,
                days_end=90,
                batch_size=500,
                run_embeddings=False,
                run_llm=False,
            ),
        ]

        async for progress in self.sync_progressive(
            user_id=user_id,
            config=config,
            phases=background_phases,
        ):
            yield progress
