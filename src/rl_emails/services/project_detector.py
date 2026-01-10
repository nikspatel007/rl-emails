"""Project detection service for identifying project clusters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.cluster_metadata import ClusterMetadata
from rl_emails.repositories.cluster_metadata import ClusterMetadataRepository

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class ProjectDetectionConfig:
    """Configuration for project detection."""

    # Minimum cluster size to consider as project
    min_size: int = 5

    # Minimum engagement rate (pct_replied / 100) for projects
    min_engagement: float = 0.2

    # Days without activity before project is marked stale
    stale_days: int = 14

    # Days without activity before project is marked completed
    completed_days: int = 30

    # Minimum coherence score (if available) for projects
    min_coherence: float = 0.5

    # Minimum participant count for projects (collaborative indicator)
    min_participants: int = 2


@dataclass
class ProjectDetectionResult:
    """Result of project detection."""

    cluster_id: int
    is_project: bool
    status: str | None  # active, stale, completed
    confidence: float
    reason: str


@dataclass
class ProjectDetectionSummary:
    """Summary of project detection run."""

    clusters_analyzed: int
    projects_detected: int
    active_projects: int
    stale_projects: int
    completed_projects: int
    detection_results: list[ProjectDetectionResult]


class ProjectDetectorService:
    """Service for detecting and managing project clusters."""

    def __init__(
        self,
        session: AsyncSession,
        config: ProjectDetectionConfig | None = None,
    ) -> None:
        """Initialize the project detector service.

        Args:
            session: Async database session.
            config: Project detection configuration.
        """
        self.session = session
        self.repository = ClusterMetadataRepository(session)
        self.config = config or ProjectDetectionConfig()

    async def _get_cluster_activity(
        self,
        cluster_id: int,
        user_id: UUID | None = None,
    ) -> dict[str, datetime | int | None]:
        """Get activity information for a cluster.

        Args:
            cluster_id: Content cluster ID.
            user_id: Optional user ID for multi-tenant.

        Returns:
            Dictionary with last_activity, email_count, participant_count.
        """
        query = text(
            """
            SELECT
                MAX(e.date_parsed) as last_activity,
                COUNT(DISTINCT e.id) as email_count,
                COUNT(DISTINCT e.from_email) as participant_count
            FROM emails e
            JOIN email_clusters ec ON ec.email_id = e.id
            WHERE ec.content_cluster_id = :cluster_id
            """
            + (" AND e.user_id = :user_id" if user_id else "")
        )

        params: dict[str, int | str] = {"cluster_id": cluster_id}
        if user_id:
            params["user_id"] = str(user_id)

        result = await self.session.execute(query, params)
        row = result.fetchone()

        if not row:
            return {
                "last_activity": None,
                "email_count": 0,
                "participant_count": 0,
            }

        return {
            "last_activity": row[0],
            "email_count": row[1] or 0,
            "participant_count": row[2] or 0,
        }

    def _calculate_project_score(
        self,
        metadata: ClusterMetadata,
        activity: dict[str, datetime | int | None],
    ) -> tuple[float, str]:
        """Calculate a project likelihood score.

        Args:
            metadata: Cluster metadata.
            activity: Activity information.

        Returns:
            Tuple of (score, reason).
        """
        score = 0.0
        reasons = []

        # Size factor (more emails = more likely a project)
        size = metadata.size or 0
        if size >= self.config.min_size:
            size_score = min(size / 50, 1.0) * 0.25
            score += size_score
            reasons.append(f"size={size}")

        # Engagement factor (higher reply rate = more active project)
        engagement = (metadata.pct_replied or 0) / 100.0
        if engagement >= self.config.min_engagement:
            engagement_score = min(engagement / 0.5, 1.0) * 0.3
            score += engagement_score
            reasons.append(f"engagement={engagement:.1%}")

        # Participant factor (more participants = collaborative project)
        participants = activity.get("participant_count") or 0
        if isinstance(participants, int) and participants >= self.config.min_participants:
            participant_score = min(participants / 10, 1.0) * 0.2
            score += participant_score
            reasons.append(f"participants={participants}")

        # Recency factor (recent activity = active project)
        last_activity = activity.get("last_activity")
        if isinstance(last_activity, datetime):
            days_since = (datetime.now(UTC) - last_activity.replace(tzinfo=UTC)).days
            if days_since < self.config.stale_days:
                recency_score = max(0, 1 - (days_since / self.config.stale_days)) * 0.25
                score += recency_score
                reasons.append(f"days_since={days_since}")

        # Coherence factor (if available)
        coherence = metadata.coherence_score
        if coherence is not None and coherence >= self.config.min_coherence:
            coherence_score = coherence * 0.1
            score += coherence_score
            reasons.append(f"coherence={coherence:.2f}")

        return score, ", ".join(reasons)

    def _determine_project_status(
        self,
        last_activity: datetime | None,
    ) -> str:
        """Determine project status based on last activity.

        Args:
            last_activity: Last activity timestamp.

        Returns:
            Project status (active, stale, completed).
        """
        if last_activity is None:
            return "stale"

        # Ensure timezone-aware comparison
        if last_activity.tzinfo is None:
            last_activity = last_activity.replace(tzinfo=UTC)

        now = datetime.now(UTC)
        days_since = (now - last_activity).days

        if days_since > self.config.completed_days:
            return "completed"
        if days_since > self.config.stale_days:
            return "stale"
        return "active"

    async def detect_projects(
        self,
        user_id: UUID | None = None,
        *,
        min_score: float = 0.4,
        limit: int = 100,
    ) -> ProjectDetectionSummary:
        """Detect and flag project clusters.

        Args:
            user_id: Optional user ID for multi-tenant.
            min_score: Minimum score to consider as project.
            limit: Maximum clusters to analyze.

        Returns:
            ProjectDetectionSummary with results.
        """
        # Get content clusters (projects are detected from content clusters)
        clusters, total = await self.repository.list_by_dimension(
            dimension="content",
            user_id=user_id,
            limit=limit,
        )

        results: list[ProjectDetectionResult] = []
        projects_detected = 0
        active_count = 0
        stale_count = 0
        completed_count = 0

        for metadata in clusters:
            # Get activity info
            activity = await self._get_cluster_activity(
                cluster_id=metadata.cluster_id,
                user_id=user_id,
            )

            # Calculate project score
            score, reason = self._calculate_project_score(metadata, activity)

            # Determine if it's a project
            is_project = score >= min_score
            status = None

            if is_project:
                last_activity = activity.get("last_activity")
                if isinstance(last_activity, datetime):
                    status = self._determine_project_status(last_activity)
                else:
                    status = "stale"

                projects_detected += 1
                if status == "active":
                    active_count += 1
                elif status == "stale":
                    stale_count += 1
                else:
                    completed_count += 1

                # Update cluster metadata
                await self.repository.mark_as_project(
                    dimension="content",
                    cluster_id=metadata.cluster_id,
                    status=status,
                    user_id=user_id,
                )

                # Update last_activity_at
                last_activity = activity.get("last_activity")
                if isinstance(last_activity, datetime):
                    metadata.last_activity_at = last_activity
                    await self.session.commit()

            results.append(
                ProjectDetectionResult(
                    cluster_id=metadata.cluster_id,
                    is_project=is_project,
                    status=status,
                    confidence=score,
                    reason=reason,
                )
            )

        return ProjectDetectionSummary(
            clusters_analyzed=len(clusters),
            projects_detected=projects_detected,
            active_projects=active_count,
            stale_projects=stale_count,
            completed_projects=completed_count,
            detection_results=results,
        )

    async def update_project_statuses(
        self,
        user_id: UUID | None = None,
    ) -> dict[str, int]:
        """Update status of existing projects based on activity.

        Args:
            user_id: Optional user ID for multi-tenant.

        Returns:
            Dictionary with count of status changes.
        """
        # Get all projects
        projects, _ = await self.repository.list_projects(
            user_id=user_id,
            limit=500,
        )

        status_changes = {
            "to_active": 0,
            "to_stale": 0,
            "to_completed": 0,
            "unchanged": 0,
        }

        for project in projects:
            # Get current activity
            activity = await self._get_cluster_activity(
                cluster_id=project.cluster_id,
                user_id=user_id,
            )

            last_activity = activity.get("last_activity")
            if isinstance(last_activity, datetime):
                new_status = self._determine_project_status(last_activity)
            else:
                new_status = "stale"

            if new_status != project.project_status:
                await self.repository.mark_as_project(
                    dimension="content",
                    cluster_id=project.cluster_id,
                    status=new_status,
                    user_id=user_id,
                )
                status_changes[f"to_{new_status}"] += 1
            else:
                status_changes["unchanged"] += 1

        return status_changes

    async def get_project_summary(
        self,
        user_id: UUID | None = None,
    ) -> dict[str, object]:
        """Get summary of detected projects.

        Args:
            user_id: Optional user ID for multi-tenant.

        Returns:
            Summary with counts and top projects.
        """
        # Get stats
        stats = await self.repository.get_stats("content", user_id)

        # Get active projects
        active_projects, _ = await self.repository.list_projects(
            user_id=user_id,
            status="active",
            limit=10,
        )

        top_active: list[dict[str, object]] = [
            {
                "cluster_id": p.cluster_id,
                "label": p.auto_label,
                "size": p.size,
                "engagement": p.pct_replied,
            }
            for p in active_projects
        ]

        return {
            "total_projects": stats["project_clusters"],
            "active_projects": stats["active_projects"],
            "total_content_clusters": stats["total_clusters"],
            "top_active_projects": top_active,
        }

    async def enrich_cluster_metadata(
        self,
        user_id: UUID | None = None,
    ) -> int:
        """Enrich cluster metadata with computed fields.

        Updates participant_count and last_activity_at for all clusters.

        Args:
            user_id: Optional user ID for multi-tenant.

        Returns:
            Number of clusters updated.
        """
        # Get all content clusters
        clusters, _ = await self.repository.list_by_dimension(
            dimension="content",
            user_id=user_id,
            limit=500,
        )

        updated = 0
        for metadata in clusters:
            activity = await self._get_cluster_activity(
                cluster_id=metadata.cluster_id,
                user_id=user_id,
            )

            # Update fields
            participant_count = activity.get("participant_count")
            last_activity = activity.get("last_activity")

            if isinstance(participant_count, int):
                metadata.participant_count = participant_count
            if isinstance(last_activity, datetime):
                metadata.last_activity_at = last_activity

            updated += 1

        await self.session.commit()
        return updated

    async def _get_temporal_activity(
        self,
        cluster_id: int,
        user_id: UUID | None = None,
    ) -> dict[str, datetime | int | float | None]:
        """Get temporal activity information for a cluster.

        Args:
            cluster_id: Content cluster ID.
            user_id: Optional user ID for multi-tenant.

        Returns:
            Dictionary with temporal metrics.
        """
        query = text(
            """
            WITH cluster_emails AS (
                SELECT e.date_parsed
                FROM emails e
                JOIN email_clusters ec ON ec.email_id = e.id
                WHERE ec.content_cluster_id = :cluster_id
            """
            + (" AND e.user_id = :user_id" if user_id else "")
            + """
            )
            SELECT
                MIN(date_parsed) as first_activity,
                MAX(date_parsed) as last_activity,
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE date_parsed > NOW() - INTERVAL '7 days') as emails_7d,
                COUNT(*) FILTER (WHERE date_parsed > NOW() - INTERVAL '30 days') as emails_30d,
                COUNT(*) FILTER (WHERE date_parsed > NOW() - INTERVAL '14 days'
                                  AND date_parsed <= NOW() - INTERVAL '7 days') as emails_prev_7d
            FROM cluster_emails
            """
        )

        params: dict[str, int | str] = {"cluster_id": cluster_id}
        if user_id:
            params["user_id"] = str(user_id)

        result = await self.session.execute(query, params)
        row = result.fetchone()

        if not row:
            return {
                "first_activity": None,
                "last_activity": None,
                "total_count": 0,
                "emails_7d": 0,
                "emails_30d": 0,
                "emails_prev_7d": 0,
            }

        return {
            "first_activity": row[0],
            "last_activity": row[1],
            "total_count": row[2] or 0,
            "emails_7d": row[3] or 0,
            "emails_30d": row[4] or 0,
            "emails_prev_7d": row[5] or 0,
        }

    def _calculate_temporal_strength(
        self,
        temporal: dict[str, datetime | int | float | None],
    ) -> tuple[float, float, float]:
        """Calculate temporal strength metrics.

        Args:
            temporal: Temporal activity data.

        Returns:
            Tuple of (temporal_strength, velocity, trend).
        """
        import math

        emails_7d = temporal.get("emails_7d") or 0
        emails_prev_7d = temporal.get("emails_prev_7d") or 0
        last_activity = temporal.get("last_activity")

        # Velocity: emails per day in last 7 days (normalized to 0-1)
        velocity = min(emails_7d / 7.0, 5.0) / 5.0 if isinstance(emails_7d, int) else 0.0

        # Trend: compare last 7 days vs previous 7 days
        if isinstance(emails_7d, int) and isinstance(emails_prev_7d, int):
            if emails_prev_7d > 0:
                trend = (emails_7d - emails_prev_7d) / emails_prev_7d
                trend = max(-1.0, min(1.0, trend))  # Clamp to [-1, 1]
            elif emails_7d > 0:
                trend = 1.0  # New activity where there was none
            else:
                trend = 0.0
        else:
            trend = 0.0

        # Recency: exponential decay based on days since last activity
        if isinstance(last_activity, datetime):
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=UTC)
            days_since = (datetime.now(UTC) - last_activity).days
            half_life = 7.0  # Half the value every 7 days
            recency = math.exp(-days_since * math.log(2) / half_life)
        else:
            recency = 0.0

        # Combined temporal strength
        temporal_strength = (
            recency * 0.35
            + velocity * 0.35
            + (trend + 1) / 2 * 0.30  # Normalize trend from [-1,1] to [0,1]
        )

        return temporal_strength, velocity, trend

    async def compute_temporal_strength(
        self,
        user_id: UUID | None = None,
    ) -> int:
        """Compute temporal strength for all clusters.

        Args:
            user_id: Optional user ID for multi-tenant.

        Returns:
            Number of clusters updated.
        """
        # Get all content clusters
        clusters, _ = await self.repository.list_by_dimension(
            dimension="content",
            user_id=user_id,
            limit=500,
        )

        updated = 0
        for metadata in clusters:
            temporal = await self._get_temporal_activity(
                cluster_id=metadata.cluster_id,
                user_id=user_id,
            )

            # Calculate temporal strength
            strength, velocity, trend = self._calculate_temporal_strength(temporal)

            # Update metadata
            metadata.temporal_strength = strength
            metadata.activity_velocity = velocity
            metadata.activity_trend = trend

            first_activity = temporal.get("first_activity")
            last_activity = temporal.get("last_activity")
            emails_7d = temporal.get("emails_7d")
            emails_30d = temporal.get("emails_30d")

            if isinstance(first_activity, datetime):
                metadata.first_activity_at = first_activity
            if isinstance(last_activity, datetime):
                metadata.last_activity_at = last_activity
            if isinstance(emails_7d, int):
                metadata.emails_last_7d = emails_7d
            if isinstance(emails_30d, int):
                metadata.emails_last_30d = emails_30d

            updated += 1

        await self.session.commit()
        logger.info(f"Computed temporal strength for {updated} clusters")
        return updated
