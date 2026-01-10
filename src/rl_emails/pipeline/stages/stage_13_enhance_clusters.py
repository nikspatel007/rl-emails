"""Stage 13: Enhanced cluster analysis with labeling, project detection, and temporal strength.

This stage runs after clustering (stage 9) to:
1. Compute temporal strength metrics for all clusters
2. Detect projects from content clusters
3. Label clusters using LLM (if API keys available)
4. Enrich cluster metadata with participant counts and activity data
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from rl_emails.pipeline.stages.base import StageResult

if TYPE_CHECKING:
    from rl_emails.core.config import Config

logger = logging.getLogger(__name__)


def run(config: Config) -> StageResult:
    """Run enhanced cluster analysis.

    Args:
        config: Pipeline configuration.

    Returns:
        StageResult with processing stats.
    """
    return asyncio.run(_run_async(config))


async def _run_async(config: Config) -> StageResult:
    """Async implementation of enhanced cluster analysis."""
    import os

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from rl_emails.services.cluster_labeler import ClusterLabelerService
    from rl_emails.services.project_detector import ProjectDetectorService

    start = time.time()
    stats: dict[str, int] = {}

    # Get async database URL
    db_url = config.database_url
    if "postgresql://" in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

    engine = create_async_engine(db_url)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            # Get user_id for multi-tenant mode
            user_id = config.user_id if config.is_multi_tenant else None

            # 1. Compute temporal strength
            logger.info("Computing temporal strength for clusters...")
            detector = ProjectDetectorService(session)
            temporal_count = await detector.compute_temporal_strength(user_id=user_id)
            stats["temporal_enriched"] = temporal_count
            logger.info(f"Computed temporal strength for {temporal_count} clusters")

            # 2. Enrich cluster metadata (participant counts, last activity)
            logger.info("Enriching cluster metadata...")
            enriched_count = await detector.enrich_cluster_metadata(user_id=user_id)
            stats["metadata_enriched"] = enriched_count
            logger.info(f"Enriched metadata for {enriched_count} clusters")

            # 3. Detect projects from content clusters
            logger.info("Detecting projects from clusters...")
            summary = await detector.detect_projects(user_id=user_id, min_score=0.3)
            stats["clusters_analyzed"] = summary.clusters_analyzed
            stats["projects_detected"] = summary.projects_detected
            stats["active_projects"] = summary.active_projects
            stats["stale_projects"] = summary.stale_projects
            stats["completed_projects"] = summary.completed_projects
            logger.info(
                f"Detected {summary.projects_detected} projects "
                f"({summary.active_projects} active, {summary.stale_projects} stale)"
            )

            # 4. Label clusters with LLM (if API keys available)
            has_openai = bool(os.environ.get("OPENAI_API_KEY"))
            has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

            if has_openai or has_anthropic:
                logger.info("Labeling clusters with LLM...")
                labeler = ClusterLabelerService(session)
                label_results = await labeler.label_unlabeled_clusters(
                    dimension="content",
                    user_id=user_id,
                    limit=50,  # Label top 50 clusters
                )
                labeled_count = sum(1 for r in label_results if r.success)
                stats["clusters_labeled"] = labeled_count
                logger.info(f"Labeled {labeled_count} clusters")
            else:
                logger.info("Skipping LLM labeling (no API keys)")
                stats["clusters_labeled"] = 0

    finally:
        await engine.dispose()

    duration = time.time() - start

    message = (
        f"Enhanced {stats.get('temporal_enriched', 0)} clusters: "
        f"{stats.get('projects_detected', 0)} projects detected, "
        f"{stats.get('clusters_labeled', 0)} labeled"
    )

    return StageResult(
        success=True,
        records_processed=stats.get("temporal_enriched", 0),
        duration_seconds=duration,
        message=message,
    )
