"""Pipeline orchestrator.

Coordinates the execution of all pipeline stages in order.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import (
    StageResult,
    stage_01_parse_mbox,
    stage_02_import_postgres,
    stage_03_populate_threads,
    stage_04_enrich_emails,
    stage_05_compute_features,
    stage_06_compute_embeddings,
    stage_07_classify_handleability,
    stage_08_populate_users,
    stage_09_cluster_emails,
    stage_10_compute_priority,
    stage_11_llm_classification,
)
from rl_emails.pipeline.status import PipelineStatus, check_postgres, get_status


@dataclass
class PipelineOptions:
    """Options for running the pipeline."""

    workers: int = 10
    batch_size: int = 100
    skip_embeddings: bool = False
    skip_llm: bool = False
    start_from: int = 0
    llm_model: str = "gpt5"
    llm_limit: int | None = None


@dataclass
class PipelineResult:
    """Result of running the pipeline."""

    success: bool
    stages_completed: list[int] = field(default_factory=list)
    stages_skipped: list[int] = field(default_factory=list)
    stages_failed: list[int] = field(default_factory=list)
    duration_seconds: float = 0.0
    final_status: PipelineStatus | None = None
    error: str | None = None

    @property
    def message(self) -> str:
        """Generate summary message."""
        if self.error:
            return f"Pipeline failed: {self.error}"
        if self.stages_failed:
            return f"Pipeline failed at stage {self.stages_failed[0]}"
        return f"Pipeline completed: {len(self.stages_completed)} stages in {self.duration_seconds:.1f}s"


StageRunner = Callable[[Config], StageResult]


@dataclass
class StageDefinition:
    """Definition of a pipeline stage."""

    number: int
    name: str
    description: str
    runner: StageRunner
    requires_openai: bool = False
    requires_llm: bool = False


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline stages."""

    def __init__(self, config: Config, options: PipelineOptions | None = None) -> None:
        """Initialize the orchestrator.

        Args:
            config: Application configuration.
            options: Pipeline options.
        """
        self.config = config
        self.options = options or PipelineOptions()
        self._stages = self._build_stages()
        self._callbacks: list[Callable[[int, str, StageResult | None], None]] = []

    def _build_stages(self) -> list[StageDefinition]:
        """Build the list of stage definitions."""
        return [
            StageDefinition(
                number=1,
                name="parse_mbox",
                description="Parse MBOX to JSONL",
                runner=self._run_stage_01,
            ),
            StageDefinition(
                number=2,
                name="import_postgres",
                description="Import to PostgreSQL",
                runner=self._run_stage_02,
            ),
            StageDefinition(
                number=3,
                name="populate_threads",
                description="Build thread relationships",
                runner=self._run_stage_03,
            ),
            StageDefinition(
                number=4,
                name="enrich_emails",
                description="Compute action labels",
                runner=self._run_stage_04,
            ),
            StageDefinition(
                number=5,
                name="compute_features",
                description="Compute ML features",
                runner=self._run_stage_05,
            ),
            StageDefinition(
                number=6,
                name="compute_embeddings",
                description="Generate embeddings",
                runner=self._run_stage_06,
                requires_openai=True,
            ),
            StageDefinition(
                number=7,
                name="classify_handleability",
                description="Rule-based classification",
                runner=self._run_stage_07,
            ),
            StageDefinition(
                number=8,
                name="populate_users",
                description="Populate user profiles",
                runner=self._run_stage_08,
            ),
            StageDefinition(
                number=9,
                name="cluster_emails",
                description="Multi-dimensional clustering",
                runner=self._run_stage_09,
            ),
            StageDefinition(
                number=10,
                name="compute_priority",
                description="Hybrid priority ranking",
                runner=self._run_stage_10,
            ),
            StageDefinition(
                number=11,
                name="llm_classification",
                description="LLM classification",
                runner=self._run_stage_11,
                requires_llm=True,
            ),
            StageDefinition(
                number=12,
                name="entity_extraction",
                description="Extract projects and tasks",
                runner=self._run_stage_12,
            ),
            StageDefinition(
                number=13,
                name="enhance_clusters",
                description="Enhanced clustering analysis",
                runner=self._run_stage_13,
            ),
        ]

    def add_callback(self, callback: Callable[[int, str, StageResult | None], None]) -> None:
        """Add a callback for stage events.

        Args:
            callback: Function called with (stage_number, event, result).
                     event is 'start', 'complete', 'skip', or 'fail'.
        """
        self._callbacks.append(callback)

    def _notify(self, stage_num: int, event: str, result: StageResult | None) -> None:
        """Notify callbacks of a stage event."""
        for callback in self._callbacks:
            callback(stage_num, event, result)

    def _run_stage_01(self, config: Config) -> StageResult:
        """Run stage 1: Parse MBOX."""
        return stage_01_parse_mbox.run(config)

    def _run_stage_02(self, config: Config) -> StageResult:
        """Run stage 2: Import to PostgreSQL."""
        return stage_02_import_postgres.run(config, batch_size=self.options.batch_size)

    def _run_stage_03(self, config: Config) -> StageResult:
        """Run stage 3: Populate threads."""
        return stage_03_populate_threads.run(config, batch_size=self.options.batch_size)

    def _run_stage_04(self, config: Config) -> StageResult:
        """Run stage 4: Enrich emails."""
        return stage_04_enrich_emails.run(config)

    def _run_stage_05(self, config: Config) -> StageResult:
        """Run stage 5: Compute features."""
        return stage_05_compute_features.run(config, batch_size=self.options.batch_size)

    def _run_stage_06(self, config: Config) -> StageResult:
        """Run stage 6: Compute embeddings."""
        return stage_06_compute_embeddings.run(
            config,
            workers=self.options.workers,
            batch_size=self.options.batch_size,
        )

    def _run_stage_07(self, config: Config) -> StageResult:
        """Run stage 7: Classify handleability."""
        return stage_07_classify_handleability.run(config)

    def _run_stage_08(self, config: Config) -> StageResult:
        """Run stage 8: Populate users."""
        return stage_08_populate_users.run(config)

    def _run_stage_09(self, config: Config) -> StageResult:
        """Run stage 9: Cluster emails."""
        return stage_09_cluster_emails.run(config)

    def _run_stage_10(self, config: Config) -> StageResult:
        """Run stage 10: Compute priority."""
        return stage_10_compute_priority.run(config)

    def _run_stage_11(self, config: Config) -> StageResult:
        """Run stage 11: LLM classification."""
        return stage_11_llm_classification.run(
            config,
            model_name=self.options.llm_model,
            workers=self.options.workers,
            batch_size=self.options.batch_size,
            limit=self.options.llm_limit,
        )

    def _run_stage_12(self, config: Config) -> StageResult:
        """Run stage 12: Entity extraction."""
        from rl_emails.pipeline.stages import stage_12_entity_extraction

        return stage_12_entity_extraction.run(config)

    def _run_stage_13(self, config: Config) -> StageResult:
        """Run stage 13: Enhanced clustering analysis."""
        from rl_emails.pipeline.stages import stage_13_enhance_clusters

        return stage_13_enhance_clusters.run(config)

    def validate(self) -> list[str]:
        """Validate configuration before running.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []

        if not self.config.database_url:
            errors.append("DATABASE_URL not configured")

        if not self.config.mbox_path:
            errors.append("MBOX_PATH not configured")
        elif not Path(self.config.mbox_path).exists():
            errors.append(f"MBOX file not found: {self.config.mbox_path}")

        if not self.config.your_email:
            errors.append("YOUR_EMAIL not configured")

        if not self.options.skip_embeddings and not self.config.openai_api_key:
            errors.append("OPENAI_API_KEY not configured (required for embeddings)")

        if not self.options.skip_llm and not self.config.has_llm():
            errors.append("No LLM API key configured (need OPENAI_API_KEY or ANTHROPIC_API_KEY)")

        if not check_postgres(self.config):
            errors.append("Cannot connect to PostgreSQL")

        return errors

    def run_migrations(self) -> bool:
        """Run alembic database migrations.

        Returns:
            True if migrations succeeded.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def run(self, run_migrations: bool = True) -> PipelineResult:
        """Run the complete pipeline.

        Args:
            run_migrations: Whether to run alembic migrations first.

        Returns:
            PipelineResult with execution details.
        """
        start_time = time.time()
        result = PipelineResult(success=True)

        # Run migrations if requested
        if run_migrations and self.options.start_from == 0:
            if not self.run_migrations():
                result.success = False
                result.error = "Database migrations failed"
                result.duration_seconds = time.time() - start_time
                return result

        # Run each stage
        for stage in self._stages:
            # Skip if before start point
            if stage.number < self.options.start_from:
                result.stages_skipped.append(stage.number)
                self._notify(stage.number, "skip", None)
                continue

            # Skip embeddings if requested
            if self.options.skip_embeddings and stage.requires_openai:
                result.stages_skipped.append(stage.number)
                self._notify(stage.number, "skip", None)
                continue

            # Skip LLM if requested
            if self.options.skip_llm and stage.requires_llm:
                result.stages_skipped.append(stage.number)
                self._notify(stage.number, "skip", None)
                continue

            # Run the stage
            self._notify(stage.number, "start", None)
            try:
                stage_result = stage.runner(self.config)
            except Exception as e:
                stage_result = StageResult(
                    success=False,
                    records_processed=0,
                    duration_seconds=0,
                    message=str(e),
                )

            if stage_result.success:
                result.stages_completed.append(stage.number)
                self._notify(stage.number, "complete", stage_result)
            else:
                result.success = False
                result.stages_failed.append(stage.number)
                result.error = f"Stage {stage.number} failed: {stage_result.message}"
                self._notify(stage.number, "fail", stage_result)
                break

        result.duration_seconds = time.time() - start_time
        result.final_status = get_status(self.config)

        return result

    def run_stage(self, stage_number: int) -> StageResult:
        """Run a single stage by number.

        Args:
            stage_number: Stage number (1-11).

        Returns:
            StageResult from the stage.

        Raises:
            ValueError: If stage number is invalid.
        """
        for stage in self._stages:
            if stage.number == stage_number:
                return stage.runner(self.config)

        raise ValueError(f"Invalid stage number: {stage_number}")

    def get_stage_info(self) -> list[dict[str, Any]]:
        """Get information about all stages.

        Returns:
            List of stage info dictionaries.
        """
        return [
            {
                "number": stage.number,
                "name": stage.name,
                "description": stage.description,
                "requires_openai": stage.requires_openai,
                "requires_llm": stage.requires_llm,
            }
            for stage in self._stages
        ]
