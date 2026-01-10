"""Cluster labeling service using LLM for auto-generating cluster labels."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.cluster_metadata import ClusterMetadata
from rl_emails.repositories.cluster_metadata import ClusterMetadataRepository

if TYPE_CHECKING:
    from uuid import UUID

# Suppress LiteLLM debug messages
os.environ["LITELLM_LOG"] = "ERROR"

logger = logging.getLogger(__name__)

# Model mapping (LiteLLM format)
MODELS = {
    "gpt5": "gpt-5-mini",
    "haiku": "anthropic/claude-haiku-4-5",
    "sonnet": "anthropic/claude-sonnet-4-5",
}

DEFAULT_MODEL = "haiku"
DEFAULT_WORKERS = 5

# Prompt templates for different dimensions
LABEL_PROMPTS = {
    "content": """Analyze this email and generate a short label for the email cluster it belongs to.

## Email
From: {from_email}
Subject: {subject}
Date: {date}

---
{body_preview}
---

## Cluster Stats
- Cluster size: {size} emails
- Reply rate: {pct_replied}%
- Avg response time: {avg_response_hours} hours

Generate a 2-5 word label that describes what this group of emails is about.
Focus on the topic, project, or purpose (e.g., "Q4 Budget Review", "Customer Support", "Weekly Standup").

Respond with JSON only:
{{"label": "Your 2-5 word label here", "confidence": 0.0-1.0}}""",
    "people": """Analyze this sender and generate a label for emails from them.

## Sender
Email: {from_email}
Relationship strength: {relationship_strength}
Reply rate to sender: {pct_replied}%

## Sample Email
Subject: {subject}
Date: {date}

---
{body_preview}
---

Generate a 2-4 word label for this sender category (e.g., "Team Lead", "External Client", "Newsletter").

Respond with JSON only:
{{"label": "Your 2-4 word label here", "confidence": 0.0-1.0}}""",
    "service": """Analyze this automated/service email and categorize it.

## Email
From: {from_email}
Subject: {subject}
Date: {date}

---
{body_preview}
---

Determine what type of service email this is.
Generate a 2-4 word label (e.g., "GitHub Notifications", "Billing Alerts", "Newsletter Digest").

Respond with JSON only:
{{"label": "Your 2-4 word label here", "confidence": 0.0-1.0}}""",
    "behavior": """Analyze this email and label the response pattern.

## Email
Subject: {subject}
Action taken: {action}
Response time: {response_time_hours} hours

---
{body_preview}
---

Generate a 2-4 word label for this behavior pattern (e.g., "Quick Replies", "Pending Review", "Auto-archived").

Respond with JSON only:
{{"label": "Your 2-4 word label here", "confidence": 0.0-1.0}}""",
    "temporal": """Analyze this email's timing pattern.

## Email
Subject: {subject}
Day: {day_of_week}
Hour: {hour_of_day}
Is business hours: {is_business_hours}

Generate a 2-4 word label for this time pattern (e.g., "Morning Rush", "Weekend Updates", "After Hours").

Respond with JSON only:
{{"label": "Your 2-4 word label here", "confidence": 0.0-1.0}}""",
}


@dataclass
class LabelResult:
    """Result of labeling a cluster."""

    cluster_id: int
    dimension: str
    label: str | None
    confidence: float
    success: bool
    error: str | None = None


class ClusterLabelerError(Exception):
    """Exception raised for cluster labeling errors."""

    pass


class ClusterLabelerService:
    """Service for auto-generating cluster labels using LLM."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the cluster labeler service.

        Args:
            session: Async database session.
            openai_api_key: OpenAI API key (optional).
            anthropic_api_key: Anthropic API key (optional).
            model: Model to use (gpt5, haiku, sonnet).
        """
        self.session = session
        self.repository = ClusterMetadataRepository(session)
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._completion_func: Any = None

    def _get_model(self) -> str:
        """Get the LLM model to use.

        Returns:
            LiteLLM model string.

        Raises:
            ClusterLabelerError: If no API key is available.
        """
        if self._model and self._model in MODELS:
            return MODELS[self._model]

        if self.anthropic_api_key:
            return MODELS["haiku"]
        if self.openai_api_key:
            return MODELS["gpt5"]

        raise ClusterLabelerError(
            "No LLM API key available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    def _get_completion_func(self) -> Any:
        """Get the LiteLLM completion function.

        Returns:
            LiteLLM completion function.

        Raises:
            ClusterLabelerError: If litellm is not installed.
        """
        if self._completion_func is not None:
            return self._completion_func

        try:
            import litellm

            litellm.suppress_debug_info = True
            from litellm import completion

            self._completion_func = completion
            return completion
        except ImportError as e:  # pragma: no cover
            raise ClusterLabelerError("litellm package not installed") from e

    async def _fetch_representative_email(
        self,
        email_id: int,
    ) -> dict[str, Any] | None:
        """Fetch representative email data for labeling.

        Args:
            email_id: Email ID.

        Returns:
            Email data dictionary or None if not found.
        """
        from sqlalchemy import text

        query = text(
            """
            SELECT
                e.id,
                e.from_email,
                e.subject,
                e.date_parsed,
                SUBSTRING(e.body_text, 1, 500) as body_preview,
                e.action,
                ef.relationship_strength,
                e.response_time_seconds
            FROM emails e
            LEFT JOIN email_features ef ON ef.email_id = e.id
            WHERE e.id = :email_id
        """
        )

        result = await self.session.execute(query, {"email_id": email_id})
        row = result.fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "from_email": row[1] or "unknown",
            "subject": row[2] or "(no subject)",
            "date": str(row[3]) if row[3] else "unknown",
            "body_preview": row[4] or "",
            "action": row[5] or "unknown",
            "relationship_strength": row[6] or 0.0,
            "response_time_hours": (row[7] or 0) / 3600.0,
        }

    def _build_prompt(
        self,
        dimension: str,
        email_data: dict[str, Any],
        cluster_metadata: ClusterMetadata,
    ) -> str:
        """Build the prompt for labeling.

        Args:
            dimension: Clustering dimension.
            email_data: Email data dictionary.
            cluster_metadata: Cluster metadata record.

        Returns:
            Formatted prompt string.
        """
        template = LABEL_PROMPTS.get(dimension, LABEL_PROMPTS["content"])

        return template.format(
            from_email=email_data.get("from_email", "unknown"),
            subject=email_data.get("subject", "(no subject)"),
            date=email_data.get("date", "unknown"),
            body_preview=email_data.get("body_preview", "")[:500],
            size=cluster_metadata.size or 0,
            pct_replied=cluster_metadata.pct_replied or 0,
            avg_response_hours=round(cluster_metadata.avg_response_time_hours or 0, 1),
            relationship_strength=round(email_data.get("relationship_strength", 0), 2),
            action=email_data.get("action", "unknown"),
            response_time_hours=round(email_data.get("response_time_hours", 0), 1),
            day_of_week="Monday",  # Placeholder for temporal
            hour_of_day=9,  # Placeholder for temporal
            is_business_hours=True,  # Placeholder for temporal
        )

    def _call_llm(self, prompt: str, model: str) -> dict[str, Any]:
        """Call the LLM to generate a label.

        Args:
            prompt: The prompt to send.
            model: LiteLLM model string.

        Returns:
            Parsed result dictionary.
        """
        completion_func = self._get_completion_func()

        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        }
        if model.startswith("gpt-5"):
            completion_kwargs["reasoning_effort"] = "minimal"
        else:
            completion_kwargs["temperature"] = 0

        response = completion_func(**completion_kwargs)
        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            if "```" in result_text:
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result_text)
                if json_match:
                    result_text = json_match.group(1)
            result = json.loads(result_text)
            return {
                "label": result.get("label", "").strip()[:100],
                "confidence": float(result.get("confidence", 0.5)),
                "success": True,
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Try to extract label from raw text
            label = result_text[:50].strip()
            return {
                "label": label if label else None,
                "confidence": 0.3,
                "success": bool(label),
            }

    async def label_cluster(
        self,
        cluster_id: int,
        dimension: str,
        user_id: UUID | None = None,
        *,
        force_relabel: bool = False,
    ) -> LabelResult:
        """Label a single cluster.

        Args:
            cluster_id: Cluster ID.
            dimension: Clustering dimension.
            user_id: Optional user ID for multi-tenant.
            force_relabel: Whether to relabel if already labeled.

        Returns:
            LabelResult with the generated label.
        """
        # Get cluster metadata
        metadata = await self.repository.get_by_dimension_and_cluster(
            dimension=dimension,
            cluster_id=cluster_id,
            user_id=user_id,
        )

        if not metadata:
            return LabelResult(
                cluster_id=cluster_id,
                dimension=dimension,
                label=None,
                confidence=0.0,
                success=False,
                error=f"Cluster metadata not found for {dimension}/{cluster_id}",
            )

        # Skip if already labeled (unless force_relabel)
        if metadata.auto_label and not force_relabel:
            return LabelResult(
                cluster_id=cluster_id,
                dimension=dimension,
                label=metadata.auto_label,
                confidence=1.0,
                success=True,
            )

        # Get representative email
        if not metadata.representative_email_id:
            return LabelResult(
                cluster_id=cluster_id,
                dimension=dimension,
                label=None,
                confidence=0.0,
                success=False,
                error="No representative email for cluster",
            )

        email_data = await self._fetch_representative_email(metadata.representative_email_id)
        if not email_data:
            return LabelResult(
                cluster_id=cluster_id,
                dimension=dimension,
                label=None,
                confidence=0.0,
                success=False,
                error="Representative email not found",
            )

        # Build prompt and call LLM
        try:
            model = self._get_model()
            prompt = self._build_prompt(dimension, email_data, metadata)
            result = self._call_llm(prompt, model)

            if result["success"] and result["label"]:
                # Update cluster metadata with label
                await self.repository.update_label(
                    dimension=dimension,
                    cluster_id=cluster_id,
                    label=result["label"],
                    user_id=user_id,
                )

                return LabelResult(
                    cluster_id=cluster_id,
                    dimension=dimension,
                    label=result["label"],
                    confidence=result["confidence"],
                    success=True,
                )

            return LabelResult(
                cluster_id=cluster_id,
                dimension=dimension,
                label=None,
                confidence=0.0,
                success=False,
                error="LLM did not return a valid label",
            )

        except Exception as e:
            logger.error(f"Error labeling cluster {dimension}/{cluster_id}: {e}")
            return LabelResult(
                cluster_id=cluster_id,
                dimension=dimension,
                label=None,
                confidence=0.0,
                success=False,
                error=str(e),
            )

    async def label_unlabeled_clusters(
        self,
        dimension: str | None = None,
        user_id: UUID | None = None,
        *,
        limit: int = 50,
        workers: int = DEFAULT_WORKERS,
    ) -> list[LabelResult]:
        """Label all unlabeled clusters.

        Args:
            dimension: Optional dimension filter.
            user_id: Optional user ID for multi-tenant.
            limit: Maximum clusters to label.
            workers: Number of parallel workers.

        Returns:
            List of LabelResult for each cluster.
        """
        # Get unlabeled clusters
        unlabeled = await self.repository.list_unlabeled(
            dimension=dimension,
            user_id=user_id,
            limit=limit,
        )

        if not unlabeled:
            return []

        results: list[LabelResult] = []

        # Process in parallel using thread pool for LLM calls
        for metadata in unlabeled:
            result = await self.label_cluster(
                cluster_id=metadata.cluster_id,
                dimension=metadata.dimension,
                user_id=user_id,
            )
            results.append(result)

        return results

    async def get_labeling_stats(
        self,
        user_id: UUID | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get labeling statistics per dimension.

        Args:
            user_id: Optional user ID for multi-tenant.

        Returns:
            Dictionary of stats per dimension.
        """
        dimensions = ["people", "content", "behavior", "service", "temporal"]
        stats = {}

        for dim in dimensions:
            dim_stats = await self.repository.get_stats(dim, user_id)
            total = int(dim_stats["total_clusters"])
            labeled = int(dim_stats["labeled_clusters"])
            stats[dim] = {
                "total_clusters": total,
                "labeled_clusters": labeled,
                "unlabeled_clusters": total - labeled,
            }

        return stats
