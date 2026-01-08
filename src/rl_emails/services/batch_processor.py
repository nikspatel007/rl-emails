"""Batch processor for running emails through the full pipeline.

This processor handles batch-oriented pipeline execution:
1. Store emails to database
2. Compute ML features
3. Generate embeddings (OpenAI)
4. LLM classification
5. Rule-based classification
6. Clustering
7. Priority ranking

Designed for use with progressive sync to process emails as they arrive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from rl_emails.core.config import Config
    from rl_emails.core.types import EmailData


@dataclass
class BatchResult:
    """Result of processing a batch of emails."""

    success: bool
    emails_stored: int
    features_computed: int
    embeddings_generated: int
    llm_classified: int
    error: str | None = None


@dataclass
class StageResult:
    """Result of a single pipeline stage."""

    success: bool
    records_processed: int
    error: str | None = None


class BatchProcessor:
    """Processes email batches through the full pipeline.

    This class provides batch-oriented processing for the pipeline stages,
    optimized for streaming/progressive sync workflows.

    Example:
        processor = BatchProcessor(session, config)

        # Process a batch of emails
        result = await processor.process_batch(
            emails=email_list,
            run_embeddings=True,
            run_llm=True,
        )

        # Update clusters after all batches
        await processor.update_clusters()
        await processor.compute_priority()
    """

    def __init__(
        self,
        session: AsyncSession,
        config: Config,
    ) -> None:
        """Initialize batch processor.

        Args:
            session: Async database session.
            config: Pipeline configuration.
        """
        self.session = session
        self.config = config

    async def process_batch(
        self,
        emails: list[EmailData],
        run_embeddings: bool = True,
        run_llm: bool = True,
        llm_limit: int | None = None,
    ) -> BatchResult:
        """Process a batch of emails through the pipeline.

        Args:
            emails: List of EmailData to process.
            run_embeddings: Whether to generate embeddings.
            run_llm: Whether to run LLM classification.
            llm_limit: Max emails to classify with LLM.

        Returns:
            BatchResult with processing stats.
        """
        if not emails:
            return BatchResult(
                success=True,
                emails_stored=0,
                features_computed=0,
                embeddings_generated=0,
                llm_classified=0,
            )

        try:
            # Step 1: Store emails and commit immediately
            # (sync stages need to see the data via their own connections)
            stored_count = await self._store_emails(emails)
            await self.session.commit()

            # Step 2: Compute features for new emails
            features_count = await self._compute_features()

            # Step 3: Rule-based classification
            await self._classify_rule_based()

            # Step 4: Generate embeddings (if enabled)
            embeddings_count = 0
            if run_embeddings:
                embeddings_count = await self._generate_embeddings()

            # Step 5: LLM classification (if enabled)
            llm_count = 0
            if run_llm:
                llm_count = await self._classify_llm(limit=llm_limit)

            return BatchResult(
                success=True,
                emails_stored=stored_count,
                features_computed=features_count,
                embeddings_generated=embeddings_count,
                llm_classified=llm_count,
            )

        except Exception as e:
            await self.session.rollback()
            return BatchResult(
                success=False,
                emails_stored=0,
                features_computed=0,
                embeddings_generated=0,
                llm_classified=0,
                error=str(e),
            )

    async def _store_emails(self, emails: list[EmailData]) -> int:
        """Store emails in the database.

        Args:
            emails: Emails to store.

        Returns:
            Number of emails stored.
        """
        from datetime import datetime

        stored = 0

        for email_data in emails:
            message_id = str(email_data.get("message_id", ""))
            if not message_id:
                continue

            subject = str(email_data.get("subject", ""))
            from_email = str(email_data.get("from_email", ""))
            from_name = email_data.get("from_name")
            date_str = str(email_data.get("date_str", ""))
            body_text = str(email_data.get("body_text", ""))
            body_html = email_data.get("body_html")
            labels_raw = email_data.get("labels", [])
            to_emails_raw = email_data.get("to_emails", [])
            cc_emails_raw = email_data.get("cc_emails", [])
            in_reply_to = email_data.get("in_reply_to")
            references_raw = email_data.get("references", [])

            # Safely cast to lists
            labels: list[str] = list(labels_raw) if isinstance(labels_raw, list) else []
            to_emails: list[str] = list(to_emails_raw) if isinstance(to_emails_raw, list) else []
            cc_emails: list[str] = list(cc_emails_raw) if isinstance(cc_emails_raw, list) else []
            references: list[str] = list(references_raw) if isinstance(references_raw, list) else []

            # Parse date
            date_parsed = None
            if date_str:
                try:
                    date_parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Insert into raw_emails
            raw_result = await self.session.execute(
                text(
                    """
                    INSERT INTO raw_emails (
                        message_id, in_reply_to, references_raw,
                        date_raw, from_raw, to_raw, cc_raw,
                        subject_raw, body_text, body_html, labels_raw
                    ) VALUES (
                        :message_id, :in_reply_to, :references_raw,
                        :date_raw, :from_raw, :to_raw, :cc_raw,
                        :subject_raw, :body_text, :body_html, :labels_raw
                    )
                    ON CONFLICT (message_id) DO NOTHING
                    RETURNING id
                """
                ),
                {
                    "message_id": message_id,
                    "in_reply_to": in_reply_to,
                    "references_raw": " ".join(references) if references else None,
                    "date_raw": date_str or None,
                    "from_raw": f"{from_name} <{from_email}>" if from_name else from_email,
                    "to_raw": ", ".join(to_emails) if to_emails else None,
                    "cc_raw": ", ".join(cc_emails) if cc_emails else None,
                    "subject_raw": subject,
                    "body_text": body_text,
                    "body_html": body_html,
                    "labels_raw": ",".join(labels) if labels else None,
                },
            )

            raw_row = raw_result.fetchone()
            if raw_row is None:
                # Already exists
                continue

            raw_id = raw_row[0]
            is_sent = "SENT" in labels

            # Insert into emails
            await self.session.execute(
                text(
                    """
                    INSERT INTO emails (
                        raw_email_id, message_id, in_reply_to, date_parsed,
                        from_email, from_name, to_emails, cc_emails, subject,
                        body_text, body_preview, word_count, labels,
                        has_attachments, is_sent, enriched_at
                    ) VALUES (
                        :raw_email_id, :message_id, :in_reply_to, :date_parsed,
                        :from_email, :from_name, :to_emails, :cc_emails, :subject,
                        :body_text, :body_preview, :word_count, :labels,
                        :has_attachments, :is_sent, NOW()
                    )
                    ON CONFLICT (message_id) DO NOTHING
                """
                ),
                {
                    "raw_email_id": raw_id,
                    "message_id": message_id,
                    "in_reply_to": in_reply_to,
                    "date_parsed": date_parsed,
                    "from_email": from_email,
                    "from_name": from_name,
                    "to_emails": to_emails if to_emails else None,
                    "cc_emails": cc_emails if cc_emails else None,
                    "subject": subject,
                    "body_text": body_text,
                    "body_preview": body_text[:200] if body_text else None,
                    "word_count": len(body_text.split()) if body_text else 0,
                    "labels": labels if labels else None,
                    "has_attachments": False,
                    "is_sent": is_sent,
                },
            )

            stored += 1

        return stored

    async def _compute_features(self) -> int:
        """Compute ML features for emails without features.

        Returns:
            Number of emails processed.
        """
        import asyncio

        from rl_emails.pipeline.stages import stage_05_compute_features

        # Run sync stage in thread pool to avoid nested event loop
        result = await asyncio.to_thread(stage_05_compute_features.run, self.config)
        return result.records_processed

    async def _classify_rule_based(self) -> int:
        """Run rule-based classification.

        Returns:
            Number of emails classified.
        """
        import asyncio

        from rl_emails.pipeline.stages import stage_07_classify_handleability

        result = await asyncio.to_thread(stage_07_classify_handleability.run, self.config)
        return result.records_processed

    async def _generate_embeddings(self) -> int:
        """Generate embeddings for emails without them.

        Returns:
            Number of embeddings generated.
        """
        if not self.config.openai_api_key:
            return 0

        import asyncio

        from rl_emails.pipeline.stages import stage_06_compute_embeddings

        result = await asyncio.to_thread(stage_06_compute_embeddings.run, self.config)
        return result.records_processed

    async def _classify_llm(self, limit: int | None = None) -> int:
        """Run LLM classification.

        Args:
            limit: Max emails to classify.

        Returns:
            Number of emails classified.
        """
        if not self.config.openai_api_key and not self.config.anthropic_api_key:
            return 0

        import asyncio

        from rl_emails.pipeline.stages import stage_11_llm_classification

        def run_llm() -> int:
            result = stage_11_llm_classification.run(self.config, limit=limit)
            return result.records_processed

        return await asyncio.to_thread(run_llm)

    async def update_clusters(self) -> StageResult:
        """Update email clusters.

        Should be called after processing batches to update clustering.

        Returns:
            StageResult with clustering outcome.
        """
        import asyncio

        try:
            from rl_emails.pipeline.stages import stage_09_cluster_emails

            result = await asyncio.to_thread(stage_09_cluster_emails.run, self.config)
            return StageResult(
                success=result.success,
                records_processed=result.records_processed,
                error=result.message if not result.success else None,
            )
        except Exception as e:
            return StageResult(success=False, records_processed=0, error=str(e))

    async def compute_priority(self) -> StageResult:
        """Compute email priority rankings.

        Should be called after clustering to rank emails.

        Returns:
            StageResult with priority computation outcome.
        """
        import asyncio

        try:
            from rl_emails.pipeline.stages import stage_10_compute_priority

            result = await asyncio.to_thread(stage_10_compute_priority.run, self.config)
            return StageResult(
                success=result.success,
                records_processed=result.records_processed,
                error=result.message if not result.success else None,
            )
        except Exception as e:
            return StageResult(success=False, records_processed=0, error=str(e))

    async def populate_users(self) -> StageResult:
        """Populate user profiles.

        Returns:
            StageResult with user population outcome.
        """
        import asyncio

        try:
            from rl_emails.pipeline.stages import stage_08_populate_users

            result = await asyncio.to_thread(stage_08_populate_users.run, self.config)
            return StageResult(
                success=result.success,
                records_processed=result.records_processed,
                error=result.message if not result.success else None,
            )
        except Exception as e:
            return StageResult(success=False, records_processed=0, error=str(e))

    async def build_threads(self) -> StageResult:
        """Build thread relationships.

        Returns:
            StageResult with thread building outcome.
        """
        import asyncio

        try:
            from rl_emails.pipeline.stages import stage_03_populate_threads

            result = await asyncio.to_thread(stage_03_populate_threads.run, self.config)
            return StageResult(
                success=result.success,
                records_processed=result.records_processed,
                error=result.message if not result.success else None,
            )
        except Exception as e:
            return StageResult(success=False, records_processed=0, error=str(e))

    async def enrich_emails(self) -> StageResult:
        """Enrich emails with action labels.

        Returns:
            StageResult with enrichment outcome.
        """
        import asyncio

        try:
            from rl_emails.pipeline.stages import stage_04_enrich_emails

            result = await asyncio.to_thread(stage_04_enrich_emails.run, self.config)
            return StageResult(
                success=result.success,
                records_processed=result.records_processed,
                error=result.message if not result.success else None,
            )
        except Exception as e:
            return StageResult(success=False, records_processed=0, error=str(e))
