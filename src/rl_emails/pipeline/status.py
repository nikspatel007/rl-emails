"""Pipeline status utilities.

Provides functions to check pipeline progress and status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg2

from rl_emails.core.config import Config


@dataclass
class PipelineStatus:
    """Status of the pipeline processing."""

    emails: int = 0
    sent_emails: int = 0
    threads: int = 0
    features: int = 0
    embeddings: int = 0
    ai_classification: int = 0
    llm_classification: int = 0
    needs_llm: int = 0
    clusters: int = 0
    priority: int = 0
    users: int = 0
    error: str | None = None

    @property
    def received_emails(self) -> int:
        """Number of received (non-sent) emails."""
        return self.emails - self.sent_emails

    @property
    def llm_remaining(self) -> int:
        """Number of emails still needing LLM classification."""
        return max(0, self.needs_llm - self.llm_classification)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "emails": self.emails,
            "sent_emails": self.sent_emails,
            "received_emails": self.received_emails,
            "threads": self.threads,
            "features": self.features,
            "embeddings": self.embeddings,
            "ai_classification": self.ai_classification,
            "llm_classification": self.llm_classification,
            "needs_llm": self.needs_llm,
            "llm_remaining": self.llm_remaining,
            "clusters": self.clusters,
            "priority": self.priority,
            "users": self.users,
            "error": self.error,
        }


def _safe_count(cur: psycopg2.extensions.cursor, query: str) -> int:
    """Safely execute a count query and return result."""
    try:
        cur.execute(query)
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def get_status(config: Config) -> PipelineStatus:
    """Get current pipeline status from the database.

    Args:
        config: Application configuration.

    Returns:
        PipelineStatus with current counts.
    """
    status = PipelineStatus()

    try:
        conn = psycopg2.connect(config.database_url)
        cur = conn.cursor()

        # Emails
        status.emails = _safe_count(cur, "SELECT COUNT(*) FROM emails")
        status.sent_emails = _safe_count(cur, "SELECT COUNT(*) FROM emails WHERE is_sent = TRUE")

        # Threads
        status.threads = _safe_count(cur, "SELECT COUNT(*) FROM threads")

        # Features
        status.features = _safe_count(cur, "SELECT COUNT(*) FROM email_features")

        # Embeddings
        status.embeddings = _safe_count(cur, "SELECT COUNT(*) FROM email_embeddings")

        # AI Classification
        status.ai_classification = _safe_count(cur, "SELECT COUNT(*) FROM email_ai_classification")

        # Needs LLM
        status.needs_llm = _safe_count(
            cur,
            """
            SELECT COUNT(*) FROM email_ai_classification
            WHERE predicted_handleability = 'needs_llm'
            """,
        )

        # LLM Classification
        status.llm_classification = _safe_count(
            cur, "SELECT COUNT(*) FROM email_llm_classification"
        )

        # Clusters
        status.clusters = _safe_count(cur, "SELECT COUNT(*) FROM email_clusters")

        # Priority
        status.priority = _safe_count(cur, "SELECT COUNT(*) FROM email_priority")

        # Users
        status.users = _safe_count(cur, "SELECT COUNT(*) FROM users")

        conn.close()

    except Exception as e:
        status.error = str(e)

    return status


def check_postgres(config: Config) -> bool:
    """Check if PostgreSQL is accessible.

    Args:
        config: Application configuration.

    Returns:
        True if connection succeeds.
    """
    try:
        conn = psycopg2.connect(config.database_url)
        conn.close()
        return True
    except Exception:
        return False


def format_status(status: PipelineStatus) -> str:
    """Format pipeline status for display.

    Args:
        status: Pipeline status.

    Returns:
        Formatted status string.
    """
    if status.error:
        return f"Error getting status: {status.error}"

    lines = [
        "=" * 60,
        "PIPELINE STATUS",
        "=" * 60,
        f"Emails imported:        {status.emails:,}",
        f"  - Received:           {status.received_emails:,}",
        f"  - Sent:               {status.sent_emails:,}",
        f"Threads built:          {status.threads:,}",
        f"ML features computed:   {status.features:,}",
        f"Embeddings generated:   {status.embeddings:,}",
        f"Rule-based classified:  {status.ai_classification:,}",
        f"LLM classified:         {status.llm_classification:,}",
        f"Clusters created:       {status.clusters:,}",
        f"Priority computed:      {status.priority:,}",
        f"User profiles:          {status.users:,}",
    ]

    if status.needs_llm > 0:
        lines.append(f"Needs LLM (remaining):  {status.llm_remaining:,}")

    lines.append("=" * 60)
    return "\n".join(lines)
