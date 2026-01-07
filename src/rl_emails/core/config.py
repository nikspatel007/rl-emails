"""Configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import dotenv_values


@dataclass
class Config:
    """Application configuration."""

    database_url: str
    mbox_path: Path | None = None
    parsed_jsonl: Path | None = None
    your_email: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    # Multi-tenant context (optional for backward compatibility)
    org_id: UUID | None = None
    user_id: UUID | None = None
    # Google OAuth (Phase 2)
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Config:
        """Load configuration from environment and .env file.

        Args:
            env_file: Path to .env file. If None, searches for .env in current
                     directory and parent directories.

        Returns:
            Config instance with loaded values.

        Raises:
            ValueError: If required DATABASE_URL is not set.
        """
        # Load from .env file if provided
        config: dict[str, Any] = {}
        if env_file and env_file.exists():
            config = dict(dotenv_values(env_file))

        # Environment variables override .env file
        database_url = os.environ.get("DATABASE_URL") or config.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is required")

        mbox_path_str = os.environ.get("MBOX_PATH") or config.get("MBOX_PATH")
        parsed_jsonl_str = os.environ.get("PARSED_JSONL") or config.get("PARSED_JSONL")

        return cls(
            database_url=database_url,
            mbox_path=Path(mbox_path_str) if mbox_path_str else None,
            parsed_jsonl=Path(parsed_jsonl_str) if parsed_jsonl_str else None,
            your_email=os.environ.get("YOUR_EMAIL") or config.get("YOUR_EMAIL"),
            openai_api_key=os.environ.get("OPENAI_API_KEY") or config.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
            or config.get("ANTHROPIC_API_KEY"),
            google_client_id=os.environ.get("GOOGLE_CLIENT_ID") or config.get("GOOGLE_CLIENT_ID"),
            google_client_secret=os.environ.get("GOOGLE_CLIENT_SECRET")
            or config.get("GOOGLE_CLIENT_SECRET"),
            google_redirect_uri=os.environ.get("GOOGLE_REDIRECT_URI")
            or config.get("GOOGLE_REDIRECT_URI")
            or "http://localhost:8000/auth/google/callback",
        )

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of missing required field names.
        """
        missing = []
        if not self.database_url:
            missing.append("DATABASE_URL")
        return missing

    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)

    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    def has_llm(self) -> bool:
        """Check if any LLM API key is configured."""
        return self.has_openai() or self.has_anthropic()

    def has_google_oauth(self) -> bool:
        """Check if Google OAuth credentials are configured."""
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def is_multi_tenant(self) -> bool:
        """Check if running in multi-tenant mode."""
        return self.user_id is not None

    def with_user(self, user_id: UUID, org_id: UUID | None = None) -> Config:
        """Create a new config with user context.

        Args:
            user_id: User UUID for multi-tenant filtering.
            org_id: Optional organization UUID.

        Returns:
            New Config instance with user context.
        """
        from dataclasses import replace

        return replace(self, user_id=user_id, org_id=org_id)
