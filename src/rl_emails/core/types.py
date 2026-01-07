"""Shared type definitions."""

from __future__ import annotations

from typing import TypedDict


class EmailData(TypedDict, total=False):
    """Email data structure from parsing."""

    message_id: str
    from_email: str
    from_name: str | None
    to_emails: list[str]
    cc_emails: list[str]
    bcc_emails: list[str]
    subject: str | None
    date_str: str | None
    body_text: str
    body_html: str | None
    headers: dict[str, str]
    labels: list[str]
    in_reply_to: str | None
    references: list[str]


class EmailFeatures(TypedDict, total=False):
    """ML features for an email."""

    email_id: int
    relationship_strength: float
    urgency_score: float
    is_service_email: bool
    service_type: str | None
    service_importance: float


class PriorityScores(TypedDict):
    """Priority scoring components."""

    feature_score: float
    replied_similarity: float
    cluster_novelty: float
    sender_novelty: float
    priority_score: float


class LLMFlags(TypedDict):
    """LLM analysis flags."""

    needs_llm: bool
    reason: str | None


class ClusterAssignment(TypedDict, total=False):
    """Cluster assignments across dimensions."""

    email_id: int
    people_cluster_id: int | None
    content_cluster_id: int | None
    behavior_cluster_id: int | None
    service_cluster_id: int | None
    temporal_cluster_id: int | None


# =============================================================================
# MULTI-TENANT TYPES
# =============================================================================


class OrganizationData(TypedDict, total=False):
    """Organization data structure for multi-tenant support."""

    id: str  # UUID as string
    name: str
    slug: str
    settings: dict[str, object]
    created_at: str  # ISO format datetime
    updated_at: str  # ISO format datetime


class OrgUserData(TypedDict, total=False):
    """Organization user data structure."""

    id: str  # UUID as string
    org_id: str  # UUID as string
    email: str
    name: str | None
    role: str  # 'admin' or 'member'
    gmail_connected: bool
    created_at: str  # ISO format datetime
    updated_at: str  # ISO format datetime


class OAuthTokenData(TypedDict, total=False):
    """OAuth token data structure for Gmail API."""

    id: str  # UUID as string
    user_id: str  # UUID as string
    provider: str  # 'google'
    access_token: str
    refresh_token: str
    expires_at: str  # ISO format datetime
    scopes: list[str]
    created_at: str  # ISO format datetime
    updated_at: str  # ISO format datetime


class SyncStateData(TypedDict, total=False):
    """Gmail sync state data structure."""

    id: str  # UUID as string
    user_id: str  # UUID as string
    last_history_id: str | None
    last_sync_at: str | None  # ISO format datetime
    sync_status: str  # 'idle', 'syncing', 'error'
    error_message: str | None
    emails_synced: int
    created_at: str  # ISO format datetime
    updated_at: str  # ISO format datetime


class ClusterMetadataData(TypedDict, total=False):
    """Cluster metadata for auto-labeling and project detection."""

    id: str  # UUID as string
    org_id: str | None  # UUID as string
    user_id: str | None  # UUID as string
    dimension: str  # 'people', 'content', 'behavior', 'service', 'temporal'
    cluster_id: int
    size: int
    representative_email_id: int | None
    auto_label: str | None
    pct_replied: float | None
    avg_response_time_hours: float | None
    avg_relationship_strength: float | None
    is_project: bool
    project_status: str | None  # 'active', 'stale', 'completed'
    last_activity_at: str | None  # ISO format datetime
    created_at: str  # ISO format datetime
