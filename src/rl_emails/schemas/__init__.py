"""Pydantic schemas for rl-emails."""

from rl_emails.schemas.cluster_metadata import (
    ClusterDimension,
    ClusterLabelRequest,
    ClusterLabelResponse,
    ClusterLabelResult,
    ClusterMetadataCreate,
    ClusterMetadataListResponse,
    ClusterMetadataResponse,
    ClusterMetadataUpdate,
    ClusterStatsResponse,
    ProjectClusterResponse,
    ProjectDetectionRequest,
    ProjectDetectionResponse,
    ProjectStatus,
)
from rl_emails.schemas.cluster_metadata import (
    ProjectListResponse as ClusterProjectListResponse,
)
from rl_emails.schemas.inbox import (
    EmailSummary,
    InboxStats,
    PriorityContext,
    PriorityEmail,
    PriorityInboxResponse,
)
from rl_emails.schemas.oauth_token import (
    OAuthTokenCreate,
    OAuthTokenResponse,
    OAuthTokenStatus,
    OAuthTokenUpdate,
)
from rl_emails.schemas.org_user import (
    OrgUserCreate,
    OrgUserResponse,
    OrgUserUpdate,
)
from rl_emails.schemas.organization import (
    OrganizationCreate,
    OrganizationResponse,
    OrganizationUpdate,
)
from rl_emails.schemas.project import (
    ProjectCreate,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)
from rl_emails.schemas.sync import SyncStateResponse, SyncStateUpdate
from rl_emails.schemas.task import (
    TaskCreate,
    TaskDetailResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatusUpdate,
    TaskUpdate,
)
from rl_emails.schemas.watch import (
    WatchSubscriptionCreate,
    WatchSubscriptionResponse,
    WatchSubscriptionStatus,
    WatchSubscriptionUpdate,
)

__all__ = [
    "ClusterDimension",
    "ClusterLabelRequest",
    "ClusterLabelResponse",
    "ClusterLabelResult",
    "ClusterMetadataCreate",
    "ClusterMetadataListResponse",
    "ClusterMetadataResponse",
    "ClusterMetadataUpdate",
    "ClusterProjectListResponse",
    "ClusterStatsResponse",
    "ProjectClusterResponse",
    "ProjectDetectionRequest",
    "ProjectDetectionResponse",
    "ProjectStatus",
    "EmailSummary",
    "InboxStats",
    "OAuthTokenCreate",
    "OAuthTokenResponse",
    "OAuthTokenStatus",
    "OAuthTokenUpdate",
    "OrganizationCreate",
    "OrganizationResponse",
    "OrganizationUpdate",
    "OrgUserCreate",
    "OrgUserResponse",
    "OrgUserUpdate",
    "PriorityContext",
    "PriorityEmail",
    "PriorityInboxResponse",
    "ProjectCreate",
    "ProjectDetailResponse",
    "ProjectListResponse",
    "ProjectResponse",
    "ProjectUpdate",
    "SyncStateResponse",
    "SyncStateUpdate",
    "TaskCreate",
    "TaskDetailResponse",
    "TaskListResponse",
    "TaskResponse",
    "TaskStatusUpdate",
    "TaskUpdate",
    "WatchSubscriptionCreate",
    "WatchSubscriptionResponse",
    "WatchSubscriptionStatus",
    "WatchSubscriptionUpdate",
]
