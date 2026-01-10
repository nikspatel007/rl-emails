"""Repository classes for data access."""

from rl_emails.repositories.cluster_metadata import ClusterMetadataRepository
from rl_emails.repositories.oauth_token import OAuthTokenRepository
from rl_emails.repositories.org_user import OrgUserRepository
from rl_emails.repositories.organization import OrganizationRepository
from rl_emails.repositories.project import ProjectRepository
from rl_emails.repositories.sync_state import SyncStateRepository
from rl_emails.repositories.task import TaskRepository
from rl_emails.repositories.watch_subscription import WatchSubscriptionRepository

__all__ = [
    "ClusterMetadataRepository",
    "OAuthTokenRepository",
    "OrganizationRepository",
    "OrgUserRepository",
    "ProjectRepository",
    "SyncStateRepository",
    "TaskRepository",
    "WatchSubscriptionRepository",
]
