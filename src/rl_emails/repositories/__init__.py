"""Repository classes for data access."""

from rl_emails.repositories.org_user import OrgUserRepository
from rl_emails.repositories.organization import OrganizationRepository
from rl_emails.repositories.sync_state import SyncStateRepository

__all__ = [
    "OrganizationRepository",
    "OrgUserRepository",
    "SyncStateRepository",
]
