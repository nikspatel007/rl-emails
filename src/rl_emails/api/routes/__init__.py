"""API route modules."""

from rl_emails.api.routes.connections import router as connections_router
from rl_emails.api.routes.connections import set_connection_service
from rl_emails.api.routes.health import router as health_router
from rl_emails.api.routes.inbox import router as inbox_router
from rl_emails.api.routes.inbox import set_session_factory as set_inbox_session
from rl_emails.api.routes.projects import router as projects_router
from rl_emails.api.routes.projects import set_session_factory as set_projects_session
from rl_emails.api.routes.tasks import router as tasks_router
from rl_emails.api.routes.tasks import set_session_factory as set_tasks_session
from rl_emails.api.routes.webhooks import router as webhooks_router
from rl_emails.api.routes.webhooks import set_push_service

__all__ = [
    "connections_router",
    "health_router",
    "inbox_router",
    "projects_router",
    "set_connection_service",
    "set_inbox_session",
    "set_projects_session",
    "set_push_service",
    "set_tasks_session",
    "tasks_router",
    "webhooks_router",
]
