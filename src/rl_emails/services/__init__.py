"""Service layer for business logic orchestration."""

from rl_emails.services.auth_service import AuthService
from rl_emails.services.batch_processor import BatchProcessor, BatchResult
from rl_emails.services.cluster_labeler import (
    ClusterLabelerError,
    ClusterLabelerService,
    LabelResult,
)
from rl_emails.services.entity_extraction import (
    ExtractionResult,
    PriorityContextBuilder,
    ProjectExtractor,
    TaskExtractor,
    extract_all_entities,
)
from rl_emails.services.inbox_service import InboxService
from rl_emails.services.progressive_sync import (
    PhaseConfig,
    ProgressiveSyncResult,
    ProgressiveSyncService,
    SyncPhase,
    SyncProgress,
)
from rl_emails.services.project_detector import (
    ProjectDetectionConfig,
    ProjectDetectionResult,
    ProjectDetectionSummary,
    ProjectDetectorService,
)
from rl_emails.services.project_service import ProjectNotFoundError, ProjectService
from rl_emails.services.push_notification import (
    InvalidNotificationError,
    NotificationData,
    NotificationDeduplicator,
    NotificationResult,
    PushNotificationError,
    PushNotificationService,
    UserNotFoundError,
)
from rl_emails.services.sync_service import SyncResult, SyncService
from rl_emails.services.task_service import TaskNotFoundError, TaskService

__all__ = [
    "AuthService",
    "BatchProcessor",
    "BatchResult",
    "ClusterLabelerError",
    "ClusterLabelerService",
    "ExtractionResult",
    "InboxService",
    "LabelResult",
    "InvalidNotificationError",
    "NotificationData",
    "NotificationDeduplicator",
    "NotificationResult",
    "PhaseConfig",
    "PriorityContextBuilder",
    "ProgressiveSyncResult",
    "ProgressiveSyncService",
    "ProjectDetectionConfig",
    "ProjectDetectionResult",
    "ProjectDetectionSummary",
    "ProjectDetectorService",
    "ProjectExtractor",
    "ProjectNotFoundError",
    "ProjectService",
    "PushNotificationError",
    "PushNotificationService",
    "SyncPhase",
    "SyncProgress",
    "SyncResult",
    "SyncService",
    "TaskExtractor",
    "TaskNotFoundError",
    "TaskService",
    "UserNotFoundError",
    "extract_all_entities",
]
