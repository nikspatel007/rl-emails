"""Feature extraction modules for email RL system.

This package provides feature extraction for the email RL policy network:
- project: Project mentions, deadlines, action items
- topic: Topic classification and content type detection
- task: Task extraction and deadline urgency
- people: Sender importance and relationship scoring
- temporal: Time-based features (hour, day, freshness, thread timing)
- content: Semantic embeddings from email body text
- relationship: Communication graph and relational priority model
- service: Service domain classification (transactional, newsletter, etc.)
- combined: Unified feature vector for policy network input
"""

from .project import (
    ProjectFeatures,
    ProjectMention,
    Deadline,
    ActionItem,
    extract_project_features,
    detect_project_mentions,
    detect_deadlines,
    extract_action_items,
)

from .topic import (
    TopicFeatures,
    classify_topic,
    compute_topic_score,
)

from .task import (
    TaskFeatures,
    ExtractedDeadline,
    ExtractedActionItem,
    extract_tasks,
    compute_task_score,
)

from .people import (
    PeopleFeatures,
    extract_people_features,
    compute_people_score,
)

from .temporal import (
    TemporalFeatures,
    extract_temporal_features,
    compute_temporal_score,
)

from .content import (
    ContentFeatures,
    ContentFeatureExtractor,
    extract_content_features,
    get_content_extractor,
    DEFAULT_EMBEDDING_DIM,
)

from .relationship import (
    CommunicationGraph,
    RelationshipFeatures,
    UserBaseline,
    build_communication_graph,
    process_email_dataset,
)

from .sender_frequency import (
    SenderFrequency,
    SenderFrequencyIndex,
    compute_sender_frequency_from_emails,
    compute_sender_frequency_from_db,
    compute_sender_frequency_from_db_sync,
)

from .combined import (
    CombinedFeatures,
    CombinedFeatureExtractor,
    extract_combined_features,
    extract_batch,
    build_feature_matrix,
    compute_overall_priority,
    FEATURE_DIMS,
)

from .service import (
    ServiceType,
    ServiceFeatures,
    classify_service,
    compute_service_score,
    is_automated_sender,
    get_service_type_description,
)

__all__ = [
    # Project features
    'ProjectFeatures',
    'ProjectMention',
    'Deadline',
    'ActionItem',
    'extract_project_features',
    'detect_project_mentions',
    'detect_deadlines',
    'extract_action_items',
    # Topic features
    'TopicFeatures',
    'classify_topic',
    'compute_topic_score',
    # Task features
    'TaskFeatures',
    'ExtractedDeadline',
    'ExtractedActionItem',
    'extract_tasks',
    'compute_task_score',
    # People features
    'PeopleFeatures',
    'extract_people_features',
    'compute_people_score',
    # Temporal features
    'TemporalFeatures',
    'extract_temporal_features',
    'compute_temporal_score',
    # Content features
    'ContentFeatures',
    'ContentFeatureExtractor',
    'extract_content_features',
    'get_content_extractor',
    'DEFAULT_EMBEDDING_DIM',
    # Relationship features
    'CommunicationGraph',
    'RelationshipFeatures',
    'UserBaseline',
    'build_communication_graph',
    'process_email_dataset',
    # Sender frequency features
    'SenderFrequency',
    'SenderFrequencyIndex',
    'compute_sender_frequency_from_emails',
    'compute_sender_frequency_from_db',
    'compute_sender_frequency_from_db_sync',
    # Combined features
    'CombinedFeatures',
    'CombinedFeatureExtractor',
    'extract_combined_features',
    'extract_batch',
    'build_feature_matrix',
    'compute_overall_priority',
    'FEATURE_DIMS',
    # Service features
    'ServiceType',
    'ServiceFeatures',
    'classify_service',
    'compute_service_score',
    'is_automated_sender',
    'get_service_type_description',
]
