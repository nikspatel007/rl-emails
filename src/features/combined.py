#!/usr/bin/env python3
"""Combined feature vector builder for email RL system.

Combines all feature extraction modules into a unified feature vector
suitable for the policy network input.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from .project import ProjectFeatures, extract_project_features
from .topic import TopicFeatures, classify_topic, compute_topic_score
from .task import TaskFeatures, extract_tasks, compute_task_score
from .people import PeopleFeatures, extract_people_features, compute_people_score
from .temporal import TemporalFeatures, extract_temporal_features, compute_temporal_score
from .content import (
    ContentFeatures,
    ContentFeatureExtractor,
    get_content_extractor,
    DEFAULT_EMBEDDING_DIM,
)
from .relationship import (
    RelationshipFeatures,
    CommunicationGraph,
)


@dataclass
class CombinedFeatures:
    """Combined feature set from all extractors."""
    # Individual feature sets
    project: ProjectFeatures
    topic: TopicFeatures
    task: TaskFeatures
    people: PeopleFeatures
    temporal: TemporalFeatures
    content: Optional[ContentFeatures] = None  # Optional for backward compat
    relationship: Optional[RelationshipFeatures] = None  # Optional for backward compat

    # Computed scores
    project_score: float = 0.0
    topic_score: float = 0.0
    task_score: float = 0.0
    people_score: float = 0.0
    temporal_score: float = 0.0
    relationship_score: float = 0.0
    overall_priority: float = 0.0

    def to_feature_vector(
        self,
        include_content: bool = True,
        include_relationship: bool = True,
    ) -> Union["np.ndarray", list[float]]:
        """Convert to unified feature vector for ML pipeline.

        Returns concatenated vector of all features plus computed scores.

        Vector structure (full):
        - Project features: 8 dims
        - Topic features: 20 dims
        - Task features: 12 dims
        - People features: 15 dims
        - Temporal features: 8 dims
        - Relationship features: 11 dims (if include_relationship=True and available)
        - Computed scores: 7 dims (project, topic, task, people, temporal, relationship, overall)
        - Content embedding: 384 dims (if include_content=True and content is available)
        - Total: 69 dims (base) + 11 (relationship) + 384 (content) = 464 dims (full)

        Args:
            include_content: Whether to include content embeddings (default True)
            include_relationship: Whether to include relationship features (default True)

        Returns:
            Feature vector as numpy array or list
        """
        # Get individual vectors
        project_vec = self.project.to_feature_vector()
        topic_vec = self.topic.to_feature_vector()
        task_vec = self.task.to_feature_vector()
        people_vec = self.people.to_feature_vector()
        temporal_vec = self.temporal.to_feature_vector()

        # Computed scores (now includes relationship_score)
        scores = [
            self.project_score,
            self.topic_score,
            self.task_score,
            self.people_score,
            self.temporal_score,
            self.relationship_score,
            self.overall_priority,
        ]

        if HAS_NUMPY:
            base_vec = np.concatenate([
                np.asarray(project_vec),
                np.asarray(topic_vec),
                np.asarray(task_vec),
                np.asarray(people_vec),
                np.asarray(temporal_vec),
                np.array(scores, dtype=np.float32),
            ])
            # Add relationship features if available and requested
            if include_relationship and self.relationship is not None:
                relationship_vec = self.relationship.to_feature_vector()
                base_vec = np.concatenate([base_vec, np.asarray(relationship_vec)])
            # Add content embedding if available and requested
            if include_content and self.content is not None:
                content_vec = self.content.to_feature_vector()
                return np.concatenate([base_vec, np.asarray(content_vec)])
            return base_vec
        else:
            # List concatenation
            combined = []
            for vec in [project_vec, topic_vec, task_vec, people_vec, temporal_vec]:
                if isinstance(vec, list):
                    combined.extend(vec)
                else:
                    combined.extend(list(vec))
            combined.extend(scores)
            # Add relationship features if available and requested
            if include_relationship and self.relationship is not None:
                relationship_vec = self.relationship.to_feature_vector()
                if isinstance(relationship_vec, list):
                    combined.extend(relationship_vec)
                else:
                    combined.extend(list(relationship_vec))
            # Add content embedding if available and requested
            if include_content and self.content is not None:
                content_vec = self.content.to_feature_vector()
                if isinstance(content_vec, list):
                    combined.extend(content_vec)
                else:
                    combined.extend(list(content_vec))
            return combined

    def to_dict(self) -> dict:
        """Convert to dictionary representation for inspection/logging."""
        result = {
            'scores': {
                'project': self.project_score,
                'topic': self.topic_score,
                'task': self.task_score,
                'people': self.people_score,
                'temporal': self.temporal_score,
                'relationship': self.relationship_score,
                'overall_priority': self.overall_priority,
            },
            'project': {
                'mention_count': self.project.project_mention_count,
                'deadline_count': self.project.deadline_count,
                'action_count': self.project.action_item_count,
                'urgency': self.project.urgency_score,
            },
            'topic': {
                'primary': self.topic.primary_topic,
                'confidence': self.topic.topic_confidence,
                'is_question': self.topic.is_question,
                'is_action_request': self.topic.is_action_request,
                'urgency': self.topic.urgency_score,
            },
            'task': {
                'has_deadline': self.task.has_deadline,
                'deadline_urgency': self.task.deadline_urgency,
                'has_deliverable': self.task.has_deliverable,
                'is_assigned': self.task.is_assigned_to_user,
                'effort': self.task.estimated_effort,
            },
            'people': {
                'sender_org_level': self.people.sender_org_level,
                'is_internal': self.people.sender_is_internal,
                'is_direct_to': self.people.is_direct_to,
                'includes_executives': self.people.includes_executives,
                'sender_importance': self.people.sender_importance,
            },
            'temporal': {
                'hour_of_day': self.temporal.hour_of_day,
                'day_of_week': self.temporal.day_of_week,
                'time_since_receipt_hours': self.temporal.time_since_receipt_hours,
                'is_business_hours': self.temporal.is_business_hours,
                'is_weekend': self.temporal.is_weekend,
                'urgency': self.temporal.temporal_urgency,
            },
            'vector_dims': len(self.to_feature_vector(
                include_content=self.content is not None,
                include_relationship=self.relationship is not None,
            )),
            'vector_dims_base': len(self.to_feature_vector(
                include_content=False,
                include_relationship=False,
            )),
        }
        if self.relationship is not None:
            result['relationship'] = {
                'sender_response_deviation': self.relationship.sender_response_deviation,
                'sender_frequency_rank': self.relationship.sender_frequency_rank,
                'inferred_hierarchy': self.relationship.inferred_hierarchy,
                'relationship_strength': self.relationship.relationship_strength,
                'emails_from_sender_30d': self.relationship.emails_from_sender_30d,
                'response_rate_to_sender': self.relationship.response_rate_to_sender,
                'communication_asymmetry': self.relationship.communication_asymmetry,
            }
        if self.content is not None:
            result['content'] = {
                'subject_length': self.content.subject_length,
                'body_length': self.content.body_length,
                'body_word_count': self.content.body_word_count,
                'embedding_dim': self.content.embedding_dim,
            }
        return result


def compute_relationship_score(features: RelationshipFeatures) -> float:
    """Compute relationship score from relationship features.

    Combines hierarchy, strength, and response patterns into a 0-1 score.

    Args:
        features: RelationshipFeatures from relationship analysis

    Returns:
        Relationship score 0-1
    """
    # Weight hierarchy and strength heavily
    score = (
        features.inferred_hierarchy * 0.30 +
        features.relationship_strength * 0.25 +
        features.sender_frequency_rank * 0.20 +
        features.response_rate_to_sender * 0.15 +
        (0.5 + features.sender_response_deviation * 0.5) * 0.10  # Normalize deviation
    )
    return min(1.0, max(0.0, score))


def compute_overall_priority(
    project_score: float,
    topic_score: float,
    task_score: float,
    people_score: float,
    temporal_score: float = 0.5,
    relationship_score: float = 0.5,
    *,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute weighted overall priority score.

    Args:
        project_score: Score from project features
        topic_score: Score from topic classification
        task_score: Score from task extraction
        people_score: Score from people analysis
        temporal_score: Score from temporal features (default 0.5 for backward compat)
        relationship_score: Score from relationship features (default 0.5 for backward compat)
        weights: Optional custom weights (must sum to 1.0)

    Returns:
        Overall priority score 0-1
    """
    if weights is None:
        weights = {
            'people': 0.20,
            'project': 0.18,
            'topic': 0.18,
            'task': 0.18,
            'temporal': 0.13,
            'relationship': 0.13,
        }

    priority = (
        weights.get('people', 0.20) * people_score +
        weights.get('project', 0.18) * project_score +
        weights.get('topic', 0.18) * topic_score +
        weights.get('task', 0.18) * task_score +
        weights.get('temporal', 0.13) * temporal_score +
        weights.get('relationship', 0.13) * relationship_score
    )

    return min(1.0, max(0.0, priority))


def extract_combined_features(
    email: dict,
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
    thread_context: Optional[dict] = None,
    reference_time: Optional["datetime"] = None,
    weights: Optional[dict[str, float]] = None,
    include_content: bool = False,
    content_extractor: Optional[ContentFeatureExtractor] = None,
    relationship_graph: Optional[CommunicationGraph] = None,
) -> CombinedFeatures:
    """Extract all features from an email and combine into unified representation.

    Args:
        email: Email dictionary with subject, body, from, to, cc, x_from, x_to, date fields
        user_email: The user's email address for context
        user_context: Optional dict with historical interaction data
        thread_context: Optional dict with thread timing info (for temporal features)
        reference_time: Time to compute temporal features from (default: now)
        weights: Optional custom priority weights
        include_content: Whether to extract content embeddings (default False for backward compat)
        content_extractor: Optional content extractor instance (uses global if None)
        relationship_graph: Optional CommunicationGraph for relationship features

    Returns:
        CombinedFeatures with all extracted information
    """
    subject = email.get('subject', '')
    body = email.get('body', '')

    # Extract individual feature sets
    project_features = extract_project_features(subject, body)
    topic_features = classify_topic(subject, body)
    task_features = extract_tasks(subject, body)
    people_features = extract_people_features(
        email,
        user_email=user_email,
        user_context=user_context,
    )
    temporal_features = extract_temporal_features(
        email,
        reference_time=reference_time,
        thread_context=thread_context,
    )

    # Extract content features if requested
    content_features = None
    if include_content:
        if content_extractor is not None:
            content_features = content_extractor.extract(email)
        else:
            content_features = get_content_extractor().extract(email)

    # Extract relationship features if graph provided
    relationship_features = None
    relationship_score = 0.5  # Default neutral score
    if relationship_graph is not None and user_email:
        relationship_features = relationship_graph.get_relationship_features(email, user_email)
        relationship_score = compute_relationship_score(relationship_features)

    # Compute scores
    # Project score uses the project_features internal score
    project_score = project_features.project_score
    topic_score = compute_topic_score(topic_features)
    task_score = compute_task_score(task_features)
    people_score = compute_people_score(people_features)
    temporal_score = compute_temporal_score(temporal_features)

    # Compute overall priority
    overall_priority = compute_overall_priority(
        project_score,
        topic_score,
        task_score,
        people_score,
        temporal_score,
        relationship_score,
        weights=weights,
    )

    return CombinedFeatures(
        project=project_features,
        topic=topic_features,
        task=task_features,
        people=people_features,
        temporal=temporal_features,
        content=content_features,
        relationship=relationship_features,
        project_score=project_score,
        topic_score=topic_score,
        task_score=task_score,
        people_score=people_score,
        temporal_score=temporal_score,
        relationship_score=relationship_score,
        overall_priority=overall_priority,
    )


def extract_batch(
    emails: list[dict],
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
    thread_context: Optional[dict] = None,
    reference_time: Optional["datetime"] = None,
    weights: Optional[dict[str, float]] = None,
    include_content: bool = False,
    content_extractor: Optional[ContentFeatureExtractor] = None,
    relationship_graph: Optional[CommunicationGraph] = None,
) -> list[CombinedFeatures]:
    """Extract features from a batch of emails.

    Args:
        emails: List of email dictionaries
        user_email: The user's email address
        user_context: Optional historical context
        thread_context: Optional thread timing info
        reference_time: Time to compute temporal features from
        weights: Optional priority weights
        include_content: Whether to extract content embeddings
        content_extractor: Optional content extractor instance
        relationship_graph: Optional CommunicationGraph for relationship features

    Returns:
        List of CombinedFeatures, one per email
    """
    # For content features, batch extraction is more efficient
    content_features_list = None
    if include_content:
        extractor = content_extractor or get_content_extractor()
        content_features_list = extractor.extract_batch(emails)

    results = []
    for i, email in enumerate(emails):
        subject = email.get('subject', '')
        body = email.get('body', '')

        # Extract individual feature sets
        project_features = extract_project_features(subject, body)
        topic_features = classify_topic(subject, body)
        task_features = extract_tasks(subject, body)
        people_features = extract_people_features(
            email,
            user_email=user_email,
            user_context=user_context,
        )
        temporal_features = extract_temporal_features(
            email,
            reference_time=reference_time,
            thread_context=thread_context,
        )

        # Get content features from pre-computed batch
        content_features = content_features_list[i] if content_features_list else None

        # Extract relationship features if graph provided
        relationship_features = None
        relationship_score = 0.5  # Default neutral score
        if relationship_graph is not None and user_email:
            relationship_features = relationship_graph.get_relationship_features(email, user_email)
            relationship_score = compute_relationship_score(relationship_features)

        # Compute scores
        project_score = project_features.project_score
        topic_score = compute_topic_score(topic_features)
        task_score = compute_task_score(task_features)
        people_score = compute_people_score(people_features)
        temporal_score = compute_temporal_score(temporal_features)

        overall_priority = compute_overall_priority(
            project_score, topic_score, task_score, people_score, temporal_score,
            relationship_score,
            weights=weights
        )

        results.append(CombinedFeatures(
            project=project_features,
            topic=topic_features,
            task=task_features,
            people=people_features,
            temporal=temporal_features,
            content=content_features,
            relationship=relationship_features,
            project_score=project_score,
            topic_score=topic_score,
            task_score=task_score,
            people_score=people_score,
            temporal_score=temporal_score,
            relationship_score=relationship_score,
            overall_priority=overall_priority,
        ))

    return results


def build_feature_matrix(
    emails: list[dict],
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
    include_content: bool = False,
    include_relationship: bool = False,
    relationship_graph: Optional[CommunicationGraph] = None,
) -> Union["np.ndarray", list[list[float]]]:
    """Build feature matrix from batch of emails.

    Args:
        emails: List of email dictionaries
        user_email: The user's email address
        user_context: Optional historical context
        include_content: Whether to include content embeddings
        include_relationship: Whether to include relationship features in vector
        relationship_graph: Optional CommunicationGraph for relationship features

    Returns:
        Matrix of shape (n_emails, n_features)
    """
    features_list = extract_batch(
        emails,
        user_email=user_email,
        user_context=user_context,
        include_content=include_content,
        relationship_graph=relationship_graph,
    )
    vectors = [f.to_feature_vector(
        include_content=include_content,
        include_relationship=include_relationship,
    ) for f in features_list]

    if HAS_NUMPY:
        return np.stack(vectors)
    return vectors


class CombinedFeatureExtractor:
    """Stateful feature extractor for consistent processing.

    Stores user context and configuration for reuse across multiple emails.
    """

    def __init__(
        self,
        user_email: str = '',
        user_context: Optional[dict] = None,
        weights: Optional[dict[str, float]] = None,
        include_content: bool = False,
        include_relationship: bool = False,
        content_model: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        relationship_graph: Optional[CommunicationGraph] = None,
    ):
        """Initialize extractor with user context.

        Args:
            user_email: The user's email address
            user_context: Historical interaction data
            weights: Priority computation weights
            include_content: Whether to include content embeddings
            include_relationship: Whether to include relationship features
            content_model: Sentence transformer model name for content embeddings
            device: Device for content model ('cpu', 'cuda', 'mps', or None for auto)
            relationship_graph: Pre-built CommunicationGraph for relationship features
        """
        self.user_email = user_email
        self.user_context = user_context or {}
        self.weights = weights
        self.include_content = include_content
        self.include_relationship = include_relationship
        self._content_extractor = None
        self._relationship_graph = relationship_graph

        if include_content:
            self._content_extractor = ContentFeatureExtractor(
                model_name=content_model,
                device=device,
            )

    def extract(self, email: dict) -> CombinedFeatures:
        """Extract features from a single email."""
        return extract_combined_features(
            email,
            user_email=self.user_email,
            user_context=self.user_context,
            weights=self.weights,
            include_content=self.include_content,
            content_extractor=self._content_extractor,
            relationship_graph=self._relationship_graph,
        )

    def extract_batch(self, emails: list[dict]) -> list[CombinedFeatures]:
        """Extract features from multiple emails (optimized for batch)."""
        return extract_batch(
            emails,
            user_email=self.user_email,
            user_context=self.user_context,
            weights=self.weights,
            include_content=self.include_content,
            content_extractor=self._content_extractor,
            relationship_graph=self._relationship_graph,
        )

    def to_vector(self, email: dict) -> Union["np.ndarray", list[float]]:
        """Extract and convert to feature vector."""
        return self.extract(email).to_feature_vector(
            include_content=self.include_content,
            include_relationship=self.include_relationship,
        )

    def to_matrix(self, emails: list[dict]) -> Union["np.ndarray", list[list[float]]]:
        """Extract and convert to feature matrix."""
        features_list = self.extract_batch(emails)
        vectors = [f.to_feature_vector(
            include_content=self.include_content,
            include_relationship=self.include_relationship,
        ) for f in features_list]
        if HAS_NUMPY:
            return np.stack(vectors)
        return vectors

    def update_context(self, user_context: dict) -> None:
        """Update user context for future extractions."""
        self.user_context.update(user_context)

    def set_relationship_graph(self, graph: CommunicationGraph) -> None:
        """Set or update the relationship graph for feature extraction."""
        self._relationship_graph = graph

    @property
    def feature_dim(self) -> int:
        """Return dimensionality of feature vector."""
        # Base: 8 + 20 + 12 + 15 + 8 + 7 = 70 (scores now include relationship)
        base_dim = 70
        if self.include_relationship:
            # Add relationship feature dimension (11 dims)
            base_dim += 11
        if self.include_content:
            # Add content embedding dimension
            if self._content_extractor is not None:
                return base_dim + self._content_extractor.embedding_dim
            return base_dim + DEFAULT_EMBEDDING_DIM
        return base_dim


# Feature dimension constants for external reference
FEATURE_DIMS = {
    'project': 8,
    'topic': 20,
    'task': 12,
    'people': 15,
    'temporal': 8,
    'relationship': 11,
    'scores': 7,  # project, topic, task, people, temporal, relationship, overall
    'content': DEFAULT_EMBEDDING_DIM,  # 384 for all-MiniLM-L6-v2
    'total_base': 70,  # Without relationship or content (8+20+12+15+8+7)
    'total_with_relationship': 81,  # With relationship (70+11)
    'total_with_content': 70 + DEFAULT_EMBEDDING_DIM,  # 454 with content
    'total_full': 81 + DEFAULT_EMBEDDING_DIM,  # 465 with relationship and content
}


if __name__ == '__main__':
    # Example usage
    sample_email = {
        'from': 'john.smith@enron.com',
        'to': 'jane.doe@enron.com',
        'cc': 'bob.jones@enron.com',
        'x_from': 'Smith, John (Vice President, Trading)',
        'x_to': 'Doe, Jane',
        'subject': 'URGENT: Project Eagle Phase II - Review Required by Friday',
        'body': """
        Hi Jane,

        I need your decision on the Project Eagle Phase II proposal by end of day Friday.

        Please review the attached document and let me know if you have any concerns.
        This is time-sensitive as we need to submit to the board by Monday.

        Key items for your review:
        - Budget estimates (Contract #12345)
        - Timeline for Q2 deliverables
        - Resource allocation for the team

        Can you also send me the updated risk assessment? We need it before we can
        proceed with the client meeting next week.

        Thanks,
        John
        """,
    }

    # Create extractor with user context
    extractor = CombinedFeatureExtractor(
        user_email='jane.doe@enron.com',
        user_context={
            'reply_rate_to_sender': 0.85,
            'avg_response_time_hours': 3.0,
            'emails_from_sender_30d': 25,
            'last_interaction_days': 1,
        }
    )

    # Extract features
    features = extractor.extract(sample_email)

    print("=" * 60)
    print("COMBINED FEATURE EXTRACTION")
    print("=" * 60)
    print()
    print("SCORES:")
    print(f"  Project:  {features.project_score:.2f}")
    print(f"  Topic:    {features.topic_score:.2f}")
    print(f"  Task:     {features.task_score:.2f}")
    print(f"  People:   {features.people_score:.2f}")
    print(f"  Temporal: {features.temporal_score:.2f}")
    print(f"  Overall:  {features.overall_priority:.2f}")
    print()
    print("PROJECT FEATURES:")
    print(f"  Mentions: {features.project.project_mention_count}")
    print(f"  Deadlines: {features.project.deadline_count}")
    print(f"  Actions: {features.project.action_item_count}")
    print()
    print("TOPIC FEATURES:")
    print(f"  Primary: {features.topic.primary_topic}")
    print(f"  Question: {features.topic.is_question}")
    print(f"  Action request: {features.topic.is_action_request}")
    print(f"  Decision needed: {features.topic.is_decision_needed}")
    print()
    print("TASK FEATURES:")
    print(f"  Has deadline: {features.task.has_deadline}")
    if features.task.deadline_text:
        print(f"    Text: {features.task.deadline_text}")
    print(f"  Urgency: {features.task.deadline_urgency:.2f}")
    print(f"  Assigned: {features.task.is_assigned_to_user}")
    print(f"  Effort: {features.task.estimated_effort}")
    print()
    print("PEOPLE FEATURES:")
    print(f"  Sender: {features.people.sender_email}")
    print(f"  Org level: {features.people.sender_org_level}")
    print(f"  Direct to: {features.people.is_direct_to}")
    print(f"  Importance: {features.people.sender_importance:.2f}")
    print()
    print("TEMPORAL FEATURES:")
    print(f"  Hour of day: {features.temporal.hour_of_day}")
    print(f"  Day of week: {features.temporal.day_of_week}")
    print(f"  Business hours: {features.temporal.is_business_hours}")
    print(f"  Weekend: {features.temporal.is_weekend}")
    print(f"  Time since receipt: {features.temporal.time_since_receipt_hours:.1f}h")
    print(f"  Urgency: {features.temporal.temporal_urgency:.2f}")
    print()
    print("FEATURE VECTOR:")
    vec = features.to_feature_vector()
    print(f"  Dimensions: {len(vec)}")
    if HAS_NUMPY:
        print(f"  Shape: {vec.shape}")
        print(f"  Dtype: {vec.dtype}")
    print(f"  First 10 values: {list(vec[:10])}")
    print()
    print("=" * 60)
