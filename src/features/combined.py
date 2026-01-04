#!/usr/bin/env python3
"""Combined feature vector builder for email RL system.

Combines all feature extraction modules into a unified feature vector
suitable for the policy network input.
"""

from dataclasses import dataclass
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


@dataclass
class CombinedFeatures:
    """Combined feature set from all extractors."""
    # Individual feature sets
    project: ProjectFeatures
    topic: TopicFeatures
    task: TaskFeatures
    people: PeopleFeatures

    # Computed scores
    project_score: float
    topic_score: float
    task_score: float
    people_score: float
    overall_priority: float

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to unified feature vector for ML pipeline.

        Returns concatenated vector of all features plus computed scores.

        Vector structure:
        - Project features: 8 dims
        - Topic features: 20 dims
        - Task features: 12 dims
        - People features: 15 dims
        - Computed scores: 5 dims (project, topic, task, people, overall)
        - Total: 60 dims
        """
        # Get individual vectors
        project_vec = self.project.to_feature_vector()
        topic_vec = self.topic.to_feature_vector()
        task_vec = self.task.to_feature_vector()
        people_vec = self.people.to_feature_vector()

        # Computed scores
        scores = [
            self.project_score,
            self.topic_score,
            self.task_score,
            self.people_score,
            self.overall_priority,
        ]

        if HAS_NUMPY:
            return np.concatenate([
                np.asarray(project_vec),
                np.asarray(topic_vec),
                np.asarray(task_vec),
                np.asarray(people_vec),
                np.array(scores, dtype=np.float32),
            ])
        else:
            # List concatenation
            combined = []
            for vec in [project_vec, topic_vec, task_vec, people_vec]:
                if isinstance(vec, list):
                    combined.extend(vec)
                else:
                    combined.extend(list(vec))
            combined.extend(scores)
            return combined

    def to_dict(self) -> dict:
        """Convert to dictionary representation for inspection/logging."""
        return {
            'scores': {
                'project': self.project_score,
                'topic': self.topic_score,
                'task': self.task_score,
                'people': self.people_score,
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
            'vector_dims': len(self.to_feature_vector()),
        }


def compute_overall_priority(
    project_score: float,
    topic_score: float,
    task_score: float,
    people_score: float,
    *,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute weighted overall priority score.

    Args:
        project_score: Score from project features
        topic_score: Score from topic classification
        task_score: Score from task extraction
        people_score: Score from people analysis
        weights: Optional custom weights (must sum to 1.0)

    Returns:
        Overall priority score 0-1
    """
    if weights is None:
        weights = {
            'people': 0.30,
            'project': 0.25,
            'topic': 0.25,
            'task': 0.20,
        }

    priority = (
        weights.get('people', 0.30) * people_score +
        weights.get('project', 0.25) * project_score +
        weights.get('topic', 0.25) * topic_score +
        weights.get('task', 0.20) * task_score
    )

    return min(1.0, max(0.0, priority))


def extract_combined_features(
    email: dict,
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
    weights: Optional[dict[str, float]] = None,
) -> CombinedFeatures:
    """Extract all features from an email and combine into unified representation.

    Args:
        email: Email dictionary with subject, body, from, to, cc, x_from, x_to fields
        user_email: The user's email address for context
        user_context: Optional dict with historical interaction data
        weights: Optional custom priority weights

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

    # Compute scores
    # Project score uses the project_features internal score
    project_score = project_features.project_score
    topic_score = compute_topic_score(topic_features)
    task_score = compute_task_score(task_features)
    people_score = compute_people_score(people_features)

    # Compute overall priority
    overall_priority = compute_overall_priority(
        project_score,
        topic_score,
        task_score,
        people_score,
        weights=weights,
    )

    return CombinedFeatures(
        project=project_features,
        topic=topic_features,
        task=task_features,
        people=people_features,
        project_score=project_score,
        topic_score=topic_score,
        task_score=task_score,
        people_score=people_score,
        overall_priority=overall_priority,
    )


def extract_batch(
    emails: list[dict],
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
    weights: Optional[dict[str, float]] = None,
) -> list[CombinedFeatures]:
    """Extract features from a batch of emails.

    Args:
        emails: List of email dictionaries
        user_email: The user's email address
        user_context: Optional historical context
        weights: Optional priority weights

    Returns:
        List of CombinedFeatures, one per email
    """
    return [
        extract_combined_features(
            email,
            user_email=user_email,
            user_context=user_context,
            weights=weights,
        )
        for email in emails
    ]


def build_feature_matrix(
    emails: list[dict],
    *,
    user_email: str = '',
    user_context: Optional[dict] = None,
) -> Union["np.ndarray", list[list[float]]]:
    """Build feature matrix from batch of emails.

    Args:
        emails: List of email dictionaries
        user_email: The user's email address
        user_context: Optional historical context

    Returns:
        Matrix of shape (n_emails, n_features)
    """
    features_list = extract_batch(emails, user_email=user_email, user_context=user_context)
    vectors = [f.to_feature_vector() for f in features_list]

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
    ):
        """Initialize extractor with user context.

        Args:
            user_email: The user's email address
            user_context: Historical interaction data
            weights: Priority computation weights
        """
        self.user_email = user_email
        self.user_context = user_context or {}
        self.weights = weights

    def extract(self, email: dict) -> CombinedFeatures:
        """Extract features from a single email."""
        return extract_combined_features(
            email,
            user_email=self.user_email,
            user_context=self.user_context,
            weights=self.weights,
        )

    def extract_batch(self, emails: list[dict]) -> list[CombinedFeatures]:
        """Extract features from multiple emails."""
        return [self.extract(email) for email in emails]

    def to_vector(self, email: dict) -> Union["np.ndarray", list[float]]:
        """Extract and convert to feature vector."""
        return self.extract(email).to_feature_vector()

    def to_matrix(self, emails: list[dict]) -> Union["np.ndarray", list[list[float]]]:
        """Extract and convert to feature matrix."""
        vectors = [self.to_vector(email) for email in emails]
        if HAS_NUMPY:
            return np.stack(vectors)
        return vectors

    def update_context(self, user_context: dict) -> None:
        """Update user context for future extractions."""
        self.user_context.update(user_context)

    @property
    def feature_dim(self) -> int:
        """Return dimensionality of feature vector."""
        # 8 + 20 + 12 + 15 + 5 = 60
        return 60


# Feature dimension constants for external reference
FEATURE_DIMS = {
    'project': 8,
    'topic': 20,
    'task': 12,
    'people': 15,
    'scores': 5,
    'total': 60,
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
    print("FEATURE VECTOR:")
    vec = features.to_feature_vector()
    print(f"  Dimensions: {len(vec)}")
    if HAS_NUMPY:
        print(f"  Shape: {vec.shape}")
        print(f"  Dtype: {vec.dtype}")
    print(f"  First 10 values: {list(vec[:10])}")
    print()
    print("=" * 60)
