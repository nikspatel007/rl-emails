#!/usr/bin/env python3
"""Topic classification for email content.

Classifies emails by:
- Primary topic category
- Content type flags (meeting, status update, question, etc.)
- Sentiment and urgency signals
"""

import re
from dataclasses import dataclass
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# Topic categories with associated keywords
TOPIC_KEYWORDS = {
    'meeting_scheduling': [
        r'\b(?:meet|meeting|calendar|schedule|availability|conference|call)\b',
        r'\b(?:invite|invitation|rsvp|attend)\b',
        r'\b(?:reschedule|postpone|cancel)\b',
    ],
    'project_update': [
        r'\b(?:update|status|progress|report)\b',
        r'\b(?:milestone|deliverable|phase)\b',
        r'\b(?:completed|finished|done|accomplished)\b',
    ],
    'task_assignment': [
        r'\b(?:task|assignment|assigned|delegate)\b',
        r'\b(?:please|need you to|can you|could you)\b.*\b(?:review|prepare|send|complete)\b',
        r'\b(?:action\s*item|action\s*required)\b',
    ],
    'information_sharing': [
        r'\b(?:fyi|for your information|heads up|just letting you know)\b',
        r'\b(?:attached|enclosed|see below)\b',
        r'\b(?:thought you|might interest|wanted to share)\b',
    ],
    'decision_request': [
        r'\b(?:decision|decide|approval|approve|sign off)\b',
        r'\b(?:your input|your thoughts|what do you think)\b',
        r'\b(?:go ahead|proceed|move forward)\b',
    ],
    'problem_report': [
        r'\b(?:issue|problem|error|bug|failure)\b',
        r'\b(?:broken|not working|failed|crashed)\b',
        r'\b(?:urgent|critical|blocker|blocked)\b',
    ],
    'follow_up': [
        r'\b(?:follow up|following up|circling back)\b',
        r'\b(?:reminder|reminding|just checking)\b',
        r'\b(?:any update|status on|heard back)\b',
    ],
    'social_administrative': [
        r'\b(?:birthday|anniversary|congratulations|welcome)\b',
        r'\b(?:lunch|coffee|happy hour|team)\b',
        r'\b(?:vacation|out of office|pto|holiday)\b',
    ],
    'external_communication': [
        r'\b(?:customer|client|vendor|partner)\b',
        r'\b(?:contract|proposal|quote|bid)\b',
        r'\b(?:sales|marketing|pr)\b',
    ],
    'legal_compliance': [
        r'\b(?:legal|compliance|regulation|policy)\b',
        r'\b(?:confidential|privileged|attorney)\b',
        r'\b(?:nda|agreement|terms|liability)\b',
    ],
}

# Content type detection patterns
MEETING_PATTERNS = [
    r'\b(?:meeting|meet|call|conference)\b',
    r'\b(?:calendar|schedule|availability)\b',
    r'\b(?:invite|invitation)\b',
    r'\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b',
]

STATUS_UPDATE_PATTERNS = [
    r'\b(?:update|status|progress)\b',
    r'\b(?:completed|finished|done)\b',
    r'\b(?:weekly|daily|monthly)\s+(?:report|update|status)\b',
]

QUESTION_PATTERNS = [
    r'\?',
    r'\b(?:what|when|where|who|why|how)\s+(?:is|are|do|does|did|will|would|should|can|could)\b',
    r'\b(?:can you|could you|would you)\b',
    r'\b(?:do you|are you|have you)\b',
]

FYI_PATTERNS = [
    r'\b(?:fyi|for your information)\b',
    r'\b(?:just letting you know|thought you should know)\b',
    r'\b(?:no action needed|no response needed)\b',
]

ACTION_REQUEST_PATTERNS = [
    r'\b(?:please|kindly)\s+(?:review|send|update|confirm|check|prepare|complete)\b',
    r'\b(?:need you to|can you|could you|would you)\b',
    r'\b(?:action required|action needed)\b',
]

DECISION_PATTERNS = [
    r'\b(?:decision|decide|approval|approve)\b',
    r'\b(?:sign off|green light|go ahead)\b',
    r'\b(?:your call|up to you|your decision)\b',
]

ESCALATION_PATTERNS = [
    r'\b(?:escalate|escalating|escalation)\b',
    r'\b(?:urgent|critical|priority)\b',
    r'\b(?:immediately|asap|right away)\b',
]

# Sentiment patterns
POSITIVE_PATTERNS = [
    r'\b(?:thanks|thank you|appreciate|grateful)\b',
    r'\b(?:great|excellent|wonderful|fantastic)\b',
    r'\b(?:good job|well done|nice work)\b',
]

NEGATIVE_PATTERNS = [
    r'\b(?:problem|issue|error|failure)\b',
    r'\b(?:unfortunately|regret|sorry)\b',
    r'\b(?:disappointed|frustrated|concerned)\b',
]

URGENCY_PATTERNS = [
    r'\b(?:urgent|asap|immediately|critical)\b',
    r'\b(?:deadline|time.?sensitive|priority)\b',
    r'\b(?:as soon as possible|right away)\b',
]


@dataclass
class TopicFeatures:
    """Topic classification features for an email."""
    # Primary classification
    primary_topic: str
    topic_confidence: float
    topic_distribution: dict[str, float]

    # Content type flags
    is_meeting_request: bool
    is_status_update: bool
    is_question: bool
    is_fyi_only: bool
    is_action_request: bool
    is_decision_needed: bool
    is_escalation: bool

    # Sentiment
    sentiment_score: float  # -1 to 1
    urgency_score: float    # 0 to 1

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array for ML pipeline.

        Returns 20-dimensional vector:
        - 10 topic probabilities
        - 7 content type flags
        - 1 sentiment score
        - 1 urgency score
        - 1 topic confidence
        """
        # Topic probabilities (10 topics)
        topic_order = [
            'meeting_scheduling', 'project_update', 'task_assignment',
            'information_sharing', 'decision_request', 'problem_report',
            'follow_up', 'social_administrative', 'external_communication',
            'legal_compliance'
        ]
        topic_probs = [self.topic_distribution.get(t, 0.0) for t in topic_order]

        # Content type flags (7)
        flags = [
            1.0 if self.is_meeting_request else 0.0,
            1.0 if self.is_status_update else 0.0,
            1.0 if self.is_question else 0.0,
            1.0 if self.is_fyi_only else 0.0,
            1.0 if self.is_action_request else 0.0,
            1.0 if self.is_decision_needed else 0.0,
            1.0 if self.is_escalation else 0.0,
        ]

        # Scores (3)
        scores = [
            (self.sentiment_score + 1.0) / 2.0,  # Normalize to 0-1
            self.urgency_score,
            self.topic_confidence,
        ]

        values = topic_probs + flags + scores
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count matches for a list of regex patterns."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def _compute_topic_scores(text: str) -> dict[str, float]:
    """Compute raw scores for each topic based on keyword matches."""
    scores = {}
    text_lower = text.lower()
    text_len = max(len(text.split()), 1)

    for topic, patterns in TOPIC_KEYWORDS.items():
        match_count = 0
        for pattern in patterns:
            match_count += len(re.findall(pattern, text_lower))
        # Normalize by text length with diminishing returns
        scores[topic] = min(1.0, match_count / (text_len ** 0.3))

    return scores


def _normalize_distribution(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to a probability distribution."""
    total = sum(scores.values())
    if total == 0:
        # Uniform distribution if no matches
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


def classify_topic(
    subject: str,
    body: str,
    *,
    subject_weight: float = 2.0,
) -> TopicFeatures:
    """Classify email into topic categories.

    Args:
        subject: Email subject line
        body: Email body text
        subject_weight: Weight multiplier for subject matches

    Returns:
        TopicFeatures with classification results
    """
    # Combine text with weighted subject
    combined_text = f"{subject} " * int(subject_weight) + body
    text_lower = combined_text.lower()

    # Compute topic distribution
    raw_scores = _compute_topic_scores(combined_text)
    topic_distribution = _normalize_distribution(raw_scores)

    # Find primary topic
    primary_topic = max(topic_distribution, key=topic_distribution.get)
    topic_confidence = topic_distribution[primary_topic]

    # Detect content types
    is_meeting_request = _count_pattern_matches(text_lower, MEETING_PATTERNS) >= 2
    is_status_update = _count_pattern_matches(text_lower, STATUS_UPDATE_PATTERNS) >= 2
    is_question = _count_pattern_matches(combined_text, QUESTION_PATTERNS) >= 1
    is_fyi_only = _count_pattern_matches(text_lower, FYI_PATTERNS) >= 1
    is_action_request = _count_pattern_matches(text_lower, ACTION_REQUEST_PATTERNS) >= 1
    is_decision_needed = _count_pattern_matches(text_lower, DECISION_PATTERNS) >= 1
    is_escalation = _count_pattern_matches(text_lower, ESCALATION_PATTERNS) >= 1

    # Compute sentiment
    positive_count = _count_pattern_matches(text_lower, POSITIVE_PATTERNS)
    negative_count = _count_pattern_matches(text_lower, NEGATIVE_PATTERNS)
    total_sentiment = positive_count + negative_count
    if total_sentiment > 0:
        sentiment_score = (positive_count - negative_count) / total_sentiment
    else:
        sentiment_score = 0.0

    # Compute urgency
    urgency_matches = _count_pattern_matches(text_lower, URGENCY_PATTERNS)
    urgency_score = min(1.0, urgency_matches * 0.3)

    return TopicFeatures(
        primary_topic=primary_topic,
        topic_confidence=topic_confidence,
        topic_distribution=topic_distribution,
        is_meeting_request=is_meeting_request,
        is_status_update=is_status_update,
        is_question=is_question,
        is_fyi_only=is_fyi_only,
        is_action_request=is_action_request,
        is_decision_needed=is_decision_needed,
        is_escalation=is_escalation,
        sentiment_score=sentiment_score,
        urgency_score=urgency_score,
    )


def compute_topic_score(features: TopicFeatures) -> float:
    """Compute overall topic importance score.

    Higher scores for topics requiring action.

    Args:
        features: TopicFeatures from classify_topic()

    Returns:
        Score from 0 to 1
    """
    # Base score by topic type
    topic_weights = {
        'decision_request': 0.9,
        'problem_report': 0.85,
        'task_assignment': 0.8,
        'follow_up': 0.7,
        'meeting_scheduling': 0.6,
        'project_update': 0.5,
        'external_communication': 0.5,
        'legal_compliance': 0.6,
        'information_sharing': 0.3,
        'social_administrative': 0.2,
    }

    score = topic_weights.get(features.primary_topic, 0.5)

    # Modifiers
    if features.is_question:
        score += 0.1
    if features.is_action_request:
        score += 0.15
    if features.is_decision_needed:
        score += 0.2
    if features.urgency_score > 0.5:
        score += 0.15
    if features.is_escalation:
        score += 0.2

    # FYI-only reduces score
    if features.is_fyi_only and not features.is_action_request:
        score -= 0.2

    return max(0.0, min(1.0, score))


if __name__ == '__main__':
    # Example usage
    sample_subject = "URGENT: Need your approval on Q4 budget proposal"
    sample_body = """
    Hi John,

    I need your decision on the Q4 budget proposal by end of day.

    Please review the attached spreadsheet and let me know if you approve
    or have any concerns. This is time-sensitive as we need to submit
    to finance by tomorrow.

    Key items for your review:
    - Marketing spend increase
    - New headcount request
    - Office renovation budget

    Can you confirm receipt of this email?

    Thanks,
    Sarah
    """

    features = classify_topic(sample_subject, sample_body)

    print("Topic Classification:")
    print(f"  Primary topic: {features.primary_topic}")
    print(f"  Confidence: {features.topic_confidence:.2f}")
    print()
    print("  Topic distribution:")
    for topic, prob in sorted(features.topic_distribution.items(), key=lambda x: -x[1])[:5]:
        print(f"    {topic}: {prob:.3f}")
    print()
    print("  Content flags:")
    print(f"    Meeting request: {features.is_meeting_request}")
    print(f"    Status update: {features.is_status_update}")
    print(f"    Question: {features.is_question}")
    print(f"    FYI only: {features.is_fyi_only}")
    print(f"    Action request: {features.is_action_request}")
    print(f"    Decision needed: {features.is_decision_needed}")
    print(f"    Escalation: {features.is_escalation}")
    print()
    print(f"  Sentiment: {features.sentiment_score:.2f}")
    print(f"  Urgency: {features.urgency_score:.2f}")
    print()
    print(f"  Topic score: {compute_topic_score(features):.2f}")
    print(f"  Feature vector ({len(features.to_feature_vector())} dims)")
