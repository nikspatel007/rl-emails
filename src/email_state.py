"""EmailState dataclass for RL agent state representation.

This module defines the core state representation used by the RL agent
to make decisions about email handling. The state captures:
- Email content and metadata
- Sender/recipient features
- Thread context
- Temporal features
- Derived scores (people, project, topic, task)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

if TYPE_CHECKING:
    import numpy as np


@dataclass
class EmailMetadata:
    """Raw email metadata from parsed email."""
    message_id: str
    date: str
    sender: str
    to: list[str]
    cc: list[str]
    bcc: list[str]
    subject: str
    body: str
    in_reply_to: Optional[str]
    references: list[str]
    folder: str
    user: str
    attachments: list[str]
    file_path: str


@dataclass
class SenderFeatures:
    """Features derived from sender analysis."""
    email: str
    domain: str
    frequency: float  # How often this sender emails user (emails per day)
    importance: float  # Derived from org hierarchy (0-1)
    reply_rate: float  # % of emails from sender that get replies
    org_level: int  # 0=external, 1=peer, 2=manager, 3=executive
    last_interaction_days: int  # Days since last email from sender


@dataclass
class ThreadFeatures:
    """Features about the email thread/conversation."""
    is_reply: bool
    thread_length: int  # Number of emails in thread
    thread_participants: int  # Unique participants in thread
    user_already_replied: bool  # Has user replied in this thread
    thread_age_hours: float  # How old is this thread


@dataclass
class ContentFeatures:
    """Features extracted from email content."""
    has_question: bool  # Contains question marks or question words
    has_deadline: bool  # Mentions dates, "by EOD", "ASAP"
    has_attachment: bool
    email_length: int  # Character count of body
    urgency_signals: float  # 0-1 score for urgent/important/ASAP keywords
    is_automated: bool  # Newsletter, auto-generated
    is_meeting_request: bool
    is_action_request: bool  # Contains "please", "can you", etc.


@dataclass
class TemporalFeatures:
    """Time-based features."""
    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6, Monday=0
    time_since_last_email: float  # Hours since last email from anyone
    timestamp: datetime


@dataclass
class UserContext:
    """Context about the user for personalized predictions."""
    user_email: str
    user_department: str
    user_role: str
    user_manager: Optional[str]
    frequent_contacts: dict[str, float]  # email -> importance score
    typical_daily_volume: int  # Average emails per day
    current_inbox_size: int  # Unread emails


@dataclass
class EmailState:
    """Complete state representation for the RL agent.

    This is the core state that the RL policy network receives when
    making predictions about email handling actions.

    Attributes:
        email: Raw email metadata
        sender: Sender-related features
        thread: Thread/conversation features
        content: Content analysis features
        temporal: Time-based features
        user_context: User profile and context

        subject_embedding: Semantic embedding of subject (768-dim)
        body_embedding: Semantic embedding of body (768-dim)
        topic_vector: Topic model distribution

        people_score: Computed people importance score (0-1)
        project_score: Computed project relevance score (0-1)
        topic_score: Computed topic importance score (0-1)
        task_score: Computed task creation score (0-1)

        action_label: Ground truth action (for training)
    """
    # Structured features
    email: EmailMetadata
    sender: SenderFeatures
    thread: ThreadFeatures
    content: ContentFeatures
    temporal: TemporalFeatures
    user_context: UserContext

    # Embeddings (initialized as None, computed during feature extraction)
    subject_embedding: Optional[np.ndarray] = None
    body_embedding: Optional[np.ndarray] = None
    topic_vector: Optional[np.ndarray] = None

    # Computed scores (0-1 range)
    people_score: float = 0.0
    project_score: float = 0.0
    topic_score: float = 0.0
    task_score: float = 0.0

    # Ground truth for training (from labeled data)
    action_label: Optional[str] = None

    @property
    def priority_score(self) -> float:
        """Combined priority score from all dimensions."""
        weights = {
            'people': 0.30,
            'project': 0.25,
            'topic': 0.25,
            'task': 0.20,
        }
        return (
            weights['people'] * self.people_score +
            weights['project'] * self.project_score +
            weights['topic'] * self.topic_score +
            weights['task'] * self.task_score
        )

    def to_feature_vector(self) -> np.ndarray:
        """Convert state to a flat feature vector for the policy network.

        Returns:
            Concatenated numpy array of all features suitable for neural network input.
        """
        numerical_features = np.array([
            self.people_score,
            self.project_score,
            self.topic_score,
            self.task_score,
            self.sender.frequency,
            self.sender.importance,
            self.sender.reply_rate,
            self.sender.org_level / 3.0,  # Normalized
            self.sender.last_interaction_days / 365.0,  # Normalized
            self.thread.thread_length / 10.0,  # Normalized
            self.thread.thread_participants / 10.0,  # Normalized
            float(self.thread.is_reply),
            float(self.thread.user_already_replied),
            float(self.content.has_question),
            float(self.content.has_deadline),
            float(self.content.has_attachment),
            self.content.email_length / 10000.0,  # Normalized
            self.content.urgency_signals,
            float(self.content.is_automated),
            float(self.content.is_meeting_request),
            float(self.content.is_action_request),
            self.temporal.hour_of_day / 24.0,  # Normalized
            self.temporal.day_of_week / 7.0,  # Normalized
            min(self.temporal.time_since_last_email / 24.0, 1.0),  # Capped at 1 day
            self.user_context.current_inbox_size / 100.0,  # Normalized
        ], dtype=np.float32)

        # Concatenate embeddings if available
        components = [numerical_features]

        if self.subject_embedding is not None:
            components.append(self.subject_embedding)
        if self.body_embedding is not None:
            components.append(self.body_embedding)
        if self.topic_vector is not None:
            components.append(self.topic_vector)

        return np.concatenate(components)


# Action types aligned with labeled data
ACTION_TYPES = [
    'REPLIED',
    'ARCHIVED',
    'DELETED',
    'FORWARDED',
    'COMPOSED',
    'KEPT',
    'JUNK',
]


def parse_email_addresses(addr_string: str) -> list[str]:
    """Parse comma-separated email address string into list."""
    if not addr_string:
        return []
    return [addr.strip() for addr in addr_string.split(',') if addr.strip()]


def extract_domain(email_addr: str) -> str:
    """Extract domain from email address."""
    if '@' in email_addr:
        return email_addr.split('@')[-1].lower()
    return ''


def detect_question(text: str) -> bool:
    """Detect if text contains a question."""
    import re
    if '?' in text:
        return True
    question_words = r'\b(what|when|where|why|how|can you|could you|would you|will you)\b'
    return bool(re.search(question_words, text, re.I))


def detect_deadline(text: str) -> bool:
    """Detect if text mentions a deadline."""
    import re
    deadline_patterns = [
        r'\b(by|before|until|due)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        r'\b(eod|cob|eow|asap|urgent|deadline)\b',
        r'\bby\s+\d{1,2}[/\-]\d{1,2}',
        r'\bdue\s+(date|by)',
    ]
    combined = '|'.join(deadline_patterns)
    return bool(re.search(combined, text, re.I))


def detect_urgency(text: str) -> float:
    """Compute urgency signal score (0-1)."""
    import re
    urgency_words = ['urgent', 'asap', 'immediately', 'critical', 'important', 'priority', 'deadline']
    text_lower = text.lower()
    count = sum(1 for word in urgency_words if word in text_lower)
    return min(count / 3.0, 1.0)


def detect_action_request(text: str) -> bool:
    """Detect if text contains an action request."""
    import re
    patterns = r'\b(please|can you|could you|need you to|action required|kindly)\b'
    return bool(re.search(patterns, text, re.I))


def detect_meeting_request(text: str) -> bool:
    """Detect if text is a meeting request."""
    import re
    patterns = r'\b(meeting|calendar|schedule|availability|invite|conference call)\b'
    return bool(re.search(patterns, text, re.I))


def detect_automated(text: str, sender: str) -> bool:
    """Detect if email is automated/newsletter."""
    import re
    auto_senders = ['noreply', 'no-reply', 'donotreply', 'mailer-daemon', 'postmaster']
    sender_lower = sender.lower()
    if any(auto in sender_lower for auto in auto_senders):
        return True
    auto_patterns = r'\b(unsubscribe|mailing list|automated message|do not reply)\b'
    return bool(re.search(auto_patterns, text, re.I))


def parse_date(date_str: str) -> datetime:
    """Parse email date string to datetime."""
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return datetime.now()


def create_email_state_from_json(
    email_data: dict,
    user_context: Optional[UserContext] = None,
    sender_stats: Optional[dict] = None,
) -> EmailState:
    """Create EmailState from parsed email JSON.

    Args:
        email_data: Dictionary from parse_emails.py output, with 'action' label
        user_context: Optional user context (uses defaults if not provided)
        sender_stats: Optional dict with sender statistics (frequency, reply_rate, etc.)

    Returns:
        EmailState ready for feature extraction and training
    """
    # Parse basic email data
    sender = email_data.get('from', '')
    subject = email_data.get('subject', '')
    body = email_data.get('body', '')
    full_text = f"{subject} {body}"

    # Create EmailMetadata
    metadata = EmailMetadata(
        message_id=email_data.get('message_id', ''),
        date=email_data.get('date', ''),
        sender=sender,
        to=parse_email_addresses(email_data.get('to', '')),
        cc=parse_email_addresses(email_data.get('cc', '')),
        bcc=parse_email_addresses(email_data.get('bcc', '')),
        subject=subject,
        body=body,
        in_reply_to=email_data.get('in_reply_to') or None,
        references=email_data.get('references', '').split() if email_data.get('references') else [],
        folder=email_data.get('folder', ''),
        user=email_data.get('user', ''),
        attachments=email_data.get('attachments', []),
        file_path=email_data.get('file_path', ''),
    )

    # Create SenderFeatures
    stats = sender_stats or {}
    sender_features = SenderFeatures(
        email=sender,
        domain=extract_domain(sender),
        frequency=stats.get('frequency', 0.0),
        importance=stats.get('importance', 0.5),
        reply_rate=stats.get('reply_rate', 0.0),
        org_level=stats.get('org_level', 0),
        last_interaction_days=stats.get('last_interaction_days', 0),
    )

    # Create ThreadFeatures
    is_reply = bool(email_data.get('in_reply_to'))
    thread_features = ThreadFeatures(
        is_reply=is_reply,
        thread_length=len(email_data.get('references', '').split()) + 1 if is_reply else 1,
        thread_participants=len(set(metadata.to + metadata.cc + [sender])),
        user_already_replied=False,  # Would need to check sent folder
        thread_age_hours=0.0,  # Would need thread start time
    )

    # Create ContentFeatures
    content_features = ContentFeatures(
        has_question=detect_question(full_text),
        has_deadline=detect_deadline(full_text),
        has_attachment=len(metadata.attachments) > 0,
        email_length=len(body),
        urgency_signals=detect_urgency(full_text),
        is_automated=detect_automated(full_text, sender),
        is_meeting_request=detect_meeting_request(full_text),
        is_action_request=detect_action_request(full_text),
    )

    # Create TemporalFeatures
    email_dt = parse_date(metadata.date)
    temporal_features = TemporalFeatures(
        hour_of_day=email_dt.hour,
        day_of_week=email_dt.weekday(),
        time_since_last_email=0.0,  # Would need inbox state
        timestamp=email_dt,
    )

    # Create or use provided UserContext
    if user_context is None:
        user_context = UserContext(
            user_email=f"{metadata.user}@enron.com",
            user_department='',
            user_role='',
            user_manager=None,
            frequent_contacts={},
            typical_daily_volume=50,
            current_inbox_size=0,
        )

    # Create EmailState
    return EmailState(
        email=metadata,
        sender=sender_features,
        thread=thread_features,
        content=content_features,
        temporal=temporal_features,
        user_context=user_context,
        action_label=email_data.get('action'),
    )
