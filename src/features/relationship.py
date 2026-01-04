#!/usr/bin/env python3
"""Relational priority model based on response patterns and relationships.

This module builds a priority model that normalizes response times by:
1. User's baseline response time (some people are just fast)
2. Relationship-specific baseline (respond faster to manager)
3. Content signals (deadlines, questions)

Key insight: Response time alone doesn't indicate urgency - it indicates
prioritization. We need to understand the *relative* priority a user gives
to different senders.

Output features for EmailState:
- sender_response_deviation: How much faster/slower than normal?
- sender_frequency_rank: Top 10% of senders? Bottom 50%?
- inferred_hierarchy: Manager (0.9), Peer (0.5), Report (0.3)
- relationship_strength: Based on communication volume and reciprocity
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

from email.utils import parsedate_to_datetime


# Email parsing utilities
def normalize_email(email_addr: str) -> str:
    """Extract and normalize email address from a From/To header.

    Handles formats like:
    - "John Doe <john@example.com>"
    - "john@example.com"
    - "<john@example.com>"
    """
    if not email_addr:
        return ""

    # Extract email from angle brackets
    match = re.search(r'<([^>]+)>', email_addr)
    if match:
        email_addr = match.group(1)

    # Clean up and lowercase
    email_addr = email_addr.strip().lower()

    # Remove any remaining special chars at start/end
    email_addr = email_addr.strip('<>"\'')

    return email_addr


def parse_recipients(to_field: str) -> list[str]:
    """Parse To/CC/BCC field into list of normalized email addresses."""
    if not to_field:
        return []

    # Split by comma, handling quoted names
    emails = []
    for part in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', to_field):
        normalized = normalize_email(part.strip())
        if normalized and '@' in normalized:
            emails.append(normalized)

    return emails


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse email date string to datetime."""
    if not date_str:
        return None

    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        # Try common formats
        for fmt in [
            '%a, %d %b %Y %H:%M:%S %z',
            '%d %b %Y %H:%M:%S %z',
            '%Y-%m-%d %H:%M:%S',
        ]:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
    return None


def extract_thread_id(subject: str) -> str:
    """Extract normalized thread ID from subject by removing Re:/Fwd: prefixes."""
    if not subject:
        return ""

    # Remove Re:, Fwd:, Fw: prefixes (case insensitive, multiple)
    cleaned = re.sub(r'^(?:re|fwd?|fw):\s*', '', subject.strip(), flags=re.IGNORECASE)
    while cleaned != subject:
        subject = cleaned
        cleaned = re.sub(r'^(?:re|fwd?|fw):\s*', '', subject.strip(), flags=re.IGNORECASE)

    # Normalize whitespace and lowercase
    return ' '.join(cleaned.lower().split())


@dataclass
class EmailEdge:
    """Represents a single email exchange for graph building."""
    sender: str
    recipients: list[str]
    timestamp: Optional[datetime]
    subject: str
    message_id: str
    in_reply_to: Optional[str]
    thread_id: str  # Normalized subject for matching


@dataclass
class ResponseEvent:
    """A detected response from user to sender."""
    responder: str  # Who responded
    original_sender: str  # Who they responded to
    response_time_hours: float  # Time between original and response
    thread_id: str  # Which thread
    original_timestamp: datetime
    response_timestamp: datetime


@dataclass
class CommunicationStats:
    """Statistics for communication between two parties."""
    emails_sent: int = 0
    emails_received: int = 0
    total_responses: int = 0
    response_times_hours: list[float] = field(default_factory=list)
    first_contact: Optional[datetime] = None
    last_contact: Optional[datetime] = None

    @property
    def avg_response_time_hours(self) -> Optional[float]:
        """Average response time in hours."""
        if not self.response_times_hours:
            return None
        return sum(self.response_times_hours) / len(self.response_times_hours)

    @property
    def median_response_time_hours(self) -> Optional[float]:
        """Median response time in hours."""
        if not self.response_times_hours:
            return None
        sorted_times = sorted(self.response_times_hours)
        mid = len(sorted_times) // 2
        if len(sorted_times) % 2 == 0:
            return (sorted_times[mid - 1] + sorted_times[mid]) / 2
        return sorted_times[mid]

    @property
    def communication_ratio(self) -> float:
        """Ratio of sent to received. >1 means sends more than receives."""
        if self.emails_received == 0:
            return float('inf') if self.emails_sent > 0 else 0.0
        return self.emails_sent / self.emails_received


@dataclass
class UserBaseline:
    """Per-user baseline communication patterns."""
    user_email: str
    total_emails_sent: int = 0
    total_emails_received: int = 0
    total_responses: int = 0

    # Response time statistics (in hours)
    all_response_times: list[float] = field(default_factory=list)

    # Temporal patterns
    response_times_by_hour: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    response_times_by_dow: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))

    # Relationship counts
    unique_senders: int = 0
    unique_recipients: int = 0

    @property
    def avg_response_time_hours(self) -> Optional[float]:
        """User's overall average response time."""
        if not self.all_response_times:
            return None
        return sum(self.all_response_times) / len(self.all_response_times)

    @property
    def median_response_time_hours(self) -> Optional[float]:
        """User's overall median response time."""
        if not self.all_response_times:
            return None
        sorted_times = sorted(self.all_response_times)
        mid = len(sorted_times) // 2
        if len(sorted_times) % 2 == 0:
            return (sorted_times[mid - 1] + sorted_times[mid]) / 2
        return sorted_times[mid]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'user_email': self.user_email,
            'total_emails_sent': self.total_emails_sent,
            'total_emails_received': self.total_emails_received,
            'total_responses': self.total_responses,
            'avg_response_time_hours': self.avg_response_time_hours,
            'median_response_time_hours': self.median_response_time_hours,
            'unique_senders': self.unique_senders,
            'unique_recipients': self.unique_recipients,
        }


@dataclass
class RelationshipFeatures:
    """Features extracted from relationship analysis for a single email."""
    # Core relationship features
    sender_response_deviation: float  # How much faster/slower than user's baseline
    sender_frequency_rank: float  # Percentile rank of sender frequency (0-1)
    inferred_hierarchy: float  # Manager=0.9, Peer=0.5, Report=0.3
    relationship_strength: float  # Based on communication volume and reciprocity

    # Communication patterns
    emails_from_sender_30d: int  # Volume from this sender (estimated)
    total_emails_from_sender: int
    response_rate_to_sender: float  # How often user responds to this sender
    avg_response_time_to_sender: Optional[float]  # User's typical response time to sender

    # Recency
    days_since_last_email: Optional[float]

    # Asymmetry signals
    communication_asymmetry: float  # >0 if user initiates more, <0 if sender does
    response_time_asymmetry: float  # >0 if user responds faster than sender

    def to_feature_vector(self) -> list[float]:
        """Convert to numerical vector for ML pipeline.

        Returns 11-dimensional vector.
        """
        return [
            self.sender_response_deviation,
            self.sender_frequency_rank,
            self.inferred_hierarchy,
            self.relationship_strength,
            min(self.emails_from_sender_30d, 100) / 100.0,  # Normalized
            min(self.total_emails_from_sender, 500) / 500.0,
            self.response_rate_to_sender,
            (self.avg_response_time_to_sender or 24.0) / 168.0,  # Normalize by week
            min(self.days_since_last_email or 365, 365) / 365.0,
            self.communication_asymmetry,
            self.response_time_asymmetry,
        ]


class CommunicationGraph:
    """Builds and analyzes communication patterns from email data."""

    def __init__(self):
        # Graph edges: (sender, recipient) -> CommunicationStats
        self.edges: dict[tuple[str, str], CommunicationStats] = defaultdict(CommunicationStats)

        # User baselines: user_email -> UserBaseline
        self.user_baselines: dict[str, UserBaseline] = {}

        # Message index for response matching
        self.messages_by_id: dict[str, EmailEdge] = {}
        self.messages_by_thread: dict[str, list[EmailEdge]] = defaultdict(list)

        # Response events for analysis
        self.response_events: list[ResponseEvent] = []

        # Sender frequency for ranking
        self.sender_counts: dict[str, int] = defaultdict(int)

    def add_email(self, email: dict) -> None:
        """Add an email to the communication graph."""
        sender = normalize_email(email.get('from', ''))
        if not sender:
            return

        recipients = parse_recipients(email.get('to', ''))
        cc_recipients = parse_recipients(email.get('cc', ''))
        all_recipients = recipients + cc_recipients

        timestamp = parse_date(email.get('date', ''))
        subject = email.get('subject', '')
        message_id = email.get('message_id', '')
        in_reply_to = email.get('in_reply_to', '')
        thread_id = extract_thread_id(subject)

        edge = EmailEdge(
            sender=sender,
            recipients=all_recipients,
            timestamp=timestamp,
            subject=subject,
            message_id=message_id,
            in_reply_to=in_reply_to,
            thread_id=thread_id,
        )

        # Index by message ID and thread
        if message_id:
            self.messages_by_id[message_id] = edge
        if thread_id:
            self.messages_by_thread[thread_id].append(edge)

        # Update sender counts
        self.sender_counts[sender] += 1

        # Update edge statistics
        for recipient in all_recipients:
            key = (sender, recipient)
            stats = self.edges[key]
            stats.emails_sent += 1

            if timestamp:
                if stats.first_contact is None or timestamp < stats.first_contact:
                    stats.first_contact = timestamp
                if stats.last_contact is None or timestamp > stats.last_contact:
                    stats.last_contact = timestamp

            # Update reverse edge (recipient received from sender)
            reverse_key = (recipient, sender)
            reverse_stats = self.edges[reverse_key]
            reverse_stats.emails_received += 1

    def detect_responses(self) -> None:
        """Detect response events from email threads."""
        # Method 1: Use in_reply_to field
        for msg_id, edge in self.messages_by_id.items():
            if edge.in_reply_to and edge.in_reply_to in self.messages_by_id:
                original = self.messages_by_id[edge.in_reply_to]
                if edge.timestamp and original.timestamp:
                    response_time = (edge.timestamp - original.timestamp).total_seconds() / 3600.0

                    # Only count positive response times (chronologically correct)
                    if 0 < response_time < 720:  # Max 30 days
                        event = ResponseEvent(
                            responder=edge.sender,
                            original_sender=original.sender,
                            response_time_hours=response_time,
                            thread_id=edge.thread_id,
                            original_timestamp=original.timestamp,
                            response_timestamp=edge.timestamp,
                        )
                        self.response_events.append(event)

        # Method 2: Match by thread ID (Re: subjects)
        for thread_id, messages in self.messages_by_thread.items():
            if len(messages) < 2:
                continue

            # Sort by timestamp
            sorted_msgs = [m for m in messages if m.timestamp]
            sorted_msgs.sort(key=lambda m: m.timestamp)

            # Track who needs to respond to whom
            for i in range(1, len(sorted_msgs)):
                current = sorted_msgs[i]

                # Look for previous message from different sender
                for j in range(i - 1, -1, -1):
                    prev = sorted_msgs[j]
                    if prev.sender != current.sender:
                        # Found a response - current is responding to prev
                        response_time = (current.timestamp - prev.timestamp).total_seconds() / 3600.0

                        if 0 < response_time < 720:
                            event = ResponseEvent(
                                responder=current.sender,
                                original_sender=prev.sender,
                                response_time_hours=response_time,
                                thread_id=thread_id,
                                original_timestamp=prev.timestamp,
                                response_timestamp=current.timestamp,
                            )
                            self.response_events.append(event)
                        break

        # Deduplicate response events
        seen = set()
        unique_events = []
        for event in self.response_events:
            key = (event.responder, event.original_sender, event.thread_id,
                   event.response_timestamp.isoformat() if event.response_timestamp else '')
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        self.response_events = unique_events

    def compute_baselines(self) -> None:
        """Compute per-user baseline response times."""
        # Collect all users
        all_users = set()
        for (sender, recipient) in self.edges.keys():
            all_users.add(sender)
            all_users.add(recipient)

        # Initialize baselines
        for user in all_users:
            self.user_baselines[user] = UserBaseline(user_email=user)

        # Count emails sent/received
        for (sender, recipient), stats in self.edges.items():
            if sender in self.user_baselines:
                self.user_baselines[sender].total_emails_sent += stats.emails_sent
            if recipient in self.user_baselines:
                self.user_baselines[recipient].total_emails_received += stats.emails_received

        # Process response events
        for event in self.response_events:
            if event.responder in self.user_baselines:
                baseline = self.user_baselines[event.responder]
                baseline.total_responses += 1
                baseline.all_response_times.append(event.response_time_hours)

                # Temporal patterns
                hour = event.response_timestamp.hour
                dow = event.response_timestamp.weekday()
                baseline.response_times_by_hour[hour].append(event.response_time_hours)
                baseline.response_times_by_dow[dow].append(event.response_time_hours)

            # Update edge-level response stats
            key = (event.responder, event.original_sender)
            if key in self.edges:
                stats = self.edges[key]
                stats.total_responses += 1
                stats.response_times_hours.append(event.response_time_hours)

        # Count unique contacts
        for user, baseline in self.user_baselines.items():
            senders = set()
            recipients = set()
            for (sender, recipient) in self.edges.keys():
                if recipient == user:
                    senders.add(sender)
                if sender == user:
                    recipients.add(recipient)
            baseline.unique_senders = len(senders)
            baseline.unique_recipients = len(recipients)

    def get_sender_frequency_rank(self, sender: str, user: str) -> float:
        """Get percentile rank of sender frequency for a user (0-1).

        1.0 = top sender, 0.0 = least frequent.
        """
        # Get all senders for this user
        sender_volumes = {}
        for (s, r), stats in self.edges.items():
            if r == user and stats.emails_sent > 0:
                sender_volumes[s] = stats.emails_sent

        if not sender_volumes:
            return 0.5  # Default to middle

        if sender not in sender_volumes:
            return 0.0  # Unknown sender

        # Calculate percentile
        sender_count = sender_volumes[sender]
        lower_count = sum(1 for v in sender_volumes.values() if v < sender_count)
        return lower_count / len(sender_volumes)

    def infer_hierarchy(self, sender: str, user: str) -> float:
        """Infer organizational hierarchy from communication patterns.

        Returns:
            0.9 for inferred manager (user responds faster, less volume)
            0.5 for peer (balanced communication)
            0.3 for inferred report (user responds slower, more volume to them)
        """
        key_from = (sender, user)  # sender -> user
        key_to = (user, sender)    # user -> sender

        stats_from = self.edges.get(key_from, CommunicationStats())
        stats_to = self.edges.get(key_to, CommunicationStats())

        # Get user's baseline response time
        user_baseline = self.user_baselines.get(user)
        user_avg_response = user_baseline.avg_response_time_hours if user_baseline else None

        # Get response time to this sender
        sender_response_time = stats_to.avg_response_time_hours

        # Calculate hierarchy score
        score = 0.5  # Default to peer

        if sender_response_time is not None and user_avg_response is not None:
            # If user responds faster than baseline -> sender is likely higher priority
            if sender_response_time < user_avg_response * 0.7:
                score += 0.2
            elif sender_response_time > user_avg_response * 1.5:
                score -= 0.2

        # Communication volume asymmetry
        # If sender emails user more than user emails sender -> sender is likely manager
        if stats_from.emails_sent > stats_to.emails_sent * 1.5:
            score += 0.1
        elif stats_to.emails_sent > stats_from.emails_sent * 1.5:
            score -= 0.1

        # Response rate asymmetry
        # If user responds more often to sender -> sender is likely higher priority
        sender_response_rate = stats_to.total_responses / max(stats_from.emails_sent, 1)
        if sender_response_rate > 0.6:
            score += 0.1
        elif sender_response_rate < 0.3:
            score -= 0.1

        return max(0.1, min(0.95, score))

    def compute_relationship_strength(self, sender: str, user: str) -> float:
        """Compute relationship strength based on volume and reciprocity.

        Returns 0-1 score where higher = stronger relationship.
        """
        key_from = (sender, user)
        key_to = (user, sender)

        stats_from = self.edges.get(key_from, CommunicationStats())
        stats_to = self.edges.get(key_to, CommunicationStats())

        # Total communication volume (log scale)
        total_volume = stats_from.emails_sent + stats_to.emails_sent
        volume_score = min(1.0, (1 + total_volume) ** 0.3 / 10)

        # Reciprocity (both directions active)
        if stats_from.emails_sent > 0 and stats_to.emails_sent > 0:
            reciprocity = min(stats_from.emails_sent, stats_to.emails_sent) / max(stats_from.emails_sent, stats_to.emails_sent)
        else:
            reciprocity = 0.0

        # Response engagement
        response_engagement = 0.0
        if stats_from.emails_sent > 0:
            response_engagement = stats_to.total_responses / stats_from.emails_sent

        # Combine factors
        strength = (volume_score * 0.4 + reciprocity * 0.3 + response_engagement * 0.3)
        return min(1.0, strength)

    def get_relationship_features(self, email: dict, user: str) -> RelationshipFeatures:
        """Extract relationship features for a single email and user.

        Args:
            email: Email dictionary with 'from', 'date', etc.
            user: The user's email address (the recipient)

        Returns:
            RelationshipFeatures for this email
        """
        sender = normalize_email(email.get('from', ''))
        user = normalize_email(user)
        email_date = parse_date(email.get('date', ''))

        # Get communication stats
        key_from = (sender, user)
        key_to = (user, sender)
        stats_from = self.edges.get(key_from, CommunicationStats())
        stats_to = self.edges.get(key_to, CommunicationStats())

        # Get user baseline
        user_baseline = self.user_baselines.get(user)
        user_avg_response = user_baseline.avg_response_time_hours if user_baseline else None

        # Sender response deviation
        user_response_to_sender = stats_to.avg_response_time_hours
        if user_response_to_sender is not None and user_avg_response is not None and user_avg_response > 0:
            deviation = (user_avg_response - user_response_to_sender) / user_avg_response
        else:
            deviation = 0.0

        # Response rate to sender
        if stats_from.emails_sent > 0:
            response_rate = stats_to.total_responses / stats_from.emails_sent
        else:
            response_rate = 0.0

        # Days since last email
        days_since = None
        if email_date and stats_from.last_contact:
            delta = email_date - stats_from.last_contact
            days_since = max(0, delta.total_seconds() / 86400)

        # Communication asymmetry
        if stats_from.emails_sent + stats_to.emails_sent > 0:
            asymmetry = (stats_to.emails_sent - stats_from.emails_sent) / (stats_from.emails_sent + stats_to.emails_sent)
        else:
            asymmetry = 0.0

        # Response time asymmetry
        sender_response_time = stats_from.avg_response_time_hours  # How fast sender responds to user
        if user_response_to_sender is not None and sender_response_time is not None:
            if user_response_to_sender + sender_response_time > 0:
                rt_asymmetry = (sender_response_time - user_response_to_sender) / (user_response_to_sender + sender_response_time)
            else:
                rt_asymmetry = 0.0
        else:
            rt_asymmetry = 0.0

        return RelationshipFeatures(
            sender_response_deviation=deviation,
            sender_frequency_rank=self.get_sender_frequency_rank(sender, user),
            inferred_hierarchy=self.infer_hierarchy(sender, user),
            relationship_strength=self.compute_relationship_strength(sender, user),
            emails_from_sender_30d=min(stats_from.emails_sent, 100),  # Rough estimate
            total_emails_from_sender=stats_from.emails_sent,
            response_rate_to_sender=min(1.0, response_rate),
            avg_response_time_to_sender=user_response_to_sender,
            days_since_last_email=days_since,
            communication_asymmetry=asymmetry,
            response_time_asymmetry=rt_asymmetry,
        )

    def compute_priority_score(
        self,
        email: dict,
        user: str,
        content_urgency: float = 0.5,
    ) -> float:
        """Compute priority score combining relationship and content signals.

        Formula:
            priority = content_urgency * relationship_weight * response_deviation_factor

        Args:
            email: Email dictionary
            user: User's email address
            content_urgency: 0-1 score from content analysis (deadlines, questions)

        Returns:
            Priority score 0-1
        """
        features = self.get_relationship_features(email, user)

        # Relationship weight (higher for stronger relationships + hierarchy)
        relationship_weight = (
            features.relationship_strength * 0.4 +
            features.inferred_hierarchy * 0.3 +
            features.sender_frequency_rank * 0.3
        )

        # Response deviation factor (if user typically responds fast, it's high priority)
        deviation_factor = 1.0 + features.sender_response_deviation * 0.5
        deviation_factor = max(0.5, min(1.5, deviation_factor))

        # Combine with content urgency
        priority = content_urgency * relationship_weight * deviation_factor

        return max(0.0, min(1.0, priority))

    def export_relationship_graph(self, output_path: Path) -> None:
        """Export relationship graph to JSON file."""
        graph_data = {
            'edges': [],
            'node_count': len(self.user_baselines),
            'edge_count': len(self.edges),
            'response_events_count': len(self.response_events),
        }

        for (sender, recipient), stats in self.edges.items():
            if stats.emails_sent == 0 and stats.emails_received == 0:
                continue

            graph_data['edges'].append({
                'sender': sender,
                'recipient': recipient,
                'emails_sent': stats.emails_sent,
                'emails_received': stats.emails_received,
                'total_responses': stats.total_responses,
                'avg_response_time_hours': stats.avg_response_time_hours,
                'median_response_time_hours': stats.median_response_time_hours,
                'communication_ratio': stats.communication_ratio if stats.communication_ratio != float('inf') else None,
                'first_contact': stats.first_contact.isoformat() if stats.first_contact else None,
                'last_contact': stats.last_contact.isoformat() if stats.last_contact else None,
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)

    def export_user_baselines(self, output_path: Path) -> None:
        """Export user baselines to JSON file."""
        baselines_data = {
            'user_count': len(self.user_baselines),
            'users': [baseline.to_dict() for baseline in self.user_baselines.values()],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(baselines_data, f, indent=2)


def build_communication_graph(emails: list[dict]) -> CommunicationGraph:
    """Build communication graph from list of emails.

    Args:
        emails: List of email dictionaries with fields:
            - from: sender email
            - to: recipients
            - cc: CC recipients
            - date: email date
            - subject: email subject
            - message_id: unique message ID
            - in_reply_to: ID of parent message

    Returns:
        Populated CommunicationGraph
    """
    graph = CommunicationGraph()

    print(f"Building communication graph from {len(emails)} emails...")

    for email in emails:
        graph.add_email(email)

    print("Detecting response patterns...")
    graph.detect_responses()

    print("Computing user baselines...")
    graph.compute_baselines()

    print(f"Graph built: {len(graph.user_baselines)} users, "
          f"{len(graph.edges)} edges, {len(graph.response_events)} response events")

    return graph


def process_email_dataset(
    input_path: Path,
    output_dir: Path,
    limit: Optional[int] = None,
) -> CommunicationGraph:
    """Process email dataset and generate relationship analysis.

    Args:
        input_path: Path to emails.json
        output_dir: Directory for output files
        limit: Optional limit on emails to process

    Returns:
        CommunicationGraph with analysis
    """
    print(f"Loading emails from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    if limit:
        emails = emails[:limit]
        print(f"Limited to {limit} emails")

    graph = build_communication_graph(emails)

    # Export results
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / 'relationship_graph.json'
    baselines_path = output_dir / 'user_baselines.json'

    print(f"Exporting relationship graph to {graph_path}...")
    graph.export_relationship_graph(graph_path)

    print(f"Exporting user baselines to {baselines_path}...")
    graph.export_user_baselines(baselines_path)

    return graph


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build relational priority model from email data'
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Path to emails.json'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('data'),
        help='Output directory for relationship files (default: data)'
    )
    parser.add_argument(
        '-n', '--limit',
        type=int,
        help='Limit number of emails to process'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        exit(1)

    graph = process_email_dataset(args.input, args.output_dir, args.limit)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RELATIONSHIP ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total users: {len(graph.user_baselines)}")
    print(f"Total edges: {len(graph.edges)}")
    print(f"Total response events: {len(graph.response_events)}")

    # Top 10 most active users
    print("\nTop 10 most active users (by emails sent):")
    sorted_users = sorted(
        graph.user_baselines.values(),
        key=lambda u: u.total_emails_sent,
        reverse=True
    )[:10]
    for user in sorted_users:
        avg_rt = f"{user.avg_response_time_hours:.1f}h" if user.avg_response_time_hours else "N/A"
        print(f"  {user.user_email}: {user.total_emails_sent} sent, "
              f"{user.total_emails_received} received, avg response: {avg_rt}")

    print("=" * 60)
