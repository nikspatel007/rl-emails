# Feature Extraction Enhancement Plan for Email Action Prediction

## Overview

This plan defines the feature extraction enhancement work for the rl-emails project. The goal is to build rich features that help predict what action a user will take on an email.

## User Requirements Summary

**Priority Order:**
1. **Relationship Features** - For ALL senders (human AND service)
2. **Sender Classification** - Service vs People identification
3. **Email Type Classification** - Transactional/FYI (services) vs Long-running/Short (people)
4. **Task Extraction** - Extract tasks from emails (one email can have multiple tasks)
5. **Urgency Analysis** - At both email AND task level

**LLM Strategy:** LM Studio with OpenAI 120B model locally, fallback to Claude API

**Documentation:** Per-category docs (separate file per feature type)

---

## Critical: Temporal Context & Sliding Windows

**All features must be computed within temporal context.** Importance is not static - it changes over time:

### The Problem with Static Features
- A relationship that was critical 6 months ago may be irrelevant now
- A project that dominated Q1 might be completely finished by Q3
- Today's urgent task came from a thread that started 2 weeks ago
- The user's behavior toward a sender may have changed over time

### Temporal Modeling Requirements

#### 1. Relationship Decay
Relationships have lifecycles - they emerge, strengthen, and fade:
```python
@dataclass
class TemporalRelationshipFeatures:
    # Current state (at email timestamp)
    current_relationship_strength: float  # 0-1 at this point in time

    # Trajectory
    relationship_trend: str  # 'emerging', 'stable', 'fading', 'dormant'
    days_since_peak_interaction: int
    interaction_velocity: float  # rate of change in frequency

    # Historical context
    relationship_age_days: int  # first contact to now
    peak_interaction_period: str  # "2023-Q2", when most active
    total_lifetime_emails: int

    # Decay modeling
    decay_rate: float  # how fast this relationship fades without contact
    reactivation_count: int  # times relationship went dormant then restarted
```

#### 2. Project/Topic Lifecycle
Projects and topics have beginnings, peaks, and endings:
```python
@dataclass
class TopicLifecycleFeatures:
    # Topic detection
    topic_id: str  # extracted topic/project name

    # Lifecycle stage
    lifecycle_stage: str  # 'emerging', 'active', 'winding_down', 'closed'
    topic_start_date: datetime
    topic_last_activity: datetime
    days_active: int

    # Activity pattern
    emails_in_topic_7d: int
    emails_in_topic_30d: int
    peak_activity_date: datetime
    current_vs_peak_ratio: float  # current activity / peak activity

    # Participant evolution
    participants_added_recently: int
    participants_dropped_recently: int
    core_participants: list[str]  # consistent throughout
```

#### 3. Timeline Reconstruction
Reconstruct state at any point in time for training:
```python
def get_features_at_timestamp(email_id: str, timestamp: datetime) -> Features:
    """
    Compute features AS THEY WOULD HAVE BEEN at the given timestamp.

    Critical for training: We can't use future information!
    When training on an email from 2023-06-15, we must only use
    data available on 2023-06-15, not data from 2024.
    """
    # Filter all source data to <= timestamp
    # Compute relationship stats using only historical emails
    # Compute topic stats using only historical mentions
    # Return features as of that moment
```

#### 4. Sliding Window Aggregations
Compute features over multiple time windows simultaneously:
```python
WINDOWS = [
    ('1d', timedelta(days=1)),
    ('7d', timedelta(days=7)),
    ('30d', timedelta(days=30)),
    ('90d', timedelta(days=90)),
    ('365d', timedelta(days=365)),
]

@dataclass
class SlidingWindowFeatures:
    # Per-sender stats at each window
    sender_emails_1d: int
    sender_emails_7d: int
    sender_emails_30d: int
    sender_emails_90d: int
    sender_emails_365d: int

    # Derived trends
    sender_frequency_trend: float  # (7d/30d) / (30d/90d) ratio
    sender_acceleration: float  # is frequency increasing or decreasing?

    # Topic/project windows
    topic_mentions_7d: int
    topic_mentions_30d: int
    topic_activity_trend: float

    # User behavior windows
    user_reply_rate_7d: float
    user_reply_rate_30d: float
    user_behavior_changing: bool  # significant difference in recent vs historical
```

### Implementation: Temporal Feature Store

To support timeline reconstruction efficiently, we need a temporal feature store:

```sql
-- Pre-computed daily snapshots for efficient queries
CREATE TABLE daily_relationship_snapshots (
    snapshot_date DATE,
    user_email TEXT,
    contact_email TEXT,

    -- Rolling counts (as of snapshot_date)
    emails_sent_7d INT,
    emails_sent_30d INT,
    emails_sent_90d INT,
    emails_received_7d INT,
    emails_received_30d INT,
    emails_received_90d INT,

    -- Derived metrics
    relationship_strength FLOAT,
    trend VARCHAR(20),

    PRIMARY KEY (snapshot_date, user_email, contact_email)
);

CREATE TABLE daily_topic_snapshots (
    snapshot_date DATE,
    topic_id TEXT,

    -- Activity counts
    emails_7d INT,
    emails_30d INT,
    participants_7d INT,

    -- Lifecycle
    lifecycle_stage VARCHAR(20),
    days_since_start INT,

    PRIMARY KEY (snapshot_date, topic_id)
);
```

### New Beads for Temporal Features

| Bead ID | Title | Approach | Deps |
|---------|-------|----------|------|
| `rle-TEMP-001` | Build daily relationship snapshots | SQL + Python | REL-007 |
| `rle-TEMP-002` | Implement relationship decay model | Python | TEMP-001 |
| `rle-TEMP-003` | Topic lifecycle detection | Traditional + LLM | - |
| `rle-TEMP-004` | Build daily topic snapshots | SQL + Python | TEMP-003 |
| `rle-TEMP-005` | Timeline reconstruction API | Python | TEMP-001, TEMP-004 |
| `rle-TEMP-006` | Sliding window feature aggregator | Python | TEMP-005 |

---

## Current Architecture Analysis

### Existing Feature Infrastructure
Location: `rl-emails/polecats/glory/src/features/`

| Module | Dims | Status | Coverage |
|--------|------|--------|----------|
| `relationship.py` | 11 | **NOT integrated** | Response patterns, reciprocity, hierarchy inference |
| `people.py` | 15 | Integrated | Sender importance, org level, basic automated detection |
| `task.py` | 12 | Integrated | Deadlines, action items, deliverables |
| `topic.py` | 20 | Integrated | Meeting, task, decision, FYI classification |
| `temporal.py` | 8 | Integrated | Hour, day, business hours |
| `content.py` | 384 | Integrated | Sentence-transformer embeddings |
| `project.py` | 8 | Integrated | Project mentions |
| `combined.py` | - | Orchestrator | Aggregates all features |

**Critical Gap:** `relationship.py` exists (880 lines) with good features but is NOT integrated into `combined.py` pipeline.

### Data Available
- PostgreSQL: `localhost:5433/rl_emails` with 41,373 emails
- Actions: 3,408 REPLIED, 12,025 ARCHIVED, 19,930 IGNORED
- 30 unique Gmail labels, thread information, response times

---

## Implementation Plan by Priority

### Priority 1: Relationship Features (ALL senders)

| Bead ID | Title | Approach | Deps |
|---------|-------|----------|------|
| `rle-REL-001` | Time-windowed email frequency (7d/30d/90d) | SQL + Python | - |
| `rle-REL-002` | User response history per sender | SQL + Python | REL-001 |
| `rle-REL-003` | Initiation pattern detection | Python | REL-001 |
| `rle-REL-004` | CC relationship patterns | Python | - |
| `rle-REL-005` | Thread participation depth | SQL + Python | REL-001 |
| `rle-REL-006` | Recency and reciprocity scores | Python | REL-001, REL-002 |
| `rle-REL-007` | Integrate relationship features into combined.py | Python | REL-001..006 |

**Output dataclass:**
```python
@dataclass
class RelationshipFeatures:
    emails_from_sender_7d: int
    emails_from_sender_30d: int
    emails_from_sender_90d: int
    user_replied_to_sender_rate: float
    avg_response_time_hours: float
    user_initiated_ratio: float
    cc_affinity_score: float
    avg_thread_depth: float
    days_since_last_interaction: int
    reciprocity_score: float
    relationship_momentum: float  # 30d vs 90d trend
```

**Details:**

#### rle-REL-001: Time-Windowed Email Frequency
Extend `CommunicationStats` in `relationship.py` to compute email counts within time windows:
- `emails_from_sender_7d` - Emails received from sender in last 7 days
- `emails_from_sender_30d` - Emails received from sender in last 30 days
- `emails_from_sender_90d` - Emails received from sender in last 90 days
- `emails_to_sender_7d/30d/90d` - Emails sent to sender

Implementation: PostgreSQL queries counting emails by sender within date ranges, cached in CommunicationGraph.

#### rle-REL-002: User Response History
Track how often user replies to each sender:
- `user_replied_to_sender_count` - Total replies to this sender
- `user_replied_to_sender_rate` - Replies / emails received (0.0-1.0)
- `avg_response_time_to_sender_hours` - User's typical response time
- `median_response_time_to_sender_hours`

Implementation: Join emails on thread_id and in_reply_to to identify user's responses.

#### rle-REL-003: Initiation Pattern Detection
Detect who typically starts email threads:
- `user_initiated_threads_with_sender` - Threads started by user
- `sender_initiated_threads_with_user` - Threads started by sender
- `initiation_ratio` - user_initiated / total (0.0-1.0)
- `avg_thread_depth_when_user_initiated`
- `avg_thread_depth_when_sender_initiated`

Implementation: Identify thread starters (emails with no in_reply_to), group by (user, other_party) pairs.

#### rle-REL-004: CC Relationship Patterns
Track CC co-occurrence patterns:
- `times_cc_together` - How often both appear in CC
- `times_user_cced_by_sender` - Times sender CC'd user
- `times_sender_cced_by_user` - Times user CC'd sender
- `cc_affinity_score` - Normalized 0-1 score

Implementation: Parse CC fields, build co-occurrence matrix, compute affinity scores.

#### rle-REL-005: Thread Participation Depth
Compute thread depth metrics with sender:
- `avg_thread_depth_with_sender` - Average depth of threads involving both
- `max_thread_depth_with_sender`
- `threads_participated_with_sender` - Count of shared threads
- `avg_participants_in_shared_threads`

Implementation: Query threads table for shared participation, compute depth as chain length.

#### rle-REL-006: Recency and Reciprocity Scores
Final relationship score computation:
- `days_since_last_interaction` - Recency factor
- `reciprocity_score` - Balance of sent vs received (0.5 = balanced)
- `relationship_momentum` - Trend in communication frequency (30d vs 90d)

Implementation: Combine signals from REL-001 and REL-002, compute momentum as frequency change.

#### rle-REL-007: Integration
Integrate all relationship features into `CombinedFeatures`:
- Update `RelationshipFeatures.to_feature_vector()` to include all new dimensions
- Add to `combined.py` feature aggregation
- Update `FEATURE_DIMS` constants

---

### Priority 2: Service vs People Classification

| Bead ID | Title | Approach | Deps |
|---------|-------|----------|------|
| `rle-SVC-001` | Service domain classification | Rule-based + ML | - |
| `rle-SVC-002` | Unsubscribe link detection | Regex | - |
| `rle-SVC-003` | Template pattern detection | Regex + LLM | - |
| `rle-SVC-004` | Service email type classification | Traditional + LLM | SVC-001, SVC-003 |

**Service types:**
- TRANSACTIONAL (receipts, confirmations, shipping)
- FYI (notifications, alerts, status updates)
- NEWSLETTER (content digests, daily/weekly)
- FINANCIAL (statements, bills, investment alerts)
- SOCIAL (LinkedIn, Twitter, social media notifications)
- MARKETING (promotions, sales, product announcements)
- SYSTEM (password resets, security alerts, verification)
- CALENDAR (invites, reminders, RSVPs)

**Details:**

#### rle-SVC-001: Service Domain Classification
Create service classification module:
```python
@dataclass
class ServiceClassification:
    is_service_email: bool
    service_confidence: float  # 0.0-1.0
    service_type: Optional[str]
    classification_signals: list[str]
```

Implementation:
- Build known service domain list (google.com/calendar, github.com/notifications, etc.)
- Pattern matching on sender addresses (noreply@, notifications@, no-reply@)
- Detect common service email patterns

#### rle-SVC-002: Unsubscribe Link Detection
Detect unsubscribe links in email body/headers:
- `has_unsubscribe_link` - Boolean
- `unsubscribe_type` - 'header', 'body_link', 'text_mention'
- `list_unsubscribe_header` - Value from List-Unsubscribe header

Implementation:
- Check List-Unsubscribe header (RFC 2369)
- Regex for unsubscribe links in body
- Look for "unsubscribe" text patterns

#### rle-SVC-003: Template Pattern Detection
Detect templated/automated emails:
```python
@dataclass
class TemplateSignals:
    has_template_markers: bool  # {{name}}, %RECIPIENT%, etc.
    has_tracking_pixels: bool   # 1x1 images, tracking URLs
    has_repeated_structure: bool
    personalization_score: float  # 0=fully templated, 1=fully personal
```

Implementation:
- Regex for template markers
- Image pixel detection (1x1 images, tracking URLs)
- For repeated structure: hash email structure, compare to sender history
- LLM fallback for ambiguous cases

#### rle-SVC-004: Service Email Type Classification
For identified service emails, classify type:
```python
class ServiceEmailType(Enum):
    TRANSACTIONAL = "transactional"
    FYI = "fyi"
    NEWSLETTER = "newsletter"
    FINANCIAL = "financial"
    SOCIAL = "social"
    MARKETING = "marketing"
    SYSTEM = "system"
    CALENDAR = "calendar"
```

Implementation:
- Rule-based for clear cases (calendar invites via MIME type, financial from known domains)
- LLM classification for ambiguous (LM Studio local or Claude API)

---

### Priority 3: Email Type Classification (People)

| Bead ID | Title | Approach | Deps |
|---------|-------|----------|------|
| `rle-TYPE-001` | Conversation classification for people | Traditional + LLM | REL-005 |

**Types for people emails:**
- LONG_RUNNING - Ongoing project discussions (5+ messages over 2+ weeks)
- SHORT_EXCHANGE - Quick back-and-forth (2-4 messages within hours/days)
- SINGLE_MESSAGE - No thread (standalone email)
- BROADCAST - Sent to many recipients, low interaction expected

**Details:**

#### rle-TYPE-001: Conversation Classification
Classify person-to-person email threads:
```python
class ConversationType(Enum):
    LONG_RUNNING = "long_running"
    SHORT_EXCHANGE = "short_exchange"
    SINGLE_MESSAGE = "single_message"
    BROADCAST = "broadcast"

@dataclass
class ConversationClassification:
    conversation_type: ConversationType
    thread_length: int
    unique_participants: int
    avg_message_length: int
    topic_drift_score: float  # how much subject changed
```

Implementation:
- Query thread metadata (length, participants, timespan)
- Compute participant count, message count, avg length
- LLM for topic drift detection (compare first and last message)

---

### Priority 4: Task Extraction

| Bead ID | Title | Approach | Deps |
|---------|-------|----------|------|
| `rle-TASK-001` | Multi-task extraction per email | LLM primary | - |
| `rle-TASK-002` | Enhanced deadline parsing | Traditional + LLM | - |
| `rle-TASK-003` | Task complexity estimation | LLM | TASK-001 |

**Output:**
```python
@dataclass
class ExtractedTask:
    task_id: str            # generated unique ID
    description: str        # what needs to be done
    deadline: Optional[datetime]
    deadline_text: Optional[str]  # original text ("next Tuesday")
    assignee_hint: Optional[str]  # "you", "John", etc.
    complexity: str         # 'trivial', 'quick', 'medium', 'substantial'
    task_type: str          # 'review', 'send', 'schedule', 'decision'
    urgency_score: float    # 0.0-1.0
    source_text: str        # the sentence(s) this came from
```

**Details:**

#### rle-TASK-001: Multi-Task Extraction
Extract ALL tasks from a single email (one email can have multiple tasks):

Implementation:
- Use LLM (LM Studio local) with structured output schema
- Prompt: "Extract all action items from this email. For each, identify description, deadline (if any), assignee, and task type."
- Traditional regex as fallback/validation
- Deduplicate similar tasks

Task types to detect:
- **review** - Look at document/PR/proposal
- **send** - Send email/document/update
- **schedule** - Set up meeting/call
- **decision** - Make a choice/approval
- **research** - Find information
- **create** - Write/build something
- **follow_up** - Check back on something

#### rle-TASK-002: Enhanced Deadline Parsing
Improve deadline extraction beyond existing task.py:
- Relative dates ("next Tuesday", "in two weeks", "end of month")
- Time-of-day ("by 3pm", "before the meeting", "COB")
- Implicit deadlines ("before the board meeting on Friday")
- Fuzzy deadlines ("soon", "when you can", "ASAP")

Implementation:
- Enhance regex patterns in task.py
- LLM for implicit deadline resolution
- Calendar integration hints (detect meeting references)

#### rle-TASK-003: Task Complexity Estimation
Estimate effort required for each task:
```python
class TaskComplexity(Enum):
    TRIVIAL = "trivial"      # < 5 minutes
    QUICK = "quick"          # 5-30 minutes
    MEDIUM = "medium"        # 30 min - 2 hours
    SUBSTANTIAL = "substantial"  # 2+ hours
    UNKNOWN = "unknown"

@dataclass
class ComplexityEstimate:
    complexity: TaskComplexity
    estimated_minutes: Optional[int]
    requires_research: bool
    requires_others: bool    # needs input from other people
    blocking_factor: float   # how much this blocks other work
```

Implementation:
- LLM with few-shot examples
- Keywords for complexity hints ("simple", "quick", "detailed analysis", "comprehensive")
- Factor in deliverable type (email reply vs report vs code)

---

### Priority 5: Urgency Analysis

| Bead ID | Title | Approach | Deps |
|---------|-------|----------|------|
| `rle-URG-001` | Email-level urgency scoring | Traditional + LLM | REL-006 |
| `rle-URG-002` | Task-level urgency scoring | Traditional + LLM | TASK-001, URG-001 |

**Details:**

#### rle-URG-001: Email-Level Urgency
Comprehensive email urgency scoring:
```python
@dataclass
class EmailUrgency:
    urgency_score: float        # 0.0-1.0 overall
    urgency_signals: list[str]  # what triggered the score
    urgency_source: str         # 'explicit', 'implicit', 'contextual'
    deadline_driven: bool       # urgency from deadline
    sender_driven: bool         # urgency from sender (executive/manager)
    content_driven: bool        # urgency from language
```

Email urgency signals:
- **Explicit**: Keywords ("URGENT", "ASAP", "time-sensitive", "critical")
- **Sender hierarchy**: From manager/executive (higher importance)
- **Deadline proximity**: Task deadline within 24/48 hours
- **Thread activity**: Many rapid responses indicate urgency
- **Subject markers**: "[URGENT]", "ACTION REQUIRED"

Implementation:
- Combine signals from topic.py urgency_score
- Add sender hierarchy factor from relationship features
- LLM for implicit urgency detection ("I really need this")

#### rle-URG-002: Task-Level Urgency
Per-task urgency scoring:
```python
@dataclass
class TaskUrgency:
    task_id: str
    urgency_score: float
    deadline_urgency: float     # based on time until deadline
    dependency_urgency: float   # blocks others
    explicit_urgency: float     # stated as urgent in text
    combined_urgency: float     # weighted combination
```

Implementation:
- Inherit email urgency as baseline
- Add task-specific deadline factor (urgency increases as deadline approaches)
- Add dependency/blocking factor (from task description)
- LLM for task-specific urgency hints ("this is the blocker")

---

## Dependency Graph

```
Priority 1 (Relationship):
rle-REL-001 ─┬─→ rle-REL-002 ───┐
             ├─→ rle-REL-003    │
             ├─→ rle-REL-005 ───┼──→ rle-REL-007
             └─→ rle-REL-006 ◄──┘
rle-REL-004 (standalone) ─────────→ rle-REL-007

Priority 2 (Service):
rle-SVC-001 ──┬──→ rle-SVC-004
rle-SVC-002   │
rle-SVC-003 ──┘

Priority 3 (Type):
rle-TYPE-001 ◄── rle-REL-005

Priority 4 (Task):
rle-TASK-001 ──→ rle-TASK-003
rle-TASK-002 (standalone)

Priority 5 (Urgency):
rle-URG-001 ◄── rle-REL-006
rle-URG-002 ◄── rle-TASK-001, rle-URG-001

Temporal Features (Cross-cutting):
rle-REL-007 ──→ rle-TEMP-001 ──→ rle-TEMP-002
                              └─→ rle-TEMP-005
rle-TEMP-003 ──→ rle-TEMP-004 ──→ rle-TEMP-005 ──→ rle-TEMP-006
```

---

## LLM Integration Strategy

**Primary:** LM Studio with OpenAI 120B locally
**Fallback:** Claude API

**Usage by Bead:**
| Bead | LLM Usage | Purpose |
|------|-----------|---------|
| SVC-003 | Edge cases | Template detection for ambiguous emails |
| SVC-004 | Classification | Service type when rules insufficient |
| TYPE-001 | Analysis | Topic drift detection in threads |
| TASK-001 | Core | Multi-task extraction from email text |
| TASK-002 | Enhancement | Implicit deadline resolution |
| TASK-003 | Estimation | Complexity assessment |
| URG-001 | Detection | Implicit urgency in language |
| URG-002 | Assessment | Task-specific urgency hints |

**Implementation Pattern:**
```python
from typing import TypeVar, Generic
import json

T = TypeVar('T')

class LLMExtractor(Generic[T]):
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1"):
        self.lm_studio_url = lm_studio_url
        self.claude_client = None  # lazy init

    def extract(self, email: dict, prompt_template: str, output_schema: type[T]) -> T:
        # Try LM Studio first
        try:
            result = self._call_lm_studio(prompt_template.format(**email))
            return self._parse_structured(result, output_schema)
        except Exception as e:
            # Fallback to Claude
            if self.claude_client is None:
                self.claude_client = anthropic.Anthropic()
            result = self._call_claude(prompt_template.format(**email))
            return self._parse_structured(result, output_schema)

    def _call_lm_studio(self, prompt: str) -> str:
        # OpenAI-compatible API call
        import openai
        client = openai.OpenAI(base_url=self.lm_studio_url)
        response = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def _call_claude(self, prompt: str) -> str:
        response = self.claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _parse_structured(self, text: str, schema: type[T]) -> T:
        # Parse JSON from LLM output into dataclass
        data = json.loads(text)
        return schema(**data)
```

---

## Documentation Structure

Create these per-category docs in `docs/v2/`:

| File | Content |
|------|---------|
| `docs/v2/01-relationship-features.md` | Time-windowed frequency, response patterns, reciprocity, CC analysis |
| `docs/v2/02-service-classification.md` | Service vs human detection, service type taxonomy, template detection |
| `docs/v2/03-task-extraction.md` | Multi-task extraction, deadline parsing, complexity estimation |
| `docs/v2/04-urgency-model.md` | Email-level urgency, task-level urgency, signal weighting |
| `docs/v2/05-temporal-features.md` | Sliding windows, relationship decay, topic lifecycle, timeline reconstruction |

---

## Critical Files to Modify

1. **`src/features/relationship.py`** - Extend with time-windowed stats, CC patterns
2. **`src/features/service.py`** - NEW: Service email classifier
3. **`src/features/task.py`** - Enhance for multi-task extraction
4. **`src/features/urgency.py`** - NEW: Urgency scoring at email + task level
5. **`src/features/temporal.py`** - NEW: Sliding windows, decay models, lifecycle
6. **`src/features/combined.py`** - Integrate all new features
7. **`db/dataset.py`** - Database queries for relationship analysis
8. **`scripts/create_schema.sql`** - Add temporal snapshot tables

---

## Implementation Order

**Phase 1 (Foundation):** REL-001 → REL-004 → REL-007 (integrate relationship.py)
**Phase 2 (Classification):** SVC-001 → SVC-002 → SVC-003 → SVC-004
**Phase 3 (Type):** TYPE-001
**Phase 4 (Tasks):** TASK-001 → TASK-002 → TASK-003
**Phase 5 (Urgency):** URG-001 → URG-002
**Phase 6 (Temporal):** TEMP-001 → TEMP-006 (sliding windows, decay, timeline)
**Phase 7 (Docs):** Create per-category documentation

---

## Total Beads Summary

| Priority | Beads | Count |
|----------|-------|-------|
| 1. Relationship | REL-001 to REL-007 | 7 |
| 2. Service | SVC-001 to SVC-004 | 4 |
| 3. Type | TYPE-001 | 1 |
| 4. Task | TASK-001 to TASK-003 | 3 |
| 5. Urgency | URG-001 to URG-002 | 2 |
| 6. Temporal | TEMP-001 to TEMP-006 | 6 |
| **Total** | | **23 beads** |
