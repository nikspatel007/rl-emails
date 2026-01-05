#!/usr/bin/env python3
"""Human verification UI for LLM-extracted email features.

A Streamlit-based interface for verifying and correcting LLM-extracted
features. Shows 3-column layout: LLM Extracted | AI Predicted | Human Verified.

Usage:
    # Start PostgreSQL first
    ./scripts/start_db.sh

    # Run the UI
    streamlit run scripts/labeling_ui_v2.py

    # With custom labeler name
    streamlit run scripts/labeling_ui_v2.py -- --labeler "nik"

Features:
    - 3-column comparison layout
    - One-click 'All Correct' button
    - Keyboard shortcuts (j/k/y/n)
    - Progress tracking with AI accuracy %
    - Confidence-based filtering
    - Collapsible email content
"""

import argparse
import asyncio
import sys
from datetime import datetime
from typing import Optional

import asyncpg
import streamlit as st

# Database configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Field options for corrections
SERVICE_TYPES = ["notification", "marketing", "transactional", "newsletter", "system", "personal", None]
TOPIC_CATEGORIES = ["project", "admin", "meeting", "task", "social", "notification", "spam", "other"]
TASK_TYPES = ["review", "send", "schedule", "decision", "research", "create", "follow_up", "other"]
COMPLEXITY_LEVELS = ["trivial", "quick", "medium", "substantial", "unknown"]

# CSS for the verification UI
VERIFICATION_CSS = """
<style>
/* Main container */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
}

/* Email card */
.email-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.email-header {
    border-bottom: 1px solid #f0f0f0;
    padding-bottom: 12px;
    margin-bottom: 16px;
}

.email-subject {
    font-weight: 600;
    font-size: 1.1em;
    color: #1a1a1a;
    margin-bottom: 8px;
}

.email-meta {
    color: #666;
    font-size: 0.85em;
    line-height: 1.6;
}

.email-body {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    font-family: -apple-system, system-ui, sans-serif;
    font-size: 0.9em;
    line-height: 1.6;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* Column headers */
.col-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 16px;
}

.col-header-llm {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.col-header-ai {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.col-header-human {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

/* Feature cards */
.feature-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}

.feature-label {
    font-size: 0.75em;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.feature-value {
    font-weight: 500;
    color: #1a1a1a;
}

.feature-value-service {
    color: #e53e3e;
}

.feature-value-personal {
    color: #38a169;
}

/* Task list */
.task-item {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 8px;
}

.task-description {
    font-weight: 500;
    margin-bottom: 4px;
}

.task-meta {
    font-size: 0.8em;
    color: #666;
}

/* Score badges */
.score-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
}

.score-high {
    background: #c6f6d5;
    color: #22543d;
}

.score-medium {
    background: #fefcbf;
    color: #744210;
}

.score-low {
    background: #fed7d7;
    color: #742a2a;
}

/* Action buttons */
.action-btn {
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-correct {
    background: #48bb78;
    color: white;
    border: none;
}

.btn-correct:hover {
    background: #38a169;
}

.btn-fix {
    background: #ed8936;
    color: white;
    border: none;
}

/* Progress bar */
.progress-container {
    background: #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    height: 24px;
    margin: 16px 0;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 0.85em;
}

/* Stats cards */
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.stat-value {
    font-size: 2em;
    font-weight: 700;
    color: #2d3748;
}

.stat-label {
    font-size: 0.85em;
    color: #718096;
    margin-top: 4px;
}

/* Keyboard shortcuts hint */
.shortcuts-hint {
    background: #edf2f7;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.85em;
    color: #4a5568;
}

.kbd {
    background: white;
    border: 1px solid #cbd5e0;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: monospace;
    font-size: 0.9em;
}

/* Match/mismatch indicators */
.match-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

.match-yes {
    background: #48bb78;
}

.match-no {
    background: #f56565;
}

.match-pending {
    background: #ed8936;
}
</style>
"""


class VerificationStore:
    """Store human verifications in PostgreSQL."""

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url

    async def ensure_table(self):
        """Create human_verifications table if it doesn't exist."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS human_verifications (
                        id SERIAL PRIMARY KEY,
                        email_id INTEGER REFERENCES emails(id),

                        -- LLM extraction fields (what was extracted)
                        llm_is_service_email BOOLEAN,
                        llm_service_type TEXT,
                        llm_topic_category TEXT,
                        llm_overall_urgency FLOAT,
                        llm_requires_response BOOLEAN,
                        llm_summary TEXT,
                        llm_tasks JSONB,

                        -- Human verified values
                        verified_is_service_email BOOLEAN,
                        verified_service_type TEXT,
                        verified_topic_category TEXT,
                        verified_overall_urgency FLOAT,
                        verified_requires_response BOOLEAN,
                        verified_tasks JSONB,

                        -- Verification metadata
                        all_correct BOOLEAN DEFAULT FALSE,
                        fields_corrected TEXT[],
                        labeler TEXT,
                        verification_time_seconds INTEGER,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                ''')
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_human_verifications_email_id
                    ON human_verifications(email_id)
                ''')
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Error creating table: {e}")

    async def save_verification(
        self,
        email_id: int,
        llm_data: dict,
        verified_data: dict,
        all_correct: bool,
        fields_corrected: list[str],
        labeler: str,
        verification_time_seconds: int,
    ) -> bool:
        """Save a verification to the database."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                import json
                await conn.execute(
                    '''
                    INSERT INTO human_verifications (
                        email_id,
                        llm_is_service_email, llm_service_type, llm_topic_category,
                        llm_overall_urgency, llm_requires_response, llm_summary, llm_tasks,
                        verified_is_service_email, verified_service_type, verified_topic_category,
                        verified_overall_urgency, verified_requires_response, verified_tasks,
                        all_correct, fields_corrected, labeler, verification_time_seconds
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8,
                        $9, $10, $11, $12, $13, $14,
                        $15, $16, $17, $18
                    )
                    ''',
                    email_id,
                    llm_data.get('is_service_email'),
                    llm_data.get('service_type'),
                    llm_data.get('topic_category'),
                    llm_data.get('overall_urgency'),
                    llm_data.get('requires_response'),
                    llm_data.get('summary'),
                    json.dumps(llm_data.get('tasks', [])),
                    verified_data.get('is_service_email'),
                    verified_data.get('service_type'),
                    verified_data.get('topic_category'),
                    verified_data.get('overall_urgency'),
                    verified_data.get('requires_response'),
                    json.dumps(verified_data.get('tasks', [])),
                    all_correct,
                    fields_corrected,
                    labeler,
                    verification_time_seconds,
                )
                return True
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Error saving verification: {e}")
            return False

    async def get_verified_email_ids(self) -> set[int]:
        """Get IDs of emails that have been verified."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                rows = await conn.fetch('SELECT DISTINCT email_id FROM human_verifications')
                return {row['email_id'] for row in rows}
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Error loading verified IDs: {e}")
            return set()

    async def get_stats(self) -> dict:
        """Get verification statistics."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                row = await conn.fetchrow('''
                    SELECT
                        COUNT(*) as total_verified,
                        COUNT(*) FILTER (WHERE all_correct = true) as all_correct_count,
                        AVG(verification_time_seconds) as avg_time_seconds
                    FROM human_verifications
                ''')

                # Get field-level accuracy
                field_stats = await conn.fetchrow('''
                    SELECT
                        COUNT(*) FILTER (WHERE llm_is_service_email = verified_is_service_email) as service_matches,
                        COUNT(*) FILTER (WHERE llm_service_type = verified_service_type) as service_type_matches,
                        COUNT(*) FILTER (WHERE llm_topic_category = verified_topic_category) as topic_matches,
                        COUNT(*) as total
                    FROM human_verifications
                ''')

                result = dict(row) if row else {}
                if field_stats and field_stats['total'] > 0:
                    total = field_stats['total']
                    result['service_accuracy'] = field_stats['service_matches'] / total
                    result['service_type_accuracy'] = field_stats['service_type_matches'] / total
                    result['topic_accuracy'] = field_stats['topic_matches'] / total

                return result
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Error loading stats: {e}")
            return {}


class EmailDataLoader:
    """Load emails with LLM features for verification."""

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self._cache = {}

    async def load_email_with_features(self, email_id: int) -> Optional[dict]:
        """Load email with its LLM-extracted and ML features."""
        if email_id in self._cache:
            return self._cache[email_id]

        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                # Get email
                email = await conn.fetchrow('''
                    SELECT id, message_id, subject, body_text, from_email, to_emails,
                           date_parsed, labels, action
                    FROM emails WHERE id = $1
                ''', email_id)

                if not email:
                    return None

                # Get ML features
                features = await conn.fetchrow('''
                    SELECT project_score, topic_score, task_score, people_score,
                           temporal_score, service_score, relationship_score,
                           overall_priority, is_service_email, service_type,
                           urgency_score
                    FROM email_features WHERE email_id = $1
                ''', email_id)

                # Get extracted tasks
                tasks = await conn.fetch('''
                    SELECT description, deadline_text, assignee_hint, task_type,
                           complexity, urgency_score
                    FROM tasks WHERE email_id = $1
                ''', email_id)

                result = {
                    'email': dict(email),
                    'ml_features': dict(features) if features else {},
                    'extracted_tasks': [dict(t) for t in tasks],
                }
                self._cache[email_id] = result
                return result

            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
            return None

    async def get_emails_for_verification(
        self,
        verified_ids: set[int],
        limit: int = 50,
        sort_by_confidence: bool = True,
    ) -> list[dict]:
        """Get emails that need verification, optionally sorted by confidence."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                # Get emails with features that haven't been verified
                query = '''
                    SELECT e.id, e.subject, ef.overall_priority,
                           ef.service_email_confidence, ef.is_service_email
                    FROM emails e
                    LEFT JOIN email_features ef ON ef.email_id = e.id
                    WHERE e.body_text IS NOT NULL
                      AND length(e.body_text) > 50
                '''

                if verified_ids:
                    query += f" AND e.id NOT IN ({','.join(str(i) for i in verified_ids)})"

                if sort_by_confidence:
                    # Lower confidence first (needs more human review)
                    query += " ORDER BY COALESCE(ef.service_email_confidence, 0.5) ASC"
                else:
                    query += " ORDER BY e.date_parsed DESC NULLS LAST"

                query += f" LIMIT {limit}"

                rows = await conn.fetch(query)
                return [dict(r) for r in rows]

            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
            return []


def render_score_badge(score: Optional[float], label: str) -> str:
    """Render a score as a colored badge."""
    if score is None:
        return f'<span class="score-badge score-low">{label}: N/A</span>'

    if score >= 0.7:
        css_class = "score-high"
    elif score >= 0.4:
        css_class = "score-medium"
    else:
        css_class = "score-low"

    return f'<span class="score-badge {css_class}">{label}: {score:.0%}</span>'


def render_task_card(task: dict, index: int) -> str:
    """Render a task as an HTML card."""
    desc = task.get('description', 'No description')
    task_type = task.get('task_type', 'other')
    urgency = task.get('urgency_score') or task.get('urgency', 0)
    deadline = task.get('deadline_text') or task.get('deadline', '')

    return f'''
    <div class="task-item">
        <div class="task-description">{index + 1}. {desc}</div>
        <div class="task-meta">
            Type: {task_type} | Urgency: {urgency:.0%}
            {f" | Deadline: {deadline}" if deadline else ""}
        </div>
    </div>
    '''


def init_session_state(labeler: str):
    """Initialize Streamlit session state."""
    if 'loader' not in st.session_state:
        st.session_state.loader = EmailDataLoader()

    if 'store' not in st.session_state:
        st.session_state.store = VerificationStore()

    if 'labeler' not in st.session_state:
        st.session_state.labeler = labeler

    if 'verified_ids' not in st.session_state:
        st.session_state.verified_ids = set()

    if 'email_queue' not in st.session_state:
        st.session_state.email_queue = []

    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    if 'current_email' not in st.session_state:
        st.session_state.current_email = None

    if 'editing_field' not in st.session_state:
        st.session_state.editing_field = None

    if 'corrections' not in st.session_state:
        st.session_state.corrections = {}

    if 'start_time' not in st.session_state:
        st.session_state.start_time = None

    if 'stats' not in st.session_state:
        st.session_state.stats = {}

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if 'show_email_body' not in st.session_state:
        st.session_state.show_email_body = False

    if 'filter_low_confidence' not in st.session_state:
        st.session_state.filter_low_confidence = True


async def initialize_data():
    """Initialize data from the database."""
    store = st.session_state.store
    await store.ensure_table()

    st.session_state.verified_ids = await store.get_verified_ids()
    st.session_state.stats = await store.get_stats()

    # Load email queue
    loader = st.session_state.loader
    st.session_state.email_queue = await loader.get_emails_for_verification(
        st.session_state.verified_ids,
        limit=100,
        sort_by_confidence=st.session_state.filter_low_confidence,
    )


async def load_current_email():
    """Load the current email for verification."""
    if not st.session_state.email_queue:
        st.session_state.current_email = None
        return

    if st.session_state.current_index >= len(st.session_state.email_queue):
        st.session_state.current_index = 0

    email_info = st.session_state.email_queue[st.session_state.current_index]
    loader = st.session_state.loader
    st.session_state.current_email = await loader.load_email_with_features(email_info['id'])
    st.session_state.start_time = datetime.now()
    st.session_state.corrections = {}
    st.session_state.editing_field = None


async def save_and_next(all_correct: bool):
    """Save verification and move to next email."""
    email_data = st.session_state.current_email
    if not email_data:
        return

    email = email_data['email']
    ml_features = email_data.get('ml_features', {})
    tasks = email_data.get('extracted_tasks', [])

    # Build LLM data from what was extracted
    llm_data = {
        'is_service_email': ml_features.get('is_service_email', False),
        'service_type': ml_features.get('service_type'),
        'topic_category': None,  # From LLM extraction
        'overall_urgency': ml_features.get('urgency_score', 0.0),
        'requires_response': False,
        'summary': None,
        'tasks': tasks,
    }

    # Build verified data
    corrections = st.session_state.corrections
    if all_correct:
        verified_data = llm_data.copy()
        fields_corrected = []
    else:
        verified_data = {
            'is_service_email': corrections.get('is_service_email', llm_data['is_service_email']),
            'service_type': corrections.get('service_type', llm_data['service_type']),
            'topic_category': corrections.get('topic_category', llm_data['topic_category']),
            'overall_urgency': corrections.get('overall_urgency', llm_data['overall_urgency']),
            'requires_response': corrections.get('requires_response', llm_data['requires_response']),
            'tasks': corrections.get('tasks', llm_data['tasks']),
        }
        fields_corrected = list(corrections.keys())

    # Calculate verification time
    time_seconds = int((datetime.now() - st.session_state.start_time).total_seconds())

    # Save
    store = st.session_state.store
    success = await store.save_verification(
        email_id=email['id'],
        llm_data=llm_data,
        verified_data=verified_data,
        all_correct=all_correct,
        fields_corrected=fields_corrected,
        labeler=st.session_state.labeler,
        verification_time_seconds=time_seconds,
    )

    if success:
        st.session_state.verified_ids.add(email['id'])
        st.session_state.current_index += 1

        # Update stats
        total = st.session_state.stats.get('total_verified', 0) + 1
        correct = st.session_state.stats.get('all_correct_count', 0) + (1 if all_correct else 0)
        st.session_state.stats['total_verified'] = total
        st.session_state.stats['all_correct_count'] = correct

        await load_current_email()


def main():
    st.set_page_config(
        page_title="Email Verification UI",
        page_icon="✅",
        layout="wide"
    )

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeler', type=str, default='anonymous')

    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args([])

    # Initialize
    init_session_state(args.labeler)

    if not st.session_state.initialized:
        asyncio.run(initialize_data())
        asyncio.run(load_current_email())
        st.session_state.initialized = True

    # Custom CSS
    st.markdown(VERIFICATION_CSS, unsafe_allow_html=True)

    # Header
    st.title("✅ Email Feature Verification")

    # Stats row
    stats = st.session_state.stats
    total = stats.get('total_verified', 0)
    correct = stats.get('all_correct_count', 0)
    accuracy = (correct / total * 100) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Verified", total)
    with col2:
        st.metric("AI Correct", f"{accuracy:.1f}%")
    with col3:
        st.metric("In Queue", len(st.session_state.email_queue) - st.session_state.current_index)
    with col4:
        st.metric("Labeler", st.session_state.labeler)

    st.divider()

    # Keyboard shortcuts hint
    st.markdown('''
    <div class="shortcuts-hint">
        <strong>Keyboard:</strong>
        <span class="kbd">Y</span> All Correct |
        <span class="kbd">N</span> Needs Fix |
        <span class="kbd">J</span> Next |
        <span class="kbd">K</span> Previous |
        <span class="kbd">E</span> Toggle Email
    </div>
    ''', unsafe_allow_html=True)

    # Current email
    email_data = st.session_state.current_email
    if not email_data:
        st.success("All emails have been verified!")
        return

    email = email_data['email']
    ml_features = email_data.get('ml_features', {})
    tasks = email_data.get('extracted_tasks', [])

    # Email header
    st.markdown(f'''
    <div class="email-card">
        <div class="email-header">
            <div class="email-subject">{email.get('subject', '(No subject)')}</div>
            <div class="email-meta">
                <strong>From:</strong> {email.get('from_email', 'Unknown')} |
                <strong>Date:</strong> {email.get('date_parsed', 'Unknown')}
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Collapsible email body
    with st.expander("📧 Email Body", expanded=st.session_state.show_email_body):
        body = email.get('body_text', '')[:1500]
        if len(email.get('body_text', '')) > 1500:
            body += '...'
        st.text(body)

    st.divider()

    # 3-column layout
    col_llm, col_ai, col_human = st.columns(3)

    with col_llm:
        st.markdown('<div class="col-header col-header-llm">LLM Extracted</div>', unsafe_allow_html=True)

        # Service email
        is_service = ml_features.get('is_service_email', False)
        st.markdown(f'''
        <div class="feature-card">
            <div class="feature-label">Service Email</div>
            <div class="feature-value {'feature-value-service' if is_service else 'feature-value-personal'}">
                {'Yes' if is_service else 'No'}
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Service type
        service_type = ml_features.get('service_type', 'N/A')
        st.markdown(f'''
        <div class="feature-card">
            <div class="feature-label">Service Type</div>
            <div class="feature-value">{service_type or 'N/A'}</div>
        </div>
        ''', unsafe_allow_html=True)

        # Tasks
        st.markdown('<div class="feature-card"><div class="feature-label">Extracted Tasks</div></div>', unsafe_allow_html=True)
        if tasks:
            for i, task in enumerate(tasks):
                st.markdown(render_task_card(task, i), unsafe_allow_html=True)
        else:
            st.caption("No tasks extracted")

    with col_ai:
        st.markdown('<div class="col-header col-header-ai">AI Predicted</div>', unsafe_allow_html=True)

        # Priority score
        priority = ml_features.get('overall_priority', 0)
        st.markdown(render_score_badge(priority, 'Priority'), unsafe_allow_html=True)

        # Other scores
        scores = [
            ('People', ml_features.get('people_score')),
            ('Project', ml_features.get('project_score')),
            ('Task', ml_features.get('task_score')),
            ('Urgency', ml_features.get('urgency_score')),
        ]
        for label, score in scores:
            st.markdown(render_score_badge(score, label), unsafe_allow_html=True)

    with col_human:
        st.markdown('<div class="col-header col-header-human">Human Verified</div>', unsafe_allow_html=True)

        # Service email correction
        is_service_current = st.session_state.corrections.get(
            'is_service_email', ml_features.get('is_service_email', False)
        )
        verified_service = st.checkbox(
            "Is Service Email?",
            value=is_service_current,
            key="verify_service"
        )
        if verified_service != ml_features.get('is_service_email'):
            st.session_state.corrections['is_service_email'] = verified_service

        # Service type correction
        current_type = st.session_state.corrections.get(
            'service_type', ml_features.get('service_type')
        )
        type_options = SERVICE_TYPES
        type_index = type_options.index(current_type) if current_type in type_options else 0
        verified_type = st.selectbox(
            "Service Type",
            options=type_options,
            index=type_index,
            key="verify_type",
            format_func=lambda x: x if x else "None"
        )
        if verified_type != ml_features.get('service_type'):
            st.session_state.corrections['service_type'] = verified_type

        # Topic category
        verified_topic = st.selectbox(
            "Topic Category",
            options=TOPIC_CATEGORIES,
            key="verify_topic"
        )
        st.session_state.corrections['topic_category'] = verified_topic

        # Urgency
        current_urgency = st.session_state.corrections.get(
            'overall_urgency', ml_features.get('urgency_score', 0.5)
        ) or 0.5
        verified_urgency = st.slider(
            "Urgency",
            min_value=0.0,
            max_value=1.0,
            value=float(current_urgency),
            step=0.1,
            key="verify_urgency"
        )
        if abs(verified_urgency - (ml_features.get('urgency_score', 0.5) or 0.5)) > 0.05:
            st.session_state.corrections['overall_urgency'] = verified_urgency

    st.divider()

    # Action buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 2, 1, 1])

    with btn_col1:
        if st.button("✅ All Correct", type="primary", use_container_width=True):
            asyncio.run(save_and_next(all_correct=True))
            st.rerun()

    with btn_col2:
        if st.button("💾 Save Corrections", use_container_width=True):
            asyncio.run(save_and_next(all_correct=False))
            st.rerun()

    with btn_col3:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                asyncio.run(load_current_email())
                st.rerun()

    with btn_col4:
        if st.button("➡️ Skip", use_container_width=True):
            st.session_state.current_index += 1
            asyncio.run(load_current_email())
            st.rerun()

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        st.session_state.filter_low_confidence = st.checkbox(
            "Show low confidence first",
            value=st.session_state.filter_low_confidence
        )

        if st.button("Refresh Queue"):
            asyncio.run(initialize_data())
            asyncio.run(load_current_email())
            st.rerun()

        st.divider()
        st.header("Accuracy by Field")

        if 'service_accuracy' in stats:
            st.metric("Service Email", f"{stats['service_accuracy']:.1%}")
        if 'service_type_accuracy' in stats:
            st.metric("Service Type", f"{stats['service_type_accuracy']:.1%}")
        if 'topic_accuracy' in stats:
            st.metric("Topic Category", f"{stats['topic_accuracy']:.1%}")


if __name__ == '__main__':
    main()
