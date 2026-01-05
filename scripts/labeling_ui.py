#!/usr/bin/env python3
"""Human preference labeling UI for email prioritization.

A Streamlit-based interface for collecting human preference judgments on email
pairs. Supports smart sampling to focus human effort on uncertain/edge cases.

Usage:
    # Start PostgreSQL first
    ./scripts/start_db.sh

    # Run the UI
    streamlit run scripts/labeling_ui.py

    # With custom labeler name
    streamlit run scripts/labeling_ui.py -- --labeler "nik"
"""

import argparse
import asyncio
import random
import sys
from typing import Optional

import asyncpg
import streamlit as st

# Database configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"


# Preference labels
PREFERENCE_LEFT = "left"
PREFERENCE_RIGHT = "right"
PREFERENCE_SAME = "same"
PREFERENCE_SKIP = "skip"

# Confidence levels
CONFIDENCE_CERTAIN = "certain"
CONFIDENCE_UNSURE = "unsure"

# CSS for better email display
EMAIL_CSS = """
<style>
.email-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background: #fafafa;
}
.email-header {
    border-bottom: 1px solid #eee;
    padding-bottom: 8px;
    margin-bottom: 12px;
}
.email-subject {
    font-weight: bold;
    font-size: 1.1em;
    margin-bottom: 4px;
}
.email-meta {
    color: #666;
    font-size: 0.9em;
}
.email-body {
    white-space: pre-wrap;
    font-family: -apple-system, system-ui, sans-serif;
    line-height: 1.5;
    max-height: 300px;
    overflow-y: auto;
}
.email-labels {
    margin-top: 8px;
}
.label-tag {
    display: inline-block;
    background: #e0e0e0;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    margin-right: 4px;
}
.label-starred {
    background: #ffd700;
    color: #333;
}
.label-important {
    background: #ff9800;
    color: white;
}
.selected-card {
    border: 3px solid #4CAF50;
    background: #f0fff0;
}
</style>
"""


class EmailLoader:
    """Load emails from PostgreSQL for labeling."""

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self._emails_cache = {}
        self._id_cache = {}  # message_id -> id mapping

    async def load_email(self, email_id: int) -> Optional[dict]:
        """Load a single email by id."""
        if email_id in self._emails_cache:
            return self._emails_cache[email_id]

        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                row = await conn.fetchrow(
                    '''
                    SELECT
                        id,
                        message_id,
                        subject,
                        body_text as body,
                        from_email,
                        to_emails,
                        date_parsed,
                        labels,
                        action
                    FROM emails
                    WHERE id = $1
                    ''',
                    email_id
                )

                if row:
                    email = dict(row)
                    # Format date for display
                    if email.get('date_parsed'):
                        email['date_str'] = email['date_parsed'].strftime('%Y-%m-%d %H:%M')
                    self._emails_cache[email_id] = email
                    return email

                return None
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
            return None

    async def load_emails_batch(self, email_ids: list[int]) -> dict[int, dict]:
        """Load multiple emails efficiently."""
        # Check cache first
        missing = [eid for eid in email_ids if eid not in self._emails_cache]

        if missing:
            try:
                conn = await asyncpg.connect(self.db_url)
                try:
                    rows = await conn.fetch(
                        '''
                        SELECT
                            id,
                            message_id,
                            subject,
                            body_text as body,
                            from_email,
                            to_emails,
                            date_parsed,
                            labels,
                            action
                        FROM emails
                        WHERE id = ANY($1)
                        ''',
                        missing
                    )

                    for row in rows:
                        email = dict(row)
                        if email.get('date_parsed'):
                            email['date_str'] = email['date_parsed'].strftime('%Y-%m-%d %H:%M')
                        self._emails_cache[email['id']] = email
                finally:
                    await conn.close()
            except Exception as e:
                st.error(f"Database error: {e}")

        return {eid: self._emails_cache.get(eid) for eid in email_ids}

    async def get_email_count(self) -> int:
        """Get total number of emails in the database."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                count = await conn.fetchval('SELECT COUNT(*) FROM emails')
                return count or 0
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
            return 0

    async def get_random_pair(self, exclude_ids: set[tuple[int, int]]) -> Optional[tuple[int, int]]:
        """Get a random pair of email IDs for comparison, excluding already labeled pairs."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                # Get two random emails with decent content (not service emails)
                rows = await conn.fetch(
                    '''
                    SELECT e.id
                    FROM emails e
                    LEFT JOIN email_features ef ON ef.email_id = e.id
                    WHERE e.body_text IS NOT NULL
                      AND length(e.body_text) > 50
                      AND (ef.is_service_email IS NULL OR ef.is_service_email = false)
                    ORDER BY RANDOM()
                    LIMIT 100
                    '''
                )

                if len(rows) < 2:
                    return None

                # Find a pair not in exclude_ids
                ids = [r['id'] for r in rows]
                random.shuffle(ids)

                for i in range(len(ids)):
                    for j in range(i + 1, min(i + 20, len(ids))):
                        pair = (ids[i], ids[j])
                        reverse_pair = (ids[j], ids[i])
                        if pair not in exclude_ids and reverse_pair not in exclude_ids:
                            return pair

                return None
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
            return None


class PreferenceStore:
    """Store human preferences in PostgreSQL."""

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url

    async def ensure_table(self):
        """Create human_preferences table if it doesn't exist."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS human_preferences (
                        id SERIAL PRIMARY KEY,
                        email_left_id INTEGER REFERENCES emails(id),
                        email_right_id INTEGER REFERENCES emails(id),
                        preference TEXT CHECK (preference IN ('left', 'right', 'same', 'skip')),
                        confidence TEXT CHECK (confidence IN ('certain', 'unsure')),
                        labeler TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                ''')
                # Create index for faster duplicate checking
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_human_preferences_pair
                    ON human_preferences(email_left_id, email_right_id)
                ''')
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Error creating table: {e}")

    async def save_preference(
        self,
        email_left_id: int,
        email_right_id: int,
        preference: str,
        confidence: str,
        labeler: str,
    ) -> bool:
        """Save a single preference to the database."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute(
                    '''
                    INSERT INTO human_preferences
                        (email_left_id, email_right_id, preference, confidence, labeler)
                    VALUES ($1, $2, $3, $4, $5)
                    ''',
                    email_left_id, email_right_id, preference, confidence, labeler
                )
                return True
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Error saving preference: {e}")
            return False

    async def get_labeled_pairs(self) -> set[tuple[int, int]]:
        """Get all pairs that have already been labeled."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                rows = await conn.fetch(
                    'SELECT email_left_id, email_right_id FROM human_preferences'
                )
                pairs = set()
                for row in rows:
                    pairs.add((row['email_left_id'], row['email_right_id']))
                    # Also add reverse to avoid duplicate comparisons
                    pairs.add((row['email_right_id'], row['email_left_id']))
                return pairs
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Error loading labeled pairs: {e}")
            return set()

    async def get_stats(self) -> dict:
        """Get labeling statistics."""
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                row = await conn.fetchrow('''
                    SELECT
                        COUNT(*) as total_labeled,
                        COUNT(*) FILTER (WHERE preference = 'left') as left_chosen,
                        COUNT(*) FILTER (WHERE preference = 'right') as right_chosen,
                        COUNT(*) FILTER (WHERE preference = 'same') as same,
                        COUNT(*) FILTER (WHERE preference = 'skip') as skipped,
                        COUNT(*) FILTER (WHERE confidence = 'certain') as certain,
                        COUNT(*) FILTER (WHERE confidence = 'unsure') as unsure
                    FROM human_preferences
                ''')
                return dict(row) if row else {}
            finally:
                await conn.close()
        except Exception as e:
            st.error(f"Error loading stats: {e}")
            return {}


def render_email_card(email: dict, side: str, selected: bool = False) -> str:
    """Render an email as an HTML card."""
    if not email:
        return "<div class='email-card'>Email not found</div>"

    subject = email.get('subject', '(No subject)')
    from_email = email.get('from_email', 'Unknown')
    to_emails = email.get('to_emails', [])
    to_str = ', '.join(to_emails[:3]) + ('...' if len(to_emails) > 3 else '')
    date_str = email.get('date_str', '')
    body = email.get('body', '')[:500]  # Truncate for display
    if len(email.get('body', '')) > 500:
        body += '...'
    labels = email.get('labels', [])
    action = email.get('action', '')

    # Escape HTML
    import html
    subject = html.escape(subject)
    from_email = html.escape(from_email)
    to_str = html.escape(to_str)
    body = html.escape(body)

    # Build labels HTML
    labels_html = ""
    for label in labels[:5]:
        cls = "label-tag"
        if 'STARRED' in label.upper():
            cls += " label-starred"
        elif 'IMPORTANT' in label.upper():
            cls += " label-important"
        labels_html += f"<span class='{cls}'>{html.escape(label)}</span>"

    if action:
        labels_html += f"<span class='label-tag'>Action: {html.escape(action)}</span>"

    card_class = "email-card selected-card" if selected else "email-card"

    return f"""
    <div class='{card_class}'>
        <div class='email-header'>
            <div class='email-subject'>{subject}</div>
            <div class='email-meta'>
                <strong>From:</strong> {from_email}<br>
                <strong>To:</strong> {to_str}<br>
                <strong>Date:</strong> {date_str}
            </div>
        </div>
        <div class='email-body'>{body}</div>
        <div class='email-labels'>{labels_html}</div>
    </div>
    """


def init_session_state(labeler: str):
    """Initialize Streamlit session state."""
    if 'loader' not in st.session_state:
        st.session_state.loader = EmailLoader()

    if 'store' not in st.session_state:
        st.session_state.store = PreferenceStore()

    if 'labeler' not in st.session_state:
        st.session_state.labeler = labeler

    if 'labeled_pairs' not in st.session_state:
        st.session_state.labeled_pairs = set()

    if 'current_pair' not in st.session_state:
        st.session_state.current_pair = None

    if 'current_left' not in st.session_state:
        st.session_state.current_left = None

    if 'current_right' not in st.session_state:
        st.session_state.current_right = None

    if 'selection' not in st.session_state:
        st.session_state.selection = None

    if 'confidence' not in st.session_state:
        st.session_state.confidence = CONFIDENCE_CERTAIN

    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_labeled': 0,
            'left_chosen': 0,
            'right_chosen': 0,
            'same': 0,
            'skipped': 0,
            'certain': 0,
            'unsure': 0,
        }

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False


async def initialize_data():
    """Initialize data from the database."""
    store = st.session_state.store
    await store.ensure_table()

    # Load existing labeled pairs
    st.session_state.labeled_pairs = await store.get_labeled_pairs()

    # Load stats
    stats = await store.get_stats()
    if stats:
        st.session_state.stats = stats


async def get_next_pair():
    """Get the next pair to label."""
    loader = st.session_state.loader
    labeled_pairs = st.session_state.labeled_pairs

    pair = await loader.get_random_pair(labeled_pairs)
    if not pair:
        st.session_state.current_pair = None
        st.session_state.current_left = None
        st.session_state.current_right = None
        return

    left_id, right_id = pair

    # Load emails
    emails = await loader.load_emails_batch([left_id, right_id])

    st.session_state.current_pair = {
        'left_id': left_id,
        'right_id': right_id,
    }
    st.session_state.current_left = emails.get(left_id)
    st.session_state.current_right = emails.get(right_id)
    st.session_state.selection = None
    st.session_state.confidence = CONFIDENCE_CERTAIN


async def record_label(preference: str, confidence: str):
    """Record a human preference label to the database."""
    pair = st.session_state.current_pair
    if not pair:
        return

    store = st.session_state.store
    left_id = pair['left_id']
    right_id = pair['right_id']
    labeler = st.session_state.labeler

    # Save to database
    success = await store.save_preference(
        email_left_id=left_id,
        email_right_id=right_id,
        preference=preference,
        confidence=confidence,
        labeler=labeler,
    )

    if success:
        # Add to local cache to avoid showing same pair
        st.session_state.labeled_pairs.add((left_id, right_id))
        st.session_state.labeled_pairs.add((right_id, left_id))

        # Update stats
        st.session_state.stats['total_labeled'] += 1
        if preference == PREFERENCE_LEFT:
            st.session_state.stats['left_chosen'] += 1
        elif preference == PREFERENCE_RIGHT:
            st.session_state.stats['right_chosen'] += 1
        elif preference == PREFERENCE_SAME:
            st.session_state.stats['same'] += 1
        else:
            st.session_state.stats['skipped'] += 1

        if confidence == CONFIDENCE_CERTAIN:
            st.session_state.stats['certain'] += 1
        else:
            st.session_state.stats['unsure'] += 1


def main():
    st.set_page_config(
        page_title="Email Preference Labeling",
        page_icon="📧",
        layout="wide"
    )

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--labeler',
        type=str,
        default='anonymous',
        help='Name of the labeler for tracking'
    )

    # Streamlit passes args after --
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args([])

    # Initialize state
    init_session_state(args.labeler)

    # Initialize database on first run
    if not st.session_state.initialized:
        asyncio.run(initialize_data())
        st.session_state.initialized = True

    # Custom CSS
    st.markdown(EMAIL_CSS, unsafe_allow_html=True)

    # Header
    st.title("📧 Email Preference Labeling")
    st.markdown("**Which email would you handle first?** Pick the more important/urgent one.")

    # Stats sidebar
    with st.sidebar:
        st.header("Progress")
        labeled = st.session_state.stats.get('total_labeled', 0)

        st.metric("Labeled", labeled)

        st.divider()
        st.header("Statistics")
        stats = st.session_state.stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Left", stats.get('left_chosen', 0))
            st.metric("Same", stats.get('same', 0))
            st.metric("Certain", stats.get('certain', 0))
        with col2:
            st.metric("Right", stats.get('right_chosen', 0))
            st.metric("Skipped", stats.get('skipped', 0))
            st.metric("Unsure", stats.get('unsure', 0))

        st.divider()
        st.caption(f"Labeler: {st.session_state.labeler}")
        st.caption(f"Database: PostgreSQL (localhost:5433)")

    # Load next pair if needed
    if st.session_state.current_pair is None:
        asyncio.run(get_next_pair())

    if st.session_state.current_pair is None:
        st.success("🎉 No more pairs available! All sampled pairs have been labeled.")
        return

    # Display emails side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Email A")
        is_selected = st.session_state.selection == PREFERENCE_LEFT
        st.markdown(
            render_email_card(st.session_state.current_left, 'left', is_selected),
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("Email B")
        is_selected = st.session_state.selection == PREFERENCE_RIGHT
        st.markdown(
            render_email_card(st.session_state.current_right, 'right', is_selected),
            unsafe_allow_html=True
        )

    st.divider()

    # Preference selection
    st.subheader("Which email would you handle first?")

    pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)

    with pref_col1:
        if st.button("⬅️ Email A", use_container_width=True, type="primary" if st.session_state.selection == PREFERENCE_LEFT else "secondary"):
            st.session_state.selection = PREFERENCE_LEFT

    with pref_col2:
        if st.button("➡️ Email B", use_container_width=True, type="primary" if st.session_state.selection == PREFERENCE_RIGHT else "secondary"):
            st.session_state.selection = PREFERENCE_RIGHT

    with pref_col3:
        if st.button("🤝 Same Priority", use_container_width=True, type="primary" if st.session_state.selection == PREFERENCE_SAME else "secondary"):
            st.session_state.selection = PREFERENCE_SAME

    with pref_col4:
        if st.button("⏭️ Skip (Spam/Obvious)", use_container_width=True, type="primary" if st.session_state.selection == PREFERENCE_SKIP else "secondary"):
            st.session_state.selection = PREFERENCE_SKIP

    # Confidence selection
    st.subheader("How confident are you?")

    conf_col1, conf_col2, _ = st.columns([1, 1, 2])

    with conf_col1:
        if st.button("✅ Certain", use_container_width=True, type="primary" if st.session_state.confidence == CONFIDENCE_CERTAIN else "secondary"):
            st.session_state.confidence = CONFIDENCE_CERTAIN

    with conf_col2:
        if st.button("❓ Unsure", use_container_width=True, type="primary" if st.session_state.confidence == CONFIDENCE_UNSURE else "secondary"):
            st.session_state.confidence = CONFIDENCE_UNSURE

    st.divider()

    # Submit button
    _, submit_col, _ = st.columns([1, 1, 1])
    with submit_col:
        if st.button("✓ Submit & Next", use_container_width=True, type="primary", disabled=st.session_state.selection is None):
            asyncio.run(record_label(st.session_state.selection, st.session_state.confidence))
            asyncio.run(get_next_pair())
            st.rerun()

    # Debug info (collapsible)
    with st.expander("Debug Info"):
        pair = st.session_state.current_pair
        if pair:
            st.json({
                'left_id': pair.get('left_id'),
                'right_id': pair.get('right_id'),
            })


if __name__ == '__main__':
    main()
