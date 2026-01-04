#!/usr/bin/env python3
"""Human preference labeling UI for email prioritization.

A Streamlit-based interface for collecting human preference judgments on email
pairs. Supports smart sampling to focus human effort on uncertain/edge cases.

Usage:
    # Start SurrealDB first
    ./scripts/start_db.sh gmail

    # Run the UI
    streamlit run scripts/labeling_ui.py

    # With custom implicit preferences file
    streamlit run scripts/labeling_ui.py -- --implicit-prefs data/preferences_implicit.json
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from surrealdb import AsyncSurreal


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
    """Load emails from SurrealDB for labeling."""

    def __init__(
        self,
        url: str = 'ws://localhost:8001/rpc',
        namespace: str = 'rl_emails',
        database: str = 'gmail',
    ):
        self.url = url
        self.namespace = namespace
        self.database = database
        self._emails_cache = {}

    async def load_email(self, message_id: str) -> Optional[dict]:
        """Load a single email by message_id."""
        if message_id in self._emails_cache:
            return self._emails_cache[message_id]

        db = AsyncSurreal(self.url)
        try:
            await db.connect()
            await db.signin({'username': 'root', 'password': 'root'})
            await db.use(self.namespace, self.database)

            result = await db.query(
                '''
                SELECT
                    message_id,
                    subject,
                    body,
                    from_email,
                    to_emails,
                    date_str,
                    labels,
                    action
                FROM emails
                WHERE message_id = $message_id
                LIMIT 1
                ''',
                {'message_id': message_id}
            )

            if result and isinstance(result, list) and len(result) > 0:
                email = result[0]
                self._emails_cache[message_id] = email
                return email

            return None
        except Exception as e:
            st.error(f"Database error: {e}")
            return None
        finally:
            await db.close()

    async def load_emails_batch(self, message_ids: list[str]) -> dict[str, dict]:
        """Load multiple emails efficiently."""
        # Check cache first
        missing = [mid for mid in message_ids if mid not in self._emails_cache]

        if missing:
            db = AsyncSurreal(self.url)
            try:
                await db.connect()
                await db.signin({'username': 'root', 'password': 'root'})
                await db.use(self.namespace, self.database)

                # Load in batches of 50
                for i in range(0, len(missing), 50):
                    batch = missing[i:i + 50]
                    result = await db.query(
                        '''
                        SELECT
                            message_id,
                            subject,
                            body,
                            from_email,
                            to_emails,
                            date_str,
                            labels,
                            action
                        FROM emails
                        WHERE message_id IN $ids
                        ''',
                        {'ids': batch}
                    )

                    if result and isinstance(result, list):
                        for email in result:
                            mid = email.get('message_id')
                            if mid:
                                self._emails_cache[mid] = email
            except Exception as e:
                st.error(f"Database error: {e}")
            finally:
                await db.close()

        return {mid: self._emails_cache.get(mid) for mid in message_ids}


class SmartSampler:
    """Smart sampling to prioritize uncertain/edge case pairs."""

    def __init__(
        self,
        implicit_pairs: list[dict],
        labeled_pairs: list[dict],
    ):
        self.implicit_pairs = implicit_pairs
        self.labeled_pairs = labeled_pairs
        self._labeled_set = {
            (p['chosen_id'], p['rejected_id'])
            for p in labeled_pairs
        }

    def get_unlabeled_pairs(self) -> list[dict]:
        """Get pairs that haven't been labeled yet."""
        return [
            p for p in self.implicit_pairs
            if (p['chosen_id'], p['rejected_id']) not in self._labeled_set
        ]

    def sample_next_pair(self) -> Optional[dict]:
        """Sample the next pair to label using smart sampling.

        Prioritizes:
        1. Low confidence implicit pairs (uncertain)
        2. Pairs from underrepresented signal types
        3. Random sampling among remaining
        """
        unlabeled = self.get_unlabeled_pairs()
        if not unlabeled:
            return None

        # Bucket by confidence
        low_conf = [p for p in unlabeled if p.get('confidence', 1.0) < 0.65]
        med_conf = [p for p in unlabeled if 0.65 <= p.get('confidence', 1.0) < 0.8]
        high_conf = [p for p in unlabeled if p.get('confidence', 1.0) >= 0.8]

        # Prioritize low confidence (80%), then medium (15%), then high (5%)
        r = random.random()
        if r < 0.8 and low_conf:
            pool = low_conf
        elif r < 0.95 and med_conf:
            pool = med_conf
        elif high_conf:
            pool = high_conf
        else:
            pool = unlabeled

        # Balance signal types
        signal_counts = {}
        for p in self.labeled_pairs:
            sig = p.get('signal_type', 'unknown')
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        # Favor underrepresented signals
        min_count = min(signal_counts.values()) if signal_counts else 0
        underrepresented = [
            sig for sig, count in signal_counts.items()
            if count <= min_count + 10
        ]

        if underrepresented:
            underrep_pool = [
                p for p in pool
                if p.get('signal_type') in underrepresented
            ]
            if underrep_pool:
                pool = underrep_pool

        return random.choice(pool) if pool else None


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


def init_session_state():
    """Initialize Streamlit session state."""
    if 'loader' not in st.session_state:
        st.session_state.loader = EmailLoader()

    if 'implicit_pairs' not in st.session_state:
        st.session_state.implicit_pairs = []

    if 'human_labels' not in st.session_state:
        st.session_state.human_labels = []

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


def load_implicit_preferences(path: Path) -> list[dict]:
    """Load implicit preference pairs from JSON file."""
    if not path.exists():
        return []

    try:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict) and 'pairs' in data:
            return data['pairs']
        elif isinstance(data, list):
            return data
        else:
            st.error(f"Unexpected format in {path}")
            return []
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return []


def save_human_labels(path: Path, labels: list[dict]):
    """Save human preference labels to JSON."""
    output = {
        'metadata': {
            'total_labels': len(labels),
            'last_updated': datetime.now().isoformat(),
        },
        'labels': labels
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


def load_human_labels(path: Path) -> list[dict]:
    """Load existing human labels."""
    if not path.exists():
        return []

    try:
        with open(path) as f:
            data = json.load(f)
        return data.get('labels', [])
    except Exception:
        return []


async def get_next_pair():
    """Get the next pair to label."""
    sampler = SmartSampler(
        st.session_state.implicit_pairs,
        st.session_state.human_labels
    )

    pair = sampler.sample_next_pair()
    if not pair:
        st.session_state.current_pair = None
        st.session_state.current_left = None
        st.session_state.current_right = None
        return

    # Randomly assign left/right
    if random.random() < 0.5:
        left_id = pair['chosen_id']
        right_id = pair['rejected_id']
        implicit_choice = 'left'
    else:
        left_id = pair['rejected_id']
        right_id = pair['chosen_id']
        implicit_choice = 'right'

    # Load emails
    loader = st.session_state.loader
    emails = await loader.load_emails_batch([left_id, right_id])

    st.session_state.current_pair = {
        **pair,
        'left_id': left_id,
        'right_id': right_id,
        'implicit_choice': implicit_choice,
    }
    st.session_state.current_left = emails.get(left_id)
    st.session_state.current_right = emails.get(right_id)
    st.session_state.selection = None
    st.session_state.confidence = CONFIDENCE_CERTAIN


def record_label(preference: str, confidence: str):
    """Record a human preference label."""
    pair = st.session_state.current_pair
    if not pair:
        return

    # Map preference to chosen/rejected
    if preference == PREFERENCE_LEFT:
        chosen_id = pair['left_id']
        rejected_id = pair['right_id']
    elif preference == PREFERENCE_RIGHT:
        chosen_id = pair['right_id']
        rejected_id = pair['left_id']
    else:
        # Same or skip - no clear preference
        chosen_id = None
        rejected_id = None

    label = {
        'chosen_id': chosen_id,
        'rejected_id': rejected_id,
        'preference': preference,
        'confidence': confidence,
        'implicit_signal_type': pair.get('signal_type'),
        'implicit_confidence': pair.get('confidence'),
        'agrees_with_implicit': (
            preference == PREFERENCE_LEFT and pair['implicit_choice'] == 'left'
        ) or (
            preference == PREFERENCE_RIGHT and pair['implicit_choice'] == 'right'
        ),
        'labeled_at': datetime.now().isoformat(),
    }

    st.session_state.human_labels.append(label)

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
        '--implicit-prefs',
        type=Path,
        default=Path('data/preferences_implicit.json'),
        help='Path to implicit preferences JSON'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/preferences_human.json'),
        help='Path to save human labels'
    )

    # Streamlit passes args after --
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args([])

    # Initialize state
    init_session_state()

    # Load data on first run
    if not st.session_state.implicit_pairs:
        st.session_state.implicit_pairs = load_implicit_preferences(args.implicit_prefs)
        st.session_state.human_labels = load_human_labels(args.output)

    # Custom CSS
    st.markdown(EMAIL_CSS, unsafe_allow_html=True)

    # Header
    st.title("📧 Email Preference Labeling")
    st.markdown("**Which email would you handle first?** Pick the more important/urgent one.")

    # Stats sidebar
    with st.sidebar:
        st.header("Progress")
        total_pairs = len(st.session_state.implicit_pairs)
        labeled = len(st.session_state.human_labels)
        remaining = total_pairs - labeled

        st.metric("Labeled", labeled)
        st.metric("Remaining", remaining)

        if total_pairs > 0:
            progress = labeled / total_pairs
            st.progress(progress)
            st.caption(f"{progress * 100:.1f}% complete")

        st.divider()
        st.header("Statistics")
        stats = st.session_state.stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Left", stats['left_chosen'])
            st.metric("Same", stats['same'])
            st.metric("Certain", stats['certain'])
        with col2:
            st.metric("Right", stats['right_chosen'])
            st.metric("Skipped", stats['skipped'])
            st.metric("Unsure", stats['unsure'])

        st.divider()

        # Export button
        if st.button("💾 Save Labels", type="primary"):
            save_human_labels(args.output, st.session_state.human_labels)
            st.success(f"Saved {len(st.session_state.human_labels)} labels to {args.output}")

        # Data info
        st.divider()
        st.caption(f"Implicit pairs: {args.implicit_prefs}")
        st.caption(f"Output: {args.output}")

    # Main content
    if not st.session_state.implicit_pairs:
        st.warning("No implicit preference pairs found. Run extract_implicit_preferences.py first.")
        st.code(f"python scripts/extract_implicit_preferences.py -o {args.implicit_prefs}")
        return

    # Load next pair if needed
    if st.session_state.current_pair is None:
        asyncio.run(get_next_pair())

    if st.session_state.current_pair is None:
        st.success("🎉 All pairs have been labeled!")
        save_human_labels(args.output, st.session_state.human_labels)
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
            record_label(st.session_state.selection, st.session_state.confidence)

            # Auto-save every 10 labels
            if len(st.session_state.human_labels) % 10 == 0:
                save_human_labels(args.output, st.session_state.human_labels)

            asyncio.run(get_next_pair())
            st.rerun()

    # Debug info (collapsible)
    with st.expander("Debug Info"):
        pair = st.session_state.current_pair
        if pair:
            st.json({
                'signal_type': pair.get('signal_type'),
                'implicit_confidence': pair.get('confidence'),
                'implicit_choice': pair.get('implicit_choice'),
            })


if __name__ == '__main__':
    main()
