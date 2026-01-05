#!/usr/bin/env python3
"""
Task Labeling UI v2 - Human Verification Workflow

A redesigned Streamlit app for efficient human verification of AI-extracted data.

Features:
- 3-column layout: LLM Extracted | AI Predicted | Human Verified
- One-click "All Correct" for quick approval
- Keyboard shortcuts: j/k navigate, y=correct, n=needs fix
- Progress tracker with AI accuracy %
- Collapsible email content
- Confidence-based filtering

Usage:
    streamlit run apps/labeling_ui_v2.py
"""

import streamlit as st
import psycopg2
import psycopg2.extras
from datetime import datetime
import json

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Labeling options
TRIAGE_CATEGORIES = [
    ("fyi", "FYI - Info only"),
    ("quick_win", "Quick Win - <2 min"),
    ("ai_doable", "AI Doable"),
    ("human_high_value", "Human Required"),
    ("waiting", "Waiting/Blocked"),
]

RELEVANCY_OPTIONS = [
    ("high", "High - Core to project"),
    ("medium", "Medium - Related"),
    ("low", "Low - Tangential"),
    ("none", "None - Wrong association"),
]

EXTRACTION_QUALITY = [
    ("good", "Good - Correctly extracted"),
    ("partial", "Partial - Missing context"),
    ("wrong", "Wrong - Not a real task"),
]

# Email statuses - these are lifecycle states, NOT projects
# They should never appear in project selection dropdowns
EMAIL_STATUSES = [
    "archived", "read", "unread", "starred", "snoozed",
    "inbox", "spam", "trash", "draft", "sent",
]

# Custom CSS for better styling
CUSTOM_CSS = """
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .verification-column {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 1rem;
        border: 2px solid #4a90d9;
    }
    .llm-column {
        background-color: #fff7e6;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #d9a441;
    }
    .ai-column {
        background-color: #f0fff4;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #48bb78;
    }
    .email-header {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .task-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .accuracy-high { color: #22c55e; font-weight: bold; }
    .accuracy-medium { color: #f59e0b; font-weight: bold; }
    .accuracy-low { color: #ef4444; font-weight: bold; }
    .shortcut-hint {
        font-size: 0.75rem;
        color: #666;
        background-color: #f0f0f0;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 4px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
"""

# JavaScript for keyboard shortcuts
KEYBOARD_JS = """
<script>
document.addEventListener('keydown', function(e) {
    // Only trigger if not in an input field
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const key = e.key.toLowerCase();

    if (key === 'j') {
        // Next email
        const nextBtn = document.querySelector('[data-testid="next-button"]') ||
                       Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Next'));
        if (nextBtn) nextBtn.click();
    } else if (key === 'k') {
        // Previous email
        const prevBtn = document.querySelector('[data-testid="prev-button"]') ||
                       Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Previous'));
        if (prevBtn) prevBtn.click();
    } else if (key === 'y') {
        // Approve all
        const approveBtn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('All Correct'));
        if (approveBtn) approveBtn.click();
    } else if (key === 'n') {
        // Needs fix - expand verification section
        const expanders = document.querySelectorAll('[data-testid="stExpander"]');
        if (expanders.length > 0) expanders[0].click();
    }
});
</script>
"""


def get_connection():
    """Get database connection."""
    return psycopg2.connect(DB_URL)


def get_emails_for_verification(conn, limit=50, offset=0, unlabeled_only=True,
                                 confidence_filter=None, sort_by='date'):
    """Get emails with LLM features and AI predictions for verification."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        where_clauses = ["llm.tasks IS NOT NULL", "jsonb_array_length(llm.tasks) > 0"]

        if unlabeled_only:
            where_clauses.append("""
                NOT EXISTS (
                    SELECT 1 FROM human_task_labels htl WHERE htl.email_id = e.id
                )
            """)

        if confidence_filter == 'low':
            where_clauses.append("(ef.overall_priority IS NULL OR ef.overall_priority < 0.3)")
        elif confidence_filter == 'medium':
            where_clauses.append("ef.overall_priority >= 0.3 AND ef.overall_priority < 0.7")
        elif confidence_filter == 'high':
            where_clauses.append("ef.overall_priority >= 0.7")

        where_sql = " AND ".join(where_clauses)

        order_sql = "e.date_parsed DESC"
        if sort_by == 'priority':
            order_sql = "COALESCE(ef.overall_priority, 0) DESC"
        elif sort_by == 'confidence_asc':
            order_sql = "COALESCE(ef.overall_priority, 0) ASC"

        cur.execute(f"""
            SELECT DISTINCT
                e.id, e.message_id, e.subject, e.from_email, e.from_name,
                e.date_parsed, e.body_text, e.body_preview, e.thread_id,
                llm.tasks, llm.overall_urgency, llm.topic_category,
                llm.summary,
                ef.overall_priority as ai_priority,
                ef.task_score as ai_task_score,
                ef.urgency_score as ai_urgency,
                ef.project_score as ai_project_score,
                ef.is_service_email as ai_is_service
            FROM emails e
            JOIN email_llm_features llm ON llm.email_id = e.message_id
            LEFT JOIN email_features ef ON ef.email_id = e.id
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT %s OFFSET %s
        """, (limit, offset))
        return cur.fetchall()


def get_email_projects(conn, email_id: int):
    """Get projects associated with an email.

    Excludes email statuses (Read, Unread, Archived, etc.) which are
    lifecycle states, not topical project groupings.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Build exclusion list - case insensitive matching
        status_exclusions = ", ".join([f"'{s}'" for s in EMAIL_STATUSES])

        cur.execute(f"""
            SELECT p.id, p.name, p.source, p.project_type,
                   epl.confidence, epl.source as link_source
            FROM email_project_links epl
            JOIN projects p ON p.id = epl.project_id
            WHERE epl.email_id = %s
              AND p.merged_into IS NULL
              AND LOWER(p.name) NOT IN ({status_exclusions})
            ORDER BY epl.confidence DESC
        """, (email_id,))
        return cur.fetchall()


def get_all_projects(conn):
    """Get all active projects for selection.

    Excludes email statuses (Read, Unread, Archived, etc.) which are
    lifecycle states, not topical project groupings.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Build exclusion list - case insensitive matching
        status_exclusions = ", ".join([f"'{s}'" for s in EMAIL_STATUSES])

        cur.execute(f"""
            SELECT id, name, source, project_type, email_count
            FROM projects
            WHERE merged_into IS NULL
              AND LOWER(name) NOT IN ({status_exclusions})
            ORDER BY email_count DESC
            LIMIT 500
        """)
        return cur.fetchall()


def get_labeling_stats(conn):
    """Get comprehensive labeling statistics."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Basic stats
        cur.execute("""
            SELECT
                COUNT(*) as total_labels,
                COUNT(DISTINCT email_id) as emails_labeled,
                COUNT(DISTINCT labeler) as labelers
            FROM human_task_labels
        """)
        basic = cur.fetchone()

        # AI accuracy (when human agrees with AI)
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE extraction_quality = 'good') as good_extractions,
                COUNT(*) as total_extractions
            FROM human_task_labels
        """)
        accuracy = cur.fetchone()

        # Pending count
        cur.execute("""
            SELECT COUNT(DISTINCT e.id) as pending
            FROM emails e
            JOIN email_llm_features llm ON llm.email_id = e.message_id
            WHERE llm.tasks IS NOT NULL
              AND jsonb_array_length(llm.tasks) > 0
              AND NOT EXISTS (
                  SELECT 1 FROM human_task_labels htl WHERE htl.email_id = e.id
              )
        """)
        pending = cur.fetchone()

        return {
            **basic,
            'ai_accuracy': (accuracy['good_extractions'] / accuracy['total_extractions'] * 100)
                          if accuracy['total_extractions'] > 0 else 0,
            'pending': pending['pending']
        }


def save_task_label(conn, email_id: int, task_index: int, task_description: str,
                    project_id: int, project_relevancy: str, triage_category: str,
                    extraction_quality: str, notes: str, labeler: str,
                    ai_agreed: bool = True):
    """Save a task label to the database."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO human_task_labels (
                email_id, task_index, task_description, project_id,
                project_relevancy, triage_category, extraction_quality,
                notes, labeler, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT DO NOTHING
            RETURNING id
        """, (email_id, task_index, task_description, project_id,
              project_relevancy, triage_category, extraction_quality,
              notes, labeler))
        conn.commit()
        result = cur.fetchone()
        return result[0] if result else None


def save_all_tasks_approved(conn, email_id: int, tasks: list, project_id: int,
                            labeler: str):
    """Save all tasks for an email as approved (AI was correct)."""
    saved_count = 0
    for idx, task in enumerate(tasks):
        label_id = save_task_label(
            conn,
            email_id=email_id,
            task_index=idx,
            task_description=task.get('description', ''),
            project_id=project_id,
            project_relevancy='high',
            triage_category=task.get('task_type', 'quick_win'),
            extraction_quality='good',
            notes='Quick approved - AI correct',
            labeler=labeler,
            ai_agreed=True
        )
        if label_id:
            saved_count += 1
    return saved_count


def update_email_project_link(conn, email_id: int, project_id: int, confidence: float = 1.0):
    """Update or create email-project link."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO email_project_links (email_id, project_id, confidence, source)
            VALUES (%s, %s, %s, 'human')
            ON CONFLICT (email_id, project_id) DO UPDATE
            SET confidence = GREATEST(email_project_links.confidence, EXCLUDED.confidence),
                source = 'human'
        """, (email_id, project_id, confidence))
        conn.commit()


def render_task_column(task: dict, column_type: str, task_idx: int, email_id: int):
    """Render a task in a specific column format."""
    desc = task.get('description', 'No description')
    urgency = task.get('urgency', 0)
    task_type = task.get('task_type', 'unknown')

    if column_type == 'llm':
        st.markdown(f"**{desc}**")
        st.caption(f"Type: {task_type} | Urgency: {int(urgency * 100)}%")
    elif column_type == 'ai':
        st.markdown(f"*{desc}*")
        st.caption(f"Predicted: {task_type}")
    else:
        st.markdown(f"_{desc}_")


def main():
    st.set_page_config(
        page_title="Labeling UI v2",
        page_icon="✓",
        layout="wide"
    )

    # Inject custom CSS and keyboard shortcuts
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.components.v1.html(KEYBOARD_JS, height=0)

    st.title("✓ Human Verification Workflow")
    st.caption("Keyboard: **j/k** navigate | **y** approve all | **n** needs fix")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        labeler = st.text_input("Your name", value="anonymous")
        unlabeled_only = st.checkbox("Unlabeled only", value=True)

        st.subheader("Filters")
        confidence_filter = st.selectbox(
            "Confidence level",
            options=[None, 'low', 'medium', 'high'],
            format_func=lambda x: {
                None: "All",
                'low': "Low (<30%) - Review first",
                'medium': "Medium (30-70%)",
                'high': "High (>70%)"
            }.get(x, x)
        )

        sort_by = st.selectbox(
            "Sort by",
            options=['date', 'priority', 'confidence_asc'],
            format_func=lambda x: {
                'date': "Date (newest first)",
                'priority': "Priority (highest first)",
                'confidence_asc': "Confidence (lowest first)"
            }.get(x, x)
        )

        page_size = st.slider("Emails per page", 10, 100, 25)

        st.divider()
        st.header("Progress")

        try:
            conn = get_connection()
            stats = get_labeling_stats(conn)

            # Progress metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Labeled", stats['emails_labeled'])
            with col2:
                st.metric("Pending", stats['pending'])

            # AI Accuracy with color coding
            accuracy = stats['ai_accuracy']
            accuracy_class = 'high' if accuracy >= 80 else ('medium' if accuracy >= 60 else 'low')
            st.markdown(f"**AI Accuracy:** <span class='accuracy-{accuracy_class}'>{accuracy:.1f}%</span>",
                       unsafe_allow_html=True)

            # Progress bar
            total = stats['emails_labeled'] + stats['pending']
            if total > 0:
                progress = stats['emails_labeled'] / total
                st.progress(progress)
                st.caption(f"{progress*100:.1f}% complete")

            conn.close()
        except Exception as e:
            st.error(f"Stats error: {e}")

    # Initialize session state
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'page' not in st.session_state:
        st.session_state.page = 0

    # Main content
    try:
        conn = get_connection()

        # Get emails
        offset = st.session_state.page * page_size
        emails = get_emails_for_verification(
            conn,
            limit=page_size,
            offset=offset,
            unlabeled_only=unlabeled_only,
            confidence_filter=confidence_filter,
            sort_by=sort_by
        )

        if not emails:
            st.success("All caught up! No emails to verify.")
            conn.close()
            return

        # Ensure valid index
        if st.session_state.current_idx >= len(emails):
            st.session_state.current_idx = 0

        email = emails[st.session_state.current_idx]

        # Navigation bar
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 2, 2, 1])

        with nav_col1:
            if st.button("← Previous (k)", disabled=st.session_state.current_idx == 0,
                        key="prev-button"):
                st.session_state.current_idx -= 1
                st.rerun()

        with nav_col2:
            st.markdown(f"**Email {st.session_state.current_idx + 1} of {len(emails)}**")

        with nav_col3:
            # Quick jump
            new_idx = st.number_input("Go to #", min_value=1, max_value=len(emails),
                                       value=st.session_state.current_idx + 1,
                                       label_visibility="collapsed")
            if new_idx - 1 != st.session_state.current_idx:
                st.session_state.current_idx = new_idx - 1
                st.rerun()

        with nav_col4:
            if st.button("Next (j) →", disabled=st.session_state.current_idx >= len(emails) - 1,
                        key="next-button"):
                st.session_state.current_idx += 1
                st.rerun()

        st.divider()

        # Email header
        with st.container():
            header_col1, header_col2 = st.columns([3, 1])

            with header_col1:
                st.subheader(email['subject'] or "(No Subject)")
                st.caption(f"**From:** {email['from_name'] or email['from_email']}")
                if email['date_parsed']:
                    st.caption(f"**Date:** {email['date_parsed'].strftime('%Y-%m-%d %H:%M')}")

            with header_col2:
                # AI scores
                if email['ai_priority'] is not None:
                    st.metric("AI Priority", f"{int(email['ai_priority'] * 100)}%")
                if email['ai_is_service']:
                    st.warning("Service Email")

        # Collapsible email content
        with st.expander("📧 Email Content", expanded=False):
            if email['summary']:
                st.info(f"**AI Summary:** {email['summary']}")

            if email['body_text']:
                # Show first 500 chars with option to expand
                body = email['body_text']
                if len(body) > 500:
                    st.text(body[:500] + "...")
                    if st.checkbox("Show full content", key=f"full_{email['id']}"):
                        st.text(body)
                else:
                    st.text(body)
            else:
                st.text(email['body_preview'] or "(No content)")

        st.divider()

        # Get projects for this email
        projects = get_email_projects(conn, email['id'])
        all_projects = get_all_projects(conn)

        # Project selection
        st.subheader("Project Association")
        proj_col1, proj_col2 = st.columns(2)

        with proj_col1:
            if projects:
                st.caption("**Auto-detected:**")
                for p in projects[:3]:
                    conf = int(p['confidence'] * 100) if p['confidence'] else 0
                    st.write(f"• {p['name']} ({conf}% conf)")
            else:
                st.caption("_No projects auto-detected_")

        with proj_col2:
            project_options = {0: "-- No project --"}
            project_options.update({p['id']: p['name'] for p in all_projects})
            default_project = projects[0]['id'] if projects else 0

            selected_project_id = st.selectbox(
                "Confirm/select project:",
                options=list(project_options.keys()),
                format_func=lambda x: project_options[x],
                index=list(project_options.keys()).index(default_project)
                      if default_project in project_options else 0,
                key=f"proj_{email['id']}"
            )

        st.divider()

        # 3-Column Task Verification Layout
        st.subheader("Task Verification")

        tasks = email['tasks'] or []

        if not tasks:
            st.info("No tasks extracted from this email")
        else:
            # Column headers
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### 🤖 LLM Extracted")
                st.caption("What Haiku found")
            with col2:
                st.markdown("### 📊 AI Predicted")
                st.caption("ML model scores")
            with col3:
                st.markdown("### ✓ Human Verified")
                st.caption("Your confirmation")

            # Quick approve button
            st.markdown("---")
            approve_col1, approve_col2, approve_col3 = st.columns([2, 1, 2])
            with approve_col2:
                if st.button("✓ All Correct (y)", type="primary", use_container_width=True):
                    project_id = selected_project_id if selected_project_id > 0 else None
                    saved = save_all_tasks_approved(conn, email['id'], tasks, project_id, labeler)

                    if project_id:
                        update_email_project_link(conn, email['id'], project_id)

                    st.success(f"Saved {saved} task(s) as approved!")

                    # Move to next
                    if st.session_state.current_idx < len(emails) - 1:
                        st.session_state.current_idx += 1
                    st.rerun()

            st.markdown("---")

            # Task rows
            for idx, task in enumerate(tasks):
                with st.container():
                    col1, col2, col3 = st.columns(3)

                    # LLM Extracted column
                    with col1:
                        st.markdown(f"**Task {idx + 1}:** {task.get('description', 'No description')}")
                        st.caption(f"Type: `{task.get('task_type', 'unknown')}`")
                        st.caption(f"Urgency: {int(task.get('urgency', 0) * 100)}%")
                        if task.get('deadline_text'):
                            st.caption(f"Deadline: {task.get('deadline_text')}")

                    # AI Predicted column
                    with col2:
                        if email['ai_task_score'] is not None:
                            st.metric("Task Score", f"{int(email['ai_task_score'] * 100)}%")
                        if email['ai_urgency'] is not None:
                            st.metric("Urgency Score", f"{int(email['ai_urgency'] * 100)}%")
                        if email['topic_category']:
                            st.caption(f"Category: {email['topic_category']}")

                    # Human Verified column
                    with col3:
                        with st.expander("Edit Details", expanded=False):
                            triage = st.selectbox(
                                "Triage",
                                options=[t[0] for t in TRIAGE_CATEGORIES],
                                format_func=lambda x: next((t[1] for t in TRIAGE_CATEGORIES if t[0] == x), x),
                                key=f"triage_{email['id']}_{idx}"
                            )

                            relevancy = st.selectbox(
                                "Relevancy",
                                options=[r[0] for r in RELEVANCY_OPTIONS],
                                format_func=lambda x: next((r[1] for r in RELEVANCY_OPTIONS if r[0] == x), x),
                                key=f"rel_{email['id']}_{idx}"
                            )

                            quality = st.selectbox(
                                "Extraction Quality",
                                options=[q[0] for q in EXTRACTION_QUALITY],
                                format_func=lambda x: next((q[1] for q in EXTRACTION_QUALITY if q[0] == x), x),
                                key=f"qual_{email['id']}_{idx}"
                            )

                            notes = st.text_input("Notes", key=f"notes_{email['id']}_{idx}")

                            if st.button(f"Save Task {idx + 1}", key=f"save_{email['id']}_{idx}"):
                                project_id = selected_project_id if selected_project_id > 0 else None
                                label_id = save_task_label(
                                    conn,
                                    email_id=email['id'],
                                    task_index=idx,
                                    task_description=task.get('description', ''),
                                    project_id=project_id,
                                    project_relevancy=relevancy,
                                    triage_category=triage,
                                    extraction_quality=quality,
                                    notes=notes,
                                    labeler=labeler,
                                    ai_agreed=(quality == 'good')
                                )

                                if project_id:
                                    update_email_project_link(conn, email['id'], project_id)

                                if label_id:
                                    st.success(f"Saved!")
                                else:
                                    st.warning("Already labeled")

                    st.markdown("---")

        # Bottom navigation
        st.divider()
        bottom_col1, bottom_col2 = st.columns([1, 1])

        with bottom_col1:
            if st.button("⏭️ Skip to Next", use_container_width=True):
                if st.session_state.current_idx < len(emails) - 1:
                    st.session_state.current_idx += 1
                else:
                    st.session_state.page += 1
                    st.session_state.current_idx = 0
                st.rerun()

        with bottom_col2:
            if st.button("✓ Done & Next", type="primary", use_container_width=True):
                if st.session_state.current_idx < len(emails) - 1:
                    st.session_state.current_idx += 1
                else:
                    st.session_state.page += 1
                    st.session_state.current_idx = 0
                st.rerun()

        conn.close()

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
