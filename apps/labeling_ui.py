#!/usr/bin/env python3
"""
Task Labeling UI with Project Context

A Streamlit app for human labeling of email tasks with project associations.

Features:
- View email content and extracted tasks
- See associated projects from auto-discovery
- Confirm or correct project associations
- Rate task relevancy and triage category
- Store labels in human_task_labels table

Usage:
    streamlit run apps/labeling_ui.py
"""

import streamlit as st
import psycopg2
import psycopg2.extras
from datetime import datetime

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Labeling options
TRIAGE_CATEGORIES = [
    ("fyi", "FYI - Info only, no action needed"),
    ("quick_win", "Quick Win - Can complete in <2 min"),
    ("ai_doable", "AI Doable - Can be delegated to assistant"),
    ("human_high_value", "Human Required - Needs judgment/expertise"),
    ("waiting", "Waiting - Blocked on someone else"),
]

RELEVANCY_OPTIONS = [
    ("high", "High - Core to this project"),
    ("medium", "Medium - Related to project"),
    ("low", "Low - Tangentially related"),
    ("none", "None - Wrong project association"),
]

EXTRACTION_QUALITY = [
    ("good", "Good - Task correctly extracted"),
    ("partial", "Partial - Missing some context"),
    ("wrong", "Wrong - Not a real task"),
]


def get_connection():
    """Get database connection."""
    return psycopg2.connect(DB_URL)


@st.cache_data(ttl=60)
def get_emails_with_tasks(_conn, limit=100, offset=0, unlabeled_only=True):
    """Get emails that have extracted tasks."""
    with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if unlabeled_only:
            cur.execute("""
                SELECT DISTINCT e.id, e.subject, e.from_email, e.date_parsed,
                       e.body_preview, llm.tasks, llm.overall_urgency,
                       llm.topic_category, llm.summary
                FROM emails e
                JOIN email_llm_features llm ON llm.email_id = e.message_id
                WHERE llm.tasks IS NOT NULL
                  AND jsonb_array_length(llm.tasks) > 0
                  AND NOT EXISTS (
                      SELECT 1 FROM human_task_labels htl WHERE htl.email_id = e.id
                  )
                ORDER BY e.date_parsed DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
        else:
            cur.execute("""
                SELECT DISTINCT e.id, e.subject, e.from_email, e.date_parsed,
                       e.body_preview, llm.tasks, llm.overall_urgency,
                       llm.topic_category, llm.summary
                FROM emails e
                JOIN email_llm_features llm ON llm.email_id = e.message_id
                WHERE llm.tasks IS NOT NULL
                  AND jsonb_array_length(llm.tasks) > 0
                ORDER BY e.date_parsed DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
        return cur.fetchall()


@st.cache_data(ttl=60)
def get_email_projects(_conn, email_id: int):
    """Get projects associated with an email."""
    with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT p.id, p.name, p.source, p.project_type, p.email_count,
                   epl.confidence, epl.source as link_source
            FROM email_project_links epl
            JOIN projects p ON p.id = epl.project_id
            WHERE epl.email_id = %s
              AND p.merged_into IS NULL
            ORDER BY epl.confidence DESC, p.email_count DESC
        """, (email_id,))
        return cur.fetchall()


@st.cache_data(ttl=300)
def get_all_projects(_conn):
    """Get all active projects for selection."""
    with _conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, name, source, project_type, email_count
            FROM projects
            WHERE merged_into IS NULL
            ORDER BY email_count DESC
            LIMIT 500
        """)
        return cur.fetchall()


def save_task_label(conn, email_id: int, task_index: int, task_description: str,
                    project_id: int, project_relevancy: str, triage_category: str,
                    extraction_quality: str, notes: str, labeler: str):
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


def get_labeling_stats(conn):
    """Get labeling statistics."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total_labels,
                COUNT(DISTINCT email_id) as emails_labeled,
                COUNT(DISTINCT labeler) as labelers
            FROM human_task_labels
        """)
        return cur.fetchone()


def main():
    st.set_page_config(
        page_title="Task Labeling UI",
        page_icon="📧",
        layout="wide"
    )

    st.title("📧 Task Labeling with Project Context")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        labeler = st.text_input("Your name", value="anonymous")
        unlabeled_only = st.checkbox("Show unlabeled only", value=True)
        page_size = st.slider("Emails per page", 10, 50, 20)

        st.divider()
        st.header("Statistics")
        try:
            conn = get_connection()
            stats = get_labeling_stats(conn)
            st.metric("Labels Created", stats['total_labels'])
            st.metric("Emails Labeled", stats['emails_labeled'])
            st.metric("Labelers", stats['labelers'])
            conn.close()
        except Exception as e:
            st.error(f"Stats error: {e}")

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'current_email_idx' not in st.session_state:
        st.session_state.current_email_idx = 0

    # Main content
    try:
        conn = get_connection()

        # Get emails
        offset = st.session_state.current_page * page_size
        emails = get_emails_with_tasks(conn, limit=page_size, offset=offset,
                                        unlabeled_only=unlabeled_only)

        if not emails:
            st.info("No emails to label. All caught up!")
            conn.close()
            return

        # Navigation
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Previous") and st.session_state.current_email_idx > 0:
                st.session_state.current_email_idx -= 1
                st.rerun()
        with col2:
            st.write(f"Email {st.session_state.current_email_idx + 1} of {len(emails)}")
        with col3:
            if st.button("Next ➡️") and st.session_state.current_email_idx < len(emails) - 1:
                st.session_state.current_email_idx += 1
                st.rerun()

        # Ensure valid index
        if st.session_state.current_email_idx >= len(emails):
            st.session_state.current_email_idx = 0

        email = emails[st.session_state.current_email_idx]

        # Email header
        st.divider()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(email['subject'] or "(No Subject)")
            st.caption(f"From: {email['from_email']}")
            if email['date_parsed']:
                st.caption(f"Date: {email['date_parsed'].strftime('%Y-%m-%d %H:%M')}")
        with col2:
            if email['overall_urgency']:
                urgency_pct = int(email['overall_urgency'] * 100)
                st.metric("Urgency", f"{urgency_pct}%")
            if email['topic_category']:
                st.caption(f"Category: {email['topic_category']}")

        # Email content
        with st.expander("📄 Email Content", expanded=False):
            if email['summary']:
                st.info(f"**Summary:** {email['summary']}")
            st.text(email['body_preview'] or "(No preview)")

        # Project associations
        st.divider()
        st.subheader("🏷️ Project Associations")

        projects = get_email_projects(conn, email['id'])
        all_projects = get_all_projects(conn)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Auto-detected projects:**")
            if projects:
                for proj in projects:
                    confidence = int(proj['confidence'] * 100) if proj['confidence'] else 0
                    st.write(f"- {proj['name']} ({proj['source']}, {confidence}% conf)")
            else:
                st.write("_No projects auto-detected_")

        with col2:
            st.write("**Select/correct project:**")
            project_options = {0: "-- No project --"}
            project_options.update({p['id']: f"{p['name']} ({p['source']})" for p in all_projects})

            # Default to first auto-detected project if available
            default_project = projects[0]['id'] if projects else 0

            selected_project_id = st.selectbox(
                "Project",
                options=list(project_options.keys()),
                format_func=lambda x: project_options[x],
                index=list(project_options.keys()).index(default_project) if default_project in project_options else 0,
                key=f"project_{email['id']}"
            )

        # Tasks section
        st.divider()
        st.subheader("📋 Extracted Tasks")

        tasks = email['tasks'] or []

        if not tasks:
            st.info("No tasks extracted from this email")
        else:
            for idx, task in enumerate(tasks):
                with st.container():
                    st.markdown(f"**Task {idx + 1}:** {task.get('description', 'No description')}")

                    task_cols = st.columns([1, 1, 1, 1])

                    with task_cols[0]:
                        task_urgency = task.get('urgency', 0)
                        st.caption(f"Urgency: {int(task_urgency * 100)}%")
                        st.caption(f"Type: {task.get('task_type', 'unknown')}")

                    with task_cols[1]:
                        triage = st.selectbox(
                            "Triage Category",
                            options=[t[0] for t in TRIAGE_CATEGORIES],
                            format_func=lambda x: next((t[1] for t in TRIAGE_CATEGORIES if t[0] == x), x),
                            key=f"triage_{email['id']}_{idx}"
                        )

                    with task_cols[2]:
                        relevancy = st.selectbox(
                            "Project Relevancy",
                            options=[r[0] for r in RELEVANCY_OPTIONS],
                            format_func=lambda x: next((r[1] for r in RELEVANCY_OPTIONS if r[0] == x), x),
                            key=f"relevancy_{email['id']}_{idx}"
                        )

                    with task_cols[3]:
                        quality = st.selectbox(
                            "Extraction Quality",
                            options=[q[0] for q in EXTRACTION_QUALITY],
                            format_func=lambda x: next((q[1] for q in EXTRACTION_QUALITY if q[0] == x), x),
                            key=f"quality_{email['id']}_{idx}"
                        )

                    notes = st.text_input("Notes (optional)", key=f"notes_{email['id']}_{idx}")

                    if st.button(f"💾 Save Task {idx + 1}", key=f"save_{email['id']}_{idx}"):
                        # Save task label
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
                            labeler=labeler
                        )

                        # Update project link if project selected
                        if project_id:
                            update_email_project_link(conn, email['id'], project_id)

                        if label_id:
                            st.success(f"Saved label #{label_id}")
                            # Clear cache to refresh stats
                            get_labeling_stats.clear()
                        else:
                            st.warning("Label may already exist")

                    st.divider()

        # Save all and next button
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("✅ Done with this email - Next", type="primary"):
                if st.session_state.current_email_idx < len(emails) - 1:
                    st.session_state.current_email_idx += 1
                else:
                    st.session_state.current_page += 1
                    st.session_state.current_email_idx = 0
                # Clear cache
                get_emails_with_tasks.clear()
                st.rerun()

        conn.close()

    except Exception as e:
        st.error(f"Database error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
