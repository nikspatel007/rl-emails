#!/usr/bin/env python3
"""
Task Labeling UI v2 - Action-First Workflow

A redesigned Streamlit app with action-first email triage.

Flow:
1. See email → What's my action? (Delete / Archive / Act On)
2. Delete/Archive: Save + move to next (no task verification)
3. Act On: Expand task panel with sub-actions and verification

Usage:
    streamlit run apps/labeling_ui_v2.py
"""

import streamlit as st
import psycopg2
import psycopg2.extras
from datetime import datetime

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Action types for quick triage
PRIMARY_ACTIONS = {
    "delete": {"label": "Delete", "icon": "🗑️", "color": "green", "desc": "Trash, promotional, spam"},
    "archive": {"label": "Archive", "icon": "📁", "color": "blue", "desc": "Reference only, no action needed"},
    "act_on": {"label": "Act On", "icon": "⚡", "color": "orange", "desc": "Needs work - expand to see tasks"},
}

# Sub-actions when "Act On" is selected
ACT_ON_SUBACTIONS = [
    ("reply_now", "Reply Now", "Respond immediately"),
    ("reply_later", "Reply Later", "Queue for later response"),
    ("forward", "Forward", "Delegate to someone else"),
    ("create_task", "Create Task", "Add to task list"),
    ("snooze", "Snooze", "Remind me later"),
]

# Extraction quality for task verification
EXTRACTION_QUALITY = [
    ("good", "Correct"),
    ("partial", "Partial"),
    ("wrong", "Wrong"),
]

# Custom CSS for big action buttons
CUSTOM_CSS = """
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Big action button styling */
    .action-btn-delete button {
        background-color: #22c55e !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        height: auto !important;
        min-height: 80px !important;
    }
    .action-btn-archive button {
        background-color: #3b82f6 !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        height: auto !important;
        min-height: 80px !important;
    }
    .action-btn-act button {
        background-color: #f97316 !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        height: auto !important;
        min-height: 80px !important;
    }

    /* Task panel styling */
    .task-panel {
        background-color: #fff7ed;
        border: 2px solid #f97316;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    /* Email header */
    .email-header {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Progress colors */
    .accuracy-high { color: #22c55e; font-weight: bold; }
    .accuracy-medium { color: #f59e0b; font-weight: bold; }
    .accuracy-low { color: #ef4444; font-weight: bold; }

    /* Subaction buttons */
    .subaction-selected {
        border: 3px solid #f97316 !important;
        background-color: #fff7ed !important;
    }
</style>
"""

# Keyboard shortcuts
KEYBOARD_JS = """
<script>
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const key = e.key.toLowerCase();

    if (key === 'd') {
        // Delete
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Delete'));
        if (btn) btn.click();
    } else if (key === 'a') {
        // Archive
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Archive'));
        if (btn) btn.click();
    } else if (key === 'e') {
        // Act on (execute)
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Act On'));
        if (btn) btn.click();
    } else if (key === 'j') {
        // Next
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Next'));
        if (btn) btn.click();
    } else if (key === 'k') {
        // Previous
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Previous'));
        if (btn) btn.click();
    }
});
</script>
"""


def get_connection():
    """Get database connection."""
    return psycopg2.connect(DB_URL)


def get_emails_for_verification(conn, limit=50, offset=0, unlabeled_only=True,
                                 confidence_filter=None, sort_by='date'):
    """Get emails with LLM features for verification."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        where_clauses = ["e.body_text IS NOT NULL"]

        if unlabeled_only:
            where_clauses.append("""
                NOT EXISTS (
                    SELECT 1 FROM email_actions ea WHERE ea.email_id = e.id
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
                COALESCE(llm.tasks, '[]'::jsonb) as tasks,
                llm.overall_urgency, llm.topic_category, llm.summary,
                ef.overall_priority as ai_priority,
                ef.task_score as ai_task_score,
                ef.urgency_score as ai_urgency,
                ef.is_service_email as ai_is_service
            FROM emails e
            LEFT JOIN email_llm_features llm ON llm.email_id = e.message_id
            LEFT JOIN email_features ef ON ef.email_id = e.id
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT %s OFFSET %s
        """, (limit, offset))
        return cur.fetchall()


def get_email_projects(conn, email_id: int):
    """Get projects associated with an email."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT p.id, p.name, p.source, p.project_type,
                   epl.confidence, epl.source as link_source
            FROM email_project_links epl
            JOIN projects p ON p.id = epl.project_id
            WHERE epl.email_id = %s
              AND p.merged_into IS NULL
            ORDER BY epl.confidence DESC
        """, (email_id,))
        return cur.fetchall()


def get_all_projects(conn):
    """Get all active projects for selection."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, name, source, project_type, email_count
            FROM projects
            WHERE merged_into IS NULL
            ORDER BY email_count DESC
            LIMIT 500
        """)
        return cur.fetchall()


def get_labeling_stats(conn):
    """Get labeling statistics."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Action stats
        cur.execute("""
            SELECT
                COUNT(*) as total_actions,
                COUNT(DISTINCT email_id) as emails_labeled,
                COUNT(*) FILTER (WHERE action = 'delete') as deletes,
                COUNT(*) FILTER (WHERE action = 'archive') as archives,
                COUNT(*) FILTER (WHERE action = 'act_on') as act_ons
            FROM email_actions
        """)
        actions = cur.fetchone() or {
            'total_actions': 0, 'emails_labeled': 0,
            'deletes': 0, 'archives': 0, 'act_ons': 0
        }

        # Pending count
        cur.execute("""
            SELECT COUNT(*) as pending
            FROM emails e
            WHERE e.body_text IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM email_actions ea WHERE ea.email_id = e.id
              )
        """)
        pending = cur.fetchone()

        return {
            **actions,
            'pending': pending['pending'] if pending else 0
        }


def save_email_action(conn, email_id: int, action: str, sub_action: str = None,
                      project_id: int = None, notes: str = None, labeler: str = "anonymous"):
    """Save primary email action (delete/archive/act_on)."""
    with conn.cursor() as cur:
        # Create table if not exists (for first run)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_actions (
                id SERIAL PRIMARY KEY,
                email_id INTEGER NOT NULL REFERENCES emails(id),
                action VARCHAR(20) NOT NULL,
                sub_action VARCHAR(30),
                project_id INTEGER REFERENCES projects(id),
                notes TEXT,
                labeler VARCHAR(100),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(email_id)
            )
        """)

        cur.execute("""
            INSERT INTO email_actions (email_id, action, sub_action, project_id, notes, labeler)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (email_id) DO UPDATE SET
                action = EXCLUDED.action,
                sub_action = EXCLUDED.sub_action,
                project_id = EXCLUDED.project_id,
                notes = EXCLUDED.notes,
                labeler = EXCLUDED.labeler,
                created_at = NOW()
            RETURNING id
        """, (email_id, action, sub_action, project_id, notes, labeler))
        conn.commit()
        result = cur.fetchone()
        return result[0] if result else None


def save_task_verification(conn, email_id: int, task_index: int,
                           extraction_quality: str, labeler: str):
    """Save task verification for act_on emails."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO human_task_labels (
                email_id, task_index, task_description,
                extraction_quality, labeler, created_at
            ) VALUES (%s, %s, '', %s, %s, NOW())
            ON CONFLICT DO NOTHING
            RETURNING id
        """, (email_id, task_index, extraction_quality, labeler))
        conn.commit()
        return cur.fetchone()


def advance_to_next(emails):
    """Move to next email."""
    if st.session_state.current_idx < len(emails) - 1:
        st.session_state.current_idx += 1
    else:
        st.session_state.page += 1
        st.session_state.current_idx = 0
    # Clear act_on state
    st.session_state.show_act_panel = False
    st.session_state.selected_subaction = None


def main():
    st.set_page_config(
        page_title="Email Triage",
        page_icon="📧",
        layout="wide"
    )

    # Inject CSS and JS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.components.v1.html(KEYBOARD_JS, height=0)

    st.title("📧 Email Triage")
    st.caption("**Shortcuts:** d=Delete | a=Archive | e=Act | j/k=Navigate")

    # Initialize session state
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'page' not in st.session_state:
        st.session_state.page = 0
    if 'show_act_panel' not in st.session_state:
        st.session_state.show_act_panel = False
    if 'selected_subaction' not in st.session_state:
        st.session_state.selected_subaction = None

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        labeler = st.text_input("Your name", value="anonymous")
        unlabeled_only = st.checkbox("Unlabeled only", value=True)

        st.subheader("Filters")
        confidence_filter = st.selectbox(
            "AI Confidence",
            options=[None, 'low', 'medium', 'high'],
            format_func=lambda x: {
                None: "All",
                'low': "Low (<30%)",
                'medium': "Medium (30-70%)",
                'high': "High (>70%)"
            }.get(x, x)
        )

        sort_by = st.selectbox(
            "Sort by",
            options=['date', 'priority', 'confidence_asc'],
            format_func=lambda x: {
                'date': "Newest first",
                'priority': "Highest priority",
                'confidence_asc': "Lowest confidence"
            }.get(x, x)
        )

        page_size = st.slider("Per page", 10, 100, 25)

        st.divider()
        st.header("Progress")

        try:
            conn = get_connection()
            stats = get_labeling_stats(conn)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Done", stats['emails_labeled'])
            with col2:
                st.metric("Pending", stats['pending'])

            # Action breakdown
            if stats['emails_labeled'] > 0:
                st.caption(f"🗑️ {stats['deletes']} | 📁 {stats['archives']} | ⚡ {stats['act_ons']}")

            total = stats['emails_labeled'] + stats['pending']
            if total > 0:
                progress = stats['emails_labeled'] / total
                st.progress(progress)
                st.caption(f"{progress*100:.1f}% complete")

            conn.close()
        except Exception as e:
            st.warning(f"Stats unavailable: {e}")

    # Main content
    try:
        conn = get_connection()

        offset = st.session_state.page * page_size
        emails = get_emails_for_verification(
            conn, limit=page_size, offset=offset,
            unlabeled_only=unlabeled_only,
            confidence_filter=confidence_filter,
            sort_by=sort_by
        )

        if not emails:
            st.success("🎉 All caught up! No emails to triage.")
            conn.close()
            return

        if st.session_state.current_idx >= len(emails):
            st.session_state.current_idx = 0

        email = emails[st.session_state.current_idx]

        # Navigation
        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("← Previous (k)", disabled=st.session_state.current_idx == 0):
                st.session_state.current_idx -= 1
                st.session_state.show_act_panel = False
                st.rerun()
        with nav2:
            st.markdown(f"### Email {st.session_state.current_idx + 1} of {len(emails)}")
        with nav3:
            if st.button("Next (j) →", disabled=st.session_state.current_idx >= len(emails) - 1):
                st.session_state.current_idx += 1
                st.session_state.show_act_panel = False
                st.rerun()

        st.divider()

        # Email header
        col_header, col_meta = st.columns([3, 1])
        with col_header:
            st.subheader(email['subject'] or "(No Subject)")
            st.caption(f"**From:** {email['from_name'] or email['from_email']}")
            if email['date_parsed']:
                st.caption(f"**Date:** {email['date_parsed'].strftime('%Y-%m-%d %H:%M')}")
        with col_meta:
            if email['ai_priority'] is not None:
                priority_pct = int(email['ai_priority'] * 100)
                st.metric("AI Priority", f"{priority_pct}%")
            if email['ai_is_service']:
                st.info("📬 Service Email")

        # Email content (collapsible)
        with st.expander("📧 View Email Content", expanded=False):
            if email['summary']:
                st.info(f"**Summary:** {email['summary']}")
            body = email['body_text'] or email['body_preview'] or "(No content)"
            st.text(body[:2000] + ("..." if len(body) > 2000 else ""))

        # Project selection (compact)
        projects = get_email_projects(conn, email['id'])
        all_projects = get_all_projects(conn)

        with st.expander("📂 Project", expanded=False):
            project_options = {0: "-- No project --"}
            project_options.update({p['id']: p['name'] for p in all_projects})
            default_project = projects[0]['id'] if projects else 0

            selected_project_id = st.selectbox(
                "Project:",
                options=list(project_options.keys()),
                format_func=lambda x: project_options[x],
                index=list(project_options.keys()).index(default_project)
                      if default_project in project_options else 0,
                key=f"proj_{email['id']}",
                label_visibility="collapsed"
            )

        st.divider()

        # ============================================
        # PRIMARY ACTION BUTTONS - Big and prominent
        # ============================================
        st.markdown("### What's your action?")

        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            st.markdown('<div class="action-btn-delete">', unsafe_allow_html=True)
            if st.button("🗑️ DELETE\n\nTrash / Promo / Spam",
                        key="btn_delete", use_container_width=True, type="primary"):
                project_id = selected_project_id if selected_project_id > 0 else None
                save_email_action(conn, email['id'], 'delete',
                                 project_id=project_id, labeler=labeler)
                advance_to_next(emails)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with btn_col2:
            st.markdown('<div class="action-btn-archive">', unsafe_allow_html=True)
            if st.button("📁 ARCHIVE\n\nReference / FYI",
                        key="btn_archive", use_container_width=True, type="secondary"):
                project_id = selected_project_id if selected_project_id > 0 else None
                save_email_action(conn, email['id'], 'archive',
                                 project_id=project_id, labeler=labeler)
                advance_to_next(emails)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with btn_col3:
            st.markdown('<div class="action-btn-act">', unsafe_allow_html=True)
            if st.button("⚡ ACT ON\n\nNeeds Work",
                        key="btn_act", use_container_width=True, type="secondary"):
                st.session_state.show_act_panel = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # ============================================
        # ACT ON PANEL - Only shown when Act On clicked
        # ============================================
        if st.session_state.show_act_panel:
            st.divider()
            st.markdown("### ⚡ Action Details")

            # Sub-action selection
            st.markdown("**What will you do?**")
            subaction_cols = st.columns(len(ACT_ON_SUBACTIONS))

            for i, (key, label, desc) in enumerate(ACT_ON_SUBACTIONS):
                with subaction_cols[i]:
                    is_selected = st.session_state.selected_subaction == key
                    btn_type = "primary" if is_selected else "secondary"
                    if st.button(f"{label}", key=f"sub_{key}",
                                use_container_width=True, type=btn_type):
                        st.session_state.selected_subaction = key
                        st.rerun()
                    st.caption(desc)

            # Task verification (if tasks exist)
            tasks = email['tasks'] or []
            if tasks:
                st.markdown("---")
                st.markdown("**LLM-Extracted Tasks:**")

                for idx, task in enumerate(tasks):
                    task_col1, task_col2 = st.columns([3, 1])
                    with task_col1:
                        st.markdown(f"**{idx+1}.** {task.get('description', 'No description')}")
                        urgency = int(task.get('urgency', 0) * 100)
                        st.caption(f"Type: {task.get('task_type', 'unknown')} | Urgency: {urgency}%")
                    with task_col2:
                        quality = st.radio(
                            "Quality",
                            options=[q[0] for q in EXTRACTION_QUALITY],
                            format_func=lambda x: next((q[1] for q in EXTRACTION_QUALITY if q[0] == x), x),
                            key=f"qual_{email['id']}_{idx}",
                            horizontal=True,
                            label_visibility="collapsed"
                        )

            # Notes
            notes = st.text_input("Notes (optional)", key=f"notes_{email['id']}")

            # Save & Continue button
            st.markdown("---")
            save_col1, save_col2, save_col3 = st.columns([1, 2, 1])
            with save_col2:
                can_save = st.session_state.selected_subaction is not None
                if st.button("✓ Save & Continue", type="primary",
                            use_container_width=True, disabled=not can_save):
                    project_id = selected_project_id if selected_project_id > 0 else None

                    # Save action
                    save_email_action(
                        conn, email['id'], 'act_on',
                        sub_action=st.session_state.selected_subaction,
                        project_id=project_id,
                        notes=notes,
                        labeler=labeler
                    )

                    # Save task verifications
                    for idx, task in enumerate(tasks):
                        quality_key = f"qual_{email['id']}_{idx}"
                        if quality_key in st.session_state:
                            save_task_verification(
                                conn, email['id'], idx,
                                st.session_state[quality_key],
                                labeler
                            )

                    advance_to_next(emails)
                    st.rerun()

                if not can_save:
                    st.caption("Select a sub-action above")

        conn.close()

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
