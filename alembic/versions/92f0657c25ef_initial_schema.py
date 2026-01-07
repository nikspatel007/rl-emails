"""initial_schema

Revision ID: 92f0657c25ef
Revises:
Create Date: 2026-01-05 14:09:29.550275

This is the baseline migration capturing the existing database schema.
For existing databases, run: alembic stamp 92f0657c25ef
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '92f0657c25ef'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema for rl-emails database."""

    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')  # pgvector for embeddings

    # raw_emails (Source of Truth)
    op.create_table(
        'raw_emails',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('message_id', sa.Text, unique=True, nullable=False),
        sa.Column('thread_id', sa.Text),
        sa.Column('in_reply_to', sa.Text),
        sa.Column('references_raw', sa.Text),
        sa.Column('date_raw', sa.Text),
        sa.Column('from_raw', sa.Text),
        sa.Column('to_raw', sa.Text),
        sa.Column('cc_raw', sa.Text),
        sa.Column('bcc_raw', sa.Text),
        sa.Column('subject_raw', sa.Text),
        sa.Column('body_text', sa.Text),
        sa.Column('body_html', sa.Text),
        sa.Column('labels_raw', sa.Text),
        sa.Column('mbox_offset', sa.BigInteger),
        sa.Column('raw_size_bytes', sa.Integer),
        sa.Column('imported_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_raw_emails_message_id', 'raw_emails', ['message_id'])
    op.create_index('idx_raw_emails_thread_id', 'raw_emails', ['thread_id'])

    # emails (Enriched/Derived)
    op.create_table(
        'emails',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('raw_email_id', sa.Integer, sa.ForeignKey('raw_emails.id')),
        sa.Column('message_id', sa.Text, unique=True, nullable=False),
        sa.Column('thread_id', sa.Text),
        sa.Column('in_reply_to', sa.Text),
        sa.Column('date_parsed', sa.TIMESTAMP(timezone=True)),
        sa.Column('from_email', sa.Text),
        sa.Column('from_name', sa.Text),
        sa.Column('to_emails', sa.ARRAY(sa.Text)),
        sa.Column('cc_emails', sa.ARRAY(sa.Text)),
        sa.Column('subject', sa.Text),
        sa.Column('body_text', sa.Text),
        sa.Column('body_preview', sa.Text),
        sa.Column('word_count', sa.Integer),
        sa.Column('labels', sa.ARRAY(sa.Text)),
        sa.Column('has_attachments', sa.Boolean, server_default='false'),
        sa.Column('attachment_count', sa.Integer, server_default='0'),
        sa.Column('attachment_types', sa.ARRAY(sa.Text)),
        sa.Column('total_attachment_bytes', sa.BigInteger, server_default='0'),
        sa.Column('is_sent', sa.Boolean, server_default='false'),
        sa.Column('action', sa.Text),
        sa.Column('timing', sa.Text),
        sa.Column('response_time_seconds', sa.Integer),
        sa.Column('priority_score', sa.Float),
        sa.Column('enriched_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('enrichment_version', sa.Integer, server_default='1'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_emails_thread_id', 'emails', ['thread_id'])
    op.create_index('idx_emails_from_email', 'emails', ['from_email'])
    op.create_index('idx_emails_date', 'emails', ['date_parsed'])
    op.create_index('idx_emails_in_reply_to', 'emails', ['in_reply_to'])
    op.create_index('idx_emails_is_sent', 'emails', ['is_sent'])
    op.execute('CREATE INDEX idx_emails_labels ON emails USING GIN(labels)')
    op.execute('CREATE INDEX idx_emails_subject_trgm ON emails USING GIN(subject gin_trgm_ops)')

    # attachments
    op.create_table(
        'attachments',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('raw_email_id', sa.Integer, sa.ForeignKey('raw_emails.id')),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id')),
        sa.Column('filename', sa.Text),
        sa.Column('content_type', sa.Text),
        sa.Column('size_bytes', sa.Integer),
        sa.Column('content_disposition', sa.Text),
        sa.Column('content_hash', sa.Text),
        sa.Column('stored_path', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_attachments_raw_email_id', 'attachments', ['raw_email_id'])
    op.create_index('idx_attachments_email_id', 'attachments', ['email_id'])
    op.create_index('idx_attachments_content_type', 'attachments', ['content_type'])

    # threads (Aggregated)
    op.create_table(
        'threads',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('thread_id', sa.Text, unique=True, nullable=False),
        sa.Column('subject', sa.Text),
        sa.Column('participants', sa.ARRAY(sa.Text)),
        sa.Column('your_role', sa.Text),
        sa.Column('email_count', sa.Integer, server_default='0'),
        sa.Column('your_email_count', sa.Integer, server_default='0'),
        sa.Column('your_reply_count', sa.Integer, server_default='0'),
        sa.Column('has_attachments', sa.Boolean, server_default='false'),
        sa.Column('total_attachment_count', sa.Integer, server_default='0'),
        sa.Column('attachment_types', sa.ARRAY(sa.Text)),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('last_activity', sa.TIMESTAMP(timezone=True)),
        sa.Column('your_first_reply_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('avg_response_time_seconds', sa.Integer),
        sa.Column('thread_duration_seconds', sa.Integer),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_threads_thread_id', 'threads', ['thread_id'])
    op.create_index('idx_threads_has_attachments', 'threads', ['has_attachments'])

    # users (Aggregated)
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email', sa.Text, unique=True, nullable=False),
        sa.Column('name', sa.Text),
        sa.Column('is_you', sa.Boolean, server_default='false'),
        sa.Column('emails_from', sa.Integer, server_default='0'),
        sa.Column('emails_to', sa.Integer, server_default='0'),
        sa.Column('threads_with', sa.Integer, server_default='0'),
        sa.Column('reply_count', sa.Integer, server_default='0'),
        sa.Column('avg_response_time_seconds', sa.Integer),
        sa.Column('reply_rate', sa.Float),
        sa.Column('is_important_sender', sa.Boolean, server_default='false'),
        sa.Column('first_contact', sa.TIMESTAMP(timezone=True)),
        sa.Column('last_contact', sa.TIMESTAMP(timezone=True)),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_is_you', 'users', ['is_you'])

    # email_features (Pre-computed ML Features)
    # Merged schema from alembic + compute_basic_features.py
    op.create_table(
        'email_features',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE'), unique=True),

        # Relationship Features (from compute_basic_features.py)
        sa.Column('emails_from_sender_7d', sa.Integer),
        sa.Column('emails_from_sender_30d', sa.Integer),
        sa.Column('emails_from_sender_90d', sa.Integer),
        sa.Column('emails_from_sender_all', sa.Integer),
        sa.Column('user_replied_to_sender_count', sa.Integer),
        sa.Column('user_replied_to_sender_rate', sa.Float),
        sa.Column('avg_response_time_hours', sa.Float),
        sa.Column('user_initiated_ratio', sa.Float),
        sa.Column('days_since_last_interaction', sa.Integer),
        sa.Column('sender_replies_to_you_rate', sa.Float),
        sa.Column('relationship_strength', sa.Float),

        # Service Detection (from compute_basic_features.py)
        sa.Column('is_service_email', sa.Boolean),
        sa.Column('service_confidence', sa.Float),
        sa.Column('service_type', sa.Text),
        sa.Column('service_importance', sa.Float),
        sa.Column('has_unsubscribe_link', sa.Boolean),
        sa.Column('has_list_unsubscribe_header', sa.Boolean),
        sa.Column('from_common_service_domain', sa.Boolean),

        # Temporal Features (from compute_basic_features.py)
        sa.Column('hour_of_day', sa.Integer),
        sa.Column('day_of_week', sa.Integer),
        sa.Column('is_weekend', sa.Boolean),
        sa.Column('is_business_hours', sa.Boolean),
        sa.Column('days_since_received', sa.Integer),
        sa.Column('is_recent', sa.Boolean),
        sa.Column('time_bucket', sa.Text),
        sa.Column('urgency_score', sa.Float),

        # Content Basic (from compute_basic_features.py)
        sa.Column('subject_word_count', sa.Integer),
        sa.Column('body_word_count', sa.Integer),
        sa.Column('has_attachments', sa.Boolean),
        sa.Column('attachment_count', sa.Integer),
        sa.Column('recipient_count', sa.Integer),

        # Additional features from original alembic schema
        sa.Column('sender_response_deviation', sa.Float),
        sa.Column('sender_frequency_rank', sa.Float),
        sa.Column('inferred_hierarchy', sa.Float),
        sa.Column('avg_thread_depth', sa.Float),
        sa.Column('days_since_last_email', sa.Float),
        sa.Column('cc_affinity_score', sa.Float),
        sa.Column('unsubscribe_phrase_count', sa.Integer),
        sa.Column('task_count', sa.Integer),
        sa.Column('has_deadline', sa.Boolean),
        sa.Column('deadline_urgency', sa.Float),
        sa.Column('is_assigned_to_user', sa.Boolean),
        sa.Column('estimated_effort', sa.Text),
        sa.Column('has_deliverable', sa.Boolean),
        sa.Column('urgency_bucket', sa.Text),
        sa.Column('project_score', sa.Float),
        sa.Column('topic_score', sa.Float),
        sa.Column('task_score', sa.Float),
        sa.Column('people_score', sa.Float),
        sa.Column('temporal_score', sa.Float),
        sa.Column('service_score', sa.Float),
        sa.Column('relationship_score', sa.Float),
        sa.Column('overall_priority', sa.Float),
        sa.Column('feature_vector', sa.ARRAY(sa.Float)),
        sa.Column('content_embedding', sa.ARRAY(sa.Float)),
        sa.Column('embedding_model', sa.Text),
        sa.Column('embedding_dim', sa.Integer),
        sa.Column('computed_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('feature_version', sa.Integer, server_default='1'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_email_features_email_id', 'email_features', ['email_id'], unique=True)
    op.create_index('idx_email_features_is_service', 'email_features', ['is_service_email'])
    op.create_index('idx_email_features_urgency', 'email_features', ['urgency_score'])
    op.create_index('idx_email_features_priority', 'email_features', ['overall_priority'])
    op.create_index('idx_email_features_service_type', 'email_features', ['service_type'])

    # tasks (Extracted from emails)
    op.create_table(
        'tasks',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('task_id', sa.Text, unique=True, nullable=False),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE')),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('deadline', sa.TIMESTAMP(timezone=True)),
        sa.Column('deadline_text', sa.Text),
        sa.Column('assignee_hint', sa.Text),
        sa.Column('complexity', sa.Text),  # CHECK constraint added via execute
        sa.Column('task_type', sa.Text),   # CHECK constraint added via execute
        sa.Column('urgency_score', sa.Float),  # CHECK constraint added via execute
        sa.Column('source_text', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.execute("ALTER TABLE tasks ADD CONSTRAINT tasks_complexity_check CHECK (complexity IN ('trivial', 'quick', 'medium', 'substantial', 'unknown'))")
    op.execute("ALTER TABLE tasks ADD CONSTRAINT tasks_task_type_check CHECK (task_type IN ('review', 'send', 'schedule', 'decision', 'research', 'create', 'follow_up', 'other'))")
    op.execute("ALTER TABLE tasks ADD CONSTRAINT tasks_urgency_score_check CHECK (urgency_score >= 0 AND urgency_score <= 1)")
    op.create_index('idx_tasks_email_id', 'tasks', ['email_id'])
    op.create_index('idx_tasks_task_type', 'tasks', ['task_type'])
    op.create_index('idx_tasks_complexity', 'tasks', ['complexity'])
    op.create_index('idx_tasks_deadline', 'tasks', ['deadline'])
    op.create_index('idx_tasks_urgency_score', 'tasks', ['urgency_score'])

    # projects
    op.create_table(
        'projects',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.Text, unique=True, nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('keywords', sa.ARRAY(sa.Text)),
        sa.Column('related_domains', sa.ARRAY(sa.Text)),
        sa.Column('is_active', sa.Boolean, server_default='true'),
        sa.Column('priority', sa.Integer, server_default='3'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_projects_name', 'projects', ['name'])
    op.create_index('idx_projects_is_active', 'projects', ['is_active'])

    # email_project_links
    op.create_table(
        'email_project_links',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE')),
        sa.Column('project_id', sa.Integer, sa.ForeignKey('projects.id', ondelete='CASCADE')),
        sa.Column('confidence', sa.Float),
        sa.Column('match_reason', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_epl_email_id', 'email_project_links', ['email_id'])
    op.create_index('idx_epl_project_id', 'email_project_links', ['project_id'])

    # priority_contexts
    op.create_table(
        'priority_contexts',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE')),
        sa.Column('context_type', sa.Text),
        sa.Column('context_value', sa.Text),
        sa.Column('weight', sa.Float),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_pc_email_id', 'priority_contexts', ['email_id'])
    op.create_index('idx_pc_context_type', 'priority_contexts', ['context_type'])

    # email_embeddings (matches compute_embeddings.py)
    # Use raw SQL because SQLAlchemy doesn't have native pgvector support
    op.execute("""
        CREATE TABLE email_embeddings (
            id SERIAL PRIMARY KEY,
            email_id INTEGER UNIQUE REFERENCES emails(id) ON DELETE CASCADE,
            embedding vector(1536),
            model TEXT DEFAULT 'text-embedding-3-small',
            token_count INTEGER,
            content_hash TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.create_index('idx_ee_email_id', 'email_embeddings', ['email_id'], unique=True)

    # email_llm_features
    op.create_table(
        'email_llm_features',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE'), unique=True),
        sa.Column('is_service_email', sa.Boolean),
        sa.Column('service_type', sa.Text),
        sa.Column('task_count', sa.Integer),
        sa.Column('has_deadline', sa.Boolean),
        sa.Column('urgency_level', sa.Text),
        sa.Column('summary', sa.Text),
        sa.Column('model_used', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_elf_email_id', 'email_llm_features', ['email_id'], unique=True)

    # human_task_labels
    op.create_table(
        'human_task_labels',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE')),
        sa.Column('task_index', sa.Integer),
        sa.Column('task_description', sa.Text),
        sa.Column('project_id', sa.Integer, sa.ForeignKey('projects.id')),
        sa.Column('project_relevancy', sa.Text),
        sa.Column('triage_category', sa.Text),
        sa.Column('extraction_quality', sa.Text),
        sa.Column('action', sa.Text),
        sa.Column('notes', sa.Text),
        sa.Column('labeler', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_htl_email_id', 'human_task_labels', ['email_id'])
    op.create_index('idx_htl_labeler', 'human_task_labels', ['labeler'])

    # email_ai_classification (matches classify_ai_handleability.py)
    op.create_table(
        'email_ai_classification',
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE'), primary_key=True),
        # Rule-based classification
        sa.Column('predicted_handleability', sa.Text),
        sa.Column('classification_reason', sa.Text),
        sa.Column('classification_metadata', JSONB),
        # Pattern flags
        sa.Column('has_question', sa.Boolean),
        sa.Column('has_request', sa.Boolean),
        sa.Column('has_scheduling', sa.Boolean),
        sa.Column('has_deadline', sa.Boolean),
        sa.Column('has_approval', sa.Boolean),
        sa.Column('has_confirm', sa.Boolean),
        sa.Column('is_newsletter', sa.Boolean),
        sa.Column('is_fyi', sa.Boolean),
        sa.Column('is_calendar_response', sa.Boolean),
        sa.Column('has_attachment_ref', sa.Boolean),
        # LLM processing flags
        sa.Column('needs_llm_classification', sa.Boolean),
        sa.Column('llm_priority', sa.Integer),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_eac_handleability', 'email_ai_classification', ['predicted_handleability'])
    op.create_index('idx_eac_needs_llm', 'email_ai_classification', ['needs_llm_classification'])
    op.create_index('idx_eac_llm_priority', 'email_ai_classification', ['llm_priority'])

    # email_llm_classification (matches run_llm_classification.py)
    op.create_table(
        'email_llm_classification',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE'), unique=True),
        sa.Column('thread_id', sa.Text),
        # Raw LLM data for ML training
        sa.Column('raw_prompt', sa.Text),
        sa.Column('raw_response', sa.Text),
        # Parsed classification
        sa.Column('action_type', sa.Text),
        sa.Column('urgency', sa.Text),
        sa.Column('ai_can_handle', sa.Text),
        sa.Column('next_step', sa.Text),
        sa.Column('suggested_action', sa.Text),
        sa.Column('one_liner', sa.Text),
        # Model metadata
        sa.Column('model', sa.Text),
        sa.Column('prompt_tokens', sa.Integer),
        sa.Column('completion_tokens', sa.Integer),
        sa.Column('total_tokens', sa.Integer),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_elc_email_id', 'email_llm_classification', ['email_id'], unique=True)
    op.create_index('idx_elc_action_type', 'email_llm_classification', ['action_type'])
    op.create_index('idx_elc_urgency', 'email_llm_classification', ['urgency'])


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.execute('DROP TABLE IF EXISTS email_llm_classification CASCADE')
    op.execute('DROP TABLE IF EXISTS email_ai_classification CASCADE')
    op.execute('DROP TABLE IF EXISTS human_task_labels CASCADE')
    op.execute('DROP TABLE IF EXISTS email_llm_features CASCADE')
    op.execute('DROP TABLE IF EXISTS email_embeddings CASCADE')
    op.execute('DROP TABLE IF EXISTS priority_contexts CASCADE')
    op.execute('DROP TABLE IF EXISTS email_project_links CASCADE')
    op.execute('DROP TABLE IF EXISTS projects CASCADE')
    op.execute('DROP TABLE IF EXISTS tasks CASCADE')
    op.execute('DROP TABLE IF EXISTS email_features CASCADE')
    op.execute('DROP TABLE IF EXISTS users CASCADE')
    op.execute('DROP TABLE IF EXISTS threads CASCADE')
    op.execute('DROP TABLE IF EXISTS attachments CASCADE')
    op.execute('DROP TABLE IF EXISTS emails CASCADE')
    op.execute('DROP TABLE IF EXISTS raw_emails CASCADE')
