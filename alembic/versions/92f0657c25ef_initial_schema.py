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


# revision identifiers, used by Alembic.
revision: str = '92f0657c25ef'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema for rl-emails database."""

    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')

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
    op.create_table(
        'email_features',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email_id', sa.Integer, sa.ForeignKey('emails.id', ondelete='CASCADE'), unique=True),
        sa.Column('message_id', sa.Text, nullable=False),
        sa.Column('sender_response_deviation', sa.Float),
        sa.Column('sender_frequency_rank', sa.Float),
        sa.Column('inferred_hierarchy', sa.Float),
        sa.Column('relationship_strength', sa.Float),
        sa.Column('emails_from_sender_7d', sa.Integer, server_default='0'),
        sa.Column('emails_from_sender_30d', sa.Integer, server_default='0'),
        sa.Column('emails_from_sender_90d', sa.Integer, server_default='0'),
        sa.Column('response_rate_to_sender', sa.Float),
        sa.Column('avg_thread_depth', sa.Float),
        sa.Column('days_since_last_email', sa.Float),
        sa.Column('cc_affinity_score', sa.Float),
        sa.Column('is_service_email', sa.Boolean, server_default='false'),
        sa.Column('service_type', sa.Text),
        sa.Column('service_email_confidence', sa.Float),
        sa.Column('has_list_unsubscribe_header', sa.Boolean, server_default='false'),
        sa.Column('has_unsubscribe_url', sa.Boolean, server_default='false'),
        sa.Column('unsubscribe_phrase_count', sa.Integer, server_default='0'),
        sa.Column('task_count', sa.Integer, server_default='0'),
        sa.Column('has_deadline', sa.Boolean, server_default='false'),
        sa.Column('deadline_urgency', sa.Float),
        sa.Column('is_assigned_to_user', sa.Boolean, server_default='false'),
        sa.Column('estimated_effort', sa.Text),
        sa.Column('has_deliverable', sa.Boolean, server_default='false'),
        sa.Column('urgency_score', sa.Float),
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
    op.create_index('idx_email_features_message_id', 'email_features', ['message_id'])
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


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.execute('DROP TABLE IF EXISTS tasks CASCADE')
    op.execute('DROP TABLE IF EXISTS email_features CASCADE')
    op.execute('DROP TABLE IF EXISTS users CASCADE')
    op.execute('DROP TABLE IF EXISTS threads CASCADE')
    op.execute('DROP TABLE IF EXISTS attachments CASCADE')
    op.execute('DROP TABLE IF EXISTS emails CASCADE')
    op.execute('DROP TABLE IF EXISTS raw_emails CASCADE')
