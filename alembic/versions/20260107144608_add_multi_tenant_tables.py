"""add_multi_tenant_tables

Revision ID: 20260107144608
Revises: 92f0657c25ef
Create Date: 2026-01-07 14:46:08

Add multi-tenant support tables for organizations, users, OAuth tokens,
and sync state. Also adds optional org_id/user_id columns to existing tables.

This migration is ADDITIVE and backward compatible:
- Existing single-tenant usage continues to work
- New multi-tenant features are opt-in via org_id/user_id
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


# revision identifiers, used by Alembic.
revision: str = "20260107144608"
down_revision: Union[str, Sequence[str], None] = "92f0657c25ef"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create multi-tenant tables and add tenant columns to existing tables."""

    # Enable UUID extension if not already enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # =================================================================
    # CORE MULTI-TENANT TABLES
    # =================================================================

    # organizations - Top-level tenant
    op.create_table(
        "organizations",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("slug", sa.Text, unique=True, nullable=False),
        sa.Column("settings", JSONB, server_default="{}"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_organizations_slug", "organizations", ["slug"], unique=True)

    # org_users - Users within an organization
    op.create_table(
        "org_users",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.Text, nullable=False),
        sa.Column("name", sa.Text),
        sa.Column("role", sa.Text, server_default="member"),
        sa.Column("gmail_connected", sa.Boolean, server_default="false"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("org_id", "email", name="uq_org_users_org_email"),
    )
    op.create_index("idx_org_users_org_id", "org_users", ["org_id"])
    op.create_index("idx_org_users_email", "org_users", ["email"])

    # oauth_tokens - OAuth2 tokens for Gmail API
    op.create_table(
        "oauth_tokens",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.Text, server_default="google"),
        sa.Column("access_token", sa.Text, nullable=False),
        sa.Column("refresh_token", sa.Text, nullable=False),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("scopes", sa.ARRAY(sa.Text)),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_oauth_tokens_user_id", "oauth_tokens", ["user_id"])
    op.create_index(
        "idx_oauth_tokens_user_provider",
        "oauth_tokens",
        ["user_id", "provider"],
        unique=True,
    )

    # sync_state - Gmail sync state per user
    op.create_table(
        "sync_state",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("last_history_id", sa.Text),
        sa.Column("last_sync_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("sync_status", sa.Text, server_default="idle"),
        sa.Column("error_message", sa.Text),
        sa.Column("emails_synced", sa.Integer, server_default="0"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_sync_state_user_id", "sync_state", ["user_id"], unique=True)
    op.create_index("idx_sync_state_status", "sync_state", ["sync_status"])

    # =================================================================
    # ADD MULTI-TENANT COLUMNS TO EXISTING TABLES
    # These columns are NULLABLE for backward compatibility
    # =================================================================

    # Add org_id and user_id to emails table
    op.add_column(
        "emails",
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "emails",
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "emails",
        sa.Column("gmail_id", sa.Text, nullable=True),
    )
    op.create_index("idx_emails_org_id", "emails", ["org_id"])
    op.create_index("idx_emails_user_id", "emails", ["user_id"])
    op.create_index(
        "idx_emails_user_gmail",
        "emails",
        ["user_id", "gmail_id"],
        unique=True,
        postgresql_where=sa.text("gmail_id IS NOT NULL"),
    )

    # Add org_id and user_id to raw_emails table
    op.add_column(
        "raw_emails",
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "raw_emails",
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.create_index("idx_raw_emails_org_id", "raw_emails", ["org_id"])
    op.create_index("idx_raw_emails_user_id", "raw_emails", ["user_id"])

    # Add org_id and user_id to threads table
    op.add_column(
        "threads",
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "threads",
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.create_index("idx_threads_org_id", "threads", ["org_id"])
    op.create_index("idx_threads_user_id", "threads", ["user_id"])

    # Add org_id and user_id to users table (contact profiles)
    op.add_column(
        "users",
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "owner_user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.create_index("idx_users_org_id", "users", ["org_id"])
    op.create_index("idx_users_owner_user_id", "users", ["owner_user_id"])

    # =================================================================
    # CLUSTER METADATA TABLE (for Phase 3 clustering enhancements)
    # =================================================================

    op.create_table(
        "email_clusters",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "email_id",
            sa.Integer,
            sa.ForeignKey("emails.id", ondelete="CASCADE"),
            unique=True,
        ),
        sa.Column("people_cluster_id", sa.Integer),
        sa.Column("content_cluster_id", sa.Integer),
        sa.Column("behavior_cluster_id", sa.Integer),
        sa.Column("service_cluster_id", sa.Integer),
        sa.Column("temporal_cluster_id", sa.Integer),
        # Cluster probability columns (used by stage_09)
        sa.Column("people_cluster_prob", sa.Float),
        sa.Column("content_cluster_prob", sa.Float),
        sa.Column("behavior_cluster_prob", sa.Float),
        sa.Column("service_cluster_prob", sa.Float),
        sa.Column("temporal_cluster_prob", sa.Float),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_email_clusters_email_id", "email_clusters", ["email_id"], unique=True)
    op.create_index("idx_email_clusters_people", "email_clusters", ["people_cluster_id"])
    op.create_index("idx_email_clusters_content", "email_clusters", ["content_cluster_id"])

    op.create_table(
        "cluster_metadata",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "org_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("org_users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("dimension", sa.Text, nullable=False),
        sa.Column("cluster_id", sa.Integer, nullable=False),
        sa.Column("size", sa.Integer),
        sa.Column("representative_email_id", sa.Integer, sa.ForeignKey("emails.id")),
        sa.Column("auto_label", sa.Text),
        sa.Column("pct_replied", sa.Float),
        sa.Column("avg_response_time_hours", sa.Float),
        sa.Column("avg_relationship_strength", sa.Float),
        sa.Column("is_project", sa.Boolean, server_default="false"),
        sa.Column("project_status", sa.Text),
        sa.Column("last_activity_at", sa.TIMESTAMP(timezone=True)),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_cluster_metadata_user_dim", "cluster_metadata", ["user_id", "dimension"])
    op.create_index(
        "idx_cluster_metadata_project",
        "cluster_metadata",
        ["user_id", "is_project", "last_activity_at"],
    )
    # Unique constraint for ON CONFLICT (used by stage_09)
    op.create_index(
        "idx_cluster_metadata_dimension_cluster",
        "cluster_metadata",
        ["dimension", "cluster_id"],
        unique=True,
    )

    # =================================================================
    # EMAIL PRIORITY TABLE (for Phase 3 priority scoring)
    # =================================================================

    op.create_table(
        "email_priority",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "email_id",
            sa.Integer,
            sa.ForeignKey("emails.id", ondelete="CASCADE"),
            unique=True,
        ),
        sa.Column("feature_score", sa.Float),
        sa.Column("replied_similarity", sa.Float),
        sa.Column("cluster_novelty", sa.Float),
        sa.Column("sender_novelty", sa.Float),
        sa.Column("priority_score", sa.Float),
        sa.Column("priority_rank", sa.Integer),
        # LLM analysis columns (used by stage_10)
        sa.Column("needs_llm_analysis", sa.Boolean, server_default="false"),
        sa.Column("llm_reason", sa.Text),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_email_priority_email_id", "email_priority", ["email_id"], unique=True)
    op.create_index("idx_email_priority_rank", "email_priority", ["priority_rank"])
    op.create_index("idx_email_priority_score", "email_priority", ["priority_score"])


def downgrade() -> None:
    """Remove multi-tenant tables and columns."""

    # Drop priority table
    op.drop_table("email_priority")

    # Drop cluster tables
    op.drop_table("cluster_metadata")
    op.drop_table("email_clusters")

    # Remove tenant columns from users table
    op.drop_index("idx_users_owner_user_id")
    op.drop_index("idx_users_org_id")
    op.drop_column("users", "owner_user_id")
    op.drop_column("users", "org_id")

    # Remove tenant columns from threads table
    op.drop_index("idx_threads_user_id")
    op.drop_index("idx_threads_org_id")
    op.drop_column("threads", "user_id")
    op.drop_column("threads", "org_id")

    # Remove tenant columns from raw_emails table
    op.drop_index("idx_raw_emails_user_id")
    op.drop_index("idx_raw_emails_org_id")
    op.drop_column("raw_emails", "user_id")
    op.drop_column("raw_emails", "org_id")

    # Remove tenant columns from emails table
    op.drop_index("idx_emails_user_gmail")
    op.drop_index("idx_emails_user_id")
    op.drop_index("idx_emails_org_id")
    op.drop_column("emails", "gmail_id")
    op.drop_column("emails", "user_id")
    op.drop_column("emails", "org_id")

    # Drop multi-tenant tables
    op.drop_table("sync_state")
    op.drop_table("oauth_tokens")
    op.drop_table("org_users")
    op.drop_table("organizations")
