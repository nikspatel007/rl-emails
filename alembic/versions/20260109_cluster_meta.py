"""Enhance cluster_metadata table with project detection fields.

Revision ID: 20260109_cluster_meta
Revises: 20260109_watch
Create Date: 2026-01-09

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    pass

# revision identifiers, used by Alembic.
revision: str = "20260109_cluster_meta"
down_revision: str = "20260109_watch"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add project detection and multi-tenant fields to cluster_metadata."""
    # Add new columns to cluster_metadata table
    # Using raw SQL since table may or may not exist (created by stage_09)
    op.execute(
        """
        DO $$
        BEGIN
            -- Create table if it doesn't exist
            CREATE TABLE IF NOT EXISTS cluster_metadata (
                id SERIAL PRIMARY KEY,
                dimension TEXT,
                cluster_id INTEGER,
                size INTEGER,
                representative_email_id INTEGER,
                auto_label TEXT,
                pct_replied FLOAT,
                avg_response_time_hours FLOAT,
                avg_relationship_strength FLOAT,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(dimension, cluster_id)
            );

            -- Add org_id column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'org_id'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN org_id UUID;
            END IF;

            -- Add user_id column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'user_id'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN user_id UUID;
            END IF;

            -- Add is_project column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'is_project'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN is_project BOOLEAN DEFAULT FALSE;
            END IF;

            -- Add project_status column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'project_status'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN project_status TEXT;
            END IF;

            -- Add last_activity_at column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'last_activity_at'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN last_activity_at TIMESTAMPTZ;
            END IF;

            -- Add coherence_score column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'coherence_score'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN coherence_score FLOAT;
            END IF;

            -- Add participant_count column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'participant_count'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN participant_count INTEGER;
            END IF;

            -- Add updated_at column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'updated_at'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
            END IF;

            -- Update unique constraint to include org_id and user_id for multi-tenant
            -- Drop old constraint and create new one
            ALTER TABLE cluster_metadata DROP CONSTRAINT IF EXISTS cluster_metadata_dimension_cluster_id_key;

            -- Create index for multi-tenant queries
            CREATE INDEX IF NOT EXISTS idx_cluster_metadata_user_dimension
                ON cluster_metadata(user_id, dimension);

            -- Create index for project queries
            CREATE INDEX IF NOT EXISTS idx_cluster_metadata_project
                ON cluster_metadata(user_id, is_project, last_activity_at DESC);

            -- Create new unique constraint including user_id
            CREATE UNIQUE INDEX IF NOT EXISTS idx_cluster_metadata_unique
                ON cluster_metadata(COALESCE(user_id, '00000000-0000-0000-0000-000000000000'::UUID), dimension, cluster_id);
        END $$;
        """
    )


def downgrade() -> None:
    """Remove project detection fields from cluster_metadata."""
    op.execute(
        """
        DO $$
        BEGIN
            -- Drop indexes
            DROP INDEX IF EXISTS idx_cluster_metadata_unique;
            DROP INDEX IF EXISTS idx_cluster_metadata_project;
            DROP INDEX IF EXISTS idx_cluster_metadata_user_dimension;

            -- Drop new columns
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS updated_at;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS participant_count;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS coherence_score;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS last_activity_at;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS project_status;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS is_project;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS user_id;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS org_id;

            -- Restore original unique constraint
            ALTER TABLE cluster_metadata
                ADD CONSTRAINT cluster_metadata_dimension_cluster_id_key
                UNIQUE (dimension, cluster_id);
        END $$;
        """
    )
