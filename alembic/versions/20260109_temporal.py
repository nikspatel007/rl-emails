"""Add temporal strength fields to cluster_metadata.

Revision ID: 20260109_temporal
Revises: 20260109_cluster_meta
Create Date: 2026-01-09

"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260109_temporal"
down_revision: str = "20260109_cluster_meta"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add temporal strength fields to cluster_metadata."""
    op.execute(
        """
        DO $$
        BEGIN
            -- Add temporal_strength column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'temporal_strength'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN temporal_strength FLOAT;
            END IF;

            -- Add activity_velocity column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'activity_velocity'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN activity_velocity FLOAT;
            END IF;

            -- Add activity_trend column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'activity_trend'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN activity_trend FLOAT;
            END IF;

            -- Add first_activity_at column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'first_activity_at'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN first_activity_at TIMESTAMPTZ;
            END IF;

            -- Add emails_last_7d column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'emails_last_7d'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN emails_last_7d INTEGER;
            END IF;

            -- Add emails_last_30d column if not exists
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'cluster_metadata' AND column_name = 'emails_last_30d'
            ) THEN
                ALTER TABLE cluster_metadata ADD COLUMN emails_last_30d INTEGER;
            END IF;

            -- Create index for temporal queries
            CREATE INDEX IF NOT EXISTS idx_cluster_metadata_temporal
                ON cluster_metadata(temporal_strength DESC NULLS LAST);
        END $$;
        """
    )


def downgrade() -> None:
    """Remove temporal strength fields from cluster_metadata."""
    op.execute(
        """
        DO $$
        BEGIN
            DROP INDEX IF EXISTS idx_cluster_metadata_temporal;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS emails_last_30d;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS emails_last_7d;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS first_activity_at;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS activity_trend;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS activity_velocity;
            ALTER TABLE cluster_metadata DROP COLUMN IF EXISTS temporal_strength;
        END $$;
        """
    )
