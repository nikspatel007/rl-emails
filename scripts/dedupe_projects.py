#!/usr/bin/env python3
"""
Merge and deduplicate discovered projects.

This script:
1. Adds a merged_into column to track project merges
2. Identifies duplicate participant groups by participant overlap
3. Finds similar cluster projects by email overlap
4. Merges duplicates by pointing to a canonical project
5. Updates email_project_links to use canonical projects

Usage:
    python scripts/dedupe_projects.py [--dry-run]

The --dry-run flag shows what would be merged without making changes.
"""

import asyncio
import sys
from collections import defaultdict

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Thresholds
PARTICIPANT_OVERLAP_THRESHOLD = 0.7  # 70% participant overlap -> merge
EMAIL_OVERLAP_THRESHOLD = 0.5  # 50% email overlap within same source -> merge
MIN_EMAILS_FOR_CANONICAL = 10  # Prefer projects with more emails as canonical


async def ensure_merged_into_column(conn: asyncpg.Connection) -> bool:
    """Add merged_into column if it doesn't exist."""
    exists = await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'projects' AND column_name = 'merged_into'
        )
    """)

    if not exists:
        await conn.execute("""
            ALTER TABLE projects ADD COLUMN merged_into INTEGER REFERENCES projects(id)
        """)
        await conn.execute("""
            CREATE INDEX idx_projects_merged_into ON projects(merged_into)
        """)
        return True
    return False


async def column_exists(conn: asyncpg.Connection, column: str) -> bool:
    """Check if a column exists in the projects table."""
    return await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'projects' AND column_name = $1
        )
    """, column)


async def get_participant_projects(conn: asyncpg.Connection) -> list[dict]:
    """Get all participant projects with their key_people."""
    has_merged = await column_exists(conn, 'merged_into')
    if has_merged:
        rows = await conn.fetch("""
            SELECT id, name, key_people, email_count
            FROM projects
            WHERE source = 'participant' AND merged_into IS NULL
            ORDER BY email_count DESC
        """)
    else:
        rows = await conn.fetch("""
            SELECT id, name, key_people, email_count
            FROM projects
            WHERE source = 'participant'
            ORDER BY email_count DESC
        """)
    return [dict(row) for row in rows]


async def get_project_emails(conn: asyncpg.Connection, project_id: int) -> set[int]:
    """Get all email IDs linked to a project."""
    rows = await conn.fetch("""
        SELECT email_id FROM email_project_links WHERE project_id = $1
    """, project_id)
    return {row['email_id'] for row in rows}


def compute_participant_overlap(people1: list, people2: list) -> float:
    """Compute Jaccard similarity between two participant lists."""
    if not people1 or not people2:
        return 0.0
    set1 = set(people1) if people1 else set()
    set2 = set(people2) if people2 else set()
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_email_overlap(emails1: set[int], emails2: set[int]) -> float:
    """Compute Jaccard similarity between two email sets."""
    if not emails1 or not emails2:
        return 0.0
    intersection = len(emails1 & emails2)
    union = len(emails1 | emails2)
    return intersection / union if union > 0 else 0.0


async def find_participant_duplicates(conn: asyncpg.Connection) -> list[tuple[int, int]]:
    """Find participant projects that should be merged based on participant overlap."""
    projects = await get_participant_projects(conn)

    # Build merge pairs (smaller -> larger)
    merge_pairs = []
    merged_ids = set()

    for i, p1 in enumerate(projects):
        if p1['id'] in merged_ids:
            continue

        for j, p2 in enumerate(projects[i+1:], i+1):
            if p2['id'] in merged_ids:
                continue

            overlap = compute_participant_overlap(p1['key_people'], p2['key_people'])

            if overlap >= PARTICIPANT_OVERLAP_THRESHOLD:
                # Merge smaller into larger (by email count)
                if p1['email_count'] >= p2['email_count']:
                    merge_pairs.append((p2['id'], p1['id']))
                    merged_ids.add(p2['id'])
                else:
                    merge_pairs.append((p1['id'], p2['id']))
                    merged_ids.add(p1['id'])

    return merge_pairs


async def find_cluster_duplicates(conn: asyncpg.Connection) -> list[tuple[int, int]]:
    """Find cluster projects that have high email overlap."""
    # Get cluster projects
    has_merged = await column_exists(conn, 'merged_into')
    if has_merged:
        rows = await conn.fetch("""
            SELECT id, name, email_count
            FROM projects
            WHERE source = 'cluster' AND merged_into IS NULL
            ORDER BY email_count DESC
        """)
    else:
        rows = await conn.fetch("""
            SELECT id, name, email_count
            FROM projects
            WHERE source = 'cluster'
            ORDER BY email_count DESC
        """)
    projects = [dict(row) for row in rows]

    # Pre-fetch email sets for all clusters
    email_sets = {}
    for p in projects:
        email_sets[p['id']] = await get_project_emails(conn, p['id'])

    merge_pairs = []
    merged_ids = set()

    for i, p1 in enumerate(projects):
        if p1['id'] in merged_ids:
            continue

        for j, p2 in enumerate(projects[i+1:], i+1):
            if p2['id'] in merged_ids:
                continue

            overlap = compute_email_overlap(email_sets[p1['id']], email_sets[p2['id']])

            if overlap >= EMAIL_OVERLAP_THRESHOLD:
                # Merge smaller into larger
                if p1['email_count'] >= p2['email_count']:
                    merge_pairs.append((p2['id'], p1['id']))
                    merged_ids.add(p2['id'])
                else:
                    merge_pairs.append((p1['id'], p2['id']))
                    merged_ids.add(p1['id'])

    return merge_pairs


async def execute_merge(conn: asyncpg.Connection, from_id: int, into_id: int):
    """Merge one project into another."""
    # Update project's merged_into field
    await conn.execute("""
        UPDATE projects SET merged_into = $1 WHERE id = $2
    """, into_id, from_id)

    # Move email links to canonical project (update existing, skip conflicts)
    await conn.execute("""
        INSERT INTO email_project_links (email_id, project_id, confidence, source)
        SELECT email_id, $1, confidence, source
        FROM email_project_links
        WHERE project_id = $2
        ON CONFLICT (email_id, project_id) DO UPDATE SET
            confidence = GREATEST(email_project_links.confidence, EXCLUDED.confidence)
    """, into_id, from_id)

    # Update email count on canonical project
    await conn.execute("""
        UPDATE projects
        SET email_count = (
            SELECT COUNT(DISTINCT email_id)
            FROM email_project_links
            WHERE project_id = $1
        )
        WHERE id = $1
    """, into_id)


async def main(dry_run: bool = False):
    """Main entry point."""
    print("Merge and Deduplicate Projects")
    print("=" * 40)

    if dry_run:
        print("\n[DRY RUN - no changes will be made]\n")

    print(f"Connecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Ensure merged_into column exists
        if not dry_run:
            created = await ensure_merged_into_column(conn)
            if created:
                print("Added merged_into column to projects table")

        # Get initial counts
        total_projects = await conn.fetchval("SELECT COUNT(*) FROM projects")
        print(f"\nTotal projects: {total_projects}")

        # Find participant duplicates
        print("\nFinding participant group duplicates...")
        participant_merges = await find_participant_duplicates(conn)
        print(f"  Found {len(participant_merges)} pairs to merge")

        # Find cluster duplicates
        print("\nFinding cluster duplicates by email overlap...")
        cluster_merges = await find_cluster_duplicates(conn)
        print(f"  Found {len(cluster_merges)} pairs to merge")

        all_merges = participant_merges + cluster_merges

        if not all_merges:
            print("\nNo duplicates found!")
            return

        # Show merge preview
        print(f"\n=== Merge Preview ({len(all_merges)} total) ===")

        # Get project names for display
        all_ids = set()
        for from_id, into_id in all_merges:
            all_ids.add(from_id)
            all_ids.add(into_id)

        rows = await conn.fetch("""
            SELECT id, name, source, email_count FROM projects WHERE id = ANY($1)
        """, list(all_ids))
        project_info = {row['id']: dict(row) for row in rows}

        # Group by source for cleaner display
        participant_count = 0
        cluster_count = 0

        for from_id, into_id in all_merges[:15]:
            from_p = project_info.get(from_id, {})
            into_p = project_info.get(into_id, {})
            source = from_p.get('source', '?')

            if source == 'participant':
                participant_count += 1
            else:
                cluster_count += 1

            from_name = from_p.get('name', '?')[:35]
            into_name = into_p.get('name', '?')[:35]
            print(f"  [{source[:4]}] {from_name:<35} -> {into_name:<35}")

        if len(all_merges) > 15:
            print(f"  ... and {len(all_merges) - 15} more merges")

        print(f"\nSummary: {len(participant_merges)} participant + {len(cluster_merges)} cluster merges")

        if dry_run:
            print("\n[DRY RUN - stopping here]")
            return

        # Execute merges
        print("\nExecuting merges...")
        for i, (from_id, into_id) in enumerate(all_merges):
            await execute_merge(conn, from_id, into_id)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(all_merges)} merges...")

        print(f"\nCompleted {len(all_merges)} merges")

        # Verification
        print("\n=== Verification ===")
        active_projects = await conn.fetchval(
            "SELECT COUNT(*) FROM projects WHERE merged_into IS NULL"
        )
        merged_projects = await conn.fetchval(
            "SELECT COUNT(*) FROM projects WHERE merged_into IS NOT NULL"
        )
        print(f"Active (canonical) projects: {active_projects}")
        print(f"Merged (non-canonical) projects: {merged_projects}")

        # Stats by source
        rows = await conn.fetch("""
            SELECT source,
                   COUNT(*) FILTER (WHERE merged_into IS NULL) as active,
                   COUNT(*) FILTER (WHERE merged_into IS NOT NULL) as merged
            FROM projects
            GROUP BY source
            ORDER BY source
        """)
        print("\nBy source:")
        for row in rows:
            print(f"  {row['source']}: {row['active']} active, {row['merged']} merged")

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry_run))
