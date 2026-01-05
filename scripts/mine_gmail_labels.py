#!/usr/bin/env python3
"""
Mine Gmail labels for project discovery.

This script:
1. Extracts all unique Gmail labels from the emails table
2. Filters out system labels (CATEGORY_*, INBOX, SENT, etc.)
3. Creates project records for meaningful labels
4. Links emails to their projects via email_project_links

Usage:
    python scripts/mine_gmail_labels.py [--dry-run]

The --dry-run flag shows what would be created without making changes.
"""

import asyncio
import sys
from datetime import datetime

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Gmail system labels to exclude (not user-created, not useful as projects)
SYSTEM_LABELS = {
    "INBOX",
    "SENT",
    "TRASH",
    "SPAM",
    "DRAFTS",
    "STARRED",
    "UNREAD",
    "OPENED",
    "ARCHIVED",
    "SNOOZED",
    "IMPORTANT",
}

# Gmail category labels to exclude
CATEGORY_PREFIX = "CATEGORY "


def classify_label(label: str) -> dict | None:
    """
    Classify a label and return project metadata.

    Returns None for labels that should be skipped.
    Returns dict with: name, project_type, description
    """
    # Skip system labels
    if label in SYSTEM_LABELS:
        return None

    # Skip category labels (e.g., "CATEGORY UPDATES")
    if label.startswith(CATEGORY_PREFIX):
        return None

    # Superhuman AI labels -> topic type
    if label.startswith("[SUPERHUMAN]/AI/"):
        ai_type = label.replace("[SUPERHUMAN]/AI/", "")
        return {
            "name": f"AI: {ai_type.title()}",
            "project_type": "topic",
            "description": f"Emails classified as {ai_type.lower()} by Superhuman AI",
            "source_detail": label,
        }

    # Custom user labels -> project type
    return {
        "name": label.title(),
        "project_type": "project",
        "description": f"User-created Gmail label: {label}",
        "source_detail": label,
    }


async def get_labels_with_counts(conn: asyncpg.Connection) -> list[dict]:
    """Get all labels with their email counts and date ranges."""
    rows = await conn.fetch("""
        SELECT
            unnest(labels) as label,
            COUNT(*) as email_count,
            MIN(date_parsed) as first_activity,
            MAX(date_parsed) as last_activity
        FROM emails
        WHERE labels IS NOT NULL
        GROUP BY label
        ORDER BY COUNT(*) DESC
    """)
    return [dict(row) for row in rows]


async def check_existing_projects(conn: asyncpg.Connection) -> set[str]:
    """Get source_detail values for existing gmail_label projects."""
    rows = await conn.fetch("""
        SELECT source_detail FROM projects WHERE source = 'gmail_label'
    """)
    return {row["source_detail"] for row in rows}


async def create_project(
    conn: asyncpg.Connection, label_data: dict, classification: dict
) -> int:
    """Create a project record and return its ID."""
    project_id = await conn.fetchval(
        """
        INSERT INTO projects (
            name, description, source, source_detail,
            project_type, status, email_count,
            started_at, last_activity, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
        RETURNING id
    """,
        classification["name"],
        classification["description"],
        "gmail_label",
        classification["source_detail"],
        classification["project_type"],
        "active",
        label_data["email_count"],
        label_data["first_activity"],
        label_data["last_activity"],
    )
    return project_id


async def link_emails_to_project(
    conn: asyncpg.Connection, label: str, project_id: int
) -> int:
    """Create email_project_links for all emails with this label."""
    result = await conn.execute(
        """
        INSERT INTO email_project_links (email_id, project_id, confidence, source)
        SELECT id, $1, 1.0, 'label'
        FROM emails
        WHERE $2 = ANY(labels)
        ON CONFLICT (email_id, project_id) DO NOTHING
    """,
        project_id,
        label,
    )
    # Parse "INSERT 0 N" to get count
    count = int(result.split()[-1]) if result else 0
    return count


async def main(dry_run: bool = False):
    """Main entry point."""
    print("Mine Gmail Labels for Project Discovery")
    print("=" * 45)

    if dry_run:
        print("\n[DRY RUN - no changes will be made]\n")

    print(f"Connecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Get all labels with counts
        print("\nQuerying labels...")
        labels = await get_labels_with_counts(conn)
        print(f"Found {len(labels)} unique labels")

        # Check existing projects
        existing = await check_existing_projects(conn)
        if existing:
            print(f"Already have {len(existing)} gmail_label projects")

        # Classify labels
        projects_to_create = []
        for label_data in labels:
            label = label_data["label"]

            # Skip if already exists
            if label in existing:
                continue

            classification = classify_label(label)
            if classification:
                projects_to_create.append((label_data, classification))

        print(f"\nLabels to convert to projects: {len(projects_to_create)}")

        if not projects_to_create:
            print("Nothing to do!")
            return

        # Show what will be created
        print("\nProjects to create:")
        print("-" * 60)
        for label_data, classification in projects_to_create:
            print(
                f"  {classification['name']:<30} "
                f"({classification['project_type']}) "
                f"[{label_data['email_count']} emails]"
            )

        if dry_run:
            print("\n[DRY RUN - stopping here]")
            return

        # Create projects and link emails
        print("\nCreating projects...")
        total_links = 0

        for label_data, classification in projects_to_create:
            label = label_data["label"]

            # Create project
            project_id = await create_project(conn, label_data, classification)
            print(f"  Created: {classification['name']} (id={project_id})")

            # Link emails
            links_created = await link_emails_to_project(conn, label, project_id)
            total_links += links_created
            print(f"    Linked {links_created} emails")

        print(f"\n=== Summary ===")
        print(f"Projects created: {len(projects_to_create)}")
        print(f"Email links created: {total_links}")

        # Verify
        print("\n=== Verification ===")
        project_count = await conn.fetchval(
            "SELECT COUNT(*) FROM projects WHERE source = 'gmail_label'"
        )
        link_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM email_project_links epl
            JOIN projects p ON p.id = epl.project_id
            WHERE p.source = 'gmail_label'
        """
        )
        print(f"Total gmail_label projects: {project_count}")
        print(f"Total email links: {link_count}")

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry_run))
