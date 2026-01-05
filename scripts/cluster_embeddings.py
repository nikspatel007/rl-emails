#!/usr/bin/env python3
"""
Cluster emails by embedding similarity for project discovery.

This script:
1. Fetches email embeddings from PostgreSQL
2. Runs KMeans clustering to find semantic topic groups
3. Analyzes each cluster to generate descriptive names
4. Creates project records with source='cluster'
5. Links emails to their cluster projects

Usage:
    python scripts/cluster_embeddings.py [--n-clusters N] [--dry-run]

Options:
    --n-clusters N  Number of clusters (default: 30)
    --dry-run       Show what would be created without making changes
"""

import argparse
import asyncio
import sys
from collections import Counter

import asyncpg
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"
DEFAULT_N_CLUSTERS = 30
MIN_CLUSTER_SIZE = 10  # Skip clusters smaller than this


async def fetch_embeddings(conn: asyncpg.Connection) -> tuple[list[int], np.ndarray]:
    """Fetch all embeddings from the database."""
    print("Fetching embeddings...")

    rows = await conn.fetch("""
        SELECT ee.email_id, ee.embedding::text
        FROM email_embeddings ee
        WHERE ee.email_id IS NOT NULL
        ORDER BY ee.email_id
    """)

    email_ids = []
    embeddings = []

    for row in rows:
        email_ids.append(row["email_id"])
        # Parse vector string "[0.1, 0.2, ...]" to numpy array
        vec_str = row["embedding"].strip("[]")
        vec = np.fromstring(vec_str, sep=",")
        embeddings.append(vec)

    return email_ids, np.array(embeddings)


async def get_cluster_samples(
    conn: asyncpg.Connection, email_ids: list[int], n_samples: int = 5
) -> list[dict]:
    """Get sample emails from a cluster for naming."""
    if not email_ids:
        return []

    sample_ids = email_ids[:n_samples]
    rows = await conn.fetch(
        """
        SELECT id, subject, from_email, body_preview
        FROM emails
        WHERE id = ANY($1)
    """,
        sample_ids,
    )

    return [dict(row) for row in rows]


def generate_cluster_name(samples: list[dict], cluster_id: int) -> tuple[str, str]:
    """
    Generate a name and description for a cluster based on sample emails.

    Returns (name, description)
    """
    if not samples:
        return f"Cluster {cluster_id}", "No samples available"

    # Extract subjects and senders
    subjects = [s.get("subject", "") or "" for s in samples]
    senders = [s.get("from_email", "") or "" for s in samples]

    # Find common words in subjects
    all_words = []
    for subj in subjects:
        words = [w.lower() for w in subj.split() if len(w) > 3]
        all_words.extend(words)

    word_counts = Counter(all_words)
    common_words = [w for w, c in word_counts.most_common(3) if c > 1]

    # Find common sender domains
    domains = []
    for sender in senders:
        if "@" in sender:
            domain = sender.split("@")[1].split(".")[0]
            if domain not in ("gmail", "yahoo", "hotmail", "outlook"):
                domains.append(domain)

    domain_counts = Counter(domains)
    common_domains = [d for d, c in domain_counts.most_common(2) if c > 1]

    # Build name
    if common_words:
        name = " ".join(common_words[:2]).title()
    elif common_domains:
        name = f"From {common_domains[0].title()}"
    else:
        # Use first subject as fallback
        first_subj = subjects[0][:40] if subjects else f"Topic {cluster_id}"
        name = first_subj.strip()

    # Clean up name
    name = name.strip()
    if not name:
        name = f"Cluster {cluster_id}"

    # Limit length
    if len(name) > 50:
        name = name[:47] + "..."

    # Build description
    desc_parts = []
    if common_words:
        desc_parts.append(f"Common terms: {', '.join(common_words)}")
    if common_domains:
        desc_parts.append(f"Common senders: {', '.join(common_domains)}")

    sample_subjects = [s[:50] for s in subjects[:3] if s]
    if sample_subjects:
        desc_parts.append(f"Sample subjects: {'; '.join(sample_subjects)}")

    description = ". ".join(desc_parts) if desc_parts else "Semantically similar emails"

    return name, description


async def create_project(
    conn: asyncpg.Connection,
    name: str,
    description: str,
    cluster_id: int,
    email_count: int,
    first_date,
    last_date,
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
        name,
        description,
        "cluster",
        f"kmeans_cluster_{cluster_id}",
        "topic",
        "active",
        email_count,
        first_date,
        last_date,
    )
    return project_id


async def link_emails_to_project(
    conn: asyncpg.Connection, email_ids: list[int], project_id: int, confidences: list[float]
) -> int:
    """Create email_project_links for emails in this cluster."""
    # Batch insert
    await conn.executemany(
        """
        INSERT INTO email_project_links (email_id, project_id, confidence, source)
        VALUES ($1, $2, $3, 'cluster')
        ON CONFLICT (email_id, project_id) DO NOTHING
    """,
        [(eid, project_id, conf) for eid, conf in zip(email_ids, confidences)],
    )
    return len(email_ids)


async def get_cluster_date_range(
    conn: asyncpg.Connection, email_ids: list[int]
) -> tuple:
    """Get first and last email dates for a cluster."""
    if not email_ids:
        return None, None

    row = await conn.fetchrow(
        """
        SELECT MIN(date_parsed) as first_date, MAX(date_parsed) as last_date
        FROM emails
        WHERE id = ANY($1)
    """,
        email_ids,
    )

    return row["first_date"], row["last_date"]


def compute_cluster_confidences(
    embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """
    Compute confidence scores based on distance to cluster center.

    Returns normalized confidence (0-1) where 1 = closest to center.
    """
    # Compute distances to assigned cluster centers
    distances = np.zeros(len(embeddings))
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        distances[i] = np.linalg.norm(emb - centers[label])

    # Normalize to 0-1 (inverse: closer = higher confidence)
    max_dist = distances.max()
    if max_dist > 0:
        confidences = 1 - (distances / max_dist)
    else:
        confidences = np.ones(len(embeddings))

    return confidences


async def main(n_clusters: int = DEFAULT_N_CLUSTERS, dry_run: bool = False):
    """Main entry point."""
    print("Email Embedding Clustering for Project Discovery")
    print("=" * 50)

    if dry_run:
        print("\n[DRY RUN - no changes will be made]\n")

    print(f"Connecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Fetch embeddings
        email_ids, embeddings = await fetch_embeddings(conn)
        print(f"Loaded {len(email_ids)} embeddings ({embeddings.shape[1]} dimensions)")

        if len(email_ids) < n_clusters:
            print(f"Warning: Only {len(email_ids)} emails, reducing clusters")
            n_clusters = max(5, len(email_ids) // 10)

        # Run KMeans clustering
        print(f"\nRunning KMeans with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Compute silhouette score (quality metric)
        if len(set(labels)) > 1:
            sil_score = silhouette_score(embeddings, labels, sample_size=min(5000, len(labels)))
            print(f"Silhouette score: {sil_score:.3f}")

        # Compute confidences
        confidences = compute_cluster_confidences(embeddings, labels, kmeans.cluster_centers_)

        # Analyze clusters
        print(f"\nAnalyzing {n_clusters} clusters...")
        cluster_info = []

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_email_ids = [eid for eid, m in zip(email_ids, mask) if m]
            cluster_confidences = confidences[mask].tolist()

            if len(cluster_email_ids) < MIN_CLUSTER_SIZE:
                continue

            # Get samples for naming
            samples = await get_cluster_samples(conn, cluster_email_ids)
            name, description = generate_cluster_name(samples, cluster_id)

            # Get date range
            first_date, last_date = await get_cluster_date_range(conn, cluster_email_ids)

            cluster_info.append({
                "cluster_id": cluster_id,
                "name": name,
                "description": description,
                "email_ids": cluster_email_ids,
                "confidences": cluster_confidences,
                "email_count": len(cluster_email_ids),
                "first_date": first_date,
                "last_date": last_date,
            })

        # Sort by size
        cluster_info.sort(key=lambda x: x["email_count"], reverse=True)

        print(f"\nValid clusters (>= {MIN_CLUSTER_SIZE} emails): {len(cluster_info)}")

        # Show what will be created
        print("\nClusters to create as projects:")
        print("-" * 70)
        for ci in cluster_info[:20]:  # Show top 20
            print(f"  {ci['name']:<40} [{ci['email_count']:>5} emails]")
        if len(cluster_info) > 20:
            print(f"  ... and {len(cluster_info) - 20} more clusters")

        if dry_run:
            print("\n[DRY RUN - stopping here]")
            return

        # Create projects and link emails
        print("\nCreating projects...")
        total_links = 0

        for ci in cluster_info:
            # Create project
            project_id = await create_project(
                conn,
                ci["name"],
                ci["description"],
                ci["cluster_id"],
                ci["email_count"],
                ci["first_date"],
                ci["last_date"],
            )
            print(f"  Created: {ci['name'][:40]:<40} (id={project_id}, {ci['email_count']} emails)")

            # Link emails
            links_created = await link_emails_to_project(
                conn, ci["email_ids"], project_id, ci["confidences"]
            )
            total_links += links_created

        print(f"\n=== Summary ===")
        print(f"Projects created: {len(cluster_info)}")
        print(f"Email links created: {total_links}")

        # Verify
        print("\n=== Verification ===")
        project_count = await conn.fetchval(
            "SELECT COUNT(*) FROM projects WHERE source = 'cluster'"
        )
        link_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM email_project_links epl
            JOIN projects p ON p.id = epl.project_id
            WHERE p.source = 'cluster'
        """
        )
        print(f"Total cluster projects: {project_count}")
        print(f"Total email links: {link_count}")

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster emails by embedding similarity")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help=f"Number of clusters (default: {DEFAULT_N_CLUSTERS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without making changes",
    )
    args = parser.parse_args()

    asyncio.run(main(n_clusters=args.n_clusters, dry_run=args.dry_run))
