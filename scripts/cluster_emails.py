#!/usr/bin/env python3
"""Multi-dimensional email clustering for Phase 4A.

Performs clustering across 5 dimensions:
1. People - relationship patterns
2. Content - semantic similarity (embeddings)
3. Behavior - action patterns
4. Service - automated email types
5. Temporal - time patterns

Usage:
    python scripts/cluster_emails.py                    # Run all dimensions
    python scripts/cluster_emails.py --dimension people # Run specific dimension
    python scripts/cluster_emails.py --analyze          # Show cluster analysis
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import Any

import numpy as np
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

load_dotenv()

try:
    import hdbscan  # type: ignore[import-not-found]
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    import umap  # type: ignore[import-not-found]
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Configuration from environment
DB_URL = os.environ.get("DATABASE_URL")


# =============================================================================
# Database Setup
# =============================================================================

def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create clustering tables if they don't exist."""
    with conn.cursor() as cur:
        # Main clustering results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_clusters (
                id SERIAL PRIMARY KEY,
                email_id INTEGER REFERENCES emails(id),

                -- Cluster assignments per dimension
                people_cluster_id INTEGER,
                content_cluster_id INTEGER,
                behavior_cluster_id INTEGER,
                service_cluster_id INTEGER,
                temporal_cluster_id INTEGER,

                -- Confidence scores
                people_cluster_prob FLOAT,
                content_cluster_prob FLOAT,

                created_at TIMESTAMP DEFAULT NOW(),

                UNIQUE(email_id)
            )
        """)

        # Cluster metadata
        cur.execute("""
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
            )
        """)

        conn.commit()
        print("✓ Tables created/verified")


# =============================================================================
# Data Loading
# =============================================================================

def load_people_features(conn: psycopg2.extensions.connection) -> tuple[list[int], np.ndarray]:
    """Load features for people clustering."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ef.email_id,
                COALESCE(ef.relationship_strength, 0) as relationship_strength,
                COALESCE(ef.user_replied_to_sender_rate, 0) as reply_rate,
                COALESCE(ef.avg_response_time_hours, 168) as avg_response_hours,
                COALESCE(ef.emails_from_sender_all, 1) as emails_from_sender,
                COALESCE(ef.sender_replies_to_you_rate, 0) as sender_reply_rate,
                CASE WHEN ef.is_service_email THEN 1 ELSE 0 END as is_service
            FROM email_features ef
            JOIN emails e ON e.id = ef.email_id
            WHERE e.is_sent = FALSE
            ORDER BY ef.email_id
        """)
        rows = cur.fetchall()

    email_ids = [r[0] for r in rows]
    features = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
    return email_ids, features


def load_embeddings(conn: psycopg2.extensions.connection) -> tuple[list[int], np.ndarray]:
    """Load embeddings for content clustering."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT email_id, embedding::text
            FROM email_embeddings
            ORDER BY email_id
        """)
        rows = cur.fetchall()

    email_ids = [r[0] for r in rows]
    embeddings = []
    for r in rows:
        # Parse embedding from string "[0.1, 0.2, ...]"
        emb_str = r[1].strip("[]")
        emb = [float(x) for x in emb_str.split(",")]
        embeddings.append(emb)

    return email_ids, np.array(embeddings)


def load_behavior_features(conn: psycopg2.extensions.connection) -> tuple[list[int], np.ndarray]:
    """Load features for behavior clustering."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                e.id,
                CASE e.action
                    WHEN 'REPLIED' THEN 1
                    WHEN 'ARCHIVED' THEN 2
                    WHEN 'IGNORED' THEN 3
                    WHEN 'PENDING' THEN 4
                    ELSE 5
                END as action_code,
                COALESCE(e.response_time_seconds / 3600.0, 168) as response_hours,
                CASE e.timing
                    WHEN 'immediate' THEN 1
                    WHEN 'same_day' THEN 2
                    WHEN 'next_day' THEN 3
                    WHEN 'delayed' THEN 4
                    ELSE 5
                END as timing_code
            FROM emails e
            WHERE e.is_sent = FALSE
            ORDER BY e.id
        """)
        rows = cur.fetchall()

    email_ids = [r[0] for r in rows]
    features = np.array([[r[1], r[2], r[3]] for r in rows])
    return email_ids, features


def load_service_features(conn: psycopg2.extensions.connection) -> tuple[list[int], np.ndarray]:
    """Load features for service clustering."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ef.email_id,
                CASE WHEN ef.is_service_email THEN 1 ELSE 0 END as is_service,
                COALESCE(ef.service_importance, 0.5) as service_importance,
                CASE WHEN ef.has_unsubscribe_link THEN 1 ELSE 0 END as has_unsub,
                CASE WHEN ef.from_common_service_domain THEN 1 ELSE 0 END as common_domain,
                CASE ef.service_type
                    WHEN 'transactional' THEN 1
                    WHEN 'notification' THEN 2
                    WHEN 'newsletter' THEN 3
                    WHEN 'marketing' THEN 4
                    WHEN 'social' THEN 5
                    ELSE 0
                END as service_type_code
            FROM email_features ef
            JOIN emails e ON e.id = ef.email_id
            WHERE e.is_sent = FALSE
            ORDER BY ef.email_id
        """)
        rows = cur.fetchall()

    email_ids = [r[0] for r in rows]
    features = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    return email_ids, features


def load_temporal_features(conn: psycopg2.extensions.connection) -> tuple[list[int], np.ndarray]:
    """Load features for temporal clustering."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ef.email_id,
                ef.hour_of_day,
                ef.day_of_week,
                CASE WHEN ef.is_business_hours THEN 1 ELSE 0 END as is_business,
                CASE WHEN ef.is_weekend THEN 1 ELSE 0 END as is_weekend
            FROM email_features ef
            JOIN emails e ON e.id = ef.email_id
            WHERE e.is_sent = FALSE
            ORDER BY ef.email_id
        """)
        rows = cur.fetchall()

    email_ids = [r[0] for r in rows]
    features = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    return email_ids, features


# =============================================================================
# Clustering Functions
# =============================================================================

def cluster_people(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by relationship patterns."""
    print("\n" + "=" * 60)
    print("PEOPLE CLUSTERING")
    print("=" * 60)

    email_ids, features = load_people_features(conn)
    print(f"Loaded {len(email_ids)} emails with {features.shape[1]} features")

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means with k=15 (relationship tiers)
    n_clusters = 15
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    # Compute distances to centroids for confidence
    distances = kmeans.transform(features_scaled)
    min_distances = distances.min(axis=1)
    max_dist = min_distances.max()
    probs = 1 - (min_distances / max_dist)  # Inverse distance as probability

    print(f"Created {n_clusters} people clusters")

    # Show cluster distribution
    cluster_counts = Counter(labels)
    print("\nCluster sizes:")
    for cid, count in sorted(cluster_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  Cluster {cid}: {count} emails")

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": probs,
        "n_clusters": n_clusters
    }


def cluster_content(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by semantic similarity (embeddings)."""
    print("\n" + "=" * 60)
    print("CONTENT CLUSTERING")
    print("=" * 60)

    email_ids, embeddings = load_embeddings(conn)
    print(f"Loaded {len(email_ids)} embeddings ({embeddings.shape[1]} dims)")

    # Reduce dimensionality with UMAP for HDBSCAN
    if HAS_UMAP:
        print("Reducing dimensions with UMAP...")
        reducer = umap.UMAP(n_components=50, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_reduced = reducer.fit_transform(embeddings)
        print(f"Reduced to {embeddings_reduced.shape[1]} dimensions")
    else:
        print("UMAP not available, using raw embeddings (slower)")
        embeddings_reduced = embeddings

    # HDBSCAN for content clustering
    if HAS_HDBSCAN:
        print("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(embeddings_reduced)
        probs = clusterer.probabilities_
    else:
        print("HDBSCAN not available, using K-Means fallback")
        n_clusters = 100
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_reduced)
        distances = kmeans.transform(embeddings_reduced)
        min_distances = distances.min(axis=1)
        probs = 1 - (min_distances / min_distances.max())

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"Created {n_clusters} content clusters")
    print(f"Noise points (outliers): {n_noise}")

    # Show cluster distribution
    cluster_counts = Counter(labels)
    print("\nLargest clusters:")
    for cid, count in sorted(cluster_counts.items(), key=lambda x: -x[1])[:10]:
        if cid == -1:
            print(f"  Noise: {count} emails")
        else:
            print(f"  Cluster {cid}: {count} emails")

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": probs,
        "n_clusters": n_clusters
    }


def cluster_behavior(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by action patterns."""
    print("\n" + "=" * 60)
    print("BEHAVIOR CLUSTERING")
    print("=" * 60)

    email_ids, features = load_behavior_features(conn)
    print(f"Loaded {len(email_ids)} emails with {features.shape[1]} features")

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means with k=8 (behavior patterns)
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    print(f"Created {n_clusters} behavior clusters")

    # Show cluster distribution
    cluster_counts = Counter(labels)
    print("\nCluster sizes:")
    for cid, count in sorted(cluster_counts.items(), key=lambda x: -x[1]):
        print(f"  Cluster {cid}: {count} emails")

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": np.ones(len(labels)),  # K-Means doesn't give probs
        "n_clusters": n_clusters
    }


def cluster_service(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by service type."""
    print("\n" + "=" * 60)
    print("SERVICE CLUSTERING")
    print("=" * 60)

    email_ids, features = load_service_features(conn)
    print(f"Loaded {len(email_ids)} emails with {features.shape[1]} features")

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means with k=8 (service types + person)
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    print(f"Created {n_clusters} service clusters")

    # Show cluster distribution
    cluster_counts = Counter(labels)
    print("\nCluster sizes:")
    for cid, count in sorted(cluster_counts.items(), key=lambda x: -x[1]):
        print(f"  Cluster {cid}: {count} emails")

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": np.ones(len(labels)),
        "n_clusters": n_clusters
    }


def cluster_temporal(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by time patterns."""
    print("\n" + "=" * 60)
    print("TEMPORAL CLUSTERING")
    print("=" * 60)

    email_ids, features = load_temporal_features(conn)
    print(f"Loaded {len(email_ids)} emails with {features.shape[1]} features")

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means with k=6 (time patterns)
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    print(f"Created {n_clusters} temporal clusters")

    # Show cluster distribution
    cluster_counts = Counter(labels)
    print("\nCluster sizes:")
    for cid, count in sorted(cluster_counts.items(), key=lambda x: -x[1]):
        print(f"  Cluster {cid}: {count} emails")

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": np.ones(len(labels)),
        "n_clusters": n_clusters
    }


# =============================================================================
# Save Results
# =============================================================================

def save_clusters(conn: psycopg2.extensions.connection, results: dict[str, Any]) -> None:
    """Save all clustering results to database."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Build email_id -> cluster mappings
    people = {eid: (lbl, prob) for eid, lbl, prob in
              zip(results["people"]["email_ids"],
                  results["people"]["labels"],
                  results["people"]["probs"])}

    content = {eid: (lbl, prob) for eid, lbl, prob in
               zip(results["content"]["email_ids"],
                   results["content"]["labels"],
                   results["content"]["probs"])}

    behavior = {eid: lbl for eid, lbl in
                zip(results["behavior"]["email_ids"],
                    results["behavior"]["labels"])}

    service = {eid: lbl for eid, lbl in
               zip(results["service"]["email_ids"],
                   results["service"]["labels"])}

    temporal = {eid: lbl for eid, lbl in
                zip(results["temporal"]["email_ids"],
                    results["temporal"]["labels"])}

    # Get all email IDs
    all_ids = set(people.keys())

    # Prepare data
    data = []
    for eid in all_ids:
        p_lbl, p_prob = people.get(eid, (None, None))
        c_lbl, c_prob = content.get(eid, (None, None))
        b_lbl = behavior.get(eid)
        s_lbl = service.get(eid)
        t_lbl = temporal.get(eid)

        data.append((
            eid,
            int(p_lbl) if p_lbl is not None else None,
            int(c_lbl) if c_lbl is not None else None,
            int(b_lbl) if b_lbl is not None else None,
            int(s_lbl) if s_lbl is not None else None,
            int(t_lbl) if t_lbl is not None else None,
            float(p_prob) if p_prob is not None else None,
            float(c_prob) if c_prob is not None else None
        ))

    # Clear and insert
    with conn.cursor() as cur:
        cur.execute("TRUNCATE email_clusters")
        execute_values(
            cur,
            """
            INSERT INTO email_clusters
            (email_id, people_cluster_id, content_cluster_id, behavior_cluster_id,
             service_cluster_id, temporal_cluster_id, people_cluster_prob, content_cluster_prob)
            VALUES %s
            """,
            data
        )
        conn.commit()

    print(f"✓ Saved {len(data)} cluster assignments")


def compute_cluster_metadata(conn: psycopg2.extensions.connection) -> None:
    """Compute and save cluster metadata."""
    print("\nComputing cluster metadata...")

    dimensions = ['people', 'content', 'behavior', 'service', 'temporal']

    with conn.cursor() as cur:
        cur.execute("TRUNCATE cluster_metadata")

        for dim in dimensions:
            col = f"{dim}_cluster_id"

            # Get cluster stats
            cur.execute(f"""
                SELECT
                    ec.{col} as cluster_id,
                    COUNT(*) as size,
                    ROUND(AVG(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) * 100, 1) as pct_replied,
                    ROUND(AVG(COALESCE(e.response_time_seconds / 3600.0, 0))::numeric, 1) as avg_response_hours,
                    ROUND(AVG(COALESCE(ef.relationship_strength, 0))::numeric, 3) as avg_relationship
                FROM email_clusters ec
                JOIN emails e ON e.id = ec.email_id
                LEFT JOIN email_features ef ON ef.email_id = ec.email_id
                WHERE ec.{col} IS NOT NULL
                GROUP BY ec.{col}
            """)

            rows = cur.fetchall()

            for row in rows:
                cluster_id, size, pct_replied, avg_response, avg_rel = row

                # Find representative (closest to centroid - just pick first for now)
                cur.execute(f"""
                    SELECT email_id FROM email_clusters
                    WHERE {col} = %s
                    LIMIT 1
                """, (cluster_id,))
                rep_row = cur.fetchone()
                rep_id = rep_row[0] if rep_row else None

                cur.execute("""
                    INSERT INTO cluster_metadata
                    (dimension, cluster_id, size, representative_email_id,
                     pct_replied, avg_response_time_hours, avg_relationship_strength)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dimension, cluster_id) DO UPDATE SET
                        size = EXCLUDED.size,
                        pct_replied = EXCLUDED.pct_replied,
                        avg_response_time_hours = EXCLUDED.avg_response_time_hours,
                        avg_relationship_strength = EXCLUDED.avg_relationship_strength
                """, (dim, cluster_id, size, rep_id, pct_replied, avg_response, avg_rel))

        conn.commit()

    print("✓ Cluster metadata computed")


# =============================================================================
# Analysis
# =============================================================================

def analyze_clusters(conn: psycopg2.extensions.connection) -> None:
    """Show cluster analysis."""
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS")
    print("=" * 70)

    dimensions = ['people', 'content', 'behavior', 'service', 'temporal']

    for dim in dimensions:
        print(f"\n{'=' * 50}")
        print(f"{dim.upper()} CLUSTERS")
        print(f"{'=' * 50}")

        with conn.cursor() as cur:
            cur.execute("""
                SELECT cluster_id, size, pct_replied, avg_response_time_hours, avg_relationship_strength
                FROM cluster_metadata
                WHERE dimension = %s
                ORDER BY size DESC
            """, (dim,))

            rows = cur.fetchall()

            print(f"{'Cluster':<10} {'Size':<8} {'Reply%':<10} {'Avg Resp(h)':<12} {'Avg Rel':<10}")
            print("-" * 50)

            for row in rows[:15]:  # Top 15
                cid, size, pct, resp, rel = row
                print(f"{cid:<10} {size:<8} {pct or 0:<10.1f} {resp or 0:<12.1f} {rel or 0:<10.3f}")

    # Cross-dimensional analysis
    print(f"\n{'=' * 70}")
    print("CROSS-DIMENSIONAL INSIGHTS")
    print(f"{'=' * 70}")

    with conn.cursor() as cur:
        # People clusters with highest reply rates
        print("\nTop People Clusters (by reply rate):")
        cur.execute("""
            SELECT cluster_id, size, pct_replied, avg_relationship_strength
            FROM cluster_metadata
            WHERE dimension = 'people' AND size >= 100
            ORDER BY pct_replied DESC
            LIMIT 5
        """)
        for row in cur.fetchall():
            print(f"  Cluster {row[0]}: {row[1]} emails, {row[2]:.1f}% replied, rel={row[3]:.3f}")

        # Content clusters with highest reply rates
        print("\nTop Content Clusters (by reply rate):")
        cur.execute("""
            SELECT cluster_id, size, pct_replied
            FROM cluster_metadata
            WHERE dimension = 'content' AND size >= 50 AND cluster_id != -1
            ORDER BY pct_replied DESC
            LIMIT 5
        """)
        for row in cur.fetchall():
            print(f"  Cluster {row[0]}: {row[1]} emails, {row[2]:.1f}% replied")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-dimensional email clustering")
    parser.add_argument("--dimension", "-d", choices=["people", "content", "behavior", "service", "temporal"],
                        help="Run specific dimension only")
    parser.add_argument("--analyze", "-a", action="store_true", help="Show analysis only")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4A: Multi-Dimensional Email Clustering")
    print("=" * 70)
    print(f"HDBSCAN available: {HAS_HDBSCAN}")
    print(f"UMAP available: {HAS_UMAP}")

    try:
        conn = psycopg2.connect(DB_URL)
        print("✓ Connected to database")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    try:
        if args.analyze:
            analyze_clusters(conn)
            return 0

        create_tables(conn)

        start_time = time.time()

        if args.dimension:
            # Run single dimension
            if args.dimension == "people":
                results = {"people": cluster_people(conn)}
            elif args.dimension == "content":
                results = {"content": cluster_content(conn)}
            elif args.dimension == "behavior":
                results = {"behavior": cluster_behavior(conn)}
            elif args.dimension == "service":
                results = {"service": cluster_service(conn)}
            elif args.dimension == "temporal":
                results = {"temporal": cluster_temporal(conn)}
        else:
            # Run all dimensions
            results = {
                "people": cluster_people(conn),
                "content": cluster_content(conn),
                "behavior": cluster_behavior(conn),
                "service": cluster_service(conn),
                "temporal": cluster_temporal(conn)
            }

            save_clusters(conn, results)
            compute_cluster_metadata(conn)

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("CLUSTERING COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed:.1f}s")

        if not args.dimension:
            print(f"\nCluster counts:")
            print(f"  People:   {results['people']['n_clusters']} clusters")
            print(f"  Content:  {results['content']['n_clusters']} clusters")
            print(f"  Behavior: {results['behavior']['n_clusters']} clusters")
            print(f"  Service:  {results['service']['n_clusters']} clusters")
            print(f"  Temporal: {results['temporal']['n_clusters']} clusters")

            analyze_clusters(conn)

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
