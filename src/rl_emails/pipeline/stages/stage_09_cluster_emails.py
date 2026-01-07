"""Stage 9: Multi-dimensional email clustering.

Clusters emails across 5 dimensions:
1. People - relationship patterns
2. Content - semantic similarity (embeddings)
3. Behavior - action patterns
4. Service - automated email types
5. Temporal - time patterns
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult

# Optional imports
try:
    import hdbscan  # type: ignore[import-not-found]

    HAS_HDBSCAN = True  # pragma: no cover
except ImportError:
    hdbscan = None
    HAS_HDBSCAN = False

try:
    import umap  # type: ignore[import-not-found]

    HAS_UMAP = True  # pragma: no cover
except ImportError:
    umap = None
    HAS_UMAP = False


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create clustering tables if they don't exist.

    Args:
        conn: Database connection.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_clusters (
                id SERIAL PRIMARY KEY,
                email_id INTEGER REFERENCES emails(id),
                people_cluster_id INTEGER,
                content_cluster_id INTEGER,
                behavior_cluster_id INTEGER,
                service_cluster_id INTEGER,
                temporal_cluster_id INTEGER,
                people_cluster_prob FLOAT,
                content_cluster_prob FLOAT,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(email_id)
            )
        """
        )

        cur.execute(
            """
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
        """
        )

        conn.commit()


def load_people_features(
    conn: psycopg2.extensions.connection,
) -> tuple[list[int], np.ndarray]:
    """Load features for people clustering.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (email_ids, feature_array).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
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
        """
        )
        rows = cur.fetchall()

    if not rows:
        return [], np.array([])

    email_ids = [r[0] for r in rows]
    features = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
    return email_ids, features


def load_embeddings(
    conn: psycopg2.extensions.connection,
) -> tuple[list[int], np.ndarray]:
    """Load embeddings for content clustering.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (email_ids, embeddings_array).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT email_id, embedding::text
            FROM email_embeddings
            ORDER BY email_id
        """
        )
        rows = cur.fetchall()

    if not rows:
        return [], np.array([])

    email_ids = [r[0] for r in rows]
    embeddings = []
    for r in rows:
        emb_str = r[1].strip("[]")
        emb = [float(x) for x in emb_str.split(",")]
        embeddings.append(emb)

    return email_ids, np.array(embeddings)


def load_behavior_features(
    conn: psycopg2.extensions.connection,
) -> tuple[list[int], np.ndarray]:
    """Load features for behavior clustering.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (email_ids, feature_array).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
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
        """
        )
        rows = cur.fetchall()

    if not rows:
        return [], np.array([])

    email_ids = [r[0] for r in rows]
    features = np.array([[r[1], r[2], r[3]] for r in rows])
    return email_ids, features


def load_service_features(
    conn: psycopg2.extensions.connection,
) -> tuple[list[int], np.ndarray]:
    """Load features for service clustering.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (email_ids, feature_array).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
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
        """
        )
        rows = cur.fetchall()

    if not rows:
        return [], np.array([])

    email_ids = [r[0] for r in rows]
    features = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    return email_ids, features


def load_temporal_features(
    conn: psycopg2.extensions.connection,
) -> tuple[list[int], np.ndarray]:
    """Load features for temporal clustering.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (email_ids, feature_array).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
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
        """
        )
        rows = cur.fetchall()

    if not rows:
        return [], np.array([])

    email_ids = [r[0] for r in rows]
    features = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    return email_ids, features


def cluster_people(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by relationship patterns.

    Args:
        conn: Database connection.

    Returns:
        Clustering results dictionary.
    """
    email_ids, features = load_people_features(conn)

    if len(email_ids) == 0:
        return {"email_ids": [], "labels": np.array([]), "probs": np.array([]), "n_clusters": 0}

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = min(15, len(email_ids))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    distances = kmeans.transform(features_scaled)
    min_distances = distances.min(axis=1)
    max_dist = min_distances.max()
    probs = 1 - (min_distances / max_dist) if max_dist > 0 else np.ones(len(labels))

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": probs,
        "n_clusters": n_clusters,
    }


def cluster_content(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by semantic similarity (embeddings).

    Args:
        conn: Database connection.

    Returns:
        Clustering results dictionary.
    """
    email_ids, embeddings = load_embeddings(conn)

    if len(email_ids) == 0:
        return {"email_ids": [], "labels": np.array([]), "probs": np.array([]), "n_clusters": 0}

    if HAS_UMAP and umap is not None and len(email_ids) > 50:
        reducer = umap.UMAP(n_components=50, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_reduced = reducer.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    if HAS_HDBSCAN and hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings_reduced)
        probs = clusterer.probabilities_
    else:
        n_clusters = min(100, len(email_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_reduced)
        distances = kmeans.transform(embeddings_reduced)
        min_distances = distances.min(axis=1)
        max_dist = min_distances.max()
        probs = 1 - (min_distances / max_dist) if max_dist > 0 else np.ones(len(labels))

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": probs,
        "n_clusters": n_clusters,
    }


def cluster_behavior(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by action patterns.

    Args:
        conn: Database connection.

    Returns:
        Clustering results dictionary.
    """
    email_ids, features = load_behavior_features(conn)

    if len(email_ids) == 0:
        return {"email_ids": [], "labels": np.array([]), "probs": np.array([]), "n_clusters": 0}

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = min(8, len(email_ids))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": np.ones(len(labels)),
        "n_clusters": n_clusters,
    }


def cluster_service(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by service type.

    Args:
        conn: Database connection.

    Returns:
        Clustering results dictionary.
    """
    email_ids, features = load_service_features(conn)

    if len(email_ids) == 0:
        return {"email_ids": [], "labels": np.array([]), "probs": np.array([]), "n_clusters": 0}

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = min(8, len(email_ids))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": np.ones(len(labels)),
        "n_clusters": n_clusters,
    }


def cluster_temporal(conn: psycopg2.extensions.connection) -> dict[str, Any]:
    """Cluster by time patterns.

    Args:
        conn: Database connection.

    Returns:
        Clustering results dictionary.
    """
    email_ids, features = load_temporal_features(conn)

    if len(email_ids) == 0:
        return {"email_ids": [], "labels": np.array([]), "probs": np.array([]), "n_clusters": 0}

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = min(6, len(email_ids))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    return {
        "email_ids": email_ids,
        "labels": labels,
        "probs": np.ones(len(labels)),
        "n_clusters": n_clusters,
    }


def save_clusters(conn: psycopg2.extensions.connection, results: dict[str, Any]) -> int:
    """Save all clustering results to database.

    Args:
        conn: Database connection.
        results: Dictionary with clustering results per dimension.

    Returns:
        Number of cluster assignments saved.
    """
    people = dict(
        zip(
            results["people"]["email_ids"],
            zip(results["people"]["labels"], results["people"]["probs"], strict=True),
            strict=True,
        )
    )

    content = dict(
        zip(
            results["content"]["email_ids"],
            zip(results["content"]["labels"], results["content"]["probs"], strict=True),
            strict=True,
        )
    )

    behavior = dict(
        zip(results["behavior"]["email_ids"], results["behavior"]["labels"], strict=True)
    )

    service = dict(zip(results["service"]["email_ids"], results["service"]["labels"], strict=True))

    temporal = dict(
        zip(results["temporal"]["email_ids"], results["temporal"]["labels"], strict=True)
    )

    all_ids = set(people.keys())

    data = []
    for eid in all_ids:
        p_lbl, p_prob = people.get(eid, (None, None))
        c_lbl, c_prob = content.get(eid, (None, None))
        b_lbl = behavior.get(eid)
        s_lbl = service.get(eid)
        t_lbl = temporal.get(eid)

        data.append(
            (
                eid,
                int(p_lbl) if p_lbl is not None else None,
                int(c_lbl) if c_lbl is not None else None,
                int(b_lbl) if b_lbl is not None else None,
                int(s_lbl) if s_lbl is not None else None,
                int(t_lbl) if t_lbl is not None else None,
                float(p_prob) if p_prob is not None else None,
                float(c_prob) if c_prob is not None else None,
            )
        )

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
            data,
        )
        conn.commit()

    return len(data)


def compute_cluster_metadata(conn: psycopg2.extensions.connection) -> None:
    """Compute and save cluster metadata.

    Args:
        conn: Database connection.
    """
    dimensions = ["people", "content", "behavior", "service", "temporal"]

    with conn.cursor() as cur:
        cur.execute("TRUNCATE cluster_metadata")

        for dim in dimensions:
            col = f"{dim}_cluster_id"

            cur.execute(
                f"""
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
            """
            )

            rows = cur.fetchall()

            for row in rows:
                cluster_id, size, pct_replied, avg_response, avg_rel = row

                cur.execute(
                    f"""
                    SELECT email_id FROM email_clusters
                    WHERE {col} = %s
                    LIMIT 1
                """,
                    (cluster_id,),
                )
                rep_row = cur.fetchone()
                rep_id = rep_row[0] if rep_row else None

                cur.execute(
                    """
                    INSERT INTO cluster_metadata
                    (dimension, cluster_id, size, representative_email_id,
                     pct_replied, avg_response_time_hours, avg_relationship_strength)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dimension, cluster_id) DO UPDATE SET
                        size = EXCLUDED.size,
                        pct_replied = EXCLUDED.pct_replied,
                        avg_response_time_hours = EXCLUDED.avg_response_time_hours,
                        avg_relationship_strength = EXCLUDED.avg_relationship_strength
                """,
                    (dim, cluster_id, size, rep_id, pct_replied, avg_response, avg_rel),
                )

        conn.commit()


def run(config: Config) -> StageResult:
    """Run Stage 9: Cluster emails.

    Args:
        config: Application configuration.

    Returns:
        StageResult with clustering statistics.
    """
    start_time = time.time()

    conn = psycopg2.connect(config.database_url)
    try:
        create_tables(conn)

        results = {
            "people": cluster_people(conn),
            "content": cluster_content(conn),
            "behavior": cluster_behavior(conn),
            "service": cluster_service(conn),
            "temporal": cluster_temporal(conn),
        }

        saved = save_clusters(conn, results)
        compute_cluster_metadata(conn)

        duration = time.time() - start_time

        return StageResult(
            success=True,
            records_processed=saved,
            duration_seconds=duration,
            message=f"Clustered {saved} emails across 5 dimensions",
            metadata={
                "people_clusters": results["people"]["n_clusters"],
                "content_clusters": results["content"]["n_clusters"],
                "behavior_clusters": results["behavior"]["n_clusters"],
                "service_clusters": results["service"]["n_clusters"],
                "temporal_clusters": results["temporal"]["n_clusters"],
                "has_hdbscan": HAS_HDBSCAN,
                "has_umap": HAS_UMAP,
            },
        )
    finally:
        conn.close()
