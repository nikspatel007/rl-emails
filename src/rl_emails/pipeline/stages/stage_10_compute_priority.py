"""Stage 10: Compute hybrid priority rankings.

Combines multiple signals to score email importance:
- Feature scores (relationship, urgency)
- Replied email similarity (embeddings)
- Cluster novelty
- Sender novelty
"""

from __future__ import annotations

import time

import psycopg2
from psycopg2.extras import execute_values

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult

# Weights for combining scores
PRIORITY_WEIGHTS = {
    "feature": 0.35,
    "replied_sim": 0.30,
    "cluster_nov": 0.20,
    "sender_nov": 0.15,
}


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create priority table if it doesn't exist.

    Args:
        conn: Database connection.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_priority (
                id SERIAL PRIMARY KEY,
                email_id INTEGER REFERENCES emails(id) UNIQUE,
                feature_score FLOAT,
                replied_similarity FLOAT,
                cluster_novelty FLOAT,
                sender_novelty FLOAT,
                priority_score FLOAT,
                priority_rank INTEGER,
                needs_llm_analysis BOOLEAN DEFAULT FALSE,
                llm_reason TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_email_priority_score
            ON email_priority(priority_score DESC)
        """
        )

        conn.commit()


def compute_feature_scores(conn: psycopg2.extensions.connection) -> dict[int, float]:
    """Compute feature-based scores from Phase 2 data.

    Args:
        conn: Database connection.

    Returns:
        Dictionary mapping email_id to feature score.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                ef.email_id,
                COALESCE(ef.relationship_strength, 0) as rel,
                COALESCE(ef.urgency_score, 0) as urgency,
                CASE WHEN ef.is_service_email THEN 0 ELSE 1 END as is_person,
                COALESCE(ef.service_importance, 0.5) as service_imp
            FROM email_features ef
            JOIN emails e ON e.id = ef.email_id
            WHERE e.is_sent = FALSE
        """
        )
        rows = cur.fetchall()

    scores: dict[int, float] = {}
    for row in rows:
        email_id, rel, urgency, is_person, service_imp = row

        if is_person:
            score = rel * 0.6 + urgency * 0.4
        else:
            score = service_imp * 0.5 + urgency * 0.5

        scores[email_id] = min(1.0, max(0.0, score))

    return scores


def compute_replied_similarity(conn: psycopg2.extensions.connection) -> dict[int, float]:
    """Compute similarity to replied emails using embeddings.

    Args:
        conn: Database connection.

    Returns:
        Dictionary mapping email_id to similarity score.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT AVG(ee.embedding) as centroid
            FROM email_embeddings ee
            JOIN emails e ON e.id = ee.email_id
            WHERE e.action = 'REPLIED'
        """
        )
        result = cur.fetchone()

        if result is None or result[0] is None:
            return {}

        cur.execute(
            """
            SELECT
                ee.email_id,
                1 - (ee.embedding <=> (
                    SELECT AVG(ee2.embedding)
                    FROM email_embeddings ee2
                    JOIN emails e2 ON e2.id = ee2.email_id
                    WHERE e2.action = 'REPLIED'
                )) as similarity
            FROM email_embeddings ee
        """
        )
        rows = cur.fetchall()

    similarities = {row[0]: row[1] for row in rows}

    if similarities:
        min_sim = min(similarities.values())
        max_sim = max(similarities.values())
        range_sim = max_sim - min_sim if max_sim > min_sim else 1

        for eid in similarities:
            similarities[eid] = (similarities[eid] - min_sim) / range_sim

    return similarities


def compute_cluster_novelty(conn: psycopg2.extensions.connection) -> dict[int, float]:
    """Compute how novel each email is within its content cluster.

    Args:
        conn: Database connection.

    Returns:
        Dictionary mapping email_id to novelty score.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH cluster_centroids AS (
                SELECT
                    ec.content_cluster_id,
                    AVG(ee.embedding) as centroid
                FROM email_clusters ec
                JOIN email_embeddings ee ON ee.email_id = ec.email_id
                WHERE ec.content_cluster_id IS NOT NULL AND ec.content_cluster_id != -1
                GROUP BY ec.content_cluster_id
            )
            SELECT
                ec.email_id,
                CASE
                    WHEN ec.content_cluster_id = -1 THEN 1.0
                    WHEN cc.centroid IS NULL THEN 0.5
                    ELSE (ee.embedding <=> cc.centroid)
                END as novelty
            FROM email_clusters ec
            JOIN email_embeddings ee ON ee.email_id = ec.email_id
            LEFT JOIN cluster_centroids cc ON cc.content_cluster_id = ec.content_cluster_id
        """
        )
        rows = cur.fetchall()

    novelties = {row[0]: row[1] for row in rows if row[1] is not None}

    if novelties:
        min_nov = min(novelties.values())
        max_nov = max(novelties.values())
        range_nov = max_nov - min_nov if max_nov > min_nov else 1

        for eid in novelties:
            novelties[eid] = (novelties[eid] - min_nov) / range_nov

    return novelties


def compute_sender_novelty(conn: psycopg2.extensions.connection) -> dict[int, float]:
    """Compute how unusual each email is for its sender.

    Args:
        conn: Database connection.

    Returns:
        Dictionary mapping email_id to novelty score.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH sender_centroids AS (
                SELECT
                    e.from_email,
                    AVG(ee.embedding) as centroid,
                    COUNT(*) as email_count
                FROM emails e
                JOIN email_embeddings ee ON ee.email_id = e.id
                WHERE e.is_sent = FALSE
                GROUP BY e.from_email
                HAVING COUNT(*) >= 3
            )
            SELECT
                e.id,
                CASE
                    WHEN sc.centroid IS NULL THEN 0.5
                    ELSE (ee.embedding <=> sc.centroid)
                END as novelty
            FROM emails e
            JOIN email_embeddings ee ON ee.email_id = e.id
            LEFT JOIN sender_centroids sc ON sc.from_email = e.from_email
            WHERE e.is_sent = FALSE
        """
        )
        rows = cur.fetchall()

    novelties = {row[0]: row[1] for row in rows if row[1] is not None}

    if novelties:
        min_nov = min(novelties.values())
        max_nov = max(novelties.values())
        range_nov = max_nov - min_nov if max_nov > min_nov else 1

        for eid in novelties:
            novelties[eid] = (novelties[eid] - min_nov) / range_nov

    return novelties


def compute_priority_scores(
    feature_scores: dict[int, float],
    replied_similarity: dict[int, float],
    cluster_novelty: dict[int, float],
    sender_novelty: dict[int, float],
) -> dict[int, dict[str, float]]:
    """Combine component scores into final priority.

    Args:
        feature_scores: Feature-based scores.
        replied_similarity: Similarity to replied emails.
        cluster_novelty: Cluster novelty scores.
        sender_novelty: Sender novelty scores.

    Returns:
        Dictionary mapping email_id to score components.
    """
    all_ids = set(feature_scores.keys())
    results: dict[int, dict[str, float]] = {}

    for eid in all_ids:
        feat = feature_scores.get(eid, 0.5)
        replied = replied_similarity.get(eid, 0.5)
        cluster = cluster_novelty.get(eid, 0.5)
        sender = sender_novelty.get(eid, 0.5)

        priority = (
            feat * PRIORITY_WEIGHTS["feature"]
            + replied * PRIORITY_WEIGHTS["replied_sim"]
            + cluster * PRIORITY_WEIGHTS["cluster_nov"]
            + sender * PRIORITY_WEIGHTS["sender_nov"]
        )

        results[eid] = {
            "feature_score": feat,
            "replied_similarity": replied,
            "cluster_novelty": cluster,
            "sender_novelty": sender,
            "priority_score": priority,
        }

    return results


def determine_llm_flags(
    conn: psycopg2.extensions.connection, priorities: dict[int, dict[str, float]]
) -> dict[int, tuple[bool, str | None]]:
    """Determine which emails need LLM analysis.

    Args:
        conn: Database connection.
        priorities: Priority scores.

    Returns:
        Dictionary mapping email_id to (needs_llm, reason).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.id,
                e.action,
                ef.is_service_email,
                ec.content_cluster_id
            FROM emails e
            LEFT JOIN email_features ef ON ef.email_id = e.id
            LEFT JOIN email_clusters ec ON ec.email_id = e.id
            WHERE e.is_sent = FALSE
        """
        )
        rows = cur.fetchall()

    email_meta = {
        row[0]: {"action": row[1], "is_service": row[2], "content_cluster": row[3]} for row in rows
    }

    flags: dict[int, tuple[bool, str | None]] = {}
    for eid, scores in priorities.items():
        meta = email_meta.get(eid, {})
        needs_llm = False
        reason: str | None = None

        priority = scores["priority_score"]

        if priority >= 0.7:
            needs_llm = True
            reason = "high_priority"
        elif meta.get("action") == "REPLIED":
            needs_llm = True
            reason = "replied_training"
        elif priority >= 0.5 and meta.get("action") == "IGNORED":
            needs_llm = True
            reason = "high_priority_ignored"
        elif meta.get("content_cluster") == -1 and priority >= 0.4:
            needs_llm = True
            reason = "novel_content"

        flags[eid] = (needs_llm, reason)

    return flags


def save_priorities(
    conn: psycopg2.extensions.connection,
    priorities: dict[int, dict[str, float]],
    llm_flags: dict[int, tuple[bool, str | None]],
) -> int:
    """Save priority scores to database.

    Args:
        conn: Database connection.
        priorities: Priority scores.
        llm_flags: LLM analysis flags.

    Returns:
        Number of priorities saved.
    """
    sorted_priorities = sorted(priorities.items(), key=lambda x: -x[1]["priority_score"])

    data = []
    for rank, (eid, scores) in enumerate(sorted_priorities, 1):
        needs_llm, llm_reason = llm_flags.get(eid, (False, None))
        data.append(
            (
                eid,
                scores["feature_score"],
                scores["replied_similarity"],
                scores["cluster_novelty"],
                scores["sender_novelty"],
                scores["priority_score"],
                rank,
                needs_llm,
                llm_reason,
            )
        )

    with conn.cursor() as cur:
        cur.execute("TRUNCATE email_priority")
        execute_values(
            cur,
            """
            INSERT INTO email_priority
            (email_id, feature_score, replied_similarity, cluster_novelty,
             sender_novelty, priority_score, priority_rank, needs_llm_analysis, llm_reason)
            VALUES %s
            """,
            data,
        )
        conn.commit()

    return len(data)


def run(config: Config) -> StageResult:
    """Run Stage 10: Compute priority rankings.

    Args:
        config: Application configuration.

    Returns:
        StageResult with computation statistics.
    """
    start_time = time.time()

    conn = psycopg2.connect(config.database_url)
    try:
        create_tables(conn)

        feature_scores = compute_feature_scores(conn)
        replied_similarity = compute_replied_similarity(conn)
        cluster_novelty = compute_cluster_novelty(conn)
        sender_novelty = compute_sender_novelty(conn)

        priorities = compute_priority_scores(
            feature_scores, replied_similarity, cluster_novelty, sender_novelty
        )

        llm_flags = determine_llm_flags(conn, priorities)
        saved_count = save_priorities(conn, priorities, llm_flags)

        duration = time.time() - start_time

        llm_count = sum(1 for v in llm_flags.values() if v[0])

        return StageResult(
            success=True,
            records_processed=saved_count,
            duration_seconds=duration,
            message=f"Computed priority for {saved_count} emails ({llm_count} flagged for LLM)",
            metadata={
                "feature_scores": len(feature_scores),
                "replied_similarities": len(replied_similarity),
                "cluster_novelties": len(cluster_novelty),
                "sender_novelties": len(sender_novelty),
                "llm_flagged": llm_count,
            },
        )
    finally:
        conn.close()
