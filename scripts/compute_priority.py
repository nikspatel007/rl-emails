#!/usr/bin/env python3
"""Phase 4B: Hybrid Priority Ranking.

Combines multiple signals to score email importance:
- Phase 2 features (relationship, urgency)
- Phase 3 embeddings (similarity to replied emails)
- Phase 4A clusters (novelty within cluster)

Usage:
    python scripts/compute_priority.py              # Compute all priorities
    python scripts/compute_priority.py --analyze   # Show analysis only
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv()

# Configuration from environment
DB_URL = os.environ.get("DATABASE_URL")


# =============================================================================
# Database Setup
# =============================================================================

def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create priority table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_priority (
                id SERIAL PRIMARY KEY,
                email_id INTEGER REFERENCES emails(id) UNIQUE,

                -- Component scores (0-1)
                feature_score FLOAT,
                replied_similarity FLOAT,
                cluster_novelty FLOAT,
                sender_novelty FLOAT,

                -- Final scores
                priority_score FLOAT,
                priority_rank INTEGER,

                -- LLM selection flags
                needs_llm_analysis BOOLEAN DEFAULT FALSE,
                llm_reason TEXT,

                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Index for fast ranking queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_email_priority_score
            ON email_priority(priority_score DESC)
        """)

        conn.commit()
        print("✓ Tables created/verified")


# =============================================================================
# Component Score Computations
# =============================================================================

def compute_feature_scores(conn: psycopg2.extensions.connection) -> dict[int, float]:
    """Compute feature-based scores from Phase 2 data."""
    print("\nComputing feature scores...")

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ef.email_id,
                COALESCE(ef.relationship_strength, 0) as rel,
                COALESCE(ef.urgency_score, 0) as urgency,
                CASE WHEN ef.is_service_email THEN 0 ELSE 1 END as is_person,
                COALESCE(ef.service_importance, 0.5) as service_imp
            FROM email_features ef
            JOIN emails e ON e.id = ef.email_id
            WHERE e.is_sent = FALSE
        """)
        rows = cur.fetchall()

    scores = {}
    for row in rows:
        email_id, rel, urgency, is_person, service_imp = row

        # Weighted combination
        if is_person:
            # For person emails: relationship + urgency
            score = rel * 0.6 + urgency * 0.4
        else:
            # For service emails: service importance + urgency
            score = service_imp * 0.5 + urgency * 0.5

        scores[email_id] = min(1.0, max(0.0, score))

    print(f"  Computed {len(scores)} feature scores")
    return scores


def compute_replied_similarity(conn: psycopg2.extensions.connection) -> dict[int, float]:
    """Compute similarity to replied emails using embeddings."""
    print("\nComputing replied similarity...")

    with conn.cursor() as cur:
        # First, compute centroid of replied emails
        cur.execute("""
            SELECT AVG(ee.embedding) as centroid
            FROM email_embeddings ee
            JOIN emails e ON e.id = ee.email_id
            WHERE e.action = 'REPLIED'
        """)
        result = cur.fetchone()

        if result is None or result[0] is None:
            print("  WARNING: No replied emails found")
            return {}

        # Get centroid as string and parse
        centroid_str = str(result[0])

        # Now compute similarity for all emails
        # Using cosine similarity: 1 - cosine_distance
        cur.execute("""
            SELECT
                ee.email_id,
                1 - (ee.embedding <=> (
                    SELECT AVG(ee2.embedding)
                    FROM email_embeddings ee2
                    JOIN emails e2 ON e2.id = ee2.email_id
                    WHERE e2.action = 'REPLIED'
                )) as similarity
            FROM email_embeddings ee
        """)
        rows = cur.fetchall()

    # Normalize to 0-1
    similarities = {row[0]: row[1] for row in rows}

    if similarities:
        min_sim = min(similarities.values())
        max_sim = max(similarities.values())
        range_sim = max_sim - min_sim if max_sim > min_sim else 1

        for eid in similarities:
            similarities[eid] = (similarities[eid] - min_sim) / range_sim

    print(f"  Computed {len(similarities)} similarity scores")
    return similarities


def compute_cluster_novelty(conn: psycopg2.extensions.connection) -> dict[int, float]:
    """Compute how novel each email is within its content cluster."""
    print("\nComputing cluster novelty...")

    with conn.cursor() as cur:
        # For each email, compute distance to its cluster centroid
        # Emails far from centroid are "novel"
        cur.execute("""
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
                    WHEN ec.content_cluster_id = -1 THEN 1.0  -- Noise points are novel
                    WHEN cc.centroid IS NULL THEN 0.5
                    ELSE (ee.embedding <=> cc.centroid)  -- Distance = novelty
                END as novelty
            FROM email_clusters ec
            JOIN email_embeddings ee ON ee.email_id = ec.email_id
            LEFT JOIN cluster_centroids cc ON cc.content_cluster_id = ec.content_cluster_id
        """)
        rows = cur.fetchall()

    # Normalize to 0-1
    novelties = {row[0]: row[1] for row in rows if row[1] is not None}

    if novelties:
        min_nov = min(novelties.values())
        max_nov = max(novelties.values())
        range_nov = max_nov - min_nov if max_nov > min_nov else 1

        for eid in novelties:
            novelties[eid] = (novelties[eid] - min_nov) / range_nov

    print(f"  Computed {len(novelties)} novelty scores")
    return novelties


def compute_sender_novelty(conn: psycopg2.extensions.connection) -> dict[int, float]:
    """Compute how unusual each email is for its sender."""
    print("\nComputing sender novelty...")

    with conn.cursor() as cur:
        # For senders with multiple emails, compute distance from sender's average
        cur.execute("""
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
                    WHEN sc.centroid IS NULL THEN 0.5  -- Unknown sender
                    ELSE (ee.embedding <=> sc.centroid)
                END as novelty
            FROM emails e
            JOIN email_embeddings ee ON ee.email_id = e.id
            LEFT JOIN sender_centroids sc ON sc.from_email = e.from_email
            WHERE e.is_sent = FALSE
        """)
        rows = cur.fetchall()

    # Normalize to 0-1
    novelties = {row[0]: row[1] for row in rows if row[1] is not None}

    if novelties:
        min_nov = min(novelties.values())
        max_nov = max(novelties.values())
        range_nov = max_nov - min_nov if max_nov > min_nov else 1

        for eid in novelties:
            novelties[eid] = (novelties[eid] - min_nov) / range_nov

    print(f"  Computed {len(novelties)} sender novelty scores")
    return novelties


# =============================================================================
# Priority Computation
# =============================================================================

def compute_priority_scores(
    feature_scores: dict[int, float],
    replied_similarity: dict[int, float],
    cluster_novelty: dict[int, float],
    sender_novelty: dict[int, float]
) -> dict[int, dict[str, float]]:
    """Combine component scores into final priority."""
    print("\nComputing final priority scores...")

    # Weights for combining scores
    WEIGHTS = {
        "feature": 0.35,
        "replied_sim": 0.30,
        "cluster_nov": 0.20,
        "sender_nov": 0.15
    }

    all_ids = set(feature_scores.keys())
    results = {}

    for eid in all_ids:
        feat = feature_scores.get(eid, 0.5)
        replied = replied_similarity.get(eid, 0.5)
        cluster = cluster_novelty.get(eid, 0.5)
        sender = sender_novelty.get(eid, 0.5)

        priority = (
            feat * WEIGHTS["feature"] +
            replied * WEIGHTS["replied_sim"] +
            cluster * WEIGHTS["cluster_nov"] +
            sender * WEIGHTS["sender_nov"]
        )

        results[eid] = {
            "feature_score": feat,
            "replied_similarity": replied,
            "cluster_novelty": cluster,
            "sender_novelty": sender,
            "priority_score": priority
        }

    print(f"  Computed {len(results)} priority scores")
    return results


def determine_llm_flags(conn: psycopg2.extensions.connection, priorities: dict[int, dict[str, float]]) -> dict[int, tuple[bool, str | None]]:
    """Determine which emails need LLM analysis."""
    print("\nDetermining LLM analysis flags...")

    # Get email metadata for decision making
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                e.id,
                e.action,
                ef.is_service_email,
                ec.content_cluster_id
            FROM emails e
            LEFT JOIN email_features ef ON ef.email_id = e.id
            LEFT JOIN email_clusters ec ON ec.email_id = e.id
            WHERE e.is_sent = FALSE
        """)
        rows = cur.fetchall()

    email_meta = {row[0]: {"action": row[1], "is_service": row[2], "content_cluster": row[3]} for row in rows}

    flags = {}
    for eid, scores in priorities.items():
        meta = email_meta.get(eid, {})
        needs_llm = False
        reason = None

        priority = scores["priority_score"]

        # High priority emails
        if priority >= 0.7:
            needs_llm = True
            reason = "high_priority"

        # Replied emails (training signal)
        elif meta.get("action") == "REPLIED":
            needs_llm = True
            reason = "replied_training"

        # High priority but ignored (interesting case)
        elif priority >= 0.5 and meta.get("action") == "IGNORED":
            needs_llm = True
            reason = "high_priority_ignored"

        # Noise/outlier content (novel)
        elif meta.get("content_cluster") == -1 and priority >= 0.4:
            needs_llm = True
            reason = "novel_content"

        # Cluster representatives (sample from each cluster)
        # This would be done separately

        flags[eid] = (needs_llm, reason)

    llm_count = sum(1 for v in flags.values() if v[0])
    print(f"  Flagged {llm_count} emails for LLM analysis")

    return flags


# =============================================================================
# Save Results
# =============================================================================

def save_priorities(conn: psycopg2.extensions.connection, priorities: dict[int, dict[str, float]], llm_flags: dict[int, tuple[bool, str | None]]) -> None:
    """Save priority scores to database."""
    print("\nSaving priorities to database...")

    # Prepare data with rankings
    sorted_priorities = sorted(priorities.items(), key=lambda x: -x[1]["priority_score"])

    data = []
    for rank, (eid, scores) in enumerate(sorted_priorities, 1):
        needs_llm, llm_reason = llm_flags.get(eid, (False, None))
        data.append((
            eid,
            scores["feature_score"],
            scores["replied_similarity"],
            scores["cluster_novelty"],
            scores["sender_novelty"],
            scores["priority_score"],
            rank,
            needs_llm,
            llm_reason
        ))

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
            data
        )
        conn.commit()

    print(f"✓ Saved {len(data)} priority scores")


# =============================================================================
# Analysis
# =============================================================================

def analyze_priorities(conn: psycopg2.extensions.connection) -> None:
    """Show priority analysis."""
    print("\n" + "=" * 70)
    print("PRIORITY ANALYSIS")
    print("=" * 70)

    with conn.cursor() as cur:
        # Distribution
        print("\n1. SCORE DISTRIBUTION:")
        cur.execute("""
            SELECT
                CASE
                    WHEN ep.priority_score >= 0.8 THEN 'Very High (0.8+)'
                    WHEN ep.priority_score >= 0.6 THEN 'High (0.6-0.8)'
                    WHEN ep.priority_score >= 0.4 THEN 'Medium (0.4-0.6)'
                    WHEN ep.priority_score >= 0.2 THEN 'Low (0.2-0.4)'
                    ELSE 'Very Low (<0.2)'
                END as tier,
                COUNT(*) as cnt,
                ROUND(AVG(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) * 100, 1) as reply_pct
            FROM email_priority ep
            JOIN emails e ON e.id = ep.email_id
            GROUP BY
                CASE
                    WHEN ep.priority_score >= 0.8 THEN 'Very High (0.8+)'
                    WHEN ep.priority_score >= 0.6 THEN 'High (0.6-0.8)'
                    WHEN ep.priority_score >= 0.4 THEN 'Medium (0.4-0.6)'
                    WHEN ep.priority_score >= 0.2 THEN 'Low (0.2-0.4)'
                    ELSE 'Very Low (<0.2)'
                END
            ORDER BY MIN(ep.priority_score) DESC
        """)
        print(f"{'Tier':<25} {'Count':<10} {'Reply %':<10}")
        print("-" * 45)
        for row in cur.fetchall():
            print(f"{row[0]:<25} {row[1]:<10} {row[2]:<10}")

        # Top priority emails
        print("\n2. TOP 20 PRIORITY EMAILS:")
        cur.execute("""
            SELECT
                ep.priority_rank,
                ep.priority_score,
                e.action,
                e.from_email,
                e.subject
            FROM email_priority ep
            JOIN emails e ON e.id = ep.email_id
            ORDER BY ep.priority_rank
            LIMIT 20
        """)
        print(f"{'Rank':<6} {'Score':<8} {'Action':<10} {'From':<30} {'Subject':<40}")
        print("-" * 94)
        for row in cur.fetchall():
            from_email = (row[3][:28] + '..') if len(row[3]) > 30 else row[3]
            subject = (row[4][:38] + '..') if row[4] and len(row[4]) > 40 else (row[4] or '')
            print(f"{row[0]:<6} {row[1]:<8.3f} {row[2]:<10} {from_email:<30} {subject:<40}")

        # LLM analysis breakdown
        print("\n3. LLM ANALYSIS FLAGS:")
        cur.execute("""
            SELECT
                llm_reason,
                COUNT(*) as cnt,
                ROUND(AVG(priority_score)::numeric, 3) as avg_priority
            FROM email_priority
            WHERE needs_llm_analysis = TRUE
            GROUP BY llm_reason
            ORDER BY cnt DESC
        """)
        print(f"{'Reason':<25} {'Count':<10} {'Avg Priority':<12}")
        print("-" * 47)
        for row in cur.fetchall():
            print(f"{row[0] or 'None':<25} {row[1]:<10} {row[2]:<12}")

        cur.execute("""
            SELECT COUNT(*) FROM email_priority WHERE needs_llm_analysis = TRUE
        """)
        llm_row = cur.fetchone()
        total_llm = llm_row[0] if llm_row else 0
        cur.execute("SELECT COUNT(*) FROM email_priority")
        total_row = cur.fetchone()
        total = total_row[0] if total_row else 1  # Avoid division by zero
        print(f"\nTotal for LLM: {total_llm} / {total} ({total_llm * 100 / total:.1f}%)")

        # Component score correlations with reply
        print("\n4. COMPONENT SCORE VS REPLY RATE:")
        for component in ['feature_score', 'replied_similarity', 'cluster_novelty', 'sender_novelty']:
            cur.execute(f"""
                SELECT
                    CASE
                        WHEN {component} >= 0.8 THEN 'High (0.8+)'
                        WHEN {component} >= 0.5 THEN 'Medium (0.5-0.8)'
                        ELSE 'Low (<0.5)'
                    END as tier,
                    COUNT(*) as cnt,
                    ROUND(AVG(CASE WHEN e.action = 'REPLIED' THEN 1 ELSE 0 END) * 100, 1) as reply_pct
                FROM email_priority ep
                JOIN emails e ON e.id = ep.email_id
                GROUP BY
                    CASE
                        WHEN {component} >= 0.8 THEN 'High (0.8+)'
                        WHEN {component} >= 0.5 THEN 'Medium (0.5-0.8)'
                        ELSE 'Low (<0.5)'
                    END
                ORDER BY MIN({component}) DESC
            """)
            print(f"\n  {component}:")
            for row in cur.fetchall():
                print(f"    {row[0]}: {row[1]:,} emails, {row[2]}% replied")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Compute hybrid priority rankings")
    parser.add_argument("--analyze", "-a", action="store_true", help="Show analysis only")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4B: Hybrid Priority Ranking")
    print("=" * 70)

    try:
        conn = psycopg2.connect(DB_URL)
        print("✓ Connected to database")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    try:
        if args.analyze:
            analyze_priorities(conn)
            return 0

        create_tables(conn)

        start_time = time.time()

        # Compute component scores
        feature_scores = compute_feature_scores(conn)
        replied_similarity = compute_replied_similarity(conn)
        cluster_novelty = compute_cluster_novelty(conn)
        sender_novelty = compute_sender_novelty(conn)

        # Combine into final priority
        priorities = compute_priority_scores(
            feature_scores,
            replied_similarity,
            cluster_novelty,
            sender_novelty
        )

        # Determine LLM flags
        llm_flags = determine_llm_flags(conn, priorities)

        # Save results
        save_priorities(conn, priorities, llm_flags)

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("PRIORITY COMPUTATION COMPLETE")
        print("=" * 70)
        print(f"Time: {elapsed:.1f}s")
        print(f"Emails scored: {len(priorities)}")

        # Show analysis
        analyze_priorities(conn)

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
