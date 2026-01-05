#!/usr/bin/env python3
"""STORE-004: Feature computation pipeline.

Computes ML features for all emails and stores in email_features table.
Supports batch processing, progress reporting, and incremental updates.

Usage:
    python scripts/compute_features.py                # Incremental update
    python scripts/compute_features.py --force        # Recompute all
    python scripts/compute_features.py --batch-size 500
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.combined import (
    extract_combined_features,
    CombinedFeatureExtractor,
    FEATURE_DIMS,
)
from src.features.relationship import (
    CommunicationGraph,
    build_communication_graph,
)
from src.features.urgency import (
    compute_email_urgency,
    urgency_to_priority_bucket,
)

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"
FEATURE_VERSION = 1
DEFAULT_BATCH_SIZE = 100
USER_EMAIL = "nik@nik-patel.com"  # Primary user email


async def get_connection() -> asyncpg.Connection:
    """Get database connection."""
    return await asyncpg.connect(DB_URL)


async def load_all_emails_for_graph(conn: asyncpg.Connection) -> list[dict]:
    """Load all emails for building CommunicationGraph.

    Returns minimal data needed for graph construction.
    """
    rows = await conn.fetch("""
        SELECT
            e.message_id,
            e.thread_id,
            e.in_reply_to,
            e.from_email as "from",
            array_to_string(e.to_emails, ',') as "to",
            array_to_string(e.cc_emails, ',') as "cc",
            e.subject,
            e.date_parsed as date
        FROM emails e
        ORDER BY e.date_parsed ASC
    """)

    emails = []
    for row in rows:
        email = dict(row)
        # Convert datetime to string for parser
        if email.get('date'):
            email['date'] = email['date'].isoformat()
        emails.append(email)

    return emails


async def get_emails_needing_features(
    conn: asyncpg.Connection,
    force: bool = False,
) -> list[int]:
    """Get email IDs that need feature computation.

    Args:
        conn: Database connection
        force: If True, return all emails. If False, only unprocessed or outdated.

    Returns:
        List of email IDs
    """
    if force:
        # Return all email IDs
        rows = await conn.fetch("SELECT id FROM emails ORDER BY date_parsed ASC")
    else:
        # Return only emails without features or with older version
        rows = await conn.fetch("""
            SELECT e.id
            FROM emails e
            LEFT JOIN email_features ef ON ef.email_id = e.id
            WHERE ef.id IS NULL
               OR ef.feature_version < $1
            ORDER BY e.date_parsed ASC
        """, FEATURE_VERSION)

    return [row['id'] for row in rows]


async def load_email_batch(
    conn: asyncpg.Connection,
    email_ids: list[int],
) -> list[dict]:
    """Load full email data for a batch of IDs."""
    rows = await conn.fetch("""
        SELECT
            e.id,
            e.message_id,
            e.thread_id,
            e.in_reply_to,
            e.date_parsed,
            e.from_email as "from",
            e.from_name,
            array_to_string(e.to_emails, ',') as "to",
            array_to_string(e.cc_emails, ',') as "cc",
            e.subject,
            e.body_text as body,
            e.labels,
            e.is_sent
        FROM emails e
        WHERE e.id = ANY($1)
        ORDER BY e.date_parsed ASC
    """, email_ids)

    emails = []
    for row in rows:
        email = dict(row)
        # Convert datetime to string
        if email.get('date_parsed'):
            email['date'] = email['date_parsed'].isoformat()
        # Add x_from/x_to for people features
        email['x_from'] = email.get('from_name', '')
        email['x_to'] = ''
        emails.append(email)

    return emails


async def store_features_batch(
    conn: asyncpg.Connection,
    features_data: list[dict],
) -> int:
    """Store computed features to database.

    Uses INSERT ... ON CONFLICT to handle both inserts and updates.

    Returns:
        Number of rows affected
    """
    if not features_data:
        return 0

    # Use executemany with a single transaction
    async with conn.transaction():
        result = await conn.executemany("""
            INSERT INTO email_features (
                email_id, message_id,
                -- Relationship features
                sender_response_deviation, sender_frequency_rank,
                inferred_hierarchy, relationship_strength,
                emails_from_sender_7d, emails_from_sender_30d,
                emails_from_sender_90d, response_rate_to_sender,
                avg_thread_depth, days_since_last_email, cc_affinity_score,
                -- Service classification
                is_service_email, service_type, service_email_confidence,
                has_list_unsubscribe_header, has_unsubscribe_url,
                unsubscribe_phrase_count,
                -- Task features
                task_count, has_deadline, deadline_urgency,
                is_assigned_to_user, estimated_effort, has_deliverable,
                -- Urgency scoring
                urgency_score, urgency_bucket,
                -- Priority scores
                project_score, topic_score, task_score,
                people_score, temporal_score, service_score,
                relationship_score, overall_priority,
                -- Embeddings
                feature_vector, embedding_model, embedding_dim,
                -- Metadata
                computed_at, feature_version
            ) VALUES (
                $1, $2,
                $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                $14, $15, $16, $17, $18, $19,
                $20, $21, $22, $23, $24, $25,
                $26, $27,
                $28, $29, $30, $31, $32, $33, $34, $35,
                $36, $37, $38,
                $39, $40
            )
            ON CONFLICT (email_id) DO UPDATE SET
                sender_response_deviation = EXCLUDED.sender_response_deviation,
                sender_frequency_rank = EXCLUDED.sender_frequency_rank,
                inferred_hierarchy = EXCLUDED.inferred_hierarchy,
                relationship_strength = EXCLUDED.relationship_strength,
                emails_from_sender_7d = EXCLUDED.emails_from_sender_7d,
                emails_from_sender_30d = EXCLUDED.emails_from_sender_30d,
                emails_from_sender_90d = EXCLUDED.emails_from_sender_90d,
                response_rate_to_sender = EXCLUDED.response_rate_to_sender,
                avg_thread_depth = EXCLUDED.avg_thread_depth,
                days_since_last_email = EXCLUDED.days_since_last_email,
                cc_affinity_score = EXCLUDED.cc_affinity_score,
                is_service_email = EXCLUDED.is_service_email,
                service_type = EXCLUDED.service_type,
                service_email_confidence = EXCLUDED.service_email_confidence,
                has_list_unsubscribe_header = EXCLUDED.has_list_unsubscribe_header,
                has_unsubscribe_url = EXCLUDED.has_unsubscribe_url,
                unsubscribe_phrase_count = EXCLUDED.unsubscribe_phrase_count,
                task_count = EXCLUDED.task_count,
                has_deadline = EXCLUDED.has_deadline,
                deadline_urgency = EXCLUDED.deadline_urgency,
                is_assigned_to_user = EXCLUDED.is_assigned_to_user,
                estimated_effort = EXCLUDED.estimated_effort,
                has_deliverable = EXCLUDED.has_deliverable,
                urgency_score = EXCLUDED.urgency_score,
                urgency_bucket = EXCLUDED.urgency_bucket,
                project_score = EXCLUDED.project_score,
                topic_score = EXCLUDED.topic_score,
                task_score = EXCLUDED.task_score,
                people_score = EXCLUDED.people_score,
                temporal_score = EXCLUDED.temporal_score,
                service_score = EXCLUDED.service_score,
                relationship_score = EXCLUDED.relationship_score,
                overall_priority = EXCLUDED.overall_priority,
                feature_vector = EXCLUDED.feature_vector,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dim = EXCLUDED.embedding_dim,
                computed_at = EXCLUDED.computed_at,
                feature_version = EXCLUDED.feature_version
        """, [
            (
                d['email_id'], d['message_id'],
                d['sender_response_deviation'], d['sender_frequency_rank'],
                d['inferred_hierarchy'], d['relationship_strength'],
                d['emails_from_sender_7d'], d['emails_from_sender_30d'],
                d['emails_from_sender_90d'], d['response_rate_to_sender'],
                d['avg_thread_depth'], d['days_since_last_email'], d['cc_affinity_score'],
                d['is_service_email'], d['service_type'], d['service_email_confidence'],
                d['has_list_unsubscribe_header'], d['has_unsubscribe_url'],
                d['unsubscribe_phrase_count'],
                d['task_count'], d['has_deadline'], d['deadline_urgency'],
                d['is_assigned_to_user'], d['estimated_effort'], d['has_deliverable'],
                d['urgency_score'], d['urgency_bucket'],
                d['project_score'], d['topic_score'], d['task_score'],
                d['people_score'], d['temporal_score'], d['service_score'],
                d['relationship_score'], d['overall_priority'],
                d['feature_vector'], d['embedding_model'], d['embedding_dim'],
                d['computed_at'], d['feature_version'],
            )
            for d in features_data
        ])

    return len(features_data)


def extract_features_for_email(
    email: dict,
    graph: CommunicationGraph,
    user_email: str,
) -> dict:
    """Extract all features for a single email.

    Returns dict ready for database insertion.
    """
    # Extract combined features
    combined = extract_combined_features(
        email,
        user_email=user_email,
        communication_graph=graph,
        include_content=False,  # Content embeddings handled separately
    )

    # Extract urgency features
    urgency = compute_email_urgency(email, user_email=user_email)

    # Get relationship features
    rel = combined.relationship if combined.relationship else None

    # Get service features
    svc = combined.service if combined.service else None

    # Get task features
    task = combined.task

    # Build feature vector (without content for now)
    feature_vec = list(combined.to_feature_vector(include_content=False))

    return {
        'email_id': email['id'],
        'message_id': email['message_id'],
        # Relationship features
        'sender_response_deviation': rel.sender_response_deviation if rel else 0.0,
        'sender_frequency_rank': rel.sender_frequency_rank if rel else 0.0,
        'inferred_hierarchy': rel.inferred_hierarchy if rel else 0.5,
        'relationship_strength': rel.relationship_strength if rel else 0.0,
        'emails_from_sender_7d': rel.emails_from_sender_7d if rel else 0,
        'emails_from_sender_30d': rel.emails_from_sender_30d if rel else 0,
        'emails_from_sender_90d': rel.emails_from_sender_90d if rel else 0,
        'response_rate_to_sender': rel.response_rate_to_sender if rel else 0.0,
        'avg_thread_depth': rel.avg_thread_depth if rel else None,
        'days_since_last_email': rel.days_since_last_email if rel else None,
        'cc_affinity_score': rel.cc_affinity_score if rel else 0.0,
        # Service classification
        'is_service_email': svc.is_service_email if svc else False,
        'service_type': svc.service_type.value if svc and svc.is_service_email else None,
        'service_email_confidence': svc.confidence if svc else 0.0,
        'has_list_unsubscribe_header': svc.domain_match if svc else False,  # Proxy
        'has_unsubscribe_url': svc.body_match if svc else False,  # Proxy
        'unsubscribe_phrase_count': len(svc.matched_patterns) if svc else 0,
        # Task features
        'task_count': len(task.action_items),
        'has_deadline': task.has_deadline,
        'deadline_urgency': task.deadline_urgency,
        'is_assigned_to_user': task.is_assigned_to_user,
        'estimated_effort': task.estimated_effort,
        'has_deliverable': task.has_deliverable,
        # Urgency scoring
        'urgency_score': urgency.overall_urgency,
        'urgency_bucket': urgency_to_priority_bucket(urgency.overall_urgency),
        # Priority scores
        'project_score': combined.project_score,
        'topic_score': combined.topic_score,
        'task_score': combined.task_score,
        'people_score': combined.people_score,
        'temporal_score': combined.temporal_score,
        'service_score': combined.service_score,
        'relationship_score': combined.relationship_score,
        'overall_priority': combined.overall_priority,
        # Feature vector
        'feature_vector': feature_vec,
        'embedding_model': None,  # Content embeddings not computed
        'embedding_dim': FEATURE_DIMS['total_base'],
        # Metadata
        'computed_at': datetime.now(),
        'feature_version': FEATURE_VERSION,
    }


async def run_pipeline(
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
) -> dict:
    """Run the feature computation pipeline.

    Args:
        force: If True, recompute all features
        batch_size: Number of emails per batch
        limit: Optional limit on total emails to process

    Returns:
        Summary statistics
    """
    print("=" * 60)
    print("STORE-004: Feature Computation Pipeline")
    print("=" * 60)
    print()

    start_time = datetime.now()

    # Connect to database
    print(f"Connecting to database...")
    conn = await get_connection()

    try:
        # Step 1: Build CommunicationGraph from all emails
        print("\nStep 1: Building CommunicationGraph...")
        graph_emails = await load_all_emails_for_graph(conn)
        print(f"  Loaded {len(graph_emails)} emails for graph construction")

        graph = build_communication_graph(graph_emails)
        print(f"  Graph built: {len(graph.user_baselines)} users, "
              f"{len(graph.edges)} edges, {len(graph.response_events)} responses")

        # Step 2: Get emails needing features
        print("\nStep 2: Identifying emails needing features...")
        email_ids = await get_emails_needing_features(conn, force=force)

        if limit:
            email_ids = email_ids[:limit]

        total_emails = len(email_ids)
        print(f"  Found {total_emails} emails to process")

        if total_emails == 0:
            print("\nNo emails need feature computation. Done!")
            return {'processed': 0, 'duration_seconds': 0}

        # Step 3: Process in batches
        print(f"\nStep 3: Computing features (batch size: {batch_size})...")

        processed = 0
        failed = 0

        with tqdm(total=total_emails, desc="Computing features") as pbar:
            for batch_start in range(0, total_emails, batch_size):
                batch_ids = email_ids[batch_start:batch_start + batch_size]

                # Load email data
                emails = await load_email_batch(conn, batch_ids)

                # Extract features for each email
                features_data = []
                for email in emails:
                    try:
                        features = extract_features_for_email(
                            email, graph, USER_EMAIL
                        )
                        features_data.append(features)
                    except Exception as e:
                        failed += 1
                        tqdm.write(f"Error processing {email.get('message_id', 'unknown')}: {e}")

                # Store to database
                stored = await store_features_batch(conn, features_data)
                processed += stored
                pbar.update(len(batch_ids))

        # Summary
        duration = (datetime.now() - start_time).total_seconds()

        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Processed: {processed} emails")
        print(f"  Failed: {failed} emails")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Rate: {processed / duration:.1f} emails/second")

        return {
            'processed': processed,
            'failed': failed,
            'duration_seconds': duration,
            'rate': processed / duration if duration > 0 else 0,
        }

    finally:
        await conn.close()


async def verify_features(conn: Optional[asyncpg.Connection] = None) -> dict:
    """Verify feature computation results.

    Returns summary statistics.
    """
    close_conn = False
    if conn is None:
        conn = await get_connection()
        close_conn = True

    try:
        # Count emails and features
        email_count = await conn.fetchval("SELECT COUNT(*) FROM emails")
        feature_count = await conn.fetchval("SELECT COUNT(*) FROM email_features")

        # Feature coverage
        coverage = feature_count / email_count * 100 if email_count > 0 else 0

        # Priority distribution
        priority_dist = await conn.fetch("""
            SELECT urgency_bucket, COUNT(*) as cnt
            FROM email_features
            GROUP BY urgency_bucket
            ORDER BY cnt DESC
        """)

        # Service email stats
        service_stats = await conn.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE is_service_email) as service_count,
                COUNT(*) FILTER (WHERE NOT is_service_email) as personal_count
            FROM email_features
        """)

        print("\n=== Feature Verification ===")
        print(f"Total emails: {email_count}")
        print(f"Features computed: {feature_count}")
        print(f"Coverage: {coverage:.1f}%")
        print()
        print("Urgency distribution:")
        for row in priority_dist:
            print(f"  {row['urgency_bucket'] or 'NULL'}: {row['cnt']}")
        print()
        print("Service email classification:")
        print(f"  Service emails: {service_stats['service_count']}")
        print(f"  Personal emails: {service_stats['personal_count']}")

        return {
            'email_count': email_count,
            'feature_count': feature_count,
            'coverage': coverage,
        }

    finally:
        if close_conn:
            await conn.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='STORE-004: Feature computation pipeline'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Recompute all features (ignore existing)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        help='Limit number of emails to process'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing features, do not compute'
    )

    args = parser.parse_args()

    if args.verify:
        await verify_features()
    else:
        result = await run_pipeline(
            force=args.force,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        # Run verification after processing
        if result['processed'] > 0:
            conn = await get_connection()
            try:
                await verify_features(conn)
            finally:
                await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
