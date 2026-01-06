#!/usr/bin/env python3
"""Phase 2: Basic ML Features Computation.

Computes 30 features per email for smart ranking and filtering:
- Relationship Features (11 dimensions)
- Service Detection (6 dimensions)
- Temporal Features (8 dimensions)
- Content Basic (5 dimensions)

Usage:
    python scripts/compute_basic_features.py
    python scripts/compute_basic_features.py --batch-size 500
    python scripts/compute_basic_features.py --verify

Required .env variables:
    DB_URL - PostgreSQL connection URL
    YOUR_EMAIL - Your email address
"""

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Optional

import asyncpg
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

DB_URL = os.environ.get('DB_URL')
YOUR_EMAIL = os.environ.get('YOUR_EMAIL', '').lower()

# Service email detection patterns
SERVICE_DOMAINS = [
    'noreply', 'no-reply', 'donotreply', 'do-not-reply',
    'notifications', 'notification', 'alerts', 'alert',
    'mailer', 'mail', 'newsletter', 'updates', 'info',
    'support', 'service', 'system', 'automated', 'auto',
    'bounce', 'postmaster', 'mailer-daemon'
]

SERVICE_TYPES = {
    'newsletter': ['newsletter', 'digest', 'weekly', 'daily', 'monthly', 'subscribe'],
    'transactional': ['order', 'receipt', 'invoice', 'payment', 'shipping', 'delivery', 'confirmation'],
    'notification': ['notification', 'alert', 'reminder', 'update'],
    'marketing': ['offer', 'discount', 'sale', 'promo', 'deal', 'exclusive'],
    'social': ['linkedin', 'facebook', 'twitter', 'instagram', 'social'],
}

# Service importance detection
IMPORTANT_SUBJECT_KEYWORDS = [
    'order', 'shipped', 'delivered', 'delivery', 'payment', 'transaction',
    'confirm', 'confirmed', 'confirmation', 'receipt', 'invoice', 'alert',
    'security', 'verification', 'verify', 'password', 'login', 'suspicious',
    'fraud', 'unauthorized', 'expired', 'expiring', 'renewal', 'bill',
    'due', 'overdue', 'failed', 'declined', 'refund', 'return'
]

LOW_IMPORTANCE_KEYWORDS = [
    'offer', 'sale', 'deal', 'save', 'discount', 'promo', 'promotional',
    'newsletter', 'digest', 'weekly', 'daily', 'tips', 'recommendation',
    'suggested', 'trending', 'popular', 'new arrivals', 'just for you',
    'exclusive', 'limited time', 'don\'t miss', 'check out'
]

IMPORTANT_SENDER_PATTERNS = [
    'order', 'shipping', 'shipment', 'tracking', 'confirm', 'alert',
    'security', 'verify', 'transaction', 'payment', 'billing', 'receipt',
    'return', 'refund', 'delivery', 'support'
]


def extract_plain_text(html_body: str) -> str:
    """Extract plain text from HTML email body."""
    if not html_body:
        return ""

    # Parse HTML
    soup = BeautifulSoup(html_body, 'html.parser')

    # Remove script and style elements
    for element in soup(['script', 'style', 'head', 'meta', 'link']):
        element.decompose()

    # Get text
    text = soup.get_text(separator=' ')

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())


def detect_service_email(
    from_email: str,
    subject: str,
    body: str,
    headers: Optional[dict] = None,
) -> tuple[bool, float, Optional[str]]:
    """Detect if email is from a service/automated sender.

    Returns:
        (is_service, confidence, service_type)
    """
    from_email = (from_email or '').lower()
    subject = (subject or '').lower()
    body = (body or '').lower()

    signals = []

    # Check for service domain patterns
    for pattern in SERVICE_DOMAINS:
        if pattern in from_email:
            signals.append(('domain', 0.3))
            break

    # Check for List-Unsubscribe header (if available)
    has_list_unsubscribe = False
    if headers and isinstance(headers, dict):
        has_list_unsubscribe = 'list-unsubscribe' in str(headers).lower()
        if has_list_unsubscribe:
            signals.append(('list_header', 0.4))

    # Check for unsubscribe in body
    if 'unsubscribe' in body:
        signals.append(('unsubscribe_body', 0.3))

    # Determine service type
    detected_type = None
    for stype, keywords in SERVICE_TYPES.items():
        for kw in keywords:
            if kw in subject or kw in from_email:
                detected_type = stype
                signals.append(('type_match', 0.2))
                break
        if detected_type:
            break

    # Calculate confidence
    if not signals:
        return False, 0.0, None

    confidence = min(1.0, sum(s[1] for s in signals))
    is_service = confidence >= 0.3

    return is_service, confidence, detected_type


def get_time_bucket(hour: int) -> str:
    """Convert hour (0-23) to time bucket."""
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'


def compute_service_importance(
    from_email: str,
    subject: str,
    is_service: bool,
) -> float:
    """Compute importance score for service emails (0-1).

    Higher = more likely to be important (transaction, alert, security)
    Lower = more likely to be noise (marketing, newsletter)
    """
    if not is_service:
        return 0.0  # Not applicable for non-service emails

    from_email = (from_email or '').lower()
    subject = (subject or '').lower()

    # Subject importance (0.5 weight)
    important_matches = sum(1 for kw in IMPORTANT_SUBJECT_KEYWORDS if kw in subject)
    low_matches = sum(1 for kw in LOW_IMPORTANCE_KEYWORDS if kw in subject)

    if important_matches > 0 and low_matches == 0:
        subject_score = min(1.0, 0.5 + important_matches * 0.2)
    elif low_matches > 0 and important_matches == 0:
        subject_score = max(0.0, 0.3 - low_matches * 0.1)
    elif important_matches > low_matches:
        subject_score = 0.6
    elif low_matches > important_matches:
        subject_score = 0.2
    else:
        subject_score = 0.4  # Neutral

    # Sender pattern importance (0.5 weight)
    sender_matches = sum(1 for p in IMPORTANT_SENDER_PATTERNS if p in from_email)
    if sender_matches > 0:
        sender_score = min(1.0, 0.5 + sender_matches * 0.25)
    else:
        sender_score = 0.3  # Neutral for generic noreply addresses

    # Combined score
    importance = 0.5 * subject_score + 0.5 * sender_score

    return min(1.0, max(0.0, importance))


def is_business_hours(hour: int, day_of_week: int) -> bool:
    """Check if time is during business hours (9-17, Mon-Fri)."""
    return 0 <= day_of_week <= 4 and 9 <= hour < 17


async def create_email_features_table(conn: asyncpg.Connection) -> None:
    """Create the email_features table if it doesn't exist."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS email_features (
            id SERIAL PRIMARY KEY,
            email_id INTEGER REFERENCES emails(id) UNIQUE,

            -- Relationship Features (11 dimensions)
            emails_from_sender_7d INTEGER,
            emails_from_sender_30d INTEGER,
            emails_from_sender_90d INTEGER,
            emails_from_sender_all INTEGER,
            user_replied_to_sender_count INTEGER,
            user_replied_to_sender_rate FLOAT,
            avg_response_time_hours FLOAT,
            user_initiated_ratio FLOAT,
            days_since_last_interaction INTEGER,
            sender_replies_to_you_rate FLOAT,
            relationship_strength FLOAT,

            -- Service Detection (7 dimensions)
            is_service_email BOOLEAN,
            service_confidence FLOAT,
            service_type TEXT,
            service_importance FLOAT,
            has_unsubscribe_link BOOLEAN,
            has_list_unsubscribe_header BOOLEAN,
            from_common_service_domain BOOLEAN,

            -- Temporal Features (8 dimensions)
            hour_of_day INTEGER,
            day_of_week INTEGER,
            is_weekend BOOLEAN,
            is_business_hours BOOLEAN,
            days_since_received INTEGER,
            is_recent BOOLEAN,
            time_bucket TEXT,
            urgency_score FLOAT,

            -- Content Basic (5 dimensions)
            subject_word_count INTEGER,
            body_word_count INTEGER,
            has_attachments BOOLEAN,
            attachment_count INTEGER,
            recipient_count INTEGER,

            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # Create index on email_id for fast joins
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_email_features_email_id
        ON email_features(email_id)
    """)

    print("email_features table created/verified")


async def compute_sender_stats(conn: asyncpg.Connection) -> dict:
    """Pre-compute sender statistics for relationship features.

    Returns dict: sender_email -> {
        emails_7d, emails_30d, emails_90d, emails_all,
        user_replied_count, total_received,
        avg_response_time, user_initiated_count,
        last_interaction_date, sender_replies_count
    }
    """
    print("Computing sender statistics...")

    # Get reference date (latest email)
    ref_date = await conn.fetchval("""
        SELECT MAX(date_parsed) FROM emails WHERE date_parsed IS NOT NULL
    """)
    if ref_date is None:
        ref_date = datetime.now()

    # Compute all sender stats in one query (using subquery for ref_date to avoid timezone issues)
    rows = await conn.fetch("""
        WITH ref AS (
            SELECT COALESCE(MAX(date_parsed), NOW()) as ref_date FROM emails
        ),
        sender_counts AS (
            SELECT
                LOWER(from_email) as sender,
                COUNT(*) as emails_all,
                COUNT(*) FILTER (WHERE date_parsed >= (SELECT ref_date - INTERVAL '7 days' FROM ref)) as emails_7d,
                COUNT(*) FILTER (WHERE date_parsed >= (SELECT ref_date - INTERVAL '30 days' FROM ref)) as emails_30d,
                COUNT(*) FILTER (WHERE date_parsed >= (SELECT ref_date - INTERVAL '90 days' FROM ref)) as emails_90d,
                MAX(date_parsed) as last_received
            FROM emails
            WHERE is_sent = FALSE
            GROUP BY LOWER(from_email)
        ),
        user_replies AS (
            -- Count how many times user replied to each sender
            SELECT
                LOWER(received.from_email) as sender,
                COUNT(DISTINCT received.id) as replied_count,
                AVG(EXTRACT(EPOCH FROM (sent.date_parsed - received.date_parsed))/3600) as avg_response_hours
            FROM emails received
            JOIN emails sent ON sent.in_reply_to = received.message_id
            WHERE received.is_sent = FALSE
              AND sent.is_sent = TRUE
            GROUP BY LOWER(received.from_email)
        ),
        user_initiated AS (
            -- Emails user sent TO each address (user initiated)
            SELECT
                LOWER(to_email) as recipient,
                COUNT(*) as initiated_count
            FROM emails, UNNEST(to_emails) as to_email
            WHERE is_sent = TRUE
            GROUP BY LOWER(to_email)
        ),
        sender_replies AS (
            -- How often sender replies to user's emails
            SELECT
                LOWER(sent.from_email) as sender,
                COUNT(*) FILTER (
                    WHERE EXISTS (
                        SELECT 1 FROM emails reply
                        WHERE reply.in_reply_to = sent.message_id
                          AND reply.is_sent = FALSE
                    )
                )::FLOAT / NULLIF(COUNT(*), 0) as reply_rate
            FROM emails sent
            WHERE sent.is_sent = TRUE
            GROUP BY LOWER(sent.from_email)
        )
        SELECT
            sc.sender,
            sc.emails_all,
            sc.emails_7d,
            sc.emails_30d,
            sc.emails_90d,
            sc.last_received,
            COALESCE(ur.replied_count, 0) as user_replied_count,
            COALESCE(ur.avg_response_hours, 0) as avg_response_hours,
            COALESCE(ui.initiated_count, 0) as user_initiated_count,
            COALESCE(sr.reply_rate, 0) as sender_reply_rate
        FROM sender_counts sc
        LEFT JOIN user_replies ur ON ur.sender = sc.sender
        LEFT JOIN user_initiated ui ON ui.recipient = sc.sender
        LEFT JOIN sender_replies sr ON sr.sender = sc.sender
    """)

    stats = {}
    for row in rows:
        stats[row['sender']] = {
            'emails_all': row['emails_all'],
            'emails_7d': row['emails_7d'],
            'emails_30d': row['emails_30d'],
            'emails_90d': row['emails_90d'],
            'last_received': row['last_received'],
            'user_replied_count': row['user_replied_count'],
            'avg_response_hours': row['avg_response_hours'],
            'user_initiated_count': row['user_initiated_count'],
            'sender_reply_rate': row['sender_reply_rate'],
        }

    print(f"  Computed stats for {len(stats)} senders")
    return stats, ref_date


def compute_relationship_strength(
    emails_30d: int,
    user_replied_rate: float,
    sender_reply_rate: float,
    days_since_interaction: int,
    user_initiated_ratio: float,
) -> float:
    """Compute relationship strength using user-specified weights.

    Formula (from user):
        0.35 * frequency +
        0.45 * engagement +
        0.10 * reciprocity +
        0.05 * recency +
        0.05 * balance
    """
    # Frequency: normalized emails in last 30 days (cap at 10)
    frequency = min(emails_30d / 10.0, 1.0)

    # Engagement: user's reply rate to this sender
    engagement = user_replied_rate

    # Reciprocity: sender's reply rate to user
    reciprocity = sender_reply_rate

    # Recency: inverse of days since last interaction (cap at 90)
    recency = 1.0 - min(days_since_interaction / 90.0, 1.0)

    # Balance: user initiated ratio
    balance = user_initiated_ratio

    strength = (
        0.35 * frequency +
        0.45 * engagement +
        0.10 * reciprocity +
        0.05 * recency +
        0.05 * balance
    )

    return min(1.0, max(0.0, strength))


def compute_urgency_score(
    relationship_strength: float,
    days_since_received: int,
    is_business_hours_flag: bool,
    has_attachments: bool,
) -> float:
    """Compute urgency score (0-1).

    Formula:
        0.4 * relationship_strength +
        0.3 * recency +
        0.2 * business_hours +
        0.1 * attachments
    """
    # Recency: 1.0 for today, decays over 7 days
    recency = max(0.0, 1.0 - (days_since_received / 7.0))

    urgency = (
        0.4 * relationship_strength +
        0.3 * recency +
        0.2 * (1.0 if is_business_hours_flag else 0.5) +
        0.1 * (1.0 if has_attachments else 0.0)
    )

    return min(1.0, max(0.0, urgency))


async def compute_features_batch(
    conn: asyncpg.Connection,
    email_ids: list[int],
    sender_stats: dict,
    ref_date: datetime,
) -> list[dict]:
    """Compute features for a batch of emails."""

    # Load email data
    rows = await conn.fetch("""
        SELECT
            e.id,
            e.from_email,
            e.subject,
            e.body_text,
            e.date_parsed,
            e.to_emails,
            e.cc_emails,
            e.labels,
            e.is_sent
        FROM emails e
        WHERE e.id = ANY($1)
    """, email_ids)

    features_list = []

    for row in rows:
        email_id = row['id']
        from_email = (row['from_email'] or '').lower()
        subject = row['subject'] or ''
        body = row['body_text'] or ''
        date_parsed = row['date_parsed']
        to_emails = row['to_emails'] or []
        cc_emails = row['cc_emails'] or []

        # Get sender stats
        sender = sender_stats.get(from_email, {})
        emails_all = sender.get('emails_all', 0)
        emails_7d = sender.get('emails_7d', 0)
        emails_30d = sender.get('emails_30d', 0)
        emails_90d = sender.get('emails_90d', 0)
        user_replied_count = sender.get('user_replied_count', 0)
        avg_response_hours = sender.get('avg_response_hours', 0)
        user_initiated_count = sender.get('user_initiated_count', 0)
        sender_reply_rate = sender.get('sender_reply_rate', 0)
        last_received = sender.get('last_received')

        # Compute relationship features
        user_replied_rate = user_replied_count / emails_all if emails_all > 0 else 0.0
        total_interactions = emails_all + user_initiated_count
        user_initiated_ratio = user_initiated_count / total_interactions if total_interactions > 0 else 0.0

        days_since_last = 0
        if last_received:
            days_since_last = (ref_date - last_received).days

        relationship_strength = compute_relationship_strength(
            emails_30d, user_replied_rate, sender_reply_rate,
            days_since_last, user_initiated_ratio
        )

        # Service detection (no headers available, rely on body/from/subject)
        is_service, service_conf, service_type = detect_service_email(
            from_email, subject, body, None
        )

        # Compute service importance (only meaningful for service emails)
        service_importance = compute_service_importance(from_email, subject, is_service)

        # Check for common service domain patterns
        from_service_domain = any(p in from_email for p in SERVICE_DOMAINS)

        # Check for unsubscribe in body
        plain_body = extract_plain_text(body)
        has_unsub_link = 'unsubscribe' in plain_body.lower()

        # Check List-Unsubscribe header (not available without raw headers, default to False)
        has_list_unsub = False

        # Temporal features
        hour = date_parsed.hour if date_parsed else 0
        dow = date_parsed.weekday() if date_parsed else 0
        is_weekend = dow >= 5
        is_biz_hours = is_business_hours(hour, dow)

        days_since = (ref_date - date_parsed).days if date_parsed else 0
        is_recent = days_since < 7
        time_bucket = get_time_bucket(hour)

        # Content features
        subject_words = count_words(subject)
        body_words = count_words(plain_body)

        # Attachments - check labels for has_attachments
        labels = row['labels'] or []
        has_attach = 'Has attachment' in labels or 'has:attachment' in str(labels).lower()
        attach_count = 1 if has_attach else 0  # Can't determine exact count from labels

        # Recipient count
        recipient_count = len(to_emails) + len(cc_emails)

        # Urgency score
        urgency = compute_urgency_score(
            relationship_strength, days_since, is_biz_hours, has_attach
        )

        features_list.append({
            'email_id': email_id,
            # Relationship
            'emails_from_sender_7d': emails_7d,
            'emails_from_sender_30d': emails_30d,
            'emails_from_sender_90d': emails_90d,
            'emails_from_sender_all': emails_all,
            'user_replied_to_sender_count': user_replied_count,
            'user_replied_to_sender_rate': user_replied_rate,
            'avg_response_time_hours': avg_response_hours,
            'user_initiated_ratio': user_initiated_ratio,
            'days_since_last_interaction': days_since_last,
            'sender_replies_to_you_rate': sender_reply_rate,
            'relationship_strength': relationship_strength,
            # Service
            'is_service_email': is_service,
            'service_confidence': service_conf,
            'service_type': service_type,
            'service_importance': service_importance,
            'has_unsubscribe_link': has_unsub_link,
            'has_list_unsubscribe_header': has_list_unsub,
            'from_common_service_domain': from_service_domain,
            # Temporal
            'hour_of_day': hour,
            'day_of_week': dow,
            'is_weekend': is_weekend,
            'is_business_hours': is_biz_hours,
            'days_since_received': days_since,
            'is_recent': is_recent,
            'time_bucket': time_bucket,
            'urgency_score': urgency,
            # Content
            'subject_word_count': subject_words,
            'body_word_count': body_words,
            'has_attachments': has_attach,
            'attachment_count': attach_count,
            'recipient_count': recipient_count,
        })

    return features_list


async def store_features_batch(
    conn: asyncpg.Connection,
    features_list: list[dict],
) -> int:
    """Store computed features to database."""
    if not features_list:
        return 0

    async with conn.transaction():
        await conn.executemany("""
            INSERT INTO email_features (
                email_id,
                emails_from_sender_7d, emails_from_sender_30d,
                emails_from_sender_90d, emails_from_sender_all,
                user_replied_to_sender_count, user_replied_to_sender_rate,
                avg_response_time_hours, user_initiated_ratio,
                days_since_last_interaction, sender_replies_to_you_rate,
                relationship_strength,
                is_service_email, service_confidence, service_type,
                service_importance,
                has_unsubscribe_link, has_list_unsubscribe_header,
                from_common_service_domain,
                hour_of_day, day_of_week, is_weekend, is_business_hours,
                days_since_received, is_recent, time_bucket, urgency_score,
                subject_word_count, body_word_count, has_attachments,
                attachment_count, recipient_count
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16, $17, $18, $19,
                $20, $21, $22, $23, $24, $25, $26, $27,
                $28, $29, $30, $31, $32
            )
            ON CONFLICT (email_id) DO UPDATE SET
                emails_from_sender_7d = EXCLUDED.emails_from_sender_7d,
                emails_from_sender_30d = EXCLUDED.emails_from_sender_30d,
                emails_from_sender_90d = EXCLUDED.emails_from_sender_90d,
                emails_from_sender_all = EXCLUDED.emails_from_sender_all,
                user_replied_to_sender_count = EXCLUDED.user_replied_to_sender_count,
                user_replied_to_sender_rate = EXCLUDED.user_replied_to_sender_rate,
                avg_response_time_hours = EXCLUDED.avg_response_time_hours,
                user_initiated_ratio = EXCLUDED.user_initiated_ratio,
                days_since_last_interaction = EXCLUDED.days_since_last_interaction,
                sender_replies_to_you_rate = EXCLUDED.sender_replies_to_you_rate,
                relationship_strength = EXCLUDED.relationship_strength,
                is_service_email = EXCLUDED.is_service_email,
                service_confidence = EXCLUDED.service_confidence,
                service_type = EXCLUDED.service_type,
                service_importance = EXCLUDED.service_importance,
                has_unsubscribe_link = EXCLUDED.has_unsubscribe_link,
                has_list_unsubscribe_header = EXCLUDED.has_list_unsubscribe_header,
                from_common_service_domain = EXCLUDED.from_common_service_domain,
                hour_of_day = EXCLUDED.hour_of_day,
                day_of_week = EXCLUDED.day_of_week,
                is_weekend = EXCLUDED.is_weekend,
                is_business_hours = EXCLUDED.is_business_hours,
                days_since_received = EXCLUDED.days_since_received,
                is_recent = EXCLUDED.is_recent,
                time_bucket = EXCLUDED.time_bucket,
                urgency_score = EXCLUDED.urgency_score,
                subject_word_count = EXCLUDED.subject_word_count,
                body_word_count = EXCLUDED.body_word_count,
                has_attachments = EXCLUDED.has_attachments,
                attachment_count = EXCLUDED.attachment_count,
                recipient_count = EXCLUDED.recipient_count,
                created_at = NOW()
        """, [
            (
                f['email_id'],
                f['emails_from_sender_7d'], f['emails_from_sender_30d'],
                f['emails_from_sender_90d'], f['emails_from_sender_all'],
                f['user_replied_to_sender_count'], f['user_replied_to_sender_rate'],
                f['avg_response_time_hours'], f['user_initiated_ratio'],
                f['days_since_last_interaction'], f['sender_replies_to_you_rate'],
                f['relationship_strength'],
                f['is_service_email'], f['service_confidence'], f['service_type'],
                f['service_importance'],
                f['has_unsubscribe_link'], f['has_list_unsubscribe_header'],
                f['from_common_service_domain'],
                f['hour_of_day'], f['day_of_week'], f['is_weekend'], f['is_business_hours'],
                f['days_since_received'], f['is_recent'], f['time_bucket'], f['urgency_score'],
                f['subject_word_count'], f['body_word_count'], f['has_attachments'],
                f['attachment_count'], f['recipient_count'],
            )
            for f in features_list
        ])

    return len(features_list)


async def verify_results(conn: asyncpg.Connection) -> None:
    """Verify feature computation results."""
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    # Coverage
    total = await conn.fetchval("SELECT COUNT(*) FROM emails WHERE is_sent = FALSE")
    with_features = await conn.fetchval("SELECT COUNT(*) FROM email_features")
    coverage = with_features * 100.0 / total if total > 0 else 0

    print(f"\nCoverage:")
    print(f"  Total received emails: {total:,}")
    print(f"  With features: {with_features:,}")
    print(f"  Coverage: {coverage:.1f}%")

    # Feature distributions
    stats = await conn.fetchrow("""
        SELECT
            AVG(relationship_strength) as avg_relationship,
            AVG(urgency_score) as avg_urgency,
            SUM(CASE WHEN is_service_email THEN 1 ELSE 0 END) as service_count,
            COUNT(DISTINCT service_type) as service_types
        FROM email_features
    """)

    print(f"\nFeature Distributions:")
    print(f"  Avg relationship strength: {stats['avg_relationship']:.3f}")
    print(f"  Avg urgency score: {stats['avg_urgency']:.3f}")
    print(f"  Service emails: {stats['service_count']:,}")
    print(f"  Service types: {stats['service_types']}")

    # Service type breakdown
    service_types = await conn.fetch("""
        SELECT service_type, COUNT(*) as cnt
        FROM email_features
        WHERE is_service_email = TRUE
        GROUP BY service_type
        ORDER BY cnt DESC
    """)
    print(f"\n  Service type breakdown:")
    for row in service_types:
        print(f"    {row['service_type'] or 'unknown':15s}: {row['cnt']:,}")

    # Service importance distribution
    importance_dist = await conn.fetch("""
        SELECT
            CASE
                WHEN service_importance >= 0.7 THEN 'HIGH (>=0.7)'
                WHEN service_importance >= 0.4 THEN 'MEDIUM (0.4-0.7)'
                ELSE 'LOW (<0.4)'
            END as importance_level,
            COUNT(*) as cnt
        FROM email_features
        WHERE is_service_email = TRUE
        GROUP BY importance_level
        ORDER BY importance_level
    """)
    print(f"\n  Service importance distribution:")
    for row in importance_dist:
        print(f"    {row['importance_level']:20s}: {row['cnt']:,}")

    # Top relationships
    print(f"\nTop 10 Relationships:")
    top = await conn.fetch("""
        SELECT e.from_email,
               ef.relationship_strength,
               ef.emails_from_sender_all,
               ef.user_replied_to_sender_rate
        FROM emails e
        JOIN email_features ef ON ef.email_id = e.id
        WHERE ef.relationship_strength > 0.3
        GROUP BY e.from_email, ef.relationship_strength,
                 ef.emails_from_sender_all, ef.user_replied_to_sender_rate
        ORDER BY ef.relationship_strength DESC
        LIMIT 10
    """)
    for row in top:
        print(f"  {row['from_email'][:40]:40s} "
              f"str={row['relationship_strength']:.2f} "
              f"n={row['emails_from_sender_all']} "
              f"rate={row['user_replied_to_sender_rate']:.2f}")

    print("=" * 60)


async def run_pipeline(batch_size: int = 1000) -> dict:
    """Run the feature computation pipeline."""
    print("=" * 60)
    print("Phase 2: Basic ML Features Computation")
    print("=" * 60)
    print()

    start_time = datetime.now()

    # Validate environment
    if not DB_URL:
        print("Error: DB_URL not set in .env")
        sys.exit(1)

    print(f"Database: {DB_URL}")
    print(f"Your email: {YOUR_EMAIL}")
    print()

    # Connect
    print("Connecting to database...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Create table
        await create_email_features_table(conn)

        # Pre-compute sender stats
        sender_stats, ref_date = await compute_sender_stats(conn)

        # Get email IDs to process (only received emails)
        print("\nFetching emails to process...")
        email_ids = await conn.fetch("""
            SELECT id FROM emails
            WHERE is_sent = FALSE
            ORDER BY date_parsed ASC
        """)
        email_ids = [r['id'] for r in email_ids]
        total = len(email_ids)
        print(f"  Found {total:,} received emails")

        # Process in batches
        print(f"\nComputing features (batch size: {batch_size})...")
        processed = 0

        with tqdm(total=total, desc="Processing") as pbar:
            for i in range(0, total, batch_size):
                batch_ids = email_ids[i:i + batch_size]

                features = await compute_features_batch(
                    conn, batch_ids, sender_stats, ref_date
                )
                stored = await store_features_batch(conn, features)

                processed += stored
                pbar.update(len(batch_ids))

        # Verify
        await verify_results(conn)

        duration = (datetime.now() - start_time).total_seconds()

        print(f"\nPipeline complete!")
        print(f"  Processed: {processed:,} emails")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Rate: {processed/duration:.1f} emails/sec")

        return {'processed': processed, 'duration': duration}

    finally:
        await conn.close()


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Compute basic ML features'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1000,
        help='Batch size (default: 1000)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing features'
    )

    args = parser.parse_args()

    if args.verify:
        conn = await asyncpg.connect(DB_URL)
        try:
            await verify_results(conn)
        finally:
            await conn.close()
    else:
        await run_pipeline(batch_size=args.batch_size)


if __name__ == "__main__":
    asyncio.run(main())
