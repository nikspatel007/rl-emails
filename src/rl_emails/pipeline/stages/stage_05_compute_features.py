"""Stage 5: Compute ML features for emails.

Computes 30+ features per email for smart ranking and filtering:
- Relationship Features (11 dimensions)
- Service Detection (7 dimensions)
- Temporal Features (8 dimensions)
- Content Basic (5 dimensions)
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from typing import Any

import asyncpg
from bs4 import BeautifulSoup

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult

# Service email detection patterns
SERVICE_DOMAINS = [
    "noreply",
    "no-reply",
    "donotreply",
    "do-not-reply",
    "notifications",
    "notification",
    "alerts",
    "alert",
    "mailer",
    "mail",
    "newsletter",
    "updates",
    "info",
    "support",
    "service",
    "system",
    "automated",
    "auto",
    "bounce",
    "postmaster",
    "mailer-daemon",
]

SERVICE_TYPES = {
    "newsletter": ["newsletter", "digest", "weekly", "daily", "monthly", "subscribe"],
    "transactional": [
        "order",
        "receipt",
        "invoice",
        "payment",
        "shipping",
        "delivery",
        "confirmation",
    ],
    "notification": ["notification", "alert", "reminder", "update"],
    "marketing": ["offer", "discount", "sale", "promo", "deal", "exclusive"],
    "social": ["linkedin", "facebook", "twitter", "instagram", "social"],
}

IMPORTANT_SUBJECT_KEYWORDS = [
    "order",
    "shipped",
    "delivered",
    "delivery",
    "payment",
    "transaction",
    "confirm",
    "confirmed",
    "confirmation",
    "receipt",
    "invoice",
    "alert",
    "security",
    "verification",
    "verify",
    "password",
    "login",
    "suspicious",
    "fraud",
    "unauthorized",
    "expired",
    "expiring",
    "renewal",
    "bill",
    "due",
    "overdue",
    "failed",
    "declined",
    "refund",
    "return",
]

LOW_IMPORTANCE_KEYWORDS = [
    "offer",
    "sale",
    "deal",
    "save",
    "discount",
    "promo",
    "promotional",
    "newsletter",
    "digest",
    "weekly",
    "daily",
    "tips",
    "recommendation",
    "suggested",
    "trending",
    "popular",
    "new arrivals",
    "just for you",
    "exclusive",
    "limited time",
    "don't miss",
    "check out",
]

IMPORTANT_SENDER_PATTERNS = [
    "order",
    "shipping",
    "shipment",
    "tracking",
    "confirm",
    "alert",
    "security",
    "verify",
    "transaction",
    "payment",
    "billing",
    "receipt",
    "return",
    "refund",
    "delivery",
    "support",
]


def extract_plain_text(html_body: str | None) -> str:
    """Extract plain text from HTML email body.

    Args:
        html_body: HTML content.

    Returns:
        Plain text extracted from HTML.
    """
    if not html_body:
        return ""

    soup = BeautifulSoup(html_body, "html.parser")

    for element in soup(["script", "style", "head", "meta", "link"]):
        element.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def count_words(text: str | None) -> int:
    """Count words in text.

    Args:
        text: Text to count words in.

    Returns:
        Word count.
    """
    if not text:
        return 0
    return len(text.split())


def detect_service_email(
    from_email: str | None,
    subject: str | None,
    body: str | None,
) -> tuple[bool, float, str | None]:
    """Detect if email is from a service/automated sender.

    Args:
        from_email: Sender email address.
        subject: Email subject.
        body: Email body text.

    Returns:
        Tuple of (is_service, confidence, service_type).
    """
    from_email = (from_email or "").lower()
    subject = (subject or "").lower()
    body = (body or "").lower()

    signals: list[tuple[str, float]] = []

    # Check for service domain patterns
    for pattern in SERVICE_DOMAINS:
        if pattern in from_email:
            signals.append(("domain", 0.3))
            break

    # Check for unsubscribe in body
    if "unsubscribe" in body:
        signals.append(("unsubscribe_body", 0.3))

    # Determine service type
    detected_type = None
    for stype, keywords in SERVICE_TYPES.items():
        for kw in keywords:
            if kw in subject or kw in from_email:
                detected_type = stype
                signals.append(("type_match", 0.2))
                break
        if detected_type:
            break

    if not signals:
        return False, 0.0, None

    confidence = min(1.0, sum(s[1] for s in signals))
    is_service = confidence >= 0.3

    return is_service, confidence, detected_type


def get_time_bucket(hour: int) -> str:
    """Convert hour (0-23) to time bucket.

    Args:
        hour: Hour of day (0-23).

    Returns:
        Time bucket string.
    """
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def compute_service_importance(
    from_email: str | None,
    subject: str | None,
    is_service: bool,
) -> float:
    """Compute importance score for service emails (0-1).

    Args:
        from_email: Sender email address.
        subject: Email subject.
        is_service: Whether email is from a service.

    Returns:
        Importance score (0.0-1.0).
    """
    if not is_service:
        return 0.0

    from_email = (from_email or "").lower()
    subject = (subject or "").lower()

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
        subject_score = 0.4

    sender_matches = sum(1 for p in IMPORTANT_SENDER_PATTERNS if p in from_email)
    if sender_matches > 0:
        sender_score = min(1.0, 0.5 + sender_matches * 0.25)
    else:
        sender_score = 0.3

    importance = 0.5 * subject_score + 0.5 * sender_score
    return min(1.0, max(0.0, importance))


def is_business_hours(hour: int, day_of_week: int) -> bool:
    """Check if time is during business hours (9-17, Mon-Fri).

    Args:
        hour: Hour of day (0-23).
        day_of_week: Day of week (0=Monday, 6=Sunday).

    Returns:
        True if business hours.
    """
    return 0 <= day_of_week <= 4 and 9 <= hour < 17


def compute_relationship_strength(
    emails_30d: int,
    user_replied_rate: float,
    sender_reply_rate: float,
    days_since_interaction: int,
    user_initiated_ratio: float,
) -> float:
    """Compute relationship strength score.

    Args:
        emails_30d: Emails from sender in last 30 days.
        user_replied_rate: Rate user replies to sender.
        sender_reply_rate: Rate sender replies to user.
        days_since_interaction: Days since last interaction.
        user_initiated_ratio: Ratio of user-initiated emails.

    Returns:
        Relationship strength (0.0-1.0).
    """
    frequency = min(emails_30d / 10.0, 1.0)
    engagement = user_replied_rate
    reciprocity = sender_reply_rate
    recency = 1.0 - min(days_since_interaction / 90.0, 1.0)
    balance = user_initiated_ratio

    strength = (
        0.35 * frequency + 0.45 * engagement + 0.10 * reciprocity + 0.05 * recency + 0.05 * balance
    )

    return min(1.0, max(0.0, strength))


def compute_urgency_score(
    relationship_strength: float,
    days_since_received: int,
    is_business_hours_flag: bool,
    has_attachments: bool,
) -> float:
    """Compute urgency score (0-1).

    Args:
        relationship_strength: Relationship strength score.
        days_since_received: Days since email was received.
        is_business_hours_flag: Whether received during business hours.
        has_attachments: Whether email has attachments.

    Returns:
        Urgency score (0.0-1.0).
    """
    recency = max(0.0, 1.0 - (days_since_received / 7.0))

    urgency = (
        0.4 * relationship_strength
        + 0.3 * recency
        + 0.2 * (1.0 if is_business_hours_flag else 0.5)
        + 0.1 * (1.0 if has_attachments else 0.0)
    )

    return min(1.0, max(0.0, urgency))


async def create_email_features_table(conn: asyncpg.Connection) -> None:
    """Create the email_features table if it doesn't exist.

    Args:
        conn: Database connection.
    """
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS email_features (
            id SERIAL PRIMARY KEY,
            email_id INTEGER REFERENCES emails(id) UNIQUE,
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
            is_service_email BOOLEAN,
            service_confidence FLOAT,
            service_type TEXT,
            service_importance FLOAT,
            has_unsubscribe_link BOOLEAN,
            has_list_unsubscribe_header BOOLEAN,
            from_common_service_domain BOOLEAN,
            hour_of_day INTEGER,
            day_of_week INTEGER,
            is_weekend BOOLEAN,
            is_business_hours BOOLEAN,
            days_since_received INTEGER,
            is_recent BOOLEAN,
            time_bucket TEXT,
            urgency_score FLOAT,
            subject_word_count INTEGER,
            body_word_count INTEGER,
            has_attachments BOOLEAN,
            attachment_count INTEGER,
            recipient_count INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """
    )

    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_email_features_email_id
        ON email_features(email_id)
    """
    )


async def compute_sender_stats(
    conn: asyncpg.Connection,
) -> tuple[dict[str, dict[str, Any]], datetime]:
    """Pre-compute sender statistics for relationship features.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (stats_dict, reference_date).
    """
    ref_date = await conn.fetchval(
        """
        SELECT MAX(date_parsed) FROM emails WHERE date_parsed IS NOT NULL
    """
    )
    if ref_date is None:
        ref_date = datetime.now()

    rows = await conn.fetch(
        """
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
            SELECT
                LOWER(to_email) as recipient,
                COUNT(*) as initiated_count
            FROM emails, UNNEST(to_emails) as to_email
            WHERE is_sent = TRUE
            GROUP BY LOWER(to_email)
        ),
        sender_replies AS (
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
    """
    )

    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        stats[row["sender"]] = {
            "emails_all": row["emails_all"],
            "emails_7d": row["emails_7d"],
            "emails_30d": row["emails_30d"],
            "emails_90d": row["emails_90d"],
            "last_received": row["last_received"],
            "user_replied_count": row["user_replied_count"],
            "avg_response_hours": row["avg_response_hours"],
            "user_initiated_count": row["user_initiated_count"],
            "sender_reply_rate": row["sender_reply_rate"],
        }

    return stats, ref_date


async def compute_features_batch(
    conn: asyncpg.Connection,
    email_ids: list[int],
    sender_stats: dict[str, dict[str, Any]],
    ref_date: datetime,
) -> list[dict[str, Any]]:
    """Compute features for a batch of emails.

    Args:
        conn: Database connection.
        email_ids: List of email IDs to process.
        sender_stats: Pre-computed sender statistics.
        ref_date: Reference date for calculations.

    Returns:
        List of feature dictionaries.
    """
    rows = await conn.fetch(
        """
        SELECT
            e.id,
            e.from_email,
            e.subject,
            e.body_text,
            e.date_parsed,
            e.to_emails,
            e.cc_emails,
            e.labels
        FROM emails e
        WHERE e.id = ANY($1)
    """,
        email_ids,
    )

    features_list: list[dict[str, Any]] = []

    for row in rows:
        email_id = row["id"]
        from_email = (row["from_email"] or "").lower()
        subject = row["subject"] or ""
        body = row["body_text"] or ""
        date_parsed = row["date_parsed"]
        to_emails = row["to_emails"] or []
        cc_emails = row["cc_emails"] or []

        sender = sender_stats.get(from_email, {})
        emails_all = sender.get("emails_all", 0)
        emails_7d = sender.get("emails_7d", 0)
        emails_30d = sender.get("emails_30d", 0)
        emails_90d = sender.get("emails_90d", 0)
        user_replied_count = sender.get("user_replied_count", 0)
        avg_response_hours = sender.get("avg_response_hours", 0)
        user_initiated_count = sender.get("user_initiated_count", 0)
        sender_reply_rate = sender.get("sender_reply_rate", 0)
        last_received = sender.get("last_received")

        user_replied_rate = user_replied_count / emails_all if emails_all > 0 else 0.0
        total_interactions = emails_all + user_initiated_count
        user_initiated_ratio = (
            user_initiated_count / total_interactions if total_interactions > 0 else 0.0
        )

        days_since_last = 0
        if last_received:
            days_since_last = (ref_date - last_received).days

        relationship_str = compute_relationship_strength(
            emails_30d,
            user_replied_rate,
            sender_reply_rate,
            days_since_last,
            user_initiated_ratio,
        )

        is_service, service_conf, service_type = detect_service_email(from_email, subject, body)

        service_importance = compute_service_importance(from_email, subject, is_service)
        from_service_domain = any(p in from_email for p in SERVICE_DOMAINS)

        plain_body = extract_plain_text(body)
        has_unsub_link = "unsubscribe" in plain_body.lower()

        hour = date_parsed.hour if date_parsed else 0
        dow = date_parsed.weekday() if date_parsed else 0
        is_weekend = dow >= 5
        is_biz_hours = is_business_hours(hour, dow)

        days_since = (ref_date - date_parsed).days if date_parsed else 0
        is_recent = days_since < 7
        time_bucket = get_time_bucket(hour)

        subject_words = count_words(subject)
        body_words = count_words(plain_body)

        labels = row["labels"] or []
        has_attach = "Has attachment" in labels or "has:attachment" in str(labels).lower()
        attach_count = 1 if has_attach else 0

        recipient_count = len(to_emails) + len(cc_emails)

        urgency = compute_urgency_score(relationship_str, days_since, is_biz_hours, has_attach)

        features_list.append(
            {
                "email_id": email_id,
                "emails_from_sender_7d": emails_7d,
                "emails_from_sender_30d": emails_30d,
                "emails_from_sender_90d": emails_90d,
                "emails_from_sender_all": emails_all,
                "user_replied_to_sender_count": user_replied_count,
                "user_replied_to_sender_rate": user_replied_rate,
                "avg_response_time_hours": avg_response_hours,
                "user_initiated_ratio": user_initiated_ratio,
                "days_since_last_interaction": days_since_last,
                "sender_replies_to_you_rate": sender_reply_rate,
                "relationship_strength": relationship_str,
                "is_service_email": is_service,
                "service_confidence": service_conf,
                "service_type": service_type,
                "service_importance": service_importance,
                "has_unsubscribe_link": has_unsub_link,
                "has_list_unsubscribe_header": False,
                "from_common_service_domain": from_service_domain,
                "hour_of_day": hour,
                "day_of_week": dow,
                "is_weekend": is_weekend,
                "is_business_hours": is_biz_hours,
                "days_since_received": days_since,
                "is_recent": is_recent,
                "time_bucket": time_bucket,
                "urgency_score": urgency,
                "subject_word_count": subject_words,
                "body_word_count": body_words,
                "has_attachments": has_attach,
                "attachment_count": attach_count,
                "recipient_count": recipient_count,
            }
        )

    return features_list


async def store_features_batch(
    conn: asyncpg.Connection,
    features_list: list[dict[str, Any]],
) -> int:
    """Store computed features to database.

    Args:
        conn: Database connection.
        features_list: List of feature dictionaries.

    Returns:
        Number of features stored.
    """
    if not features_list:
        return 0

    async with conn.transaction():
        await conn.executemany(
            """
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
        """,
            [
                (
                    f["email_id"],
                    f["emails_from_sender_7d"],
                    f["emails_from_sender_30d"],
                    f["emails_from_sender_90d"],
                    f["emails_from_sender_all"],
                    f["user_replied_to_sender_count"],
                    f["user_replied_to_sender_rate"],
                    f["avg_response_time_hours"],
                    f["user_initiated_ratio"],
                    f["days_since_last_interaction"],
                    f["sender_replies_to_you_rate"],
                    f["relationship_strength"],
                    f["is_service_email"],
                    f["service_confidence"],
                    f["service_type"],
                    f["service_importance"],
                    f["has_unsubscribe_link"],
                    f["has_list_unsubscribe_header"],
                    f["from_common_service_domain"],
                    f["hour_of_day"],
                    f["day_of_week"],
                    f["is_weekend"],
                    f["is_business_hours"],
                    f["days_since_received"],
                    f["is_recent"],
                    f["time_bucket"],
                    f["urgency_score"],
                    f["subject_word_count"],
                    f["body_word_count"],
                    f["has_attachments"],
                    f["attachment_count"],
                    f["recipient_count"],
                )
                for f in features_list
            ],
        )

    return len(features_list)


def _convert_db_url(db_url: str) -> str:
    """Convert postgresql:// URL to postgres:// for asyncpg compatibility.

    Args:
        db_url: Database URL.

    Returns:
        URL with postgres:// prefix.
    """
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgres://", 1)
    return db_url


async def run_async(db_url: str, batch_size: int = 1000) -> dict[str, Any]:
    """Async implementation of feature computation.

    Args:
        db_url: Database URL.
        batch_size: Number of emails per batch.

    Returns:
        Statistics dictionary.
    """
    conn = await asyncpg.connect(db_url)
    try:
        await create_email_features_table(conn)

        sender_stats, ref_date = await compute_sender_stats(conn)

        email_ids = await conn.fetch(
            """
            SELECT id FROM emails
            WHERE is_sent = FALSE
            ORDER BY date_parsed ASC
        """
        )
        email_ids_list = [r["id"] for r in email_ids]
        total = len(email_ids_list)

        processed = 0
        for i in range(0, total, batch_size):
            batch_ids = email_ids_list[i : i + batch_size]
            features = await compute_features_batch(conn, batch_ids, sender_stats, ref_date)
            stored = await store_features_batch(conn, features)
            processed += stored

        return {
            "total_emails": total,
            "processed": processed,
            "senders": len(sender_stats),
        }
    finally:
        await conn.close()


def run(config: Config, *, batch_size: int = 1000) -> StageResult:
    """Run Stage 5: Compute ML features for emails.

    Args:
        config: Application configuration.
        batch_size: Number of emails per batch.

    Returns:
        StageResult with computation statistics.
    """
    start_time = time.time()

    db_url = _convert_db_url(config.database_url)
    stats = asyncio.run(run_async(db_url, batch_size))
    duration = time.time() - start_time

    return StageResult(
        success=True,
        records_processed=stats["processed"],
        duration_seconds=duration,
        message=f"Computed features for {stats['processed']} emails",
        metadata=stats,
    )
