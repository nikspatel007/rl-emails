"""Stage 6: Compute email embeddings using OpenAI.

Generates embeddings for emails using text-embedding-3-small model.
Stores results in email_embeddings table with pgvector.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any

import psycopg2
from bs4 import BeautifulSoup
from psycopg2.extras import execute_values

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_TOKENS = 8000
DEFAULT_WORKERS = 10
DEFAULT_BATCH_SIZE = 100


def count_tokens(text: str) -> int:
    """Count tokens using conservative character estimate.

    Args:
        text: Text to count tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // 3


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit.

    Args:
        text: Text to truncate.
        max_tokens: Maximum tokens allowed.

    Returns:
        Truncated text.
    """
    if not text:
        return ""

    max_chars = max_tokens * 3
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def strip_html(html_body: str) -> str:
    """Extract plain text from HTML email body.

    Args:
        html_body: Raw HTML body.

    Returns:
        Plain text content.
    """
    if not html_body:
        return ""

    try:
        soup = BeautifulSoup(html_body, "html.parser")

        for element in soup(["script", "style", "head", "meta", "link", "noscript"]):
            element.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return html_body


def strip_template_syntax(text: str) -> str:
    """Remove Liquid/Jinja template syntax from text.

    Args:
        text: Text with potential template syntax.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    text = re.sub(r"\{%.*?%\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{#.*?#\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def strip_quoted_replies(text: str) -> str:
    """Remove quoted reply chains and signatures from email text.

    Args:
        text: Email text with potential quotes.

    Returns:
        Cleaned text without quotes/signatures.
    """
    if not text:
        return ""

    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        if line.strip().startswith(">"):
            continue
        if re.match(r"^On .+ wrote:$", line.strip()):
            continue
        if re.match(r"^(From|Sent|To|Subject|Date):", line.strip()):
            continue
        if line.strip() in ["--", "---", "-- ", "Sent from my iPhone", "Sent from my iPad"]:
            break
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def build_embedding_text(
    subject: str,
    body: str,
    is_service: bool,
    service_importance: float,
    relationship_strength: float,
) -> str:
    """Build embedding text with importance metadata.

    Args:
        subject: Email subject.
        body: Email body.
        is_service: Whether this is a service email.
        service_importance: Service importance score.
        relationship_strength: Relationship strength score.

    Returns:
        Formatted text for embedding.
    """
    if is_service:
        email_type = "SERVICE"
        if service_importance >= 0.7:
            priority = "HIGH"
        elif service_importance >= 0.4:
            priority = "MEDIUM"
        else:
            priority = "LOW"
    else:
        email_type = "PERSON"
        if relationship_strength > 0.5:
            priority = "HIGH"
        elif relationship_strength > 0.2:
            priority = "MEDIUM"
        else:
            priority = "LOW"

    metadata = f"[TYPE: {email_type}] [PRIORITY: {priority}]"
    subject_line = f"[SUBJECT] {subject.strip()}" if subject else ""

    header_text = metadata + "\n" + subject_line if subject_line else metadata
    header_tokens = count_tokens(header_text)
    body_token_budget = MAX_TOKENS - header_tokens - 20

    if body and body_token_budget > 100:
        clean_body = strip_html(body)
        clean_body = strip_template_syntax(clean_body)
        clean_body = strip_quoted_replies(clean_body)

        if clean_body:
            clean_body = truncate_to_tokens(clean_body, body_token_budget)
            body_line = f"[BODY] {clean_body}"
        else:
            body_line = ""
    else:
        body_line = ""

    parts = [metadata]
    if subject_line:
        parts.append(subject_line)
    if body_line:
        parts.append(body_line)

    return "\n".join(parts)


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of content for deduplication.

    Args:
        text: Text to hash.

    Returns:
        First 16 characters of SHA256 hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create email_embeddings table if it doesn't exist.

    Args:
        conn: Database connection.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_embeddings (
                id SERIAL PRIMARY KEY,
                email_id INTEGER REFERENCES emails(id) UNIQUE,
                embedding vector(1536),
                model TEXT,
                token_count INTEGER,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_email_embeddings_email_id
            ON email_embeddings(email_id)
        """
        )

        conn.commit()


def get_unprocessed_emails(
    conn: psycopg2.extensions.connection, limit: int = 1000
) -> list[dict[str, Any]]:
    """Get emails that don't have embeddings yet.

    Args:
        conn: Database connection.
        limit: Maximum number of emails to return.

    Returns:
        List of email dictionaries with features.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, e.subject, e.body_text,
                   COALESCE(ef.is_service_email, FALSE) as is_service,
                   COALESCE(ef.service_importance, 0.5) as service_importance,
                   COALESCE(ef.relationship_strength, 0.5) as relationship_strength
            FROM emails e
            LEFT JOIN email_embeddings ee ON e.id = ee.email_id
            LEFT JOIN email_features ef ON e.id = ef.email_id
            WHERE ee.id IS NULL
              AND e.is_sent = FALSE
            ORDER BY e.id
            LIMIT %s
        """,
            (limit,),
        )

        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "subject": row[1],
                "body": row[2],
                "is_service": row[3],
                "service_importance": row[4],
                "relationship_strength": row[5],
            }
            for row in rows
        ]


def get_embedding_counts(conn: psycopg2.extensions.connection) -> tuple[int, int]:
    """Get total eligible and embedded counts.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (total_eligible, already_embedded).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM emails WHERE is_sent = FALSE")
        row = cur.fetchone()
        total = row[0] if row else 0

        cur.execute("SELECT COUNT(*) FROM email_embeddings")
        row = cur.fetchone()
        embedded = row[0] if row else 0

        return total, embedded


def save_embeddings_to_db(
    conn: psycopg2.extensions.connection,
    embeddings_data: list[tuple[int, list[float], int, str]],
) -> int:
    """Save embeddings to email_embeddings table.

    Args:
        conn: Database connection.
        embeddings_data: List of (email_id, embedding, token_count, content_hash).

    Returns:
        Number of embeddings saved.
    """
    if not embeddings_data:
        return 0

    with conn.cursor() as cur:
        values = []
        for email_id, emb, token_count, content_hash in embeddings_data:
            embedding_str = f"[{','.join(str(x) for x in emb)}]"
            values.append((email_id, embedding_str, EMBEDDING_MODEL, token_count, content_hash))

        execute_values(
            cur,
            """
            INSERT INTO email_embeddings (email_id, embedding, model, token_count, content_hash)
            VALUES %s
            ON CONFLICT (email_id) DO NOTHING
            """,
            values,
            template="(%s, %s::vector, %s, %s, %s)",
        )

        conn.commit()
        return len(values)


async def generate_single_embedding(
    text: str, semaphore: asyncio.Semaphore, embedding_func: Any
) -> list[float]:
    """Generate embedding for a single text.

    Args:
        text: Text to embed.
        semaphore: Semaphore for rate limiting.
        embedding_func: Function to generate embedding.

    Returns:
        List of embedding floats.
    """
    async with semaphore:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: embedding_func(model=EMBEDDING_MODEL, input=[text])
        )
        return list(response.data[0]["embedding"])


async def process_emails_parallel(
    emails: list[dict[str, Any]], workers: int, embedding_func: Any
) -> list[tuple[int, list[float], int, str]]:
    """Process emails in parallel with specified number of workers.

    Args:
        emails: List of email dictionaries.
        workers: Number of parallel workers.
        embedding_func: Function to generate embeddings.

    Returns:
        List of (email_id, embedding, token_count, content_hash).
    """
    semaphore = asyncio.Semaphore(workers)

    async def process_one(
        email: dict[str, Any],
    ) -> tuple[int, list[float], int, str] | None:
        try:
            text = build_embedding_text(
                email["subject"] or "",
                email["body"] or "",
                email["is_service"],
                email["service_importance"],
                email["relationship_strength"],
            )

            if not text.strip() or len(text) < 10:
                return None

            content_hash = compute_content_hash(text)
            token_count = count_tokens(text)

            emb = await generate_single_embedding(text, semaphore, embedding_func)

            return (email["id"], emb, token_count, content_hash)
        except Exception:
            return None

    tasks = [process_one(email) for email in emails]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


async def compute_embeddings_async(
    conn: psycopg2.extensions.connection,
    embedding_func: Any,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compute embeddings for all unprocessed emails.

    Args:
        conn: Database connection.
        embedding_func: Function to generate embeddings.
        workers: Number of parallel workers.
        batch_size: Emails per batch.
        limit: Maximum emails to process (None for all).

    Returns:
        Statistics dictionary.
    """
    total, already_embedded = get_embedding_counts(conn)
    remaining = total - already_embedded

    if limit:
        remaining = min(remaining, limit)

    if remaining == 0:
        return {
            "total_emails": total,
            "already_embedded": already_embedded,
            "processed": 0,
            "batches": 0,
        }

    total_processed = 0
    batch_num = 0

    while total_processed < remaining:
        batch_num += 1
        fetch_limit = min(batch_size, remaining - total_processed)
        emails = get_unprocessed_emails(conn, limit=fetch_limit)

        if not emails:
            break

        embeddings_data = await process_emails_parallel(emails, workers, embedding_func)
        saved = save_embeddings_to_db(conn, embeddings_data)
        total_processed += saved

    return {
        "total_emails": total,
        "already_embedded": already_embedded,
        "processed": total_processed,
        "batches": batch_num,
    }


def run(
    config: Config,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> StageResult:
    """Run Stage 6: Compute email embeddings.

    Args:
        config: Application configuration.
        workers: Number of parallel workers.
        batch_size: Emails per batch.
        limit: Maximum emails to process.

    Returns:
        StageResult with computation statistics.
    """
    start_time = time.time()

    if not config.openai_api_key:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="OPENAI_API_KEY not configured",
        )

    try:
        from litellm import embedding as litellm_embedding
    except ImportError:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="litellm package not installed",
        )

    conn = psycopg2.connect(config.database_url)
    try:
        create_tables(conn)

        stats = asyncio.run(
            compute_embeddings_async(
                conn,
                litellm_embedding,
                workers=workers,
                batch_size=batch_size,
                limit=limit,
            )
        )

        duration = time.time() - start_time

        return StageResult(
            success=True,
            records_processed=stats["processed"],
            duration_seconds=duration,
            message=f"Generated embeddings for {stats['processed']} emails",
            metadata=stats,
        )
    finally:
        conn.close()
