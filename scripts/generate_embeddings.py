#!/usr/bin/env python3
"""Generate OpenAI embeddings for emails and store in pgvector.

Uses text-embedding-3-small (1536 dims) for efficient similarity search.
Processes in batches with rate limiting.
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass

import asyncpg

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100  # OpenAI batch limit
WORKERS = 8  # Concurrent API calls


def truncate_text(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to approximate token limit (rough estimate: 4 chars/token)."""
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def prepare_email_text(subject: str, body: str, from_email: str = None) -> str:
    """Prepare email text for embedding."""
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if from_email:
        parts.append(f"From: {from_email}")
    if body:
        parts.append(body[:4000])  # Limit body to ~1000 tokens

    text = "\n".join(parts)
    return truncate_text(text, 8000)


async def get_unprocessed_emails(conn: asyncpg.Connection, limit: int = 1000) -> list[dict]:
    """Get emails that don't have embeddings yet."""
    rows = await conn.fetch("""
        SELECT e.id, e.message_id, e.subject, e.body_text, e.from_email
        FROM emails e
        JOIN email_features ef ON e.id = ef.email_id
        WHERE ef.content_embedding IS NULL
          AND e.body_text IS NOT NULL
          AND length(e.body_text) > 50
        ORDER BY e.date_parsed DESC
        LIMIT $1
    """, limit)

    return [dict(row) for row in rows]


async def get_embedding_counts(conn: asyncpg.Connection) -> tuple[int, int]:
    """Get total eligible and embedded counts."""
    total = await conn.fetchval("""
        SELECT COUNT(*) FROM emails e
        JOIN email_features ef ON e.id = ef.email_id
        WHERE e.body_text IS NOT NULL AND length(e.body_text) > 50
    """)
    embedded = await conn.fetchval("""
        SELECT COUNT(*) FROM email_features
        WHERE content_embedding IS NOT NULL
    """)
    return total, embedded


async def generate_embeddings_batch(
    client: "AsyncOpenAI",
    texts: list[str],
    semaphore: asyncio.Semaphore
) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    async with semaphore:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]


async def save_embeddings(
    conn: asyncpg.Connection,
    embeddings_data: list[tuple[int, str, list[float]]]
) -> int:
    """Save embeddings to email_features table."""
    saved = 0
    for email_id, message_id, embedding in embeddings_data:
        try:
            # Convert list to pgvector format
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            await conn.execute("""
                UPDATE email_features SET
                    content_embedding = $2::vector,
                    embedding_model = $3,
                    embedding_dim = $4,
                    computed_at = NOW()
                WHERE email_id = $1
            """, email_id, embedding_str, EMBEDDING_MODEL, EMBEDDING_DIM)
            saved += 1
        except Exception as e:
            print(f"  Error saving embedding for {email_id}: {e}", file=sys.stderr)
    return saved


async def process_batch(
    client: "AsyncOpenAI",
    conn: asyncpg.Connection,
    emails: list[dict],
    semaphore: asyncio.Semaphore
) -> tuple[int, float]:
    """Process a batch of emails: generate embeddings and save."""
    start_time = time.time()

    # Prepare texts
    texts = [
        prepare_email_text(
            e["subject"] or "",
            e["body_text"] or "",
            e["from_email"]
        )
        for e in emails
    ]

    # Generate embeddings
    try:
        embeddings = await generate_embeddings_batch(client, texts, semaphore)
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return 0, time.time() - start_time

    # Save to database
    embeddings_data = [
        (emails[i]["id"], emails[i]["message_id"], embeddings[i])
        for i in range(len(emails))
    ]
    saved = await save_embeddings(conn, embeddings_data)

    return saved, time.time() - start_time


async def main_async(args):
    """Main async entry point."""
    batch_size = args.batch_size

    print("=" * 60)
    print("OpenAI Embedding Generation for Emails")
    print("=" * 60)
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Dimensions: {EMBEDDING_DIM}")
    print(f"  Batch size: {batch_size}")
    print()

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1

    print("1. API key found")

    # Connect to database
    conn = await asyncpg.connect(DB_URL)

    try:
        total, already_embedded = await get_embedding_counts(conn)
        remaining = total - already_embedded

        print()
        print(f"Database status:")
        print(f"  Total eligible emails: {total:,}")
        print(f"  Already embedded:      {already_embedded:,}")
        print(f"  Remaining:             {remaining:,}")
        print()

        if remaining == 0:
            print("All emails already have embeddings!")
            return 0

        # Initialize OpenAI client
        client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(WORKERS)

        total_processed = 0
        start_time = time.time()
        batch_num = 0

        while True:
            batch_num += 1
            emails = await get_unprocessed_emails(conn, limit=batch_size)

            if not emails:
                print("\nNo more unprocessed emails.")
                break

            print(f"\nBatch {batch_num}: Processing {len(emails)} emails...")
            sys.stdout.flush()

            saved, batch_time = await process_batch(client, conn, emails, semaphore)
            total_processed += saved

            # Progress report
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining_now = remaining - total_processed
            eta_seconds = remaining_now / rate if rate > 0 else 0

            print(f"  Saved {saved} embeddings in {batch_time:.1f}s")
            print(f"  Progress: {total_processed:,}/{remaining:,} ({total_processed/remaining*100:.1f}%)")
            print(f"  Rate: {rate:.2f} emails/s | ETA: {eta_seconds/60:.0f} min")
            sys.stdout.flush()

        total_time = time.time() - start_time

        print()
        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print()
        print(f"Total processed: {total_processed:,}")
        print(f"Total time:      {total_time/60:.1f} minutes")
        print(f"Avg rate:        {total_processed/total_time:.2f} emails/s")
        print("=" * 60)

    finally:
        await conn.close()

    return 0


def main():
    if not HAS_OPENAI:
        print("ERROR: openai package not installed")
        print("Install with: pip install openai")
        return 1

    parser = argparse.ArgumentParser(description="Generate OpenAI embeddings for emails")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Emails per batch")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
