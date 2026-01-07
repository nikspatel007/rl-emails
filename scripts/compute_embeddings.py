#!/usr/bin/env python3
"""Generate embeddings for emails using LiteLLM and store in pgvector.

Phase 3 of the email enrichment pipeline.
Uses text-embedding-3-small (1536 dims) via LiteLLM for unified API access.
Stores embeddings in email_embeddings table + local JSONL backup.

Features:
- Parallel processing with configurable workers (default: 10)
- Proper email cleanup (HTML strip, quote removal, signature removal)
- Includes importance metadata in embedding text
- Truncates long emails to fit context window
- Resume capability (skips already-embedded emails)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

if TYPE_CHECKING:
    import tiktoken

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

import psycopg2
from psycopg2.extras import execute_values
from bs4 import BeautifulSoup

try:
    from mailparser_reply import EmailReplyParser  # type: ignore[import-not-found]
    HAS_MAIL_PARSER = True
except ImportError:
    HAS_MAIL_PARSER = False
    class EmailReplyParser:  # type: ignore[no-redef]
        """Stub for when mail-parser-reply is not installed."""
        @staticmethod
        def read(text: str) -> Any:
            return None

try:
    from litellm import embedding
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

try:
    import tiktoken as _tiktoken
    TOKENIZER: tiktoken.Encoding | None = _tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    TOKENIZER = None
    HAS_TIKTOKEN = False

# Configuration - DATABASE_URL is required
DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    print("ERROR: DATABASE_URL environment variable is required")
    sys.exit(1)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
DEFAULT_WORKERS = 10  # Parallel API calls
MAX_TOKENS = 8000  # Leave buffer under 8191 limit

# Backup directory
BACKUP_DIR = Path(__file__).parent.parent / "backups" / "embeddings"


# =============================================================================
# Token Counting Utilities
# =============================================================================

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (accurate) or estimate (fallback)."""
    if HAS_TIKTOKEN and TOKENIZER:
        return len(TOKENIZER.encode(text))
    # Fallback: conservative estimate (3 chars per token)
    return len(text) // 3


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit."""
    if not text:
        return ""

    if HAS_TIKTOKEN and TOKENIZER:
        tokens = TOKENIZER.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return TOKENIZER.decode(truncated_tokens) + "..."
    else:
        # Fallback: conservative character limit
        max_chars = max_tokens * 3
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."


# =============================================================================
# Email Text Cleaning Utilities (Reusable)
# =============================================================================

def strip_html(html_body: str) -> str:
    """Extract plain text from HTML email body.

    Removes:
    - Script and style elements
    - HTML tags
    - Extra whitespace
    """
    if not html_body:
        return ""

    try:
        soup = BeautifulSoup(html_body, 'html.parser')

        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'head', 'meta', 'link', 'noscript']):
            element.decompose()

        # Get text with space separator
        text = soup.get_text(separator=' ')

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception:
        return html_body


def strip_template_syntax(text: str) -> str:
    """Remove Liquid/Jinja template syntax from text.

    These often appear in marketing emails as raw template code.
    """
    if not text:
        return ""

    # Remove {% ... %} blocks
    text = re.sub(r'\{%.*?%\}', '', text, flags=re.DOTALL)

    # Remove {{ ... }} blocks
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)

    # Remove {# ... #} comments
    text = re.sub(r'\{#.*?#\}', '', text, flags=re.DOTALL)

    # Clean up resulting whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def strip_quoted_replies(text: str) -> str:
    """Remove quoted reply chains and signatures from email text.

    Uses mail-parser-reply library for robust detection.
    Falls back to regex patterns if library not available.
    """
    if not text:
        return ""

    if HAS_MAIL_PARSER and EmailReplyParser is not None:
        try:
            parsed = EmailReplyParser.read(text)
            # Get just the latest reply, stripped of quotes/signatures
            clean: str = str(parsed.text_reply) if parsed.text_reply else ""
            if clean and len(clean.strip()) > 10:
                return clean.strip()
        except Exception:
            pass  # Fall back to regex

    # Fallback: regex-based quote removal
    lines = text.split('\n')
    clean_lines = []

    for line in lines:
        # Skip quoted lines
        if line.strip().startswith('>'):
            continue
        # Skip "On ... wrote:" lines
        if re.match(r'^On .+ wrote:$', line.strip()):
            continue
        # Skip "From: ... Sent: ..." headers
        if re.match(r'^(From|Sent|To|Subject|Date):', line.strip()):
            continue
        # Skip signature markers
        if line.strip() in ['--', '---', '— ', 'Sent from my iPhone', 'Sent from my iPad']:
            break  # Stop at signature
        clean_lines.append(line)

    return '\n'.join(clean_lines).strip()


def clean_email_text(html_body: str, max_tokens: int = MAX_TOKENS) -> str:
    """Full email text cleaning pipeline.

    Steps:
    1. Strip HTML tags
    2. Strip template syntax (Liquid/Jinja)
    3. Strip quoted replies and signatures
    4. Truncate to max tokens

    Args:
        html_body: Raw email body (may be HTML or plain text)
        max_tokens: Maximum tokens to return

    Returns:
        Clean plain text suitable for embedding
    """
    if not html_body:
        return ""

    # Step 1: Strip HTML
    text = strip_html(html_body)

    # Step 2: Strip template syntax
    text = strip_template_syntax(text)

    # Step 3: Strip quoted replies and signatures
    text = strip_quoted_replies(text)

    # Step 4: Truncate to token limit
    text = truncate_to_tokens(text, max_tokens)

    return text


# =============================================================================
# Embedding Text Construction
# =============================================================================

def build_embedding_text(subject: str, body: str, is_service: bool,
                         service_importance: float, relationship_strength: float) -> str:
    """Build embedding text with importance metadata.

    Format:
    [TYPE: PERSON|SERVICE] [PRIORITY: HIGH|MEDIUM|LOW]
    [SUBJECT] ...
    [BODY] ...

    Uses token-based truncation to stay under MAX_TOKENS limit.
    """
    # Determine type and priority
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

    # Build metadata line
    metadata = f"[TYPE: {email_type}] [PRIORITY: {priority}]"

    # Build subject line
    subject_line = f"[SUBJECT] {subject.strip()}" if subject else ""

    # Calculate tokens used by metadata + subject
    header_text = metadata + "\n" + subject_line if subject_line else metadata
    header_tokens = count_tokens(header_text)

    # Reserve tokens for body (leave buffer for [BODY] prefix and newlines)
    body_token_budget = MAX_TOKENS - header_tokens - 20

    # Clean and truncate body to fit token budget
    if body and body_token_budget > 100:
        # First clean the body (HTML, templates, quotes)
        clean_body = strip_html(body)
        clean_body = strip_template_syntax(clean_body)
        clean_body = strip_quoted_replies(clean_body)

        if clean_body:
            # Truncate to token budget
            clean_body = truncate_to_tokens(clean_body, body_token_budget)
            body_line = f"[BODY] {clean_body}"
        else:
            body_line = ""
    else:
        body_line = ""

    # Assemble final text
    parts = [metadata]
    if subject_line:
        parts.append(subject_line)
    if body_line:
        parts.append(body_line)

    return "\n".join(parts)


# =============================================================================
# Database Operations
# =============================================================================

def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def get_unprocessed_emails(
    conn: psycopg2.extensions.connection, limit: int = 1000
) -> list[dict[str, Any]]:
    """Get emails that don't have embeddings yet, with features."""
    with conn.cursor() as cur:
        cur.execute("""
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
        """, (limit,))

        rows = cur.fetchall()
        return [{
            "id": row[0],
            "subject": row[1],
            "body": row[2],
            "is_service": row[3],
            "service_importance": row[4],
            "relationship_strength": row[5]
        } for row in rows]


def get_embedding_counts(conn: psycopg2.extensions.connection) -> tuple[int, int]:
    """Get total eligible and embedded counts."""
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
    """Save embeddings to email_embeddings table."""
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
            template="(%s, %s::vector, %s, %s, %s)"
        )

        conn.commit()
        return len(values)


# =============================================================================
# Backup Operations
# =============================================================================

def save_embeddings_to_jsonl(
    embeddings_data: list[tuple[int, list[float], int, str]], batch_num: int
) -> Path | None:
    """Save embeddings to local JSONL file for backup."""
    if not embeddings_data:
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_file = BACKUP_DIR / f"embeddings_batch_{batch_num:04d}.jsonl"

    with open(backup_file, 'w') as f:
        for email_id, emb, token_count, content_hash in embeddings_data:
            record = {
                "email_id": email_id,
                "embedding": emb,
                "token_count": token_count,
                "content_hash": content_hash,
                "model": EMBEDDING_MODEL
            }
            f.write(json.dumps(record) + "\n")

    return backup_file


def save_metadata(
    total_processed: int, total_time: float, total_emails: int, workers: int
) -> Path:
    """Save metadata about the embedding run."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIM,
        "workers": workers,
        "total_processed": total_processed,
        "total_emails": total_emails,
        "processing_time_seconds": total_time,
        "emails_per_second": total_processed / total_time if total_time > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }

    metadata_file = BACKUP_DIR / "embeddings_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata_file


# =============================================================================
# Parallel Processing
# =============================================================================

async def generate_single_embedding(
    text: str, semaphore: asyncio.Semaphore
) -> list[float]:
    """Generate embedding for a single text using LiteLLM."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: embedding(model=EMBEDDING_MODEL, input=[text])
        )
        return list(response.data[0]["embedding"])


async def process_emails_parallel(
    emails: list[dict[str, Any]], workers: int
) -> list[tuple[int, list[float], int, str]]:
    """Process emails in parallel with specified number of workers."""
    semaphore = asyncio.Semaphore(workers)

    async def process_one(
        email: dict[str, Any]
    ) -> tuple[int, list[float], int, str] | None:
        try:
            text = build_embedding_text(
                email["subject"] or "",
                email["body"] or "",
                email["is_service"],
                email["service_importance"],
                email["relationship_strength"]
            )

            if not text.strip() or len(text) < 10:
                return None

            content_hash = compute_content_hash(text)
            token_count = count_tokens(text)  # Accurate count via tiktoken

            emb = await generate_single_embedding(text, semaphore)

            return (email["id"], emb, token_count, content_hash)
        except Exception as e:
            print(f"  Error processing email {email['id']}: {e}", file=sys.stderr)
            return None

    tasks = [process_one(email) for email in emails]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


# =============================================================================
# Main Entry Point
# =============================================================================

async def main_async(args: argparse.Namespace) -> int:
    """Main async entry point."""
    workers = args.workers
    batch_size = args.batch_size

    print("=" * 60)
    print("Phase 3: Email Embeddings (LiteLLM + Parallel)")
    print("=" * 60)
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Dimensions: {EMBEDDING_DIM}")
    print(f"  Workers: {workers} parallel")
    print(f"  Batch size: {batch_size}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Email cleanup: HTML + templates + quotes/signatures")
    print(f"  Backup dir: {BACKUP_DIR}")
    print()

    # Check dependencies
    if not HAS_TIKTOKEN:
        print("WARNING: tiktoken not installed, using fallback token estimation")
        print("  Install with: uv pip install tiktoken")
    else:
        print("✓ tiktoken available (accurate token counting)")

    if not HAS_MAIL_PARSER:
        print("WARNING: mail-parser-reply not installed, using fallback quote removal")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1

    print("✓ OPENAI_API_KEY found")

    try:
        conn = psycopg2.connect(DB_URL)
        print("✓ Connected to database")
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        return 1

    try:
        total, already_embedded = get_embedding_counts(conn)
        remaining = total - already_embedded

        if args.limit:
            remaining = min(remaining, args.limit)

        print()
        print("Database status:")
        print(f"  Total received emails: {total:,}")
        print(f"  Already embedded:      {already_embedded:,}")
        print(f"  To process:            {remaining:,}")
        print()

        if remaining == 0:
            print("All emails already have embeddings!")
            return 0

        total_processed = 0
        start_time = time.time()
        batch_num = 0

        while total_processed < remaining:
            batch_num += 1
            fetch_limit = min(batch_size, remaining - total_processed)
            emails = get_unprocessed_emails(conn, limit=fetch_limit)

            if not emails:
                print("\nNo more unprocessed emails.")
                break

            print(f"\nBatch {batch_num}: Processing {len(emails)} emails with {workers} workers...")
            batch_start = time.time()

            embeddings_data = await process_emails_parallel(emails, workers)

            saved = save_embeddings_to_db(conn, embeddings_data)
            save_embeddings_to_jsonl(embeddings_data, batch_num)

            total_processed += saved
            batch_time = time.time() - batch_start

            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining_now = remaining - total_processed
            eta_seconds = remaining_now / rate if rate > 0 else 0

            print(f"  ✓ Saved {saved}/{len(emails)} embeddings in {batch_time:.1f}s")
            print(f"  Progress: {total_processed:,}/{remaining:,} ({total_processed/remaining*100:.1f}%)")
            print(f"  Rate: {rate:.1f} emails/s | ETA: {eta_seconds/60:.1f} min")

        total_time = time.time() - start_time
        metadata_file = save_metadata(total_processed, total_time, total, workers)

        print()
        print("=" * 60)
        print("PHASE 3 COMPLETE")
        print("=" * 60)
        print()
        print(f"Total processed:  {total_processed:,}")
        print(f"Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Avg rate:         {total_processed/total_time:.1f} emails/s")
        print(f"Workers used:     {workers}")
        print(f"Backup location:  {BACKUP_DIR}")
        print("=" * 60)

    finally:
        conn.close()

    return 0


def main() -> int:
    if not HAS_LITELLM:
        print("ERROR: litellm package not installed")
        print("Install with: uv pip install litellm")
        return 1

    parser = argparse.ArgumentParser(description="Generate embeddings for emails using LiteLLM")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--batch-size", "-b", type=int, default=100,
                        help="Emails per batch (default: 100)")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Max emails to process")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
