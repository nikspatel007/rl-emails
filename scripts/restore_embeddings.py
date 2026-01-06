#!/usr/bin/env python3
"""Restore email embeddings from JSONL backup files.

This script restores embeddings from the local JSONL backup files created
by compute_embeddings.py. Use this when you need to:
- Restore embeddings after database reset
- Migrate embeddings to a new database
- Recover from data loss

Usage:
    python scripts/restore_embeddings.py                    # Restore all
    python scripts/restore_embeddings.py --verify           # Verify only
    python scripts/restore_embeddings.py --clear --restore  # Clear and restore
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/gmail_twoyrs"
BACKUP_DIR = Path(__file__).parent.parent / "backups" / "embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_backup_files() -> list[Path]:
    """Get all JSONL backup files sorted by batch number."""
    files = list(BACKUP_DIR.glob("embeddings_batch_*.jsonl"))
    return sorted(files, key=lambda f: int(f.stem.split("_")[-1]))


def count_backup_records() -> int:
    """Count total records in backup files."""
    total = 0
    for f in get_backup_files():
        with open(f) as fp:
            total += sum(1 for _ in fp)
    return total


def verify_backups() -> dict:
    """Verify backup files and return stats."""
    files = get_backup_files()

    stats = {
        "backup_dir": str(BACKUP_DIR),
        "num_files": len(files),
        "total_records": 0,
        "email_ids": set(),
        "models": set(),
        "valid": True,
        "errors": []
    }

    for f in files:
        try:
            with open(f) as fp:
                for line_num, line in enumerate(fp, 1):
                    try:
                        record = json.loads(line)
                        stats["total_records"] += 1
                        stats["email_ids"].add(record["email_id"])
                        stats["models"].add(record.get("model", "unknown"))

                        # Verify embedding dimensions
                        if len(record.get("embedding", [])) != 1536:
                            stats["errors"].append(f"{f.name}:{line_num}: wrong dimensions")
                            stats["valid"] = False
                    except json.JSONDecodeError as e:
                        stats["errors"].append(f"{f.name}:{line_num}: JSON error: {e}")
                        stats["valid"] = False
        except Exception as e:
            stats["errors"].append(f"{f.name}: {e}")
            stats["valid"] = False

    stats["unique_emails"] = len(stats["email_ids"])
    del stats["email_ids"]  # Don't include full set in output
    stats["models"] = list(stats["models"])

    return stats


def clear_embeddings(conn) -> int:
    """Clear all embeddings from database."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM email_embeddings")
        count = cur.fetchone()[0]

        if count > 0:
            cur.execute("TRUNCATE email_embeddings")
            conn.commit()

        return count


def restore_from_jsonl(conn, batch_size: int = 1000) -> int:
    """Restore embeddings from JSONL backup files."""
    files = get_backup_files()

    if not files:
        print("ERROR: No backup files found")
        return 0

    total_restored = 0
    batch = []

    for f in files:
        with open(f) as fp:
            for line in fp:
                record = json.loads(line)

                embedding_str = f"[{','.join(str(x) for x in record['embedding'])}]"
                batch.append((
                    record["email_id"],
                    embedding_str,
                    record.get("model", EMBEDDING_MODEL),
                    record.get("token_count", 0),
                    record.get("content_hash", "")
                ))

                if len(batch) >= batch_size:
                    saved = save_batch(conn, batch)
                    total_restored += saved
                    print(f"  Restored {total_restored:,} embeddings...")
                    batch = []

    # Save remaining
    if batch:
        saved = save_batch(conn, batch)
        total_restored += saved

    return total_restored


def save_batch(conn, batch: list[tuple]) -> int:
    """Save a batch of embeddings to database."""
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO email_embeddings (email_id, embedding, model, token_count, content_hash)
            VALUES %s
            ON CONFLICT (email_id) DO NOTHING
            """,
            batch,
            template="(%s, %s::vector, %s, %s, %s)"
        )
        conn.commit()
        return len(batch)


def get_db_count(conn) -> int:
    """Get current embedding count in database."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM email_embeddings")
        return cur.fetchone()[0]


def main():
    parser = argparse.ArgumentParser(description="Restore email embeddings from backup")
    parser.add_argument("--verify", action="store_true", help="Verify backups only")
    parser.add_argument("--clear", action="store_true", help="Clear existing embeddings first")
    parser.add_argument("--restore", action="store_true", help="Restore from backup")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for restore")
    args = parser.parse_args()

    print("=" * 60)
    print("Email Embeddings Restore Tool")
    print("=" * 60)
    print(f"Backup directory: {BACKUP_DIR}")
    print()

    # Always verify first
    print("Verifying backups...")
    stats = verify_backups()
    print(f"  Files: {stats['num_files']}")
    print(f"  Records: {stats['total_records']:,}")
    print(f"  Unique emails: {stats['unique_emails']:,}")
    print(f"  Models: {stats['models']}")
    print(f"  Valid: {'Yes' if stats['valid'] else 'No'}")

    if stats["errors"]:
        print(f"  Errors ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            print(f"    - {err}")
        if len(stats["errors"]) > 5:
            print(f"    ... and {len(stats['errors']) - 5} more")

    if args.verify:
        return 0 if stats["valid"] else 1

    if not args.restore:
        print()
        print("Use --restore to restore embeddings from backup")
        print("Use --clear --restore to clear existing and restore")
        return 0

    if not stats["valid"]:
        print()
        print("ERROR: Backup files have errors. Fix before restoring.")
        return 1

    # Connect to database
    try:
        conn = psycopg2.connect(DB_URL)
        print()
        print(f"Connected to database")
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        return 1

    try:
        current_count = get_db_count(conn)
        print(f"Current embeddings in database: {current_count:,}")

        if args.clear:
            print()
            print("Clearing existing embeddings...")
            cleared = clear_embeddings(conn)
            print(f"  Cleared {cleared:,} embeddings")

        print()
        print("Restoring from backup...")
        restored = restore_from_jsonl(conn, args.batch_size)

        final_count = get_db_count(conn)

        print()
        print("=" * 60)
        print("RESTORE COMPLETE")
        print("=" * 60)
        print(f"Restored: {restored:,}")
        print(f"Total in database: {final_count:,}")
        print("=" * 60)

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
