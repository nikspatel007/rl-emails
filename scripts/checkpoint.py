#!/usr/bin/env python3
"""Create or restore checkpoints of email embeddings data.

This script manages checkpoints that include:
- SQL dump of email_embeddings table
- JSONL backup files
- Metadata about the checkpoint

Usage:
    python scripts/checkpoint.py create              # Create checkpoint
    python scripts/checkpoint.py create --name v1   # Create named checkpoint
    python scripts/checkpoint.py list               # List checkpoints
    python scripts/checkpoint.py restore <name>     # Restore from checkpoint
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import psycopg2

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/gmail_twoyrs"
BACKUP_DIR = Path(__file__).parent.parent / "backups"
EMBEDDINGS_DIR = BACKUP_DIR / "embeddings"
CHECKPOINTS_DIR = BACKUP_DIR / "checkpoints"
PG_DUMP = "/opt/homebrew/Cellar/postgresql@16/16.11/bin/pg_dump"
PSQL = "/opt/homebrew/Cellar/postgresql@16/16.11/bin/psql"


def get_db_stats() -> dict:
    """Get current database statistics."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    stats = {}

    cur.execute("SELECT COUNT(*) FROM email_embeddings")
    stats["embedding_count"] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM emails WHERE is_sent = FALSE")
    stats["email_count"] = cur.fetchone()[0]

    if stats["embedding_count"] > 0:
        cur.execute("SELECT MIN(token_count), AVG(token_count)::int, MAX(token_count) FROM email_embeddings")
        row = cur.fetchone()
        stats["token_min"] = row[0]
        stats["token_avg"] = row[1]
        stats["token_max"] = row[2]

    conn.close()
    return stats


def create_checkpoint(name: str = None) -> Path:
    """Create a checkpoint with SQL dump and metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = name or f"checkpoint_{timestamp}"

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Creating checkpoint: {checkpoint_name}")
    print()

    # Get stats
    print("Gathering database stats...")
    stats = get_db_stats()
    print(f"  Embeddings: {stats['embedding_count']:,}")
    print(f"  Emails: {stats['email_count']:,}")

    # Create SQL dump
    sql_file = CHECKPOINTS_DIR / f"{checkpoint_name}.sql"
    print(f"\nCreating SQL dump...")

    env = os.environ.copy()
    env["PGPASSWORD"] = "postgres"

    result = subprocess.run(
        [PG_DUMP, "-h", "localhost", "-p", "5433", "-U", "postgres",
         "-d", "gmail_twoyrs", "-t", "email_embeddings",
         "--no-owner", "--no-acl"],
        capture_output=True,
        text=True,
        env=env
    )

    if result.returncode != 0:
        print(f"ERROR: pg_dump failed: {result.stderr}")
        return None

    with open(sql_file, "w") as f:
        f.write(result.stdout)

    sql_size = sql_file.stat().st_size / (1024 * 1024)
    print(f"  SQL dump: {sql_file.name} ({sql_size:.1f} MB)")

    # Create metadata
    metadata = {
        "name": checkpoint_name,
        "timestamp": datetime.now().isoformat(),
        "database": "gmail_twoyrs",
        "table": "email_embeddings",
        "stats": stats,
        "sql_file": sql_file.name,
        "jsonl_dir": "embeddings",
        "jsonl_files": len(list(EMBEDDINGS_DIR.glob("embeddings_batch_*.jsonl"))),
    }

    metadata_file = CHECKPOINTS_DIR / f"{checkpoint_name}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata: {metadata_file.name}")

    # Create tar.gz archive with SQL dump and JSONL files
    archive_file = CHECKPOINTS_DIR / f"{checkpoint_name}.tar.gz"
    print(f"\nCreating archive...")

    with tarfile.open(archive_file, "w:gz") as tar:
        tar.add(sql_file, arcname=sql_file.name)
        tar.add(metadata_file, arcname=metadata_file.name)

        # Add JSONL files
        for jsonl_file in sorted(EMBEDDINGS_DIR.glob("embeddings_batch_*.jsonl")):
            tar.add(jsonl_file, arcname=f"embeddings/{jsonl_file.name}")

        # Add metadata from embeddings dir
        emb_meta = EMBEDDINGS_DIR / "embeddings_metadata.json"
        if emb_meta.exists():
            tar.add(emb_meta, arcname="embeddings/embeddings_metadata.json")

    archive_size = archive_file.stat().st_size / (1024 * 1024)
    print(f"  Archive: {archive_file.name} ({archive_size:.1f} MB)")

    print()
    print("=" * 60)
    print(f"CHECKPOINT CREATED: {checkpoint_name}")
    print("=" * 60)
    print(f"  Archive: {archive_file}")
    print(f"  Size: {archive_size:.1f} MB")
    print(f"  Embeddings: {stats['embedding_count']:,}")
    print()
    print("Restore with:")
    print(f"  python scripts/checkpoint.py restore {checkpoint_name}")
    print("=" * 60)

    return archive_file


def list_checkpoints():
    """List available checkpoints."""
    print("Available checkpoints:")
    print()

    archives = sorted(CHECKPOINTS_DIR.glob("*.tar.gz"))

    if not archives:
        print("  No checkpoints found")
        return

    for archive in archives:
        name = archive.stem.replace(".tar", "")
        size = archive.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(archive.stat().st_mtime)

        # Try to load metadata
        metadata_file = CHECKPOINTS_DIR / f"{name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
                count = meta.get("stats", {}).get("embedding_count", "?")
                print(f"  {name}")
                print(f"    Created: {mtime.strftime('%Y-%m-%d %H:%M')}")
                print(f"    Size: {size:.1f} MB")
                print(f"    Embeddings: {count:,}" if isinstance(count, int) else f"    Embeddings: {count}")
                print()
        else:
            print(f"  {name} ({size:.1f} MB) - {mtime.strftime('%Y-%m-%d %H:%M')}")


def restore_checkpoint(name: str, method: str = "sql"):
    """Restore from a checkpoint."""
    archive_file = CHECKPOINTS_DIR / f"{name}.tar.gz"

    if not archive_file.exists():
        # Try without .tar.gz
        archive_file = CHECKPOINTS_DIR / f"{name}"
        if not archive_file.exists():
            print(f"ERROR: Checkpoint not found: {name}")
            return False

    print(f"Restoring checkpoint: {name}")
    print(f"Method: {method}")
    print()

    # Extract archive to temp dir
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("Extracting archive...")
        with tarfile.open(archive_file, "r:gz") as tar:
            tar.extractall(tmpdir)

        # Load metadata
        metadata_file = tmpdir / f"{name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
            print(f"  Checkpoint from: {meta.get('timestamp', 'unknown')}")
            print(f"  Embeddings: {meta.get('stats', {}).get('embedding_count', '?'):,}")

        if method == "sql":
            # Restore from SQL dump
            sql_file = tmpdir / f"{name}.sql"
            if not sql_file.exists():
                print(f"ERROR: SQL file not found in archive")
                return False

            print()
            print("Clearing existing embeddings...")
            conn = psycopg2.connect(DB_URL)
            cur = conn.cursor()
            cur.execute("TRUNCATE email_embeddings")
            conn.commit()
            conn.close()

            print("Restoring from SQL dump...")
            env = os.environ.copy()
            env["PGPASSWORD"] = "postgres"

            result = subprocess.run(
                [PSQL, "-h", "localhost", "-p", "5433", "-U", "postgres",
                 "-d", "gmail_twoyrs", "-f", str(sql_file)],
                capture_output=True,
                text=True,
                env=env
            )

            if result.returncode != 0:
                print(f"ERROR: psql failed: {result.stderr}")
                return False

        else:
            # Restore from JSONL (uses restore_embeddings.py logic)
            print("Restoring from JSONL...")
            # Copy JSONL files to backup dir temporarily and use restore script
            jsonl_dir = tmpdir / "embeddings"
            if jsonl_dir.exists():
                # Use the restore_embeddings module
                from restore_embeddings import clear_embeddings, restore_from_jsonl

                conn = psycopg2.connect(DB_URL)
                clear_embeddings(conn)

                # Temporarily swap EMBEDDINGS_DIR
                import restore_embeddings
                old_dir = restore_embeddings.BACKUP_DIR
                restore_embeddings.BACKUP_DIR = jsonl_dir

                restored = restore_from_jsonl(conn, batch_size=1000)
                restore_embeddings.BACKUP_DIR = old_dir

                conn.close()
                print(f"  Restored {restored:,} embeddings from JSONL")

    # Verify
    print()
    print("Verifying restore...")
    stats = get_db_stats()

    print()
    print("=" * 60)
    print("RESTORE COMPLETE")
    print("=" * 60)
    print(f"  Embeddings in database: {stats['embedding_count']:,}")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Checkpoint management for email embeddings")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a checkpoint")
    create_parser.add_argument("--name", help="Checkpoint name (default: timestamp)")

    # List command
    subparsers.add_parser("list", help="List checkpoints")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from checkpoint")
    restore_parser.add_argument("name", help="Checkpoint name")
    restore_parser.add_argument("--method", choices=["sql", "jsonl"], default="sql",
                                help="Restore method (default: sql)")

    args = parser.parse_args()

    if args.command == "create":
        create_checkpoint(args.name)
    elif args.command == "list":
        list_checkpoints()
    elif args.command == "restore":
        restore_checkpoint(args.name, args.method)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
