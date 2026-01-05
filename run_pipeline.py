#!/usr/bin/env python3
"""
Email ML Pipeline Orchestrator

Runs the complete email analysis pipeline from a Gmail MBOX export.

Usage:
    python run_pipeline.py /path/to/your.mbox [--data-dir ./data] [--skip-embeddings]

Pipeline stages:
    1. parse_mbox.py        - Parse MBOX to JSONL
    2. import_to_postgres.py - Import to PostgreSQL
    3. populate_threads.py   - Build thread relationships
    4. generate_embeddings.py - Create OpenAI embeddings (optional)
    5. mine_gmail_labels.py  - Extract projects from labels
    6. discover_participant_projects.py - Find recurring groups
    7. cluster_embeddings.py - Semantic topic clustering
    8. dedupe_projects.py    - Merge duplicate projects
    9. detect_priority_contexts.py - Find high-engagement periods

After pipeline completes, run the labeling UI:
    streamlit run apps/labeling_ui.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_path: Path, args: list[str] = None, env: dict = None) -> bool:
    """Run a Python script and return success status."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    script_env = os.environ.copy()
    if env:
        script_env.update(env)

    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=script_env)

    if result.returncode != 0:
        print(f"ERROR: {script_path.name} failed with code {result.returncode}")
        return False

    print(f"SUCCESS: {script_path.name} completed")
    return True


def check_postgres() -> bool:
    """Check if PostgreSQL is running and accessible."""
    try:
        import asyncpg
        import asyncio

        async def check():
            conn = await asyncpg.connect(
                "postgresql://postgres:postgres@localhost:5433/rl_emails"
            )
            await conn.close()
            return True

        return asyncio.run(check())
    except Exception as e:
        print(f"PostgreSQL check failed: {e}")
        return False


def check_openai_key() -> bool:
    """Check if OpenAI API key is set."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def main():
    parser = argparse.ArgumentParser(
        description="Run the email ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "mbox_path",
        type=Path,
        help="Path to Gmail MBOX file"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/gmail"),
        help="Directory for intermediate files (default: ./data/gmail)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (requires OpenAI API key)"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip embedding-based clustering"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        choices=range(1, 10),
        help="Start from step N (1-9, for resuming failed runs)"
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://postgres:postgres@localhost:5433/rl_emails",
        help="PostgreSQL connection URL"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.mbox_path.exists():
        print(f"ERROR: MBOX file not found: {args.mbox_path}")
        sys.exit(1)

    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    parsed_jsonl = args.data_dir / "parsed_emails.jsonl"
    enriched_jsonl = args.data_dir / "enriched_emails.jsonl"

    # Check prerequisites
    print("\n=== Checking Prerequisites ===")

    if not check_postgres():
        print("\nERROR: PostgreSQL is not running or not accessible.")
        print("Start the database with: docker compose up -d postgres")
        print("Or: ./scripts/start_db.sh")
        sys.exit(1)
    print("PostgreSQL: OK")

    has_openai = check_openai_key()
    if has_openai:
        print("OpenAI API Key: OK")
    else:
        print("OpenAI API Key: Not set (embeddings will be skipped)")
        args.skip_embeddings = True

    # Scripts directory
    scripts_dir = Path(__file__).parent / "scripts"

    # Environment variables for scripts
    env = {
        "MBOX_PATH": str(args.mbox_path.absolute()),
        "DATA_DIR": str(args.data_dir.absolute()),
        "PARSED_JSONL": str(parsed_jsonl.absolute()),
        "ENRICHED_JSONL": str(enriched_jsonl.absolute()),
        "DB_URL": args.db_url,
    }

    print(f"\n=== Pipeline Configuration ===")
    print(f"MBOX Path: {args.mbox_path}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Database: {args.db_url}")
    print(f"Skip Embeddings: {args.skip_embeddings}")
    print(f"Starting from step: {args.start_from}")

    start_time = datetime.now()
    print(f"\n=== Starting Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    steps = [
        # Step 1: Parse MBOX
        (1, "parse_mbox.py", [], "Parse MBOX to JSONL"),

        # Step 2: Import to PostgreSQL
        (2, "import_to_postgres.py", [], "Import to PostgreSQL"),

        # Step 3: Populate threads
        (3, "populate_threads.py", [], "Build thread relationships"),

        # Step 4: Generate embeddings (optional)
        (4, "generate_embeddings.py", [], "Generate embeddings"),

        # Step 5: Mine Gmail labels
        (5, "mine_gmail_labels.py", [], "Extract label-based projects"),

        # Step 6: Discover participant projects
        (6, "discover_participant_projects.py", [], "Find participant groups"),

        # Step 7: Cluster embeddings
        (7, "cluster_embeddings.py", [], "Semantic clustering"),

        # Step 8: Dedupe projects
        (8, "dedupe_projects.py", [], "Merge duplicates"),

        # Step 9: Detect priority contexts
        (9, "detect_priority_contexts.py", [], "Find high-engagement periods"),
    ]

    failed = False
    for step_num, script_name, script_args, description in steps:
        if step_num < args.start_from:
            print(f"\nSkipping step {step_num}: {description}")
            continue

        # Skip embedding-related steps if requested
        if args.skip_embeddings and script_name == "generate_embeddings.py":
            print(f"\nSkipping step {step_num}: {description} (--skip-embeddings)")
            continue

        if args.skip_clustering and script_name == "cluster_embeddings.py":
            print(f"\nSkipping step {step_num}: {description} (--skip-clustering)")
            continue

        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"\nWARNING: Script not found: {script_path}")
            continue

        success = run_script(script_path, script_args, env)
        if not success:
            failed = True
            print(f"\nPipeline failed at step {step_num}: {script_name}")
            print(f"Fix the issue and resume with: python run_pipeline.py {args.mbox_path} --start-from {step_num}")
            break

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    if failed:
        print(f"Pipeline FAILED after {duration}")
    else:
        print(f"Pipeline COMPLETED in {duration}")
        print(f"\nNext steps:")
        print(f"  1. Run the labeling UI: streamlit run apps/labeling_ui.py")
        print(f"  2. Label some emails to build training data")
        print(f"  3. Train models (see docs/training.md)")
    print(f"{'='*60}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
