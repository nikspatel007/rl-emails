#!/usr/bin/env python3
"""Filter mbox file to emails within a date range.

Usage:
    python scripts/filter_mbox_by_date.py /path/to/source.mbox /path/to/output.mbox --months 6
"""

import argparse
import mailbox
import email.utils
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Default configuration (can be overridden by env vars or CLI args)
DEFAULT_MBOX_PATH = "/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/takeout/extracted/All mail Including Spam and Trash.mbox"


def parse_email_date(date_str: str) -> datetime | None:
    """Parse email date header to datetime."""
    if not date_str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed.replace(tzinfo=None)  # Remove timezone for comparison
    except (ValueError, TypeError):
        pass
    return None


def filter_mbox(
    source_path: Path,
    output_path: Path,
    months: int = 6,
    dry_run: bool = False,
) -> dict:
    """Filter mbox to emails within the last N months.

    Args:
        source_path: Path to source mbox file
        output_path: Path to output mbox file
        months: Number of months to include
        dry_run: If True, just count without writing

    Returns:
        Stats dict with counts
    """
    cutoff_date = datetime.now() - timedelta(days=months * 30)

    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Filtering to emails after: {cutoff_date.strftime('%Y-%m-%d')}")
    print()

    stats = {
        "total": 0,
        "included": 0,
        "excluded": 0,
        "no_date": 0,
        "parse_errors": 0,
    }

    source_mbox = mailbox.mbox(str(source_path))

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_mbox = mailbox.mbox(str(output_path))
        output_mbox.lock()

    try:
        for i, message in enumerate(source_mbox):
            stats["total"] += 1

            if i % 10000 == 0:
                print(f"Processed {i:,} emails, included {stats['included']:,}...")

            date_str = message.get("Date", "")
            email_date = parse_email_date(date_str)

            if email_date is None:
                stats["no_date"] += 1
                # Include emails without dates (might be recent)
                if not dry_run:
                    output_mbox.add(message)
                stats["included"] += 1
                continue

            if email_date >= cutoff_date:
                stats["included"] += 1
                if not dry_run:
                    output_mbox.add(message)
            else:
                stats["excluded"] += 1

    except Exception as e:
        stats["parse_errors"] += 1
        print(f"Error processing message: {e}")
    finally:
        if not dry_run:
            output_mbox.unlock()
            output_mbox.close()
        source_mbox.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter mbox by date range")
    parser.add_argument(
        "source", type=Path, nargs="?",
        default=Path(os.environ.get("MBOX_PATH", DEFAULT_MBOX_PATH)),
        help="Source mbox file (default: $MBOX_PATH or built-in default)"
    )
    parser.add_argument(
        "output", type=Path, nargs="?",
        default=None,
        help="Output mbox file (default: source directory + filtered_<months>m.mbox)"
    )
    parser.add_argument(
        "--months", type=int, default=6,
        help="Include emails from last N months (default: 6)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count emails without writing output"
    )

    args = parser.parse_args()

    # Generate default output path if not provided
    if args.output is None:
        args.output = args.source.parent / f"filtered_{args.months}m.mbox"

    if not args.source.exists():
        print(f"Error: Source file not found: {args.source}")
        sys.exit(1)

    print("=" * 60)
    print("MBOX DATE FILTER")
    print("=" * 60)

    stats = filter_mbox(
        args.source,
        args.output,
        months=args.months,
        dry_run=args.dry_run,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total emails:    {stats['total']:,}")
    print(f"Included:        {stats['included']:,}")
    print(f"Excluded:        {stats['excluded']:,}")
    print(f"No date header:  {stats['no_date']:,}")
    print(f"Parse errors:    {stats['parse_errors']:,}")

    if not args.dry_run and args.output.exists():
        size_mb = args.output.stat().st_size / (1024 * 1024)
        print(f"Output size:     {size_mb:.1f} MB")

    print("=" * 60)


if __name__ == "__main__":
    main()
