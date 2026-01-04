#!/usr/bin/env python3
"""Create train/val/test splits for Gmail dataset.

For single-user Gmail exports, splits data chronologically instead of by user.
This ensures no temporal leakage (model can't see future emails during training).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_date(date_str: str) -> datetime:
    """Parse email date string to datetime for sorting."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S",
        "%a %b %d %H:%M:%S %Y",
    ]

    # Clean timezone abbreviations
    cleaned = date_str
    for tz in ['(PDT)', '(PST)', '(EDT)', '(EST)', '(CDT)', '(CST)', '(MDT)', '(MST)', '(UTC)']:
        cleaned = cleaned.replace(f' {tz}', '')

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned.strip(), fmt)
            # Make all datetimes timezone-aware for consistent comparison
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # Fallback: return epoch if parsing fails
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def create_splits(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """Create chronological train/val/test splits.

    Args:
        input_path: Path to labeled emails JSON file
        output_dir: Directory for output files
        train_ratio: Fraction of emails for training (oldest)
        val_ratio: Fraction of emails for validation
        test_ratio: Fraction of emails for testing (newest)

    Returns:
        Statistics dictionary
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path) as f:
        emails = json.load(f)
    print(f"Loaded {len(emails)} emails")

    # Sort by date (oldest first)
    print("Sorting by date...")
    emails.sort(key=lambda e: parse_date(e.get("date", "")))

    # Split chronologically
    n = len(emails)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_emails = emails[:n_train]
    val_emails = emails[n_train:n_train + n_val]
    test_emails = emails[n_train + n_val:]

    print(f"Split: {len(train_emails)} train / {len(val_emails)} val / {len(test_emails)} test")

    # Show date ranges
    if train_emails:
        train_start = train_emails[0].get('date', 'N/A')[:20]
        train_end = train_emails[-1].get('date', 'N/A')[:20]
        print(f"Train: {train_start} to {train_end}")
    if val_emails:
        val_start = val_emails[0].get('date', 'N/A')[:20]
        val_end = val_emails[-1].get('date', 'N/A')[:20]
        print(f"Val:   {val_start} to {val_end}")
    if test_emails:
        test_start = test_emails[0].get('date', 'N/A')[:20]
        test_end = test_emails[-1].get('date', 'N/A')[:20]
        print(f"Test:  {test_start} to {test_end}")

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_emails), ("val", val_emails), ("test", test_emails)]:
        output_path = output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {output_path}")

    # Class distribution per split
    from collections import Counter
    print("\nClass distribution per split:")
    for name, data in [("train", train_emails), ("val", val_emails), ("test", test_emails)]:
        counts = Counter(e['action'] for e in data)
        print(f"  {name}:")
        for action, count in sorted(counts.items()):
            pct = 100 * count / len(data) if data else 0
            print(f"    {action}: {count} ({pct:.1f}%)")

    # Write split metadata
    meta = {
        "split_method": "chronological",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_emails": {
            "train": len(train_emails),
            "val": len(val_emails),
            "test": len(test_emails)
        },
    }
    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Create chronological train/val/test splits for Gmail data")
    parser.add_argument("input", type=Path, help="Path to labeled emails JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/gmail_splits"),
        help="Output directory (default: data/gmail_splits)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    create_splits(
        args.input,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    return 0


if __name__ == "__main__":
    exit(main())
