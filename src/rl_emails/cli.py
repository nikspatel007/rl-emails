"""CLI entry point for rl-emails pipeline.

This module provides a CLI entry point that wraps scripts/onboard_data.py,
allowing installation via `pip install -e .` and execution via `rl-emails`.

Usage:
    rl-emails                    # Run full pipeline
    rl-emails --status           # Check pipeline status
    rl-emails --help             # Show all options
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """CLI entry point for the email onboarding pipeline.

    Delegates to scripts/onboard_data.py with all arguments passed through.
    """
    # Find the project root (where pyproject.toml is)
    cli_module = Path(__file__).resolve()
    src_dir = cli_module.parent.parent  # src/
    project_root = src_dir.parent  # project root

    script_path = project_root / "scripts" / "onboard_data.py"

    if not script_path.exists():
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        print("Ensure you're running from an installed rl-emails package.", file=sys.stderr)
        sys.exit(1)

    # Run the script with all arguments passed through
    result = subprocess.run(
        [sys.executable, str(script_path), *sys.argv[1:]],
        cwd=project_root,
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
