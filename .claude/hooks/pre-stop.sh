#!/bin/bash
# Pre-stop hook: Run full quality gates before Claude finishes
# Exit code 2 blocks and shows message to Claude

set -e
cd "$CLAUDE_PROJECT_DIR" || exit 0

echo "Running quality gates..."

# Check if there are any staged or modified Python files in src/tests
changed_files=$(git diff --name-only HEAD 2>/dev/null | grep -E '^(src|tests)/.*\.py$' || true)
staged_files=$(git diff --cached --name-only 2>/dev/null | grep -E '^(src|tests)/.*\.py$' || true)

# Only run full checks if src/ or tests/ were modified
if [ -z "$changed_files" ] && [ -z "$staged_files" ]; then
    echo "No src/ or tests/ changes detected, skipping quality gates"
    exit 0
fi

# Type checking (strict)
echo "Running type checks..."
if ! uv run mypy src/ --strict --no-error-summary 2>/dev/null; then
    echo "❌ Type check failed - fix errors before continuing" >&2
    exit 2
fi

# Unit tests
echo "Running unit tests..."
if ! uv run pytest tests/unit/ -q --tb=line 2>/dev/null; then
    echo "❌ Unit tests failed - fix tests before continuing" >&2
    exit 2
fi

# Coverage check
echo "Checking coverage..."
coverage_output=$(uv run pytest tests/unit/ -q --cov=src/rl_emails --cov-report=term --cov-fail-under=100 2>&1)
if [ $? -ne 0 ]; then
    echo "❌ Coverage below 100% - add tests before continuing" >&2
    echo "$coverage_output" | tail -5 >&2
    exit 2
fi

echo "✅ All quality gates passed"
exit 0
