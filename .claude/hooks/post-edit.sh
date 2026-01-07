#!/bin/bash
# Post-edit hook: Quick lint/format after file edits
# Runs after Edit or Write tool calls

cd "$CLAUDE_PROJECT_DIR" || exit 0

# Read tool input from stdin
input=$(cat)

# Extract file path from JSON input
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

if [ -z "$file_path" ]; then
    exit 0
fi

# Only check Python files in src/ or tests/
if [[ "$file_path" =~ ^.*(src|tests)/.*\.py$ ]]; then
    # Auto-fix with ruff (non-blocking)
    uv run ruff check "$file_path" --fix --quiet 2>/dev/null || true
    uv run ruff format "$file_path" --quiet 2>/dev/null || true
fi

exit 0
