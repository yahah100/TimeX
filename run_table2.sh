#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
# Use the managed environment when present; uv handles normal fresh checkouts.
if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python experiments/table2_workflow.py "$@"
fi
exec uv run python experiments/table2_workflow.py "$@"
