#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python experiments/synth_workflow.py --table 1 "$@"
fi
exec uv run python experiments/synth_workflow.py --table 1 "$@"
