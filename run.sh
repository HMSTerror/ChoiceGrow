#!/usr/bin/env bash
# Linux entry point: prefer explicit Python, then the project virtual environment, then system python3.
# exec preserves signals and exit status; forward all CLI arguments to run.py.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${CHOICE_PYTHON:-}" ]]; then
  python_bin="$CHOICE_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  python_bin=python3
fi
exec "$python_bin" -B -u run.py "$@"
