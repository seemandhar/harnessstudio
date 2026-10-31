#!/usr/bin/env bash
# Install every method package under methods/ (editable) into the repo venv.
set -euo pipefail
cd "$(dirname "$0")/.."
for m in methods/*/ examples/plugin_method/; do
  [ -f "$m/pyproject.toml" ] && uv pip install -q --python .venv/bin/python -e "$m" && echo "installed $m"
done
.venv/bin/hs list
