#!/usr/bin/env bash
# One-command demo: five favorite songs from different genres -> 12 album recommendations.
# Usage: ./scripts/demo.sh            (CLI)
#        ./scripts/demo.sh app        (Streamlit UI)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"

if [ "${1:-}" = "app" ]; then
  exec "$PY" -m streamlit run src/app.py
fi
exec "$PY" -m src.cli \
  "Cocteau Twins - Know Who You Are at Every Age" \
  "Joy Division - Disorder" \
  "David Bowie - Heroes" \
  "Boards of Canada - Roygbiv" \
  "Nujabes - Feather" -k 12
