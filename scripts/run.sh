#!/bin/zsh
# Run the dashboard in the foreground, keeping the Mac awake while it runs.
#   ./scripts/run.sh            -> http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"   # tesseract
HOST="${DASHBOARD_HOST:-127.0.0.1}"  # not $HOST: zsh sets that to the hostname
PORT="${DASHBOARD_PORT:-8000}"
exec /usr/bin/caffeinate -ims ./.venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT"
