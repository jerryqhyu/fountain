#!/bin/zsh
# Keep the dashboard running: restart uvicorn if it exits, and keep the Mac awake meanwhile.
#   nohup ./scripts/serve_forever.sh > logs/server.log 2>&1 &
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
while true; do
  /usr/bin/caffeinate -ims ./.venv/bin/uvicorn app.main:app --host "${DASHBOARD_HOST:-127.0.0.1}" --port "${DASHBOARD_PORT:-8000}"
  echo "$(date) uvicorn exited ($?); restarting in 10 s"
  sleep 10
done
