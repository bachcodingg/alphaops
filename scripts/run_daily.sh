#!/usr/bin/env bash
# Daily pipeline: ingest -> evaluate the alpha library -> health check.
#
# Designed to be run by cron or a systemd timer:
#   * never two runs at once (flock), and a second start exits cleanly;
#   * one log file per day, pruned after ALPHAOPS_LOG_RETENTION_DAYS;
#   * a degraded ingest (some symbols failed / rows quarantined) still evaluates alphas;
#   * a failed ingest stops the run and exits non-zero, so cron/systemd records the failure.
#
# Configuration via environment (or the optional .env file in the project root):
#   ALPHAOPS_SOURCE      data source (default: yahoo)
#   ALPHAOPS_UNIVERSE    universe CSV (default: config/universe_hose.csv; empty = none)
#   ALPHAOPS_ALPHAS      alpha library (default: examples/alphas.toml)
#   ALPHAOPS_LOG_DIR     log directory (default: logs/)
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SOURCE="${ALPHAOPS_SOURCE:-yahoo}"
# An empty ALPHAOPS_UNIVERSE means "let the source decide" (e.g. synthetic).
UNIVERSE="${ALPHAOPS_UNIVERSE-config/universe_hose.csv}"
ALPHAS="${ALPHAOPS_ALPHAS:-examples/alphas.toml}"
LOG_DIR="${ALPHAOPS_LOG_DIR:-$PROJECT_DIR/logs}"
RETENTION_DAYS="${ALPHAOPS_LOG_RETENTION_DAYS:-30}"
LOCK_FILE="${ALPHAOPS_LOCK_FILE:-${TMPDIR:-/tmp}/alphaops-daily.lock}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily-$(date +%F).log"
exec >>"$LOG_FILE" 2>&1

log() { printf '%s run_daily: %s\n' "$(date -u +%FT%TZ)" "$*"; }
trap 'log "ERROR: line $LINENO exited with status $?"' ERR

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "another run holds $LOCK_FILE; skipping"
  exit 0
fi

if [[ -x .venv/bin/alphaops ]]; then
  ALPHAOPS=.venv/bin/alphaops
elif command -v alphaops >/dev/null 2>&1; then
  ALPHAOPS=alphaops
else
  log "alphaops not found (create .venv or install the package)"
  exit 1
fi

log "start source=$SOURCE universe=$UNIVERSE"

ingest_args=(--source "$SOURCE")
if [[ -n "$UNIVERSE" ]]; then
  ingest_args+=(--universe "$UNIVERSE")
fi

rc=0
"$ALPHAOPS" --log-format json ingest "${ingest_args[@]}" || rc=$?
case "$rc" in
  0) log "ingest ok" ;;
  3) log "ingest degraded; continuing (see alerts / alphaops dq)" ;;
  *) log "ingest failed (exit $rc); stopping"; exit "$rc" ;;
esac

batch_rc=0
"$ALPHAOPS" --log-format json batch "$ALPHAS" || batch_rc=$?
[[ "$batch_rc" -eq 0 || "$batch_rc" -eq 3 ]] || log "batch failed (exit $batch_rc)"

health=0
"$ALPHAOPS" health || health=$?
log "health exit=$health (0 ok, 1 warning, 2 critical)"

find "$LOG_DIR" -name 'daily-*.log' -mtime +"$RETENTION_DAYS" -delete
log "done"

# Fail the unit if alpha evaluation failed outright; health warnings do not fail the run.
if [[ "$batch_rc" -ne 0 && "$batch_rc" -ne 3 ]]; then
  exit "$batch_rc"
fi
exit 0
