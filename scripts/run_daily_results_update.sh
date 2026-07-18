#!/bin/bash
# scripts/run_daily_results_update.sh
#
# launchd (または手動) から呼び出すラッパー。
# venvのPythonでscripts/update_results_daily.pyを実行し、実行ログを
# data/logs/daily_update_run_logs/ 以下に残す。
#
# 手動で試すとき:
#   ./scripts/run_daily_results_update.sh --date 20260717 --venue 戸田 --skip-pipeline
#
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/data/logs/daily_update_run_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

echo "=== run_daily_results_update.sh start: $(date) ===" >> "$LOG_FILE"
"$PYTHON_BIN" scripts/update_results_daily.py "$@" >> "$LOG_FILE" 2>&1
STATUS=$?
echo "=== exit code: $STATUS ($(date)) ===" >> "$LOG_FILE"
exit $STATUS
