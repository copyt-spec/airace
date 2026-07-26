#!/bin/bash
# scripts/run_notify_rank_a.sh
#
# launchd(または手動)から呼び出すラッパー。
# venvのPythonでscripts/notify_rank_a_races.pyを実行し、実行ログを
# data/logs/notify_rank_a_run_logs/ 以下に残す。
#
# 手動で試すとき:
#   ./scripts/run_notify_rank_a.sh --dry-run   (LINE送信せず判定結果だけ確認)
#   ./scripts/run_notify_rank_a.sh             (通常運用)
#
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/data/logs/notify_rank_a_run_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

echo "=== run_notify_rank_a.sh start: $(date) ===" >> "$LOG_FILE"
"$PYTHON_BIN" scripts/notify_rank_a_races.py "$@" >> "$LOG_FILE" 2>&1
STATUS=$?
echo "=== exit code: $STATUS ($(date)) ===" >> "$LOG_FILE"
exit $STATUS
