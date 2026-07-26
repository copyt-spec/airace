#!/bin/bash
# scripts/run_daily_k_fetch.sh
#
# launchd (または手動) から呼び出すラッパー。
# venvのPythonでscripts/fetch_and_unpack_k_files_daily.pyを実行し、実行ログを
# data/logs/daily_k_fetch_run_logs/ 以下に残す。
#
# これはdata/raw_txt (Kファイル)を最新化するだけの処理。選手成績特徴量
# (racer_point_in_time_stats.csv)の再生成はCowork側のboat-ai-racer-stats-refresh
# スケジュールタスク(毎日06:09 JST)が別途行うので、このスクリプトはそれより
# 前の時間に実行されるようlaunchdで設定すること。
#
# 手動で試すとき:
#   ./scripts/run_daily_k_fetch.sh --days-back 14   (久しぶりに動かす/取りこぼし解消)
#   ./scripts/run_daily_k_fetch.sh                  (通常運用、直近5日分だけ確認)
#
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/data/logs/daily_k_fetch_run_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

echo "=== run_daily_k_fetch.sh start: $(date) ===" >> "$LOG_FILE"
"$PYTHON_BIN" -m scripts.fetch_and_unpack_k_files_daily "$@" >> "$LOG_FILE" 2>&1
STATUS=$?
echo "=== exit code: $STATUS ($(date)) ===" >> "$LOG_FILE"
exit $STATUS
