# scripts/update_results_daily.py
# -*- coding: utf-8 -*-
"""
data/logs/results.csv を boatrace.jp から自動取得して更新する日次バッチ。

ユーザーのMac上でlaunchd等から毎日1回(レース終了後、例: 23:00頃)実行する想定。
このCoworkサンドボックス環境からはboatrace.jpに到達できないため、ここでは実行できない
(ユーザーの手元環境で実行・動作確認する必要がある)。

処理内容:
  1. 対象日(デフォルト: 前日)について、全24会場×最大12Rの結果を
     engine.raceresult_fetcher 経由で取得し、まだresults.csvに無い分だけ追記する。
  2. --skip-pipeline が無ければ、続けて
     merge_prediction_results.py → build_sim_light_csv.py → build_kelly_bankroll_daily.py
     を自動実行し、/sim ダッシュボードまで一気通貫で最新化する。

使い方の例:
  # 前日分だけ取得して/simまで更新
  python scripts/update_results_daily.py

  # 直近3日分をまとめて後追い(取りこぼし対策)
  python scripts/update_results_daily.py --days-back 3

  # 特定の日・会場だけ試す(動作確認用)
  python scripts/update_results_daily.py --date 20260717 --venue 戸田 --skip-pipeline

  # パース失敗時に生HTMLを保存してデバッグしたい場合
  python scripts/update_results_daily.py --date 20260717 --venue 戸田 --debug
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.raceresult_fetcher import fetch_race_result  # noqa: E402
from engine.venue_registry import VENUE_ORDER, VENUE_NAME_TO_JCD  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
RESULTS_CSV = LOG_DIR / "results.csv"

RESULT_FIELDS = [
    "logged_at",
    "date",
    "venue",
    "race_no",
    "actual_combo",
    "payout",
    "source",
    "combo_3puku",
    "payout_3puku",
    "combo_2tan",
    "payout_2tan",
    "combo_2puku",
    "payout_2puku",
]


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _load_existing_keys() -> set:
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV, dtype=str, low_memory=False)
    if df.empty:
        return set()
    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    race_no = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)
    return set(zip(df["date"], df["venue"], race_no))


def fetch_date(
    target: str,
    venues: Optional[List[str]] = None,
    sleep_sec: float = 0.6,
    debug: bool = False,
) -> List[dict]:
    existing = _load_existing_keys()
    venues = venues or VENUE_ORDER
    rows: List[dict] = []

    for venue in venues:
        jcd = VENUE_NAME_TO_JCD[venue]
        miss_streak = 0

        for rno in range(1, 13):
            key = (target, venue, rno)
            if key in existing:
                continue

            result = fetch_race_result(
                rno, target, jcd, save_debug=debug, sleep_sec=sleep_sec
            )

            if not result or not result.get("actual_combo"):
                miss_streak += 1
                # 序盤のレースが連続で取れない = その会場はこの日開催が無かった
                # 可能性が高いので、その会場は打ち切って次に進む
                if miss_streak >= 2 and rno <= 3:
                    break
                continue

            miss_streak = 0
            result["logged_at"] = datetime.now().isoformat(timespec="seconds")
            result["venue"] = venue
            rows.append(result)
            print(f"[OK] {target} {venue} {rno}R -> {result.get('actual_combo')} {result.get('payout')}円")

    return rows


def append_rows(rows: List[dict]) -> None:
    if not rows:
        print("no new rows to append")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame(rows)
    for col in RESULT_FIELDS:
        if col not in df_new.columns:
            df_new[col] = ""
    df_new = df_new[RESULT_FIELDS]

    if RESULTS_CSV.exists():
        df_old = pd.read_csv(RESULTS_CSV, low_memory=False)
        combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        combined = df_new

    combined["date"] = combined["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    combined["race_no"] = pd.to_numeric(combined["race_no"], errors="coerce").fillna(0).astype(int)
    combined["payout"] = pd.to_numeric(combined["payout"], errors="coerce").fillna(0.0)

    combined = combined.sort_values(
        by=["date", "venue", "race_no", "payout"],
        ascending=[True, True, True, False],
    ).drop_duplicates(subset=["date", "venue", "race_no"], keep="first")
    combined = combined.sort_values(["date", "venue", "race_no"]).reset_index(drop=True)

    combined.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"appended {len(df_new)} new rows -> results.csv now has {len(combined)} rows total")


def run_downstream_pipeline() -> None:
    steps = [
        [sys.executable, str(PROJECT_ROOT / "scripts" / "merge_prediction_results.py")],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_sim_light_csv.py")],
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_kelly_bankroll_daily.py")],
    ]
    for cmd in steps:
        print("=" * 80)
        print("running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        except subprocess.CalledProcessError as e:
            print(f"[WARN] step failed, skipping remaining downstream steps: {e}")
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch daily race results from boatrace.jp and update results.csv"
    )
    parser.add_argument("--date", help="YYYYMMDD (省略時は前日)")
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="今日から何日前まで遡って埋めるか(取りこぼし後追い用、デフォルト1=前日のみ)",
    )
    parser.add_argument(
        "--venue",
        action="append",
        help="対象会場を絞り込む(複数指定可、例: --venue 戸田 --venue 桐生)。省略時は全24会場",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="merge_prediction_results / build_sim_light_csv / build_kelly_bankroll_daily の再実行をスキップ",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="パース失敗時に生HTMLをdata/logs/raceresult_debug/に保存する",
    )
    args = parser.parse_args()

    if args.date:
        targets = [args.date]
    else:
        today = date.today()
        targets = [_yyyymmdd(today - timedelta(days=i)) for i in range(1, args.days_back + 1)]
        targets.reverse()

    all_rows: List[dict] = []
    for target in targets:
        print("=" * 80)
        print("fetching date:", target)
        rows = fetch_date(target, venues=args.venue, debug=args.debug)
        all_rows.extend(rows)

    append_rows(all_rows)

    if not args.skip_pipeline and all_rows:
        run_downstream_pipeline()


if __name__ == "__main__":
    main()
