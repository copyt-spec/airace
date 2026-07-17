from __future__ import annotations

"""
scripts/backtest_kelly_sizing.py が出力する レース単位の資産推移
(data/logs/kelly_sizing_backtest.csv) を日次に集約し、
/sim ダッシュボードのKelly資産推移チャート用の軽量csvを作る。

出力: data/logs/kelly_bankroll_daily.csv
  columns: date, flat_bankroll, kelly_bankroll, flat_invested, flat_returned,
           kelly_invested, kelly_returned
  (flat/kelly_bankroll はその日の最終値、invested/returnedはその日の合計)
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"

RACE_LEVEL_CSV = LOG_DIR / "kelly_sizing_backtest.csv"
DAILY_CSV = LOG_DIR / "kelly_bankroll_daily.csv"


def main() -> None:
    if not RACE_LEVEL_CSV.exists():
        raise FileNotFoundError(f"Missing: {RACE_LEVEL_CSV}")

    df = pd.read_csv(RACE_LEVEL_CSV, low_memory=False)
    if df.empty:
        raise RuntimeError("kelly_sizing_backtest.csv is empty")

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)

    # 元CSVは日付順のレース単位で並んでいる前提(backtest_kelly_sizing.py側で保証)
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            flat_invested=("flat_invested", "sum"),
            flat_returned=("flat_returned", "sum"),
            kelly_invested=("kelly_invested", "sum"),
            kelly_returned=("kelly_returned", "sum"),
            flat_bankroll=("flat_bankroll", "last"),
            kelly_bankroll=("kelly_bankroll", "last"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")

    print("saved:", DAILY_CSV)
    print("rows :", len(daily))
    print("date range:", daily["date"].min(), "-", daily["date"].max())
    print("final flat_bankroll :", daily["flat_bankroll"].iloc[-1])
    print("final kelly_bankroll:", daily["kelly_bankroll"].iloc[-1])


if __name__ == "__main__":
    main()
