from __future__ import annotations

"""
predictions.csv (全組み合わせのprob/odds付きログ) と 修正済み results.csv を突き合わせ、
engine.buy_selector.select_best_bets を「今のコード」でオフライン再現し、
実際の運用ログ(is_selected)に依存しない、公正なROIバックテストを行う。

過去ログの is_selected は当時のコードやオッズ取得タイミングに依存しているため、
「今のロジックを過去データに適用したら何点買って、いくら勝っていたか」を
再計算することで、ロジック変更の効果を検証できる。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
RESULTS_CSV = LOG_DIR / "results.csv"

UNIT_BET_YEN = 100


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def load_data(start_date: str = "", end_date: str = "") -> pd.DataFrame:
    pred = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    res = pd.read_csv(RESULTS_CSV, low_memory=False)

    pred["date"] = _normalize_date(pred["date"])
    res["date"] = _normalize_date(res["date"])
    pred["venue"] = pred["venue"].astype(str).str.strip()
    res["venue"] = res["venue"].astype(str).str.strip()
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()

    if start_date:
        pred = pred[pred["date"] >= start_date]
    if end_date:
        pred = pred[pred["date"] <= end_date]

    merged = pred.merge(
        res[["date", "venue", "race_no", "actual_combo", "payout"]],
        on=["date", "venue", "race_no"],
        how="left",
    )
    return merged


def run_backtest(
    df: pd.DataFrame,
    selector_fn=None,
    selector_kwargs: Optional[Dict[str, Any]] = None,
    calibrate_fn=None,
) -> pd.DataFrame:
    selector_fn = selector_fn or buy_selector.select_best_bets
    selector_kwargs = selector_kwargs or {}

    results: List[Dict[str, Any]] = []

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]

        if pd.isna(actual_combo):
            continue

        ai_preds = []
        for _, r in race_df.iterrows():
            prob = float(r["prob"])
            odds = float(r["odds"])
            if calibrate_fn is not None:
                prob = calibrate_fn(prob, odds)
            ai_preds.append({"combo": r["combo"], "prob": prob, "odds": odds})

        best_bets = selector_fn(ai_preds, venue=venue, **selector_kwargs)
        buy_combos = [b["combo"] for b in best_bets]
        buy_count = len(buy_combos)
        invested = buy_count * UNIT_BET_YEN

        hit = 1 if actual_combo in buy_combos else 0
        returned = int(float(payout)) if hit and pd.notna(payout) else 0

        results.append({
            "date": date,
            "venue": venue,
            "race_no": race_no,
            "buy_count": buy_count,
            "invested": invested,
            "returned": returned,
            "profit": returned - invested,
            "hit": hit,
        })

    return pd.DataFrame(results)


def summarize(result_df: pd.DataFrame, label: str = "") -> Dict[str, Any]:
    if result_df.empty:
        return {}

    total_invested = result_df["invested"].sum()
    total_returned = result_df["returned"].sum()
    roi = total_returned / total_invested if total_invested > 0 else 0.0

    summary = {
        "label": label,
        "races": len(result_df),
        "hit_rate": result_df["hit"].mean(),
        "avg_points": result_df["buy_count"].mean(),
        "invested": total_invested,
        "returned": total_returned,
        "profit": total_returned - total_invested,
        "roi": roi,
    }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"--- {summary.get('label','')} ---")
    print(f"races      : {summary.get('races')}")
    print(f"hit_rate   : {summary.get('hit_rate'):.4f}")
    print(f"avg_points : {summary.get('avg_points'):.2f}")
    print(f"invested   : {summary.get('invested'):.0f}")
    print(f"returned   : {summary.get('returned'):.0f}")
    print(f"profit     : {summary.get('profit'):.0f}")
    print(f"roi        : {summary.get('roi')*100:.2f}%")
    print()


if __name__ == "__main__":
    print("loading data...")
    df = load_data()
    print("rows:", len(df), "races:", df.drop_duplicates(["date", "venue", "race_no"]).shape[0])

    print("running backtest with CURRENT buy_selector...")
    res = run_backtest(df)
    print_summary(summarize(res, "current buy_selector.py"))

    print("\nby venue:")
    by_venue = res.groupby("venue").apply(
        lambda g: pd.Series(summarize(g))
    ).reset_index()
    print(by_venue[["venue", "races", "hit_rate", "avg_points", "roi", "profit"]].sort_values("profit").to_string(index=False))
