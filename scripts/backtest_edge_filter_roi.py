# scripts/backtest_edge_filter_roi.py
# -*- coding: utf-8 -*-
"""
「buy_selectorが選んだ買い目のうち、真のケリー的エッジ(prob*odds>1)が無いものを
除外して買ったら回収率(ROI)はどう変わるか」を、scripts/backtest_buy_selector.pyと
同じデータ(predictions_combined_for_sim.csv + results.csv、/simと同じ土台)で
検証する。

やること(1レースずつ):
  1. engine.buy_selector.select_best_bets() で今のロジック通りに買い目を選ぶ
     (= 現状の「固定100円ベット」と全く同じ選定)
  2. そのうちprob*odds<=1(エッジ無し)の点を除外した「エッジ絞り込み版」を作る
  3. 両方とも100円均等ベットとして、ROI/的中率/レース当たり平均点数/
     買い目0になったレース数を比較する

使い方:
  python scripts/backtest_edge_filter_roi.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402
from scripts.backtest_buy_selector import load_data  # noqa: E402

UNIT_BET_YEN = 100


def run(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    baseline_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]
        if pd.isna(actual_combo):
            continue

        ai_preds = []
        for _, r in race_df.iterrows():
            ai_preds.append({"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])})

        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)

        # --- baseline: 今の固定100円ベット(全選定点) ---
        buy_combos_all = [b["combo"] for b in best_bets]
        n_all = len(buy_combos_all)
        invested_all = n_all * UNIT_BET_YEN
        hit_all = 1 if actual_combo in buy_combos_all else 0
        returned_all = int(float(payout)) if hit_all and pd.notna(payout) else 0
        baseline_rows.append({
            "date": date, "venue": venue, "race_no": race_no,
            "buy_count": n_all, "invested": invested_all,
            "returned": returned_all, "profit": returned_all - invested_all, "hit": hit_all,
        })

        # --- edge filter: prob*odds>1 の点だけ残す ---
        buy_combos_edge = [b["combo"] for b in best_bets if float(b.get("prob", 0.0)) * float(b.get("odds", 0.0)) > 1.0]
        n_edge = len(buy_combos_edge)
        invested_edge = n_edge * UNIT_BET_YEN
        hit_edge = 1 if actual_combo in buy_combos_edge else 0
        returned_edge = int(float(payout)) if hit_edge and pd.notna(payout) else 0
        edge_rows.append({
            "date": date, "venue": venue, "race_no": race_no,
            "buy_count": n_edge, "invested": invested_edge,
            "returned": returned_edge, "profit": returned_edge - invested_edge, "hit": hit_edge,
        })

    return {
        "baseline": pd.DataFrame(baseline_rows),
        "edge": pd.DataFrame(edge_rows),
    }


def summarize(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    if df.empty:
        return {"label": label}
    total_invested = df["invested"].sum()
    total_returned = df["returned"].sum()
    roi = total_returned / total_invested if total_invested > 0 else 0.0
    zero_bet_races = int((df["buy_count"] == 0).sum())
    return {
        "label": label,
        "races": len(df),
        "races_with_bet": int((df["buy_count"] > 0).sum()),
        "zero_bet_races": zero_bet_races,
        "hit_rate": df.loc[df["buy_count"] > 0, "hit"].mean() if (df["buy_count"] > 0).any() else 0.0,
        "avg_points": df.loc[df["buy_count"] > 0, "buy_count"].mean() if (df["buy_count"] > 0).any() else 0.0,
        "invested": int(total_invested),
        "returned": int(total_returned),
        "profit": int(total_returned - total_invested),
        "roi": roi,
    }


def print_summary(s: Dict[str, Any]) -> None:
    print(f"--- {s.get('label')} ---")
    print(f"races               : {s.get('races')}")
    print(f"races_with_bet      : {s.get('races_with_bet')}  (buy_count=0のレース: {s.get('zero_bet_races')})")
    print(f"hit_rate(bet有りのみ): {s.get('hit_rate', 0)*100:.2f}%")
    print(f"avg_points(bet有りのみ): {s.get('avg_points', 0):.2f}")
    print(f"invested            : {s.get('invested'):,}円")
    print(f"returned            : {s.get('returned'):,}円")
    print(f"profit              : {s.get('profit'):,}円")
    print(f"roi                 : {s.get('roi', 0)*100:.2f}%")
    print()


def main() -> None:
    combined_predictions = PROJECT_ROOT / "data" / "logs" / "predictions_combined_for_sim.csv"
    print("loading:", combined_predictions)
    df = load_data(predictions_path=combined_predictions)
    print("rows:", len(df), "races:", df.drop_duplicates(["date", "venue", "race_no"]).shape[0])
    print()

    results = run(df)

    baseline_summary = summarize(results["baseline"], "baseline(今の固定100円ベット、全選定点)")
    edge_summary = summarize(results["edge"], "edge filter(prob*odds>1の点だけ)")

    print("=" * 80)
    print_summary(baseline_summary)
    print_summary(edge_summary)

    print("=" * 80)
    print("全体比較")
    print(f"{'':22s} {'baseline':>14s} {'edge filter':>14s} {'差分':>10s}")
    print(f"{'races_with_bet':22s} {baseline_summary['races_with_bet']:14d} {edge_summary['races_with_bet']:14d} {edge_summary['races_with_bet']-baseline_summary['races_with_bet']:10d}")
    print(f"{'hit_rate':22s} {baseline_summary['hit_rate']*100:13.2f}% {edge_summary['hit_rate']*100:13.2f}% {(edge_summary['hit_rate']-baseline_summary['hit_rate'])*100:9.2f}%")
    print(f"{'roi':22s} {baseline_summary['roi']*100:13.2f}% {edge_summary['roi']*100:13.2f}% {(edge_summary['roi']-baseline_summary['roi'])*100:9.2f}%")
    print(f"{'profit':22s} {baseline_summary['profit']:14,d} {edge_summary['profit']:14,d} {edge_summary['profit']-baseline_summary['profit']:10,d}")

    # 会場別
    print()
    print("会場別比較:")
    b_by_v = results["baseline"].groupby("venue").apply(lambda g: pd.Series(summarize(g, ""))).reset_index()
    e_by_v = results["edge"].groupby("venue").apply(lambda g: pd.Series(summarize(g, ""))).reset_index()
    merged = b_by_v[["venue", "races", "roi", "profit"]].merge(
        e_by_v[["venue", "races", "roi", "profit"]], on="venue", suffixes=("_baseline", "_edge")
    )
    merged["roi_diff"] = merged["roi_edge"] - merged["roi_baseline"]
    merged = merged.sort_values("roi_diff")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
