from __future__ import annotations

"""
フォーメーション買い(engine.formation_builder.build_formation)と
現行の「最強の買い目」(engine.buy_selector.select_best_bets)を、
同じレース集合・同じprob_map/odds_mapで比較するバックテスト。

データ: data/logs/predictions_combined_for_sim.csv
  (新モデル[展開特徴量追加、2026-07-22本番投入]の真のholdout結果
  + デプロイ後の本物のログを結合したもの。scripts/build_combined_predictions_for_sim.py参照)
  + data/logs/results.csv

各レースについて:
  - best_bets: select_best_bets(ai_preds, venue=venue) をそのまま適用(現行本番ロジック)
  - formation: build_formation(prob_map, odds_map, max_points=budget) を
    budget=4,8,12 それぞれで適用(app側のbudgets=(4,8,12)と同じ)

投資額は「買い点数 x 100円」、的中時の払戻はresults.csvのpayout(100円あたり)を
実際の買い点数に応じてそのまま加算(1点100円なのでpayoutをそのまま使う)。
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "engine").exists() else Path("/sessions/practical-magical-edison/mnt/boat_ai")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402
from engine import formation_builder  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions_combined_for_sim.csv"
RESULTS_CSV = LOG_DIR / "results.csv"

UNIT_BET_YEN = 100
FORMATION_BUDGETS = (4, 8, 12)


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def load_data() -> pd.DataFrame:
    pred = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    res = pd.read_csv(RESULTS_CSV, low_memory=False)

    pred["date"] = _normalize_date(pred["date"])
    res["date"] = _normalize_date(res["date"])
    pred["venue"] = pred["venue"].astype(str).str.strip()
    res["venue"] = res["venue"].astype(str).str.strip()
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()

    merged = pred.merge(
        res[["date", "venue", "race_no", "actual_combo", "payout"]],
        on=["date", "venue", "race_no"],
        how="left",
    )
    return merged


def main() -> None:
    print("loading data...")
    df = load_data()
    n_races_total = df.drop_duplicates(["date", "venue", "race_no"]).shape[0]
    print(f"rows: {len(df)}, races (all): {n_races_total}")

    stats = {
        "best_bets": {"races": 0, "buy_points": 0, "invested": 0, "returned": 0, "hits": 0},
    }
    for budget in FORMATION_BUDGETS:
        stats[f"formation_{budget}"] = {"races": 0, "buy_points": 0, "invested": 0, "returned": 0, "hits": 0}

    n_resulted = 0
    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]
        if pd.isna(actual_combo):
            continue
        n_resulted += 1

        payout_val = int(float(payout)) if pd.notna(payout) else 0

        ai_preds = []
        prob_map = {}
        odds_map = {}
        for _, r in race_df.iterrows():
            combo = r["combo"]
            prob = float(r["prob"])
            odds = float(r["odds"])
            ai_preds.append({"combo": combo, "prob": prob, "odds": odds})
            prob_map[combo] = prob
            odds_map[combo] = odds

        # --- 最強の買い目(現行本番ロジック) ---
        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)
        if best_bets:
            combos = [b["combo"] for b in best_bets]
            invested = len(combos) * UNIT_BET_YEN
            hit = actual_combo in combos
            returned = payout_val if hit else 0
            s = stats["best_bets"]
            s["races"] += 1
            s["buy_points"] += len(combos)
            s["invested"] += invested
            s["returned"] += returned
            s["hits"] += 1 if hit else 0

        # --- フォーメーション買い(各予算) ---
        for budget in FORMATION_BUDGETS:
            f = formation_builder.build_formation(prob_map, odds_map=odds_map, max_points=budget)
            combos = f.get("combos", [])
            if not combos:
                continue
            invested = len(combos) * UNIT_BET_YEN
            hit = actual_combo in combos
            returned = payout_val if hit else 0
            s = stats[f"formation_{budget}"]
            s["races"] += 1
            s["buy_points"] += len(combos)
            s["invested"] += invested
            s["returned"] += returned
            s["hits"] += 1 if hit else 0

    print(f"races with results: {n_resulted}")
    print()
    header = f"{'method':16s} {'races':>7s} {'avg_pt':>7s} {'invested':>12s} {'returned':>12s} {'profit':>12s} {'roi%':>8s} {'hit_rate%':>10s}"
    print(header)
    print("-" * len(header))
    for name, s in stats.items():
        races = s["races"]
        if races == 0:
            print(f"{name:16s} (no bets)")
            continue
        avg_pt = s["buy_points"] / races
        profit = s["returned"] - s["invested"]
        roi = s["returned"] / s["invested"] * 100 if s["invested"] > 0 else 0.0
        hit_rate = s["hits"] / races * 100
        print(f"{name:16s} {races:7d} {avg_pt:7.2f} {s['invested']:12,d} {s['returned']:12,d} {profit:12,d} {roi:8.2f} {hit_rate:10.2f}")


if __name__ == "__main__":
    main()
