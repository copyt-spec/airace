from __future__ import annotations

"""
3連単だけを買う現行戦略と、3連複(モデルの3連単確率を集約し、3連単オッズから
概算した3連複オッズを使う)を買う戦略を、同じ「レースあたりの点数」で
公平に比較するバックテスト。

前提・注意:
- 3連複の「オッズ」は実測データが無いため、engine.bet_type_converter の
  近似式(3連単オッズの調和平均的な変換)を使っている。実際の3連複プールとは
  ズレる可能性があるため、この結果はあくまで方向性の参考値。
  本番導入前に実オッズ(engine.odds_fetcher.fetch_odds_sanrenpuku)で
  数週間分ログを取り、実データで再検証すること。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402
from engine.bet_type_converter import (  # noqa: E402
    sanrenpuku_probs,
    approx_box_odds_from_3tan_odds,
)

LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
RESULTS_CSV = LOG_DIR / "results.csv"

UNIT_BET_YEN = 100


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

    keep_res_cols = ["date", "venue", "race_no", "actual_combo", "payout"]
    for c in ["combo_3puku", "payout_3puku"]:
        if c in res.columns:
            keep_res_cols.append(c)

    merged = pred.merge(res[keep_res_cols], on=["date", "venue", "race_no"], how="left")
    return merged


def select_box_bets_topn(prob_map: Dict[str, float], odds_map: Dict[str, float], n: int) -> List[str]:
    """3連複候補をEV(prob*odds)降順でn点選ぶ、シンプルな比較用ロジック。"""
    candidates = []
    for combo, prob in prob_map.items():
        odds = odds_map.get(combo, 0.0)
        if odds <= 0 or prob <= 0:
            continue
        ev = prob * odds
        candidates.append((combo, prob, odds, ev))

    candidates.sort(key=lambda x: x[3], reverse=True)
    return [c[0] for c in candidates[:n]]


def run(df: pd.DataFrame) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]

        if pd.isna(actual_combo):
            continue

        actual_combo_3puku = race_df["combo_3puku"].iloc[0] if "combo_3puku" in race_df.columns else None
        payout_3puku = race_df["payout_3puku"].iloc[0] if "payout_3puku" in race_df.columns else None

        ai_preds = [
            {"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])}
            for _, r in race_df.iterrows()
        ]

        # --- 現行: 3連単戦略(実オッズ・現行buy_selector) ---
        best_bets_3tan = buy_selector.select_best_bets(ai_preds, venue=venue)
        n_points = len(best_bets_3tan)
        buy_3tan = [b["combo"] for b in best_bets_3tan]

        invested_3tan = n_points * UNIT_BET_YEN
        hit_3tan = 1 if actual_combo in buy_3tan else 0
        returned_3tan = int(float(payout)) if hit_3tan and pd.notna(payout) else 0

        # --- 比較: 3連複戦略(同じ点数、確率は正確に集約、オッズは近似) ---
        prob_map_3tan = {r["combo"]: float(r["prob"]) for _, r in race_df.iterrows()}
        odds_map_3tan = {r["combo"]: float(r["odds"]) for _, r in race_df.iterrows()}

        if n_points > 0:
            box_probs = sanrenpuku_probs(prob_map_3tan)
            box_odds_approx = approx_box_odds_from_3tan_odds(odds_map_3tan)
            buy_3puku = select_box_bets_topn(box_probs, box_odds_approx, n_points)
        else:
            buy_3puku = []

        invested_3puku = len(buy_3puku) * UNIT_BET_YEN
        hit_3puku = 0
        returned_3puku = 0
        if actual_combo_3puku is not None and pd.notna(actual_combo_3puku):
            hit_3puku = 1 if str(actual_combo_3puku) in buy_3puku else 0
            if hit_3puku and pd.notna(payout_3puku):
                returned_3puku = int(float(payout_3puku))

        results.append({
            "date": date, "venue": venue, "race_no": race_no,
            "n_points": n_points,
            "invested_3tan": invested_3tan, "returned_3tan": returned_3tan, "hit_3tan": hit_3tan,
            "invested_3puku": invested_3puku, "returned_3puku": returned_3puku, "hit_3puku": hit_3puku,
            "has_3puku_truth": int(pd.notna(actual_combo_3puku)) if actual_combo_3puku is not None else 0,
        })

    return pd.DataFrame(results)


def summarize(res: pd.DataFrame) -> None:
    valid = res[res["has_3puku_truth"] == 1].copy()
    print(f"races (with 3puku ground truth): {len(valid)} / {len(res)}")

    for label, inv_col, ret_col, hit_col in [
        ("3連単(現行)", "invested_3tan", "returned_3tan", "hit_3tan"),
        ("3連複(近似オッズでEV選択)", "invested_3puku", "returned_3puku", "hit_3puku"),
    ]:
        inv = valid[inv_col].sum()
        ret = valid[ret_col].sum()
        roi = ret / inv if inv > 0 else 0.0
        hit_rate = valid[hit_col].mean()
        print(f"--- {label} ---")
        print(f"  invested={inv:.0f} returned={ret:.0f} profit={ret-inv:.0f} roi={roi*100:.2f}% hit_rate={hit_rate*100:.2f}%")


if __name__ == "__main__":
    print("loading data...")
    df = load_data()
    print("running backtest...")
    res = run(df)
    summarize(res)
    res.to_csv(PROJECT_ROOT / "data" / "logs" / "analysis" / "bet_type_backtest.csv", index=False, encoding="utf-8-sig")
