from __future__ import annotations

"""
buy_selector.select_best_bets()の下限(min_points=3にクランプされている)を
緩めたらROIは変わるか?を検証する。

scripts/backtest_point_cap_sweep.pyのrank別集計から、cum_roi(1..K)は
K=2でピーク(119.26%)、K=3で113.92%に下落、以降ゆるやかに下がって
K=12(現行実質上限)で107.56%になることが分かった。つまり「毎レース
最低3点」という下限自体が、buy_scoreの低いrank3の点を無理に買わせて
ROIを薄めている可能性がある。

ここでは、固定K(=1,2,3,5,現行相当の可変)を全24会場に適用したときの
会場別ROI・投資額・利益を集計し、K=2 vs 現行の可変ポイント方式(select_best_bets
そのまま)の会場別利益差に95%信頼区間を付けて、外れ値会場に依存していないか
確認する。
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "engine").exists() else Path("/sessions/practical-magical-edison/mnt/boat_ai")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions_combined_for_sim.csv"
RESULTS_CSV = LOG_DIR / "results.csv"

UNIT_BET_YEN = 100
FIXED_KS = (1, 2, 3, 5)


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


def ranked_candidates(ai_preds, venue: str):
    if buy_selector.should_skip_race(ai_preds, venue=venue):
        return []

    cfg = buy_selector._resolve_config(venue=venue)
    rows = buy_selector._prepare_candidate_rows(ai_preds, cfg, use_min_prob=True, use_min_ev=True)

    if len(rows) < 3:
        relaxed_cfg = dict(cfg)
        relaxed_cfg["min_ev"] = min(float(cfg.get("min_ev", 0.85)), 0.70)
        rows = buy_selector._prepare_candidate_rows(ai_preds, relaxed_cfg, use_min_prob=True, use_min_ev=True)

    if len(rows) < 3:
        rows = buy_selector._prepare_candidate_rows(ai_preds, cfg, use_min_prob=False, use_min_ev=False)

    if not rows:
        return []

    prob_norms = buy_selector._minmax_norm([r["prob"] for r in rows])
    odds_norms = buy_selector._minmax_norm([r["odds_capped"] for r in rows])
    ev_norms = buy_selector._minmax_norm([r["ev_capped"] for r in rows])

    for i, r in enumerate(rows):
        r["prob_norm"] = prob_norms[i]
        r["odds_norm"] = odds_norms[i]
        r["ev_norm"] = ev_norms[i]
        r["buy_score"] = (
            (r["prob_norm"] * cfg["weight_prob"]) +
            (r["ev_norm"] * cfg["weight_ev"]) +
            (r["odds_norm"] * cfg["weight_odds"]) +
            buy_selector._odds_band_score_adj(r["odds_raw"])
        )

    rows.sort(
        key=lambda x: (float(x.get("buy_score", 0.0)), float(x.get("prob", 0.0)), float(x.get("ev_raw", 0.0))),
        reverse=True,
    )
    return rows


def main() -> None:
    print("loading data...")
    df = load_data()
    print(f"rows: {len(df)}, races: {df.drop_duplicates(['date','venue','race_no']).shape[0]}")

    # venue -> method -> {invested, returned}
    venue_stats = {}

    n_resulted = 0
    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]
        if pd.isna(actual_combo):
            continue
        n_resulted += 1
        payout_val = int(float(payout)) if pd.notna(payout) else 0

        ai_preds = [{"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])} for _, r in race_df.iterrows()]

        v = venue_stats.setdefault(venue, {})

        # --- 現行の可変ポイント方式(select_best_betsそのまま) ---
        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)
        if best_bets:
            combos = [b["combo"] for b in best_bets]
            invested = len(combos) * UNIT_BET_YEN
            returned = payout_val if actual_combo in combos else 0
            m = v.setdefault("current_dynamic", {"invested": 0, "returned": 0, "races": 0})
            m["invested"] += invested
            m["returned"] += returned
            m["races"] += 1

        # --- 固定K点(buy_score上位K点のみ) ---
        ranked = ranked_candidates(ai_preds, venue=venue)
        for k in FIXED_KS:
            if len(ranked) < k:
                continue
            combos_k = [r["combo"] for r in ranked[:k]]
            invested = k * UNIT_BET_YEN
            returned = payout_val if actual_combo in combos_k else 0
            m = v.setdefault(f"fixed_{k}", {"invested": 0, "returned": 0, "races": 0})
            m["invested"] += invested
            m["returned"] += returned
            m["races"] += 1

    print(f"races with results: {n_resulted}")
    print()

    methods = ["current_dynamic"] + [f"fixed_{k}" for k in FIXED_KS]

    # 全体集計
    print("=== 全体(全会場合算) ===")
    header = f"{'method':16s} {'races':>7s} {'invested':>12s} {'returned':>12s} {'profit':>12s} {'roi%':>8s}"
    print(header)
    print("-" * len(header))
    overall = {m: {"invested": 0, "returned": 0, "races": 0} for m in methods}
    for venue, vs in venue_stats.items():
        for m in methods:
            if m in vs:
                overall[m]["invested"] += vs[m]["invested"]
                overall[m]["returned"] += vs[m]["returned"]
                overall[m]["races"] += vs[m]["races"]
    for m in methods:
        s = overall[m]
        roi = s["returned"] / s["invested"] * 100 if s["invested"] > 0 else 0.0
        print(f"{m:16s} {s['races']:7d} {s['invested']:12,d} {s['returned']:12,d} {s['returned']-s['invested']:12,d} {roi:8.2f}")

    # 会場別 profit_delta (fixed_2 - current_dynamic) の95%信頼区間(正規近似)
    import math
    print()
    print("=== 会場別 profit_delta (fixed_2 - current_dynamic) の分布 ===")
    deltas = []
    for venue, vs in venue_stats.items():
        if "current_dynamic" not in vs or "fixed_2" not in vs:
            continue
        cur_profit = vs["current_dynamic"]["returned"] - vs["current_dynamic"]["invested"]
        fx2_profit = vs["fixed_2"]["returned"] - vs["fixed_2"]["invested"]
        delta = fx2_profit - cur_profit
        deltas.append((venue, delta, vs["current_dynamic"]["races"]))

    deltas.sort(key=lambda x: x[1])
    for venue, delta, races in deltas:
        print(f"  {venue:6s} races={races:5d} profit_delta={delta:+,.0f}円")

    vals = [d for _, d, _ in deltas]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 0 else 0.0
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    n_positive = sum(1 for v in vals if v > 0)
    print()
    print(f"venues n={n}, mean profit_delta={mean:+.0f}円, 95%CI=[{ci_lo:+.0f}, {ci_hi:+.0f}]円, positive_venues={n_positive}/{n}")


if __name__ == "__main__":
    main()
