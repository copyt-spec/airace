from __future__ import annotations

"""
scripts/analyze_optimal_k_drivers.pyで見つかった「entropy(確率分布の
散らばり)でhit_rankに必要な点数が変わる」という知見を、実際に
buy_selector.select_best_bets()の点数下限・上限(現行: 全会場一律3〜12点)に
反映したらROIが改善するかを、リーク無し(時系列split)で検証する。

分割: 20260706より前をtrain(entropy tercile境界の決定に使うだけ、
ROI評価には使わない)、20260706以降をtest(この結果だけを最終評価とする)。

ポリシー(train側の分析結果から設定、test側でチューニングはしない):
  - entropy tercile境界はtrain側の分布から計算
  - 低entropy(堅い)  : min_points=1, max_points=6
  - 中entropy       : min_points=3, max_points=10 (現行のデフォルト相当)
  - 高entropy(荒れ) : min_points=4, max_points=16

点数自体はbuy_selector._calc_dynamic_points()と同じspread_score式を
再利用し、min/max境界だけをentropy tierに応じて差し替える
(_calc_dynamic_points_flexとして再実装、3〜12のハードクランプを除去)。

比較対象: 現行本番ロジック(buy_selector.select_best_bets、venue別固定3〜12点)
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "engine").exists() else Path("/sessions/practical-magical-edison/mnt/boat_ai")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions_combined_for_sim.csv"
RESULTS_CSV = LOG_DIR / "results.csv"

UNIT_BET_YEN = 100
CUTOFF_DATE = "20260703"  # train(< this) / test(>= this)。2026-07-23追加:
# テスト期間を長く取り会場別の確認をしやすくするため、20260706→20260703に短縮
# (train側は7/1-7/2+それ以前のスパースな期間のみで境界を決める)。

TIER_BOUNDS = {
    "low": (1, 6),
    "mid": (3, 10),
    "high": (4, 16),
}


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
        on=["date", "venue", "race_no"], how="left",
    )
    return merged


def prepared_rows(ai_preds, venue: str, cfg):
    rows = buy_selector._prepare_candidate_rows(ai_preds, cfg, use_min_prob=True, use_min_ev=True)
    if len(rows) < 3:
        relaxed_cfg = dict(cfg)
        relaxed_cfg["min_ev"] = min(float(cfg.get("min_ev", 0.85)), 0.70)
        rows = buy_selector._prepare_candidate_rows(ai_preds, relaxed_cfg, use_min_prob=True, use_min_ev=True)
    if len(rows) < 3:
        rows = buy_selector._prepare_candidate_rows(ai_preds, cfg, use_min_prob=False, use_min_ev=False)
    return rows


def score_and_sort(rows, cfg):
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


def entropy_top10(probs_sorted_desc):
    vals = [p for p in probs_sorted_desc[:10] if p > 0]
    if not vals:
        return 1.0
    total = sum(vals)
    if total <= 0:
        return 1.0
    norm = [p / total for p in vals]
    h = -sum(p * math.log(p + 1e-12) for p in norm)
    max_h = math.log(len(norm) + 1e-12)
    return max(0.0, min(1.0, h / max_h)) if max_h > 0 else 0.0


def calc_dynamic_points_flex(rows, venue, min_points, max_points):
    """buy_selector._calc_dynamic_pointsと同じspread_score式だが、
    3〜12のハードクランプを外し、呼び出し側が渡すmin/maxをそのまま使う。"""
    probs = sorted([buy_selector._safe_float(r.get("prob", 0.0), 0.0) for r in rows], reverse=True)
    if not probs:
        return min_points

    top1 = probs[0]
    top3_sum = sum(probs[:3])
    top5_sum = sum(probs[:5])
    entropy = buy_selector._calc_distribution_entropy(probs[:10])
    gap12 = max(0.0, top1 - (probs[1] if len(probs) >= 2 else 0.0))

    spread_score = 0.0
    spread_score += (0.12 - min(top1, 0.12)) / 0.12 * 0.36
    spread_score += (0.24 - min(top3_sum, 0.24)) / 0.24 * 0.20
    spread_score += (0.36 - min(top5_sum, 0.36)) / 0.36 * 0.12
    spread_score += (0.03 - min(gap12, 0.03)) / 0.03 * 0.08
    spread_score += entropy * 0.24

    v = buy_selector._normalize_venue_name(venue)
    if v == "丸亀":
        spread_score *= 0.72
    elif v == "戸田":
        spread_score *= 1.08
    elif v == "児島":
        spread_score *= 0.96
    elif v == "住之江":
        spread_score *= 0.90

    spread_score = max(0.0, min(1.0, spread_score))
    points = min_points + round((max_points - min_points) * spread_score)

    if top1 >= 0.10 and top3_sum >= 0.23:
        points = min(points, max(min_points, 3))
    elif top1 >= 0.08 and top5_sum >= 0.34:
        points = min(points, max(min_points, min(4, max_points)))

    if entropy >= 0.88 and top1 <= 0.05:
        points = min(max_points, points + 1)

    return max(min_points, min(max_points, points))


def race_entropy(ai_preds):
    probs_sorted = sorted([float(p["prob"]) for p in ai_preds], reverse=True)
    return entropy_top10(probs_sorted)


def main() -> None:
    print("loading data...")
    df = load_data()
    print(f"rows: {len(df)}, races: {df.drop_duplicates(['date','venue','race_no']).shape[0]}")

    train_entropies = []
    race_cache = {}  # key -> (ai_preds, venue, actual_combo, payout_val, entropy)

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]
        if pd.isna(actual_combo):
            continue
        payout_val = int(float(payout)) if pd.notna(payout) else 0
        ai_preds = [{"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])} for _, r in race_df.iterrows()]
        ent = race_entropy(ai_preds)
        key = (date, venue, race_no)
        race_cache[key] = (ai_preds, venue, actual_combo, payout_val, ent, date)
        if date < CUTOFF_DATE:
            train_entropies.append(ent)

    q33, q66 = np.percentile(train_entropies, [33.33, 66.67])
    print(f"train races: {len(train_entropies)}, entropy tercile境界(train由来): q33={q33:.4f}, q66={q66:.4f}")

    def tier_of(ent: float) -> str:
        if ent <= q33:
            return "low"
        elif ent <= q66:
            return "mid"
        return "high"

    # ---------- test期間で baseline(現行) vs entropy_adaptive を比較 ----------
    baseline = {"invested": 0, "returned": 0, "races": 0}
    adaptive = {"invested": 0, "returned": 0, "races": 0}
    venue_baseline = {}
    venue_adaptive = {}
    venue_race_counts = {}
    tier_counts = {"low": 0, "mid": 0, "high": 0}

    n_test = 0
    for key, (ai_preds, venue, actual_combo, payout_val, ent, date) in race_cache.items():
        if date < CUTOFF_DATE:
            continue
        n_test += 1
        venue_race_counts[venue] = venue_race_counts.get(venue, 0) + 1

        # --- baseline: 現行本番ロジックそのまま ---
        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)
        if best_bets:
            combos = [b["combo"] for b in best_bets]
            inv = len(combos) * UNIT_BET_YEN
            ret = payout_val if actual_combo in combos else 0
            baseline["invested"] += inv
            baseline["returned"] += ret
            baseline["races"] += 1
            vb = venue_baseline.setdefault(venue, {"invested": 0, "returned": 0})
            vb["invested"] += inv
            vb["returned"] += ret

        # --- entropy adaptive ---
        if buy_selector.should_skip_race(ai_preds, venue=venue):
            continue
        cfg = buy_selector._resolve_config(venue=venue)
        rows = prepared_rows(ai_preds, venue, cfg)
        if not rows:
            continue
        rows = score_and_sort(rows, cfg)

        tier = tier_of(ent)
        tier_counts[tier] += 1
        min_k, max_k = TIER_BOUNDS[tier]
        k = calc_dynamic_points_flex(rows, venue, min_k, max_k)
        k = max(1, min(k, len(rows)))
        selected = rows[:k]
        combos = [r["combo"] for r in selected]
        inv = len(combos) * UNIT_BET_YEN
        ret = payout_val if actual_combo in combos else 0
        adaptive["invested"] += inv
        adaptive["returned"] += ret
        adaptive["races"] += 1
        va = venue_adaptive.setdefault(venue, {"invested": 0, "returned": 0})
        va["invested"] += inv
        va["returned"] += ret

    print(f"\ntest races (total, resulted): {n_test}")
    print(f"tier分布(test): {tier_counts}")
    print()
    for name, s in [("baseline(現行)", baseline), ("entropy_adaptive", adaptive)]:
        roi = s["returned"] / s["invested"] * 100 if s["invested"] > 0 else 0.0
        profit = s["returned"] - s["invested"]
        print(f"{name:20s} races={s['races']:5d} invested={s['invested']:10,d} returned={s['returned']:10,d} profit={profit:+10,d} roi={roi:7.2f}%")

    # ---------- 会場別の詳細(baseline/adaptive両方のROI・利益) ----------
    print()
    print("=== 会場別 詳細(baseline vs entropy_adaptive) ===")
    header = f"{'venue':6s} {'races':>6s} {'base_roi%':>9s} {'base_profit':>12s} {'adp_roi%':>9s} {'adp_profit':>12s} {'delta_profit':>13s}"
    print(header)
    print("-" * len(header))
    deltas = []
    detail_rows = []
    for venue in set(list(venue_baseline.keys()) + list(venue_adaptive.keys())):
        vb = venue_baseline.get(venue, {"invested": 0, "returned": 0})
        va = venue_adaptive.get(venue, {"invested": 0, "returned": 0})
        b_profit = vb["returned"] - vb["invested"]
        a_profit = va["returned"] - va["invested"]
        b_roi = vb["returned"] / vb["invested"] * 100 if vb["invested"] > 0 else 0.0
        a_roi = va["returned"] / va["invested"] * 100 if va["invested"] > 0 else 0.0
        delta = a_profit - b_profit
        races_v = venue_race_counts.get(venue, 0)
        deltas.append((venue, delta))
        detail_rows.append((venue, races_v, b_roi, b_profit, a_roi, a_profit, delta))
    detail_rows.sort(key=lambda x: x[6])
    for venue, races_v, b_roi, b_profit, a_roi, a_profit, delta in detail_rows:
        print(f"{venue:6s} {races_v:6d} {b_roi:9.2f} {b_profit:12,.0f} {a_roi:9.2f} {a_profit:12,.0f} {delta:+13,.0f}")

    vals = [d for _, d in deltas]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 0 else 0.0
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    n_pos = sum(1 for v in vals if v > 0)
    print()
    print(f"venues n={n}, mean profit_delta={mean:+.0f}円, 95%CI=[{ci_lo:+.0f},{ci_hi:+.0f}]円, positive_venues={n_pos}/{n}")


if __name__ == "__main__":
    main()
