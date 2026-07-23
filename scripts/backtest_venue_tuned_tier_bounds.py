from __future__ import annotations

"""
scripts/backtest_entropy_adaptive_points.pyで「entropy閾値を会場別に
チューニングしても、負けている大口会場(平和島・丸亀・住之江・多摩川・
下関・三国・桐生)の顔ぶれは変わらない」ことが分かった。つまり問題は
entropy閾値の置き場所ではなく、tierごとのmin/max点数バウンドそのもの。

このスクリプトでは、tierごとの点数バウンド自体を複数プロファイルとして
用意し、train期間のデータだけを使って:
  - train側のサンプル数が十分ある会場(戸田・江戸川・桐生。閾値は
    MIN_VENUE_TRAIN_RACESで調整可)は、その会場の train profit を
    最大化するプロファイルを個別に選ぶ。
  - それ以外の(サンプルが少ない)会場は、全会場プールしたtrain profit を
    最大化する「全体最適プロファイル」を採用する(=事実上の縮小推定。
    小サンプル会場に個別最適化を許すと過学習するため)。
プロファイルの中には「baseline_equiv」(=現行と同じ一律3〜10点、
実質的に適応方式を使わない)を含めており、どの会場についても
適応方式がtrainで現行を上回れない場合は自動的にオプトアウトできる
ようにしてある。

決めたプロファイルをtest期間(train集計には一切使っていない、真の
out-of-sample)に適用し、現行本番ロジック(buy_selector.select_best_bets)
と比較する。

データ: data/logs/predictions_combined_for_sim.csv + data/logs/results.csv
分割: train(< 20260703) / test(>= 20260703)
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
CUTOFF_DATE = "20260703"
ENTROPY_SHRINKAGE_K = 30
MIN_VENUE_TRAIN_RACES = 60  # これ未満の会場は個別最適化せず全体最適プロファイルを使う

PROFILES = {
    "baseline_equiv":      {"low": (3, 10), "mid": (3, 10), "high": (3, 10)},
    "current_adaptive":    {"low": (1, 6),  "mid": (3, 10), "high": (4, 16)},
    "mild_adaptive":       {"low": (2, 7),  "mid": (3, 10), "high": (4, 13)},
    "aggressive_low_cut":  {"low": (1, 4),  "mid": (2, 8),  "high": (4, 16)},
    "aggressive_high_lift": {"low": (1, 6), "mid": (3, 10), "high": (6, 20)},
    "low_only_adaptive":   {"low": (1, 6),  "mid": (3, 10), "high": (3, 10)},
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


def prepared_rows(ai_preds, cfg):
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


def race_entropy(ai_preds):
    probs_sorted = sorted([float(p["prob"]) for p in ai_preds], reverse=True)
    return entropy_top10(probs_sorted)


def calc_dynamic_points_flex(rows, venue, min_points, max_points):
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


def simulate_race(ai_preds, venue, tier, profile):
    """指定プロファイル・tierで1レース分の(invested, returned, hit)を計算する。
    should_skip_raceは全プロファイル共通(現行ロジックのまま)。"""
    if buy_selector.should_skip_race(ai_preds, venue=venue):
        return None
    cfg = buy_selector._resolve_config(venue=venue)
    rows = prepared_rows(ai_preds, cfg)
    if not rows:
        return None
    rows = score_and_sort(rows, cfg)
    min_k, max_k = profile[tier]
    k = calc_dynamic_points_flex(rows, venue, min_k, max_k)
    k = max(1, min(k, len(rows)))
    return rows[:k]


def main() -> None:
    print("loading data...")
    df = load_data()
    print(f"rows: {len(df)}, races: {df.drop_duplicates(['date','venue','race_no']).shape[0]}")

    race_cache = {}
    train_entropies = []
    train_entropies_by_venue: dict = {}

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
            train_entropies_by_venue.setdefault(venue, []).append(ent)

    q33, q66 = np.percentile(train_entropies, [33.33, 66.67])
    print(f"train races: {len(train_entropies)}, 全体entropy tercile: q33={q33:.4f}, q66={q66:.4f}")

    venue_bounds = {}
    for venue, vals in train_entropies_by_venue.items():
        n_v = len(vals)
        if n_v >= 2:
            vq33, vq66 = np.percentile(vals, [33.33, 66.67])
        else:
            vq33 = vq66 = vals[0]
        w = n_v / (n_v + ENTROPY_SHRINKAGE_K)
        venue_bounds[venue] = (w * vq33 + (1 - w) * q33, w * vq66 + (1 - w) * q66)

    def tier_of(venue: str, ent: float) -> str:
        vq33, vq66 = venue_bounds.get(venue, (q33, q66))
        if ent <= vq33:
            return "low"
        elif ent <= vq66:
            return "mid"
        return "high"

    # ---------- STEP1: train上で各プロファイルのprofitを計算(全体プール + 会場別) ----------
    print("\ncomputing train profit for each profile...")
    profile_global = {name: {"invested": 0, "returned": 0} for name in PROFILES}
    profile_by_venue = {name: {} for name in PROFILES}

    for key, (ai_preds, venue, actual_combo, payout_val, ent, date) in race_cache.items():
        if date >= CUTOFF_DATE:
            continue
        tier = tier_of(venue, ent)
        for name, profile in PROFILES.items():
            selected = simulate_race(ai_preds, venue, tier, profile)
            if not selected:
                continue
            combos = [r["combo"] for r in selected]
            inv = len(combos) * UNIT_BET_YEN
            ret = payout_val if actual_combo in combos else 0
            profile_global[name]["invested"] += inv
            profile_global[name]["returned"] += ret
            pv = profile_by_venue[name].setdefault(venue, {"invested": 0, "returned": 0})
            pv["invested"] += inv
            pv["returned"] += ret

    print("\n全体プール train成績(プロファイル別):")
    for name, s in profile_global.items():
        profit = s["returned"] - s["invested"]
        roi = s["returned"] / s["invested"] * 100 if s["invested"] > 0 else 0.0
        print(f"  {name:20s} invested={s['invested']:9,d} profit={profit:+9,d} roi={roi:7.2f}%")

    best_global_name = max(profile_global, key=lambda n: profile_global[n]["returned"] - profile_global[n]["invested"])
    print(f"\n全体最適プロファイル: {best_global_name}")

    # ---------- STEP2: 会場ごとにプロファイルを決定 ----------
    venue_train_race_count = {v: len(vals) for v, vals in train_entropies_by_venue.items()}
    selected_profile_by_venue = {}
    print(f"\n会場別プロファイル選択(MIN_VENUE_TRAIN_RACES={MIN_VENUE_TRAIN_RACES}):")
    for venue in sorted(set(v for _, v, *_ in race_cache.values())):
        n_v = venue_train_race_count.get(venue, 0)
        if n_v >= MIN_VENUE_TRAIN_RACES:
            venue_profiles = profile_by_venue
            best_name = max(
                PROFILES,
                key=lambda n: (venue_profiles[n].get(venue, {"invested": 0, "returned": 0})["returned"]
                               - venue_profiles[n].get(venue, {"invested": 0, "returned": 0})["invested"]),
            )
            selected_profile_by_venue[venue] = best_name
            print(f"  {venue:6s} n_train={n_v:4d} -> 個別選択: {best_name}")
        else:
            selected_profile_by_venue[venue] = best_global_name
            print(f"  {venue:6s} n_train={n_v:4d} -> 全体最適採用: {best_global_name}")

    # ---------- STEP3: testで評価(baseline vs 選択済みプロファイル) ----------
    baseline = {"invested": 0, "returned": 0, "races": 0}
    tuned = {"invested": 0, "returned": 0, "races": 0}
    venue_baseline = {}
    venue_tuned = {}
    venue_race_counts = {}

    n_test = 0
    for key, (ai_preds, venue, actual_combo, payout_val, ent, date) in race_cache.items():
        if date < CUTOFF_DATE:
            continue
        n_test += 1
        venue_race_counts[venue] = venue_race_counts.get(venue, 0) + 1

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

        tier = tier_of(venue, ent)
        profile = PROFILES[selected_profile_by_venue[venue]]
        selected = simulate_race(ai_preds, venue, tier, profile)
        if not selected:
            continue
        combos = [r["combo"] for r in selected]
        inv = len(combos) * UNIT_BET_YEN
        ret = payout_val if actual_combo in combos else 0
        tuned["invested"] += inv
        tuned["returned"] += ret
        tuned["races"] += 1
        vt = venue_tuned.setdefault(venue, {"invested": 0, "returned": 0})
        vt["invested"] += inv
        vt["returned"] += ret

    print(f"\ntest races: {n_test}")
    print()
    for name, s in [("baseline(現行)", baseline), ("venue_tuned_profile", tuned)]:
        roi = s["returned"] / s["invested"] * 100 if s["invested"] > 0 else 0.0
        profit = s["returned"] - s["invested"]
        print(f"{name:22s} races={s['races']:5d} invested={s['invested']:10,d} returned={s['returned']:10,d} profit={profit:+10,d} roi={roi:7.2f}%")

    print()
    print("=== 会場別 詳細(baseline vs venue_tuned_profile) ===")
    header = f"{'venue':6s} {'races':>6s} {'profile':>18s} {'base_roi%':>9s} {'base_profit':>12s} {'tun_roi%':>9s} {'tun_profit':>12s} {'delta':>10s}"
    print(header)
    print("-" * len(header))
    detail_rows = []
    for venue in set(list(venue_baseline.keys()) + list(venue_tuned.keys())):
        vb = venue_baseline.get(venue, {"invested": 0, "returned": 0})
        vt = venue_tuned.get(venue, {"invested": 0, "returned": 0})
        b_profit = vb["returned"] - vb["invested"]
        t_profit = vt["returned"] - vt["invested"]
        b_roi = vb["returned"] / vb["invested"] * 100 if vb["invested"] > 0 else 0.0
        t_roi = vt["returned"] / vt["invested"] * 100 if vt["invested"] > 0 else 0.0
        delta = t_profit - b_profit
        races_v = venue_race_counts.get(venue, 0)
        detail_rows.append((venue, races_v, selected_profile_by_venue.get(venue, "?"), b_roi, b_profit, t_roi, t_profit, delta))
    detail_rows.sort(key=lambda x: x[7])
    for venue, races_v, prof, b_roi, b_profit, t_roi, t_profit, delta in detail_rows:
        print(f"{venue:6s} {races_v:6d} {prof:>18s} {b_roi:9.2f} {b_profit:12,.0f} {t_roi:9.2f} {t_profit:12,.0f} {delta:+10,.0f}")

    deltas = [d for *_, d in detail_rows]
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 0 else 0.0
    n_pos = sum(1 for v in deltas if v > 0)
    print()
    print(f"venues n={n}, mean profit_delta={mean:+.0f}円, 95%CI=[{mean-1.96*se:+.0f},{mean+1.96*se:+.0f}]円, positive_venues={n_pos}/{n}")

    large = [row for row in detail_rows if row[1] >= 70]
    large_pos = sum(1 for row in large if row[7] > 0)
    print(f"races>=70の大口会場: {len(large)}会場中 {large_pos}会場がプラス")


if __name__ == "__main__":
    main()
