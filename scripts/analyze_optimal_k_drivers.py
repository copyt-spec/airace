from __future__ import annotations

"""
仮説先行(会場の荒れやすさ、prob×odds)ではなく、逆方向から攻める:
「そのレースで実際に当たった組み合わせは、buy_score順で何番目だったか」
(hit_rank)を全レースで計算し、hit_rankがレースのどんな特徴(prob_gap、
確率分布のエントロピー、top1_prob、top1_odds)と相関するかを直接調べる。

hit_rankが小さい(1〜数点で当たる)レースを予測できる特徴が見つかれば、
そこでは点数を絞り(下限を下げる)、hit_rankが大きくなりがちな特徴の
レースでは点数を増やす、という条件付きmin_pointsの設計に使える。

データ: data/logs/predictions_combined_for_sim.csv + data/logs/results.csv
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


def ranked_all(ai_preds, venue: str):
    """フィルタ無し(min_prob/min_evを無視)で、prob>0&odds>0の全候補を
    buy_score降順に並べる。「下限を無視して何点あれば当たったか」を測るため。"""
    cfg = buy_selector._resolve_config(venue=venue)
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


def fit_ols_hc1(X: np.ndarray, y: np.ndarray, col_names: list) -> pd.DataFrame:
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    u2 = resid ** 2
    XtSX = (X * u2[:, None]).T @ X
    V = XtX_inv @ XtSX @ XtX_inv * (n / max(n - k, 1))
    se = np.sqrt(np.diag(V))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    result = pd.DataFrame({"coef": beta, "se": se, "ci_low": ci_low, "ci_high": ci_high}, index=col_names)
    result["significant"] = (result["ci_low"] > 0) | (result["ci_high"] < 0)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    result.attrs["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    result.attrs["n"] = n
    return result


def main() -> None:
    print("loading data...")
    df = load_data()
    print(f"rows: {len(df)}, races: {df.drop_duplicates(['date','venue','race_no']).shape[0]}")

    records = []
    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        if pd.isna(actual_combo):
            continue

        ai_preds = [{"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])} for _, r in race_df.iterrows()]
        ranked = ranked_all(ai_preds, venue=venue)
        if len(ranked) < 3:
            continue

        combos_in_order = [r["combo"] for r in ranked]
        try:
            hit_rank = combos_in_order.index(actual_combo) + 1
        except ValueError:
            continue  # 実際の結果comboがodds/prob>0の候補に無い(異常系)

        probs_sorted = sorted([r["prob"] for r in ranked], reverse=True)
        top1_prob = probs_sorted[0]
        top2_prob = probs_sorted[1] if len(probs_sorted) > 1 else 0.0
        prob_gap = top1_prob - top2_prob
        top1_odds = ranked[0]["odds_raw"]
        entropy = entropy_top10(probs_sorted)

        records.append({
            "venue": venue,
            "hit_rank": hit_rank,
            "n_candidates": len(ranked),
            "top1_prob": top1_prob,
            "top1_odds": top1_odds,
            "prob_gap": prob_gap,
            "entropy": entropy,
        })

    rdf = pd.DataFrame(records)
    print(f"races analyzed: {len(rdf)}")
    print()
    print("hit_rankの分布:")
    print(rdf["hit_rank"].describe().to_string())
    print()
    for k in [1, 2, 3, 5, 8, 12, 20, 30]:
        pct = (rdf["hit_rank"] <= k).mean() * 100
        print(f"  hit_rank<=  {k:3d}: {pct:6.2f}%のレースがカバーされる")

    # ---------- OLS: hit_rank ~ prob_gap + entropy + top1_prob + top1_odds ----------
    print()
    print("=" * 90)
    print("OLS(HC1頑健標準誤差): hit_rank ~ prob_gap + entropy + top1_prob + top1_odds")
    print("=" * 90)
    valid = rdf.dropna(subset=["hit_rank", "prob_gap", "entropy", "top1_prob", "top1_odds"])
    y = valid["hit_rank"].values.astype(float)
    X = np.column_stack([
        np.ones(len(valid)),
        valid["prob_gap"].values.astype(float),
        valid["entropy"].values.astype(float),
        valid["top1_prob"].values.astype(float),
        valid["top1_odds"].values.astype(float),
    ])
    col_names = ["const", "prob_gap", "entropy", "top1_prob", "top1_odds"]
    result = fit_ols_hc1(X, y, col_names)
    print(result.to_string(float_format=lambda x: f"{x:,.4f}"))
    print(f"n={result.attrs['n']}, R2={result.attrs['r2']:.4f}")

    # ---------- 単純相関 ----------
    print()
    print("=== 単純相関(hit_rankとの) ===")
    for col in ["prob_gap", "entropy", "top1_prob", "top1_odds"]:
        corr = rdf["hit_rank"].corr(rdf[col])
        print(f"  {col:12s}: r={corr:+.3f}")

    # ---------- entropy tercile別のカバレッジカーブ ----------
    print()
    print("=" * 90)
    print("entropy(確率分布の散らばり)3分位ごとの hit_rank<=K カバー率")
    print("=" * 90)
    rdf["entropy_tercile"] = pd.qcut(rdf["entropy"], 3, labels=["低(堅い)", "中", "高(荒れ)"])
    cov_rows = []
    for terc, g in rdf.groupby("entropy_tercile", observed=True):
        row = {"entropy_tercile": terc, "races": len(g), "entropy_mean": g["entropy"].mean()}
        for k in [1, 2, 3, 5, 8, 12]:
            row[f"K<={k}"] = (g["hit_rank"] <= k).mean() * 100
        cov_rows.append(row)
    print(pd.DataFrame(cov_rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---------- prob_gap tercile別のカバレッジカーブ ----------
    print()
    print("=" * 90)
    print("prob_gap(1位と2位の確率差)3分位ごとの hit_rank<=K カバー率")
    print("=" * 90)
    rdf["gap_tercile"] = pd.qcut(rdf["prob_gap"], 3, labels=["小(僅差)", "中", "大(頭一つ抜け)"])
    cov_rows2 = []
    for terc, g in rdf.groupby("gap_tercile", observed=True):
        row = {"gap_tercile": terc, "races": len(g), "prob_gap_mean": g["prob_gap"].mean()}
        for k in [1, 2, 3, 5, 8, 12]:
            row[f"K<={k}"] = (g["hit_rank"] <= k).mean() * 100
        cov_rows2.append(row)
    print(pd.DataFrame(cov_rows2).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
