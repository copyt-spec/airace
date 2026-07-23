from __future__ import annotations

"""
buy_selector.select_best_bets()内部の「3〜12点」クランプを緩めたら回収率は
上がるのか?を検証する。

やること: 各レースについて、現行のスコアリング(_resolve_config + buy_score:
prob/ev/oddsの加重 + オッズ帯補正)で候補をbuy_score降順に並べ替えるところまでは
現行ロジックと全く同じにする。そのうえで「买い目の何番目(rank)を追加するか」を
1点ずつ切り出し、rankごとに「そのrankの点だけを全レースで買ったら回収率は
何%か」を集計する。これにより、12点の壁を超えて13点目・14点目…を追加したら
その追加分だけでROIがどう変わるかを直接、race-by-raceの周辺(marginal)効果として
測定できる(累積の母集団構成がrankごとに変わることによる歪みを避けるため)。

データ: data/logs/predictions_combined_for_sim.csv (新モデルの真のholdout結果
+ デプロイ後ログ) + data/logs/results.csv
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
MAX_RANK = 30


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
    """select_best_betsと同じ候補生成・スコアリングを再現(点数上限なし)。"""
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

    # rank(1-indexed)ごとの「その順位の1点だけ買った」集計
    rank_stats = {k: {"races": 0, "hits": 0, "returned": 0} for k in range(1, MAX_RANK + 1)}
    skip_races = 0
    n_resulted = 0

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]
        if pd.isna(actual_combo):
            continue
        n_resulted += 1
        payout_val = int(float(payout)) if pd.notna(payout) else 0

        ai_preds = [{"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])} for _, r in race_df.iterrows()]

        ranked = ranked_candidates(ai_preds, venue=venue)
        if not ranked:
            skip_races += 1
            continue

        for k in range(1, min(len(ranked), MAX_RANK) + 1):
            combo_k = ranked[k - 1]["combo"]
            s = rank_stats[k]
            s["races"] += 1
            if combo_k == actual_combo:
                s["hits"] += 1
                s["returned"] += payout_val

    print(f"races with results: {n_resulted}, skipped(should_skip_race/no candidates): {skip_races}")
    print()
    header = f"{'rank':>4s} {'races':>7s} {'hit_rate%':>10s} {'roi_this_rank%':>15s} {'cum_roi_1..k%':>14s}"
    print(header)
    print("-" * len(header))

    cum_invested = 0
    cum_returned = 0
    for k in range(1, MAX_RANK + 1):
        s = rank_stats[k]
        races = s["races"]
        if races == 0:
            continue
        invested_k = races * UNIT_BET_YEN
        returned_k = s["returned"]
        roi_k = returned_k / invested_k * 100
        hit_rate_k = s["hits"] / races * 100

        cum_invested += invested_k
        cum_returned += returned_k
        cum_roi = cum_returned / cum_invested * 100 if cum_invested > 0 else 0.0

        print(f"{k:4d} {races:7d} {hit_rate_k:10.2f} {roi_k:15.2f} {cum_roi:14.2f}")


if __name__ == "__main__":
    main()
