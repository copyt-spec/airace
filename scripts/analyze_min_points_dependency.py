from __future__ import annotations

"""
「下限(min_points)を会場・レースの条件で可変にすべきか」を検証する。

仮説1(会場依存): 荒れやすい(=favoriteが当たりにくい)会場ほど、点数を
増やす(rank2/rank3を追加する)ことのメリットが大きいはず。堅い会場では
逆に1点(favorite)だけで十分で、追加点はコストを増やすだけ。
  -> 会場ごとに「favorite的中率」を計算し、rank2/rank3追加の限界利益との
     相関を見る。

仮説2(レース依存): top1(一番buy_scoreが高い===実質favorite)のodds/probの
組み合わせで、rank2/rank3を追加することの限界利益が変わるはず。
特に「確率は高いがoddsが低い」レース(堅い本命レース)では、追加点の
限界利益がゼロ以下になっているのではないか。
  -> top1_prob x top1_odds でレースを2x2(中央値split)に分け、
     rank2/rank3追加の限界ROIをbucketごとに集計する。

データ: data/logs/predictions_combined_for_sim.csv + data/logs/results.csv
(engine.buy_selectorの現行スコアリングロジックをそのまま再利用、点数上限なし)
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

    race_records = []

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]
        if pd.isna(actual_combo):
            continue
        payout_val = int(float(payout)) if pd.notna(payout) else 0

        ai_preds = [{"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])} for _, r in race_df.iterrows()]
        ranked = ranked_candidates(ai_preds, venue=venue)
        if len(ranked) < 3:
            continue

        top1 = ranked[0]
        rank1_hit = 1 if top1["combo"] == actual_combo else 0
        rank1_return = payout_val if rank1_hit else 0
        rank2_hit = 1 if ranked[1]["combo"] == actual_combo else 0
        rank2_return = payout_val if rank2_hit else 0
        rank3_hit = 1 if ranked[2]["combo"] == actual_combo else 0
        rank3_return = payout_val if rank3_hit else 0

        race_records.append({
            "venue": venue,
            "top1_prob": top1["prob"],
            "top1_odds": top1["odds_raw"],
            "rank1_hit": rank1_hit,
            "rank1_return": rank1_return,
            "rank2_hit": rank2_hit,
            "rank2_return": rank2_return,
            "rank3_hit": rank3_hit,
            "rank3_return": rank3_return,
        })

    rdf = pd.DataFrame(race_records)
    print(f"races used (>=3 candidates): {len(rdf)}")
    print()

    # ---------- 仮説1: 会場別 favorite的中率 と rank2/rank3追加の限界利益の関係 ----------
    print("=" * 100)
    print("仮説1: 会場別 favorite的中率(rank1_hit_rate) と rank2/rank3追加の限界利益")
    print("=" * 100)
    rows = []
    for venue, g in rdf.groupby("venue"):
        races = len(g)
        fav_hit_rate = g["rank1_hit"].mean() * 100
        rank2_profit = g["rank2_return"].sum() - races * UNIT_BET_YEN
        rank2_roi = g["rank2_return"].sum() / (races * UNIT_BET_YEN) * 100
        rank3_profit = g["rank3_return"].sum() - races * UNIT_BET_YEN
        rank3_roi = g["rank3_return"].sum() / (races * UNIT_BET_YEN) * 100
        rows.append({
            "venue": venue, "races": races, "fav_hit_rate%": fav_hit_rate,
            "rank2_roi%": rank2_roi, "rank2_profit": rank2_profit,
            "rank3_roi%": rank3_roi, "rank3_profit": rank3_profit,
        })
    vdf = pd.DataFrame(rows).sort_values("fav_hit_rate%")
    pd.set_option("display.width", 160)
    print(vdf.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    corr_r2 = vdf["fav_hit_rate%"].corr(vdf["rank2_roi%"])
    corr_r3 = vdf["fav_hit_rate%"].corr(vdf["rank3_roi%"])
    print()
    print(f"相関係数: fav_hit_rate% vs rank2_roi% = {corr_r2:+.3f}")
    print(f"相関係数: fav_hit_rate% vs rank3_roi% = {corr_r3:+.3f}")
    print("(負の相関なら「favoriteが当たりにくい=荒れる会場ほどrank2/3の価値が高い」という仮説と整合)")

    # ---------- 仮説2: top1_prob x top1_odds でのrank2/rank3限界ROI ----------
    print()
    print("=" * 100)
    print("仮説2: top1_prob(高/低) x top1_odds(高/低) 別のrank2/rank3限界ROI")
    print("=" * 100)

    prob_median = rdf["top1_prob"].median()
    odds_median = rdf["top1_odds"].median()
    print(f"top1_prob中央値={prob_median:.4f}, top1_odds中央値={odds_median:.2f}")

    rdf["prob_bucket"] = pd.cut(rdf["top1_prob"], bins=[-1, prob_median, 999], labels=["prob_low", "prob_high"])
    rdf["odds_bucket"] = pd.cut(rdf["top1_odds"], bins=[-1, odds_median, 99999], labels=["odds_low", "odds_high"])

    rows2 = []
    for (pb, ob), g in rdf.groupby(["prob_bucket", "odds_bucket"], observed=True):
        races = len(g)
        if races == 0:
            continue
        rank1_roi = g["rank1_return"].sum() / (races * UNIT_BET_YEN) * 100
        rank2_roi = g["rank2_return"].sum() / (races * UNIT_BET_YEN) * 100
        rank3_roi = g["rank3_return"].sum() / (races * UNIT_BET_YEN) * 100
        rows2.append({
            "prob_bucket": pb, "odds_bucket": ob, "races": races,
            "rank1_roi%": rank1_roi, "rank2_roi%": rank2_roi, "rank3_roi%": rank3_roi,
        })
    b2 = pd.DataFrame(rows2)
    print(b2.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print()
    print("注目: prob_high x odds_low (確率高いがoddsは低い『堅い本命』レース)の行で、")
    print("rank2/rank3のROIがrank1より大きく劣化していれば、ユーザーの仮説")
    print("(堅い本命レースでは追加点の下限を強制するとムダ)を支持する結果になる。")


if __name__ == "__main__":
    main()
