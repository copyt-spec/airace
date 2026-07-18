from __future__ import annotations

"""
raw_txt(K-file)由来の data/datasets/startk_dataset.csv (1レース1行, 全24会場,
2023-03-31〜) を使って、現行の with_racer_stats CatBoostモデルで「もしその場で
予測していたら」を再現し、過去分の的中率/回収率をバックテストする。

【なぜこれが必要か】
- data/logs/predictions.csv はライブでアクセスした会場だけしかログが無く、
  蒲郡/びわこ/鳴門/徳山/下関/若松/福岡の7会場は一度もログが無い。
- しかし raw_txt(K-file)は全国・全会場の結果を毎日含むので、
  この7会場についても「出走表+天候+選手成績」は2023年から揃っている。
- 過去の「その時点でのオッズ」は存在しないため、engine.buy_selector の
  EV(期待値)ベースの点数可変選定は再現できない。その代わりここでは
  「モデル予測確率が高い上位K点を100円均一で買う」という単純な戦略で
  的中率と回収率(円)を計算する。オッズが無くても、外れは常に-100円、
  的中時は実際の払戻金(trifecta_payout)がそのまま使えるため、この戦略に
  限っては回収率も計算できる(選定にオッズを使わないだけ)。
  ライブ運用中の可変点数EV選定とは異なる簡易版である点に注意。

【リーク防止】
engine.model_loader_catboost_binary.BinaryCatBoostVenueModel.predict_proba() は
ライブ推論用に「選手ごとに一番新しい行」を選手成績特徴量として使う実装になって
おり、過去日付に対して使うと未来のデータが混ざる(リーク)。
このスクリプトでは data/datasets/racer_point_in_time_stats.csv を
(racer_no, date) の完全一致で結合し、その日時点で分かっていたはずの
prior統計だけを使う。

使い方:
  python3 scripts/backtest_from_raw_dataset.py --venue 戸田 --top-k 5
  python3 scripts/backtest_from_raw_dataset.py --venue 蒲郡,びわこ,鳴門 --top-k 5
  python3 scripts/backtest_from_raw_dataset.py --all --top-k 5
"""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.model_loader_catboost_binary import (  # noqa: E402
    BinaryCatBoostVenueModel,
    RACER_STATS_FEATURE_SUFFIXES,
)
from engine.venue_registry import VENUE_ORDER  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
STARTK_CSV = DATASET_DIR / "startk_dataset.csv"
RACER_STATS_CSV = DATASET_DIR / "racer_point_in_time_stats.csv"
OUT_DIR = PROJECT_ROOT / "data" / "logs"

TRIFECTA_COMBOS: List[str] = [
    f"{a}-{b}-{c}" for a, b, c in itertools.permutations([1, 2, 3, 4, 5, 6], 3)
]


def _norm_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def load_startk(venues: List[str]) -> pd.DataFrame:
    df = pd.read_csv(STARTK_CSV, low_memory=False)
    df["date"] = _norm_date(df["date"])
    df = df[df["venue"].isin(venues)].copy()
    return df.reset_index(drop=True)


def load_racer_stats() -> pd.DataFrame:
    stats = pd.read_csv(RACER_STATS_CSV, low_memory=False)
    stats["date"] = _norm_date(stats["date"])
    stats["racer_no"] = pd.to_numeric(stats["racer_no"], errors="coerce").fillna(0).astype(int)
    return stats


def expand_to_120(base_df: pd.DataFrame) -> pd.DataFrame:
    """1レース1行のbase_dfを、comboごとに120行へ展開する(全lane情報はそのままコピー)。"""
    n = len(base_df)
    combos = TRIFECTA_COMBOS
    k = len(combos)

    repeated = base_df.loc[base_df.index.repeat(k)].reset_index(drop=True)
    repeated["combo"] = combos * n
    return repeated


def join_point_in_time_racer_stats(work: pd.DataFrame, racer_stats: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    lane{n}_racer_no + date の完全一致で racer_point_in_time_stats.csv を結合する
    (未来データが混ざらないよう、その日付の行だけを厳密に使う)。
    """
    out = work.copy()

    base_cols = ["racer_no", "date", "races_prior", "win_rate_rate_prior_placeholder"]  # unused, kept for clarity
    core_cols = ["racer_no", "date", "races_prior", "win_rate_prior", "place_rate_prior", "avg_st_prior"]
    course_cols = []
    for c in range(1, 7):
        course_cols += [f"course{c}_races_prior", f"course{c}_place_rate_prior", f"course{c}_avg_st_prior"]

    stats_small = racer_stats[["racer_no", "date"] + core_cols[2:] + course_cols].copy()
    dup_before = len(stats_small)
    stats_small = stats_small.drop_duplicates(subset=["racer_no", "date"], keep="last")
    if debug and dup_before != len(stats_small):
        print(f"  [WARN] racer_point_in_time_stats had {dup_before - len(stats_small)} duplicate (racer_no,date) rows, kept last")

    if debug:
        total_rows = 0
        matched_rows = 0

    for lane in range(1, 7):
        racer_col = f"lane{lane}_racer_no"
        course_col = f"lane{lane}_course"

        out[racer_col] = pd.to_numeric(out.get(racer_col, 0), errors="coerce").fillna(0).astype(int)

        merged = out[[racer_col, "date"]].merge(
            stats_small,
            left_on=[racer_col, "date"],
            right_on=["racer_no", "date"],
            how="left",
        )

        if debug:
            total_rows += len(merged)
            matched_rows += int(merged["races_prior"].notna().sum())

        out[f"lane{lane}_races_prior"] = merged["races_prior"].values
        out[f"lane{lane}_win_rate_prior"] = merged["win_rate_prior"].values
        out[f"lane{lane}_place_rate_prior"] = merged["place_rate_prior"].values
        out[f"lane{lane}_avg_st_prior"] = merged["avg_st_prior"].values

        course_vals = pd.to_numeric(out.get(course_col, 0), errors="coerce").fillna(0).astype(int).clip(1, 6)

        place_rate_by_course = np.full(len(out), np.nan)
        avg_st_by_course = np.full(len(out), np.nan)
        for c in range(1, 7):
            mask = (course_vals == c).values
            if not mask.any():
                continue
            place_rate_by_course[mask] = merged[f"course{c}_place_rate_prior"].values[mask]
            avg_st_by_course[mask] = merged[f"course{c}_avg_st_prior"].values[mask]

        out[f"lane{lane}_course_place_rate_prior"] = place_rate_by_course
        out[f"lane{lane}_course_avg_st_prior"] = avg_st_by_course

    for pos, pos_name in [
        ("first", "combo_first_lane"),
        ("second", "combo_second_lane"),
        ("third", "combo_third_lane"),
    ]:
        for suf in RACER_STATS_FEATURE_SUFFIXES:
            out[f"{pos}_{suf}"] = np.nan
            for lane in range(1, 7):
                mask = out[pos_name] == lane
                out.loc[mask, f"{pos}_{suf}"] = out.loc[mask, f"lane{lane}_{suf}"]

    return out


def run_backtest_for_venue(
    loader: BinaryCatBoostVenueModel,
    racer_stats: pd.DataFrame,
    venue: str,
    base_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    if venue not in loader.models:
        print(f"[SKIP] no model for venue={venue}")
        return pd.DataFrame()

    model = loader.models[venue]
    meta = loader.metas[venue]
    feature_cols = list(meta.get("feature_cols", []))
    temperature = loader._get_temperature_by_venue(venue)

    rows_out = []

    CHUNK = 300  # races per chunk (=300*120=36000 rows) to keep memory/time bounded
    for start in range(0, len(base_df), CHUNK):
        chunk = base_df.iloc[start:start + CHUNK].reset_index(drop=True)
        df120 = expand_to_120(chunk)

        work = loader._add_feature_block(df120)
        work = join_point_in_time_racer_stats(work, racer_stats)
        x = loader._prepare_x(work, feature_cols)

        raw_scores = model.predict(x, prediction_type="RawFormulaVal")
        work["raw_score"] = raw_scores

        # レースごとにsoftmaxして上位K点を選ぶ
        for race_idx, race_key in enumerate(chunk["race_key"].tolist() if "race_key" in chunk.columns else range(len(chunk))):
            race_rows = work.iloc[race_idx * 120:(race_idx + 1) * 120]
            scores = race_rows["raw_score"].tolist()
            probs = loader._softmax(scores, temperature=temperature)
            combos = race_rows["combo"].tolist()
            prob_map = dict(zip(combos, probs))

            top_combos = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            top_combo_set = {c for c, _ in top_combos}

            base_row = chunk.iloc[race_idx]
            y_combo = str(base_row.get("y_combo", base_row.get("trifecta", ""))).strip()
            payout = base_row.get("trifecta_payout", 0)
            try:
                payout_val = float(payout)
            except Exception:
                payout_val = 0.0

            hit = y_combo in top_combo_set
            bet_cost = 100.0 * len(top_combo_set)
            ret = payout_val if hit else 0.0

            rows_out.append({
                "venue": venue,
                "date": base_row.get("date", ""),
                "race_no": base_row.get("race_no", ""),
                "top_k": len(top_combo_set),
                "is_hit": int(hit),
                "bet_cost_yen": bet_cost,
                "return_yen": ret,
                "profit_yen": ret - bet_cost,
                "top1_prob": max(prob_map.values()) if prob_map else 0.0,
            })

    return pd.DataFrame(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", type=str, default="", help="comma separated venue names")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if args.all:
        venues = VENUE_ORDER
    elif args.venue:
        venues = [v.strip() for v in args.venue.split(",") if v.strip()]
    else:
        raise SystemExit("specify --venue A,B,C or --all")

    print("loading models...")
    loader = BinaryCatBoostVenueModel(use_racer_stats=True, debug=False)

    print("loading startk dataset...")
    base_df = load_startk(venues)
    print("races loaded:", len(base_df))

    print("loading racer point-in-time stats...")
    racer_stats = load_racer_stats()

    all_results = []
    for venue in venues:
        vdf = base_df[base_df["venue"] == venue].sort_values(["date", "race_no"]).reset_index(drop=True)
        if vdf.empty:
            print(f"[SKIP] no rows for venue={venue}")
            continue
        print(f"--- {venue}: {len(vdf)} races ---")
        res = run_backtest_for_venue(loader, racer_stats, venue, vdf, args.top_k)
        if res.empty:
            continue
        all_results.append(res)

        buy_count = res["top_k"].sum()
        hit_count = res["is_hit"].sum()
        total_bets = res["bet_cost_yen"].sum()
        total_return = res["return_yen"].sum()
        roi = total_return / total_bets if total_bets > 0 else 0.0
        print(f"  races={len(res)} hit_rate={hit_count/len(res)*100:.2f}% roi={roi:.3f}")

    if not all_results:
        print("no results")
        return

    out_df = pd.concat(all_results, ignore_index=True)
    out_path = Path(args.out) if args.out else OUT_DIR / "raw_dataset_backtest.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path, "rows:", len(out_df))


if __name__ == "__main__":
    main()
