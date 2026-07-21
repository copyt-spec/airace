# scripts/analyze_racer_class_patterns.py
# -*- coding: utf-8 -*-
"""
選手の級別(A1/A2/B1/B2)構成による「レースパターン」を分析する。

ユーザー仮説: 「1号艇がA1で他がB2のような、クラスではっきり格上が
1艇だけいるパターンは、絞り込みやすく勝ちやすい。ただしオッズが
つかない(=市場も織り込み済み)。逆に、このパターンなのにオッズが
高いレースがあれば、それは狙い目では」というもの。

このスクリプトは2段階で検証する:
  1. 「lane1_clear_favorite」(1号艇がA1/A2で、他の5艇より級別が
     明確に上)というパターンを定義し、実際の1号艇1着率を、
     このパターンに該当するレースとしないレースで比較する。
     -> 前半の仮説(絞り込みやすい=勝ちやすい)の検証。
  2. そのパターンのレースで実際に1号艇1着に賭けた場合の損益を
     オッズ帯別に見る。
     -> 後半の仮説(オッズが高い時が狙い目)の検証。

注意点(重要):
  - 選手の級別(grade)は data/masters/racer_stats.csv の「現在時点」の
    値を使っている。級別は概ね半年ごとに見直されるため、2023年など
    古いレースに現在の級別を当てはめると、当時と異なる級別だった
    選手が一定数いるはずである(=多少のノイズを許容した近似)。
  - 級別×選手番号のひも付けは data/datasets/trifecta_train_by_venue/
     の学習データ(構築時に学習用データセットに入っていた期間=
     7月より前)にしか無い。7月以降にバックフィルしたレース
     (scripts/backfill_predictions_from_results.py)には選手番号を
     ログしていないため、このスクリプトでは追跡できない。
     今後もこの分析を続けたいなら、predictions.csvへの記録時に
     lane別racer_no/gradeも一緒に残すようにする改修が要る。
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_BY_VENUE_DIR = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_by_venue"
RACER_STATS_PATH = PROJECT_ROOT / "data" / "masters" / "racer_stats.csv"
LIGHT_CSV = PROJECT_ROOT / "data" / "logs" / "prediction_results_sim_1y_light.csv"
RESULTS_CSV = PROJECT_ROOT / "data" / "logs" / "results.csv"
# 2026-07-21追加: ライブ予測/バックフィル時にengine.prediction_logger.save_lane_racer_no
# が書き出す、1レース=1行のレーン別racer_noログ(7月以降の分もここに溜まっていく)。
LIVE_LANE_RACER_NO_CSV = PROJECT_ROOT / "data" / "logs" / "predictions_lane_racer_no.csv"

GRADE_RANK = {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0}
LANE_RACER_COLS = [f"lane{i}_racer_no" for i in range(1, 7)]
USECOLS = ["date", "race_no"] + LANE_RACER_COLS


def load_race_lane_racer_no() -> pd.DataFrame:
    """
    学習データセット(7月より前)由来のレーン別racer_noに加え、
    engine.prediction_logger.save_lane_racer_noが継続的に書き足している
    ライブ/バックフィル分(predictions_lane_racer_no.csv、7月以降も蓄積中)を
    合わせて返す。同じレースが両方にあれば学習データセット側を優先する
    (由来がはっきりしているため)。
    """
    files = sorted(glob.glob(str(SPLIT_BY_VENUE_DIR / "*.csv")))
    all_races = []
    for fp in files:
        venue = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp, usecols=USECOLS, low_memory=False)
        df["date"] = df["date"].astype(str).str.zfill(8)
        races = df.drop_duplicates(subset=["date", "race_no"]).copy()
        races["venue"] = venue
        all_races.append(races)
    races = pd.concat(all_races, ignore_index=True)

    if LIVE_LANE_RACER_NO_CSV.exists():
        live = pd.read_csv(LIVE_LANE_RACER_NO_CSV, low_memory=False)
        live["date"] = live["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
        live["race_no"] = pd.to_numeric(live["race_no"], errors="coerce").fillna(0).astype(int)
        live = live[["date", "venue", "race_no"] + LANE_RACER_COLS].copy()
        combined = pd.concat([races, live], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "venue", "race_no"], keep="first")
        print(f"(ライブ/バックフィルログから{len(live)}レース分を追加、"
              f"重複除去後の合計は{len(combined)}レース)")
        return combined.reset_index(drop=True)

    return races


def load_grade_map() -> Dict[int, str]:
    stats = pd.read_csv(RACER_STATS_PATH, low_memory=False, dtype=str)
    stats["racer_no"] = pd.to_numeric(stats["racer_no"], errors="coerce").fillna(0).astype(int)
    return dict(zip(stats["racer_no"], stats["grade"].fillna("")))


def classify_pattern(races: pd.DataFrame, grade_map: Dict[int, str]) -> pd.DataFrame:
    for i in range(1, 7):
        races[f"lane{i}_grade"] = races[f"lane{i}_racer_no"].map(grade_map).fillna("")

    grade_cols = [f"lane{i}_grade" for i in range(1, 7)]
    ranks = races[grade_cols].apply(lambda col: col.map(GRADE_RANK))

    lane1_rank = ranks["lane1_grade"]
    other_max = ranks[[c for c in grade_cols if c != "lane1_grade"]].max(axis=1)

    races["lane1_grade"] = races["lane1_grade"]
    races["lane1_clear_favorite"] = (
        races["lane1_grade"].isin(["A1", "A2"]) & (lane1_rank > other_max)
    )
    return races


def ci_profit(g: pd.DataFrame, z: float = 1.96) -> pd.Series:
    n = len(g)
    m = g["profit_yen"].mean()
    s = g["profit_yen"].std(ddof=1) if n > 1 else np.nan
    se = s / np.sqrt(n) if n > 1 and s == s else np.nan
    return pd.Series({
        "n": n,
        "mean_profit": m,
        "ci_low": m - z * se if se == se else np.nan,
        "ci_high": m + z * se if se == se else np.nan,
        "roi_est_pct": m + 100.0,
    })


def main() -> None:
    print("loading lane-level racer_no from training dataset (7月より前のみ)...")
    races = load_race_lane_racer_no()
    print("races:", len(races))

    grade_map = load_grade_map()
    races = classify_pattern(races, grade_map)

    print()
    print("=== パターン内訳 ===")
    print(races["lane1_clear_favorite"].value_counts())

    # --- (1) 実際の1号艇1着率(絞り込みやすさの検証) ---
    res = pd.read_csv(RESULTS_CSV, low_memory=False)
    res["date"] = res["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    res["venue"] = res["venue"].astype(str).str.strip()
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)
    res["lane1_won"] = res["actual_combo"].astype(str).str.startswith("1-")

    m = races.merge(res[["date", "venue", "race_no", "lane1_won"]], on=["date", "venue", "race_no"], how="inner")
    print()
    print(f"=== 実際の1号艇1着率(結果と突合できた{len(m)}レース) ===")
    print(m.groupby("lane1_clear_favorite")["lane1_won"].agg(["mean", "count"]).round(4))

    # --- (2) そのパターンで1号艇に賭けた場合の損益(オッズパラドックス仮説の検証) ---
    light = pd.read_csv(LIGHT_CSV, low_memory=False)
    light["date"] = light["date"].astype(str)
    light["pred_1st"] = light["combo"].astype(str).str.split("-").str[0]

    merged = light.merge(
        races[["date", "venue", "race_no", "lane1_clear_favorite"]],
        on=["date", "venue", "race_no"], how="inner",
    )
    lane1_bets = merged[merged["pred_1st"] == "1"].copy()

    print()
    print(f"=== 1号艇1着に賭けたベット(n={len(lane1_bets)})、パターン別損益 ===")
    print(lane1_bets.groupby("lane1_clear_favorite").apply(ci_profit, include_groups=False).round(2))

    print()
    print("=== lane1_clear_favorite=True の時、オッズ帯別(仮説: 高オッズなら儲かる) ===")
    sub = lane1_bets[lane1_bets["lane1_clear_favorite"] == True]  # noqa: E712
    g = sub.groupby("odds_band").apply(ci_profit, include_groups=False)
    print(g.round(2).sort_values("mean_profit", ascending=False))
    print()
    print("(信頼区間が広い=まだ結論を出すには材料不足の可能性が高い区分に注意)")


if __name__ == "__main__":
    main()
