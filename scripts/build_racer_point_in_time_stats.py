from __future__ import annotations

"""
data/datasets/racer_race_history.csv (選手×レースのファクトテーブル)から、
「その日より前の実績のみ」を使ったリーク無しの選手成績特徴量を計算する。

日付粒度でカットオフする(同日内のレース順序は使わない/信用しない)ことで、
同日内のレース順の曖昧さによるリークを避ける、保守的な設計。

出力: data/datasets/racer_point_in_time_stats.csv
  racer_no, date, races_prior, win_rate_prior, place_rate_prior, avg_st_prior,
  course{1-6}_races_prior, course{1-6}_place_rate_prior, course{1-6}_avg_st_prior

date は「この日"から"適用される、この日より前の実績に基づく統計」という意味。
学習データにmerge_asof(direction='backward', 厳密にはこの日"以前"の最新の値)で
結合すればよい。ただし当日分がリークしないよう、学習データ側では
「同日は対象外」にするため、このファイルの date は「対象レースの日付」そのものにして、
値は必ずその日より前の実績のみで計算している。
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"
OUT_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_point_in_time_stats.csv"


def main() -> None:
    print("loading:", HISTORY_CSV)
    df = pd.read_csv(HISTORY_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)

    # 日付粒度に集約(同日に複数レースがあっても「その日の実績」としてまとめる)
    df["is_win"] = (df["finish"] == 1).astype(int)
    df["is_top3"] = (df["finish"] <= 3).astype(int)

    daily = (
        df.groupby(["racer_no", "date"], as_index=False)
        .agg(
            races=("finish", "count"),
            wins=("is_win", "sum"),
            top3=("is_top3", "sum"),
            st_sum=("st", "sum"),
            st_count=("st", "count"),
        )
    )

    # コース別の日次集計(1-6)
    course_daily_list = []
    for c in range(1, 7):
        sub = df[df["course"] == c].copy()
        sub["is_win_c"] = (sub["finish"] == 1).astype(int)
        sub["is_top3_c"] = (sub["finish"] <= 3).astype(int)
        g = (
            sub.groupby(["racer_no", "date"], as_index=False)
            .agg(
                **{
                    f"course{c}_races": ("finish", "count"),
                    f"course{c}_top3": ("is_top3_c", "sum"),
                    f"course{c}_st_sum": ("st", "sum"),
                    f"course{c}_st_count": ("st", "count"),
                }
            )
        )
        course_daily_list.append(g)

    daily = daily.sort_values(["racer_no", "date"]).reset_index(drop=True)
    for g in course_daily_list:
        daily = daily.merge(g, on=["racer_no", "date"], how="left")

    fill_cols = [c for c in daily.columns if c.startswith("course")]
    daily[fill_cols] = daily[fill_cols].fillna(0)

    # ---- 累積(当日を含む)を計算してから、当日分を引いて「当日より前」にする ----
    daily = daily.sort_values(["racer_no", "date"]).reset_index(drop=True)
    g = daily.groupby("racer_no")

    cum_races = g["races"].cumsum()
    cum_wins = g["wins"].cumsum()
    cum_top3 = g["top3"].cumsum()
    cum_st_sum = g["st_sum"].cumsum()
    cum_st_count = g["st_count"].cumsum()

    # 「当日より前」= 累積(当日含む) - 当日分
    daily["races_prior"] = cum_races - daily["races"]
    daily["wins_prior"] = cum_wins - daily["wins"]
    daily["top3_prior"] = cum_top3 - daily["top3"]
    daily["st_sum_prior"] = cum_st_sum - daily["st_sum"]
    daily["st_count_prior"] = cum_st_count - daily["st_count"]

    daily["win_rate_prior"] = np.where(daily["races_prior"] > 0, daily["wins_prior"] / daily["races_prior"], np.nan)
    daily["place_rate_prior"] = np.where(daily["races_prior"] > 0, daily["top3_prior"] / daily["races_prior"], np.nan)
    daily["avg_st_prior"] = np.where(daily["st_count_prior"] > 0, daily["st_sum_prior"] / daily["st_count_prior"], np.nan)

    for c in range(1, 7):
        cum_c_races = g[f"course{c}_races"].cumsum()
        cum_c_top3 = g[f"course{c}_top3"].cumsum()
        cum_c_st_sum = g[f"course{c}_st_sum"].cumsum()
        cum_c_st_count = g[f"course{c}_st_count"].cumsum()

        races_prior = cum_c_races - daily[f"course{c}_races"]
        top3_prior = cum_c_top3 - daily[f"course{c}_top3"]
        st_sum_prior = cum_c_st_sum - daily[f"course{c}_st_sum"]
        st_count_prior = cum_c_st_count - daily[f"course{c}_st_count"]

        daily[f"course{c}_races_prior"] = races_prior
        daily[f"course{c}_place_rate_prior"] = np.where(races_prior > 0, top3_prior / races_prior, np.nan)
        daily[f"course{c}_avg_st_prior"] = np.where(st_count_prior > 0, st_sum_prior / st_count_prior, np.nan)

    keep_cols = ["racer_no", "date", "races_prior", "win_rate_prior", "place_rate_prior", "avg_st_prior"]
    for c in range(1, 7):
        keep_cols += [f"course{c}_races_prior", f"course{c}_place_rate_prior", f"course{c}_avg_st_prior"]

    out = daily[keep_cols].copy()
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("saved:", OUT_CSV)
    print("rows:", len(out))
    print("racers:", out["racer_no"].nunique())
    print(out[out["races_prior"] > 50].head(3).to_string())


if __name__ == "__main__":
    main()
