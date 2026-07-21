# scripts/build_motor_point_in_time_stats.py
# -*- coding: utf-8 -*-
"""
data/datasets/racer_race_history.csv (選手×レースのファクトテーブル)から、
「(会場, モーター番号)ごとの直近の調子」をリーク無しで計算する。

scripts/analyze_motor_recent_form.py で検証した「モーターの直近8走の
複勝率は、選手自身の実力とは別に将来の成績を追加的に予測する」という
結果を、実際にモデルの特徴量として使えるようにするためのビルダー。

モーター番号は会場ごとに割り振られる識別子であり、同じ番号でも会場が
違えば別のモーターを指す。また物理的なモーター交換(整備・積み替え)が
定期的に起きるため、選手の通算成績のような「全期間平均」ではなく、
直近の使用実績(トレイリングウィンドウ)を使う
(analyze_motor_recent_form.pyのコメント参照)。

出力: data/datasets/motor_point_in_time_stats.csv
  venue, motor, date, motor_recent8_races_prior, motor_recent8_place_rate_prior

date は「この日の最初のレースより前」の直近8走(同一venue×motor)を集計した
値、という意味(racer_point_in_time_stats.csvのrecent5_place_rate_priorと
同じ設計思想: 同日内の後半レースの影響を混ぜない)。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"
OUT_CSV = PROJECT_ROOT / "data" / "datasets" / "motor_point_in_time_stats.csv"

RECENT_WINDOW = 8


def main() -> None:
    print("loading:", HISTORY_CSV)
    df = pd.read_csv(HISTORY_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["motor"] = pd.to_numeric(df["motor"], errors="coerce")
    df = df[df["motor"].notna()].copy()
    df["motor"] = df["motor"].astype(int)
    df["is_top3"] = (df["finish"] <= 3).astype(int)

    race_level = df.sort_values(["venue", "motor", "date", "race_no"]).reset_index(drop=True)
    gm = race_level.groupby(["venue", "motor"])["is_top3"]

    race_level["motor_recent_place_rate_prior"] = gm.transform(
        lambda s: s.shift(1).rolling(RECENT_WINDOW, min_periods=RECENT_WINDOW).mean()
    )
    race_level["motor_recent_races_prior"] = gm.transform(
        lambda s: s.shift(1).expanding().count()
    ).clip(upper=RECENT_WINDOW)

    # (venue, motor, date)の「その日最初のレースより前」の値を代表値として使う
    daily = (
        race_level.groupby(["venue", "motor", "date"], as_index=False)
        .first()[["venue", "motor", "date", "motor_recent_races_prior", "motor_recent_place_rate_prior"]]
    )
    daily = daily.rename(columns={
        "motor_recent_races_prior": f"motor_recent{RECENT_WINDOW}_races_prior",
        "motor_recent_place_rate_prior": f"motor_recent{RECENT_WINDOW}_place_rate_prior",
    })

    tmp_path = OUT_CSV.with_suffix(".csv.tmp")
    daily.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(OUT_CSV)  # atomic

    print("saved:", OUT_CSV)
    print("rows:", len(daily))
    print("venue x motor pairs:", daily[["venue", "motor"]].drop_duplicates().shape[0])
    print(daily[daily[f"motor_recent{RECENT_WINDOW}_races_prior"] >= RECENT_WINDOW].head(3).to_string())


if __name__ == "__main__":
    main()
