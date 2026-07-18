from __future__ import annotations

"""
2026-07-18追加: 「展開予想」ページの妥当性を上げるため、raw_txt(K-file)由来の
data/datasets/startk_dataset.csv (全24会場・2023-03-31〜のレース結果、進入
コース情報込み)から、実際の「進入コース別の1着/2着になる確率(決まり手の実データ版)」
を会場ごとに集計する。

これはモデルの推論ではなく、単純な過去の集計(=実際に起きたことの統計)なので、
学習データを採点してしまう心配([[boat_ai_backtest_methodology_limit]]で
判明した問題)は無い。展開予想のヒューリスティックをこの実データで裏付けることで、
「1コースがこの会場では実際に何%逃げているか」「2コースの差しは何%か」を
反映できるようにする。

出力: data/datasets/venue_course_stats.csv
  columns: venue, course, win_rate, place_top3_rate, race_count
  (winner_course, top3_course の意味は下記参照)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
STARTK_CSV = DATASET_DIR / "startk_dataset.csv"
OUT_CSV = DATASET_DIR / "venue_course_stats.csv"

VENUE_ORDER = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
    "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
    "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
]


def _courses_for_lanes(df: pd.DataFrame, lane_series: pd.Series, course_matrix: np.ndarray) -> np.ndarray:
    """
    lane_series(1-6, NaN可)を行インデックスとして、事前に作った
    course_matrix(shape=(n_rows,6), 列=lane1..6のcourse)から一括で
    「そのレーンの進入コース」を取り出す(1レースずつのPythonループを回避)。
    """
    idx = pd.to_numeric(lane_series, errors="coerce")
    out = np.full(len(idx), np.nan)
    valid = idx.between(1, 6).values
    row_pos = np.nonzero(valid)[0]
    col_pos = (idx.values[valid] - 1).astype(int)
    out[row_pos] = course_matrix[row_pos, col_pos]
    return out


def main() -> None:
    if not STARTK_CSV.exists():
        raise FileNotFoundError(f"missing: {STARTK_CSV}")

    print("loading:", STARTK_CSV)
    df = pd.read_csv(STARTK_CSV, low_memory=False)
    print("rows:", len(df))

    def parse_y_combo(s):
        parts = str(s).split("-")
        if len(parts) != 3:
            return None, None, None
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None, None, None

    parsed = df["y_combo"].map(parse_y_combo)
    df["lane_1st"] = parsed.map(lambda t: t[0])
    df["lane_2nd"] = parsed.map(lambda t: t[1])
    df["lane_3rd"] = parsed.map(lambda t: t[2])

    course_matrix = np.column_stack([
        pd.to_numeric(df[f"lane{i}_course"], errors="coerce").fillna(0).astype(int).values
        for i in range(1, 7)
    ])

    df["course_1st"] = _courses_for_lanes(df, df["lane_1st"], course_matrix)
    df["course_2nd"] = _courses_for_lanes(df, df["lane_2nd"], course_matrix)
    df["course_3rd"] = _courses_for_lanes(df, df["lane_3rd"], course_matrix)
    for col in ["course_1st", "course_2nd", "course_3rd"]:
        df.loc[df[col] <= 0, col] = np.nan

    rows = []
    for venue in VENUE_ORDER:
        vdf = df[df["venue"] == venue]
        n = len(vdf)
        if n == 0:
            for c in range(1, 7):
                rows.append({"venue": venue, "course": c, "win_rate": None, "place_top3_rate": None, "race_count": 0})
            continue

        valid = vdf[vdf["course_1st"].notna()]
        n_valid = len(valid)

        for c in range(1, 7):
            win_rate = float((valid["course_1st"] == c).sum() / n_valid) if n_valid else None
            top3_mask = (
                (valid["course_1st"] == c) | (valid["course_2nd"] == c) | (valid["course_3rd"] == c)
            )
            place_rate = float(top3_mask.sum() / n_valid) if n_valid else None
            rows.append({
                "venue": venue, "course": c,
                "win_rate": win_rate, "place_top3_rate": place_rate,
                "race_count": n_valid,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("saved:", OUT_CSV)
    print(out.to_string())


if __name__ == "__main__":
    main()
