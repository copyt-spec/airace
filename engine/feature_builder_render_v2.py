from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import pandas as pd


DROP_COLS = {
    "race_key",
    "date",
    "venue",
    "race_no",
    "combo",
    "y_combo",
    "y_class",
}


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=False)


def _rank_asc(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=True)


def _add_rank_features(df120: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.DataFrame:
    if col not in df120.columns:
        return df120

    s = _safe_numeric_series(df120[col])
    if higher_is_better:
        rank = _rank_desc(s)
        best_val = s.max()
    else:
        rank = _rank_asc(s)
        best_val = s.min()

    df120[f"{col}_rank"] = rank
    df120[f"{col}_diff_from_best"] = s - best_val
    if not higher_is_better:
        df120[f"{col}_diff_from_best"] = best_val - s

    return df120


def _extract_lane_from_combo(combo: str, pos: int) -> int:
    try:
        parts = str(combo).split("-")
        if len(parts) != 3:
            return 0
        return int(parts[pos])
    except Exception:
        return 0


def _build_lane_master_map(df120: pd.DataFrame) -> pd.DataFrame:
    """
    laneごとの特徴量代表を作る。
    想定:
      lane1_win_rate, lane2_win_rate ... のような列が既にある場合はそれを使う。
      無ければ buildしない。
    """
    return df120


def _inside_outside_gap(df120: pd.DataFrame) -> pd.DataFrame:
    targets = [
        "ability_index",
        "prev_ability_index",
        "win_rate",
        "place_rate",
        "exhibit",
        "st",
        "avg_st",
    ]

    for t in targets:
        cols = [f"lane{i}_{t}" for i in range(1, 7) if f"lane{i}_{t}" in df120.columns]
        if len(cols) != 6:
            continue

        inside_cols = cols[:3]
        outside_cols = cols[3:]

        inside_mean = df120[inside_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)
        outside_mean = df120[outside_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)

        df120[f"inside_{t}_mean_v2"] = inside_mean
        df120[f"outside_{t}_mean_v2"] = outside_mean
        df120[f"inside_outside_{t}_gap_v2"] = inside_mean - outside_mean

    return df120


def _lane1_gap_features(df120: pd.DataFrame) -> pd.DataFrame:
    targets = [
        "win_rate",
        "place_rate",
        "ability_index",
        "prev_ability_index",
        "motor_rate",
        "boat_rate",
        "exhibit",
        "st",
        "avg_st",
    ]

    combo_col = "combo"
    if combo_col not in df120.columns:
        return df120

    first_lanes = df120[combo_col].astype(str).map(lambda x: _extract_lane_from_combo(x, 0))
    second_lanes = df120[combo_col].astype(str).map(lambda x: _extract_lane_from_combo(x, 1))
    third_lanes = df120[combo_col].astype(str).map(lambda x: _extract_lane_from_combo(x, 2))

    df120["combo_first_lane"] = first_lanes
    df120["combo_second_lane"] = second_lanes
    df120["combo_third_lane"] = third_lanes

    for t in targets:
        lane_cols = {i: f"lane{i}_{t}" for i in range(1, 7) if f"lane{i}_{t}" in df120.columns}
        if 1 not in lane_cols:
            continue

        lane1_col = lane_cols[1]
        lane1_val = _safe_numeric_series(df120[lane1_col])

        for i in range(2, 7):
            c = lane_cols.get(i)
            if not c:
                continue
            iv = _safe_numeric_series(df120[c])
            df120[f"lane{i}_{t}_gap_from_lane1"] = iv - lane1_val

    return df120


def _combo_position_feature(df120: pd.DataFrame) -> pd.DataFrame:
    """
    comboそのものの構造特徴
    """
    if "combo" not in df120.columns:
        return df120

    first_lane = df120["combo"].astype(str).map(lambda x: _extract_lane_from_combo(x, 0))
    second_lane = df120["combo"].astype(str).map(lambda x: _extract_lane_from_combo(x, 1))
    third_lane = df120["combo"].astype(str).map(lambda x: _extract_lane_from_combo(x, 2))

    df120["combo_first_lane"] = first_lane
    df120["combo_second_lane"] = second_lane
    df120["combo_third_lane"] = third_lane

    df120["combo_first_inside_flag"] = (first_lane <= 2).astype(int)
    df120["combo_first_center_flag"] = ((first_lane >= 3) & (first_lane <= 4)).astype(int)
    df120["combo_first_outside_flag"] = (first_lane >= 5).astype(int)

    df120["combo_lane_sum"] = first_lane + second_lane + third_lane
    df120["combo_lane_range"] = pd.concat([first_lane, second_lane, third_lane], axis=1).max(axis=1) - pd.concat([first_lane, second_lane, third_lane], axis=1).min(axis=1)

    return df120


def _top_lane_strength_features(df120: pd.DataFrame) -> pd.DataFrame:
    """
    1着候補 lane の強さを直接見る特徴
    """
    if "combo" not in df120.columns:
        return df120

    first_lane = df120["combo"].astype(str).map(lambda x: _extract_lane_from_combo(x, 0))

    targets = [
        "win_rate",
        "place_rate",
        "ability_index",
        "prev_ability_index",
        "motor_rate",
        "boat_rate",
        "exhibit",
        "st",
        "avg_st",
        "grade_num",
    ]

    for t in targets:
        lane_cols = {i: f"lane{i}_{t}" for i in range(1, 7) if f"lane{i}_{t}" in df120.columns}
        if not lane_cols:
            continue

        picked_vals = []
        for idx, lane in enumerate(first_lane.tolist()):
            c = lane_cols.get(int(lane))
            if not c:
                picked_vals.append(0.0)
            else:
                picked_vals.append(_safe_numeric_series(df120[c]).iloc[idx])

        df120[f"combo_first_{t}"] = picked_vals

    return df120


def add_render_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    trifecta_train_features.csv 相当のデータへ、追加の相対特徴量を付与する。
    """
    if df is None or df.empty:
        return df.copy()

    out = df.copy()

    # 既存の代表列に対するレース内順位
    rank_targets = [
        ("race_win_rate_mean", True),
        ("race_place_rate_mean", True),
        ("race_exhibit_mean", False),
        ("race_st_mean", False),
        ("race_avg_st_mean", False),
        ("race_ability_index_mean", True),
        ("race_prev_ability_index_mean", True),
    ]
    for col, higher_is_better in rank_targets:
        out = _add_rank_features(out, col, higher_is_better=higher_is_better)

    out = _combo_position_feature(out)
    out = _lane1_gap_features(out)
    out = _inside_outside_gap(out)
    out = _top_lane_strength_features(out)

    # 数値整理
    for c in out.columns:
        if c in {"race_key", "date", "venue", "combo", "y_combo"}:
            continue
        try:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        except Exception:
            pass

    out = out.replace([np.inf, -np.inf], 0.0)
    return out
