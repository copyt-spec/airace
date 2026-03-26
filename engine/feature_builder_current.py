from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


LEAK_EXACT_COLS = {
    "race_id",
    "race_key",
    "date",
    "venue",
    "race_no",
    "combo",
    "trifecta",
    "trifecta_payout",
    "payout",
    "y",
    "is_hit",
    "y_class",
    "y_combo",
    "first_only_hit",
    "first2_hit",
    "weather",
    "wind_dir",
}

LEAK_KEYWORDS = [
    "finish",
    "result",
    "payout",
    "pay",
    "return",
    "odds",
    "hit",
    "y_class",
    "y_combo",
]


def safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def normalize_venue(v: str) -> str:
    s = safe_str(v)
    if "丸亀" in s:
        return "丸亀"
    if "児島" in s:
        return "児島"
    if "戸田" in s:
        return "戸田"
    return s


def weather_to_code(s: str) -> int:
    x = safe_str(s)
    table = {
        "晴": 1,
        "晴れ": 1,
        "曇": 2,
        "くもり": 2,
        "雨": 3,
        "雪": 4,
    }
    return table.get(x, 0)


def wind_dir_to_code(s: str) -> int:
    x = safe_str(s)
    table = {
        "無風": 0,
        "北": 1,
        "北東": 2,
        "東": 3,
        "南東": 4,
        "南": 5,
        "南西": 6,
        "西": 7,
        "北西": 8,
    }
    return table.get(x, 0)


def split_combo(combo: str) -> Tuple[int, int, int]:
    try:
        a, b, c = str(combo).split("-")
        return int(a), int(b), int(c)
    except Exception:
        return 0, 0, 0


def ensure_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def pick_lane_metric(df: pd.DataFrame, lane_series: pd.Series, metric: str) -> pd.Series:
    vals: List[float] = []
    for idx, lane in lane_series.items():
        lane_i = int(lane) if pd.notna(lane) else 0
        col = f"lane{lane_i}_{metric}"
        if lane_i in range(1, 7) and col in df.columns:
            vals.append(safe_float(df.at[idx, col], 0.0))
        else:
            vals.append(0.0)
    return pd.Series(vals, index=df.index, dtype=float)


def lane_mean(df: pd.DataFrame, metric: str) -> pd.Series:
    cols = [f"lane{i}_{metric}" for i in range(1, 7) if f"lane{i}_{metric}" in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index, dtype=float)
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x.mean(axis=1)


def lane_std(df: pd.DataFrame, metric: str) -> pd.Series:
    cols = [f"lane{i}_{metric}" for i in range(1, 7) if f"lane{i}_{metric}" in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index, dtype=float)
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x.std(axis=1)


def build_is_hit(df: pd.DataFrame) -> pd.Series:
    combo = df["combo"].astype(str)
    y_combo = df["y_combo"].astype(str)
    return (combo == y_combo).astype(int)


def add_feature_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    現在の trifecta_train.csv / 推論用 DataFrame に対して、
    v3特徴をまとめて作成する。
    DataFrame fragmentation を避けるため、追加列は最後に一括結合する。
    """
    out = df.copy()

    extra: Dict[str, pd.Series] = {}

    if "weather" in out.columns:
        extra["weather_code"] = out["weather"].astype(str).map(weather_to_code)
    else:
        extra["weather_code"] = pd.Series(0, index=out.index, dtype=int)

    if "wind_dir" in out.columns:
        extra["wind_dir_code"] = out["wind_dir"].astype(str).map(wind_dir_to_code)
    else:
        extra["wind_dir_code"] = pd.Series(0, index=out.index, dtype=int)

    if (
        "combo_first_lane" not in out.columns
        or "combo_second_lane" not in out.columns
        or "combo_third_lane" not in out.columns
    ):
        combo_parts = out["combo"].astype(str).map(split_combo)
        extra["combo_first_lane"] = pd.Series([x[0] for x in combo_parts], index=out.index, dtype=float)
        extra["combo_second_lane"] = pd.Series([x[1] for x in combo_parts], index=out.index, dtype=float)
        extra["combo_third_lane"] = pd.Series([x[2] for x in combo_parts], index=out.index, dtype=float)

    combo_first = ensure_col(out if "combo_first_lane" in out.columns else pd.concat([out, pd.DataFrame(extra)], axis=1), "combo_first_lane", 0)
    combo_second = ensure_col(out if "combo_second_lane" in out.columns else pd.concat([out, pd.DataFrame(extra)], axis=1), "combo_second_lane", 0)
    combo_third = ensure_col(out if "combo_third_lane" in out.columns else pd.concat([out, pd.DataFrame(extra)], axis=1), "combo_third_lane", 0)

    base_metrics = [
        "boat",
        "course",
        "exhibit",
        "motor",
        "racer_no",
        "st",
    ]

    picked: Dict[str, Dict[str, pd.Series]] = {
        "first": {},
        "second": {},
        "third": {},
    }

    work_df = pd.concat([out, pd.DataFrame(extra)], axis=1)

    for metric in base_metrics:
        picked["first"][metric] = pick_lane_metric(work_df, combo_first, metric)
        picked["second"][metric] = pick_lane_metric(work_df, combo_second, metric)
        picked["third"][metric] = pick_lane_metric(work_df, combo_third, metric)

    race_mean = {metric: lane_mean(work_df, metric) for metric in base_metrics}
    race_std_map = {metric: lane_std(work_df, metric) for metric in base_metrics}

    for metric in base_metrics:
        f = picked["first"][metric]
        s = picked["second"][metric]
        t = picked["third"][metric]

        extra[f"combo_sum_{metric}_v3"] = f + s + t
        extra[f"combo_mean_{metric}_v3"] = (f + s + t) / 3.0
        extra[f"combo_max_{metric}_v3"] = pd.concat([f, s, t], axis=1).max(axis=1)
        extra[f"combo_min_{metric}_v3"] = pd.concat([f, s, t], axis=1).min(axis=1)
        extra[f"combo_range_{metric}_v3"] = extra[f"combo_max_{metric}_v3"] - extra[f"combo_min_{metric}_v3"]

        extra[f"combo_diff_first_second_{metric}_v3"] = f - s
        extra[f"combo_diff_first_third_{metric}_v3"] = f - t
        extra[f"combo_diff_second_third_{metric}_v3"] = s - t

        extra[f"combo_absdiff_first_second_{metric}_v3"] = (f - s).abs()
        extra[f"combo_absdiff_first_third_{metric}_v3"] = (f - t).abs()
        extra[f"combo_absdiff_second_third_{metric}_v3"] = (s - t).abs()

        combo_mean = extra[f"combo_mean_{metric}_v3"]
        extra[f"combo_vs_race_{metric}_diff_v3"] = combo_mean - race_mean[metric]

        denom = race_std_map[metric].replace(0, np.nan)
        extra[f"combo_vs_race_{metric}_z_v3"] = (
            (combo_mean - race_mean[metric]) / denom
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        combo_sum = extra[f"combo_sum_{metric}_v3"]
        total_sum = race_mean[metric] * 6.0
        other_mean = (total_sum - combo_sum) / 3.0
        extra[f"combo_vs_others_{metric}_diff_v3"] = combo_mean - other_mean

    lane_triplet = pd.concat([combo_first, combo_second, combo_third], axis=1)

    extra["combo_lane_sum_v3"] = combo_first + combo_second + combo_third
    extra["combo_lane_range_v3"] = lane_triplet.max(axis=1) - lane_triplet.min(axis=1)
    extra["combo_lane_std_v3"] = lane_triplet.std(axis=1)
    extra["combo_lane_mean_v3"] = (combo_first + combo_second + combo_third) / 3.0

    extra["combo_forward_order_flag_v3"] = ((combo_first < combo_second) & (combo_second < combo_third)).astype(int)
    extra["combo_reverse_order_flag_v3"] = ((combo_first > combo_second) & (combo_second > combo_third)).astype(int)
    extra["combo_first_inner2_flag_v3"] = combo_first.isin([1, 2]).astype(int)
    extra["combo_second_inner3_flag_v3"] = combo_second.isin([1, 2, 3]).astype(int)
    extra["combo_third_outer3_flag_v3"] = combo_third.isin([4, 5, 6]).astype(int)

    extra["combo_exhibit_adv_v3"] = race_mean["exhibit"] - extra["combo_mean_exhibit_v3"]
    extra["combo_st_adv_v3"] = race_mean["st"] - extra["combo_mean_st_v3"]

    extra["combo_first_push_score_v3"] = (
        picked["first"]["motor"] * 0.20
        + picked["first"]["boat"] * 0.10
        + (race_mean["exhibit"] - picked["first"]["exhibit"]) * 10.0 * 0.30
        + (race_mean["st"] - picked["first"]["st"]) * 10.0 * 0.30
        + (6.0 - picked["first"]["course"]) * 0.10
    )

    extra["combo_follow_score_v3"] = (
        (picked["second"]["motor"] + picked["third"]["motor"]) * 0.15
        + (picked["second"]["boat"] + picked["third"]["boat"]) * 0.10
        + ((race_mean["exhibit"] - picked["second"]["exhibit"]) + (race_mean["exhibit"] - picked["third"]["exhibit"])) * 10.0 * 0.25
        + ((race_mean["st"] - picked["second"]["st"]) + (race_mean["st"] - picked["third"]["st"])) * 10.0 * 0.25
        + ((6.0 - picked["second"]["course"]) + (6.0 - picked["third"]["course"])) * 0.15
    )

    extra["combo_balance_score_v3"] = (
        extra["combo_absdiff_first_second_exhibit_v3"]
        + extra["combo_absdiff_first_third_exhibit_v3"]
        + extra["combo_absdiff_second_third_exhibit_v3"]
        + extra["combo_absdiff_first_second_st_v3"]
        + extra["combo_absdiff_first_third_st_v3"]
        + extra["combo_absdiff_second_third_st_v3"]
    )

    extra["wind_st_pressure_v3"] = ensure_col(work_df, "wind_speed_mps", 0.0) * extra["combo_mean_st_v3"]
    extra["wave_exhibit_pressure_v3"] = ensure_col(work_df, "wave_cm", 0.0) * extra["combo_mean_exhibit_v3"]

    extra_df = pd.DataFrame(extra, index=out.index)

    for c in extra_df.columns:
        extra_df[c] = pd.to_numeric(extra_df[c], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out = pd.concat([out, extra_df], axis=1)
    return out.copy()


def is_leak_feature(col: str) -> bool:
    if col in LEAK_EXACT_COLS:
        return True

    c = col.lower()

    if any(k in c for k in LEAK_KEYWORDS):
        return True

    if c.startswith("y_class"):
        return True

    if "finish" in c:
        return True

    return False


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    for c in df.columns:
        if is_leak_feature(c):
            continue

        if c.startswith("lane") or c.startswith("combo_") or c.startswith("race_") or c.startswith("inside_") or c.startswith("outside_") or c.startswith("inside_outside_"):
            cols.append(c)
            continue

        if c in {"wave_cm", "wind_speed_mps", "weather_code", "wind_dir_code", "venue_code"}:
            cols.append(c)
            continue

        if c.startswith("wind_") or c.startswith("wave_") or c.startswith("weather_"):
            cols.append(c)
            continue

    cols = list(dict.fromkeys(cols))
    return cols


def sanitize_x(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    x = df[feature_cols].copy()

    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    x = x.replace([np.inf, -np.inf], 0.0)
    return x
