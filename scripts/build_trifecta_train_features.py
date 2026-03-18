from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "datasets" / "startk_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"

LANES = [1, 2, 3, 4, 5, 6]
ALL_COMBOS = [f"{a}-{b}-{c}" for a, b, c in itertools.permutations(LANES, 3)]

VENUE_CODE = {
    "桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5,
    "浜名湖": 6, "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10,
    "びわこ": 11, "住之江": 12, "尼崎": 13, "鳴門": 14, "丸亀": 15,
    "児島": 16, "宮島": 17, "徳山": 18, "下関": 19, "若松": 20,
    "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24,
}

GRADE_CODE = {
    "A1": 4,
    "A2": 3,
    "B1": 2,
    "B2": 1,
}

WEATHER_CODE = {
    "晴": 1,
    "晴れ": 1,
    "曇": 2,
    "曇り": 2,
    "くもり": 2,
    "雨": 3,
    "雪": 4,
}

WIND_DIR_CODE = {
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    m = _mean(vals)
    return float((sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5)


def _rank_desc(values: Dict[int, float]) -> Dict[int, int]:
    ordered = sorted(values.items(), key=lambda x: (-x[1], x[0]))
    return {lane: rank + 1 for rank, (lane, _) in enumerate(ordered)}


def _rank_asc(values: Dict[int, float]) -> Dict[int, int]:
    ordered = sorted(values.items(), key=lambda x: (x[1], x[0]))
    return {lane: rank + 1 for rank, (lane, _) in enumerate(ordered)}


def _encode_grade(v: Any) -> int:
    return GRADE_CODE.get(_safe_str(v), 0)


def _encode_weather(v: Any) -> int:
    return WEATHER_CODE.get(_safe_str(v), 0)


def _encode_wind_dir(v: Any) -> int:
    return WIND_DIR_CODE.get(_safe_str(v), 0)


def _encode_venue(v: Any) -> int:
    return VENUE_CODE.get(_safe_str(v), 0)


def _combo_to_class_id(combo: str) -> int:
    return ALL_COMBOS.index(combo)


def _pick_lane_value(
    row: pd.Series,
    lane: int,
    candidates: List[str],
    kind: str = "float",
) -> float | int | str:
    for suffix in candidates:
        col = f"lane{lane}_{suffix}"
        if col in row.index:
            if kind == "float":
                return _safe_float(row[col], 0.0)
            if kind == "int":
                return _safe_int(row[col], 0)
            if kind == "str":
                return _safe_str(row[col], "")
    if kind == "float":
        return 0.0
    if kind == "int":
        return 0
    return ""


def _build_lane_dict(row: pd.Series) -> Dict[int, Dict[str, float]]:
    lane_info: Dict[int, Dict[str, float]] = {}

    for lane in LANES:
        lane_info[lane] = {
            "lane": float(lane),
            "win_rate": _pick_lane_value(row, lane, ["win_rate", "nation_win_rate"], "float"),
            "place_rate": _pick_lane_value(row, lane, ["place_rate", "nation_place_rate", "two_rate"], "float"),
            "exhibit": _pick_lane_value(row, lane, ["exhibit"], "float"),
            "st": _pick_lane_value(row, lane, ["st", "avg_st"], "float"),
            "course": _pick_lane_value(row, lane, ["course"], "float"),
            "grade": float(_encode_grade(_pick_lane_value(row, lane, ["grade"], "str"))),
            "motor": _pick_lane_value(row, lane, ["motor"], "float"),
            "boat": _pick_lane_value(row, lane, ["boat"], "float"),
            "age": _pick_lane_value(row, lane, ["age"], "float"),
            "weight": _pick_lane_value(row, lane, ["weight"], "float"),
        }

        if lane_info[lane]["course"] <= 0:
            lane_info[lane]["course"] = float(lane)

    return lane_info


def _build_race_features(row: pd.Series, boats: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    win_rates = {i: boats[i]["win_rate"] for i in LANES}
    place_rates = {i: boats[i]["place_rate"] for i in LANES}
    exhibits = {i: boats[i]["exhibit"] for i in LANES}
    sts = {i: boats[i]["st"] for i in LANES}
    grades = {i: boats[i]["grade"] for i in LANES}

    win_rank = _rank_desc(win_rates)
    place_rank = _rank_desc(place_rates)
    exhibit_rank = _rank_asc(exhibits)
    st_rank = _rank_asc(sts)
    grade_rank = _rank_desc(grades)

    feat: Dict[str, float] = {}

    feat["weather_code"] = float(_encode_weather(row.get("weather")))
    feat["wind_dir_code"] = float(_encode_wind_dir(row.get("wind_dir")))
    feat["wind_speed_mps"] = _safe_float(row.get("wind_speed_mps"))
    feat["wave_cm"] = _safe_float(row.get("wave_cm"))
    feat["venue_code"] = float(_encode_venue(row.get("venue")))

    for key in ["win_rate", "place_rate", "exhibit", "st", "grade", "course"]:
        vals = [boats[i][key] for i in LANES]
        feat[f"race_{key}_mean"] = _mean(vals)
        feat[f"race_{key}_std"] = _std(vals)

    feat["inside_exhibit_mean"] = _mean([boats[i]["exhibit"] for i in [1, 2, 3]])
    feat["outside_exhibit_mean"] = _mean([boats[i]["exhibit"] for i in [4, 5, 6]])
    feat["inside_st_mean"] = _mean([boats[i]["st"] for i in [1, 2, 3]])
    feat["outside_st_mean"] = _mean([boats[i]["st"] for i in [4, 5, 6]])

    feat["course_move_abs_sum"] = sum(abs(boats[i]["course"] - i) for i in LANES)
    feat["course_move_abs_mean"] = feat["course_move_abs_sum"] / 6.0

    for i in LANES:
        feat[f"rank_win_rate_lane{i}"] = float(win_rank[i])
        feat[f"rank_place_rate_lane{i}"] = float(place_rank[i])
        feat[f"rank_exhibit_lane{i}"] = float(exhibit_rank[i])
        feat[f"rank_st_lane{i}"] = float(st_rank[i])
        feat[f"rank_grade_lane{i}"] = float(grade_rank[i])

        feat[f"rel_win_rate_lane{i}"] = boats[i]["win_rate"] - feat["race_win_rate_mean"]
        feat[f"rel_place_rate_lane{i}"] = boats[i]["place_rate"] - feat["race_place_rate_mean"]
        feat[f"rel_exhibit_lane{i}"] = boats[i]["exhibit"] - feat["race_exhibit_mean"]
        feat[f"rel_st_lane{i}"] = boats[i]["st"] - feat["race_st_mean"]
        feat[f"rel_grade_lane{i}"] = boats[i]["grade"] - feat["race_grade_mean"]
        feat[f"rel_course_lane{i}"] = boats[i]["course"] - feat["race_course_mean"]

    return feat


def _add_triplet_features(
    out: Dict[str, Any],
    prefix: str,
    boat: Dict[str, float],
    lane: int,
    race_feat: Dict[str, float],
) -> None:
    out[f"{prefix}_course"] = boat["course"]
    out[f"{prefix}_win_rate"] = boat["win_rate"]
    out[f"{prefix}_place_rate"] = boat["place_rate"]
    out[f"{prefix}_exhibit"] = boat["exhibit"]
    out[f"{prefix}_st"] = boat["st"]
    out[f"{prefix}_grade"] = boat["grade"]
    out[f"{prefix}_motor"] = boat["motor"]
    out[f"{prefix}_boat"] = boat["boat"]
    out[f"{prefix}_age"] = boat["age"]
    out[f"{prefix}_weight"] = boat["weight"]

    out[f"{prefix}_win_rate_rank"] = race_feat[f"rank_win_rate_lane{lane}"]
    out[f"{prefix}_place_rate_rank"] = race_feat[f"rank_place_rate_lane{lane}"]
    out[f"{prefix}_exhibit_rank"] = race_feat[f"rank_exhibit_lane{lane}"]
    out[f"{prefix}_st_rank"] = race_feat[f"rank_st_lane{lane}"]
    out[f"{prefix}_grade_rank"] = race_feat[f"rank_grade_lane{lane}"]

    out[f"{prefix}_win_rate_rel"] = race_feat[f"rel_win_rate_lane{lane}"]
    out[f"{prefix}_place_rate_rel"] = race_feat[f"rel_place_rate_lane{lane}"]
    out[f"{prefix}_exhibit_rel"] = race_feat[f"rel_exhibit_lane{lane}"]
    out[f"{prefix}_st_rel"] = race_feat[f"rel_st_lane{lane}"]
    out[f"{prefix}_grade_rel"] = race_feat[f"rel_grade_lane{lane}"]
    out[f"{prefix}_course_rel"] = race_feat[f"rel_course_lane{lane}"]


def _build_one_feature_row(row: pd.Series, combo: str) -> Dict[str, Any]:
    boats = _build_lane_dict(row)
    race_feat = _build_race_features(row, boats)

    a, b, c = map(int, combo.split("-"))
    f = boats[a]
    s = boats[b]
    t = boats[c]

    out: Dict[str, Any] = {
        "race_key": _safe_str(row.get("race_key")),
        "date": _safe_str(row.get("date")),
        "venue": _safe_str(row.get("venue")),
        "race_no": _safe_int(row.get("race_no")),
        "combo": combo,
        "y_combo": _safe_str(row.get("y_combo")),
    }
    out["y_class"] = _combo_to_class_id(out["y_combo"]) if out["y_combo"] in ALL_COMBOS else -1

    out.update(race_feat)

    _add_triplet_features(out, "first", f, a, race_feat)
    _add_triplet_features(out, "second", s, b, race_feat)
    _add_triplet_features(out, "third", t, c, race_feat)

    for key in ["win_rate", "place_rate", "exhibit", "st", "grade", "course"]:
        vals = [f[key], s[key], t[key]]
        out[f"top3_{key}_sum"] = sum(vals)
        out[f"top3_{key}_mean"] = _mean(vals)
        out[f"top3_{key}_std"] = _std(vals)

    def _add_diff(prefix: str, x: Dict[str, float], y: Dict[str, float]) -> None:
        for k in ["win_rate", "place_rate", "exhibit", "st", "grade", "course"]:
            out[f"{prefix}_{k}_diff"] = x[k] - y[k]

    _add_diff("f_s", f, s)
    _add_diff("f_t", f, t)
    _add_diff("s_t", s, t)

    for key in ["win_rate", "place_rate", "exhibit", "st", "grade"]:
        field_avg = race_feat[f"race_{key}_mean"]
        out[f"f_{key}_vs_field"] = f[key] - field_avg
        out[f"s_{key}_vs_field"] = s[key] - field_avg
        out[f"t_{key}_vs_field"] = t[key] - field_avg

    out["f_stronger_than_s"] = int(f["win_rate"] > s["win_rate"])
    out["f_stronger_than_t"] = int(f["win_rate"] > t["win_rate"])
    out["s_stronger_than_t"] = int(s["win_rate"] > t["win_rate"])

    out["f_fastest_st"] = int(f["st"] < s["st"] and f["st"] < t["st"])
    out["s_fastest_st"] = int(s["st"] < f["st"] and s["st"] < t["st"])
    out["t_fastest_st"] = int(t["st"] < f["st"] and t["st"] < s["st"])

    out["f_best_exhibit"] = int(f["exhibit"] < s["exhibit"] and f["exhibit"] < t["exhibit"])
    out["s_best_exhibit"] = int(s["exhibit"] < f["exhibit"] and s["exhibit"] < t["exhibit"])
    out["t_best_exhibit"] = int(t["exhibit"] < f["exhibit"] and t["exhibit"] < s["exhibit"])

    out["course_move_abs_sum_top3"] = (
        abs(f["course"] - a) +
        abs(s["course"] - b) +
        abs(t["course"] - c)
    )

    return out


def build():
    print("===== BUILD TRIFECTA FEATURES (ANTI-LANE 120 VERSION) =====")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")

    src = pd.read_csv(INPUT_PATH, low_memory=False)

    required_cols = ["race_key", "date", "venue", "race_no", "y_combo"]
    missing = [c for c in required_cols if c not in src.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    src = src[src["y_combo"].astype(str).isin(ALL_COMBOS)].copy().reset_index(drop=True)
    if src.empty:
        raise RuntimeError("No valid y_combo rows found in startk_dataset.csv")

    sample_row = _build_one_feature_row(src.iloc[0], ALL_COMBOS[0])
    header = list(sample_row.keys())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    built_rows = 0
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for i, (_, row) in enumerate(src.iterrows(), start=1):
            for combo in ALL_COMBOS:
                writer.writerow(_build_one_feature_row(row, combo))
                built_rows += 1

            if i % 500 == 0:
                print(f"processed races: {i}/{len(src)}  built_rows={built_rows}")

    print(f"DONE: {OUTPUT_PATH} rows= {built_rows}")


if __name__ == "__main__":
    build()
