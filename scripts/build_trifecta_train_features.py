from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Dict, List, Any

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


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        s = str(v).strip()
        if s == "":
            return default
        s = s.upper()
        s = s.replace("F.", "0.")
        s = s.replace("L.", "0.")
        if s.startswith("."):
            s = "0" + s
        if s.endswith("M"):
            s = s[:-1].strip()
        if s.endswith("CM"):
            s = s[:-2].strip()
        if s.endswith("℃"):
            s = s[:-1].strip()
        x = float(s)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        s = str(v).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _safe_str(v, default: str = "") -> str:
    if pd.isna(v):
        return default
    s = str(v).strip()
    return s if s else default


def _encode_grade(v) -> int:
    return GRADE_CODE.get(_safe_str(v).upper(), 0)


def _encode_weather(v) -> int:
    return WEATHER_CODE.get(_safe_str(v), 0)


def _encode_wind_dir(v) -> int:
    return WIND_DIR_CODE.get(_safe_str(v), 0)


def _encode_venue(v) -> int:
    s = _safe_str(v)
    if s in VENUE_CODE:
        return VENUE_CODE[s]
    try:
        return int(float(s))
    except Exception:
        return 0


def _combo_to_class_id(combo: str) -> int:
    try:
        return ALL_COMBOS.index(str(combo))
    except ValueError:
        return -1


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


def _first_existing(row: pd.Series, candidates: List[str], default=None):
    for c in candidates:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip() != "":
                return v
    return default


def _course_key(lane: int, suffix: str) -> str:
    return f"racer_course{lane}_{suffix}"


def _extract_lane_row(row: pd.Series, lane: int) -> Dict[str, float]:
    prefix = f"lane{lane}_"

    def pick_float(names: List[str], default=0.0) -> float:
        for n in names:
            col = prefix + n
            if col in row.index:
                return _safe_float(row[col], default)
        return default

    def pick_str(names: List[str], default="") -> str:
        for n in names:
            col = prefix + n
            if col in row.index:
                return _safe_str(row[col], default)
        return default

    info: Dict[str, float] = {
        "lane": float(lane),
        "course": pick_float(["course", "entry_course", "course_no"], float(lane)),
        "win_rate": pick_float(["win_rate", "nation_win_rate"], 0.0),
        "place_rate": pick_float(["place_rate", "quinella_rate", "nation_place_rate"], 0.0),
        "exhibit": pick_float(["exhibit", "exhibit_time", "tenji_time"], 0.0),
        "st": pick_float(["start_timing", "st", "avg_st"], 0.0),
        "avg_st": pick_float(["avg_st"], 0.0),
        "grade": float(_encode_grade(pick_str(["grade"], ""))),
        "motor": pick_float(["motor", "motor_no"], 0.0),
        "boat": pick_float(["boat", "boat_no"], 0.0),
        "age": pick_float(["age"], 0.0),
        "weight": pick_float(["weight"], 0.0),
        "ability_index": pick_float(["ability_index"], 0.0),
        "prev_ability_index": pick_float(["prev_ability_index"], 0.0),
        "self_course_entry_count": pick_float([_course_key(lane, "entry_count")], 0.0),
        "self_course_place_rate": pick_float([_course_key(lane, "place_rate")], 0.0),
        "self_course_avg_st": pick_float([_course_key(lane, "avg_st")], 0.0),
    }

    for c in range(1, 7):
        info[f"course{c}_entry_count"] = pick_float([_course_key(c, "entry_count")], 0.0)
        info[f"course{c}_place_rate"] = pick_float([_course_key(c, "place_rate")], 0.0)
        info[f"course{c}_avg_st"] = pick_float([_course_key(c, "avg_st")], 0.0)

    return info


def _add_triplet_features(
    out: Dict[str, Any],
    prefix: str,
    boat: Dict[str, float],
    lane: int,
    race_feat: Dict[str, float],
) -> None:
    base_keys = [
        "course", "win_rate", "place_rate", "exhibit", "st", "avg_st",
        "grade", "motor", "boat", "age", "weight",
        "ability_index", "prev_ability_index",
        "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
        "course1_entry_count", "course1_place_rate", "course1_avg_st",
        "course2_entry_count", "course2_place_rate", "course2_avg_st",
        "course3_entry_count", "course3_place_rate", "course3_avg_st",
        "course4_entry_count", "course4_place_rate", "course4_avg_st",
        "course5_entry_count", "course5_place_rate", "course5_avg_st",
        "course6_entry_count", "course6_place_rate", "course6_avg_st",
    ]
    for key in base_keys:
        out[f"{prefix}_{key}"] = boat[key]

    out[f"{prefix}_win_rate_rank"] = race_feat[f"rank_win_rate_lane{lane}"]
    out[f"{prefix}_place_rate_rank"] = race_feat[f"rank_place_rate_lane{lane}"]
    out[f"{prefix}_exhibit_rank"] = race_feat[f"rank_exhibit_lane{lane}"]
    out[f"{prefix}_st_rank"] = race_feat[f"rank_st_lane{lane}"]
    out[f"{prefix}_grade_rank"] = race_feat[f"rank_grade_lane{lane}"]
    out[f"{prefix}_ability_rank"] = race_feat[f"rank_ability_lane{lane}"]
    out[f"{prefix}_prev_ability_rank"] = race_feat[f"rank_prev_ability_lane{lane}"]
    out[f"{prefix}_self_course_place_rate_rank"] = race_feat[f"rank_self_course_place_rate_lane{lane}"]
    out[f"{prefix}_self_course_avg_st_rank"] = race_feat[f"rank_self_course_avg_st_lane{lane}"]

    rel_keys = [
        "win_rate", "place_rate", "exhibit", "st", "avg_st",
        "grade", "course", "age", "weight",
        "ability_index", "prev_ability_index",
        "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
    ]
    for key in rel_keys:
        out[f"{prefix}_{key}_rel"] = race_feat[f"rel_{key}_lane{lane}"]


def _add_diff_features(
    out: Dict[str, Any],
    prefix: str,
    x: Dict[str, float],
    y: Dict[str, float],
) -> None:
    diff_keys = [
        "win_rate", "place_rate", "exhibit", "st", "avg_st",
        "grade", "course", "age", "weight",
        "ability_index", "prev_ability_index",
        "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
    ]
    for k in diff_keys:
        out[f"{prefix}_{k}_diff"] = x[k] - y[k]


def build_features_for_one_race(row: pd.Series) -> List[Dict[str, Any]]:
    lane_info = {lane: _extract_lane_row(row, lane) for lane in LANES}

    venue = _first_existing(row, ["venue", "jyo", "stadium"], "")
    race_no = _safe_int(_first_existing(row, ["race_no", "rno"], 0), 0)
    date = _safe_str(_first_existing(row, ["date", "hd"], ""), "")
    y_combo = _safe_str(_first_existing(row, ["y_combo", "result_combo", "combo_result"], ""), "")

    weather = _first_existing(row, ["weather"], "")
    wind_dir = _first_existing(row, ["wind_dir", "wind_direction"], "")
    wind_speed_mps = _safe_float(_first_existing(row, ["wind_speed_mps", "wind_speed"], 0.0), 0.0)
    wave_cm = _safe_float(_first_existing(row, ["wave_cm", "wave"], 0.0), 0.0)

    win_rates = {i: lane_info[i]["win_rate"] for i in LANES}
    place_rates = {i: lane_info[i]["place_rate"] for i in LANES}
    exhibits = {i: lane_info[i]["exhibit"] for i in LANES}
    sts = {i: lane_info[i]["st"] for i in LANES}
    grades = {i: lane_info[i]["grade"] for i in LANES}
    abilities = {i: lane_info[i]["ability_index"] for i in LANES}
    prev_abilities = {i: lane_info[i]["prev_ability_index"] for i in LANES}
    self_course_rates = {i: lane_info[i]["self_course_place_rate"] for i in LANES}
    self_course_sts = {i: lane_info[i]["self_course_avg_st"] for i in LANES}

    race_feat: Dict[str, float] = {
        "weather_code": float(_encode_weather(weather)),
        "wind_dir_code": float(_encode_wind_dir(wind_dir)),
        "wind_speed_mps": wind_speed_mps,
        "wave_cm": wave_cm,
        "venue_code": float(_encode_venue(venue)),
    }

    summary_keys = [
        "win_rate", "place_rate", "exhibit", "st", "avg_st",
        "grade", "course", "age", "weight",
        "ability_index", "prev_ability_index",
        "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
    ]
    for key in summary_keys:
        vals = [lane_info[i][key] for i in LANES]
        race_feat[f"race_{key}_mean"] = _mean(vals)
        race_feat[f"race_{key}_std"] = _std(vals)

    race_feat["inside_exhibit_mean"] = _mean([lane_info[i]["exhibit"] for i in [1, 2, 3]])
    race_feat["outside_exhibit_mean"] = _mean([lane_info[i]["exhibit"] for i in [4, 5, 6]])
    race_feat["inside_st_mean"] = _mean([lane_info[i]["st"] for i in [1, 2, 3]])
    race_feat["outside_st_mean"] = _mean([lane_info[i]["st"] for i in [4, 5, 6]])
    race_feat["inside_ability_mean"] = _mean([lane_info[i]["ability_index"] for i in [1, 2, 3]])
    race_feat["outside_ability_mean"] = _mean([lane_info[i]["ability_index"] for i in [4, 5, 6]])
    race_feat["course_move_abs_sum"] = sum(abs(lane_info[i]["course"] - i) for i in LANES)
    race_feat["course_move_abs_mean"] = race_feat["course_move_abs_sum"] / 6.0

    win_rank = _rank_desc(win_rates)
    place_rank = _rank_desc(place_rates)
    exhibit_rank = _rank_asc(exhibits)
    st_rank = _rank_asc(sts)
    grade_rank = _rank_desc(grades)
    ability_rank = _rank_desc(abilities)
    prev_ability_rank = _rank_desc(prev_abilities)
    self_course_rate_rank = _rank_desc(self_course_rates)
    self_course_st_rank = _rank_asc(self_course_sts)

    for i in LANES:
        race_feat[f"rank_win_rate_lane{i}"] = float(win_rank[i])
        race_feat[f"rank_place_rate_lane{i}"] = float(place_rank[i])
        race_feat[f"rank_exhibit_lane{i}"] = float(exhibit_rank[i])
        race_feat[f"rank_st_lane{i}"] = float(st_rank[i])
        race_feat[f"rank_grade_lane{i}"] = float(grade_rank[i])
        race_feat[f"rank_ability_lane{i}"] = float(ability_rank[i])
        race_feat[f"rank_prev_ability_lane{i}"] = float(prev_ability_rank[i])
        race_feat[f"rank_self_course_place_rate_lane{i}"] = float(self_course_rate_rank[i])
        race_feat[f"rank_self_course_avg_st_lane{i}"] = float(self_course_st_rank[i])

        rel_keys = [
            "win_rate", "place_rate", "exhibit", "st", "avg_st",
            "grade", "course", "age", "weight",
            "ability_index", "prev_ability_index",
            "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
        ]
        for key in rel_keys:
            race_feat[f"rel_{key}_lane{i}"] = lane_info[i][key] - race_feat[f"race_{key}_mean"]

    out_rows: List[Dict[str, Any]] = []

    for combo in ALL_COMBOS:
        a, b, c = map(int, combo.split("-"))
        f = lane_info[a]
        s = lane_info[b]
        t = lane_info[c]

        out: Dict[str, Any] = {
            "race_key": f"{date}_{venue}_{race_no}",
            "date": date,
            "venue": venue,
            "race_no": race_no,
            "combo": combo,
            "y_combo": y_combo,
            "y_class": _combo_to_class_id(y_combo),
            "is_hit": int(combo == y_combo),
        }
        out.update(race_feat)

        _add_triplet_features(out, "first", f, a, race_feat)
        _add_triplet_features(out, "second", s, b, race_feat)
        _add_triplet_features(out, "third", t, c, race_feat)

        top3_keys = [
            "win_rate", "place_rate", "exhibit", "st", "avg_st", "grade", "course",
            "age", "weight", "ability_index", "prev_ability_index",
            "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
        ]
        for key in top3_keys:
            vals = [f[key], s[key], t[key]]
            out[f"top3_{key}_sum"] = sum(vals)
            out[f"top3_{key}_mean"] = _mean(vals)
            out[f"top3_{key}_std"] = _std(vals)

        _add_diff_features(out, "f_s", f, s)
        _add_diff_features(out, "f_t", f, t)
        _add_diff_features(out, "s_t", s, t)

        field_vs_keys = [
            "win_rate", "place_rate", "exhibit", "st", "avg_st",
            "grade", "ability_index", "prev_ability_index",
            "self_course_place_rate", "self_course_avg_st",
        ]
        for key in field_vs_keys:
            field_avg = race_feat[f"race_{key}_mean"]
            out[f"f_{key}_vs_field"] = f[key] - field_avg
            out[f"s_{key}_vs_field"] = s[key] - field_avg
            out[f"t_{key}_vs_field"] = t[key] - field_avg

        out["f_stronger_than_s"] = int(f["win_rate"] > s["win_rate"])
        out["f_stronger_than_t"] = int(f["win_rate"] > t["win_rate"])
        out["s_stronger_than_t"] = int(s["win_rate"] > t["win_rate"])

        out["f_higher_ability_than_s"] = int(f["ability_index"] > s["ability_index"])
        out["f_higher_ability_than_t"] = int(f["ability_index"] > t["ability_index"])
        out["s_higher_ability_than_t"] = int(s["ability_index"] > t["ability_index"])

        out["f_better_self_course_than_s"] = int(f["self_course_place_rate"] > s["self_course_place_rate"])
        out["f_better_self_course_than_t"] = int(f["self_course_place_rate"] > t["self_course_place_rate"])
        out["s_better_self_course_than_t"] = int(s["self_course_place_rate"] > t["self_course_place_rate"])

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

        out["first_lane_fit_place_rate"] = f[f"course{a}_place_rate"]
        out["second_lane_fit_place_rate"] = s[f"course{b}_place_rate"]
        out["third_lane_fit_place_rate"] = t[f"course{c}_place_rate"]

        out["first_lane_fit_avg_st"] = f[f"course{a}_avg_st"]
        out["second_lane_fit_avg_st"] = s[f"course{b}_avg_st"]
        out["third_lane_fit_avg_st"] = t[f"course{c}_avg_st"]

        out["top3_lane_fit_place_rate_mean"] = _mean([
            out["first_lane_fit_place_rate"],
            out["second_lane_fit_place_rate"],
            out["third_lane_fit_place_rate"],
        ])
        out["top3_lane_fit_avg_st_mean"] = _mean([
            out["first_lane_fit_avg_st"],
            out["second_lane_fit_avg_st"],
            out["third_lane_fit_avg_st"],
        ])

        out["f_s_lane_fit_place_rate_diff"] = out["first_lane_fit_place_rate"] - out["second_lane_fit_place_rate"]
        out["f_t_lane_fit_place_rate_diff"] = out["first_lane_fit_place_rate"] - out["third_lane_fit_place_rate"]
        out["s_t_lane_fit_place_rate_diff"] = out["second_lane_fit_place_rate"] - out["third_lane_fit_place_rate"]

        out["f_s_lane_fit_avg_st_diff"] = out["first_lane_fit_avg_st"] - out["second_lane_fit_avg_st"]
        out["f_t_lane_fit_avg_st_diff"] = out["first_lane_fit_avg_st"] - out["third_lane_fit_avg_st"]
        out["s_t_lane_fit_avg_st_diff"] = out["second_lane_fit_avg_st"] - out["third_lane_fit_avg_st"]

        out_rows.append(out)

    return out_rows


def main() -> None:
    print("===== BUILD TRIFECTA TRAIN FEATURES START =====")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"input shape: {df.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample_rows = build_features_for_one_race(df.iloc[0])
    fieldnames = list(sample_rows[0].keys())

    total = len(df)
    written = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            try:
                feat_rows = build_features_for_one_race(row)
                writer.writerows(feat_rows)
                written += len(feat_rows)
            except Exception as e:
                print(f"[WARN] skipped row {idx}/{total}: {e}")

            if idx % 100 == 0:
                print(f"processed: {idx}/{total} written_rows={written}")

    print(f"saved: {OUTPUT_PATH}")
    print(f"written_rows: {written}")
    print("===== BUILD TRIFECTA TRAIN FEATURES END =====")


if __name__ == "__main__":
    main()
