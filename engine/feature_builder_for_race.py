from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List


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


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        s = str(v).strip()
        if s == "":
            return default
        x = float(s)
        if math.isfinite(x):
            return x
        return default
    except Exception:
        return default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _encode_grade(v: Any) -> int:
    return GRADE_CODE.get(_safe_str(v).upper(), 0)


def _encode_weather(v: Any) -> int:
    return WEATHER_CODE.get(_safe_str(v), 0)


def _encode_wind_dir(v: Any) -> int:
    return WIND_DIR_CODE.get(_safe_str(v), 0)


def _encode_venue(v: Any) -> int:
    return VENUE_CODE.get(_safe_str(v), 0)


def _rank_desc(values: Dict[int, float]) -> Dict[int, int]:
    ordered = sorted(values.items(), key=lambda x: (-x[1], x[0]))
    return {lane: rank + 1 for rank, (lane, _) in enumerate(ordered)}


def _rank_asc(values: Dict[int, float]) -> Dict[int, int]:
    ordered = sorted(values.items(), key=lambda x: (x[1], x[0]))
    return {lane: rank + 1 for rank, (lane, _) in enumerate(ordered)}


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    m = _mean(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def _course_key(lane: int, suffix: str) -> str:
    return f"racer_course{lane}_{suffix}"


def build_120_features_for_race(
    *,
    date: str,
    venue: str,
    race_no: int,
    entries: List[Dict[str, Any]],
    weather: str = "",
    wind_dir: str = "",
    wind_speed_mps: float = 0.0,
    wave_cm: float = 0.0,
) -> List[Dict[str, Any]]:
    if len(entries) != 6:
        raise ValueError("entries must contain exactly 6 boats.")

    lane_info: Dict[int, Dict[str, float]] = {}

    for row in entries:
        lane = int(row["lane"])
        lane_info[lane] = {
            "lane": float(lane),
            "course": _safe_float(row.get("course", lane), float(lane)),
            "win_rate": _safe_float(row.get("win_rate", 0.0), 0.0),
            "place_rate": _safe_float(row.get("place_rate", 0.0), 0.0),
            "exhibit": _safe_float(row.get("exhibit", 0.0), 0.0),
            "st": _safe_float(
                row.get("start_timing", row.get("st", row.get("avg_st", 0.0))),
                0.0,
            ),
            "avg_st": _safe_float(row.get("avg_st", 0.0), 0.0),
            "grade": float(_encode_grade(row.get("grade", ""))),
            "motor": _safe_float(row.get("motor", 0.0), 0.0),
            "boat": _safe_float(row.get("boat", 0.0), 0.0),
            "age": _safe_float(row.get("age", 0.0), 0.0),
            "weight": _safe_float(row.get("weight", 0.0), 0.0),
            "ability_index": _safe_float(row.get("ability_index", 0.0), 0.0),
            "prev_ability_index": _safe_float(row.get("prev_ability_index", 0.0), 0.0),

            # 自コース適性
            "self_course_entry_count": _safe_float(row.get(_course_key(lane, "entry_count"), 0.0), 0.0),
            "self_course_place_rate": _safe_float(row.get(_course_key(lane, "place_rate"), 0.0), 0.0),
            "self_course_avg_st": _safe_float(row.get(_course_key(lane, "avg_st"), 0.0), 0.0),

            # 1〜6コース全部持つ
            "course1_entry_count": _safe_float(row.get("racer_course1_entry_count", 0.0), 0.0),
            "course1_place_rate": _safe_float(row.get("racer_course1_place_rate", 0.0), 0.0),
            "course1_avg_st": _safe_float(row.get("racer_course1_avg_st", 0.0), 0.0),

            "course2_entry_count": _safe_float(row.get("racer_course2_entry_count", 0.0), 0.0),
            "course2_place_rate": _safe_float(row.get("racer_course2_place_rate", 0.0), 0.0),
            "course2_avg_st": _safe_float(row.get("racer_course2_avg_st", 0.0), 0.0),

            "course3_entry_count": _safe_float(row.get("racer_course3_entry_count", 0.0), 0.0),
            "course3_place_rate": _safe_float(row.get("racer_course3_place_rate", 0.0), 0.0),
            "course3_avg_st": _safe_float(row.get("racer_course3_avg_st", 0.0), 0.0),

            "course4_entry_count": _safe_float(row.get("racer_course4_entry_count", 0.0), 0.0),
            "course4_place_rate": _safe_float(row.get("racer_course4_place_rate", 0.0), 0.0),
            "course4_avg_st": _safe_float(row.get("racer_course4_avg_st", 0.0), 0.0),

            "course5_entry_count": _safe_float(row.get("racer_course5_entry_count", 0.0), 0.0),
            "course5_place_rate": _safe_float(row.get("racer_course5_place_rate", 0.0), 0.0),
            "course5_avg_st": _safe_float(row.get("racer_course5_avg_st", 0.0), 0.0),

            "course6_entry_count": _safe_float(row.get("racer_course6_entry_count", 0.0), 0.0),
            "course6_place_rate": _safe_float(row.get("racer_course6_place_rate", 0.0), 0.0),
            "course6_avg_st": _safe_float(row.get("racer_course6_avg_st", 0.0), 0.0),
        }

    win_rates = {i: lane_info[i]["win_rate"] for i in LANES}
    place_rates = {i: lane_info[i]["place_rate"] for i in LANES}
    exhibits = {i: lane_info[i]["exhibit"] for i in LANES}
    sts = {i: lane_info[i]["st"] for i in LANES}
    grades = {i: lane_info[i]["grade"] for i in LANES}
    abilities = {i: lane_info[i]["ability_index"] for i in LANES}
    prev_abilities = {i: lane_info[i]["prev_ability_index"] for i in LANES}
    self_course_rates = {i: lane_info[i]["self_course_place_rate"] for i in LANES}
    self_course_sts = {i: lane_info[i]["self_course_avg_st"] for i in LANES}

    win_rank = _rank_desc(win_rates)
    place_rank = _rank_desc(place_rates)
    exhibit_rank = _rank_asc(exhibits)
    st_rank = _rank_asc(sts)
    grade_rank = _rank_desc(grades)
    ability_rank = _rank_desc(abilities)
    prev_ability_rank = _rank_desc(prev_abilities)
    self_course_rate_rank = _rank_desc(self_course_rates)
    self_course_st_rank = _rank_asc(self_course_sts)

    race_feat: Dict[str, float] = {
        "weather_code": float(_encode_weather(weather)),
        "wind_dir_code": float(_encode_wind_dir(wind_dir)),
        "wind_speed_mps": _safe_float(wind_speed_mps, 0.0),
        "wave_cm": _safe_float(wave_cm, 0.0),
        "venue_code": float(_encode_venue(venue)),
    }

    race_summary_keys = [
        "win_rate", "place_rate", "exhibit", "st", "avg_st",
        "grade", "course", "age", "weight",
        "ability_index", "prev_ability_index",
        "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
    ]
    for key in race_summary_keys:
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

    rows: List[Dict[str, Any]] = []

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
        }

        out.update(race_feat)

        def add_triplet(prefix: str, boat: Dict[str, float], lane: int) -> None:
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

        add_triplet("first", f, a)
        add_triplet("second", s, b)
        add_triplet("third", t, c)

        summary_keys = [
            "win_rate", "place_rate", "exhibit", "st", "avg_st", "grade", "course",
            "age", "weight", "ability_index", "prev_ability_index",
            "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
        ]
        for key in summary_keys:
            vals = [f[key], s[key], t[key]]
            out[f"top3_{key}_sum"] = sum(vals)
            out[f"top3_{key}_mean"] = _mean(vals)
            out[f"top3_{key}_std"] = _std(vals)

        def add_diff(prefix: str, x: Dict[str, float], y: Dict[str, float]) -> None:
            diff_keys = [
                "win_rate", "place_rate", "exhibit", "st", "avg_st", "grade", "course",
                "age", "weight", "ability_index", "prev_ability_index",
                "self_course_entry_count", "self_course_place_rate", "self_course_avg_st",
            ]
            for k in diff_keys:
                out[f"{prefix}_{k}_diff"] = x[k] - y[k]

        add_diff("f_s", f, s)
        add_diff("f_t", f, t)
        add_diff("s_t", s, t)

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

        # 進入コースではなく「その艇が今回の枠でどれくらい得意か」
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

        rows.append(out)

    return rows
