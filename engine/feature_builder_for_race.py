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
        x = float(v)
        if math.isfinite(x):
            return x
        return default
    except Exception:
        return default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v).strip()


def _encode_grade(v: Any) -> int:
    return GRADE_CODE.get(_safe_str(v), 0)


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
    """
    entries は6艇分の辞書配列を想定。
    各dictに最低限ほしいキー例:
      lane, racer_no, motor, boat, exhibit, st,
      win_rate, place_rate, age, weight, grade, course
    """

    if len(entries) != 6:
        raise ValueError("entries must contain exactly 6 boats.")

    lane_info: Dict[int, Dict[str, float]] = {}

    for row in entries:
        lane = int(row["lane"])
        lane_info[lane] = {
            "lane": float(lane),
            "racer_no": _safe_float(row.get("racer_no", 0)),
            "motor": _safe_float(row.get("motor", 0)),
            "boat": _safe_float(row.get("boat", 0)),
            "exhibit": _safe_float(row.get("exhibit", 0)),
            "st": _safe_float(row.get("st", 0)),
            "course": _safe_float(row.get("course", lane)),
            "win_rate": _safe_float(row.get("win_rate", 0)),
            "place_rate": _safe_float(row.get("place_rate", 0)),
            "age": _safe_float(row.get("age", 0)),
            "weight": _safe_float(row.get("weight", 0)),
            "grade_code": float(_encode_grade(row.get("grade", ""))),
        }

    win_rates = {lane: lane_info[lane]["win_rate"] for lane in LANES}
    place_rates = {lane: lane_info[lane]["place_rate"] for lane in LANES}
    exhibits = {lane: lane_info[lane]["exhibit"] for lane in LANES}
    sts = {lane: lane_info[lane]["st"] for lane in LANES}
    grade_codes = {lane: lane_info[lane]["grade_code"] for lane in LANES}

    win_rate_rank = _rank_desc(win_rates)
    place_rate_rank = _rank_desc(place_rates)
    exhibit_rank = _rank_asc(exhibits)
    st_rank = _rank_asc(sts)
    grade_rank = _rank_desc(grade_codes)

    lane1 = lane_info[1]

    race_feat: Dict[str, float] = {
        "weather_code": float(_encode_weather(weather)),
        "wind_dir_code": float(_encode_wind_dir(wind_dir)),
        "wind_speed_mps": _safe_float(wind_speed_mps),
        "wave_cm": _safe_float(wave_cm),
        "venue_code": float(_encode_venue(venue)),
    }

    all_win = [lane_info[l]["win_rate"] for l in LANES]
    all_place = [lane_info[l]["place_rate"] for l in LANES]
    all_ex = [lane_info[l]["exhibit"] for l in LANES]
    all_st = [lane_info[l]["st"] for l in LANES]

    race_feat["race_win_rate_mean"] = _mean(all_win)
    race_feat["race_win_rate_std"] = _std(all_win)
    race_feat["race_place_rate_mean"] = _mean(all_place)
    race_feat["race_place_rate_std"] = _std(all_place)
    race_feat["race_exhibit_mean"] = _mean(all_ex)
    race_feat["race_exhibit_std"] = _std(all_ex)
    race_feat["race_st_mean"] = _mean(all_st)
    race_feat["race_st_std"] = _std(all_st)

    race_feat["inside_win_rate_sum"] = sum(lane_info[l]["win_rate"] for l in [1, 2, 3])
    race_feat["outside_win_rate_sum"] = sum(lane_info[l]["win_rate"] for l in [4, 5, 6])
    race_feat["inside_place_rate_sum"] = sum(lane_info[l]["place_rate"] for l in [1, 2, 3])
    race_feat["outside_place_rate_sum"] = sum(lane_info[l]["place_rate"] for l in [4, 5, 6])

    race_feat["inside_exhibit_mean"] = _mean([lane_info[l]["exhibit"] for l in [1, 2, 3]])
    race_feat["outside_exhibit_mean"] = _mean([lane_info[l]["exhibit"] for l in [4, 5, 6]])
    race_feat["inside_st_mean"] = _mean([lane_info[l]["st"] for l in [1, 2, 3]])
    race_feat["outside_st_mean"] = _mean([lane_info[l]["st"] for l in [4, 5, 6]])

    race_feat["inner_advantage_win"] = race_feat["inside_win_rate_sum"] - race_feat["outside_win_rate_sum"]
    race_feat["inner_advantage_place"] = race_feat["inside_place_rate_sum"] - race_feat["outside_place_rate_sum"]
    race_feat["inner_advantage_exhibit"] = race_feat["outside_exhibit_mean"] - race_feat["inside_exhibit_mean"]
    race_feat["inner_advantage_st"] = race_feat["outside_st_mean"] - race_feat["inside_st_mean"]

    for lane in LANES:
        li = lane_info[lane]
        race_feat[f"lane{lane}_win_rate"] = li["win_rate"]
        race_feat[f"lane{lane}_place_rate"] = li["place_rate"]
        race_feat[f"lane{lane}_exhibit"] = li["exhibit"]
        race_feat[f"lane{lane}_st"] = li["st"]
        race_feat[f"lane{lane}_motor"] = li["motor"]
        race_feat[f"lane{lane}_boat"] = li["boat"]
        race_feat[f"lane{lane}_age"] = li["age"]
        race_feat[f"lane{lane}_weight"] = li["weight"]
        race_feat[f"lane{lane}_grade_code"] = li["grade_code"]
        race_feat[f"lane{lane}_course"] = li["course"]
        race_feat[f"lane{lane}_racer_no"] = li["racer_no"]

        race_feat[f"lane{lane}_win_rate_rank"] = float(win_rate_rank[lane])
        race_feat[f"lane{lane}_place_rate_rank"] = float(place_rate_rank[lane])
        race_feat[f"lane{lane}_exhibit_rank"] = float(exhibit_rank[lane])
        race_feat[f"lane{lane}_st_rank"] = float(st_rank[lane])
        race_feat[f"lane{lane}_grade_rank"] = float(grade_rank[lane])

        race_feat[f"lane{lane}_win_rate_diff_from_lane1"] = li["win_rate"] - lane1["win_rate"]
        race_feat[f"lane{lane}_place_rate_diff_from_lane1"] = li["place_rate"] - lane1["place_rate"]
        race_feat[f"lane{lane}_exhibit_diff_from_lane1"] = li["exhibit"] - lane1["exhibit"]
        race_feat[f"lane{lane}_st_diff_from_lane1"] = li["st"] - lane1["st"]
        race_feat[f"lane{lane}_grade_diff_from_lane1"] = li["grade_code"] - lane1["grade_code"]

        race_feat[f"lane{lane}_win_rate_rel"] = li["win_rate"] - race_feat["race_win_rate_mean"]
        race_feat[f"lane{lane}_place_rate_rel"] = li["place_rate"] - race_feat["race_place_rate_mean"]
        race_feat[f"lane{lane}_exhibit_rel"] = li["exhibit"] - race_feat["race_exhibit_mean"]
        race_feat[f"lane{lane}_st_rel"] = li["st"] - race_feat["race_st_mean"]

    out_rows: List[Dict[str, Any]] = []

    for combo in ALL_COMBOS:
        a, b, c = map(int, combo.split("-"))
        first = lane_info[a]
        second = lane_info[b]
        third = lane_info[c]

        row: Dict[str, Any] = {
            "race_key": f"{date}_{venue}_{race_no}",
            "date": date,
            "venue": venue,
            "race_no": race_no,
            "combo": combo,
        }

        row.update(race_feat)

        row["first_lane"] = float(a)
        row["second_lane"] = float(b)
        row["third_lane"] = float(c)

        row["is_head_inside"] = 1.0 if a in [1, 2] else 0.0
        row["is_head_outer"] = 1.0 if a in [5, 6] else 0.0
        row["has_lane1"] = 1.0 if 1 in [a, b, c] else 0.0
        row["has_lane2"] = 1.0 if 2 in [a, b, c] else 0.0
        row["has_lane3"] = 1.0 if 3 in [a, b, c] else 0.0
        row["has_lane4"] = 1.0 if 4 in [a, b, c] else 0.0
        row["has_lane5"] = 1.0 if 5 in [a, b, c] else 0.0
        row["has_lane6"] = 1.0 if 6 in [a, b, c] else 0.0

        row["first_minus_second_lane"] = float(a - b)
        row["first_minus_third_lane"] = float(a - c)
        row["second_minus_third_lane"] = float(b - c)

        for prefix, src in [("first", first), ("second", second), ("third", third)]:
            lane = int(src["lane"])

            row[f"{prefix}_win_rate"] = src["win_rate"]
            row[f"{prefix}_place_rate"] = src["place_rate"]
            row[f"{prefix}_exhibit"] = src["exhibit"]
            row[f"{prefix}_st"] = src["st"]
            row[f"{prefix}_motor"] = src["motor"]
            row[f"{prefix}_boat"] = src["boat"]
            row[f"{prefix}_age"] = src["age"]
            row[f"{prefix}_weight"] = src["weight"]
            row[f"{prefix}_grade_code"] = src["grade_code"]
            row[f"{prefix}_course"] = src["course"]
            row[f"{prefix}_racer_no"] = src["racer_no"]

            row[f"{prefix}_win_rate_rank"] = race_feat[f"lane{lane}_win_rate_rank"]
            row[f"{prefix}_place_rate_rank"] = race_feat[f"lane{lane}_place_rate_rank"]
            row[f"{prefix}_exhibit_rank"] = race_feat[f"lane{lane}_exhibit_rank"]
            row[f"{prefix}_st_rank"] = race_feat[f"lane{lane}_st_rank"]
            row[f"{prefix}_grade_rank"] = race_feat[f"lane{lane}_grade_rank"]

            row[f"{prefix}_win_rate_diff_from_lane1"] = race_feat[f"lane{lane}_win_rate_diff_from_lane1"]
            row[f"{prefix}_place_rate_diff_from_lane1"] = race_feat[f"lane{lane}_place_rate_diff_from_lane1"]
            row[f"{prefix}_exhibit_diff_from_lane1"] = race_feat[f"lane{lane}_exhibit_diff_from_lane1"]
            row[f"{prefix}_st_diff_from_lane1"] = race_feat[f"lane{lane}_st_diff_from_lane1"]
            row[f"{prefix}_grade_diff_from_lane1"] = race_feat[f"lane{lane}_grade_diff_from_lane1"]

            row[f"{prefix}_win_rate_rel"] = race_feat[f"lane{lane}_win_rate_rel"]
            row[f"{prefix}_place_rate_rel"] = race_feat[f"lane{lane}_place_rate_rel"]
            row[f"{prefix}_exhibit_rel"] = race_feat[f"lane{lane}_exhibit_rel"]
            row[f"{prefix}_st_rel"] = race_feat[f"lane{lane}_st_rel"]

        row["top3_win_rate_sum"] = first["win_rate"] + second["win_rate"] + third["win_rate"]
        row["top3_place_rate_sum"] = first["place_rate"] + second["place_rate"] + third["place_rate"]
        row["top3_grade_sum"] = first["grade_code"] + second["grade_code"] + third["grade_code"]

        row["top3_win_rate_mean"] = _mean([first["win_rate"], second["win_rate"], third["win_rate"]])
        row["top3_place_rate_mean"] = _mean([first["place_rate"], second["place_rate"], third["place_rate"]])
        row["top3_exhibit_mean"] = _mean([first["exhibit"], second["exhibit"], third["exhibit"]])
        row["top3_st_mean"] = _mean([first["st"], second["st"], third["st"]])

        row["top3_win_rate_std"] = _std([first["win_rate"], second["win_rate"], third["win_rate"]])
        row["top3_place_rate_std"] = _std([first["place_rate"], second["place_rate"], third["place_rate"]])
        row["top3_exhibit_std"] = _std([first["exhibit"], second["exhibit"], third["exhibit"]])
        row["top3_st_std"] = _std([first["st"], second["st"], third["st"]])

        row["first_second_win_rate_diff"] = first["win_rate"] - second["win_rate"]
        row["first_third_win_rate_diff"] = first["win_rate"] - third["win_rate"]
        row["second_third_win_rate_diff"] = second["win_rate"] - third["win_rate"]

        row["first_second_place_rate_diff"] = first["place_rate"] - second["place_rate"]
        row["first_third_place_rate_diff"] = first["place_rate"] - third["place_rate"]
        row["second_third_place_rate_diff"] = second["place_rate"] - third["place_rate"]

        row["first_second_exhibit_diff"] = first["exhibit"] - second["exhibit"]
        row["first_third_exhibit_diff"] = first["exhibit"] - third["exhibit"]
        row["second_third_exhibit_diff"] = second["exhibit"] - third["exhibit"]

        row["first_second_st_diff"] = first["st"] - second["st"]
        row["first_third_st_diff"] = first["st"] - third["st"]
        row["second_third_st_diff"] = second["st"] - third["st"]

        row["head_inner_advantage_win"] = first["win_rate"] - race_feat["race_win_rate_mean"]
        row["head_inner_advantage_place"] = first["place_rate"] - race_feat["race_place_rate_mean"]
        row["head_inner_advantage_exhibit"] = race_feat["race_exhibit_mean"] - first["exhibit"]
        row["head_inner_advantage_st"] = race_feat["race_st_mean"] - first["st"]

        row["pattern_1_2"] = 1.0 if [a, b] == [1, 2] else 0.0
        row["pattern_1_3"] = 1.0 if [a, b] == [1, 3] else 0.0
        row["pattern_2_1"] = 1.0 if [a, b] == [2, 1] else 0.0
        row["pattern_3_1"] = 1.0 if [a, b] == [3, 1] else 0.0

        out_rows.append(row)

    return out_rows
