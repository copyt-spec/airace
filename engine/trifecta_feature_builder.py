# engine/trifecta_feature_builder.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd


# ★ odds_fetcher と同じ固定順（絶対触らない）
FIXED_ORDER = """1-2-3 2-1-3 3-1-2 4-1-2 5-1-2 6-1-2
1-2-4 2-1-4 3-1-4 4-1-3 5-1-3 6-1-3
1-2-5 2-1-5 3-1-5 4-1-5 5-1-4 6-1-4
1-2-6 2-1-6 3-1-6 4-1-6 5-1-6 6-1-5
1-3-2 2-3-1 3-2-1 4-2-1 5-2-1 6-2-1
1-3-4 2-3-4 3-2-4 4-2-3 5-2-3 6-2-3
1-3-5 2-3-5 3-2-5 4-2-5 5-2-4 6-2-4
1-3-6 2-3-6 3-2-6 4-2-6 5-2-6 6-2-5
1-4-2 2-4-1 3-4-1 4-3-1 5-3-1 6-3-1
1-4-3 2-4-3 3-4-2 4-3-2 5-3-2 6-3-2
1-4-5 2-4-5 3-4-5 4-3-5 5-3-4 6-3-4
1-4-6 2-4-6 3-4-6 4-3-6 5-3-6 6-3-5
1-5-2 2-5-1 3-5-1 4-5-1 5-4-1 6-4-1
1-5-3 2-5-3 3-5-2 4-5-2 5-4-2 6-4-2
1-5-4 2-5-4 3-5-4 4-5-3 5-4-3 6-4-3
1-5-6 2-5-6 3-5-6 4-5-6 5-4-6 6-4-5
1-6-2 2-6-1 3-6-1 4-6-1 5-6-1 6-5-1
1-6-3 2-6-3 3-6-2 4-6-2 5-6-2 6-5-2
1-6-4 2-6-4 3-6-4 4-6-3 5-6-3 6-5-3
1-6-5 2-6-5 3-6-5 4-6-5 5-6-4 6-5-4""".split()


# ===== leakage guard =====
try:
    from engine.leakage_guard import drop_odds_leakage  # type: ignore
except Exception:
    drop_odds_leakage = None  # type: ignore


def _to_float(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("value", "val", "score", "num", "x", "y"):
            if k in v:
                return _to_float(v.get(k))
        return 0.0
    if isinstance(v, (list, tuple)):
        return _to_float(v[0]) if len(v) > 0 else 0.0

    s = str(v).strip()
    if s == "" or s == "-" or s.lower() in ("nan", "none"):
        return 0.0
    if s.startswith("."):
        s = "0" + s
    try:
        return float(s)
    except Exception:
        if len(s) >= 2 and s[0].upper() in ("F", "L"):
            try:
                return float(s[1:])
            except Exception:
                return 0.0
        return 0.0


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return a / b


def _wind_dir_to_num(s: Any) -> float:
    t = _to_str(s)
    mapping = {
        "北": 1.0, "北東": 2.0, "東": 3.0, "南東": 4.0,
        "南": 5.0, "南西": 6.0, "西": 7.0, "北西": 8.0,
    }
    return mapping.get(t, 0.0)


def _weather_to_num(s: Any) -> float:
    t = _to_str(s)
    mapping = {
        "晴れ": 1.0,
        "くもり": 2.0,
        "曇り": 2.0,
        "雨": 3.0,
        "雪": 4.0,
    }
    return mapping.get(t, 0.0)


def _lane_advantage(lane: int) -> float:
    mapping = {
        1: 3.0,
        2: 2.0,
        3: 1.0,
        4: 0.0,
        5: -1.0,
        6: -2.0,
    }
    return mapping.get(int(lane), 0.0)


def _inside_bias(lane: int) -> float:
    mapping = {
        1: 1.00,
        2: 0.80,
        3: 0.55,
        4: 0.25,
        5: 0.10,
        6: 0.00,
    }
    return mapping.get(int(lane), 0.0)


def _bi_get(bi: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in bi:
            return bi.get(k)
    return None


def _bi_lane_dict(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if v is not None:
        return {
            "exhibit_time": getattr(v, "exhibit_time", None),
            "st": getattr(v, "st", None),
            "course": getattr(v, "course", None),
        }
    return {}


def _entry_value(entry: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in entry:
            return entry.get(k)
    return None


def _pack_lane(entry: Dict[str, Any], lane: int, prefix: str) -> Dict[str, float]:
    motor = _to_float(_entry_value(entry, "motor", "motor_no"))
    boat = _to_float(_entry_value(entry, "boat", "boat_no"))
    exhibit = _to_float(_entry_value(entry, "exhibit", "exhibit_time"))
    st = _to_float(_entry_value(entry, "start_timing", "st"))
    course = _to_float(_entry_value(entry, "course", "course_no"))

    racer_win_rate = _to_float(_entry_value(entry, "racer_win_rate"))
    racer_place_rate = _to_float(_entry_value(entry, "racer_place_rate"))
    racer_avg_st_base = _to_float(_entry_value(entry, "racer_avg_st_base"))
    racer_ability_index = _to_float(_entry_value(entry, "racer_ability_index"))
    racer_prev_ability_index = _to_float(_entry_value(entry, "racer_prev_ability_index"))
    racer_grade_score = _to_float(_entry_value(entry, "racer_grade_score"))
    racer_prev_grade_score = _to_float(_entry_value(entry, "racer_prev_grade_score"))
    racer_age = _to_float(_entry_value(entry, "racer_age"))
    racer_height = _to_float(_entry_value(entry, "racer_height"))
    racer_weight = _to_float(_entry_value(entry, "racer_weight"))

    racer_course_place_rate = _to_float(_entry_value(entry, f"racer_course{lane}_place_rate"))
    racer_course_avg_st = _to_float(_entry_value(entry, f"racer_course{lane}_avg_st"))
    racer_course_avg_st_rank = _to_float(_entry_value(entry, f"racer_course{lane}_avg_st_rank"))
    racer_course_entry_count = _to_float(_entry_value(entry, f"racer_course{lane}_entry_count"))

    lane_power = (
        1.8 * _lane_advantage(lane)
        + 0.55 * racer_win_rate
        + 0.08 * racer_ability_index
        + 0.07 * racer_course_place_rate
        + 0.03 * racer_grade_score
        - 6.0 * racer_avg_st_base
        - 4.5 * racer_course_avg_st
        - 0.8 * exhibit
    )

    motor_power = motor
    boat_power = boat
    start_power = -st
    exhibit_power = -exhibit

    one_head_score = (
        1.3 * _inside_bias(lane)
        + 0.45 * racer_win_rate
        + 0.08 * racer_ability_index
        + 0.06 * racer_course_place_rate
        + 0.04 * racer_grade_score
        + 0.015 * motor_power
        - 5.5 * racer_avg_st_base
        - 2.8 * st
        - 0.6 * exhibit
    )

    return {
        f"{prefix}_lane": float(lane),
        f"{prefix}_boat": boat,
        f"{prefix}_course": course,
        f"{prefix}_exhibit": exhibit,
        f"{prefix}_motor": motor,
        f"{prefix}_racer_no": _to_float(_entry_value(entry, "racer_no")),
        f"{prefix}_st": st,

        f"{prefix}_lane_advantage": _lane_advantage(lane),
        f"{prefix}_inside_bias": _inside_bias(lane),
        f"{prefix}_is_inner": 1.0 if lane <= 3 else 0.0,
        f"{prefix}_is_outer": 1.0 if lane >= 4 else 0.0,

        f"{prefix}_racer_win_rate": racer_win_rate,
        f"{prefix}_racer_place_rate": racer_place_rate,
        f"{prefix}_racer_avg_st_base": racer_avg_st_base,
        f"{prefix}_racer_ability_index": racer_ability_index,
        f"{prefix}_racer_prev_ability_index": racer_prev_ability_index,
        f"{prefix}_racer_grade_score": racer_grade_score,
        f"{prefix}_racer_prev_grade_score": racer_prev_grade_score,
        f"{prefix}_racer_age": racer_age,
        f"{prefix}_racer_height": racer_height,
        f"{prefix}_racer_weight": racer_weight,

        f"{prefix}_racer_course_place_rate": racer_course_place_rate,
        f"{prefix}_racer_course_avg_st": racer_course_avg_st,
        f"{prefix}_racer_course_avg_st_rank": racer_course_avg_st_rank,
        f"{prefix}_racer_course_entry_count": racer_course_entry_count,

        f"{prefix}_motor_power": motor_power,
        f"{prefix}_boat_power": boat_power,
        f"{prefix}_start_power": start_power,
        f"{prefix}_exhibit_power": exhibit_power,
        f"{prefix}_lane_power": lane_power,
        f"{prefix}_one_head_score": one_head_score,
    }


def build_trifecta_features(
    race_entries: List[Dict[str, Any]],
    before_info: Optional[Dict[str, Any]] = None,
    venue: str = "",
    race_no: int = 0,
    date: str = "",
) -> pd.DataFrame:
    """
    1レース6艇 -> 120行（combo候補）
    学習側 build_trifecta_train_features.py と極力そろえる
    """
    bi = before_info or {}

    lane_map: Dict[int, Dict[str, Any]] = {}
    for e in race_entries:
        lane = int(_to_float(e.get("lane")))
        if 1 <= lane <= 6:
            lane_map[lane] = dict(e)

    def _bi_lane(lane: int) -> Dict[str, Any]:
        v = bi.get(lane)
        if v is None:
            v = bi.get(str(lane))
        d = _bi_lane_dict(v)

        if "exhibit_time" not in d:
            for k in ("exhibit", "ex_time", "exhibitTime"):
                if k in d:
                    d["exhibit_time"] = d.get(k)
                    break

        if "st" not in d:
            for k in ("start_timing", "start", "st_time"):
                if k in d:
                    d["st"] = d.get(k)
                    break

        if "course" not in d:
            for k in ("course_no", "cource", "courseNo"):
                if k in d:
                    d["course"] = d.get(k)
                    break

        return d

    wave_cm = _to_float(_bi_get(bi, "wave_cm", "wave", "waveCm"))
    wind_speed_mps = _to_float(_bi_get(bi, "wind_speed_mps", "wind_speed", "wind", "windSpeed", "wind_speed_ms"))
    wind_dir = _wind_dir_to_num(_bi_get(bi, "wind_dir", "wind_direction", "windDirection", "wind_dir_name"))
    weather = _weather_to_num(_bi_get(bi, "weather", "tenki", "weather_name"))

    rows: List[Dict[str, Any]] = []

    for combo in FIXED_ORDER:
        a_s, b_s, c_s = combo.split("-")
        a, b, c = int(a_s), int(b_s), int(c_s)

        ea = dict(lane_map.get(a, {}))
        eb = dict(lane_map.get(b, {}))
        ec = dict(lane_map.get(c, {}))

        # beforeinfo があれば推論時も展示/ST/進入を上書き補助
        ba = _bi_lane(a)
        bb = _bi_lane(b)
        bc = _bi_lane(c)

        if ba:
            if ea.get("exhibit") in (None, "", 0, "0") and ba.get("exhibit_time") not in (None, "", 0, "0"):
                ea["exhibit"] = ba.get("exhibit_time")
            if ea.get("start_timing") in (None, "", 0, "0") and ba.get("st") not in (None, "", 0, "0"):
                ea["start_timing"] = ba.get("st")
            if ea.get("course") in (None, "", 0, "0") and ba.get("course") not in (None, "", 0, "0"):
                ea["course"] = ba.get("course")

        if bb:
            if eb.get("exhibit") in (None, "", 0, "0") and bb.get("exhibit_time") not in (None, "", 0, "0"):
                eb["exhibit"] = bb.get("exhibit_time")
            if eb.get("start_timing") in (None, "", 0, "0") and bb.get("st") not in (None, "", 0, "0"):
                eb["start_timing"] = bb.get("st")
            if eb.get("course") in (None, "", 0, "0") and bb.get("course") not in (None, "", 0, "0"):
                eb["course"] = bb.get("course")

        if bc:
            if ec.get("exhibit") in (None, "", 0, "0") and bc.get("exhibit_time") not in (None, "", 0, "0"):
                ec["exhibit"] = bc.get("exhibit_time")
            if ec.get("start_timing") in (None, "", 0, "0") and bc.get("st") not in (None, "", 0, "0"):
                ec["start_timing"] = bc.get("st")
            if ec.get("course") in (None, "", 0, "0") and bc.get("course") not in (None, "", 0, "0"):
                ec["course"] = bc.get("course")

        feat = {
            "date": date,
            "venue": venue,
            "race_no": float(race_no),
            "combo": combo,
            "wave_cm": wave_cm,
            "wind_speed_mps": wind_speed_mps,
            "wind_dir": wind_dir,
            "weather": weather,
        }

        feat.update(_pack_lane(ea, a, "a"))
        feat.update(_pack_lane(eb, b, "b"))
        feat.update(_pack_lane(ec, c, "c"))

        # ===== 既存差分 =====
        feat.update({
            "ab_exhibit_diff": feat["a_exhibit"] - feat["b_exhibit"],
            "ac_exhibit_diff": feat["a_exhibit"] - feat["c_exhibit"],
            "bc_exhibit_diff": feat["b_exhibit"] - feat["c_exhibit"],

            "ab_st_diff": feat["a_st"] - feat["b_st"],
            "ac_st_diff": feat["a_st"] - feat["c_st"],
            "bc_st_diff": feat["b_st"] - feat["c_st"],

            "ab_course_diff": feat["a_course"] - feat["b_course"],
            "ac_course_diff": feat["a_course"] - feat["c_course"],
            "bc_course_diff": feat["b_course"] - feat["c_course"],

            "ab_motor_diff": feat["a_motor"] - feat["b_motor"],
            "ac_motor_diff": feat["a_motor"] - feat["c_motor"],
            "bc_motor_diff": feat["b_motor"] - feat["c_motor"],
        })

        # ===== 能力差分 =====
        feat.update({
            "ab_racer_win_rate_diff": feat["a_racer_win_rate"] - feat["b_racer_win_rate"],
            "ac_racer_win_rate_diff": feat["a_racer_win_rate"] - feat["c_racer_win_rate"],
            "bc_racer_win_rate_diff": feat["b_racer_win_rate"] - feat["c_racer_win_rate"],

            "ab_racer_place_rate_diff": feat["a_racer_place_rate"] - feat["b_racer_place_rate"],
            "ac_racer_place_rate_diff": feat["a_racer_place_rate"] - feat["c_racer_place_rate"],
            "bc_racer_place_rate_diff": feat["b_racer_place_rate"] - feat["c_racer_place_rate"],

            "ab_racer_avg_st_base_diff": feat["a_racer_avg_st_base"] - feat["b_racer_avg_st_base"],
            "ac_racer_avg_st_base_diff": feat["a_racer_avg_st_base"] - feat["c_racer_avg_st_base"],
            "bc_racer_avg_st_base_diff": feat["b_racer_avg_st_base"] - feat["c_racer_avg_st_base"],

            "ab_racer_ability_index_diff": feat["a_racer_ability_index"] - feat["b_racer_ability_index"],
            "ac_racer_ability_index_diff": feat["a_racer_ability_index"] - feat["c_racer_ability_index"],
            "bc_racer_ability_index_diff": feat["b_racer_ability_index"] - feat["c_racer_ability_index"],

            "ab_racer_grade_score_diff": feat["a_racer_grade_score"] - feat["b_racer_grade_score"],
            "ac_racer_grade_score_diff": feat["a_racer_grade_score"] - feat["c_racer_grade_score"],
            "bc_racer_grade_score_diff": feat["b_racer_grade_score"] - feat["c_racer_grade_score"],

            "ab_racer_course_place_rate_diff": feat["a_racer_course_place_rate"] - feat["b_racer_course_place_rate"],
            "ac_racer_course_place_rate_diff": feat["a_racer_course_place_rate"] - feat["c_racer_course_place_rate"],
            "bc_racer_course_place_rate_diff": feat["b_racer_course_place_rate"] - feat["c_racer_course_place_rate"],

            "ab_racer_course_avg_st_diff": feat["a_racer_course_avg_st"] - feat["b_racer_course_avg_st"],
            "ac_racer_course_avg_st_diff": feat["a_racer_course_avg_st"] - feat["c_racer_course_avg_st"],
            "bc_racer_course_avg_st_diff": feat["b_racer_course_avg_st"] - feat["c_racer_course_avg_st"],

            "ab_racer_course_avg_st_rank_diff": feat["a_racer_course_avg_st_rank"] - feat["b_racer_course_avg_st_rank"],
            "ac_racer_course_avg_st_rank_diff": feat["a_racer_course_avg_st_rank"] - feat["c_racer_course_avg_st_rank"],
            "bc_racer_course_avg_st_rank_diff": feat["b_racer_course_avg_st_rank"] - feat["c_racer_course_avg_st_rank"],
        })

        # ===== 競艇っぽい強化特徴 =====
        feat.update({
            "ab_lane_advantage_diff": feat["a_lane_advantage"] - feat["b_lane_advantage"],
            "ac_lane_advantage_diff": feat["a_lane_advantage"] - feat["c_lane_advantage"],
            "bc_lane_advantage_diff": feat["b_lane_advantage"] - feat["c_lane_advantage"],

            "ab_inside_bias_diff": feat["a_inside_bias"] - feat["b_inside_bias"],
            "ac_inside_bias_diff": feat["a_inside_bias"] - feat["c_inside_bias"],
            "bc_inside_bias_diff": feat["b_inside_bias"] - feat["c_inside_bias"],

            "ab_lane_power_diff": feat["a_lane_power"] - feat["b_lane_power"],
            "ac_lane_power_diff": feat["a_lane_power"] - feat["c_lane_power"],
            "bc_lane_power_diff": feat["b_lane_power"] - feat["c_lane_power"],

            "ab_one_head_score_diff": feat["a_one_head_score"] - feat["b_one_head_score"],
            "ac_one_head_score_diff": feat["a_one_head_score"] - feat["c_one_head_score"],
            "bc_one_head_score_diff": feat["b_one_head_score"] - feat["c_one_head_score"],

            "ab_start_power_diff": feat["a_start_power"] - feat["b_start_power"],
            "ac_start_power_diff": feat["a_start_power"] - feat["c_start_power"],
            "bc_start_power_diff": feat["b_start_power"] - feat["c_start_power"],

            "ab_exhibit_power_diff": feat["a_exhibit_power"] - feat["b_exhibit_power"],
            "ac_exhibit_power_diff": feat["a_exhibit_power"] - feat["c_exhibit_power"],
            "bc_exhibit_power_diff": feat["b_exhibit_power"] - feat["c_exhibit_power"],

            "ab_motor_power_diff": feat["a_motor_power"] - feat["b_motor_power"],
            "ac_motor_power_diff": feat["a_motor_power"] - feat["c_motor_power"],
            "bc_motor_power_diff": feat["b_motor_power"] - feat["c_motor_power"],

            "ab_boat_power_diff": feat["a_boat_power"] - feat["b_boat_power"],
            "ac_boat_power_diff": feat["a_boat_power"] - feat["c_boat_power"],
            "bc_boat_power_diff": feat["b_boat_power"] - feat["c_boat_power"],
        })

        # ===== 合計/比率 =====
        ability_sum = (
            feat["a_racer_ability_index"]
            + feat["b_racer_ability_index"]
            + feat["c_racer_ability_index"]
        )

        feat.update({
            "abc_win_rate_sum": feat["a_racer_win_rate"] + feat["b_racer_win_rate"] + feat["c_racer_win_rate"],
            "abc_ability_sum": ability_sum,
            "abc_lane_power_sum": feat["a_lane_power"] + feat["b_lane_power"] + feat["c_lane_power"],
            "abc_one_head_score_sum": feat["a_one_head_score"] + feat["b_one_head_score"] + feat["c_one_head_score"],
            "abc_motor_sum": feat["a_motor"] + feat["b_motor"] + feat["c_motor"],
            "abc_exhibit_sum": feat["a_exhibit"] + feat["b_exhibit"] + feat["c_exhibit"],
            "abc_st_sum": feat["a_st"] + feat["b_st"] + feat["c_st"],

            "a_share_of_ability": _safe_div(feat["a_racer_ability_index"], ability_sum),
            "b_share_of_ability": _safe_div(feat["b_racer_ability_index"], ability_sum),
            "c_share_of_ability": _safe_div(feat["c_racer_ability_index"], ability_sum),
        })

        # ===== 順位関係フラグ =====
        feat.update({
            "a_head_is_inner": 1.0 if feat["a_lane"] <= 2 else 0.0,
            "a_head_is_center": 1.0 if feat["a_lane"] in (3, 4) else 0.0,
            "a_head_is_outer": 1.0 if feat["a_lane"] >= 5 else 0.0,

            "a_head_best_winrate": 1.0 if feat["a_racer_win_rate"] >= max(feat["b_racer_win_rate"], feat["c_racer_win_rate"]) else 0.0,
            "a_head_best_ability": 1.0 if feat["a_racer_ability_index"] >= max(feat["b_racer_ability_index"], feat["c_racer_ability_index"]) else 0.0,
            "a_head_best_motor": 1.0 if feat["a_motor"] >= max(feat["b_motor"], feat["c_motor"]) else 0.0,

            "a_head_best_st": 1.0 if feat["a_st"] <= min(feat["b_st"], feat["c_st"]) else 0.0,
            "a_head_best_exhibit": 1.0 if feat["a_exhibit"] <= min(feat["b_exhibit"], feat["c_exhibit"]) else 0.0,

            "b_second_inner_than_c": 1.0 if feat["b_lane"] < feat["c_lane"] else 0.0,
            "b_second_better_winrate_than_c": 1.0 if feat["b_racer_win_rate"] > feat["c_racer_win_rate"] else 0.0,
            "b_second_better_st_than_c": 1.0 if feat["b_st"] < feat["c_st"] else 0.0,
        })

        rows.append(feat)

    df = pd.DataFrame(rows)

    if drop_odds_leakage is not None:
        df = drop_odds_leakage(df, verbose=False, context="trifecta_feature_builder")

    return df
