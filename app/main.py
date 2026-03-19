from __future__ import annotations

import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, render_template, request

try:
    from app.controller import RaceController
except Exception as e1:
    print("[IMPORT_ERROR] from app.controller import RaceController failed:", e1)
    try:
        from controller import RaceController  # type: ignore
    except Exception as e2:
        print("[IMPORT_ERROR] from controller import RaceController failed:", e2)
        RaceController = None  # type: ignore

app = Flask(__name__, template_folder="templates", static_folder="static")

VENUE_CODE_MAP: Dict[str, int] = {
    "丸亀": 15,
    "戸田": 2,
    "児島": 16,
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.floating)):
            v = float(x)
            return v if np.isfinite(v) else default
        s = str(x).strip()
        if s == "":
            return default
        v = float(s)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _is_debug_request() -> bool:
    return request.args.get("debug", "").strip() in ("1", "true", "True", "yes", "on")


def _today_yyyymmdd_tokyo() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")


def _get_date_default() -> str:
    return request.args.get("date", "").strip() or _today_yyyymmdd_tokyo()


def _blank_races() -> List[Dict[str, Any]]:
    return [{"race_no": rn, "entries": []} for rn in range(1, 13)]


def _calc_ev_cutoffs(ev_result: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    if not ev_result:
        return None, None
    arr = np.array([float(v) for v in ev_result.values() if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return None, None
    return float(np.quantile(arr, 0.90)), float(np.quantile(arr, 0.95))


def _normalize_beforeinfo_dict(beforeinfo_raw: Any) -> Dict[str, Any]:
    if not beforeinfo_raw:
        return {}
    if isinstance(beforeinfo_raw, dict):
        return beforeinfo_raw

    out: Dict[str, Any] = {}
    for k in ("weather", "wind_speed", "wind_direction", "wind_dir", "wave_cm", "wind_speed_mps"):
        if hasattr(beforeinfo_raw, k):
            out[k] = getattr(beforeinfo_raw, k)

    if hasattr(beforeinfo_raw, "lanes"):
        lanes = getattr(beforeinfo_raw, "lanes")
        if isinstance(lanes, dict):
            for ln, v in lanes.items():
                out[ln] = v

    return out


def _pre_info_from_beforeinfo(beforeinfo: Dict[str, Any]) -> Dict[str, Any]:
    if not beforeinfo:
        return {
            "weather": "",
            "wind_dir": "",
            "wind_direction": "",
            "wind_speed": 0.0,
            "wind_speed_mps": 0.0,
            "wave_cm": 0.0,
        }

    wind_dir = str(beforeinfo.get("wind_dir") or beforeinfo.get("wind_direction") or "").strip()
    wind_speed = _safe_float(
        beforeinfo.get("wind_speed_mps", beforeinfo.get("wind_speed", 0.0)),
        0.0,
    )
    wave_cm = _safe_float(beforeinfo.get("wave_cm", beforeinfo.get("wave", 0.0)), 0.0)
    weather = str(beforeinfo.get("weather") or "").strip()

    return {
        "weather": weather,
        "wind_dir": wind_dir,
        "wind_direction": wind_dir,
        "wind_speed": wind_speed,
        "wind_speed_mps": wind_speed,
        "wave_cm": wave_cm,
    }


def _inject_exhibit_and_st_from_beforeinfo(entries: List[Dict[str, Any]], beforeinfo: Dict[str, Any]) -> None:
    if not entries or not beforeinfo:
        return

    for e in entries:
        lane = _to_int(e.get("lane", 0), 0)
        if lane <= 0:
            continue

        info = beforeinfo.get(lane)
        if info is None:
            info = beforeinfo.get(str(lane))
        if not isinstance(info, dict):
            continue

        ex = info.get("exhibit_time")
        if ex is None:
            ex = info.get("exhibit")

        st = info.get("st")
        if st is None:
            st = info.get("start_timing")

        course = info.get("course")
        if course is None:
            course = info.get("course_no")

        if e.get("exhibit") in (None, "", 0, "0") and ex not in (None, "", 0, "0"):
            e["exhibit"] = ex
        if e.get("start_timing") in (None, "", 0, "0") and st not in (None, "", 0, "0"):
            e["start_timing"] = st
        if e.get("course") in (None, "", 0, "0") and course not in (None, "", 0, "0"):
            e["course"] = course


def _preds_to_probabilities(ai_preds: List[Dict[str, Any]]) -> Dict[str, float]:
    probabilities: Dict[str, float] = {}
    for row in ai_preds:
        combo = str(row.get("combo", "")).strip()
        if not combo:
            continue
        probabilities[combo] = _safe_float(row.get("score", 0.0), 0.0)

    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v / total for k, v in probabilities.items()}
    return probabilities


def _preds_to_ev_result(ai_preds: List[Dict[str, Any]]) -> Dict[str, float]:
    ev_result: Dict[str, float] = {}
    for row in ai_preds:
        combo = str(row.get("combo", "")).strip()
        if not combo:
            continue
        if row.get("ev") is None:
            continue
        ev_result[combo] = _safe_float(row.get("ev", 0.0), 0.0)
    return ev_result


def _debug_print_top10(probabilities: Dict[str, float]) -> None:
    if not probabilities:
        print("[DBG] TOP10 probs: (empty)")
        return
    top = sorted(probabilities.items(), key=lambda kv: float(kv[1]), reverse=True)[:10]
    print("[DBG] TOP10 probs:")
    for c, p in top:
        print(f"  {c} {float(p):.6f}")


def _render_venue_page(venue_name: str):
    date = _get_date_default()
    race_str = request.args.get("race", "").strip()
    mode = request.args.get("mode", "").strip().lower()

    if RaceController is None:
        return "RaceController import failed. Check app/controller.py", 500

    controller = RaceController()
    _ = VENUE_CODE_MAP.get(venue_name, 0)

    grouped_odds = None
    probabilities: Dict[str, float] = {}
    ev_result: Dict[str, float] = {}
    ev_cutoff_90 = None
    ev_cutoff_95 = None
    selected_race = 0

    pre_info: Dict[str, Any] = {
        "weather": "",
        "wind_dir": "",
        "wind_direction": "",
        "wind_speed": 0.0,
        "wind_speed_mps": 0.0,
        "wave_cm": 0.0,
    }
    beforeinfo_for_template: Dict[str, Any] = {}

    if not (mode == "full" and race_str.isdigit()):
        races = _blank_races()
        return render_template(
            "index.html",
            venue=venue_name,
            date=date,
            races=races,
            selected_race=0,
            grouped_odds=None,
            probabilities={},
            ev_result={},
            ev_cutoff_90=None,
            ev_cutoff_95=None,
            pre_info=pre_info,
            beforeinfo={},
        )

    race_no = int(race_str)
    selected_race = race_no

    try:
        if venue_name == "戸田":
            entries = controller.get_entries_toda_race(date, race_no)
        elif venue_name == "児島":
            entries = controller.get_entries_kojima_race(date, race_no)
        else:
            entries = controller.get_entries_race(date, race_no)
    except Exception as e:
        return f"get_entries_race failed: {e}", 500

    entries = [dict(x) for x in entries]
    races = _blank_races()
    races[race_no - 1]["entries"] = entries

    try:
        if venue_name == "戸田":
            entries = controller.enrich_entries_toda(entries, date=date, race_no=race_no)
        elif venue_name == "児島":
            entries = controller.enrich_entries_kojima(entries, date=date, race_no=race_no)
        else:
            entries = controller.enrich_entries_marugame(entries, date=date, race_no=race_no)
    except Exception as e:
        if _is_debug_request():
            print("[ENRICH_ERROR]", e)

    try:
        if venue_name == "戸田":
            bi_raw = controller.get_beforeinfo_only_toda(race_no=race_no, date=date)
        elif venue_name == "児島":
            bi_raw = controller.get_beforeinfo_only_kojima(race_no=race_no, date=date)
        else:
            bi_raw = controller.get_beforeinfo_only(race_no=race_no, date=date)

        beforeinfo_for_template = _normalize_beforeinfo_dict(bi_raw)
    except Exception as e:
        beforeinfo_for_template = {}
        if _is_debug_request():
            print("[BEFOREINFO_ERROR]", e)

    _inject_exhibit_and_st_from_beforeinfo(entries, beforeinfo_for_template)
    pre_info = _pre_info_from_beforeinfo(beforeinfo_for_template)

    try:
        if venue_name == "戸田":
            grouped_odds = controller.get_odds_only_toda(race_no=race_no, date=date)
        elif venue_name == "児島":
            grouped_odds = controller.get_odds_only_kojima(race_no=race_no, date=date)
        else:
            grouped_odds = controller.get_odds_only(race_no=race_no, date=date)
    except Exception as e:
        grouped_odds = None
        if _is_debug_request():
            print("[ODDS_ERROR]", e)

    ai_preds: List[Dict[str, Any]] = []
    try:
        if venue_name == "戸田":
            ai_preds = controller.get_ai_predictions_toda(
                date=date,
                race_no=race_no,
                top_n=120,
                with_odds=True,
            )
        elif venue_name == "児島":
            ai_preds = controller.get_ai_predictions_kojima(
                date=date,
                race_no=race_no,
                top_n=120,
                with_odds=True,
            )
        else:
            ai_preds = controller.get_ai_predictions_marugame(
                date=date,
                race_no=race_no,
                top_n=120,
                with_odds=True,
            )
    except Exception as e:
        print("[AI_ERROR]", e)
        print(traceback.format_exc())
        ai_preds = []

    probabilities = _preds_to_probabilities(ai_preds)
    ev_result = _preds_to_ev_result(ai_preds)
    ev_cutoff_90, ev_cutoff_95 = _calc_ev_cutoffs(ev_result)

    if _is_debug_request():
        _debug_print_top10(probabilities)

    races[race_no - 1]["entries"] = entries

    return render_template(
        "index.html",
        venue=venue_name,
        date=date,
        races=races,
        selected_race=selected_race,
        grouped_odds=grouped_odds,
        probabilities=probabilities,
        ev_result=ev_result,
        ev_cutoff_90=ev_cutoff_90,
        ev_cutoff_95=ev_cutoff_95,
        pre_info=pre_info,
        beforeinfo=beforeinfo_for_template,
    )


@app.route("/")
def home():
    date = request.args.get("date", "").strip() or _today_yyyymmdd_tokyo()
    return render_template("home.html", date=date)


@app.route("/marugame")
def marugame():
    return _render_venue_page("丸亀")


@app.route("/toda")
def toda():
    return _render_venue_page("戸田")


@app.route("/kojima")
def kojima():
    return _render_venue_page("児島")


if __name__ == "__main__":
    app.run(debug=True)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
