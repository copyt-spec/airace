from __future__ import annotations

import os
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, render_template, request

try:
    from app.controller import RaceController
except Exception as e:
    print("[IMPORT_ERROR]", e)
    RaceController = None  # type: ignore

from engine.prediction_logger import build_prediction_rows, save_prediction_rows


app = Flask(__name__, template_folder="templates", static_folder="static")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "prediction_results_merged.csv"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _today() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")


def _blank_races() -> List[Dict[str, Any]]:
    return [{"race_no": i, "entries": []} for i in range(1, 13)]


def _calc_ev(prob: float, odds: float) -> float:
    if odds <= 0 or prob <= 0:
        return 0.0
    return float(prob * odds)


def _build_all_trifecta_labels() -> List[str]:
    labels: List[str] = []
    for a in range(1, 7):
        for b in range(1, 7):
            if b == a:
                continue
            for c in range(1, 7):
                if c == a or c == b:
                    continue
                labels.append(f"{a}-{b}-{c}")
    return labels


def _complete_probabilities(probabilities: Dict[str, float] | None) -> Dict[str, float]:
    src = probabilities or {}
    out: Dict[str, float] = {}

    for combo in _build_all_trifecta_labels():
        out[combo] = _safe_float(src.get(combo, 0.0), 0.0)

    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}

    return out


def _complete_odds_map(odds_map: Dict[str, float] | None) -> Dict[str, float]:
    src = odds_map or {}
    out: Dict[str, float] = {}

    for combo in _build_all_trifecta_labels():
        out[combo] = _safe_float(src.get(combo, 0.0), 0.0)

    return out


def _build_ev_map_from_prob_and_odds(
    probabilities: Dict[str, float],
    odds_map: Dict[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for combo in _build_all_trifecta_labels():
        prob = _safe_float(probabilities.get(combo, 0.0), 0.0)
        odds = _safe_float(odds_map.get(combo, 0.0), 0.0)
        out[combo] = _calc_ev(prob, odds)
    return out


def _grouped_odds_to_flat_map(grouped_odds: Dict[str, Any] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not grouped_odds:
        return out

    godata = grouped_odds.get("data", {})
    if not isinstance(godata, dict):
        return out

    for a in range(1, 7):
        col = godata.get(a, {})
        if not isinstance(col, dict):
            continue

        for b in range(1, 7):
            if b == a:
                continue
            for c in range(1, 7):
                if c == a or c == b:
                    continue
                combo = f"{a}-{b}-{c}"
                odd = col.get((b, c))
                if odd is None:
                    odd = col.get(f"{b}{c}")
                out[combo] = _safe_float(odd, 0.0)

    return out


def _normalize_venue_name(v: str) -> str:
    s = str(v or "").strip()
    if "丸亀" in s:
        return "丸亀"
    if "戸田" in s:
        return "戸田"
    if "児島" in s:
        return "児島"
    return s


def _build_home_stats() -> Dict[str, Dict[str, Any]]:
    venue_order = ["丸亀", "戸田", "児島"]
    empty_row = {
        "buy_count": 0,
        "hit_count": 0,
        "hit_rate": 0.0,
        "total_bets": 0.0,
        "total_return": 0.0,
        "total_profit": 0.0,
        "roi": 0.0,
    }
    out = {v: dict(empty_row) for v in venue_order}

    if not MERGED_LOG_PATH.exists():
        return out

    try:
        df = pd.read_csv(MERGED_LOG_PATH)
    except Exception as e:
        print("[WARN] failed to read merged log:", e)
        return out

    if df.empty:
        return out

    if "venue" not in df.columns or "is_selected" not in df.columns:
        return out

    df["venue_norm"] = df["venue"].astype(str).map(_normalize_venue_name)
    df = df[df["is_selected"] == 1].copy()

    if df.empty:
        return out

    for venue in venue_order:
        vdf = df[df["venue_norm"] == venue].copy()
        if vdf.empty:
            continue

        buy_count = int(len(vdf))
        hit_count = int((vdf["is_hit"] == 1).sum()) if "is_hit" in vdf.columns else 0
        total_bets = float(vdf["bet_cost_yen"].fillna(0).sum()) if "bet_cost_yen" in vdf.columns else 0.0
        total_return = float(vdf["return_yen"].fillna(0).sum()) if "return_yen" in vdf.columns else 0.0
        total_profit = total_return - total_bets
        hit_rate = (hit_count / buy_count) if buy_count > 0 else 0.0
        roi = (total_return / total_bets) if total_bets > 0 else 0.0

        out[venue] = {
            "buy_count": buy_count,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "total_bets": total_bets,
            "total_return": total_return,
            "total_profit": total_profit,
            "roi": roi,
        }

    return out


def _debug_render_log(
    venue: str,
    date: str,
    race_no: int,
    mode: str | None,
    entries: List[Dict[str, Any]],
    grouped_odds: Dict[str, Any],
    best_bets: List[Dict[str, Any]],
    probabilities: Dict[str, float],
    ev_result: Dict[str, float],
) -> None:
    print("\n" + "=" * 80)
    print("[DEBUG] RENDER TEMPLATE")
    print("=" * 80)
    print("[DEBUG] venue         :", venue)
    print("[DEBUG] date          :", date)
    print("[DEBUG] selected_race :", race_no)
    print("[DEBUG] mode          :", mode)
    print("[DEBUG] entries_len   :", len(entries))
    print("[DEBUG] best_bets_len :", len(best_bets))
    print("[DEBUG] probabilities :", len(probabilities))
    print("[DEBUG] ev_result     :", len(ev_result))
    print("[DEBUG] odds exists   :", bool(grouped_odds and grouped_odds.get("data")))


def _get_entries_and_beforeinfo(controller: RaceController, venue: str, date: str, race_no: int):
    if venue == "戸田":
        entries = controller.get_entries_toda_race(date, race_no)
        entries = controller.enrich_entries_toda(entries, date=date, race_no=race_no)
        beforeinfo = controller.get_beforeinfo_only_toda(race_no=race_no, date=date)
    elif venue == "児島":
        entries = controller.get_entries_kojima_race(date, race_no)
        entries = controller.enrich_entries_kojima(entries, date=date, race_no=race_no)
        beforeinfo = controller.get_beforeinfo_only_kojima(race_no=race_no, date=date)
    else:
        entries = controller.get_entries_race(date, race_no)
        entries = controller.enrich_entries_marugame(entries, date=date, race_no=race_no)
        beforeinfo = controller.get_beforeinfo_only(race_no=race_no, date=date)

    return [dict(e) for e in entries], beforeinfo


def _get_full_bundle(controller: RaceController, venue: str, date: str, race_no: int):
    if venue == "戸田":
        grouped_odds = controller.get_odds_only_toda(race_no=race_no, date=date)
        bundle = controller.get_ai_prediction_bundle_toda(date, race_no, top_n=20, with_odds=True)
    elif venue == "児島":
        grouped_odds = controller.get_odds_only_kojima(race_no=race_no, date=date)
        bundle = controller.get_ai_prediction_bundle_kojima(date, race_no, top_n=20, with_odds=True)
    else:
        grouped_odds = controller.get_odds_only(race_no=race_no, date=date)
        bundle = controller.get_ai_prediction_bundle_marugame(date, race_no, top_n=20, with_odds=True)

    return grouped_odds, (bundle or {})


def _save_prediction_log(
    *,
    date: str,
    venue: str,
    race_no: int,
    best_bets: List[Dict[str, Any]],
    probabilities: Dict[str, float],
    grouped_odds: Dict[str, Any],
) -> None:
    try:
        prediction_rows = build_prediction_rows(
            date=date,
            venue=venue,
            race_no=race_no,
            best_bets=best_bets,
            probabilities=probabilities,
            grouped_odds=grouped_odds,
            model_name="binary_catboost_venue",
        )
        save_prediction_rows(prediction_rows)
    except Exception as log_e:
        print("[WARN] prediction log save failed:", log_e)


def _render(venue: str):
    if RaceController is None:
        return "RaceController import error", 500

    date = request.args.get("date") or _today()
    race_str = request.args.get("race")
    mode = request.args.get("mode")

    controller = RaceController()

    if not race_str:
        return render_template(
            "index.html",
            venue=venue,
            date=date,
            races=_blank_races(),
            selected_race=0,
        )

    race_no = int(race_str)

    try:
        entries, beforeinfo = _get_entries_and_beforeinfo(controller, venue, date, race_no)

        grouped_odds: Dict[str, Any] = {}
        best_bets: List[Dict[str, Any]] = []
        probabilities: Dict[str, float] = {}
        ev_result: Dict[str, float] = {}

        if mode == "full":
            grouped_odds, bundle = _get_full_bundle(controller, venue, date, race_no)

            raw_prob_map = bundle.get("prob_map", {}) or {}
            probabilities = _complete_probabilities(raw_prob_map)

            raw_odds_map = bundle.get("odds_map", {}) or {}
            grouped_flat_odds = _grouped_odds_to_flat_map(grouped_odds)

            merged_odds_map = dict(grouped_flat_odds)
            merged_odds_map.update(raw_odds_map)
            odds_map = _complete_odds_map(merged_odds_map)

            ev_result = _build_ev_map_from_prob_and_odds(probabilities, odds_map)
            best_bets = bundle.get("best_bets", []) or []

            _save_prediction_log(
                date=date,
                venue=venue,
                race_no=race_no,
                best_bets=best_bets,
                probabilities=probabilities,
                grouped_odds=grouped_odds,
            )

        races = _blank_races()
        races[race_no - 1]["entries"] = entries

        _debug_render_log(
            venue=venue,
            date=date,
            race_no=race_no,
            mode=mode,
            entries=entries,
            grouped_odds=grouped_odds,
            best_bets=best_bets,
            probabilities=probabilities,
            ev_result=ev_result,
        )

        return render_template(
            "index.html",
            venue=venue,
            date=date,
            races=races,
            selected_race=race_no,
            grouped_odds=grouped_odds,
            probabilities=probabilities,
            ev_result=ev_result,
            best_bets=best_bets,
            beforeinfo=beforeinfo,
            before_info=beforeinfo,
            beforeinfo_data=beforeinfo,
            mode=mode,
        )

    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] _render failed")
        print("=" * 80)
        print("venue:", venue)
        print("date :", date)
        print("race :", race_no)
        print("mode :", mode)
        traceback.print_exc()

        return render_template(
            "index.html",
            venue=venue,
            date=date,
            races=_blank_races(),
            selected_race=race_no,
            grouped_odds={},
            probabilities={},
            ev_result={},
            best_bets=[],
            beforeinfo={},
            error_message=str(e),
            mode=mode,
        )


@app.route("/")
def home():
    home_stats = _build_home_stats()
    return render_template(
        "home.html",
        date=_today(),
        home_stats=home_stats,
    )


@app.route("/ping")
def ping():
    now = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    return jsonify({
        "ok": True,
        "ts": now,
    })


@app.route("/marugame")
def marugame():
    return _render("丸亀")


@app.route("/toda")
def toda():
    return _render("戸田")


@app.route("/kojima")
def kojima():
    return _render("児島")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
