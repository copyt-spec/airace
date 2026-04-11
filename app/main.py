from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Tuple

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
LOG_DIR = PROJECT_ROOT / "data" / "logs"

HOME_STATS_JSON_PATH = LOG_DIR / "home_stats_1y.json"
SIM_LIGHT_CSV_PATH = LOG_DIR / "prediction_results_sim_1y_light.csv"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _today() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")


def _ymd_to_html_date(ymd: str) -> str:
    s = str(ymd or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return ""


def _html_date_to_ymd(s: str) -> str:
    return str(s or "").replace("-", "").strip()


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
    if "住之江" in s:
        return "住之江"
    return s


def _empty_stats_map() -> Dict[str, Dict[str, Any]]:
    venue_order = ["丸亀", "戸田", "児島", "住之江"]
    empty_row = {
        "race_count": 0,
        "buy_count": 0,
        "hit_count": 0,
        "hit_rate": 0.0,
        "avg_points": 0.0,
        "total_bets": 0.0,
        "total_return": 0.0,
        "total_profit": 0.0,
        "roi": 0.0,
    }
    return {v: dict(empty_row) for v in venue_order}


def _load_home_stats_json() -> Dict[str, Dict[str, Any]]:
    if not HOME_STATS_JSON_PATH.exists():
        return _empty_stats_map()

    try:
        with open(HOME_STATS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("[WARN] failed to read home_stats json:", e)
        return _empty_stats_map()

    stats = data.get("stats", {})
    out = _empty_stats_map()
    for venue in ["丸亀", "戸田", "児島", "住之江"]:
        if venue in stats and isinstance(stats[venue], dict):
            out[venue].update(stats[venue])
    return out


def _load_sim_light_df() -> pd.DataFrame:
    if not SIM_LIGHT_CSV_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            SIM_LIGHT_CSV_PATH,
            dtype={
                "date": "string",
                "venue": "string",
                "race_no": "Int64",
                "is_selected": "Int64",
                "is_hit": "Int64",
            },
            low_memory=False,
        )
    except Exception as e:
        print("[WARN] failed to read sim light csv:", e)
        return pd.DataFrame()

    if df.empty:
        return df

    if "venue" in df.columns:
        df["venue_norm"] = df["venue"].astype(str).map(_normalize_venue_name)
    else:
        df["venue_norm"] = ""

    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)

    for col in ["is_selected", "is_hit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    for col in ["bet_cost_yen", "return_yen", "profit_yen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    return df


def _aggregate_stats_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out = _empty_stats_map()

    if df.empty:
        return out

    selected_df = df[df["is_selected"] == 1].copy()
    if selected_df.empty:
        return out

    for venue in ["丸亀", "戸田", "児島", "住之江"]:
        vdf = selected_df[selected_df["venue_norm"] == venue].copy()
        if vdf.empty:
            continue

        buy_count = int(len(vdf))
        hit_count = int((vdf["is_hit"] == 1).sum())
        total_bets = float(vdf["bet_cost_yen"].sum())
        total_return = float(vdf["return_yen"].sum())
        total_profit = total_return - total_bets
        hit_rate = hit_count / buy_count if buy_count > 0 else 0.0
        roi = total_return / total_bets if total_bets > 0 else 0.0

        race_count = 0
        if "date" in vdf.columns and "race_no" in vdf.columns:
            race_count = int(vdf[["date", "race_no"]].drop_duplicates().shape[0])

        avg_points = buy_count / race_count if race_count > 0 else 0.0

        out[venue] = {
            "race_count": race_count,
            "buy_count": buy_count,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "avg_points": avg_points,
            "total_bets": total_bets,
            "total_return": total_return,
            "total_profit": total_profit,
            "roi": roi,
        }

    return out


def _build_sim_stats(start_date: str, end_date: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    df = _load_sim_light_df()

    meta = {
        "start_date": start_date,
        "end_date": end_date,
        "available_min_date": "",
        "available_max_date": "",
        "row_count": 0,
    }

    if df.empty:
        return _aggregate_stats_from_df(df), meta

    if "date" in df.columns and not df["date"].empty:
        meta["available_min_date"] = str(df["date"].min())
        meta["available_max_date"] = str(df["date"].max())

    if start_date:
        df = df[df["date"] >= start_date].copy()
    if end_date:
        df = df[df["date"] <= end_date].copy()

    meta["row_count"] = int(len(df))
    stats = _aggregate_stats_from_df(df)
    return stats, meta


def _build_race_signal(
    probabilities: Dict[str, float],
    ev_result: Dict[str, float],
    best_bets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ranked_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True) if probabilities else []
    top_prob = float(ranked_probs[0][1]) if len(ranked_probs) >= 1 else 0.0
    second_prob = float(ranked_probs[1][1]) if len(ranked_probs) >= 2 else 0.0
    prob_gap = top_prob - second_prob

    top_ev = max([_safe_float(v, 0.0) for v in ev_result.values()], default=0.0)

    best_count = len(best_bets)
    avg_best_ev = 0.0
    if best_bets:
        avg_best_ev = sum(_safe_float(x.get("ev_raw", x.get("ev", 0.0)), 0.0) for x in best_bets) / len(best_bets)

    score = (
        top_prob * 100.0 * 0.42
        + prob_gap * 100.0 * 0.33
        + min(top_ev, 2.0) * 10.0 * 0.17
        + max(0.0, 7.0 - float(best_count)) * 0.18
    )

    if top_prob >= 0.11 and prob_gap >= 0.020 and top_ev >= 0.85:
        rank = "A"
        label = "勝負レース"
        tone = "good"
        comment = "上位の信頼度が高く、買い目も絞りやすいです。"
    elif top_prob >= 0.08 and prob_gap >= 0.012 and top_ev >= 0.65:
        rank = "B"
        label = "狙い目"
        tone = "good"
        comment = "十分狙えるレースです。買い目バランスも悪くありません。"
    elif top_prob >= 0.055 and top_ev >= 0.50:
        rank = "C"
        label = "様子見"
        tone = "warn"
        comment = "買えなくはないですが、過信はしにくいです。"
    else:
        rank = "D"
        label = "見送り寄り"
        tone = "bad"
        comment = "確率差か期待値の押しが弱く、無理に触らない方が無難です。"

    return {
        "rank": rank,
        "label": label,
        "tone": tone,
        "comment": comment,
        "score": round(score, 2),
        "top_prob": round(top_prob * 100.0, 2),
        "second_prob": round(second_prob * 100.0, 2),
        "prob_gap": round(prob_gap * 100.0, 2),
        "top_ev": round(top_ev, 2),
        "avg_best_ev": round(avg_best_ev, 2),
        "best_count": best_count,
    }


def _get_entries_and_beforeinfo(controller: RaceController, venue: str, date: str, race_no: int):
    if venue == "戸田":
        entries = controller.get_entries_toda_race(date, race_no)
        entries = controller.enrich_entries_toda(entries, date=date, race_no=race_no)
        beforeinfo = controller.get_beforeinfo_only_toda(race_no=race_no, date=date)

    elif venue == "児島":
        entries = controller.get_entries_kojima_race(date, race_no)
        entries = controller.enrich_entries_kojima(entries, date=date, race_no=race_no)
        beforeinfo = controller.get_beforeinfo_only_kojima(race_no=race_no, date=date)

    elif venue == "住之江":
        entries = controller.get_entries_suminoe_race(date, race_no)
        entries = controller.enrich_entries_suminoe(entries, date=date, race_no=race_no)
        beforeinfo = controller.get_beforeinfo_only_suminoe(race_no=race_no, date=date)

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

    elif venue == "住之江":
        grouped_odds = controller.get_odds_only_suminoe(race_no=race_no, date=date)
        bundle = controller.get_ai_prediction_bundle_suminoe(date, race_no, top_n=20, with_odds=True)

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
        race_signal: Dict[str, Any] = {}

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

            race_signal = _build_race_signal(
                probabilities=probabilities,
                ev_result=ev_result,
                best_bets=best_bets,
            )

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
            race_signal=race_signal,
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
            race_signal={},
            beforeinfo={},
            error_message=str(e),
            mode=mode,
        )


@app.route("/")
def home():
    home_stats = _load_home_stats_json()
    return render_template(
        "home.html",
        date=_today(),
        home_stats=home_stats,
    )


@app.route("/sim")
def sim_stats():
    df = _load_sim_light_df()

    available_min = ""
    available_max = ""
    if not df.empty and "date" in df.columns:
        available_min = str(df["date"].min())
        available_max = str(df["date"].max())

    start_arg = request.args.get("start", "").strip()
    end_arg = request.args.get("end", "").strip()

    start_date = _html_date_to_ymd(start_arg) if start_arg else available_min
    end_date = _html_date_to_ymd(end_arg) if end_arg else available_max

    stats, meta = _build_sim_stats(start_date=start_date, end_date=end_date)

    return render_template(
        "sim_stats.html",
        sim_stats=stats,
        start_date_html=_ymd_to_html_date(start_date),
        end_date_html=_ymd_to_html_date(end_date),
        available_min_html=_ymd_to_html_date(meta.get("available_min_date", "")),
        available_max_html=_ymd_to_html_date(meta.get("available_max_date", "")),
        sim_meta=meta,
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


@app.route("/suminoe")
def suminoe():
    return _render("住之江")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
