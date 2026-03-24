from __future__ import annotations

import os
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

import numpy as np
from flask import Flask, render_template, request

try:
    from app.controller import RaceController
except Exception as e:
    print("[IMPORT_ERROR]", e)
    RaceController = None  # type: ignore

app = Flask(__name__, template_folder="templates", static_folder="static")


# =========================
# 基本ユーティリティ
# =========================
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _today() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")


def _blank_races() -> List[Dict[str, Any]]:
    return [{"race_no": i, "entries": []} for i in range(1, 13)]


def _debug_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"[DEBUG] {title}")
    print("=" * 80)


def _debug_print(label: str, value: Any) -> None:
    print(f"[DEBUG] {label}: {value}")


def _debug_sample_list(label: str, rows: Any, max_items: int = 2) -> None:
    if not isinstance(rows, list):
        print(f"[DEBUG] {label}: <not list> {type(rows)}")
        return

    print(f"[DEBUG] {label} len = {len(rows)}")
    for i, row in enumerate(rows[:max_items]):
        try:
            print(f"[DEBUG] {label}[{i}] keys = {list(row.keys()) if isinstance(row, dict) else type(row)}")
            print(f"[DEBUG] {label}[{i}] = {row}")
        except Exception as e:
            print(f"[DEBUG] {label}[{i}] print error = {e}")


def _debug_sample_dict(label: str, d: Any, max_items: int = 5) -> None:
    if not isinstance(d, dict):
        print(f"[DEBUG] {label}: <not dict> {type(d)}")
        return

    print(f"[DEBUG] {label} len = {len(d)}")
    try:
        items = list(d.items())[:max_items]
        for k, v in items:
            print(f"[DEBUG] {label}[{k}] = {v}")
    except Exception as e:
        print(f"[DEBUG] {label} print error = {e}")


def _normalize_beforeinfo_map(beforeinfo: Any) -> Dict[Any, Any]:
    """
    template 側で beforeinfo[lane] / beforeinfo["1"] 両対応に寄せる
    """
    if not isinstance(beforeinfo, dict):
        return {}

    out: Dict[Any, Any] = {}

    # 元データをそのまま保持
    for k, v in beforeinfo.items():
        out[k] = v
        try:
            ik = int(k)
            out[ik] = v
            out[str(ik)] = v
        except Exception:
            pass

    rows = beforeinfo.get("entries") or beforeinfo.get("data") or beforeinfo.get("rows") or []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue

            lane = (
                row.get("lane")
                or row.get("艇番")
                or row.get("艇")
                or row.get("枠")
                or row.get("teiban")
            )
            lane = _safe_int(lane, 0)
            if lane in range(1, 7):
                out[lane] = row
                out[str(lane)] = row

    return out


# =========================
# EV改善ロジック
# =========================
def _calc_ev(prob: float, odds: float) -> float:
    if odds <= 0:
        return 0.0

    odds_adj = np.log1p(odds)
    ev = prob * odds_adj

    if odds > 80:
        ev *= 0.7
    if odds > 150:
        ev *= 0.5

    return float(ev)


def _extract_best_bets_from_preds(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: List[Dict[str, Any]] = []

    for p in preds:
        if p.get("is_best_bet"):
            best.append(dict(p))

    best.sort(
        key=lambda x: (
            _safe_int(x.get("buy_rank", 999), 999),
            -_safe_float(x.get("buy_score", 0.0), 0.0),
        )
    )
    return best


# =========================
# メイン描画
# =========================
def _render(venue: str):
    _debug_title("START _render")

    if RaceController is None:
        print("[FATAL] RaceController import failed")
        return "RaceController の import に失敗しています。", 500

    date = request.args.get("date") or _today()
    race_str = request.args.get("race")
    mode = request.args.get("mode")

    _debug_print("venue", venue)
    _debug_print("date", date)
    _debug_print("race_str", race_str)
    _debug_print("mode", mode)

    controller = RaceController()

    if not race_str:
        _debug_print("render_mode", "race未指定 -> blank page")
        return render_template(
            "index.html",
            venue=venue,
            date=date,
            races=_blank_races(),
            selected_race=0,
            grouped_odds={},
            probabilities={},
            ev_result={},
            preds=[],
            best_bets=[],
            beforeinfo={},
        )

    race_no = _safe_int(race_str, 0)
    _debug_print("race_no", race_no)

    if race_no not in range(1, 13):
        _debug_print("invalid_race_no", race_no)
        return render_template(
            "index.html",
            venue=venue,
            date=date,
            races=_blank_races(),
            selected_race=0,
            grouped_odds={},
            probabilities={},
            ev_result={},
            preds=[],
            best_bets=[],
            beforeinfo={},
        )

    # =========================
    # 出走表
    # =========================
    _debug_title("ENTRIES FETCH")

    try:
        if venue == "戸田":
            entries = controller.get_entries_toda_race(date, race_no)
        elif venue == "児島":
            entries = controller.get_entries_kojima_race(date, race_no)
        else:
            entries = controller.get_entries_race(date, race_no)
    except Exception as e:
        print("[ERROR] entries fetch failed")
        traceback.print_exc()
        entries = []

    entries = [dict(e) for e in entries] if entries else []
    _debug_sample_list("entries(before enrich)", entries, max_items=3)

    try:
        if venue == "戸田":
            entries = controller.enrich_entries_toda(entries, date=date, race_no=race_no)
        elif venue == "児島":
            entries = controller.enrich_entries_kojima(entries, date=date, race_no=race_no)
        else:
            entries = controller.enrich_entries_marugame(entries, date=date, race_no=race_no)
    except Exception:
        print("[ERROR] enrich failed")
        traceback.print_exc()

    entries = [dict(e) for e in entries] if entries else []
    _debug_sample_list("entries(after enrich)", entries, max_items=3)

    # =========================
    # beforeinfo
    # =========================
    _debug_title("BEFOREINFO FETCH")

    try:
        if venue == "戸田":
            beforeinfo = controller.get_beforeinfo_only_toda(race_no=race_no, date=date)
        elif venue == "児島":
            beforeinfo = controller.get_beforeinfo_only_kojima(race_no=race_no, date=date)
        else:
            beforeinfo = controller.get_beforeinfo_only(race_no=race_no, date=date)
    except Exception:
        print("[ERROR] beforeinfo fetch failed")
        traceback.print_exc()
        beforeinfo = {}

    _debug_sample_dict("beforeinfo(raw)", beforeinfo, max_items=10)

    beforeinfo = _normalize_beforeinfo_map(beforeinfo)
    _debug_sample_dict("beforeinfo(normalized)", beforeinfo, max_items=10)

    # =========================
    # オッズ
    # =========================
    _debug_title("ODDS FETCH")

    try:
        if venue == "戸田":
            odds = controller.get_odds_only_toda(race_no=race_no, date=date)
        elif venue == "児島":
            odds = controller.get_odds_only_kojima(race_no=race_no, date=date)
        else:
            odds = controller.get_odds_only(race_no=race_no, date=date)
    except Exception:
        print("[ERROR] odds fetch failed")
        traceback.print_exc()
        odds = {}

    _debug_sample_dict("grouped_odds", odds, max_items=10)

    godata = odds.get("data", {}) if isinstance(odds, dict) else {}
    _debug_print("godata_exists", bool(godata))
    _debug_print("godata_cols", list(godata.keys()) if isinstance(godata, dict) else [])

    # =========================
    # AI
    # =========================
    _debug_title("AI PREDICT")

    try:
        if venue == "戸田":
            preds = controller.get_ai_predictions_toda(date, race_no, 120, True)
        elif venue == "児島":
            preds = controller.get_ai_predictions_kojima(date, race_no, 120, True)
        else:
            preds = controller.get_ai_predictions_marugame(date, race_no, 120, True)
    except Exception:
        print("[ERROR] AI predict failed")
        traceback.print_exc()
        preds = []

    preds = [dict(p) for p in preds] if preds else []
    _debug_sample_list("preds", preds, max_items=5)

    probabilities: Dict[str, float] = {}
    ev_result: Dict[str, float] = {}

    for p in preds:
        combo = str(p.get("combo", "")).strip()
        if not combo:
            continue

        prob = _safe_float(p.get("score", p.get("prob", 0.0)), 0.0)

        parts = combo.split("-")
        if len(parts) != 3:
            probabilities[combo] = prob
            ev_result[combo] = 0.0
            continue

        a = _safe_int(parts[0], 0)
        b = _safe_int(parts[1], 0)
        c = _safe_int(parts[2], 0)

        odd = 0.0
        if isinstance(godata, dict) and a in godata:
            col = godata.get(a, {}) or {}
            odd = _safe_float(col.get((b, c)) or col.get(f"{b}{c}") or 0, 0.0)

        ev = _calc_ev(prob, odd)

        probabilities[combo] = prob
        ev_result[combo] = ev

        if "odds" not in p or _safe_float(p.get("odds", 0.0), 0.0) <= 0:
            p["odds"] = odd
        if "ev" not in p or _safe_float(p.get("ev", 0.0), 0.0) <= 0:
            p["ev"] = ev

    _debug_print("probabilities_len(before norm)", len(probabilities))
    _debug_print("ev_result_len", len(ev_result))

    total = sum(probabilities.values())
    _debug_print("probabilities_total(before norm)", total)

    if total > 0:
        probabilities = {k: v / total for k, v in probabilities.items()}

    _debug_sample_dict("probabilities(after norm)", probabilities, max_items=10)
    _debug_sample_dict("ev_result", ev_result, max_items=10)

    # =========================
    # 最強買い目
    # =========================
    _debug_title("BEST BETS EXTRACT")

    best_bets = _extract_best_bets_from_preds(preds)
    _debug_sample_list("best_bets", best_bets, max_items=5)

    races = _blank_races()
    races[race_no - 1]["entries"] = entries

    _debug_title("RENDER TEMPLATE")
    _debug_print("selected_race", race_no)
    _debug_print("entries_len", len(entries))
    _debug_print("preds_len", len(preds))
    _debug_print("best_bets_len", len(best_bets))
    _debug_print("probabilities_len", len(probabilities))
    _debug_print("ev_result_len", len(ev_result))

    return render_template(
        "index.html",
        venue=venue,
        date=date,
        races=races,
        selected_race=race_no,
        grouped_odds=odds,
        probabilities=probabilities,
        ev_result=ev_result,
        preds=preds,
        best_bets=best_bets,
        beforeinfo=beforeinfo,
        mode=mode,
    )


# =========================
# ルーティング
# =========================
@app.route("/")
def home():
    return render_template("home.html", date=_today())


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
    app.run(host="0.0.0.0", port=port)
