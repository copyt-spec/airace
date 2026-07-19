from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timedelta
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
from engine.race_scenario import build_race_scenario
from engine.venue_registry import (
    VENUE_MASTER,
    VENUE_ORDER,
    get_venue_name_by_key,
    normalize_venue_name,
)


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
    return normalize_venue_name(v)


def _resolve_model_mode(v: str | None) -> str:
    s = str(v or "").strip().lower()
    if s not in {"venue", "global_auto"}:
        return "venue"
    return s


def _empty_stats_map() -> Dict[str, Dict[str, Any]]:
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
    return {v: dict(empty_row) for v in VENUE_ORDER}


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
    for venue in VENUE_ORDER:
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

    for col in ["bet_cost_yen", "return_yen", "profit_yen", "prob", "odds", "top1_prob", "prob_gap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["odds_band", "conf_tier", "combo"]:
        if col not in df.columns:
            df[col] = ""

    return df


def _aggregate_stats_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out = _empty_stats_map()

    if df.empty:
        return out

    selected_df = df[df["is_selected"] == 1].copy()
    if selected_df.empty:
        return out

    for venue in VENUE_ORDER:
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


def _build_daily_timeseries(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    work = df.copy()

    if "date" not in work.columns:
        return []

    for col in ["bet_cost_yen", "return_yen", "profit_yen", "is_selected", "is_hit"]:
        if col not in work.columns:
            work[col] = 0

    work["bet_cost_yen"] = pd.to_numeric(work["bet_cost_yen"], errors="coerce").fillna(0.0)
    work["return_yen"] = pd.to_numeric(work["return_yen"], errors="coerce").fillna(0.0)
    work["profit_yen"] = pd.to_numeric(work["profit_yen"], errors="coerce").fillna(0.0)
    work["is_selected"] = pd.to_numeric(work["is_selected"], errors="coerce").fillna(0).astype(int)
    work["is_hit"] = pd.to_numeric(work["is_hit"], errors="coerce").fillna(0).astype(int)

    selected = work[work["is_selected"] == 1].copy()
    if selected.empty:
        return []

    g = (
        selected.groupby("date", as_index=False)
        .agg(
            buy_count=("is_selected", "sum"),
            hit_count=("is_hit", "sum"),
            total_bets=("bet_cost_yen", "sum"),
            total_return=("return_yen", "sum"),
            total_profit=("profit_yen", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    g["roi"] = g.apply(
        lambda r: float(r["total_return"] / r["total_bets"]) if r["total_bets"] > 0 else 0.0,
        axis=1,
    )
    g["hit_rate"] = g.apply(
        lambda r: float(r["hit_count"] / r["buy_count"]) if r["buy_count"] > 0 else 0.0,
        axis=1,
    )
    g["cum_profit"] = g["total_profit"].cumsum()
    g["cum_bets"] = g["total_bets"].cumsum()
    g["cum_return"] = g["total_return"].cumsum()

    rows: List[Dict[str, Any]] = []
    for _, row in g.iterrows():
        rows.append({
            "date": str(row["date"]),
            "buy_count": int(row["buy_count"]),
            "hit_count": int(row["hit_count"]),
            "total_bets": float(row["total_bets"]),
            "total_return": float(row["total_return"]),
            "total_profit": float(row["total_profit"]),
            "roi": float(row["roi"]),
            "hit_rate": float(row["hit_rate"]),
            "cum_profit": float(row["cum_profit"]),
            "cum_bets": float(row["cum_bets"]),
            "cum_return": float(row["cum_return"]),
        })

    return rows


def _build_venue_daily_timeseries(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}

    if df.empty or "venue_norm" not in df.columns:
        return out

    for venue in VENUE_ORDER:
        vdf = df[df["venue_norm"] == venue].copy()
        out[venue] = _build_daily_timeseries(vdf)

    return out


def _build_venue_summary_cards(df: pd.DataFrame) -> List[Dict[str, Any]]:
    stats = _aggregate_stats_from_df(df)
    rows: List[Dict[str, Any]] = []

    for venue in VENUE_ORDER:
        s = stats.get(venue, {})
        rows.append({
            "venue": venue,
            "race_count": int(s.get("race_count", 0) or 0),
            "buy_count": int(s.get("buy_count", 0) or 0),
            "hit_count": int(s.get("hit_count", 0) or 0),
            "hit_rate": float(s.get("hit_rate", 0.0) or 0.0),
            "avg_points": float(s.get("avg_points", 0.0) or 0.0),
            "total_bets": float(s.get("total_bets", 0.0) or 0.0),
            "total_return": float(s.get("total_return", 0.0) or 0.0),
            "total_profit": float(s.get("total_profit", 0.0) or 0.0),
            "roi": float(s.get("roi", 0.0) or 0.0),
        })
    return rows


CONF_TIER_ORDER = ["A", "B", "C", "D"]
CONF_TIER_LABEL = {
    "A": "A(近似・勝負レース相当)",
    "B": "B(近似・狙い目相当)",
    "C": "C(近似・様子見相当)",
    "D": "D(近似・見送り寄り相当)",
}

ODDS_BAND_ORDER = ["〜10倍", "10〜30倍", "30〜50倍", "50〜100倍", "100倍〜"]


def _breakdown_by_key(df: pd.DataFrame, key: str, order: List[str], label_map: Dict[str, str] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty or key not in df.columns:
        return rows

    work = df.copy()
    for col in ["bet_cost_yen", "return_yen", "profit_yen", "is_hit"]:
        if col not in work.columns:
            work[col] = 0

    present = [v for v in order if v in set(work[key].astype(str))]
    for v in present:
        sub = work[work[key].astype(str) == v]
        buy_count = int(len(sub))
        hit_count = int((sub["is_hit"] == 1).sum())
        total_bets = float(sub["bet_cost_yen"].sum())
        total_return = float(sub["return_yen"].sum())
        total_profit = total_return - total_bets
        hit_rate = hit_count / buy_count if buy_count > 0 else 0.0
        roi = total_return / total_bets if total_bets > 0 else 0.0
        rows.append({
            "key": v,
            "label": (label_map or {}).get(v, v),
            "buy_count": buy_count,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "total_bets": total_bets,
            "total_return": total_return,
            "total_profit": total_profit,
            "roi": roi,
        })
    return rows


def _build_conf_tier_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return _breakdown_by_key(df, "conf_tier", CONF_TIER_ORDER, CONF_TIER_LABEL)


def _build_odds_band_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return _breakdown_by_key(df, "odds_band", ODDS_BAND_ORDER)


KELLY_BANKROLL_DAILY_CSV = LOG_DIR / "kelly_bankroll_daily.csv"


def _load_kelly_bankroll_daily(start_date: str = "", end_date: str = "") -> Dict[str, Any]:
    if not KELLY_BANKROLL_DAILY_CSV.exists():
        return {"rows": [], "available": False}

    try:
        df = pd.read_csv(KELLY_BANKROLL_DAILY_CSV, low_memory=False)
    except Exception as e:
        print("[WARN] failed to read kelly bankroll daily csv:", e)
        return {"rows": [], "available": False}

    if df.empty:
        return {"rows": [], "available": False}

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df = df.sort_values("date").reset_index(drop=True)

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    if df.empty:
        return {"rows": [], "available": False}

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append({
            "date": str(r["date"]),
            "flat_bankroll": float(r.get("flat_bankroll", 0.0) or 0.0),
            "kelly_bankroll": float(r.get("kelly_bankroll", 0.0) or 0.0),
        })

    return {
        "rows": rows,
        "available": True,
        "initial_bankroll": 100000,
        "final_flat_bankroll": rows[-1]["flat_bankroll"] if rows else 0.0,
        "final_kelly_bankroll": rows[-1]["kelly_bankroll"] if rows else 0.0,
    }


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


def _save_prediction_log(
    *,
    date: str,
    venue: str,
    race_no: int,
    best_bets: List[Dict[str, Any]],
    probabilities: Dict[str, float],
    grouped_odds: Dict[str, Any],
    model_mode: str,
) -> None:
    try:
        prediction_rows = build_prediction_rows(
            date=date,
            venue=venue,
            race_no=race_no,
            best_bets=best_bets,
            probabilities=probabilities,
            grouped_odds=grouped_odds,
            model_name=f"binary_catboost_{model_mode}",
        )
        save_prediction_rows(prediction_rows)
    except Exception as log_e:
        print("[WARN] prediction log save failed:", log_e)


def _render_venue_page(venue_name: str):
    if RaceController is None:
        return "RaceController import error", 500

    venue_name = _normalize_venue_name(venue_name)

    date = request.args.get("date") or _today()
    race_str = request.args.get("race")
    mode = request.args.get("mode")
    model_mode = _resolve_model_mode(request.args.get("model_mode"))

    controller = RaceController(model_mode=model_mode)

    if not race_str:
        return render_template(
            "index.html",
            venue=venue_name,
            venue_key=VENUE_MASTER[venue_name]["venue_key"],
            date=date,
            races=_blank_races(),
            selected_race=0,
            all_venues=VENUE_ORDER,
            venue_config=VENUE_MASTER,
            model_mode=model_mode,
        )

    race_no = int(race_str)

    try:
        entries = controller.get_entries_race(venue_name, date, race_no)
        entries = controller.enrich_entries(venue_name, entries, date=date, race_no=race_no)
        entries = [dict(e) for e in entries]

        try:
            beforeinfo = controller.get_beforeinfo_only(venue_name, race_no=race_no, date=date)
        except Exception:
            beforeinfo = {}

        grouped_odds: Dict[str, Any] = {}
        best_bets: List[Dict[str, Any]] = []
        probabilities: Dict[str, float] = {}
        ev_result: Dict[str, float] = {}
        race_signal: Dict[str, Any] = {}
        formation_options: List[Dict[str, Any]] = []
        resolved_model_mode = model_mode

        if mode == "full":
            grouped_odds = controller.get_odds_only(venue_name, race_no=race_no, date=date)
            bundle = controller.get_ai_prediction_bundle(
                venue_name=venue_name,
                date=date,
                race_no=race_no,
                top_n=20,
                with_odds=True,
                model_mode=model_mode,
            ) or {}

            resolved_model_mode = _resolve_model_mode(bundle.get("model_mode", model_mode))

            raw_prob_map = bundle.get("prob_map", {}) or {}
            probabilities = _complete_probabilities(raw_prob_map)

            raw_odds_map = bundle.get("odds_map", {}) or {}
            grouped_flat_odds = _grouped_odds_to_flat_map(grouped_odds)

            merged_odds_map = dict(grouped_flat_odds)
            merged_odds_map.update(raw_odds_map)
            odds_map = _complete_odds_map(merged_odds_map)

            ev_result = _build_ev_map_from_prob_and_odds(probabilities, odds_map)
            best_bets = bundle.get("best_bets", []) or []
            formation_options = bundle.get("formation_options", []) or []

            race_signal = _build_race_signal(
                probabilities=probabilities,
                ev_result=ev_result,
                best_bets=best_bets,
            )

            _save_prediction_log(
                date=date,
                venue=venue_name,
                race_no=race_no,
                best_bets=best_bets,
                probabilities=probabilities,
                grouped_odds=grouped_odds,
                model_mode=resolved_model_mode,
            )

        races = _blank_races()
        races[race_no - 1]["entries"] = entries

        return render_template(
            "index.html",
            venue=venue_name,
            venue_key=VENUE_MASTER[venue_name]["venue_key"],
            date=date,
            races=races,
            selected_race=race_no,
            grouped_odds=grouped_odds,
            probabilities=probabilities,
            ev_result=ev_result,
            best_bets=best_bets,
            race_signal=race_signal,
            formation_options=formation_options,
            beforeinfo=beforeinfo,
            before_info=beforeinfo,
            beforeinfo_data=beforeinfo,
            mode=mode,
            all_venues=VENUE_ORDER,
            venue_config=VENUE_MASTER,
            model_mode=resolved_model_mode,
        )

    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] _render_venue_page failed")
        print("=" * 80)
        print("venue     :", venue_name)
        print("date      :", date)
        print("race      :", race_no)
        print("mode      :", mode)
        print("model_mode:", model_mode)
        traceback.print_exc()

        return render_template(
            "index.html",
            venue=venue_name,
            venue_key=VENUE_MASTER[venue_name]["venue_key"],
            date=date,
            races=_blank_races(),
            selected_race=race_no,
            grouped_odds={},
            probabilities={},
            ev_result={},
            best_bets=[],
            race_signal={},
            formation_options=[],
            beforeinfo={},
            error_message=str(e),
            mode=mode,
            all_venues=VENUE_ORDER,
            venue_config=VENUE_MASTER,
            model_mode=model_mode,
        )


@app.route("/")
def home():
    home_stats = _load_home_stats_json()
    return render_template(
        "home.html",
        date=_today(),
        home_stats=home_stats,
        all_venues=VENUE_ORDER,
        venue_config=VENUE_MASTER,
    )


MODEL_COMPARISON_CSV = LOG_DIR / "model_comparison_all_venues_holdout.csv"


def _load_model_comparison() -> Dict[str, Any]:
    """
    scripts/compare_racer_stats_holdout_all_venues.py が出力する
    data/logs/model_comparison_all_venues_holdout.csv (全24会場、旧モデル
    with_racer_no vs 新モデルwith_racer_stats+天候修正の、真のholdout期間
    ROI比較)を読み込む。実オッズが無い会場はオフラインのlogloss/aucのみの
    行になる(ROI未検証)。
    """
    empty = {"available": False, "validated_rows": [], "offline_rows": [], "validated_count": 0, "offline_count": 0}

    if not MODEL_COMPARISON_CSV.exists():
        return empty

    try:
        df = pd.read_csv(MODEL_COMPARISON_CSV, low_memory=False)
    except Exception as e:
        print("[WARN] failed to read model comparison csv:", e)
        return empty

    if df.empty:
        return empty

    validated_rows: List[Dict[str, Any]] = []
    offline_rows: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        venue = str(r.get("venue", "")).strip()
        real_odds = str(r.get("real_odds_available", "")).strip().lower() in ("true", "1", "1.0")

        if real_odds:
            races_new = int(_safe_float(r.get("races_new", 0), 0.0))
            valid_start = str(r.get("valid_start", "")).replace(".0", "", 1) if str(r.get("valid_start", "")).endswith(".0") else str(r.get("valid_start", ""))
            valid_end = str(r.get("valid_end", "")).replace(".0", "", 1) if str(r.get("valid_end", "")).endswith(".0") else str(r.get("valid_end", ""))
            validated_rows.append({
                "venue": venue,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "races": races_new,
                "roi_old_pct": _safe_float(r.get("roi_old_pct", 0.0), 0.0),
                "roi_new_pct": _safe_float(r.get("roi_new_pct", 0.0), 0.0),
                "roi_delta_pct": _safe_float(r.get("roi_delta_pct", 0.0), 0.0),
                "low_confidence": races_new < 50,
            })
        else:
            offline_rows.append({
                "venue": venue,
                "note": str(r.get("note", "")),
                "offline_valid_logloss": _safe_float(r.get("offline_valid_logloss", 0.0), 0.0),
                "offline_valid_auc": _safe_float(r.get("offline_valid_auc", 0.0), 0.0),
            })

    validated_rows.sort(key=lambda x: x["races"], reverse=True)
    offline_rows.sort(key=lambda x: VENUE_ORDER.index(x["venue"]) if x["venue"] in VENUE_ORDER else 999)

    return {
        "available": True,
        "validated_rows": validated_rows,
        "offline_rows": offline_rows,
        "validated_count": len(validated_rows),
        "offline_count": len(offline_rows),
    }


PREDICTIONS_CSV_PATH = LOG_DIR / "predictions.csv"


def _load_daily_crawl_status(recent_days: int = 14) -> Dict[str, Any]:
    """
    data/logs/predictions.csv を軽量(date, venueの2列のみ)に読み込み、
    daily-prediction-crawl(全24会場を毎日自動巡回してmode=fullで予測ログを
    残す仕組み)が実際に直近も動いているかを可視化する。
    このFlaskアプリが動いている環境(ローカルMac or Render)ごとに
    predictions.csvは別ファイルなので、この状況もその環境ごとに変わる点に注意。
    """
    empty = {"available": False}

    if not PREDICTIONS_CSV_PATH.exists():
        return empty

    try:
        df = pd.read_csv(PREDICTIONS_CSV_PATH, usecols=["date", "venue"], low_memory=False)
    except Exception as e:
        print("[WARN] failed to read predictions.csv for daily crawl status:", e)
        return empty

    if df.empty:
        return empty

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df["venue"] = df["venue"].astype(str).str.strip().map(normalize_venue_name)

    total_rows = int(len(df))
    latest_date = str(df["date"].max())

    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    today_str = now_jst.strftime("%Y%m%d")
    cutoff_date = (now_jst - timedelta(days=recent_days)).strftime("%Y%m%d")

    recent_df = df[df["date"] >= cutoff_date]
    covered_venues = sorted(
        {v for v in recent_df["venue"].unique().tolist() if v in VENUE_ORDER},
        key=lambda v: VENUE_ORDER.index(v),
    )
    missing_venues = [v for v in VENUE_ORDER if v not in covered_venues]

    return {
        "available": True,
        "total_rows": total_rows,
        "latest_date": latest_date,
        "today": today_str,
        "recent_days": recent_days,
        "covered_venue_count": len(covered_venues),
        "total_venue_count": len(VENUE_ORDER),
        "missing_venues": missing_venues,
        "is_stale": latest_date < cutoff_date,
    }


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
    venue_arg = request.args.get("venue", "").strip()

    start_date = _html_date_to_ymd(start_arg) if start_arg else available_min
    end_date = _html_date_to_ymd(end_arg) if end_arg else available_max

    filtered = df.copy()

    if start_date:
        filtered = filtered[filtered["date"] >= start_date].copy()
    if end_date:
        filtered = filtered[filtered["date"] <= end_date].copy()

    if venue_arg and venue_arg in VENUE_ORDER:
        filtered = filtered[filtered["venue_norm"] == venue_arg].copy()

    stats = _aggregate_stats_from_df(filtered)

    meta = {
        "start_date": start_date,
        "end_date": end_date,
        "available_min_date": available_min,
        "available_max_date": available_max,
        "row_count": int(len(filtered)),
        "selected_venue": venue_arg,
    }

    daily_all = _build_daily_timeseries(filtered)
    venue_daily = _build_venue_daily_timeseries(filtered)
    summary_cards = _build_venue_summary_cards(filtered)
    conf_tier_breakdown = _build_conf_tier_breakdown(filtered)
    odds_band_breakdown = _build_odds_band_breakdown(filtered)
    kelly_bankroll = _load_kelly_bankroll_daily(start_date, end_date)
    model_comparison = _load_model_comparison()
    daily_crawl_status = _load_daily_crawl_status()

    return render_template(
        "sim_stats.html",
        sim_stats=stats,
        sim_summary_cards=summary_cards,
        sim_daily_all=daily_all,
        sim_daily_by_venue=venue_daily,
        conf_tier_breakdown=conf_tier_breakdown,
        odds_band_breakdown=odds_band_breakdown,
        kelly_bankroll=kelly_bankroll,
        model_comparison=model_comparison,
        daily_crawl_status=daily_crawl_status,
        start_date_html=_ymd_to_html_date(start_date),
        end_date_html=_ymd_to_html_date(end_date),
        available_min_html=_ymd_to_html_date(meta.get("available_min_date", "")),
        available_max_html=_ymd_to_html_date(meta.get("available_max_date", "")),
        sim_meta=meta,
        selected_venue=venue_arg,
        all_venues=VENUE_ORDER,
    )


@app.route("/ping")
def ping():
    now = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    return jsonify({
        "ok": True,
        "ts": now,
    })


@app.route("/venue/<venue_key>")
def venue_page(venue_key: str):
    try:
        venue_name = get_venue_name_by_key(venue_key)
    except Exception:
        return f"Unknown venue_key: {venue_key}", 404

    return _render_venue_page(venue_name)


def _render_scenario_page(venue_name: str):
    if RaceController is None:
        return "RaceController import error", 500

    venue_name = _normalize_venue_name(venue_name)

    date = request.args.get("date") or _today()
    race_str = request.args.get("race")
    model_mode = _resolve_model_mode(request.args.get("model_mode"))

    if not race_str:
        return render_template(
            "scenario.html",
            venue=venue_name,
            venue_key=VENUE_MASTER[venue_name]["venue_key"],
            date=date,
            selected_race=0,
            scenario={"available": False},
            error_message="レースを選択してください(?race=N を指定)",
            all_venues=VENUE_ORDER,
            venue_config=VENUE_MASTER,
            model_mode=model_mode,
        )

    race_no = int(race_str)
    controller = RaceController(model_mode=model_mode)

    try:
        entries = controller.get_entries_race(venue_name, date, race_no)
        entries = controller.enrich_entries(venue_name, entries, date=date, race_no=race_no)
        entries = [dict(e) for e in entries]

        try:
            beforeinfo = controller.get_beforeinfo_only(venue_name, race_no=race_no, date=date)
        except Exception:
            beforeinfo = {}

        bundle = controller.get_ai_prediction_bundle(
            venue_name=venue_name,
            date=date,
            race_no=race_no,
            top_n=20,
            with_odds=True,
            model_mode=model_mode,
        ) or {}

        raw_prob_map = bundle.get("prob_map", {}) or {}
        probabilities = _complete_probabilities(raw_prob_map)

        scenario = build_race_scenario(entries, beforeinfo, probabilities, venue=venue_name)

        return render_template(
            "scenario.html",
            venue=venue_name,
            venue_key=VENUE_MASTER[venue_name]["venue_key"],
            date=date,
            selected_race=race_no,
            scenario=scenario,
            error_message=None,
            all_venues=VENUE_ORDER,
            venue_config=VENUE_MASTER,
            model_mode=model_mode,
        )

    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] _render_scenario_page failed")
        print("=" * 80)
        print("venue:", venue_name, "date:", date, "race:", race_no)
        traceback.print_exc()

        return render_template(
            "scenario.html",
            venue=venue_name,
            venue_key=VENUE_MASTER[venue_name]["venue_key"],
            date=date,
            selected_race=race_no,
            scenario={"available": False},
            error_message=str(e),
            all_venues=VENUE_ORDER,
            venue_config=VENUE_MASTER,
            model_mode=model_mode,
        )


@app.route("/venue/<venue_key>/scenario")
def venue_scenario_page(venue_key: str):
    try:
        venue_name = get_venue_name_by_key(venue_key)
    except Exception:
        return f"Unknown venue_key: {venue_key}", 404

    return _render_scenario_page(venue_name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
