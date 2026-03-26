from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


CONFIG_JSON_PATH = Path("data/models/venue_buy_config.optimized.json")

BASE_VENUE_BUY_CONFIG: Dict[str, Dict[str, float]] = {
    "丸亀": {
        "min_prob": 0.018,
        "max_odds_cap": 55.0,
        "max_ev_cap": 2.2,
        "weight_prob": 0.68,
        "weight_ev": 0.22,
        "weight_odds": 0.10,
        "prob_exp": 1.45,
        "ev_exp": 0.70,
        "base_bonus": 0.66,
        "fixed_points": 5,
    },
    "児島": {
        "min_prob": 0.015,
        "max_odds_cap": 80.0,
        "max_ev_cap": 2.8,
        "weight_prob": 0.58,
        "weight_ev": 0.28,
        "weight_odds": 0.14,
        "prob_exp": 1.28,
        "ev_exp": 0.82,
        "base_bonus": 0.64,
        "fixed_points": 7,
    },
    "戸田": {
        "min_prob": 0.014,
        "max_odds_cap": 90.0,
        "max_ev_cap": 3.0,
        "weight_prob": 0.56,
        "weight_ev": 0.29,
        "weight_odds": 0.15,
        "prob_exp": 1.24,
        "ev_exp": 0.86,
        "base_bonus": 0.63,
        "fixed_points": 8,
    },
    "default": {
        "min_prob": 0.015,
        "max_odds_cap": 80.0,
        "max_ev_cap": 2.5,
        "weight_prob": 0.62,
        "weight_ev": 0.26,
        "weight_odds": 0.12,
        "prob_exp": 1.35,
        "ev_exp": 0.75,
        "base_bonus": 0.65,
        "fixed_points": 6,
    },
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _normalize_venue_name(venue: str) -> str:
    s = str(venue or "").strip()
    if "丸亀" in s:
        return "丸亀"
    if "児島" in s:
        return "児島"
    if "戸田" in s:
        return "戸田"
    return "default"


def _load_external_config() -> Dict[str, Dict[str, float]]:
    if not CONFIG_JSON_PATH.exists():
        return {}

    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            out[str(k)] = {str(kk): float(vv) for kk, vv in v.items()}
        return out
    except Exception:
        return {}


def _resolve_config(
    venue: str | None = None,
    min_prob: float | None = None,
    max_odds_cap: float | None = None,
    max_ev_cap: float | None = None,
    weight_prob: float | None = None,
    weight_ev: float | None = None,
    weight_odds: float | None = None,
) -> Dict[str, float]:
    key = _normalize_venue_name(venue or "")
    cfg = dict(BASE_VENUE_BUY_CONFIG.get(key, BASE_VENUE_BUY_CONFIG["default"]))

    ext = _load_external_config()
    if key in ext:
        cfg.update(ext[key])

    if min_prob is not None:
        cfg["min_prob"] = float(min_prob)
    if max_odds_cap is not None:
        cfg["max_odds_cap"] = float(max_odds_cap)
    if max_ev_cap is not None:
        cfg["max_ev_cap"] = float(max_ev_cap)
    if weight_prob is not None:
        cfg["weight_prob"] = float(weight_prob)
    if weight_ev is not None:
        cfg["weight_ev"] = float(weight_ev)
    if weight_odds is not None:
        cfg["weight_odds"] = float(weight_odds)

    return cfg


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _calc_high_odds_penalty(odds: float, venue: str = "default") -> float:
    v = _normalize_venue_name(venue)
    if odds <= 0:
        return 0.0

    if v == "丸亀":
        if odds <= 15:
            return 1.00
        if odds <= 30:
            return 0.95
        if odds <= 50:
            return 0.86
        if odds <= 80:
            return 0.70
        if odds <= 120:
            return 0.54
        return 0.38

    if v == "児島":
        if odds <= 20:
            return 1.00
        if odds <= 40:
            return 0.97
        if odds <= 70:
            return 0.90
        if odds <= 120:
            return 0.78
        if odds <= 200:
            return 0.63
        return 0.48

    if v == "戸田":
        if odds <= 20:
            return 1.00
        if odds <= 45:
            return 0.98
        if odds <= 80:
            return 0.91
        if odds <= 130:
            return 0.80
        if odds <= 220:
            return 0.66
        return 0.50

    if odds <= 20:
        return 1.00
    if odds <= 40:
        return 0.96
    if odds <= 80:
        return 0.88
    if odds <= 120:
        return 0.76
    if odds <= 200:
        return 0.62
    return 0.45


def _calc_low_odds_penalty(odds: float, venue: str = "default") -> float:
    v = _normalize_venue_name(venue)
    if odds <= 0:
        return 0.0

    if v == "丸亀":
        if odds < 4:
            return 0.75
        if odds < 6:
            return 0.86
        if odds < 8:
            return 0.94
        return 1.00

    if v == "児島":
        if odds < 4:
            return 0.82
        if odds < 6:
            return 0.90
        if odds < 8:
            return 0.96
        return 1.00

    if v == "戸田":
        if odds < 4:
            return 0.86
        if odds < 6:
            return 0.93
        if odds < 8:
            return 0.97
        return 1.00

    if odds < 4:
        return 0.78
    if odds < 6:
        return 0.88
    if odds < 8:
        return 0.95
    return 1.00


def _calc_odds_zone_bonus(odds: float, venue: str = "default") -> float:
    v = _normalize_venue_name(venue)
    if odds <= 0:
        return 0.0

    if v == "丸亀":
        if 8 <= odds <= 14:
            return 1.08
        if 14 < odds <= 24:
            return 1.10
        if 24 < odds <= 35:
            return 1.04
        return 1.00

    if v == "児島":
        if 10 <= odds <= 18:
            return 1.07
        if 18 < odds <= 35:
            return 1.12
        if 35 < odds <= 60:
            return 1.08
        return 1.00

    if v == "戸田":
        if 10 <= odds <= 20:
            return 1.05
        if 20 < odds <= 45:
            return 1.12
        if 45 < odds <= 70:
            return 1.10
        return 1.00

    if 8 <= odds <= 15:
        return 1.08
    if 15 < odds <= 30:
        return 1.12
    if 30 < odds <= 50:
        return 1.06
    return 1.00


def _calc_point_count(rows: List[Dict[str, Any]], venue: str = "default") -> int:
    cfg = _resolve_config(venue=venue)
    fixed_points = _safe_int(cfg.get("fixed_points", 6), 6)
    return max(3, min(16, fixed_points))


def select_best_bets(
    ai_preds: List[Dict[str, Any]],
    min_prob: float | None = None,
    max_odds_cap: float | None = None,
    max_ev_cap: float | None = None,
    top_n: int = 16,
    weight_prob: float | None = None,
    weight_ev: float | None = None,
    weight_odds: float | None = None,
    venue: str | None = None,
) -> List[Dict[str, Any]]:
    cfg = _resolve_config(
        venue=venue,
        min_prob=min_prob,
        max_odds_cap=max_odds_cap,
        max_ev_cap=max_ev_cap,
        weight_prob=weight_prob,
        weight_ev=weight_ev,
        weight_odds=weight_odds,
    )

    rows: List[Dict[str, Any]] = []

    for row in ai_preds:
        combo = str(row.get("combo", "")).strip()
        prob = _safe_float(row.get("prob", row.get("score", 0.0)), 0.0)
        odds = _safe_float(row.get("odds", row.get("odds_raw", 0.0)), 0.0)
        ev = _safe_float(row.get("ev", row.get("ev_raw", prob * odds)), 0.0)

        if not combo:
            continue
        if prob < cfg["min_prob"]:
            continue
        if odds <= 0:
            continue
        if ev <= 0:
            continue

        row2 = dict(row)
        row2["prob"] = prob
        row2["odds_raw"] = odds
        row2["ev_raw"] = ev
        row2["odds_capped"] = min(odds, cfg["max_odds_cap"])
        row2["ev_capped"] = min(ev, cfg["max_ev_cap"])
        rows.append(row2)

    if not rows:
        return []

    prob_vals = [r["prob"] for r in rows]
    odds_vals = [r["odds_capped"] for r in rows]
    ev_vals = [r["ev_capped"] for r in rows]

    prob_norms = _minmax_norm(prob_vals)
    odds_norms = _minmax_norm(odds_vals)
    ev_norms = _minmax_norm(ev_vals)

    for i, r in enumerate(rows):
        r["prob_norm"] = prob_norms[i]
        r["odds_norm"] = odds_norms[i]
        r["ev_norm"] = ev_norms[i]

        odds_raw = r["odds_raw"]
        high_odds_penalty = _calc_high_odds_penalty(odds_raw, venue=venue or "default")
        low_odds_penalty = _calc_low_odds_penalty(odds_raw, venue=venue or "default")
        odds_zone_bonus = _calc_odds_zone_bonus(odds_raw, venue=venue or "default")

        r["high_odds_penalty"] = high_odds_penalty
        r["low_odds_penalty"] = low_odds_penalty
        r["odds_zone_bonus"] = odds_zone_bonus

        linear_score = (
            (r["prob_norm"] * cfg["weight_prob"]) +
            (r["ev_norm"] * cfg["weight_ev"]) +
            (r["odds_norm"] * cfg["weight_odds"])
        )

        r["buy_score"] = (
            max(r["prob_norm"], 1e-9) ** cfg["prob_exp"] *
            max(r["ev_norm"], 1e-9) ** cfg["ev_exp"] *
            (cfg["base_bonus"] + linear_score) *
            odds_zone_bonus *
            high_odds_penalty *
            low_odds_penalty
        )

    rows.sort(
        key=lambda x: (
            float(x["buy_score"]),
            float(x["prob"]),
            float(x["ev_raw"]),
        ),
        reverse=True,
    )

    auto_points = _calc_point_count(rows=rows, venue=venue or "default")
    final_n = max(3, min(top_n, auto_points))

    selected = rows[:final_n]
    for idx, r in enumerate(selected, start=1):
        r["buy_rank"] = idx
        r["is_best_bet"] = True

    return selected
