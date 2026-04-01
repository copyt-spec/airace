# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIOR_PATH = PROJECT_ROOT / "data" / "models" / "lane_priors_by_venue.json"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def normalize_venue(v: Any) -> str:
    s = str(v or "").strip()
    if "丸亀" in s:
        return "丸亀"
    if "戸田" in s:
        return "戸田"
    if "児島" in s:
        return "児島"
    return s


def load_lane_priors() -> Dict[str, Any]:
    if not PRIOR_PATH.exists():
        return {}
    with open(PRIOR_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_lane_bias(
    prob_map: Dict[str, float],
    venue: str,
    alpha_first: float = 0.55,
    alpha_pair: float = 0.25,
) -> Dict[str, float]:
    """
    raw prob を会場別レーン傾向で軽く補正する
    """
    priors = load_lane_priors()
    venue_key = normalize_venue(venue)

    if venue_key not in priors:
        return prob_map

    first_prior = priors[venue_key].get("first", {})
    pair_prior = priors[venue_key].get("pair", {})

    adjusted: Dict[str, float] = {}

    for combo, raw_prob in prob_map.items():
        parts = str(combo).split("-")
        if len(parts) != 3:
            adjusted[combo] = _safe_float(raw_prob, 0.0)
            continue

        a, b, _ = parts

        p_first = _safe_float(first_prior.get(a, 1.0), 1.0)
        p_pair = _safe_float(pair_prior.get(f"{a}-{b}", 1.0), 1.0)

        value = (
            _safe_float(raw_prob, 0.0)
            * (p_first ** alpha_first)
            * (p_pair ** alpha_pair)
        )
        adjusted[combo] = value

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted
