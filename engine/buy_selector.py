from __future__ import annotations

import json
import math
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
        "min_points": 1,
        "max_points": 12,
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
        "min_points": 1,
        "max_points": 12,
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
        "min_points": 1,
        "max_points": 12,
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
        "min_points": 1,
        "max_points": 12,
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
            out[str(k)] = {}
            for kk, vv in v.items():
                try:
                    out[str(k)][str(kk)] = float(vv)
                except Exception:
                    continue
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

    if "fixed_points" in cfg:
        fp = _safe_int(cfg["fixed_points"], 6)
        cfg["min_points"] = float(fp)
        cfg["max_points"] = float(fp)

    if "min_points" not in cfg:
        cfg["min_points"] = 3.0
    if "max_points" not in cfg:
        cfg["max_points"] = 12.0

    # 必ず 3〜12 に収める
    cfg["min_points"] = float(max(3, min(12, _safe_int(cfg["min_points"], 3))))
    cfg["max_points"] = float(max(3, min(12, _safe_int(cfg["max_points"], 12))))
    if cfg["max_points"] < cfg["min_points"]:
        cfg["max_points"] = cfg["min_points"]

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


def _calc_distribution_entropy(probs: List[float]) -> float:
    vals = [p for p in probs if p > 0]
    if not vals:
        return 1.0

    total = sum(vals)
    if total <= 0:
        return 1.0

    norm = [p / total for p in vals]
    h = 0.0
    for p in norm:
        h -= p * math.log(p + 1e-12)

    max_h = math.log(len(norm) + 1e-12)
    if max_h <= 0:
        return 0.0

    score = h / max_h
    return max(0.0, min(1.0, score))


def _calc_dynamic_points(rows: List[Dict[str, Any]], venue: str, cfg: Dict[str, float]) -> int:
    """
    必ず 3〜12点の範囲で可変
    """
    min_points = max(3, min(12, _safe_int(cfg.get("min_points", 3), 3)))
    max_points = max(min_points, min(12, _safe_int(cfg.get("max_points", 12), 12)))

    probs = sorted([_safe_float(r.get("prob", 0.0), 0.0) for r in rows], reverse=True)
    if not probs:
        return min_points

    top1 = probs[0] if len(probs) >= 1 else 0.0
    top2 = probs[1] if len(probs) >= 2 else 0.0
    top3_sum = sum(probs[:3])
    top5_sum = sum(probs[:5])
    entropy = _calc_distribution_entropy(probs[:10])

    spread_score = 0.0

    # 本命が強いほど絞る
    spread_score += (0.12 - min(top1, 0.12)) / 0.12 * 0.42

    # 上位3つが薄いほど広げる
    spread_score += (0.24 - min(top3_sum, 0.24)) / 0.24 * 0.22

    # 上位5つが薄いほど広げる
    spread_score += (0.36 - min(top5_sum, 0.36)) / 0.36 * 0.14

    # 1位と2位の差が小さいほど広げる
    gap12 = max(0.0, top1 - top2)
    spread_score += (0.03 - min(gap12, 0.03)) / 0.03 * 0.08

    # 分布ばらつき
    spread_score += entropy * 0.28

    v = _normalize_venue_name(venue)
    if v == "丸亀":
        spread_score *= 0.82
    elif v == "児島":
        spread_score *= 1.00
    elif v == "戸田":
        spread_score *= 1.08

    spread_score = max(0.0, min(1.0, spread_score))

    points = min_points + round((max_points - min_points) * spread_score)

    # 強本命時はさらに絞る
    if top1 >= 0.10 and top3_sum >= 0.23:
        points = min(points, 3)
    elif top1 >= 0.08 and top5_sum >= 0.34:
        points = min(points, max(4, min_points))

    # 超分散時は少し増やす
    if entropy >= 0.88 and top1 <= 0.05:
        points = min(max_points, points + 1)

    return max(3, min(12, points))


def _prepare_candidate_rows(
    ai_preds: List[Dict[str, Any]],
    cfg: Dict[str, float],
    use_min_prob: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in ai_preds:
        combo = str(row.get("combo", "")).strip()
        prob = _safe_float(row.get("prob", row.get("score", 0.0)), 0.0)
        odds = _safe_float(row.get("odds", row.get("odds_raw", 0.0)), 0.0)
        ev = _safe_float(row.get("ev", row.get("ev_raw", prob * odds)), 0.0)

        if not combo:
            continue
        if use_min_prob and prob < cfg["min_prob"]:
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

    return rows


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

    # まず min_prob ありで候補作成
    rows = _prepare_candidate_rows(ai_preds, cfg, use_min_prob=True)

    # 候補が少なすぎたら min_prob 無視で補充用候補を使う
    backup_rows = _prepare_candidate_rows(ai_preds, cfg, use_min_prob=False)

    if not rows and not backup_rows:
        return []

    if not rows:
        rows = backup_rows[:]

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

    dynamic_points = _calc_dynamic_points(
        rows=rows,
        venue=venue or "default",
        cfg=cfg,
    )
    final_n = max(3, min(12, min(top_n, dynamic_points)))

    # 候補不足なら backup から補充
    if len(rows) < final_n:
        existing = {str(r.get("combo", "")) for r in rows}
        backup_rows_sorted = sorted(
            backup_rows,
            key=lambda x: _safe_float(x.get("prob", 0.0), 0.0),
            reverse=True,
        )

        for br in backup_rows_sorted:
            combo = str(br.get("combo", ""))
            if combo in existing:
                continue

            # 補充行にも最低限の派生値を持たせる
            prob = _safe_float(br.get("prob", br.get("score", 0.0)), 0.0)
            odds = _safe_float(br.get("odds_raw", br.get("odds", 0.0)), 0.0)
            ev = _safe_float(br.get("ev_raw", br.get("ev", prob * odds)), 0.0)

            row2 = dict(br)
            row2["prob"] = prob
            row2["odds_raw"] = odds
            row2["ev_raw"] = ev
            row2["buy_score"] = prob  # 補充時はprob順でOK
            rows.append(row2)
            existing.add(combo)

            if len(rows) >= final_n:
                break

        rows.sort(
            key=lambda x: (
                float(x.get("buy_score", 0.0)),
                float(x.get("prob", 0.0)),
                float(x.get("ev_raw", 0.0)),
            ),
            reverse=True,
        )

    selected = rows[:final_n]

    for idx, r in enumerate(selected, start=1):
        r["buy_rank"] = idx
        r["is_best_bet"] = True

    return selected
