from __future__ import annotations

from typing import List, Dict, Any


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []

    vmin = min(values)
    vmax = max(values)

    if vmax <= vmin:
        return [0.0 for _ in values]

    return [(v - vmin) / (vmax - vmin) for v in values]


def _clamp(v: float, low: float, high: float) -> float:
    if v < low:
        return low
    if v > high:
        return high
    return v


def select_best_bets(
    ai_preds: List[Dict[str, Any]],
    min_prob: float = 0.025,
    max_odds_cap: float = 50.0,
    max_ev_cap: float = 2.5,
    top_n: int = 5,
    weight_prob: float = 0.60,
    weight_ev: float = 0.25,
    weight_odds: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    ai_preds の各行に最低でも以下がある前提:
      - combo
      - score   (= probability)
      - odds
      - ev

    最強買い目向けに、
    EV偏重を抑えて prob 重視で上位選定する。

    スコア思想:
      - prob を最重要
      - ev は補助
      - odds は軽め
      - 極端な高オッズ / 高EV は cap して暴走抑制
    """
    rows: List[Dict[str, Any]] = []

    for row in ai_preds:
        combo = str(row.get("combo", "")).strip()
        prob = _safe_float(row.get("prob", row.get("score", 0.0)), 0.0)
        odds = _safe_float(row.get("odds", 0.0), 0.0)
        ev = _safe_float(row.get("ev", 0.0), 0.0)

        if not combo:
            continue
        if prob < min_prob:
            continue
        if odds <= 0:
            continue
        if ev <= 0:
            continue

        row2 = dict(row)
        row2["prob"] = prob
        row2["odds_raw"] = odds
        row2["ev_raw"] = ev
        row2["odds_capped"] = min(odds, max_odds_cap)
        row2["ev_capped"] = min(ev, max_ev_cap)
        rows.append(row2)

    if not rows:
        return []

    prob_vals = [r["prob"] for r in rows]
    odds_vals = [r["odds_capped"] for r in rows]
    ev_vals = [r["ev_capped"] for r in rows]

    prob_norms = _minmax_norm(prob_vals)
    odds_norms = _minmax_norm(odds_vals)
    ev_norms = _minmax_norm(ev_vals)

    total_weight = weight_prob + weight_ev + weight_odds
    if total_weight <= 0:
        weight_prob, weight_ev, weight_odds = 0.60, 0.25, 0.15
        total_weight = 1.0

    weight_prob = weight_prob / total_weight
    weight_ev = weight_ev / total_weight
    weight_odds = weight_odds / total_weight

    for i, r in enumerate(rows):
        r["prob_norm"] = _clamp(prob_norms[i], 0.0, 1.0)
        r["odds_norm"] = _clamp(odds_norms[i], 0.0, 1.0)
        r["ev_norm"] = _clamp(ev_norms[i], 0.0, 1.0)

        # 確率をやや強調、EVはほどほど、オッズは軽め
        prob_boost = r["prob_norm"] ** 1.20
        ev_boost = r["ev_norm"] ** 0.90
        odds_boost = r["odds_norm"] ** 0.85

        r["buy_score"] = (
            prob_boost * weight_prob +
            ev_boost * weight_ev +
            odds_boost * weight_odds
        )

    rows.sort(
        key=lambda x: (
            float(x.get("buy_score", 0.0)),
            float(x.get("prob", 0.0)),
            float(x.get("ev_capped", 0.0)),
        ),
        reverse=True,
    )
    return rows[:top_n]
