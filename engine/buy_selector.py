from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List


CONFIG_JSON_PATH = Path("data/models/venue_buy_config.optimized.json")

# 「モデルが一番自信を持つ組み合わせ(top1_prob)がこの値未満のレースは賭けない」戦略。
# 2026-07-15: 戸田・桐生・江戸川(with_racer_statsモデル、真のholdout期間1370レース)で検証。
# 前半685レースで閾値を決め、後半685レース(未使用データ)に適用した結果、
# ROIが78.75% -> 84.31%に改善(スキップ約30%、ベット数509)。他の会場・モデルでは
# 未検証のため、デフォルトは無効(0.0)にし、検証済みの3会場のみ有効にする。
SKIP_TOP1_PROB_THRESHOLDS: Dict[str, float] = {
    "戸田": 0.0473,
    "桐生": 0.0473,
    "江戸川": 0.0473,
}

BASE_VENUE_BUY_CONFIG: Dict[str, Dict[str, float]] = {
    "丸亀": {
        "min_prob": 0.050,
        "min_ev": 0.95,
        "max_odds_cap": 25.0,
        "max_ev_cap": 1.8,
        "weight_prob": 0.74,
        "weight_ev": 0.18,
        "weight_odds": 0.08,
        "prob_exp": 1.65,
        "ev_exp": 0.62,
        "base_bonus": 0.64,
        "min_points": 3,
        "max_points": 8,
    },
    "児島": {
        "min_prob": 0.030,
        "min_ev": 0.85,
        "max_odds_cap": 50.0,
        "max_ev_cap": 2.2,
        "weight_prob": 0.64,
        "weight_ev": 0.25,
        "weight_odds": 0.11,
        "prob_exp": 1.42,
        "ev_exp": 0.72,
        "base_bonus": 0.65,
        "min_points": 3,
        "max_points": 10,
    },
    "戸田": {
        "min_prob": 0.026,
        "min_ev": 0.78,
        "max_odds_cap": 70.0,
        "max_ev_cap": 2.6,
        "weight_prob": 0.58,
        "weight_ev": 0.30,
        "weight_odds": 0.12,
        "prob_exp": 1.28,
        "ev_exp": 0.84,
        "base_bonus": 0.66,
        "min_points": 4,
        "max_points": 12,
    },
    "住之江": {
        "min_prob": 0.035,
        "min_ev": 0.85,
        "max_odds_cap": 45.0,
        "max_ev_cap": 2.2,
        "weight_prob": 0.66,
        "weight_ev": 0.24,
        "weight_odds": 0.10,
        "prob_exp": 1.42,
        "ev_exp": 0.72,
        "base_bonus": 0.65,
        "min_points": 3,
        "max_points": 10,
    },
    "default": {
        "min_prob": 0.032,
        "min_ev": 0.85,
        "max_odds_cap": 50.0,
        "max_ev_cap": 2.2,
        "weight_prob": 0.64,
        "weight_ev": 0.25,
        "weight_odds": 0.11,
        "prob_exp": 1.40,
        "ev_exp": 0.74,
        "base_bonus": 0.65,
        "min_points": 3,
        "max_points": 10,
    },
}


VENUE_NAMES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
    "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
    "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
]


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
    for v in VENUE_NAMES:
        if v in s:
            return v
    return "default"


def _config_key(venue: str | None) -> str:
    v = _normalize_venue_name(venue or "")
    if v in BASE_VENUE_BUY_CONFIG:
        return v
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
    key = _config_key(venue)
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

    if "min_ev" not in cfg:
        cfg["min_ev"] = 0.85

    if "fixed_points" in cfg:
        fp = _safe_int(cfg["fixed_points"], 6)
        cfg["min_points"] = float(fp)
        cfg["max_points"] = float(fp)

    cfg["min_points"] = float(max(3, min(12, _safe_int(cfg.get("min_points", 3), 3))))
    cfg["max_points"] = float(max(3, min(12, _safe_int(cfg.get("max_points", 10), 10))))
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

    return max(0.0, min(1.0, h / max_h))


def _calc_dynamic_points(rows: List[Dict[str, Any]], venue: str, cfg: Dict[str, float]) -> int:
    min_points = max(3, min(12, _safe_int(cfg.get("min_points", 3), 3)))
    max_points = max(min_points, min(12, _safe_int(cfg.get("max_points", 10), 10)))

    probs = sorted([_safe_float(r.get("prob", 0.0), 0.0) for r in rows], reverse=True)
    if not probs:
        return min_points

    top1 = probs[0]
    top2 = probs[1] if len(probs) >= 2 else 0.0
    top3_sum = sum(probs[:3])
    top5_sum = sum(probs[:5])
    entropy = _calc_distribution_entropy(probs[:10])
    gap12 = max(0.0, top1 - top2)

    spread_score = 0.0
    spread_score += (0.12 - min(top1, 0.12)) / 0.12 * 0.36
    spread_score += (0.24 - min(top3_sum, 0.24)) / 0.24 * 0.20
    spread_score += (0.36 - min(top5_sum, 0.36)) / 0.36 * 0.12
    spread_score += (0.03 - min(gap12, 0.03)) / 0.03 * 0.08
    spread_score += entropy * 0.24

    v = _normalize_venue_name(venue)
    if v == "丸亀":
        spread_score *= 0.72
    elif v == "戸田":
        spread_score *= 1.08
    elif v == "児島":
        spread_score *= 0.96
    elif v == "住之江":
        spread_score *= 0.90

    spread_score = max(0.0, min(1.0, spread_score))
    points = min_points + round((max_points - min_points) * spread_score)

    if top1 >= 0.10 and top3_sum >= 0.23:
        points = min(points, 3)
    elif top1 >= 0.08 and top5_sum >= 0.34:
        points = min(points, max(4, min_points))

    if entropy >= 0.88 and top1 <= 0.05:
        points = min(max_points, points + 1)

    return max(3, min(12, points))


def _load_external_skip_thresholds() -> Dict[str, float]:
    ext = _load_external_config()
    out: Dict[str, float] = {}
    for k, v in ext.items():
        if "skip_top1_prob_threshold" in v:
            out[str(k)] = float(v["skip_top1_prob_threshold"])
    return out


def get_skip_top1_prob_threshold(venue: str | None = None) -> float:
    key = _normalize_venue_name(venue or "")
    overrides = _load_external_skip_thresholds()
    if key in overrides:
        return overrides[key]
    return SKIP_TOP1_PROB_THRESHOLDS.get(key, 0.0)


def should_skip_race(
    ai_preds: List[Dict[str, Any]],
    venue: str | None = None,
    threshold: float | None = None,
) -> bool:
    """このレースを丸ごと見送るべきかどうかを判定する。

    「賭けるべき候補が(閾値未満で)見つからない」という理由でbest_betsが
    空になるケースと区別するため、独立した関数として公開する。
    呼び出し側(app/controller.pyなど)は、select_best_betsが空リストを
    返したときに素朴なフォールバックへ進む前に、必ずこの関数で
    「意図的な見送りかどうか」を確認すること。
    """
    thr = threshold if threshold is not None else get_skip_top1_prob_threshold(venue)
    if thr <= 0:
        return False

    top1_prob = 0.0
    for row in ai_preds:
        p = _safe_float(row.get("prob", row.get("score", 0.0)), 0.0)
        if p > top1_prob:
            top1_prob = p

    return top1_prob < thr


def _prepare_candidate_rows(
    ai_preds: List[Dict[str, Any]],
    cfg: Dict[str, float],
    use_min_prob: bool = True,
    use_min_ev: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in ai_preds:
        combo = str(row.get("combo", "")).strip()
        prob = _safe_float(row.get("prob", row.get("score", 0.0)), 0.0)
        odds = _safe_float(row.get("odds", row.get("odds_raw", 0.0)), 0.0)
        ev = _safe_float(row.get("ev", row.get("ev_raw", prob * odds)), 0.0)

        if not combo:
            continue
        if prob <= 0:
            continue
        if odds <= 0:
            continue
        if ev <= 0:
            continue
        if use_min_prob and prob < cfg["min_prob"]:
            continue
        if use_min_ev and ev < cfg.get("min_ev", 0.85):
            continue

        row2 = dict(row)
        row2["prob"] = prob
        row2["odds_raw"] = odds
        row2["odds"] = odds
        row2["ev_raw"] = ev
        row2["ev"] = ev
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
    if should_skip_race(ai_preds, venue=venue):
        return []

    cfg = _resolve_config(
        venue=venue,
        min_prob=min_prob,
        max_odds_cap=max_odds_cap,
        max_ev_cap=max_ev_cap,
        weight_prob=weight_prob,
        weight_ev=weight_ev,
        weight_odds=weight_odds,
    )

    rows = _prepare_candidate_rows(ai_preds, cfg, use_min_prob=True, use_min_ev=True)

    if len(rows) < 3:
        # 本来のmin_evを満たす候補が3点未満のときだけ、min_evを大きく緩めすぎず
        # 最低ラインの0.70までに限定して救済する(min_prob条件は維持)。
        # ここでもまだ3点に満たない場合のみ、min_prob/min_evを両方無視した
        # backup_rowsで最終的に埋める(app/controller.py側の
        # 「best_betsが空ならscore降順で単純にtop_nを買う」というさらに粗い
        # フォールバックに落ちるより、buy_score(prob/ev/oddsの加重)で選ぶ方が良いため維持)。
        relaxed_cfg = dict(cfg)
        relaxed_cfg["min_ev"] = min(float(cfg.get("min_ev", 0.85)), 0.70)
        rows = _prepare_candidate_rows(ai_preds, relaxed_cfg, use_min_prob=True, use_min_ev=True)

    backup_rows = _prepare_candidate_rows(ai_preds, cfg, use_min_prob=False, use_min_ev=False)
    if len(rows) < 3:
        rows = backup_rows[:]

    if not rows:
        return []

    prob_norms = _minmax_norm([r["prob"] for r in rows])
    odds_norms = _minmax_norm([r["odds_capped"] for r in rows])
    ev_norms = _minmax_norm([r["ev_capped"] for r in rows])

    for i, r in enumerate(rows):
        r["prob_norm"] = prob_norms[i]
        r["odds_norm"] = odds_norms[i]
        r["ev_norm"] = ev_norms[i]

        r["buy_score"] = (
            (r["prob_norm"] * cfg["weight_prob"]) +
            (r["ev_norm"] * cfg["weight_ev"]) +
            (r["odds_norm"] * cfg["weight_odds"])
        )

    rows.sort(
        key=lambda x: (
            float(x.get("buy_score", 0.0)),
            float(x.get("prob", 0.0)),
            float(x.get("ev_raw", 0.0)),
        ),
        reverse=True,
    )

    dynamic_points = _calc_dynamic_points(rows, venue or "default", cfg)
    final_n = max(3, min(12, min(_safe_int(top_n, 12), dynamic_points)))

    selected = rows[:final_n]

    for idx, r in enumerate(selected, start=1):
        r["buy_rank"] = idx
        r["is_best_bet"] = True

    return selected
