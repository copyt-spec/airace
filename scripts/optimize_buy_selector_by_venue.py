from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from engine.feature_builder_current import add_feature_block, normalize_venue, sanitize_x
from engine.buy_selector import BASE_VENUE_BUY_CONFIG, select_best_bets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
OUTPUT_JSON = MODEL_DIR / "venue_buy_config.optimized.json"

ROWS_PER_RACE = 120
TARGET_VENUES = ["丸亀", "児島", "戸田"]
VALID_TO = "20260228"

BEST_MODEL_SUFFIX = {
    "丸亀": "with_racer_no",
    "児島": "without_racer_no",
    "戸田": "with_racer_no",
}


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError as e:
        raise RuntimeError("catboost が入っていません。 `pip install catboost`") from e


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _time_split_test(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].astype(str)
    return out[out["date"] > VALID_TO].copy()


def _load_model_and_meta(venue: str):
    CatBoostClassifier = _require_catboost()
    suffix = BEST_MODEL_SUFFIX[venue]
    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_{suffix}.cbm"
    meta_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_{suffix}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = list(meta.get("feature_cols", []))
    if not feature_cols:
        raise RuntimeError(f"feature_cols missing in {meta_path}")

    return model, feature_cols


def _build_prob_map(model, feature_cols: List[str], df120: pd.DataFrame) -> Dict[str, float]:
    work = add_feature_block(df120.copy())

    missing_cols = [c for c in feature_cols if c not in work.columns]
    for c in missing_cols:
        work[c] = 0.0

    x = sanitize_x(work, feature_cols)
    prob = model.predict_proba(x)[:, 1]
    combos = work["combo"].astype(str).tolist()

    prob_map = {combo: float(p) for combo, p in zip(combos, prob)}
    total = sum(prob_map.values())
    if total > 0:
        prob_map = {k: v / total for k, v in prob_map.items()}
    return prob_map


def _build_ai_preds(prob_map: Dict[str, float], odds_proxy_scale: float = 100.0) -> List[Dict[str, Any]]:
    """
    過去の120通りオッズが無いので、optimizer段階では
    prob中心で点数戦略を最適化する。
    odds/ev は buy_selector が落ちないための軽い疑似値。
    """
    rows: List[Dict[str, Any]] = []

    for combo, prob in prob_map.items():
        # 疑似 odds:
        # prob が低いほど高い値になるようにする
        pseudo_odds = 1.0 / max(prob, 1e-6)
        pseudo_odds = min(pseudo_odds, odds_proxy_scale)

        rows.append({
            "combo": combo,
            "prob": float(prob),
            "score": float(prob),
            "odds": float(pseudo_odds),
            "ev": float(prob) * float(pseudo_odds),
        })

    return rows


def _simulate_for_config(
    venue_df: pd.DataFrame,
    model,
    feature_cols: List[str],
    venue: str,
    cfg: Dict[str, float],
) -> Dict[str, float]:
    races = 0
    hits = 0
    total_bets = 0
    total_return = 0.0

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    points_hist: List[int] = []

    for race_key, df120 in venue_df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue

        prob_map = _build_prob_map(model, feature_cols, df120)
        ai_preds = _build_ai_preds(prob_map)

        selected_rows = select_best_bets(
            ai_preds=ai_preds,
            venue=venue,
            min_prob=cfg.get("min_prob"),
            max_odds_cap=cfg.get("max_odds_cap"),
            max_ev_cap=cfg.get("max_ev_cap"),
            weight_prob=cfg.get("weight_prob"),
            weight_ev=cfg.get("weight_ev"),
            weight_odds=cfg.get("weight_odds"),
            top_n=int(_safe_float(cfg.get("max_points", 16), 16)),
        )

        selected = [str(r.get("combo", "")) for r in selected_rows]
        y_combo = _safe_str(df120.iloc[0].get("y_combo"))
        payout = _safe_float(df120.iloc[0].get("trifecta_payout", 0.0), 0.0)

        races += 1
        total_bets += len(selected)
        points_hist.append(len(selected))

        ranked = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
        top1 = [c for c, _ in ranked[:1]]
        top3 = [c for c, _ in ranked[:3]]
        top5 = [c for c, _ in ranked[:5]]

        if y_combo in top1:
            top1_hits += 1
        if y_combo in top3:
            top3_hits += 1
        if y_combo in top5:
            top5_hits += 1

        if y_combo in selected:
            hits += 1
            total_return += payout

    total_cost = total_bets * 100.0
    hit_rate = hits / races if races else 0.0
    avg_points = total_bets / races if races else 0.0
    roi = total_return / total_cost if total_cost > 0 else 0.0

    top1_rate = top1_hits / races if races else 0.0
    top3_rate = top3_hits / races if races else 0.0
    top5_rate = top5_hits / races if races else 0.0

    # 点数が多すぎるとわずかに減点
    point_penalty = 0.0
    if avg_points > 10:
        point_penalty = (avg_points - 10) * 0.01

    # ROI最重視 + 的中率 + top3/5も少し加点
    score = (
        roi * 0.62
        + hit_rate * 0.18
        + top3_rate * 0.10
        + top5_rate * 0.10
        - point_penalty
    )

    return {
        "races": races,
        "hits": hits,
        "hit_rate": hit_rate,
        "avg_points": avg_points,
        "total_bets": total_bets,
        "total_cost": total_cost,
        "total_return": total_return,
        "roi": roi,
        "top1": top1_rate,
        "top3": top3_rate,
        "top5": top5_rate,
        "score": score,
        "min_points_used": min(points_hist) if points_hist else 0,
        "max_points_used": max(points_hist) if points_hist else 0,
    }


def _sample_config(base_cfg: Dict[str, float], venue: str) -> Dict[str, float]:
    cfg = dict(base_cfg)

    if venue == "丸亀":
        cfg["min_prob"] = round(random.uniform(0.016, 0.030), 4)
        cfg["prob_exp"] = round(random.uniform(1.20, 1.70), 3)
        cfg["ev_exp"] = round(random.uniform(0.60, 0.85), 3)
        cfg["weight_prob"] = round(random.uniform(0.58, 0.78), 3)
        cfg["weight_ev"] = round(random.uniform(0.15, 0.30), 3)
        cfg["weight_odds"] = round(max(0.05, 1.0 - cfg["weight_prob"] - cfg["weight_ev"]), 3)
        cfg["min_points"] = random.randint(3, 5)
        cfg["max_points"] = random.randint(max(int(cfg["min_points"]), 5), 9)
        cfg["max_odds_cap"] = round(random.uniform(35, 70), 1)
        cfg["max_ev_cap"] = round(random.uniform(1.8, 2.8), 2)
        cfg["base_bonus"] = round(random.uniform(0.60, 0.72), 3)

    elif venue == "児島":
        cfg["min_prob"] = round(random.uniform(0.012, 0.024), 4)
        cfg["prob_exp"] = round(random.uniform(1.05, 1.45), 3)
        cfg["ev_exp"] = round(random.uniform(0.70, 0.95), 3)
        cfg["weight_prob"] = round(random.uniform(0.48, 0.68), 3)
        cfg["weight_ev"] = round(random.uniform(0.18, 0.34), 3)
        cfg["weight_odds"] = round(max(0.08, 1.0 - cfg["weight_prob"] - cfg["weight_ev"]), 3)
        cfg["min_points"] = random.randint(4, 7)
        cfg["max_points"] = random.randint(max(int(cfg["min_points"]) + 1, 7), 13)
        cfg["max_odds_cap"] = round(random.uniform(45, 100), 1)
        cfg["max_ev_cap"] = round(random.uniform(2.0, 3.4), 2)
        cfg["base_bonus"] = round(random.uniform(0.58, 0.70), 3)

    elif venue == "戸田":
        cfg["min_prob"] = round(random.uniform(0.011, 0.023), 4)
        cfg["prob_exp"] = round(random.uniform(1.00, 1.35), 3)
        cfg["ev_exp"] = round(random.uniform(0.75, 1.00), 3)
        cfg["weight_prob"] = round(random.uniform(0.45, 0.65), 3)
        cfg["weight_ev"] = round(random.uniform(0.20, 0.36), 3)
        cfg["weight_odds"] = round(max(0.08, 1.0 - cfg["weight_prob"] - cfg["weight_ev"]), 3)
        cfg["min_points"] = random.randint(5, 8)
        cfg["max_points"] = random.randint(max(int(cfg["min_points"]) + 1, 8), 15)
        cfg["max_odds_cap"] = round(random.uniform(55, 120), 1)
        cfg["max_ev_cap"] = round(random.uniform(2.2, 3.8), 2)
        cfg["base_bonus"] = round(random.uniform(0.56, 0.68), 3)

    else:
        cfg["min_prob"] = round(random.uniform(0.012, 0.025), 4)
        cfg["prob_exp"] = round(random.uniform(1.05, 1.55), 3)
        cfg["ev_exp"] = round(random.uniform(0.70, 0.95), 3)
        cfg["weight_prob"] = round(random.uniform(0.50, 0.70), 3)
        cfg["weight_ev"] = round(random.uniform(0.18, 0.32), 3)
        cfg["weight_odds"] = round(max(0.08, 1.0 - cfg["weight_prob"] - cfg["weight_ev"]), 3)
        cfg["min_points"] = random.randint(4, 7)
        cfg["max_points"] = random.randint(max(int(cfg["min_points"]) + 1, 7), 12)
        cfg["max_odds_cap"] = round(random.uniform(45, 95), 1)
        cfg["max_ev_cap"] = round(random.uniform(2.0, 3.2), 2)
        cfg["base_bonus"] = round(random.uniform(0.58, 0.70), 3)

    if cfg["max_points"] < cfg["min_points"]:
        cfg["max_points"] = cfg["min_points"]

    total_w = cfg["weight_prob"] + cfg["weight_ev"] + cfg["weight_odds"]
    if total_w > 0:
        cfg["weight_prob"] = round(cfg["weight_prob"] / total_w, 3)
        cfg["weight_ev"] = round(cfg["weight_ev"] / total_w, 3)
        cfg["weight_odds"] = round(cfg["weight_odds"] / total_w, 3)

        # 丸め誤差補正
        remain = round(1.0 - (cfg["weight_prob"] + cfg["weight_ev"] + cfg["weight_odds"]), 3)
        cfg["weight_odds"] = round(cfg["weight_odds"] + remain, 3)

    return cfg


def optimize_one_venue(venue: str, df: pd.DataFrame, n_trials: int = 160) -> Tuple[Dict[str, float], Dict[str, float]]:
    venue_df = df[df["venue"].astype(str).map(normalize_venue) == venue].copy()
    venue_df = _time_split_test(venue_df)

    if venue_df.empty:
        raise RuntimeError(f"{venue}: test rows empty")

    model, feature_cols = _load_model_and_meta(venue)
    base_cfg = dict(BASE_VENUE_BUY_CONFIG.get(venue, BASE_VENUE_BUY_CONFIG["default"]))

    best_cfg = dict(base_cfg)
    best_eval = _simulate_for_config(venue_df, model, feature_cols, venue, best_cfg)

    print(f"\n===== {venue} =====")
    print("test rows :", len(venue_df))
    print("test races:", venue_df["race_key"].nunique())
    print("base_eval :", json.dumps(best_eval, ensure_ascii=False, indent=2))

    for trial in range(1, n_trials + 1):
        cfg = _sample_config(base_cfg, venue)
        ev = _simulate_for_config(venue_df, model, feature_cols, venue, cfg)

        if ev["score"] > best_eval["score"]:
            best_cfg = cfg
            best_eval = ev

        if trial % 20 == 0:
            print(
                f"[{venue} trial {trial}] "
                f"best score={best_eval['score']:.4f} "
                f"roi={best_eval['roi']:.4f} "
                f"hit_rate={best_eval['hit_rate']:.4f} "
                f"avg_points={best_eval['avg_points']:.2f} "
                f"used_points={best_eval['min_points_used']}-{best_eval['max_points_used']}"
            )

    print("\nBEST CONFIG:")
    print(json.dumps(best_cfg, ensure_ascii=False, indent=2))
    print("BEST EVAL:")
    print(json.dumps(best_eval, ensure_ascii=False, indent=2))

    return best_cfg, best_eval


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    random.seed(42)
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    required = {"date", "venue", "race_key", "combo", "y_combo", "trifecta_payout"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    out_cfg: Dict[str, Dict[str, float]] = {}
    out_eval: Dict[str, Dict[str, float]] = {}

    for venue in TARGET_VENUES:
        best_cfg, best_eval = optimize_one_venue(venue, df, n_trials=160)
        out_cfg[venue] = best_cfg
        out_eval[venue] = best_eval

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print("\n===== SAVED CONFIG =====")
    print(OUTPUT_JSON)
    print(json.dumps(out_cfg, ensure_ascii=False, indent=2))

    print("\n===== FINAL EVAL =====")
    print(json.dumps(out_eval, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
