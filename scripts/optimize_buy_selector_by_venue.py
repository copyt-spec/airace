from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from engine.feature_builder_current import add_feature_block, normalize_venue, sanitize_x
from engine.buy_selector import BASE_VENUE_BUY_CONFIG

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


def _build_prob_rows(prob_map: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    for combo, prob in prob_map.items():
        rows.append({
            "combo": combo,
            "prob": prob,
        })
    rows.sort(key=lambda x: float(x["prob"]), reverse=True)
    return rows


def _select_prob_bets(
    prob_rows: List[Dict[str, Any]],
    cfg: Dict[str, float],
) -> List[str]:
    min_prob = _safe_float(cfg.get("min_prob", 0.015), 0.015)
    fixed_points = max(3, min(16, int(_safe_float(cfg.get("fixed_points", 6), 6))))
    prob_exp = _safe_float(cfg.get("prob_exp", 1.35), 1.35)

    rows = []
    for r in prob_rows:
        prob = _safe_float(r.get("prob", 0.0), 0.0)
        if prob < min_prob:
            continue

        score = max(prob, 1e-9) ** prob_exp
        rr = dict(r)
        rr["score"] = score
        rows.append(rr)

    rows.sort(key=lambda x: float(x["score"]), reverse=True)
    selected = rows[:fixed_points]
    return [str(x["combo"]) for x in selected]


def _simulate_roi_for_config(
    venue_df: pd.DataFrame,
    model,
    feature_cols: List[str],
    cfg: Dict[str, float],
) -> Dict[str, float]:
    races = 0
    hits = 0
    total_bets = 0
    total_return = 0.0

    for race_key, df120 in venue_df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue

        prob_map = _build_prob_map(model, feature_cols, df120)
        prob_rows = _build_prob_rows(prob_map)
        selected = _select_prob_bets(prob_rows, cfg)

        y_combo = _safe_str(df120.iloc[0].get("y_combo"))
        payout = _safe_float(df120.iloc[0].get("trifecta_payout", 0.0), 0.0)

        races += 1
        total_bets += len(selected)

        if y_combo in selected:
            hits += 1
            total_return += payout

    total_cost = total_bets * 100.0
    hit_rate = hits / races if races else 0.0
    avg_points = total_bets / races if races else 0.0
    roi = total_return / total_cost if total_cost > 0 else 0.0

    # ROI重視 + 的中率も少し加点
    score = roi * 0.75 + hit_rate * 0.25

    return {
        "races": races,
        "hits": hits,
        "hit_rate": hit_rate,
        "avg_points": avg_points,
        "total_bets": total_bets,
        "total_cost": total_cost,
        "total_return": total_return,
        "roi": roi,
        "score": score,
    }


def _sample_config(base_cfg: Dict[str, float], venue: str) -> Dict[str, float]:
    rng = random.random

    cfg = dict(base_cfg)

    if venue == "丸亀":
        cfg["min_prob"] = round(random.uniform(0.016, 0.030), 4)
        cfg["fixed_points"] = random.randint(3, 7)
        cfg["prob_exp"] = round(random.uniform(1.20, 1.65), 3)
    elif venue == "児島":
        cfg["min_prob"] = round(random.uniform(0.012, 0.025), 4)
        cfg["fixed_points"] = random.randint(4, 10)
        cfg["prob_exp"] = round(random.uniform(1.05, 1.45), 3)
    elif venue == "戸田":
        cfg["min_prob"] = round(random.uniform(0.011, 0.023), 4)
        cfg["fixed_points"] = random.randint(5, 12)
        cfg["prob_exp"] = round(random.uniform(1.00, 1.40), 3)
    else:
        cfg["min_prob"] = round(random.uniform(0.012, 0.025), 4)
        cfg["fixed_points"] = random.randint(4, 9)
        cfg["prob_exp"] = round(random.uniform(1.05, 1.50), 3)

    return cfg


def optimize_one_venue(venue: str, df: pd.DataFrame, n_trials: int = 120) -> Tuple[Dict[str, float], Dict[str, float]]:
    venue_df = df[df["venue"].astype(str).map(normalize_venue) == venue].copy()
    venue_df = _time_split_test(venue_df)

    if venue_df.empty:
        raise RuntimeError(f"{venue}: test rows empty")

    model, feature_cols = _load_model_and_meta(venue)
    base_cfg = dict(BASE_VENUE_BUY_CONFIG.get(venue, BASE_VENUE_BUY_CONFIG["default"]))

    best_cfg = dict(base_cfg)
    best_eval = _simulate_roi_for_config(venue_df, model, feature_cols, best_cfg)

    print(f"\n===== {venue} =====")
    print("test rows:", len(venue_df))
    print("test races:", venue_df['race_key'].nunique())
    print("base_eval:", best_eval)

    for trial in range(1, n_trials + 1):
        cfg = _sample_config(base_cfg, venue)
        ev = _simulate_roi_for_config(venue_df, model, feature_cols, cfg)

        if ev["score"] > best_eval["score"]:
            best_cfg = cfg
            best_eval = ev

        if trial % 20 == 0:
            print(
                f"[{venue} trial {trial}] "
                f"best score={best_eval['score']:.4f} "
                f"roi={best_eval['roi']:.4f} "
                f"hit_rate={best_eval['hit_rate']:.4f} "
                f"avg_points={best_eval['avg_points']:.2f}"
            )

    print("\nBEST CONFIG:", best_cfg)
    print("BEST EVAL  :", best_eval)

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
        best_cfg, best_eval = optimize_one_venue(venue, df, n_trials=120)
        out_cfg[venue] = best_cfg
        out_eval[venue] = best_eval

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print("\n===== SAVED =====")
    print("config:", OUTPUT_JSON)
    print(json.dumps(out_cfg, ensure_ascii=False, indent=2))
    print("\n===== EVAL =====")
    print(json.dumps(out_eval, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
