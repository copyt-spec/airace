# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from engine.buy_selector import BASE_VENUE_BUY_CONFIG, select_best_bets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
OUTPUT_JSON = MODEL_DIR / "venue_buy_config.optimized.json"

VENUE_DATASETS = {
    "丸亀": DATASET_DIR / "trifecta_train_small_marugame.csv",
    "戸田": DATASET_DIR / "trifecta_train_small_toda.csv",
    "児島": DATASET_DIR / "trifecta_train_small_kojima.csv",
}

BEST_MODEL_SUFFIX = {
    "丸亀": "with_racer_no",
    "戸田": "with_racer_no",
    "児島": "with_racer_no",
}

ROWS_PER_RACE = 120


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


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
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


def sanitize_x(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0

    out = out[feature_cols].copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)
        else:
            out[col] = out[col].astype(str).fillna("")

    return out


def add_feature_block(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "combo" in out.columns:
        parts = out["combo"].astype(str).str.split("-", expand=True)
        if parts.shape[1] >= 3:
            out["combo_first_lane"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype("int16")
            out["combo_second_lane"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype("int16")
            out["combo_third_lane"] = pd.to_numeric(parts[2], errors="coerce").fillna(0).astype("int16")
        else:
            out["combo_first_lane"] = 0
            out["combo_second_lane"] = 0
            out["combo_third_lane"] = 0

    wind_map = {
        "無風": 0,
        "北": 1, "北東": 2, "東": 3, "南東": 4,
        "南": 5, "南西": 6, "西": 7, "北西": 8,
    }
    if "wind_dir" in out.columns:
        out["wind_dir_code"] = out["wind_dir"].astype(str).map(wind_map).fillna(0).astype("int16")
    else:
        out["wind_dir_code"] = 0

    weather_map = {
        "晴": 1, "晴れ": 1,
        "曇": 2, "曇り": 2,
        "雨": 3,
        "雪": 4,
    }
    if "weather" in out.columns:
        out["weather_code"] = out["weather"].astype(str).map(weather_map).fillna(0).astype("int16")
    else:
        out["weather_code"] = 0

    for lane in range(1, 7):
        st_col = f"lane{lane}_st"
        ex_col = f"lane{lane}_exhibit"
        course_col = f"lane{lane}_course"
        motor_col = f"lane{lane}_motor"
        boat_col = f"lane{lane}_boat"
        racer_col = f"lane{lane}_racer_no"

        for c in [st_col, ex_col, course_col, motor_col, boat_col, racer_col]:
            if c not in out.columns:
                out[c] = 0

        out[st_col] = pd.to_numeric(out[st_col], errors="coerce").fillna(0).astype("float32")
        out[ex_col] = pd.to_numeric(out[ex_col], errors="coerce").fillna(0).astype("float32")
        out[course_col] = pd.to_numeric(out[course_col], errors="coerce").fillna(0).astype("int16")
        out[motor_col] = pd.to_numeric(out[motor_col], errors="coerce").fillna(0).astype("int16")
        out[boat_col] = pd.to_numeric(out[boat_col], errors="coerce").fillna(0).astype("int16")
        out[racer_col] = pd.to_numeric(out[racer_col], errors="coerce").fillna(0).astype("int32")

        out[f"lane{lane}_st_inv"] = (0.3 - out[st_col]).clip(lower=-1, upper=1).astype("float32")
        out[f"lane{lane}_exhibit_inv"] = (7.5 - out[ex_col]).clip(lower=-2, upper=2).astype("float32")
        out[f"lane{lane}_course_diff"] = (out[course_col] - lane).astype("int16")

    for pos, pos_name in [
        ("first", "combo_first_lane"),
        ("second", "combo_second_lane"),
        ("third", "combo_third_lane"),
    ]:
        out[f"{pos}_st"] = 0.0
        out[f"{pos}_exhibit"] = 0.0
        out[f"{pos}_course"] = 0
        out[f"{pos}_motor"] = 0
        out[f"{pos}_boat"] = 0
        out[f"{pos}_racer_no"] = 0

        for lane in range(1, 7):
            mask = out[pos_name] == lane
            out.loc[mask, f"{pos}_st"] = out.loc[mask, f"lane{lane}_st"]
            out.loc[mask, f"{pos}_exhibit"] = out.loc[mask, f"lane{lane}_exhibit"]
            out.loc[mask, f"{pos}_course"] = out.loc[mask, f"lane{lane}_course"]
            out.loc[mask, f"{pos}_motor"] = out.loc[mask, f"lane{lane}_motor"]
            out.loc[mask, f"{pos}_boat"] = out.loc[mask, f"lane{lane}_boat"]
            out.loc[mask, f"{pos}_racer_no"] = out.loc[mask, f"lane{lane}_racer_no"]

    out["first_second_st_diff"] = (out["first_st"] - out["second_st"]).astype("float32")
    out["first_third_st_diff"] = (out["first_st"] - out["third_st"]).astype("float32")
    out["first_second_exhibit_diff"] = (out["first_exhibit"] - out["second_exhibit"]).astype("float32")
    out["first_third_exhibit_diff"] = (out["first_exhibit"] - out["third_exhibit"]).astype("float32")

    if "wind_speed_mps" not in out.columns:
        out["wind_speed_mps"] = 0.0
    if "wave_cm" not in out.columns:
        out["wave_cm"] = 0.0

    out["wind_speed_mps"] = pd.to_numeric(out["wind_speed_mps"], errors="coerce").fillna(0).astype("float32")
    out["wave_cm"] = pd.to_numeric(out["wave_cm"], errors="coerce").fillna(0).astype("float32")

    return out


def get_feature_cols(with_racer_no: bool) -> List[str]:
    cols = [
        "combo_first_lane",
        "combo_second_lane",
        "combo_third_lane",
        "wind_dir_code",
        "weather_code",
        "wind_speed_mps",
        "wave_cm",

        "first_st",
        "second_st",
        "third_st",
        "first_exhibit",
        "second_exhibit",
        "third_exhibit",
        "first_course",
        "second_course",
        "third_course",
        "first_motor",
        "second_motor",
        "third_motor",
        "first_boat",
        "second_boat",
        "third_boat",

        "first_second_st_diff",
        "first_third_st_diff",
        "first_second_exhibit_diff",
        "first_third_exhibit_diff",
    ]

    for lane in range(1, 7):
        cols += [
            f"lane{lane}_st",
            f"lane{lane}_exhibit",
            f"lane{lane}_course",
            f"lane{lane}_motor",
            f"lane{lane}_boat",
            f"lane{lane}_st_inv",
            f"lane{lane}_exhibit_inv",
            f"lane{lane}_course_diff",
        ]

    if with_racer_no:
        cols += [
            "first_racer_no",
            "second_racer_no",
            "third_racer_no",
        ]
        for lane in range(1, 7):
            cols.append(f"lane{lane}_racer_no")

    return cols


def load_model_and_meta(venue: str):
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


def load_dataset(csv_path: Path, venue: str, max_races: int = 300) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    usecols = [
        "date", "venue", "race_key", "race_no", "combo", "y",
        "trifecta_payout", "wave_cm", "weather", "wind_dir", "wind_speed_mps",
        "lane1_boat", "lane1_course", "lane1_exhibit", "lane1_motor", "lane1_racer_no", "lane1_st",
        "lane2_boat", "lane2_course", "lane2_exhibit", "lane2_motor", "lane2_racer_no", "lane2_st",
        "lane3_boat", "lane3_course", "lane3_exhibit", "lane3_motor", "lane3_racer_no", "lane3_st",
        "lane4_boat", "lane4_course", "lane4_exhibit", "lane4_motor", "lane4_racer_no", "lane4_st",
        "lane5_boat", "lane5_course", "lane5_exhibit", "lane5_motor", "lane5_racer_no", "lane5_st",
        "lane6_boat", "lane6_course", "lane6_exhibit", "lane6_motor", "lane6_racer_no", "lane6_st",
    ]

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    df["venue"] = df["venue"].astype(str).map(normalize_venue)
    df = df[df["venue"] == venue].copy()

    if df.empty:
        raise RuntimeError(f"{venue}: dataset empty")

    df["date"] = df["date"].astype(str)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype("int8")
    df["trifecta_payout"] = pd.to_numeric(df["trifecta_payout"], errors="coerce").fillna(0).astype("float32")

    race_keys = (
        df[["race_key", "date"]]
        .drop_duplicates()
        .sort_values("date")
        .tail(max_races)["race_key"]
        .tolist()
    )
    df = df[df["race_key"].isin(race_keys)].copy()

    return df.reset_index(drop=True)


def build_prob_map(model, feature_cols: List[str], df120: pd.DataFrame) -> Dict[str, float]:
    work = add_feature_block(df120.copy())
    x = sanitize_x(work, feature_cols)
    prob = model.predict_proba(x)[:, 1]
    combos = work["combo"].astype(str).tolist()

    prob_map = {combo: float(p) for combo, p in zip(combos, prob)}
    total = sum(prob_map.values())
    if total > 0:
        prob_map = {k: v / total for k, v in prob_map.items()}
    return prob_map


def build_ai_preds(prob_map: Dict[str, float], odds_proxy_scale: float = 100.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for combo, prob in prob_map.items():
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


def simulate_for_config(
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

    for _, df120 in venue_df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue

        prob_map = build_prob_map(model, feature_cols, df120)
        ai_preds = build_ai_preds(prob_map)

        selected_rows = select_best_bets(
            ai_preds=ai_preds,
            venue=venue,
            min_prob=cfg.get("min_prob"),
            max_odds_cap=cfg.get("max_odds_cap"),
            max_ev_cap=cfg.get("max_ev_cap"),
            weight_prob=cfg.get("weight_prob"),
            weight_ev=cfg.get("weight_ev"),
            weight_odds=cfg.get("weight_odds"),
            top_n=int(_safe_float(cfg.get("max_points", 12), 12)),
        )

        selected = [str(r.get("combo", "")) for r in selected_rows]
        y_combo = str(df120.loc[df120["y"] == 1, "combo"].iloc[0]) if (df120["y"] == 1).any() else ""
        payout = _safe_float(df120["trifecta_payout"].iloc[0], 0.0)

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

    point_penalty = 0.0
    if avg_points > 10:
        point_penalty = (avg_points - 10) * 0.01

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


def sample_config(base_cfg: Dict[str, float], venue: str) -> Dict[str, float]:
    cfg = dict(base_cfg)

    if venue == "丸亀":
        cfg["min_prob"] = round(random.uniform(0.016, 0.030), 4)
        cfg["prob_exp"] = round(random.uniform(1.20, 1.70), 3)
        cfg["ev_exp"] = round(random.uniform(0.60, 0.90), 3)
        cfg["weight_prob"] = round(random.uniform(0.55, 0.78), 3)
        cfg["weight_ev"] = round(random.uniform(0.10, 0.28), 3)
        cfg["weight_odds"] = round(max(0.05, 1.0 - cfg["weight_prob"] - cfg["weight_ev"]), 3)
        cfg["min_points"] = random.randint(3, 5)
        cfg["max_points"] = random.randint(max(int(cfg["min_points"]), 5), 8)
        cfg["max_odds_cap"] = round(random.uniform(35, 70), 1)
        cfg["max_ev_cap"] = round(random.uniform(1.8, 3.0), 2)
        cfg["base_bonus"] = round(random.uniform(0.58, 0.72), 3)

    elif venue == "戸田":
        cfg["min_prob"] = round(random.uniform(0.011, 0.024), 4)
        cfg["prob_exp"] = round(random.uniform(1.00, 1.40), 3)
        cfg["ev_exp"] = round(random.uniform(0.70, 1.00), 3)
        cfg["weight_prob"] = round(random.uniform(0.45, 0.68), 3)
        cfg["weight_ev"] = round(random.uniform(0.18, 0.34), 3)
        cfg["weight_odds"] = round(max(0.08, 1.0 - cfg["weight_prob"] - cfg["weight_ev"]), 3)
        cfg["min_points"] = random.randint(4, 7)
        cfg["max_points"] = random.randint(max(int(cfg["min_points"]) + 1, 6), 10)
        cfg["max_odds_cap"] = round(random.uniform(50, 110), 1)
        cfg["max_ev_cap"] = round(random.uniform(2.0, 3.8), 2)
        cfg["base_bonus"] = round(random.uniform(0.55, 0.68), 3)

    elif venue == "児島":
        cfg["min_prob"] = round(random.uniform(0.012, 0.024), 4)
        cfg["prob_exp"] = round(random.uniform(1.05, 1.50), 3)
        cfg["ev_exp"] = round(random.uniform(0.70, 0.98), 3)
        cfg["weight_prob"] = round(random.uniform(0.48, 0.70), 3)
        cfg["weight_ev"] = round(random.uniform(0.16, 0.34), 3)
        cfg["weight_odds"] = round(max(0.08, 1.0 - cfg["weight_prob"] - cfg["weight_ev"]), 3)
        cfg["min_points"] = random.randint(4, 6)
        cfg["max_points"] = random.randint(max(int(cfg["min_points"]) + 1, 7), 11)
        cfg["max_odds_cap"] = round(random.uniform(40, 95), 1)
        cfg["max_ev_cap"] = round(random.uniform(2.0, 3.5), 2)
        cfg["base_bonus"] = round(random.uniform(0.58, 0.70), 3)

    if cfg["max_points"] < cfg["min_points"]:
        cfg["max_points"] = cfg["min_points"]

    total_w = cfg["weight_prob"] + cfg["weight_ev"] + cfg["weight_odds"]
    if total_w > 0:
        cfg["weight_prob"] = round(cfg["weight_prob"] / total_w, 3)
        cfg["weight_ev"] = round(cfg["weight_ev"] / total_w, 3)
        cfg["weight_odds"] = round(cfg["weight_odds"] / total_w, 3)
        remain = round(1.0 - (cfg["weight_prob"] + cfg["weight_ev"] + cfg["weight_odds"]), 3)
        cfg["weight_odds"] = round(cfg["weight_odds"] + remain, 3)

    return cfg


def optimize_one_venue(
    venue: str,
    dataset_path: Path,
    n_trials: int = 20,
    max_races: int = 300,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    venue_df = load_dataset(dataset_path, venue=venue, max_races=max_races)
    model, feature_cols = load_model_and_meta(venue)

    base_cfg = dict(BASE_VENUE_BUY_CONFIG.get(venue, BASE_VENUE_BUY_CONFIG["default"]))
    if "min_points" not in base_cfg:
        base_cfg["min_points"] = 3
    if "max_points" not in base_cfg:
        base_cfg["max_points"] = 12

    best_cfg = dict(base_cfg)
    best_eval = simulate_for_config(venue_df, model, feature_cols, venue, best_cfg)

    print(f"\n===== {venue} =====")
    print("dataset    :", dataset_path)
    print("rows       :", len(venue_df))
    print("race_count :", venue_df['race_key'].nunique())
    print("base_eval  :", json.dumps(best_eval, ensure_ascii=False, indent=2))

    for trial in range(1, n_trials + 1):
        cfg = sample_config(base_cfg, venue)
        ev = simulate_for_config(venue_df, model, feature_cols, venue, cfg)

        if ev["score"] > best_eval["score"]:
            best_cfg = cfg
            best_eval = ev

        print(
            f"[{venue} trial {trial:02d}/{n_trials}] "
            f"score={ev['score']:.4f} "
            f"best={best_eval['score']:.4f} "
            f"roi={ev['roi']:.4f} "
            f"hit={ev['hit_rate']:.4f} "
            f"pts={ev['avg_points']:.2f}"
        )

    print("\nBEST CONFIG:")
    print(json.dumps(best_cfg, ensure_ascii=False, indent=2))
    print("BEST EVAL:")
    print(json.dumps(best_eval, ensure_ascii=False, indent=2))

    return best_cfg, best_eval


def main() -> None:
    random.seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    out_cfg: Dict[str, Dict[str, float]] = {}
    out_eval: Dict[str, Dict[str, float]] = {}

    for venue in ["丸亀", "戸田", "児島"]:
        dataset_path = VENUE_DATASETS[venue]
        best_cfg, best_eval = optimize_one_venue(
            venue=venue,
            dataset_path=dataset_path,
            n_trials=20,
            max_races=300,
        )
        out_cfg[venue] = best_cfg
        out_eval[venue] = best_eval

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print("\n===== SAVED CONFIG =====")
    print(OUTPUT_JSON)
    print(json.dumps(out_cfg, ensure_ascii=False, indent=2))

    print("\n===== FINAL EVAL =====")
    print(json.dumps(out_eval, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
