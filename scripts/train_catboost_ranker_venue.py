from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

TARGET_VENUES = ["丸亀", "児島", "戸田"]
ROWS_PER_RACE = 120

# 時系列分割
TRAIN_TO = "20251231"
VALID_TO = "20260228"

WHITELIST_PREFIXES = [
    "lane1_", "lane2_", "lane3_", "lane4_", "lane5_", "lane6_",
    "race_", "inside_", "outside_", "inside_outside_",
    "combo_", "weather_", "wind_", "wave_",
]

WHITELIST_EXACT_EXCLUDE = {
    "date",
    "venue",
    "race_key",
    "race_no",
    "combo",
    "y_combo",
    "y_class",
}


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError as e:
        raise RuntimeError("catboost が入っていません。 `pip install catboost`") from e


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _normalize_venue(v: str) -> str:
    s = _safe_str(v)
    if "丸亀" in s:
        return "丸亀"
    if "児島" in s:
        return "児島"
    if "戸田" in s:
        return "戸田"
    return s


def _build_is_hit(df: pd.DataFrame) -> pd.Series:
    combo = df["combo"].astype(str)
    y_combo = df["y_combo"].astype(str)
    return (combo == y_combo).astype(int)


def _time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["date"] = out["date"].astype(str)

    train_df = out[out["date"] <= TRAIN_TO].copy()
    valid_df = out[(out["date"] > TRAIN_TO) & (out["date"] <= VALID_TO)].copy()
    test_df = out[out["date"] > VALID_TO].copy()
    return train_df, valid_df, test_df


def _is_allowed_col(col: str) -> bool:
    if col in WHITELIST_EXACT_EXCLUDE:
        return False
    return any(col.startswith(p) for p in WHITELIST_PREFIXES)


def _sanitize_x(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if _is_allowed_col(c)]
    x = df[feature_cols].copy()

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)

    x = x.replace([np.inf, -np.inf], 0.0)
    return x


def _topk_hits(prob_map: Dict[str, float], y_combo: str, k: int) -> int:
    if not prob_map or not y_combo:
        return 0
    topk = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
    combos = [c for c, _ in topk]
    return 1 if y_combo in combos else 0


def _split_combo(combo: str) -> Tuple[int, int, int]:
    try:
        a, b, c = str(combo).split("-")
        return int(a), int(b), int(c)
    except Exception:
        return 0, 0, 0


def _ensure_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _pick_lane_metric(df: pd.DataFrame, lane_series: pd.Series, metric: str) -> pd.Series:
    vals: List[float] = []
    for idx, lane in lane_series.items():
        lane_i = int(lane) if pd.notna(lane) else 0
        col = f"lane{lane_i}_{metric}"
        if lane_i in range(1, 7) and col in df.columns:
            vals.append(_safe_float(df.at[idx, col], 0.0))
        else:
            vals.append(0.0)
    return pd.Series(vals, index=df.index, dtype=float)


def _lane_mean(df: pd.DataFrame, metric: str) -> pd.Series:
    cols = [f"lane{i}_{metric}" for i in range(1, 7) if f"lane{i}_{metric}" in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index, dtype=float)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)


def _add_binary_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    combo全体を見る特徴を追加
    """
    out = df.copy()

    combo_first = _ensure_col(out, "combo_first_lane", 0)
    combo_second = _ensure_col(out, "combo_second_lane", 0)
    combo_third = _ensure_col(out, "combo_third_lane", 0)

    metrics = [
        "win_rate",
        "place_rate",
        "motor_rate",
        "boat_rate",
        "ability_index",
        "grade_num",
        "exhibit",
        "st",
    ]

    picked: Dict[str, Dict[str, pd.Series]] = {
        "first": {},
        "second": {},
        "third": {},
    }

    for metric in metrics:
        picked["first"][metric] = _pick_lane_metric(out, combo_first, metric)
        picked["second"][metric] = _pick_lane_metric(out, combo_second, metric)
        picked["third"][metric] = _pick_lane_metric(out, combo_third, metric)

    # race mean
    race_mean = {metric: _lane_mean(out, metric) for metric in metrics}

    # 3艇 aggregate
    for metric in metrics:
        f = picked["first"][metric]
        s = picked["second"][metric]
        t = picked["third"][metric]

        out[f"combo_sum_{metric}_v2"] = f + s + t
        out[f"combo_mean_{metric}_v2"] = (f + s + t) / 3.0
        out[f"combo_max_{metric}_v2"] = pd.concat([f, s, t], axis=1).max(axis=1)
        out[f"combo_min_{metric}_v2"] = pd.concat([f, s, t], axis=1).min(axis=1)
        out[f"combo_range_{metric}_v2"] = out[f"combo_max_{metric}_v2"] - out[f"combo_min_{metric}_v2"]

    # 差分（大きい方が良い系）
    for metric in ["win_rate", "place_rate", "motor_rate", "boat_rate", "ability_index", "grade_num"]:
        f = picked["first"][metric]
        s = picked["second"][metric]
        t = picked["third"][metric]

        out[f"combo_diff_first_second_{metric}_v2"] = f - s
        out[f"combo_diff_first_third_{metric}_v2"] = f - t
        out[f"combo_diff_second_third_{metric}_v2"] = s - t

    # 差分（小さい方が良い系）
    for metric in ["exhibit", "st"]:
        f = picked["first"][metric]
        s = picked["second"][metric]
        t = picked["third"][metric]

        out[f"combo_diff_first_second_{metric}_v2"] = s - f
        out[f"combo_diff_first_third_{metric}_v2"] = t - f
        out[f"combo_diff_second_third_{metric}_v2"] = t - s

    # 3艇 vs race平均との差
    for metric in metrics:
        combo_mean = out[f"combo_mean_{metric}_v2"]
        rm = race_mean[metric]

        if metric in {"exhibit", "st"}:
            # 小さい方が良い → race mean からの優位
            out[f"combo_vs_race_{metric}_adv_v2"] = rm - combo_mean
        else:
            out[f"combo_vs_race_{metric}_adv_v2"] = combo_mean - rm

    # 残り3艇平均との差
    for metric in metrics:
        combo_sum = out[f"combo_sum_{metric}_v2"]
        total_mean = race_mean[metric]
        total_sum = total_mean * 6.0
        other_mean = (total_sum - combo_sum) / 3.0

        if metric in {"exhibit", "st"}:
            out[f"combo_vs_others_{metric}_adv_v2"] = other_mean - out[f"combo_mean_{metric}_v2"]
        else:
            out[f"combo_vs_others_{metric}_adv_v2"] = out[f"combo_mean_{metric}_v2"] - other_mean

    # 順序の自然さ
    out["combo_forward_order_flag_v2"] = (
        (combo_first < combo_second) & (combo_second < combo_third)
    ).astype(int)
    out["combo_reverse_order_flag_v2"] = (
        (combo_first > combo_second) & (combo_second > combo_third)
    ).astype(int)

    out["combo_first_inner2_flag_v2"] = combo_first.isin([1, 2]).astype(int)
    out["combo_second_inner3_flag_v2"] = combo_second.isin([1, 2, 3]).astype(int)
    out["combo_third_outer3_flag_v2"] = combo_third.isin([4, 5, 6]).astype(int)

    # lane構成のバランス
    out["combo_lane_std_v2"] = pd.concat([combo_first, combo_second, combo_third], axis=1).std(axis=1)
    out["combo_lane_mean_v2"] = (combo_first + combo_second + combo_third) / 3.0

    # 3艇の総合スコア
    out["combo_strength_score_v2"] = (
        out["combo_mean_ability_index_v2"] * 0.35
        + out["combo_mean_win_rate_v2"] * 0.20
        + out["combo_mean_place_rate_v2"] * 0.15
        + out["combo_mean_motor_rate_v2"] * 0.10
        + out["combo_mean_boat_rate_v2"] * 0.05
        + out["combo_vs_others_exhibit_adv_v2"] * 10.0 * 0.075
        + out["combo_vs_others_st_adv_v2"] * 10.0 * 0.075
    )

    # 1着候補主導の押し切りスコア
    out["combo_first_push_score_v2"] = (
        picked["first"]["ability_index"] * 0.35
        + picked["first"]["win_rate"] * 0.20
        + picked["first"]["place_rate"] * 0.10
        + picked["first"]["motor_rate"] * 0.10
        + picked["first"]["boat_rate"] * 0.05
        + (race_mean["st"] - picked["first"]["st"]) * 10.0 * 0.10
        + (race_mean["exhibit"] - picked["first"]["exhibit"]) * 10.0 * 0.10
    )

    # 2着3着の厚み
    out["combo_follow_score_v2"] = (
        picked["second"]["ability_index"] * 0.30
        + picked["third"]["ability_index"] * 0.25
        + picked["second"]["place_rate"] * 0.20
        + picked["third"]["place_rate"] * 0.15
        + (race_mean["st"] - picked["second"]["st"]) * 10.0 * 0.05
        + (race_mean["st"] - picked["third"]["st"]) * 10.0 * 0.05
    )

    # 数値整形
    for c in out.columns:
        if c not in {"race_key", "date", "venue", "combo", "y_combo"}:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out.copy()


def _eval_binary_model(model, feature_cols: List[str], df: pd.DataFrame) -> Dict[str, float]:
    races = 0
    top1 = 0
    top3 = 0
    top5 = 0

    if df.empty:
        return {"races": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0}

    for race_key, df120 in df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue

        x = df120[feature_cols].copy()
        for c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
        x = x.replace([np.inf, -np.inf], 0.0)

        prob = model.predict_proba(x)[:, 1]
        prob_map = {c: float(p) for c, p in zip(df120["combo"].astype(str).tolist(), prob)}
        s = sum(prob_map.values())
        if s > 0:
            prob_map = {k: v / s for k, v in prob_map.items()}

        y_combo = _safe_str(df120.iloc[0].get("y_combo"))

        races += 1
        top1 += _topk_hits(prob_map, y_combo, 1)
        top3 += _topk_hits(prob_map, y_combo, 3)
        top5 += _topk_hits(prob_map, y_combo, 5)

    return {
        "races": races,
        "top1": top1 / races if races else 0.0,
        "top3": top3 / races if races else 0.0,
        "top5": top5 / races if races else 0.0,
    }


def train_one_venue(df: pd.DataFrame, venue: str) -> None:
    CatBoostClassifier = _require_catboost()

    vdf = df[df["venue"].astype(str).map(_normalize_venue) == venue].copy()
    if vdf.empty:
        print(f"[SKIP] {venue}: no rows")
        return

    train_df, valid_df, test_df = _time_split(vdf)

    print(f"\n===== {venue} =====")
    print("train rows:", len(train_df))
    print("valid rows:", len(valid_df))
    print("test rows :", len(test_df))

    if train_df.empty:
        print(f"[SKIP] {venue}: train empty")
        return

    train_df = _add_binary_v2_features(train_df)
    if not valid_df.empty:
        valid_df = _add_binary_v2_features(valid_df)
    if not test_df.empty:
        test_df = _add_binary_v2_features(test_df)

    train_df["is_hit"] = _build_is_hit(train_df)
    if not valid_df.empty:
        valid_df["is_hit"] = _build_is_hit(valid_df)
    if not test_df.empty:
        test_df["is_hit"] = _build_is_hit(test_df)

    x_train = _sanitize_x(train_df)
    y_train = train_df["is_hit"].astype(int)

    feature_cols = x_train.columns.tolist()
    if not feature_cols:
        raise RuntimeError(f"{venue}: feature_cols empty")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=500,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
    )

    model.fit(x_train, y_train)

    valid_eval = _eval_binary_model(model, feature_cols, valid_df) if not valid_df.empty else {}
    test_eval = _eval_binary_model(model, feature_cols, test_df) if not test_df.empty else {}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}.cbm"
    meta_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_meta.json"

    model.save_model(str(model_path))

    meta = {
        "model_type": "catboost_binary_v2",
        "venue": venue,
        "feature_cols": feature_cols,
        "train_to": TRAIN_TO,
        "valid_to": VALID_TO,
        "n_rows_train": int(len(train_df)),
        "n_rows_valid": int(len(valid_df)),
        "n_rows_test": int(len(test_df)),
        "feature_count": len(feature_cols),
        "valid_eval": valid_eval,
        "test_eval": test_eval,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved:", model_path)
    print("feature_count:", len(feature_cols))
    print("valid_eval:", valid_eval)
    print("test_eval :", test_eval)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    required_cols = {"date", "venue", "race_key", "combo", "y_combo"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    for venue in TARGET_VENUES:
        try:
            train_one_venue(df, venue)
        except Exception as e:
            print(f"[ERROR] {venue}: {e}")


if __name__ == "__main__":
    main()
