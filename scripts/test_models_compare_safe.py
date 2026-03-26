from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.feature_builder_render_v2 import add_render_features_v2  # type: ignore
from engine.model_loader import BoatRaceModel  # type: ignore


FEATURES_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

ROWS_PER_RACE = 120
TARGET_VENUES = ["丸亀", "児島", "戸田"]
CHUNK_SIZE = 30000
RANDOM_STATE = 42

MAX_RACES_PER_VENUE_TEST = 60
TEST_FROM = "20260101"

LEAK_KEYWORDS = [
    "result",
    "finish",
    "rank_result",
    "pay",
    "payout",
    "return",
    "odds",
    "払戻",
    "配当",
    "着順",
    "確定",
    "的中",
    "hit",
    "y_",
]


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _normalize_venue(v: str) -> str:
    s = _safe_str(v)
    if "丸亀" in s:
        return "丸亀"
    if "児島" in s:
        return "児島"
    if "戸田" in s:
        return "戸田"
    return s


def _topk_hits(prob_map: Dict[str, float], y_combo: str, k: int) -> int:
    if not prob_map or not y_combo:
        return 0
    topk = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
    combos = [c for c, _ in topk]
    return 1 if y_combo in combos else 0


def _get_columns(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _split_complete_races(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    counts = df.groupby("race_key").size()
    complete_keys = counts[counts == ROWS_PER_RACE].index.tolist()
    partial_keys = counts[counts != ROWS_PER_RACE].index.tolist()

    complete_df = df[df["race_key"].isin(complete_keys)].copy()
    partial_df = df[df["race_key"].isin(partial_keys)].copy()
    return complete_df, partial_df


def _load_sampled_venue_races_for_test(path: Path, venue: str, max_races: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    rng = np.random.default_rng(RANDOM_STATE)
    usecols = _get_columns(path)

    sampled_races: List[pd.DataFrame] = []
    kept_race_total = 0
    carry_df = pd.DataFrame()

    print(f"\n===== TEST LOAD VENUE: {venue} =====")
    print(f"MAX_RACES_PER_VENUE_TEST={max_races}")

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            low_memory=True,
        ),
        start=1,
    ):
        required = {"venue", "race_key", "y_class", "combo", "date"}
        missing = [c for c in required if c not in chunk.columns]
        if missing:
            raise RuntimeError(f"missing columns: {missing}")

        chunk["venue"] = chunk["venue"].astype(str).map(_normalize_venue)
        chunk = chunk[chunk["venue"] == venue].copy()

        if chunk.empty:
            continue

        chunk["date"] = chunk["date"].astype(str)
        chunk = chunk[chunk["date"] >= TEST_FROM].copy()
        if chunk.empty:
            continue

        chunk["y_class"] = pd.to_numeric(chunk["y_class"], errors="coerce")
        chunk = chunk[chunk["y_class"].notna()]
        chunk = chunk[chunk["y_class"] >= 0]
        if chunk.empty:
            continue

        chunk["y_class"] = chunk["y_class"].astype(np.int16)

        if "combo" in chunk.columns:
            chunk["combo"] = chunk["combo"].astype(str)
        if "y_combo" in chunk.columns:
            chunk["y_combo"] = chunk["y_combo"].astype(str)

        if not carry_df.empty:
            chunk = pd.concat([carry_df, chunk], ignore_index=True)
            carry_df = pd.DataFrame()

        complete_df, partial_df = _split_complete_races(chunk)
        carry_df = partial_df.copy()

        if complete_df.empty:
            continue

        race_keys = complete_df["race_key"].drop_duplicates().tolist()
        remain_races = max_races - kept_race_total
        if remain_races <= 0:
            break

        if len(race_keys) > remain_races:
            picked_keys = rng.choice(np.array(race_keys), size=remain_races, replace=False).tolist()
        else:
            picked_keys = race_keys

        picked_df = complete_df[complete_df["race_key"].isin(picked_keys)].copy()

        valid_counts = picked_df.groupby("race_key").size()
        valid_keys = valid_counts[valid_counts == ROWS_PER_RACE].index.tolist()
        picked_df = picked_df[picked_df["race_key"].isin(valid_keys)].copy()

        sampled_races.append(picked_df)
        kept_race_total += picked_df["race_key"].nunique()

        print(f"chunk {i}: kept_races={kept_race_total}/{max_races}")

        if kept_race_total >= max_races:
            break

    if not sampled_races:
        raise RuntimeError(f"{venue}: test sampled races not found")

    df = pd.concat(sampled_races, ignore_index=True)

    race_sizes = df.groupby("race_key").size()
    bad = race_sizes[race_sizes != ROWS_PER_RACE]
    if len(bad) > 0:
        raise RuntimeError(f"{venue}: incomplete races found: {bad.head().to_dict()}")

    return df


def _is_leak_col(col: str) -> bool:
    c = str(col).lower()
    base_drop = {
        "race_key", "date", "venue", "race_no", "combo", "y_combo", "y_class"
    }
    if col in base_drop:
        return True
    return any(k.lower() in c for k in LEAK_KEYWORDS)


def _predict_lgbm_prob_map(model, meta: Dict[str, Any], labels: List[str], df120: pd.DataFrame) -> Dict[str, float]:
    x = df120.copy()
    feature_cols = meta["feature_cols"]

    for c in feature_cols:
        if c not in x.columns:
            x[c] = 0.0

    x = x[feature_cols].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x = x.replace([np.inf, -np.inf], 0.0)

    proba = model.predict_proba(x)
    classes = [int(c) for c in model.classes_]
    class_pos_map = {cls: pos for pos, cls in enumerate(classes)}
    label_to_class_id = {label: i for i, label in enumerate(labels)}

    combos = df120["combo"].astype(str).tolist()
    out: Dict[str, float] = {}

    for row_idx, combo in enumerate(combos):
        class_id = label_to_class_id.get(combo)
        if class_id is None:
            out[combo] = 0.0
            continue
        class_pos = class_pos_map.get(class_id)
        if class_pos is None:
            out[combo] = 0.0
            continue
        out[combo] = float(proba[row_idx, class_pos])

    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


def _eval_rf_for_venue(rf_model: BoatRaceModel, df: pd.DataFrame, venue: str) -> Dict[str, float]:
    races = 0
    top1 = 0
    top3 = 0
    top5 = 0

    for race_key, df120 in df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue

        y_combo = _safe_str(df120.iloc[0].get("y_combo"))
        prob_map = rf_model.predict_proba(df120)

        races += 1
        top1 += _topk_hits(prob_map, y_combo, 1)
        top3 += _topk_hits(prob_map, y_combo, 3)
        top5 += _topk_hits(prob_map, y_combo, 5)

    return {
        "venue": venue,
        "races": races,
        "top1": top1 / races if races else 0.0,
        "top3": top3 / races if races else 0.0,
        "top5": top5 / races if races else 0.0,
    }


def _eval_lgbm_for_venue(df: pd.DataFrame, venue: str) -> Dict[str, float]:
    model_path = MODEL_DIR / f"trifecta120_lgbm_{venue}_safe.joblib"
    meta_path = MODEL_DIR / f"trifecta120_lgbm_{venue}_safe_meta.json"
    labels_path = MODEL_DIR / f"trifecta120_lgbm_{venue}_safe_labels.json"

    if not (model_path.exists() and meta_path.exists() and labels_path.exists()):
        return {
            "venue": venue,
            "races": 0,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
        }

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    races = 0
    top1 = 0
    top3 = 0
    top5 = 0

    for race_key, df120 in df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue

        y_combo = _safe_str(df120.iloc[0].get("y_combo"))
        prob_map = _predict_lgbm_prob_map(model, meta, labels, df120)

        races += 1
        top1 += _topk_hits(prob_map, y_combo, 1)
        top3 += _topk_hits(prob_map, y_combo, 3)
        top5 += _topk_hits(prob_map, y_combo, 5)

    return {
        "venue": venue,
        "races": races,
        "top1": top1 / races if races else 0.0,
        "top3": top3 / races if races else 0.0,
        "top5": top5 / races if races else 0.0,
    }


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)

    rf_model = BoatRaceModel(
        model_path=str(MODEL_DIR / "trifecta120_model_render.joblib"),
        meta_path=str(MODEL_DIR / "trifecta120_model_render_meta.json"),
        debug=False,
    )

    summary: Dict[str, Any] = {
        "RandomForest": {},
        "LightGBM_safe": {},
    }

    for venue in TARGET_VENUES:
        try:
            vdf = _load_sampled_venue_races_for_test(
                FEATURES_PATH,
                venue=venue,
                max_races=MAX_RACES_PER_VENUE_TEST,
            )
            print(f"{venue} raw shape:", vdf.shape)

            vdf = add_render_features_v2(vdf)
            print(f"{venue} feature-added shape:", vdf.shape)

            # 念のため怪しい列確認
            suspicious = [c for c in vdf.columns if _is_leak_col(c)]
            print(f"{venue} suspicious cols count:", len(suspicious))

            rf_result = _eval_rf_for_venue(rf_model, vdf, venue)
            lgbm_result = _eval_lgbm_for_venue(vdf, venue)

            summary["RandomForest"][venue] = rf_result
            summary["LightGBM_safe"][venue] = lgbm_result

        except Exception as e:
            summary["RandomForest"][venue] = {"error": str(e)}
            summary["LightGBM_safe"][venue] = {"error": str(e)}

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
