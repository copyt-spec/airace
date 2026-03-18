# scripts/test_render_model.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render.joblib"
META_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_meta.json"
LABELS_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_labels.json"
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"

CHUNK_SIZE = 50000


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0

    out = out[feature_cols].copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out.replace([np.inf, -np.inf], 0.0)
    return out


def _load_one_race_chunk() -> pd.DataFrame:
    print("[2/5] loading ONE race only (chunk)...")

    for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE, low_memory=True):
        if "race_key" not in chunk.columns:
            continue

        keys = chunk["race_key"].astype(str).unique().tolist()
        for race_key in keys:
            race_df = chunk[chunk["race_key"].astype(str) == str(race_key)].copy()
            if len(race_df) >= 100:
                print(f"picked race_key: {race_key} rows={len(race_df)}")
                return race_df

    raise RuntimeError("No valid race found in dataset.")


def _build_combo_score_map(
    race_df: pd.DataFrame,
    proba: np.ndarray,
    model_classes: np.ndarray,
    all_labels: List[str],
) -> Dict[str, float]:
    """
    model.predict_proba の列順は model.classes_ 基準。
    labels.json の順番とは一致しない可能性がある。
    """
    class_to_col = {int(cls): col_idx for col_idx, cls in enumerate(model_classes)}
    combo_to_class = {combo: idx for idx, combo in enumerate(all_labels)}

    combo_to_score: Dict[str, float] = {}

    for i, (_, row) in enumerate(race_df.iterrows()):
        combo = str(row["combo"])

        if combo not in combo_to_class:
            combo_to_score[combo] = 0.0
            continue

        class_id = combo_to_class[combo]

        if class_id not in class_to_col:
            # 学習サンプル内に存在しなかったクラス
            combo_to_score[combo] = 0.0
            continue

        col_idx = class_to_col[class_id]
        combo_to_score[combo] = float(proba[i, col_idx])

    return combo_to_score


def main():
    print("===== TEST RENDER MODEL START =====")

    print("[1/5] loading model/meta/labels...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing meta: {META_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")

    model = joblib.load(MODEL_PATH)
    meta = _load_json(META_PATH)
    labels = _load_json(LABELS_PATH)

    feature_cols: List[str] = meta.get("feature_cols", [])
    if not feature_cols:
        raise RuntimeError("feature_cols not found in meta json.")

    if not hasattr(model, "classes_"):
        raise RuntimeError("Loaded model does not have classes_ attribute.")

    model_classes = np.array(model.classes_)
    print("feature cols:", len(feature_cols))
    print("labels:", len(labels))
    print("model classes:", len(model_classes))

    race_df = _load_one_race_chunk()

    print("[3/5] preparing features...")
    x = _sanitize_features(race_df, feature_cols)

    print("[4/5] predicting...")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba().")

    proba = model.predict_proba(x)
    if proba.ndim != 2:
        raise RuntimeError(f"Unexpected predict_proba output shape: {proba.shape}")

    combo_to_score = _build_combo_score_map(
        race_df=race_df,
        proba=proba,
        model_classes=model_classes,
        all_labels=labels,
    )

    result = race_df[["race_key", "date", "venue", "race_no", "combo", "y_combo"]].copy()
    result["score"] = result["combo"].astype(str).map(combo_to_score).fillna(0.0)
    result["is_hit"] = result["combo"].astype(str) == result["y_combo"].astype(str)
    result = result.sort_values("score", ascending=False).reset_index(drop=True)

    print("\n===== TOP 20 PREDICTIONS =====")
    print(result.head(20).to_string(index=False))

    y_combo = str(result["y_combo"].iloc[0])
    hit_rows = result[result["combo"].astype(str) == y_combo].copy()

    print("\n===== HIT INFO =====")
    print(f"actual combo : {y_combo}")

    if hit_rows.empty:
        print("hit rank     : NOT FOUND")
        print("hit score    : NOT FOUND")
    else:
        hit_rank = int(hit_rows.index[0]) + 1
        hit_score = float(hit_rows["score"].iloc[0])
        print(f"hit rank     : {hit_rank}")
        print(f"hit score    : {hit_score:.6f}")

    missing_classes = sorted(
        set(range(len(labels))) - set(int(c) for c in model_classes.tolist())
    )
    print("\n===== CLASS COVERAGE =====")
    print(f"total labels      : {len(labels)}")
    print(f"trained classes   : {len(model_classes)}")
    print(f"missing classes   : {len(missing_classes)}")
    if missing_classes:
        print("example missing class ids:", missing_classes[:10])

    print("\n===== TEST RENDER MODEL END =====")


if __name__ == "__main__":
    main()
