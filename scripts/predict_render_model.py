from __future__ import annotations

import gc
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
OUTPUT_PATH = OUTPUT_DIR / "render_model_predictions.csv"

# ここは環境に応じて調整OK
CHUNK_SIZE = 100000


def _load_json(path: Path):
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
    return out.astype(np.float32)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main():
    print("===== PREDICT RENDER MODEL START =====")

    print("[1/7] checking files...")
    _require_file(MODEL_PATH, "model")
    _require_file(META_PATH, "meta")
    _require_file(LABELS_PATH, "labels")
    _require_file(DATASET_PATH, "dataset")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[2/7] loading model/meta/labels...")
    model = joblib.load(MODEL_PATH)
    meta = _load_json(META_PATH)
    labels = _load_json(LABELS_PATH)

    feature_cols: List[str] = meta.get("feature_cols", [])
    if not feature_cols:
        raise RuntimeError("feature_cols not found in meta json.")

    if not isinstance(labels, list) or len(labels) == 0:
        raise RuntimeError("labels json is empty or invalid.")

    label_to_index: Dict[str, int] = {str(label): i for i, label in enumerate(labels)}

    print(f"model       : {MODEL_PATH}")
    print(f"meta        : {META_PATH}")
    print(f"labels      : {LABELS_PATH}")
    print(f"feature_cols: {len(feature_cols)}")
    print(f"num_labels  : {len(labels)}")

    print("[3/7] counting dataset rows...")
    try:
        total_rows = sum(1 for _ in open(DATASET_PATH, "r", encoding="utf-8", errors="ignore")) - 1
    except Exception:
        total_rows = -1

    if total_rows >= 0:
        print(f"dataset rows: {total_rows}")
    else:
        print("dataset rows: unknown")

    print("[4/7] removing old output if exists...")
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    print("[5/7] chunk prediction start...")
    required_cols = ["race_key", "date", "venue", "race_no", "combo", "y_combo"]

    total_written = 0
    chunk_no = 0

    reader = pd.read_csv(DATASET_PATH, low_memory=False, chunksize=CHUNK_SIZE)

    for chunk in reader:
        chunk_no += 1
        print(f"\n--- chunk {chunk_no} start ---")
        print(f"raw chunk rows: {len(chunk)}")

        for c in required_cols:
            if c not in chunk.columns:
                raise RuntimeError(f"Required column missing in dataset: {c}")

        x = _sanitize_features(chunk, feature_cols)
        print(f"feature matrix shape: {x.shape}")

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Loaded model does not support predict_proba().")

        proba = model.predict_proba(x)
        if proba.ndim != 2:
            raise RuntimeError(f"Unexpected predict_proba output shape: {proba.shape}")

        combo_series = chunk["combo"].astype(str)
        class_idx_arr = combo_series.map(label_to_index).fillna(-1).astype(int).to_numpy()

        row_idx_arr = np.arange(len(chunk), dtype=np.int64)
        score_arr = np.zeros(len(chunk), dtype=np.float32)

        valid_mask = (class_idx_arr >= 0) & (class_idx_arr < proba.shape[1])
        score_arr[valid_mask] = proba[row_idx_arr[valid_mask], class_idx_arr[valid_mask]]

        out_df = chunk[required_cols].copy()
        out_df["score"] = score_arr
        out_df["is_hit"] = out_df["combo"].astype(str) == out_df["y_combo"].astype(str)

        out_df.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=(not OUTPUT_PATH.exists()),
            index=False,
            encoding="utf-8-sig",
        )

        total_written += len(out_df)

        print(f"chunk {chunk_no} written rows: {len(out_df)}")
        print(f"total written rows         : {total_written}")

        del chunk
        del x
        del proba
        del out_df
        del combo_series
        del class_idx_arr
        del row_idx_arr
        del score_arr
        gc.collect()

    print("\n[6/7] final check...")
    if not OUTPUT_PATH.exists():
        raise RuntimeError(f"Output was not created: {OUTPUT_PATH}")

    pred_df = pd.read_csv(OUTPUT_PATH, low_memory=False)
    print(f"output shape: {pred_df.shape}")

    print("[7/7] done")
    print(f"saved: {OUTPUT_PATH}")
    print("===== PREDICT RENDER MODEL END =====")


if __name__ == "__main__":
    main()
