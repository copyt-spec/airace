from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render.joblib"
META_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_meta.json"
LABELS_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_labels.json"

DROP_COLS = {
    "race_key",
    "date",
    "venue",
    "race_no",
    "combo",
    "y_combo",
    "y_class",
}

# まずは軽く確認
MAX_ROWS = 120000
CHUNK_SIZE = 50000
RANDOM_STATE = 42


def _get_columns(path: Path) -> List[str]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return cols


def _build_usecols(all_cols: List[str]) -> List[str]:
    # y_combo は学習に不要なので最初から読まない
    usecols = [c for c in all_cols if c != "y_combo"]
    return usecols


def _sanitize_x(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    x = df[feature_cols].copy()

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)

    x = x.replace([np.inf, -np.inf], 0.0)
    return x


def _load_sampled_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    all_cols = _get_columns(path)
    usecols = _build_usecols(all_cols)

    chunks = []
    kept_total = 0
    seen_total = 0

    rng = np.random.default_rng(RANDOM_STATE)

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            low_memory=True,
        ),
        start=1,
    ):
        seen_total += len(chunk)

        if "y_class" not in chunk.columns:
            raise RuntimeError("y_class column missing in chunk.")

        chunk["y_class"] = pd.to_numeric(chunk["y_class"], errors="coerce")
        chunk = chunk[chunk["y_class"].notna()]
        chunk = chunk[chunk["y_class"] >= 0]

        if chunk.empty:
            print(f"chunk {i}: empty after y_class filter")
            continue

        chunk["y_class"] = chunk["y_class"].astype(np.int16)

        remain = MAX_ROWS - kept_total
        if remain <= 0:
            break

        if len(chunk) > remain:
            # 最後は必要分だけランダムに取る
            idx = rng.choice(chunk.index.to_numpy(), size=remain, replace=False)
            chunk = chunk.loc[idx].copy()

        chunks.append(chunk)
        kept_total += len(chunk)

        print(
            f"chunk {i}: seen_total={seen_total} "
            f"chunk_kept={len(chunk)} kept_total={kept_total}/{MAX_ROWS}"
        )

        if kept_total >= MAX_ROWS:
            break

    if not chunks:
        raise RuntimeError("No valid chunks loaded.")

    df = pd.concat(chunks, ignore_index=True)
    return df


def train():
    print("===== TRAIN TRIFECTA 120CLS START =====")

    print("[1/8] loading sampled chunks...")
    df = _load_sampled_chunks(DATASET_PATH)
    print("sampled shape:", df.shape)

    required_cols = ["race_key", "combo", "y_class"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    print("[2/8] building X / y...")
    x = _sanitize_x(df)
    y = df["y_class"].astype(int).copy()

    leaked = [c for c in x.columns if c in DROP_COLS]
    if leaked:
        raise RuntimeError(f"Leak columns still included in X: {leaked}")

    print("feature matrix shape:", x.shape)
    print("target shape:", y.shape)
    print("feature cols count:", len(x.columns))

    print("[3/8] fitting model...")
    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x, y)

    print("[4/8] preparing artifacts...")
    labels = sorted(df["combo"].astype(str).unique().tolist())
    feature_cols = x.columns.tolist()

    meta = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_rows": int(len(df)),
        "model_name": "trifecta120_model_render",
        "drop_cols": sorted(list(DROP_COLS)),
        "max_rows": MAX_ROWS,
        "chunk_size": CHUNK_SIZE,
    }

    print("[5/8] saving model/meta/labels...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("saved model :", MODEL_PATH)
    print("saved meta  :", META_PATH)
    print("saved labels:", LABELS_PATH)

    print("[6/8] verifying saved feature cols...")
    ng = [c for c in feature_cols if c in {
        "combo", "y_combo", "y_class", "race_key", "date", "venue", "race_no"
    }]
    print("NG cols in feature_cols:", ng)

    print("[7/8] done")
    print("===== TRAIN TRIFECTA 120CLS END =====")


if __name__ == "__main__":
    train()
