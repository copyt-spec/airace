from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from engine.feature_builder_render_v2 import add_render_features_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

DROP_COLS = {
    "race_key",
    "date",
    "venue",
    "race_no",
    "combo",
    "y_combo",
    "y_class",
}

TARGET_VENUES = ["丸亀", "児島", "戸田"]
ROWS_PER_RACE = 120
CHUNK_SIZE = 30000
RANDOM_STATE = 42

# 1会場あたり最大何レースまで使うか
MAX_ROWS_PER_VENUE = 36000
MAX_RACES_PER_VENUE = MAX_ROWS_PER_VENUE // ROWS_PER_RACE


def _require_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError as e:
        raise RuntimeError(
            "lightgbm が入っていません。`pip install lightgbm` してください。"
        ) from e


def _normalize_venue(v: str) -> str:
    s = str(v or "").strip()
    if "丸亀" in s:
        return "丸亀"
    if "児島" in s:
        return "児島"
    if "戸田" in s:
        return "戸田"
    return s


def _get_columns(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _split_complete_races(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    counts = df.groupby("race_key").size()
    complete_keys = counts[counts == ROWS_PER_RACE].index.tolist()
    partial_keys = counts[counts != ROWS_PER_RACE].index.tolist()

    complete_df = df[df["race_key"].isin(complete_keys)].copy()
    partial_df = df[df["race_key"].isin(partial_keys)].copy()
    return complete_df, partial_df


def _load_sampled_venue_races(path: Path, venue: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    rng = np.random.default_rng(RANDOM_STATE)
    usecols = _get_columns(path)

    sampled_races: List[pd.DataFrame] = []
    kept_race_total = 0
    carry_df = pd.DataFrame()

    print(f"\n===== LOAD VENUE: {venue} =====")
    print(f"MAX_RACES_PER_VENUE={MAX_RACES_PER_VENUE}")

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            low_memory=True,
        ),
        start=1,
    ):
        if "venue" not in chunk.columns:
            raise RuntimeError("venue column missing")
        if "race_key" not in chunk.columns:
            raise RuntimeError("race_key column missing")
        if "y_class" not in chunk.columns:
            raise RuntimeError("y_class column missing")
        if "combo" not in chunk.columns:
            raise RuntimeError("combo column missing")

        chunk["venue"] = chunk["venue"].astype(str).map(_normalize_venue)
        chunk = chunk[chunk["venue"] == venue].copy()

        if chunk.empty:
            print(f"chunk {i}: no {venue}")
            continue

        chunk["y_class"] = pd.to_numeric(chunk["y_class"], errors="coerce")
        chunk = chunk[chunk["y_class"].notna()]
        chunk = chunk[chunk["y_class"] >= 0]

        if chunk.empty:
            print(f"chunk {i}: empty after y_class filter")
            continue

        chunk["y_class"] = chunk["y_class"].astype(np.int16)

        if "date" in chunk.columns:
            chunk["date"] = chunk["date"].astype(str)
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
            print(f"chunk {i}: no complete races")
            continue

        race_keys = complete_df["race_key"].drop_duplicates().tolist()
        remain_races = MAX_RACES_PER_VENUE - kept_race_total
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

        print(f"chunk {i}: kept_races={kept_race_total}/{MAX_RACES_PER_VENUE}")

        if kept_race_total >= MAX_RACES_PER_VENUE:
            break

    if not sampled_races:
        raise RuntimeError(f"{venue}: valid sampled races not found")

    df = pd.concat(sampled_races, ignore_index=True)

    race_sizes = df.groupby("race_key").size()
    bad = race_sizes[race_sizes != ROWS_PER_RACE]
    if len(bad) > 0:
        raise RuntimeError(f"{venue}: incomplete races found: {bad.head().to_dict()}")

    return df


def _sanitize_x(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    x = df[feature_cols].copy()

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0).astype(np.float32)

    x = x.replace([np.inf, -np.inf], 0.0)
    return x


def _build_all_trifecta_labels() -> List[str]:
    labels: List[str] = []
    for a in range(1, 7):
        for b in range(1, 7):
            if b == a:
                continue
            for c in range(1, 7):
                if c == a or c == b:
                    continue
                labels.append(f"{a}-{b}-{c}")
    return labels


def train_one_venue(venue: str) -> None:
    lgb = _require_lightgbm()

    vdf = _load_sampled_venue_races(DATASET_PATH, venue)
    print(f"{venue} raw shape:", vdf.shape)

    # 特徴量追加
    vdf = add_render_features_v2(vdf)
    print(f"{venue} feature-added shape:", vdf.shape)

    x = _sanitize_x(vdf)
    y = vdf["y_class"].astype(int).copy()

    feature_cols = x.columns.tolist()
    labels = _build_all_trifecta_labels()

    print(f"{venue} X shape:", x.shape)
    print(f"{venue} y shape:", y.shape)
    print(f"{venue} feature count:", len(feature_cols))

    # 軽量パラメータ
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=120,
        n_estimators=120,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=-1,
    )
    model.fit(x, y)

    out_model = MODEL_DIR / f"trifecta120_lgbm_{venue}.joblib"
    out_meta = MODEL_DIR / f"trifecta120_lgbm_{venue}_meta.json"
    out_labels = MODEL_DIR / f"trifecta120_lgbm_{venue}_labels.json"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)

    meta = {
        "model_type": "lightgbm",
        "venue": venue,
        "feature_cols": feature_cols,
        "n_rows": int(len(vdf)),
        "n_features": len(feature_cols),
        "num_class": 120,
        "max_rows_per_venue": MAX_ROWS_PER_VENUE,
        "chunk_size": CHUNK_SIZE,
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {venue}")
    print(" saved:", out_model)


def main() -> None:
    for venue in TARGET_VENUES:
        try:
            train_one_venue(venue)
        except Exception as e:
            print(f"[ERROR] {venue}: {e}")


if __name__ == "__main__":
    main()
