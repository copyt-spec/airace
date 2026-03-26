from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from engine.feature_builder_render_v2 import add_render_features_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

BASE_DROP_COLS = {
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

MAX_ROWS_PER_VENUE = 36000
MAX_RACES_PER_VENUE = MAX_ROWS_PER_VENUE // ROWS_PER_RACE

# 時系列分割
VALID_FROM = "20251201"
TEST_FROM = "20260301"

# リーク疑い文字列
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


def _require_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError as e:
        raise RuntimeError("lightgbm が入っていません。`pip install lightgbm`") from e


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


def _split_complete_races(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        required = {"venue", "race_key", "y_class", "combo", "date"}
        missing = [c for c in required if c not in chunk.columns]
        if missing:
            raise RuntimeError(f"missing columns in chunk: {missing}")

        chunk["venue"] = chunk["venue"].astype(str).map(_normalize_venue)
        chunk = chunk[chunk["venue"] == venue].copy()
        if chunk.empty:
            continue

        chunk["date"] = chunk["date"].astype(str)
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


def _is_leak_col(col: str) -> bool:
    c = str(col).lower()
    if col in BASE_DROP_COLS:
        return True
    return any(k.lower() in c for k in LEAK_KEYWORDS)


def _sanitize_x(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if not _is_leak_col(c)]
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


def _time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = df["date"].astype(str)

    train_df = df[df["date"] < VALID_FROM].copy()
    valid_df = df[(df["date"] >= VALID_FROM) & (df["date"] < TEST_FROM)].copy()
    test_df = df[df["date"] >= TEST_FROM].copy()

    return train_df, valid_df, test_df


def _topk_hits(prob_map: dict[str, float], y_combo: str, k: int) -> int:
    if not prob_map or not y_combo:
        return 0
    topk = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
    combos = [c for c, _ in topk]
    return 1 if y_combo in combos else 0


def _predict_lgbm_prob_map(model, feature_cols: list[str], labels: list[str], df120: pd.DataFrame) -> dict[str, float]:
    x = df120.copy()

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
    out: dict[str, float] = {}

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


def _eval_lgbm(model, feature_cols: list[str], labels: list[str], df: pd.DataFrame) -> dict[str, float]:
    races = 0
    top1 = 0
    top3 = 0
    top5 = 0

    for race_key, df120 in df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue
        y_combo = str(df120.iloc[0].get("y_combo", "")).strip()
        prob_map = _predict_lgbm_prob_map(model, feature_cols, labels, df120)

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


def train_one_venue(venue: str) -> None:
    lgb = _require_lightgbm()

    vdf = _load_sampled_venue_races(DATASET_PATH, venue)
    print(f"{venue} raw shape:", vdf.shape)

    vdf = add_render_features_v2(vdf)
    print(f"{venue} feature-added shape:", vdf.shape)

    train_df, valid_df, test_df = _time_split(vdf)

    print(f"{venue} train rows:", len(train_df))
    print(f"{venue} valid rows:", len(valid_df))
    print(f"{venue} test  rows:", len(test_df))

    if train_df.empty:
        raise RuntimeError(f"{venue}: train_df empty")
    if valid_df.empty:
        print(f"[WARN] {venue}: valid_df empty")
    if test_df.empty:
        print(f"[WARN] {venue}: test_df empty")

    x_train = _sanitize_x(train_df)
    y_train = train_df["y_class"].astype(int).copy()

    feature_cols = x_train.columns.tolist()
    labels = _build_all_trifecta_labels()

    print(f"{venue} X_train shape:", x_train.shape)
    print(f"{venue} feature count:", len(feature_cols))

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
    model.fit(x_train, y_train)

    valid_eval = _eval_lgbm(model, feature_cols, labels, valid_df) if not valid_df.empty else {}
    test_eval = _eval_lgbm(model, feature_cols, labels, test_df) if not test_df.empty else {}

    out_model = MODEL_DIR / f"trifecta120_lgbm_{venue}_safe.joblib"
    out_meta = MODEL_DIR / f"trifecta120_lgbm_{venue}_safe_meta.json"
    out_labels = MODEL_DIR / f"trifecta120_lgbm_{venue}_safe_labels.json"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)

    meta = {
        "model_type": "lightgbm_safe",
        "venue": venue,
        "feature_cols": feature_cols,
        "n_rows_total": int(len(vdf)),
        "n_rows_train": int(len(train_df)),
        "n_rows_valid": int(len(valid_df)),
        "n_rows_test": int(len(test_df)),
        "n_features": len(feature_cols),
        "num_class": 120,
        "max_rows_per_venue": MAX_ROWS_PER_VENUE,
        "chunk_size": CHUNK_SIZE,
        "valid_from": VALID_FROM,
        "test_from": TEST_FROM,
        "leak_keywords": LEAK_KEYWORDS,
        "valid_eval": valid_eval,
        "test_eval": test_eval,
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {venue}")
    print(" saved:", out_model)
    print(" valid_eval:", valid_eval)
    print(" test_eval :", test_eval)


def main() -> None:
    for venue in TARGET_VENUES:
        try:
            train_one_venue(venue)
        except Exception as e:
            print(f"[ERROR] {venue}: {e}")


if __name__ == "__main__":
    main()
