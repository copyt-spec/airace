from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"
MODEL_OUT_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_search_tmp.joblib"
SEARCH_RESULT_PATH = PROJECT_ROOT / "data" / "models" / "manual_weight_search_result_light.json"

DROP_COLS = {
    "race_key",
    "date",
    "venue",
    "race_no",
    "combo",
    "y_combo",
    "y_class",
}

TARGET_VENUES = {"丸亀", "児島", "戸田"}

ROWS_PER_RACE = 120
CHUNK_SIZE = 30000
RANDOM_STATE = 42

MAX_ROWS = 36000
MAX_RACES = MAX_ROWS // ROWS_PER_RACE
N_TRIALS = 120

BASE_MANUAL_WEIGHTS = {
    "blend_ratio": 0.08,
    "weights": {
        "lane": 0.23,
        "motor": 0.09,
        "boat": 0.05,
        "st": 0.02,
        "exhibit": 0.05,
        "win_rate": 0.10,
        "place_rate": 0.10,
        "grade": 0.03,
        "ability": 0.10,
        "course_fit": 0.18,
        "weather": 0.01,
        "other": 0.04,
    },
}

BASE_VENUE_MANUAL_WEIGHTS = {
    "丸亀": {
        "blend_ratio": 0.08,
        "weights": {
            "lane": 0.26,
            "motor": 0.08,
            "boat": 0.04,
            "st": 0.02,
            "exhibit": 0.05,
            "win_rate": 0.10,
            "place_rate": 0.09,
            "grade": 0.03,
            "ability": 0.09,
            "course_fit": 0.19,
            "weather": 0.01,
            "other": 0.04,
        },
    },
    "児島": {
        "blend_ratio": 0.08,
        "weights": {
            "lane": 0.18,
            "motor": 0.10,
            "boat": 0.06,
            "st": 0.03,
            "exhibit": 0.07,
            "win_rate": 0.10,
            "place_rate": 0.10,
            "grade": 0.03,
            "ability": 0.11,
            "course_fit": 0.15,
            "weather": 0.03,
            "other": 0.04,
        },
    },
    "戸田": {
        "blend_ratio": 0.08,
        "weights": {
            "lane": 0.21,
            "motor": 0.09,
            "boat": 0.05,
            "st": 0.03,
            "exhibit": 0.06,
            "win_rate": 0.10,
            "place_rate": 0.10,
            "grade": 0.03,
            "ability": 0.10,
            "course_fit": 0.17,
            "weather": 0.02,
            "other": 0.04,
        },
    },
}


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _format_venue(v: str) -> str:
    s = str(v or "").strip()
    if not s:
        return "不明"

    s2 = s.replace("BOAT RACE", "").replace("ボートレース", "").strip()

    if "丸亀" in s or "丸亀" in s2 or s in {"15", "015"}:
        return "丸亀"
    if "児島" in s or "児島" in s2 or s in {"16", "016"}:
        return "児島"
    if "戸田" in s or "戸田" in s2 or s in {"2", "02", "002"}:
        return "戸田"

    return s


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


def _build_combo_to_class() -> Dict[str, int]:
    labels = _build_all_trifecta_labels()
    return {combo: i for i, combo in enumerate(labels)}


def _get_columns(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _build_usecols(all_cols: List[str]) -> List[str]:
    return all_cols


def _sanitize_x(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    x = df[feature_cols].copy()

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0).astype(np.float32)

    x = x.replace([np.inf, -np.inf], 0.0)
    return x


def _split_complete_races(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    counts = df.groupby("race_key").size()
    complete_keys = counts[counts == ROWS_PER_RACE].index.tolist()
    partial_keys = counts[counts != ROWS_PER_RACE].index.tolist()

    complete_df = df[df["race_key"].isin(complete_keys)].copy()
    partial_df = df[df["race_key"].isin(partial_keys)].copy()
    return complete_df, partial_df


def _load_sampled_races_light(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    all_cols = _get_columns(path)
    usecols = _build_usecols(all_cols)

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_races: List[pd.DataFrame] = []
    kept_race_total = 0
    carry_df = pd.DataFrame()

    print("loading sampled races (light mode)...")

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            low_memory=True,
        ),
        start=1,
    ):
        if "race_key" not in chunk.columns:
            raise RuntimeError("race_key column missing.")
        if "y_class" not in chunk.columns:
            raise RuntimeError("y_class column missing.")
        if "combo" not in chunk.columns:
            raise RuntimeError("combo column missing.")

        chunk["venue"] = chunk["venue"].astype(str).map(_format_venue)
        chunk = chunk[chunk["venue"].isin(TARGET_VENUES)].copy()

        chunk["y_class"] = pd.to_numeric(chunk["y_class"], errors="coerce")
        chunk = chunk[chunk["y_class"].notna()]
        chunk = chunk[chunk["y_class"] >= 0]

        if chunk.empty:
            print(f"chunk {i}: no target venue rows")
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
        remain_races = MAX_RACES - kept_race_total
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

        print(f"chunk {i}: kept_races={kept_race_total}/{MAX_RACES}")

        if kept_race_total >= MAX_RACES:
            break

    if not sampled_races:
        raise RuntimeError("No sampled target-venue races loaded.")

    df = pd.concat(sampled_races, ignore_index=True)

    for col in df.columns:
        if col in {"date", "venue", "combo", "y_combo", "race_key"}:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if str(df[col].dtype).startswith(("float", "int")):
                df[col] = df[col].astype(np.float32)
        except Exception:
            pass

    return df


def _infer_feature_group(col: str) -> str:
    c = col.lower()

    if "motor" in c:
        return "motor"
    if "boat" in c:
        return "boat"
    if "exhibit" in c:
        return "exhibit"
    if c.endswith("_st") or "avg_st" in c or "_st_" in c:
        return "st"
    if "win_rate" in c:
        return "win_rate"
    if "place_rate" in c or "quinella" in c or "two_rate" in c:
        return "place_rate"
    if "grade" in c:
        return "grade"
    if "ability" in c:
        return "ability"
    if "course" in c or "lane_fit" in c:
        return "course_fit"
    if "weather" in c or "wind" in c or "wave" in c:
        return "weather"
    if "lane" in c:
        return "lane"
    return "other"


def _build_feature_groups(feature_cols: List[str]) -> Dict[str, str]:
    return {col: _infer_feature_group(col) for col in feature_cols}


def _normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in d.values())
    if s <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: max(v, 0.0) / s for k, v in d.items()}


def _split_races_train_valid_test(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    race_meta = (
        df.groupby("race_key", as_index=False)
        .agg(date=("date", "first"), venue=("venue", "first"))
        .sort_values(["date", "race_key"])
        .reset_index(drop=True)
    )

    n = len(race_meta)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_keys = set(race_meta.iloc[:n_train]["race_key"].tolist())
    valid_keys = set(race_meta.iloc[n_train:n_train + n_valid]["race_key"].tolist())
    test_keys = set(race_meta.iloc[n_train + n_valid:]["race_key"].tolist())

    train_df = df[df["race_key"].isin(train_keys)].copy()
    valid_df = df[df["race_key"].isin(valid_keys)].copy()
    test_df = df[df["race_key"].isin(test_keys)].copy()

    return train_df, valid_df, test_df


def _group_softmax(xs: np.ndarray) -> np.ndarray:
    xs = xs.astype(np.float64)
    m = np.max(xs)
    ex = np.exp(xs - m)
    s = np.sum(ex)
    if s <= 0:
        return np.ones_like(xs) / len(xs)
    return ex / s


def _calc_manual_score_matrix(
    race_df: pd.DataFrame,
    weights: Dict[str, float],
    feature_groups: Dict[str, str],
    feature_cols: List[str],
) -> np.ndarray:
    x = race_df[feature_cols].copy()

    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0).astype(np.float32)

    mat = x.to_numpy(dtype=np.float32)
    col_weights = np.zeros(len(feature_cols), dtype=np.float32)

    for i, col in enumerate(feature_cols):
        g = feature_groups.get(col, "other")
        w = weights.get(g, 0.0)

        col_lower = col.lower()
        if "st" in col_lower or "exhibit" in col_lower:
            w = -w

        col_weights[i] = np.float32(w)

    scores = mat @ col_weights
    return scores.astype(np.float64)


def _blend_probs(
    model_probs: np.ndarray,
    manual_probs: np.ndarray,
    blend_ratio: float,
) -> np.ndarray:
    blend_ratio = max(0.0, min(1.0, float(blend_ratio)))
    out = (1.0 - blend_ratio) * model_probs + blend_ratio * manual_probs
    s = float(out.sum())
    if s > 0:
        out = out / s
    return out


def _topk_hit_from_arrays(final_probs: np.ndarray, y_index: int, k: int) -> int:
    top_idx = np.argsort(final_probs)[::-1][:k]
    return 1 if y_index in top_idx else 0


def _score_metrics(top1: float, top3: float, top5: float) -> float:
    return top1 * 3.0 + top3 * 1.0 + top5 * 0.4


def _sample_weights_like(base: Dict[str, float], scale: float = 0.35) -> Dict[str, float]:
    out = {}
    for k, v in base.items():
        low = max(0.001, v * (1.0 - scale))
        high = max(low + 1e-9, v * (1.0 + scale))
        out[k] = random.uniform(low, high)
    return _normalize_weights(out)


def _sample_blend_ratio(base: float, scale: float = 0.4) -> float:
    low = max(0.02, base * (1.0 - scale))
    high = min(0.12, base * (1.0 + scale))
    if high < low:
        high = low
    return random.uniform(low, high)


def _prepare_eval_cache(
    df_eval: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: List[str],
    combo_to_class: Dict[str, int],
) -> List[Dict[str, Any]]:
    caches: List[Dict[str, Any]] = []

    model_classes = list(model.classes_)
    class_index_map = {int(c): i for i, c in enumerate(model_classes)}

    for race_key, race_df in df_eval.groupby("race_key", sort=False):
        if len(race_df) != ROWS_PER_RACE:
            continue

        row0 = race_df.iloc[0]
        venue = _format_venue(_safe_str(row0.get("venue")))
        y_combo = _safe_str(row0.get("y_combo"))

        x = race_df[feature_cols].copy()
        for col in feature_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0).astype(np.float32)

        proba = model.predict_proba(x)
        if proba.ndim == 1:
            proba = np.expand_dims(proba, axis=0)

        model_prob_arr = np.zeros(ROWS_PER_RACE, dtype=np.float64)
        combos = race_df["combo"].astype(str).tolist()

        for i, combo in enumerate(combos):
            class_id = combo_to_class.get(combo)
            if class_id is None:
                model_prob_arr[i] = 0.0
                continue

            class_pos = class_index_map.get(class_id)
            if class_pos is None:
                model_prob_arr[i] = 0.0
                continue

            model_prob_arr[i] = float(proba[i][class_pos])

        s = float(model_prob_arr.sum())
        if s > 0:
            model_prob_arr = model_prob_arr / s

        y_index = 0
        for idx, combo in enumerate(combos):
            if combo == y_combo:
                y_index = idx
                break

        caches.append({
            "race_key": race_key,
            "venue": venue,
            "y_combo": y_combo,
            "y_index": y_index,
            "race_df": race_df.copy(),
            "model_prob_arr": model_prob_arr,
            "combos": combos,
        })

    return caches


def _evaluate_setting_from_cache(
    eval_caches: List[Dict[str, Any]],
    feature_cols: List[str],
    feature_groups: Dict[str, str],
    global_setting: Dict[str, Any],
    venue_settings: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    total_races = 0
    top1 = 0
    top3 = 0
    top5 = 0

    venue_stats = {
        "丸亀": {"races": 0, "top1": 0, "top3": 0, "top5": 0},
        "児島": {"races": 0, "top1": 0, "top3": 0, "top5": 0},
        "戸田": {"races": 0, "top1": 0, "top3": 0, "top5": 0},
    }

    for cache in eval_caches:
        venue = cache["venue"]
        race_df = cache["race_df"]
        model_prob_arr = cache["model_prob_arr"]
        y_index = cache["y_index"]

        venue_setting = venue_settings.get(venue, global_setting)

        manual_scores = _calc_manual_score_matrix(
            race_df=race_df,
            weights=venue_setting["weights"],
            feature_groups=feature_groups,
            feature_cols=feature_cols,
        )
        manual_probs = _group_softmax(manual_scores)

        final_probs = _blend_probs(
            model_probs=model_prob_arr,
            manual_probs=manual_probs,
            blend_ratio=venue_setting["blend_ratio"],
        )

        h1 = _topk_hit_from_arrays(final_probs, y_index, 1)
        h3 = _topk_hit_from_arrays(final_probs, y_index, 3)
        h5 = _topk_hit_from_arrays(final_probs, y_index, 5)

        total_races += 1
        top1 += h1
        top3 += h3
        top5 += h5

        if venue in venue_stats:
            venue_stats[venue]["races"] += 1
            venue_stats[venue]["top1"] += h1
            venue_stats[venue]["top3"] += h3
            venue_stats[venue]["top5"] += h5

    if total_races == 0:
        return {
            "score": -999.0,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "venue_stats": venue_stats,
        }

    top1_rate = top1 / total_races
    top3_rate = top3 / total_races
    top5_rate = top5 / total_races

    score = _score_metrics(top1_rate, top3_rate, top5_rate)

    return {
        "score": score,
        "top1": top1_rate,
        "top3": top3_rate,
        "top5": top5_rate,
        "venue_stats": venue_stats,
    }


def main() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    combo_to_class = _build_combo_to_class()

    print("===== SEARCH MANUAL WEIGHTS LIGHT START =====")
    print(f"TARGET_VENUES={sorted(TARGET_VENUES)}")
    print(f"MAX_RACES={MAX_RACES}, N_TRIALS={N_TRIALS}")

    df = _load_sampled_races_light(DATASET_PATH)
    print("loaded shape:", df.shape)
    print("race count :", df["race_key"].nunique())

    required = {"race_key", "combo", "y_combo", "y_class", "venue", "date"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    print("\n===== ALL TARGET VENUE COUNTS =====")
    print(df["venue"].astype(str).value_counts())

    train_df, valid_df, test_df = _split_races_train_valid_test(df)

    print("\n===== VENUE CHECK =====")
    print("train unique venue:", sorted(train_df["venue"].astype(str).unique().tolist()))
    print("valid unique venue:", sorted(valid_df["venue"].astype(str).unique().tolist()))
    print("test  unique venue:", sorted(test_df["venue"].astype(str).unique().tolist()))

    print("train venue counts:")
    print(train_df["venue"].astype(str).value_counts())

    print("valid venue counts:")
    print(valid_df["venue"].astype(str).value_counts())

    print("test venue counts:")
    print(test_df["venue"].astype(str).value_counts())

    print("train races:", train_df["race_key"].nunique())
    print("valid races:", valid_df["race_key"].nunique())
    print("test  races:", test_df["race_key"].nunique())

    print("training base model...")
    x_train = _sanitize_x(train_df)
    y_train = train_df["y_class"].astype(int).copy()

    feature_cols = x_train.columns.tolist()
    feature_groups = _build_feature_groups(feature_cols)

    model = RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    joblib.dump(model, MODEL_OUT_PATH)

    print("precomputing valid/test cache...")
    valid_caches = _prepare_eval_cache(valid_df, model, feature_cols, combo_to_class)
    test_caches = _prepare_eval_cache(test_df, model, feature_cols, combo_to_class)
    print("valid cache races:", len(valid_caches))
    print("test  cache races:", len(test_caches))

    print("searching...")
    best_rows: List[Dict[str, Any]] = []

    for trial in range(1, N_TRIALS + 1):
        global_setting = {
            "blend_ratio": _sample_blend_ratio(BASE_MANUAL_WEIGHTS["blend_ratio"], scale=0.4),
            "weights": _sample_weights_like(BASE_MANUAL_WEIGHTS["weights"], scale=0.40),
        }

        venue_settings = {}
        for venue, conf in BASE_VENUE_MANUAL_WEIGHTS.items():
            venue_settings[venue] = {
                "blend_ratio": _sample_blend_ratio(conf["blend_ratio"], scale=0.4),
                "weights": _sample_weights_like(conf["weights"], scale=0.40),
            }

        result = _evaluate_setting_from_cache(
            eval_caches=valid_caches,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            global_setting=global_setting,
            venue_settings=venue_settings,
        )

        row = {
            "trial": trial,
            "score": result["score"],
            "top1": result["top1"],
            "top3": result["top3"],
            "top5": result["top5"],
            "venue_stats": result["venue_stats"],
            "global_setting": global_setting,
            "venue_settings": venue_settings,
        }
        best_rows.append(row)
        best_rows = sorted(best_rows, key=lambda x: x["score"], reverse=True)[:20]

        if trial % 10 == 0:
            best = best_rows[0]
            print(
                f"[trial {trial}] "
                f"best_score={best['score']:.6f} "
                f"top1={best['top1']:.4f} "
                f"top3={best['top3']:.4f} "
                f"top5={best['top5']:.4f}"
            )

    best = best_rows[0]

    print("\n===== BEST ON VALID =====")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    print("\n===== FINAL TEST WITH BEST VALID SETTING =====")
    final_test = _evaluate_setting_from_cache(
        eval_caches=test_caches,
        feature_cols=feature_cols,
        feature_groups=feature_groups,
        global_setting=best["global_setting"],
        venue_settings=best["venue_settings"],
    )
    print(json.dumps(final_test, ensure_ascii=False, indent=2))

    out = {
        "target_venues": sorted(TARGET_VENUES),
        "best_valid": best,
        "final_test": final_test,
        "n_trials": N_TRIALS,
        "max_races": MAX_RACES,
    }
    with open(SEARCH_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nsaved:", SEARCH_RESULT_PATH)
    print("===== SEARCH MANUAL WEIGHTS LIGHT END =====")


if __name__ == "__main__":
    main()
