from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

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

ROWS_PER_RACE = 120
MAX_ROWS = 120000
MAX_RACES = MAX_ROWS // ROWS_PER_RACE
CHUNK_SIZE = 50000
RANDOM_STATE = 42

# ===== 共通手動重み（探索結果反映版）=====
MANUAL_WEIGHTS = {
    "blend_ratio": 0.11199410102458938,
    "memo_blend_ratio": "学習モデル確率に対して、手動補正確率をどれだけ混ぜるか。探索結果ベース。",
    "weights": {
        "lane": 0.21780787676400223,
        "motor": 0.10848856366976276,
        "boat": 0.06591435640121725,
        "st": 0.024136301383069034,
        "exhibit": 0.064752105205749,
        "win_rate": 0.14632496800196412,
        "place_rate": 0.08173347406059221,
        "grade": 0.0198973381768106,
        "ability": 0.07768895997293125,
        "course_fit": 0.13578596014468514,
        "weather": 0.012220571676520733,
        "other": 0.04524952454269569
    },
    "memo": {
        "lane": "枠・艇番の影響。探索結果では強め。",
        "motor": "モーター2連率などモーター性能。",
        "boat": "ボート2連率などボート性能。",
        "st": "ST・平均ST。小さい方が有利。",
        "exhibit": "展示タイム。小さい方が有利。",
        "win_rate": "勝率系。探索結果でやや強め。",
        "place_rate": "2連率・連対率系。",
        "grade": "級別系。",
        "ability": "ability_index, prev_ability_index 系。",
        "course_fit": "コース適性・lane_fit・course別成績。",
        "weather": "風・波・天候。",
        "other": "上記に入らない特徴量。"
    }
}

# ===== 会場別手動重み（探索結果反映版）=====
VENUE_MANUAL_WEIGHTS = {
    "丸亀": {
        "blend_ratio": 0.06194973061801623,
        "weights": {
            "lane": 0.3118628164236701,
            "motor": 0.10042201195518549,
            "boat": 0.030379836466106445,
            "st": 0.022463197133241356,
            "exhibit": 0.06197874625347187,
            "win_rate": 0.07154248729639569,
            "place_rate": 0.11687761303219206,
            "grade": 0.042569366596550924,
            "ability": 0.06390904132848654,
            "course_fit": 0.12196011846962714,
            "weather": 0.008787990044293568,
            "other": 0.04724677500077888
        },
        "memo": "探索結果ベース。丸亀は lane を強く、place_rate もやや重視。"
    },
    "児島": {
        "blend_ratio": 0.10932306164517733,
        "weights": {
            "lane": 0.1691707926618211,
            "motor": 0.12007768471721138,
            "boat": 0.040620921128822506,
            "st": 0.03542332572620059,
            "exhibit": 0.07901849333778699,
            "win_rate": 0.0698247847021226,
            "place_rate": 0.12478781206839898,
            "grade": 0.03934967472123105,
            "ability": 0.12175285989955857,
            "course_fit": 0.10709203987575469,
            "weather": 0.042633568387874944,
            "other": 0.05024804277321662
        },
        "memo": "探索結果ベース。児島は motor / exhibit / ability / place_rate をやや重視。"
    },
    "戸田": {
        "blend_ratio": 0.07022104097974073,
        "weights": {
            "lane": 0.1878341837930803,
            "motor": 0.07655129589246386,
            "boat": 0.04766688721912631,
            "st": 0.02484898912591752,
            "exhibit": 0.07284952680434177,
            "win_rate": 0.11934786684385927,
            "place_rate": 0.06493972613436667,
            "grade": 0.03895722797680701,
            "ability": 0.10517291547297446,
            "course_fit": 0.2037141744215618,
            "weather": 0.02212341892643188,
            "other": 0.035993787389069026
        },
        "memo": "探索結果ベース。戸田は course_fit と win_rate、exhibit をやや重視。"
    }
}


def _get_columns(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _build_usecols(all_cols: List[str]) -> List[str]:
    return [c for c in all_cols if c != "y_combo"]


def _sanitize_x(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    x = df[feature_cols].copy()

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)

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


def _split_complete_races(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    counts = df.groupby("race_key").size()
    complete_keys = counts[counts == ROWS_PER_RACE].index.tolist()
    partial_keys = counts[counts != ROWS_PER_RACE].index.tolist()

    complete_df = df[df["race_key"].isin(complete_keys)].copy()
    partial_df = df[df["race_key"].isin(partial_keys)].copy()
    return complete_df, partial_df


def _load_sampled_races(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    all_cols = _get_columns(path)
    usecols = _build_usecols(all_cols)

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_races: List[pd.DataFrame] = []
    kept_race_total = 0
    seen_total_rows = 0
    carry_df = pd.DataFrame()

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            low_memory=True,
        ),
        start=1,
    ):
        seen_total_rows += len(chunk)

        if "race_key" not in chunk.columns:
            raise RuntimeError("race_key column missing in chunk.")
        if "y_class" not in chunk.columns:
            raise RuntimeError("y_class column missing in chunk.")
        if "combo" not in chunk.columns:
            raise RuntimeError("combo column missing in chunk.")

        chunk["y_class"] = pd.to_numeric(chunk["y_class"], errors="coerce")
        chunk = chunk[chunk["y_class"].notna()]
        chunk = chunk[chunk["y_class"] >= 0]

        if chunk.empty:
            print(f"chunk {i}: empty after y_class filter")
            continue

        chunk["y_class"] = chunk["y_class"].astype(np.int16)

        if not carry_df.empty:
            chunk = pd.concat([carry_df, chunk], ignore_index=True)
            carry_df = pd.DataFrame()

        complete_df, partial_df = _split_complete_races(chunk)
        carry_df = partial_df.copy()

        if complete_df.empty:
            print(f"chunk {i}: no complete races yet")
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

        race_count = picked_df["race_key"].nunique()
        sampled_races.append(picked_df)
        kept_race_total += race_count

        print(
            f"chunk {i}: seen_rows={seen_total_rows} "
            f"picked_races={race_count} kept_races={kept_race_total}/{MAX_RACES}"
        )

        if kept_race_total >= MAX_RACES:
            break

    if not sampled_races:
        raise RuntimeError("No valid sampled races loaded.")

    df = pd.concat(sampled_races, ignore_index=True)

    race_sizes = df.groupby("race_key").size()
    bad = race_sizes[race_sizes != ROWS_PER_RACE]
    if len(bad) > 0:
        raise RuntimeError(f"Found incomplete races after sampling: {bad.head().to_dict()}")

    return df


def _print_race_sampling_debug(df: pd.DataFrame) -> None:
    race_count = df["race_key"].nunique()
    row_count = len(df)
    combo_count = df["combo"].nunique()
    class_count = df["y_class"].nunique()

    print("sampled race count :", race_count)
    print("sampled row count  :", row_count)
    print("unique combos      :", combo_count)
    print("unique y_class     :", class_count)

    if race_count > 0:
        sample_key = df["race_key"].iloc[0]
        sample_df = df[df["race_key"] == sample_key]
        print("sample race_key    :", sample_key)
        print("sample race rows   :", len(sample_df))
        print("sample race combos :", sample_df["combo"].nunique())


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


def train():
    print("===== TRAIN TRIFECTA 120CLS START =====")

    print("[1/8] loading sampled races...")
    df = _load_sampled_races(DATASET_PATH)
    print("sampled shape:", df.shape)
    _print_race_sampling_debug(df)

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
    labels = _build_all_trifecta_labels()
    feature_cols = x.columns.tolist()
    feature_groups = _build_feature_groups(feature_cols)

    missing_classes = sorted(set(range(ROWS_PER_RACE)) - set(pd.Series(y).unique().tolist()))

    meta = {
        "feature_cols": feature_cols,
        "feature_groups": feature_groups,
        "manual_weights": MANUAL_WEIGHTS,
        "venue_manual_weights": VENUE_MANUAL_WEIGHTS,
        "n_features": len(feature_cols),
        "n_rows": int(len(df)),
        "n_races": int(df["race_key"].nunique()),
        "model_name": "trifecta120_model_render",
        "drop_cols": sorted(list(DROP_COLS)),
        "rows_per_race": ROWS_PER_RACE,
        "max_rows": MAX_ROWS,
        "max_races": MAX_RACES,
        "chunk_size": CHUNK_SIZE,
        "missing_classes": missing_classes,
        "manual_weight_source": {
            "search_result": "manual_weight_search_result_light.json",
            "final_test_score": 0.5288888888888889,
            "final_test_top1": 0.1111111111111111,
            "final_test_top3": 0.13333333333333333,
            "final_test_top5": 0.15555555555555556
        }
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

    print("[7/8] class coverage...")
    print("labels count        :", len(labels))
    print("trained classes     :", len(model.classes_))
    print("missing classes     :", len(missing_classes))
    if missing_classes:
        print("example missing ids :", missing_classes[:10])

    print("[8/8] done")
    print("===== TRAIN TRIFECTA 120CLS END =====")


if __name__ == "__main__":
    train()
