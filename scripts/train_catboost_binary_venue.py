# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

VENUE_DATASETS = {
    "丸亀": DATASET_DIR / "trifecta_train_small_marugame.csv",
    "戸田": DATASET_DIR / "trifecta_train_small_toda.csv",
    "児島": DATASET_DIR / "trifecta_train_small_kojima.csv",
}


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError as e:
        raise RuntimeError("catboost が入っていません。 `pip install catboost`") from e


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

    # comboを split して、first/second/third の対象艇を引くためだけに使う
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

    # comboで指定された1着/2着/3着候補の特徴量を作る
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
    # 重要:
    # combo_first_lane / combo_second_lane / combo_third_lane は
    # 学習特徴量から外す
    cols = [
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


def load_dataset(csv_path: Path, venue: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    usecols = [
        "date", "venue", "race_key", "race_no", "combo", "y",
        "wave_cm", "weather", "wind_dir", "wind_speed_mps",
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
        raise RuntimeError(f"{venue}: dataset empty after venue filter")

    df["date"] = df["date"].astype(str)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype("int8")

    return df.reset_index(drop=True)


def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    race_dates = (
        df[["race_key", "date"]]
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )

    n = len(race_dates)
    if n < 10:
        raise RuntimeError("race count too small for split")

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_keys = set(race_dates.iloc[:train_end]["race_key"].tolist())
    valid_keys = set(race_dates.iloc[train_end:valid_end]["race_key"].tolist())
    test_keys = set(race_dates.iloc[valid_end:]["race_key"].tolist())

    train_df = df[df["race_key"].isin(train_keys)].copy()
    valid_df = df[df["race_key"].isin(valid_keys)].copy()
    test_df = df[df["race_key"].isin(test_keys)].copy()

    return train_df, valid_df, test_df


def evaluate_topk(prob: List[float], race_keys: List[str], combos: List[str], y: List[int]) -> Dict[str, float]:
    eval_df = pd.DataFrame({
        "race_key": race_keys,
        "combo": combos,
        "prob": prob,
        "y": y,
    })

    races = 0
    top1 = 0
    top3 = 0
    top5 = 0

    for _, g in eval_df.groupby("race_key", sort=False):
        if len(g) != 120:
            continue

        races += 1
        g = g.sort_values("prob", ascending=False).reset_index(drop=True)

        top1 += int(g.iloc[:1]["y"].max())
        top3 += int(g.iloc[:3]["y"].max())
        top5 += int(g.iloc[:5]["y"].max())

    if races == 0:
        return {"races": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0}

    return {
        "races": races,
        "top1": top1 / races,
        "top3": top3 / races,
        "top5": top5 / races,
    }


def train_one_pattern(
    venue: str,
    df: pd.DataFrame,
    with_racer_no: bool,
) -> Tuple[Any, Dict[str, Any]]:
    CatBoostClassifier = _require_catboost()

    work = add_feature_block(df.copy())
    feature_cols = get_feature_cols(with_racer_no)

    train_df, valid_df, test_df = split_by_date(work)

    x_train = sanitize_x(train_df, feature_cols)
    y_train = train_df["y"].astype("int8")

    x_valid = sanitize_x(valid_df, feature_cols)
    y_valid = valid_df["y"].astype("int8")

    x_test = sanitize_x(test_df, feature_cols)
    y_test = test_df["y"].astype("int8")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=600,
        depth=6,
        learning_rate=0.04,
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
        class_weights=[1.0, 10.0],
    )

    model.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )

    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    valid_eval = evaluate_topk(
        prob=valid_prob.tolist(),
        race_keys=valid_df["race_key"].astype(str).tolist(),
        combos=valid_df["combo"].astype(str).tolist(),
        y=valid_df["y"].astype(int).tolist(),
    )
    test_eval = evaluate_topk(
        prob=test_prob.tolist(),
        race_keys=test_df["race_key"].astype(str).tolist(),
        combos=test_df["combo"].astype(str).tolist(),
        y=test_df["y"].astype(int).tolist(),
    )

    info = {
        "venue": venue,
        "with_racer_no": with_racer_no,
        "feature_cols": feature_cols,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "valid_eval": valid_eval,
        "test_eval": test_eval,
        "train_params": {
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.04,
            "class_weights": [1.0, 10.0],
        },
    }
    return model, info


def save_model_and_meta(model: Any, info: Dict[str, Any], suffix: str) -> Tuple[Path, Path]:
    venue = info["venue"]
    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_{suffix}.cbm"
    meta_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_{suffix}_meta.json"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return model_path, meta_path


def main() -> None:
    print("===== train_catboost_binary_venue NO-COMBO-LANE START =====")

    for venue, csv_path in VENUE_DATASETS.items():
        print(f"\n===== {venue} =====")
        print("dataset:", csv_path)

        df = load_dataset(csv_path, venue)
        print("rows:", len(df))
        print("races:", df["race_key"].nunique())

        results: List[Tuple[str, Any, Dict[str, Any]]] = []

        for suffix, with_racer_no in [
            ("with_racer_no", True),
            ("without_racer_no", False),
        ]:
            print(f"\n--- pattern: {suffix} ---")
            model, info = train_one_pattern(
                venue=venue,
                df=df,
                with_racer_no=with_racer_no,
            )

            model_path, meta_path = save_model_and_meta(model, info, suffix=suffix)

            print("train rows:", info["train_rows"])
            print("valid rows:", info["valid_rows"])
            print("test rows :", info["test_rows"])
            print("saved model:", model_path)
            print("saved meta :", meta_path)
            print("valid_eval:", info["valid_eval"])
            print("test_eval :", info["test_eval"])

            results.append((suffix, model, info))

        best_suffix, _, best_info = sorted(
            results,
            key=lambda x: (
                x[2]["test_eval"]["top5"],
                x[2]["test_eval"]["top3"],
                x[2]["test_eval"]["top1"],
            ),
            reverse=True,
        )[0]

        print(f"\n>>> BEST for {venue}: {best_suffix}")
        print("best test_eval:", best_info["test_eval"])

    print("\n===== train_catboost_binary_venue NO-COMBO-LANE DONE =====")


if __name__ == "__main__":
    main()
