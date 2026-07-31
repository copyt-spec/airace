from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from engine.venue_registry import VENUE_ORDER, normalize_venue_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"


def _load_dataset_for_venue(src: Path, venue: str, chunksize: int = 200_000) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(src)

    venue = normalize_venue_name(venue)
    print(f"Loading dataset for venue: {venue}")
    print(f"source: {src}")

    chunks: List[pd.DataFrame] = []
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(src, low_memory=False, chunksize=chunksize):
        total_rows += len(chunk)

        if "venue" not in chunk.columns:
            raise ValueError("dataset must contain venue column")

        chunk["venue"] = chunk["venue"].astype(str).map(normalize_venue_name)
        hit = chunk[chunk["venue"] == venue].copy()

        if not hit.empty:
            matched_rows += len(hit)
            chunks.append(hit)

        print(f"scanned={total_rows:,} matched={matched_rows:,}")

    if not chunks:
        raise RuntimeError(f"no data for venue: {venue}")

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        raise RuntimeError(f"no data for venue: {venue}")

    return df


def _add_feature_block(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    candidate_cols = [
        "combo_first_lane", "combo_second_lane", "combo_third_lane",
        "wind_dir_code", "weather_code", "wind_speed_mps", "wave_cm",

        "lane1_boat", "lane1_course", "lane1_exhibit", "lane1_motor", "lane1_racer_no", "lane1_st",
        "lane2_boat", "lane2_course", "lane2_exhibit", "lane2_motor", "lane2_racer_no", "lane2_st",
        "lane3_boat", "lane3_course", "lane3_exhibit", "lane3_motor", "lane3_racer_no", "lane3_st",
        "lane4_boat", "lane4_course", "lane4_exhibit", "lane4_motor", "lane4_racer_no", "lane4_st",
        "lane5_boat", "lane5_course", "lane5_exhibit", "lane5_motor", "lane5_racer_no", "lane5_st",
        "lane6_boat", "lane6_course", "lane6_exhibit", "lane6_motor", "lane6_racer_no", "lane6_st",

        "lane1_st_inv", "lane2_st_inv", "lane3_st_inv", "lane4_st_inv", "lane5_st_inv", "lane6_st_inv",
        "lane1_exhibit_inv", "lane2_exhibit_inv", "lane3_exhibit_inv", "lane4_exhibit_inv", "lane5_exhibit_inv", "lane6_exhibit_inv",
        "lane1_course_diff", "lane2_course_diff", "lane3_course_diff", "lane4_course_diff", "lane5_course_diff", "lane6_course_diff",

        "first_st", "second_st", "third_st",
        "first_exhibit", "second_exhibit", "third_exhibit",
        "first_course", "second_course", "third_course",
        "first_motor", "second_motor", "third_motor",
        "first_boat", "second_boat", "third_boat",
        "first_racer_no", "second_racer_no", "third_racer_no",

        "first_second_st_diff", "first_third_st_diff",
        "first_second_exhibit_diff", "first_third_exhibit_diff",
    ]
    return [c for c in candidate_cols if c in df.columns]


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if "y" not in df.columns:
        raise ValueError("dataset must contain y column")

    work = _add_feature_block(df.copy())
    feature_cols = _get_feature_cols(work)
    if not feature_cols:
        raise RuntimeError("no feature columns found")

    x = work[feature_cols].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    y = pd.to_numeric(work["y"], errors="coerce").fillna(0).astype(int)
    return x, y, feature_cols


def _build_save_paths(model_dir: Path, venue: str) -> Tuple[Path, Path]:
    model_path = model_dir / f"trifecta_binary_catboost_{venue}_with_racer_no.cbm"
    meta_path = model_dir / f"trifecta_binary_catboost_{venue}_with_racer_no_meta.json"
    return model_path, meta_path


def _train_one_venue(
    src: Path,
    venue: str,
    test_size: float,
    random_state: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    verbose_eval: int,
) -> None:
    venue = normalize_venue_name(venue)
    print("\n" + "=" * 80)
    print("TRAIN VENUE:", venue)
    print("=" * 80)

    df = _load_dataset_for_venue(src, venue)

    print("rows           :", len(df))
    if "date" in df.columns:
        print("min date       :", str(df["date"].astype(str).min()))
        print("max date       :", str(df["date"].astype(str).max()))

    x, y, feature_cols = _prepare_xy(df)

    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    print("positive rows  :", pos_count)
    print("negative rows  :", neg_count)

    if pos_count == 0 or neg_count == 0:
        raise RuntimeError(f"class balance invalid for venue: {venue}")

    stratify_y = y if y.nunique() >= 2 else None

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    pos_train = int((y_train == 1).sum())
    neg_train = int((y_train == 0).sum())
    scale_pos_weight = float(neg_train / max(pos_train, 1))

    print("train rows      :", len(x_train))
    print("valid rows      :", len(x_valid))
    print("scale_pos_weight:", round(scale_pos_weight, 4))
    print("feature count   :", len(feature_cols))

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=random_state,
        verbose=verbose_eval,
        scale_pos_weight=scale_pos_weight,
        allow_writing_files=False,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )

    valid_pred = model.predict_proba(x_valid)[:, 1]
    valid_logloss = float(log_loss(y_valid, valid_pred))

    try:
        valid_auc = float(roc_auc_score(y_valid, valid_pred))
    except Exception:
        valid_auc = 0.0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = _build_save_paths(MODEL_DIR, venue)
    model.save_model(str(model_path))

    meta = {
        "venue": venue,
        "model_type": "catboost_binary",
        "with_racer_no": True,
        "dataset_path": str(src),
        "rows_total": int(len(df)),
        "rows_train": int(len(x_train)),
        "rows_valid": int(len(x_valid)),
        "positive_total": pos_count,
        "negative_total": neg_count,
        "feature_cols": feature_cols,
        "params": {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "test_size": test_size,
            "scale_pos_weight": scale_pos_weight,
        },
        "metrics": {
            "valid_logloss": valid_logloss,
            "valid_auc": valid_auc,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved model     :", model_path)
    print("saved meta      :", meta_path)
    print("valid logloss   :", round(valid_logloss, 6))
    print("valid auc       :", round(valid_auc, 6))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DATASET_PATH))
    parser.add_argument("--venue", default="", help="会場名")
    parser.add_argument("--all", action="store_true", help="全会場学習")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--verbose_eval", type=int, default=100)
    args = parser.parse_args()

    src = Path(args.src)

    if args.all:
        venues = list(VENUE_ORDER)
    elif args.venue:
        venues = [normalize_venue_name(args.venue)]
    else:
        raise ValueError("either --venue or --all is required")

    for venue in venues:
        try:
            _train_one_venue(
                src=src,
                venue=venue,
                test_size=args.test_size,
                random_state=args.random_state,
                iterations=args.iterations,
                depth=args.depth,
                learning_rate=args.learning_rate,
                verbose_eval=args.verbose_eval,
            )
        except Exception as e:
            print(f"[TRAIN_SKIP] venue={venue} reason={e}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
