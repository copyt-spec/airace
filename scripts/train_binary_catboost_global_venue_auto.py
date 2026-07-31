from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _normalize_venue_name(v: Any) -> str:
    s = str(v or "").strip()
    for venue in [
        "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
        "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
        "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
    ]:
        if venue in s:
            return venue
    return s


def _load_dataset(src: Path, chunksize: int = 200_000) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(src)

    chunks: List[pd.DataFrame] = []
    total = 0

    for chunk in pd.read_csv(src, low_memory=False, chunksize=chunksize):
        total += len(chunk)
        if "venue" not in chunk.columns:
            raise ValueError("dataset must contain venue column")

        chunk["venue"] = chunk["venue"].astype(str).map(_normalize_venue_name)
        chunks.append(chunk)
        print(f"loaded={total:,}")

    if not chunks:
        raise RuntimeError("dataset empty")

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        raise RuntimeError("dataset empty after concat")

    return df


def _split_combo_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "combo" not in out.columns:
        raise ValueError("combo column is required")

    parts = out["combo"].astype(str).str.split("-", expand=True)
    if parts.shape[1] < 3:
        raise ValueError("combo split failed")

    out["combo_first_lane"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype("int16")
    out["combo_second_lane"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype("int16")
    out["combo_third_lane"] = pd.to_numeric(parts[2], errors="coerce").fillna(0).astype("int16")

    return out


def _build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _split_combo_cols(df.copy())

    wind_map = {
        "無風": 0,
        "北": 1, "北東": 2, "東": 3, "南東": 4,
        "南": 5, "南西": 6, "西": 7, "北西": 8,
    }
    weather_map = {
        "晴": 1, "晴れ": 1,
        "曇": 2, "曇り": 2,
        "雨": 3,
        "雪": 4,
    }

    if "wind_dir" not in out.columns:
        out["wind_dir"] = ""
    if "weather" not in out.columns:
        out["weather"] = ""

    out["wind_dir_code"] = out["wind_dir"].astype(str).map(wind_map).fillna(0).astype("int16")
    out["weather_code"] = out["weather"].astype(str).map(weather_map).fillna(0).astype("int16")

    if "wind_speed_mps" not in out.columns:
        out["wind_speed_mps"] = 0.0
    if "wave_cm" not in out.columns:
        out["wave_cm"] = 0.0

    out["wind_speed_mps"] = pd.to_numeric(out["wind_speed_mps"], errors="coerce").fillna(0).astype("float32")
    out["wave_cm"] = pd.to_numeric(out["wave_cm"], errors="coerce").fillna(0).astype("float32")

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

    return out


def _build_venue_auto_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "y" not in work.columns:
        raise ValueError("y column is required")

    work["y"] = pd.to_numeric(work["y"], errors="coerce").fillna(0).astype(int)

    if "trifecta_payout" not in work.columns:
        work["trifecta_payout"] = 0.0
    work["trifecta_payout"] = pd.to_numeric(work["trifecta_payout"], errors="coerce").fillna(0.0)

    pos = work[work["y"] == 1].copy()
    if pos.empty:
        raise RuntimeError("positive rows not found")

    venue_base = pos.groupby("venue").agg(
        venue_hit_count=("y", "sum"),
        venue_avg_payout=("trifecta_payout", "mean"),
        venue_med_payout=("trifecta_payout", "median"),
        venue_std_payout=("trifecta_payout", "std"),
    ).reset_index()

    venue_base["venue_std_payout"] = venue_base["venue_std_payout"].fillna(0.0)

    first_rate = (
        pos.groupby(["venue", "combo_first_lane"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4, 5, 6], fill_value=0)
    )
    first_rate = first_rate.div(first_rate.sum(axis=1), axis=0)
    first_rate.columns = [f"venue_first_lane_{c}_rate" for c in first_rate.columns]
    first_rate = first_rate.reset_index()

    second_rate = (
        pos.groupby(["venue", "combo_second_lane"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4, 5, 6], fill_value=0)
    )
    second_rate = second_rate.div(second_rate.sum(axis=1), axis=0)
    second_rate.columns = [f"venue_second_lane_{c}_rate" for c in second_rate.columns]
    second_rate = second_rate.reset_index()

    third_rate = (
        pos.groupby(["venue", "combo_third_lane"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4, 5, 6], fill_value=0)
    )
    third_rate = third_rate.div(third_rate.sum(axis=1), axis=0)
    third_rate.columns = [f"venue_third_lane_{c}_rate" for c in third_rate.columns]
    third_rate = third_rate.reset_index()

    stats = venue_base.merge(first_rate, on="venue", how="left")
    stats = stats.merge(second_rate, on="venue", how="left")
    stats = stats.merge(third_rate, on="venue", how="left")

    stats = stats.fillna(0.0)

    # 会場の荒れ度っぽい指標
    # 平均配当 × 標準偏差 を軽く正規化したもの
    stats["venue_chaos_index"] = (
        np.log1p(stats["venue_avg_payout"].clip(lower=0))
        * (1.0 + np.log1p(stats["venue_std_payout"].clip(lower=0)))
    )

    return stats


def _attach_venue_auto_features(df: pd.DataFrame, venue_stats: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(venue_stats, on="venue", how="left")

    auto_cols = [c for c in venue_stats.columns if c != "venue"]
    for c in auto_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 自分のcombo laneに対応する会場傾向を拾う
    out["venue_first_lane_match_rate"] = 0.0
    out["venue_second_lane_match_rate"] = 0.0
    out["venue_third_lane_match_rate"] = 0.0

    for lane in range(1, 7):
        out.loc[out["combo_first_lane"] == lane, "venue_first_lane_match_rate"] = out.loc[
            out["combo_first_lane"] == lane, f"venue_first_lane_{lane}_rate"
        ]
        out.loc[out["combo_second_lane"] == lane, "venue_second_lane_match_rate"] = out.loc[
            out["combo_second_lane"] == lane, f"venue_second_lane_{lane}_rate"
        ]
        out.loc[out["combo_third_lane"] == lane, "venue_third_lane_match_rate"] = out.loc[
            out["combo_third_lane"] == lane, f"venue_third_lane_{lane}_rate"
        ]

    out["venue_combo_lane_prior"] = (
        out["venue_first_lane_match_rate"]
        + out["venue_second_lane_match_rate"]
        + out["venue_third_lane_match_rate"]
    ).astype("float32")

    return out


def _downsample_for_binary(df: pd.DataFrame, neg_pos_ratio: int, random_state: int) -> pd.DataFrame:
    work = df.copy()
    work["y"] = pd.to_numeric(work["y"], errors="coerce").fillna(0).astype(int)

    pos = work[work["y"] == 1].copy()
    neg = work[work["y"] == 0].copy()

    if pos.empty:
        raise RuntimeError("positive rows not found")
    if neg.empty:
        raise RuntimeError("negative rows not found")

    max_neg = len(pos) * int(neg_pos_ratio)
    if len(neg) > max_neg:
        neg = neg.sample(n=max_neg, random_state=random_state)

    out = pd.concat([pos, neg], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def _split_by_race_key(df: pd.DataFrame, valid_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "race_key" not in df.columns:
        if {"venue", "date", "race_no"}.issubset(df.columns):
            df = df.copy()
            df["race_key"] = (
                df["venue"].astype(str) + "_"
                + df["date"].astype(str) + "_"
                + df["race_no"].astype(str)
            )
        else:
            raise ValueError("race_key or (venue,date,race_no) is required")

    race_keys = df["race_key"].drop_duplicates().sample(frac=1.0, random_state=random_state).tolist()
    cut = int(len(race_keys) * (1.0 - valid_ratio))

    train_keys = set(race_keys[:cut])
    valid_keys = set(race_keys[cut:])

    train_df = df[df["race_key"].isin(train_keys)].copy()
    valid_df = df[df["race_key"].isin(valid_keys)].copy()

    return train_df, valid_df


def _build_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feature_cols = [
        "venue",
        "combo_first_lane", "combo_second_lane", "combo_third_lane",
        "wind_dir", "weather",
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

        "first_second_st_diff",
        "first_third_st_diff",
        "first_second_exhibit_diff",
        "first_third_exhibit_diff",

        "venue_hit_count",
        "venue_avg_payout",
        "venue_med_payout",
        "venue_std_payout",
        "venue_chaos_index",
        "venue_first_lane_match_rate",
        "venue_second_lane_match_rate",
        "venue_third_lane_match_rate",
        "venue_combo_lane_prior",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    cat_cols = [c for c in ["venue", "wind_dir", "weather"] if c in feature_cols]

    return feature_cols, cat_cols


def _prepare_xy(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    x = df[feature_cols].copy()

    for col in feature_cols:
        if col in cat_cols:
            x[col] = x[col].astype(str).fillna("")
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DATASET_PATH))
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--valid_ratio", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--neg_pos_ratio", type=int, default=12)
    parser.add_argument("--verbose_eval", type=int, default=100)
    parser.add_argument("--model_name", default="trifecta_binary_catboost_global_venue_auto")
    args = parser.parse_args()

    src = Path(args.src)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOAD DATASET")
    print("=" * 80)
    df = _load_dataset(src)

    print("rows total:", len(df))
    print("min date  :", str(df["date"].astype(str).min()) if "date" in df.columns else "-")
    print("max date  :", str(df["date"].astype(str).max()) if "date" in df.columns else "-")

    print("=" * 80)
    print("BUILD BASIC FEATURES")
    print("=" * 80)
    df = _build_basic_features(df)

    print("=" * 80)
    print("BUILD VENUE AUTO STATS")
    print("=" * 80)
    venue_stats = _build_venue_auto_stats(df)
    print("venue stats rows:", len(venue_stats))

    df = _attach_venue_auto_features(df, venue_stats)

    print("=" * 80)
    print("DOWNSAMPLE")
    print("=" * 80)
    df_trainable = _downsample_for_binary(
        df=df,
        neg_pos_ratio=args.neg_pos_ratio,
        random_state=args.random_state,
    )
    print("trainable rows:", len(df_trainable))
    print("positive rows :", int((pd.to_numeric(df_trainable["y"], errors="coerce").fillna(0).astype(int) == 1).sum()))
    print("negative rows :", int((pd.to_numeric(df_trainable["y"], errors="coerce").fillna(0).astype(int) == 0).sum()))

    print("=" * 80)
    print("SPLIT")
    print("=" * 80)
    train_df, valid_df = _split_by_race_key(
        df_trainable,
        valid_ratio=args.valid_ratio,
        random_state=args.random_state,
    )
    print("train rows:", len(train_df))
    print("valid rows:", len(valid_df))

    feature_cols, cat_cols = _build_feature_columns(df_trainable)
    print("feature count:", len(feature_cols))
    print("cat cols     :", cat_cols)

    x_train, y_train = _prepare_xy(train_df, feature_cols, cat_cols)
    x_valid, y_valid = _prepare_xy(valid_df, feature_cols, cat_cols)

    pos_train = int((y_train == 1).sum())
    neg_train = int((y_train == 0).sum())
    scale_pos_weight = float(neg_train / max(pos_train, 1))
    print("scale_pos_weight:", round(scale_pos_weight, 4))

    train_pool = Pool(
        x_train,
        y_train,
        cat_features=cat_cols,
    )
    valid_pool = Pool(
        x_valid,
        y_valid,
        cat_features=cat_cols,
    )

    print("=" * 80)
    print("TRAIN")
    print("=" * 80)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=args.random_state,
        verbose=args.verbose_eval,
        scale_pos_weight=scale_pos_weight,
        allow_writing_files=False,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
    )

    valid_pred = model.predict_proba(valid_pool)[:, 1]
    valid_logloss = float(log_loss(y_valid, valid_pred))
    try:
        valid_auc = float(roc_auc_score(y_valid, valid_pred))
    except Exception:
        valid_auc = 0.0

    model_path = MODEL_DIR / f"{args.model_name}.cbm"
    meta_path = MODEL_DIR / f"{args.model_name}_meta.json"
    venue_stats_path = MODEL_DIR / f"{args.model_name}_venue_stats.csv"

    model.save_model(str(model_path))
    venue_stats.to_csv(venue_stats_path, index=False, encoding="utf-8-sig")

    meta = {
        "model_type": "catboost_binary_global_venue_auto",
        "dataset_path": str(src),
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "venue_stats_path": str(venue_stats_path),
        "params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "valid_ratio": args.valid_ratio,
            "random_state": args.random_state,
            "neg_pos_ratio": args.neg_pos_ratio,
            "scale_pos_weight": scale_pos_weight,
        },
        "metrics": {
            "valid_logloss": valid_logloss,
            "valid_auc": valid_auc,
        },
        "rows": {
            "all_rows": int(len(df)),
            "trainable_rows": int(len(df_trainable)),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("saved model     :", model_path)
    print("saved meta      :", meta_path)
    print("saved venue stat:", venue_stats_path)
    print("valid logloss   :", round(valid_logloss, 6))
    print("valid auc       :", round(valid_auc, 6))


if __name__ == "__main__":
    main()
