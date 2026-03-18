from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render.joblib"
META_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_meta.json"

OUT_DIR = PROJECT_ROOT / "data" / "analysis"
OUT_CSV = OUT_DIR / "feature_importance_full.csv"
OUT_TOP_CSV = OUT_DIR / "feature_importance_top100.csv"
OUT_GROUP_CSV = OUT_DIR / "feature_importance_group_summary.csv"


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _detect_group(feature_name: str) -> str:
    name = feature_name.lower()

    if "lane" in name:
        return "lane"
    if "course" in name:
        return "course"
    if "win_rate" in name:
        return "win_rate"
    if "place_rate" in name:
        return "place_rate"
    if "exhibit" in name:
        return "exhibit"
    if name.endswith("_st") or "_st_" in name or name == "st":
        return "st"
    if "grade" in name:
        return "grade"
    if "weather" in name:
        return "weather"
    if "wind" in name:
        return "wind"
    if "wave" in name:
        return "wave"
    if "motor" in name:
        return "motor"
    if "boat" in name:
        return "boat"
    if "weight" in name:
        return "weight"
    if "age" in name:
        return "age"
    if "racer_no" in name:
        return "racer_no"
    if "venue" in name:
        return "venue"

    return "other"


def _print_top(df: pd.DataFrame, n: int = 50) -> None:
    print(f"\n===== TOP {n} FEATURES =====")
    print(df.head(n).to_string(index=False))


def _print_group_summary(group_df: pd.DataFrame) -> None:
    print("\n===== GROUP SUMMARY =====")
    print(group_df.to_string(index=False))


def main():
    print("===== ANALYZE FEATURE IMPORTANCE START =====")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing meta: {META_PATH}")

    print("[1/5] loading model...")
    model = joblib.load(MODEL_PATH)

    print("[2/5] loading meta...")
    meta = _load_json(META_PATH)
    feature_cols: List[str] = meta.get("feature_cols", [])
    if not feature_cols:
        raise RuntimeError("feature_cols not found in meta json.")

    if not hasattr(model, "feature_importances_"):
        raise RuntimeError("Loaded model does not have feature_importances_.")

    importances = list(model.feature_importances_)

    if len(importances) != len(feature_cols):
        raise RuntimeError(
            f"feature count mismatch: importances={len(importances)} feature_cols={len(feature_cols)}"
        )

    print("[3/5] building dataframe...")
    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
        }
    )
    df["group"] = df["feature"].map(_detect_group)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    print("[4/5] building group summary...")
    group_df = (
        df.groupby("group", as_index=False)
        .agg(
            total_importance=("importance", "sum"),
            avg_importance=("importance", "mean"),
            feature_count=("feature", "count"),
        )
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    df.head(100).to_csv(OUT_TOP_CSV, index=False, encoding="utf-8-sig")
    group_df.to_csv(OUT_GROUP_CSV, index=False, encoding="utf-8-sig")

    print("[5/5] output")
    print(f"saved: {OUT_CSV}")
    print(f"saved: {OUT_TOP_CSV}")
    print(f"saved: {OUT_GROUP_CSV}")

    _print_top(df, 50)
    _print_group_summary(group_df)

    lane_total = float(group_df.loc[group_df["group"] == "lane", "total_importance"].sum())
    course_total = float(group_df.loc[group_df["group"] == "course", "total_importance"].sum())
    win_total = float(group_df.loc[group_df["group"] == "win_rate", "total_importance"].sum())
    place_total = float(group_df.loc[group_df["group"] == "place_rate", "total_importance"].sum())
    exhibit_total = float(group_df.loc[group_df["group"] == "exhibit", "total_importance"].sum())
    st_total = float(group_df.loc[group_df["group"] == "st", "total_importance"].sum())
    grade_total = float(group_df.loc[group_df["group"] == "grade", "total_importance"].sum())

    print("\n===== KEY TOTALS =====")
    print(f"lane      : {lane_total:.6f}")
    print(f"course    : {course_total:.6f}")
    print(f"win_rate  : {win_total:.6f}")
    print(f"place_rate: {place_total:.6f}")
    print(f"exhibit   : {exhibit_total:.6f}")
    print(f"st        : {st_total:.6f}")
    print(f"grade     : {grade_total:.6f}")

    print("===== ANALYZE FEATURE IMPORTANCE END =====")


if __name__ == "__main__":
    main()
