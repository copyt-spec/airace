# scripts/analyze_render_model.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "render_model_predictions.csv"
OUT_DIR = PROJECT_ROOT / "data" / "analysis"
TOPK = 20


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def _normalize_prediction_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # analyze_render_model.py は pred_proba を前提にしているので
    # 実ファイル側の列名ゆれをここで吸収する
    candidate_cols = [
        "pred_proba",
        "score",
        "prob",
        "proba",
        "pred_score",
        "prediction",
    ]

    if "pred_proba" not in out.columns:
        for c in candidate_cols:
            if c in out.columns:
                out["pred_proba"] = out[c]
                break

    return out


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = [
        "race_key",
        "date",
        "venue",
        "race_no",
        "combo",
        "y_combo",
        "pred_proba",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def _to_numeric_safe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _build_ranked_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["race_key"] = out["race_key"].astype(str)
    out["combo"] = out["combo"].astype(str)
    out["y_combo"] = out["y_combo"].astype(str)
    out["pred_proba"] = pd.to_numeric(out["pred_proba"], errors="coerce").fillna(0.0)

    out = out.sort_values(
        ["race_key", "pred_proba"],
        ascending=[True, False]
    ).reset_index(drop=True)

    out["pred_rank"] = out.groupby("race_key").cumcount() + 1
    out["is_hit"] = (out["combo"] == out["y_combo"]).astype(int)

    return out


def _calc_hit_summary(df: pd.DataFrame) -> pd.DataFrame:
    hit_rows = df[df["is_hit"] == 1].copy()

    if hit_rows.empty:
        return pd.DataFrame(
            [{
                "num_races": 0,
                "top1_hit_rate": 0.0,
                "top3_hit_rate": 0.0,
                "top5_hit_rate": 0.0,
                "top10_hit_rate": 0.0,
                "avg_hit_rank": None,
                "median_hit_rank": None,
            }]
        )

    num_races = hit_rows["race_key"].nunique()

    summary = pd.DataFrame([{
        "num_races": int(num_races),
        "top1_hit_rate": float((hit_rows["pred_rank"] <= 1).mean()),
        "top3_hit_rate": float((hit_rows["pred_rank"] <= 3).mean()),
        "top5_hit_rate": float((hit_rows["pred_rank"] <= 5).mean()),
        "top10_hit_rate": float((hit_rows["pred_rank"] <= 10).mean()),
        "avg_hit_rank": float(hit_rows["pred_rank"].mean()),
        "median_hit_rank": float(hit_rows["pred_rank"].median()),
    }])

    return summary


def _calc_top1_bias(df: pd.DataFrame) -> pd.DataFrame:
    top1 = df[df["pred_rank"] == 1].copy()
    if top1.empty:
        return pd.DataFrame(columns=["combo", "count", "rate"])

    agg = (
        top1.groupby("combo", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    agg["rate"] = agg["count"] / agg["count"].sum()
    return agg


def _calc_hit_combo_stats(df: pd.DataFrame) -> pd.DataFrame:
    hit_rows = df[df["is_hit"] == 1].copy()
    if hit_rows.empty:
        return pd.DataFrame(columns=["combo", "count", "avg_rank", "top1_rate", "top3_rate"])

    agg = (
        hit_rows.groupby("combo", dropna=False)
        .agg(
            count=("combo", "size"),
            avg_rank=("pred_rank", "mean"),
        )
        .reset_index()
    )

    top1_rate = (
        hit_rows.assign(top1=(hit_rows["pred_rank"] <= 1).astype(int))
        .groupby("combo")["top1"]
        .mean()
        .reset_index(name="top1_rate")
    )

    top3_rate = (
        hit_rows.assign(top3=(hit_rows["pred_rank"] <= 3).astype(int))
        .groupby("combo")["top3"]
        .mean()
        .reset_index(name="top3_rate")
    )

    agg = agg.merge(top1_rate, on="combo", how="left")
    agg = agg.merge(top3_rate, on="combo", how="left")
    agg = agg.sort_values(["count", "avg_rank"], ascending=[False, True]).reset_index(drop=True)

    return agg


def _calc_worst_hits(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    hit_rows = df[df["is_hit"] == 1].copy()
    if hit_rows.empty:
        return pd.DataFrame(columns=df.columns.tolist())

    cols = ["race_key", "date", "venue", "race_no", "combo", "y_combo", "pred_proba", "pred_rank"]
    cols = [c for c in cols if c in hit_rows.columns]

    out = (
        hit_rows[cols]
        .sort_values(["pred_rank", "pred_proba"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def _calc_best_hits(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    hit_rows = df[df["is_hit"] == 1].copy()
    if hit_rows.empty:
        return pd.DataFrame(columns=df.columns.tolist())

    cols = ["race_key", "date", "venue", "race_no", "combo", "y_combo", "pred_proba", "pred_rank"]
    cols = [c for c in cols if c in hit_rows.columns]

    out = (
        hit_rows[cols]
        .sort_values(["pred_rank", "pred_proba"], ascending=[True, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def analyze() -> None:
    print("===== ANALYZE RENDER MODEL START =====")

    print("[1/6] loading prediction csv...")
    df = _safe_read_csv(PRED_PATH)
    print(f"loaded shape: {df.shape}")

    print("[2/6] normalizing columns...")
    df = _normalize_prediction_column(df)

    print("[3/6] checking required columns...")
    _ensure_required_columns(df)

    print("[4/6] ranking predictions...")
    df = _to_numeric_safe(df, ["race_no", "pred_proba"])
    ranked = _build_ranked_frame(df)

    print("[5/6] building reports...")
    summary = _calc_hit_summary(ranked)
    top1_bias = _calc_top1_bias(ranked)
    hit_combo_stats = _calc_hit_combo_stats(ranked)
    best_hits = _calc_best_hits(ranked, top_n=100)
    worst_hits = _calc_worst_hits(ranked, top_n=100)

    print("[6/6] saving outputs...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ranked_path = OUT_DIR / "render_model_ranked_predictions.csv"
    summary_path = OUT_DIR / "render_model_summary.csv"
    bias_path = OUT_DIR / "render_model_top1_bias.csv"
    combo_stats_path = OUT_DIR / "render_model_hit_combo_stats.csv"
    best_hits_path = OUT_DIR / "render_model_best_hits.csv"
    worst_hits_path = OUT_DIR / "render_model_worst_hits.csv"

    ranked.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    top1_bias.to_csv(bias_path, index=False, encoding="utf-8-sig")
    hit_combo_stats.to_csv(combo_stats_path, index=False, encoding="utf-8-sig")
    best_hits.to_csv(best_hits_path, index=False, encoding="utf-8-sig")
    worst_hits.to_csv(worst_hits_path, index=False, encoding="utf-8-sig")

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== TOP1 BIAS TOP 20 =====")
    if len(top1_bias) == 0:
        print("no data")
    else:
        print(top1_bias.head(TOPK).to_string(index=False))

    print("\n===== HIT COMBO STATS TOP 20 =====")
    if len(hit_combo_stats) == 0:
        print("no data")
    else:
        print(hit_combo_stats.head(TOPK).to_string(index=False))

    print("\nSaved files:")
    print(f"- {ranked_path}")
    print(f"- {summary_path}")
    print(f"- {bias_path}")
    print(f"- {combo_stats_path}")
    print(f"- {best_hits_path}")
    print(f"- {worst_hits_path}")

    print("===== ANALYZE RENDER MODEL END =====")


if __name__ == "__main__":
    analyze()
