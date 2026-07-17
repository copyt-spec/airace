from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"

PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
RESULTS_CSV = LOG_DIR / "results.csv"
MERGED_CSV = LOG_DIR / "prediction_results_merged.csv"


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _normalize_date_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def _normalize_venue_name(v: str) -> str:
    s = str(v or "").strip()
    venue_order = [
        "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
        "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
        "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
    ]
    for venue in venue_order:
        if venue in s:
            return venue
    return s


def _dedupe_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    同一 (date, venue, race_no) が重複していたら、
    payout が大きいものを優先
    """
    if df.empty:
        return df

    work = df.copy()
    work["payout"] = pd.to_numeric(work["payout"], errors="coerce").fillna(0.0)

    work = work.sort_values(
        by=["date", "venue", "race_no", "payout"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    work = work.drop_duplicates(subset=["date", "venue", "race_no"], keep="first")
    work = work.reset_index(drop=True)
    return work


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Missing predictions csv: {PREDICTIONS_CSV}")
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing results csv: {RESULTS_CSV}")

    print("loading predictions:", PREDICTIONS_CSV)
    pred = pd.read_csv(PREDICTIONS_CSV, low_memory=False)

    print("loading results    :", RESULTS_CSV)
    res = pd.read_csv(RESULTS_CSV, low_memory=False)

    if pred.empty:
        raise RuntimeError("predictions.csv is empty")
    if res.empty:
        raise RuntimeError("results.csv is empty")

    pred["date"] = _normalize_date_series(pred["date"])
    res["date"] = _normalize_date_series(res["date"])

    pred["venue"] = pred["venue"].astype(str).map(_normalize_venue_name)
    res["venue"] = res["venue"].astype(str).map(_normalize_venue_name)

    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)

    pred["combo"] = pred["combo"].astype(str).str.strip()
    res["actual_combo"] = res["actual_combo"].astype(str).str.strip()
    res["payout"] = pd.to_numeric(res["payout"], errors="coerce").fillna(0.0)

    res = _dedupe_results(res)

    merged = pred.merge(
        res[["date", "venue", "race_no", "actual_combo", "payout", "source"]],
        on=["date", "venue", "race_no"],
        how="left",
    )

    merged["is_selected"] = pd.to_numeric(merged.get("is_selected", 0), errors="coerce").fillna(0).astype(int)

    # そのレースにまだ結果(K-file)が届いていない場合、resultとのマージは
    # actual_combo/source がNaNになる。これを「未確定」として区別しないと
    # 「結果待ちのレース」が「全部ハズレ」として集計されてしまい、
    # ROIが実態より不当に低く出てしまう(過去に発覚した問題の再発防止)。
    merged["has_result"] = merged["source"].notna()

    merged["is_hit"] = (
        (merged["combo"] == merged["actual_combo"])
        & (merged["is_selected"] == 1)
        & merged["has_result"]
    ).astype(int)

    merged["bet_cost_yen"] = merged["is_selected"].astype(float) * 100.0 * merged["has_result"].astype(float)
    merged["return_yen"] = (
        ((merged["is_selected"] == 1) & (merged["is_hit"] == 1)).astype(int)
        * merged["payout"].fillna(0.0)
    ).astype(float)
    merged["profit_yen"] = merged["return_yen"] - merged["bet_cost_yen"]

    if "prob" in merged.columns:
        merged["prob"] = pd.to_numeric(merged["prob"], errors="coerce").fillna(0.0)
    if "odds" in merged.columns:
        merged["odds"] = pd.to_numeric(merged["odds"], errors="coerce").fillna(0.0)
    if "expected_return_yen" in merged.columns:
        merged["expected_return_yen"] = pd.to_numeric(merged["expected_return_yen"], errors="coerce").fillna(0.0)

    merged = merged.sort_values(
        by=["date", "venue", "race_no", "rank_prob", "combo"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("saved:", MERGED_CSV)
    print("rows :", len(merged))
    print("pred races:", pred[["date", "venue", "race_no"]].drop_duplicates().shape[0])
    print("result races:", res[["date", "venue", "race_no"]].drop_duplicates().shape[0])
    print("merged races:", merged[["date", "venue", "race_no"]].drop_duplicates().shape[0])

    hit_selected = merged[(merged["is_selected"] == 1) & (merged["is_hit"] == 1)]
    print("selected hits:", len(hit_selected))


if __name__ == "__main__":
    main()
