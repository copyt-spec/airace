# scripts/compare_racer_stats_holdout_all_venues.py
# -*- coding: utf-8 -*-
"""
全24会場について、旧モデル(with_racer_no、predictions.csvに実際に残っているログ)と
新モデル(with_racer_stats、日付分割の真のholdout期間だけを対象に
generate_predictions_with_racer_stats_model.pyで再生成した予測)を、
「今のbuy_selector.select_best_bets」で同じ土俵に立たせて比較する。

前提として、以下があらかじめ実行済みであること:
  1) python scripts/train_binary_catboost_per_venue_with_racer_stats.py --all --skip_baseline_rerun
     (全24会場のwith_racer_statsモデルを日付分割で学習。本番では実施済み)
  2) python scripts/generate_predictions_with_racer_stats_model.py --all
     (predictions.csvに実オッズが5レース以上残っている会場だけ
      data/logs/predictions_with_racer_stats_{venue}.csv が生成される)

このスクリプト自体はcatboostを使わない(既存CSVをpandasで突き合わせるだけ)ため、
このサンドボックス環境でも実行できる。

実オッズが無い会場(=predictions_with_racer_stats_{venue}.csvが存在しない会場)は
「real_odds_available=False」としてスキップし、代わりにモデルmeta.jsonの
オフライン指標(logloss/auc)だけを参考値として載せる(=ROIの検証にはならない点に注意)。

出力: data/logs/model_comparison_all_venues_holdout.csv (全会場分に更新・上書き)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402
from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402
from scripts.backtest_buy_selector import run_backtest, summarize  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
RESULTS_CSV = LOG_DIR / "results.csv"
OUT_CSV = LOG_DIR / "model_comparison_all_venues_holdout.csv"

MIN_RACES = 5


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def _load_results() -> pd.DataFrame:
    res = pd.read_csv(RESULTS_CSV, low_memory=False)
    res["date"] = _normalize_date(res["date"])
    res["venue"] = res["venue"].astype(str).str.strip().map(normalize_venue_name)
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)
    return res


def _load_predictions() -> pd.DataFrame:
    pred = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    pred["date"] = _normalize_date(pred["date"])
    pred["venue"] = pred["venue"].astype(str).str.strip().map(normalize_venue_name)
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()
    return pred


def _read_meta_metrics(venue: str) -> Optional[Dict[str, Any]]:
    meta_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_with_racer_stats_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("metrics")
    except Exception:
        return None


def _score(df_with_results: pd.DataFrame) -> Dict[str, Any]:
    res = run_backtest(df_with_results, selector_fn=buy_selector.select_best_bets)
    return summarize(res)


def compare_venue(venue: str, all_predictions: pd.DataFrame, results: pd.DataFrame) -> Dict[str, Any]:
    venue = normalize_venue_name(venue)
    new_path = LOG_DIR / f"predictions_with_racer_stats_{venue}.csv"

    row: Dict[str, Any] = {"venue": venue, "real_odds_available": False}

    if not new_path.exists():
        metrics = _read_meta_metrics(venue)
        row["note"] = "実オッズデータ不足のためROI検証不可(predictions_with_racer_stats未生成)"
        if metrics:
            row["offline_valid_logloss"] = metrics.get("valid_logloss")
            row["offline_valid_auc"] = metrics.get("valid_auc")
        return row

    new_df = pd.read_csv(new_path, low_memory=False)
    if new_df.empty:
        row["note"] = "predictions_with_racer_statsが空"
        return row

    new_df["date"] = _normalize_date(new_df["date"])
    new_df["venue"] = venue
    new_df["race_no"] = pd.to_numeric(new_df["race_no"], errors="coerce").fillna(0).astype(int)
    new_df["combo"] = new_df["combo"].astype(str).str.strip()
    new_df["odds"] = pd.to_numeric(new_df.get("odds", 0.0), errors="coerce").fillna(0.0)

    holdout_races = new_df[["date", "race_no"]].drop_duplicates()
    if len(holdout_races) < MIN_RACES:
        row["note"] = f"holdoutレース数が{MIN_RACES}件未満({len(holdout_races)}件)"
        return row

    old_df = all_predictions[all_predictions["venue"] == venue].copy()
    old_df = old_df.merge(holdout_races, on=["date", "race_no"], how="inner")

    res_cols = results[["date", "venue", "race_no", "actual_combo", "payout"]]

    old_merged = old_df.merge(res_cols, on=["date", "venue", "race_no"], how="left")
    new_merged = new_df.merge(res_cols, on=["date", "venue", "race_no"], how="left")

    old_races = old_merged.drop_duplicates(["date", "race_no"]).shape[0]
    new_races = new_merged.drop_duplicates(["date", "race_no"]).shape[0]

    old_summary = _score(old_merged) if not old_merged.empty else {}
    new_summary = _score(new_merged) if not new_merged.empty else {}

    row["real_odds_available"] = True
    row["valid_start"] = str(holdout_races["date"].min())
    row["valid_end"] = str(holdout_races["date"].max())
    row["races_old"] = old_summary.get("races", 0)
    row["races_new"] = new_summary.get("races", 0)
    row["roi_old_pct"] = round(old_summary.get("roi", 0.0) * 100, 2)
    row["roi_new_pct"] = round(new_summary.get("roi", 0.0) * 100, 2)
    row["roi_delta_pct"] = round(row["roi_new_pct"] - row["roi_old_pct"], 2)
    row["profit_old"] = old_summary.get("profit", 0)
    row["profit_new"] = new_summary.get("profit", 0)
    row["hit_rate_old_pct"] = round(old_summary.get("hit_rate", 0.0) * 100, 2)
    row["hit_rate_new_pct"] = round(new_summary.get("hit_rate", 0.0) * 100, 2)

    return row


def main() -> None:
    print("loading predictions.csv / results.csv ...")
    all_predictions = _load_predictions()
    results = _load_results()

    rows = []
    for venue in VENUE_ORDER:
        print(f"--- {venue} ---")
        try:
            row = compare_venue(venue, all_predictions, results)
        except Exception as e:
            row = {"venue": venue, "real_odds_available": False, "note": f"ERROR: {e}"}
        rows.append(row)
        print(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\nsaved:", OUT_CSV)

    print("\n" + "=" * 100)
    print("real-odds validated venues (real ROI backtest):")
    validated = out_df[out_df["real_odds_available"] == True].copy()  # noqa: E712
    if not validated.empty:
        cols = ["venue", "valid_start", "valid_end", "races_new", "roi_old_pct", "roi_new_pct", "roi_delta_pct"]
        print(validated[cols].sort_values("roi_delta_pct", ascending=False).to_string(index=False))
    else:
        print("(none)")

    print("\nno real-odds data (offline metrics only, NOT a real ROI validation):")
    not_validated = out_df[out_df["real_odds_available"] == False]  # noqa: E712
    if not not_validated.empty:
        cols = [c for c in ["venue", "note", "offline_valid_logloss", "offline_valid_auc"] if c in not_validated.columns]
        print(not_validated[cols].to_string(index=False))


if __name__ == "__main__":
    main()
