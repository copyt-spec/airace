# scripts/compare_model_suffixes.py
# -*- coding: utf-8 -*-
"""
generate_predictions_with_racer_stats_model.py --model_suffix X で生成した
2種類の predictions_with_racer_stats_{venue}{suffix}.csv (同じholdout期間、
同じ実オッズに対する予測確率だけが違う) を、今の buy_selector.select_best_bets
で同じ土俵に立たせて比較する。

想定用途: 選手/モーターの直近フォーム特徴量を追加した検証用モデル
(--model_suffix _formv2 で学習・生成)が、本番モデル(suffix無し)より
実際のROIが良いかどうかを確認する(scripts/train_binary_catboost_per_venue_with_racer_stats.py
と scripts/generate_predictions_with_racer_stats_model.py の続き)。

使い方:
  python scripts/compare_model_suffixes.py --new_suffix _formv2
  (baseline_suffixは省略時""=本番モデル)

出力: data/logs/model_comparison{baseline_suffix}_vs{new_suffix}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402
from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402
from scripts.backtest_buy_selector import run_backtest, summarize  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
RESULTS_CSV = LOG_DIR / "results.csv"

MIN_RACES = 5


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def _load_results() -> pd.DataFrame:
    res = pd.read_csv(RESULTS_CSV, low_memory=False)
    res["date"] = _normalize_date(res["date"])
    res["venue"] = res["venue"].astype(str).str.strip().map(normalize_venue_name)
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)
    return res


def _load_pred_file(path: Path, venue: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = _normalize_date(df["date"])
    df["venue"] = venue
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)
    df["combo"] = df["combo"].astype(str).str.strip()
    df["odds"] = pd.to_numeric(df.get("odds", 0.0), errors="coerce").fillna(0.0)
    return df


def _score(df_with_results: pd.DataFrame) -> Dict[str, Any]:
    res = run_backtest(df_with_results, selector_fn=buy_selector.select_best_bets)
    return summarize(res)


def compare_venue(
    venue: str, baseline_suffix: str, new_suffix: str, results: pd.DataFrame
) -> Dict[str, Any]:
    venue = normalize_venue_name(venue)
    baseline_path = LOG_DIR / f"predictions_with_racer_stats_{venue}{baseline_suffix}.csv"
    new_path = LOG_DIR / f"predictions_with_racer_stats_{venue}{new_suffix}.csv"

    row: Dict[str, Any] = {"venue": venue, "comparable": False}

    if not baseline_path.exists():
        row["note"] = f"baseline未生成: {baseline_path.name}"
        return row
    if not new_path.exists():
        row["note"] = f"new未生成: {new_path.name}"
        return row

    baseline_df = _load_pred_file(baseline_path, venue)
    new_df = _load_pred_file(new_path, venue)
    if baseline_df.empty or new_df.empty:
        row["note"] = "predictionsファイルが空"
        return row

    # 両方に共通する(date, race_no)だけを対象にし、完全に同じレース集合で比較する
    common_races = pd.merge(
        baseline_df[["date", "race_no"]].drop_duplicates(),
        new_df[["date", "race_no"]].drop_duplicates(),
        on=["date", "race_no"], how="inner",
    )
    if len(common_races) < MIN_RACES:
        row["note"] = f"共通holdoutレースが{MIN_RACES}件未満({len(common_races)}件)"
        return row

    res_cols = results[["date", "venue", "race_no", "actual_combo", "payout"]]

    baseline_merged = baseline_df.merge(common_races, on=["date", "race_no"], how="inner")
    baseline_merged = baseline_merged.merge(res_cols, on=["date", "venue", "race_no"], how="left")

    new_merged = new_df.merge(common_races, on=["date", "race_no"], how="inner")
    new_merged = new_merged.merge(res_cols, on=["date", "venue", "race_no"], how="left")

    baseline_summary = _score(baseline_merged) if not baseline_merged.empty else {}
    new_summary = _score(new_merged) if not new_merged.empty else {}

    row["comparable"] = True
    row["valid_start"] = str(common_races["date"].min())
    row["valid_end"] = str(common_races["date"].max())
    row["races"] = new_summary.get("races", 0)
    row["invested_old"] = baseline_summary.get("invested", 0)
    row["invested_new"] = new_summary.get("invested", 0)
    row["returned_old"] = baseline_summary.get("returned", 0)
    row["returned_new"] = new_summary.get("returned", 0)
    row["roi_old_pct"] = round(baseline_summary.get("roi", 0.0) * 100, 2)
    row["roi_new_pct"] = round(new_summary.get("roi", 0.0) * 100, 2)
    row["roi_delta_pct"] = round(row["roi_new_pct"] - row["roi_old_pct"], 2)
    row["profit_old"] = baseline_summary.get("profit", 0)
    row["profit_new"] = new_summary.get("profit", 0)
    row["hit_rate_old_pct"] = round(baseline_summary.get("hit_rate", 0.0) * 100, 2)
    row["hit_rate_new_pct"] = round(new_summary.get("hit_rate", 0.0) * 100, 2)

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_suffix", default="", help="比較元(通常は本番モデル=空文字)")
    parser.add_argument("--new_suffix", default="_formv2", help="比較先(検証用モデルのsuffix)")
    args = parser.parse_args()

    print("loading results.csv ...")
    results = _load_results()

    rows = []
    for venue in VENUE_ORDER:
        try:
            row = compare_venue(venue, args.baseline_suffix, args.new_suffix, results)
        except Exception as e:
            row = {"venue": venue, "comparable": False, "note": f"ERROR: {e}"}
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_name = f"model_comparison{args.baseline_suffix or '_base'}_vs{args.new_suffix}.csv"
    out_path = LOG_DIR / out_name
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)

    comparable = out_df[out_df["comparable"] == True].copy()  # noqa: E712
    print("\n" + "=" * 100)
    print(f"会場別比較(baseline='{args.baseline_suffix or '(本番)'}' vs new='{args.new_suffix}'):")
    if not comparable.empty:
        cols = ["venue", "valid_start", "valid_end", "races", "roi_old_pct", "roi_new_pct", "roi_delta_pct", "profit_old", "profit_new"]
        print(comparable[cols].sort_values("roi_delta_pct", ascending=False).to_string(index=False))
    else:
        print("(比較可能な会場なし)")

    not_comparable = out_df[out_df["comparable"] == False]  # noqa: E712
    if not not_comparable.empty:
        print("\n比較不可の会場:")
        print(not_comparable[["venue", "note"]].to_string(index=False))

    if not comparable.empty:
        total_invested_old = comparable["invested_old"].sum()
        total_returned_old = comparable["returned_old"].sum()
        total_invested_new = comparable["invested_new"].sum()
        total_returned_new = comparable["returned_new"].sum()
        roi_old = total_returned_old / total_invested_old * 100 if total_invested_old > 0 else 0.0
        roi_new = total_returned_new / total_invested_new * 100 if total_invested_new > 0 else 0.0
        print("\n" + "=" * 100)
        print(f"全会場合算(比較可能な{len(comparable)}会場, races={int(comparable['races'].sum())}):")
        print(f"  baseline ROI : {roi_old:.2f}%  (invested={total_invested_old:.0f}, returned={total_returned_old:.0f})")
        print(f"  new(new_suffix) ROI : {roi_new:.2f}%  (invested={total_invested_new:.0f}, returned={total_returned_new:.0f})")
        print(f"  delta        : {roi_new - roi_old:+.2f}pt")


if __name__ == "__main__":
    main()
