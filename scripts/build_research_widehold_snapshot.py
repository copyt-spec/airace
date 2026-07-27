# scripts/build_research_widehold_snapshot.py
# -*- coding: utf-8 -*-
"""
A~Dランク精度分析(analyze_rank_signal_improvement.py等)のため、過去データを
安全に増やす目的で作った「分析専用・広holdout」モデル群の予測結果を1本の
CSVにまとめるビルダー。

背景: 本番モデル(trifecta_binary_catboost_{venue}_with_racer_stats.cbm)は
test_size=0.20の日付分割で学習されており、真のholdout(学習に一度も使って
いない期間)は各会場ともに直近20%の日付だけ(戸田だと2025-11-05〜)しかない。
これより前のデータは学習に使われているため、そのままバックテストに使うと
「学習データを採点してしまいROIが実力より大幅に良く出る」事故になる
(2026-07-19に実際に発生: 桐生217%・戸田229%等)。

過去データをもっと増やして分析したい場合、本番モデルはそのままに、
test_sizeを大きくした「分析専用モデル」を別名(--model_suffix)で
学習し直すことで、安全にholdout期間を過去へ広げられる。

前提(すべてローカル実行、catboost必須。このリポジトリではsandboxに
catboostが無いため実行不可):
  1. python scripts/train_binary_catboost_per_venue_with_racer_stats.py \
       --all --test_size 0.4 --model_suffix _widehold --skip_baseline_rerun
  2. python scripts/generate_predictions_with_racer_stats_model.py \
       --all --model_suffix _widehold --test_size 0.4
     (--test_sizeは学習時と必ず同じ値にすること。ずれると
      _compute_holdout_dates が誤ったholdout日付を計算し、リークする)
  3. python scripts/build_research_widehold_snapshot.py --suffix _widehold

出力: data/logs/predictions_research_widehold_combined.csv
  (date, venue, race_no, combo, prob, odds の列。predictions_combined_for_sim.csv
  と違い is_selected 等は含まない生の予測データ。分析側で
  engine.buy_selector.select_best_bets を適用して評価する)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix", default="_widehold",
        help="generate_predictions_with_racer_stats_model.py --model_suffix と揃える",
    )
    parser.add_argument(
        "--out", default=str(LOG_DIR / "predictions_research_widehold_combined.csv"),
    )
    args = parser.parse_args()

    suffix = args.suffix
    out_path = Path(args.out)

    frames = []
    missing = []
    for path in sorted(LOG_DIR.glob(f"predictions_with_racer_stats_*{suffix}.csv")):
        venue = path.stem.replace("predictions_with_racer_stats_", "").replace(suffix, "")
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            missing.append(venue)
            continue
        df["venue"] = venue
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"'{suffix}' サフィックス付きの predictions_with_racer_stats_*.csv が"
            "見つかりません。先に train/generate スクリプトを実行してください。"
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["date"] = combined["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("saved:", out_path)
    print("rows :", len(combined))
    print("races:", combined[["date", "venue", "race_no"]].drop_duplicates().shape[0])
    print("venues covered:", combined["venue"].nunique(), "/ 24")
    if missing:
        print("[WARN] 空/未生成だった会場:", missing)
    print("date range:", combined["date"].min(), "〜", combined["date"].max())


if __name__ == "__main__":
    main()
