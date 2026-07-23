# scripts/build_combined_predictions_for_sim.py
# -*- coding: utf-8 -*-
"""
/simのグラフを「新モデル(展開特徴量追加後、2026-07-22デプロイ)」の結果だけで
継続的に最新化するための合成データを作る。

背景: data/logs/predictions.csv は長期間・複数モデルバージョンが混在した
「本物の履歴ログ」で、日々の運用(アプリ閲覧やdaily-prediction-crawl)で
これからも増え続ける。一方 scripts/build_new_model_predictions_snapshot.py
が2026-07-22時点で作った data/logs/predictions_new_model_snapshot.csv は、
新モデル自身の学習時test_sizeに基づく「真のholdout期間」(2025-09-25〜
2026-07-14)の実オッズ付き予測を、一度だけ再現した静的なスナップショット。

この2つを日付で単純に結合する:
  - CUTOFF_DATE (デフォルト20260721。新モデル投入日は20260722だが、
    daily-prediction-crawlによる0721分のバックフィルがlogged_at的に投入後
    [2026-07-23 23:31以降]に新モデルで行われたことを確認済みのため前倒し)
    より前は静的スナップショット(新モデルの検証用holdout結果)を使う
  - CUTOFF_DATE 以降は data/logs/predictions.csv の実ログを使う
    (この日以降にログされた行は、新モデルデプロイ後の本物の実績なので、
    そのまま使ってよい)

これにより、/simは「新モデルの真のholdout実績」を土台にしつつ、日々の
実運用が増えるほど本物のデータで上書き・拡張されていく。
scripts/update_results_daily.py の日次パイプラインから呼ばれる想定
(merge_prediction_results.py --predictions <このスクリプトの出力> という形で使う)。

出力: data/logs/predictions_combined_for_sim.csv
  (predictions.csvと同じ列構成)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"

SNAPSHOT_CSV = LOG_DIR / "predictions_new_model_snapshot.csv"
LIVE_PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
OUT_CSV = LOG_DIR / "predictions_combined_for_sim.csv"

# 新モデル(展開特徴量追加)を本番投入した日は20260722だが、daily-prediction-crawl
# による0721分のバックフィルは0723 23:31以降(=投入後)に新モデルで実行された
# ことをlogged_atで確認済みのため、2026-07-24にカットオフを0721まで前倒しした。
# この日以降のpredictions.csvの行は新モデルによる本物のログなのでそのまま使う。
CUTOFF_DATE = "20260721"


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-date", default=CUTOFF_DATE, help="YYYYMMDD")
    parser.add_argument("--snapshot", default=str(SNAPSHOT_CSV))
    parser.add_argument("--live", default=str(LIVE_PREDICTIONS_CSV))
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    cutoff = str(args.cutoff_date)
    snapshot_path = Path(args.snapshot)
    live_path = Path(args.live)
    out_path = Path(args.out)

    frames = []

    if snapshot_path.exists():
        snap = pd.read_csv(snapshot_path, low_memory=False)
        snap["date"] = _normalize_date(snap["date"])
        snap = snap[snap["date"] < cutoff].copy()
        print(f"snapshot rows (date < {cutoff}): {len(snap)}")
        frames.append(snap)
    else:
        print(f"[WARN] snapshot not found: {snapshot_path}")

    if live_path.exists():
        live = pd.read_csv(live_path, low_memory=False)
        live["date"] = _normalize_date(live["date"])
        live = live[live["date"] >= cutoff].copy()
        print(f"live rows (date >= {cutoff}): {len(live)}")
        frames.append(live)
    else:
        print(f"[WARN] live predictions not found: {live_path}")

    if not frames:
        raise RuntimeError("合成対象データが0件です(snapshot/liveどちらも見つかりません)")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["date", "venue", "race_no", "rank_prob"]).reset_index(drop=True)

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("saved:", out_path)
    print("rows :", len(combined))
    print("races:", combined[["date", "venue", "race_no"]].drop_duplicates().shape[0])
    print("date range:", str(combined["date"].min()), "〜", str(combined["date"].max()))


if __name__ == "__main__":
    main()
