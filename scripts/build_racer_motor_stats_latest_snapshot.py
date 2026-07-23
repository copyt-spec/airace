# scripts/build_racer_motor_stats_latest_snapshot.py
# -*- coding: utf-8 -*-
"""
2026-07-24追加: ライブ推論(engine/model_loader_catboost_binary.py)は
data/datasets/racer_point_in_time_stats.csv (208MB, 675,094行) と
motor_point_in_time_stats.csv (19MB, 669,710行) を丸ごと読み込んで、
選手/モーターごとに「一番新しい日付の行」だけを使っている
(_load_latest_racer_stats / _load_latest_motor_stats参照)。

つまりライブ推論には各選手・各(会場,モーター)につき最新1行だけあれば十分。
しかしこの2ファイルはdata/datasets/*.csvとして丸ごとgitignoreされており
(GitHubの1ファイル100MB上限もあり208MBは物理的にコミット不可能)、
Renderには本番モデルの学習・推論に必要なこのデータが一切届いていなかった
(ローカルとRenderで確率算出結果が食い違っていた一因)。

このスクリプトは、ローカルの巨大な履歴ファイルから「選手ごとの最新行」
「(会場,モーター)ごとの最新行」だけを抜き出した軽量スナップショットを作る:
  - data/datasets/racer_point_in_time_stats_latest.csv (約1,760行、500KB程度)
  - data/datasets/motor_point_in_time_stats_latest.csv (約1,560行、50KB程度)

この2ファイルは(.gitignoreに例外追加済みで)通常通りgitコミットでき、
Renderにも配布できる。engine/model_loader_catboost_binary.py側は、
フルの履歴ファイルがあれば従来通りそこから都度tail(1)を計算し(ローカルでは
常に最新)、無ければ(Render等)このスナップショットファイルをそのまま使う
ようフォールバック実装済み。

運用: racer_point_in_time_stats.csv / motor_point_in_time_stats.csv を
再生成した後、Renderにも新しい選手・モーター成績を反映させたい場合は
このスクリプトを再実行してからgit commit & pushする。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "datasets"

RACER_STATS_CSV = DATASET_DIR / "racer_point_in_time_stats.csv"
MOTOR_STATS_CSV = DATASET_DIR / "motor_point_in_time_stats.csv"

RACER_STATS_LATEST_CSV = DATASET_DIR / "racer_point_in_time_stats_latest.csv"
MOTOR_STATS_LATEST_CSV = DATASET_DIR / "motor_point_in_time_stats_latest.csv"


def build_racer_latest() -> None:
    if not RACER_STATS_CSV.exists():
        print(f"[SKIP] {RACER_STATS_CSV} が見つかりません")
        return

    print("loading:", RACER_STATS_CSV)
    df = pd.read_csv(RACER_STATS_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["racer_no"] = pd.to_numeric(df["racer_no"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values(["racer_no", "date"])
    latest = df.groupby("racer_no", as_index=False).tail(1)
    latest.to_csv(RACER_STATS_LATEST_CSV, index=False, encoding="utf-8-sig")

    print(f"saved: {RACER_STATS_LATEST_CSV} ({len(latest)} racers, "
          f"{RACER_STATS_LATEST_CSV.stat().st_size / 1024:.1f} KB)")


def build_motor_latest() -> None:
    if not MOTOR_STATS_CSV.exists():
        print(f"[SKIP] {MOTOR_STATS_CSV} が見つかりません")
        return

    print("loading:", MOTOR_STATS_CSV)
    df = pd.read_csv(MOTOR_STATS_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)

    df = df.sort_values(["venue", "motor", "date"])
    latest = df.groupby(["venue", "motor"], as_index=False).tail(1)
    latest.to_csv(MOTOR_STATS_LATEST_CSV, index=False, encoding="utf-8-sig")

    print(f"saved: {MOTOR_STATS_LATEST_CSV} ({len(latest)} venue x motor, "
          f"{MOTOR_STATS_LATEST_CSV.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    build_racer_latest()
    build_motor_latest()


if __name__ == "__main__":
    main()
