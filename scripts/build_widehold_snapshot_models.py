from __future__ import annotations

"""
「直近一年間を最新の予想モデルで実施した結果にsimを直したい」(2026-07-30)
に応えるための、retrospective full-history retraining driver。

背景: 通常のtest_size=0.20では真のholdout期間が数ヶ月しかなく、/simの
「直近1年」表示を新モデル基準で埋めるには短すぎる。より大きなtest_sizeで
学習しholdoutを広げる「widehold」アプローチ(scripts/build_research_widehold_snapshot.py
の手法を流用)を、各会場の「現在の本番モデルが実際に使っている特徴量構成」
(scripts/sweep_hyperparameters.py の _load_baseline_feature_flags を再利用)+
「現在のdepth選択」(VENUE_DEPTH_OVERRIDES / --use_tuned_depth)をそのまま
引き継いだ形で全24会場に対して行う。

model_suffix="_widehold" で保存するため、本番の"_with_racer_stats.cbm"は
一切上書きしない。

重要: 各会場のholdout期間はここでは決め打ちしない。generate_for_venue側が
学習時と全く同じ_split_by_dateロジックで自動的にholdout日付を復元するため、
このスクリプトではvalid_datesを明示的に渡さない(sweep_hyperparameters.pyとの
違い)。test_sizeさえ学習・生成で揃えれば、渡さなくても正しいholdoutになる。

前提: catboost/sklearnがこのサンドボックスに無いため実行できない。
catboostが使えるユーザーのローカル環境で実行すること。

使い方:
  python -m scripts.build_widehold_snapshot_models --all --test_size 0.32 --use_tuned_depth
  python -m scripts.build_widehold_snapshot_models --venue 戸田 --test_size 0.32 --use_tuned_depth

出力:
  - モデル: data/models/trifecta_binary_catboost_{venue}_with_racer_stats_widehold.cbm (+meta.json)
  - holdout予測(実オッズ付き): data/logs/predictions_with_racer_stats_{venue}_widehold.csv
    このあと scripts/build_new_model_predictions_snapshot.py --pred-suffix _widehold で
    /sim用の predictions_new_model_snapshot.csv に合成する。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402
from scripts.train_binary_catboost_per_venue_with_racer_stats import (  # noqa: E402
    DATASET_PATH,
    VENUE_DEPTH_OVERRIDES,
    _load_racer_stats,
    _load_motor_stats,
    _train_one_venue,
)
from scripts.sweep_hyperparameters import _load_baseline_feature_flags  # noqa: E402
from scripts.generate_predictions_with_racer_stats_model import generate_for_venue  # noqa: E402

MODEL_SUFFIX = "_widehold"


def build_one_venue(
    venue: str,
    test_size: float,
    random_state: int,
    iterations: int,
    learning_rate: float,
    use_tuned_depth: bool,
    default_depth: int,
    racer_stats: pd.DataFrame,
    motor_stats: pd.DataFrame,
) -> None:
    venue = normalize_venue_name(venue)
    print(f"\n{'#' * 80}\n# widehold venue={venue}\n{'#' * 80}")

    feature_flags = _load_baseline_feature_flags(venue)
    depth = default_depth
    if use_tuned_depth:
        depth = VENUE_DEPTH_OVERRIDES.get(venue, default_depth)
    print(f"depth={depth} (use_tuned_depth={use_tuned_depth}) feature_flags={feature_flags}")

    _train_one_venue(
        src=DATASET_PATH, venue=venue, racer_stats=racer_stats,
        test_size=test_size, random_state=random_state,
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        verbose_eval=0,
        skip_baseline_rerun=True,
        motor_stats=motor_stats,
        model_suffix=MODEL_SUFFIX,
        **feature_flags,
    )

    print(f"\n=== 予測生成(真のholdout全体、実オッズ付きのみ): {venue} ===")
    generate_for_venue(
        venue, temperature=0.90, min_races=5,
        model_suffix=MODEL_SUFFIX, out_suffix=MODEL_SUFFIX,
        test_size=test_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="", help="会場名(例: 戸田)")
    parser.add_argument("--all", action="store_true", help="全会場")
    parser.add_argument(
        "--test_size", type=float, required=True,
        help="widehold期間を広げるためのtest_size(例: 0.32で全期間の約32%%=約1年分)",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8, help="--use_tuned_depth対象外の会場に使うdepth")
    parser.add_argument(
        "--use_tuned_depth", action="store_true",
        help="2026-07-30に選択導入したdepth=6の13会場(VENUE_DEPTH_OVERRIDES)にはdepth=6を、"
             "それ以外は--depthを使う(=現在の本番構成を再現)。",
    )
    args = parser.parse_args()

    racer_stats = _load_racer_stats()
    print("racer_stats rows:", len(racer_stats))
    motor_stats = _load_motor_stats()
    print("motor_stats rows:", len(motor_stats))

    if args.all:
        venues = list(VENUE_ORDER)
    elif args.venue:
        venues = [args.venue]
    else:
        raise SystemExit("--venue か --all を指定してください")

    for v in venues:
        try:
            build_one_venue(
                v, args.test_size, args.random_state, args.iterations,
                args.learning_rate, args.use_tuned_depth, args.depth,
                racer_stats, motor_stats,
            )
        except Exception as e:
            print(f"[ERROR] venue={v}: {e}")


if __name__ == "__main__":
    main()
