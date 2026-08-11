from __future__ import annotations

"""
CatBoostのハイパーパラメータ(depth / iterations / learning_rate)を
複数組み合わせて学習し、真のholdout(学習に一切使っていない日付)での
hit_rate/ROIを比較するスイープスクリプト。設計は
scripts/sweep_racer_stats_weight.py(選手勝率の特徴量重み探索)と同じ。

背景: これまでの特徴量追加(展開特徴量、今節成績など)はいずれも「何を
入力するか」の改善だったが、モデル自体のハイパーパラメータ(木の深さ・
本数・学習率)は現在全会場・全モデル共通のデフォルト(depth=8,
iterations=600, learning_rate=0.05)のまま一度も体系的に探索していない。

CatBoostのハイパーパラメータに解析的な最適値(黄金比)は無く、深さを
上げすぎると過学習(学習データには強いが未知データに弱くなる)、
学習率を上げすぎると収束が粗くなる、といったトレードオフがあるため、
実際に複数組み合わせを学習してholdoutのROIで比較するしかない。

前提: このサンドボックスにはcatboost/sklearnが無いため実行できない。
catboostが使えるユーザーのローカル環境で実行すること。

使い方(デフォルトの候補セットで戸田を試す):
  python -m scripts.sweep_hyperparameters --venue 戸田

候補セットを指定したい場合(depth:iterations:learning_rate のカンマ区切り):
  python -m scripts.sweep_hyperparameters --venue 戸田 \
      --configs "8:600:0.05,6:600:0.05,10:600:0.05,8:1000:0.03,8:400:0.08"

全会場で回したい場合(時間がかかるので候補セットを絞ることを推奨):
  python -m scripts.sweep_hyperparameters --all --configs "8:600:0.05,6:800:0.04"

出力:
  - 各候補の学習済みモデル: data/models/trifecta_binary_catboost_{venue}_with_racer_stats_hp_d{depth}_i{iterations}_lr{lr}.cbm
    (現行デフォルトと完全一致する候補は再学習せず、既存の本番モデルをbaselineとして使う)
  - 眞のholdout日付に絞った予測CSV: data/logs/predictions_with_racer_stats_{venue}_sweep_hp_*_holdout.csv
  - 比較サマリー: data/logs/hyperparameter_sweep_{venue}.csv (候補ごとのhit_rate/ROI/avg_points/valid_logloss/valid_auc)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402
from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402
from scripts.train_binary_catboost_per_venue_with_racer_stats import (  # noqa: E402
    DATASET_PATH,
    _load_dataset_for_venue,
    _load_racer_stats,
    _load_motor_stats,
    _add_racer_stats_block,
    _split_by_date,
    _train_one_venue,
)
from scripts.generate_predictions_with_racer_stats_model import generate_for_venue  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
RESULTS_CSV = LOG_DIR / "results.csv"

# 現行の本番デフォルト(train_binary_catboost_per_venue_with_racer_stats.pyのargparseと一致)
PROD_DEPTH = 8
PROD_ITERATIONS = 600
PROD_LEARNING_RATE = 0.05

DEFAULT_CONFIGS = [
    (PROD_DEPTH, PROD_ITERATIONS, PROD_LEARNING_RATE),  # baseline(現行本番)
    (6, PROD_ITERATIONS, PROD_LEARNING_RATE),   # 浅め(過学習抑制)
    (10, PROD_ITERATIONS, PROD_LEARNING_RATE),  # 深め
    (PROD_DEPTH, 1000, 0.03),                   # 木を増やして学習率を落とす
    (PROD_DEPTH, 400, 0.08),                    # 木を減らして学習率を上げる
    (6, 1000, 0.03),                            # 浅め+多め+低学習率の組み合わせ
]


def _lr_tag(lr: float) -> str:
    return f"{lr:g}".replace(".", "p")


def _config_suffix(depth: int, iterations: int, lr: float) -> str:
    return f"_hp_d{depth}_i{iterations}_lr{_lr_tag(lr)}"


def _is_prod_default(depth: int, iterations: int, lr: float) -> bool:
    return depth == PROD_DEPTH and iterations == PROD_ITERATIONS and abs(lr - PROD_LEARNING_RATE) < 1e-9


def _parse_configs(s: str) -> List[Tuple[int, int, float]]:
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"--configsの形式が不正です(depth:iterations:learning_rateで指定): {chunk!r}")
        depth, iterations, lr = int(parts[0]), int(parts[1]), float(parts[2])
        out.append((depth, iterations, lr))
    if not out:
        raise ValueError("--configsが空です")
    return out


def _load_baseline_feature_flags(venue: str) -> dict:
    """
    現行本番モデルのmeta.jsonから特徴量構成フラグを読み取る。

    重要: 2026-07-29に判明したことだが、本番モデルの特徴量構成は現在
    24会場で統一されていない(例: 戸田/江戸川/丸亀はwith_recent_form_features=True
    だが桐生はFalseのまま)。ハイパーパラメータの効果だけを公平に比較するには、
    「その会場の現行本番モデルが実際に使っている特徴量構成」を非baseline候補にも
    そのまま引き継ぐ必要がある(でないとハイパーパラメータの差なのか特徴量構成の
    差なのか切り分けられなくなる)。meta.jsonが無い/キーが無い場合は
    _train_one_venueの関数デフォルト(True/True/True/False/False)にフォールバックする。
    """
    meta_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_with_racer_stats_meta.json"
    defaults = {
        "include_form_features": True,
        "include_development_features": True,
        "include_meeting_form_features": True,
        "include_wind_course_interaction_features": False,
        "include_st_std_feature": False,
        # 2026-08-10追加: 潮位特徴量・風速非線形項も他のフラグと同様に本番
        # meta.jsonから引き継ぐ。これを追加し忘れると[[boat_ai_feature_flag_reset_bug]]
        # と同じパターンで、widehold再構築時に全会場が「無効」にリセットされてしまう
        # (_train_one_venueの関数デフォルトがFalseのため)。
        "include_tide_features": False,
        "include_wind_nonlinear_features": False,
    }
    if not meta_path.exists():
        return defaults
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return defaults

    def _flag(key: str, default: bool) -> bool:
        v = meta.get(key)
        return default if v is None else bool(v)

    return {
        "include_form_features": _flag("with_recent_form_features", defaults["include_form_features"]),
        "include_development_features": _flag("with_development_features", defaults["include_development_features"]),
        "include_meeting_form_features": _flag("with_meeting_form_features", defaults["include_meeting_form_features"]),
        "include_wind_course_interaction_features": _flag(
            "with_wind_course_interaction_features", defaults["include_wind_course_interaction_features"]
        ),
        "include_st_std_feature": _flag("with_st_std_feature", defaults["include_st_std_feature"]),
        "include_tide_features": _flag("with_tide_features", defaults["include_tide_features"]),
        "include_wind_nonlinear_features": _flag(
            "with_wind_nonlinear_features", defaults["include_wind_nonlinear_features"]
        ),
    }


def _compute_valid_dates(venue: str, test_size: float) -> set:
    """学習時の_split_by_dateと同じロジックでholdout側の日付集合を復元する。
    ハイパーパラメータを変えてもデータの日付分割自体は変わらないため、
    どの候補でもこの日付集合は同一になる。"""
    venue = normalize_venue_name(venue)
    df = _load_dataset_for_venue(DATASET_PATH, venue)
    racer_stats = _load_racer_stats()
    df = _add_racer_stats_block(df, racer_stats)
    df["date"] = df["date"].astype(str)
    _, valid_raw = _split_by_date(df, test_size)
    valid_dates = set(valid_raw["date"].astype(str).unique())
    return valid_dates


def _backtest_variant(pred_path: Path, venue: str) -> dict:
    if not pred_path.exists():
        return {}

    pred = pd.read_csv(pred_path, low_memory=False)
    if pred.empty:
        return {}

    pred["date"] = pred["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()

    res = pd.read_csv(RESULTS_CSV, low_memory=False)
    res["date"] = res["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    res["venue"] = res["venue"].astype(str).str.strip()
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)

    merged = pred.merge(
        res[["date", "venue", "race_no", "actual_combo", "payout"]],
        on=["date", "venue", "race_no"],
        how="left",
    )

    UNIT_BET_YEN = 100
    rows = []
    for (date, race_no), race_df in merged.groupby(["date", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]
        if pd.isna(actual_combo):
            continue

        ai_preds = []
        for _, r in race_df.iterrows():
            odds = pd.to_numeric(r.get("odds", 0.0), errors="coerce")
            if pd.isna(odds) or odds <= 0:
                continue
            ai_preds.append({"combo": r["combo"], "prob": float(r["prob"]), "odds": float(odds)})

        if not ai_preds:
            continue

        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)
        buy_combos = [b["combo"] for b in best_bets]
        invested = len(buy_combos) * UNIT_BET_YEN
        hit = actual_combo in buy_combos
        returned = int(float(payout)) if hit and pd.notna(payout) else 0

        rows.append({
            "buy_count": len(buy_combos),
            "invested": invested,
            "returned": returned,
            "hit": int(hit),
        })

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    total_invested = df["invested"].sum()
    total_returned = df["returned"].sum()

    return {
        "races": len(df),
        "hit_rate": float(df["hit"].mean()),
        "avg_points": float(df["buy_count"].mean()),
        "invested": int(total_invested),
        "returned": int(total_returned),
        "roi": float(total_returned / total_invested) if total_invested > 0 else 0.0,
    }


def sweep_one_venue(
    venue: str,
    configs: List[Tuple[int, int, float]],
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    venue = normalize_venue_name(venue)
    print(f"\n{'#' * 80}\n# venue={venue}\n{'#' * 80}")

    print("真のholdout日付を復元中(学習時のsplitと同一ロジック)...")
    valid_dates = _compute_valid_dates(venue, test_size)
    print(f"valid dates: {len(valid_dates)}日分 ({min(valid_dates)}..{max(valid_dates)})")

    racer_stats = _load_racer_stats()
    motor_stats = _load_motor_stats()

    feature_flags = _load_baseline_feature_flags(venue)
    print(f"現行本番モデルの特徴量構成を再現(ハイパーパラメータ以外は揃える): {feature_flags}")

    summary_rows = []

    for depth, iterations, lr in configs:
        label = f"depth={depth} iters={iterations} lr={lr:g}"
        is_baseline = _is_prod_default(depth, iterations, lr)

        if is_baseline:
            model_suffix = ""
            out_suffix = "_sweep_hp_baseline"
            label += " (baseline=現行本番)"
        else:
            model_suffix = _config_suffix(depth, iterations, lr)
            out_suffix = f"_sweep{model_suffix}"

            print(f"\n=== 学習: {label} ===")
            _train_one_venue(
                src=DATASET_PATH, venue=venue, racer_stats=racer_stats,
                test_size=test_size, random_state=random_state,
                iterations=iterations, depth=depth, learning_rate=lr,
                verbose_eval=0,
                skip_baseline_rerun=True,
                motor_stats=motor_stats,
                model_suffix=model_suffix,
                **feature_flags,
            )

        print(f"\n=== 予測生成(真のholdoutのみ): {label} ===")
        generate_for_venue(
            venue, temperature=0.90, min_races=5,
            model_suffix=model_suffix, valid_dates=valid_dates, out_suffix=out_suffix,
            test_size=test_size,
        )

        pred_path = LOG_DIR / f"predictions_with_racer_stats_{venue}{out_suffix}.csv"
        stats = _backtest_variant(pred_path, venue)
        stats["depth"] = depth
        stats["iterations"] = iterations
        stats["learning_rate"] = lr
        stats["label"] = label
        stats["is_baseline"] = is_baseline
        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    cols = ["label", "depth", "iterations", "learning_rate", "is_baseline",
            "races", "hit_rate", "avg_points", "roi", "invested", "returned"]
    cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df[cols]

    out_path = LOG_DIR / f"hyperparameter_sweep_{venue}.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"SWEEP RESULT: {venue}")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\n保存先: {out_path}")

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="", help="会場名(例: 戸田)")
    parser.add_argument("--all", action="store_true", help="全会場で実行(時間がかかるので候補セットを絞ることを推奨)")
    parser.add_argument(
        "--configs", default="",
        help="depth:iterations:learning_rate のカンマ区切りリスト。"
             "未指定ならデフォルト候補セット(現行本番+浅め/深め/多め低学習率/少なめ高学習率)を使う。",
    )
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    configs = _parse_configs(args.configs) if args.configs else DEFAULT_CONFIGS
    print("試す候補:")
    for depth, iterations, lr in configs:
        tag = " (baseline)" if _is_prod_default(depth, iterations, lr) else ""
        print(f"  depth={depth} iterations={iterations} learning_rate={lr:g}{tag}")

    if args.all:
        venues = list(VENUE_ORDER)
    elif args.venue:
        venues = [args.venue]
    else:
        raise SystemExit("--venue か --all を指定してください")

    all_results = []
    for v in venues:
        try:
            df = sweep_one_venue(v, configs, args.test_size, args.random_state)
            df["venue"] = normalize_venue_name(v)
            all_results.append(df)
        except Exception as e:
            print(f"[ERROR] venue={v}: {e}")

    if len(venues) > 1 and all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = LOG_DIR / "hyperparameter_sweep_all_venues.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"\n全会場まとめ保存先: {combined_path}")

        print("\n" + "=" * 80)
        print("候補ごとの全会場合算ROI")
        print("=" * 80)
        agg = combined.groupby("label", sort=False).agg(
            races=("races", "sum"), invested=("invested", "sum"), returned=("returned", "sum"),
        )
        agg["roi"] = agg["returned"] / agg["invested"]
        print(agg.to_string())


if __name__ == "__main__":
    main()
