from __future__ import annotations

"""
「選手の勝率(win_rate_prior)の比重をもっと重くしたら的中率(hit_rate)は
上がるか?」というユーザーの疑問に、実際に手を動かして答えるためのスイープ
スクリプト。

背景(重要): CatBoostのような決定木ベースのモデルは分割点(閾値)で学習する
ため、特徴量の値をスケーリングしても分割自体は変わらない
(win_rate>0.35 も win_rate*2>0.70 も同じ情報)。したがって「値に定数を
掛けるだけ」では的中率は一切変わらない。本当に重みを変えるには、CatBoostの
feature_weights(学習時に特徴量ごとの分割探索の優先度を変えるパラメータ)を
使う必要があり、scripts/train_binary_catboost_per_venue_with_racer_stats.py
に --feature_weight_targets / --feature_weight_multiplier として追加済み。

ただし、これは解析的に最適値(黄金比)が求まるようなものではなく、
複数の倍率で再学習し、眞のholdout(学習に一切使っていない日付)で
hit_rate/ROIを比較するグリッドサーチでしか良し悪しを判断できない。
このスクリプトはそのグリッドサーチを自動化する。

前提: このサンドボックスにはcatboost/sklearnが無いため実行できない。
catboostが使えるユーザーのローカル環境で実行すること。

使い方:
  python -m scripts.sweep_racer_stats_weight --venue 戸田 \
      --targets win_rate_prior --multipliers 1,2,4,8,16

出力:
  - 各倍率の学習済みモデル: data/models/trifecta_binary_catboost_{venue}_with_racer_stats_w{m}x.cbm
    (multiplier=1.0は既存の本番モデルをそのままbaselineとして使い、再学習しない)
  - 眞のholdout日付に絞った予測CSV: data/logs/predictions_with_racer_stats_{venue}_sweep_w{m}x_holdout.csv
  - 比較サマリー: data/logs/racer_stats_weight_sweep_{venue}.csv (倍率ごとのhit_rate/ROI/avg_points)
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402
from engine.venue_registry import normalize_venue_name  # noqa: E402
from scripts.train_binary_catboost_per_venue_with_racer_stats import (  # noqa: E402
    DATASET_PATH,
    VENUE_DEPTH_OVERRIDES,
    _load_dataset_for_venue,
    _load_racer_stats,
    _load_motor_stats,
    _add_racer_stats_block,
    _default_feature_flags_for_venue,
    _split_by_date,
    _train_one_venue,
)
from scripts.generate_predictions_with_racer_stats_model import generate_for_venue  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
RESULTS_CSV = LOG_DIR / "results.csv"


def _compute_valid_dates(venue: str, test_size: float) -> set:
    """
    学習時の_split_by_dateと全く同じロジック(同じ日付ソート・同じtest_size)で
    valid(holdout)側の日付集合を復元する。特徴量重みを変えてもデータの
    日付分割自体は変わらないため、どの倍率でもこの日付集合は同一になる。
    """
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="戸田")
    parser.add_argument("--targets", default="win_rate_prior",
                         help="重みを上げる特徴量名の部分一致文字列(カンマ区切り)")
    parser.add_argument("--multipliers", default="1,2,4,8,16",
                         help="試す倍率のカンマ区切りリスト(1は既存の本番モデルをbaselineとして使う)")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument(
        "--depth", type=int, default=None,
        help="未指定の場合、VENUE_DEPTH_OVERRIDES(本番でdepth=6にチューニング済みの"
             "13会場)に従って自動選択する。それ以外の会場はdepth=8。"
             "明示的に指定した場合はその値を全倍率に一律適用する。",
    )
    parser.add_argument("--learning_rate", type=float, default=0.05)
    args = parser.parse_args()

    venue = normalize_venue_name(args.venue)
    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    multipliers: List[float] = [float(s.strip()) for s in args.multipliers.split(",") if s.strip()]

    # 2026-08-03バグ修正: 以前はここでdepth=8固定・motor_statsを渡さず・
    # 特徴量構成(include_form_features等)も全会場一律Trueで再学習していたため、
    # depth=6にチューニング済みの会場や、桐生のようにform/motor/meeting_form特徴量を
    # 意図的に無効化している会場で、multiplier=1.0(既存本番モデル)との比較条件が
    # ズレていた(depthやモーター特徴量の有無が結果に混ざり込み、重みの効果だけを
    # 切り分けられなかった)。本番と同じ_default_feature_flags_for_venue()・
    # VENUE_DEPTH_OVERRIDES・実際のmotor_statsを使うよう修正。
    depth = args.depth if args.depth is not None else VENUE_DEPTH_OVERRIDES.get(venue, 8)
    feature_flags = _default_feature_flags_for_venue(venue)
    print(f"venue={venue} targets={targets} multipliers={multipliers} depth={depth} feature_flags={feature_flags}")

    print("\n真のholdout日付を復元中(学習時のsplitと同一ロジック)...")
    valid_dates = _compute_valid_dates(venue, args.test_size)
    print(f"valid dates: {len(valid_dates)}日分 ({min(valid_dates)}..{max(valid_dates)})")

    racer_stats = _load_racer_stats()
    motor_stats = _load_motor_stats()

    summary_rows = []

    for m in multipliers:
        if m == 1.0:
            # baseline: 既存の本番モデル(_with_racer_stats.cbm)をそのまま使う。
            # 再学習しない(学習し直しても同条件なら理論上同じ結果になるはずだが、
            # 既存の検証済みモデルをbaselineにする方が安全・高速)。
            model_suffix = ""
            out_suffix = "_sweep_baseline"
            label = "1x (baseline, 既存本番モデル)"
        else:
            model_suffix = f"_w{m:g}x"
            out_suffix = f"_sweep_w{m:g}x"
            label = f"{m:g}x"

            print(f"\n=== 学習: multiplier={m} ===")
            _train_one_venue(
                src=DATASET_PATH, venue=venue, racer_stats=racer_stats,
                test_size=args.test_size, random_state=args.random_state,
                iterations=args.iterations, depth=depth,
                learning_rate=args.learning_rate, verbose_eval=0,
                skip_baseline_rerun=True,
                feature_weight_targets=targets,
                feature_weight_multiplier=m,
                motor_stats=motor_stats,
                **feature_flags,
            )

        print(f"\n=== 予測生成(真のholdoutのみ): multiplier={m} ===")
        generate_for_venue(
            venue, temperature=0.90, min_races=5,
            model_suffix=model_suffix, valid_dates=valid_dates, out_suffix=out_suffix,
        )

        pred_path = LOG_DIR / f"predictions_with_racer_stats_{venue}{out_suffix}.csv"
        stats = _backtest_variant(pred_path, venue)
        stats["multiplier"] = m
        stats["label"] = label
        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    cols = ["label", "multiplier", "races", "hit_rate", "avg_points", "roi", "invested", "returned"]
    cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df[cols].sort_values("multiplier")

    out_path = LOG_DIR / f"racer_stats_weight_sweep_{venue}.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"SWEEP RESULT: {venue} (targets={targets})")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\n保存先: {out_path}")


if __name__ == "__main__":
    main()
