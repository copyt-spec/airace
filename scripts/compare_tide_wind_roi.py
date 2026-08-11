from __future__ import annotations

"""
scripts/generate_predictions_with_racer_stats_model.py --model_suffix "_tidewind" / "_tidewind_off"
で会場について生成した predictions_with_racer_stats_{venue}_tidewind(.off).csv を読み込み、
今のbuy_selector.select_best_bets ロジックで backtest_buy_selector.py と同じ計算をして、
潮位特徴量(tide_level_cm/tide_trend_cmph/course_tide_interaction)+風速非線形項
(wind_speed_mps_sq/wave_cm_sq)あり/なしのROIを公正に比較する。

両方とも generate_predictions...py 側で「学習に使っていない日付(holdout)」だけに
自動で絞り込み済みなので、ここでの比較はリーク無し。

使い方:
  python scripts/compare_tide_wind_roi.py --venue 児島
  python scripts/compare_tide_wind_roi.py --venue 下関 --with_suffix _tidewind_seed2 --without_suffix _tidewind_off_seed2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_buy_selector import run_backtest, summarize, print_summary  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
RESULTS_CSV = LOG_DIR / "results.csv"


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def load_variant(venue: str, suffix: str) -> pd.DataFrame:
    path = LOG_DIR / f"predictions_with_racer_stats_{venue}{suffix}.csv"
    if not path.exists():
        raise RuntimeError(f"{suffix}: 予測ファイルが見つかりません: {path}")

    pred = pd.read_csv(path, low_memory=False)
    pred["date"] = _normalize_date(pred["date"])
    pred["venue"] = pred["venue"].astype(str).str.strip()
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()

    res = pd.read_csv(RESULTS_CSV, low_memory=False)
    res["date"] = _normalize_date(res["date"])
    res["venue"] = res["venue"].astype(str).str.strip()
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)

    merged = pred.merge(
        res[["date", "venue", "race_no", "actual_combo", "payout"]],
        on=["date", "venue", "race_no"],
        how="left",
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", required=True, help="会場名(例: 児島)")
    parser.add_argument(
        "--with_suffix", default="_tidewind",
        help="あり側のmodel_suffix(例: 別random_stateで頑健性確認する場合は_tidewind_seed2など)",
    )
    parser.add_argument(
        "--without_suffix", default="_tidewind_off",
        help="なし側(baseline)のmodel_suffix",
    )
    args = parser.parse_args()
    venue = args.venue

    variants = [
        (args.with_suffix, "潮位+風速非線形あり(with tide/wind_nonlinear)"),
        (args.without_suffix, "潮位+風速非線形なし(baseline)"),
    ]

    summaries = {}

    for suffix, label in variants:
        print("\n" + "=" * 80)
        print(label, f"(suffix={suffix})")
        print("=" * 80)
        path = LOG_DIR / f"predictions_with_racer_stats_{venue}{suffix}.csv"
        if not path.exists():
            print(f"[SKIP] {venue}{suffix}: 予測ファイルが見つかりません({path})。"
                  f"実オッズ付きholdoutレースが少なすぎてgenerate_predictions側でスキップされた"
                  f"可能性があります。")
            summaries = None
            break
        df = load_variant(venue, suffix)
        print("rows:", len(df), "races:", df.drop_duplicates(["date", "venue", "race_no"]).shape[0])
        print("date range:", df["date"].min(), "..", df["date"].max())

        res = run_backtest(df)
        summary = summarize(res, label)
        print_summary(summary)
        summaries[suffix] = summary

    if not summaries:
        print(f"\n[SKIP] {venue}: 比較に必要な予測ファイルが揃わなかったため、比較をスキップしました。")
        return

    print("\n" + "=" * 80)
    print(f"{venue} 全体比較")
    print("=" * 80)
    a = summaries[args.with_suffix]
    b = summaries[args.without_suffix]
    print(f"{'':20s} {'あり':>12s} {'なし':>12s} {'差分':>12s}")
    print(f"{'races':20s} {a['races']:12d} {b['races']:12d} {a['races']-b['races']:12d}")
    print(f"{'hit_rate':20s} {a['hit_rate']*100:11.2f}% {b['hit_rate']*100:11.2f}% {(a['hit_rate']-b['hit_rate'])*100:11.2f}%")
    print(f"{'roi':20s} {a['roi']*100:11.2f}% {b['roi']*100:11.2f}% {(a['roi']-b['roi'])*100:11.2f}%")
    print(f"{'profit':20s} {a['profit']:12.0f} {b['profit']:12.0f} {a['profit']-b['profit']:12.0f}")


if __name__ == "__main__":
    main()
