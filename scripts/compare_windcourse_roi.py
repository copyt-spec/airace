from __future__ import annotations

"""
scripts/generate_predictions_with_racer_stats_model.py --model_suffix "_windcourse" / "_windcourse_off"
で丸亀について生成した predictions_with_racer_stats_丸亀_windcourse(.off).csv を読み込み、
今のbuy_selector.select_best_bets ロジックで backtest_buy_selector.py と同じ計算をして、
コース×風速/波高 交互作用特徴量あり/なしのROIを公正に比較する(丸亀のみ)。

両方とも generate_predictions...py 側で「学習に使っていない日付(holdout)」だけに
自動で絞り込み済みなので、ここでの比較はリーク無し。

使い方:
  python scripts/compare_windcourse_roi.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_buy_selector import run_backtest, summarize, print_summary  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
RESULTS_CSV = LOG_DIR / "results.csv"
VENUE = "丸亀"


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def load_variant(suffix: str) -> pd.DataFrame:
    path = LOG_DIR / f"predictions_with_racer_stats_{VENUE}{suffix}.csv"
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
    variants = [
        ("_windcourse", "コース×風速/波高あり(with wind_course_interaction)"),
        ("_windcourse_off", "コース×風速/波高なし(baseline, without wind_course_interaction)"),
    ]

    summaries = {}

    for suffix, label in variants:
        print("\n" + "=" * 80)
        print(label, f"(suffix={suffix})")
        print("=" * 80)
        df = load_variant(suffix)
        print("rows:", len(df), "races:", df.drop_duplicates(["date", "venue", "race_no"]).shape[0])
        print("date range:", df["date"].min(), "..", df["date"].max())

        res = run_backtest(df)
        summary = summarize(res, label)
        print_summary(summary)
        summaries[suffix] = summary

    print("\n" + "=" * 80)
    print("丸亀 全体比較")
    print("=" * 80)
    a = summaries["_windcourse"]
    b = summaries["_windcourse_off"]
    print(f"{'':20s} {'あり':>12s} {'なし':>12s} {'差分':>12s}")
    print(f"{'races':20s} {a['races']:12d} {b['races']:12d} {a['races']-b['races']:12d}")
    print(f"{'hit_rate':20s} {a['hit_rate']*100:11.2f}% {b['hit_rate']*100:11.2f}% {(a['hit_rate']-b['hit_rate'])*100:11.2f}%")
    print(f"{'roi':20s} {a['roi']*100:11.2f}% {b['roi']*100:11.2f}% {(a['roi']-b['roi'])*100:11.2f}%")
    print(f"{'profit':20s} {a['profit']:12.0f} {b['profit']:12.0f} {a['profit']-b['profit']:12.0f}")


if __name__ == "__main__":
    main()
