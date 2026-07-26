from __future__ import annotations

"""
scripts/generate_predictions_with_racer_stats_model.py --model_suffix "_meetingform" / "_meetingform_off"
で会場ごとに生成した predictions_with_racer_stats_{venue}_meetingform(.off).csv を全会場分まとめて読み込み、
今のbuy_selector.select_best_bets ロジックで backtest_buy_selector.py と同じ計算をして、
「今節これまでの成績」特徴量あり/なしのROIを公正に比較する。

両方とも generate_predictions...py 側で「学習に使っていない日付(holdout)」だけに
自動で絞り込み済みなので、ここでの比較はリーク無し。

使い方:
  python scripts/compare_meeting_form_roi.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import VENUE_ORDER  # noqa: E402
from scripts.backtest_buy_selector import run_backtest, summarize, print_summary  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
RESULTS_CSV = LOG_DIR / "results.csv"


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def load_variant(suffix: str) -> pd.DataFrame:
    frames = []
    missing = []
    for venue in VENUE_ORDER:
        path = LOG_DIR / f"predictions_with_racer_stats_{venue}{suffix}.csv"
        if not path.exists():
            missing.append(venue)
            continue
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)

    if missing:
        print(f"[{suffix}] スキップ(ファイル無し、holdoutレース不足等): {missing}")

    if not frames:
        raise RuntimeError(f"{suffix}: 予測ファイルが1件も見つかりません")

    pred = pd.concat(frames, ignore_index=True)
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
        ("_meetingform", "今節成績あり(with meeting_form)"),
        ("_meetingform_off", "今節成績なし(baseline, without meeting_form)"),
    ]

    summaries = {}
    per_venue = {}

    for suffix, label in variants:
        print("\n" + "=" * 80)
        print(label, f"(suffix={suffix})")
        print("=" * 80)
        df = load_variant(suffix)
        print("rows:", len(df), "races:", df.drop_duplicates(["date", "venue", "race_no"]).shape[0])

        res = run_backtest(df)
        summary = summarize(res, label)
        print_summary(summary)
        summaries[suffix] = summary

        by_venue = res.groupby("venue").apply(lambda g: pd.Series(summarize(g))).reset_index()
        per_venue[suffix] = by_venue

    print("\n" + "=" * 80)
    print("全体比較")
    print("=" * 80)
    a = summaries["_meetingform"]
    b = summaries["_meetingform_off"]
    print(f"{'':20s} {'あり':>12s} {'なし':>12s} {'差分':>12s}")
    print(f"{'races':20s} {a['races']:12d} {b['races']:12d} {a['races']-b['races']:12d}")
    print(f"{'hit_rate':20s} {a['hit_rate']*100:11.2f}% {b['hit_rate']*100:11.2f}% {(a['hit_rate']-b['hit_rate'])*100:11.2f}%")
    print(f"{'roi':20s} {a['roi']*100:11.2f}% {b['roi']*100:11.2f}% {(a['roi']-b['roi'])*100:11.2f}%")
    print(f"{'profit':20s} {a['profit']:12.0f} {b['profit']:12.0f} {a['profit']-b['profit']:12.0f}")

    print("\n会場別ROI比較(共通で存在する会場のみ):")
    merged_venue = per_venue["_meetingform"][["venue", "races", "roi", "profit"]].merge(
        per_venue["_meetingform_off"][["venue", "races", "roi", "profit"]],
        on="venue", suffixes=("_あり", "_なし"),
    )
    merged_venue["roi_diff"] = merged_venue["roi_あり"] - merged_venue["roi_なし"]
    merged_venue = merged_venue.sort_values("roi_diff")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(merged_venue.to_string(index=False))


if __name__ == "__main__":
    main()
