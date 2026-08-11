from __future__ import annotations

"""
task#54: 風×コース利き方(include_wind_course_interaction_features)を
全24会場で再検証するための一括ドライバー。

丸亀だけは2026-07-26に検証済み(ROI -3.63pt、[[boat_ai_marugame_wind_course_feature]]で
不採用)だったが、holdout102レースのみでの単発検証だったため、
[[boat_ai_tide_wind_all_venue_roi_and_robustness]]と同じ方法論
(会場ごとに現在の本番特徴量構成を再現した上でwind_course特徴量だけ
on/offしてholdout ROIを比較)で全24会場を検証する。

会場ごとの「現在の本番特徴量構成」は
scripts/train_binary_catboost_per_venue_with_racer_stats.py の
_default_feature_flags_for_venue()と同じルール(VENUE_DEPTH_OVERRIDES /
MEETING_FORM_VENUES / FORM_FEATURE_DISABLED_VENUES / TIDE_WIND_VENUES)から
再現する。--venue単発実行はこれらのデフォルトを自動適用しない
([[boat_ai_feature_flag_reset_bug]]と同じ理由でCLIフラグを明示する必要がある)ため、
このスクリプトがその変換を代行する。

事前準備: このサンドボックスにはcatboostが無いため、ユーザーの手元環境
(catboostが使える場所)で実行すること。

使い方:
  python scripts/run_windcourse_all_venue_check.py                 # 全24会場
  python scripts/run_windcourse_all_venue_check.py --venue 丸亀      # 単一会場のみ
  python scripts/run_windcourse_all_venue_check.py 2>&1 | tee data/logs/windcourse_all_venue.log

各会場ごとに:
  1) --enable_wind_course_interaction_features 付きで学習 (--model_suffix _windcourse)
  2) 付けずに学習 (--model_suffix _windcourse_off)
     ※どちらも会場ごとの本番構成(depth/meeting_form/tide_wind)は共通で揃える
  3) 両方についてholdout予測を生成 (generate_predictions_with_racer_stats_model.py)
  4) scripts/compare_windcourse_roi.py --venue X でROI比較を表示
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_binary_catboost_per_venue_with_racer_stats import (  # noqa: E402
    VENUE_DEPTH_OVERRIDES,
    MEETING_FORM_VENUES,
    FORM_FEATURE_DISABLED_VENUES,
    TIDE_WIND_VENUES,
)
from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_binary_catboost_per_venue_with_racer_stats.py"
GEN_SCRIPT = PROJECT_ROOT / "scripts" / "generate_predictions_with_racer_stats_model.py"
COMPARE_SCRIPT = PROJECT_ROOT / "scripts" / "compare_windcourse_roi.py"


def _production_flags(venue: str) -> list[str]:
    """会場ごとの現在の本番特徴量構成を、--venue単発実行用のCLIフラグに変換する。
    (_default_feature_flags_for_venue()と同じルールをここで再現。
    wind_course自体はここに含めない、呼び出し側でon/offする)"""
    venue = normalize_venue_name(venue)
    flags: list[str] = []

    depth = VENUE_DEPTH_OVERRIDES.get(venue, 8)
    flags += ["--depth", str(depth)]

    if venue in FORM_FEATURE_DISABLED_VENUES:
        flags.append("--disable_form_features")

    if venue not in MEETING_FORM_VENUES:
        flags.append("--disable_meeting_form_features")

    if venue in TIDE_WIND_VENUES:
        flags.append("--enable_tide_features")
        flags.append("--enable_wind_nonlinear_features")

    return flags


def _run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def process_venue(venue: str) -> None:
    venue = normalize_venue_name(venue)
    print("\n" + "#" * 100)
    print(f"# {venue}")
    print("#" * 100)

    base_flags = _production_flags(venue)
    print(f"[{venue}] 本番構成フラグ: {base_flags}")

    # 1) with wind_course
    rc = _run([
        sys.executable, str(TRAIN_SCRIPT),
        "--venue", venue,
        "--model_suffix", "_windcourse",
        "--enable_wind_course_interaction_features",
        *base_flags,
    ])
    if rc != 0:
        print(f"[ERROR] {venue}: 学習(with)に失敗しました(rc={rc})。この会場をスキップします。")
        return

    # 2) without wind_course (baseline, 他は同じ構成)
    rc = _run([
        sys.executable, str(TRAIN_SCRIPT),
        "--venue", venue,
        "--model_suffix", "_windcourse_off",
        *base_flags,
    ])
    if rc != 0:
        print(f"[ERROR] {venue}: 学習(without)に失敗しました(rc={rc})。この会場をスキップします。")
        return

    # 3) holdout予測生成(両方)
    for suffix in ("_windcourse", "_windcourse_off"):
        rc = _run([
            sys.executable, str(GEN_SCRIPT),
            "--venue", venue,
            "--model_suffix", suffix,
        ])
        if rc != 0:
            print(f"[ERROR] {venue}: 予測生成({suffix})に失敗しました(rc={rc})。")

    # 4) ROI比較
    _run([sys.executable, str(COMPARE_SCRIPT), "--venue", venue])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="", help="単一会場のみ実行したい場合に指定(例: 丸亀)")
    args = parser.parse_args()

    venues = [args.venue] if args.venue else list(VENUE_ORDER)

    for v in venues:
        try:
            process_venue(v)
        except Exception as e:
            print(f"[ERROR] venue={v}: {e}")

    print("\n" + "=" * 100)
    print("全会場の学習・比較が完了しました。上のROI比較(会場ごとの「全体比較」ブロック)を確認してください。")
    print("=" * 100)


if __name__ == "__main__":
    main()
