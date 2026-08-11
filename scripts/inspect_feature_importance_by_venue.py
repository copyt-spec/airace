from __future__ import annotations

"""
指定会場の本番CatBoostモデル(_with_racer_stats.cbm)から実際の特徴量重要度
(PredictionValuesChange %)を取り出し、個別上位ランキングとカテゴリ別合計の
両方を表示する。「今、確率算出にどのパラメータがどれくらい効いているか」を
確認するための汎用ツール(scripts/inspect_tide_wind_feature_importance.pyの
潮位/風速版を全特徴量に一般化したもの)。

このサンドボックスにはcatboostが無いため実行できない。ユーザーの手元環境で
実行すること。

使い方:
  python scripts/inspect_feature_importance_by_venue.py --venue 下関
  python scripts/inspect_feature_importance_by_venue.py --venue 下関 --top 30
  python scripts/inspect_feature_importance_by_venue.py --all   # 24会場まとめて上位10件+カテゴリ内訳
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "data" / "models"


_POS_PREFIX = r"(?:lane\d|first|second|third)"


def categorize(c: str) -> str:
    if re.search(r"course_wind_interaction|course_wave_interaction", c):
        return "コース×風/波 交互作用"
    if "tide" in c:
        return "潮位"
    if c in ("wind_speed_mps_sq", "wave_cm_sq"):
        return "風速/波高(非線形2乗項)"
    if c in ("wind_dir", "wind_dir_code", "wind_speed_mps", "wave_cm", "weather", "weather_code"):
        return "天候(生値)"
    if "meeting_so_far" in c:
        return "今節成績"
    if "recent5" in c or "motor_recent8" in c:
        return "選手/モーター直近成績"
    if "st_std" in c:
        return "ST安定性"
    # 2026-08-11修正: combo候補の1着/2着/3着に割り当てた艇の特徴量
    # (first_course/second_win_rate_prior等、combo_first_laneでlaneを引いた後の
    # 値)がlane\d_接頭辞版と同じ意味なのに未分類の"その他"に落ちていたバグを修正。
    # first_/second_/third_接頭辞もlane\d_接頭辞と同じカテゴリ判定にまとめる。
    if re.match(rf"^{_POS_PREFIX}_(win_rate_prior|place_rate_prior|avg_st_prior|races_prior|course_place_rate_prior|course_avg_st_prior)$", c):
        return "選手通算成績(事前)"
    if re.match(rf"^{_POS_PREFIX}_course$", c):
        return "コース(枠番)"
    if re.match(rf"^{_POS_PREFIX}_(boat|racer_no|motor|exhibit|st|grade_num|st_inv)$", c):
        return "基本属性(艇番/選手番号/モーター/展示/ST/級別)"
    if re.search(r"exhibit_rank|st_rank|st_diff$", c):
        return "展開特徴量(展示/ST順位・差分)"
    if c.startswith("combo_"):
        return "組み合わせ形状(3連単の並び方)"
    return "その他"


def inspect_venue(venue: str, top_n: int) -> None:
    venue = normalize_venue_name(venue)
    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_with_racer_stats.cbm"
    if not model_path.exists():
        print(f"[SKIP] {venue}: モデルが見つかりません({model_path})")
        return

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    feature_names = model.feature_names_
    importances = model.get_feature_importance()

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df["category"] = df["feature"].map(categorize)
    df = df.sort_values("importance", ascending=False)

    print("\n" + "=" * 80)
    print(f"{venue} (n_features={len(df)})")
    print("=" * 80)
    print(f"--- 上位{top_n}件(個別) ---")
    print(df.head(top_n).to_string(index=False, formatters={"importance": "{:.2f}".format}))

    print("\n--- カテゴリ別合計(PredictionValuesChange % の合計) ---")
    cat = df.groupby("category")["importance"].sum().sort_values(ascending=False)
    cat_pct = (cat / cat.sum() * 100).round(2)
    for name, pct in cat_pct.items():
        print(f"  {name:30s} {pct:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="", help="会場名(例: 下関)")
    parser.add_argument("--all", action="store_true", help="全24会場")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if args.all:
        for v in VENUE_ORDER:
            inspect_venue(v, top_n=10)
    elif args.venue:
        inspect_venue(args.venue, top_n=args.top)
    else:
        raise SystemExit("--venue か --all を指定してください")


if __name__ == "__main__":
    main()
