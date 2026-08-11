from __future__ import annotations

"""
scripts/compare_tide_wind_roi.py で会場別ROIを比較した際、会場によって
潮位+風速非線形特徴量ありのROIが大きく上下(+35pt〜-31pt)にばらついた。
「向きの逆転(上潮でインが有利な会場にアウト有利の補正を入れてしまった等)」の
可能性をユーザーから指摘されたが、この特徴量は会場ごとの固定ルールを人間が
書いたものではなく、tide_level_cm・tide_trend_cmph・lane{n}_course_tide_interaction
を生の数値として渡し、会場ごとに独立学習しているCatBoostモデル自身に向きを
学習させる設計になっている([[boat_ai_tide_wind_nonlinear_feature_impl]]参照)。

このスクリプトは各会場の検証用モデル(--model_suffix _tidewind で学習済み)を
読み込み、潮位/風速非線形の各特徴量がCatBoostのfeature importance(PredictionValuesChange、
%単位、全特徴量で合計100)の何位・何%かを一覧にする。

「重要度が低いのにROIが悪化した会場」→ ノイズを拾って過学習しただけの可能性が高い。
「重要度が高いのにROIが悪化した会場」→ 学習はしているが訓練期間とholdout期間で
関係が安定しない(汎化していない)可能性が高く、概算発走時刻のズレなど別の原因を
疑うべき。

使い方:
  python -m scripts.inspect_tide_wind_feature_importance
"""

import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "data" / "models"
SUFFIX = "_tidewind"

TIDE_WIND_COLS = [
    "tide_level_cm",
    "tide_trend_cmph",
    "wind_speed_mps_sq",
    "wave_cm_sq",
]
# lane1〜lane6のコース×潮位トレンド交互作用も対象にする(接頭辞一致で拾う)
TIDE_WIND_PREFIXES = ["lane", "course_tide_interaction"]

# 2026-08-10時点のROIバックテスト結果(compare_tide_wind_roi.pyの出力より、roi差分pt)
ROI_DIFF = {
    "津": 35.82, "下関": 35.64, "若松": 35.05, "三国": 27.16, "宮島": 21.07,
    "蒲郡": 12.32, "児島": 10.22, "平和島": 8.33, "福岡": 6.33, "江戸川": 5.45,
    "浜名湖": 0.09, "芦屋": -4.30, "徳山": -4.38, "丸亀": -4.49, "常滑": -13.39,
    "尼崎": -22.99, "住之江": -29.37, "多摩川": -30.87,
}


def _is_tide_wind_col(col: str) -> bool:
    if col in TIDE_WIND_COLS:
        return True
    return "tide" in col.lower()


def inspect_venue(venue: str) -> pd.DataFrame | None:
    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_with_racer_stats{SUFFIX}.cbm"
    if not model_path.exists():
        print(f"[SKIP] {venue}: モデルファイルが見つかりません({model_path})")
        return None

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    feature_names = model.feature_names_
    importances = model.get_feature_importance()
    df = pd.DataFrame({"feature": feature_names, "importance_pct": importances})
    df = df.sort_values("importance_pct", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["total_features"] = len(df)

    tide_wind_rows = df[df["feature"].apply(_is_tide_wind_col)]
    return tide_wind_rows


def main() -> None:
    print(f"{'venue':8s} {'roi_diff':>9s}  {'feature':30s} {'rank':>6s} {'importance%':>12s}")
    print("-" * 80)
    for venue in sorted(ROI_DIFF, key=lambda v: ROI_DIFF[v], reverse=True):
        rows = inspect_venue(venue)
        if rows is None or rows.empty:
            print(f"{venue:8s} {ROI_DIFF[venue]:9.2f}  (潮位/風速特徴量が見つかりません)")
            continue
        for i, r in rows.iterrows():
            venue_label = venue if i == rows.index[0] else ""
            roi_label = f"{ROI_DIFF[venue]:9.2f}" if i == rows.index[0] else " " * 9
            print(f"{venue_label:8s} {roi_label}  {r['feature']:30s} {r['rank']:6.0f}/{r['total_features']:.0f} {r['importance_pct']:11.2f}%")
        print()


if __name__ == "__main__":
    main()
