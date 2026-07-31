# scripts/build_combined_predictions_for_sim.py
# -*- coding: utf-8 -*-
"""
/simのグラフを「新モデル(展開特徴量追加後、2026-07-22デプロイ)」の結果だけで
継続的に最新化するための合成データを作る。

背景: data/logs/predictions.csv は長期間・複数モデルバージョンが混在した
「本物の履歴ログ」で、日々の運用(アプリ閲覧やdaily-prediction-crawl)で
これからも増え続ける。一方 scripts/build_new_model_predictions_snapshot.py
が2026-07-22時点で作った data/logs/predictions_new_model_snapshot.csv は、
新モデル自身の学習時test_sizeに基づく「真のholdout期間」(2025-09-25〜
2026-07-14)の実オッズ付き予測を、一度だけ再現した静的なスナップショット。

この2つを日付で単純に結合する:
  - CUTOFF_DATE (デフォルト20260721。新モデル投入日は20260722だが、
    daily-prediction-crawlによる0721分のバックフィルがlogged_at的に投入後
    [2026-07-23 23:31以降]に新モデルで行われたことを確認済みのため前倒し)
    より前は静的スナップショット(新モデルの検証用holdout結果)を使う
  - CUTOFF_DATE 以降は data/logs/predictions.csv の実ログを使う
    (この日以降にログされた行は、新モデルデプロイ後の本物の実績なので、
    そのまま使ってよい)

これにより、/simは「新モデルの真のholdout実績」を土台にしつつ、日々の
実運用が増えるほど本物のデータで上書き・拡張されていく。
scripts/update_results_daily.py の日次パイプラインから呼ばれる想定
(merge_prediction_results.py --predictions <このスクリプトの出力> という形で使う)。

出力: data/logs/predictions_combined_for_sim.csv
  (predictions.csvと同じ列構成)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import normalize_venue_name  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"

SNAPSHOT_CSV = LOG_DIR / "predictions_new_model_snapshot.csv"
LIVE_PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
OUT_CSV = LOG_DIR / "predictions_combined_for_sim.csv"

# 2026-07-31追加(重要): 2026-07-30にdepth=6+特徴量構成修正版モデルへの
# widehold全期間再学習を行い、/simを新モデル基準に作り直した([[boat_ai_sim_widehold_rebuild]]、
# scripts/build_widehold_snapshot_models.py参照)。この新モデルはまだRenderにpush
# していないため、predictions.csv(実ログ)は当面古いモデルのままログされ続ける。
# この状態でCUTOFF_DATEを過去日のままにしておくと、scripts/update_results_daily.py
# の日次自動実行(cronでこのスクリプトを引数無しで呼ぶ)のたびに古いモデルの
# 実ログがsimに混入してしまう(実際に2026-07-31の自動実行で発生し、
# predictions_combined_for_sim.csvが汚染される事故があった)。
# そのため、実際にRenderへpushして本番が新モデルに切り替わるまでは、
# CUTOFF_DATEを未来の安全な日付に固定し、常にwidehold静的スナップショットだけを
# 使うようにする。**Renderへpushしたら、この値を実際のpush(デプロイ)日に
# 更新すること。** 更新を忘れると、新モデルが本番稼働しているのにsimがいつまでも
# 静的スナップショットのままになってしまう。
CUTOFF_DATE = "20261231"

# 2026-07-26追加、2026-07-31にデフォルト無効化: 「今節成績」特徴量は全会場一律
# ではなく9会場だけ選択導入していた(scripts/build_meeting_form_predictions_snapshot.py)。
# widehold再学習(2026-07-30〜)は各会場の現行本番構成(9会場のmeeting_form含む)を
# そのまま引き継いでいるため、この2段目レイヤーは不要になった。二重適用すると
# 古い(test_size=0.20の狭い)meeting_formスナップショットでwidehold分が上書きされて
# しまうため、既定では適用しない(--apply-meeting-form-layerを明示した時だけ適用)。
MEETING_FORM_SNAPSHOT_CSV = LOG_DIR / "predictions_meetingform_snapshot.csv"
MEETING_FORM_VENUES = [
    "戸田", "江戸川", "丸亀", "びわこ", "若松", "下関", "芦屋", "鳴門", "浜名湖",
]
MEETING_FORM_CUTOFF_DATE = "20260726"


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-date", default=CUTOFF_DATE, help="YYYYMMDD")
    parser.add_argument("--snapshot", default=str(SNAPSHOT_CSV))
    parser.add_argument("--live", default=str(LIVE_PREDICTIONS_CSV))
    parser.add_argument("--out", default=str(OUT_CSV))
    parser.add_argument("--meeting-form-snapshot", default=str(MEETING_FORM_SNAPSHOT_CSV))
    parser.add_argument("--meeting-form-cutoff-date", default=MEETING_FORM_CUTOFF_DATE, help="YYYYMMDD")
    parser.add_argument(
        "--apply-meeting-form-layer", action="store_true",
        help="2026-07-31以前のデフォルト挙動に戻し、9会場のmeeting_form 2段目レイヤーを"
             "適用する。widehold再学習後は不要(会場ごとの構成をwidehold側が既に反映済み)"
             "なので、通常は指定しないこと。",
    )
    parser.add_argument(
        "--skip-meeting-form-layer", action="store_true",
        help="(後方互換のため残置、2026-07-31以降は何もしない。既定でスキップになった)",
    )
    args = parser.parse_args()

    cutoff = str(args.cutoff_date)
    snapshot_path = Path(args.snapshot)
    live_path = Path(args.live)
    out_path = Path(args.out)
    meeting_form_snapshot_path = Path(args.meeting_form_snapshot)
    meeting_form_cutoff = str(args.meeting_form_cutoff_date)

    frames = []

    if snapshot_path.exists():
        snap = pd.read_csv(snapshot_path, low_memory=False)
        snap["date"] = _normalize_date(snap["date"])
        snap = snap[snap["date"] < cutoff].copy()
        print(f"snapshot rows (date < {cutoff}): {len(snap)}")
        frames.append(snap)
    else:
        print(f"[WARN] snapshot not found: {snapshot_path}")

    if live_path.exists():
        live = pd.read_csv(live_path, low_memory=False)
        live["date"] = _normalize_date(live["date"])
        live = live[live["date"] >= cutoff].copy()
        print(f"live rows (date >= {cutoff}): {len(live)}")
        frames.append(live)
    else:
        print(f"[WARN] live predictions not found: {live_path}")

    if not frames:
        raise RuntimeError("合成対象データが0件です(snapshot/liveどちらも見つかりません)")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if args.apply_meeting_form_layer:
        if meeting_form_snapshot_path.exists():
            mf_venues = [normalize_venue_name(v) for v in MEETING_FORM_VENUES]
            mf_snap = pd.read_csv(meeting_form_snapshot_path, low_memory=False)
            mf_snap["date"] = _normalize_date(mf_snap["date"])
            mf_snap["venue"] = mf_snap["venue"].astype(str).map(normalize_venue_name)
            mf_snap = mf_snap[mf_snap["date"] < meeting_form_cutoff].copy()
            print(f"meeting_form snapshot rows (venue in {mf_venues}, date < {meeting_form_cutoff}): {len(mf_snap)}")

            # 土台側(snapshot+live)から、この9会場×このカットオフより前の行を除去してから、
            # meeting_form版のholdoutスナップショットに置き換える。
            combined["venue"] = combined["venue"].astype(str).map(normalize_venue_name)
            before_rows = len(combined)
            remove_mask = combined["venue"].isin(mf_venues) & (combined["date"] < meeting_form_cutoff)
            print(f"removing {int(remove_mask.sum())} base rows for meeting_form venues before {meeting_form_cutoff}")
            combined = combined[~remove_mask].copy()
            combined = pd.concat([combined, mf_snap], ignore_index=True, sort=False)
            print(f"combined rows: {before_rows} -> {len(combined)} (after meeting_form layer)")
        else:
            print(f"[WARN] meeting_form snapshot not found: {meeting_form_snapshot_path}(9会場のholdout層はスキップされます)")

    combined = combined.sort_values(["date", "venue", "race_no", "rank_prob"]).reset_index(drop=True)

    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("saved:", out_path)
    print("rows :", len(combined))
    print("races:", combined[["date", "venue", "race_no"]].drop_duplicates().shape[0])
    print("date range:", str(combined["date"].min()), "〜", str(combined["date"].max()))


if __name__ == "__main__":
    main()
