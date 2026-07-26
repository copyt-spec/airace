# scripts/build_meeting_form_predictions_snapshot.py
# -*- coding: utf-8 -*-
"""
scripts/build_new_model_predictions_snapshot.py と同じ考え方で、
「今節成績特徴量」を選択導入した9会場(戸田・江戸川・丸亀・びわこ・若松・下関・
芦屋・鳴門・浜名湖、2026-07-26デプロイ)だけを対象にした/sim用の合成スナップショットを作る。

この9会場は全会場一律デプロイではなく個別選択導入なので、既存の
build_new_model_predictions_snapshot.py(全会場一律の展開特徴量ロールアウト、
2026-07-22デプロイ)とは別ファイルとして分離し、
scripts/build_combined_predictions_for_sim.py 側で
「9会場だけ、このスナップショット + デプロイ日以降の本物ログ」で上書きする
2段目のレイヤーとして使う。

前提: 事前に
  python scripts/generate_predictions_with_racer_stats_model.py --all --model_suffix "_meetingform"
を実行し、data/logs/predictions_with_racer_stats_{venue}_meetingform.csv が
9会場分そろっていること(2026-07-26時点で既に生成・検証済み)。

出力: data/logs/predictions_meetingform_snapshot.csv
  (predictions.csvと同じ列: logged_at,date,venue,race_no,combo,rank_prob,
   prob,prob_pct,odds,expected_return_yen,is_selected,model_name)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402
from engine.venue_registry import normalize_venue_name  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
OUT_CSV = LOG_DIR / "predictions_meetingform_snapshot.csv"
MODEL_NAME = "binary_catboost_venue_meetingform_snapshot"

# 2026-07-26: /simの選択導入対象9会場(scripts/compare_meeting_form_roi.pyで
# holdout ROIがベースラインを上回った会場のみ)。全会場一律ではない点に注意。
MEETING_FORM_VENUES = [
    "戸田", "江戸川", "丸亀", "びわこ", "若松", "下関", "芦屋", "鳴門", "浜名湖",
]

FIELDNAMES = [
    "logged_at", "date", "venue", "race_no", "combo",
    "rank_prob", "prob", "prob_pct", "odds",
    "expected_return_yen", "is_selected", "model_name",
]


def _load_venue_predictions(venue: str) -> pd.DataFrame:
    path = LOG_DIR / f"predictions_with_racer_stats_{venue}_meetingform.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df
    df["venue"] = venue
    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)
    df["combo"] = df["combo"].astype(str).str.strip()
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)
    df["odds"] = pd.to_numeric(df.get("odds", 0.0), errors="coerce").fillna(0.0)
    return df


def _build_race_rows(date: str, venue: str, race_no: int, race_df: pd.DataFrame, logged_at: str) -> List[Dict[str, Any]]:
    if len(race_df) != 120:
        return []

    ranked = race_df.sort_values("prob", ascending=False).reset_index(drop=True)

    ai_preds = [
        {"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])}
        for _, r in ranked.iterrows()
    ]
    try:
        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)
    except Exception:
        best_bets = []
    selected_combos = {b["combo"] for b in best_bets}

    rows: List[Dict[str, Any]] = []
    for rank, r in enumerate(ranked.itertuples(index=False), start=1):
        combo = getattr(r, "combo")
        prob = float(getattr(r, "prob"))
        odds = float(getattr(r, "odds"))
        expected_return_yen = round(prob * odds * 100.0, 4) if odds > 0 else 0.0
        rows.append({
            "logged_at": logged_at,
            "date": date,
            "venue": venue,
            "race_no": race_no,
            "combo": combo,
            "rank_prob": rank,
            "prob": round(prob, 8),
            "prob_pct": round(prob * 100.0, 4),
            "odds": round(odds, 4),
            "expected_return_yen": expected_return_yen,
            "is_selected": 1 if combo in selected_combos else 0,
            "model_name": MODEL_NAME,
        })
    return rows


def main() -> None:
    logged_at = datetime.now().isoformat(timespec="seconds")
    all_rows: List[Dict[str, Any]] = []
    total_races = 0

    for venue in MEETING_FORM_VENUES:
        venue = normalize_venue_name(venue)
        df = _load_venue_predictions(venue)
        if df.empty:
            print(f"[SKIP] {venue}: predictions_with_racer_stats_{venue}_meetingform.csv が無いか空です")
            continue

        n_races = 0
        for (date, race_no), race_df in df.groupby(["date", "race_no"], sort=False):
            rows = _build_race_rows(str(date), venue, int(race_no), race_df, logged_at)
            if rows:
                all_rows.extend(rows)
                n_races += 1
        total_races += n_races
        print(f"[OK] {venue}: {n_races}レース")

    if not all_rows:
        raise RuntimeError(
            "合成対象データが0件でした。先に "
            '`python scripts/generate_predictions_with_racer_stats_model.py --all --model_suffix "_meetingform"` '
            "を実行してpredictions_with_racer_stats_{venue}_meetingform.csvを最新化してください。"
        )

    out_df = pd.DataFrame(all_rows, columns=FIELDNAMES)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("saved:", OUT_CSV)
    print("rows       :", len(out_df))
    print("races      :", total_races)
    print("selected   :", int(out_df["is_selected"].sum()))
    print(
        "date range :",
        str(out_df["date"].min()), "〜", str(out_df["date"].max()),
    )


if __name__ == "__main__":
    main()
