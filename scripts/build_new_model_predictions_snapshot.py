# scripts/build_new_model_predictions_snapshot.py
# -*- coding: utf-8 -*-
"""
「/simのグラフ(累積損益・日別ROI・会場別ROI・信頼度/オッズ帯別内訳・Kelly資産
推移)を新モデル(展開特徴量追加後)の結果に書き換えたい」という要望に応える
ためのビルダー。

背景: /simのグラフ類は最終的に data/logs/predictions.csv (実際にアプリが
稼働した時点でログされた、その時点のモデルによる予測)を起点に
scripts/merge_prediction_results.py → scripts/build_sim_light_csv.py という
パイプラインで作られる。predictions.csvは長期間・複数のモデルバージョンが
混在した「本物の履歴ログ」なので、ここに新モデルの予測を追記/上書きすると
過去の実履歴を壊してしまう(過去にも「学習データを採点してリークする」
事故が起きているため、ログの真正性は特に慎重に扱うべき箇所)。

そこでこのスクリプトは、predictions.csvには一切触れず、
scripts/generate_predictions_with_racer_stats_model.py --all (suffix無し
=現在の本番モデル)が生成する data/logs/predictions_with_racer_stats_{venue}.csv
(モデル自身の学習時test_sizeに基づく、真のholdout期間だけの実オッズ付き
予測、120通り/レース)を材料に、predictions.csvと同じ列構成の「新モデル版
predictions.csv」を合成する:
  1. 各レースの120通りのprob降順でrank_probを付与
  2. engine.buy_selector.select_best_bets(現在のロジック)をそのレースに
     適用し、実際に選ばれる買い目にis_selected=1を立てる
  3. model_name列に "binary_catboost_venue_devfeatures_snapshot" を入れ、
     どの合成データか後から分かるようにする

出力: data/logs/predictions_new_model_snapshot.csv
  (predictions.csvと同じ列: logged_at,date,venue,race_no,combo,rank_prob,
   prob,prob_pct,odds,expected_return_yen,is_selected,model_name)

このファイルを scripts/merge_prediction_results.py --predictions <このファイル>
に渡すことで、/simの本物の出力パス(prediction_results_merged.csv →
prediction_results_sim_1y_light.csv)を「新モデルの真のholdout期間だけ」の
データで安全に更新できる(predictions.csv自体は無傷のまま)。

前提: 事前に `python scripts/generate_predictions_with_racer_stats_model.py --all`
(suffix無し=本番モデル、デフォルトtest_size=0.20)を実行し、
data/logs/predictions_with_racer_stats_{venue}.csv が最新化されていること。
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
from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
OUT_CSV = LOG_DIR / "predictions_new_model_snapshot.csv"
MODEL_NAME = "binary_catboost_venue_devfeatures_snapshot"

FIELDNAMES = [
    "logged_at", "date", "venue", "race_no", "combo",
    "rank_prob", "prob", "prob_pct", "odds",
    "expected_return_yen", "is_selected", "model_name",
]


def _load_venue_predictions(venue: str) -> pd.DataFrame:
    path = LOG_DIR / f"predictions_with_racer_stats_{venue}.csv"
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

    for venue in VENUE_ORDER:
        venue = normalize_venue_name(venue)
        df = _load_venue_predictions(venue)
        if df.empty:
            print(f"[SKIP] {venue}: predictions_with_racer_stats_{venue}.csv が無いか空です")
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
            "`python scripts/generate_predictions_with_racer_stats_model.py --all` "
            "を実行してpredictions_with_racer_stats_{venue}.csvを最新化してください。"
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
