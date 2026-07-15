from __future__ import annotations

"""
学習済みの trifecta_binary_catboost_{venue}_with_racer_stats.cbm を使って、
data/logs/predictions.csv に載っている実際の対戦カード(過去に実オッズ付きで
ログされたレース)と同じ (date, venue, race_no) の組み合わせについて、
120通り全部の予測確率を再生成する。

これにより、旧モデル(predictions.csv内の既存prob, with_racer_no)と
新モデル(with_racer_stats)を、同じ実オッズ・同じ実際の結果で
公平にバックテスト比較できる(scripts/backtest_buy_selector.py で使える形式)。

使い方:
  python scripts/generate_predictions_with_racer_stats_model.py --venue 戸田

出力:
  data/logs/predictions_with_racer_stats_{venue}.csv
  (date, venue, race_no, combo, prob, odds 列。oddsはpredictions.csvから転記)
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from engine.venue_registry import VENUE_ORDER, normalize_venue_name
from scripts.train_binary_catboost_per_venue_with_racer_stats import (
    SPLIT_BY_VENUE_DIR,
    _add_feature_block,
    _add_racer_stats_block,
    _load_racer_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "logs" / "predictions.csv"


def _softmax(xs: List[float], temperature: float = 1.0) -> List[float]:
    if not xs:
        return []
    t = max(float(temperature), 1e-6)
    scaled = [x / t for x in xs]
    m = max(scaled)
    exps = [np.exp(x - m) for x in scaled]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def _load_venue_dataset(venue: str) -> pd.DataFrame:
    pre_split_path = SPLIT_BY_VENUE_DIR / f"{venue}.csv"
    if pre_split_path.exists():
        print(f"loading pre-split dataset <- {pre_split_path}")
        df = pd.read_csv(pre_split_path, low_memory=False)
        df["venue"] = df["venue"].astype(str).map(normalize_venue_name)
        return df[df["venue"] == venue].copy()

    print("loading training dataset (chunked, no pre-split found)...")
    chunks = []
    for chunk in pd.read_csv(DATASET_PATH, low_memory=False, chunksize=200_000):
        chunk["venue"] = chunk["venue"].astype(str).map(normalize_venue_name)
        hit = chunk[chunk["venue"] == venue].copy()
        if not hit.empty:
            chunks.append(hit)
    if not chunks:
        raise RuntimeError(f"no data for venue: {venue}")
    return pd.concat(chunks, ignore_index=True)


def generate_for_venue(venue: str, temperature: float, min_races: int = 5) -> None:
    venue = normalize_venue_name(venue)
    print("\n" + "=" * 80)
    print("GENERATE PREDICTIONS:", venue)
    print("=" * 80)

    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_with_racer_stats.cbm"
    if not model_path.exists():
        print(f"[SKIP] model not found: {model_path}")
        return

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    pred = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    pred["date"] = pred["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    pred["venue"] = pred["venue"].astype(str).str.strip()
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()

    target_pred = pred[pred["venue"] == venue].copy()
    target_races = target_pred[["date", "race_no"]].drop_duplicates()
    print("target races (real odds available):", len(target_races))

    if len(target_races) < min_races:
        print(f"[SKIP] {venue}: 実オッズ付きレースが{min_races}件未満のためバックテスト対象外")
        return

    df = _load_venue_dataset(venue)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)
    df = df.merge(target_races, on=["date", "race_no"], how="inner")

    if df.empty:
        print(f"[SKIP] {venue}: 学習データセットとpredictions.csvが重ならない")
        return

    print("matched rows:", len(df), "races:", df[["date", "race_no"]].drop_duplicates().shape[0])

    racer_stats = _load_racer_stats()
    df = _add_racer_stats_block(df, racer_stats)
    df = _add_feature_block(df)

    feature_cols = list(model.feature_names_)
    x = df[feature_cols].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    raw_scores = model.predict(x, prediction_type="RawFormulaVal")
    df["raw_score"] = raw_scores

    out_rows = []
    for (date, race_no), g in df.groupby(["date", "race_no"], sort=False):
        if len(g) != 120:
            continue
        probs = _softmax(list(g["raw_score"]), temperature=temperature)
        for combo, p in zip(g["combo"], probs):
            out_rows.append({"date": date, "venue": venue, "race_no": race_no, "combo": combo, "prob": p})

    out_df = pd.DataFrame(out_rows)

    odds_lookup = (
        target_pred[["date", "race_no", "combo", "odds"]]
        .drop_duplicates(subset=["date", "race_no", "combo"], keep="last")
    )
    out_df = out_df.merge(odds_lookup, on=["date", "race_no", "combo"], how="left")

    out_path = PROJECT_ROOT / "data" / "logs" / f"predictions_with_racer_stats_{venue}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("saved:", out_path)
    print("rows:", len(out_df))
    print("races:", out_df[["date", "race_no"]].drop_duplicates().shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--min_races", type=int, default=5)
    args = parser.parse_args()

    if args.all:
        venues = list(VENUE_ORDER)
    elif args.venue:
        venues = [args.venue]
    else:
        raise SystemExit("--venue か --all を指定してください")

    for v in venues:
        try:
            generate_for_venue(v, temperature=args.temperature, min_races=args.min_races)
        except Exception as e:
            print(f"[ERROR] venue={v}: {e}")


if __name__ == "__main__":
    main()
