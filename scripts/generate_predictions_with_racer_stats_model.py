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
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402
from scripts.train_binary_catboost_per_venue_with_racer_stats import (  # noqa: E402
    SPLIT_BY_VENUE_DIR,
    _add_feature_block,
    _add_racer_stats_block,
    _load_racer_stats,
)
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


def _compute_holdout_dates(df: pd.DataFrame, test_size: float) -> set:
    """
    scripts/train_binary_catboost_per_venue_with_racer_stats.py の
    _split_by_date と全く同じロジック(日付でソートし、末尾test_size割合を
    holdoutとする)で、その会場のデータセットから「モデルが学習時に一度も
    見ていないはずの日付集合」を再現する。

    これを使わずに predictions.csv にある全レースを対象にしてしまうと、
    学習に使った期間まで評価に含めてしまい(=モデルが答えを記憶している
    データを採点することになり)、ROIが実力より大幅に良く出る事故につながる
    (2026-07-19に実際に発生し、桐生217%・戸田229%等の異常値が出たため
    このガードを追加した)。
    """
    dates = sorted(df["date"].astype(str).str.zfill(8).unique())
    n = len(dates)
    if n < 5:
        return set()
    cutoff_idx = max(1, int(n * (1.0 - test_size)))
    return set(dates[cutoff_idx:])


def generate_for_venue(
    venue: str,
    temperature: float,
    min_races: int = 5,
    model_suffix: str = "",
    valid_dates: set | None = None,
    out_suffix: str = "",
    test_size: float = 0.20,
    full_range: bool = False,
) -> None:
    """
    model_suffix: 通常は""(本番の_with_racer_statsモデル)。
      2026-07-16追加: engine.kelly_allocator...ではなく、選手勝率などの
      特徴量重みを変えた実験モデル(例: "_w4x")を評価したい場合に指定する
      (scripts/sweep_racer_stats_weight.py から使う)。
    valid_dates: 指定した場合、その日付集合に含まれるレースだけを対象にする。
      Noneの場合、full_range=Falseなら自動的にそのモデルの日付分割holdout
      期間(学習に使っていない末尾test_size割合の日付)だけに絞る(デフォルト、
      2026-07-19に安全側へ変更)。full_range=Trueを明示した場合のみ、
      predictions.csvにある全期間を対象にする(学習データを含む可能性があり、
      ROI検証としては無効。オフラインの動作確認等の用途のみに使うこと)。
    out_suffix: 出力ファイル名に追加するサフィックス(重み実験の結果を
      本番用の比較ファイルと混同しないようにするため)。
    """
    venue = normalize_venue_name(venue)
    print("\n" + "=" * 80)
    print("GENERATE PREDICTIONS:", venue, f"(model_suffix={model_suffix!r})" if model_suffix else "")
    print("=" * 80)

    out_path = PROJECT_ROOT / "data" / "logs" / f"predictions_with_racer_stats_{venue}{out_suffix}.csv"

    def _skip(reason: str) -> None:
        # 2026-07-19追加: 以前のファイル(全期間版など)がここに残っていると、
        # 今回スキップしたにもかかわらず古い(=リーク混入の可能性がある)
        # ファイルが「有効な検証結果」として誤って使われ続けてしまう事故が
        # 実際に発生した(常滑で発覚)。スキップ時は必ず古い出力を削除する。
        print(f"[SKIP] {venue}: {reason}")
        if out_path.exists():
            out_path.unlink()
            print(f"[SKIP] {venue}: 古い出力ファイルを削除しました -> {out_path}")

    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_with_racer_stats{model_suffix}.cbm"
    if not model_path.exists():
        _skip(f"model not found: {model_path}")
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

    df = _load_venue_dataset(venue)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)

    if valid_dates is None and not full_range:
        valid_dates = _compute_holdout_dates(df, test_size)
        print(f"[{venue}] valid_dates未指定 -> 自動でholdout期間に限定します "
              f"({len(valid_dates)}日, {min(valid_dates) if valid_dates else '-'}〜"
              f"{max(valid_dates) if valid_dates else '-'})")

    if valid_dates is not None:
        target_races = target_races[target_races["date"].isin(valid_dates)].copy()
        target_pred = target_pred.merge(target_races, on=["date", "race_no"], how="inner")

    print("target races (real odds available, holdout範囲適用後):", len(target_races))

    if len(target_races) < min_races:
        _skip(f"実オッズ付きholdoutレースが{min_races}件未満のためバックテスト対象外")
        return

    df = df.merge(target_races, on=["date", "race_no"], how="inner")

    if df.empty:
        _skip("学習データセットとpredictions.csvが重ならない")
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
    parser.add_argument(
        "--test_size", type=float, default=0.20,
        help="学習時に指定したtest_sizeと必ず合わせること(train_binary_catboost_per_venue_with_racer_stats.pyのデフォルトは0.20)",
    )
    parser.add_argument(
        "--full_range", action="store_true",
        help="危険: holdout期間に絞らず、predictions.csvにある全期間を対象にする"
             "(学習データを含みROI検証としては無効。オフライン確認用途のみ)",
    )
    args = parser.parse_args()

    if args.all:
        venues = list(VENUE_ORDER)
    elif args.venue:
        venues = [args.venue]
    else:
        raise SystemExit("--venue か --all を指定してください")

    for v in venues:
        try:
            generate_for_venue(
                v,
                temperature=args.temperature,
                min_races=args.min_races,
                test_size=args.test_size,
                full_range=args.full_range,
            )
        except Exception as e:
            print(f"[ERROR] venue={v}: {e}")


if __name__ == "__main__":
    main()
