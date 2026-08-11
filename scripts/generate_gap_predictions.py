# scripts/generate_gap_predictions.py
# -*- coding: utf-8 -*-
"""
新モデルがRenderにデプロイされた直後など、trifecta_train_by_venue/*.csv
(重い・全期間フルスクラッチ再構築が必要)を更新せずに、特定の数日分だけ
「新モデルなら何が出ていたか」を実オッズ付きで再現するための軽量スナップショット
生成スクリプト。

背景: 2026-07-30にdepth=6+特徴量バグ修正版モデルへ切り替え、2026-07-31 21:56に
Renderへ実際にデプロイされた。しかしその前後(7/30・7/31)は実運用ログ
(data/logs/predictions.csv)がまだ旧モデルで記録されているため、/simの
CUTOFF_DATEを2026-08-01に設定しても7/30・7/31は静的スナップショット
(predictions_new_model_snapshot.csv、〜2026-07-29までしかカバーしない)にも
実運用ログにも含まれず、穴になってしまう([[boat_ai_...]]参照)。

このスクリプトは、指定した日付範囲のraw_txt(K-file)だけを軽量にパースし
(6.9GBのtrifecta_train.csv全体は一切読まない)、現行本番モデルで推論して
実オッズ(data/logs/predictions.csvに既にログ済みの本物のオッズ)と組み合わせた
predictions_with_racer_stats_{venue}_gapfill.csv を書き出す。

出力形式は scripts/generate_predictions_with_racer_stats_model.py の出力
(date, venue, race_no, combo, prob, odds)と互換なので、そのまま
scripts/build_new_model_predictions_snapshot.py --pred-suffix _gapfill で
組み立てられる。

使い方:
  python scripts/generate_gap_predictions.py --dates 20260730,20260731

注意: motor_point_in_time_stats.csv / racer_point_in_time_stats.csv が
対象日付をカバーしていない場合、モーター/選手特徴量はNaN(→0埋め)に
フォールバックする(_add_motor_stats_block/_add_racer_stats_blockの既存挙動)。
meeting_form特徴量も同様にNaNフォールバックする(_add_meeting_form_block、
join_meeting_so_far_features.pyは実行しない=軽量優先)。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402
from engine.txt_dataset_builder import build_dataset_from_txt, TxtDatasetConfig  # noqa: E402
from scripts.train_binary_catboost_per_venue_with_racer_stats import (  # noqa: E402
    _add_feature_block,
    _add_racer_stats_block,
    _add_motor_stats_block,
    _add_meeting_form_block,
    _load_racer_stats,
    _load_motor_stats,
)

RAW_TXT_DIR = PROJECT_ROOT / "data" / "raw_txt"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions.csv"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True, help="カンマ区切りYYYYMMDD (例: 20260730,20260731)")
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--out-suffix", default="_gapfill")
    args = parser.parse_args()

    target_dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    print("target dates:", target_dates)

    tmp_dir = Path(tempfile.mkdtemp(prefix="gapfill_"))
    raw_tmp = tmp_dir / "raw_txt"
    raw_tmp.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for d in target_dates:
        # K-fileの命名規則: K{yy}{mm}{dd}.TXT
        yy = d[2:4]
        mm = d[4:6]
        dd = d[6:8]
        fn = f"K{yy}{mm}{dd}.TXT"
        src = RAW_TXT_DIR / fn
        if not src.exists():
            print(f"[WARN] raw_txtに見つかりません: {src} (この日はスキップされます)")
            continue
        shutil.copy(src, raw_tmp / fn)
        n_copied += 1
    if n_copied == 0:
        raise SystemExit("対象日付のK-fileが1つも見つかりませんでした")

    startk_csv = tmp_dir / "startk.csv"
    build_dataset_from_txt(TxtDatasetConfig(
        raw_txt_dir=str(raw_tmp), out_csv_path=str(startk_csv),
        keep_unlabeled=False, verbose=True,
    ))

    trifecta_csv = tmp_dir / "trifecta.csv"
    import subprocess
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_trifecta_training_csv.py"),
         "--in", str(startk_csv), "--out", str(trifecta_csv), "--require-label"],
        check=True,
    )

    df_all = pd.read_csv(trifecta_csv, low_memory=False)
    df_all["date"] = df_all["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df_all["venue"] = df_all["venue"].astype(str).map(normalize_venue_name)
    df_all["race_no"] = pd.to_numeric(df_all["race_no"], errors="coerce").fillna(0).astype(int)

    racer_stats = _load_racer_stats()
    motor_stats = _load_motor_stats()

    odds_src = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    odds_src["date"] = odds_src["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    odds_src["venue"] = odds_src["venue"].astype(str).str.strip()
    odds_src["race_no"] = pd.to_numeric(odds_src["race_no"], errors="coerce").fillna(0).astype(int)
    odds_src["combo"] = odds_src["combo"].astype(str).str.strip()
    odds_src = odds_src[odds_src["date"].isin(target_dates)].copy()

    for venue in VENUE_ORDER:
        venue = normalize_venue_name(venue)
        df = df_all[df_all["venue"] == venue].copy()
        out_path = LOG_DIR / f"predictions_with_racer_stats_{venue}{args.out_suffix}.csv"
        if df.empty:
            print(f"[SKIP] {venue}: 対象日にレース無し")
            if out_path.exists():
                out_path.unlink()
            continue

        model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_with_racer_stats.cbm"
        if not model_path.exists():
            print(f"[SKIP] {venue}: モデルが見つかりません {model_path}")
            continue

        model = CatBoostClassifier()
        model.load_model(str(model_path))

        df = _add_racer_stats_block(df, racer_stats)
        df = _add_motor_stats_block(df, motor_stats)
        df = _add_feature_block(df)
        df = _add_meeting_form_block(df)

        feature_cols = list(model.feature_names_)
        missing = [c for c in feature_cols if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        x = df[feature_cols].copy()
        for col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

        raw_scores = model.predict(x, prediction_type="RawFormulaVal")
        df["raw_score"] = raw_scores

        out_rows = []
        for (date, race_no), g in df.groupby(["date", "race_no"], sort=False):
            if len(g) != 120:
                continue
            probs = _softmax(list(g["raw_score"]), temperature=args.temperature)
            for combo, p in zip(g["combo"], probs):
                out_rows.append({"date": date, "venue": venue, "race_no": race_no, "combo": combo, "prob": p})

        out_df = pd.DataFrame(out_rows)
        if out_df.empty:
            print(f"[SKIP] {venue}: 有効な120通りレースが無し")
            if out_path.exists():
                out_path.unlink()
            continue

        odds_lookup = (
            odds_src[odds_src["venue"] == venue][["date", "race_no", "combo", "odds"]]
            .drop_duplicates(subset=["date", "race_no", "combo"], keep="last")
        )
        out_df = out_df.merge(odds_lookup, on=["date", "race_no", "combo"], how="left")
        out_df["odds"] = pd.to_numeric(out_df["odds"], errors="coerce").fillna(0.0)

        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        n_races = out_df[["date", "race_no"]].drop_duplicates().shape[0]
        print(f"[OK] {venue}: {n_races}レース -> {out_path}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("DONE")


if __name__ == "__main__":
    main()
