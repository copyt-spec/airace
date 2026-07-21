# scripts/analyze_racer_venue_affinity.py
# -*- coding: utf-8 -*-
"""
「選手ごとに得意な会場がある」という仮説を、リーク無し(時系列分割)で検証する。

やり方:
  1. data/datasets/racer_race_history.csv(raw_txt全件から構築した選手×レースの
     実績ロング形式テーブル、build_racer_history.py参照)を学習期間(CUTOFF以前)
     と検証期間(CUTOFF以降、実際に起きた未来のレース)に分割する。
  2. 学習期間だけを使って、選手ごとの「全体複勝率」と「会場別複勝率」を計算し、
     affinity(会場別複勝率 - 全体複勝率)を出す。サンプルが少なすぎる
     選手×会場ペア(MIN_STARTS_VENUE未満)は除外する。
  3. 検証期間の実際の結果(複勝したかどうか)を、学習期間で計算した
     overall_place_rateとaffinityの両方を説明変数にした線形確率モデル
     (OLS, HC1頑健標準誤差)で予測し、affinityの係数が統計的に有意かどうかを見る。
     -> 有意なら「選手の得意会場」は全体の実力とは別に効く本物のシグナル。
        有意でなければ、見かけ上の得意/不得意は再現しない(=ノイズ)。

2026-07時点の結果(CUTOFF=2026-06-01、検証期間2026-06-01〜07-14):
  overall_place_rateの係数は強く有意(0.87、選手の総合力はちゃんと未来を予測する)。
  一方でaffinityの係数は0.019、信頼区間[-0.039, 0.078]で0を含み、有意ではなかった。
  -> 「この選手はこの会場が得意」という過去の複勝率の上振れは、多くの場合
     単なるサンプルノイズであり、将来の成績を追加的に予測する材料にはならない
     、というのが現時点の結論。過信して「得意会場だから買う」という戦略には
     まだ根拠が無い。

このスクリプトはCUTOFFやMIN_STARTS_VENUEを変えて再実行できるので、データが
増えたら(特に検証期間を長くとれるようになったら)再検証すること。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACER_HISTORY_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"

CUTOFF = "20260601"  # これより前を学習期間、以降を検証期間(実際に起きた未来)にする
MIN_STARTS_VENUE = 15
MIN_STARTS_OVERALL = 30


def load_history() -> pd.DataFrame:
    df = pd.read_csv(RACER_HISTORY_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["racer_no"] = pd.to_numeric(df["racer_no"], errors="coerce").fillna(0).astype(int)
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce").fillna(99).astype(int)
    df = df[(df["racer_no"] > 0) & (df["finish"].between(1, 6))].copy()
    df["is_place"] = (df["finish"] <= 3).astype(int)
    df["is_win"] = (df["finish"] == 1).astype(int)
    return df


def compute_venue_affinity(train: pd.DataFrame) -> pd.DataFrame:
    overall = train.groupby("racer_no").agg(
        overall_starts=("is_place", "size"), overall_place_rate=("is_place", "mean")
    )
    venue_stats = train.groupby(["racer_no", "venue"]).agg(
        venue_starts=("is_place", "size"), venue_place_rate=("is_place", "mean")
    ).reset_index()
    venue_stats = venue_stats.merge(overall, on="racer_no", how="left")
    venue_stats = venue_stats[
        (venue_stats["venue_starts"] >= MIN_STARTS_VENUE)
        & (venue_stats["overall_starts"] >= MIN_STARTS_OVERALL)
    ].copy()
    venue_stats["affinity"] = venue_stats["venue_place_rate"] - venue_stats["overall_place_rate"]
    return venue_stats


def fit_ols_hc1(X: np.ndarray, y: np.ndarray, col_names: list) -> pd.DataFrame:
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    u2 = resid ** 2
    XtSX = (X * u2[:, None]).T @ X
    V = XtX_inv @ XtSX @ XtX_inv * (n / max(n - k, 1))
    se = np.sqrt(np.diag(V))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    result = pd.DataFrame({"coef": beta, "se": se, "ci_low": ci_low, "ci_high": ci_high}, index=col_names)
    result["significant"] = (result["ci_low"] > 0) | (result["ci_high"] < 0)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    result.attrs["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    result.attrs["n"] = n
    return result


def main() -> None:
    df = load_history()
    train = df[df["date"] < CUTOFF].copy()
    test = df[df["date"] >= CUTOFF].copy()
    print(f"学習期間: {train['date'].min()} 〜 {train['date'].max()} ({len(train)}行)")
    print(f"検証期間: {test['date'].min()} 〜 {test['date'].max()} ({len(test)}行)  <- 実際に起きた未来")
    print()

    venue_stats = compute_venue_affinity(train)
    print(f"十分なサンプル(会場>={MIN_STARTS_VENUE}走, 全体>={MIN_STARTS_OVERALL}走)がある選手x会場ペア: {len(venue_stats)}")
    print()
    print("=== affinity上位(過去データ上「得意」に見える選手x会場) ===")
    print(venue_stats.sort_values("affinity", ascending=False).head(10)
          [["racer_no", "venue", "venue_starts", "venue_place_rate", "overall_place_rate", "affinity"]]
          .round(3).to_string(index=False))

    merged = test.merge(
        venue_stats[["racer_no", "venue", "overall_place_rate", "affinity", "venue_starts"]],
        on=["racer_no", "venue"], how="inner",
    )
    print()
    print(f"検証期間のうち、学習期間のaffinityが計算できたレコード: {len(merged)}/{len(test)}")

    merged["affinity_bucket"] = pd.cut(
        merged["affinity"], bins=[-1, -0.1, -0.02, 0.02, 0.1, 1],
        labels=["苦手(-10pt以下)", "やや苦手", "中立", "やや得意", "得意(+10pt以上)"],
    )
    print()
    print("=== affinityバケット別、検証期間の実際の複勝率 ===")
    g = merged.groupby("affinity_bucket", observed=True).agg(
        n=("is_place", "size"), actual_place_rate=("is_place", "mean"),
        avg_overall_place_rate=("overall_place_rate", "mean"),
    )
    print(g.round(3))

    y = merged["is_place"].values.astype(float)
    X = np.column_stack([
        np.ones(len(merged)),
        merged["overall_place_rate"].values.astype(float),
        merged["affinity"].values.astype(float),
    ])
    result = fit_ols_hc1(X, y, ["intercept", "overall_place_rate", "venue_affinity"])
    print()
    print(f"=== 重回帰(n={result.attrs['n']}, R^2={result.attrs['r2']:.4f}) ===")
    print("目的変数: 検証期間の実際の複勝(0/1)。説明変数はどちらも学習期間だけで計算。")
    print(result.round(4))
    print()
    if result.loc["venue_affinity", "significant"]:
        print("-> venue_affinityは統計的に有意。選手の得意会場は本物のシグナルの可能性がある。")
    else:
        print("-> venue_affinityは統計的に有意でない。過去の得意/不得意は将来を追加的に予測しない"
              "(=多くはサンプルノイズ)。現時点では『得意会場だから買う』根拠は無い。")


if __name__ == "__main__":
    main()
