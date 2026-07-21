# scripts/analyze_racer_recent_form.py
# -*- coding: utf-8 -*-
"""
「選手の直近の調子(直近N走の複勝率)」が、通算の実力(overall_place_rate)
とは別に、将来の成績を追加的に予測するか(=いわゆる「ホットハンド」効果が
実在するか)を、リーク無しの時系列特徴量で検証する。

やり方:
  1. data/datasets/racer_race_history.csv を選手ごとに日付順に並べる。
  2. 各レースについて、「そのレースより前の全履歴」だけを使った
     expanding_place_rate(通算複勝率、当該レースは含まない)と、
     recent5_place_rate(直近5走の複勝率、当該レースは含まない、shift(1)で
     厳密に未来情報を除外)を計算する。
  3. 検証期間(CUTOFF以降、実際に起きた未来のレース)に絞り、
     is_place ~ expanding_place_rate + recent5_place_rate
     をOLS(HC1頑健標準誤差)で推定する。recent5の係数が有意なら、
     「直近の調子」は通算実力を差し引いてもなお追加の予測力を持つ、
     という意味で本物のシグナル。

2026-07時点の結果(検証期間2026-06-01〜07-14、n=39,550):
  expanding_place_rateの係数0.741(強く有意)に加えて、
  recent5_place_rateの係数も0.136、信頼区間[0.115, 0.157]で
  はっきり有意だった(R^2=0.085、選手x会場のaffinity検証時のR^2=0.065より高い)。
  通算実力ゾーンごとに見ても、どのゾーンでも「直近絶好調」は「直近絶不調」より
  実際の複勝率が20〜28ポイントほど高く、単調な関係がきれいに出ている。
  -> 「直近の調子」は現状のモデル(通算のwin_rate/place_rate/ability_indexが
     中心)にはおそらく無い、追加できる価値のある特徴量。

このスクリプトはCUTOFFやウィンドウ幅(5走)を変えて再実行できる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACER_HISTORY_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"

CUTOFF = "20260601"
RECENT_WINDOW = 5
MIN_PRIOR_STARTS = 30


def load_history_with_rolling() -> pd.DataFrame:
    df = pd.read_csv(RACER_HISTORY_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["racer_no"] = pd.to_numeric(df["racer_no"], errors="coerce").fillna(0).astype(int)
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce").fillna(99).astype(int)
    df = df[(df["racer_no"] > 0) & (df["finish"].between(1, 6))].copy()
    df["is_place"] = (df["finish"] <= 3).astype(int)

    df = df.sort_values(["racer_no", "date", "venue", "race_no"]).reset_index(drop=True)

    g = df.groupby("racer_no")["is_place"]
    # shift(1)で「このレースより前」だけを使う(リーク防止)
    df["expanding_place_rate"] = g.transform(lambda s: s.shift(1).expanding().mean())
    df["recent_place_rate"] = g.transform(lambda s: s.shift(1).rolling(RECENT_WINDOW, min_periods=RECENT_WINDOW).mean())
    df["prior_starts"] = g.transform(lambda s: s.shift(1).expanding().count())
    return df


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
    df = load_history_with_rolling()
    test = df[
        (df["date"] >= CUTOFF)
        & (df["prior_starts"] >= MIN_PRIOR_STARTS)
        & df["recent_place_rate"].notna()
    ].copy()
    print(f"検証期間: {test['date'].min()} 〜 {test['date'].max()} (n={len(test)}, 十分な履歴がある選手のみ)")

    y = test["is_place"].values.astype(float)
    X = np.column_stack([
        np.ones(len(test)),
        test["expanding_place_rate"].values.astype(float),
        test["recent_place_rate"].values.astype(float),
    ])
    result = fit_ols_hc1(X, y, ["intercept", "expanding_place_rate(通算)", f"recent{RECENT_WINDOW}_place_rate(直近{RECENT_WINDOW}走)"])
    print()
    print(f"=== 重回帰(R^2={result.attrs['r2']:.4f}) ===")
    print(result.round(4))
    print()

    test["form_bucket"] = pd.cut(
        test["recent_place_rate"], bins=[-0.01, 0.0, 0.2, 0.4, 0.6, 1.01],
        labels=["絶不調(0/5)", "不調(1/5)", "普通(2/5)", "好調(3/5)", "絶好調(4-5/5)"],
    )
    test["overall_bucket"] = pd.cut(
        test["expanding_place_rate"], bins=[0, 0.3, 0.4, 0.5, 0.6, 1.0],
        labels=["~30%", "30-40%", "40-50%", "50-60%", "60%~"],
    )
    print("=== 通算実力ゾーン x 直近の調子、実際の複勝率(交絡調整版、n>=100のみ) ===")
    g = test.groupby(["overall_bucket", "form_bucket"], observed=True).agg(n=("is_place", "size"), actual=("is_place", "mean"))
    g = g[g["n"] >= 100]
    print(g.round(3).to_string())

    row = result.loc[f"recent{RECENT_WINDOW}_place_rate(直近{RECENT_WINDOW}走)"]
    print()
    if row["significant"]:
        print(f"-> recent{RECENT_WINDOW}_place_rateは統計的に有意(係数{row['coef']:.4f})。"
              "直近の調子は通算実力とは別に将来を予測する本物のシグナル。モデルへの特徴量追加を推奨。")
    else:
        print(f"-> recent{RECENT_WINDOW}_place_rateは有意でない。直近の調子だけでは追加の予測力は無さそう。")


if __name__ == "__main__":
    main()
