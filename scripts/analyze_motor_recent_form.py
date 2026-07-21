# scripts/analyze_motor_recent_form.py
# -*- coding: utf-8 -*-
"""
「モーター相性(直近の調子)」が、選手自身の実力とは別に将来の成績を
追加的に予測するかを、リーク無しの時系列特徴量で検証する。

背景: 競艇のモーターは会場ごとに番号が振られ、既存のracer_no同様の
考え方で(venue, motor)単位の識別子として履歴を追える。ただし物理的な
モーター交換(整備・積み替え)が定期的に起きるため、選手の通算成績のような
「全期間平均」ではなく、直近の使用実績(トレイリングウィンドウ)で
「今のこのモーターの調子」を近似するのが妥当と考え、直近8走の複勝率を使う。

やり方:
  1. data/datasets/racer_race_history.csv を (venue, motor) 単位で日付順に
     並べ、shift(1)で「その使用より前」だけを使った直近8走の複勝率
     (motor_recent8_place_rate)を計算する(リーク防止)。
  2. 別途、選手ごとの通算複勝率(racer_expanding_place_rate、
     analyze_racer_recent_form.pyと同じ考え方)も計算する。
  3. 検証期間(2026-06-01〜07-14、実際に起きた未来)で
     is_place ~ racer_expanding_place_rate + motor_recent8_place_rate
     をOLS(HC1)で推定し、motor_recent8_place_rateの係数が選手の実力を
     差し引いてもなお有意かどうかを見る。

2026-07時点の結果(n=39,534):
  racer_expanding_place_rateの係数0.811(強く有意)に加え、
  motor_recent8_place_rateの係数も0.120、有意(信頼区間が0を含まない)。
  R^2=0.084で、選手の直近の調子(analyze_racer_recent_form.py, R^2=0.085)と
  ほぼ同水準。モーター調子バケット別の実際の複勝率も単調
  (絶不調40.0% -> 絶好調59.5%)。
  -> 「モーターの直近の調子を見る」という、競艇でよく言われる経験則
     (2連率チェック)は、このデータでも裏付けられた。選手の直近の調子と
     並んで、モデルに追加する価値のある特徴量候補。

注意: 物理的なモーター交換のタイミングを厳密には特定していないため、
直近8走の中に交換をまたぐケースが混ざっている可能性がある(その場合は
むしろ効果を過小評価する方向に働くはずで、それでも有意という点は
シグナルの強さを裏付ける)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACER_HISTORY_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"

CUTOFF = "20260601"
MOTOR_WINDOW = 8
MIN_RACER_PRIOR_STARTS = 30
MIN_MOTOR_PRIOR_USES = 8


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RACER_HISTORY_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["racer_no"] = pd.to_numeric(df["racer_no"], errors="coerce").fillna(0).astype(int)
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce").fillna(99).astype(int)
    df["motor"] = pd.to_numeric(df["motor"], errors="coerce")
    df = df[(df["racer_no"] > 0) & (df["finish"].between(1, 6)) & df["motor"].notna()].copy()
    df["motor"] = df["motor"].astype(int)
    df["is_place"] = (df["finish"] <= 3).astype(int)

    df = df.sort_values(["racer_no", "date", "venue", "race_no"]).reset_index(drop=True)
    gr = df.groupby("racer_no")["is_place"]
    df["racer_expanding_place_rate"] = gr.transform(lambda s: s.shift(1).expanding().mean())
    df["racer_prior_starts"] = gr.transform(lambda s: s.shift(1).expanding().count())

    df = df.sort_values(["venue", "motor", "date", "race_no"]).reset_index(drop=True)
    gm = df.groupby(["venue", "motor"])["is_place"]
    df["motor_recent_place_rate"] = gm.transform(lambda s: s.shift(1).rolling(MOTOR_WINDOW, min_periods=MOTOR_WINDOW).mean())
    df["motor_prior_uses"] = gm.transform(lambda s: s.shift(1).expanding().count())
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
    df = load_data()
    test = df[
        (df["date"] >= CUTOFF)
        & (df["racer_prior_starts"] >= MIN_RACER_PRIOR_STARTS)
        & (df["motor_prior_uses"] >= MIN_MOTOR_PRIOR_USES)
        & df["motor_recent_place_rate"].notna()
    ].copy()
    print(f"検証期間: {test['date'].min()} 〜 {test['date'].max()} (n={len(test)})")

    y = test["is_place"].values.astype(float)
    X = np.column_stack([
        np.ones(len(test)),
        test["racer_expanding_place_rate"].values.astype(float),
        test["motor_recent_place_rate"].values.astype(float),
    ])
    result = fit_ols_hc1(X, y, ["intercept", "racer_expanding_place_rate(選手通算)", f"motor_recent{MOTOR_WINDOW}_place_rate(モーター直近{MOTOR_WINDOW}走)"])
    print()
    print(f"=== 重回帰(R^2={result.attrs['r2']:.4f}) ===")
    print(result.round(4))

    test["motor_bucket"] = pd.cut(
        test["motor_recent_place_rate"], bins=[-0.01, 0.25, 0.375, 0.5, 0.625, 1.01],
        labels=["絶不調(~2/8)", "不調(3/8)", "普通(4/8)", "好調(5/8)", "絶好調(6-8/8)"],
    )
    print()
    print("=== モーター調子バケット別、実際の複勝率 ===")
    g = test.groupby("motor_bucket", observed=True).agg(n=("is_place", "size"), actual=("is_place", "mean"))
    print(g.round(3))

    row = result.loc[f"motor_recent{MOTOR_WINDOW}_place_rate(モーター直近{MOTOR_WINDOW}走)"]
    print()
    if row["significant"]:
        print(f"-> motor_recent{MOTOR_WINDOW}_place_rateは統計的に有意(係数{row['coef']:.4f})。"
              "選手の実力とは別に、モーターの直近の調子は将来を予測する本物のシグナル。")
    else:
        print(f"-> motor_recent{MOTOR_WINDOW}_place_rateは有意でない。")


if __name__ == "__main__":
    main()
