# scripts/analyze_bet_factors_ols.py
# -*- coding: utf-8 -*-
"""
「勝てるレースとダメなレースの傾向」を、会場・レース番号・オッズ・コース・
曜日など複数の要因を同時に投入した重回帰(OLS, HC1頑健標準誤差)で分析する。

これまでの分析(会場別/オッズ帯別/曜日別など)は1要因ずつの単純集計だった
ため、要因同士の交絡(例: 特定の曜日にたまたま調子の良い会場のレースが
多かっただけ、等)を区別できなかった。この重回帰は全要因を同時に投入する
ことで、他の要因を固定した上での「純粋な」効果と95%信頼区間を出す。

対象データ: data/logs/prediction_results_sim_1y_light.csv
  (is_selected==1の実際の購入ログのみ。generate/backfillスクリプト等で
   継続的に増える想定なので、このスクリプトは都度再実行してよい)

目的変数: profit_yen (1点=100円賭けた場合の損益)。払戻が0か大きいかの
  二極構造で分散が非常に大きい(std >> mean)ため、決定係数は低くなりがちで、
  「効果が無い」ように見えても実際にはサンプル不足で検出力が無いだけの
  ケースが多い点に注意。有意でない = 効果が無い、と早合点しないこと。

statsmodels/scikit-learnがサンドボックスに無い(pipインストール不可)ため、
numpyのみで手実装している。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIGHT_CSV = PROJECT_ROOT / "data" / "logs" / "prediction_results_sim_1y_light.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(LIGHT_CSV, low_memory=False)
    df["date"] = df["date"].astype(str)
    df["dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["weekday"] = df["dt"].dt.dayofweek
    df["pred_lane"] = df["combo"].astype(str).str.split("-").str[0].astype(int)
    df["log_odds"] = np.log(df["odds"].clip(lower=0.1))
    return df


def build_design_matrix(df: pd.DataFrame):
    X_parts = [np.ones((len(df), 1))]
    col_names = ["intercept"]

    for col in ["log_odds", "prob", "prob_gap"]:
        X_parts.append(df[col].values.reshape(-1, 1).astype(float))
        col_names.append(col)

    cat_cols = {
        "race_no": df["race_no"].astype(int),
        "pred_lane": df["pred_lane"],
        "weekday": df["weekday"],
        "venue": df["venue"],
    }
    for name, series in cat_cols.items():
        dummies = pd.get_dummies(series, prefix=name, drop_first=True)
        X_parts.append(dummies.values.astype(float))
        col_names.extend(dummies.columns.tolist())

    X = np.hstack(X_parts)
    return X, col_names


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
    tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se != 0)
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    result = pd.DataFrame(
        {"coef": beta, "se": se, "t": tstat, "ci_low": ci_low, "ci_high": ci_high},
        index=col_names,
    )
    result["significant"] = (result["ci_low"] > 0) | (result["ci_high"] < 0)

    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    result.attrs["r2"] = r2
    result.attrs["n"] = n
    result.attrs["k"] = k
    return result


def main() -> None:
    df = load_data()
    y = df["profit_yen"].values.astype(float)
    X, col_names = build_design_matrix(df)
    result = fit_ols_hc1(X, y, col_names)

    pd.set_option("display.width", 140)
    print("n =", result.attrs["n"], " k =", result.attrs["k"], " R^2 =", round(result.attrs["r2"], 4))
    print("y mean =", round(float(y.mean()), 2), " y std =", round(float(y.std()), 2),
          "(標準偏差が平均に比べて非常に大きい=1点ずつのブレが大きい宝くじ的構造)")
    print()
    print("=== 統計的に有意な係数(95%信頼区間が0を跨がない) ===")
    sig = result[result["significant"]].sort_values("coef", ascending=False)
    if sig.empty:
        print("(有意な係数なし)")
    else:
        print(sig[["coef", "se", "ci_low", "ci_high"]].round(2).to_string())
    print()
    print("=== 全係数(参考) ===")
    print(result[["coef", "se", "ci_low", "ci_high", "significant"]].round(2).to_string())


if __name__ == "__main__":
    main()
