# scripts/analyze_race_development_patterns.py
# -*- coding: utf-8 -*-
"""
「展開予想」(engine/race_scenario.py)は現状、3連単の予測確率(prob_map)の
最上位組み合わせを起点に、そこに至るもっともらしい筋書き(スタート→第一ターン
→ホームストレッチ→ゴール)を"逆算"で作っているだけで、独立に着順を予測する
仕組みにはなっていない(パターン判定=1着コースが1なら「逃げ」、2なら「差し」…
という単純なラベル付けのみ)。

このスクリプトは、その逆――「展開(コース取り・ST・展示タイムから見た攻め手)」
から実際に着順を予測できるか、リーク無し(時系列分割)で検証する。

具体的には、以下の競艇セオリーをテストする:
  - 「まくり成立」条件: 3コース以上(アウト側)の艇が、そのレースで展示タイムが
    一番速い(exhibit_rank==1)とき、実際に勝ちやすいか。
  - 「1コース逃げ」条件: 1コースの艇のSTが、そのレースの中で相対的に良い
    (st_rank上位)とき、実際に勝ちやすいか。
  - 全体的に、レース内の展示タイム順位(exhibit_rank)・ST順位(st_rank)が、
    選手自身の通算能力(win_rate_prior、リーク無しの時系列選手成績)を
    差し引いてもなお勝敗を追加的に予測するか。

選手の通算能力を必ず説明変数に含めることで、「単に強い選手が展示も速い」
という交絡(だから見かけ上『展示が速い艇が勝つ』ように見えるだけ)を
区別できるようにしている。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIST_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"
RACER_STATS_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_point_in_time_stats.csv"

CUTOFF = "20260601"
MIN_PRIOR_STARTS = 30


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


def load_data() -> pd.DataFrame:
    hist = pd.read_csv(HIST_CSV, low_memory=False)
    hist["date"] = hist["date"].astype(str).str.zfill(8)
    hist["racer_no"] = pd.to_numeric(hist["racer_no"], errors="coerce").fillna(0).astype(int)
    hist["finish"] = pd.to_numeric(hist["finish"], errors="coerce")
    hist["course"] = pd.to_numeric(hist["course"], errors="coerce")
    hist["st"] = pd.to_numeric(hist["st"], errors="coerce")
    hist["exhibit"] = pd.to_numeric(hist["exhibit"], errors="coerce")
    hist = hist[
        (hist["racer_no"] > 0)
        & hist["finish"].between(1, 6)
        & hist["course"].between(1, 6)
        & hist["exhibit"].between(5.5, 8.5)  # 明らかな異常値(未計測0埋め等)を除外
    ].copy()
    hist["is_win"] = (hist["finish"] == 1).astype(int)
    hist["is_place"] = (hist["finish"] <= 3).astype(int)

    key = ["date", "venue", "race_no"]
    hist["exhibit_rank"] = hist.groupby(key)["exhibit"].rank(method="min", ascending=True)
    hist["st_rank"] = hist.groupby(key)["st"].rank(method="min", ascending=True, na_option="bottom")

    stats = pd.read_csv(RACER_STATS_CSV, low_memory=False)
    stats["date"] = stats["date"].astype(str).str.zfill(8)
    stats["racer_no"] = pd.to_numeric(stats["racer_no"], errors="coerce").fillna(0).astype(int)
    stats = stats[["racer_no", "date", "races_prior", "win_rate_prior"]]

    df = hist.merge(stats, on=["racer_no", "date"], how="left")
    df = df[df["races_prior"].fillna(0) >= MIN_PRIOR_STARTS].copy()
    df["win_rate_prior"] = df["win_rate_prior"].fillna(0.0)
    return df


def report_ols(test: pd.DataFrame, extra_col: str, label: str) -> None:
    valid = test.dropna(subset=["win_rate_prior", extra_col, "is_win"])
    y = valid["is_win"].values.astype(float)
    X = np.column_stack([
        np.ones(len(valid)),
        valid["win_rate_prior"].values.astype(float),
        valid[extra_col].values.astype(float),
    ])
    result = fit_ols_hc1(X, y, ["intercept", "win_rate_prior(通算)", label])
    print(f"\n=== is_win ~ win_rate_prior + {label} (n={result.attrs['n']}, R^2={result.attrs['r2']:.4f}) ===")
    print(result.round(5))
    row = result.loc[label]
    if row["significant"]:
        print(f"-> {label}は統計的に有意。選手の通算能力を差し引いても、追加的に勝敗を予測する。")
    else:
        print(f"-> {label}は有意でない。選手の通算能力で説明できる範囲を超えていない。")


def main() -> None:
    print("loading & preparing data (leak-free join: racer_no+date exact match)...")
    df = load_data()
    train = df[df["date"] < CUTOFF]
    test = df[df["date"] >= CUTOFF]
    print(f"学習期間: {train['date'].min()} 〜 {train['date'].max()} ({len(train)}行)")
    print(f"検証期間: {test['date'].min()} 〜 {test['date'].max()} ({len(test)}行)  <- 実際に起きた未来")

    # --- 1. 展示タイム順位(exhibit_rank)は通算能力を差し引いても勝敗を予測するか ---
    report_ols(test, "exhibit_rank", "exhibit_rank(展示順位、1=最速)")

    # --- 2. STランク(st_rank)は通算能力を差し引いても勝敗を予測するか ---
    report_ols(test, "st_rank", "st_rank(ST順位、1=最良)")

    # --- 3. 「まくり成立」条件: アウト(3コース以上)かつ展示最速 ---
    test = test.copy()
    test["is_outer_fastest_exhibit"] = (
        (test["course"] >= 3) & (test["exhibit_rank"] == 1)
    ).astype(int)
    report_ols(test, "is_outer_fastest_exhibit", "is_outer_fastest_exhibit(3コース以上×展示最速=まくり候補)")

    print("\n--- 参考: is_outer_fastest_exhibit有無別の実際の勝率(通算能力ゾーン別) ---")
    test["ability_bucket"] = pd.cut(
        test["win_rate_prior"], bins=[0, 0.2, 0.35, 0.5, 1.0],
        labels=["低(~20%)", "中(20-35%)", "高(35-50%)", "最上位(50%~)"],
    )
    g = test.groupby(["ability_bucket", "is_outer_fastest_exhibit"], observed=True).agg(
        n=("is_win", "size"), actual_win_rate=("is_win", "mean")
    )
    print(g[g["n"] >= 50].round(3).to_string())

    # --- 4. 「1コース逃げ」条件: 1コースかつSTが好調(そのレースでst_rank<=2) ---
    test["is_course1_good_st"] = ((test["course"] == 1) & (test["st_rank"] <= 2)).astype(int)
    course1_only = test[test["course"] == 1].copy()
    report_ols(course1_only, "st_rank", "st_rank(1コース限定、STが良いほど逃げやすいか)")


if __name__ == "__main__":
    main()
