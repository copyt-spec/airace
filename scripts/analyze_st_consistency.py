from __future__ import annotations

"""
「ST精度・安定性」を深掘りする第一歩: 現行モデルは選手の平均ST(avg_st_prior)
しか特徴量化していない。ST の"ばらつき"(標準偏差=安定性)が、平均STや
勝率とは独立に着順を予測するか、リーク無し(その日より前の実績のみ)の
OLSで検証する。

方法: build_racer_point_in_time_stats.py と全く同じ「日付粒度カットオフ・
累積-当日=当日より前」の方式でstd_st_priorを追加計算し、
avg_st_prior/win_rate_prior/コース固定効果を統制したOLSで
is_top3 ~ win_rate_prior + avg_st_prior + std_st_prior + course_dummies
を実行する。statsmodels/scipy/sklearnがサンドボックスに無いため、
numpyのlstsqで手動OLS(係数・SE・95%CI)を計算する。

結果(2026-07-29、n=1,050,137): std_st_prior係数-0.907 [95%CI -1.016,-0.798]、
t=-16.29で強く有意。win_rate_prior・avg_st_prior・course固定効果を統制しても
残る独立した信号と確認できたため、build_racer_point_in_time_stats.py に
std_st_prior列を追加し、学習・ライブ推論に配線した(デフォルトOFF、
--enable_st_std_feature で有効化。ローカル再学習+ROIホールドアウト検証待ち)。
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"

MIN_RACES_PRIOR = 20


def manual_ols(X: np.ndarray, y: np.ndarray, col_names: list[str]) -> None:
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stat = beta / se
    # 大サンプルなのでt分布≒正規分布として1.96を使う
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se

    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())

    print(f"n={n}, R^2={r2:.5f}")
    print(f"{'term':30s} {'coef':>12s} {'se':>12s} {'t':>10s} {'ci_lo':>12s} {'ci_hi':>12s}")
    for name, b, s, t, lo, hi in zip(col_names, beta, se, t_stat, ci_lo, ci_hi):
        print(f"{name:30s} {b:12.6f} {s:12.6f} {t:10.2f} {lo:12.6f} {hi:12.6f}")


def main() -> None:
    print("loading:", HISTORY_CSV)
    df = pd.read_csv(HISTORY_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["racer_no"] = pd.to_numeric(df["racer_no"], errors="coerce").fillna(0).astype(int)
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce").fillna(99).astype(int)
    df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0).astype(int)
    df["st"] = pd.to_numeric(df["st"], errors="coerce")

    df["is_win"] = (df["finish"] == 1).astype(int)
    df["is_top3"] = (df["finish"] <= 3).astype(int)

    # ---- build_racer_point_in_time_stats.py と同じ日付粒度カットオフ方式 ----
    df["st_sq"] = df["st"] ** 2

    daily = (
        df.groupby(["racer_no", "date"], as_index=False)
        .agg(
            races=("finish", "count"),
            wins=("is_win", "sum"),
            top3=("is_top3", "sum"),
            st_sum=("st", "sum"),
            st_sq_sum=("st_sq", "sum"),
            st_count=("st", "count"),
        )
    )
    daily = daily.sort_values(["racer_no", "date"]).reset_index(drop=True)
    g = daily.groupby("racer_no")

    cum_races = g["races"].cumsum()
    cum_wins = g["wins"].cumsum()
    cum_top3 = g["top3"].cumsum()
    cum_st_sum = g["st_sum"].cumsum()
    cum_st_sq_sum = g["st_sq_sum"].cumsum()
    cum_st_count = g["st_count"].cumsum()

    daily["races_prior"] = cum_races - daily["races"]
    daily["wins_prior"] = cum_wins - daily["wins"]
    daily["top3_prior"] = cum_top3 - daily["top3"]
    daily["st_sum_prior"] = cum_st_sum - daily["st_sum"]
    daily["st_sq_sum_prior"] = cum_st_sq_sum - daily["st_sq_sum"]
    daily["st_count_prior"] = cum_st_count - daily["st_count"]

    daily["win_rate_prior"] = np.where(daily["races_prior"] > 0, daily["wins_prior"] / daily["races_prior"], np.nan)
    daily["avg_st_prior"] = np.where(daily["st_count_prior"] > 0, daily["st_sum_prior"] / daily["st_count_prior"], np.nan)
    mean_sq_prior = np.where(daily["st_count_prior"] > 0, daily["st_sq_sum_prior"] / daily["st_count_prior"], np.nan)
    var_prior = mean_sq_prior - daily["avg_st_prior"] ** 2
    daily["std_st_prior"] = np.sqrt(np.clip(var_prior, a_min=0.0, a_max=None))

    keep_cols = ["racer_no", "date", "races_prior", "win_rate_prior", "avg_st_prior", "std_st_prior", "st_count_prior"]
    pit = daily[keep_cols].copy()

    print("point-in-time stats built, rows:", len(pit))
    print(pit[pit["races_prior"] > 100][["avg_st_prior", "std_st_prior"]].describe())

    # ---- レースレベルにマージ ----
    merged = df.merge(pit, on=["racer_no", "date"], how="left")
    merged = merged[merged["races_prior"] >= MIN_RACES_PRIOR].copy()
    merged = merged[merged["st_count_prior"] >= MIN_RACES_PRIOR].copy()
    merged = merged.dropna(subset=["win_rate_prior", "avg_st_prior", "std_st_prior"])
    merged = merged[merged["course"].between(1, 6)]

    print()
    print("regression sample size:", len(merged))
    print()

    y = merged["is_top3"].to_numpy(dtype=float)

    # コース固定効果(course1を基準、2-6のダミー)
    course_dummies = pd.get_dummies(merged["course"], prefix="course", drop_first=True).astype(float)

    X_df = pd.DataFrame({
        "intercept": 1.0,
        "win_rate_prior": merged["win_rate_prior"].to_numpy(dtype=float),
        "avg_st_prior": merged["avg_st_prior"].to_numpy(dtype=float),
        "std_st_prior": merged["std_st_prior"].to_numpy(dtype=float),
    })
    X_df = pd.concat([X_df.reset_index(drop=True), course_dummies.reset_index(drop=True)], axis=1)

    X = X_df.to_numpy(dtype=float)
    col_names = list(X_df.columns)

    print("=== is_top3 ~ win_rate_prior + avg_st_prior + std_st_prior + course dummies ===")
    manual_ols(X, y, col_names)

    print()
    print("=== 参考: std_st_prior単独の相関 ===")
    print("corr(std_st_prior, is_top3):", np.corrcoef(merged["std_st_prior"], merged["is_top3"])[0, 1])
    print("corr(std_st_prior, avg_st_prior):", np.corrcoef(merged["std_st_prior"], merged["avg_st_prior"])[0, 1])
    print("corr(std_st_prior, win_rate_prior):", np.corrcoef(merged["std_st_prior"], merged["win_rate_prior"])[0, 1])


if __name__ == "__main__":
    main()
