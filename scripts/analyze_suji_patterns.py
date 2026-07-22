# scripts/analyze_suji_patterns.py
# -*- coding: utf-8 -*-
"""
競艇の「筋(スジ)舟券」セオリー(=1着コースが決まると、2着/3着に来やすいコースの
パターンがある、というもの)を、実データで検証する。

やり方:
  1. data/datasets/racer_race_history.csv (course, finish列) を使い、学習期間
     (CUTOFF以前)で「1着コースiのとき、2着コースがどこに来やすいか」の
     実際の分布と、最頻の(2着,3着)コースの組み合わせを計算する(=セオリーが
     実データでも成立するかどうかの記述的な確認)。
  2. data/logs/predictions.csv(実際にログされた予測確率・実オッズ)+
     data/logs/results.csv(実際の結果・払戻)を検証期間(CUTOFF以降、実際に
     起きた未来)に絞り、レースごとに「モデルの1着予想レーン」を基準にした
     セオリー通りの(2着,3着)パターンに一致する買い目(suji_match)が、
     モデルの確率(prob)を差し引いてもなお当たりやすいか(=市場やモデルが
     まだ織り込んでいない追加のシグナルかどうか)をOLS(HC1)で検証する。
  3. さらに、セオリー通りの買い目だけに絞った実オッズでのROIを計算し、
     [[boat_ai-roi-findings]]の選手クラスパターン検証と同じ基準
     (95%信頼区間が損益分岐点をまたぐかどうか)で「儲かるかどうか」を判定する。

注意: 「コース」(進入コース、course列)と「枠番」(艇番、predictions.csvの
comboが使っている番号)は実際には多くの場合一致するが、前づけ・スタート
展示での進入変化により異なるケースがある。predictions.csvのcomboは枠番
ベースのため、このスクリプトのPart 2は「進入コース」ではなく「枠番」を
セオリー適用の単位として扱っている(近似)。Part 1(racer_race_history.csv
のcourse列を使った集計)は実際の進入コースそのものなので、より厳密。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIST_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"
PRED_CSV = PROJECT_ROOT / "data" / "logs" / "predictions.csv"
RES_CSV = PROJECT_ROOT / "data" / "logs" / "results.csv"

CUTOFF = "20260601"
UNIT_BET_YEN = 100


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


def part1_theory_check() -> pd.DataFrame:
    print("=" * 100)
    print("Part 1: 「筋」セオリーは実データで成立するか(racer_race_history.csv, 学習期間のみ)")
    print("=" * 100)

    df = pd.read_csv(HIST_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df["course"] = pd.to_numeric(df["course"], errors="coerce")
    df = df[df["finish"].between(1, 6) & df["course"].between(1, 6)].copy()

    train = df[df["date"] < CUTOFF]

    key = ["date", "venue", "race_no"]
    piv = (
        train[train["finish"].isin([1, 2, 3])]
        .pivot_table(index=key, columns="finish", values="course", aggfunc="first")
    )
    piv.columns = [f"c{int(c)}" for c in piv.columns]
    piv = piv.dropna(subset=["c1", "c2", "c3"]).reset_index()
    piv[["c1", "c2", "c3"]] = piv[["c1", "c2", "c3"]].astype(int)

    print(f"学習期間の完走レース数: {len(piv)}")
    print()
    print("--- 1着コース別、2着コースの分布(%) ---")
    for i in range(1, 7):
        sub = piv[piv["c1"] == i]
        n = len(sub)
        if n == 0:
            continue
        dist = (sub["c2"].value_counts(normalize=True).sort_index() * 100).round(1)
        dist_str = ", ".join([f"{int(j)}コース:{v}%" for j, v in dist.items()])
        print(f"1着コース{i} (n={n}): {dist_str}")

    triple = piv.groupby(["c1", "c2", "c3"]).size().reset_index(name="n")
    triple["total_given_1st"] = triple.groupby("c1")["n"].transform("sum")
    triple["freq_given_1st"] = triple["n"] / triple["total_given_1st"]
    best = triple.sort_values("freq_given_1st", ascending=False).groupby("c1").first().reset_index()

    print()
    print("--- 1着コース別、最頻の(2着,3着)パターン(=学習期間データが示す「筋」) ---")
    print(best[["c1", "c2", "c3", "n", "freq_given_1st"]].round(3).to_string(index=False))

    return best.set_index("c1")[["c2", "c3"]].rename(columns={"c2": "best_c2", "c3": "best_c3"})


def part2_market_pricing_check(best_map: pd.DataFrame) -> None:
    print()
    print("=" * 100)
    print("Part 2: 「筋」通りの買い目は、実オッズ・実結果で見て儲かるか(検証期間のみ)")
    print("=" * 100)

    pred = pd.read_csv(PRED_CSV, low_memory=False)
    pred["date"] = pred["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    pred["venue"] = pred["venue"].astype(str).str.strip()
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()
    parts = pred["combo"].str.split("-", expand=True)
    pred["c1"] = pd.to_numeric(parts[0], errors="coerce")
    pred["c2"] = pd.to_numeric(parts[1], errors="coerce")
    pred["c3"] = pd.to_numeric(parts[2], errors="coerce")
    pred["prob"] = pd.to_numeric(pred["prob"], errors="coerce").fillna(0.0)
    pred["odds"] = pd.to_numeric(pred["odds"], errors="coerce")
    pred["rank_prob"] = pd.to_numeric(pred["rank_prob"], errors="coerce")

    res = pd.read_csv(RES_CSV, low_memory=False)
    res["date"] = res["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    res["venue"] = res["venue"].astype(str).str.strip()
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)

    merged = pred.merge(
        res[["date", "venue", "race_no", "actual_combo", "payout"]],
        on=["date", "venue", "race_no"], how="inner",
    )
    merged["is_hit"] = (merged["combo"] == merged["actual_combo"]).astype(int)

    test = merged[merged["date"] >= CUTOFF].copy()
    print(f"検証期間: {test['date'].min()} 〜 {test['date'].max()} (n={len(test)}行, "
          f"{test[['date','venue','race_no']].drop_duplicates().shape[0]}レース)")

    # モデルの1着予想レーン(=そのレースでrank_prob==1の買い目の1着レーン)を
    # 「セオリー適用の基準となる1着」として使う(実際にベットする時点で分かる情報のみ使用)
    key = ["date", "venue", "race_no"]
    top1 = (
        test[test["rank_prob"] == 1][key + ["c1"]]
        .rename(columns={"c1": "assumed_1st"})
        .drop_duplicates(subset=key)
    )
    test = test.merge(top1, on=key, how="left")

    test["best_c2"] = test["assumed_1st"].map(best_map["best_c2"])
    test["best_c3"] = test["assumed_1st"].map(best_map["best_c3"])
    test["suji_match"] = (
        (test["c2"] == test["best_c2"]) & (test["c3"] == test["best_c3"])
    ).astype(int)

    print(f"suji_match=1(セオリー通りの買い目)の行数: {int(test['suji_match'].sum())} / {len(test)}")

    # --- OLS: is_hit ~ prob + suji_match ---
    # モデルの確率(prob)を差し引いてもなお、セオリー通りの目が追加的に当たりやすいかを見る
    valid = test.dropna(subset=["prob", "assumed_1st"]).copy()
    y = valid["is_hit"].values.astype(float)
    X = np.column_stack([
        np.ones(len(valid)),
        valid["prob"].values.astype(float),
        valid["suji_match"].values.astype(float),
    ])
    result = fit_ols_hc1(X, y, ["intercept", "model_prob", "suji_match"])
    print()
    print(f"=== 重回帰: is_hit ~ model_prob + suji_match (n={result.attrs['n']}, R^2={result.attrs['r2']:.4f}) ===")
    print(result.round(6))
    row = result.loc["suji_match"]
    if row["significant"]:
        print("-> suji_matchは統計的に有意。モデルの確率を差し引いても、セオリー通りの目は追加的に当たりやすい"
              "(本物のシグナルの可能性がある)。")
    else:
        print("-> suji_matchは有意でない。セオリー通りの目が当たりやすいとしても、それはモデルの確率(prob)"
              "が既に織り込んでいる可能性が高い。")

    # --- セオリー通りの買い目だけに絞った実ROI ---
    print()
    print("--- suji_match有無別、実オッズでのROI(100円均等買い想定) ---")
    for flag, label in [(1, "セオリー通り(suji_match=1)"), (0, "セオリー外(suji_match=0)")]:
        sub = valid[valid["suji_match"] == flag]
        n = len(sub)
        invested = n * UNIT_BET_YEN
        returned = float((sub["is_hit"] * sub["payout"]).sum())
        roi = returned / invested * 100 if invested > 0 else 0.0
        hit_rate = sub["is_hit"].mean() if n > 0 else 0.0
        # ROIの95%信頼区間(profit per bet の平均に対する単純なCI)
        profit_per_bet = sub["is_hit"] * sub["payout"] - UNIT_BET_YEN
        se = profit_per_bet.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        mean_profit = profit_per_bet.mean() if n > 0 else 0.0
        ci_low = mean_profit - 1.96 * se
        ci_high = mean_profit + 1.96 * se
        print(f"[{label}] n={n}, hit_rate={hit_rate*100:.2f}%, ROI={roi:.2f}%, "
              f"平均損益/点={mean_profit:.1f}円 (95%CI=[{ci_low:.1f}, {ci_high:.1f}], "
              f"損益分岐(0)をまたぐ: {ci_low < 0 < ci_high})")


def main() -> None:
    best_map = part1_theory_check()
    part2_market_pricing_check(best_map)


if __name__ == "__main__":
    main()
