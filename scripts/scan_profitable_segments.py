# scripts/scan_profitable_segments.py
# -*- coding: utf-8 -*-
"""
「本当に100%(損益分岐)を超えると言えるセグメントがあるか」を、多数の切り口
(単独要因・2要因の組み合わせ)について機械的にスキャンし、95%信頼区間付きで
判定する「監視リスト」ツール。

背景:
  - 単純集計だけだと、小サンプルでたまたま良く見えているだけの数字を
    「儲かる」と誤認するリスクが高い(このプロジェクトで過去に何度か
    痛い目にあっている: 常滑の263%異常値がリーク混入だった件など)。
  - かといって「有意でないから何も無い」と切り捨てると、本当に良い
    セグメントを早期に見逃す。
  - そこでこのスクリプトは各セグメントを3段階に分類する:
      CONFIRMED : 95%信頼区間の下限が0円(=ROI100%)を超える
                  -> 本番のA+的な扱いにして良い
      CANDIDATE : 点推定(平均損益)はプラスだが信頼区間は0をまたぐ
                  -> 「監視リスト」に載せて、データが増えるたびに再判定する
      (該当なし): 点推定もマイナス

  - 多重比較(たくさんの切り口を同時に試すと偶然有意に見えるものが増える)
    には要注意。ここでは検定数を毎回表示し、過度に細かい切り口
    (n が小さすぎるもの)は最初から除外することである程度対策している。

実行するたびに data/logs/edge_scan_history.json に結果を追記保存するので、
「このセグメントが時間とともにCANDIDATE -> CONFIRMEDに育ったか、
それとも尻すぼみで消えたか」を後から追える。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIGHT_CSV = PROJECT_ROOT / "data" / "logs" / "prediction_results_sim_1y_light.csv"
HISTORY_JSON = PROJECT_ROOT / "data" / "logs" / "edge_scan_history.json"

MIN_N = 100  # これ未満のセグメントは最初からスキャン対象外(過学習防止)
Z = 1.96  # 95%信頼区間


def load_data() -> pd.DataFrame:
    df = pd.read_csv(LIGHT_CSV, low_memory=False)
    df["date"] = df["date"].astype(str)
    df["dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["weekday"] = df["dt"].dt.day_name()
    df["month"] = df["dt"].dt.to_period("M").astype(str)

    parts = df["combo"].astype(str).str.split("-")
    df["pred_1st"] = parts.str[0]
    df["pred_2nd"] = parts.str[1]
    df["pred_3rd"] = parts.str[2]

    # レースごとの購入点数(モデルがそのレースにどれだけ強気/弱気だったか)
    race_key = df["date"] + "_" + df["venue"] + "_" + df["race_no"].astype(str)
    df["race_key"] = race_key
    points = df.groupby("race_key")["combo"].transform("count")
    df["race_points"] = points
    df["race_points_band"] = pd.cut(
        df["race_points"], bins=[0, 3, 5, 8, 100], labels=["3点以下", "4-5点", "6-8点", "9点以上"]
    )

    # 直近60日かどうか(recency: 最新モデル/直近相場での再現性チェック用)
    max_date = df["dt"].max()
    df["is_recent60"] = (max_date - df["dt"]).dt.days <= 60

    return df


def ci_profit(g: pd.DataFrame, z: float = Z) -> pd.Series:
    n = len(g)
    m = g["profit_yen"].mean()
    s = g["profit_yen"].std(ddof=1) if n > 1 else np.nan
    se = s / np.sqrt(n) if n > 1 and s == s else np.nan
    ci_low = m - z * se if se == se else np.nan
    ci_high = m + z * se if se == se else np.nan
    return pd.Series({
        "n": n,
        "mean_profit": m,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "roi_est_pct": m + 100.0,
    })


SEGMENT_DEFS: List[Dict[str, Any]] = [
    {"name": "venue", "cols": ["venue"]},
    {"name": "odds_band", "cols": ["odds_band"]},
    {"name": "conf_tier", "cols": ["conf_tier"]},
    {"name": "pred_1st(コース)", "cols": ["pred_1st"]},
    {"name": "pred_2nd(2着コース)", "cols": ["pred_2nd"]},
    {"name": "pred_3rd(3着コース)", "cols": ["pred_3rd"]},
    {"name": "race_no", "cols": ["race_no"]},
    {"name": "weekday", "cols": ["weekday"]},
    {"name": "race_points_band(買い点数)", "cols": ["race_points_band"]},
    {"name": "venue x odds_band", "cols": ["venue", "odds_band"]},
    {"name": "venue x pred_1st", "cols": ["venue", "pred_1st"]},
    {"name": "odds_band x pred_1st", "cols": ["odds_band", "pred_1st"]},
    {"name": "odds_band x race_no", "cols": ["odds_band", "race_no"]},
    {"name": "venue x race_points_band", "cols": ["venue", "race_points_band"]},
]


def scan(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_tests = 0
    for spec in SEGMENT_DEFS:
        cols = spec["cols"]
        g = df.groupby(cols, observed=True).apply(ci_profit, include_groups=False)
        g = g[g["n"] >= MIN_N].copy()
        n_tests += len(g)
        g["segment_type"] = spec["name"]
        g["segment_value"] = [
            str(idx) if not isinstance(idx, tuple) else " / ".join(str(x) for x in idx)
            for idx in g.index
        ]
        rows.append(g.reset_index(drop=True))

    result = pd.concat(rows, ignore_index=True)

    def classify(r):
        if pd.notna(r["ci_low"]) and r["ci_low"] > 0:
            return "CONFIRMED"
        if r["mean_profit"] > 0:
            return "CANDIDATE"
        return ""

    result["status"] = result.apply(classify, axis=1)
    result.attrs["n_tests"] = n_tests
    return result


def save_history(result: pd.DataFrame, total_n: int) -> None:
    HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if HISTORY_JSON.exists():
        try:
            history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        except Exception:
            history = []

    interesting = result[result["status"].isin(["CONFIRMED", "CANDIDATE"])].copy()
    entry = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "total_bets": int(total_n),
        "n_tests": int(result.attrs.get("n_tests", 0)),
        "segments": [
            {
                "segment_type": r["segment_type"],
                "segment_value": r["segment_value"],
                "n": int(r["n"]),
                "mean_profit": round(float(r["mean_profit"]), 2),
                "ci_low": round(float(r["ci_low"]), 2) if pd.notna(r["ci_low"]) else None,
                "ci_high": round(float(r["ci_high"]), 2) if pd.notna(r["ci_high"]) else None,
                "roi_est_pct": round(float(r["roi_est_pct"]), 2),
                "status": r["status"],
            }
            for _, r in interesting.sort_values("mean_profit", ascending=False).iterrows()
        ],
    }
    history.append(entry)
    HISTORY_JSON.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    df = load_data()
    result = scan(df)

    pd.set_option("display.width", 160)
    print(f"総ベット数: {len(df)}  スキャンしたセグメント数(n>={MIN_N}のみ): {result.attrs.get('n_tests', 0)}")
    print(f"(多重比較注意: これだけの数を同時に検定しているので、単独のp<0.05は割り引いて見ること)")
    print()

    confirmed = result[result["status"] == "CONFIRMED"].sort_values("mean_profit", ascending=False)
    print("=== CONFIRMED(信頼区間下限が0円=ROI100%を超える) ===")
    if confirmed.empty:
        print("(該当なし)")
    else:
        print(confirmed[["segment_type", "segment_value", "n", "mean_profit", "ci_low", "ci_high", "roi_est_pct"]]
              .round(2).to_string(index=False))
    print()

    candidate = result[result["status"] == "CANDIDATE"].sort_values("mean_profit", ascending=False)
    print("=== CANDIDATE(点推定はプラスだが信頼区間は0をまたぐ、監視リスト) ===")
    if candidate.empty:
        print("(該当なし)")
    else:
        print(candidate[["segment_type", "segment_value", "n", "mean_profit", "ci_low", "ci_high", "roi_est_pct"]]
              .round(2).head(20).to_string(index=False))

    save_history(result, len(df))
    print()
    print(f"履歴を保存しました -> {HISTORY_JSON}")


if __name__ == "__main__":
    main()
