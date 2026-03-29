from __future__ import annotations

import pandas as pd
from pathlib import Path


LOG_PATH = Path("data/logs/prediction_results_merged.csv")


def categorize_ev(ev: float) -> str:
    """
    EVを帯で分類
    """
    if ev >= 3.0:
        return "EV>=3.0"
    elif ev >= 2.0:
        return "EV 2.0-3.0"
    elif ev >= 1.5:
        return "EV 1.5-2.0"
    elif ev >= 1.2:
        return "EV 1.2-1.5"
    elif ev >= 1.0:
        return "EV 1.0-1.2"
    else:
        return "EV<1.0"


def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError("merged log not found. run merge first.")

    df = pd.read_csv(LOG_PATH)

    # 買ったやつだけ
    df = df[df["is_selected"] == 1].copy()

    if len(df) == 0:
        print("No selected bets found.")
        return

    # EV列（100円基準なので expected_return_yen / 100）
    df["ev"] = df["expected_return_yen"] / 100.0

    # EV帯分類
    df["ev_bin"] = df["ev"].apply(categorize_ev)

    # 集計
    grouped = df.groupby("ev_bin").agg(
        bets=("bet_cost_yen", "sum"),
        returns=("return_yen", "sum"),
        hits=("is_hit", "sum"),
        count=("is_hit", "count"),
    ).reset_index()

    # 指標
    grouped["roi"] = grouped["returns"] / grouped["bets"]
    grouped["hit_rate"] = grouped["hits"] / grouped["count"]

    # 並び替え（EV順）
    order = [
        "EV>=3.0",
        "EV 2.0-3.0",
        "EV 1.5-2.0",
        "EV 1.2-1.5",
        "EV 1.0-1.2",
        "EV<1.0",
    ]
    grouped["order"] = grouped["ev_bin"].apply(lambda x: order.index(x))
    grouped = grouped.sort_values("order").drop(columns=["order"])

    print("\n===== EV帯分析 =====\n")
    print(grouped.to_string(index=False))

    print("\n===== 総合 =====\n")
    total_bets = df["bet_cost_yen"].sum()
    total_return = df["return_yen"].sum()
    roi = total_return / total_bets if total_bets > 0 else 0

    print(f"total_bets   : {total_bets}")
    print(f"total_return : {total_return}")
    print(f"ROI          : {round(roi, 4)}")


if __name__ == "__main__":
    main()
