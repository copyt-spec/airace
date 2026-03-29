from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.prediction_logger import PREDICTIONS_CSV, RESULTS_CSV, MERGED_CSV


def main() -> None:
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"missing predictions log: {PREDICTIONS_CSV}")
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"missing results log: {RESULTS_CSV}")

    pred = pd.read_csv(PREDICTIONS_CSV)
    res = pd.read_csv(RESULTS_CSV)

    res = res.sort_values(["date", "venue", "race_no", "logged_at"])
    res = res.drop_duplicates(subset=["date", "venue", "race_no"], keep="last")

    merged = pred.merge(
        res,
        on=["date", "venue", "race_no"],
        how="left",
    )

    merged["is_hit"] = (merged["combo"].astype(str) == merged["actual_combo"].astype(str)).astype(int)
    merged["bet_cost_yen"] = merged["is_selected"].fillna(0).astype(int) * 100
    merged["return_yen"] = 0.0

    hit_mask = (merged["is_selected"] == 1) & (merged["is_hit"] == 1)
    merged.loc[hit_mask, "return_yen"] = merged.loc[hit_mask, "payout"].fillna(0.0)

    merged["profit_yen"] = merged["return_yen"] - merged["bet_cost_yen"]

    MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")

    selected = merged[merged["is_selected"] == 1].copy()

    total_bets = float(selected["bet_cost_yen"].sum())
    total_return = float(selected["return_yen"].sum())
    total_profit = float(selected["profit_yen"].sum())
    hit_count = int(((selected["is_hit"] == 1)).sum())
    buy_count = int(len(selected))

    roi = (total_return / total_bets) if total_bets > 0 else 0.0
    hit_rate = (hit_count / buy_count) if buy_count > 0 else 0.0

    print("===== MERGED DONE =====")
    print("merged file   :", MERGED_CSV)
    print("buy_count     :", buy_count)
    print("hit_count     :", hit_count)
    print("hit_rate      :", round(hit_rate, 4))
    print("total_bets    :", round(total_bets, 2))
    print("total_return  :", round(total_return, 2))
    print("total_profit  :", round(total_profit, 2))
    print("roi           :", round(roi, 4))


if __name__ == "__main__":
    main()
