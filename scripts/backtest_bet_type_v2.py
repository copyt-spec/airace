from __future__ import annotations

"""
アプローチ1(backtest_bet_type.py): 3連単オッズから3連複オッズを近似して、
3連複を独自にEV選択する → 近似誤差が大きく、あまり信用できない結果になった。

アプローチ2(このスクリプト): 買い目の"選び方"は今のbuy_selector(3連単の
実オッズ・実確率)のまま変えず、選ばれた3連単の買い目を"3連複の箱"に
変換して買った場合にどうなるかを見る。

例: 今の買い目が 1-2-3, 2-1-3, 1-3-2 の3点(=同じ3艇{1,2,3}の異なる順序)
    だったとする。3連単のままなら「結果が1-2-3の順で来た時だけ」的中。
    3連複に変換すると 箱"1-2-3" 1点にまとまり(コストは3点→1点に減る)、
    結果が 1-2-3 でも 2-1-3 でも 3-1-2 でも(=1,2,3のどれかの順序)的中する。

このアプローチはオッズの近似を一切使わず、実測の3連複払戻(payout_3puku)
だけで判定するので、結果の信頼性が高い。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import buy_selector  # noqa: E402

LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
RESULTS_CSV = LOG_DIR / "results.csv"

UNIT_BET_YEN = 100


def _normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def _box_key(combo: str) -> str:
    a, b, c = combo.split("-")
    return "-".join(sorted([a, b, c], key=int))


def load_data() -> pd.DataFrame:
    pred = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    res = pd.read_csv(RESULTS_CSV, low_memory=False)

    pred["date"] = _normalize_date(pred["date"])
    res["date"] = _normalize_date(res["date"])
    pred["venue"] = pred["venue"].astype(str).str.strip()
    res["venue"] = res["venue"].astype(str).str.strip()
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()

    keep_res_cols = ["date", "venue", "race_no", "actual_combo", "payout", "combo_3puku", "payout_3puku"]
    keep_res_cols = [c for c in keep_res_cols if c in res.columns or c in ["date", "venue", "race_no"]]

    merged = pred.merge(res[keep_res_cols], on=["date", "venue", "race_no"], how="left")
    return merged


def run(df: pd.DataFrame) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]
        actual_combo_3puku = race_df["combo_3puku"].iloc[0]
        payout_3puku = race_df["payout_3puku"].iloc[0]

        if pd.isna(actual_combo) or pd.isna(actual_combo_3puku):
            continue

        ai_preds = [
            {"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])}
            for _, r in race_df.iterrows()
        ]

        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)
        buy_3tan = [b["combo"] for b in best_bets]
        n_tan = len(buy_3tan)

        # 3連単のまま
        invested_3tan = n_tan * UNIT_BET_YEN
        hit_3tan = 1 if actual_combo in buy_3tan else 0
        returned_3tan = int(float(payout)) if hit_3tan and pd.notna(payout) else 0

        # 同じ買い目を3連複の箱に変換(重複除去)
        buy_3puku_boxes = sorted(set(_box_key(c) for c in buy_3tan))
        n_box = len(buy_3puku_boxes)
        invested_3puku = n_box * UNIT_BET_YEN
        hit_3puku = 1 if str(actual_combo_3puku) in buy_3puku_boxes else 0
        returned_3puku = int(float(payout_3puku)) if hit_3puku and pd.notna(payout_3puku) else 0

        results.append({
            "date": date, "venue": venue, "race_no": race_no,
            "n_tan": n_tan, "n_box": n_box,
            "invested_3tan": invested_3tan, "returned_3tan": returned_3tan, "hit_3tan": hit_3tan,
            "invested_3puku": invested_3puku, "returned_3puku": returned_3puku, "hit_3puku": hit_3puku,
        })

    return pd.DataFrame(results)


def summarize(res: pd.DataFrame) -> None:
    print(f"races: {len(res)}")
    print(f"avg points  3tan={res['n_tan'].mean():.2f}  3puku(box, dedup後)={res['n_box'].mean():.2f}")
    print()

    for label, inv_col, ret_col, hit_col in [
        ("3連単(現行のまま)", "invested_3tan", "returned_3tan", "hit_3tan"),
        ("同じ買い目を3連複箱に変換", "invested_3puku", "returned_3puku", "hit_3puku"),
    ]:
        inv = res[inv_col].sum()
        ret = res[ret_col].sum()
        roi = ret / inv if inv > 0 else 0.0
        hit_rate = res[hit_col].mean()
        print(f"--- {label} ---")
        print(f"  invested={inv:.0f} returned={ret:.0f} profit={ret-inv:.0f} roi={roi*100:.2f}% hit_rate={hit_rate*100:.2f}%")

    print()
    # 点数が減った分の効率(1点あたり回収率)も見る
    for label, inv_col, ret_col in [("3連単", "invested_3tan", "returned_3tan"), ("3連複箱", "invested_3puku", "returned_3puku")]:
        pass


if __name__ == "__main__":
    print("loading data...")
    df = load_data()
    print("running backtest...")
    res = run(df)
    summarize(res)
    res.to_csv(PROJECT_ROOT / "data" / "logs" / "analysis" / "bet_type_backtest_v2.csv", index=False, encoding="utf-8-sig")
