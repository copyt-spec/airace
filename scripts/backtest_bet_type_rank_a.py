from __future__ import annotations

"""
RANK Aレース限定で、現行の3連単買い目を2連単/2連複/3連複箱に変換したら
収支が上向くか検証する。

方式は scripts/backtest_bet_type_v2.py と同じ「オッズ近似を使わない」
アプローチ: 買い目の"選び方"自体は現行buy_selector(実オッズ・実確率での
3連単選定)のまま変えず、選ばれた3連単の買い目を他の券種のキーに変換
(重複除去)して、実測payout列(payout_2tan/payout_2puku/payout_3puku)
だけで的中判定する。オッズの近似が一切要らないので信頼度が高い。

RANK A判定は app/main.py の _build_race_signal と全く同じ式を再現する
(top_prob>=0.11, prob_gap>=0.020, top_ev>=0.85)。
"""

import argparse
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


def _box3_key(combo: str) -> str:
    a, b, c = combo.split("-")
    return "-".join(sorted([a, b, c], key=int))


def _nirentan_key(combo: str) -> str:
    a, b, _c = combo.split("-")
    return f"{a}-{b}"


def _nirenpuku_key(combo: str) -> str:
    a, b, _c = combo.split("-")
    return "-".join(sorted([a, b], key=int))


def load_data(predictions_path: Path = PREDICTIONS_CSV) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path, low_memory=False)
    res = pd.read_csv(RESULTS_CSV, low_memory=False)

    pred["date"] = _normalize_date(pred["date"])
    res["date"] = _normalize_date(res["date"])
    pred["venue"] = pred["venue"].astype(str).str.strip()
    res["venue"] = res["venue"].astype(str).str.strip()
    pred["race_no"] = pd.to_numeric(pred["race_no"], errors="coerce").fillna(0).astype(int)
    res["race_no"] = pd.to_numeric(res["race_no"], errors="coerce").fillna(0).astype(int)
    pred["combo"] = pred["combo"].astype(str).str.strip()
    pred["prob"] = pd.to_numeric(pred["prob"], errors="coerce").fillna(0.0)
    pred["odds"] = pd.to_numeric(pred["odds"], errors="coerce").fillna(0.0)

    keep_res_cols = [
        "date", "venue", "race_no",
        "actual_combo", "payout",
        "combo_3puku", "payout_3puku",
        "combo_2tan", "payout_2tan",
        "combo_2puku", "payout_2puku",
    ]
    keep_res_cols = [c for c in keep_res_cols if c in res.columns]

    merged = pred.merge(res[keep_res_cols], on=["date", "venue", "race_no"], how="left")
    return merged


def run(df: pd.DataFrame) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    n_groups = 0
    n_rank_a = 0

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        n_groups += 1
        actual_combo = race_df["actual_combo"].iloc[0]
        if pd.isna(actual_combo):
            continue

        payout = race_df["payout"].iloc[0]

        probs_sorted = race_df["prob"].sort_values(ascending=False).tolist()
        top_prob = probs_sorted[0] if len(probs_sorted) >= 1 else 0.0
        second_prob = probs_sorted[1] if len(probs_sorted) >= 2 else 0.0
        prob_gap = top_prob - second_prob
        ev_series = race_df["prob"] * race_df["odds"]
        top_ev = float(ev_series.max()) if len(ev_series) else 0.0

        is_rank_a = (top_prob >= 0.11) and (prob_gap >= 0.020) and (top_ev >= 0.85)
        if not is_rank_a:
            continue
        n_rank_a += 1

        ai_preds = [
            {"combo": r["combo"], "prob": float(r["prob"]), "odds": float(r["odds"])}
            for _, r in race_df.iterrows()
        ]

        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)
        buy_3tan = [b["combo"] for b in best_bets]
        n_tan = len(buy_3tan)

        invested_3tan = n_tan * UNIT_BET_YEN
        hit_3tan = 1 if actual_combo in buy_3tan else 0
        returned_3tan = int(float(payout)) if hit_3tan and pd.notna(payout) else 0

        row_out: Dict[str, Any] = {
            "date": date, "venue": venue, "race_no": race_no,
            "top_prob": top_prob, "prob_gap": prob_gap, "top_ev": top_ev,
            "n_tan": n_tan,
            "invested_3tan": invested_3tan, "returned_3tan": returned_3tan, "hit_3tan": hit_3tan,
        }

        # 3連複箱
        actual_3puku = race_df["combo_3puku"].iloc[0] if "combo_3puku" in race_df.columns else None
        payout_3puku = race_df["payout_3puku"].iloc[0] if "payout_3puku" in race_df.columns else None
        buy_3puku = sorted(set(_box3_key(c) for c in buy_3tan))
        n_3puku = len(buy_3puku)
        invested_3puku = n_3puku * UNIT_BET_YEN
        hit_3puku = 0
        returned_3puku = 0
        if actual_3puku is not None and pd.notna(actual_3puku):
            hit_3puku = 1 if str(actual_3puku) in buy_3puku else 0
            if hit_3puku and pd.notna(payout_3puku):
                returned_3puku = int(float(payout_3puku))
        row_out.update({
            "n_3puku": n_3puku,
            "invested_3puku": invested_3puku, "returned_3puku": returned_3puku, "hit_3puku": hit_3puku,
        })

        # 2連単
        actual_2tan = race_df["combo_2tan"].iloc[0] if "combo_2tan" in race_df.columns else None
        payout_2tan = race_df["payout_2tan"].iloc[0] if "payout_2tan" in race_df.columns else None
        buy_2tan = sorted(set(_nirentan_key(c) for c in buy_3tan))
        n_2tan = len(buy_2tan)
        invested_2tan = n_2tan * UNIT_BET_YEN
        hit_2tan = 0
        returned_2tan = 0
        if actual_2tan is not None and pd.notna(actual_2tan):
            hit_2tan = 1 if str(actual_2tan) in buy_2tan else 0
            if hit_2tan and pd.notna(payout_2tan):
                returned_2tan = int(float(payout_2tan))
        row_out.update({
            "n_2tan": n_2tan,
            "invested_2tan": invested_2tan, "returned_2tan": returned_2tan, "hit_2tan": hit_2tan,
        })

        # 2連複
        actual_2puku = race_df["combo_2puku"].iloc[0] if "combo_2puku" in race_df.columns else None
        payout_2puku = race_df["payout_2puku"].iloc[0] if "payout_2puku" in race_df.columns else None
        buy_2puku = sorted(set(_nirenpuku_key(c) for c in buy_3tan))
        n_2puku = len(buy_2puku)
        invested_2puku = n_2puku * UNIT_BET_YEN
        hit_2puku = 0
        returned_2puku = 0
        if actual_2puku is not None and pd.notna(actual_2puku):
            hit_2puku = 1 if str(actual_2puku) in buy_2puku else 0
            if hit_2puku and pd.notna(payout_2puku):
                returned_2puku = int(float(payout_2puku))
        row_out.update({
            "n_2puku": n_2puku,
            "invested_2puku": invested_2puku, "returned_2puku": returned_2puku, "hit_2puku": hit_2puku,
        })

        results.append(row_out)

    print(f"total race-groups scanned: {n_groups}, RANK A races: {n_rank_a}, with valid actual_combo & bets: {len(results)}")
    return pd.DataFrame(results)


def summarize(res: pd.DataFrame) -> None:
    if res.empty:
        print("no data")
        return
    print(f"races (RANK A, evaluated): {len(res)}")
    print()

    for label, inv_col, ret_col, hit_col, n_col in [
        ("3連単(現行のまま)", "invested_3tan", "returned_3tan", "hit_3tan", "n_tan"),
        ("同じ買い目を3連複箱に変換", "invested_3puku", "returned_3puku", "hit_3puku", "n_3puku"),
        ("同じ買い目を2連単に変換", "invested_2tan", "returned_2tan", "hit_2tan", "n_2tan"),
        ("同じ買い目を2連複に変換", "invested_2puku", "returned_2puku", "hit_2puku", "n_2puku"),
    ]:
        inv = res[inv_col].sum()
        ret = res[ret_col].sum()
        roi = ret / inv if inv > 0 else 0.0
        hit_rate = res[hit_col].mean()
        avg_n = res[n_col].mean()
        print(f"--- {label} ---")
        print(f"  avg_points={avg_n:.2f} invested={inv:.0f} returned={ret:.0f} profit={ret-inv:.0f} roi={roi*100:.2f}% hit_rate={hit_rate*100:.2f}%")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=str(PREDICTIONS_CSV),
        help="predictions.csv or predictions_combined_for_sim.csv (最近のholdoutクロスチェック用)",
    )
    parser.add_argument("--out_suffix", default="")
    args = parser.parse_args()

    print("loading data from", args.predictions, "...")
    df = load_data(Path(args.predictions))
    print("running backtest (RANK A only)...")
    res = run(df)
    summarize(res)
    out_path = PROJECT_ROOT / "data" / "logs" / "analysis" / f"bet_type_backtest_rank_a{args.out_suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)
