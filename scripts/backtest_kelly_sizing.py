from __future__ import annotations

"""
scripts/backtest_buy_selector.py と同じデータ(data/logs/predictions.csv +
data/logs/results.csv)を使い、「今のbuy_selectorの選定はそのまま」で、
賭け金だけを固定100円 vs engine.kelly_allocator.kelly_size_selected_bets()
にした場合とで、バンクロール(資産)の推移がどう変わるかを比較する。

ROI%(トータルの回収率)自体はKellyサイジングでほぼ変わらない前提の
施策なので、ここで見るべきなのは:
  - 最終バンクロール(複利で賭け続けた場合の資産の伸び)
  - 最大ドローダウン(資産が一時的にどれだけ減ったか)
  - 破産(バンクロールが0や運用不能な水準まで減った)が起きなかったか

date順にレースを並べ、各レースで:
  1. buy_selector.select_best_bets(...) で買い目を選ぶ(現行ロジックのまま)
  2. 固定シミュレーション: 1点100円で賭ける(現状の運用)
  3. Kellyシミュレーション: その時点のバンクロールに対してkelly_size_selected_betsで
     賭け金を決める(複利、bankrollは毎レース更新される)
を両方計算し、レース単位・日次・累計で比較する。
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
from engine.kelly_allocator import kelly_size_selected_bets  # noqa: E402
from scripts.backtest_buy_selector import load_data  # noqa: E402

UNIT_BET_YEN = 100
INITIAL_BANKROLL = 100_000


def simulate(df: pd.DataFrame, initial_bankroll: int = INITIAL_BANKROLL) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    flat_bankroll = float(initial_bankroll)
    kelly_bankroll = float(initial_bankroll)
    flat_peak = flat_bankroll
    kelly_peak = kelly_bankroll

    # レースを日付順に並べて処理する(複利計算のため順序が重要)
    race_keys = (
        df[["date", "venue", "race_no"]]
        .drop_duplicates()
        .sort_values(["date", "venue", "race_no"])
        .itertuples(index=False, name=None)
    )

    grouped = {k: g for k, g in df.groupby(["date", "venue", "race_no"], sort=False)}

    for key in race_keys:
        race_df = grouped.get(key)
        if race_df is None or race_df.empty:
            continue

        date, venue, race_no = key
        actual_combo = race_df["actual_combo"].iloc[0]
        payout = race_df["payout"].iloc[0]

        if pd.isna(actual_combo):
            continue

        ai_preds = []
        for _, r in race_df.iterrows():
            ai_preds.append({
                "combo": r["combo"],
                "prob": float(r["prob"]),
                "odds": float(r["odds"]),
            })

        best_bets = buy_selector.select_best_bets(ai_preds, venue=venue)
        if not best_bets:
            rows.append({
                "date": date, "venue": venue, "race_no": race_no,
                "buy_count": 0,
                "flat_invested": 0, "flat_returned": 0,
                "kelly_invested": 0, "kelly_returned": 0,
                "flat_bankroll": flat_bankroll, "kelly_bankroll": kelly_bankroll,
            })
            continue

        hit = actual_combo in [b["combo"] for b in best_bets]
        payout_val = int(float(payout)) if pd.notna(payout) else 0

        # --- 固定100円ベット ---
        flat_invested = len(best_bets) * UNIT_BET_YEN
        flat_returned = payout_val if hit else 0
        flat_bankroll += (flat_returned - flat_invested)
        flat_peak = max(flat_peak, flat_bankroll)

        # --- Kellyサイジング(その時点のkelly_bankrollを使って複利) ---
        sized = kelly_size_selected_bets(best_bets, bankroll=max(0, int(kelly_bankroll)))
        kelly_invested = sum(int(b.get("bet_amount_yen", 0)) for b in sized)

        kelly_returned = 0
        if hit:
            hit_bet = next(
                (b for b in sized if b.get("combo") == actual_combo),
                None,
            )
            if hit_bet is not None:
                bet_yen = int(hit_bet.get("bet_amount_yen", 0))
                # payoutは「100円払戻金」なので、実際の賭け金に比例させる
                kelly_returned = int(payout_val * (bet_yen / 100.0)) if bet_yen > 0 else 0

        kelly_bankroll += (kelly_returned - kelly_invested)
        kelly_peak = max(kelly_peak, kelly_bankroll)

        rows.append({
            "date": date, "venue": venue, "race_no": race_no,
            "buy_count": len(best_bets),
            "flat_invested": flat_invested, "flat_returned": flat_returned,
            "kelly_invested": kelly_invested, "kelly_returned": kelly_returned,
            "flat_bankroll": flat_bankroll, "kelly_bankroll": kelly_bankroll,
        })

    return pd.DataFrame(rows)


def summarize(sim_df: pd.DataFrame, initial_bankroll: int = INITIAL_BANKROLL) -> None:
    if sim_df.empty:
        print("no data")
        return

    n_races = len(sim_df)

    flat_total_inv = sim_df["flat_invested"].sum()
    flat_total_ret = sim_df["flat_returned"].sum()
    flat_final = sim_df["flat_bankroll"].iloc[-1]
    flat_roi = flat_total_ret / flat_total_inv if flat_total_inv > 0 else 0.0

    kelly_total_inv = sim_df["kelly_invested"].sum()
    kelly_total_ret = sim_df["kelly_returned"].sum()
    kelly_final = sim_df["kelly_bankroll"].iloc[-1]
    kelly_roi = kelly_total_ret / kelly_total_inv if kelly_total_inv > 0 else 0.0

    # 最大ドローダウン(その時点までの最高資産からの下落率)
    def max_drawdown(bankroll_series: pd.Series) -> float:
        running_max = bankroll_series.cummax()
        dd = (bankroll_series - running_max) / running_max.replace(0, pd.NA)
        return float(dd.min()) if dd.notna().any() else 0.0

    flat_mdd = max_drawdown(sim_df["flat_bankroll"])
    kelly_mdd = max_drawdown(sim_df["kelly_bankroll"])

    flat_min = sim_df["flat_bankroll"].min()
    kelly_min = sim_df["kelly_bankroll"].min()

    print(f"races                : {n_races}")
    print(f"initial bankroll     : {initial_bankroll:,}円")
    print()
    print("--- 固定100円ベット ---")
    print(f"  total invested     : {flat_total_inv:,.0f}円")
    print(f"  total returned     : {flat_total_ret:,.0f}円")
    print(f"  roi                : {flat_roi*100:.2f}%")
    print(f"  final bankroll     : {flat_final:,.0f}円 (初期比 {flat_final/initial_bankroll*100:.1f}%)")
    print(f"  min bankroll       : {flat_min:,.0f}円")
    print(f"  max drawdown       : {flat_mdd*100:.2f}%")
    print()
    print("--- Kellyサイジング(1/4ケリー, 総額上限5%, 1点上限2%) ---")
    print(f"  total invested     : {kelly_total_inv:,.0f}円")
    print(f"  total returned     : {kelly_total_ret:,.0f}円")
    print(f"  roi                : {kelly_roi*100:.2f}%")
    print(f"  final bankroll     : {kelly_final:,.0f}円 (初期比 {kelly_final/initial_bankroll*100:.1f}%)")
    print(f"  min bankroll       : {kelly_min:,.0f}円")
    print(f"  max drawdown       : {kelly_mdd*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", default="",
        help="入力predictions(未指定ならdata/logs/predictions.csv。"
             "新モデルの合成データ(scripts/build_new_model_predictions_snapshot.py)"
             "等、別ファイルを指定することもできる)。",
    )
    parser.add_argument("--results", default="", help="未指定ならdata/logs/results.csv")
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "data" / "logs" / "kelly_sizing_backtest.csv"),
        help="出力先(デフォルトは/simが読む本物のkelly_sizing_backtest.csvを上書きする)。",
    )
    args = parser.parse_args()

    print("loading data...")
    df = load_data(
        predictions_path=Path(args.predictions).expanduser().resolve() if args.predictions else None,
        results_path=Path(args.results).expanduser().resolve() if args.results else None,
    )
    print("rows:", len(df), "races:", df.drop_duplicates(["date", "venue", "race_no"]).shape[0])

    print("\nrunning bankroll simulation...")
    sim_df = simulate(df)

    out_path = Path(args.out).expanduser().resolve()
    sim_df.to_csv(out_path, index=False)
    print(f"saved race-by-race sim to {out_path}")

    print()
    summarize(sim_df)
