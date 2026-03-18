from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.ev_calculator import (
    EVCalcConfig,
    merge_predictions_and_odds,
    calc_ev_table,
    build_ev_summary,
    build_top_ev_picks,
)


PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "render_model_predictions.csv"
ODDS_PATH = PROJECT_ROOT / "data" / "predictions" / "race_odds.csv"

OUT_DIR = PROJECT_ROOT / "data" / "analysis"
OUT_EV_ALL = OUT_DIR / "ev_all_predictions.csv"
OUT_EV_SUMMARY = OUT_DIR / "ev_summary.csv"
OUT_EV_TOP5 = OUT_DIR / "ev_top5_per_race.csv"
OUT_EV_TOP5_OVER_12 = OUT_DIR / "ev_top5_over_1_2.csv"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Could not read CSV: {path}\n{e}") from e


def _print_head(title: str, df: pd.DataFrame, n: int = 10) -> None:
    print(f"\n===== {title} =====")
    if df.empty:
        print("(empty)")
        return
    print(df.head(n).to_string(index=False))


def main() -> None:
    print("===== CALC EV FROM PREDICTIONS START =====")

    print("[1/7] loading prediction csv...")
    pred_df = _safe_read_csv(PRED_PATH)
    print(f"prediction shape: {pred_df.shape}")

    print("[2/7] loading odds csv...")
    odds_df = _safe_read_csv(ODDS_PATH)
    print(f"odds shape: {odds_df.shape}")

    print("[3/7] merging predictions and odds...")
    cfg = EVCalcConfig(
        pred_prob_col="pred_proba",
        odds_col="odds",
        combo_col="combo",
        race_key_col="race_key",
        min_odds=1.0,
        ev_col="ev",
        expected_return_col="expected_return",
        edge_col="edge",
        rank_col="ev_rank",
        hit_col="is_hit",
    )

    merged = merge_predictions_and_odds(pred_df, odds_df, cfg=cfg)
    print(f"merged shape: {merged.shape}")

    print("[4/7] calculating EV table...")
    ev_df = calc_ev_table(merged, cfg=cfg)
    print(f"ev table shape: {ev_df.shape}")

    print("[5/7] building reports...")
    summary_df = build_ev_summary(ev_df, cfg=cfg)
    top5_df = build_top_ev_picks(ev_df, cfg=cfg, top_n_per_race=5, min_ev=None)
    top5_over_12_df = build_top_ev_picks(ev_df, cfg=cfg, top_n_per_race=5, min_ev=1.2)

    print("[6/7] saving outputs...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ev_df.to_csv(OUT_EV_ALL, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_EV_SUMMARY, index=False, encoding="utf-8-sig")
    top5_df.to_csv(OUT_EV_TOP5, index=False, encoding="utf-8-sig")
    top5_over_12_df.to_csv(OUT_EV_TOP5_OVER_12, index=False, encoding="utf-8-sig")

    print("[7/7] done")

    _print_head("SUMMARY", summary_df, n=10)

    display_cols = [
        c for c in [
            "race_key",
            "date",
            "venue",
            "race_no",
            "combo",
            "pred_proba",
            "odds",
            "ev",
            "edge",
            "ev_rank",
            "is_hit",
        ]
        if c in top5_over_12_df.columns
    ]
    _print_head("TOP EV PICKS (EV >= 1.2)", top5_over_12_df[display_cols], n=30)

    print("\nSaved files:")
    print(f"- {OUT_EV_ALL}")
    print(f"- {OUT_EV_SUMMARY}")
    print(f"- {OUT_EV_TOP5}")
    print(f"- {OUT_EV_TOP5_OVER_12}")
    print("===== CALC EV FROM PREDICTIONS END =====")


if __name__ == "__main__":
    main()
