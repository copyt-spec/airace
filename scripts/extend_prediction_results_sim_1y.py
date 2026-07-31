from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"

BASE_SIM_CSV = LOG_DIR / "prediction_results_merged_sim_1y.csv"
INCREMENTAL_CSV = LOG_DIR / "prediction_results_merged.csv"
OUT_CSV = LOG_DIR / "prediction_results_merged_sim_1y.csv"

DEDUP_KEYS: List[str] = ["date", "venue", "race_no", "combo"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        out["date"] = (
            out["date"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(8)
            .str.strip()
        )

    if "venue" in out.columns:
        out["venue"] = out["venue"].astype(str).str.strip()

    if "combo" in out.columns:
        out["combo"] = out["combo"].astype(str).str.strip()

    if "actual_combo" in out.columns:
        out["actual_combo"] = out["actual_combo"].astype(str).str.strip()

    if "race_no" in out.columns:
        out["race_no"] = pd.to_numeric(out["race_no"], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        "rank_prob",
        "prob",
        "prob_pct",
        "odds",
        "expected_return_yen",
        "is_selected",
        "is_hit",
        "bet_cost_yen",
        "return_yen",
        "profit_yen",
        "payout",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _ensure_same_columns(base: pd.DataFrame, inc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_cols = list(dict.fromkeys(list(base.columns) + list(inc.columns)))

    for col in all_cols:
        if col not in base.columns:
            base[col] = pd.NA
        if col not in inc.columns:
            inc[col] = pd.NA

    base = base[all_cols].copy()
    inc = inc[all_cols].copy()
    return base, inc


def main() -> None:
    if not INCREMENTAL_CSV.exists():
        raise FileNotFoundError(f"missing incremental file: {INCREMENTAL_CSV}")

    if BASE_SIM_CSV.exists():
        base = pd.read_csv(BASE_SIM_CSV, low_memory=False)
        print("loaded base       :", BASE_SIM_CSV)
        print("base rows         :", len(base))
    else:
        base = pd.DataFrame()
        print("base file not found, start from empty.")

    inc = pd.read_csv(INCREMENTAL_CSV, low_memory=False)
    print("loaded incremental:", INCREMENTAL_CSV)
    print("incremental rows   :", len(inc))

    base = _normalize(base)
    inc = _normalize(inc)

    if base.empty:
        merged = inc.copy()
    else:
        base, inc = _ensure_same_columns(base, inc)

        merged = pd.concat([base, inc], ignore_index=True)

        sort_cols = DEDUP_KEYS.copy()
        if "logged_at" in merged.columns:
            sort_cols.append("logged_at")

        merged = merged.sort_values(sort_cols, na_position="last")
        merged = merged.drop_duplicates(subset=DEDUP_KEYS, keep="last")

    merged = _normalize(merged)

    final_sort_cols = [c for c in ["date", "venue", "race_no", "rank_prob", "combo"] if c in merged.columns]
    if final_sort_cols:
        merged = merged.sort_values(final_sort_cols, na_position="last").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("===== EXTEND DONE =====")
    print("saved            :", OUT_CSV)
    print("rows             :", len(merged))

    if "date" in merged.columns and not merged.empty:
        print("min date         :", merged["date"].min())
        print("max date         :", merged["date"].max())

    if "date" in inc.columns and not inc.empty:
        print("inc min date     :", inc["date"].min())
        print("inc max date     :", inc["date"].max())

    if "venue" in merged.columns and not merged.empty:
        print("\nrows by venue:")
        print(merged["venue"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
