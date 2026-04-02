from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"

MERGED_LOG_PATH = LOG_DIR / "prediction_results_merged_sim_1y.csv"

HOME_STATS_JSON_PATH = LOG_DIR / "home_stats_1y.json"
SIM_LIGHT_CSV_PATH = LOG_DIR / "prediction_results_sim_1y_light.csv"


def _normalize_venue_name(v: str) -> str:
    s = str(v or "").strip()
    if "丸亀" in s:
        return "丸亀"
    if "戸田" in s:
        return "戸田"
    if "児島" in s:
        return "児島"
    return s


def _empty_stats_map() -> Dict[str, Dict[str, Any]]:
    venue_order = ["丸亀", "戸田", "児島"]
    empty_row = {
        "race_count": 0,
        "buy_count": 0,
        "hit_count": 0,
        "hit_rate": 0.0,
        "avg_points": 0.0,
        "total_bets": 0.0,
        "total_return": 0.0,
        "total_profit": 0.0,
        "roi": 0.0,
    }
    return {v: dict(empty_row) for v in venue_order}


def _aggregate_stats_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out = _empty_stats_map()

    if df.empty:
        return out

    selected_df = df[df["is_selected"] == 1].copy()
    if selected_df.empty:
        return out

    for venue in ["丸亀", "戸田", "児島"]:
        vdf = selected_df[selected_df["venue_norm"] == venue].copy()
        if vdf.empty:
            continue

        buy_count = int(len(vdf))
        hit_count = int((vdf["is_hit"] == 1).sum())
        total_bets = float(vdf["bet_cost_yen"].sum())
        total_return = float(vdf["return_yen"].sum())
        total_profit = total_return - total_bets
        hit_rate = hit_count / buy_count if buy_count > 0 else 0.0
        roi = total_return / total_bets if total_bets > 0 else 0.0

        race_count = 0
        if "date" in vdf.columns and "race_no" in vdf.columns:
            race_count = int(vdf[["date", "race_no"]].drop_duplicates().shape[0])

        avg_points = buy_count / race_count if race_count > 0 else 0.0

        out[venue] = {
            "race_count": race_count,
            "buy_count": buy_count,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "avg_points": avg_points,
            "total_bets": total_bets,
            "total_return": total_return,
            "total_profit": total_profit,
            "roi": roi,
        }

    return out


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not MERGED_LOG_PATH.exists():
        raise FileNotFoundError(f"Missing merged log: {MERGED_LOG_PATH}")

    print("Loading:", MERGED_LOG_PATH)

    usecols = [
        "date",
        "venue",
        "race_no",
        "is_selected",
        "is_hit",
        "bet_cost_yen",
        "return_yen",
        "profit_yen",
    ]

    df = pd.read_csv(MERGED_LOG_PATH, usecols=usecols, low_memory=False)

    if df.empty:
        raise RuntimeError("Merged log is empty.")

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df["venue_norm"] = df["venue"].astype(str).map(_normalize_venue_name)

    for col in ["is_selected", "is_hit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["bet_cost_yen", "return_yen", "profit_yen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    stats = _aggregate_stats_from_df(df)

    home_json = {
        "source": str(MERGED_LOG_PATH.name),
        "available_min_date": str(df["date"].min()) if "date" in df.columns else "",
        "available_max_date": str(df["date"].max()) if "date" in df.columns else "",
        "row_count": int(len(df)),
        "stats": stats,
    }

    with open(HOME_STATS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(home_json, f, ensure_ascii=False, indent=2)

    print("Saved home stats json:", HOME_STATS_JSON_PATH)

    light_cols = [
        "date",
        "venue",
        "race_no",
        "is_selected",
        "is_hit",
        "bet_cost_yen",
        "return_yen",
        "profit_yen",
    ]
    light_df = df[light_cols].copy()
    light_df.to_csv(SIM_LIGHT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Saved sim light csv :", SIM_LIGHT_CSV_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
