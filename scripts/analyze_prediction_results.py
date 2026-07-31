from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"
DEFAULT_SRC = LOG_DIR / "prediction_results_merged.csv"
DEFAULT_OUT_DIR = LOG_DIR / "analysis"


VENUE_ORDER = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
    "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
    "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _normalize_venue_name(v: str) -> str:
    s = str(v or "").strip()
    for venue in VENUE_ORDER:
        if venue in s:
            return venue
    return s


def _load_merged_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing merged csv: {path}")

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError("prediction_results_merged.csv is empty")

    for col in ["date", "venue", "race_no", "combo"]:
        if col not in df.columns:
            raise RuntimeError(f"missing required column: {col}")

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df["venue"] = df["venue"].astype(str).map(_normalize_venue_name)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)
    df["combo"] = df["combo"].astype(str).str.strip()

    for col in ["is_selected", "is_hit", "rank_prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    for col in [
        "prob",
        "prob_pct",
        "odds",
        "expected_return_yen",
        "bet_cost_yen",
        "return_yen",
        "profit_yen",
        "payout",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    if "model_name" not in df.columns:
        df["model_name"] = ""

    return df


def _build_selected_df(df: pd.DataFrame) -> pd.DataFrame:
    selected = df[df["is_selected"] == 1].copy()
    if selected.empty:
        return selected

    selected["roi"] = selected.apply(
        lambda r: float(r["return_yen"] / r["bet_cost_yen"])
        if r["bet_cost_yen"] > 0 else 0.0,
        axis=1,
    )

    selected["ev_raw"] = selected.apply(
        lambda r: float(r["prob"] * r["odds"])
        if r["prob"] > 0 and r["odds"] > 0 else 0.0,
        axis=1,
    )

    selected["prob_pct_calc"] = selected["prob"] * 100.0

    return selected


def _build_race_level_df(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()

    race_df = (
        selected.groupby(["date", "venue", "race_no"], as_index=False)
        .agg(
            race_buy_count=("combo", "count"),
            hit_count=("is_hit", "sum"),
            total_bets=("bet_cost_yen", "sum"),
            total_return=("return_yen", "sum"),
            total_profit=("profit_yen", "sum"),
            avg_prob=("prob", "mean"),
            max_prob=("prob", "max"),
            avg_odds=("odds", "mean"),
            max_odds=("odds", "max"),
            avg_ev_raw=("ev_raw", "mean"),
            max_ev_raw=("ev_raw", "max"),
            model_name=("model_name", "first"),
        )
        .sort_values(["date", "venue", "race_no"])
        .reset_index(drop=True)
    )

    race_df["is_hit_race"] = (race_df["hit_count"] > 0).astype(int)
    race_df["roi"] = race_df.apply(
        lambda r: float(r["total_return"] / r["total_bets"])
        if r["total_bets"] > 0 else 0.0,
        axis=1,
    )

    return race_df

def _add_band_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def point_band(x: Any) -> str:
        n = _safe_int(x, 0)
        if n <= 0:
            return "0"
        if n <= 3:
            return "1-3"
        if n <= 5:
            return "4-5"
        if n <= 8:
            return "6-8"
        if n <= 12:
            return "9-12"
        if n <= 20:
            return "13-20"
        return "21+"

    def ev_band(x: Any) -> str:
        v = _safe_float(x, 0.0)
        if v < 0.5:
            return "<0.5"
        if v < 0.8:
            return "0.5-0.8"
        if v < 1.0:
            return "0.8-1.0"
        if v < 1.2:
            return "1.0-1.2"
        if v < 1.5:
            return "1.2-1.5"
        if v < 2.0:
            return "1.5-2.0"
        return "2.0+"

    def prob_band(x: Any) -> str:
        v = _safe_float(x, 0.0)
        if v < 0.01:
            return "<1%"
        if v < 0.02:
            return "1-2%"
        if v < 0.03:
            return "2-3%"
        if v < 0.05:
            return "3-5%"
        if v < 0.08:
            return "5-8%"
        if v < 0.12:
            return "8-12%"
        return "12%+"

    def odds_band(x: Any) -> str:
        v = _safe_float(x, 0.0)
        if v <= 0:
            return "0"
        if v < 5:
            return "<5"
        if v < 10:
            return "5-10"
        if v < 20:
            return "10-20"
        if v < 30:
            return "20-30"
        if v < 50:
            return "30-50"
        if v < 100:
            return "50-100"
        return "100+"

    if "race_buy_count" in out.columns:
        out["point_band"] = out["race_buy_count"].map(point_band)
    else:
        out["point_band"] = "NA"

    if "ev_raw" in out.columns:
        ev_src = out["ev_raw"]
    elif "max_ev_raw" in out.columns:
        ev_src = out["max_ev_raw"]
    elif "avg_ev_raw" in out.columns:
        ev_src = out["avg_ev_raw"]
    else:
        ev_src = pd.Series([0.0] * len(out), index=out.index)

    if "prob" in out.columns:
        prob_src = out["prob"]
    elif "max_prob" in out.columns:
        prob_src = out["max_prob"]
    elif "avg_prob" in out.columns:
        prob_src = out["avg_prob"]
    else:
        prob_src = pd.Series([0.0] * len(out), index=out.index)

    if "odds" in out.columns:
        odds_src = out["odds"]
    elif "max_odds" in out.columns:
        odds_src = out["max_odds"]
    elif "avg_odds" in out.columns:
        odds_src = out["avg_odds"]
    else:
        odds_src = pd.Series([0.0] * len(out), index=out.index)

    out["ev_band"] = ev_src.map(ev_band)
    out["prob_band"] = prob_src.map(prob_band)
    out["odds_band"] = odds_src.map(odds_band)

    return out


def _aggregate_selected_by_venue(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()

    out = (
        selected.groupby("venue", as_index=False)
        .agg(
            buy_count=("combo", "count"),
            hit_count=("is_hit", "sum"),
            total_bets=("bet_cost_yen", "sum"),
            total_return=("return_yen", "sum"),
            total_profit=("profit_yen", "sum"),
            avg_prob=("prob", "mean"),
            avg_odds=("odds", "mean"),
            avg_ev_raw=("ev_raw", "mean"),
            max_ev_raw=("ev_raw", "max"),
            race_count=("race_no", "nunique"),
        )
        .copy()
    )

    out["hit_rate"] = out.apply(
        lambda r: float(r["hit_count"] / r["buy_count"])
        if r["buy_count"] > 0 else 0.0,
        axis=1,
    )
    out["roi"] = out.apply(
        lambda r: float(r["total_return"] / r["total_bets"])
        if r["total_bets"] > 0 else 0.0,
        axis=1,
    )
    out["avg_points"] = out.apply(
        lambda r: float(r["buy_count"] / r["race_count"])
        if r["race_count"] > 0 else 0.0,
        axis=1,
    )

    order_map = {v: i for i, v in enumerate(VENUE_ORDER)}
    out["venue_order"] = out["venue"].map(lambda x: order_map.get(x, 999))
    out = out.sort_values(["venue_order", "venue"]).drop(columns=["venue_order"]).reset_index(drop=True)

    return out


def _aggregate_race_by_venue(race_df: pd.DataFrame) -> pd.DataFrame:
    if race_df.empty:
        return pd.DataFrame()

    out = (
        race_df.groupby("venue", as_index=False)
        .agg(
            race_count=("race_no", "count"),
            hit_race_count=("is_hit_race", "sum"),
            total_bets=("total_bets", "sum"),
            total_return=("total_return", "sum"),
            total_profit=("total_profit", "sum"),
            avg_points=("race_buy_count", "mean"),
            avg_prob=("avg_prob", "mean"),
            avg_odds=("avg_odds", "mean"),
            avg_ev_raw=("avg_ev_raw", "mean"),
            max_ev_raw=("max_ev_raw", "max"),
        )
        .copy()
    )

    out["hit_race_rate"] = out.apply(
        lambda r: float(r["hit_race_count"] / r["race_count"])
        if r["race_count"] > 0 else 0.0,
        axis=1,
    )
    out["roi"] = out.apply(
        lambda r: float(r["total_return"] / r["total_bets"])
        if r["total_bets"] > 0 else 0.0,
        axis=1,
    )

    order_map = {v: i for i, v in enumerate(VENUE_ORDER)}
    out["venue_order"] = out["venue"].map(lambda x: order_map.get(x, 999))
    out = out.sort_values(["venue_order", "venue"]).drop(columns=["venue_order"]).reset_index(drop=True)

    return out


def _aggregate_band(df: pd.DataFrame, band_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(band_col, as_index=False)
        .agg(
            race_count=("race_no", "count"),
            hit_count=("is_hit_race", "sum"),
            total_bets=("total_bets", "sum"),
            total_return=("total_return", "sum"),
            total_profit=("total_profit", "sum"),
        )
        .copy()
    )

    out["hit_rate"] = out.apply(
        lambda r: float(r["hit_count"] / r["race_count"])
        if r["race_count"] > 0 else 0.0,
        axis=1,
    )
    out["roi"] = out.apply(
        lambda r: float(r["total_return"] / r["total_bets"])
        if r["total_bets"] > 0 else 0.0,
        axis=1,
    )

    return out
def _aggregate_band_by_venue(df: pd.DataFrame, band_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(["venue", band_col], as_index=False)
        .agg(
            race_count=("race_no", "count"),
            hit_count=("is_hit_race", "sum"),
            total_bets=("total_bets", "sum"),
            total_return=("total_return", "sum"),
            total_profit=("total_profit", "sum"),
        )
        .copy()
    )

    out["hit_rate"] = out.apply(
        lambda r: float(r["hit_count"] / r["race_count"])
        if r["race_count"] > 0 else 0.0,
        axis=1,
    )
    out["roi"] = out.apply(
        lambda r: float(r["total_return"] / r["total_bets"])
        if r["total_bets"] > 0 else 0.0,
        axis=1,
    )

    return out


def _build_daily_summary(race_df: pd.DataFrame) -> pd.DataFrame:
    if race_df.empty:
        return pd.DataFrame()

    out = (
        race_df.groupby("date", as_index=False)
        .agg(
            race_count=("race_no", "count"),
            hit_race_count=("is_hit_race", "sum"),
            total_bets=("total_bets", "sum"),
            total_return=("total_return", "sum"),
            total_profit=("total_profit", "sum"),
            avg_points=("race_buy_count", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    out["hit_race_rate"] = out.apply(
        lambda r: float(r["hit_race_count"] / r["race_count"])
        if r["race_count"] > 0 else 0.0,
        axis=1,
    )
    out["roi"] = out.apply(
        lambda r: float(r["total_return"] / r["total_bets"])
        if r["total_bets"] > 0 else 0.0,
        axis=1,
    )
    out["cum_profit"] = out["total_profit"].cumsum()

    return out


def _build_top_worst_races(
    race_df: pd.DataFrame,
    top_n: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if race_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    worst = (
        race_df.sort_values(
            ["total_profit", "date", "venue", "race_no"],
            ascending=[True, True, True, True],
        )
        .head(top_n)
        .copy()
    )

    best = (
        race_df.sort_values(
            ["total_profit", "date", "venue", "race_no"],
            ascending=[False, True, True, True],
        )
        .head(top_n)
        .copy()
    )

    return worst, best


def _pick_worst(df: pd.DataFrame, band_col: str) -> Dict[str, Any]:
    if df.empty or band_col not in df.columns:
        return {}

    work = df.sort_values(
        ["roi", "total_profit"],
        ascending=[True, True],
    ).reset_index(drop=True)

    row = work.iloc[0]

    return {
        band_col: row[band_col],
        "roi": float(row["roi"]),
        "total_profit": float(row["total_profit"]),
        "race_count": int(row["race_count"]),
    }


def _pick_best(df: pd.DataFrame, band_col: str) -> Dict[str, Any]:
    if df.empty or band_col not in df.columns:
        return {}

    work = df.sort_values(
        ["roi", "total_profit"],
        ascending=[False, False],
    ).reset_index(drop=True)

    row = work.iloc[0]

    return {
        band_col: row[band_col],
        "roi": float(row["roi"]),
        "total_profit": float(row["total_profit"]),
        "race_count": int(row["race_count"]),
    }


def _build_summary_json(
    selected: pd.DataFrame,
    race_df: pd.DataFrame,
    venue_df: pd.DataFrame,
    point_band_df: pd.DataFrame,
    ev_band_df: pd.DataFrame,
    prob_band_df: pd.DataFrame,
    odds_band_df: pd.DataFrame,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    summary["selected_rows"] = int(len(selected))
    summary["race_count"] = int(len(race_df))
    summary["venue_count"] = (
        int(venue_df["venue"].nunique())
        if not venue_df.empty and "venue" in venue_df.columns
        else 0
    )

    if not selected.empty:
        total_bets = float(selected["bet_cost_yen"].sum())
        total_return = float(selected["return_yen"].sum())

        summary["selected_hit_count"] = int(selected["is_hit"].sum())
        summary["selected_hit_rate"] = float(selected["is_hit"].mean())
        summary["selected_total_bets"] = total_bets
        summary["selected_total_return"] = total_return
        summary["selected_total_profit"] = float(selected["profit_yen"].sum())
        summary["selected_roi"] = (
            float(total_return / total_bets)
            if total_bets > 0
            else 0.0
        )
    else:
        summary["selected_hit_count"] = 0
        summary["selected_hit_rate"] = 0.0
        summary["selected_total_bets"] = 0.0
        summary["selected_total_return"] = 0.0
        summary["selected_total_profit"] = 0.0
        summary["selected_roi"] = 0.0

    if not race_df.empty:
        summary["race_hit_count"] = int(race_df["is_hit_race"].sum())
        summary["race_hit_rate"] = float(race_df["is_hit_race"].mean())
        summary["avg_points"] = float(race_df["race_buy_count"].mean())
    else:
        summary["race_hit_count"] = 0
        summary["race_hit_rate"] = 0.0
        summary["avg_points"] = 0.0

    summary["worst_point_band"] = _pick_worst(point_band_df, "point_band")
    summary["best_point_band"] = _pick_best(point_band_df, "point_band")
    summary["worst_ev_band"] = _pick_worst(ev_band_df, "ev_band")
    summary["best_ev_band"] = _pick_best(ev_band_df, "ev_band")
    summary["worst_prob_band"] = _pick_worst(prob_band_df, "prob_band")
    summary["best_prob_band"] = _pick_best(prob_band_df, "prob_band")
    summary["worst_odds_band"] = _pick_worst(odds_band_df, "odds_band")
    summary["best_odds_band"] = _pick_best(odds_band_df, "odds_band")

    if not venue_df.empty:
        worst_venue = venue_df.sort_values(
            ["roi", "total_profit"],
            ascending=[True, True],
        ).iloc[0]

        best_venue = venue_df.sort_values(
            ["roi", "total_profit"],
            ascending=[False, False],
        ).iloc[0]

        summary["worst_venue"] = {
            "venue": worst_venue["venue"],
            "roi": float(worst_venue["roi"]),
            "total_profit": float(worst_venue["total_profit"]),
            "race_count": int(worst_venue["race_count"]),
        }

        summary["best_venue"] = {
            "venue": best_venue["venue"],
            "roi": float(best_venue["roi"]),
            "total_profit": float(best_venue["total_profit"]),
            "race_count": int(best_venue["race_count"]),
        }
    else:
        summary["worst_venue"] = {}
        summary["best_venue"] = {}

    return summary


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="prediction_results_merged.csv path")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="output directory")
    parser.add_argument("--start-date", default="", help="YYYYMMDD")
    parser.add_argument("--end-date", default="", help="YYYYMMDD")
    parser.add_argument("--venue", default="", help="例: 丸亀")
    parser.add_argument("--model-name", default="", help="model_name filter")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading:", src)
    df = _load_merged_csv(src)

    if args.start_date:
        df = df[df["date"] >= str(args.start_date).strip()].copy()

    if args.end_date:
        df = df[df["date"] <= str(args.end_date).strip()].copy()

    if args.venue:
        venue = _normalize_venue_name(args.venue)
        df = df[df["venue"] == venue].copy()

    if args.model_name:
        df = df[df["model_name"].astype(str) == str(args.model_name)].copy()

    if df.empty:
        raise RuntimeError("filtered dataframe is empty")

    print("rows loaded:", len(df))

    selected = _build_selected_df(df)
    if selected.empty:
        raise RuntimeError("selected rows are empty. is_selected=1 not found")

    race_df = _build_race_level_df(selected)
    race_df = _add_band_columns(race_df)

    venue_selected_df = _aggregate_selected_by_venue(selected)
    venue_race_df = _aggregate_race_by_venue(race_df)

    point_band_df = _aggregate_band(race_df, "point_band")
    ev_band_df = _aggregate_band(race_df, "ev_band")
    prob_band_df = _aggregate_band(race_df, "prob_band")
    odds_band_df = _aggregate_band(race_df, "odds_band")

    point_band_by_venue_df = _aggregate_band_by_venue(race_df, "point_band")
    ev_band_by_venue_df = _aggregate_band_by_venue(race_df, "ev_band")
    prob_band_by_venue_df = _aggregate_band_by_venue(race_df, "prob_band")
    odds_band_by_venue_df = _aggregate_band_by_venue(race_df, "odds_band")

    daily_df = _build_daily_summary(race_df)
    worst_races_df, best_races_df = _build_top_worst_races(race_df, top_n=30)

    summary = _build_summary_json(
        selected=selected,
        race_df=race_df,
        venue_df=venue_race_df,
        point_band_df=point_band_df,
        ev_band_df=ev_band_df,
        prob_band_df=prob_band_df,
        odds_band_df=odds_band_df,
    )

    _save_df(venue_selected_df, out_dir / "01_selected_by_venue.csv")
    _save_df(venue_race_df, out_dir / "02_race_by_venue.csv")
    _save_df(point_band_df, out_dir / "03_point_band_summary.csv")
    _save_df(ev_band_df, out_dir / "04_ev_band_summary.csv")
    _save_df(prob_band_df, out_dir / "05_prob_band_summary.csv")
    _save_df(odds_band_df, out_dir / "06_odds_band_summary.csv")
    _save_df(point_band_by_venue_df, out_dir / "07_point_band_by_venue.csv")
    _save_df(ev_band_by_venue_df, out_dir / "08_ev_band_by_venue.csv")
    _save_df(prob_band_by_venue_df, out_dir / "09_prob_band_by_venue.csv")
    _save_df(odds_band_by_venue_df, out_dir / "10_odds_band_by_venue.csv")
    _save_df(daily_df, out_dir / "11_daily_summary.csv")
    _save_df(worst_races_df, out_dir / "12_worst_races.csv")
    _save_df(best_races_df, out_dir / "13_best_races.csv")
    _save_df(race_df, out_dir / "14_race_level_detail.csv")

    with open(out_dir / "00_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("out_dir:", out_dir)
    print("selected rows:", len(selected))
    print("race count   :", len(race_df))
    print("venue count  :", venue_race_df["venue"].nunique() if not venue_race_df.empty else 0)

    print("-" * 80)
    print("summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
