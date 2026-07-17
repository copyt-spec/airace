from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"

MERGED_LOG_PATH = LOG_DIR / "prediction_results_merged.csv"

HOME_STATS_JSON_PATH = LOG_DIR / "home_stats_1y.json"
SIM_LIGHT_CSV_PATH = LOG_DIR / "prediction_results_sim_1y_light.csv"


VENUE_ORDER = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
    "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
    "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
]


def _normalize_venue_name(v: str) -> str:
    s = str(v or "").strip()
    for venue in VENUE_ORDER:
        if venue in s:
            return venue
    return s


def _empty_stats_map() -> Dict[str, Dict[str, Any]]:
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
    return {v: dict(empty_row) for v in VENUE_ORDER}


def _aggregate_stats_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out = _empty_stats_map()

    if df.empty:
        return out

    selected_df = df[df["is_selected"] == 1].copy()
    if selected_df.empty:
        return out

    for venue in VENUE_ORDER:
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


def _compute_race_confidence(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    レース単位の「信頼度tier」を、ログ済みの全買い目(is_selectedに関わらず)から
    近似的に算出する。

    本番の _build_race_signal (app/main.py) は top_prob / prob_gap / top_ev の
    3指標を使うが、top_ev はその時点のEVロジックに依存し過去ログから厳密に
    再現できないため、ここでは top_prob と prob_gap の2指標のみで近似する。
    (=「近似tier」であり本番のA/B/C/D判定と完全一致はしない)
    """
    key = ["date", "venue", "race_no"]
    work = full_df[key + ["rank_prob", "prob"]].copy()
    work["rank_prob"] = pd.to_numeric(work["rank_prob"], errors="coerce")
    work["prob"] = pd.to_numeric(work["prob"], errors="coerce").fillna(0.0)

    top1 = (
        work[work["rank_prob"] == 1][key + ["prob"]]
        .rename(columns={"prob": "top1_prob"})
        .drop_duplicates(subset=key)
    )
    top2 = (
        work[work["rank_prob"] == 2][key + ["prob"]]
        .rename(columns={"prob": "second_prob"})
        .drop_duplicates(subset=key)
    )

    sig = top1.merge(top2, on=key, how="left")
    sig["second_prob"] = sig["second_prob"].fillna(0.0)
    sig["prob_gap"] = sig["top1_prob"] - sig["second_prob"]

    def _tier(row) -> str:
        p, gap = row["top1_prob"], row["prob_gap"]
        if p >= 0.11 and gap >= 0.020:
            return "A"
        if p >= 0.08 and gap >= 0.012:
            return "B"
        if p >= 0.055:
            return "C"
        return "D"

    sig["conf_tier"] = sig.apply(_tier, axis=1)
    return sig


def _bucket_odds(odds: float) -> str:
    try:
        o = float(odds)
    except Exception:
        return "不明"
    if o < 10:
        return "〜10倍"
    if o < 30:
        return "10〜30倍"
    if o < 50:
        return "30〜50倍"
    if o < 100:
        return "50〜100倍"
    return "100倍〜"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not MERGED_LOG_PATH.exists():
        raise FileNotFoundError(f"Missing merged log: {MERGED_LOG_PATH}")

    print("Loading:", MERGED_LOG_PATH)

    usecols = [
        "date",
        "venue",
        "race_no",
        "combo",
        "rank_prob",
        "prob",
        "odds",
        "is_selected",
        "is_hit",
        "bet_cost_yen",
        "return_yen",
        "profit_yen",
        "has_result",
    ]

    full_df = pd.read_csv(MERGED_LOG_PATH, usecols=usecols, low_memory=False)

    if full_df.empty:
        raise RuntimeError("Merged log is empty.")

    full_df["date"] = full_df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    full_df["venue"] = full_df["venue"].astype(str).str.strip()

    print("Computing per-race confidence tier (approx)...")
    race_signal = _compute_race_confidence(full_df)

    df = full_df[pd.to_numeric(full_df["is_selected"], errors="coerce").fillna(0) == 1].copy()

    # 結果(K-file)がまだ届いていない「未確定」レースは、ハズレ扱いにされて
    # ROIが不当に下がるのを防ぐため集計から除外する。
    if "has_result" in df.columns:
        before = len(df)
        has_result_mask = df["has_result"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
        df = df[has_result_mask].copy()
        excluded = before - len(df)
        if excluded:
            print(f"excluded pending (no result yet) rows: {excluded}")

    df = df.merge(race_signal[["date", "venue", "race_no", "top1_prob", "prob_gap", "conf_tier"]],
                  on=["date", "venue", "race_no"], how="left")

    df["venue_norm"] = df["venue"].map(_normalize_venue_name)

    for col in ["is_selected", "is_hit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["bet_cost_yen", "return_yen", "profit_yen", "prob", "odds", "top1_prob", "prob_gap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["odds_band"] = df["odds"].map(_bucket_odds)
    df["conf_tier"] = df["conf_tier"].fillna("D")

    stats = _aggregate_stats_from_df(df)

    home_json = {
        "source": str(MERGED_LOG_PATH.name),
        "available_min_date": str(df["date"].min()) if "date" in df.columns else "",
        "available_max_date": str(df["date"].max()) if "date" in df.columns else "",
        "row_count": int(len(df)),
        "venue_count": int(df["venue_norm"].nunique()) if "venue_norm" in df.columns else 0,
        "stats": stats,
    }

    with open(HOME_STATS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(home_json, f, ensure_ascii=False, indent=2)

    print("Saved home stats json:", HOME_STATS_JSON_PATH)

    light_cols = [
        "date",
        "venue",
        "race_no",
        "combo",
        "prob",
        "odds",
        "odds_band",
        "top1_prob",
        "prob_gap",
        "conf_tier",
        "is_selected",
        "is_hit",
        "bet_cost_yen",
        "return_yen",
        "profit_yen",
    ]
    light_df = df[light_cols].copy()
    light_df.to_csv(SIM_LIGHT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Saved sim light csv :", SIM_LIGHT_CSV_PATH)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("row_count       :", len(df))
    print("available_min   :", home_json["available_min_date"])
    print("available_max   :", home_json["available_max_date"])
    print("venue_count     :", home_json["venue_count"])

    used_venues = [v for v in VENUE_ORDER if stats.get(v, {}).get("race_count", 0) > 0]
    print("used venues     :", ", ".join(used_venues))


if __name__ == "__main__":
    main()
