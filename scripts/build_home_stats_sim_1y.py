# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from engine.buy_selector import select_best_bets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
OUTPUT_PATH = LOG_DIR / "prediction_results_merged_sim_1y.csv"

BEST_MODEL_SUFFIX = {
    "丸亀": "with_racer_no",
    "戸田": "with_racer_no",
    "児島": "with_racer_no",
}

VENUES = ["丸亀", "戸田", "児島"]
ROWS_PER_RACE = 120


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError as e:
        raise RuntimeError("catboost が入っていません。 `pip install catboost`") from e


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


def normalize_venue(v: Any) -> str:
    s = str(v or "").strip()
    if "丸亀" in s:
        return "丸亀"
    if "戸田" in s:
        return "戸田"
    if "児島" in s:
        return "児島"
    return s


def sanitize_x(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0

    out = out[feature_cols].copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)
        else:
            out[col] = out[col].astype(str).fillna("")

    return out


def add_feature_block(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # combo
    if "combo" in out.columns:
        parts = out["combo"].astype(str).str.split("-", expand=True)
        if parts.shape[1] >= 3:
            out["combo_first_lane"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype("int16")
            out["combo_second_lane"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype("int16")
            out["combo_third_lane"] = pd.to_numeric(parts[2], errors="coerce").fillna(0).astype("int16")
        else:
            out["combo_first_lane"] = 0
            out["combo_second_lane"] = 0
            out["combo_third_lane"] = 0

    # 風向コード
    wind_map = {
        "無風": 0,
        "北": 1, "北東": 2, "東": 3, "南東": 4,
        "南": 5, "南西": 6, "西": 7, "北西": 8,
    }
    if "wind_dir" in out.columns:
        out["wind_dir_code"] = out["wind_dir"].astype(str).map(wind_map).fillna(0).astype("int16")
    else:
        out["wind_dir_code"] = 0

    # 天候コード
    weather_map = {
        "晴": 1, "晴れ": 1,
        "曇": 2, "曇り": 2,
        "雨": 3,
        "雪": 4,
    }
    if "weather" in out.columns:
        out["weather_code"] = out["weather"].astype(str).map(weather_map).fillna(0).astype("int16")
    else:
        out["weather_code"] = 0

    for lane in range(1, 7):
        st_col = f"lane{lane}_st"
        ex_col = f"lane{lane}_exhibit"
        course_col = f"lane{lane}_course"
        motor_col = f"lane{lane}_motor"
        boat_col = f"lane{lane}_boat"
        racer_col = f"lane{lane}_racer_no"

        for c in [st_col, ex_col, course_col, motor_col, boat_col, racer_col]:
            if c not in out.columns:
                out[c] = 0

        out[st_col] = pd.to_numeric(out[st_col], errors="coerce").fillna(0).astype("float32")
        out[ex_col] = pd.to_numeric(out[ex_col], errors="coerce").fillna(0).astype("float32")
        out[course_col] = pd.to_numeric(out[course_col], errors="coerce").fillna(0).astype("int16")
        out[motor_col] = pd.to_numeric(out[motor_col], errors="coerce").fillna(0).astype("int16")
        out[boat_col] = pd.to_numeric(out[boat_col], errors="coerce").fillna(0).astype("int16")
        out[racer_col] = pd.to_numeric(out[racer_col], errors="coerce").fillna(0).astype("int32")

        out[f"lane{lane}_st_inv"] = (0.3 - out[st_col]).clip(lower=-1, upper=1).astype("float32")
        out[f"lane{lane}_exhibit_inv"] = (7.5 - out[ex_col]).clip(lower=-2, upper=2).astype("float32")
        out[f"lane{lane}_course_diff"] = (out[course_col] - lane).astype("int16")

    # combo対象laneの特徴
    for pos, pos_name in [
        ("first", "combo_first_lane"),
        ("second", "combo_second_lane"),
        ("third", "combo_third_lane"),
    ]:
        out[f"{pos}_st"] = 0.0
        out[f"{pos}_exhibit"] = 0.0
        out[f"{pos}_course"] = 0
        out[f"{pos}_motor"] = 0
        out[f"{pos}_boat"] = 0
        out[f"{pos}_racer_no"] = 0

        for lane in range(1, 7):
            mask = out[pos_name] == lane
            out.loc[mask, f"{pos}_st"] = out.loc[mask, f"lane{lane}_st"]
            out.loc[mask, f"{pos}_exhibit"] = out.loc[mask, f"lane{lane}_exhibit"]
            out.loc[mask, f"{pos}_course"] = out.loc[mask, f"lane{lane}_course"]
            out.loc[mask, f"{pos}_motor"] = out.loc[mask, f"lane{lane}_motor"]
            out.loc[mask, f"{pos}_boat"] = out.loc[mask, f"lane{lane}_boat"]
            out.loc[mask, f"{pos}_racer_no"] = out.loc[mask, f"lane{lane}_racer_no"]

    # 差分
    out["first_second_st_diff"] = (out["first_st"] - out["second_st"]).astype("float32")
    out["first_third_st_diff"] = (out["first_st"] - out["third_st"]).astype("float32")
    out["first_second_exhibit_diff"] = (out["first_exhibit"] - out["second_exhibit"]).astype("float32")
    out["first_third_exhibit_diff"] = (out["first_exhibit"] - out["third_exhibit"]).astype("float32")

    if "wind_speed_mps" not in out.columns:
        out["wind_speed_mps"] = 0.0
    if "wave_cm" not in out.columns:
        out["wave_cm"] = 0.0

    out["wind_speed_mps"] = pd.to_numeric(out["wind_speed_mps"], errors="coerce").fillna(0).astype("float32")
    out["wave_cm"] = pd.to_numeric(out["wave_cm"], errors="coerce").fillna(0).astype("float32")

    return out


def load_model_and_meta(venue: str):
    CatBoostClassifier = _require_catboost()
    suffix = BEST_MODEL_SUFFIX[venue]

    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_{suffix}.cbm"
    meta_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_{suffix}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = list(meta.get("feature_cols", []))
    if not feature_cols:
        raise RuntimeError(f"feature_cols missing in {meta_path}")

    return model, feature_cols


def get_latest_date_and_cutoff() -> Tuple[str, str]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    max_date = None
    for chunk in pd.read_csv(DATASET_PATH, usecols=["date"], chunksize=300000, low_memory=False):
        s = chunk["date"].astype(str)
        local_max = s.max()
        if max_date is None or local_max > max_date:
            max_date = local_max

    if max_date is None:
        raise RuntimeError("date not found in dataset")

    max_dt = datetime.strptime(max_date, "%Y%m%d")
    cutoff_dt = max_dt - timedelta(days=365)

    return max_dt.strftime("%Y%m%d"), cutoff_dt.strftime("%Y%m%d")


def load_one_year_venue_df(venue: str, cutoff_date: str) -> pd.DataFrame:
    usecols = [
        "date", "venue", "race_key", "race_no", "combo", "y", "trifecta_payout",
        "wave_cm", "weather", "wind_dir", "wind_speed_mps",
        "lane1_boat", "lane1_course", "lane1_exhibit", "lane1_motor", "lane1_racer_no", "lane1_st",
        "lane2_boat", "lane2_course", "lane2_exhibit", "lane2_motor", "lane2_racer_no", "lane2_st",
        "lane3_boat", "lane3_course", "lane3_exhibit", "lane3_motor", "lane3_racer_no", "lane3_st",
        "lane4_boat", "lane4_course", "lane4_exhibit", "lane4_motor", "lane4_racer_no", "lane4_st",
        "lane5_boat", "lane5_course", "lane5_exhibit", "lane5_motor", "lane5_racer_no", "lane5_st",
        "lane6_boat", "lane6_course", "lane6_exhibit", "lane6_motor", "lane6_racer_no", "lane6_st",
    ]

    chunks: List[pd.DataFrame] = []
    total = 0

    for chunk in pd.read_csv(DATASET_PATH, usecols=usecols, chunksize=250000, low_memory=False):
        chunk["date"] = chunk["date"].astype(str)
        chunk["venue"] = chunk["venue"].astype(str).map(normalize_venue)

        chunk = chunk[
            (chunk["venue"] == venue) &
            (chunk["date"] >= cutoff_date)
        ].copy()

        if chunk.empty:
            continue

        chunk["y"] = pd.to_numeric(chunk["y"], errors="coerce").fillna(0).astype("int8")
        chunk["trifecta_payout"] = pd.to_numeric(chunk["trifecta_payout"], errors="coerce").fillna(0).astype("float32")

        chunks.append(chunk)
        total += len(chunk)
        print(f"[{venue}] loaded rows: {total}")

    if not chunks:
        raise RuntimeError(f"{venue}: no rows found for last 1 year")

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["date", "race_key", "combo"]).reset_index(drop=True)
    return df


def build_prob_map(model, feature_cols: List[str], df120: pd.DataFrame) -> Dict[str, float]:
    work = add_feature_block(df120.copy())
    x = sanitize_x(work, feature_cols)
    prob = model.predict_proba(x)[:, 1]
    combos = work["combo"].astype(str).tolist()

    prob_map = {combo: float(p) for combo, p in zip(combos, prob)}
    total = sum(prob_map.values())
    if total > 0:
        prob_map = {k: v / total for k, v in prob_map.items()}
    return prob_map


def build_ai_preds_with_proxy_odds(prob_map: Dict[str, float], odds_proxy_scale: float = 100.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for combo, prob in prob_map.items():
        pseudo_odds = 1.0 / max(prob, 1e-6)
        pseudo_odds = min(pseudo_odds, odds_proxy_scale)

        rows.append({
            "combo": combo,
            "prob": float(prob),
            "score": float(prob),
            "odds": float(pseudo_odds),
            "ev": float(prob) * float(pseudo_odds),
        })

    return rows


def simulate_one_venue(
    venue: str,
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows_out: List[Dict[str, Any]] = []
    race_count = 0

    for race_key, df120 in df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue

        date = str(df120["date"].iloc[0])
        race_no = _safe_int(df120["race_no"].iloc[0], 0)
        payout = _safe_float(df120["trifecta_payout"].iloc[0], 0.0)

        hit_rows = df120[df120["y"] == 1]
        if hit_rows.empty:
            continue
        actual_combo = str(hit_rows.iloc[0]["combo"])

        prob_map = build_prob_map(model, feature_cols, df120)
        ai_preds = build_ai_preds_with_proxy_odds(prob_map)

        best_bets = select_best_bets(
            ai_preds=ai_preds,
            venue=venue,
            top_n=12,
        )

        selected_combo_set = {str(x.get("combo", "")) for x in best_bets}
        ranked = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)

        for rank, (combo, prob) in enumerate(ranked, start=1):
            pseudo_odds = min(1.0 / max(prob, 1e-6), 100.0)
            expected_return_yen = prob * pseudo_odds * 100.0

            is_selected = 1 if combo in selected_combo_set else 0
            is_hit = 1 if combo == actual_combo else 0
            bet_cost_yen = 100 if is_selected == 1 else 0
            return_yen = payout if (is_selected == 1 and is_hit == 1) else 0.0
            profit_yen = return_yen - bet_cost_yen

            rows_out.append({
                "logged_at": "",
                "date": date,
                "venue": venue,
                "race_no": race_no,
                "combo": combo,
                "rank_prob": rank,
                "prob": round(prob, 8),
                "prob_pct": round(prob * 100.0, 4),
                "odds": round(pseudo_odds, 4),
                "expected_return_yen": round(expected_return_yen, 4),
                "is_selected": is_selected,
                "model_name": f"binary_catboost_{venue}_sim_1y",
                "actual_combo": actual_combo,
                "payout": round(payout, 4),
                "source": "sim_1y",
                "is_hit": is_hit,
                "bet_cost_yen": bet_cost_yen,
                "return_yen": round(return_yen, 4),
                "profit_yen": round(profit_yen, 4),
            })

        race_count += 1
        if race_count % 500 == 0:
            print(f"[{venue}] simulated races: {race_count}")

    if not rows_out:
        raise RuntimeError(f"{venue}: no simulated rows")

    return pd.DataFrame(rows_out)


def summarize_selected(df: pd.DataFrame, venue: str) -> None:
    sdf = df[df["is_selected"] == 1].copy()

    buy_count = int(len(sdf))
    hit_count = int((sdf["is_hit"] == 1).sum())
    total_bets = float(sdf["bet_cost_yen"].sum())
    total_return = float(sdf["return_yen"].sum())
    total_profit = total_return - total_bets
    hit_rate = hit_count / buy_count if buy_count > 0 else 0.0
    roi = total_return / total_bets if total_bets > 0 else 0.0

    print(f"\n===== {venue} SIM SUMMARY =====")
    print("buy_count    :", buy_count)
    print("hit_count    :", hit_count)
    print("hit_rate     :", round(hit_rate, 4))
    print("total_bets   :", round(total_bets, 2))
    print("total_return :", round(total_return, 2))
    print("total_profit :", round(total_profit, 2))
    print("roi          :", round(roi, 4))


def main() -> None:
    latest_date, cutoff_date = get_latest_date_and_cutoff()

    print("latest date :", latest_date)
    print("cutoff date :", cutoff_date)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []

    for venue in VENUES:
        print(f"\n########## {venue} ##########")
        model, feature_cols = load_model_and_meta(venue)
        df = load_one_year_venue_df(venue, cutoff_date=cutoff_date)
        print(f"[{venue}] rows: {len(df)} / races: {df['race_key'].nunique()}")

        sim_df = simulate_one_venue(
            venue=venue,
            df=df,
            model=model,
            feature_cols=feature_cols,
        )

        summarize_selected(sim_df, venue)
        all_frames.append(sim_df)

    merged = pd.concat(all_frames, ignore_index=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n===== DONE =====")
    print("saved:", OUTPUT_PATH)
    print("rows :", len(merged))


if __name__ == "__main__":
    main()
