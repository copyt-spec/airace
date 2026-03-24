from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.model_loader import BoatRaceModel  # type: ignore
from engine.buy_selector import select_best_bets  # type: ignore


FEATURES_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render.joblib"
META_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_meta.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "datasets" / "simulate_best_bets_roi_result.csv"

# ここに成績TXTのあるフォルダを入れてください
RESULT_TXT_DIRS = [
    PROJECT_ROOT / "data" / "txt",
    PROJECT_ROOT / "data" / "results_txt",
    PROJECT_ROOT / "data" / "startk",
]

ROWS_PER_RACE = 120
TARGET_VENUES = {"丸亀", "児島", "戸田"}

UNIT_BET_YEN = 100
TOP_N = 5

# buy_selector 用
MIN_PROB = 0.03
MAX_ODDS_CAP = 40.0
MAX_EV_CAP = 2.2
WEIGHT_PROB = 0.60
WEIGHT_EV = 0.25
WEIGHT_ODDS = 0.15


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        s = str(v).replace(",", "").strip()
        return float(s)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        s = str(v).replace(",", "").strip()
        return int(float(s))
    except Exception:
        return default


def _normalize_venue_text(s: str) -> str:
    t = _safe_str(s).replace("　", "").replace(" ", "")
    t = t.replace("ボートレース", "").replace("［成績］", "")
    if "丸亀" in t:
        return "丸亀"
    if "児島" in t:
        return "児島"
    if "戸田" in t:
        return "戸田"
    return t


def _find_existing_result_dirs() -> List[Path]:
    return [p for p in RESULT_TXT_DIRS if p.exists() and p.is_dir()]


def _find_odds_col(df: pd.DataFrame) -> Optional[str]:
    exact_candidates = [
        "odds",
        "trifecta_odds",
        "market_odds",
        "odds_3t",
        "odds3t",
        "final_odds",
        "result_odds",
        "pre_odds",
    ]
    for c in exact_candidates:
        if c in df.columns:
            return c

    lower_map = {c.lower(): c for c in df.columns}
    for k, orig in lower_map.items():
        if "odds" in k:
            return orig

    return None


def _print_odds_candidates(df: pd.DataFrame) -> None:
    print("\n[DEBUG] odds候補列:")
    found = False
    for c in df.columns:
        if "odds" in str(c).lower():
            print(" -", c)
            found = True
    if not found:
        print(" - oddsらしい列は見つかりませんでした。")


def _build_pred_rows_exact(
    race_df: pd.DataFrame,
    prob_map: Dict[str, float],
    odds_col: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    combo_to_odds: Dict[str, float] = {}
    for _, row in race_df.iterrows():
        combo = _safe_str(row.get("combo"))
        if not combo:
            continue
        combo_to_odds[combo] = _safe_float(row.get(odds_col), 0.0)

    for combo, prob in prob_map.items():
        odds = combo_to_odds.get(combo, 0.0)
        ev = prob * odds if odds > 0 else 0.0
        rows.append({
            "combo": combo,
            "score": float(prob),
            "prob": float(prob),
            "odds": float(odds),
            "ev": float(ev),
        })
    return rows


def _select_best_bets_fallback_by_prob(
    prob_map: Dict[str, float],
    top_n: int,
) -> List[Dict[str, Any]]:
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    out: List[Dict[str, Any]] = []
    for i, (combo, prob) in enumerate(items, start=1):
        out.append({
            "combo": combo,
            "score": float(prob),
            "prob": float(prob),
            "buy_rank": i,
            "is_best_bet": True,
        })
    return out


def _parse_result_txt_file(path: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    """
    STARTK/TXT から [払戻金] の 3連単払戻を抜く
    key = (date_yyyymmdd, venue, race_no)
    value = {"combo": "1-2-3", "payout": 600}
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    result_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    current_venue = ""
    current_date = ""

    full_date_re = re.compile(r"(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})")
    race_payout_re = re.compile(r"^\s*(\d{1,2})R\s+([1-6]-[1-6]-[1-6])\s+([\d,]+)")

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # venue更新
        if "ボートレース" in raw:
            v = _normalize_venue_text(raw)
            if v in TARGET_VENUES:
                current_venue = v

        # date更新
        m_date = full_date_re.search(raw)
        if m_date:
            yyyy = int(m_date.group(1))
            mm = int(m_date.group(2))
            dd = int(m_date.group(3))
            current_date = f"{yyyy:04d}{mm:02d}{dd:02d}"

        # 払戻金テーブル開始
        if "[払戻金]" in raw:
            j = i + 1
            while j < len(lines):
                row = lines[j]
                m = race_payout_re.match(row)
                if not m:
                    # 連続する race 行が終わったら抜ける
                    if row.strip() == "":
                        break
                    # 12R以外の行に入ったら適宜終了
                    if "R" not in row:
                        break
                    j += 1
                    continue

                race_no = int(m.group(1))
                combo = m.group(2)
                payout = _safe_int(m.group(3), 0)

                if current_date and current_venue in TARGET_VENUES:
                    key = (current_date, current_venue, race_no)
                    result_map[key] = {
                        "combo": combo,
                        "payout": payout,
                        "source_file": str(path.name),
                    }
                j += 1
            i = j
            continue

        i += 1

    return result_map


def _load_all_payouts() -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    merged: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    dirs = _find_existing_result_dirs()
    if not dirs:
        raise RuntimeError(
            "成績TXTフォルダが見つかりません。RESULT_TXT_DIRS を確認してください。"
        )

    txt_files: List[Path] = []
    for d in dirs:
        txt_files.extend(sorted(d.rglob("*.TXT")))
        txt_files.extend(sorted(d.rglob("*.txt")))

    if not txt_files:
        raise RuntimeError("成績TXTファイルが見つかりませんでした。")

    for path in txt_files:
        one = _parse_result_txt_file(path)
        merged.update(one)

    return merged


def _simulate_one_race(
    race_df: pd.DataFrame,
    model: BoatRaceModel,
    payouts_map: Dict[Tuple[str, str, int], Dict[str, Any]],
    odds_col: Optional[str],
) -> Dict[str, Any]:
    row0 = race_df.iloc[0]

    venue = _normalize_venue_text(_safe_str(row0.get("venue")))
    date = _safe_str(row0.get("date"))
    race_no = _safe_int(row0.get("race_no"), 0)
    race_key = _safe_str(row0.get("race_key"))
    y_combo_csv = _safe_str(row0.get("y_combo"))

    prob_map = model.predict_proba(race_df)

    # 払戻はTXTを正とする
    payout_key = (date, venue, race_no)
    payout_info = payouts_map.get(payout_key, {})
    y_combo_txt = _safe_str(payout_info.get("combo"))
    payout_yen = _safe_int(payout_info.get("payout"), 0)

    y_combo = y_combo_txt or y_combo_csv

    # exact mode: historical odds 列あり
    if odds_col:
        pred_rows = _build_pred_rows_exact(race_df, prob_map, odds_col)
        best_bets = select_best_bets(
            pred_rows,
            min_prob=MIN_PROB,
            max_odds_cap=MAX_ODDS_CAP,
            max_ev_cap=MAX_EV_CAP,
            top_n=TOP_N,
            weight_prob=WEIGHT_PROB,
            weight_ev=WEIGHT_EV,
            weight_odds=WEIGHT_ODDS,
        )
        mode = "exact_with_odds"
    else:
        best_bets = _select_best_bets_fallback_by_prob(prob_map, TOP_N)
        mode = "fallback_prob_only"

    buy_combos = [str(x.get("combo", "")) for x in best_bets]
    buy_count = len(buy_combos)
    invested_yen = buy_count * UNIT_BET_YEN

    hit = 1 if y_combo and y_combo in buy_combos else 0

    # payout_yen は 100円払戻金
    returned_yen = int(payout_yen * (UNIT_BET_YEN / 100.0)) if hit and payout_yen > 0 else 0
    profit_yen = returned_yen - invested_yen
    roi = (returned_yen / invested_yen) if invested_yen > 0 else 0.0

    return {
        "race_key": race_key,
        "date": date,
        "venue": venue,
        "race_no": race_no,
        "selector_mode": mode,
        "y_combo": y_combo,
        "buy_count": buy_count,
        "buy_combos": ",".join(buy_combos),
        "invested_yen": invested_yen,
        "returned_yen": returned_yen,
        "profit_yen": profit_yen,
        "hit": hit,
        "txt_payout_yen": payout_yen,
        "roi": roi,
    }


def _print_summary(result_df: pd.DataFrame) -> None:
    if result_df.empty:
        print("結果がありません。")
        return

    print("\n===== ROI SUMMARY =====")
    total_races = len(result_df)
    total_invested = int(result_df["invested_yen"].sum())
    total_returned = int(result_df["returned_yen"].sum())
    total_profit = int(result_df["profit_yen"].sum())
    hit_rate = float(result_df["hit"].mean()) if total_races > 0 else 0.0
    roi = (total_returned / total_invested) if total_invested > 0 else 0.0

    print(f"races        : {total_races}")
    print(f"invested_yen : {total_invested}")
    print(f"returned_yen : {total_returned}")
    print(f"profit_yen   : {total_profit}")
    print(f"hit_rate     : {hit_rate:.4f}")
    print(f"roi          : {roi:.4f} ({roi * 100:.2f}%)")

    print("\n===== VENUE SUMMARY =====")
    for venue in ["丸亀", "児島", "戸田"]:
        vdf = result_df[result_df["venue"] == venue].copy()
        if vdf.empty:
            continue

        races = len(vdf)
        invested = int(vdf["invested_yen"].sum())
        returned = int(vdf["returned_yen"].sum())
        profit = int(vdf["profit_yen"].sum())
        hit_rate = float(vdf["hit"].mean()) if races > 0 else 0.0
        roi = (returned / invested) if invested > 0 else 0.0
        avg_buy_count = float(vdf["buy_count"].mean()) if races > 0 else 0.0

        print(f"\n【{venue}】")
        print(f"races        : {races}")
        print(f"avg_buy_count: {avg_buy_count:.2f}")
        print(f"invested_yen : {invested}")
        print(f"returned_yen : {returned}")
        print(f"profit_yen   : {profit}")
        print(f"hit_rate     : {hit_rate:.4f}")
        print(f"roi          : {roi:.4f} ({roi * 100:.2f}%)")


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing: {FEATURES_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing: {META_PATH}")

    print("loading model...")
    model = BoatRaceModel(
        model_path=str(MODEL_PATH),
        meta_path=str(META_PATH),
        debug=False,
    )

    print("loading features csv...")
    df = pd.read_csv(FEATURES_PATH)

    required_cols = {"venue", "combo", "y_combo", "race_key", "date", "race_no"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"必要列不足: {missing}")

    df["venue"] = df["venue"].astype(str).map(_normalize_venue_text)
    df = df[df["venue"].isin(TARGET_VENUES)].copy()

    odds_col = _find_odds_col(df)
    if odds_col:
        print(f"historical odds column found: {odds_col}")
        print("selector mode: exact_with_odds")
    else:
        print("historical odds column not found.")
        _print_odds_candidates(df)
        print("selector mode: fallback_prob_only")

    print("loading payouts from TXT...")
    payouts_map = _load_all_payouts()
    print(f"payout records: {len(payouts_map)}")

    results: List[Dict[str, Any]] = []

    print("\n===== SIMULATION START =====\n")

    for race_key, race_df in df.groupby("race_key", sort=False):
        race_df = race_df.copy()

        if len(race_df) != ROWS_PER_RACE:
            continue

        venue = _normalize_venue_text(_safe_str(race_df.iloc[0].get("venue")))
        if venue not in TARGET_VENUES:
            continue

        sim_row = _simulate_one_race(
            race_df=race_df,
            model=model,
            payouts_map=payouts_map,
            odds_col=odds_col,
        )
        results.append(sim_row)

    result_df = pd.DataFrame(results)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    _print_summary(result_df)

    print(f"\nsaved: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
