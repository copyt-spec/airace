from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
RESULTS_CSV = LOG_DIR / "results.csv"
MERGED_CSV = LOG_DIR / "prediction_results_merged.csv"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    _ensure_dir()
    file_exists = path.exists()

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_prediction_rows(
    *,
    date: str,
    venue: str,
    race_no: int,
    best_bets: List[Dict[str, Any]],
    probabilities: Dict[str, float],
    grouped_odds: Optional[Dict[str, Any]] = None,
    model_name: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    odds_map: Dict[str, float] = {}
    if grouped_odds and isinstance(grouped_odds, dict):
        godata = grouped_odds.get("data", {})
        if isinstance(godata, dict):
            for a in range(1, 7):
                col = godata.get(a, {})
                if not isinstance(col, dict):
                    continue
                for b in range(1, 7):
                    if b == a:
                        continue
                    for c in range(1, 7):
                        if c == a or c == b:
                            continue
                        combo = f"{a}-{b}-{c}"
                        odd = col.get((b, c))
                        if odd is None:
                            odd = col.get(f"{b}{c}")
                        odds_map[combo] = _safe_float(odd, 0.0)

    selected_combo_set = {
        _safe_str(x.get("combo", ""))
        for x in (best_bets or [])
        if _safe_str(x.get("combo", ""))
    }

    ranked = sorted(
        [(combo, _safe_float(prob, 0.0)) for combo, prob in (probabilities or {}).items()],
        key=lambda kv: kv[1],
        reverse=True,
    )

    logged_at = _now_iso()

    for rank, (combo, prob) in enumerate(ranked, start=1):
        odds = _safe_float(odds_map.get(combo, 0.0), 0.0)
        expected_return_yen = prob * odds * 100.0 if odds > 0 else 0.0
        row = {
            "logged_at": logged_at,
            "date": _safe_str(date),
            "venue": _safe_str(venue),
            "race_no": int(race_no),
            "combo": combo,
            "rank_prob": rank,
            "prob": round(prob, 8),
            "prob_pct": round(prob * 100.0, 4),
            "odds": round(odds, 4),
            "expected_return_yen": round(expected_return_yen, 4),
            "is_selected": 1 if combo in selected_combo_set else 0,
            "model_name": _safe_str(model_name),
        }
        rows.append(row)

    return rows


def save_prediction_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = [
        "logged_at",
        "date",
        "venue",
        "race_no",
        "combo",
        "rank_prob",
        "prob",
        "prob_pct",
        "odds",
        "expected_return_yen",
        "is_selected",
        "model_name",
    ]

    for row in rows:
        _append_csv_row(PREDICTIONS_CSV, fieldnames, row)


def save_race_result(
    *,
    date: str,
    venue: str,
    race_no: int,
    actual_combo: str,
    payout: float,
    source: str = "",
) -> None:
    fieldnames = [
        "logged_at",
        "date",
        "venue",
        "race_no",
        "actual_combo",
        "payout",
        "source",
    ]

    row = {
        "logged_at": _now_iso(),
        "date": _safe_str(date),
        "venue": _safe_str(venue),
        "race_no": int(race_no),
        "actual_combo": _safe_str(actual_combo),
        "payout": round(_safe_float(payout, 0.0), 4),
        "source": _safe_str(source),
    }
    _append_csv_row(RESULTS_CSV, fieldnames, row)
