from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.model_loader import BoatRaceModel  # type: ignore


FEATURES_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_features.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render.joblib"
META_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_meta.json"
ROWS_PER_RACE = 120


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _format_venue(v: str) -> str:
    if not v:
        return "不明"
    if "丸亀" in v:
        return "丸亀"
    if "児島" in v:
        return "児島"
    if "戸田" in v:
        return "戸田"
    return v


def _topk_hits(prob_map: Dict[str, float], y_combo: str, k: int) -> int:
    if not prob_map or not y_combo:
        return 0
    topk = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
    combos = [c for c, _ in topk]
    return 1 if y_combo in combos else 0


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

    total_races = 0
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0

    venue_stats = {
        "丸亀": {"races": 0, "top1": 0, "top3": 0, "top5": 0},
        "児島": {"races": 0, "top1": 0, "top3": 0, "top5": 0},
        "戸田": {"races": 0, "top1": 0, "top3": 0, "top5": 0},
    }

    print("\n===== TEST START =====\n")

    for start in range(0, len(df), ROWS_PER_RACE):
        df120 = df.iloc[start:start + ROWS_PER_RACE].copy()
        if len(df120) < ROWS_PER_RACE:
            continue

        row0 = df120.iloc[0]
        venue = _format_venue(_safe_str(row0.get("venue")))
        y_combo = _safe_str(row0.get("y_combo"))

        prob_map = model.predict_proba(df120)

        total_races += 1
        total_top1 += _topk_hits(prob_map, y_combo, 1)
        total_top3 += _topk_hits(prob_map, y_combo, 3)
        total_top5 += _topk_hits(prob_map, y_combo, 5)

        if venue in venue_stats:
            venue_stats[venue]["races"] += 1
            venue_stats[venue]["top1"] += _topk_hits(prob_map, y_combo, 1)
            venue_stats[venue]["top3"] += _topk_hits(prob_map, y_combo, 3)
            venue_stats[venue]["top5"] += _topk_hits(prob_map, y_combo, 5)

    print("\n===== TOTAL RESULT =====")
    print(f"races : {total_races}")
    print(f"top1  : {total_top1 / total_races:.4f}")
    print(f"top3  : {total_top3 / total_races:.4f}")
    print(f"top5  : {total_top5 / total_races:.4f}")

    print("\n===== VENUE SUMMARY =====")
    for venue, stats in venue_stats.items():
        races = stats["races"]
        if races == 0:
            continue

        print(f"\n【{venue}】")
        print(f"races : {races}")
        print(f"top1  : {stats['top1'] / races:.4f}")
        print(f"top3  : {stats['top3'] / races:.4f}")
        print(f"top5  : {stats['top5'] / races:.4f}")


if __name__ == "__main__":
    main()
