# scripts/eval_trifecta_120cls.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# ===== import path 対策 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.model_loader import BoatRaceModel  # type: ignore


FEATURES_PATH = Path("data/datasets/trifecta_train_features.csv")
MODEL_PATH = Path("data/models/trifecta120_model.joblib")
META_PATH = Path("data/models/trifecta120_model_meta.json")


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _race_group_key(df120: pd.DataFrame) -> str:
    row0 = df120.iloc[0]
    date = _safe_str(row0.get("date"))
    venue = _safe_str(row0.get("venue"))
    race_no = _safe_str(row0.get("race_no"))
    y_combo = _safe_str(row0.get("y_combo"))
    return f"{date}_{venue}_{race_no}_{y_combo}"


def _topk_hits(prob_map: Dict[str, float], y_combo: str, k: int) -> int:
    if not prob_map or not y_combo:
        return 0
    topk = sorted(prob_map.items(), key=lambda kv: float(kv[1]), reverse=True)[:k]
    topk_combos = [c for c, _ in topk]
    return 1 if y_combo in topk_combos else 0


def _topk_rank(prob_map: Dict[str, float], y_combo: str) -> int:
    if not prob_map or not y_combo:
        return 999999
    ordered = sorted(prob_map.items(), key=lambda kv: float(kv[1]), reverse=True)
    for i, (c, _) in enumerate(ordered, start=1):
        if c == y_combo:
            return i
    return 999999


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

    if "combo" not in df.columns:
        raise ValueError("combo column not found")
    if "y_combo" not in df.columns:
        raise ValueError("y_combo column not found")

    total_rows = len(df)
    if total_rows % 120 != 0:
        print(f"[WARN] row count is not multiple of 120: rows={total_rows}")

    race_count = 0
    hit_top1 = 0
    hit_top3 = 0
    hit_top5 = 0
    hit_top10 = 0

    rank_sum = 0
    worst_rank = 0

    examples: List[Dict[str, Any]] = []

    print("evaluating...")
    for start in range(0, total_rows, 120):
        df120 = df.iloc[start:start + 120].copy()
        if len(df120) < 120:
            continue

        y_combo = _safe_str(df120.iloc[0].get("y_combo"))
        if not y_combo:
            continue

        prob_map = model.predict_proba(df120)

        race_count += 1
        hit_top1 += _topk_hits(prob_map, y_combo, 1)
        hit_top3 += _topk_hits(prob_map, y_combo, 3)
        hit_top5 += _topk_hits(prob_map, y_combo, 5)
        hit_top10 += _topk_hits(prob_map, y_combo, 10)

        rank = _topk_rank(prob_map, y_combo)
        rank_sum += rank
        worst_rank = max(worst_rank, rank)

        if race_count <= 5:
            top5 = sorted(prob_map.items(), key=lambda kv: float(kv[1]), reverse=True)[:5]
            examples.append(
                {
                    "race_key": _race_group_key(df120),
                    "y_combo": y_combo,
                    "top5": top5,
                    "y_rank": rank,
                }
            )

        if race_count % 1000 == 0:
            print(f"evaluated_races={race_count:,}")

    if race_count == 0:
        print("No races evaluated.")
        return

    top1_acc = hit_top1 / race_count
    top3_acc = hit_top3 / race_count
    top5_acc = hit_top5 / race_count
    top10_acc = hit_top10 / race_count
    avg_rank = rank_sum / race_count

    print("\n===== EVAL RESULT =====")
    print(f"races      : {race_count:,}")
    print(f"top1_acc   : {top1_acc:.6f}")
    print(f"top3_acc   : {top3_acc:.6f}")
    print(f"top5_acc   : {top5_acc:.6f}")
    print(f"top10_acc  : {top10_acc:.6f}")
    print(f"avg_rank   : {avg_rank:.3f}")
    print(f"worst_rank : {worst_rank}")

    print("\n===== SAMPLE TOP5 =====")
    for ex in examples:
        print(f"[{ex['race_key']}] y={ex['y_combo']} rank={ex['y_rank']}")
        for combo, p in ex["top5"]:
            print(f"  {combo}  {float(p):.6f}")
        print("")


if __name__ == "__main__":
    main()
