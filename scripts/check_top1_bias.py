# scripts/check_top1_bias.py
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

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


def _race_key(df120: pd.DataFrame) -> str:
    row0 = df120.iloc[0]
    date = _safe_str(row0.get("date"))
    venue = _safe_str(row0.get("venue"))
    race_no = _safe_str(row0.get("race_no"))
    return f"{date}_{venue}_{race_no}"


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
        print(f"[WARN] rows not multiple of 120: {total_rows}")

    pred_top1_counter: Counter[str] = Counter()
    true_counter: Counter[str] = Counter()
    hit_counter: Counter[str] = Counter()

    race_count = 0
    examples: List[Dict[str, Any]] = []

    print("checking top1 bias...")
    for start in range(0, total_rows, 120):
        df120 = df.iloc[start:start + 120].copy()
        if len(df120) < 120:
            continue

        y_combo = _safe_str(df120.iloc[0].get("y_combo"))
        if not y_combo:
            continue

        prob_map = model.predict_proba(df120)
        if not prob_map:
            continue

        ordered = sorted(prob_map.items(), key=lambda kv: float(kv[1]), reverse=True)
        pred_top1 = ordered[0][0]

        race_count += 1
        pred_top1_counter[pred_top1] += 1
        true_counter[y_combo] += 1
        if pred_top1 == y_combo:
            hit_counter[pred_top1] += 1

        if len(examples) < 10 and pred_top1 != y_combo:
            examples.append(
                {
                    "race": _race_key(df120),
                    "y_combo": y_combo,
                    "pred_top1": pred_top1,
                    "pred_top1_prob": float(ordered[0][1]),
                    "pred_top5": ordered[:5],
                }
            )

        if race_count % 1000 == 0:
            print(f"checked_races={race_count:,}")

    if race_count == 0:
        print("No races checked.")
        return

    print("\n===== SUMMARY =====")
    print(f"races: {race_count:,}")
    print(f"unique_pred_top1: {len(pred_top1_counter)}")
    print(f"unique_true_combo: {len(true_counter)}")

    print("\n===== PRED TOP1 COUNT (TOP20) =====")
    for combo, cnt in pred_top1_counter.most_common(20):
        hit = hit_counter.get(combo, 0)
        rate = cnt / race_count
        print(f"{combo:>6}  pred={cnt:>6}  rate={rate:>8.4%}  hit={hit:>5}")

    print("\n===== TRUE COMBO COUNT (TOP20) =====")
    for combo, cnt in true_counter.most_common(20):
        rate = cnt / race_count
        print(f"{combo:>6}  true={cnt:>6}  rate={rate:>8.4%}")

    print("\n===== PRED / TRUE RATIO (TOP20 by pred count) =====")
    for combo, pred_cnt in pred_top1_counter.most_common(20):
        true_cnt = true_counter.get(combo, 0)
        ratio = (pred_cnt / true_cnt) if true_cnt > 0 else float("inf")
        print(f"{combo:>6}  pred={pred_cnt:>6}  true={true_cnt:>6}  ratio={ratio:>8.3f}")

    print("\n===== MISS EXAMPLES =====")
    for ex in examples:
        print(
            f"[{ex['race']}] "
            f"y={ex['y_combo']} "
            f"pred_top1={ex['pred_top1']} "
            f"p={ex['pred_top1_prob']:.6f}"
        )
        for c, p in ex["pred_top5"]:
            print(f"  {c}  {float(p):.6f}")
        print("")


if __name__ == "__main__":
    main()
