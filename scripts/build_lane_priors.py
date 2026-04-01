# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "startk_dataset.csv"
OUT_PATH = PROJECT_ROOT / "data" / "models" / "lane_priors_by_venue.json"


def normalize_venue(v: Any) -> str:
    s = str(v or "").strip()
    if "丸亀" in s:
        return "丸亀"
    if "戸田" in s:
        return "戸田"
    if "児島" in s:
        return "児島"
    return s


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH, usecols=["venue", "y_combo"], low_memory=False)
    df["venue"] = df["venue"].astype(str).map(normalize_venue)
    df["y_combo"] = df["y_combo"].astype(str)

    venue_first = defaultdict(Counter)
    venue_pair = defaultdict(Counter)

    for _, row in df.iterrows():
        venue = row["venue"]
        combo = row["y_combo"]

        parts = combo.split("-")
        if len(parts) != 3:
            continue

        a, b, c = parts
        if a not in {"1", "2", "3", "4", "5", "6"}:
            continue
        if b not in {"1", "2", "3", "4", "5", "6"}:
            continue
        if c not in {"1", "2", "3", "4", "5", "6"}:
            continue

        venue_first[venue][a] += 1
        venue_pair[venue][f"{a}-{b}"] += 1

    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for venue in sorted(set(list(venue_first.keys()) + list(venue_pair.keys()))):
        first_counter = venue_first[venue]
        pair_counter = venue_pair[venue]

        first_total = sum(first_counter.values())
        pair_total = sum(pair_counter.values())

        first_prob = {}
        for lane in ["1", "2", "3", "4", "5", "6"]:
            # ラプラス平滑
            first_prob[lane] = (first_counter.get(lane, 0) + 1) / (first_total + 6)

        pair_prob = {}
        all_pairs = []
        for a in range(1, 7):
            for b in range(1, 7):
                if a == b:
                    continue
                all_pairs.append(f"{a}-{b}")

        for pair in all_pairs:
            pair_prob[pair] = (pair_counter.get(pair, 0) + 1) / (pair_total + len(all_pairs))

        out[venue] = {
            "first": first_prob,
            "pair": pair_prob,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
