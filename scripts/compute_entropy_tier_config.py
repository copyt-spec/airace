from __future__ import annotations

"""
scripts/backtest_venue_tuned_tier_bounds.py (2026-07-23) で検証済みの
「entropy(確率分布の散らばり)に応じてtierごとの点数レンジ(min/max)を
変える」ポリシーを本番投入するため、全期間データ(train/test分割せず)で
最終的なentropy閾値を再計算し、data/models/entropy_tier_config.json に書き出す。

検証時はtrain(〜20260702)のみで閾値・プロファイルを決めてtest(20260703〜)
で評価したが、本番投入時は「手法自体は検証済み」という前提のもと、
利用可能な全データで閾値を出し直す(標準的なやり方: 検証はholdoutで、
本番用の値は全データで再フィットする)。

tierごとの点数レンジ自体は検証結果をそのまま採用する:
  - デフォルト(ほとんどの会場): low_only_adaptive
      low(堅い)=1〜6点, mid=3〜10点(現行相当), high(荒れ)=3〜10点(現行相当)
  - 戸田・江戸川(train件数がそれぞれ264・166件と十分にあり、個別に
    aggressive_low_cutの方が良かった):
      low=1〜4点, mid=2〜8点, high=4〜16点
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "engine").exists() else Path("/sessions/practical-magical-edison/mnt/boat_ai")

LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions_combined_for_sim.csv"
OUT_JSON = PROJECT_ROOT / "data" / "models" / "entropy_tier_config.json"

SHRINKAGE_K = 30

TIER_POINT_BOUNDS_DEFAULT = {"low": [1, 6], "mid": [3, 10], "high": [3, 10]}
TIER_POINT_BOUNDS_OVERRIDE = {
    "戸田": {"low": [1, 4], "mid": [2, 8], "high": [4, 16]},
    "江戸川": {"low": [1, 4], "mid": [2, 8], "high": [4, 16]},
}


def entropy_top10(probs_sorted_desc):
    import math
    vals = [p for p in probs_sorted_desc[:10] if p > 0]
    if not vals:
        return 1.0
    total = sum(vals)
    if total <= 0:
        return 1.0
    norm = [p / total for p in vals]
    h = -sum(p * math.log(p + 1e-12) for p in norm)
    max_h = math.log(len(norm) + 1e-12)
    return max(0.0, min(1.0, h / max_h)) if max_h > 0 else 0.0


def main() -> None:
    df = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df["venue"] = df["venue"].astype(str).str.strip()
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)

    entropies = []
    entropies_by_venue: dict = {}

    for (date, venue, race_no), race_df in df.groupby(["date", "venue", "race_no"], sort=False):
        probs = sorted(race_df["prob"].astype(float).tolist(), reverse=True)
        ent = entropy_top10(probs)
        entropies.append(ent)
        entropies_by_venue.setdefault(venue, []).append(ent)

    q33, q66 = np.percentile(entropies, [33.33, 66.67])
    print(f"全体 n={len(entropies)}, q33={q33:.4f}, q66={q66:.4f}")

    thresholds = {"default": [round(float(q33), 4), round(float(q66), 4)]}
    print()
    print(f"{'venue':6s} {'n':>5s} {'q33':>8s} {'q66':>8s} {'shrunk_q33':>11s} {'shrunk_q66':>11s}")
    for venue, vals in sorted(entropies_by_venue.items()):
        n_v = len(vals)
        if n_v >= 2:
            vq33, vq66 = np.percentile(vals, [33.33, 66.67])
        else:
            vq33 = vq66 = vals[0]
        w = n_v / (n_v + SHRINKAGE_K)
        s_q33 = w * vq33 + (1 - w) * q33
        s_q66 = w * vq66 + (1 - w) * q66
        thresholds[venue] = [round(float(s_q33), 4), round(float(s_q66), 4)]
        print(f"{venue:6s} {n_v:5d} {vq33:8.4f} {vq66:8.4f} {s_q33:11.4f} {s_q66:11.4f}")

    config = {
        "_comment": (
            "2026-07-23 scripts/compute_entropy_tier_config.py で生成。"
            "根拠: scripts/backtest_venue_tuned_tier_bounds.py の検証結果。"
            "entropy_thresholds: 会場ごとの[q33, q66](無ければdefaultを使う、"
            "全会場を含む全期間predictions_combined_for_sim.csvから算出、"
            "小サンプル会場はempirical bayes shrinkage(K=30)でdefaultに寄せてある)。"
            "tier_point_bounds: entropy tier(low/mid/high)ごとのmin/max点数。"
        ),
        "entropy_thresholds": thresholds,
        "tier_point_bounds_default": TIER_POINT_BOUNDS_DEFAULT,
        "tier_point_bounds_override": TIER_POINT_BOUNDS_OVERRIDE,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print()
    print("saved:", OUT_JSON)


if __name__ == "__main__":
    main()
