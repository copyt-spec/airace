# scripts/build_odds_from_payout.py

from __future__ import annotations

import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "data" / "datasets" / "startk_dataset.csv"
OUT_PATH = PROJECT_ROOT / "data" / "predictions" / "race_odds.csv"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def build():

    print("===== BUILD ODDS FROM PAYOUT START =====")

    print("[1/4] loading dataset...")
    df = _safe_read_csv(SRC_PATH)
    print("dataset shape:", df.shape)

    required = ["race_key", "y_combo", "trifecta_payout"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

    print("[2/4] building odds rows...")

    rows = []

    for _, row in df.iterrows():

        race_key = str(row["race_key"])
        combo = str(row["y_combo"])

        payout = row["trifecta_payout"]

        if pd.isna(payout):
            continue

        try:
            payout = float(payout)
        except:
            continue

        odds = payout / 100.0

        rows.append({
            "race_key": race_key,
            "combo": combo,
            "odds": odds
        })

    odds_df = pd.DataFrame(rows)

    print("odds rows:", len(odds_df))

    print("[3/4] saving...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    odds_df.to_csv(
        OUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("saved:", OUT_PATH)

    print("[4/4] done")

    print("===== BUILD ODDS FROM PAYOUT END =====")


if __name__ == "__main__":
    build()
