from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/datasets/trifecta_train.csv")
    parser.add_argument("--out", default="data/datasets/trifecta_train_small.csv")
    parser.add_argument("--venue", default="", help="丸亀 / 戸田 / 児島")
    parser.add_argument("--max_rows", type=int, default=800000)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        raise FileNotFoundError(src)

    usecols = [
        "date", "venue", "race_key", "race_no", "combo", "y",
        "trifecta_payout", "wave_cm", "weather", "wind_dir", "wind_speed_mps",
        "lane1_boat", "lane1_course", "lane1_exhibit", "lane1_motor", "lane1_racer_no", "lane1_st",
        "lane2_boat", "lane2_course", "lane2_exhibit", "lane2_motor", "lane2_racer_no", "lane2_st",
        "lane3_boat", "lane3_course", "lane3_exhibit", "lane3_motor", "lane3_racer_no", "lane3_st",
        "lane4_boat", "lane4_course", "lane4_exhibit", "lane4_motor", "lane4_racer_no", "lane4_st",
        "lane5_boat", "lane5_course", "lane5_exhibit", "lane5_motor", "lane5_racer_no", "lane5_st",
        "lane6_boat", "lane6_course", "lane6_exhibit", "lane6_motor", "lane6_racer_no", "lane6_st",
    ]

    chunks = []
    total = 0

    for chunk in pd.read_csv(src, usecols=usecols, chunksize=200000, low_memory=False):
        if args.venue:
            chunk = chunk[chunk["venue"].astype(str) == args.venue].copy()

        if chunk.empty:
            continue

        remain = args.max_rows - total
        if remain <= 0:
            break

        if len(chunk) > remain:
            chunk = chunk.iloc[:remain].copy()

        chunks.append(chunk)
        total += len(chunk)

        print(f"collected: {total}")

        if total >= args.max_rows:
            break

    if not chunks:
        raise RuntimeError("no rows collected")

    df = pd.concat(chunks, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("saved:", out)
    print("rows :", len(df))


if __name__ == "__main__":
    main()
