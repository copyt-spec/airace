from __future__ import annotations

"""
data/datasets/trifecta_train.csv (19.5M行、全24会場ぶん、約6GB)を1回だけ
チャンク読みして、会場ごとのCSVに分割保存する。

--all で24会場を学習する際、毎回この巨大ファイルを最初から読み直すのは
非常に無駄(24回×6GBのI/O)なので、事前に1回だけ分割しておく。

出力: data/datasets/trifecta_train_by_venue/{venue}.csv
"""

from pathlib import Path

import pandas as pd

from engine.venue_registry import VENUE_ORDER, normalize_venue_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "data" / "datasets" / "trifecta_train.csv"
OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_by_venue"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    writers_header_written = {v: False for v in VENUE_ORDER}
    out_paths = {v: OUT_DIR / f"{v}.csv" for v in VENUE_ORDER}

    # 既存ファイルは一旦消す(追記型で書くため)
    for p in out_paths.values():
        if p.exists():
            p.unlink()

    total = 0
    for i, chunk in enumerate(pd.read_csv(SRC, low_memory=False, chunksize=200_000), start=1):
        chunk["venue"] = chunk["venue"].astype(str).map(normalize_venue_name)
        total += len(chunk)

        for venue, g in chunk.groupby("venue"):
            if venue not in out_paths:
                continue
            mode = "a" if writers_header_written[venue] else "w"
            header = not writers_header_written[venue]
            g.to_csv(out_paths[venue], mode=mode, header=header, index=False, encoding="utf-8-sig")
            writers_header_written[venue] = True

        print(f"chunk {i}: total rows scanned = {total:,}")

    print("=" * 80)
    print("DONE")
    for v, p in out_paths.items():
        if p.exists():
            print(v, "->", p, f"({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
