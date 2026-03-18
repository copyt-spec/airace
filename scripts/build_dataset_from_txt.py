from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from engine.txt_dataset_builder import TxtDatasetConfig, build_dataset_from_txt


def validate_output_csv(out_csv: str) -> None:
    if not os.path.exists(out_csv):
        raise RuntimeError(f"output csv not found: {out_csv}")

    df = pd.read_csv(out_csv, low_memory=False)

    if len(df) == 0:
        raise RuntimeError("startk_dataset.csv が空です。TXT解析結果が0件です。")

    required_cols = [
        "date",
        "venue",
        "venue_code",
        "race_no",
        "race_key",
        "y_combo",
        "trifecta",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"必須列不足: {missing}")

    unknown_cnt = (df["venue"].astype(str) == "UNKNOWN").sum()
    zero_code_cnt = (df["venue_code"].astype(str).str.zfill(2) == "00").sum()

    if unknown_cnt > 0 or zero_code_cnt > 0:
        sample = df.loc[
            (df["venue"].astype(str) == "UNKNOWN")
            | (df["venue_code"].astype(str).str.zfill(2) == "00"),
            ["date", "venue", "venue_code", "race_no", "race_key"]
        ].head(20)

        raise RuntimeError(
            "venue=UNKNOWN または venue_code=00 が残っています。\n"
            f"unknown_cnt={unknown_cnt}, zero_code_cnt={zero_code_cnt}\n"
            + sample.to_string(index=False)
        )

    valid = df["y_combo"].astype(str).str.match(r"^[1-6]-[1-6]-[1-6]$", na=False).sum()

    print("CSV rows =", len(df))
    print("race_key nunique =", df["race_key"].nunique())
    print("valid labeled races =", valid)
    print("zero venue_code =", zero_code_cnt)
    print("UNKNOWN venue =", unknown_cnt)
    print("OK: startk_dataset.csv validated")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    txt_dir = project_root / "data" / "raw_txt"
    out_csv = project_root / "data" / "datasets" / "startk_dataset.csv"

    print("txt_dir =", txt_dir)
    print("out_csv =", out_csv)

    txt_files = sorted(txt_dir.glob("*.TXT"))
    print("TXT file count =", len(txt_files))

    config = TxtDatasetConfig(
        raw_txt_dir=str(txt_dir),
        out_csv_path=str(out_csv),
        keep_unlabeled=False,
        verbose=True,
    )

    build_dataset_from_txt(config)
    validate_output_csv(str(out_csv))


if __name__ == "__main__":
    main()
