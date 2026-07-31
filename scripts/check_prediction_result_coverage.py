from __future__ import annotations

from pathlib import Path
import pandas as pd


LOG_DIR = Path("data/logs")
PRED_CSV = LOG_DIR / "predictions.csv"
RES_CSV = LOG_DIR / "results.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        out["date"] = (
            out["date"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(8)
            .str.strip()
        )

    if "venue" in out.columns:
        out["venue"] = out["venue"].astype(str).str.strip()

    if "race_no" in out.columns:
        out["race_no"] = pd.to_numeric(out["race_no"], errors="coerce").fillna(0).astype(int)

    return out


def main() -> None:
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"missing predictions: {PRED_CSV}")
    if not RES_CSV.exists():
        raise FileNotFoundError(f"missing results: {RES_CSV}")

    pred = pd.read_csv(PRED_CSV, usecols=["date", "venue", "race_no"], low_memory=False)
    res = pd.read_csv(RES_CSV, usecols=["date", "venue", "race_no"], low_memory=False)

    pred = _normalize(pred)
    res = _normalize(res)

    pred_keys = pred.drop_duplicates().sort_values(["date", "venue", "race_no"]).reset_index(drop=True)
    res_keys = res.drop_duplicates().sort_values(["date", "venue", "race_no"]).reset_index(drop=True)

    matched = pred_keys.merge(
        res_keys,
        on=["date", "venue", "race_no"],
        how="inner",
    )

    missing_results = pred_keys.merge(
        res_keys,
        on=["date", "venue", "race_no"],
        how="left",
        indicator=True,
    )
    missing_results = missing_results[missing_results["_merge"] == "left_only"].drop(columns=["_merge"])
    missing_results = missing_results.sort_values(["date", "venue", "race_no"]).reset_index(drop=True)

    extra_results = res_keys.merge(
        pred_keys,
        on=["date", "venue", "race_no"],
        how="left",
        indicator=True,
    )
    extra_results = extra_results[extra_results["_merge"] == "left_only"].drop(columns=["_merge"])
    extra_results = extra_results.sort_values(["date", "venue", "race_no"]).reset_index(drop=True)

    print("===== COVERAGE CHECK =====")
    print("pred unique keys :", len(pred_keys))
    print("res unique keys  :", len(res_keys))
    print("matched keys     :", len(matched))
    print("missing results  :", len(missing_results))
    print("extra results    :", len(extra_results))

    if not pred_keys.empty:
        print("\n[pred range]")
        print("min date:", pred_keys["date"].min())
        print("max date:", pred_keys["date"].max())

    if not res_keys.empty:
        print("\n[results range]")
        print("min date:", res_keys["date"].min())
        print("max date:", res_keys["date"].max())

    print("\n[pred venues]")
    print(pred_keys["venue"].value_counts().to_string())

    print("\n[result venues]")
    print(res_keys["venue"].value_counts().to_string())

    if not missing_results.empty:
        print("\n===== MISSING RESULTS (top 100) =====")
        print(missing_results.head(100).to_string(index=False))

        print("\n===== MISSING RESULTS BY VENUE =====")
        print(missing_results["venue"].value_counts().to_string())

    if not extra_results.empty:
        print("\n===== EXTRA RESULTS WITHOUT PREDICTION (top 100) =====")
        print(extra_results.head(100).to_string(index=False))

        print("\n===== EXTRA RESULTS BY VENUE =====")
        print(extra_results["venue"].value_counts().to_string())

    # 会場×日付での不足状況も見やすく出す
    if not missing_results.empty:
        by_day = (
            missing_results.groupby(["date", "venue"])
            .size()
            .reset_index(name="missing_races")
            .sort_values(["date", "venue"])
        )
        print("\n===== MISSING SUMMARY BY DATE/VENUE =====")
        print(by_day.head(100).to_string(index=False))


if __name__ == "__main__":
    main()
