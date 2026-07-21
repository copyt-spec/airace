from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.controller import RaceController  # noqa: E402
from engine.prediction_logger import build_prediction_rows, save_lane_racer_no  # noqa: E402


LOG_DIR = PROJECT_ROOT / "data" / "logs"
RESULTS_CSV = LOG_DIR / "results.csv"
PREDICTIONS_CSV = LOG_DIR / "predictions.csv"


def _normalize_venue_name(v: str) -> str:
    s = str(v or "").strip()
    venue_order = [
        "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
        "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
        "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
    ]
    for venue in venue_order:
        if venue in s:
            return venue
    return s


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    _ensure_parent(path)
    file_exists = path.exists()

    fieldnames = [
        "logged_at",
        "date",
        "venue",
        "race_no",
        "combo",
        "rank_prob",
        "prob",
        "prob_pct",
        "odds",
        "expected_return_yen",
        "is_selected",
        "model_name",
    ]

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results csv: {path}")

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError("results.csv is empty")

    required = ["date", "venue", "race_no"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"results.csv missing column: {col}")

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df["venue"] = df["venue"].astype(str).map(_normalize_venue_name)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)

    df = df[df["race_no"].between(1, 12)].copy()
    df = df[df["venue"] != ""].copy()
    df = df[df["date"] != ""].copy()

    df = df[["date", "venue", "race_no"]].drop_duplicates().reset_index(drop=True)
    df = df.sort_values(["date", "venue", "race_no"]).reset_index(drop=True)
    return df


def _load_existing_prediction_keys(path: Path) -> set[Tuple[str, str, int]]:
    if not path.exists():
        return set()

    df = pd.read_csv(path, usecols=["date", "venue", "race_no"], low_memory=False)
    if df.empty:
        return set()

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df["venue"] = df["venue"].astype(str).map(_normalize_venue_name)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)

    keys = {
        (str(r["date"]), str(r["venue"]), int(r["race_no"]))
        for _, r in df.drop_duplicates().iterrows()
        if int(r["race_no"]) in range(1, 13)
    }
    return keys


def _filter_targets(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    venue: str,
    existing_keys: set[Tuple[str, str, int]],
    skip_existing: bool,
) -> pd.DataFrame:
    work = df.copy()

    if start_date:
        work = work[work["date"] >= start_date].copy()
    if end_date:
        work = work[work["date"] <= end_date].copy()
    if venue:
        venue = _normalize_venue_name(venue)
        work = work[work["venue"] == venue].copy()

    if skip_existing:
        work = work[
            ~work.apply(lambda r: (str(r["date"]), str(r["venue"]), int(r["race_no"])) in existing_keys, axis=1)
        ].copy()

    return work.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS_CSV))
    parser.add_argument("--predictions", default=str(PREDICTIONS_CSV))
    parser.add_argument("--start-date", default="", help="YYYYMMDD")
    parser.add_argument("--end-date", default="", help="YYYYMMDD")
    parser.add_argument("--venue", default="", help="例: 丸亀")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--model-mode", default="venue", choices=["venue", "global_auto"])
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0=all")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    results_path = Path(args.results).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()

    print("loading results :", results_path)
    results_df = _load_results(results_path)

    existing_keys = _load_existing_prediction_keys(predictions_path)
    print("existing pred races:", len(existing_keys))

    targets = _filter_targets(
        df=results_df,
        start_date=str(args.start_date or "").strip(),
        end_date=str(args.end_date or "").strip(),
        venue=str(args.venue or "").strip(),
        existing_keys=existing_keys,
        skip_existing=not args.overwrite_existing,
    )

    if args.limit and int(args.limit) > 0:
        targets = targets.head(int(args.limit)).copy()

    if targets.empty:
        print("no targets to backfill")
        return

    print("=" * 80)
    print("BACKFILL TARGETS")
    print("=" * 80)
    print("rows        :", len(targets))
    print("min date    :", targets["date"].min())
    print("max date    :", targets["date"].max())
    print("venues      :", targets["venue"].nunique())
    print("model_mode  :", args.model_mode)
    print("top_n       :", args.top_n)
    print("predictions :", predictions_path)

    controller = RaceController(model_mode=args.model_mode)

    done = 0
    success = 0
    failed = 0

    for _, row in targets.iterrows():
        date = str(row["date"])
        venue = str(row["venue"])
        race_no = int(row["race_no"])
        done += 1

        try:
            bundle = controller.get_ai_prediction_bundle(
                venue_name=venue,
                date=date,
                race_no=race_no,
                top_n=int(args.top_n),
                with_odds=True,
                model_mode=args.model_mode,
            ) or {}

            best_bets = bundle.get("best_bets", []) or {}
            prob_map = bundle.get("prob_map", {}) or {}
            odds_map = bundle.get("odds_map", {}) or {}

            if not prob_map:
                print(f"[{done}/{len(targets)}] SKIP no prob_map: {date} {venue} {race_no}R")
                failed += 1
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
                continue

            grouped_odds = controller.get_odds_only(venue, race_no=race_no, date=date)

            pred_rows = build_prediction_rows(
                date=date,
                venue=venue,
                race_no=race_no,
                best_bets=list(best_bets) if isinstance(best_bets, list) else [],
                probabilities=prob_map,
                grouped_odds=grouped_odds,
                model_name=f"binary_catboost_{args.model_mode}",
            )

            _append_rows_csv(predictions_path, pred_rows)

            # 2026-07-21追加: 選手級別パターン分析(scripts/analyze_racer_class_patterns.py)
            # を今後のバックフィル分でも継続できるよう、レーン別racer_noを別ログに記録する。
            try:
                df120_rows = bundle.get("df120_rows", []) or []
                if df120_rows:
                    first_row = df120_rows[0]
                    lane_racer_no = {
                        lane: int(first_row.get(f"lane{lane}_racer_no", 0) or 0)
                        for lane in range(1, 7)
                    }
                    save_lane_racer_no(
                        date=date, venue=venue, race_no=race_no, lane_racer_no=lane_racer_no,
                    )
            except Exception as log_e:
                print(f"[WARN] lane racer_no log save failed: {date} {venue} {race_no}R -> {log_e}")

            print(f"[{done}/{len(targets)}] OK   {date} {venue} {race_no}R rows={len(pred_rows)}")
            success += 1

        except Exception as e:
            print(f"[{done}/{len(targets)}] ERR  {date} {venue} {race_no}R -> {e}")
            failed += 1

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("success:", success)
    print("failed :", failed)
    print("total  :", len(targets))


if __name__ == "__main__":
    main()
