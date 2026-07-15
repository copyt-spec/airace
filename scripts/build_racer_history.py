from __future__ import annotations

"""
data/raw_txt の全K-fileをパースし、選手×レースのロング形式ファクトテーブルを作る。
(date, venue, race_no, lane, racer_no, finish, course, st, motor, boat)

このテーブルから、リーク無し(該当レースより前の実績のみ使用)の
選手成績特徴量(累積勝率・複勝率・平均ST・コース別複勝率)を計算できる。
学習用データセットの特徴量拡張(lane{n}_racer_no を実力指標に変換)のために使う。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.txt_race_parser import parse_startk_multi_venue_txt  # noqa: E402

RAW_TXT_DIR = PROJECT_ROOT / "data" / "raw_txt"
OUT_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"


def _read_text_auto(path: Path) -> str:
    for enc in ["cp932", "shift_jis", "utf-8", "utf-8-sig", "euc_jp", "latin1"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"failed to read: {path}")


def _extract_date_from_filename(filename: str) -> str:
    import re
    m = re.search(r"(\d{4})[^\d]?(\d{1,2})[^\d]?(\d{1,2})", filename)
    if not m:
        return ""
    y, mo, d = m.groups()
    return f"{int(y):04d}{int(mo):02d}{int(d):02d}"


def build_history() -> pd.DataFrame:
    files = sorted(RAW_TXT_DIR.glob("*.TXT")) + sorted(RAW_TXT_DIR.glob("*.txt"))
    files = sorted(set(files))
    print(f"files found: {len(files)}")

    rows: List[Dict[str, Any]] = []

    for i, path in enumerate(files, start=1):
        try:
            text = _read_text_auto(path)
            fallback_date = _extract_date_from_filename(path.name)
            records = parse_startk_multi_venue_txt(text, fallback_date=fallback_date)

            for rec in records:
                for b in rec.boats:
                    rows.append({
                        "date": rec.meta.date,
                        "venue": rec.meta.venue,
                        "venue_code": rec.meta.venue_code,
                        "race_no": rec.meta.race_no,
                        "lane": b.lane,
                        "racer_no": b.racer_no,
                        "finish": b.finish,
                        "course": b.course if b.course is not None else b.lane,
                        "st": b.st,
                        "motor": b.motor,
                        "boat": b.boat,
                        "exhibit": b.exhibit,
                    })

            if i % 100 == 0:
                print(f"[{i}/{len(files)}] parsed, rows so far={len(rows)}")

        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    df = build_history()
    print("total boat-race rows:", len(df))

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df = df[df["date"].str.match(r"^\d{8}$")].copy()
    df["racer_no"] = pd.to_numeric(df["racer_no"], errors="coerce").fillna(0).astype(int)
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce").fillna(99).astype(int)
    df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0).astype(int)
    df["st"] = pd.to_numeric(df["st"], errors="coerce")

    df = df[(df["racer_no"] > 0) & (df["race_no"].between(1, 12))].copy()
    df = df.drop_duplicates(subset=["date", "venue", "race_no", "lane"]).copy()
    df = df.sort_values(["date", "venue", "race_no", "lane"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("saved:", OUT_CSV)
    print("rows :", len(df))
    print("unique racers:", df["racer_no"].nunique())
    print("date range:", df["date"].min(), df["date"].max())
    print("venues:", df["venue"].nunique())


if __name__ == "__main__":
    main()
