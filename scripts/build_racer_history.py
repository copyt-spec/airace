from __future__ import annotations

"""
data/raw_txt の全K-fileをパースし、選手×レースのロング形式ファクトテーブルを作る。
(date, venue, race_no, lane, racer_no, finish, course, st, motor, boat)

このテーブルから、リーク無し(該当レースより前の実績のみ使用)の
選手成績特徴量(累積勝率・複勝率・平均ST・コース別複勝率)を計算できる。
学習用データセットの特徴量拡張(lane{n}_racer_no を実力指標に変換)のために使う。
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.txt_race_parser import parse_startk_multi_venue_txt  # noqa: E402

RAW_TXT_DIR = PROJECT_ROOT / "data" / "raw_txt"
OUT_CSV = PROJECT_ROOT / "data" / "datasets" / "racer_race_history.csv"

# 差分更新用: 既存ファイルの最大日付からこの日数だけ遡って再パースする
# (同じ日のファイルが後から修正されるケースへの保険)
OVERLAP_DAYS = 3


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


def _extract_date_from_k_filename(filename: str) -> str:
    """K250101.TXT のような mbrace Kファイル名(K + YY + MM + DD)を厳密にパースする。
    _extract_date_from_filename は4桁年を前提にしていて2桁年のこの命名規則には
    使えない(誤った日付になり差分更新の対象選定を壊す)ため、専用の関数を用意する。
    """
    import re
    m = re.match(r"K(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        return ""
    yy, mo, d = m.groups()
    year = 2000 + int(yy)
    return f"{year:04d}{int(mo):02d}{int(d):02d}"


def build_history(files: Optional[List[Path]] = None) -> pd.DataFrame:
    if files is None:
        files = sorted(RAW_TXT_DIR.glob("*.TXT")) + sorted(RAW_TXT_DIR.glob("*.txt"))
        files = sorted(set(files))
    print(f"files to parse: {len(files)}")

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


def _select_files_for_incremental_update() -> tuple[List[Path], Optional[pd.DataFrame]]:
    """既存のOUT_CSVがあれば、その最大日付-OVERLAP_DAYS以降のファイルだけ選んで再パースする。"""
    all_files = sorted(set(sorted(RAW_TXT_DIR.glob("*.TXT")) + sorted(RAW_TXT_DIR.glob("*.txt"))))

    if not OUT_CSV.exists():
        print("既存のracer_race_history.csvが無いため、全件パースします。")
        return all_files, None

    existing = pd.read_csv(OUT_CSV, low_memory=False)
    existing["date"] = existing["date"].astype(str).str.zfill(8)
    max_date = existing["date"].max()

    from datetime import datetime, timedelta
    try:
        cutoff_dt = datetime.strptime(max_date, "%Y%m%d") - timedelta(days=OVERLAP_DAYS)
        cutoff = cutoff_dt.strftime("%Y%m%d")
    except Exception:
        cutoff = "00000000"

    # ファイル名から日付が抽出できない場合は安全側に倒して対象に含める
    target_files = [
        f for f in all_files
        if (_extract_date_from_k_filename(f.name) or "99999999") >= cutoff
    ]
    print(f"既存データの最大日付: {max_date} / 差分パース対象: {len(target_files)}件 (cutoff={cutoff}, 全{len(all_files)}件中)")
    return target_files, existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="既存ファイルを無視して全raw_txtを再パースする")
    args = parser.parse_args()

    if args.full:
        target_files = None
        existing = None
    else:
        target_files, existing = _select_files_for_incremental_update()

    df = build_history(files=target_files)
    print("newly parsed boat-race rows:", len(df))

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df = df[df["date"].str.match(r"^\d{8}$")].copy()
    df["racer_no"] = pd.to_numeric(df["racer_no"], errors="coerce").fillna(0).astype(int)
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce").fillna(99).astype(int)
    df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0).astype(int)
    df["st"] = pd.to_numeric(df["st"], errors="coerce")

    df = df[(df["racer_no"] > 0) & (df["race_no"].between(1, 12))].copy()

    if existing is not None and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)

    df["date"] = df["date"].astype(str).str.zfill(8)
    df = df.drop_duplicates(subset=["date", "venue", "race_no", "lane"], keep="last").copy()
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
