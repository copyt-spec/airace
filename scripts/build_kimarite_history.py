from __future__ import annotations

"""
data/raw_txt 全件(mbrace Kファイル形式)から決まり手(逃げ/差し/まくり/まくり差し/抜き/恵まれ)を
抽出し、data/datasets/kimarite_history.csv (date, venue, race_no, kimarite, y_combo,
trifecta_payout, winner_lane, winner_course) を作る。

engine/txt_race_parser.py の parse_startk_multi_venue_txt() が
2026-07-26に決まり手抽出(RaceRecord.kimarite)対応済み。

サンドボックスの45秒bash上限に収まるよう、--start/--end でファイルを
範囲指定して分割実行できるようにしてある(1213ファイル全部を1回でやると
40秒超で危険なため、数百件ずつ複数回に分けて呼ぶ想定)。
1回目は新規ファイル作成(ヘッダ付き)、2回目以降は追記(--append)。

使い方(例: 600件ずつ2回に分けて実行):
  python scripts/build_kimarite_history.py --start 0 --end 600
  python scripts/build_kimarite_history.py --start 600 --end 1213 --append
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.txt_race_parser import parse_startk_multi_venue_txt  # noqa: E402

RAW_TXT_DIR = PROJECT_ROOT / "data" / "raw_txt"
OUT_CSV = PROJECT_ROOT / "data" / "datasets" / "kimarite_history.csv"


def _fallback_date_from_filename(path: Path) -> str:
    # K260701.TXT -> 20260701
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) == 6:
        return "20" + digits
    return digits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10**9)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    files = sorted(RAW_TXT_DIR.glob("*.TXT"))
    subset = files[args.start:args.end]
    print(f"processing files[{args.start}:{args.end}] -> {len(subset)} files "
          f"(total available: {len(files)})")

    rows = []
    n_records = 0
    for f in subset:
        try:
            text = f.read_text(encoding="shift_jis", errors="ignore")
        except Exception as e:
            print(f"[WARN] read failed: {f.name}: {e}")
            continue
        fallback_date = _fallback_date_from_filename(f)
        try:
            records = parse_startk_multi_venue_txt(text, fallback_date=fallback_date)
        except Exception as e:
            print(f"[WARN] parse failed: {f.name}: {e}")
            continue

        for r in records:
            if len(r.boats) < 6:
                continue
            winner = next((b for b in r.boats if b.finish == 1), None)
            rows.append({
                "date": r.meta.date,
                "venue": r.meta.venue,
                "race_no": r.meta.race_no,
                "kimarite": r.kimarite,
                "y_combo": r.y_combo,
                "trifecta_payout": r.trifecta_payout,
                "winner_lane": winner.lane if winner else None,
                "winner_course": winner.course if winner else None,
                "wind_speed_mps": r.meta.wind_speed_mps,
                "wave_cm": r.meta.wave_cm,
            })
        n_records += len(records)

    df = pd.DataFrame(rows)
    print(f"parsed rows: {len(df)} (raw records incl. <6boats: {n_records})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if args.append and OUT_CSV.exists():
        df.to_csv(OUT_CSV, mode="a", header=False, index=False, encoding="utf-8")
        print("appended to:", OUT_CSV)
    else:
        tmp_path = OUT_CSV.with_suffix(".csv.tmp")
        df.to_csv(tmp_path, index=False, encoding="utf-8")
        tmp_path.replace(OUT_CSV)
        print("wrote (fresh):", OUT_CSV)


if __name__ == "__main__":
    main()
