from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from engine.txt_race_parser import parse_startk_multi_venue_txt, RaceRecord


VENUE_CODE_MAP = {
    "桐生": "01",
    "戸田": "02",
    "江戸川": "03",
    "平和島": "04",
    "多摩川": "05",
    "浜名湖": "06",
    "蒲郡": "07",
    "常滑": "08",
    "津": "09",
    "三国": "10",
    "びわこ": "11",
    "住之江": "12",
    "尼崎": "13",
    "鳴門": "14",
    "丸亀": "15",
    "児島": "16",
    "宮島": "17",
    "徳山": "18",
    "下関": "19",
    "若松": "20",
    "芦屋": "21",
    "福岡": "22",
    "唐津": "23",
    "大村": "24",
}

CODE_TO_VENUE = {v: k for k, v in VENUE_CODE_MAP.items()}


def _combo_to_class_id(combo: str) -> int:
    lanes = [1, 2, 3, 4, 5, 6]
    combos = []
    for a in lanes:
        for b in lanes:
            for c in lanes:
                if a != b and b != c and a != c:
                    combos.append(f"{a}-{b}-{c}")
    return combos.index(combo)


def _date_from_filename(filename: str) -> Optional[str]:
    m = re.search(r"([0-9]{2})([0-9]{2})([0-9]{2})", filename)
    if not m:
        return None
    yy = int(m.group(1))
    mm = int(m.group(2))
    dd = int(m.group(3))
    yyyy = 2000 + yy
    return f"{yyyy:04d}{mm:02d}{dd:02d}"


def _normalize_venue(v: Any) -> str:
    if v is None:
        return "UNKNOWN"
    s = str(v).replace("\u3000", "").replace(" ", "").strip()
    if s in VENUE_CODE_MAP:
        return s
    return "UNKNOWN"


def _normalize_venue_code(v: Any) -> str:
    if v is None:
        return "00"
    s = str(v).strip().zfill(2)
    if s in CODE_TO_VENUE:
        return s
    return "00"


def _recover_venue(venue: str, venue_code: str) -> tuple[str, str]:
    venue = _normalize_venue(venue)
    venue_code = _normalize_venue_code(venue_code)

    if venue != "UNKNOWN" and venue_code == "00":
        venue_code = VENUE_CODE_MAP.get(venue, "00")

    if venue == "UNKNOWN" and venue_code != "00":
        venue = CODE_TO_VENUE.get(venue_code, "UNKNOWN")

    return venue, venue_code


def _wide_row_from_record(rec: RaceRecord) -> Dict[str, Any]:
    venue = getattr(rec.meta, "venue", "UNKNOWN")
    venue_code = getattr(rec.meta, "venue_code", "00")
    venue, venue_code = _recover_venue(venue, venue_code)

    row: Dict[str, Any] = {}
    row["date"] = rec.meta.date
    row["venue"] = venue
    row["venue_code"] = venue_code
    row["race_no"] = rec.meta.race_no
    row["race_key"] = f"{rec.meta.date}_{venue_code}_{int(rec.meta.race_no)}"

    row["weather"] = rec.meta.weather
    row["wind_dir"] = rec.meta.wind_dir
    row["wind_speed_mps"] = rec.meta.wind_speed_mps
    row["wave_cm"] = rec.meta.wave_cm

    for b in rec.boats:
        p = f"lane{b.lane}_"
        row[p + "racer_no"] = b.racer_no
        row[p + "motor"] = b.motor
        row[p + "boat"] = b.boat
        row[p + "exhibit"] = b.exhibit
        row[p + "course"] = b.course
        row[p + "st"] = b.st
        row[p + "finish"] = b.finish

    row["y_combo"] = rec.y_combo
    row["y_class"] = _combo_to_class_id(rec.y_combo) if rec.y_combo else None
    row["trifecta"] = rec.y_combo
    row["trifecta_payout"] = rec.trifecta_payout

    return row


@dataclass
class TxtDatasetConfig:
    raw_txt_dir: str
    out_csv_path: str
    keep_unlabeled: bool = False
    verbose: bool = True


def build_dataset_from_txt(cfg: TxtDatasetConfig) -> None:
    all_rows: List[Dict[str, Any]] = []

    files = [fn for fn in sorted(os.listdir(cfg.raw_txt_dir)) if fn.lower().endswith(".txt")]
    if cfg.verbose:
        print("TXT files:", len(files))

    for fn in files:
        path = os.path.join(cfg.raw_txt_dir, fn)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        fallback_date = _date_from_filename(fn)
        records = parse_startk_multi_venue_txt(text, fallback_date=fallback_date)

        total = 0
        labeled = 0
        kept = 0
        unknown_venue = 0
        zero_code = 0

        for rec in records:
            total += 1

            row = _wide_row_from_record(rec)

            if row["venue"] == "UNKNOWN":
                unknown_venue += 1
            if row["venue_code"] == "00":
                zero_code += 1

            if (not cfg.keep_unlabeled) and (not row.get("y_combo")):
                continue

            kept += 1
            if row.get("y_combo"):
                labeled += 1

            all_rows.append(row)

        if cfg.verbose:
            print(
                f"[{fn}] total_races={total} labeled={labeled} kept={kept} "
                f"unknown_venue={unknown_venue} zero_code={zero_code}"
            )

    if not all_rows:
        raise RuntimeError("No rows parsed from TXT files.")

    if cfg.verbose:
        print("rows before dedup:", len(all_rows))

    # race_key 単位で重複除去
    # UNKNOWN/00 は後で検査するので、ここでは単純に先勝ち
    seen = set()
    deduped: List[Dict[str, Any]] = []
    dup_count = 0

    for r in all_rows:
        rk = r.get("race_key")
        if rk in seen:
            dup_count += 1
            continue
        seen.add(rk)
        deduped.append(r)

    all_rows = deduped

    if cfg.verbose:
        print("duplicate race_key rows:", dup_count)
        print("rows after dedup:", len(all_rows))
        print("race_key nunique after dedup:", len({r['race_key'] for r in all_rows}))

    remain_unknown = sum(1 for r in all_rows if r["venue"] == "UNKNOWN")
    remain_zero_code = sum(1 for r in all_rows if r["venue_code"] == "00")

    if remain_unknown > 0 or remain_zero_code > 0:
        preview = [r for r in all_rows if r["venue"] == "UNKNOWN" or r["venue_code"] == "00"][:20]
        preview_lines = []
        for r in preview:
            preview_lines.append(
                f"{r['date']} {r['venue']} {r['venue_code']} {r['race_no']} {r['race_key']}"
            )

        raise RuntimeError(
            "venue/venue_code の復元に失敗した行が残っています。\n"
            f"remain_unknown={remain_unknown}, remain_zero_code={remain_zero_code}\n"
            + "\n".join(preview_lines)
        )

    keys = set()
    for r in all_rows:
        keys.update(r.keys())
    header = sorted(keys)

    os.makedirs(os.path.dirname(cfg.out_csv_path), exist_ok=True)
    with open(cfg.out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    if cfg.verbose:
        print("DONE rows:", len(all_rows), "->", cfg.out_csv_path)
