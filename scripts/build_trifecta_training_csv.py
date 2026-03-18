# scripts/build_trifecta_training_csv.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


def norm_combo(s: str) -> str:
    """
    Normalize combo like:
      "1-2-3", "１－２－３", "1 2 3", "1=2=3" -> "1-2-3"
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    # 全角→半角っぽい最低限（数字とハイフン系）
    trans = str.maketrans({
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "－": "-", "ー": "-", "―": "-", "−": "-",
        "　": " ",
        "=": "-",
        "／": "/",
    })
    s = s.translate(trans)

    # 数字を3つ拾って "a-b-c" にする（区切り文字は何でもOK）
    nums = re.findall(r"[1-6]", s)
    if len(nums) >= 3:
        return f"{nums[0]}-{nums[1]}-{nums[2]}"

    # それでもダメなら、ハイフンっぽいものを統一
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9\-]", "-", s)
    return s


def detect_columns(fieldnames: List[str]) -> Dict[str, Optional[str]]:
    """
    startk_dataset.csv の列名が多少違っても対応できるように、
    よくある候補から探す。
    """
    # date
    date_candidates = ["date", "race_date", "ymd", "YYYYMMDD"]
    # venue
    venue_candidates = ["venue", "place", "stadium"]
    # race_no
    race_candidates = ["race_no", "rno", "race", "race_num"]

    # label
    combo_candidates = ["y_combo", "trifecta_combo", "combo", "result_combo", "3t_combo"]
    payout_candidates = ["trifecta_payout", "payout", "3t_payout", "odds_payout"]

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in fieldnames:
                return c
        return None

    return {
        "date": pick(date_candidates),
        "venue": pick(venue_candidates),
        "race_no": pick(race_candidates),
        "y_combo": pick(combo_candidates),
        "payout": pick(payout_candidates),
    }


def iter_trifecta_combos() -> List[str]:
    """All 120 permutations of 1..6 taken 3."""
    out = []
    for a, b, c in itertools.permutations([1, 2, 3, 4, 5, 6], 3):
        out.append(f"{a}-{b}-{c}")
    return out


def make_race_id(date: str, venue: str, race_no: str, serial: int) -> str:
    # venueはCSVに全角スペースや記号が混ざることがあるので簡単に正規化
    v = (venue or "UNKNOWN").strip()
    v = re.sub(r"\s+", "", v)
    v = re.sub(r"[^\w\u3000-\u9fffぁ-んァ-ン一-龠]", "", v)  # 日本語/英数を主に残す
    d = (date or "00000000").strip()
    r = (race_no or "0").strip()
    return f"{d}_{v}_{r}_{serial:06d}"


def expand_one_race(
    base_row: Dict[str, str],
    race_id: str,
    combos: List[str],
    y_combo_norm: str,
    payout_value: str,
    out_fields: List[str],
) -> Iterable[Dict[str, str]]:
    """
    1レース（1行）→120行へ展開
    """
    for cb in combos:
        row = {}
        # base features
        for k in out_fields:
            if k in ("race_id", "combo", "y", "payout"):
                continue
            row[k] = base_row.get(k, "")

        row["race_id"] = race_id
        row["combo"] = cb
        row["y"] = "1" if (y_combo_norm and cb == y_combo_norm) else "0"
        row["payout"] = payout_value
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand startk_dataset.csv into 120 trifecta rows per race.")
    parser.add_argument(
        "--in",
        dest="in_path",
        default="data/datasets/startk_dataset.csv",
        help="Input race-level CSV (default: data/datasets/startk_dataset.csv)",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/datasets/trifecta_train.csv",
        help="Output expanded CSV (default: data/datasets/trifecta_train.csv)",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=0,
        help="For quick test: limit number of races processed (0 = no limit)",
    )
    parser.add_argument(
        "--require-label",
        action="store_true",
        help="If set: skip races that don't have y_combo (label).",
    )
    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    combos = iter_trifecta_combos()

    # 読み込み
    # utf-8-sig を優先（Windows系CSVに強い）
    with open(in_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header/fieldnames.")

        fieldnames = reader.fieldnames
        colmap = detect_columns(fieldnames)

        # 必須列（date/venue/race_no）が無い場合でも動かしたいので、
        # 無ければ空扱い + serialで一意race_idにする
        date_col = colmap["date"]
        venue_col = colmap["venue"]
        race_col = colmap["race_no"]
        y_combo_col = colmap["y_combo"]
        payout_col = colmap["payout"]

        # 出力列：元の列を全部持つ（ラベル列が混在しててもOK）
        # ただし y_combo/payout は別列にしたいので、元の y_combo はそのまま残しても良いが、
        # 学習側は "combo/y" を使う前提にする。
        base_fields = list(fieldnames)

        # 追加する列
        out_fields = ["race_id", "combo", "y", "payout"] + base_fields

        with open(out_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()

            total_races = 0
            total_rows = 0
            skipped_unlabeled = 0

            for serial, row in enumerate(reader, start=1):
                total_races += 1
                if args.max_races and total_races > args.max_races:
                    break

                date_val = row.get(date_col, "") if date_col else ""
                venue_val = row.get(venue_col, "") if venue_col else ""
                race_no_val = row.get(race_col, "") if race_col else ""

                y_combo_val = row.get(y_combo_col, "") if y_combo_col else ""
                y_combo_norm = norm_combo(y_combo_val)

                if args.require_label and not y_combo_norm:
                    skipped_unlabeled += 1
                    continue

                payout_val = row.get(payout_col, "") if payout_col else ""

                race_id = make_race_id(date_val, venue_val, race_no_val, serial)

                for out_row in expand_one_race(
                    base_row=row,
                    race_id=race_id,
                    combos=combos,
                    y_combo_norm=y_combo_norm,
                    payout_value=payout_val,
                    out_fields=out_fields,
                ):
                    writer.writerow(out_row)
                    total_rows += 1

            print("===== build_trifecta_training_csv DONE =====")
            print(f"input races processed: {total_races}")
            if args.require_label:
                print(f"skipped (no label): {skipped_unlabeled}")
            print(f"output rows: {total_rows}")
            print(f"output file: {out_path}")


if __name__ == "__main__":
    main()
