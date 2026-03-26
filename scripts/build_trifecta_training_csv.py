# scripts/build_trifecta_training_csv.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    trans = str.maketrans({
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "Ａ": "A", "Ｂ": "B",
        "ａ": "a", "ｂ": "b",
        "－": "-", "ー": "-", "―": "-", "−": "-",
        "　": " ",
        "／": "/",
        "＝": "=",
    })
    return s.translate(trans)


def norm_combo(s: str) -> str:
    """
    Normalize combo like:
      "1-2-3", "１－２－３", "1 2 3", "1=2=3" -> "1-2-3"
    """
    s = norm_text(s)
    if not s:
        return ""

    s = s.replace("=", "-").replace("/", "-")
    nums = re.findall(r"[1-6]", s)
    if len(nums) >= 3:
        return f"{nums[0]}-{nums[1]}-{nums[2]}"

    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9\-]", "-", s)
    nums = re.findall(r"[1-6]", s)
    if len(nums) >= 3:
        return f"{nums[0]}-{nums[1]}-{nums[2]}"

    return ""


def detect_columns(fieldnames: List[str]) -> Dict[str, Optional[str]]:
    """
    startk_dataset.csv の列名が多少違っても対応できるように候補を探す。
    """
    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in fieldnames:
                return c
        return None

    return {
        "date": pick(["date", "race_date", "ymd", "YYYYMMDD"]),
        "venue": pick(["venue", "place", "stadium"]),
        "race_no": pick(["race_no", "rno", "race", "race_num"]),
        "y_combo": pick(["y_combo", "trifecta_combo", "combo", "result_combo", "3t_combo"]),
        "payout": pick(["trifecta_payout", "payout", "3t_payout", "odds_payout"]),
    }


def iter_trifecta_combos() -> List[str]:
    out = []
    for a, b, c in itertools.permutations([1, 2, 3, 4, 5, 6], 3):
        out.append(f"{a}-{b}-{c}")
    return out


def build_combo_to_class_map(combos: List[str]) -> Dict[str, int]:
    return {cb: i for i, cb in enumerate(combos)}


def make_race_id(date: str, venue: str, race_no: str, serial: int) -> str:
    v = norm_text(venue or "UNKNOWN")
    v = re.sub(r"\s+", "", v)
    v = re.sub(r"[^\w\u3000-\u9fffぁ-んァ-ン一-龠]", "", v)

    d = norm_text(date or "00000000")
    r = norm_text(race_no or "0")
    return f"{d}_{v}_{r}_{serial:06d}"


def split_combo(combo: str) -> Tuple[int, int, int]:
    combo = norm_combo(combo)
    if not combo:
        return 0, 0, 0
    a, b, c = combo.split("-")
    return int(a), int(b), int(c)


def grade_to_num(g: str) -> str:
    """
    A1=4, A2=3, B1=2, B2=1
    """
    s = norm_text(g).upper()
    mp = {
        "A1": "4",
        "A2": "3",
        "B1": "2",
        "B2": "1",
    }
    return mp.get(s, "")


def add_combo_features(base_row: Dict[str, str], combo: str) -> Dict[str, str]:
    """
    combo由来の特徴量を追加
    """
    out: Dict[str, str] = {}

    a, b, c = split_combo(combo)

    out["combo_first_lane"] = str(a)
    out["combo_second_lane"] = str(b)
    out["combo_third_lane"] = str(c)

    out["combo_lane_sum"] = str(a + b + c)
    out["combo_lane_range"] = str(max(a, b, c) - min(a, b, c))
    out["combo_lane_mean"] = f"{(a + b + c) / 3.0:.6f}"

    out["combo_first_inside_flag"] = "1" if a <= 2 else "0"
    out["combo_first_center_flag"] = "1" if a in (3, 4) else "0"
    out["combo_first_outside_flag"] = "1" if a >= 5 else "0"

    out["combo_forward_order_flag"] = "1" if (a < b < c) else "0"
    out["combo_reverse_order_flag"] = "1" if (a > b > c) else "0"

    out["combo_first_inner2_flag"] = "1" if a in (1, 2) else "0"
    out["combo_second_inner3_flag"] = "1" if b in (1, 2, 3) else "0"
    out["combo_third_outer3_flag"] = "1" if c in (4, 5, 6) else "0"

    return out


def add_label_features(combo: str, y_combo_norm: str, combo_to_class: Dict[str, int]) -> Dict[str, str]:
    """
    学習ターゲット系
    """
    out: Dict[str, str] = {}

    out["y_combo"] = y_combo_norm
    out["is_hit"] = "1" if (y_combo_norm and combo == y_combo_norm) else "0"
    out["y"] = out["is_hit"]  # 後方互換
    out["y_class"] = str(combo_to_class.get(y_combo_norm, -1))

    a, b, c = split_combo(combo)
    y1, y2, y3 = split_combo(y_combo_norm)

    out["first_only_hit"] = "1" if (a == y1 and y1 != 0) else "0"
    out["first2_hit"] = "1" if (a == y1 and b == y2 and y1 != 0 and y2 != 0) else "0"

    return out


def add_base_normalized_features(base_row: Dict[str, str], fieldnames: List[str]) -> Dict[str, str]:
    """
    元CSVに grade があるなら *_grade_num も追加しておく。
    lane1_grade, lane2_grade ... や grade1, grade2 的な名前に幅広く対応。
    """
    out: Dict[str, str] = {}

    # laneごとの grade_num を補助的に作る
    for lane in range(1, 7):
        candidates = [
            f"lane{lane}_grade",
            f"lane{lane}_class",
            f"grade{lane}",
            f"class{lane}",
            f"grade_{lane}",
            f"class_{lane}",
        ]
        found = ""
        for c in candidates:
            if c in fieldnames:
                found = base_row.get(c, "")
                break
        if found:
            out[f"lane{lane}_grade_num"] = grade_to_num(found)

    return out


def expand_one_race(
    base_row: Dict[str, str],
    race_id: str,
    combos: List[str],
    combo_to_class: Dict[str, int],
    y_combo_norm: str,
    payout_value: str,
    out_fields: List[str],
    fieldnames: List[str],
) -> Iterable[Dict[str, str]]:
    """
    1レース（1行）→120行へ展開
    """
    normalized_extra = add_base_normalized_features(base_row, fieldnames)

    for cb in combos:
        row: Dict[str, str] = {}

        # base features を全部コピー
        for k in fieldnames:
            row[k] = base_row.get(k, "")

        # 追加列
        row["race_id"] = race_id
        row["combo"] = cb
        row["payout"] = payout_value

        # combo構造特徴
        row.update(add_combo_features(base_row, cb))

        # target特徴
        row.update(add_label_features(cb, y_combo_norm, combo_to_class))

        # base由来の補助特徴
        row.update(normalized_extra)

        # 欠損埋め（out_fields にあるのに未設定なら空文字）
        for k in out_fields:
            if k not in row:
                row[k] = ""

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
    combo_to_class = build_combo_to_class_map(combos)

    with open(in_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header/fieldnames.")

        fieldnames = reader.fieldnames
        colmap = detect_columns(fieldnames)

        date_col = colmap["date"]
        venue_col = colmap["venue"]
        race_col = colmap["race_no"]
        y_combo_col = colmap["y_combo"]
        payout_col = colmap["payout"]

        base_fields = list(fieldnames)

        extra_fields = [
            "race_id",
            "combo",
            "payout",
            "y_combo",
            "is_hit",
            "y",
            "y_class",
            "first_only_hit",
            "first2_hit",
            "combo_first_lane",
            "combo_second_lane",
            "combo_third_lane",
            "combo_lane_sum",
            "combo_lane_range",
            "combo_lane_mean",
            "combo_first_inside_flag",
            "combo_first_center_flag",
            "combo_first_outside_flag",
            "combo_forward_order_flag",
            "combo_reverse_order_flag",
            "combo_first_inner2_flag",
            "combo_second_inner3_flag",
            "combo_third_outer3_flag",
        ]

        # grade_num補助列
        extra_grade_num_fields = [f"lane{i}_grade_num" for i in range(1, 7)]

        out_fields = extra_fields + extra_grade_num_fields + base_fields

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
                    combo_to_class=combo_to_class,
                    y_combo_norm=y_combo_norm,
                    payout_value=payout_val,
                    out_fields=out_fields,
                    fieldnames=fieldnames,
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
