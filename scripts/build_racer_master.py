# scripts/build_racer_master.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

RAW_DIR = Path("data/raw_racer_txt")
OUT_FILE = Path("data/master/racers_master.csv")

ENCODING = "shift_jis"

# =========================
# 固定長レイアウト（バイト単位）
# =========================
POS = {
    "racer_no":      (0, 4),    # 4
    "name_kanji":    (4, 20),   # 16
    "name_kana":     (20, 35),  # 15
    "branch":        (35, 39),  # 4
    "grade":         (39, 41),  # 2
    "era":           (41, 42),  # 1
    "birth_ymd":     (42, 48),  # 6
    "sex":           (48, 49),  # 1
    "age":           (49, 51),  # 2
    "height":        (51, 54),  # 3
    "weight":        (54, 56),  # 2
    "blood":         (56, 58),  # 2
    "win_rate":      (58, 62),  # 4 -> 0756 => 7.56
    "place_rate":    (62, 66),  # 4 -> 0459 => 45.9
    "win_count_1":   (66, 69),  # 3
    "win_count_2":   (69, 72),  # 3
    "race_count":    (72, 75),  # 3
    "final_count":   (75, 77),  # 2
    "champ_count":   (77, 79),  # 2
    "avg_st":        (79, 82),  # 3 -> 016 => 0.16
}

# =========================
# コース別ブロック
# 1コースごとに13バイト
# 進入回数 3
# 複勝率   4
# 平均ST   3
# 平均ST順位 3
# =========================
COURSE_BASE = 82
COURSE_UNIT = 13

# 6コース分終了位置
# 82 + 13*6 = 160
TAIL_POS = {
    "prev_grade":        (160, 162),  # 2
    "prev2_grade":       (162, 164),  # 2
    "prev3_grade":       (164, 166),  # 2
    "prev_ability":      (166, 170),  # 4 -> 7400 => 74.00
    "current_ability":   (170, 174),  # 4 -> 7500 => 75.00
    "year":              (174, 178),  # 4
    "season":            (178, 179),  # 1
    "period_from":       (179, 187),  # 8
    "period_to":         (187, 195),  # 8
    "training_term":     (195, 198),  # 3
}

# 行末近くの出身地
BIRTHPLACE_LAST_BYTES = 6


def _slice_bytes(b: bytes, start: int, end: int) -> bytes:
    if start >= len(b):
        return b""
    return b[start:min(end, len(b))]


def _decode_sjis(bs: bytes) -> str:
    return bs.decode(ENCODING, errors="ignore").strip()


def _to_int(s: str, default: int = 0) -> int:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return int(s)
    except Exception:
        return default


def _scaled_2(s: str) -> float:
    # 0756 -> 7.56
    n = _to_int(s, 0)
    return n / 100.0


def _scaled_1(s: str) -> float:
    # 0459 -> 45.9
    n = _to_int(s, 0)
    return n / 10.0


def _scaled_st(s: str) -> float:
    # 016 -> 0.16
    n = _to_int(s, 0)
    return n / 100.0


def _safe_field(b: bytes, key: str) -> str:
    start, end = POS[key]
    return _decode_sjis(_slice_bytes(b, start, end))


def _safe_tail_field(b: bytes, key: str) -> str:
    start, end = TAIL_POS[key]
    return _decode_sjis(_slice_bytes(b, start, end))


def _course_offsets(course_no: int) -> Dict[str, tuple[int, int]]:
    base = COURSE_BASE + (course_no - 1) * COURSE_UNIT
    return {
        "entry_count": (base + 0,  base + 3),
        "place_rate":  (base + 3,  base + 7),
        "avg_st":      (base + 7,  base + 10),
        "st_rank":     (base + 10, base + 13),
    }


def _parse_course_block(b: bytes, course_no: int) -> Dict[str, Any]:
    off = _course_offsets(course_no)

    entry_count = _to_int(_decode_sjis(_slice_bytes(b, *off["entry_count"])), 0)
    place_rate = _scaled_1(_decode_sjis(_slice_bytes(b, *off["place_rate"])))
    avg_st = _scaled_st(_decode_sjis(_slice_bytes(b, *off["avg_st"])))

    st_rank_raw = _to_int(_decode_sjis(_slice_bytes(b, *off["st_rank"])), 0)
    st_rank = st_rank_raw / 100.0 if st_rank_raw else 0.0

    return {
        f"course{course_no}_entry_count": entry_count,
        f"course{course_no}_place_rate": place_rate,
        f"course{course_no}_avg_st": avg_st,
        f"course{course_no}_avg_st_rank": st_rank,
    }


def parse_line_bytes(b: bytes) -> Dict[str, Any]:
    racer_no = _safe_field(b, "racer_no")
    name_kanji = _safe_field(b, "name_kanji")
    name_kana = _safe_field(b, "name_kana")
    branch = _safe_field(b, "branch")
    grade = _safe_field(b, "grade")
    era = _safe_field(b, "era")
    birth_ymd = _safe_field(b, "birth_ymd")
    sex = _safe_field(b, "sex")
    age = _to_int(_safe_field(b, "age"), 0)
    height = _to_int(_safe_field(b, "height"), 0)
    weight = _to_int(_safe_field(b, "weight"), 0)
    blood = _safe_field(b, "blood")

    win_rate = _scaled_2(_safe_field(b, "win_rate"))
    place_rate = _scaled_1(_safe_field(b, "place_rate"))
    avg_st = _scaled_st(_safe_field(b, "avg_st"))

    prev_grade = _safe_tail_field(b, "prev_grade")
    prev2_grade = _safe_tail_field(b, "prev2_grade")
    prev3_grade = _safe_tail_field(b, "prev3_grade")
    prev_ability = _scaled_2(_safe_tail_field(b, "prev_ability"))
    current_ability = _scaled_2(_safe_tail_field(b, "current_ability"))
    year = _to_int(_safe_tail_field(b, "year"), 0)
    season = _to_int(_safe_tail_field(b, "season"), 0)
    period_from = _safe_tail_field(b, "period_from")
    period_to = _safe_tail_field(b, "period_to")
    training_term = _to_int(_safe_tail_field(b, "training_term"), 0)

    birthplace = _decode_sjis(b[-BIRTHPLACE_LAST_BYTES:]) if len(b) >= BIRTHPLACE_LAST_BYTES else ""

    row: Dict[str, Any] = {
        "racer_no": racer_no,
        "name": name_kanji,
        "kana": name_kana,
        "branch": branch,
        "grade": grade,
        "era": era,
        "birth_ymd": birth_ymd,
        "sex": sex,
        "age": age,
        "height": height,
        "weight": weight,
        "blood": blood,
        "win_rate": win_rate,
        "place_rate": place_rate,
        "avg_st": avg_st,
        "prev_grade": prev_grade,
        "prev2_grade": prev2_grade,
        "prev3_grade": prev3_grade,
        "prev_ability_index": prev_ability,
        "ability_index": current_ability,
        "year": year,
        "season": season,
        "period_from": period_from,
        "period_to": period_to,
        "training_term": training_term,
        "birthplace": birthplace,
    }

    for c in range(1, 7):
        row.update(_parse_course_block(b, c))

    return row


def main() -> None:
    rows: List[Dict[str, Any]] = []

    files = sorted(list(RAW_DIR.glob("*.txt")) + list(RAW_DIR.glob("*.TXT")))
    if not files:
        raise FileNotFoundError(f"No txt files found in {RAW_DIR}")

    for file in files:
        print("loading:", file)
        with open(file, "rb") as f:
            for raw in f:
                raw = raw.rstrip(b"\r\n")
                if len(raw) < 198:
                    continue

                try:
                    row = parse_line_bytes(raw)
                    racer_no = row.get("racer_no", "")
                    if not racer_no.isdigit():
                        continue
                    rows.append(row)
                except Exception:
                    continue

    df = pd.DataFrame(rows)

    # 同じ racer_no が複数TXTにいる場合は後勝ち
    df = df.drop_duplicates(subset=["racer_no"], keep="last").reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print("saved:", OUT_FILE)
    print("rows:", len(df))
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()
