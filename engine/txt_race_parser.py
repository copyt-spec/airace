from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


VENUE_CODE_MAP = {
    "01": "桐生",
    "02": "戸田",
    "03": "江戸川",
    "04": "平和島",
    "05": "多摩川",
    "06": "浜名湖",
    "07": "蒲郡",
    "08": "常滑",
    "09": "津",
    "10": "三国",
    "11": "びわこ",
    "12": "住之江",
    "13": "尼崎",
    "14": "鳴門",
    "15": "丸亀",
    "16": "児島",
    "17": "宮島",
    "18": "徳山",
    "19": "下関",
    "20": "若松",
    "21": "芦屋",
    "22": "福岡",
    "23": "唐津",
    "24": "大村",
}


@dataclass
class RaceMeta:
    venue: str
    date: str  # YYYYMMDD
    race_no: int
    weather: Optional[str] = None
    wind_dir: Optional[str] = None
    wind_speed_mps: Optional[float] = None
    wave_cm: Optional[float] = None
    venue_code: Optional[str] = None


@dataclass
class BoatRow:
    finish: int
    lane: int
    racer_no: int
    motor: Optional[int] = None
    boat: Optional[int] = None
    exhibit: Optional[float] = None
    course: Optional[int] = None
    st: Optional[float] = None  # F は負値
    st_raw: Optional[str] = None


@dataclass
class RaceRecord:
    meta: RaceMeta
    boats: List[BoatRow]
    y_combo: Optional[str] = None
    trifecta_payout: Optional[int] = None


_NUM = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
_START_RE = re.compile(r"^\s*(\d{2})KBGN\s*$")
_END_RE = re.compile(r"^\s*(\d{2})KEND\s*$")

_ZEN2HAN = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "Ｒ": "R", "ｒ": "r",
    "［": "[", "］": "]",
    "－": "-", "ー": "-", "―": "-", "−": "-",
    "　": " ",
})


def _norm_line(s: str) -> str:
    s2 = s.translate(_ZEN2HAN)
    s2 = re.sub(r"[ \t]+", " ", s2)
    return s2.rstrip("\n")


def _to_float(s: str) -> Optional[float]:
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _date_to_yyyymmdd(s: str) -> Optional[str]:
    m = re.search(r"(\d{4})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})", s)
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2))
    d = int(m.group(3))
    return f"{y:04d}{mo:02d}{d:02d}"


def _parse_weather_line(line: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    m_weather = re.search(r"(晴れ|曇り|雨|雪|くもり)", line)
    if m_weather:
        out["weather"] = m_weather.group(1)

    m_wdir = re.search(r"(北東|南東|南西|北西|北|東|南|西|無風)", line)
    if m_wdir:
        out["wind_dir"] = m_wdir.group(1)

    m_wspd = re.search(r"(\d+(?:\.\d+)?)\s*m", line)
    if m_wspd:
        out["wind_speed_mps"] = float(m_wspd.group(1))

    m_wave = re.search(r"(\d+(?:\.\d+)?)\s*cm", line)
    if m_wave:
        out["wave_cm"] = float(m_wave.group(1))

    return out


def _parse_st(st_token: str) -> Tuple[Optional[float], str]:
    s = st_token.strip().upper()
    if s.startswith("F"):
        v = _to_float(s)
        if v is None:
            return None, s
        return -abs(v), s
    v = _to_float(s)
    return v, s


def _extract_venue_from_lines(lines: List[str], venue_code: Optional[str]) -> str:
    # まずコード優先
    if venue_code and venue_code in VENUE_CODE_MAP:
        return VENUE_CODE_MAP[venue_code]

    header_re = re.compile(r"^\s*(.+?)\s*\[成績\]")
    for ln in lines[:80]:
        m = header_re.match(ln)
        if m:
            raw = re.sub(r"\s+", "", m.group(1))
            if raw:
                return raw

    for ln in lines[:120]:
        if "ボートレース" in ln:
            raw = re.sub(r"\s+", "", ln.replace("ボートレース", ""))
            if raw:
                return raw

    return "UNKNOWN"


def _extract_trifecta_map(section_lines: List[str]) -> Dict[int, Tuple[str, int]]:
    trifecta_map: Dict[int, Tuple[str, int]] = {}
    in_pay = False

    p_line = re.compile(
        r"^\s*(\d{1,2})\s*[Rr]\b"
        r".*?"
        r"([1-6])\D+([1-6])\D+([1-6])"
        r"\s+(\d{3,})\b"
    )

    p_line_kw = re.compile(
        r"^\s*(\d{1,2})\s*[Rr]\b"
        r".*?(?:3連単|３連単|三連単).*?"
        r"([1-6])\D+([1-6])\D+([1-6])"
        r"\s+(\d{3,})\b"
    )

    for raw in section_lines:
        ln = _norm_line(raw)

        if ("[払戻金]" in ln) or ("払戻金" in ln):
            in_pay = True
            continue

        if not in_pay:
            if p_line.search(ln) or p_line_kw.search(ln):
                in_pay = True
            else:
                continue

        if re.match(r"^\s*1\s*[Rr]\b", ln) and ("H" in ln and "m" in ln):
            break

        m = p_line.search(ln) or p_line_kw.search(ln)
        if not m:
            continue

        rno = int(m.group(1))
        a, b, c = m.group(2), m.group(3), m.group(4)
        payout = int(m.group(5))
        trifecta_map[rno] = (f"{a}-{b}-{c}", payout)

    return trifecta_map


def _find_detail_race_starts(section_lines: List[str]) -> List[Tuple[int, int]]:
    head_re = re.compile(r"^\s*(\d{1,2})\s*[Rr]\b")
    starts: List[Tuple[int, int]] = []

    for i, raw in enumerate(section_lines):
        ln = _norm_line(raw)
        m = head_re.match(ln)
        if not m:
            continue

        if not ("H" in ln and "m" in ln):
            continue

        rno = int(m.group(1))
        if 1 <= rno <= 12:
            starts.append((rno, i))

    starts.sort(key=lambda x: x[1])

    seen = set()
    uniq: List[Tuple[int, int]] = []
    for rno, idx in starts:
        if rno in seen:
            continue
        seen.add(rno)
        uniq.append((rno, idx))

    return uniq


def _split_by_kbgn_kend(all_lines: List[str]) -> List[Tuple[Optional[str], List[str]]]:
    sections: List[Tuple[Optional[str], List[str]]] = []

    current_code: Optional[str] = None
    current_lines: List[str] = []
    in_block = False

    for raw in all_lines:
        ln = _norm_line(raw)

        m_start = _START_RE.match(ln)
        if m_start:
            if in_block and current_lines:
                sections.append((current_code, current_lines))
            current_code = m_start.group(1)
            current_lines = []
            in_block = True
            continue

        m_end = _END_RE.match(ln)
        if m_end and in_block:
            if current_lines:
                sections.append((current_code, current_lines))
            current_code = None
            current_lines = []
            in_block = False
            continue

        if in_block:
            current_lines.append(ln)

    if in_block and current_lines:
        sections.append((current_code, current_lines))

    if not sections:
        return [(None, list(map(_norm_line, all_lines)))]

    return sections


def _parse_single_venue_section(
    section_lines: List[str],
    fallback_date: Optional[str],
    venue_code: Optional[str],
) -> List[RaceRecord]:
    lines = [_norm_line(x) for x in section_lines]

    venue = _extract_venue_from_lines(lines, venue_code)

    date = None
    for ln in lines[:300]:
        d = _date_to_yyyymmdd(ln)
        if d:
            date = d
            break
    date = date or fallback_date or "00000000"

    trifecta_map = _extract_trifecta_map(lines)
    race_starts = _find_detail_race_starts(lines)
    if not race_starts:
        return []

    row_re = re.compile(r"^\s*([0-9SKF]{2})\s+([1-6])\s+(\d{4})\s+")
    records: List[RaceRecord] = []

    for idx, (rno, start_i) in enumerate(race_starts):
        end_i = race_starts[idx + 1][1] if idx + 1 < len(race_starts) else len(lines)
        block = lines[start_i:end_i]

        meta_kwargs: Dict[str, Any] = {
            "venue": venue,
            "date": date,
            "race_no": rno,
            "venue_code": venue_code or "00",
        }

        for ln in block[:15]:
            if "風" in ln and "波" in ln:
                meta_kwargs.update(_parse_weather_line(ln))
                break

        meta = RaceMeta(**meta_kwargs)

        boats: List[BoatRow] = []
        for ln in block:
            m = row_re.match(ln)
            if not m:
                continue

            finish_raw = m.group(1)
            lane = int(m.group(2))
            racer_no = int(m.group(3))
            tokens = ln.split()

            if finish_raw.isdigit():
                finish = int(finish_raw)
            else:
                # S1 / K0 / F1 などは完走外扱い
                finish = 99

            exhibit = None
            for t in tokens:
                if re.fullmatch(r"\d\.\d{2}", t):
                    exhibit = float(t)
                    break

            st_token = None
            for t in reversed(tokens):
                if re.fullmatch(r"F?\.\d+|F\d+\.\d+|\d+\.\d+|\.\d+", t, flags=re.IGNORECASE):
                    st_token = t
                    break

            st_val, st_raw = (None, None)
            if st_token:
                st_val, st_raw = _parse_st(st_token)

            course = None
            for t in reversed(tokens):
                if t.isdigit():
                    v = int(t)
                    if 1 <= v <= 6:
                        course = v
                        break

            motor = None
            boat = None
            ex_idx = None
            for i_t, t in enumerate(tokens):
                if re.fullmatch(r"\d\.\d{2}", t):
                    ex_idx = i_t
                    break

            if ex_idx is not None:
                prev_nums: List[int] = []
                for t in tokens[:ex_idx]:
                    if re.fullmatch(r"\d{1,3}", t):
                        prev_nums.append(int(t))
                if len(prev_nums) >= 2:
                    motor, boat = prev_nums[-2], prev_nums[-1]

            boats.append(
                BoatRow(
                    finish=finish,
                    lane=lane,
                    racer_no=racer_no,
                    motor=motor,
                    boat=boat,
                    exhibit=exhibit,
                    course=course,
                    st=st_val,
                    st_raw=st_raw,
                )
            )

        if len(boats) < 6:
            continue

        y_combo = None
        payout = None
        if rno in trifecta_map:
            y_combo, payout = trifecta_map[rno]

        records.append(
            RaceRecord(
                meta=meta,
                boats=boats[:6],
                y_combo=y_combo,
                trifecta_payout=payout,
            )
        )

    return records


def parse_startk_multi_venue_txt(text: str, fallback_date: Optional[str] = None) -> List[RaceRecord]:
    raw_lines = text.splitlines()
    kbgn_sections = _split_by_kbgn_kend(raw_lines)

    records: List[RaceRecord] = []
    for venue_code, sec_lines in kbgn_sections:
        records.extend(
            _parse_single_venue_section(
                sec_lines,
                fallback_date=fallback_date,
                venue_code=venue_code,
            )
        )

    return records
