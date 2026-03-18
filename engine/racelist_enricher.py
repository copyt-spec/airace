# engine/racelist_enricher.py
# BOATRACE racelist から motor / boat を抽出して entries(6艇)にマージする
# 使い方: enrich_entries_with_racelist(entries, date, race_no, venue_code)

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

import requests
from bs4 import BeautifulSoup, Tag


UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
}

RE_INT = re.compile(r"\d+")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _jcd_str(venue_code: int) -> str:
    return str(int(venue_code)).zfill(2)


def _safe_int_from_text(s: str) -> Optional[int]:
    s = _clean(s)
    if not s:
        return None
    m = RE_INT.search(s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _find_table_racelist(soup: BeautifulSoup) -> Optional[Tag]:
    """
    "モーター" と "ボート" を含む racelist 本体テーブルを探す
    """
    tables = soup.find_all("table")
    for t in tables:
        txt = _clean(t.get_text(" "))
        if ("モーター" in txt) and ("ボート" in txt) and ("枠" in txt):
            return t
    return None


def _extract_header_indices(table: Tag) -> Tuple[Optional[int], Optional[int]]:
    """
    thead/th から「モーター」「ボート」の列indexを推定する
    """
    motor_idx = None
    boat_idx = None

    # まず thead を優先
    thead = table.find("thead")
    header_cells: List[str] = []
    if thead:
        tr = thead.find("tr")
        if tr:
            header_cells = [_clean(th.get_text(" ")) for th in tr.find_all(["th", "td"])]

    # thead が弱い時は、最初のtrをヘッダ扱いして拾う
    if not header_cells:
        tr0 = table.find("tr")
        if tr0:
            header_cells = [_clean(x.get_text(" ")) for x in tr0.find_all(["th", "td"])]

    # 見つける
    for i, h in enumerate(header_cells):
        if motor_idx is None and ("モーター" in h):
            motor_idx = i
        if boat_idx is None and ("ボート" in h):
            boat_idx = i

    return motor_idx, boat_idx


def _iter_lane_rows(table: Tag) -> List[Tag]:
    """
    racelist は tbody 分割のことがあるので、tbody->tr を全部集める
    """
    rows: List[Tag] = []
    tbodies = table.find_all("tbody")
    if tbodies:
        for tb in tbodies:
            rows.extend(tb.find_all("tr"))
    else:
        rows.extend(table.find_all("tr"))
    return rows


def _pick_lane_from_row(tr: Tag) -> Optional[int]:
    """
    行の先頭近くに 1..6 があることが多い
    """
    tds = tr.find_all("td")
    if not tds:
        return None
    lane = _safe_int_from_text(tds[0].get_text(" "))
    if lane in (1, 2, 3, 4, 5, 6):
        return lane

    # fallback: 行全文から最初の 1..6 を拾う
    txt = _clean(tr.get_text(" "))
    m = re.search(r"(^| )([1-6])( |$)", txt)
    if m:
        return int(m.group(2))
    return None


def _pick_motor_boat_from_row(tr: Tag, motor_idx: Optional[int], boat_idx: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    """
    motor_idx / boat_idx が取れていればその列から拾う。
    取れない場合は行テキストから「2つの番号」を推定（保険）
    """
    tds = tr.find_all("td")

    motor = None
    boat = None

    if motor_idx is not None and len(tds) > motor_idx:
        motor = _safe_int_from_text(tds[motor_idx].get_text(" "))

    if boat_idx is not None and len(tds) > boat_idx:
        boat = _safe_int_from_text(tds[boat_idx].get_text(" "))

    # どっちか欠けてたら保険推定
    if motor is None or boat is None:
        txt = _clean(tr.get_text(" "))
        nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", txt)]
        # racelist内には1..6や日付等も混じるので雑だけど、
        # motor/boat は 1〜99 のことが多い → その範囲で後ろ側を優先
        cand = [n for n in nums if 1 <= n <= 99]
        if len(cand) >= 2:
            # 後ろ2個を motor/boat とみなす（保険）
            if motor is None:
                motor = cand[-2]
            if boat is None:
                boat = cand[-1]

    return motor, boat


def enrich_entries_with_racelist(
    entries: List[Dict[str, Any]],
    date: str,
    race_no: int,
    venue_code: int,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    """
    entries: 6艇（lane,racer_no...）を想定。ここに motor / boat を上書き追加して返す。
    """
    if not entries:
        return entries

    jcd = _jcd_str(venue_code)
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?hd={date}&jcd={jcd}&rno={int(race_no)}"

    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
    except Exception:
        # 取れなくても致命ではない（精度が落ちるだけ）
        return entries

    soup = BeautifulSoup(r.text, "html.parser")
    table = _find_table_racelist(soup)
    if not table:
        return entries

    motor_idx, boat_idx = _extract_header_indices(table)
    lane_map: Dict[int, Dict[str, Any]] = {int(e.get("lane")): e for e in entries if str(e.get("lane")).isdigit()}

    for tr in _iter_lane_rows(table):
        lane = _pick_lane_from_row(tr)
        if lane not in (1, 2, 3, 4, 5, 6):
            continue
        if lane not in lane_map:
            continue

        motor, boat = _pick_motor_boat_from_row(tr, motor_idx, boat_idx)
        if motor is not None:
            lane_map[lane]["motor"] = motor
        if boat is not None:
            lane_map[lane]["boat"] = boat

    # lane順で返す
    out = []
    for lane in range(1, 7):
        if lane in lane_map:
            out.append(lane_map[lane])
    return out
