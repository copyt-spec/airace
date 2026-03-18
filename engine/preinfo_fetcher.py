# engine/preinfo_fetcher.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import re

import requests
from bs4 import BeautifulSoup


@dataclass
class PreInfo:
    weather: str = ""
    wind_dir: str = ""
    wind_speed_mps: float = 0.0
    wave_cm: float = 0.0


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def fetch_racelist_preinfo_and_exhibit(
    venue_code: int,
    date: str,
    race_no: int,
    timeout: int = 10,
) -> Tuple[PreInfo, Dict[int, Dict[str, Any]]]:
    """
    BOATRACE racelist から
      - 天候/風向/風速/波
      - 各laneの展示タイム(exhibit) / ST(start_timing)
    を取得して返す。

    returns:
      (PreInfo, lane_map)
        lane_map[lane] = {"exhibit": "6.83", "start_timing": "0.18"}
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={int(race_no)}&jcd={int(venue_code)}&hd={date}"
    res = requests.get(url, timeout=timeout)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    # --- PreInfo（天候/風/波） ---
    pre = PreInfo()

    # なるべく壊れにくい抽出（テキストから拾う）
    page_text = _clean(soup.get_text(" ", strip=True))

    # 天候
    m = re.search(r"天候[:：]?\s*([^\s]+)", page_text)
    if m:
        pre.weather = _clean(m.group(1))

    # 風向
    m = re.search(r"風向[:：]?\s*([^\s]+)", page_text)
    if m:
        pre.wind_dir = _clean(m.group(1))

    # 風速
    m = re.search(r"風速[:：]?\s*([0-9.]+)\s*m", page_text)
    if m:
        pre.wind_speed_mps = _to_float(m.group(1))

    # 波高
    m = re.search(r"波高[:：]?\s*([0-9.]+)\s*cm", page_text)
    if m:
        pre.wave_cm = _to_float(m.group(1))

    # --- 各laneの展示 / ST ---
    lane_map: Dict[int, Dict[str, Any]] = {}

    # racelistの行っぽい table/tr を幅広く探索
    # 「展示」「ST」列があるテーブルを探してそこから抜く
    tables = soup.find_all("table")
    target_table = None
    for t in tables:
        head = _clean(t.get_text(" ", strip=True))
        if ("展示" in head or "展示タイム" in head) and ("ST" in head or "スタート" in head):
            # かつ「艇番/枠」っぽいのがありそうなら採用
            if "艇" in head or "枠" in head or "進入" in head:
                target_table = t
                break

    # 見つからない場合もあるので、tr を総当たりで lane/exhibit/st を拾う fallback
    rows = []
    if target_table:
        rows = target_table.find_all("tr")
    else:
        rows = soup.find_all("tr")

    for tr in rows:
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        texts = [_clean(td.get_text(" ", strip=True)) for td in tds]
        joined = " ".join(texts)

        # lane候補（1〜6が単独で出るパターン）
        lane = None
        for x in texts[:3]:  # 先頭付近に枠が出やすい
            if re.fullmatch(r"[1-6]", x):
                lane = int(x)
                break
        if lane is None:
            continue

        # exhibit候補: 5.xx〜7.xx くらいが多い（展示タイム）
        exhibit = None
        m = re.search(r"\b([4-9]\.[0-9]{2})\b", joined)
        if m:
            exhibit = m.group(1)

        # ST候補: 0.xx / F.xx / L.xx 等がありえる
        st = None
        m = re.search(r"\b([0-1]\.[0-9]{2})\b", joined)
        if m:
            st = m.group(1)

        if exhibit is None and st is None:
            continue

        lane_map.setdefault(lane, {})
        if exhibit is not None:
            lane_map[lane]["exhibit"] = exhibit
        if st is not None:
            lane_map[lane]["start_timing"] = st

    return pre, lane_map
