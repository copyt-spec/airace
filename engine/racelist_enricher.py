from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    )
}

ZEN2HAN = str.maketrans({
    "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6",
    "７": "7", "８": "8", "９": "9", "０": "0",
})

_SESSION: Optional[requests.Session] = None
_CACHE: Dict[str, Tuple[float, Dict[int, Dict[str, Any]]]] = {}
_CACHE_SECONDS = 60

# lane の先頭判定
_LANE_RE = re.compile(r"^\s*([1-6])\s")

# 公式出走表の stats 部分
# avg_st, 全国勝率, 全国2連, 全国3連, 当地勝率, 当地2連, 当地3連,
# motor_no, motor_2, motor_3, boat_no, boat_2, boat_3
_STATS_RE = re.compile(
    r"(?P<avg_st>(?:\d+\.\d+|-))\s+"
    r"(?P<nat_win>(?:\d+\.\d+|-))\s+"
    r"(?P<nat_2>(?:\d+\.\d+|-))\s+"
    r"(?P<nat_3>(?:\d+\.\d+|-))\s+"
    r"(?P<loc_win>(?:\d+\.\d+|-))\s+"
    r"(?P<loc_2>(?:\d+\.\d+|-))\s+"
    r"(?P<loc_3>(?:\d+\.\d+|-))\s+"
    r"(?P<motor_no>(?:\d{1,3}|-))\s+"
    r"(?P<motor_2>(?:\d+\.\d+|-))\s+"
    r"(?P<motor_3>(?:\d+\.\d+|-))\s+"
    r"(?P<boat_no>(?:\d{1,3}|-))\s+"
    r"(?P<boat_2>(?:\d+\.\d+|-))\s+"
    r"(?P<boat_3>(?:\d+\.\d+|-))"
)


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    session = requests.Session()
    retry = Retry(
        total=2,
        read=2,
        connect=2,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(UA)
    _SESSION = session
    return session


def _clean_text(s: str) -> str:
    s = (s or "").translate(ZEN2HAN)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = str(v).strip()
        if s in ("", "-", "－", "None", "none"):
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        s = str(v).strip()
        if s in ("", "-", "－", "None", "none"):
            return default
        return int(float(s))
    except Exception:
        return default


def _empty(v: Any) -> bool:
    return v in (None, "", "-", "－", "None", "none", 0, 0.0, "0", "0.0")


def _select_best_table(soup: BeautifulSoup) -> Optional[Tag]:
    t2 = soup.select_one("table#table2")
    if t2 is not None:
        return t2

    tables = soup.find_all("table")
    if not tables:
        return None

    best = None
    best_score = -1
    for t in tables:
        txt = _clean_text(t.get_text(" "))
        score = 0
        if "モーター" in txt:
            score += 3
        if "ボート" in txt:
            score += 3
        if "2連率" in txt:
            score += 2
        if "3連率" in txt:
            score += 1
        if score > best_score:
            best_score = score
            best = t
    return best


def _extract_lane_stats_from_tbody(tbody: Tag) -> Optional[Dict[str, Any]]:
    text = _clean_text(tbody.get_text(" "))
    if not text:
        return None

    m_lane = _LANE_RE.search(text)
    if not m_lane:
        return None
    lane = int(m_lane.group(1))

    m_stats = _STATS_RE.search(text)
    if not m_stats:
        return None

    g = m_stats.groupdict()

    return {
        "lane": lane,
        "avg_st": _safe_float(g["avg_st"], 0.0),
        "motor_no": _safe_int(g["motor_no"], 0),
        "motor_2rate": _safe_float(g["motor_2"], 0.0),
        "motor_3rate": _safe_float(g["motor_3"], 0.0),
        "boat_no": _safe_int(g["boat_no"], 0),
        "boat_2rate": _safe_float(g["boat_2"], 0.0),
        "boat_3rate": _safe_float(g["boat_3"], 0.0),
    }


def _fetch_lane_stats(date: str, race_no: int, venue_code: int) -> Dict[int, Dict[str, Any]]:
    cache_key = f"{date}_{venue_code}_{race_no}"
    item = _CACHE.get(cache_key)
    if item:
        ts, data = item
        if time.time() - ts <= _CACHE_SECONDS:
            return data

    url = (
        f"https://www.boatrace.jp/owpc/pc/race/racelist"
        f"?hd={date}&jcd={int(venue_code):02d}&rno={int(race_no)}"
    )

    session = _get_session()
    r = session.get(url, timeout=(5, 20))
    r.raise_for_status()

    try:
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        soup = BeautifulSoup(r.text, "html.parser")

    table = _select_best_table(soup)
    if table is None:
        return {}

    lane_map: Dict[int, Dict[str, Any]] = {}

    tbodies = table.find_all("tbody", recursive=False)
    if not tbodies:
        tbodies = table.find_all("tbody")

    for tbody in tbodies:
        row = _extract_lane_stats_from_tbody(tbody)
        if not row:
            continue
        lane = int(row["lane"])
        if lane in range(1, 7):
            lane_map[lane] = row

    _CACHE[cache_key] = (time.time(), lane_map)
    return lane_map


def enrich_entries_with_racelist(
    entries: List[Dict[str, Any]],
    *,
    date: str,
    race_no: int,
    venue_code: int,
) -> List[Dict[str, Any]]:
    if not entries:
        return []

    lane_stats = _fetch_lane_stats(
        date=date,
        race_no=race_no,
        venue_code=venue_code,
    )

    out: List[Dict[str, Any]] = []

    for e in entries:
        row = dict(e)
        lane = _safe_int(row.get("lane", 0), 0)
        stats = lane_stats.get(lane, {})

        if stats:
            # ===== 番号 =====
            if _empty(row.get("motor")) and not _empty(stats.get("motor_no")):
                row["motor"] = stats["motor_no"]
            row["motor_no"] = row.get("motor_no") or stats.get("motor_no")

            if _empty(row.get("boat")) and not _empty(stats.get("boat_no")):
                row["boat"] = stats["boat_no"]
            row["boat_no"] = row.get("boat_no") or stats.get("boat_no")

            # ===== 2連率 =====
            row["motor_rate"] = stats.get("motor_2rate", row.get("motor_rate", 0.0))
            row["motor_2rate"] = stats.get("motor_2rate", row.get("motor_2rate", 0.0))
            row["motor_quinella_rate"] = stats.get(
                "motor_2rate",
                row.get("motor_quinella_rate", 0.0),
            )

            row["boat_rate"] = stats.get("boat_2rate", row.get("boat_rate", 0.0))
            row["boat_2rate"] = stats.get("boat_2rate", row.get("boat_2rate", 0.0))
            row["boat_quinella_rate"] = stats.get(
                "boat_2rate",
                row.get("boat_quinella_rate", 0.0),
            )

            # ===== 3連率もついでに保持 =====
            row["motor_3rate"] = stats.get("motor_3rate", row.get("motor_3rate", 0.0))
            row["boat_3rate"] = stats.get("boat_3rate", row.get("boat_3rate", 0.0))

            # ===== avg_st が空なら補完 =====
            if _empty(row.get("avg_st")) and not _empty(stats.get("avg_st")):
                row["avg_st"] = stats["avg_st"]

        out.append(row)

    return out
