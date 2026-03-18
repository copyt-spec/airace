from __future__ import annotations

import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

JCD_KOJIMA = 16

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    )
}

RE_RACERNO_ANY = re.compile(r"(\d{4})")
RE_GRADE = re.compile(r"\b([A-Z]\d|B\d)\b")
RE_RACER_GRADE = re.compile(r"(\d{4})\s*/\s*([A-Z]\d|B\d)")

ZEN2HAN = str.maketrans({
    "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6"
})

# =========================
# cache
# =========================
_RACE_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_ALL_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

_CACHE_SECONDS_RACE = 60
_CACHE_SECONDS_ALL = 60

_SESSION: Optional[requests.Session] = None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _normalize_digits(s: str) -> str:
    return (s or "").translate(ZEN2HAN)


def _format_fl(text: str) -> str:
    parts = []

    mf = re.search(r"\bF(\d+)\b", text)
    if mf:
        parts.append(f"F{mf.group(1)}")

    ml = re.search(r"\bL(\d+)\b", text)
    if ml:
        n = int(ml.group(1))
        if n >= 1:
            parts.append(f"L{n}")

    return " ".join(parts) if parts else "-"


def _extract_lane_from_text(text: str) -> Optional[int]:
    t = _normalize_digits(_clean(text))

    m = re.search(r"(^| )([1-6])( |$)", t)
    if m:
        return int(m.group(2))

    if t and t[0] in "123456":
        return int(t[0])

    return None


def _pick_racer_no(text: str) -> Optional[str]:
    m = RE_RACERNO_ANY.search(text)
    return m.group(1) if m else None


def _parse_entry_line(text: str) -> Optional[Dict[str, Any]]:
    t = _normalize_digits(_clean(text))

    lane = _extract_lane_from_text(t)
    if lane is None:
        return None

    racer_no = _pick_racer_no(t)
    if not racer_no:
        return None

    grade = "-"
    mg = RE_RACER_GRADE.search(t)
    if mg:
        grade = mg.group(2)
    else:
        mg2 = RE_GRADE.search(t)
        if mg2:
            grade = mg2.group(1)

    name = "-"
    mname = re.search(r"([ぁ-んァ-ン一-龥]{2,12})\s+([ぁ-んァ-ン一-龥]{1,12})", t)
    if mname:
        name = _clean(f"{mname.group(1)} {mname.group(2)}")

    branch = "-"
    mbr = re.search(r"([^\s/]{1,6}/[^\s/]{1,6})", t)
    if mbr:
        branch = mbr.group(1)

    name_branch = "-"
    if name != "-" or branch != "-":
        name_branch = _clean(f"{'' if name == '-' else name} {'' if branch == '-' else branch}")

    fl = _format_fl(t)

    nums = [float(x) for x in re.findall(r"\d+\.\d+", t)]
    win_rate = "-"
    quinella = "-"

    for v in nums:
        if 2.5 <= v <= 10.5:
            win_rate = f"{v:.2f}"
            break

    for v in nums:
        if 10.0 <= v <= 100.0:
            quinella = f"{v:.2f}"
            break

    if quinella == "-":
        for v in nums:
            if 0.30 <= v <= 100.0:
                if win_rate != "-" and abs(v - float(win_rate)) < 1e-9:
                    continue
                quinella = f"{v:.2f}"
                break

    return {
        "lane": lane,
        "racer_no": racer_no,
        "name_branch": name_branch,
        "grade": grade,
        "fl": fl,
        "win_rate": win_rate,
        "quinella_rate": quinella,
        "motor": None,
        "boat": None,
        "exhibit": None,
        "start_timing": None,
    }


def _table_score(table: Tag) -> Tuple[int, int, int]:
    uniq = set()
    lane_hits = 0
    racer_hits = 0

    for tr in table.find_all("tr"):
        txt = _normalize_digits(_clean(tr.get_text(" ")))
        lane = _extract_lane_from_text(txt)
        if lane is not None and 1 <= lane <= 6:
            uniq.add(lane)
            lane_hits += 1
        if RE_RACERNO_ANY.search(txt):
            racer_hits += 1

    return len(uniq), lane_hits, racer_hits


def _select_best_table(soup: BeautifulSoup) -> Optional[Tag]:
    # まず table#2 を最優先
    t2 = soup.select_one("table#table2")
    if t2 is not None:
        return t2

    # 次に class/id に 2 を含むものを軽く探す
    for sel in ("table.is-w495", "table"):
        tables = soup.select(sel)
        if not tables:
            continue
        if len(tables) >= 2:
            return tables[1]

    # 最後の保険
    tables = soup.find_all("table")
    if not tables:
        return None

    best = None
    best_key = (-1, -1, -1)
    for t in tables:
        key = _table_score(t)
        if key > best_key:
            best_key = key
            best = t

    return best


def _get_cache(
    cache: Dict[str, Tuple[float, List[Dict[str, Any]]]],
    key: str,
    ttl: int,
) -> Optional[List[Dict[str, Any]]]:
    item = cache.get(key)
    if not item:
        return None

    ts, data = item
    if time.time() - ts > ttl:
        cache.pop(key, None)
        return None

    return data


def _set_cache(
    cache: Dict[str, Tuple[float, List[Dict[str, Any]]]],
    key: str,
    value: List[Dict[str, Any]],
) -> None:
    cache[key] = (time.time(), value)


def _create_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=2,
        read=2,
        connect=2,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=20,
        pool_maxsize=20,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(UA)
    return session


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _create_session()
    return _SESSION


def _make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def fetch_kojima_racelist(race_no: int, date: str) -> List[Dict[str, Any]]:
    cache_key = f"kojima_racelist_{date}_{race_no}"
    cached = _get_cache(_RACE_CACHE, cache_key, _CACHE_SECONDS_RACE)
    if cached is not None:
        return cached

    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?hd={date}&jcd={JCD_KOJIMA:02d}&rno={race_no}"

    session = _get_session()
    r = session.get(url, timeout=(5, 20))
    r.raise_for_status()

    soup = _make_soup(r.text)
    table = _select_best_table(soup)
    if not table:
        return []

    tbodies = table.find_all("tbody", recursive=False)
    best_by_lane: Dict[int, Dict[str, Any]] = {}

    if tbodies:
        for tbody in tbodies:
            first_tr = tbody.find("tr")
            if first_tr is None:
                continue

            e = _parse_entry_line(first_tr.get_text(" "))
            if not e:
                continue

            lane = int(e["lane"])
            if 1 <= lane <= 6 and lane not in best_by_lane:
                best_by_lane[lane] = e

            if len(best_by_lane) == 6:
                break
    else:
        for tr in table.find_all("tr"):
            e = _parse_entry_line(tr.get_text(" "))
            if not e:
                continue

            lane = int(e["lane"])
            if lane not in best_by_lane:
                best_by_lane[lane] = e

            if len(best_by_lane) == 6:
                break

    entries: List[Dict[str, Any]] = []
    for lane in range(1, 7):
        if lane in best_by_lane:
            row = best_by_lane[lane]
            row["race_no"] = race_no
            entries.append(row)

    entries.sort(key=lambda x: x["lane"])

    _set_cache(_RACE_CACHE, cache_key, entries)
    return entries


def fetch_all_kojima_entries_once(date: str, sleep_sec: float = 0.0, max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    児島12R分をまとめて取得
    - 12R全体もキャッシュ
    - 1Rごとの結果もキャッシュ
    - 並列取得で高速化
    """
    cache_key = f"kojima_all_entries_{date}"
    cached = _get_cache(_ALL_CACHE, cache_key, _CACHE_SECONDS_ALL)
    if cached is not None:
        return cached

    all_rows_by_race: Dict[int, List[Dict[str, Any]]] = {}
    race_nos = list(range(1, 13))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {
            ex.submit(fetch_kojima_racelist, rno, date): rno
            for rno in race_nos
        }

        for fut in as_completed(fut_map):
            rno = fut_map[fut]
            try:
                rows = fut.result()
            except Exception:
                rows = []
            all_rows_by_race[rno] = rows

            if sleep_sec > 0:
                time.sleep(sleep_sec)

    all_entries: List[Dict[str, Any]] = []
    for rno in race_nos:
        all_entries.extend(all_rows_by_race.get(rno, []))

    _set_cache(_ALL_CACHE, cache_key, all_entries)
    return all_entries
