from __future__ import annotations

import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from engine.venue_registry import get_jcd, normalize_venue_name
from engine.shared_page_cache import get_page as _get_shared_page


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    )
}

RE_RACERNO_ANY = re.compile(r"(\d{4})")
RE_GRADE = re.compile(r"\b([A-Z]\d|B\d)\b")
RE_RACER_GRADE = re.compile(r"(\d{4})\s*/\s*([A-Z]\d|B\d)")
RE_FLOAT = re.compile(r"\d+\.\d+")

ZEN2HAN = str.maketrans({
    "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6"
})

_RACE_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_ALL_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

_CACHE_SECONDS_RACE = 60
_CACHE_SECONDS_ALL = 60

_SESSION: Optional[requests.Session] = None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _normalize_digits(s: str) -> str:
    return (s or "").translate(ZEN2HAN)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


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
    t2 = soup.select_one("table#table2")
    if t2 is not None:
        return t2

    for sel in ("table.is-w495", "table"):
        tables = soup.select(sel)
        if not tables:
            continue
        if len(tables) >= 2:
            return tables[1]

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


def _extract_texts_from_tbody(tbody: Tag) -> List[str]:
    texts: List[str] = []
    for tr in tbody.find_all("tr", recursive=False):
        txt = _normalize_digits(_clean(tr.get_text(" ")))
        if txt:
            texts.append(txt)
    return texts


def _pick_name_branch_from_text(text: str) -> Tuple[str, str]:
    t = _normalize_digits(_clean(text))

    name = ""
    branch = ""

    mbr = re.search(r"([^\s/]{1,8}/[^\s/]{1,8})", t)
    if mbr:
        branch = mbr.group(1)

    name_matches = re.findall(r"[ぁ-んァ-ン一-龥]{2,12}", t)
    if name_matches:
        if len(name_matches) >= 2:
            name = f"{name_matches[0]} {name_matches[1]}"
        else:
            name = name_matches[0]

    return name.strip(), branch.strip()


def _pick_win_and_quinella(text: str) -> Tuple[str, str]:
    nums = [_safe_float(x, -1.0) for x in RE_FLOAT.findall(text)]

    win_rate = "-"
    quinella = "-"

    for v in nums:
        if 2.5 <= v <= 10.5:
            win_rate = f"{v:.2f}"
            break

    for v in nums:
        if 10.0 <= v <= 100.0:
            if win_rate != "-" and abs(v - float(win_rate)) < 1e-9:
                continue
            quinella = f"{v:.2f}"
            break

    if quinella == "-":
        for v in nums:
            if 0.3 <= v <= 100.0:
                if win_rate != "-" and abs(v - float(win_rate)) < 1e-9:
                    continue
                quinella = f"{v:.2f}"
                break

    return win_rate, quinella


def _parse_entry_from_tbody(tbody: Tag, race_no: int) -> Optional[Dict[str, Any]]:
    texts = _extract_texts_from_tbody(tbody)
    if not texts:
        return None

    whole = " ".join(texts)
    lane = None

    for txt in texts:
        lane = _extract_lane_from_text(txt)
        if lane is not None:
            break

    if lane is None:
        lane = _extract_lane_from_text(whole)

    if lane is None or lane not in range(1, 7):
        return None

    racer_no = _pick_racer_no(whole)
    if not racer_no:
        return None

    grade = "-"
    mg = RE_RACER_GRADE.search(whole)
    if mg:
        grade = mg.group(2)
    else:
        mg2 = RE_GRADE.search(whole)
        if mg2:
            grade = mg2.group(1)

    name, branch = _pick_name_branch_from_text(whole)
    name_branch = _clean(f"{name} {branch}") if (name or branch) else "-"

    fl = _format_fl(whole)
    win_rate, quinella_rate = _pick_win_and_quinella(whole)

    motor_no = ""
    boat_no = ""
    motor_rate = ""
    boat_rate = ""

    cells = []
    for tr in tbody.find_all("tr", recursive=False):
        for td in tr.find_all(["td", "th"], recursive=False):
            txt = _normalize_digits(_clean(td.get_text(" ")))
            if txt:
                cells.append(txt)

    ints = []
    for c in cells:
        if re.fullmatch(r"\d{1,3}", c):
            ints.append(c)

    if len(ints) >= 2:
        motor_no = ints[0]
        boat_no = ints[1]

    float_cells = [c for c in cells if re.fullmatch(r"\d+\.\d+", c)]
    if len(float_cells) >= 2:
        motor_rate = float_cells[-2]
        boat_rate = float_cells[-1]

    return {
        "race_no": int(race_no),
        "lane": int(lane),
        "racer_no": str(racer_no),
        "name": name or "",
        "branch": branch or "",
        "name_branch": name_branch,
        "grade": grade,
        "fl": fl,
        "win_rate": win_rate,
        "quinella_rate": quinella_rate,
        "motor_no": motor_no,
        "boat_no": boat_no,
        "motor_rate": motor_rate,
        "boat_rate": boat_rate,
        "motor": motor_no,
        "boat": boat_no,
        "exhibit": None,
        "start_timing": None,
    }


def fetch_racelist_by_venue(venue_name: str, race_no: int, date: str) -> List[Dict[str, Any]]:
    venue_name = normalize_venue_name(venue_name)
    venue_code = get_jcd(venue_name)

    cache_key = f"racelist_{venue_name}_{date}_{race_no}"
    cached = _get_cache(_RACE_CACHE, cache_key, _CACHE_SECONDS_RACE)
    if cached is not None:
        return cached

    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?hd={date}&jcd={venue_code:02d}&rno={race_no}"

    session = _get_session()
    # engine.racelist_enricherが同じURLを参照する際にも使う共有キャッシュ経由で取得する
    # (同一ページの二重フェッチを避けるため。2026-07-16追加)
    html = _get_shared_page(url, session, timeout=(5, 20), cache_seconds=_CACHE_SECONDS_RACE)

    soup = _make_soup(html)
    table = _select_best_table(soup)
    if not table:
        return []

    best_by_lane: Dict[int, Dict[str, Any]] = {}
    tbodies = table.find_all("tbody", recursive=False)

    if tbodies:
        for tbody in tbodies:
            e = _parse_entry_from_tbody(tbody, race_no=int(race_no))
            if not e:
                continue

            lane = int(e["lane"])
            if 1 <= lane <= 6 and lane not in best_by_lane:
                best_by_lane[lane] = e

            if len(best_by_lane) == 6:
                break
    else:
        trs = table.find_all("tr", recursive=False)
        current_group: List[Tag] = []

        for tr in trs:
            current_group.append(tr)
            text = _normalize_digits(_clean(tr.get_text(" ")))
            lane = _extract_lane_from_text(text)
            if lane in range(1, 7):
                fake_tbody = soup.new_tag("tbody")
                for gtr in current_group:
                    fake_tbody.append(gtr)
                e = _parse_entry_from_tbody(fake_tbody, race_no=int(race_no))
                current_group = []

                if not e:
                    continue

                lane2 = int(e["lane"])
                if lane2 not in best_by_lane:
                    best_by_lane[lane2] = e

                if len(best_by_lane) == 6:
                    break

    entries: List[Dict[str, Any]] = []
    for lane in range(1, 7):
        if lane in best_by_lane:
            entries.append(best_by_lane[lane])

    entries.sort(key=lambda x: int(x["lane"]))

    _set_cache(_RACE_CACHE, cache_key, entries)
    return entries


def fetch_all_entries_once_by_venue(
    venue_name: str,
    date: str,
    sleep_sec: float = 0.0,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    venue_name = normalize_venue_name(venue_name)

    cache_key = f"all_entries_{venue_name}_{date}"
    cached = _get_cache(_ALL_CACHE, cache_key, _CACHE_SECONDS_ALL)
    if cached is not None:
        return cached

    all_rows_by_race: Dict[int, List[Dict[str, Any]]] = {}
    race_nos = list(range(1, 13))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {
            ex.submit(fetch_racelist_by_venue, venue_name, rno, date): rno
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
