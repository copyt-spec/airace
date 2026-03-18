# engine/odds_fetcher.py
# FIXED_ORDER は絶対に触らない（組み合わせの基準）

from __future__ import annotations

import re
import time
from html import unescape
from typing import Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


FIXED_ORDER = """1-2-3 2-1-3 3-1-2 4-1-2 5-1-2 6-1-2
1-2-4 2-1-4 3-1-4 4-1-3 5-1-3 6-1-3
1-2-5 2-1-5 3-1-5 4-1-5 5-1-4 6-1-4
1-2-6 2-1-6 3-1-6 4-1-6 5-1-6 6-1-5
1-3-2 2-3-1 3-2-1 4-2-1 5-2-1 6-2-1
1-3-4 2-3-4 3-2-4 4-2-3 5-2-3 6-2-3
1-3-5 2-3-5 3-2-5 4-2-5 5-2-4 6-2-4
1-3-6 2-3-6 3-2-6 4-2-6 5-2-6 6-2-5
1-4-2 2-4-1 3-4-1 4-3-1 5-3-1 6-3-1
1-4-3 2-4-3 3-4-2 4-3-2 5-3-2 6-3-2
1-4-5 2-4-5 3-4-5 4-3-5 5-3-4 6-3-4
1-4-6 2-4-6 3-4-6 4-3-6 5-3-6 6-3-5
1-5-2 2-5-1 3-5-1 4-5-1 5-4-1 6-4-1
1-5-3 2-5-3 3-5-2 4-5-2 5-4-2 6-4-2
1-5-4 2-5-4 3-5-4 4-5-3 5-4-3 6-4-3
1-5-6 2-5-6 3-5-6 4-5-6 5-4-6 6-4-5
1-6-2 2-6-1 3-6-1 4-6-1 5-6-1 6-5-1
1-6-3 2-6-3 3-6-2 4-6-2 5-6-2 6-5-2
1-6-4 2-6-4 3-6-4 4-6-3 5-6-3 6-5-3
1-6-5 2-6-5 3-6-5 4-6-5 5-6-4 6-5-4""".split()


# =========================
# shared session / cache
# =========================
_SESSION: Optional[requests.Session] = None
_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}
_CACHE_SECONDS = 30  # オッズは短時間キャッシュ

# 高速抽出用 regex
_RE_ODDS_POINT = re.compile(
    r'<[^>]*class="[^"]*\boddsPoint\b[^"]*"[^>]*>(.*?)</[^>]+>',
    re.IGNORECASE | re.DOTALL,
)
_RE_TAGS = re.compile(r"<[^>]+>")
_RE_SPACE = re.compile(r"\s+")


def _cache_key(race_no: int, date: str, venue_code: int) -> str:
    return f"odds3t_{venue_code}_{date}_{race_no}"


def _get_cache(key: str) -> Optional[Dict[str, str]]:
    item = _CACHE.get(key)
    if not item:
        return None

    ts, data = item
    if time.time() - ts > _CACHE_SECONDS:
        _CACHE.pop(key, None)
        return None

    return data


def _set_cache(key: str, value: Dict[str, str]) -> None:
    _CACHE[key] = (time.time(), value)

    # たまに掃除
    if len(_CACHE) > 300:
        now = time.time()
        old_keys = [k for k, (ts, _) in _CACHE.items() if now - ts > _CACHE_SECONDS]
        for k in old_keys[:100]:
            _CACHE.pop(k, None)


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

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.boatrace.jp/",
            "Connection": "keep-alive",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        }
    )

    return session


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _create_session()
    return _SESSION


def _jcd2(venue_code: int) -> str:
    return str(int(venue_code)).zfill(2)


def _clean_html_text(s: str) -> str:
    s = unescape(s or "")
    s = _RE_TAGS.sub("", s)
    s = _RE_SPACE.sub(" ", s).strip()
    return s


def _extract_odds_fast(html: str) -> list[str]:
    """
    まず regex で oddsPoint を高速抽出。
    120件取れればこれで終わり。
    """
    hits = _RE_ODDS_POINT.findall(html)
    if not hits:
        return []

    values = [_clean_html_text(x) for x in hits]
    values = [x for x in values if x != ""]
    return values


def _make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _extract_odds_bs4(html: str) -> list[str]:
    """
    regex で足りない時の保険
    """
    soup = _make_soup(html)
    cells = soup.select(".oddsPoint")
    return [cell.get_text(strip=True) for cell in cells]


def fetch_odds(race_no, date, venue_code: int = 15) -> Dict[str, str]:
    """
    三連単120通りオッズを取得
    return:
      {
        "1-2-3": "5.4",
        ...
      }
    """
    race_no_i = int(race_no)
    date_s = str(date)
    venue_code_i = int(venue_code)

    key = _cache_key(race_no_i, date_s, venue_code_i)
    cached = _get_cache(key)
    if cached is not None:
        return cached

    jcd = _jcd2(venue_code_i)
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?jcd={jcd}&hd={date_s}&rno={race_no_i}"

    session = _get_session()
    response = session.get(url, timeout=(4, 10))
    response.raise_for_status()

    html = response.text

    # =========================
    # fast path: regex
    # =========================
    odds_values = _extract_odds_fast(html)

    # 足りない時だけ bs4 fallback
    if len(odds_values) < len(FIXED_ORDER):
        odds_values = _extract_odds_bs4(html)

    odds_dict: Dict[str, str] = {}

    # 既存仕様維持：FIXED_ORDER と zip で対応
    for combo, odd in zip(FIXED_ORDER, odds_values):
        odds_dict[combo] = odd

    # 足りない時もキーは揃える
    if len(odds_dict) < len(FIXED_ORDER):
        for combo in FIXED_ORDER:
            odds_dict.setdefault(combo, "")

    _set_cache(key, odds_dict)
    return odds_dict
