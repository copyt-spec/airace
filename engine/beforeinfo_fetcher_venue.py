# engine/beforeinfo_fetcher_venue.py
# beforeinfo_fetcher.py（丸亀固定=15）を壊さず、jcd可変版を別ファイルで用意

from __future__ import annotations

import time
from typing import Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# shared session / cache
# =========================
_SESSION: Optional[requests.Session] = None
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_SECONDS = 30  # 短時間キャッシュ


def _cache_key(race_no: int, date: str, venue_code: int) -> str:
    return f"beforeinfo_{venue_code}_{date}_{race_no}"


def _get_cache(key: str) -> Optional[Dict[str, Any]]:
    item = _CACHE.get(key)
    if not item:
        return None

    ts, data = item
    if time.time() - ts > _CACHE_SECONDS:
        _CACHE.pop(key, None)
        return None

    return data


def _set_cache(key: str, value: Dict[str, Any]) -> None:
    _CACHE[key] = (time.time(), value)


def create_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=2,
        read=2,
        connect=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.boatrace.jp/",
            "Connection": "keep-alive",
        }
    )

    return session


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = create_session()
    return _SESSION


def _jcd_str(venue_code: int) -> str:
    return str(int(venue_code)).zfill(2)


def _failed_payload() -> Dict[str, Any]:
    return {
        i: {
            "exhibit_time": "取得失敗",
            "tilt": "取得失敗",
            "parts": "",
            "st": "取得失敗",
            "course": "取得失敗",
        }
        for i in range(1, 7)
    }


def fetch_beforeinfo_venue(race_no: int, date: str, venue_code: int) -> Dict[str, Any]:
    """
    可変会場版 beforeinfo 取得
    戻り値:
      {
        1: {"exhibit_time": "...", "tilt": "...", "parts": "...", "st": "...", "course": "..."},
        ...
        "weather": "...",
        "wind_speed": "...",
        "wind_direction": "..."
      }
    """
    key = _cache_key(race_no, date, venue_code)
    cached = _get_cache(key)
    if cached is not None:
        return cached

    jcd = _jcd_str(venue_code)
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={jcd}&hd={date}"

    session = _get_session()
    data: Dict[str, Any] = {}

    try:
        res = session.get(url, timeout=(5, 15))
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("⚠ beforeinfo通信エラー:", e)
        failed = _failed_payload()
        _set_cache(key, failed)
        return failed

    soup = BeautifulSoup(res.text, "html.parser")

    # ==========================
    # 展示
    # ==========================
    tables = soup.find_all("table")
    for table in tables:
        if "展示" not in table.get_text():
            continue

        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 8:
                continue

            try:
                lane = int(cols[0].get_text(strip=True))
            except Exception:
                continue

            data[lane] = {
                "exhibit_time": cols[4].get_text(strip=True),
                "tilt": cols[5].get_text(strip=True),
                "parts": cols[7].get_text(strip=True),
                "st": "未取得",
                "course": "未取得",
            }
        break

    # ==========================
    # ST / 進入
    # ==========================
    lane_spans = soup.select("span[class*='boatImage1Number']")
    st_spans = soup.select("span[class*='boatImage1Time']")

    if len(lane_spans) == 6 and len(st_spans) == 6:
        for i in range(6):
            course = lane_spans[i].get_text(strip=True)
            st_raw = st_spans[i].get_text(strip=True)
            st = "0" + st_raw if st_raw.startswith(".") else st_raw

            try:
                lane = int(course)
            except Exception:
                continue

            if lane in data:
                data[lane]["st"] = st
                data[lane]["course"] = course
            else:
                # 展示表の抽出に漏れた時の保険
                data[lane] = {
                    "exhibit_time": "",
                    "tilt": "",
                    "parts": "",
                    "st": st,
                    "course": course,
                }

    # ==========================
    # 気象
    # ==========================
    # 天候
    weather_block = soup.select_one(".weather1_bodyUnit")
    if weather_block:
        data["weather"] = weather_block.get_text(strip=True)

    # 風速
    wind_speed_block = soup.select_one(".weather1_bodyUnit.is-wind")
    if wind_speed_block:
        speed = wind_speed_block.select_one(".weather1_bodyUnitLabelData")
        if speed:
            data["wind_speed"] = speed.get_text(strip=True)

    # 風向き
    wind_block = soup.select_one(".weather1_bodyUnit.is-windDirection")
    if wind_block:
        icon = wind_block.select_one("p.weather1_bodyUnitImage")
        if icon:
            classes = icon.get("class", [])
            for c in classes:
                if c.startswith("is-wind"):
                    num = c.replace("is-wind", "")
                    wind_map = {
                        "1": "北",
                        "2": "北東",
                        "3": "東",
                        "4": "南東",
                        "5": "南",
                        "6": "南西",
                        "7": "西",
                        "8": "北西",
                    }
                    data["wind_direction"] = wind_map.get(num, "不明")
                    break

    # ==========================
    # 足りない艇番は保険で埋める
    # ==========================
    for i in range(1, 7):
        if i not in data:
            data[i] = {
                "exhibit_time": "",
                "tilt": "",
                "parts": "",
                "st": "未取得",
                "course": "未取得",
            }

    _set_cache(key, data)
    return data
