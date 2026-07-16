# engine/beforeinfo_fetcher_venue.py
# beforeinfo_fetcher.py（丸亀固定=15）を壊さず、jcd可変版を別ファイルで用意

from __future__ import annotations

import re
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
        "wind_speed": "...",       # 生のラベル文字列(例: "1m")、後方互換用
        "wind_speed_mps": 1.0,     # 2026-07-16追加: floatとして使えるように抽出済み
        "wind_direction": "...",   # 後方互換用
        "wind_dir": "...",        # 2026-07-16追加: app/controller.pyが参照するキー名
        "wave_cm": 2.0,            # 2026-07-16追加: 波高(取得できた場合のみ存在)
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
    # 2026-07-16修正: 生のラベル文字列(例: "1m")は_safe_float()でfloat変換に失敗して
    # 常に0.0にフォールバックしていたため、ここで数値部分だけ抽出してfloatとして
    # "wind_speed_mps" にも入れておく（"wind_speed"は後方互換のため文字列のまま残す）。
    wind_speed_block = soup.select_one(".weather1_bodyUnit.is-wind")
    if wind_speed_block:
        speed = wind_speed_block.select_one(".weather1_bodyUnitLabelData")
        if speed:
            speed_text = speed.get_text(strip=True)
            data["wind_speed"] = speed_text
            m_speed = re.search(r"(\d+(?:\.\d+)?)", speed_text)
            if m_speed:
                data["wind_speed_mps"] = float(m_speed.group(1))

    # 風向き
    # 2026-07-16修正: app/controller.py の _extract_weather_set() は
    # ["wind_dir", "風向"] というキーしか見ておらず、ここで返していた
    # "wind_direction" は拾われず常に空文字になっていた。後方互換のため
    # "wind_direction" は残しつつ、"wind_dir" にも同じ値を入れる。
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
                    wind_dir_val = wind_map.get(num, "不明")
                    data["wind_direction"] = wind_dir_val
                    data["wind_dir"] = wind_dir_val
                    break

    # 波高
    # 2026-07-16修正: このファイルには波高(wave_cm)の抽出が一切存在せず、
    # app/controller.py 側では常にデフォルトの0.0になっていた。
    # 風速・風向きと同じ".weather1_bodyUnit"系クラス命名規則(is-wave)を
    # 第一候補として試し、見つからない場合はページ全文から
    # "波高 X cm" のテキストパターンを正規表現で拾うフォールバックを行う
    # （engine/preinfo_fetcher.py で実績のあるパターンと同じ方式）。
    wave_block = soup.select_one(".weather1_bodyUnit.is-wave")
    if wave_block:
        wave_label = wave_block.select_one(".weather1_bodyUnitLabelData")
        wave_text = wave_label.get_text(strip=True) if wave_label else wave_block.get_text(strip=True)
        m_wave = re.search(r"(\d+(?:\.\d+)?)", wave_text)
        if m_wave:
            data["wave_cm"] = float(m_wave.group(1))

    if "wave_cm" not in data:
        m_wave_fallback = re.search(r"波高[:：]?\s*([0-9.]+)\s*cm", res.text)
        if m_wave_fallback:
            data["wave_cm"] = float(m_wave_fallback.group(1))

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
