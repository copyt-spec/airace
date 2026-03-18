# engine/beforeinfo_fetcher_toda.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_FW_TO_ASCII = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "－": "-", "．": ".",
})

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

TIMEOUT = (4, 10)  # connect, read


class BeforeInfoRow:
    def __init__(
        self,
        exhibit_time: str = "未取得",
        st: str = "未取得",
        course: str = "未取得",
        tilt: str = "未取得",
        parts: str = "未取得",
    ):
        self.exhibit_time = exhibit_time
        self.st = st
        self.course = course
        self.tilt = tilt
        self.parts = parts


_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session

    s = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Referer": "https://www.boatrace.jp/",
            "Connection": "keep-alive",
        }
    )
    _session = s
    return s


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").translate(_FW_TO_ASCII)).strip()


def _make_default_result() -> Dict[Any, Any]:
    return {
        i: BeforeInfoRow(
            exhibit_time="未取得",
            st="未取得",
            course="未取得",
            tilt="未取得",
            parts="未取得",
        )
        for i in range(1, 7)
    }


def _safe_parser() -> str:
    # lxml が入っていれば少し速い。無ければ html.parser にフォールバック
    try:
        import lxml  # noqa: F401
        return "lxml"
    except Exception:
        return "html.parser"


def _fill_from_dom(soup: BeautifulSoup, result: Dict[Any, Any]) -> None:
    """
    beforeinfo の構造は会場共通が多いので、まず DOM から直接拾う。
    取れない時だけ後段のテキストフォールバックに流す。
    """

    # ==========================
    # 展示/チルト/部品
    # ==========================
    tables = soup.find_all("table")
    for table in tables:
        table_text = _clean_text(table.get_text(" ", strip=True))
        if "展示" not in table_text:
            continue

        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 8:
                continue

            lane_txt = _clean_text(cols[0].get_text(" ", strip=True))
            if not lane_txt.isdigit():
                continue

            lane = int(lane_txt)
            if lane not in result:
                continue

            exhibit = _clean_text(cols[4].get_text(" ", strip=True)) or "未取得"
            tilt = _clean_text(cols[5].get_text(" ", strip=True)) or "未取得"
            parts = _clean_text(cols[7].get_text(" ", strip=True)) or "未取得"

            result[lane].exhibit_time = exhibit
            result[lane].tilt = tilt
            result[lane].parts = parts
        break

    # ==========================
    # ST / 進入
    # ==========================
    lane_spans = soup.select("span[class*='boatImage1Number']")
    st_spans = soup.select("span[class*='boatImage1Time']")

    if len(lane_spans) == 6 and len(st_spans) == 6:
        for i in range(6):
            course_txt = _clean_text(lane_spans[i].get_text(" ", strip=True))
            st_txt = _clean_text(st_spans[i].get_text(" ", strip=True))

            if st_txt.startswith("."):
                st_txt = "0" + st_txt

            if course_txt.isdigit():
                lane = int(course_txt)
                if lane in result:
                    result[lane].course = course_txt
                    result[lane].st = st_txt or "未取得"

    # ==========================
    # 気温 / 風速 / 風向き
    # ==========================
    temp_block = soup.select_one(".weather1_bodyUnit")
    if temp_block:
        txt = _clean_text(temp_block.get_text(" ", strip=True))
        if txt:
            result["weather"] = txt

    wind_speed_block = soup.select_one(".weather1_bodyUnit.is-wind")
    if wind_speed_block:
        speed = wind_speed_block.select_one(".weather1_bodyUnitLabelData")
        if speed:
            result["wind_speed"] = _clean_text(speed.get_text(" ", strip=True))

    wind_block = soup.select_one(".weather1_bodyUnit.is-windDirection")
    if wind_block:
        icon = wind_block.select_one("p.weather1_bodyUnitImage")
        if icon:
            classes = icon.get("class", [])
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
            for c in classes:
                if c.startswith("is-wind"):
                    num = c.replace("is-wind", "")
                    result["wind_direction"] = wind_map.get(num, "不明")
                    break


def _needs_fallback(result: Dict[Any, Any]) -> bool:
    # 1艇でも展示が取れてないならフォールバック対象
    for lane in range(1, 7):
        row = result[lane]
        if row.exhibit_time == "未取得":
            return True
    return False


def _fill_from_text_fallback(soup: BeautifulSoup, result: Dict[Any, Any]) -> None:
    """
    DOMが崩れている日用の保険。
    ただし全走査は1回だけにする。
    """
    text = soup.get_text("\n", strip=True).translate(_FW_TO_ASCII)
    lines = [_clean_text(x) for x in text.split("\n") if _clean_text(x)]

    lane_line_pos: Dict[int, int] = {}
    lane_pat = re.compile(r"^(?:([1-6])\b|([1-6])\s+)")

    for idx, line in enumerate(lines):
        m = lane_pat.match(line)
        if not m:
            continue
        lane = int(m.group(1) or m.group(2))
        if lane not in lane_line_pos:
            lane_line_pos[lane] = idx

    part_pat = re.compile(r"(リング|キャブ|ギヤ|ピストン|シリンダ|電気|プロペラ|×)")

    for lane in range(1, 7):
        if lane not in lane_line_pos:
            continue

        i = lane_line_pos[lane]
        row = result[lane]

        # 同じ行から展示/チルト抽出
        base = lines[i]
        if row.exhibit_time == "未取得" or row.tilt == "未取得":
            m = re.search(r"\b(\d+\.\d{2})\b\s+(-?\d+(?:\.\d+)?)\b", base)
            if m:
                if row.exhibit_time == "未取得":
                    row.exhibit_time = m.group(1)
                if row.tilt == "未取得":
                    row.tilt = m.group(2)

        # 直後数行から部品抽出
        if row.parts == "未取得":
            for j in range(i + 1, min(i + 8, len(lines))):
                if lines[j] in {"1", "2", "3", "4", "5", "6"}:
                    break
                if part_pat.search(lines[j]):
                    row.parts = lines[j]
                    break

    # 気象情報が未取得なら拾う
    if "weather" not in result:
        temp = None
        weather_word = None
        for i, line in enumerate(lines):
            if line.startswith("気温"):
                temp = line.replace("気温", "").strip()
                if i + 1 < len(lines) and len(lines[i + 1]) <= 4:
                    weather_word = lines[i + 1].strip()
                break
        if temp and weather_word:
            result["weather"] = f"{temp} {weather_word}"
        elif temp:
            result["weather"] = temp

    if "wind_speed" not in result:
        for line in lines:
            if line.startswith("風速"):
                result["wind_speed"] = line.replace("風速", "").strip()
                break


def fetch_beforeinfo_toda(race_no: int, date: str) -> Dict[Any, Any]:
    """
    戸田（jcd=02）の直前情報を boatrace.jp beforeinfo から取得。

    返却形式:
      {
        1: BeforeInfoRow(...),
        ...
        6: BeforeInfoRow(...),
        "weather": "...",
        "wind_speed": "...",
        "wind_direction": "..."
      }
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?hd={date}&jcd=02&rno={race_no}"

    result = _make_default_result()
    session = _get_session()

    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, _safe_parser())

    # まず高速なDOM解析
    _fill_from_dom(soup, result)

    # 足りない時だけ保険
    if _needs_fallback(result):
        _fill_from_text_fallback(soup, result)

    return result
