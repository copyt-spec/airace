# engine/beforeinfo_fetcher.py

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session():
    session = requests.Session()

    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    return session


def fetch_beforeinfo(race_no, date, venue_code=15):
    """
    直前情報取得（デフォルト丸亀=15のまま）
    venue_code を渡せば戸田(2)などに対応できる
    """

    jcd = str(int(venue_code)).zfill(2)
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={jcd}&hd={date}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.boatrace.jp/",
        "Connection": "keep-alive",
    }

    session = create_session()

    data = {}

    try:
        res = session.get(
            url,
            headers=headers,
            timeout=(5, 20)  # (connect, read)
        )

        res.raise_for_status()

    except requests.exceptions.RequestException as e:
        print("⚠ beforeinfo通信エラー:", e)
        return {
            i: {
                "exhibit_time": "取得失敗",
                "tilt": "取得失敗",
                "parts": "",
                "st": "取得失敗",
                "course": "取得失敗"
            } for i in range(1, 7)
        }

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
                lane = int(cols[0].text.strip())
            except:
                continue

            data[lane] = {
                "exhibit_time": cols[4].text.strip(),
                "tilt": cols[5].text.strip(),
                "parts": cols[7].text.strip(),
                "st": "未取得",
                "course": "未取得"
            }

        break

    # ==========================
    # ST / 進入
    # ==========================

    lane_spans = soup.select("span[class*='boatImage1Number']")
    st_spans = soup.select("span[class*='boatImage1Time']")

    if len(lane_spans) == 6 and len(st_spans) == 6:

        for i in range(6):

            course = lane_spans[i].text.strip()
            st_raw = st_spans[i].text.strip()

            if st_raw.startswith("."):
                st = "0" + st_raw
            else:
                st = st_raw

            lane = int(course)

            if lane in data:
                data[lane]["st"] = st
                data[lane]["course"] = course

    # ==========================
    # 気温
    # ==========================

    temp_block = soup.select_one(".weather1_bodyUnit")

    if temp_block:
        data["weather"] = temp_block.get_text(strip=True)

    # ==========================
    # 風速
    # ==========================

    wind_speed_block = soup.select_one(".weather1_bodyUnit.is-wind")

    if wind_speed_block:
        speed = wind_speed_block.select_one(".weather1_bodyUnitLabelData")
        if speed:
            data["wind_speed"] = speed.text.strip()

    # ==========================
    # 風向き
    # ==========================

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
                        "8": "北西"
                    }

                    data["wind_direction"] = wind_map.get(num, "不明")

    return data
