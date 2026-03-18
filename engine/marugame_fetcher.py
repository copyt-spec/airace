# engine/marugame_fetcher.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# =========================
# in-memory cache
# =========================
_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_SECONDS = 60


def _get_cache(key: str) -> List[Dict[str, Any]] | None:
    item = _CACHE.get(key)
    if not item:
        return None

    ts, data = item
    if time.time() - ts > _CACHE_SECONDS:
        _CACHE.pop(key, None)
        return None

    return data


def _set_cache(key: str, value: List[Dict[str, Any]]) -> None:
    _CACHE[key] = (time.time(), value)


# =========================
# webdriver builder
# =========================
def _build_driver() -> webdriver.Chrome:
    options = Options()

    # headless
    options.add_argument("--headless=new")

    # stability
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # performance
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--blink-settings=imagesEnabled=false")

    options.add_argument("--window-size=1920,1080")

    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # Docker / Render / local 対応
    chrome_bin = os.getenv("CHROME_BIN")
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH")

    if chrome_bin:
        options.binary_location = chrome_bin

    # 画像ロードをさらに抑える
    prefs = {
        "profile.managed_default_content_settings.images": 2,
    }
    options.add_experimental_option("prefs", prefs)

    if chromedriver_path and os.path.exists(chromedriver_path):
        service = Service(chromedriver_path)
        return webdriver.Chrome(service=service, options=options)

    return webdriver.Chrome(options=options)


def fetch_all_entries_once(date: str) -> List[Dict[str, Any]]:
    """
    丸亀の出走表を 12R 分まとめて取得
    返却:
      [
        {
          "lane": 1,
          "racer_no": "1234",
          "name_branch": "山田太郎 香川",
          "grade": "A1",
          "fl": "F0",
          "win_rate": "6.78",
          "quinella_rate": "45.2",
          "race_no": 1
        },
        ...
      ]
    """
    cache_key = f"marugame_entries_{date}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    # ※ date は現状URLに未使用だが、将来差し替え前提で key には含める
    url = "https://www.marugameboat.jp/s_pdf/date.htm"

    driver = _build_driver()

    try:
        driver.get(url)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )

        tables = driver.find_elements(By.TAG_NAME, "table")

        all_entries: List[Dict[str, Any]] = []
        race_no = 0

        for table in tables:
            rows = table.find_elements(By.TAG_NAME, "tr")
            temp_entries: List[Dict[str, Any]] = []

            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")

                if len(cols) < 6:
                    continue

                lane = cols[0].text.strip()
                if not lane.isdigit():
                    continue

                raw_info = cols[1].text.strip()
                if len(raw_info) < 4:
                    continue

                # 登録番号
                racer_no = raw_info[:4]

                # 登録番号以降
                raw_name_part = raw_info[4:].strip()

                if "　" in raw_name_part:
                    parts = [p.strip() for p in raw_name_part.split("　") if p.strip()]
                    if len(parts) >= 3:
                        surname = parts[0]
                        firstname = parts[1]
                        branch = parts[2]
                    elif len(parts) == 2:
                        surname = parts[0]
                        firstname = parts[1]
                        branch = ""
                    elif len(parts) == 1:
                        surname = parts[0]
                        firstname = ""
                        branch = ""
                    else:
                        surname = ""
                        firstname = ""
                        branch = ""

                    full_name = f"{surname}{firstname}".strip()
                else:
                    # スペース無し例外（後ろ2文字を支部とみなす）
                    if len(raw_name_part) >= 3:
                        full_name = raw_name_part[:-2].strip()
                        branch = raw_name_part[-2:].strip()
                    else:
                        full_name = raw_name_part
                        branch = ""

                name_branch = f"{full_name} {branch}".strip()

                grade = cols[2].text.strip()
                fl = cols[3].text.strip()
                win_rate = cols[4].text.strip()
                quinella_rate = cols[5].text.strip()

                temp_entries.append(
                    {
                        "lane": int(lane),
                        "racer_no": racer_no,
                        "name_branch": name_branch,
                        "grade": grade,
                        "fl": fl,
                        "win_rate": win_rate,
                        "quinella_rate": quinella_rate,
                    }
                )

            # 6艇そろったテーブルのみ採用
            if len(temp_entries) == 6:
                race_no += 1
                for entry in temp_entries:
                    entry["race_no"] = race_no
                    all_entries.append(entry)

        _set_cache(cache_key, all_entries)
        return all_entries

    finally:
        driver.quit()
