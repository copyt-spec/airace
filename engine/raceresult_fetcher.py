# engine/raceresult_fetcher.py
# -*- coding: utf-8 -*-
"""
boatrace.jp の「レース結果」ページ(raceresult)から払戻結果(3連単/3連複/2連単/2連複)
を取得するモジュール。

engine/beforeinfo_fetcher.py と同じ requests + BeautifulSoup パターンを使う。
LZH形式のKファイルと違い、これは通常のHTMLページなので解凍ツールが不要。

【注意】このリポジトリのサンドボックス環境からは boatrace.jp への通信がブロックされて
おり、このモジュールをここで実データに対して動作確認することはできない。
ユーザーの手元環境(ネットワーク到達可能)で実行し、パースに失敗する場合は
`fetch_race_result(..., save_debug=True)` で保存される生HTMLを見ながら
正規表現/セレクタを調整する想定。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


RACERESULT_URL = "https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={hd}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = PROJECT_ROOT / "data" / "logs" / "raceresult_debug"

_SESSION: Optional[requests.Session] = None

_NO_DATA_MARKERS = [
    # 実ページで確認した実際の文言(2026-07-19)。「該当する」は付かない。
    "データがありません",
    "見つかりません",
    "ページが見つかりません",
]

_RE_YEN = re.compile(r"([\d,]+)")


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; BoatAIResultBot/1.0)",
                "Referer": "https://www.boatrace.jp/",
                "Connection": "keep-alive",
            }
        )
        _SESSION = session
    return _SESSION


_RE_COMBO3_FULL = re.compile(r"^[1-6]-[1-6]-[1-6]$")
_RE_COMBO2_FULL = re.compile(r"^[1-6]-[1-6]$")

_LABEL_TO_FIELDS = {
    "3連単": ("actual_combo", "payout", _RE_COMBO3_FULL),
    "3連複": ("combo_3puku", "payout_3puku", _RE_COMBO3_FULL),
    "2連単": ("combo_2tan", "payout_2tan", _RE_COMBO2_FULL),
    "2連複": ("combo_2puku", "payout_2puku", _RE_COMBO2_FULL),
}


def _parse_payout_section(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    「払戻金」を含むテーブル(勝式/組番/払戻金/人気の列を持つ、
    class="is-w495" のテーブル)を探し、DOM構造から直接
    3連単/3連複/2連単/2連複 の組番・払戻金を抽出する。

    実ページのDOM構造(2026-07時点で確認済み):
      <table class="is-w495">
        <tbody>  <!-- 勝式ごとに個別のtbody -->
          <tr class="is-p3-0">
            <td rowspan="2">3連単</td>
            <td><div class="numberSet1 is-small"><div class="numberSet1_row">
              <span class="numberSet1_number is-type2">2</span>
              <span class="numberSet1_text">-</span>
              <span class="numberSet1_number is-type4">4</span>
              <span class="numberSet1_text">-</span>
              <span class="numberSet1_number is-type1">1</span>
            </div></div></td>
            <td><span class="is-payout1">&yen;4,190</span></td>
            <td>17</td>
          </tr>
          <tr class="is-p3-0">...空の継続行(2着候補が無い場合の穴埋め)...</tr>
        </tbody>
        <tbody>...3連複...</tbody>
        ...
      </table>

    以前のバージョンは get_text(separator="|") で全体をテキスト化してから
    正規表現で探していたが、組番の各桁が別々の<span>に入っているため
    "2-4-1" が "2|-|4|-|1" のように分断され、正規表現が一切マッチしない
    バグがあった(2026-07-19に実際のHTMLで発覚・修正)。
    DOM構造(.numberSet1_row / .is-payout1)を直接見ることで解決する。
    """
    target = None
    for table in soup.find_all("table"):
        if "払戻金" in table.get_text():
            target = table
            break

    if target is None:
        return {}

    out: Dict[str, Any] = {}

    for tbody in target.find_all("tbody"):
        rows = tbody.find_all("tr")
        if not rows:
            continue

        label_td = rows[0].find("td")
        label = label_td.get_text(strip=True) if label_td else ""
        if label not in _LABEL_TO_FIELDS:
            continue

        combo_field, payout_field, combo_re = _LABEL_TO_FIELDS[label]
        if combo_field in out:
            continue

        for row in rows:
            number_row = row.select_one(".numberSet1_row")
            payout_span = row.select_one(".is-payout1")
            if number_row is None or payout_span is None:
                continue

            combo_text = number_row.get_text(strip=True).replace("=", "-")
            if not combo_re.match(combo_text):
                continue

            yen_m = _RE_YEN.search(payout_span.get_text(strip=True))
            if not yen_m:
                continue

            out[combo_field] = combo_text
            out[payout_field] = float(yen_m.group(1).replace(",", ""))
            break

    return out


def _save_debug_html(html: str, rno: int, date: str, venue_code: int) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"raceresult_{date}_{str(venue_code).zfill(2)}_{rno}.html"
    path.write_text(html, encoding="utf-8")
    return path


def fetch_race_result(
    rno: int,
    date: str,
    venue_code: int,
    *,
    save_debug: bool = False,
    sleep_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    boatrace.jp のレース結果ページから3連単等の払戻結果を取得する。

    戻り値: 取得できた場合は
      {"actual_combo": "1-2-3", "payout": 1830.0,
       "combo_3puku": "1-2-3", "payout_3puku": 610.0,
       "combo_2tan": "1-2", "payout_2tan": 720.0,
       "combo_2puku": "1-2", "payout_2puku": 240.0,
       "date": ..., "race_no": ..., "source": "boatrace.jp raceresult"}
    取得失敗・そのレースが未開催/未確定の場合は空dict。
    """
    jcd = str(int(venue_code)).zfill(2)
    url = RACERESULT_URL.format(rno=int(rno), jcd=jcd, hd=str(date))

    session = _get_session()
    try:
        res = session.get(url, timeout=(5, 20))
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"⚠ raceresult通信エラー: {url} :: {e}")
        return {}
    finally:
        if sleep_sec:
            time.sleep(sleep_sec)

    body_text = res.text
    if any(marker in body_text for marker in _NO_DATA_MARKERS):
        return {}

    soup = BeautifulSoup(res.text, "html.parser")
    payout = _parse_payout_section(soup)

    if not payout.get("actual_combo"):
        if save_debug:
            path = _save_debug_html(res.text, rno, date, venue_code)
            print(f"[DEBUG] parse失敗、生HTMLを保存: {path}")
        return {}

    payout["date"] = str(date)
    payout["race_no"] = int(rno)
    payout["source"] = "boatrace.jp raceresult"
    return payout
