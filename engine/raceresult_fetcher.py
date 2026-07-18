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
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


RACERESULT_URL = "https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={hd}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = PROJECT_ROOT / "data" / "logs" / "raceresult_debug"

_SESSION: Optional[requests.Session] = None

_NO_DATA_MARKERS = [
    "該当するデータがありません",
    "見つかりません",
    "ページが見つかりません",
]

_RE_COMBO_3 = re.compile(r"([1-6]-[1-6]-[1-6])")
_RE_COMBO_2 = re.compile(r"([1-6]-[1-6])")
_RE_YEN = re.compile(r"([\d,]+)\s*円")

_LABELS = ["3連単", "3連複", "2連単", "2連複", "拡連複", "単勝", "複勝"]


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


def _extract_after_label(text: str, label: str, next_labels: List[str]) -> str:
    """text 中で label が最初に現れる位置から、次のラベルが現れる位置(または末尾)までを返す"""
    idx = text.find(label)
    if idx < 0:
        return ""
    start = idx + len(label)
    end = len(text)
    for nl in next_labels:
        p = text.find(nl, start)
        if p >= 0:
            end = min(end, p)
    return text[start:end]


def _first_combo_and_payout(segment: str, combo_re: "re.Pattern") -> Optional[Tuple[str, float]]:
    combo_m = combo_re.search(segment)
    yen_m = _RE_YEN.search(segment)
    if combo_m and yen_m:
        return combo_m.group(1), float(yen_m.group(1).replace(",", ""))
    return None


def _parse_payout_section(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    「払戻金」という文字列を含むテーブルを探し、正規化したテキストから
    3連単/3連複/2連単/2連複 の組番・払戻金を抽出する。
    build_results_from_k.py の RE_PAYOUT_SUMMARY_LINE_FULL と同じ発想を
    HTMLテキスト向けに書き直したもの。
    """
    target = None
    for table in soup.find_all("table"):
        if "払戻金" in table.get_text():
            target = table
            break

    if target is None:
        return {}

    raw = target.get_text(separator="|")
    text = re.sub(r"\|+", "|", raw)

    out: Dict[str, Any] = {}

    for i, label in enumerate(_LABELS):
        seg = _extract_after_label(text, label, _LABELS[i + 1:])
        if not seg:
            continue

        if label == "3連単":
            r = _first_combo_and_payout(seg, _RE_COMBO_3)
            if r:
                out["actual_combo"], out["payout"] = r
        elif label == "3連複":
            r = _first_combo_and_payout(seg, _RE_COMBO_3)
            if r:
                out["combo_3puku"], out["payout_3puku"] = r
        elif label == "2連単":
            r = _first_combo_and_payout(seg, _RE_COMBO_2)
            if r:
                out["combo_2tan"], out["payout_2tan"] = r
        elif label == "2連複":
            r = _first_combo_and_payout(seg, _RE_COMBO_2)
            if r:
                out["combo_2puku"], out["payout_2puku"] = r

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
