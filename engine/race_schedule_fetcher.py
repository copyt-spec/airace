# engine/race_schedule_fetcher.py
# -*- coding: utf-8 -*-
"""
boatrace.jp の raceindex ページから各会場・日付の発走(締切予定)時刻を取得する。

注意: このリポジトリのサンドボックス環境からはboatrace.jpへの接続が
ブロックされているため、この実装はローカル環境で一度も実際のページに
対して動作確認できていない。ページ構造が変わっている/想定と違う場合は
fetch_post_times()の抽出ロジックを調整すること
(scripts/notify_rank_a_races.py --debug-schedule で単体確認できる)。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Optional

import requests

from engine.venue_registry import get_jcd

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    )
}

# 2026-07-25: 実際のraceindexページ(html5libが無くpd.read_htmlは使えない環境
# だったため生HTMLを直接確認)で以下を確認済み:
#   - レース番号は `...jcd=02&hd=20260725">1R</a>` のように <a> タグの中に
#     "1R".."12R" の形でページ上部に1〜12の順で並ぶ
#   - 発走(締切予定)時刻は `<td>10:47</td>` のように別セルに、同じく
#     1R〜12Rの順で並ぶ(番号とセルは離れた位置にあるため、両方をページ内の
#     出現順で別々に抜き出してから位置で対応させる)
#   - ページ内の他の場所にあるトークン文字列などに偶然 "31R" のような
#     部分一致が発生することがあるため、`>(\d{1,2})R</a>` のように
#     アンカータグの境界まで含めて厳密にマッチさせている
_RACE_NO_TAG_RE = re.compile(r">(\d{1,2})R</a>")
_TIME_TD_RE = re.compile(r"<td>([01]?\d:[0-5]\d)</td>")


def fetch_post_times(venue: str, date: str) -> Dict[int, str]:
    """
    その会場・日付の各レースの発走(締切予定)時刻を取得する。
    開催が無い会場・日付の場合は空dictを返す。

    戻り値: {race_no(int): "HH:MM"}
    """
    jcd = get_jcd(venue)
    url = f"https://www.boatrace.jp/owpc/pc/race/raceindex?jcd={jcd:02d}&hd={date}"

    resp = requests.get(url, headers=UA, timeout=(5, 20))
    resp.raise_for_status()
    html = resp.text

    race_nos = sorted({int(m.group(1)) for m in _RACE_NO_TAG_RE.finditer(html) if 1 <= int(m.group(1)) <= 12})
    times = [m.group(1) for m in _TIME_TD_RE.finditer(html)]

    if not race_nos or not times:
        return {}

    n = min(len(race_nos), len(times))
    return {race_nos[i]: times[i] for i in range(n)}


def post_datetime(venue: str, date: str, race_no: int) -> Optional[datetime]:
    times = fetch_post_times(venue, date)
    hhmm = times.get(race_no)
    if not hhmm:
        return None
    hh, mm = hhmm.split(":")
    d = datetime.strptime(date, "%Y%m%d")
    return d.replace(hour=int(hh), minute=int(mm))


if __name__ == "__main__":
    # 動作確認用: python -m engine.race_schedule_fetcher 戸田 20260725
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else "戸田"
    d = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y%m%d")
    print(fetch_post_times(v, d))
