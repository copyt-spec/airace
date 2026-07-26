# scripts/debug_race_schedule_fetch.py
# -*- coding: utf-8 -*-
"""
engine.race_schedule_fetcher が発走時刻を取得できない原因を調べるための
使い捨てデバッグスクリプト。

使い方:
  python scripts/debug_race_schedule_fetch.py 戸田 20260725

やること:
  1. raceindexページを取得し、ステータスコード・HTML長・タイトルを表示
  2. pd.read_html で取れるテーブル数と、各テーブルの先頭数行を表示
  3. 生HTML中に "HH:MM" 形式の文字列が何個あるか、周辺のテキストを表示
  4. 生HTML中に "R" を含む短い文字列(レース番号らしきもの)を何個含むか表示
  5. 取得したHTMLを data/logs/debug_raceindex_{venue}_{date}.html に保存
     (次に私が中身を見て抽出ロジックを直せるように)
"""

from __future__ import annotations

import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import get_jcd  # noqa: E402

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    )
}


def main() -> None:
    venue = sys.argv[1] if len(sys.argv) > 1 else "戸田"
    date = sys.argv[2] if len(sys.argv) > 2 else "20260725"

    jcd = get_jcd(venue)
    url = f"https://www.boatrace.jp/owpc/pc/race/raceindex?jcd={jcd:02d}&hd={date}"
    print("URL:", url)

    resp = requests.get(url, headers=UA, timeout=(5, 20))
    print("status_code:", resp.status_code)
    html = resp.text
    print("html length:", len(html))

    m_title = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
    print("title:", m_title.group(1).strip() if m_title else "(not found)")

    out_path = PROJECT_ROOT / "data" / "logs" / f"debug_raceindex_{venue}_{date}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print("saved raw html ->", out_path)

    print()
    print("=" * 60)
    print("pd.read_html結果")
    print("=" * 60)
    try:
        tables = pd.read_html(StringIO(html))
        print("table count:", len(tables))
        for i, t in enumerate(tables):
            print(f"--- table[{i}] shape={t.shape} ---")
            print(t.head(5).to_string())
    except Exception as e:
        print("pd.read_html failed:", e)

    print()
    print("=" * 60)
    print("時刻らしき文字列(HH:MM)の出現箇所(前後20文字)")
    print("=" * 60)
    time_matches = list(re.finditer(r"([01]?\d|2[0-3]):([0-5]\d)", html))
    print("count:", len(time_matches))
    for m in time_matches[:15]:
        s = max(0, m.start() - 20)
        e = min(len(html), m.end() + 20)
        snippet = html[s:e].replace("\n", " ")
        print(f"  ...{snippet}...")

    print()
    print("=" * 60)
    print("レース番号らしき文字列(数字+R)の出現箇所(前後20文字)")
    print("=" * 60)
    race_matches = list(re.finditer(r"\d{1,2}R", html))
    print("count:", len(race_matches))
    for m in race_matches[:15]:
        s = max(0, m.start() - 20)
        e = min(len(html), m.end() + 20)
        snippet = html[s:e].replace("\n", " ")
        print(f"  ...{snippet}...")


if __name__ == "__main__":
    main()
