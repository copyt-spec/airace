from __future__ import annotations

"""
複数のfetcherモジュールが同じURLを別々に取得してしまう無駄を防ぐための、
プロセス共有の生HTMLキャッシュ。

2026-07-16に発覚: engine/generic_racelist_fetcher.py の fetch_racelist_by_venue()
と engine/racelist_enricher.py の _fetch_lane_stats() が、まったく同じ
boatrace.jp の出走表ページ(https://www.boatrace.jp/owpc/pc/race/racelist?...)を
それぞれ別のセッション・別のキャッシュで二重に取得していた
(ユーザーから「レーサー情報取得が遅い」と指摘され調査して発見)。

この共有キャッシュをどちらのモジュールも参照するようにすることで、
片方が先に取得していれば、もう片方はネットワークアクセスせずに
同じHTMLを再利用できる。パース処理自体には一切手を入れない
(各モジュールは今まで通り自分でHTMLをパースする)。
"""

import time
from typing import Dict, Tuple

_PAGE_CACHE: Dict[str, Tuple[float, str]] = {}
_DEFAULT_TTL_SECONDS = 60


def get_page(url: str, session, timeout, cache_seconds: int = _DEFAULT_TTL_SECONDS) -> str:
    item = _PAGE_CACHE.get(url)
    if item is not None:
        ts, html = item
        if time.time() - ts <= cache_seconds:
            return html

    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    html = r.text

    _PAGE_CACHE[url] = (time.time(), html)

    if len(_PAGE_CACHE) > 500:
        now = time.time()
        stale_keys = [k for k, (ts, _) in _PAGE_CACHE.items() if now - ts > cache_seconds]
        for k in stale_keys[:200]:
            _PAGE_CACHE.pop(k, None)

    return html
