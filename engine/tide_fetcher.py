# engine/tide_fetcher.py
"""
気象庁(JMA)公式の潮位表テキストデータを取得・解析し、任意の会場・日時における
潮位(cm)と潮位トレンド(cm/h、正=上げ潮/満ちてくる、負=下げ潮/引いてくる)を返す。

データソース: https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{year}/{station}.txt
(気象庁の公式・無料の推算潮位データ、DRMやToS上の懸念なし。[[boat_ai_tide_data_research]]参照)

フォーマット(readme.htmlで確認済み、固定長テキスト、1行=1観測点1日):
  1-72桁   : 0時〜23時の毎時潮位(3桁×24、単位cm、欠測は"999")
  73-78桁  : 日付(YY, 半角スペース詰め2桁のMM, 半角スペース詰め2桁のDD)
  79-80桁  : 観測点記号(2文字)
  81桁以降 : 満潮・干潮の時刻/潮位(本モジュールでは未使用)

会場→観測点マッピングは2026-08-10にJMAの掲載地点一覧表・地方別ページを調べて
決定した。桐生・戸田・びわこは内陸(海洋潮汐の対象外)、鳴門・大村は近隣に
信頼できる観測点が見つからなかったため意図的にNoneにしてある(不正確な代用点を
使うより「データ無し」として0埋めにフォールバックする方が安全、という方針)。
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

CACHE_DIR = Path("data/tide_cache")

# venue_name -> JMA観測点記号(2文字)。Noneは「対象外 or 適切な観測点が未確定」。
VENUE_TIDE_STATION: Dict[str, Optional[str]] = {
    "桐生": None,    # 内陸(渡良瀬川、潮汐の影響なし)
    "戸田": None,    # 内陸(戸田ボートコース、潮汐の影響なし)
    "びわこ": None,  # 内陸・淡水湖、潮汐なし
    "江戸川": "TK",  # 東京湾岸、東京で近似
    "平和島": "TK",
    "多摩川": "TK",
    "浜名湖": "MI",  # 舞阪(浜名湖の今切口そのもの)
    "蒲郡":   "G4",  # 三河(三河湾)
    "常滑":   "ZD",  # 鬼崎(知多半島、伊勢湾)
    "津":     "TB",  # 鳥羽(四日市港(G3)も近いが緯度がやや鳥羽寄り)
    "三国":   "ZG",  # 三国(日本海側、会場名と同一観測点)
    "住之江": "OS",  # 大阪(大阪湾)
    "尼崎":   "AM",  # 尼崎(会場名と同一観測点)
    "鳴門":   None,  # 鳴門海峡特有の激しい潮流があり、近隣の代用点(洲本等)は
                      # 特性が異なりすぎるため意図的に除外。専用観測点は今回未確認。
    "丸亀":   "TX",  # 多度津
    "児島":   "UN",  # 宇野(児島・宇野フェリー航路の宇野港)
    "宮島":   "Q8",  # 広島(広島湾)
    "徳山":   "QA",  # 徳山(会場名と同一観測点)
    "下関":   "CF",  # 長府(2026-08-10修正: 会場名と同一の"DS"は気象庁の
                      # バルクテキスト配信対象外(全年404)と判明。公式観測点
                      # 一覧に載っている隣接点「長府」に変更。ボートレース下関
                      # 施設の座標(北緯34.019, 東経131.004)から長府観測点
                      # (北緯34゜01', 東経131゜00')まで約2〜3km、同じ関門海峡内。
    "若松":   "N1",  # 日明(北九州・洞海湾周辺の代用点、要精査)
    "芦屋":   "QF",  # 博多で代用(玄界灘沿岸で他に適当な掲載点が見つからず、要精査)
    "福岡":   "QF",  # 博多
    "唐津":   "KA",  # 唐津(会場名と同一観測点)
    "大村":   None,  # 大村湾は狭い湾口を持つ内海で潮汐特性が外洋と異なる。
                      # 適切な代用観測点が見つからなかったため意図的に除外。
}

_SESSION: Optional[requests.Session] = None
_HOURLY_CACHE: Dict[Tuple[str, str], Optional[List[Optional[int]]]] = {}


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        _SESSION = s
    return _SESSION


def get_tide_station(venue_name: str) -> Optional[str]:
    return VENUE_TIDE_STATION.get(venue_name)


def _cache_path(station: str, year: int) -> Path:
    return CACHE_DIR / f"{station}_{year}.txt"


def _fetch_tide_year_text(station: str, year: int) -> Optional[str]:
    """station×yearの全年テキストをローカルキャッシュ優先で取得する。
    取得できなければNone(呼び出し側は潮位特徴量を0埋めにフォールバックする)。
    """
    cache_path = _cache_path(station, year)
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    url = f"https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{year}/{station}.txt"
    try:
        res = _get_session().get(url, timeout=(5, 20))
        res.raise_for_status()
        text = res.text
    except requests.exceptions.RequestException as e:
        print(f"⚠ tide通信エラー station={station} year={year}: {e}")
        return None

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    except Exception:
        pass

    return text


def _parse_tide_line(line: str) -> Optional[Tuple[str, List[Optional[int]]]]:
    """1行をパースし、(date_str "YYYYMMDD", [0時〜23時の潮位cm(欠測はNone)]) を返す。"""
    if len(line) < 80:
        return None

    hourly_part = line[0:72]
    date_part = line[72:78]

    try:
        levels: List[Optional[int]] = []
        for i in range(0, 72, 3):
            raw = hourly_part[i:i + 3]
            v = int(raw)
            levels.append(None if v == 999 else v)
    except ValueError:
        return None

    try:
        yy = int(date_part[0:2])
        mm = int(date_part[2:4])
        dd = int(date_part[4:6])
    except ValueError:
        return None

    year = 2000 + yy
    date_str = f"{year:04d}{mm:02d}{dd:02d}"
    return date_str, levels


def get_hourly_levels(station: str, date_str: str) -> Optional[List[Optional[int]]]:
    """station×date("YYYYMMDD")の0時〜23時の潮位cm配列(24要素、欠測はNone)を返す。
    見つからなければNone。年単位でメモリキャッシュする。
    """
    date_str = str(date_str).strip()
    if len(date_str) != 8:
        return None
    year = date_str[:4]

    cache_key = (station, date_str)
    if cache_key in _HOURLY_CACHE:
        return _HOURLY_CACHE[cache_key]

    text = _fetch_tide_year_text(station, int(year))
    if text is None:
        return None

    # 同じ年のテキストから全日付分を一度にパースし、まとめてキャッシュしておく
    # (1レースごとに全文再パースするのは無駄なため)。
    for line in text.splitlines():
        parsed = _parse_tide_line(line)
        if parsed is None:
            continue
        d, levels = parsed
        _HOURLY_CACHE[(station, d)] = levels

    return _HOURLY_CACHE.get(cache_key)


def get_tide_features(
    venue_name: str, date_str: str, hh: int, mm: int
) -> Tuple[Optional[float], Optional[float]]:
    """会場・日付・時刻(hh:mm)における潮位(cm)とトレンド(cm/h、正=上げ潮)を返す。
    観測点未確定/データ取得失敗/欠測の場合は (None, None)。
    """
    station = get_tide_station(venue_name)
    if station is None:
        return None, None

    levels = get_hourly_levels(station, date_str)
    if levels is None:
        return None, None

    frac_hour = max(0.0, min(23.999, float(hh) + float(mm) / 60.0))
    h0 = int(math.floor(frac_hour))
    h1 = min(h0 + 1, 23)

    lv0 = levels[h0]
    lv1 = levels[h1]
    if lv0 is None or lv1 is None:
        return None, None

    frac = frac_hour - h0
    level = lv0 + (lv1 - lv0) * frac
    trend = float(lv1 - lv0)  # h1-h0=1時間なのでそのままcm/h

    return float(level), trend
