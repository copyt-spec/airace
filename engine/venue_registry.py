from __future__ import annotations

from typing import Dict, List


VENUE_MASTER: Dict[str, Dict[str, object]] = {
    "桐生":   {"venue_key": "kiryu",      "jcd": 1},
    "戸田":   {"venue_key": "toda",       "jcd": 2},
    "江戸川": {"venue_key": "edogawa",    "jcd": 3},
    "平和島": {"venue_key": "heiwajima",  "jcd": 4},
    "多摩川": {"venue_key": "tamagawa",   "jcd": 5},
    "浜名湖": {"venue_key": "hamanako",   "jcd": 6},
    "蒲郡":   {"venue_key": "gamagori",   "jcd": 7},
    "常滑":   {"venue_key": "tokoname",   "jcd": 8},
    "津":     {"venue_key": "tsu",        "jcd": 9},
    "三国":   {"venue_key": "mikuni",     "jcd": 10},
    "びわこ": {"venue_key": "biwako",     "jcd": 11},
    "住之江": {"venue_key": "suminoe",    "jcd": 12},
    "尼崎":   {"venue_key": "amagasaki",  "jcd": 13},
    "鳴門":   {"venue_key": "naruto",     "jcd": 14},
    "丸亀":   {"venue_key": "marugame",   "jcd": 15},
    "児島":   {"venue_key": "kojima",     "jcd": 16},
    "宮島":   {"venue_key": "miyajima",   "jcd": 17},
    "徳山":   {"venue_key": "tokuyama",   "jcd": 18},
    "下関":   {"venue_key": "shimonoseki","jcd": 19},
    "若松":   {"venue_key": "wakamatsu",  "jcd": 20},
    "芦屋":   {"venue_key": "ashiya",     "jcd": 21},
    "福岡":   {"venue_key": "fukuoka",    "jcd": 22},
    "唐津":   {"venue_key": "karatsu",    "jcd": 23},
    "大村":   {"venue_key": "omura",      "jcd": 24},
}

VENUE_ORDER: List[str] = list(VENUE_MASTER.keys())

VENUE_KEY_TO_NAME: Dict[str, str] = {
    str(v["venue_key"]): k for k, v in VENUE_MASTER.items()
}

VENUE_NAME_TO_KEY: Dict[str, str] = {
    k: str(v["venue_key"]) for k, v in VENUE_MASTER.items()
}

VENUE_NAME_TO_JCD: Dict[str, int] = {
    k: int(v["jcd"]) for k, v in VENUE_MASTER.items()
}

VENUE_JCD_TO_NAME: Dict[int, str] = {
    int(v["jcd"]): k for k, v in VENUE_MASTER.items()
}


def normalize_venue_name(v: object) -> str:
    s = str(v or "").strip()

    if s in VENUE_MASTER:
        return s

    for venue_name in VENUE_ORDER:
        if venue_name in s:
            return venue_name

    return s


def get_venue_key(venue_name: str) -> str:
    venue_name = normalize_venue_name(venue_name)
    if venue_name not in VENUE_NAME_TO_KEY:
        raise KeyError(f"unknown venue: {venue_name}")
    return VENUE_NAME_TO_KEY[venue_name]


def get_venue_name_by_key(venue_key: str) -> str:
    key = str(venue_key or "").strip()
    if key not in VENUE_KEY_TO_NAME:
        raise KeyError(f"unknown venue_key: {key}")
    return VENUE_KEY_TO_NAME[key]


def get_jcd(venue_name: str) -> int:
    venue_name = normalize_venue_name(venue_name)
    if venue_name not in VENUE_NAME_TO_JCD:
        raise KeyError(f"unknown venue: {venue_name}")
    return VENUE_NAME_TO_JCD[venue_name]
