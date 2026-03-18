# engine/leakage_guard.py
from __future__ import annotations

from typing import Iterable, List
import re


# 「これはオッズ/払戻/結果リークっぽい」と見なす列名パターン
# ※ combo は特徴量では落とすが、学習CSVでは説明用に残る場合があるのでここでは除外しない
_LEAK_PATTERNS = [
    r"\bodds\b",
    r"\bodds_", r"_odds\b",
    r"\bpayout\b",
    r"\brefund\b",
    r"\bpay\b",
    r"\breturn\b",
    r"\bdividend\b",
    r"\bwinning\b",
    r"\bresult\b",
    r"\bfinish\b",
    r"\brank\b",
    r"\bplace\b",  # データによっては「場名(place)」が venue と被るので注意。必要なら外してOK
]

# ただし「venue/place=競艇場」の意味で使ってるケースがあるので例外扱い
# ここはあなたのデータ仕様で増減OK
_SAFE_EXACT = {"venue", "place_name", "stadium", "jcd", "course"}  # courseは枠のコースではなく環境で使うなら残す


def find_odds_leak_columns(cols: Iterable[str]) -> List[str]:
    bad: List[str] = []
    for c in cols:
        cs = str(c)
        low = cs.lower().strip()

        if cs in _SAFE_EXACT:
            continue

        for pat in _LEAK_PATTERNS:
            if re.search(pat, low):
                bad.append(cs)
                break

    return sorted(set(bad))


def drop_odds_leakage(df, verbose: bool = False, context: str = ""):
    """
    DataFrameからリーク列を落とす。
    返り値は df（copyせずinplace drop）
    """
    try:
        cols = list(getattr(df, "columns", []))
    except Exception:
        return df

    bad = find_odds_leak_columns(cols)
    if bad:
        if verbose:
            tag = f"[LEAK_DROP]{context}" if context else "[LEAK_DROP]"
            print(tag, "drop_cols=", bad)
        try:
            df = df.drop(columns=bad, errors="ignore")
        except Exception:
            pass
    return df
