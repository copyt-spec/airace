from __future__ import annotations

"""
3連単(120通り)の予測確率から、他の券種(3連複・2連単・2連複)の
確率/概算オッズを導出するユーティリティ。

- 確率の変換は「同じ着順分布の周辺化・集約」なので、再学習は不要で厳密に正しい。
  例: 3連複{a,b,c}の確率 = 3連単 a-b-c, a-c-b, b-a-c, b-c-a, c-a-b, c-b-a の確率の合計

- オッズの変換(3連単オッズ→3連複オッズの概算)は、パリミュチュエル方式で
  控除率が同じだと仮定した近似(調和平均的な変換)。実際は3連単と3連複は
  別プールなので、あくまで過去データがまだ無い期間の概算・バックテスト用。
  本番では engine/odds_fetcher.py 側で3連複の実オッズを別途取得して使う。
"""

import itertools
from typing import Dict, Iterable, List, Tuple

Combo3 = Tuple[int, int, int]


def _parse_combo3(combo: str) -> Tuple[int, int, int]:
    a, b, c = combo.split("-")
    return int(a), int(b), int(c)


def _box_key(a: int, b: int, c: int) -> str:
    return "-".join(str(x) for x in sorted((a, b, c)))


def _pair_key(a: int, b: int) -> str:
    return "-".join(str(x) for x in sorted((a, b)))


def sanrenpuku_probs(prob_map_3tan: Dict[str, float]) -> Dict[str, float]:
    """3連単の確率を3連複(順不同・20通り)に集約する。"""
    out: Dict[str, float] = {}
    for combo, p in prob_map_3tan.items():
        try:
            a, b, c = _parse_combo3(combo)
        except Exception:
            continue
        key = _box_key(a, b, c)
        out[key] = out.get(key, 0.0) + float(p)
    return out


def nirentan_probs(prob_map_3tan: Dict[str, float]) -> Dict[str, float]:
    """3連単を1・2着(順序あり・30通り)に周辺化する。"""
    out: Dict[str, float] = {}
    for combo, p in prob_map_3tan.items():
        try:
            a, b, _c = _parse_combo3(combo)
        except Exception:
            continue
        key = f"{a}-{b}"
        out[key] = out.get(key, 0.0) + float(p)
    return out


def nirenpuku_probs(prob_map_3tan: Dict[str, float]) -> Dict[str, float]:
    """3連単を1・2着(順不同・15通り)に周辺化する。"""
    out: Dict[str, float] = {}
    for combo, p in prob_map_3tan.items():
        try:
            a, b, _c = _parse_combo3(combo)
        except Exception:
            continue
        key = _pair_key(a, b)
        out[key] = out.get(key, 0.0) + float(p)
    return out


def kakurenpuku_probs(prob_map_3tan: Dict[str, float]) -> Dict[str, float]:
    """
    拡連複(上位3着のうち指定した2艇が共に入る・15通り)の確率。
    「1-2着」に限定しないので、3連複の20通りから、含まれる3つのペア
    それぞれに確率を加算する(1レース内で1つの拡連複combo は最大3回の
    3着以内ヒットチャンスがあるが、賭けとしては1通りにつき1回分の的中判定)。
    """
    box_probs = sanrenpuku_probs(prob_map_3tan)
    out: Dict[str, float] = {}
    for box_key, p in box_probs.items():
        a, b, c = (int(x) for x in box_key.split("-"))
        for x, y in itertools.combinations((a, b, c), 2):
            key = _pair_key(x, y)
            out[key] = out.get(key, 0.0) + float(p)
    return out


def approx_box_odds_from_3tan_odds(odds_map_3tan: Dict[str, float]) -> Dict[str, float]:
    """
    3連単オッズから3連複オッズを近似する。

    パリミュチュエル方式では 期待払戻率(控除後) k はほぼ一定なので、
    ある組み合わせの市場が織り込む確率は概ね odds に反比例する:
        p_market(combo) ≈ k / odds(combo)
    3連複{a,b,c}が的中する市場確率は、対応する6通りの3連単の確率の合計:
        p_market(box) ≈ k * sum(1/odds(perm) for perm in 6 permutations)
    よって3連複オッズの近似値:
        odds(box) ≈ k / p_market(box) = 1 / sum(1/odds(perm))

    ※ 実際は3連単プールと3連複プールは別物で、控除率もオッズ形成も
      完全には一致しないため、この近似は「参考値」であり実オッズの
      代わりにはならない。過去データの3連複オッズが無い期間の
      概算バックテスト専用。
    """
    inv_sum: Dict[str, float] = {}
    for combo, odds in odds_map_3tan.items():
        if odds is None or odds <= 0:
            continue
        try:
            a, b, c = _parse_combo3(combo)
        except Exception:
            continue
        key = _box_key(a, b, c)
        inv_sum[key] = inv_sum.get(key, 0.0) + (1.0 / float(odds))

    out: Dict[str, float] = {}
    for key, s in inv_sum.items():
        if s > 0:
            out[key] = 1.0 / s
    return out


def all_box_combos() -> List[str]:
    """3連複の全20通りのキー一覧。"""
    out = []
    for a, b, c in itertools.combinations(range(1, 7), 3):
        out.append(f"{a}-{b}-{c}")
    return out


def all_pair_combos() -> List[str]:
    """2連複/拡連複の全15通りのキー一覧。"""
    out = []
    for a, b in itertools.combinations(range(1, 7), 2):
        out.append(f"{a}-{b}")
    return out
