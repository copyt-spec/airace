# engine/kelly_allocator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import numpy as np


@dataclass
class KellyPick:
    combo: str
    frac: float          # bankrollに対する比率
    amount: int          # 円（丸め後）
    p: float
    odds: float
    edge: float          # p*odds - 1


def _finite_pos(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _project_simplex_with_caps(v: np.ndarray, z: float, caps: Optional[np.ndarray] = None) -> np.ndarray:
    """
    v を 0<=x<=caps, sum(x)<=z に射影（近似じゃなく、ちゃんと射影）
    - capsがNoneなら 0<=x, sum(x)<=z の射影
    """
    n = v.size
    if z <= 0:
        return np.zeros_like(v)

    x = np.maximum(v, 0.0)

    if caps is not None:
        x = np.minimum(x, caps)

    s = float(x.sum())
    if s <= z:
        return x

    # sum(x)=z になるように水位調整（capsあり）
    # 方法：二分探索で threshold を探す：x_i = clip(v_i - t, 0, cap_i)
    lo, hi = -1e6, 1e6
    for _ in range(80):
        mid = (lo + hi) / 2.0
        y = v - mid
        y = np.maximum(y, 0.0)
        if caps is not None:
            y = np.minimum(y, caps)
        if float(y.sum()) > z:
            lo = mid
        else:
            hi = mid

    y = v - hi
    y = np.maximum(y, 0.0)
    if caps is not None:
        y = np.minimum(y, caps)

    # 数値誤差調整
    ss = float(y.sum())
    if ss > 0 and abs(ss - z) > 1e-9:
        y *= (z / ss)
        if caps is not None:
            y = np.minimum(y, caps)
    return y


def kelly_allocate_exclusive(
    p_dict: Dict[str, float],
    odds_dict: Dict[str, float],
    bankroll: int,
    *,
    top_n: int = 30,                 # 最適化に入れる候補数（多すぎると分散）
    fractional: float = 0.25,        # 1/4ケリー推奨
    cap_total: float = 0.05,         # 総投資 上限（資金の5%とか）
    cap_per: float = 0.02,           # 1点 上限（資金の2%とか）
    min_bet_yen: int = 100,          # 最低購入額（ボートなら100円）
    step_lr: float = 0.3,            # 勾配法の学習率
    steps: int = 200,
) -> List[KellyPick]:
    """
    排他的 outcomes（120通り）に対する多点ケリー最適化
    - odds は「的中したときの払い戻し倍率（decimal）」想定（例: 33.8）
    """
    if bankroll <= 0:
        return []

    # odds & p の整形
    items: List[Tuple[str, float, float]] = []
    for combo, p in (p_dict or {}).items():
        p = _finite_pos(p, 0.0)
        o = _finite_pos(odds_dict.get(combo, 0.0), 0.0)
        if p <= 0 or o <= 1e-12:
            continue
        items.append((combo, p, o))

    if not items:
        return []

    # p 正規化（念のため）
    ps = np.array([x[1] for x in items], dtype=float)
    s = float(ps.sum())
    if s <= 0:
        return []
    ps = ps / s

    # 期待値がプラスの候補に絞る（p*odds > 1）
    combos = [x[0] for x in items]
    odds = np.array([x[2] for x in items], dtype=float)
    edge = ps * odds - 1.0
    idx_pos = np.where(edge > 0)[0]
    if idx_pos.size == 0:
        return []

    # edge上位で top_n に絞る
    idx_sorted = idx_pos[np.argsort(edge[idx_pos])[::-1]]
    idx_use = idx_sorted[: max(1, min(top_n, idx_sorted.size))]

    combos_u = [combos[i] for i in idx_use]
    p_u = ps[idx_use]
    o_u = odds[idx_use]

    # 最適化変数 f（比率）
    # 初期値：単点ケリーを軽く入れてから制約内に射影
    f0 = (p_u * o_u - 1.0) / np.maximum(o_u - 1.0, 1e-12)
    f0 = np.maximum(f0, 0.0)

    z = float(cap_total)  # 総投資上限
    caps = np.full_like(f0, float(cap_per))
    f = _project_simplex_with_caps(f0, z=z, caps=caps)

    # 目的関数：
    # すべての outcome を考えると、betしてない outcome の倍率は (1 - sumf)
    # betしてる outcome i の倍率は (1 - sumf) + f_i * o_i
    # ここでは候補外 outcome の総確率を (1 - sum(p_u)) としてまとめる
    p_out = float(1.0 - float(p_u.sum()))

    for _ in range(int(steps)):
        sumf = float(f.sum())
        base = 1.0 - sumf  # 外れた時（候補外含む）の倍率
        if base <= 1e-12:
            base = 1e-12

        win_mult = base + f * o_u
        win_mult = np.maximum(win_mult, 1e-12)

        # 勾配：
        # d/d f_i [ p_i log(base + f_i o_i) + p_out log(base) + Σ_{j!=i} p_j log(base + f_j o_j) ]
        # base = 1 - Σ f なので、各項に -1 の影響がある
        # 結果：
        # grad_i = p_i * (o_i - 1)/win_mult_i  -  (p_out/base)  - Σ_j p_j*(1/win_mult_j)
        # ただし Σ_j の項は iに共通なのでまとめられる
        common = float((p_u / win_mult).sum()) + (p_out / base)
        grad = p_u * (o_u / win_mult) - common

        # ascent
        f = f + float(step_lr) * grad

        # 射影（上限・非負・総上限）
        f = _project_simplex_with_caps(f, z=z, caps=caps)

    # フラクショナルケリー
    f = f * float(max(0.0, min(1.0, fractional)))

    # 円に落とす（100円単位）
    picks: List[KellyPick] = []
    for combo, frac, p, o in zip(combos_u, f.tolist(), p_u.tolist(), o_u.tolist()):
        if frac <= 0:
            continue
        amt = int((bankroll * frac) // min_bet_yen) * min_bet_yen
        if amt < min_bet_yen:
            continue
        picks.append(
            KellyPick(
                combo=combo,
                frac=float(amt) / float(bankroll),
                amount=amt,
                p=float(p),
                odds=float(o),
                edge=float(p * o - 1.0),
            )
        )

    # 金額が0になったせいで総額が減るので、見た目を整えるために amount順で返す
    picks.sort(key=lambda x: x.amount, reverse=True)
    return picks


def kelly_size_selected_bets(
    selected: List[Dict[str, Any]],
    bankroll: int,
    *,
    fractional: float = 0.25,
    cap_total: float = 0.05,
    cap_per: float = 0.02,
    min_bet_yen: int = 100,
) -> List[Dict[str, Any]]:
    """
    engine.buy_selector.select_best_bets() が既に選んだ買い目(combo/prob/odds)
    に対して、賭け金だけをケリー基準で決める。

    kelly_allocate_exclusive() との違い:
    - あちらは prob_map 全体(120通り)を渡して「どれを買うか」から
      最適化する多点ケリー。ここでは選定自体はやり直さず(venue別に
      チューニング済みのbuy_selectorのロジックをそのまま尊重する)、
      「既に選ばれた点それぞれに、単点ケリー公式 f*=(p*o-1)/(o-1) で
      エッジに応じた比率を割り当てる」だけの単純な処理。
    - 単点ケリーを個別に計算したあと、合計がcap_totalを超えていれば
      比例縮小するだけなので、prob_mapを選択候補だけに絞って再正規化
      する(=確率の合計が不当に1に近づいてしまう)ような誤りを避けている。

    2026-07-16: ROI%そのものを上げる施策ではなく、「同じ選択」でも
    賭け金をエッジの大きさに応じて傾斜配分することで、資産(バンクロール)
    の成長効率を上げることを狙ったもの。[[boat_ai-roi-findings]]参照。

    戻り値: selected の各dictに "kelly_frac"(bankroll比率)と
    "bet_amount_yen"(円、100円単位に丸め)を追加したコピーのリスト。
    エッジが無い(p*o<=1)候補は bet_amount_yen=0 になる。
    """
    if bankroll <= 0 or not selected:
        return [dict(r, kelly_frac=0.0, bet_amount_yen=0) for r in (selected or [])]

    fractional = max(0.0, min(1.0, float(fractional)))

    raw_fracs: List[float] = []
    for r in selected:
        p = _finite_pos(r.get("prob", r.get("score", 0.0)), 0.0)
        o = _finite_pos(r.get("odds_raw", r.get("odds", 0.0)), 0.0)

        if p <= 0 or o <= 1.0:
            raw_fracs.append(0.0)
            continue

        f_single = (p * o - 1.0) / (o - 1.0)
        f_single = max(0.0, f_single) * fractional
        f_single = min(f_single, float(cap_per))
        raw_fracs.append(f_single)

    total = sum(raw_fracs)
    if total > float(cap_total) and total > 0:
        scale = float(cap_total) / total
        raw_fracs = [f * scale for f in raw_fracs]

    out: List[Dict[str, Any]] = []
    for r, f in zip(selected, raw_fracs):
        amt = int((bankroll * f) // min_bet_yen) * min_bet_yen
        row = dict(r)
        row["kelly_frac"] = float(f)
        row["bet_amount_yen"] = int(amt)
        out.append(row)

    return out
