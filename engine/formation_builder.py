from __future__ import annotations

"""
2026-07-17追加: ユーザーから「フォーメーション形式で案出しできる?」と要望があった
ことに対応。競艇の「フォーメーション買い」とは、1着候補・2着候補・3着候補を
それぞれ絞り込み、その組み合わせ(デカルト積)をまとめて買う方式(舟券売り場の
端末でも「1着-2着-3着」の欄にそれぞれ複数艇を指定して入力できる)。

現行のbuy_selector.select_best_bets()は「120通りの中から個別にN点選ぶ」方式
だが、これをそのまま1着/2着/3着候補の集合に変換しようとしても、選ばれた点が
綺麗な直積(フォーメーション)になっているとは限らない。そこで、既に計算済みの
prob_map(120通りの予測確率。model.predict_proba()の出力で、新たな推論や
再学習は不要)を直接使い、以下のロジックでフォーメーションを組む:

1. 各着順(1着/2着/3着)ごとに、レーン1〜6の周辺確率(marginal probability;
   そのレーンがその着順になる確率の合計)を計算する。
2. 「1着候補を周辺確率上位k1レーン、2着候補を上位k2レーン、3着候補を上位k3レーン」
   というプレフィックス集合の組み合わせを総当たり(k1,k2,k3は最大6なので216通り)
   で試し、実際に生成される組み合わせ数が指定した点数予算(max_points)以内に
   収まるものの中から、実際の同時確率(prob_mapそのもの。周辺確率の近似ではなく
   正確な値)の合計が最大になる組を選ぶ。

これは「一番当たりやすい軸の絞り方」を機械的に探すグリッドサーチであり、
buy_selector.select_best_bets()を置き換えるものではなく、同じprob_mapから
別の見せ方(フォーメーション形式)を追加で提供するもの。
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple


def _finite(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN check
            return default
        return v
    except Exception:
        return default


def compute_position_marginals(prob_map: Dict[str, float]) -> Dict[str, Dict[int, float]]:
    """
    120通りのprob_mapから、レーンごとの1着/2着/3着になる周辺確率を計算する。
    """
    marg = {
        "first": {i: 0.0 for i in range(1, 7)},
        "second": {i: 0.0 for i in range(1, 7)},
        "third": {i: 0.0 for i in range(1, 7)},
    }

    for combo, prob in (prob_map or {}).items():
        parts = str(combo).split("-")
        if len(parts) != 3:
            continue
        try:
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue

        p = _finite(prob, 0.0)
        if a in marg["first"]:
            marg["first"][a] += p
        if b in marg["second"]:
            marg["second"][b] += p
        if c in marg["third"]:
            marg["third"][c] += p

    return marg


def _ranked_lanes(marginals: Dict[int, float]) -> List[int]:
    return [lane for lane, _ in sorted(marginals.items(), key=lambda kv: kv[1], reverse=True)]


def _formation_combos(first_set: List[int], second_set: List[int], third_set: List[int]) -> List[str]:
    combos = []
    for a, b, c in itertools.product(first_set, second_set, third_set):
        if a == b or b == c or a == c:
            continue
        combos.append(f"{a}-{b}-{c}")
    return combos


def build_formation(
    prob_map: Dict[str, float],
    odds_map: Optional[Dict[str, float]] = None,
    max_points: int = 8,
    min_points: int = 1,
) -> Dict[str, Any]:
    """
    prob_map(120通りの予測確率)から、指定した点数予算(max_points)以内で
    「実際にカバーできる確率(=モデルの推定的中率)」が最大になる
    1着/2着/3着のフォーメーションを探す。

    戻り値:
      {
        "first": [レーン番号のリスト, 周辺確率降順],
        "second": [...],
        "third": [...],
        "combos": ["1-2-3", ...],  実際に生成される組み合わせ
        "point_count": len(combos),
        "covered_prob": 合計確率(=モデル推定の的中率),
        "expected_ev": sum(prob*odds) (odds_mapがあれば),
        "invested_yen": point_count * 100,
      }
    prob_mapが空、または予算内に収まる組み合わせが一つも無い場合は
    空のフォーメーション({"combos": [], ...})を返す。
    """
    if not prob_map:
        return {"first": [], "second": [], "third": [], "combos": [], "point_count": 0,
                "covered_prob": 0.0, "expected_ev": 0.0, "invested_yen": 0}

    marginals = compute_position_marginals(prob_map)
    ranked = {
        "first": _ranked_lanes(marginals["first"]),
        "second": _ranked_lanes(marginals["second"]),
        "third": _ranked_lanes(marginals["third"]),
    }

    best: Optional[Dict[str, Any]] = None

    for k1 in range(1, 7):
        first_set = ranked["first"][:k1]
        for k2 in range(1, 7):
            second_set = ranked["second"][:k2]
            for k3 in range(1, 7):
                third_set = ranked["third"][:k3]

                combos = _formation_combos(first_set, second_set, third_set)
                n = len(combos)
                if n == 0 or n > max_points or n < min_points:
                    continue

                covered_prob = sum(_finite(prob_map.get(c, 0.0)) for c in combos)

                if best is None or covered_prob > best["covered_prob"]:
                    best = {
                        "first": first_set,
                        "second": second_set,
                        "third": third_set,
                        "combos": combos,
                        "point_count": n,
                        "covered_prob": covered_prob,
                    }

    if best is None:
        return {"first": [], "second": [], "third": [], "combos": [], "point_count": 0,
                "covered_prob": 0.0, "expected_ev": 0.0, "invested_yen": 0}

    expected_ev = 0.0
    if odds_map:
        for c in best["combos"]:
            expected_ev += _finite(prob_map.get(c, 0.0)) * _finite(odds_map.get(c, 0.0))

    best["expected_ev"] = expected_ev
    best["invested_yen"] = best["point_count"] * 100
    return best


# 競艇の実際のレーン色(1号艇=白、2号艇=黒、3号艇=赤、4号艇=青、5号艇=黄、6号艇=緑)。
# app/templates/index.html の .lane-1〜.lane-6 と同じ配色に揃えている。
LANE_COLORS: Dict[int, Tuple[str, str]] = {
    1: ("#ffffff", "#111111"),
    2: ("#1a1a1a", "#ffffff"),
    3: ("#e53935", "#ffffff"),
    4: ("#1565c0", "#ffffff"),
    5: ("#f9a825", "#111111"),
    6: ("#2e7d32", "#ffffff"),
}


def build_formation_svg(formation: Dict[str, Any], width: int = 480) -> str:
    """
    2026-07-17追加: フォーメーション(1着/2着/3着候補+実際の組み合わせ)を
    ツリー状(3列のノード+実際に成立する組み合わせだけを結ぶ線)のSVGとして
    描画する。app/templates/index.html に埋め込んで使う想定なので、
    文字色などは同テンプレートの:rootで定義済みのCSS変数(--muted等)を使う。

    無効な組み合わせ(同じレーンが重複する等)は最初からformation["combos"]に
    含まれていないので、線は必ず実在する買い目だけを結ぶ。
    """
    first = formation.get("first", [])
    second = formation.get("second", [])
    third = formation.get("third", [])
    combos = formation.get("combos", [])

    if not first or not second or not third or not combos:
        return ""

    node_r = 16
    row_gap = 42
    margin_top = 34
    col_x = {"first": 60.0, "second": width / 2.0, "third": float(width - 60)}

    def positions(lanes: List[int]) -> Dict[int, float]:
        return {lane: margin_top + i * row_gap for i, lane in enumerate(lanes)}

    y_first = positions(first)
    y_second = positions(second)
    y_third = positions(third)

    height = margin_top + max(len(first), len(second), len(third)) * row_gap + 24

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" style="width:100%;height:auto;">',
        f'<title>フォーメーション買い({len(combos)}点)</title>',
        f'<text x="{col_x["first"]:.0f}" y="16" text-anchor="middle" font-size="11" '
        f'fill="var(--muted)">1着</text>',
        f'<text x="{col_x["second"]:.0f}" y="16" text-anchor="middle" font-size="11" '
        f'fill="var(--muted)">2着</text>',
        f'<text x="{col_x["third"]:.0f}" y="16" text-anchor="middle" font-size="11" '
        f'fill="var(--muted)">3着</text>',
    ]

    # 接続線: 実際に成立する組み合わせ(combos)だけを結ぶ直線(太め)。
    # 2026-07-17: ユーザー指摘によりベジェ曲線から直線・太線に変更。
    for combo in combos:
        parts_combo = combo.split("-")
        if len(parts_combo) != 3:
            continue
        try:
            a, b, c = int(parts_combo[0]), int(parts_combo[1]), int(parts_combo[2])
        except ValueError:
            continue
        if a not in y_first or b not in y_second or c not in y_third:
            continue

        x1, y1 = col_x["first"], y_first[a]
        x2, y2 = col_x["second"], y_second[b]
        x3, y3 = col_x["third"], y_third[c]

        parts.append(
            f'<line x1="{x1 + node_r:.0f}" y1="{y1:.0f}" x2="{x2 - node_r:.0f}" y2="{y2:.0f}" '
            f'stroke="rgba(255,255,255,0.35)" stroke-width="3" stroke-linecap="round"/>'
        )
        parts.append(
            f'<line x1="{x2 + node_r:.0f}" y1="{y2:.0f}" x2="{x3 - node_r:.0f}" y2="{y3:.0f}" '
            f'stroke="rgba(255,255,255,0.35)" stroke-width="3" stroke-linecap="round"/>'
        )

    def draw_nodes(lanes: List[int], x: float, ys: Dict[int, float]) -> List[str]:
        out = []
        for lane in lanes:
            y = ys[lane]
            fill, text_color = LANE_COLORS.get(lane, ("#888888", "#ffffff"))
            out.append(
                f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{node_r}" fill="{fill}" '
                f'stroke="rgba(255,255,255,0.25)" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{x:.0f}" y="{y + 4:.0f}" text-anchor="middle" font-size="13" '
                f'font-weight="700" fill="{text_color}">{lane}</text>'
            )
        return out

    parts.extend(draw_nodes(first, col_x["first"], y_first))
    parts.extend(draw_nodes(second, col_x["second"], y_second))
    parts.extend(draw_nodes(third, col_x["third"], y_third))

    parts.append("</svg>")
    return "".join(parts)


def build_formation_options(
    prob_map: Dict[str, float],
    odds_map: Optional[Dict[str, float]] = None,
    budgets: Tuple[int, ...] = (4, 8, 12),
) -> List[Dict[str, Any]]:
    """
    複数の点数予算(例: 本命4点・バランス8点・広め12点)でフォーメーションを
    それぞれ計算し、ユーザーが選べるように一覧で返す。同じ結果になる予算は
    重複除去する。
    """
    out: List[Dict[str, Any]] = []
    seen_combo_sets = set()

    for budget in sorted(set(budgets)):
        f = build_formation(prob_map, odds_map=odds_map, max_points=budget)
        if not f["combos"]:
            continue
        key = tuple(sorted(f["combos"]))
        if key in seen_combo_sets:
            continue
        seen_combo_sets.add(key)
        f = dict(f)
        f["budget_label"] = f"{budget}点まで"
        try:
            f["svg"] = build_formation_svg(f)
        except Exception:
            f["svg"] = ""
        out.append(f)

    return out
