from __future__ import annotations

"""
2026-07-18追加: 「スタートはこの艇がGA前に出て、第一ターンではこの艇がまくって
この艇が内から差して…」というシーン別の展開予想ページのための、ヒューリスティック
(統計指標からの発見的推定)エンジン。

【前提と限界】
学習データ(raw_txt由来)には「誰が誰を第一ターンで抜いた」といった展開そのものの
記録は無い。したがってここでの推定は、以下の“結果から逆算した”作り方をしている:

1. 3連単の予測確率(probabilities, 120通り)から最も確率の高い組み合わせ
   (=モデルが最終的に予想する1-2-3着)を求める。
2. その最終予想に矛盾しないように、スタート→第一ターン→ホームストレッチ→ゴール
   という4シーンで「誰がどこで順位を上げるか」を組み立てる。
3. 各シーンの説明文は、進入コース・スタートタイミング(ST)・展示タイムなど
   既存の指標から「逃げ/差し/まくり」のどれに近いパターンかを判定して生成する。

つまり「本当にその通りのレース展開になる」という保証は無く、あくまで
「モデルの最終予想に至りそうな、もっともらしい一つの筋書き」を可視化するもの。
この点はscenario.html側でも明記すること。
"""

from typing import Any, Dict, List, Optional, Tuple

from engine.formation_builder import LANE_COLORS, compute_position_marginals


def _safe_float(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if v is None or v == "":
            return default
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except Exception:
        return default


def _resolve_course(lane: int, bi_row: Dict[str, Any]) -> int:
    c = _safe_float((bi_row or {}).get("course"), None)
    if c is not None and 1 <= int(c) <= 6:
        return int(c)
    return lane


def _resolve_st(bi_row: Dict[str, Any], entry: Dict[str, Any]) -> float:
    for src, key in [(bi_row, "st"), (entry, "start_timing"), (entry, "avg_st")]:
        if src:
            v = _safe_float(src.get(key), None)
            if v is not None and v > 0:
                return v
    return 0.17


def _resolve_exhibit(bi_row: Dict[str, Any], entry: Dict[str, Any]) -> float:
    for src, key in [(bi_row, "exhibit_time"), (entry, "exhibit")]:
        if src:
            v = _safe_float(src.get(key), None)
            if v is not None and v > 0:
                return v
    return 6.80


def build_lane_context(
    entries: List[Dict[str, Any]],
    beforeinfo: Optional[Dict[Any, Any]],
) -> Dict[int, Dict[str, Any]]:
    bi_map: Dict[int, Dict[str, Any]] = {}
    for k, v in (beforeinfo or {}).items():
        try:
            bi_map[int(k)] = v
        except (TypeError, ValueError):
            continue

    ctx: Dict[int, Dict[str, Any]] = {}
    for e in entries or []:
        try:
            lane = int(e.get("lane"))
        except (TypeError, ValueError):
            continue
        if lane < 1 or lane > 6:
            continue

        bi_row = bi_map.get(lane, {})
        name = (
            e.get("racer_name") or e.get("name") or e.get("racer")
            or f"{lane}号艇"
        )

        ctx[lane] = {
            "lane": lane,
            "course": _resolve_course(lane, bi_row),
            "st": _resolve_st(bi_row, e),
            "exhibit": _resolve_exhibit(bi_row, e),
            "name": str(name),
            "win_rate": _safe_float(e.get("win_rate"), 0.0),
            "place_rate": _safe_float(e.get("quinella_rate") or e.get("place_rate"), 0.0),
        }

    # 出走表が不完全な場合でも描画は止めない(デフォルト値で埋める)
    for lane in range(1, 7):
        ctx.setdefault(lane, {
            "lane": lane, "course": lane, "st": 0.17, "exhibit": 6.80,
            "name": f"{lane}号艇", "win_rate": 0.0, "place_rate": 0.0,
        })

    return ctx


def _order_by_course(ctx: Dict[int, Dict[str, Any]]) -> List[int]:
    return [lane for lane, _ in sorted(ctx.items(), key=lambda kv: kv[1]["course"])]


def _top_combo(probabilities: Optional[Dict[str, float]]) -> Tuple[str, float]:
    if not probabilities:
        return "", 0.0
    combo, prob = max(probabilities.items(), key=lambda kv: kv[1])
    return combo, float(prob)


def _parse_combo(combo: str) -> List[int]:
    parts = str(combo).split("-")
    if len(parts) != 3:
        return []
    try:
        return [int(p) for p in parts]
    except ValueError:
        return []


def _final_order(ctx: Dict[int, Dict[str, Any]], probabilities: Optional[Dict[str, float]]) -> List[int]:
    combo, _ = _top_combo(probabilities)
    top3 = _parse_combo(combo)
    top3 = [l for l in top3 if l in ctx]

    remaining = [l for l in ctx.keys() if l not in top3]
    if probabilities:
        marg = compute_position_marginals(probabilities)

        def _score(lane: int) -> float:
            return (
                marg["first"].get(lane, 0.0)
                + marg["second"].get(lane, 0.0)
                + marg["third"].get(lane, 0.0)
            )

        remaining.sort(key=_score, reverse=True)
    else:
        remaining.sort()

    order = top3 + remaining
    if len(order) != 6:
        # フォールバック: コース順
        return _order_by_course(ctx)
    return order


def _move_to_position(order: List[int], lane: int, position: int) -> List[int]:
    if lane not in order:
        return order
    new_order = [l for l in order if l != lane]
    position = max(0, min(position, len(new_order)))
    new_order.insert(position, lane)
    return new_order


def _pattern_label(course: int) -> str:
    if course == 1:
        return "逃げ"
    if course == 2:
        return "差し"
    if course in (3, 4):
        return "まくり"
    return "大外まくり"


def _fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _build_narrative(
    ctx: Dict[int, Dict[str, Any]],
    final_order: List[int],
    pattern: str,
    combo: str,
    top_prob: float,
) -> List[Dict[str, str]]:
    l1, l2, l3 = final_order[0], final_order[1], final_order[2]
    c1, c2, c3 = ctx[l1], ctx[l2], ctx[l3]

    narrative: List[Dict[str, str]] = []

    # シーン1: スタート
    fastest_st_lane = min(ctx.keys(), key=lambda l: ctx[l]["st"])
    course1_st = ctx.get(1, c1)["st"]
    if fastest_st_lane != 1 and ctx[fastest_st_lane]["st"] <= course1_st - 0.03:
        st_text = (
            f"{fastest_st_lane}号艇 {ctx[fastest_st_lane]['name']} のスタートタイミング"
            f"({ctx[fastest_st_lane]['st']:.2f})が最も早く、隊列に変化が生まれそうです。"
        )
    else:
        st_text = "各艇ともコース取り通りにスタートし、大きな出遅れは見込みにくい展開です。"
    narrative.append({"label": "スタート想定", "text": st_text})

    # シーン2: 第一ターン(パターン判定)
    if pattern == "逃げ":
        text = f"1コースの {c1['name']} がスタートの主導権を握り、そのまま第一ターンを先頭で回る「逃げ」の形が濃厚です。"
    elif pattern == "差し":
        text = f"2コースの {c1['name']} が第一ターンで内側を差して先頭に立つ展開を予想します。"
    elif pattern == "まくり":
        text = (
            f"{c1['course']}コースの {c1['name']} が展示タイム{c1['exhibit']:.2f}秒の伸びを活かし、"
            f"第一ターンで外から「まくり」を狙う展開です。"
        )
    else:
        text = f"大外{c1['course']}コースの {c1['name']} が思い切った大外まくりを仕掛ける、荒れ含みの展開を予想します。"
    narrative.append({"label": "第一ターン", "text": text})

    # シーン3: ホームストレッチ〜最終ターン(2着争い)
    text3 = (
        f"2着争いは {l2}号艇 {c2['name']}({c2['course']}コース)が"
        f"ホームストレッチ〜最終ターンで抜け出す形を予想します。"
    )
    narrative.append({"label": "ホームストレッチ〜最終ターン", "text": text3})

    # シーン4: ゴール
    combo_label = combo or f"{l1}-{l2}-{l3}"
    text4 = (
        f"予想着順は {l1}号艇-{l2}号艇-{l3}号艇({combo_label})。"
        f"モデル推定確率は{_fmt_pct(top_prob)}です。"
    )
    narrative.append({"label": "ゴール(予想)", "text": text4})

    return narrative


def build_scenario_svg(checkpoints: List[Dict[str, Any]], width: int = 640) -> str:
    """
    フォーメーション図(engine.formation_builder.build_formation_svg)と同じ
    トーン(直線・太め・レーン配色)で、シーンごとの着順を「ポジションチャート」
    (列=シーン、行=着順、同じ艇番を線で繋ぐ)として描く。
    """
    if not checkpoints:
        return ""

    n_cols = len(checkpoints)
    node_r = 15
    row_gap = 40
    margin_top = 30
    col_gap = (width - 110) / max(1, n_cols - 1)
    col_x = [55 + i * col_gap for i in range(n_cols)]
    height = margin_top + 6 * row_gap + 20

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" style="width:100%;height:auto;">'
    ]

    for i, cp in enumerate(checkpoints):
        parts.append(
            f'<text x="{col_x[i]:.0f}" y="16" text-anchor="middle" font-size="11" '
            f'fill="var(--muted)">{cp["label"]}</text>'
        )

    col_positions: List[Dict[int, float]] = []
    for cp in checkpoints:
        order = cp["order"]
        col_positions.append({lane: margin_top + idx * row_gap for idx, lane in enumerate(order)})

    all_lanes = set()
    for cp in checkpoints:
        all_lanes.update(cp["order"])

    for lane in sorted(all_lanes):
        for i in range(n_cols - 1):
            if lane not in col_positions[i] or lane not in col_positions[i + 1]:
                continue
            x1, y1 = col_x[i], col_positions[i][lane]
            x2, y2 = col_x[i + 1], col_positions[i + 1][lane]
            parts.append(
                f'<line x1="{x1 + node_r:.0f}" y1="{y1:.0f}" x2="{x2 - node_r:.0f}" y2="{y2:.0f}" '
                f'stroke="rgba(255,255,255,0.32)" stroke-width="3" stroke-linecap="round"/>'
            )

    for i, cp in enumerate(checkpoints):
        for lane in cp["order"]:
            y = col_positions[i][lane]
            fill, text_color = LANE_COLORS.get(lane, ("#888888", "#ffffff"))
            parts.append(
                f'<circle cx="{col_x[i]:.0f}" cy="{y:.0f}" r="{node_r}" fill="{fill}" '
                f'stroke="rgba(255,255,255,0.25)" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{col_x[i]:.0f}" y="{y + 4:.0f}" text-anchor="middle" font-size="12" '
                f'font-weight="700" fill="{text_color}">{lane}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def build_race_scenario(
    entries: List[Dict[str, Any]],
    beforeinfo: Optional[Dict[Any, Any]],
    probabilities: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    if not probabilities:
        return {"available": False}

    ctx = build_lane_context(entries, beforeinfo)
    final_order = _final_order(ctx, probabilities)
    if len(final_order) != 6:
        return {"available": False}

    combo, top_prob = _top_combo(probabilities)

    start_order = _order_by_course(ctx)
    turn1_order = _move_to_position(start_order, final_order[0], 0)
    backstretch_order = _move_to_position(turn1_order, final_order[1], 1)
    finish_order = final_order

    checkpoints = [
        {"key": "start", "label": "スタート想定", "order": start_order},
        {"key": "turn1", "label": "第一ターン", "order": turn1_order},
        {"key": "backstretch", "label": "ホームストレッチ〜最終ターン", "order": backstretch_order},
        {"key": "finish", "label": "ゴール(予想)", "order": finish_order},
    ]

    pattern = _pattern_label(ctx[final_order[0]]["course"])
    narrative = _build_narrative(ctx, final_order, pattern, combo, top_prob)

    try:
        svg = build_scenario_svg(checkpoints)
    except Exception:
        svg = ""

    return {
        "available": True,
        "checkpoints": checkpoints,
        "narrative": narrative,
        "pattern": pattern,
        "predicted_combo": combo,
        "predicted_prob": top_prob,
        "svg": svg,
        "lane_context": ctx,
    }
