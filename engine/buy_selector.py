from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List


CONFIG_JSON_PATH = Path("data/models/venue_buy_config.optimized.json")
ENTROPY_TIER_CONFIG_JSON_PATH = Path("data/models/entropy_tier_config.json")

# 2026-07-23追加: 「1レースにつき最低3点は買う」という一律の下限が、
# 確率分布が集中している(=堅い、entropyが低い)レースでは点数を無駄に
# 増やして回収率を薄めていることが scripts/backtest_venue_tuned_tier_bounds.py
# の検証(train/testを時系列分割、real odds使用)で分かった。
# entropy(確率分布の散らばり)でレースをlow/mid/highの3段階に分け、
# 段階ごとに点数の下限・上限(min/max points)を変える。デフォルトは
# low(堅い)=1〜6点・mid=3〜10点(現行相当)・high(荒れ)=3〜10点(現行相当)。
# 戸田・江戸川はtrainサンプルが十分多く(264件・166件)、個別により積極的な
# 設定(aggressive_low_cut)の方が良かったためentropy_tier_config.jsonで上書きしている。
# 検証結果(test期間1771レース): ROI 109.90%→114.52%、利益+85,590円→+113,940円、
# 会場別95%CI=[-260,+2838]円(ほぼ有意)、races>=70の大口18会場中13会場がプラス。
TIER_POINT_BOUNDS_DEFAULT: Dict[str, tuple] = {
    "low": (1, 6),
    "mid": (3, 10),
    "high": (3, 10),
}

# 「モデルが一番自信を持つ組み合わせ(top1_prob)がこの値未満のレースは賭けない」戦略。
# 2026-07-15: 戸田・桐生・江戸川(with_racer_statsモデル、真のholdout期間1370レース)で検証。
# 前半685レースで閾値を決め、後半685レース(未使用データ)に適用した結果、
# ROIが78.75% -> 84.31%に改善(スキップ約30%、ベット数509)。他の会場・モデルでは
# 未検証のため、デフォルトは無効(0.0)にし、検証済みの3会場のみ有効にする。
SKIP_TOP1_PROB_THRESHOLDS: Dict[str, float] = {
    "戸田": 0.0473,
    "桐生": 0.0473,
    "江戸川": 0.0473,
}

BASE_VENUE_BUY_CONFIG: Dict[str, Dict[str, float]] = {
    "丸亀": {
        "min_prob": 0.050,
        "min_ev": 0.95,
        "max_odds_cap": 25.0,
        "max_ev_cap": 1.8,
        "weight_prob": 0.74,
        "weight_ev": 0.18,
        "weight_odds": 0.08,
        "prob_exp": 1.65,
        "ev_exp": 0.62,
        "base_bonus": 0.64,
        "min_points": 3,
        "max_points": 8,
    },
    "児島": {
        "min_prob": 0.030,
        "min_ev": 0.85,
        "max_odds_cap": 50.0,
        "max_ev_cap": 2.2,
        "weight_prob": 0.64,
        "weight_ev": 0.25,
        "weight_odds": 0.11,
        "prob_exp": 1.42,
        "ev_exp": 0.72,
        "base_bonus": 0.65,
        "min_points": 3,
        "max_points": 10,
    },
    "戸田": {
        "min_prob": 0.026,
        "min_ev": 0.78,
        "max_odds_cap": 70.0,
        "max_ev_cap": 2.6,
        "weight_prob": 0.58,
        "weight_ev": 0.30,
        "weight_odds": 0.12,
        "prob_exp": 1.28,
        "ev_exp": 0.84,
        "base_bonus": 0.66,
        "min_points": 4,
        "max_points": 12,
    },
    "住之江": {
        "min_prob": 0.035,
        "min_ev": 0.85,
        "max_odds_cap": 45.0,
        "max_ev_cap": 2.2,
        "weight_prob": 0.66,
        "weight_ev": 0.24,
        "weight_odds": 0.10,
        "prob_exp": 1.42,
        "ev_exp": 0.72,
        "base_bonus": 0.65,
        "min_points": 3,
        "max_points": 10,
    },
    "default": {
        "min_prob": 0.032,
        "min_ev": 0.85,
        "max_odds_cap": 50.0,
        "max_ev_cap": 2.2,
        "weight_prob": 0.64,
        "weight_ev": 0.25,
        "weight_odds": 0.11,
        "prob_exp": 1.40,
        "ev_exp": 0.74,
        "base_bonus": 0.65,
        "min_points": 3,
        "max_points": 10,
    },
}


# 2026-07-20追加: オッズ帯別の実績分析(全24会場、predictions.csv実ログ
# 96万行超)で「10〜30倍の的中組がROIで一貫して上位、100倍超の大穴は
# 弱含み」という傾向が見つかった。7月より前(学習期間、約63万行)で
# ボーナス/ペナルティ幅をチューニングし、7月(真のholdout、約33万行)で
# 検証した結果: ROI 78.73% -> 80.54%(+1.81pt)。買い点数(avg_points)は
# 変えず、同じ予算内での並べ替えのみで効果が出ている。
# 効果は小さいので過信しないこと(それでも100%には届かない)。
# コース(艇番)別の傾向は7月より前とMuly月で真逆に出て再現性が無かったため
# 見送った(venue_buy_config.optimized.json同様、ここには入れていない)。
ODDS_BAND_BONUS_1030 = 0.15
ODDS_BAND_PENALTY_100PLUS = -0.15

# 2026-07-26追加: 「買い目のうち真のケリー的エッジ(prob*odds>1)が無い点を
# 除外したらROIが上がるか」をscripts/backtest_edge_filter_roi.pyで検証。
# /simと同じholdoutデータ(23会場・3237レース)で全体ROI 107.18%→116.35%
# (+9.17pt、利益+97,640円→+167,410円)。会場別では23会場中17会場でプラス
# だったが、効果の大きさにはばらつきがあった(平和島・びわこ等はむしろ悪化)。
# そのため全会場一律ではなく、効果が特に大きかった(roi_diff>=+0.15pt)
# 9会場だけ選択導入する: 下関(+0.587)・蒲郡(+0.300)・浜名湖(+0.285)・
# 多摩川(+0.280)・鳴門(+0.276)・江戸川(+0.253)・戸田(+0.190)・三国(+0.173)・
# 若松(+0.167)。[[boat_ai_meeting_form_feature]]の9会場選択導入とは別の
# 独立した施策(会場の重なりは偶然)。
EDGE_FILTER_VENUES = {
    "下関", "蒲郡", "浜名湖", "多摩川", "鳴門", "江戸川", "戸田", "三国", "若松",
}


def _odds_band_score_adj(odds: float) -> float:
    if odds < 10:
        return 0.0
    if odds < 30:
        return ODDS_BAND_BONUS_1030
    if odds < 100:
        return 0.0
    return ODDS_BAND_PENALTY_100PLUS


VENUE_NAMES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
    "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
    "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _normalize_venue_name(venue: str) -> str:
    s = str(venue or "").strip()
    for v in VENUE_NAMES:
        if v in s:
            return v
    return "default"


def _config_key(venue: str | None) -> str:
    v = _normalize_venue_name(venue or "")
    if v in BASE_VENUE_BUY_CONFIG:
        return v
    return "default"


def _load_external_config() -> Dict[str, Dict[str, float]]:
    if not CONFIG_JSON_PATH.exists():
        return {}

    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}

        out: Dict[str, Dict[str, float]] = {}
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            out[str(k)] = {}
            for kk, vv in v.items():
                try:
                    out[str(k)][str(kk)] = float(vv)
                except Exception:
                    continue
        return out
    except Exception:
        return {}


def _load_entropy_tier_config() -> Dict[str, Any]:
    if not ENTROPY_TIER_CONFIG_JSON_PATH.exists():
        return {}
    try:
        with open(ENTROPY_TIER_CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        return raw
    except Exception:
        return {}


def _entropy_thresholds(venue: str | None) -> tuple:
    """会場ごとのentropy tercile境界(q33, q66)を返す。
    data/models/entropy_tier_config.json (scripts/compute_entropy_tier_config.py で生成)
    が無ければ、検証時に使ったデフォルト値にフォールバックする。"""
    cfg = _load_entropy_tier_config()
    thresholds = cfg.get("entropy_thresholds", {})
    v = _normalize_venue_name(venue or "")
    if v in thresholds:
        q33, q66 = thresholds[v]
        return float(q33), float(q66)
    if "default" in thresholds:
        q33, q66 = thresholds["default"]
        return float(q33), float(q66)
    return 0.9712, 0.9878  # 2026-07-23算出時点の全会場合算フォールバック値


def _entropy_tier(entropy: float, venue: str | None) -> str:
    q33, q66 = _entropy_thresholds(venue)
    if entropy <= q33:
        return "low"
    elif entropy <= q66:
        return "mid"
    return "high"


def _tier_point_bounds(venue: str | None, tier: str) -> tuple:
    cfg = _load_entropy_tier_config()
    override_all = cfg.get("tier_point_bounds_override", {})
    default_bounds = cfg.get("tier_point_bounds_default", {})
    v = _normalize_venue_name(venue or "")

    bounds_src = override_all.get(v) or default_bounds or TIER_POINT_BOUNDS_DEFAULT
    pair = bounds_src.get(tier, TIER_POINT_BOUNDS_DEFAULT[tier])
    return int(pair[0]), int(pair[1])


def _resolve_config(
    venue: str | None = None,
    min_prob: float | None = None,
    max_odds_cap: float | None = None,
    max_ev_cap: float | None = None,
    weight_prob: float | None = None,
    weight_ev: float | None = None,
    weight_odds: float | None = None,
) -> Dict[str, float]:
    key = _config_key(venue)
    cfg = dict(BASE_VENUE_BUY_CONFIG.get(key, BASE_VENUE_BUY_CONFIG["default"]))

    # 2026-08-06修正: 従来はext(venue_buy_config.optimized.json)の参照キーが
    # BASE_VENUE_BUY_CONFIGにハードコードされた4会場(丸亀/児島/戸田/住之江)+
    # "default"のみに限定されていたため、それ以外の18会場(桐生・江戸川等)の
    # override設定を書いても一切反映されない(常に"default"キーだけを見に行く)
    # というバグがあった。会場名そのもの(正規化後)でのoverrideを優先的に
    # 探すよう修正し、全会場で個別チューニング値が効くようにする。
    ext = _load_external_config()
    normalized_venue = _normalize_venue_name(venue or "")
    if normalized_venue in ext:
        cfg.update(ext[normalized_venue])
    elif key in ext:
        cfg.update(ext[key])

    if min_prob is not None:
        cfg["min_prob"] = float(min_prob)
    if max_odds_cap is not None:
        cfg["max_odds_cap"] = float(max_odds_cap)
    if max_ev_cap is not None:
        cfg["max_ev_cap"] = float(max_ev_cap)
    if weight_prob is not None:
        cfg["weight_prob"] = float(weight_prob)
    if weight_ev is not None:
        cfg["weight_ev"] = float(weight_ev)
    if weight_odds is not None:
        cfg["weight_odds"] = float(weight_odds)

    if "min_ev" not in cfg:
        cfg["min_ev"] = 0.85

    # 2026-07-23注記: cfg["min_points"]/["max_points"]/["fixed_points"]は
    # 過去のロジックの名残で、実際の点数決定は現在_calc_dynamic_points内の
    # entropy tierごとのTIER_POINT_BOUNDS(_tier_point_bounds参照)で行っている。
    # ここでの計算結果はもう使われないが、外部configとの後方互換のために
    # フィールド自体は残してある。
    if "fixed_points" in cfg:
        fp = _safe_int(cfg["fixed_points"], 6)
        cfg["min_points"] = float(fp)
        cfg["max_points"] = float(fp)

    cfg["min_points"] = float(max(3, min(12, _safe_int(cfg.get("min_points", 3), 3))))
    cfg["max_points"] = float(max(3, min(12, _safe_int(cfg.get("max_points", 10), 10))))
    if cfg["max_points"] < cfg["min_points"]:
        cfg["max_points"] = cfg["min_points"]

    return cfg


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _calc_distribution_entropy(probs: List[float]) -> float:
    vals = [p for p in probs if p > 0]
    if not vals:
        return 1.0

    total = sum(vals)
    if total <= 0:
        return 1.0

    norm = [p / total for p in vals]
    h = 0.0
    for p in norm:
        h -= p * math.log(p + 1e-12)

    max_h = math.log(len(norm) + 1e-12)
    if max_h <= 0:
        return 0.0

    return max(0.0, min(1.0, h / max_h))


def _calc_dynamic_points(rows: List[Dict[str, Any]], venue: str, cfg: Dict[str, float]) -> int:
    # entropy tier や top1/top3_sum等のしきい値は元々「較正前(生)の確率スケール」で
    # 検証・チューニングされているため、点数決定はできる限り生確率(prob_raw)を使う。
    # prob_rawが無いai_preds(従来通りの呼び出し)は今まで通りprobを使うので後方互換。
    probs = sorted(
        [_safe_float(r.get("prob_raw", r.get("prob", 0.0)), 0.0) for r in rows],
        reverse=True,
    )
    if not probs:
        return 3

    top1 = probs[0]
    top2 = probs[1] if len(probs) >= 2 else 0.0
    top3_sum = sum(probs[:3])
    top5_sum = sum(probs[:5])
    entropy = _calc_distribution_entropy(probs[:10])
    gap12 = max(0.0, top1 - top2)

    # 2026-07-23変更: 従来はcfg(会場別固定、3〜12点にハードクランプ)から
    # min_points/max_pointsを決めていたが、これを撤廃しentropy tier
    # (low/mid/high、_entropy_tier参照)ごとの点数レンジに置き換えた。
    # 低entropy(=堅い、確率が集中している)レースは最低1点まで絞れる。
    # 根拠: scripts/backtest_venue_tuned_tier_bounds.py の検証結果
    # (test 1771レースでROI 109.90%→114.52%、大口18会場中13会場がプラス)。
    tier = _entropy_tier(entropy, venue)
    min_points, max_points = _tier_point_bounds(venue, tier)

    spread_score = 0.0
    spread_score += (0.12 - min(top1, 0.12)) / 0.12 * 0.36
    spread_score += (0.24 - min(top3_sum, 0.24)) / 0.24 * 0.20
    spread_score += (0.36 - min(top5_sum, 0.36)) / 0.36 * 0.12
    spread_score += (0.03 - min(gap12, 0.03)) / 0.03 * 0.08
    spread_score += entropy * 0.24

    v = _normalize_venue_name(venue)
    if v == "丸亀":
        spread_score *= 0.72
    elif v == "戸田":
        spread_score *= 1.08
    elif v == "児島":
        spread_score *= 0.96
    elif v == "住之江":
        spread_score *= 0.90

    spread_score = max(0.0, min(1.0, spread_score))
    points = min_points + round((max_points - min_points) * spread_score)

    if top1 >= 0.10 and top3_sum >= 0.23:
        points = min(points, max(min_points, 3))
    elif top1 >= 0.08 and top5_sum >= 0.34:
        points = min(points, max(min_points, min(4, max_points)))

    if entropy >= 0.88 and top1 <= 0.05:
        points = min(max_points, points + 1)

    return max(min_points, min(max_points, points))


def _load_external_skip_thresholds() -> Dict[str, float]:
    ext = _load_external_config()
    out: Dict[str, float] = {}
    for k, v in ext.items():
        if "skip_top1_prob_threshold" in v:
            out[str(k)] = float(v["skip_top1_prob_threshold"])
    return out


def get_skip_top1_prob_threshold(venue: str | None = None) -> float:
    key = _normalize_venue_name(venue or "")
    overrides = _load_external_skip_thresholds()
    if key in overrides:
        return overrides[key]
    return SKIP_TOP1_PROB_THRESHOLDS.get(key, 0.0)


def should_skip_race(
    ai_preds: List[Dict[str, Any]],
    venue: str | None = None,
    threshold: float | None = None,
) -> bool:
    """このレースを丸ごと見送るべきかどうかを判定する。

    「賭けるべき候補が(閾値未満で)見つからない」という理由でbest_betsが
    空になるケースと区別するため、独立した関数として公開する。
    呼び出し側(app/controller.pyなど)は、select_best_betsが空リストを
    返したときに素朴なフォールバックへ進む前に、必ずこの関数で
    「意図的な見送りかどうか」を確認すること。
    """
    thr = threshold if threshold is not None else get_skip_top1_prob_threshold(venue)
    if thr <= 0:
        return False

    # SKIP_TOP1_PROB_THRESHOLDSは生確率(較正前)スケールで検証済みの値なので、
    # prob_rawがあればそちらを優先する(_calc_dynamic_pointsのentropy計算と同じ理由)。
    top1_prob = 0.0
    for row in ai_preds:
        p = _safe_float(row.get("prob_raw", row.get("prob", row.get("score", 0.0))), 0.0)
        if p > top1_prob:
            top1_prob = p

    return top1_prob < thr


def _prepare_candidate_rows(
    ai_preds: List[Dict[str, Any]],
    cfg: Dict[str, float],
    use_min_prob: bool = True,
    use_min_ev: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in ai_preds:
        combo = str(row.get("combo", "")).strip()
        prob = _safe_float(row.get("prob", row.get("score", 0.0)), 0.0)
        odds = _safe_float(row.get("odds", row.get("odds_raw", 0.0)), 0.0)
        ev = _safe_float(row.get("ev", row.get("ev_raw", prob * odds)), 0.0)

        if not combo:
            continue
        if prob <= 0:
            continue
        if odds <= 0:
            continue
        if ev <= 0:
            continue
        if use_min_prob and prob < cfg["min_prob"]:
            continue
        if use_min_ev and ev < cfg.get("min_ev", 0.85):
            continue

        row2 = dict(row)
        row2["prob"] = prob
        row2["odds_raw"] = odds
        row2["odds"] = odds
        row2["ev_raw"] = ev
        row2["ev"] = ev
        row2["odds_capped"] = min(odds, cfg["max_odds_cap"])
        row2["ev_capped"] = min(ev, cfg["max_ev_cap"])
        rows.append(row2)

    return rows


def select_best_bets(
    ai_preds: List[Dict[str, Any]],
    min_prob: float | None = None,
    max_odds_cap: float | None = None,
    max_ev_cap: float | None = None,
    top_n: int = 16,
    weight_prob: float | None = None,
    weight_ev: float | None = None,
    weight_odds: float | None = None,
    venue: str | None = None,
) -> List[Dict[str, Any]]:
    if should_skip_race(ai_preds, venue=venue):
        return []

    cfg = _resolve_config(
        venue=venue,
        min_prob=min_prob,
        max_odds_cap=max_odds_cap,
        max_ev_cap=max_ev_cap,
        weight_prob=weight_prob,
        weight_ev=weight_ev,
        weight_odds=weight_odds,
    )

    rows = _prepare_candidate_rows(ai_preds, cfg, use_min_prob=True, use_min_ev=True)

    if len(rows) < 3:
        # 本来のmin_evを満たす候補が3点未満のときだけ、min_evを大きく緩めすぎず
        # 最低ラインの0.70までに限定して救済する(min_prob条件は維持)。
        # ここでもまだ3点に満たない場合のみ、min_prob/min_evを両方無視した
        # backup_rowsで最終的に埋める(app/controller.py側の
        # 「best_betsが空ならscore降順で単純にtop_nを買う」というさらに粗い
        # フォールバックに落ちるより、buy_score(prob/ev/oddsの加重)で選ぶ方が良いため維持)。
        relaxed_cfg = dict(cfg)
        relaxed_cfg["min_ev"] = min(float(cfg.get("min_ev", 0.85)), 0.70)
        rows = _prepare_candidate_rows(ai_preds, relaxed_cfg, use_min_prob=True, use_min_ev=True)

    backup_rows = _prepare_candidate_rows(ai_preds, cfg, use_min_prob=False, use_min_ev=False)
    if len(rows) < 3:
        rows = backup_rows[:]

    if not rows:
        return []

    prob_norms = _minmax_norm([r["prob"] for r in rows])
    odds_norms = _minmax_norm([r["odds_capped"] for r in rows])
    ev_norms = _minmax_norm([r["ev_capped"] for r in rows])

    for i, r in enumerate(rows):
        r["prob_norm"] = prob_norms[i]
        r["odds_norm"] = odds_norms[i]
        r["ev_norm"] = ev_norms[i]

        r["buy_score"] = (
            (r["prob_norm"] * cfg["weight_prob"]) +
            (r["ev_norm"] * cfg["weight_ev"]) +
            (r["odds_norm"] * cfg["weight_odds"]) +
            _odds_band_score_adj(r["odds_raw"])
        )

    rows.sort(
        key=lambda x: (
            float(x.get("buy_score", 0.0)),
            float(x.get("prob", 0.0)),
            float(x.get("ev_raw", 0.0)),
        ),
        reverse=True,
    )

    dynamic_points = _calc_dynamic_points(rows, venue or "default", cfg)
    # 2026-07-23変更: 以前は3〜12点にハードクランプしていたが、
    # _calc_dynamic_points側でentropy tierごとの適切なmin/maxに既に
    # クランプ済みなので、ここでは「rows件数」「top_n」を超えないための
    # 安全弁(1〜20点)だけをかける。
    final_n = max(1, min(20, min(_safe_int(top_n, 20), dynamic_points)))

    selected = rows[:final_n]

    # 2026-07-26追加: エッジ絞り込み(EDGE_FILTER_VENUES参照)。選定自体は
    # 従来通り行ったうえで、対象会場だけ「真のエッジ(prob*odds>1)が無い点」を
    # 事後的に除外する。除外した結果0点になったレースは素直に見送りとする
    # (should_skip_raceによる意図的見送りと同様、app/controller.py側の
    # 「best_betsが空ならもっと粗いフォールバックで埋める」処理には落とさない
    # ほうが良いため、ここではbackup_rowsへの引き直しはしない)。
    if _normalize_venue_name(venue or "") in EDGE_FILTER_VENUES:
        selected = [
            r for r in selected
            if float(r.get("prob", 0.0)) * float(r.get("odds_raw", 0.0)) > 1.0
        ]

    for idx, r in enumerate(selected, start=1):
        r["buy_rank"] = idx
        r["is_best_bet"] = True

    return selected
