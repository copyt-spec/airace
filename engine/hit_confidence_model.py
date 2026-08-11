"""的中信頼度(hit_confidence)の二次スコアリングモデル。

2026-08-08: 従来のcombined_prob(選ばれた買い目群のprob_raw合算、
_combined_prob_from_best_bets)単体よりも、avg_ev_capped(選ばれた買い目群の
平均EV=prob_raw×min(odds,50倍)の平均、_capped_avg_best_evと同じ計算式)を
併用した簡易ロジスティック回帰の方が「実際に当たるかどうか」をより正確に
予測できることが検証で分かった。

検証方法: 全期間chronological 70/30分割(古い70%をTRAIN、新しい30%を
HOLD、モーター成績修正後のwideholdスナップショット、n=5997)でTRAINに
ロジスティック回帰をフィットし、HOLDでout-of-sample評価。
  - combined_prob単体            : Brier=0.1767  AUC=0.6093
  - combined_prob + avg_ev_capped: Brier=0.1762  AUC=0.6187 (5-fold CVでも
    0.61→0.63前後で再現、n_bets/entropy/prob_gap/top1_probなど他の特徴量を
    追加してもこれ以上の改善は無かった)

avg_ev_capped項の直感的な意味: combined_probが同じでも、高オッズ(万馬券)
寄りの買い目に偏っている(avg_ev_capptが高い)場合は実際の的中率が下がる
傾向がある(favorite-longshot bias、[[boat_ai_rank_confidence_roi_inversion]]と
同根の現象)。この項を加えることで、combined_probだけでは拾えない
「その確率の中身が堅い(本命寄り)か、荒い(穴狙い)か」を補正している。

sklearnがサンドボックス環境ではネットワーク制限でインストールできない
ため、勾配降下法によるロジスティック回帰を手動実装し、係数をJSON保存した
(data/models/hit_confidence_model.json)。詳細はメモリ
boat_ai_hit_confidence_score_model参照。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

MODEL_JSON_PATH = Path("data/models/hit_confidence_model.json")

# モデルファイルが読み込めない場合のフォールバック閾値
# (combined_prob単体スケールの旧閾値。スコアのスケールは違うが
# fail-safeとして極端な誤動作だけは避ける)
_FALLBACK_THRESHOLDS = {"C": 0.182, "B": 0.238, "A": 0.284, "A+": 0.325}

_cache: Optional[dict] = None
_cache_mtime: Optional[float] = None


def _load_model() -> Optional[dict]:
    global _cache, _cache_mtime
    try:
        if not MODEL_JSON_PATH.exists():
            return None
        mtime = MODEL_JSON_PATH.stat().st_mtime
        if _cache is not None and _cache_mtime == mtime:
            return _cache
        with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not raw.get("features") or not raw.get("coef") or not raw.get("mean") or not raw.get("std"):
            return None
        _cache = raw
        _cache_mtime = mtime
        return raw
    except Exception:
        return None


def score(combined_prob_raw: float, avg_ev_capped: float) -> float:
    """的中信頼度スコア(0〜1、較正済み確率として解釈可能)を返す。

    モデルファイルが読み込めない場合はcombined_prob_rawをそのまま返す
    (fail-safe。この場合はtier_thresholds()側もフォールバック値を返すので
    整合は取れる)。
    """
    model = _load_model()
    if model is None:
        return float(combined_prob_raw)
    try:
        vals = {"combined_prob_raw": float(combined_prob_raw), "avg_ev_capped": float(avg_ev_capped)}
        z = float(model["coef"]["intercept"])
        for feat in model["features"]:
            mean = float(model["mean"][feat])
            std = float(model["std"][feat]) or 1.0
            x = (vals[feat] - mean) / std
            z += float(model["coef"][feat]) * x
        return 1.0 / (1.0 + math.exp(-z))
    except Exception:
        return float(combined_prob_raw)


def tier_thresholds() -> Dict[str, float]:
    model = _load_model()
    if model is None or not model.get("tier_thresholds"):
        return dict(_FALLBACK_THRESHOLDS)
    return dict(model["tier_thresholds"])
