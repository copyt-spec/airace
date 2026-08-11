"""確率較正(isotonic calibration)ユーティリティ。

2026-08-06: モデルが出す確率(softmax後)の傾きがなだらかすぎる
(=本来ほぼ起こらない組み合わせにもそこそこの確率が乗ってしまう)ことが
判明したため、data/models/prob_calibration_curve.json に保存した
較正曲線(PAVA/isotonic regressionで作成、fit期間: 〜2026-05-31)を使って
モデル出力確率を較正する。

較正曲線は data/logs/prediction_results_merged.csv の実績(is_hit)を
使って手動でPAVA(pool adjacent violators)を実装しフィットしたもの
(scikit-learnはサンドボックス環境でネットワーク制限のためインストール
不可だった)。詳細はメモリ boat_ai_prob_calibration_* を参照。

重要: この較正はbuy_selectorの「どの組み合わせを買うか」の判定にのみ
使う。RANK/EV/的中信頼度など他の箇所は較正前の生確率をそのまま使い続ける
(較正確率をRANK側に流すには別途RANK閾値の再検証が必要なため、今回は
意図的にbuy_selectorのみに限定して導入する)。

2026-08-06追記: 当初は7月(TRAIN 7/1-20, HOLD 7/21-31)だけで全会場に
較正+閾値チューニングを導入したが、直近1年(widehold snapshot全期間)で
検証し直したところ7月特有のパターンへの過学習だったことが判明
(例: 戸田は7月検証+44pt改善に見えたが年間では-8〜-20pt悪化)。
そこで会場ごとにchronological 70/30分割(古い70%をTRAIN、新しい30%を
HOLD)で再検証する方式に変更した。詳細はメモリ
boat_ai_prob_calibration_fullyear_revalidationを参照。

2026-08-08追記: さらにその直後、motor_point_in_time_stats.csvが
2026-07-21〜08-05の間3週間以上古いまま(実データ最大日付7/14)だった
バグ([[boat_ai_motor_stats_stale_bug]])が発覚・修正され、widehold
snapshotをユーザーのローカル環境で再生成(2026-08-08、修正済みの
モーター成績データで再学習)。この新しいsnapshotで会場別70/30検証を
やり直したところ、モーター成績が古かった時の検証結果と大きく変わる
会場が続出した(例: 若松は較正+42.8pt改善→修正後-43.3pt悪化に逆転)。
最終的に会場別方針は下記NON_CALIBRATED_VENUESの通り(15会場で較正有効、
9会場で無効)に更新。詳細はメモリ
boat_ai_prob_calibration_motorfix_revalidationを参照。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

CALIBRATION_JSON_PATH = Path("data/models/prob_calibration_curve.json")

# モーター成績修正後のwidehold(2026-08-08)でchronological 70/30検証をやり直し、
# 較正を入れると実績(生確率のまま)を下回ることが確認された会場。
# これらはcalibrate_prob_mapを素通りさせる。
NON_CALIBRATED_VENUES = {
    "戸田", "江戸川", "下関", "蒲郡", "宮島", "若松", "三国", "大村", "唐津",
}

_cache: dict | None = None
_cache_mtime: float | None = None


def _load_curve() -> tuple[list, list] | None:
    global _cache, _cache_mtime
    try:
        if not CALIBRATION_JSON_PATH.exists():
            return None
        mtime = CALIBRATION_JSON_PATH.stat().st_mtime
        if _cache is not None and _cache_mtime == mtime:
            return _cache["cal_x"], _cache["cal_y"]
        with open(CALIBRATION_JSON_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cal_x = [float(v) for v in raw.get("cal_x", [])]
        cal_y = [float(v) for v in raw.get("cal_y", [])]
        if not cal_x or not cal_y or len(cal_x) != len(cal_y):
            return None
        _cache = {"cal_x": cal_x, "cal_y": cal_y}
        _cache_mtime = mtime
        return cal_x, cal_y
    except Exception:
        return None


def _interp(x: float, xs: list, ys: list) -> float:
    """numpy.interpの単純な純Python実装(xsは昇順ソート済み前提)。"""
    n = len(xs)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    # 二分探索
    lo, hi = 0, n - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    if x1 <= x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def calibrate_prob_map(prob_map: Dict[str, float], venue: str | None = None) -> Dict[str, float]:
    """1レース分のcombo->prob辞書を較正し、合計が1になるよう再正規化する。

    較正曲線が読み込めない場合、またはvenueがNON_CALIBRATED_VENUESに
    含まれる場合は元のprob_mapをそのまま返す(fail-safe/会場別オプトアウト)。
    """
    if venue is not None:
        v = str(venue).strip()
        for name in NON_CALIBRATED_VENUES:
            if name in v:
                return dict(prob_map)

    curve = _load_curve()
    if curve is None or not prob_map:
        return dict(prob_map)

    cal_x, cal_y = curve
    calibrated = {
        combo: _interp(float(p), cal_x, cal_y)
        for combo, p in prob_map.items()
    }
    total = sum(calibrated.values())
    if total <= 0:
        return dict(prob_map)

    return {combo: v / total for combo, v in calibrated.items()}
