from __future__ import annotations

"""
モデルが出す3連単の確率(softmax後のprob)は、オッズ帯によって系統的に
偏っている(favorite-longshot bias)。

data/logs/predictions.csv (2023, 2025年の全レース・全組み合わせのprob/odds)と
再構築したdata/logs/results.csv(正しい会場別の実際の結果)を突き合わせて集計すると、
オッズ帯ごとに「モデルの平均prob」と「実際の的中率」の比(calib_ratio)が
非常にきれいな単調減少カーブを描く:

    オッズ            calib_ratio (実際の的中率 / モデルprobの平均)
    0-5倍             3.74   (モデルは本命を過小評価)
    5-10倍            2.48
    10-20倍           1.85
    20-30倍           1.43
    30-50倍           1.19
    50-100倍          0.92   (ほぼ妥当)
    100-300倍         0.59   (モデルは大穴を過大評価)
    300倍以上          0.28   (モデルは大穴を約3.6倍も過大評価)

つまり、prob*oddsで期待値(EV)を計算するロジックは、大穴(高オッズ)の
組み合わせのEVを実態よりずっと高く見積もってしまい、そこに資金が
偏って回収率を下げる主因になっていた。

この係数は2023年+2025年の実データ(約60万行、レース約4,900走)から
算出したもので、2026年のホールドアウト検証でも同じ方向のROI改善効果を確認済み。
"""

from typing import List, Tuple

# (odds上限, calib_ratio) : オッズがこの上限未満の組み合わせに適用する係数
# 2023-03〜2025-12 のデータで学習(train)。odds<=0の行は対象外。
CALIBRATION_TABLE: List[Tuple[float, float]] = [
    (5.0, 3.744489),
    (10.0, 2.481714),
    (20.0, 1.845086),
    (30.0, 1.428486),
    (50.0, 1.194651),
    (100.0, 0.915588),
    (300.0, 0.592397),
    (float("inf"), 0.275729),
]

# 極端なオッズでの補正のしすぎ(prob=0付近での比率不安定)を避けるため、
# 補正後probが原型から大きく離れすぎないようクリップする。
MIN_RATIO = 0.15
MAX_RATIO = 4.0


def get_calibration_ratio(odds: float) -> float:
    if odds is None or odds <= 0:
        return 1.0

    for upper, ratio in CALIBRATION_TABLE:
        if odds < upper:
            return max(MIN_RATIO, min(MAX_RATIO, ratio))

    return max(MIN_RATIO, min(MAX_RATIO, CALIBRATION_TABLE[-1][1]))


def calibrate_prob(prob: float, odds: float) -> float:
    """
    生のモデルprobを、オッズ帯別の実績ベースの係数で補正する。
    的中率は超えられないので 0.0〜1.0 にクリップする。
    """
    if prob is None or prob <= 0:
        return 0.0

    ratio = get_calibration_ratio(odds)
    calibrated = prob * ratio
    return max(0.0, min(1.0, calibrated))
