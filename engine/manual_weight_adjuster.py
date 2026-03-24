from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def normalize_weight_dict(weights: Dict[str, float]) -> Dict[str, float]:
    """
    重み辞書を合計1になるように正規化する。
    負値は0扱い。
    """
    if not isinstance(weights, dict) or not weights:
        return {}

    cleaned: Dict[str, float] = {}
    total = 0.0

    for k, v in weights.items():
        try:
            x = float(v)
        except Exception:
            x = 0.0

        if not np.isfinite(x):
            x = 0.0

        x = max(0.0, x)
        cleaned[k] = x
        total += x

    if total <= 0:
        n = len(cleaned)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in cleaned.keys()}

    return {k: v / total for k, v in cleaned.items()}


def _safe_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    数値変換 + NaN/inf 除去
    """
    if X is None or X.empty:
        return pd.DataFrame()

    out = X.copy()

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out = out.replace([np.inf, -np.inf], 0.0)
    return out


def _group_sign(col: str) -> float:
    """
    特徴量ごとの有利方向を決める。
    ST/展示タイムは小さい方が有利なので -1。
    その他は大きい方が有利として +1。
    """
    c = str(col).lower()

    if "st" in c:
        return -1.0
    if "exhibit" in c:
        return -1.0

    return 1.0


def _minmax_scale_1d(arr: np.ndarray) -> np.ndarray:
    """
    0-1スケーリング
    """
    if arr.size == 0:
        return arr.astype(float)

    arr = arr.astype(float)
    mn = float(np.min(arr))
    mx = float(np.max(arr))

    if not np.isfinite(mn) or not np.isfinite(mx):
        return np.zeros_like(arr, dtype=float)

    if mx <= mn:
        return np.zeros_like(arr, dtype=float)

    return (arr - mn) / (mx - mn)


def _softmax(arr: np.ndarray) -> np.ndarray:
    """
    数値安定版 softmax
    """
    if arr.size == 0:
        return arr.astype(float)

    x = arr.astype(float)

    max_x = float(np.max(x))
    ex = np.exp(x - max_x)
    s = float(np.sum(ex))

    if s <= 0 or not np.isfinite(s):
        n = len(x)
        if n == 0:
            return np.array([], dtype=float)
        return np.ones(n, dtype=float) / n

    return ex / s


def build_manual_probability(
    X: pd.DataFrame,
    feature_groups: Dict[str, str],
    weights: Dict[str, float],
) -> np.ndarray:
    """
    手動重みから候補ごとの手動確率を作る。

    Parameters
    ----------
    X : pd.DataFrame
        120行（想定）の特徴量行列
    feature_groups : Dict[str, str]
        各列 -> グループ名 の対応
    weights : Dict[str, float]
        グループごとの手動重み

    Returns
    -------
    np.ndarray
        各行に対応する manual probability（合計1）
    """
    Xn = _safe_numeric_df(X)
    if Xn.empty:
        return np.array([], dtype=float)

    norm_weights = normalize_weight_dict(weights)
    if not norm_weights:
        n = len(Xn)
        return np.ones(n, dtype=float) / max(n, 1)

    n_rows = len(Xn)
    total_score = np.zeros(n_rows, dtype=float)

    # 各列を 0-1 正規化して、所属グループの重みで足し込む
    for col in Xn.columns:
        group = feature_groups.get(col, "other")
        w = float(norm_weights.get(group, 0.0))

        if w <= 0:
            continue

        arr = Xn[col].to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        sign = _group_sign(col)
        arr = arr * sign

        scaled = _minmax_scale_1d(arr)
        total_score += w * scaled

    # manual score -> probability
    manual_prob = _softmax(total_score)

    # 念のため再正規化
    s = float(np.sum(manual_prob))
    if s > 0 and np.isfinite(s):
        manual_prob = manual_prob / s
    else:
        manual_prob = np.ones(n_rows, dtype=float) / max(n_rows, 1)

    return manual_prob


def blend_scores(
    base_prob: np.ndarray,
    manual_prob: np.ndarray,
    blend_ratio: float,
) -> np.ndarray:
    """
    学習モデル確率と手動補正確率をブレンドする。
    final = base*(1-r) + manual*r
    """
    base = np.asarray(base_prob, dtype=float)
    manual = np.asarray(manual_prob, dtype=float)

    if base.size == 0 and manual.size == 0:
        return np.array([], dtype=float)

    if base.size == 0:
        out = manual.copy()
        s = float(np.sum(out))
        return out / s if s > 0 else out

    if manual.size == 0:
        out = base.copy()
        s = float(np.sum(out))
        return out / s if s > 0 else out

    if base.shape != manual.shape:
        raise ValueError(f"shape mismatch: base={base.shape}, manual={manual.shape}")

    r = float(blend_ratio)
    if not np.isfinite(r):
        r = 0.0
    r = max(0.0, min(1.0, r))

    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    manual = np.nan_to_num(manual, nan=0.0, posinf=0.0, neginf=0.0)

    out = (1.0 - r) * base + r * manual

    s = float(np.sum(out))
    if s <= 0 or not np.isfinite(s):
        n = len(out)
        if n == 0:
            return np.array([], dtype=float)
        return np.ones(n, dtype=float) / n

    return out / s
