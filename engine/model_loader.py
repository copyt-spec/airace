# engine/model_loader.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


VENUE_CODE_MAP = {"丸亀": 15, "戸田": 2}

DEFAULT_MODEL_PATH = "data/models/trifecta120_model.joblib"
DEFAULT_META_PATH = "data/models/trifecta120_model_meta.json"

# =========================================================
# 一時的な手動バイアス補正
# check_top1_bias の pred/true ratio を見て、
# 出しすぎ combo を少しだけ抑える
# =========================================================
DEFAULT_MANUAL_BIAS_FACTORS: Dict[str, float] = {
    "1-2-3": 0.82,
    "1-3-2": 0.88,
    "1-2-4": 0.92,
    "1-3-4": 0.95,
    "1-2-5": 0.98,
    "1-4-3": 1.00,
    "1-3-5": 0.99,
    "1-4-2": 0.98,
    "1-2-6": 1.00,
    "1-3-6": 1.00,
    "1-5-2": 1.02,
    "1-5-4": 1.03,
    "3-1-5": 1.03,
    "1-6-2": 1.01,
}


def _finite_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("", np.nan)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    s = ez.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return ez / s


def _entropy(p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    s = float(np.sum(p))
    if s <= 0:
        return 0.0
    p = p / s
    return float(-np.sum(p * np.log(p)))


def _ent_ratio(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    k = int(p.size)
    if k <= 1:
        return 0.0
    ent = _entropy(p)
    ent_max = float(np.log(k))
    if ent_max <= 0:
        return 0.0
    return float(ent / ent_max)


class BoatRaceModel:
    """
    120行（combo候補行）を受けて 120combo の確率分布を返す

    重要:
    - 学習/推論で feature_names を固定
    - LightGBM / sklearn predict_proba 両対応
    - 後段で軽い bias correction を入れる
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        meta_path: str = DEFAULT_META_PATH,
        temperature: float = 1.0,
        output_tau: float = 1.10,
        out_min_prob: float = 1e-6,
        use_geometric_mean: bool = False,
        rescue_max: float = 0.85,
        rescue_mix_cap: float = 0.20,
        bias_alpha: float = 0.30,
        bias_clip_min: float = 0.85,
        bias_clip_max: float = 1.10,
        manual_bias_factors: Optional[Dict[str, float]] = None,
        debug: bool = False,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta not found: {meta_path}")

        self.model = joblib.load(model_path)

        self.temperature = float(temperature) if np.isfinite(float(temperature)) and float(temperature) > 0 else 1.0
        self.output_tau = float(output_tau) if np.isfinite(float(output_tau)) and float(output_tau) > 0 else 1.0
        self.out_min_prob = float(out_min_prob) if np.isfinite(float(out_min_prob)) and float(out_min_prob) > 0 else 1e-6
        self.use_geometric_mean = bool(use_geometric_mean)

        self.rescue_max = float(rescue_max) if np.isfinite(float(rescue_max)) else 0.85
        self.rescue_mix_cap = float(rescue_mix_cap) if np.isfinite(float(rescue_mix_cap)) else 0.20

        self.bias_alpha = float(bias_alpha) if np.isfinite(float(bias_alpha)) else 0.55
        self.bias_clip_min = float(bias_clip_min) if np.isfinite(float(bias_clip_min)) else 0.60
        self.bias_clip_max = float(bias_clip_max) if np.isfinite(float(bias_clip_max)) else 1.20

        self.debug = bool(debug)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.feature_names: List[str] = meta.get("feature_names") or []
        self.classes_: List[str] = [str(c) for c in (meta.get("classes") or [])]

        if not self.feature_names:
            raise ValueError("meta.feature_names is missing")
        if not self.classes_:
            raise ValueError("meta.classes is missing")

        self.class_index: Dict[str, int] = {str(c): i for i, c in enumerate(self.classes_)}

        model_classes = self._get_model_classes()
        self._reorder_idx = self._build_reorder_index(model_classes, self.classes_)

        # 将来 meta から自動補正するための受け口
        self.class_true_counts: Dict[str, float] = {
            str(k): float(v)
            for k, v in (meta.get("class_true_counts") or {}).items()
        }
        self.class_pred_top1_counts: Dict[str, float] = {
            str(k): float(v)
            for k, v in (meta.get("class_pred_top1_counts") or {}).items()
        }

        # いまは manual bias を有効化
        self.manual_bias_factors = dict(DEFAULT_MANUAL_BIAS_FACTORS)
        if manual_bias_factors:
            self.manual_bias_factors.update(manual_bias_factors)

    # -----------------------------------------------------
    # classes alignment
    # -----------------------------------------------------
    def _get_model_classes(self) -> List[str]:
        est = self.model

        if hasattr(est, "steps") and getattr(est, "steps", None):
            try:
                est = est.steps[-1][1]
            except Exception:
                est = self.model

        if hasattr(est, "classes_"):
            try:
                return [str(c) for c in list(est.classes_)]
            except Exception:
                pass

        if hasattr(self.model, "classes_"):
            try:
                return [str(c) for c in list(self.model.classes_)]
            except Exception:
                pass

        return list(self.classes_)

    def _build_reorder_index(self, model_classes: List[str], meta_classes: List[str]) -> Optional[np.ndarray]:
        if not model_classes or not meta_classes:
            return None
        if len(model_classes) != len(meta_classes):
            return None
        pos = {c: i for i, c in enumerate(model_classes)}
        idx = []
        for c in meta_classes:
            if c not in pos:
                return None
            idx.append(pos[c])
        return np.asarray(idx, dtype=int)

    # -----------------------------------------------------
    # feature handling
    # -----------------------------------------------------
    def _coerce_types(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.replace("", np.nan)

        if "combo" in X.columns:
            X = X.drop(columns=["combo"])

        if "date" in X.columns:
            X["date"] = pd.to_numeric(X["date"], errors="coerce")

        if "venue" in X.columns:
            def map_venue(v: Any) -> float:
                s = str(v).strip()
                if s.isdigit():
                    return float(int(s))
                return float(VENUE_CODE_MAP.get(s, 0))
            X["venue"] = X["venue"].map(map_venue)

        if "race_no" in X.columns:
            X["race_no"] = pd.to_numeric(X["race_no"], errors="coerce")

        for c in list(X.columns):
            if not is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

        X = _finite_df(X)

        try:
            X = X.astype(float)
        except Exception:
            for c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(float)

        return X

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for c in self.feature_names:
            if c not in X.columns:
                X[c] = 0.0

        X = X[self.feature_names]
        return X

    # -----------------------------------------------------
    # core predict
    # -----------------------------------------------------
    def _predict_proba_matrix(self, X: pd.DataFrame) -> np.ndarray:
        T = self.temperature if np.isfinite(self.temperature) and self.temperature > 0 else 1.0

        if hasattr(self.model, "predict_proba"):
            P = np.asarray(self.model.predict_proba(X), dtype=float)
        elif hasattr(self.model, "decision_function"):
            z = self.model.decision_function(X)
            z = np.asarray(z, dtype=float)
            if z.ndim == 1:
                z = np.vstack([-z, z]).T
            z = z / T
            P = _softmax(z)
        else:
            raise AttributeError("Model has neither predict_proba nor decision_function")

        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        if self._reorder_idx is not None and P.ndim == 2 and P.shape[1] == len(self.classes_):
            try:
                P = P[:, self._reorder_idx]
            except Exception:
                pass

        row_sum = P.sum(axis=1, keepdims=True)
        bad = (row_sum <= 0) | (~np.isfinite(row_sum))
        if np.any(bad):
            P[bad[:, 0], :] = 1.0 / float(P.shape[1])
            row_sum = P.sum(axis=1, keepdims=True)

        P = P / np.where(row_sum <= 0, 1.0, row_sum)
        return P

    # -----------------------------------------------------
    # aggregation
    # -----------------------------------------------------
    def _aggregate_class_distribution(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        eps = 1e-15
        P = np.clip(P, eps, 1.0)

        if self.use_geometric_mean:
            logm = np.mean(np.log(P), axis=0)
            class_p = np.exp(logm)
        else:
            class_p = np.mean(P, axis=0)

        class_p = np.nan_to_num(class_p, nan=0.0, posinf=0.0, neginf=0.0)
        class_p = np.maximum(class_p, self.out_min_prob)

        s = float(class_p.sum())
        class_p = class_p / (s if s > 0 else 1.0)

        # tau > 1 で少しフラット化
        tau = self.output_tau
        if np.isfinite(tau) and tau > 0 and abs(tau - 1.0) > 1e-12:
            class_p = np.power(np.clip(class_p, 1e-15, 1.0), 1.0 / tau)
            s2 = float(class_p.sum())
            class_p = class_p / (s2 if s2 > 0 else 1.0)

        # 尖りすぎ救済（置換ではなく軽いブレンド）
        mx = float(np.max(class_p))
        er = _ent_ratio(class_p)
        if (mx >= self.rescue_max) or (er <= 0.12):
            k = int(class_p.size)
            uniform = np.full(k, 1.0 / float(k), dtype=float)

            mix = min(self.rescue_mix_cap, 0.05 + max(0.0, mx - self.rescue_max))
            mix = float(np.clip(mix, 0.04, self.rescue_mix_cap))

            class_p = (1.0 - mix) * class_p + mix * uniform
            class_p = class_p / float(class_p.sum())

            if self.debug:
                print(
                    f"[DEBUG] class_p rescue blended: "
                    f"mix={mix:.3f} max={float(np.max(class_p)):.6f} ent_ratio={_ent_ratio(class_p):.3f}"
                )

        return class_p

    # -----------------------------------------------------
    # bias correction
    # -----------------------------------------------------
    def _apply_bias_correction(self, class_p: np.ndarray) -> np.ndarray:
        class_p = np.asarray(class_p, dtype=float).copy()

        # 1) meta 由来の自動補正（将来用）
        if self.class_true_counts and self.class_pred_top1_counts:
            total_true = float(sum(self.class_true_counts.values()))
            total_pred = float(sum(self.class_pred_top1_counts.values()))

            if total_true > 0 and total_pred > 0:
                factors = np.ones_like(class_p, dtype=float)

                for combo, idx in self.class_index.items():
                    true_cnt = float(self.class_true_counts.get(combo, 0.0))
                    pred_cnt = float(self.class_pred_top1_counts.get(combo, 0.0))

                    true_rate = true_cnt / total_true if total_true > 0 else 0.0
                    pred_rate = pred_cnt / total_pred if total_pred > 0 else 0.0

                    if true_rate > 0 and pred_rate > 0:
                        ratio = true_rate / pred_rate
                        factor = float(np.power(ratio, self.bias_alpha))
                        factor = float(np.clip(factor, self.bias_clip_min, self.bias_clip_max))
                        factors[idx] *= factor

                class_p *= factors

        # 2) いま効かせる手動補正
        if self.manual_bias_factors:
            for combo, factor in self.manual_bias_factors.items():
                idx = self.class_index.get(combo)
                if idx is None:
                    continue
                class_p[idx] *= float(factor)

        class_p = np.nan_to_num(class_p, nan=0.0, posinf=0.0, neginf=0.0)
        class_p = np.maximum(class_p, self.out_min_prob)
        s = float(class_p.sum())
        class_p = class_p / (s if s > 0 else 1.0)

        if self.debug:
            top = np.argsort(-class_p)[:10]
            print("[DEBUG] after bias correction TOP10:")
            for i in top:
                print(f"  {self.classes_[i]} {class_p[i]:.6f}")

        return class_p

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------
    def predict_proba(self, features_120: pd.DataFrame) -> Dict[str, float]:
        if not isinstance(features_120, pd.DataFrame):
            raise TypeError("features_120 must be DataFrame")
        if "combo" not in features_120.columns:
            raise ValueError("features_120 must contain 'combo' column")

        combos: List[str] = [str(x) for x in features_120["combo"].values]

        X = self._coerce_types(features_120)
        X = self._align_columns(X)

        if self.debug:
            print(f"[DEBUG] X aligned shape={X.shape}")
            if len(X.columns) >= 8:
                print("[DEBUG] first feature cols:", list(X.columns[:8]))

        P = self._predict_proba_matrix(X)
        class_p = self._aggregate_class_distribution(P)
        class_p = self._apply_bias_correction(class_p)

        out_vec = np.zeros(len(combos), dtype=float)
        for i, combo in enumerate(combos):
            j = self.class_index.get(combo, -1)
            out_vec[i] = float(class_p[j]) if 0 <= j < len(class_p) else 0.0

        out_vec = np.nan_to_num(out_vec, nan=0.0, posinf=0.0, neginf=0.0)
        out_vec = np.maximum(out_vec, self.out_min_prob)
        s = float(out_vec.sum())
        out_vec = out_vec / (s if s > 0 else 1.0)

        if self.debug:
            top_idx = np.argsort(-out_vec)[:10]
            print(f"[DEBUG] out_vec stats: min={float(np.min(out_vec)):.10f} max={float(np.max(out_vec)):.10f} ent={_entropy(out_vec):.6f}")
            print("[DEBUG] final TOP10:")
            for i in top_idx:
                print(f"  {combos[i]} {out_vec[i]:.6f}")

        return {combos[i]: float(out_vec[i]) for i in range(len(combos))}
