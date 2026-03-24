from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

from engine.manual_weight_adjuster import (
    build_manual_probability,
    blend_scores,
    normalize_weight_dict,
)


class BoatRaceModel:
    def __init__(
        self,
        model_path: str,
        meta_path: str,
        debug: bool = False,
    ) -> None:
        self.debug = debug

        model_file = Path(model_path)
        meta_file = Path(meta_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Missing model file: {model_file}")
        if not meta_file.exists():
            raise FileNotFoundError(f"Missing meta file: {meta_file}")

        self.model = joblib.load(model_file)

        with open(meta_file, "r", encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)

        self.feature_cols: List[str] = list(self.meta.get("feature_cols", []))
        if not self.feature_cols:
            raise RuntimeError("feature_cols missing in meta")

        self.feature_groups: Dict[str, str] = dict(self.meta.get("feature_groups", {}))

        # 共通 manual weights
        self.manual_weights: Dict[str, Any] = dict(self.meta.get("manual_weights", {}))

        # 会場別 manual weights
        self.venue_manual_weights: Dict[str, Any] = dict(self.meta.get("venue_manual_weights", {}))

        labels_path = Path(str(meta_file).replace("_meta.json", "_labels.json"))
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels: List[str] = json.load(f)

        self.label_to_class_id: Dict[str, int] = {label: i for i, label in enumerate(self.labels)}

        if len(self.labels) != 120 and self.debug:
            print(f"[WARN] labels count is {len(self.labels)} (expected 120)")

        if self.debug:
            print("===== MODEL LOADER INIT =====")
            print("feature cols:", len(self.feature_cols))
            print("feature groups:", len(self.feature_groups))
            print("manual_weights exists:", bool(self.manual_weights))
            print("venue_manual_weights exists:", bool(self.venue_manual_weights))

    def _prepare_x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        for c in self.feature_cols:
            if c not in x.columns:
                x[c] = 0.0

        x = x[self.feature_cols].copy()

        for c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

        x = x.replace([np.inf, -np.inf], 0.0)
        return x

    def _validate_input(self, df120: pd.DataFrame) -> None:
        if df120.empty:
            raise ValueError("input df is empty")

        if "combo" not in df120.columns:
            raise ValueError("combo column missing")

        combo_count = df120["combo"].astype(str).nunique()
        if combo_count != len(df120) and self.debug:
            print(f"[WARN] combo unique count={combo_count}, rows={len(df120)}")

    def _build_base_prob(self, df120: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        combos = df120["combo"].astype(str).tolist()
        proba = self.model.predict_proba(X)
        classes = [int(c) for c in self.model.classes_]

        if self.debug:
            print("===== MODEL LOADER DEBUG =====")
            print("rows:", len(df120))
            print("feature cols:", len(self.feature_cols))
            print("labels:", len(self.labels))
            print("model classes:", len(classes))

        base_prob = np.zeros(len(df120), dtype=float)
        class_pos_map = {cls: pos for pos, cls in enumerate(classes)}

        for row_idx, combo in enumerate(combos):
            class_id = self.label_to_class_id.get(combo)
            if class_id is None:
                base_prob[row_idx] = 0.0
                continue

            class_pos = class_pos_map.get(class_id)
            if class_pos is None:
                base_prob[row_idx] = 0.0
                continue

            base_prob[row_idx] = float(proba[row_idx, class_pos])

        return base_prob

    def _normalize_venue_name(self, venue: str) -> str:
        v = str(venue or "").strip()

        if not v:
            return ""

        v2 = v.replace("BOAT RACE", "").replace("ボートレース", "").strip()

        if "丸亀" in v or "丸亀" in v2 or v in {"15", "015"}:
            return "丸亀"
        if "児島" in v or "児島" in v2 or v in {"16", "016"}:
            return "児島"
        if "戸田" in v or "戸田" in v2 or v in {"2", "02", "002"}:
            return "戸田"

        return v

    def _resolve_manual_config_for_venue(self, venue: str) -> Dict[str, Any]:
        """
        優先順:
        1. venue_manual_weights[venue]
        2. manual_weights 内の venue_overrides[venue] （旧形式互換）
        3. manual_weights の共通設定
        """
        base_blend_ratio = 0.0
        base_weights: Dict[str, float] = {}

        if self.manual_weights:
            base_blend_ratio = float(self.manual_weights.get("blend_ratio", 0.0))
            base_weights = normalize_weight_dict(self.manual_weights.get("weights", {}))

        venue_name = self._normalize_venue_name(venue)

        final_blend_ratio = base_blend_ratio
        final_weights = dict(base_weights)

        # 新形式: meta["venue_manual_weights"]
        venue_cfg = {}
        if self.venue_manual_weights and venue_name:
            venue_cfg = self.venue_manual_weights.get(venue_name, {}) or {}

        # 旧形式互換: meta["manual_weights"]["venue_overrides"]
        if not venue_cfg and self.manual_weights:
            venue_overrides = self.manual_weights.get("venue_overrides", {}) or {}
            venue_cfg = venue_overrides.get(venue_name, {}) or {}

        if venue_cfg:
            final_blend_ratio = float(venue_cfg.get("blend_ratio", final_blend_ratio))

            override_weights = venue_cfg.get("weights", {}) or {}
            if override_weights:
                final_weights.update({k: float(v) for k, v in override_weights.items()})
                final_weights = normalize_weight_dict(final_weights)

        return {
            "blend_ratio": final_blend_ratio,
            "weights": final_weights,
            "venue_name": venue_name,
        }

    def predict_proba(self, df120: pd.DataFrame) -> Dict[str, float]:
        self._validate_input(df120)

        X = self._prepare_x(df120)
        combos = df120["combo"].astype(str).tolist()

        if X.empty:
            return {}

        base_prob = self._build_base_prob(df120, X)

        venue = ""
        if "venue" in df120.columns and len(df120) > 0:
            venue = str(df120.iloc[0].get("venue", "")).strip()

        cfg = self._resolve_manual_config_for_venue(venue)

        manual_prob = np.zeros(len(X), dtype=float)
        blend_ratio = float(cfg.get("blend_ratio", 0.0))
        weights = cfg.get("weights", {})

        if self.feature_groups and weights:
            manual_prob = build_manual_probability(
                X=X,
                feature_groups=self.feature_groups,
                weights=weights,
            )

        final_prob = blend_scores(
            base_prob=base_prob,
            manual_prob=manual_prob,
            blend_ratio=blend_ratio,
        )

        out = {combo: float(p) for combo, p in zip(combos, final_prob)}

        if self.debug:
            print("venue raw:", venue)
            print("venue normalized:", cfg.get("venue_name", venue))
            print("blend_ratio:", blend_ratio)
            print("weights:", weights)
            top = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print("===== TOP10 FROM MODEL LOADER =====")
            for c, p in top:
                print(c, f"{p:.6f}")

        return out
