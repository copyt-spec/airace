from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

from engine.model_loader import BoatRaceModel


class SafeEnsembleBoatRaceModel:
    """
    本番用アンサンブル推論
    - RandomForest（既存本線）
    - LightGBM_safe（リーク除去版）
    を会場別に重み付け平均する

    使い方:
        model = SafeEnsembleBoatRaceModel(
            rf_model_path="data/models/trifecta120_model_render.joblib",
            rf_meta_path="data/models/trifecta120_model_render_meta.json",
            ensemble_meta_path="data/models/ensemble_safe_meta.json",
            debug=False,
        )
        prob_map = model.predict_proba(df120)
    """

    def __init__(
        self,
        rf_model_path: str,
        rf_meta_path: str,
        ensemble_meta_path: str,
        debug: bool = False,
    ) -> None:
        self.debug = debug

        self.rf_model = BoatRaceModel(
            model_path=rf_model_path,
            meta_path=rf_meta_path,
            debug=debug,
        )

        ensemble_meta_file = Path(ensemble_meta_path)
        if not ensemble_meta_file.exists():
            raise FileNotFoundError(f"Missing ensemble meta: {ensemble_meta_file}")

        with open(ensemble_meta_file, "r", encoding="utf-8") as f:
            self.ensemble_meta = json.load(f)

        # デフォルト重み
        self.default_rf_weight = float(self.ensemble_meta.get("rf_weight", 0.60))
        self.default_lgbm_weight = float(self.ensemble_meta.get("lgbm_weight", 0.40))

        # venue別重み
        self.venue_weights: Dict[str, Dict[str, float]] = dict(
            self.ensemble_meta.get("venue_weights", {})
        )

        self.lgbm_models: Dict[str, Any] = {}
        self.lgbm_metas: Dict[str, Dict[str, Any]] = {}
        self.lgbm_labels: Dict[str, List[str]] = {}

        self._load_lgbm_safe_models()

        if self.debug:
            print("===== SAFE ENSEMBLE INIT =====")
            print("default_rf_weight:", self.default_rf_weight)
            print("default_lgbm_weight:", self.default_lgbm_weight)
            print("venue_weights:", self.venue_weights.keys())
            print("loaded_lgbm_safe_models:", self.lgbm_models.keys())

    def _normalize_venue(self, venue: str) -> str:
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

    def _load_lgbm_safe_models(self) -> None:
        model_dir = Path(self.ensemble_meta["model_dir"])

        for venue in ["丸亀", "児島", "戸田"]:
            model_path = model_dir / f"trifecta120_lgbm_{venue}_safe.joblib"
            meta_path = model_dir / f"trifecta120_lgbm_{venue}_safe_meta.json"
            labels_path = model_dir / f"trifecta120_lgbm_{venue}_safe_labels.json"

            if model_path.exists() and meta_path.exists() and labels_path.exists():
                self.lgbm_models[venue] = joblib.load(model_path)

                with open(meta_path, "r", encoding="utf-8") as f:
                    self.lgbm_metas[venue] = json.load(f)

                with open(labels_path, "r", encoding="utf-8") as f:
                    self.lgbm_labels[venue] = json.load(f)

    def _predict_lgbm_safe_prob_map(self, venue: str, df120: pd.DataFrame) -> Dict[str, float]:
        model = self.lgbm_models.get(venue)
        meta = self.lgbm_metas.get(venue)
        labels = self.lgbm_labels.get(venue)

        if model is None or meta is None or labels is None:
            return {}

        x = df120.copy()
        feature_cols = list(meta.get("feature_cols", []))
        if not feature_cols:
            return {}

        for c in feature_cols:
            if c not in x.columns:
                x[c] = 0.0

        x = x[feature_cols].copy()
        for c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
        x = x.replace([np.inf, -np.inf], 0.0)

        proba = model.predict_proba(x)
        classes = [int(c) for c in model.classes_]
        class_pos_map = {cls: pos for pos, cls in enumerate(classes)}
        label_to_class_id = {label: i for i, label in enumerate(labels)}

        combos = df120["combo"].astype(str).tolist()
        out: Dict[str, float] = {}

        for row_idx, combo in enumerate(combos):
            class_id = label_to_class_id.get(combo)
            if class_id is None:
                out[combo] = 0.0
                continue

            class_pos = class_pos_map.get(class_id)
            if class_pos is None:
                out[combo] = 0.0
                continue

            out[combo] = float(proba[row_idx, class_pos])

        s = sum(out.values())
        if s > 0:
            out = {k: v / s for k, v in out.items()}
        return out

    def _resolve_weights_for_venue(self, venue: str) -> Dict[str, float]:
        venue_name = self._normalize_venue(venue)
        venue_cfg = self.venue_weights.get(venue_name, {}) if venue_name else {}

        rf_weight = float(venue_cfg.get("rf_weight", self.default_rf_weight))
        lgbm_weight = float(venue_cfg.get("lgbm_weight", self.default_lgbm_weight))

        total = rf_weight + lgbm_weight
        if total <= 0:
            return {"rf_weight": 1.0, "lgbm_weight": 0.0}

        return {
            "rf_weight": rf_weight / total,
            "lgbm_weight": lgbm_weight / total,
        }

    def _merge_prob_maps(
        self,
        rf_prob: Dict[str, float],
        lgbm_prob: Dict[str, float],
        rf_weight: float,
        lgbm_weight: float,
    ) -> Dict[str, float]:
        keys = set(rf_prob.keys()) | set(lgbm_prob.keys())
        merged: Dict[str, float] = {}

        for k in keys:
            rv = float(rf_prob.get(k, 0.0))
            lv = float(lgbm_prob.get(k, 0.0))
            merged[k] = rf_weight * rv + lgbm_weight * lv

        s = sum(merged.values())
        if s > 0:
            merged = {k: v / s for k, v in merged.items()}
        return merged

    def predict_proba(self, df120: pd.DataFrame) -> Dict[str, float]:
        if df120 is None or df120.empty:
            return {}

        venue = ""
        if "venue" in df120.columns and len(df120) > 0:
            venue = str(df120.iloc[0].get("venue", "")).strip()

        venue_name = self._normalize_venue(venue)

        rf_prob = self.rf_model.predict_proba(df120)
        lgbm_prob = self._predict_lgbm_safe_prob_map(venue_name, df120)

        weights = self._resolve_weights_for_venue(venue_name)
        rf_weight = float(weights["rf_weight"])
        lgbm_weight = float(weights["lgbm_weight"])

        final_prob = self._merge_prob_maps(
            rf_prob=rf_prob,
            lgbm_prob=lgbm_prob,
            rf_weight=rf_weight,
            lgbm_weight=lgbm_weight,
        )

        if self.debug:
            print("===== SAFE ENSEMBLE DEBUG =====")
            print("venue raw:", venue)
            print("venue normalized:", venue_name)
            print("rf_weight:", rf_weight)
            print("lgbm_weight:", lgbm_weight)
            print("rf_prob exists:", bool(rf_prob))
            print("lgbm_prob exists:", bool(lgbm_prob))
            top = sorted(final_prob.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print("TOP10:")
            for c, p in top:
                print(c, f"{p:.6f}")

        return final_prob
