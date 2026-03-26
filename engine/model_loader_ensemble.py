from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from engine.model_loader import BoatRaceModel


class EnsembleBoatRaceModel:
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

        self.rf_weight = float(self.ensemble_meta.get("rf_weight", 0.20))
        self.lgbm_weight = float(self.ensemble_meta.get("lgbm_weight", 0.35))
        self.catboost_weight = float(self.ensemble_meta.get("catboost_weight", 0.30))
        self.xgb_weight = float(self.ensemble_meta.get("xgb_weight", 0.15))

        self.lgbm_models: Dict[str, Any] = {}
        self.lgbm_metas: Dict[str, Dict[str, Any]] = {}
        self.lgbm_labels: Dict[str, List[str]] = {}

        self._load_optional_models()

    def _normalize_venue(self, venue: str) -> str:
        v = str(venue or "").strip()
        if "丸亀" in v:
            return "丸亀"
        if "児島" in v:
            return "児島"
        if "戸田" in v:
            return "戸田"
        return v

    def _load_optional_models(self) -> None:
        model_dir = Path(self.ensemble_meta["model_dir"])

        for venue in ["丸亀", "児島", "戸田"]:
            lgbm_model_path = model_dir / f"trifecta120_lgbm_{venue}.joblib"
            lgbm_meta_path = model_dir / f"trifecta120_lgbm_{venue}_meta.json"
            lgbm_labels_path = model_dir / f"trifecta120_lgbm_{venue}_labels.json"

            if lgbm_model_path.exists() and lgbm_meta_path.exists() and lgbm_labels_path.exists():
                self.lgbm_models[venue] = joblib.load(lgbm_model_path)
                with open(lgbm_meta_path, "r", encoding="utf-8") as f:
                    self.lgbm_metas[venue] = json.load(f)
                with open(lgbm_labels_path, "r", encoding="utf-8") as f:
                    self.lgbm_labels[venue] = json.load(f)

    def _predict_lgbm_prob_map(self, venue: str, df120: pd.DataFrame) -> Dict[str, float]:
        model = self.lgbm_models.get(venue)
        meta = self.lgbm_metas.get(venue)
        labels = self.lgbm_labels.get(venue)

        if model is None or meta is None or labels is None:
            return {}

        x = df120.copy()
        feature_cols = meta["feature_cols"]

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

    def _weighted_merge(self, maps: List[tuple[float, Dict[str, float]]]) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        total_w = 0.0

        for w, mp in maps:
            if w <= 0 or not mp:
                continue
            total_w += w
            for k, v in mp.items():
                merged[k] = merged.get(k, 0.0) + w * float(v)

        if total_w <= 0:
            return {}

        s = sum(merged.values())
        if s > 0:
            merged = {k: v / s for k, v in merged.items()}
        return merged

    def predict_proba(self, df120: pd.DataFrame) -> Dict[str, float]:
        venue = ""
        if "venue" in df120.columns and len(df120) > 0:
            venue = self._normalize_venue(str(df120.iloc[0].get("venue", "")))

        rf_prob = self.rf_model.predict_proba(df120)
        lgbm_prob = self._predict_lgbm_prob_map(venue, df120)

        # CatBoost / XGBoost はまだ未ロード。後で追加できるように空枠
        cat_prob: Dict[str, float] = {}
        xgb_prob: Dict[str, float] = {}

        final_prob = self._weighted_merge([
            (self.rf_weight, rf_prob),
            (self.lgbm_weight, lgbm_prob),
            (self.catboost_weight, cat_prob),
            (self.xgb_weight, xgb_prob),
        ])

        if self.debug:
            print("venue:", venue)
            print("rf_weight:", self.rf_weight)
            print("lgbm_weight:", self.lgbm_weight)
            print("catboost_weight:", self.catboost_weight)
            print("xgb_weight:", self.xgb_weight)
            top = sorted(final_prob.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print("===== TOP10 ENSEMBLE =====")
            for c, p in top:
                print(c, f"{p:.6f}")

        return final_prob
