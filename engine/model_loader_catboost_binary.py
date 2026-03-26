from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from engine.feature_builder_current import (
    add_feature_block,
    normalize_venue,
    sanitize_x,
)


class BinaryCatBoostVenueModel:
    def __init__(
        self,
        model_dir: str = "data/models",
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.model_dir = Path(model_dir)

        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise RuntimeError("catboost が入っていません。 `pip install catboost`") from e

        self._CatBoostClassifier = CatBoostClassifier
        self.models: Dict[str, Any] = {}
        self.metas: Dict[str, Dict[str, Any]] = {}

        # 会場別ベストモデル
        self.venue_suffix_map: Dict[str, str] = {
            "丸亀": "with_racer_no",
            "児島": "without_racer_no",
            "戸田": "with_racer_no",
        }

        for venue, suffix in self.venue_suffix_map.items():
            model_path = self.model_dir / f"trifecta_binary_catboost_{venue}_{suffix}.cbm"
            meta_path = self.model_dir / f"trifecta_binary_catboost_{venue}_{suffix}_meta.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file: {model_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing meta file: {meta_path}")

            model = self._CatBoostClassifier()
            model.load_model(str(model_path))

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.models[venue] = model
            self.metas[venue] = meta

        if self.debug:
            print("===== BinaryCatBoostVenueModel INIT =====")
            for venue in self.venue_suffix_map:
                meta = self.metas[venue]
                print(
                    venue,
                    "suffix=",
                    self.venue_suffix_map[venue],
                    "feature_count=",
                    len(meta.get("feature_cols", [])),
                )

    def _resolve_venue(self, venue: str) -> str:
        v = normalize_venue(venue)
        if v not in self.models:
            raise ValueError(f"Unsupported venue: {venue}")
        return v

    def _prepare_df(self, df120: pd.DataFrame, venue: str) -> pd.DataFrame:
        work = df120.copy()

        if "venue" not in work.columns:
            work["venue"] = venue
        if "combo" not in work.columns:
            raise ValueError("combo column missing in df120")

        work = add_feature_block(work)
        return work

    def predict_proba(self, df120: pd.DataFrame, venue: str) -> Dict[str, float]:
        venue_name = self._resolve_venue(venue)
        model = self.models[venue_name]
        meta = self.metas[venue_name]
        feature_cols: List[str] = list(meta.get("feature_cols", []))
        if not feature_cols:
            raise RuntimeError(f"feature_cols missing in meta for venue={venue_name}")

        work = self._prepare_df(df120, venue_name)

        missing_cols = [c for c in feature_cols if c not in work.columns]
        for c in missing_cols:
            work[c] = 0.0

        x = sanitize_x(work, feature_cols)
        prob = model.predict_proba(x)[:, 1]

        combos = work["combo"].astype(str).tolist()
        prob_map = {combo: float(p) for combo, p in zip(combos, prob)}

        total = sum(prob_map.values())
        if total > 0:
            prob_map = {k: v / total for k, v in prob_map.items()}

        if self.debug:
            print("\n===== BinaryCatBoostVenueModel DEBUG =====")
            print("venue:", venue_name)
            print("rows:", len(work))
            print("missing feature cols:", len(missing_cols))
            if missing_cols[:20]:
                print("sample missing:", missing_cols[:20])

            top10 = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print("top10:")
            for combo, p in top10:
                print(combo, f"{p:.6f}")

        return prob_map
