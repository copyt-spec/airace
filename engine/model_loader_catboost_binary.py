# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class BinaryCatBoostVenueModel:
    def __init__(self, model_dir: str = "data/models", debug: bool = False) -> None:
        from catboost import CatBoostClassifier

        self.model_dir = Path(model_dir)
        self.debug = debug
        self.CatBoostClassifier = CatBoostClassifier

        self.models: Dict[str, Any] = {}
        self.metas: Dict[str, Dict[str, Any]] = {}

        for venue in ["丸亀", "戸田", "児島"]:
            model_path = self.model_dir / f"trifecta_binary_catboost_{venue}_with_racer_no.cbm"
            meta_path = self.model_dir / f"trifecta_binary_catboost_{venue}_with_racer_no_meta.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file: {model_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing meta file: {meta_path}")

            model = self.CatBoostClassifier()
            model.load_model(str(model_path))

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.models[venue] = model
            self.metas[venue] = meta

    def _safe_float(self, v: Any, default: float = 0.0) -> float:
        try:
            if v is None or v == "":
                return default
            return float(v)
        except Exception:
            return default

    def _normalize_venue_name(self, venue_name: str) -> str:
        s = str(venue_name or "").strip()
        if "丸亀" in s:
            return "丸亀"
        if "戸田" in s:
            return "戸田"
        if "児島" in s:
            return "児島"
        return s

    def _softmax(self, xs: List[float], temperature: float = 1.0) -> List[float]:
        if not xs:
            return []

        t = max(float(temperature), 1e-6)
        scaled = [x / t for x in xs]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        s = sum(exps)

        if s <= 0:
            return [1.0 / len(xs)] * len(xs)

        return [e / s for e in exps]

    def _get_temperature_by_venue(self, venue_name: str) -> float:
        venue_name = self._normalize_venue_name(venue_name)

        if venue_name == "丸亀":
            return 0.88
        if venue_name == "戸田":
            return 0.92
        if venue_name == "児島":
            return 0.90
        return 0.90

    def _add_feature_block(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "combo" in out.columns:
            parts = out["combo"].astype(str).str.split("-", expand=True)
            if parts.shape[1] >= 3:
                out["combo_first_lane"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype("int16")
                out["combo_second_lane"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype("int16")
                out["combo_third_lane"] = pd.to_numeric(parts[2], errors="coerce").fillna(0).astype("int16")
            else:
                out["combo_first_lane"] = 0
                out["combo_second_lane"] = 0
                out["combo_third_lane"] = 0
        else:
            out["combo_first_lane"] = 0
            out["combo_second_lane"] = 0
            out["combo_third_lane"] = 0

        wind_map = {
            "無風": 0,
            "北": 1, "北東": 2, "東": 3, "南東": 4,
            "南": 5, "南西": 6, "西": 7, "北西": 8,
        }
        if "wind_dir" in out.columns:
            out["wind_dir_code"] = out["wind_dir"].astype(str).map(wind_map).fillna(0).astype("int16")
        else:
            out["wind_dir_code"] = 0

        weather_map = {
            "晴": 1, "晴れ": 1,
            "曇": 2, "曇り": 2,
            "雨": 3,
            "雪": 4,
        }
        if "weather" in out.columns:
            out["weather_code"] = out["weather"].astype(str).map(weather_map).fillna(0).astype("int16")
        else:
            out["weather_code"] = 0

        for lane in range(1, 7):
            st_col = f"lane{lane}_st"
            ex_col = f"lane{lane}_exhibit"
            course_col = f"lane{lane}_course"
            motor_col = f"lane{lane}_motor"
            boat_col = f"lane{lane}_boat"
            racer_col = f"lane{lane}_racer_no"

            for c in [st_col, ex_col, course_col, motor_col, boat_col, racer_col]:
                if c not in out.columns:
                    out[c] = 0

            out[st_col] = pd.to_numeric(out[st_col], errors="coerce").fillna(0).astype("float32")
            out[ex_col] = pd.to_numeric(out[ex_col], errors="coerce").fillna(0).astype("float32")
            out[course_col] = pd.to_numeric(out[course_col], errors="coerce").fillna(0).astype("int16")
            out[motor_col] = pd.to_numeric(out[motor_col], errors="coerce").fillna(0).astype("int16")
            out[boat_col] = pd.to_numeric(out[boat_col], errors="coerce").fillna(0).astype("int16")
            out[racer_col] = pd.to_numeric(out[racer_col], errors="coerce").fillna(0).astype("int32")

            out[f"lane{lane}_st_inv"] = (0.3 - out[st_col]).clip(lower=-1, upper=1).astype("float32")
            out[f"lane{lane}_exhibit_inv"] = (7.5 - out[ex_col]).clip(lower=-2, upper=2).astype("float32")
            out[f"lane{lane}_course_diff"] = (out[course_col] - lane).astype("int16")

        for pos, pos_name in [
            ("first", "combo_first_lane"),
            ("second", "combo_second_lane"),
            ("third", "combo_third_lane"),
        ]:
            out[f"{pos}_st"] = 0.0
            out[f"{pos}_exhibit"] = 0.0
            out[f"{pos}_course"] = 0
            out[f"{pos}_motor"] = 0
            out[f"{pos}_boat"] = 0
            out[f"{pos}_racer_no"] = 0

            for lane in range(1, 7):
                mask = out[pos_name] == lane
                out.loc[mask, f"{pos}_st"] = out.loc[mask, f"lane{lane}_st"]
                out.loc[mask, f"{pos}_exhibit"] = out.loc[mask, f"lane{lane}_exhibit"]
                out.loc[mask, f"{pos}_course"] = out.loc[mask, f"lane{lane}_course"]
                out.loc[mask, f"{pos}_motor"] = out.loc[mask, f"lane{lane}_motor"]
                out.loc[mask, f"{pos}_boat"] = out.loc[mask, f"lane{lane}_boat"]
                out.loc[mask, f"{pos}_racer_no"] = out.loc[mask, f"lane{lane}_racer_no"]

        out["first_second_st_diff"] = (out["first_st"] - out["second_st"]).astype("float32")
        out["first_third_st_diff"] = (out["first_st"] - out["third_st"]).astype("float32")
        out["first_second_exhibit_diff"] = (out["first_exhibit"] - out["second_exhibit"]).astype("float32")
        out["first_third_exhibit_diff"] = (out["first_exhibit"] - out["third_exhibit"]).astype("float32")

        if "wind_speed_mps" not in out.columns:
            out["wind_speed_mps"] = 0.0
        if "wave_cm" not in out.columns:
            out["wave_cm"] = 0.0

        out["wind_speed_mps"] = pd.to_numeric(out["wind_speed_mps"], errors="coerce").fillna(0).astype("float32")
        out["wave_cm"] = pd.to_numeric(out["wave_cm"], errors="coerce").fillna(0).astype("float32")

        return out

    def _prepare_x(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        x = df.copy()

        for col in feature_cols:
            if col not in x.columns:
                x[col] = 0

        x = x[feature_cols].copy()

        for col in x.columns:
            if pd.api.types.is_numeric_dtype(x[col]):
                x[col] = x[col].fillna(0)
            else:
                x[col] = x[col].astype(str).fillna("")

        return x

    def predict_proba(self, df120: pd.DataFrame, venue_name: str) -> Dict[str, float]:
        venue_name = self._normalize_venue_name(venue_name)

        if venue_name not in self.models:
            raise ValueError(f"Unsupported venue: {venue_name}")

        model = self.models[venue_name]
        meta = self.metas[venue_name]
        feature_cols = list(meta.get("feature_cols", []))

        if "combo" not in df120.columns:
            raise ValueError("df120 must have combo column")

        work = self._add_feature_block(df120.copy())
        x = self._prepare_x(work, feature_cols)

        raw_scores = model.predict(x, prediction_type="RawFormulaVal")
        raw_scores = [self._safe_float(v, 0.0) for v in raw_scores]

        temperature = self._get_temperature_by_venue(venue_name)
        probs = self._softmax(raw_scores, temperature=temperature)

        combos = df120["combo"].astype(str).tolist()
        prob_map = {combo: float(p) for combo, p in zip(combos, probs)}

        return prob_map
