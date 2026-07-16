# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from catboost import CatBoostClassifier, Pool


class GlobalVenueAutoCatBoostModel:
    def __init__(
        self,
        model_dir: str = "data/models",
        model_name: str = "trifecta_binary_catboost_global_venue_auto",
        debug: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.debug = debug

        self.model_path = self.model_dir / f"{self.model_name}.cbm"
        self.meta_path = self.model_dir / f"{self.model_name}_meta.json"
        self.venue_stats_path = self.model_dir / f"{self.model_name}_venue_stats.csv"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing meta file: {self.meta_path}")
        if not self.venue_stats_path.exists():
            raise FileNotFoundError(f"Missing venue stats file: {self.venue_stats_path}")

        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.feature_cols: List[str] = list(self.meta.get("feature_cols", []))
        self.cat_cols: List[str] = list(self.meta.get("cat_cols", []))

        self.venue_stats_df = pd.read_csv(self.venue_stats_path, low_memory=False)
        if "venue" not in self.venue_stats_df.columns:
            raise ValueError("venue stats csv must contain venue column")

        self.venue_stats_df["venue"] = self.venue_stats_df["venue"].astype(str)
        self.venue_stats_map: Dict[str, Dict[str, Any]] = (
            self.venue_stats_df.set_index("venue").to_dict(orient="index")
        )

        if self.debug:
            print("[GLOBAL_MODEL_LOADED]", self.model_path)
            print("[GLOBAL_META_LOADED]", self.meta_path)
            print("[GLOBAL_VENUE_STATS_LOADED]", self.venue_stats_path)
            print("[FEATURE_COUNT]", len(self.feature_cols))
            print("[CAT_COLS]", self.cat_cols)
            print("[VENUE_STATS_COUNT]", len(self.venue_stats_map))

    def _safe_float(self, v: Any, default: float = 0.0) -> float:
        try:
            if v is None or v == "":
                return default
            return float(v)
        except Exception:
            return default

    def _safe_int(self, v: Any, default: int = 0) -> int:
        try:
            if v is None or v == "":
                return default
            return int(float(v))
        except Exception:
            return default

    def _normalize_venue_name(self, venue_name: Any) -> str:
        s = str(venue_name or "").strip()
        for venue in [
            "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
            "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
            "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
        ]:
            if venue in s:
                return venue
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

        temp_map = {
            "丸亀": 0.88,
            "戸田": 0.92,
            "児島": 0.90,
            "住之江": 0.90,
        }
        return float(temp_map.get(venue_name, 0.90))

    def _split_combo_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "combo" not in out.columns:
            raise ValueError("df120 must contain combo column")

        parts = out["combo"].astype(str).str.split("-", expand=True)
        if parts.shape[1] < 3:
            raise ValueError("combo split failed")

        out["combo_first_lane"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype("int16")
        out["combo_second_lane"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype("int16")
        out["combo_third_lane"] = pd.to_numeric(parts[2], errors="coerce").fillna(0).astype("int16")

        return out

    def _build_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._split_combo_cols(df.copy())

        if "venue" not in out.columns:
            out["venue"] = ""

        out["venue"] = out["venue"].astype(str).map(self._normalize_venue_name)

        wind_map = {
            "無風": 0,
            "北": 1, "北東": 2, "東": 3, "南東": 4,
            "南": 5, "南西": 6, "西": 7, "北西": 8,
        }
        weather_map = {
            "晴": 1, "晴れ": 1,
            "曇": 2, "曇り": 2,
            "雨": 3,
            "雪": 4,
        }

        if "wind_dir" not in out.columns:
            out["wind_dir"] = ""
        if "weather" not in out.columns:
            out["weather"] = ""

        out["wind_dir"] = out["wind_dir"].astype(str)
        out["weather"] = out["weather"].astype(str)

        out["wind_dir_code"] = out["wind_dir"].map(wind_map).fillna(0).astype("int16")
        out["weather_code"] = out["weather"].map(weather_map).fillna(0).astype("int16")

        if "wind_speed_mps" not in out.columns:
            out["wind_speed_mps"] = 0.0
        if "wave_cm" not in out.columns:
            out["wave_cm"] = 0.0

        out["wind_speed_mps"] = pd.to_numeric(out["wind_speed_mps"], errors="coerce").fillna(0).astype("float32")
        out["wave_cm"] = pd.to_numeric(out["wave_cm"], errors="coerce").fillna(0).astype("float32")

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

        return out

    def _attach_venue_auto_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "venue" not in out.columns:
            out["venue"] = ""

        default_stats: Dict[str, Any] = {}
        if self.venue_stats_map:
            sample_venue = next(iter(self.venue_stats_map.keys()))
            for k in self.venue_stats_map[sample_venue].keys():
                default_stats[k] = 0.0

        rows = []
        for _, row in out.iterrows():
            venue = self._normalize_venue_name(row.get("venue", ""))
            stats = self.venue_stats_map.get(venue, default_stats)
            merged = row.to_dict()
            for k, v in stats.items():
                merged[k] = v
            rows.append(merged)

        out = pd.DataFrame(rows)

        # 会場 lane 傾向と combo の一致率
        out["venue_first_lane_match_rate"] = 0.0
        out["venue_second_lane_match_rate"] = 0.0
        out["venue_third_lane_match_rate"] = 0.0

        for lane in range(1, 7):
            col1 = f"venue_first_lane_{lane}_rate"
            col2 = f"venue_second_lane_{lane}_rate"
            col3 = f"venue_third_lane_{lane}_rate"

            if col1 not in out.columns:
                out[col1] = 0.0
            if col2 not in out.columns:
                out[col2] = 0.0
            if col3 not in out.columns:
                out[col3] = 0.0

            mask1 = out["combo_first_lane"] == lane
            mask2 = out["combo_second_lane"] == lane
            mask3 = out["combo_third_lane"] == lane

            out.loc[mask1, "venue_first_lane_match_rate"] = pd.to_numeric(
                out.loc[mask1, col1], errors="coerce"
            ).fillna(0.0)
            out.loc[mask2, "venue_second_lane_match_rate"] = pd.to_numeric(
                out.loc[mask2, col2], errors="coerce"
            ).fillna(0.0)
            out.loc[mask3, "venue_third_lane_match_rate"] = pd.to_numeric(
                out.loc[mask3, col3], errors="coerce"
            ).fillna(0.0)

        out["venue_combo_lane_prior"] = (
            out["venue_first_lane_match_rate"]
            + out["venue_second_lane_match_rate"]
            + out["venue_third_lane_match_rate"]
        ).astype("float32")

        return out

    def _prepare_x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        for col in self.feature_cols:
            if col not in x.columns:
                if col in self.cat_cols:
                    x[col] = ""
                else:
                    x[col] = 0

        x = x[self.feature_cols].copy()

        for col in x.columns:
            if col in self.cat_cols:
                x[col] = x[col].astype(str).fillna("")
            else:
                x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

        return x

    def predict_proba(self, df120: pd.DataFrame, venue_name: str) -> Dict[str, float]:
        if "combo" not in df120.columns:
            raise ValueError("df120 must have combo column")

        work = df120.copy()

        if "venue" not in work.columns:
            work["venue"] = self._normalize_venue_name(venue_name)
        else:
            work["venue"] = work["venue"].astype(str).map(self._normalize_venue_name)

        work = self._build_basic_features(work)
        work = self._attach_venue_auto_features(work)
        x = self._prepare_x(work)

        pool = Pool(
            x,
            cat_features=self.cat_cols,
        )

        raw_scores = self.model.predict(pool, prediction_type="RawFormulaVal")
        raw_scores = [self._safe_float(v, 0.0) for v in raw_scores]

        temperature = self._get_temperature_by_venue(venue_name)
        probs = self._softmax(raw_scores, temperature=temperature)

        combos = df120["combo"].astype(str).tolist()
        prob_map = {combo: float(p) for combo, p in zip(combos, probs)}

        if self.debug:
            print("[GLOBAL_PREDICT]")
            print("venue       :", venue_name)
            print("rows        :", len(df120))
            print("feature_cols:", len(self.feature_cols))
            print("cat_cols    :", self.cat_cols)

        return prob_map
