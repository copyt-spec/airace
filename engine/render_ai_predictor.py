from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from engine.feature_builder_for_race import build_120_features_for_race


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render.joblib"
META_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_meta.json"
LABELS_PATH = PROJECT_ROOT / "data" / "models" / "trifecta120_model_render_labels.json"


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RenderAIPredictor:
    def __init__(self, debug: bool = False):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
        if not META_PATH.exists():
            raise FileNotFoundError(f"Missing meta: {META_PATH}")
        if not LABELS_PATH.exists():
            raise FileNotFoundError(f"Missing labels: {LABELS_PATH}")

        self.model = joblib.load(MODEL_PATH)
        self.meta = _load_json(META_PATH)
        self.labels = _load_json(LABELS_PATH)
        self.feature_cols = self.meta["feature_cols"]
        self.model_classes = np.array(self.model.classes_)
        self.class_to_col = {int(cls): i for i, cls in enumerate(self.model_classes)}
        self.combo_to_class = {combo: idx for idx, combo in enumerate(self.labels)}
        self.debug = debug

    def _debug_print_feature_snapshot(self, df: pd.DataFrame) -> None:
        watch_cols = [
            "combo",
            "first_win_rate", "second_win_rate", "third_win_rate",
            "first_place_rate", "second_place_rate", "third_place_rate",
            "first_exhibit", "second_exhibit", "third_exhibit",
            "first_st", "second_st", "third_st",
            "first_course", "second_course", "third_course",
            "first_grade", "second_grade", "third_grade",
            "f_s_win_rate_diff", "f_t_win_rate_diff", "s_t_win_rate_diff",
            "f_s_exhibit_diff", "f_t_exhibit_diff", "s_t_exhibit_diff",
            "f_s_st_diff", "f_t_st_diff", "s_t_st_diff",
        ]
        cols = [c for c in watch_cols if c in df.columns]
        if not cols:
            print("[AI-DEBUG] no snapshot columns found")
            return

        print("[AI-DEBUG] raw feature snapshot:")
        print(df[cols].head(10).to_string(index=False))

    def _sanitize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()

        missing_cols = [col for col in self.feature_cols if col not in base.columns]
        extra_cols = [col for col in base.columns if col not in self.feature_cols and col != "combo"]

        if self.debug:
            print(f"[AI-DEBUG] source feature df shape: {base.shape}")
            print(f"[AI-DEBUG] missing feature cols: {len(missing_cols)}")
            if missing_cols:
                print("[AI-DEBUG] sample missing:", missing_cols[:50])
            print(f"[AI-DEBUG] extra cols not used: {len(extra_cols)}")
            if extra_cols:
                print("[AI-DEBUG] sample extra:", extra_cols[:30])

        # DataFrame fragmentation回避
        if missing_cols:
            add_df = pd.DataFrame(0.0, index=base.index, columns=missing_cols)
            base = pd.concat([base, add_df], axis=1)

        out = base[self.feature_cols].copy()

        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        out = out.replace([np.inf, -np.inf], 0.0)

        if self.debug:
            print(f"[AI-DEBUG] sanitized feature df shape: {out.shape}")
            nonzero_ratio = (out != 0).mean().mean() if len(out) > 0 else 0.0
            print(f"[AI-DEBUG] approx nonzero ratio: {nonzero_ratio:.4f}")

        return out

    def predict_race(
        self,
        *,
        date: str,
        venue: str,
        race_no: int,
        entries: List[Dict[str, Any]],
        weather: str = "",
        wind_dir: str = "",
        wind_speed_mps: float = 0.0,
        wave_cm: float = 0.0,
        top_n: int = 20,
        odds_map: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        rows = build_120_features_for_race(
            date=date,
            venue=venue,
            race_no=race_no,
            entries=entries,
            weather=weather,
            wind_dir=wind_dir,
            wind_speed_mps=wind_speed_mps,
            wave_cm=wave_cm,
        )

        df = pd.DataFrame(rows)

        if self.debug:
            self._debug_print_feature_snapshot(df)

        x = self._sanitize_features(df)
        proba = self.model.predict_proba(x)

        result: List[Dict[str, Any]] = []

        for i, row in df.iterrows():
            combo = str(row["combo"])

            if combo not in self.combo_to_class:
                score = 0.0
            else:
                class_id = self.combo_to_class[combo]
                if class_id not in self.class_to_col:
                    score = 0.0
                else:
                    col_idx = self.class_to_col[class_id]
                    score = float(proba[i, col_idx])

            odds = None
            ev = None
            if odds_map is not None and combo in odds_map:
                odds = float(odds_map[combo])
                ev = score * odds

            result.append(
                {
                    "combo": combo,
                    "score": round(score, 6),
                    "odds": odds,
                    "ev": round(ev, 6) if ev is not None else None,
                }
            )

        if odds_map is not None:
            result.sort(key=lambda x: (x["ev"] if x["ev"] is not None else -1), reverse=True)
        else:
            result.sort(key=lambda x: x["score"], reverse=True)

        if self.debug:
            print("[AI-DEBUG] prediction top10:")
            for row in result[:10]:
                print(row)

        return result[:top_n]
