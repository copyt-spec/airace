from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EVCalcConfig:
    pred_prob_col: str = "pred_proba"
    odds_col: str = "odds"
    combo_col: str = "combo"
    race_key_col: str = "race_key"
    min_odds: float = 1.0
    ev_col: str = "ev"
    expected_return_col: str = "expected_return"
    edge_col: str = "edge"
    rank_col: str = "ev_rank"
    hit_col: str = "is_hit"


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    予測CSV側の列ゆれを吸収する。
    """
    out = df.copy()

    rename_map = {}

    if "score" in out.columns and "pred_proba" not in out.columns:
        rename_map["score"] = "pred_proba"

    if "prob" in out.columns and "pred_proba" not in out.columns:
        rename_map["prob"] = "pred_proba"

    if "prediction" in out.columns and "pred_proba" not in out.columns:
        rename_map["prediction"] = "pred_proba"

    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def normalize_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    オッズCSV側の列ゆれを吸収する。
    """
    out = df.copy()

    rename_map = {}

    if "trifecta_odds" in out.columns and "odds" not in out.columns:
        rename_map["trifecta_odds"] = "odds"

    if "odds_value" in out.columns and "odds" not in out.columns:
        rename_map["odds_value"] = "odds"

    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def validate_prediction_df(df: pd.DataFrame, cfg: Optional[EVCalcConfig] = None) -> None:
    cfg = cfg or EVCalcConfig()

    required = [cfg.race_key_col, cfg.combo_col, cfg.pred_prob_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Prediction CSV missing required columns: {missing}")


def validate_odds_df(df: pd.DataFrame, cfg: Optional[EVCalcConfig] = None) -> None:
    cfg = cfg or EVCalcConfig()

    required = [cfg.race_key_col, cfg.combo_col, cfg.odds_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Odds CSV missing required columns: {missing}")


def merge_predictions_and_odds(
    pred_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    cfg: Optional[EVCalcConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EVCalcConfig()

    pred = normalize_prediction_columns(pred_df)
    odds = normalize_odds_columns(odds_df)

    validate_prediction_df(pred, cfg)
    validate_odds_df(odds, cfg)

    merged = pred.merge(
        odds[[cfg.race_key_col, cfg.combo_col, cfg.odds_col]].copy(),
        on=[cfg.race_key_col, cfg.combo_col],
        how="left",
        validate="one_to_one",
    )

    merged[cfg.pred_prob_col] = _to_numeric_safe(merged[cfg.pred_prob_col]).fillna(0.0)
    merged[cfg.odds_col] = _to_numeric_safe(merged[cfg.odds_col])

    return merged


def calc_ev_table(
    merged_df: pd.DataFrame,
    cfg: Optional[EVCalcConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EVCalcConfig()

    df = merged_df.copy()

    df[cfg.pred_prob_col] = _to_numeric_safe(df[cfg.pred_prob_col]).fillna(0.0)
    df[cfg.odds_col] = _to_numeric_safe(df[cfg.odds_col])

    valid_odds_mask = df[cfg.odds_col].notna() & (df[cfg.odds_col] >= cfg.min_odds)

    df[cfg.expected_return_col] = np.where(
        valid_odds_mask,
        df[cfg.pred_prob_col] * df[cfg.odds_col],
        np.nan,
    )

    df[cfg.ev_col] = df[cfg.expected_return_col]
    df[cfg.edge_col] = df[cfg.expected_return_col] - 1.0

    if "y_combo" in df.columns:
        df[cfg.hit_col] = df[cfg.combo_col].astype(str) == df["y_combo"].astype(str)
    else:
        df[cfg.hit_col] = False

    df[cfg.rank_col] = (
        df.groupby(cfg.race_key_col)[cfg.ev_col]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )

    df = df.sort_values(
        [cfg.race_key_col, cfg.rank_col, cfg.combo_col],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return df


def build_ev_summary(
    ev_df: pd.DataFrame,
    cfg: Optional[EVCalcConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EVCalcConfig()

    work = ev_df.copy()

    total_races = work[cfg.race_key_col].nunique()

    top1 = work[work[cfg.rank_col] == 1].copy()
    top3 = work[work[cfg.rank_col] <= 3].copy()
    top5 = work[work[cfg.rank_col] <= 5].copy()
    top10 = work[work[cfg.rank_col] <= 10].copy()

    def _hit_rate(df_part: pd.DataFrame) -> float:
        if df_part.empty or cfg.hit_col not in df_part.columns:
            return 0.0
        hit_races = df_part.loc[df_part[cfg.hit_col] == True, cfg.race_key_col].nunique()
        if total_races == 0:
            return 0.0
        return float(hit_races) / float(total_races)

    summary = {
        "num_races": total_races,
        "top1_hit_rate": _hit_rate(top1),
        "top3_hit_rate": _hit_rate(top3),
        "top5_hit_rate": _hit_rate(top5),
        "top10_hit_rate": _hit_rate(top10),
        "avg_top1_pred_proba": float(top1[cfg.pred_prob_col].mean()) if not top1.empty else 0.0,
        "avg_top1_odds": float(top1[cfg.odds_col].mean()) if not top1.empty else 0.0,
        "avg_top1_ev": float(top1[cfg.ev_col].mean()) if not top1.empty else 0.0,
        "avg_top3_ev": float(top3[cfg.ev_col].mean()) if not top3.empty else 0.0,
        "avg_top5_ev": float(top5[cfg.ev_col].mean()) if not top5.empty else 0.0,
        "avg_top10_ev": float(top10[cfg.ev_col].mean()) if not top10.empty else 0.0,
    }

    return pd.DataFrame([summary])


def build_top_ev_picks(
    ev_df: pd.DataFrame,
    cfg: Optional[EVCalcConfig] = None,
    top_n_per_race: int = 5,
    min_ev: Optional[float] = None,
) -> pd.DataFrame:
    cfg = cfg or EVCalcConfig()

    out = ev_df.copy()

    out = out[out[cfg.rank_col].notna()].copy()
    out = out[out[cfg.rank_col] <= top_n_per_race].copy()

    if min_ev is not None:
        out = out[out[cfg.ev_col] >= float(min_ev)].copy()

    out = out.sort_values(
        [cfg.race_key_col, cfg.rank_col, cfg.ev_col],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return out
