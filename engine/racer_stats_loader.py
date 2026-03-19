from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RACER_STATS_PATH = PROJECT_ROOT / "data" / "masters" / "racer_stats.csv"

_CACHE_DF: Optional[pd.DataFrame] = None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _normalize_grade(v: Any) -> str:
    s = _safe_str(v).upper()
    if s in {"A1", "A2", "B1", "B2"}:
        return s
    return ""


def _load_stats_df() -> Optional[pd.DataFrame]:
    global _CACHE_DF

    if _CACHE_DF is not None:
        return _CACHE_DF

    if not RACER_STATS_PATH.exists():
        return None

    df = pd.read_csv(RACER_STATS_PATH, dtype=str).fillna("")

    if "racer_no" not in df.columns:
        return None

    # 基本列の正規化
    df["racer_no"] = df["racer_no"].map(lambda x: _safe_int(x, 0))
    df = df[df["racer_no"] > 0].copy()

    for col in ["name", "kana", "branch", "birthplace", "sex"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_safe_str)

    if "grade" not in df.columns:
        df["grade"] = ""
    df["grade"] = df["grade"].map(_normalize_grade)

    num_float_cols = [
        "height",
        "weight",
        "win_rate",
        "place_rate",
        "avg_st",
        "prev_ability_index",
        "ability_index",
        "course1_place_rate",
        "course1_avg_st",
        "course2_place_rate",
        "course2_avg_st",
        "course3_place_rate",
        "course3_avg_st",
        "course4_place_rate",
        "course4_avg_st",
        "course5_place_rate",
        "course5_avg_st",
        "course6_place_rate",
        "course6_avg_st",
    ]
    for col in num_float_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].map(lambda x: _safe_float(x, 0.0))

    num_int_cols = [
        "age",
        "course1_entry_count",
        "course2_entry_count",
        "course3_entry_count",
        "course4_entry_count",
        "course5_entry_count",
        "course6_entry_count",
    ]
    for col in num_int_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].map(lambda x: _safe_int(x, 0))

    df = df.drop_duplicates(subset=["racer_no"], keep="last").reset_index(drop=True)

    _CACHE_DF = df
    return _CACHE_DF


def enrich_entries_with_racer_stats(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    df = _load_stats_df()
    if df is None or df.empty:
        return [dict(x) for x in entries]

    stats_map: Dict[int, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        racer_no = _safe_int(row.get("racer_no"), 0)
        if racer_no <= 0:
            continue

        stats_map[racer_no] = {
            "name": _safe_str(row.get("name"), ""),
            "kana": _safe_str(row.get("kana"), ""),
            "branch": _safe_str(row.get("branch"), ""),
            "birthplace": _safe_str(row.get("birthplace"), ""),
            "sex": _safe_str(row.get("sex"), ""),
            "grade": _normalize_grade(row.get("grade")),
            "age": _safe_int(row.get("age"), 0),
            "height": _safe_float(row.get("height"), 0.0),
            "weight": _safe_float(row.get("weight"), 0.0),
            "win_rate": _safe_float(row.get("win_rate"), 0.0),
            "place_rate": _safe_float(row.get("place_rate"), 0.0),
            "avg_st": _safe_float(row.get("avg_st"), 0.0),
            "prev_ability_index": _safe_float(row.get("prev_ability_index"), 0.0),
            "ability_index": _safe_float(row.get("ability_index"), 0.0),

            "course1_entry_count": _safe_int(row.get("course1_entry_count"), 0),
            "course1_place_rate": _safe_float(row.get("course1_place_rate"), 0.0),
            "course1_avg_st": _safe_float(row.get("course1_avg_st"), 0.0),

            "course2_entry_count": _safe_int(row.get("course2_entry_count"), 0),
            "course2_place_rate": _safe_float(row.get("course2_place_rate"), 0.0),
            "course2_avg_st": _safe_float(row.get("course2_avg_st"), 0.0),

            "course3_entry_count": _safe_int(row.get("course3_entry_count"), 0),
            "course3_place_rate": _safe_float(row.get("course3_place_rate"), 0.0),
            "course3_avg_st": _safe_float(row.get("course3_avg_st"), 0.0),

            "course4_entry_count": _safe_int(row.get("course4_entry_count"), 0),
            "course4_place_rate": _safe_float(row.get("course4_place_rate"), 0.0),
            "course4_avg_st": _safe_float(row.get("course4_avg_st"), 0.0),

            "course5_entry_count": _safe_int(row.get("course5_entry_count"), 0),
            "course5_place_rate": _safe_float(row.get("course5_place_rate"), 0.0),
            "course5_avg_st": _safe_float(row.get("course5_avg_st"), 0.0),

            "course6_entry_count": _safe_int(row.get("course6_entry_count"), 0),
            "course6_place_rate": _safe_float(row.get("course6_place_rate"), 0.0),
            "course6_avg_st": _safe_float(row.get("course6_avg_st"), 0.0),
        }

    out: List[Dict[str, Any]] = []

    for row in entries:
        new_row = dict(row)

        racer_no = _safe_int(
            new_row.get("racer_no")
            or new_row.get("register_no")
            or new_row.get("登番")
            or new_row.get("登録番号"),
            0,
        )

        if racer_no <= 0 or racer_no not in stats_map:
            out.append(new_row)
            continue

        s = stats_map[racer_no]

        # 基本情報補完
        if str(new_row.get("name", "")).strip() in ("", "-", "－", "None"):
            if s["name"]:
                new_row["name"] = s["name"]

        if str(new_row.get("branch", "")).strip() in ("", "-", "－", "None"):
            if s["branch"]:
                new_row["branch"] = s["branch"]

        if str(new_row.get("grade", "")).strip() in ("", "-", "－", "None"):
            if s["grade"]:
                new_row["grade"] = s["grade"]

        # 数値補完
        if str(new_row.get("win_rate", "")).strip() in ("", "-", "－", "None"):
            new_row["win_rate"] = s["win_rate"]

        if str(new_row.get("quinella_rate", "")).strip() in ("", "-", "－", "None"):
            new_row["quinella_rate"] = s["place_rate"]

        if str(new_row.get("place_rate", "")).strip() in ("", "-", "－", "None"):
            new_row["place_rate"] = s["place_rate"]

        if str(new_row.get("avg_st", "")).strip() in ("", "-", "－", "None"):
            new_row["avg_st"] = s["avg_st"]

        if str(new_row.get("age", "")).strip() in ("", "-", "－", "None"):
            new_row["age"] = s["age"]

        if str(new_row.get("weight", "")).strip() in ("", "-", "－", "None"):
            new_row["weight"] = s["weight"]

        # 追加能力系
        new_row["ability_index"] = s["ability_index"]
        new_row["prev_ability_index"] = s["prev_ability_index"]

        # コース別能力
        for lane in range(1, 7):
            new_row[f"racer_course{lane}_entry_count"] = s[f"course{lane}_entry_count"]
            new_row[f"racer_course{lane}_place_rate"] = s[f"course{lane}_place_rate"]
            new_row[f"racer_course{lane}_avg_st"] = s[f"course{lane}_avg_st"]

        out.append(new_row)

    return out
