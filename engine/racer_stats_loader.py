# engine/racer_stats_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


MASTER_PATH = Path("data/master/racers_master.csv")

# 短時間キャッシュ
_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_CACHE_SECONDS = 60


def _now_ts() -> float:
    import time
    return time.time()


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        s = _safe_str(v)
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        s = _safe_str(v)
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _get_cache(key: str) -> Optional[pd.DataFrame]:
    item = _CACHE.get(key)
    if not item:
        return None

    ts, df = item
    if _now_ts() - ts > _CACHE_SECONDS:
        _CACHE.pop(key, None)
        return None

    return df.copy()


def _set_cache(key: str, df: pd.DataFrame) -> None:
    _CACHE[key] = (_now_ts(), df.copy())


def _grade_to_score(grade: str) -> float:
    g = _safe_str(grade).upper()
    mapping = {
        "A1": 4.0,
        "A2": 3.0,
        "B1": 2.0,
        "B2": 1.0,
    }
    return mapping.get(g, 0.0)


def _normalize_master_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # racer_no は文字列4桁で統一
    if "racer_no" in df.columns:
        df["racer_no"] = df["racer_no"].astype(str).str.strip().str.zfill(4)

    text_cols = [
        "name",
        "kana",
        "branch",
        "grade",
        "era",
        "birth_ymd",
        "sex",
        "blood",
        "prev_grade",
        "prev2_grade",
        "prev3_grade",
        "period_from",
        "period_to",
        "birthplace",
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    num_cols = [
        "age",
        "height",
        "weight",
        "win_rate",
        "place_rate",
        "avg_st",
        "prev_ability_index",
        "ability_index",
        "year",
        "season",
        "training_term",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for i in range(1, 7):
        for suffix in ("entry_count", "place_rate", "avg_st", "avg_st_rank"):
            c = f"course{i}_{suffix}"
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 追加派生
    if "grade" in df.columns:
        df["grade_score"] = df["grade"].map(_grade_to_score).fillna(0.0)
    else:
        df["grade_score"] = 0.0

    if "prev_grade" in df.columns:
        df["prev_grade_score"] = df["prev_grade"].map(_grade_to_score).fillna(0.0)
    else:
        df["prev_grade_score"] = 0.0

    return df


def load_racer_master(master_path: Path | str = MASTER_PATH) -> pd.DataFrame:
    path = Path(master_path)
    cache_key = str(path.resolve())

    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    if not path.exists():
        raise FileNotFoundError(f"Racer master not found: {path}")

    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df = _normalize_master_df(df)

    if "racer_no" not in df.columns:
        raise ValueError("racer master must contain racer_no column")

    # 同じ racer_no がいても最後を優先
    df = df.drop_duplicates(subset=["racer_no"], keep="last").reset_index(drop=True)

    _set_cache(cache_key, df)
    return df.copy()


def build_racer_stats_map(master_path: Path | str = MASTER_PATH) -> Dict[str, Dict[str, Any]]:
    df = load_racer_master(master_path=master_path)
    stats_map: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        racer_no = _safe_str(row.get("racer_no", "")).zfill(4)
        if racer_no == "":
            continue
        stats_map[racer_no] = row.to_dict()

    return stats_map


def enrich_entries_with_racer_stats(
    entries: List[Dict[str, Any]],
    master_path: Path | str = MASTER_PATH,
) -> List[Dict[str, Any]]:
    """
    出走表 entries に選手能力を付与する
    返却は新しい list
    """
    if not entries:
        return []

    stats_map = build_racer_stats_map(master_path=master_path)
    out: List[Dict[str, Any]] = []

    for e in entries:
        row = dict(e)
        racer_no = _safe_str(row.get("racer_no", "")).zfill(4)

        stats = stats_map.get(racer_no, {})
        if stats:
            # テキスト
            row["racer_name_master"] = _safe_str(stats.get("name", ""))
            row["racer_branch_master"] = _safe_str(stats.get("branch", ""))
            row["racer_grade_master"] = _safe_str(stats.get("grade", ""))

            # 基本能力
            row["racer_win_rate"] = _safe_float(stats.get("win_rate", 0.0), 0.0)
            row["racer_place_rate"] = _safe_float(stats.get("place_rate", 0.0), 0.0)
            row["racer_avg_st_base"] = _safe_float(stats.get("avg_st", 0.0), 0.0)
            row["racer_ability_index"] = _safe_float(stats.get("ability_index", 0.0), 0.0)
            row["racer_prev_ability_index"] = _safe_float(stats.get("prev_ability_index", 0.0), 0.0)
            row["racer_grade_score"] = _safe_float(stats.get("grade_score", 0.0), 0.0)
            row["racer_prev_grade_score"] = _safe_float(stats.get("prev_grade_score", 0.0), 0.0)

            # 属性
            row["racer_age"] = _safe_int(stats.get("age", 0), 0)
            row["racer_height"] = _safe_int(stats.get("height", 0), 0)
            row["racer_weight"] = _safe_int(stats.get("weight", 0), 0)

            # コース別適性
            for i in range(1, 7):
                row[f"racer_course{i}_entry_count"] = _safe_int(stats.get(f"course{i}_entry_count", 0), 0)
                row[f"racer_course{i}_place_rate"] = _safe_float(stats.get(f"course{i}_place_rate", 0.0), 0.0)
                row[f"racer_course{i}_avg_st"] = _safe_float(stats.get(f"course{i}_avg_st", 0.0), 0.0)
                row[f"racer_course{i}_avg_st_rank"] = _safe_float(stats.get(f"course{i}_avg_st_rank", 0.0), 0.0)
        else:
            # 無い場合も列は揃える
            row["racer_name_master"] = ""
            row["racer_branch_master"] = ""
            row["racer_grade_master"] = ""

            row["racer_win_rate"] = 0.0
            row["racer_place_rate"] = 0.0
            row["racer_avg_st_base"] = 0.0
            row["racer_ability_index"] = 0.0
            row["racer_prev_ability_index"] = 0.0
            row["racer_grade_score"] = 0.0
            row["racer_prev_grade_score"] = 0.0

            row["racer_age"] = 0
            row["racer_height"] = 0
            row["racer_weight"] = 0

            for i in range(1, 7):
                row[f"racer_course{i}_entry_count"] = 0
                row[f"racer_course{i}_place_rate"] = 0.0
                row[f"racer_course{i}_avg_st"] = 0.0
                row[f"racer_course{i}_avg_st_rank"] = 0.0

        out.append(row)

    return out


if __name__ == "__main__":
    df = load_racer_master()
    print(df.head(3).to_string())
    print("rows:", len(df))
