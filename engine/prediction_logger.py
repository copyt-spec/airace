from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "data" / "logs"
PREDICTIONS_CSV = LOG_DIR / "predictions.csv"
RESULTS_CSV = LOG_DIR / "results.csv"
MERGED_CSV = LOG_DIR / "prediction_results_merged.csv"
LANE_RACER_NO_CSV = LOG_DIR / "predictions_lane_racer_no.csv"
LANE_TILT_CSV = LOG_DIR / "predictions_lane_tilt.csv"

_lane_racer_no_keys_cache: "set[tuple[str, str, int]] | None" = None
_lane_tilt_keys_cache: "set[tuple[str, str, int]] | None" = None


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    _ensure_dir()
    file_exists = path.exists()

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_prediction_rows(
    *,
    date: str,
    venue: str,
    race_no: int,
    best_bets: List[Dict[str, Any]],
    probabilities: Dict[str, float],
    grouped_odds: Optional[Dict[str, Any]] = None,
    model_name: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    odds_map: Dict[str, float] = {}
    if grouped_odds and isinstance(grouped_odds, dict):
        godata = grouped_odds.get("data", {})
        if isinstance(godata, dict):
            for a in range(1, 7):
                col = godata.get(a, {})
                if not isinstance(col, dict):
                    continue
                for b in range(1, 7):
                    if b == a:
                        continue
                    for c in range(1, 7):
                        if c == a or c == b:
                            continue
                        combo = f"{a}-{b}-{c}"
                        odd = col.get((b, c))
                        if odd is None:
                            odd = col.get(f"{b}{c}")
                        odds_map[combo] = _safe_float(odd, 0.0)

    selected_combo_set = {
        _safe_str(x.get("combo", ""))
        for x in (best_bets or [])
        if _safe_str(x.get("combo", ""))
    }

    ranked = sorted(
        [(combo, _safe_float(prob, 0.0)) for combo, prob in (probabilities or {}).items()],
        key=lambda kv: kv[1],
        reverse=True,
    )

    logged_at = _now_iso()

    for rank, (combo, prob) in enumerate(ranked, start=1):
        odds = _safe_float(odds_map.get(combo, 0.0), 0.0)
        expected_return_yen = prob * odds * 100.0 if odds > 0 else 0.0
        row = {
            "logged_at": logged_at,
            "date": _safe_str(date),
            "venue": _safe_str(venue),
            "race_no": int(race_no),
            "combo": combo,
            "rank_prob": rank,
            "prob": round(prob, 8),
            "prob_pct": round(prob * 100.0, 4),
            "odds": round(odds, 4),
            "expected_return_yen": round(expected_return_yen, 4),
            "is_selected": 1 if combo in selected_combo_set else 0,
            "model_name": _safe_str(model_name),
        }
        rows.append(row)

    return rows


def save_prediction_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = [
        "logged_at",
        "date",
        "venue",
        "race_no",
        "combo",
        "rank_prob",
        "prob",
        "prob_pct",
        "odds",
        "expected_return_yen",
        "is_selected",
        "model_name",
    ]

    for row in rows:
        _append_csv_row(PREDICTIONS_CSV, fieldnames, row)


def _load_lane_racer_no_keys() -> "set[tuple[str, str, int]]":
    """既にlane racer_noを記録済みの(date, venue, race_no)集合を返す(重複記録防止)。"""
    global _lane_racer_no_keys_cache
    if _lane_racer_no_keys_cache is not None:
        return _lane_racer_no_keys_cache

    keys: "set[tuple[str, str, int]]" = set()
    if LANE_RACER_NO_CSV.exists():
        with open(LANE_RACER_NO_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    keys.add((
                        _safe_str(row.get("date", "")),
                        _safe_str(row.get("venue", "")),
                        int(float(row.get("race_no", 0) or 0)),
                    ))
                except Exception:
                    continue

    _lane_racer_no_keys_cache = keys
    return keys


def save_lane_racer_no(
    *,
    date: str,
    venue: str,
    race_no: int,
    lane_racer_no: Dict[int, int],
) -> None:
    """
    2026-07-21追加: 選手級別(A1/A2/B1/B2)によるレースパターン分析
    (scripts/analyze_racer_class_patterns.py)を、学習データの期間だけで
    なく今後のライブ予測ぶんにも継続できるようにするためのログ。

    predictions.csv自体には追記しない(1レース=120行という既存スキーマに
    列を後から足すと、既存の行と列数が合わなくなりCSVが壊れるため)。
    代わりに1レース=1行の別ファイルに保存し、後から
    date/venue/race_noで突き合わせる(racer_stats.csvのgrade結合と同じ要領)。

    lane_racer_no: {1: 4320, 2: 4999, ...} のようなlane(1-6)->racer_noの辞書。
    6艇そろっていない(出走表が取れなかった等)場合は記録しない。
    """
    if not lane_racer_no or len(lane_racer_no) != 6:
        return
    if any(lane_racer_no.get(lane, 0) <= 0 for lane in range(1, 7)):
        return

    key = (_safe_str(date), _safe_str(venue), int(race_no))
    existing_keys = _load_lane_racer_no_keys()
    if key in existing_keys:
        return

    fieldnames = ["logged_at", "date", "venue", "race_no"] + [f"lane{i}_racer_no" for i in range(1, 7)]
    row = {
        "logged_at": _now_iso(),
        "date": _safe_str(date),
        "venue": _safe_str(venue),
        "race_no": int(race_no),
    }
    for lane in range(1, 7):
        row[f"lane{lane}_racer_no"] = int(lane_racer_no.get(lane, 0))

    _append_csv_row(LANE_RACER_NO_CSV, fieldnames, row)
    existing_keys.add(key)


def _load_lane_tilt_keys() -> "set[tuple[str, str, int]]":
    """既にlane tiltを記録済みの(date, venue, race_no)集合を返す(重複記録防止)。"""
    global _lane_tilt_keys_cache
    if _lane_tilt_keys_cache is not None:
        return _lane_tilt_keys_cache

    keys: "set[tuple[str, str, int]]" = set()
    if LANE_TILT_CSV.exists():
        with open(LANE_TILT_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    keys.add((
                        _safe_str(row.get("date", "")),
                        _safe_str(row.get("venue", "")),
                        int(float(row.get("race_no", 0) or 0)),
                    ))
                except Exception:
                    continue

    _lane_tilt_keys_cache = keys
    return keys


def save_lane_tilt(
    *,
    date: str,
    venue: str,
    race_no: int,
    lane_tilt: Dict[int, Optional[float]],
) -> None:
    """
    2026-07-29追加: チルト角度(モーターの取り付け角度、展示が良くても
    角度が大きいと操舵性が落ちるという仮説の検証用)を、今後のライブ予測ぶんに
    ついて蓄積するためのログ。

    raw_txt(過去結果ファイル)にはチルト角度の記録が無いため、過去に遡って
    バックテストすることはできない。今後このログが十分溜まった時点で初めて
    予測精度への寄与を検証できる([[boat_ai_kimarite_wind_chaos_detection]]や
    [[boat_ai_marugame_wind_course_feature]]と同じく、n>=500目安の蓄積を待つ)。

    predictions.csv本体には追記しない(save_lane_racer_noと同じ理由)。
    1レース=1行の別ファイルに保存し、後からdate/venue/race_noで突き合わせる。

    lane_tilt: {1: -0.5, 2: 0.0, ...} のようなlane(1-6)->tilt角度(度)の辞書。
    値がNone(取得失敗/未取得/パース不能)のレーンが1つでもあれば記録しない
    (中途半端なデータで学習・分析に混入させないため)。
    """
    if not lane_tilt or len(lane_tilt) != 6:
        return
    if any(lane_tilt.get(lane) is None for lane in range(1, 7)):
        return

    key = (_safe_str(date), _safe_str(venue), int(race_no))
    existing_keys = _load_lane_tilt_keys()
    if key in existing_keys:
        return

    fieldnames = ["logged_at", "date", "venue", "race_no"] + [f"lane{i}_tilt" for i in range(1, 7)]
    row = {
        "logged_at": _now_iso(),
        "date": _safe_str(date),
        "venue": _safe_str(venue),
        "race_no": int(race_no),
    }
    for lane in range(1, 7):
        row[f"lane{lane}_tilt"] = float(lane_tilt.get(lane))  # type: ignore[arg-type]

    _append_csv_row(LANE_TILT_CSV, fieldnames, row)
    existing_keys.add(key)


def save_race_result(
    *,
    date: str,
    venue: str,
    race_no: int,
    actual_combo: str,
    payout: float,
    source: str = "",
) -> None:
    fieldnames = [
        "logged_at",
        "date",
        "venue",
        "race_no",
        "actual_combo",
        "payout",
        "source",
    ]

    row = {
        "logged_at": _now_iso(),
        "date": _safe_str(date),
        "venue": _safe_str(venue),
        "race_no": int(race_no),
        "actual_combo": _safe_str(actual_combo),
        "payout": round(_safe_float(payout, 0.0), 4),
        "source": _safe_str(source),
    }
    _append_csv_row(RESULTS_CSV, fieldnames, row)
