from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from engine.feature_builder_current import (
    add_feature_block,
    build_is_hit,
    feature_columns,
    normalize_venue,
    sanitize_x,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

TARGET_VENUES = ["丸亀", "児島", "戸田"]
ROWS_PER_RACE = 120

TRAIN_TO = "20251231"
VALID_TO = "20260228"


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError as e:
        raise RuntimeError("catboost が入っていません。 `pip install catboost`") from e


def _safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["date"] = out["date"].astype(str)

    train_df = out[out["date"] <= TRAIN_TO].copy()
    valid_df = out[(out["date"] > TRAIN_TO) & (out["date"] <= VALID_TO)].copy()
    test_df = out[out["date"] > VALID_TO].copy()
    return train_df, valid_df, test_df


def _topk_hits(prob_map: Dict[str, float], y_combo: str, k: int) -> int:
    if not prob_map or not y_combo:
        return 0
    topk = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
    combos = [c for c, _ in topk]
    return 1 if y_combo in combos else 0


def _eval_binary_model(model, feature_cols: List[str], df: pd.DataFrame) -> Dict[str, float]:
    races = 0
    top1 = 0
    top3 = 0
    top5 = 0

    if df.empty:
        return {"races": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0}

    for race_key, df120 in df.groupby("race_key", sort=False):
        if len(df120) != ROWS_PER_RACE:
            continue

        x = sanitize_x(df120, feature_cols)
        prob = model.predict_proba(x)[:, 1]

        prob_map = {c: float(p) for c, p in zip(df120["combo"].astype(str).tolist(), prob)}
        s = sum(prob_map.values())
        if s > 0:
            prob_map = {k: v / s for k, v in prob_map.items()}

        y_combo = _safe_str(df120.iloc[0].get("y_combo"))

        races += 1
        top1 += _topk_hits(prob_map, y_combo, 1)
        top3 += _topk_hits(prob_map, y_combo, 3)
        top5 += _topk_hits(prob_map, y_combo, 5)

    return {
        "races": races,
        "top1": top1 / races if races else 0.0,
        "top3": top3 / races if races else 0.0,
        "top5": top5 / races if races else 0.0,
    }


def _filter_feature_cols(base_cols: List[str], use_racer_no: bool) -> List[str]:
    cols = []
    for c in base_cols:
        cl = c.lower()

        # 最終防衛線
        if "y_class" in cl:
            continue
        if "finish" in cl:
            continue
        if "payout" in cl:
            continue
        if cl in {"y", "is_hit", "y_combo"}:
            continue

        # AB条件
        if not use_racer_no and "racer_no" in cl:
            continue

        cols.append(c)

    return cols


def _train_single_variant(
    *,
    venue: str,
    variant_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_racer_no: bool,
) -> Dict[str, object]:
    CatBoostClassifier = _require_catboost()

    work_train = train_df.copy()
    work_valid = valid_df.copy()
    work_test = test_df.copy()

    work_train["is_hit"] = build_is_hit(work_train)
    if not work_valid.empty:
        work_valid["is_hit"] = build_is_hit(work_valid)
    if not work_test.empty:
        work_test["is_hit"] = build_is_hit(work_test)

    base_cols = feature_columns(work_train)
    cols = _filter_feature_cols(base_cols, use_racer_no=use_racer_no)

    if not cols:
        raise RuntimeError(f"{venue}/{variant_name}: feature_cols empty")

    suspicious = [c for c in cols if "y_class" in c.lower() or "finish" in c.lower() or "payout" in c.lower()]
    if suspicious:
        raise RuntimeError(f"{venue}/{variant_name}: leak-like cols found in feature_cols: {suspicious[:20]}")

    x_train = sanitize_x(work_train, cols)
    y_train = work_train["is_hit"].astype(int)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=700,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
    )

    model.fit(x_train, y_train)

    valid_eval = _eval_binary_model(model, cols, work_valid) if not work_valid.empty else {}
    test_eval = _eval_binary_model(model, cols, work_test) if not work_test.empty else {}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "with_racer_no" if use_racer_no else "without_racer_no"
    model_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_{suffix}.cbm"
    meta_path = MODEL_DIR / f"trifecta_binary_catboost_{venue}_{suffix}_meta.json"

    model.save_model(str(model_path))

    meta = {
        "model_type": "catboost_binary_safe_whitelist_ab",
        "venue": venue,
        "variant_name": variant_name,
        "use_racer_no": use_racer_no,
        "feature_cols": cols,
        "train_to": TRAIN_TO,
        "valid_to": VALID_TO,
        "n_rows_train": int(len(work_train)),
        "n_rows_valid": int(len(work_valid)),
        "n_rows_test": int(len(work_test)),
        "feature_count": len(cols),
        "valid_eval": valid_eval,
        "test_eval": test_eval,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n--- {venue} / {variant_name} ---")
    print("saved model   :", model_path)
    print("saved meta    :", meta_path)
    print("feature_count :", len(cols))
    print("use_racer_no  :", use_racer_no)
    print("valid_eval    :", valid_eval)
    print("test_eval     :", test_eval)

    return {
        "variant_name": variant_name,
        "use_racer_no": use_racer_no,
        "feature_count": len(cols),
        "valid_eval": valid_eval,
        "test_eval": test_eval,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
    }


def _print_compare_table(venue: str, results: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 100)
    print(f"COMPARE SUMMARY - {venue}")
    print("=" * 100)

    for r in results:
        valid_eval = r.get("valid_eval", {}) or {}
        test_eval = r.get("test_eval", {}) or {}
        print(
            f"{r['variant_name']:20s} | "
            f"features={int(r['feature_count']):3d} | "
            f"valid top1={float(valid_eval.get('top1', 0.0)):.4f} "
            f"top3={float(valid_eval.get('top3', 0.0)):.4f} "
            f"top5={float(valid_eval.get('top5', 0.0)):.4f} | "
            f"test top1={float(test_eval.get('top1', 0.0)):.4f} "
            f"top3={float(test_eval.get('top3', 0.0)):.4f} "
            f"top5={float(test_eval.get('top5', 0.0)):.4f}"
        )

    # test top3 + top5 を少し重視して勝者判定
    def score(res: Dict[str, object]) -> float:
        te = res.get("test_eval", {}) or {}
        return (
            float(te.get("top1", 0.0)) * 0.40
            + float(te.get("top3", 0.0)) * 0.35
            + float(te.get("top5", 0.0)) * 0.25
        )

    best = max(results, key=score)
    print("\nBEST BY WEIGHTED TEST SCORE:")
    print(
        f"{best['variant_name']} "
        f"(use_racer_no={best['use_racer_no']}, "
        f"score={score(best):.6f})"
    )


def train_one_venue(df: pd.DataFrame, venue: str) -> None:
    vdf = df[df["venue"].astype(str).map(normalize_venue) == venue].copy()
    if vdf.empty:
        print(f"[SKIP] {venue}: no rows")
        return

    train_df, valid_df, test_df = _time_split(vdf)

    print(f"\n===== {venue} =====")
    print("train rows:", len(train_df))
    print("valid rows:", len(valid_df))
    print("test rows :", len(test_df))

    if train_df.empty:
        print(f"[SKIP] {venue}: train empty")
        return

    # 共通特徴生成
    train_df = add_feature_block(train_df)
    if not valid_df.empty:
        valid_df = add_feature_block(valid_df)
    if not test_df.empty:
        test_df = add_feature_block(test_df)

    results: List[Dict[str, object]] = []

    # A: racer_no あり
    results.append(
        _train_single_variant(
            venue=venue,
            variant_name="with_racer_no",
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            use_racer_no=True,
        )
    )

    # B: racer_no なし
    results.append(
        _train_single_variant(
            venue=venue,
            variant_name="without_racer_no",
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            use_racer_no=False,
        )
    )

    _print_compare_table(venue, results)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    required_cols = {"date", "venue", "race_key", "combo", "y_combo"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    for venue in TARGET_VENUES:
        try:
            train_one_venue(df, venue)
        except Exception as e:
            print(f"[ERROR] {venue}: {e}")


if __name__ == "__main__":
    main()
