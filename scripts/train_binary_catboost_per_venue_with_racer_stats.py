from __future__ import annotations

"""
scripts/train_binary_catboost_per_venue.py に、選手の実績特徴量
(通算勝率・複勝率・平均ST・出走コース別複勝率/平均ST。すべてレース時点より前の
実績のみを使ったリーク無し集計)を追加したバージョン。

事前準備:
  1) python scripts/build_racer_history.py            (data/raw_txt全件から選手×レースの履歴を作成)
  2) python scripts/build_racer_point_in_time_stats.py (リーク無しの時系列成績を計算)
  3) このスクリプトを実行

このサンドボックス環境にはcatboostが無いため実行検証ができていない。
data/datasets/racer_race_history.csv と racer_point_in_time_stats.csv は
既に用意してあるので、catboostが使えるユーザーの手元環境でこのスクリプトを
実行し、venue指定 or --all で学習・比較してほしい。

保存先は "_with_racer_stats" サフィックス付きにしてあり、既存の
"_with_racer_no" モデルは上書きしない。バックテストで比較してから
本番差し替えを判断すること。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "trifecta_train.csv"
RACER_STATS_PATH = PROJECT_ROOT / "data" / "datasets" / "racer_point_in_time_stats.csv"
# 2026-07-21追加: 選手の直近の調子(recent5_*)とモーターの直近の調子
# (motor_recent8_*)。scripts/analyze_racer_recent_form.py /
# analyze_motor_recent_form.py で、選手の通算実力とは別に将来の成績を
# 追加的に予測することを確認済み(いずれも有意)。
MOTOR_STATS_PATH = PROJECT_ROOT / "data" / "datasets" / "motor_point_in_time_stats.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"


SPLIT_BY_VENUE_DIR = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_by_venue"


def _load_dataset_for_venue(src: Path, venue: str, chunksize: int = 200_000) -> pd.DataFrame:
    venue = normalize_venue_name(venue)

    # scripts/split_trifecta_train_by_venue.py で事前分割済みならそれを使う(高速)
    pre_split_path = SPLIT_BY_VENUE_DIR / f"{venue}.csv"
    if pre_split_path.exists():
        print(f"Loading pre-split dataset for venue: {venue} <- {pre_split_path}")
        df = pd.read_csv(pre_split_path, low_memory=False)
        df["venue"] = df["venue"].astype(str).map(normalize_venue_name)
        df = df[df["venue"] == venue].copy()
        if df.empty:
            raise RuntimeError(f"no data for venue: {venue}")
        return df

    if not src.exists():
        raise FileNotFoundError(src)

    print(f"Loading dataset for venue: {venue}")
    print(f"source: {src}")
    print("(ヒント: scripts/split_trifecta_train_by_venue.py を先に実行すると--allが大幅に速くなります)")

    chunks: List[pd.DataFrame] = []
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(src, low_memory=False, chunksize=chunksize):
        total_rows += len(chunk)

        if "venue" not in chunk.columns:
            raise ValueError("dataset must contain venue column")

        chunk["venue"] = chunk["venue"].astype(str).map(normalize_venue_name)
        hit = chunk[chunk["venue"] == venue].copy()

        if not hit.empty:
            matched_rows += len(hit)
            chunks.append(hit)

        print(f"scanned={total_rows:,} matched={matched_rows:,}")

    if not chunks:
        raise RuntimeError(f"no data for venue: {venue}")

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        raise RuntimeError(f"no data for venue: {venue}")

    return df


def _load_racer_stats() -> pd.DataFrame:
    if not RACER_STATS_PATH.exists():
        raise FileNotFoundError(
            f"{RACER_STATS_PATH} が見つかりません。先に "
            "scripts/build_racer_history.py と scripts/build_racer_point_in_time_stats.py を実行してください。"
        )
    stats = pd.read_csv(RACER_STATS_PATH, low_memory=False)
    stats["date"] = stats["date"].astype(str).str.zfill(8)
    stats["racer_no"] = pd.to_numeric(stats["racer_no"], errors="coerce").fillna(0).astype(int)
    return stats


def _load_motor_stats() -> pd.DataFrame:
    """
    2026-07-21追加: (venue, motor, date)単位のリーク無し直近成績。
    無い場合(build_motor_point_in_time_stats.py未実行)は空DataFrameを返し、
    呼び出し側でモーター特徴量を全てNaN(→0埋め)にフォールバックする。
    """
    if not MOTOR_STATS_PATH.exists():
        print(f"[WARN] {MOTOR_STATS_PATH} が見つかりません。モーター特徴量は空になります。"
              "先に scripts/build_motor_point_in_time_stats.py を実行してください。")
        return pd.DataFrame()
    stats = pd.read_csv(MOTOR_STATS_PATH, low_memory=False)
    stats["date"] = stats["date"].astype(str).str.zfill(8)
    stats["motor"] = pd.to_numeric(stats["motor"], errors="coerce").fillna(0).astype(int)
    return stats


def _add_feature_block(df: pd.DataFrame) -> pd.DataFrame:
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

    # 2026-07-22追加: 「展開予想」の要因(scripts/analyze_race_development_patterns.py参照)。
    # そのレース"内"での展示タイム・STの相対順位(1=最速/最良)。選手の通算能力
    # (win_rate_prior等)を差し引いても有意に勝敗を追加予測することを確認済み。
    # exhibit/stはレース前(展示走行後・出走表確定後)に既知の情報なのでリークは無い。
    # 同じ行内の6レーン分の値を横に比較するだけなので、レース単位のgroupbyは不要。
    exhibit_cols = [f"lane{i}_exhibit" for i in range(1, 7)]
    st_cols = [f"lane{i}_st" for i in range(1, 7)]
    exhibit_ranks = out[exhibit_cols].rank(axis=1, method="min", ascending=True)
    st_ranks = out[st_cols].rank(axis=1, method="min", ascending=True)
    for lane in range(1, 7):
        out[f"lane{lane}_exhibit_rank"] = exhibit_ranks[f"lane{lane}_exhibit"].fillna(0).astype("float32")
        out[f"lane{lane}_st_rank"] = st_ranks[f"lane{lane}_st"].fillna(0).astype("float32")

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
        out[f"{pos}_exhibit_rank"] = 0.0
        out[f"{pos}_st_rank"] = 0.0

        for lane in range(1, 7):
            mask = out[pos_name] == lane
            out.loc[mask, f"{pos}_st"] = out.loc[mask, f"lane{lane}_st"]
            out.loc[mask, f"{pos}_exhibit"] = out.loc[mask, f"lane{lane}_exhibit"]
            out.loc[mask, f"{pos}_course"] = out.loc[mask, f"lane{lane}_course"]
            out.loc[mask, f"{pos}_motor"] = out.loc[mask, f"lane{lane}_motor"]
            out.loc[mask, f"{pos}_boat"] = out.loc[mask, f"lane{lane}_boat"]
            out.loc[mask, f"{pos}_racer_no"] = out.loc[mask, f"lane{lane}_racer_no"]
            out.loc[mask, f"{pos}_exhibit_rank"] = out.loc[mask, f"lane{lane}_exhibit_rank"]
            out.loc[mask, f"{pos}_st_rank"] = out.loc[mask, f"lane{lane}_st_rank"]

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

    # 2026-07-26追加: 丸亀特化の検証(ユーザーが丸亀によく行く/思い入れがある
    # ことがきっかけ)。丸亀データでOLS(HC1)検証したところ、1号艇(course=1)に
    # 限定した勝率は風速が強いほど有意に下がる(win coef=-0.0164, 95%CI
    # [-0.0226,-0.0102], n=7509 / top3 coef=-0.0085, CI[-0.0132,-0.0037])。
    # wind_speed_mps自体は既存モデルにも入っているためCatBoostの分岐で暗黙に
    # 学習できる可能性はあるが、「コース×風速」の交互作用を明示的な特徴量として
    # 渡しておくことで、少ない分岐数でも同じ関係を捉えやすくなる(木モデルでも
    # 明示的な交互作用項を渡すと学習効率が上がることがある、という一般的な手法)。
    # 会場によらず計算しておき、実際に使うかはmeta.jsonのfeature_colsに委ねる
    # (他会場の展開特徴量と同じ設計)。
    for lane in range(1, 7):
        course_col = f"lane{lane}_course"
        out[f"lane{lane}_course_wind_interaction"] = (
            out[course_col].astype("float32") * out["wind_speed_mps"]
        ).astype("float32")
        out[f"lane{lane}_course_wave_interaction"] = (
            out[course_col].astype("float32") * out["wave_cm"]
        ).astype("float32")

    for pos, pos_name in [
        ("first", "combo_first_lane"),
        ("second", "combo_second_lane"),
        ("third", "combo_third_lane"),
    ]:
        out[f"{pos}_course_wind_interaction"] = 0.0
        out[f"{pos}_course_wave_interaction"] = 0.0
        for lane in range(1, 7):
            mask = out[pos_name] == lane
            out.loc[mask, f"{pos}_course_wind_interaction"] = out.loc[mask, f"lane{lane}_course_wind_interaction"]
            out.loc[mask, f"{pos}_course_wave_interaction"] = out.loc[mask, f"lane{lane}_course_wave_interaction"]

    return out


def _add_racer_stats_block(df: pd.DataFrame, racer_stats: pd.DataFrame) -> pd.DataFrame:
    """
    lane{n}_racer_no + date をキーに、リーク無しの時系列選手成績を結合する。
    course別の place_rate/avg_st は、そのレースでその艇が実際に取った
    lane{n}_course に対応する列を選んで割り当てる。
    """
    out = df.copy()
    out["date"] = out["date"].astype(str).str.zfill(8)

    stats_indexed = racer_stats.set_index(["racer_no", "date"])

    for lane in range(1, 7):
        racer_col = f"lane{lane}_racer_no"
        course_col = f"lane{lane}_course"

        key_df = out[[racer_col, "date"]].copy()
        key_df.columns = ["racer_no", "date"]
        key_df["racer_no"] = pd.to_numeric(key_df["racer_no"], errors="coerce").fillna(0).astype(int)

        joined = key_df.merge(
            racer_stats,
            on=["racer_no", "date"],
            how="left",
        )

        out[f"lane{lane}_races_prior"] = joined["races_prior"].to_numpy()
        out[f"lane{lane}_win_rate_prior"] = joined["win_rate_prior"].to_numpy()
        out[f"lane{lane}_place_rate_prior"] = joined["place_rate_prior"].to_numpy()
        out[f"lane{lane}_avg_st_prior"] = joined["avg_st_prior"].to_numpy()
        # 2026-07-21追加: 選手の直近の調子(ホットハンド、analyze_racer_recent_form.py参照)
        if "recent5_place_rate_prior" in joined.columns:
            out[f"lane{lane}_recent5_races_prior"] = joined["recent5_races_prior"].to_numpy()
            out[f"lane{lane}_recent5_place_rate_prior"] = joined["recent5_place_rate_prior"].to_numpy()
        else:
            out[f"lane{lane}_recent5_races_prior"] = np.nan
            out[f"lane{lane}_recent5_place_rate_prior"] = np.nan

        # そのレースで実際に走ったコース(1-6)別の複勝率・平均STを選ぶ
        course_vals = pd.to_numeric(out[course_col], errors="coerce").fillna(0).astype(int)
        place_rate_by_course = np.full(len(out), np.nan)
        avg_st_by_course = np.full(len(out), np.nan)

        for c in range(1, 7):
            mask = (course_vals == c).to_numpy()
            if not mask.any():
                continue
            place_rate_by_course[mask] = joined[f"course{c}_place_rate_prior"].to_numpy()[mask]
            avg_st_by_course[mask] = joined[f"course{c}_avg_st_prior"].to_numpy()[mask]

        out[f"lane{lane}_course_place_rate_prior"] = place_rate_by_course
        out[f"lane{lane}_course_avg_st_prior"] = avg_st_by_course

    # combo(1着/2着/3着候補)側にも同じ値を割り当てる
    for pos, pos_name in [
        ("first", "combo_first_lane"),
        ("second", "combo_second_lane"),
        ("third", "combo_third_lane"),
    ]:
        for feat in RACER_STATS_FEATURE_SUFFIXES:
            out[f"{pos}_{feat}"] = np.nan
            for lane in range(1, 7):
                mask = out[pos_name] == lane
                out.loc[mask, f"{pos}_{feat}"] = out.loc[mask, f"lane{lane}_{feat}"]

    return out


def _add_motor_stats_block(df: pd.DataFrame, motor_stats: pd.DataFrame) -> pd.DataFrame:
    """
    2026-07-21追加: (venue, lane{n}_motor, date)をキーに、リーク無しの
    モーター直近成績(motor_recent8_*)を結合する。scripts/analyze_motor_recent_form.py
    で「選手の実力とは別に将来を予測する」ことを確認済み。

    motor_stats が空(build_motor_point_in_time_stats.py未実行)の場合は
    全レーンNaN(学習時は0埋め)にフォールバックする。
    """
    out = df.copy()
    out["date"] = out["date"].astype(str).str.zfill(8)

    if motor_stats is None or motor_stats.empty:
        for lane in range(1, 7):
            for suf in MOTOR_STATS_FEATURE_SUFFIXES:
                out[f"lane{lane}_{suf}"] = np.nan
    else:
        for lane in range(1, 7):
            motor_col = f"lane{lane}_motor"

            key_df = out[["venue", motor_col, "date"]].copy()
            key_df.columns = ["venue", "motor", "date"]
            key_df["motor"] = pd.to_numeric(key_df["motor"], errors="coerce").fillna(0).astype(int)

            joined = key_df.merge(motor_stats, on=["venue", "motor", "date"], how="left")

            for suf in MOTOR_STATS_FEATURE_SUFFIXES:
                col = f"motor_{suf}"
                out[f"lane{lane}_motor_{suf}"] = joined[col].to_numpy() if col in joined.columns else np.nan

    for pos, pos_name in [
        ("first", "combo_first_lane"),
        ("second", "combo_second_lane"),
        ("third", "combo_third_lane"),
    ]:
        for suf in MOTOR_STATS_FEATURE_SUFFIXES:
            feat = f"motor_{suf}"
            out[f"{pos}_{feat}"] = np.nan
            for lane in range(1, 7):
                mask = out[pos_name] == lane
                out.loc[mask, f"{pos}_{feat}"] = out.loc[mask, f"lane{lane}_{feat}"]

    return out


RACER_STATS_BASE_FEATURE_SUFFIXES = [
    "races_prior", "win_rate_prior", "place_rate_prior",
    "avg_st_prior", "course_place_rate_prior", "course_avg_st_prior",
]
RACER_STATS_FORM_FEATURE_SUFFIXES = [
    "recent5_races_prior", "recent5_place_rate_prior",
]
# 後方互換: 既存コードがRACER_STATS_FEATURE_SUFFIXES(全部入り)を参照している箇所向け
RACER_STATS_FEATURE_SUFFIXES = RACER_STATS_BASE_FEATURE_SUFFIXES + RACER_STATS_FORM_FEATURE_SUFFIXES

MOTOR_STATS_FEATURE_SUFFIXES = [
    "recent8_races_prior", "recent8_place_rate_prior",
]

# 2026-07-26追加: 「今節(このシリーズ)これまでの成績」特徴量。
# scripts/join_meeting_so_far_features.py で trifecta_train_by_venue/{venue}.csv に
# lane{1-6}_meeting_so_far_{top3_rate,win_rate,avg_st,n} として既に列結合済み
# (raw_txt K-fileの「第N日」からmeeting_instance_idを判定し、節内の同一選手の
# それまでの成績をリーク無しexpandingで計算したもの)。
# OLS(HC1)で検証済み: meeting_so_far_top3_rate coef=+0.256, CI[0.253,0.259],
# n=941,164(全24会場)、レーン(コース)を統制しても有意。
MEETING_FORM_STATS_FEATURE_SUFFIXES = [
    "meeting_so_far_top3_rate", "meeting_so_far_win_rate",
    "meeting_so_far_avg_st", "meeting_so_far_n",
]


def _add_meeting_form_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    lane{n}_meeting_so_far_* は join_meeting_so_far_features.py で既に
    trifecta_train_by_venue/{venue}.csv に列として結合済みなので、ここでは
    (a) 列が無いデータセット(未結合)でも動くようフォールバックでNaN列を用意し、
    (b) 他の特徴量ブロックと同様、combo(1着/2着/3着候補)側にも同じ値を割り当てる
    だけでよい。
    """
    out = df.copy()

    for lane in range(1, 7):
        for suf in MEETING_FORM_STATS_FEATURE_SUFFIXES:
            col = f"lane{lane}_{suf}"
            if col not in out.columns:
                out[col] = np.nan

    for pos, pos_name in [
        ("first", "combo_first_lane"),
        ("second", "combo_second_lane"),
        ("third", "combo_third_lane"),
    ]:
        for suf in MEETING_FORM_STATS_FEATURE_SUFFIXES:
            out[f"{pos}_{suf}"] = np.nan
            for lane in range(1, 7):
                mask = out[pos_name] == lane
                out.loc[mask, f"{pos}_{suf}"] = out.loc[mask, f"lane{lane}_{suf}"]

    return out


def _get_feature_cols(
    df: pd.DataFrame,
    with_racer_stats: bool,
    include_form_features: bool = True,
    include_development_features: bool = True,
    include_meeting_form_features: bool = True,
    include_wind_course_interaction_features: bool = False,
) -> List[str]:
    candidate_cols = [
        "combo_first_lane", "combo_second_lane", "combo_third_lane",
        "wind_dir_code", "weather_code", "wind_speed_mps", "wave_cm",

        "lane1_boat", "lane1_course", "lane1_exhibit", "lane1_motor", "lane1_racer_no", "lane1_st",
        "lane2_boat", "lane2_course", "lane2_exhibit", "lane2_motor", "lane2_racer_no", "lane2_st",
        "lane3_boat", "lane3_course", "lane3_exhibit", "lane3_motor", "lane3_racer_no", "lane3_st",
        "lane4_boat", "lane4_course", "lane4_exhibit", "lane4_motor", "lane4_racer_no", "lane4_st",
        "lane5_boat", "lane5_course", "lane5_exhibit", "lane5_motor", "lane5_racer_no", "lane5_st",
        "lane6_boat", "lane6_course", "lane6_exhibit", "lane6_motor", "lane6_racer_no", "lane6_st",

        "lane1_st_inv", "lane2_st_inv", "lane3_st_inv", "lane4_st_inv", "lane5_st_inv", "lane6_st_inv",
        "lane1_exhibit_inv", "lane2_exhibit_inv", "lane3_exhibit_inv", "lane4_exhibit_inv", "lane5_exhibit_inv", "lane6_exhibit_inv",
        "lane1_course_diff", "lane2_course_diff", "lane3_course_diff", "lane4_course_diff", "lane5_course_diff", "lane6_course_diff",

        "first_st", "second_st", "third_st",
        "first_exhibit", "second_exhibit", "third_exhibit",
        "first_course", "second_course", "third_course",
        "first_motor", "second_motor", "third_motor",
        "first_boat", "second_boat", "third_boat",
        "first_racer_no", "second_racer_no", "third_racer_no",

        "first_second_st_diff", "first_third_st_diff",
        "first_second_exhibit_diff", "first_third_exhibit_diff",
    ]

    if include_development_features:
        # 2026-07-22追加: scripts/analyze_race_development_patterns.py で検証済みの
        # 「そのレース内での展示タイム・STの相対順位」(選手の通算能力を差し引いても
        # 有意に勝敗を追加予測する)。with_racer_statsの有無に関わらず使える特徴量。
        for lane in range(1, 7):
            candidate_cols.append(f"lane{lane}_exhibit_rank")
            candidate_cols.append(f"lane{lane}_st_rank")
        for pos in ["first", "second", "third"]:
            candidate_cols.append(f"{pos}_exhibit_rank")
            candidate_cols.append(f"{pos}_st_rank")

    if with_racer_stats:
        racer_suffixes = RACER_STATS_BASE_FEATURE_SUFFIXES + (
            RACER_STATS_FORM_FEATURE_SUFFIXES if include_form_features else []
        )
        for lane in range(1, 7):
            for suf in racer_suffixes:
                candidate_cols.append(f"lane{lane}_{suf}")
            if include_form_features:
                for suf in MOTOR_STATS_FEATURE_SUFFIXES:
                    candidate_cols.append(f"lane{lane}_motor_{suf}")
        for pos in ["first", "second", "third"]:
            for suf in racer_suffixes:
                candidate_cols.append(f"{pos}_{suf}")
            if include_form_features:
                for suf in MOTOR_STATS_FEATURE_SUFFIXES:
                    candidate_cols.append(f"{pos}_motor_{suf}")

    if include_meeting_form_features:
        for lane in range(1, 7):
            for suf in MEETING_FORM_STATS_FEATURE_SUFFIXES:
                candidate_cols.append(f"lane{lane}_{suf}")
        for pos in ["first", "second", "third"]:
            for suf in MEETING_FORM_STATS_FEATURE_SUFFIXES:
                candidate_cols.append(f"{pos}_{suf}")

    if include_wind_course_interaction_features:
        for lane in range(1, 7):
            candidate_cols.append(f"lane{lane}_course_wind_interaction")
            candidate_cols.append(f"lane{lane}_course_wave_interaction")
        for pos in ["first", "second", "third"]:
            candidate_cols.append(f"{pos}_course_wind_interaction")
            candidate_cols.append(f"{pos}_course_wave_interaction")

    return [c for c in candidate_cols if c in df.columns]


def _prepare_xy(
    df: pd.DataFrame,
    with_racer_stats: bool,
    include_form_features: bool = True,
    include_development_features: bool = True,
    include_meeting_form_features: bool = True,
    include_wind_course_interaction_features: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if "y" not in df.columns:
        raise ValueError("dataset must contain y column")

    work = _add_feature_block(df.copy())
    if include_meeting_form_features:
        work = _add_meeting_form_block(work)
    feature_cols = _get_feature_cols(
        work, with_racer_stats,
        include_form_features=include_form_features,
        include_development_features=include_development_features,
        include_meeting_form_features=include_meeting_form_features,
        include_wind_course_interaction_features=include_wind_course_interaction_features,
    )
    if not feature_cols:
        raise RuntimeError("no feature columns found")

    x = work[feature_cols].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    y = pd.to_numeric(work["y"], errors="coerce").fillna(0).astype(int)
    return x, y, feature_cols


def _split_by_date(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    行単位のランダム分割(stratify付きtrain_test_split)は、同じレースの
    組み合わせ行が学習/検証にまたがって混ざるため、実質的にどのレースも
    「答えを見たことがある」状態になりリークする。日付でスパッと分割し、
    検証期間のレースを完全に未知にする。
    """
    dates = sorted(df["date"].astype(str).unique())
    n = len(dates)
    if n < 5:
        raise RuntimeError("date count too small for split")

    cutoff_idx = max(1, int(n * (1.0 - test_size)))
    train_dates = set(dates[:cutoff_idx])
    valid_dates = set(dates[cutoff_idx:])

    train_df = df[df["date"].astype(str).isin(train_dates)].copy()
    valid_df = df[df["date"].astype(str).isin(valid_dates)].copy()

    print(f"split by date: train={min(train_dates)}..{max(train_dates)} ({len(train_dates)}日) "
          f"valid={min(valid_dates)}..{max(valid_dates)} ({len(valid_dates)}日)")

    return train_df, valid_df


def _build_save_paths(
    model_dir: Path, venue: str, weight_suffix: str = "", model_suffix: str = ""
) -> Tuple[Path, Path]:
    # model_suffix: 2026-07-21追加。選手の直近の調子/モーター直近の調子を
    # 追加した実験版を、検証が済むまで本番の"_with_racer_stats"モデルの
    # 上書きから守るためのもの(例: "_formv2")。空文字なら従来通り。
    model_path = model_dir / f"trifecta_binary_catboost_{venue}_with_racer_stats{model_suffix}{weight_suffix}.cbm"
    meta_path = model_dir / f"trifecta_binary_catboost_{venue}_with_racer_stats{model_suffix}{weight_suffix}_meta.json"
    return model_path, meta_path


def _build_feature_weights(
    feature_cols: List[str],
    weight_targets: List[str],
    multiplier: float,
) -> List[float]:
    """
    2026-07-16追加: 「選手の勝率(win_rate_prior)の比重を重くしたら的中率が
    上がるか」というユーザーの疑問を検証するための仕組み。

    注意: CatBoostなどの決定木ベースのモデルは分割点(閾値)で学習するため、
    特徴量の生の値をスケーリングしても木の分割自体は変わらない
    (例: win_rate>0.35 という分割は win_rate*2>0.70 と等価で情報量は同じ)。
    したがって「特徴量の値に定数を掛ける」やり方では的中率は一切変わらない。
    本当に重みを変えたいなら、CatBoostが学習時に特徴量ごとの分割探索の
    優先度を変えられる feature_weights パラメータ(Pool/fit時の重み)を使う
    必要がある。これはヒューリスティックな重みなので「黄金比」のような
    解析的な最適値は無く、複数の倍率で再学習して眞のholdoutで
    hit_rate/ROIを比較する(=グリッドサーチ)以外に見つける方法は無い。

    weight_targets に含まれる文字列を feature名が含む場合、multiplierを
    掛ける(例: weight_targets=["win_rate_prior"], multiplier=4.0 なら
    lane1_win_rate_prior 等の重みが4倍になり、他は1.0のまま)。
    """
    weights: List[float] = []
    for col in feature_cols:
        if any(target in col for target in weight_targets):
            weights.append(float(multiplier))
        else:
            weights.append(1.0)
    return weights


def _train_one_venue(
    src: Path,
    venue: str,
    racer_stats: pd.DataFrame,
    test_size: float,
    random_state: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    verbose_eval: int,
    skip_baseline_rerun: bool = False,
    feature_weight_targets: List[str] | None = None,
    feature_weight_multiplier: float = 1.0,
    motor_stats: pd.DataFrame | None = None,
    model_suffix: str = "",
    include_form_features: bool = True,
    include_development_features: bool = True,
    include_meeting_form_features: bool = True,
    include_wind_course_interaction_features: bool = False,
) -> None:
    venue = normalize_venue_name(venue)
    print("\n" + "=" * 80)
    print("TRAIN VENUE (with racer stats):", venue)
    print("=" * 80)

    df = _load_dataset_for_venue(src, venue)
    print("rows           :", len(df))
    if "date" in df.columns:
        print("min date       :", str(df["date"].astype(str).min()))
        print("max date       :", str(df["date"].astype(str).max()))

    df = _add_racer_stats_block(df, racer_stats)
    df = _add_motor_stats_block(df, motor_stats if motor_stats is not None else pd.DataFrame())
    df["date"] = df["date"].astype(str)

    train_raw, valid_raw = _split_by_date(df, test_size)

    variants = [(True, "with_racer_stats")]
    if not skip_baseline_rerun:
        variants.insert(0, (False, "baseline_rerun"))

    baseline_metrics = None

    for with_racer_stats, suffix in variants:
        x_train, y_train, feature_cols = _prepare_xy(
            train_raw, with_racer_stats,
            include_form_features=include_form_features,
            include_development_features=include_development_features,
            include_meeting_form_features=include_meeting_form_features,
            include_wind_course_interaction_features=include_wind_course_interaction_features,
        )
        x_valid, y_valid, _ = _prepare_xy(
            valid_raw, with_racer_stats,
            include_form_features=include_form_features,
            include_development_features=include_development_features,
            include_meeting_form_features=include_meeting_form_features,
            include_wind_course_interaction_features=include_wind_course_interaction_features,
        )

        pos_count = int((y_train == 1).sum() + (y_valid == 1).sum())
        neg_count = int((y_train == 0).sum() + (y_valid == 0).sum())
        if pos_count == 0 or neg_count == 0:
            raise RuntimeError(f"class balance invalid for venue: {venue}")

        pos_train = int((y_train == 1).sum())
        neg_train = int((y_train == 0).sum())
        scale_pos_weight = float(neg_train / max(pos_train, 1))

        print(f"\n--- variant: {suffix} (feature_count={len(feature_cols)}) ---")

        # 選手成績特徴量(主にwin_rate_prior系)の重みを上げる実験用。
        # デフォルト(multiplier=1.0, targets未指定)では全特徴量が重み1.0のままで、
        # 通常学習と完全に同じ挙動になる(後方互換)。
        feature_weights = None
        if with_racer_stats and feature_weight_targets and feature_weight_multiplier != 1.0:
            feature_weights = _build_feature_weights(
                feature_cols, feature_weight_targets, feature_weight_multiplier
            )
            print(
                f"[{suffix}] feature_weights適用: targets={feature_weight_targets} "
                f"multiplier={feature_weight_multiplier}"
            )

        catboost_kwargs: Dict[str, Any] = dict(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=random_state,
            verbose=verbose_eval,
            scale_pos_weight=scale_pos_weight,
            allow_writing_files=False,
        )
        if feature_weights is not None:
            catboost_kwargs["feature_weights"] = feature_weights

        model = CatBoostClassifier(**catboost_kwargs)

        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)

        valid_pred = model.predict_proba(x_valid)[:, 1]
        valid_logloss = float(log_loss(y_valid, valid_pred))
        try:
            valid_auc = float(roc_auc_score(y_valid, valid_pred))
        except Exception:
            valid_auc = 0.0

        print(f"[{suffix}] valid logloss: {valid_logloss:.6f}  valid auc: {valid_auc:.6f}")

        if suffix == "baseline_rerun":
            baseline_metrics = {"valid_logloss": valid_logloss, "valid_auc": valid_auc}

        if suffix == "with_racer_stats":
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            weight_suffix = ""
            if feature_weights is not None:
                # 本番の"_with_racer_stats"モデルを上書きしないよう、
                # 重み実験のモデルは別名で保存する(例: _w4.0x)
                weight_suffix = f"_w{feature_weight_multiplier:g}x"

            model_path, meta_path = _build_save_paths(
                MODEL_DIR, venue, weight_suffix=weight_suffix, model_suffix=model_suffix
            )
            model.save_model(str(model_path))

            meta = {
                "venue": venue,
                "model_type": "catboost_binary",
                "with_racer_stats": True,
                "with_recent_form_features": any("recent5_place_rate_prior" in c for c in feature_cols),
                "with_motor_stats": any("motor_recent8_place_rate_prior" in c for c in feature_cols),
                "with_development_features": any("exhibit_rank" in c for c in feature_cols),
                "with_meeting_form_features": any("meeting_so_far" in c for c in feature_cols),
                "with_wind_course_interaction_features": any(
                    "course_wind_interaction" in c for c in feature_cols
                ),
                "model_suffix": model_suffix,
                "feature_weight_targets": feature_weight_targets if feature_weights is not None else None,
                "feature_weight_multiplier": feature_weight_multiplier if feature_weights is not None else 1.0,
                "dataset_path": str(src),
                "rows_total": int(len(df)),
                "rows_train": int(len(x_train)),
                "rows_valid": int(len(x_valid)),
                "positive_total": pos_count,
                "negative_total": neg_count,
                "feature_cols": feature_cols,
                "params": {
                    "iterations": iterations, "depth": depth, "learning_rate": learning_rate,
                    "random_state": random_state, "test_size": test_size,
                    "scale_pos_weight": scale_pos_weight,
                },
                "metrics": {"valid_logloss": valid_logloss, "valid_auc": valid_auc},
                "baseline_rerun_metrics_same_split": baseline_metrics,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print("saved model:", model_path)
            print("saved meta :", meta_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DATASET_PATH))
    parser.add_argument("--venue", default="", help="会場名(例: 戸田)")
    parser.add_argument("--all", action="store_true", help="全会場学習")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--verbose_eval", type=int, default=50)
    parser.add_argument(
        "--skip_baseline_rerun", action="store_true",
        help="比較用のbaseline再学習をスキップし、with_racer_stats版だけ学習する(--all時は時間半減のため推奨)",
    )
    parser.add_argument(
        "--feature_weight_targets", default="",
        help="重みを上げたい特徴量名の部分一致文字列をカンマ区切りで指定(例: win_rate_prior)。"
             "未指定なら通常学習(重み1.0)と完全に同じ。",
    )
    parser.add_argument(
        "--feature_weight_multiplier", type=float, default=1.0,
        help="--feature_weight_targetsに一致する特徴量へ掛ける重み倍率(例: 4.0)。"
             "1.0なら無効(通常学習と同じ)。'黄金比'は解析的に求まらないため、"
             "複数の値で学習しscripts/sweep_racer_stats_weight.pyで比較する想定。",
    )
    parser.add_argument(
        "--model_suffix", default="",
        help="保存するモデルファイル名に追加するsuffix(例: _formv2)。"
             "検証用に本番の_with_racer_stats.cbmを上書きせず別名で保存したい場合に指定する。"
             "未指定なら従来と完全に同じファイル名(本番ファイルを上書きするので注意)。",
    )
    parser.add_argument(
        "--disable_form_features", action="store_true",
        help="選手の直近5走複勝率・モーターの直近8走複勝率(2026-07-21追加の新特徴量)を"
             "学習から除外する。より長い--test_sizeで『新特徴量を追加する前の本番モデル相当』を"
             "同じ日付分割・同じ現在のデータセットで再現し、公平にROI比較したい場合に使う"
             "(本番の_with_racer_statsモデルは古いtest_sizeで学習済みのため、単にgenerate_predictions側の"
             "test_sizeだけを大きくすると学習済み期間をholdoutとして採点してしまいリークする。"
             "このフラグ+同じ--test_sizeで両方を再学習すれば、リーク無しで長い期間を比較できる)。",
    )
    parser.add_argument(
        "--disable_development_features", action="store_true",
        help="レース内の展示タイム順位・ST順位(2026-07-22追加、"
             "scripts/analyze_race_development_patterns.py参照)を学習から除外する。"
             "--disable_form_featuresと同様、本番モデル相当を再現して公平比較したい場合に使う。",
    )
    parser.add_argument(
        "--disable_meeting_form_features", action="store_true",
        help="「今節(このシリーズ)これまでの成績」特徴量(2026-07-26追加、"
             "scripts/join_meeting_so_far_features.py で事前結合済みのlane{1-6}_meeting_so_far_*)"
             "を学習から除外する。--disable_form_featuresと同様、本番モデル相当を再現して"
             "公平比較したい場合に使う。",
    )
    parser.add_argument(
        "--enable_wind_course_interaction_features", action="store_true",
        help="コース×風速/波高の交互作用特徴量(2026-07-26追加、丸亀のOLS検証で"
             "1号艇の勝率/複勝率が風速上昇で有意に低下することを確認: "
             "win coef=-0.0164 [95%CI -0.0226,-0.0102] n=7509, "
             "top3 coef=-0.0085 [95%CI -0.0132,-0.0037])を学習に追加する。"
             "他の特徴量と違いデフォルトでは無効(まだROIホールドアウト検証未実施のため)。"
             "検証用に付けたい時だけこのフラグを指定する。",
    )
    args = parser.parse_args()

    feature_weight_targets = [s.strip() for s in args.feature_weight_targets.split(",") if s.strip()]

    src = Path(args.src)
    racer_stats = _load_racer_stats()
    print("racer_stats rows:", len(racer_stats))
    motor_stats = _load_motor_stats()
    print("motor_stats rows:", len(motor_stats))

    if args.all:
        venues = list(VENUE_ORDER)
        skip_baseline_rerun = True if not args.skip_baseline_rerun else args.skip_baseline_rerun
        # --all はデフォルトでbaseline再学習をスキップ(時間短縮)。
        # 明示的に比較したい場合は個別に --venue で実行してください。
    elif args.venue:
        venues = [args.venue]
        skip_baseline_rerun = args.skip_baseline_rerun
    else:
        raise SystemExit("--venue か --all を指定してください")

    for v in venues:
        try:
            _train_one_venue(
                src=src, venue=v, racer_stats=racer_stats,
                test_size=args.test_size, random_state=args.random_state,
                iterations=args.iterations, depth=args.depth,
                learning_rate=args.learning_rate, verbose_eval=args.verbose_eval,
                skip_baseline_rerun=skip_baseline_rerun,
                feature_weight_targets=feature_weight_targets,
                feature_weight_multiplier=args.feature_weight_multiplier,
                motor_stats=motor_stats,
                model_suffix=args.model_suffix,
                include_form_features=not args.disable_form_features,
                include_development_features=not args.disable_development_features,
                include_meeting_form_features=not args.disable_meeting_form_features,
                include_wind_course_interaction_features=args.enable_wind_course_interaction_features,
            )
        except Exception as e:
            print(f"[ERROR] venue={v}: {e}")


if __name__ == "__main__":
    main()
