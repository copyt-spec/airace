# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from engine.venue_registry import VENUE_ORDER, normalize_venue_name

RACER_STATS_PATH = Path("data/datasets/racer_point_in_time_stats.csv")
MOTOR_STATS_PATH = Path("data/datasets/motor_point_in_time_stats.csv")

# 2026-07-24追加: RACER_STATS_PATH/MOTOR_STATS_PATHは選手/モーター×レース日ごとの
# 全履歴(それぞれ208MB・19MB)で、GitHubの1ファイル100MB上限もありコミットできず
# gitignore対象のまま。このためRender環境にはこのデータが一切届いておらず、
# ローカルとRenderで確率算出結果が食い違う原因になっていた。
# ライブ推論では実際には「選手/モーターごとに一番新しい日付の行」しか使わないため、
# scripts/build_racer_motor_stats_latest_snapshot.py で事前に抽出した軽量版
# (数百KB程度、git管理下)をRender等フル履歴が無い環境向けのフォールバックとして使う。
RACER_STATS_LATEST_PATH = Path("data/datasets/racer_point_in_time_stats_latest.csv")
MOTOR_STATS_LATEST_PATH = Path("data/datasets/motor_point_in_time_stats_latest.csv")

RACER_STATS_FEATURE_SUFFIXES = [
    "races_prior", "win_rate_prior", "place_rate_prior",
    "avg_st_prior", "course_place_rate_prior", "course_avg_st_prior",
    "recent5_races_prior", "recent5_place_rate_prior",
]

MOTOR_STATS_FEATURE_SUFFIXES = [
    "recent8_races_prior", "recent8_place_rate_prior",
]

# 2026-07-26追加: 「今節(このシリーズ)これまでの成績」特徴量のライブ版。
# 学習側(scripts/train_binary_catboost_per_venue_with_racer_stats.py)はraw_txt
# K-fileの「第N日」から節を判定できるが、進行中の開催にはK-fileがまだ無いため、
# ライブでは代わりに「前日との日付連続性」(節は必ず連続した暦日で開催される前提)
# で節の開始日を推定する。節内の各選手の成績は、既存の2つのログファイルの
# 組み合わせから再構成する:
#   - data/logs/predictions_lane_racer_no.csv: レースごとのレーン→選手番号
#     (app/main.pyがfullモードでページ表示するたびに記録、2026-07-26時点で
#      2026-07-01以降946行・22会場分。全レースを網羅していない可能性がある点に注意)
#   - data/logs/results.csv: レースごとの着順(actual_combo、レーン単位)
# STは上記どちらにも含まれていないため、avg_stは常にNaN(→0埋め)のまま。
# どちらのファイルもRenderには無い(永続ディスク無し、[[boat_ai_results_auto_update]]
# 参照)ため、その環境では本特徴量は常に0(=データ無し扱い)に自動的にフォールバック
# する(既存のracer_stats/motor_statsの「ファイルが無ければ0埋め」と同じ設計)。
ENTRIES_LOG_PATH = Path("data/logs/predictions_lane_racer_no.csv")
RESULTS_LOG_PATH = Path("data/logs/results.csv")

MEETING_FORM_STATS_FEATURE_SUFFIXES = [
    "meeting_so_far_top3_rate", "meeting_so_far_win_rate",
    "meeting_so_far_avg_st", "meeting_so_far_n",
]


class BinaryCatBoostVenueModel:
    def __init__(
        self,
        model_dir: str = "data/models",
        debug: bool = False,
        fallback_venue: str = "丸亀",
        use_racer_stats: bool = True,
    ) -> None:
        from catboost import CatBoostClassifier

        self.model_dir = Path(model_dir)
        self.debug = debug
        self.fallback_venue = normalize_venue_name(fallback_venue)
        self.CatBoostClassifier = CatBoostClassifier
        self.use_racer_stats = use_racer_stats

        self.models: Dict[str, Any] = {}
        self.metas: Dict[str, Dict[str, Any]] = {}
        self.model_variant: Dict[str, str] = {}

        for venue in VENUE_ORDER:
            model, meta, variant = self._load_venue_model(venue)
            if model is None:
                if self.debug:
                    print(f"[SKIP_MODEL] venue={venue} model/meta missing")
                continue
            self.models[venue] = model
            self.metas[venue] = meta
            self.model_variant[venue] = variant

        if self.fallback_venue not in self.models:
            if not self.models:
                raise FileNotFoundError("No venue model files found in model_dir.")
            self.fallback_venue = list(self.models.keys())[0]

        # 選手成績特徴量(2026-07-15追加の with_racer_stats モデル用)。
        # racer_point_in_time_stats.csv は「その選手が実際にレースをした日」だけの
        # 行しか持たないため、ライブ推論(未来の日付)では日付の完全一致では
        # 引けない。ここでは選手ごとに「持っているデータの中で一番新しい行」を
        # 現時点の推定値として使う(直近レース分だけ反映が遅れるが実用上十分な近似)。
        # このファイルは定期的に再生成(build_racer_history.py →
        # build_racer_point_in_time_stats.py)しないと古くなっていく点に注意。
        self.racer_stats_latest: pd.DataFrame | None = None
        self._racer_stats_mtime: float = 0.0
        if self.use_racer_stats:
            self._reload_racer_stats_if_changed()

        # モーター直近成績特徴量(2026-07-21追加)。racer_statsと同じ理由で、
        # (venue, motor)ごとに「持っているデータの中で一番新しい行」を
        # 現時点の推定値として使う。motor_point_in_time_stats.csvが無ければ
        # 空のまま(その場合はwith_motor_stats対応モデルでも0埋めで動く)。
        self.motor_stats_latest: pd.DataFrame | None = None
        self._motor_stats_mtime: float = 0.0
        if self.use_racer_stats:
            self._reload_motor_stats_if_changed()

        # 今節成績特徴量のライブ版(2026-07-26追加)。data/logs/predictions_lane_racer_no.csv
        # とdata/logs/results.csvの両方のmtimeを見て、どちらかが変わったら再構成する。
        # どちらか一方でも無ければNone(=呼び出し側は全部0埋めにフォールバック)。
        self.meeting_history_long: pd.DataFrame | None = None
        self.meeting_raced_dates: Dict[str, List[str]] = {}
        self._meeting_entries_mtime: float = 0.0
        self._meeting_results_mtime: float = 0.0
        self._reload_meeting_history_if_changed()

        if self.debug:
            print("[LOADED_MODELS]", list(self.models.keys()))
            print("[MODEL_VARIANT]", self.model_variant)
            print("[FALLBACK_VENUE]", self.fallback_venue)
            if self.racer_stats_latest is not None:
                print("[RACER_STATS_LOADED]", len(self.racer_stats_latest), "racers")
            if self.motor_stats_latest is not None:
                print("[MOTOR_STATS_LOADED]", len(self.motor_stats_latest), "venue x motor pairs")
            if self.meeting_history_long is not None:
                print("[MEETING_HISTORY_LOADED]", len(self.meeting_history_long), "lane-race rows")

    def _load_venue_model(self, venue: str):
        """with_racer_stats モデルを優先して読み込み、無ければ with_racer_no にフォールバックする。"""
        if self.use_racer_stats:
            model_path = self.model_dir / f"trifecta_binary_catboost_{venue}_with_racer_stats.cbm"
            meta_path = self.model_dir / f"trifecta_binary_catboost_{venue}_with_racer_stats_meta.json"
            if model_path.exists() and meta_path.exists():
                model = self.CatBoostClassifier()
                model.load_model(str(model_path))
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return model, meta, "with_racer_stats"
            if self.debug:
                print(f"[FALLBACK_TO_RACER_NO] venue={venue} with_racer_stats model missing")

        model_path = self.model_dir / f"trifecta_binary_catboost_{venue}_with_racer_no.cbm"
        meta_path = self.model_dir / f"trifecta_binary_catboost_{venue}_with_racer_no_meta.json"
        if not model_path.exists() or not meta_path.exists():
            return None, None, ""

        model = self.CatBoostClassifier()
        model.load_model(str(model_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return model, meta, "with_racer_no"

    def _load_latest_racer_stats(self) -> pd.DataFrame:
        if RACER_STATS_PATH.exists():
            stats = pd.read_csv(RACER_STATS_PATH, low_memory=False)
            stats["date"] = stats["date"].astype(str).str.zfill(8)
            stats["racer_no"] = pd.to_numeric(stats["racer_no"], errors="coerce").fillna(0).astype(int)

            # 選手ごとに最新日付の行だけを残す(直近実績を「現時点の推定値」として使う)
            stats = stats.sort_values(["racer_no", "date"])
            latest = stats.groupby("racer_no", as_index=False).tail(1)
            return latest.set_index("racer_no")

        if RACER_STATS_LATEST_PATH.exists():
            # フル履歴ファイルが無い環境(Render等)向けの軽量フォールバック。
            # scripts/build_racer_motor_stats_latest_snapshot.py で事前計算済みの
            # 「選手ごとの最新1行」だけを使う(ローカルの上のパスと同じ形になる)。
            if self.debug:
                print(f"[INFO] using latest-only snapshot: {RACER_STATS_LATEST_PATH}")
            latest = pd.read_csv(RACER_STATS_LATEST_PATH, low_memory=False)
            latest["date"] = latest["date"].astype(str).str.zfill(8)
            latest["racer_no"] = pd.to_numeric(latest["racer_no"], errors="coerce").fillna(0).astype(int)
            return latest.set_index("racer_no")

        if self.debug:
            print(f"[WARN] racer stats file not found: {RACER_STATS_PATH} / {RACER_STATS_LATEST_PATH}. with_racer_stats特徴量は0で埋められます。")
        return pd.DataFrame()

    def _active_racer_stats_path(self) -> Path | None:
        if RACER_STATS_PATH.exists():
            return RACER_STATS_PATH
        if RACER_STATS_LATEST_PATH.exists():
            return RACER_STATS_LATEST_PATH
        return None

    def _reload_racer_stats_if_changed(self) -> None:
        """racer_point_in_time_stats(.csv/_latest.csv)の更新時刻をチェックし、
        変わっていれば再読み込みする。

        アプリを再起動しなくても、scripts/build_racer_history.py →
        scripts/build_racer_point_in_time_stats.py を定期実行(スケジュール)で
        更新するだけで反映されるようにするため。ファイルのstatのみの軽量チェックなので
        predict_probaのたびに呼んでも問題ない。
        """
        try:
            active_path = self._active_racer_stats_path()
            if active_path is None:
                return
            mtime = active_path.stat().st_mtime
        except Exception:
            return

        if mtime == self._racer_stats_mtime and self.racer_stats_latest is not None:
            return

        if self.debug:
            print(f"[RELOAD_RACER_STATS] mtime changed ({self._racer_stats_mtime} -> {mtime}), reloading")

        self.racer_stats_latest = self._load_latest_racer_stats()
        self._racer_stats_mtime = mtime

    def _load_latest_motor_stats(self) -> pd.DataFrame:
        if MOTOR_STATS_PATH.exists():
            stats = pd.read_csv(MOTOR_STATS_PATH, low_memory=False)
            stats["date"] = stats["date"].astype(str).str.zfill(8)
            stats["motor"] = pd.to_numeric(stats["motor"], errors="coerce").fillna(0).astype(int)

            # (venue, motor)ごとに最新日付の行だけを残す(直近実績を「現時点の推定値」として使う)
            stats = stats.sort_values(["venue", "motor", "date"])
            latest = stats.groupby(["venue", "motor"], as_index=False).tail(1)
            return latest.set_index(["venue", "motor"])

        if MOTOR_STATS_LATEST_PATH.exists():
            # フル履歴ファイルが無い環境(Render等)向けの軽量フォールバック。
            if self.debug:
                print(f"[INFO] using latest-only snapshot: {MOTOR_STATS_LATEST_PATH}")
            latest = pd.read_csv(MOTOR_STATS_LATEST_PATH, low_memory=False)
            latest["date"] = latest["date"].astype(str).str.zfill(8)
            latest["motor"] = pd.to_numeric(latest["motor"], errors="coerce").fillna(0).astype(int)
            return latest.set_index(["venue", "motor"])

        if self.debug:
            print(f"[WARN] motor stats file not found: {MOTOR_STATS_PATH} / {MOTOR_STATS_LATEST_PATH}. motor特徴量は0で埋められます。")
        return pd.DataFrame()

    def _active_motor_stats_path(self) -> Path | None:
        if MOTOR_STATS_PATH.exists():
            return MOTOR_STATS_PATH
        if MOTOR_STATS_LATEST_PATH.exists():
            return MOTOR_STATS_LATEST_PATH
        return None

    def _reload_motor_stats_if_changed(self) -> None:
        """motor_point_in_time_stats(.csv/_latest.csv)の更新時刻をチェックし、
        変わっていれば再読み込みする。

        _reload_racer_stats_if_changed()と同じ設計(mtimeのみの軽量チェック)。
        """
        try:
            active_path = self._active_motor_stats_path()
            if active_path is None:
                return
            mtime = active_path.stat().st_mtime
        except Exception:
            return

        if mtime == self._motor_stats_mtime and self.motor_stats_latest is not None:
            return

        if self.debug:
            print(f"[RELOAD_MOTOR_STATS] mtime changed ({self._motor_stats_mtime} -> {mtime}), reloading")

        self.motor_stats_latest = self._load_latest_motor_stats()
        self._motor_stats_mtime = mtime

    def _load_meeting_history(self) -> pd.DataFrame:
        """
        predictions_lane_racer_no.csv(レーン→選手番号)とresults.csv(着順)を
        (date, venue, race_no)で結合し、lane単位の「そのレースでtop3/勝ちだったか」
        のロング形式テーブルを作る。scripts/build_meeting_so_far_all_venues.pyの
        ライブ簡易版(raw_txtの代わりにこの2つのログを使う)。

        どちらかのファイルが無ければ空のDataFrameを返す(呼び出し側は0埋めにフォールバック)。
        """
        if not ENTRIES_LOG_PATH.exists() or not RESULTS_LOG_PATH.exists():
            if self.debug:
                print(f"[WARN] meeting history用ログが見つかりません: "
                      f"{ENTRIES_LOG_PATH}(exists={ENTRIES_LOG_PATH.exists()}) / "
                      f"{RESULTS_LOG_PATH}(exists={RESULTS_LOG_PATH.exists()})。"
                      "今節成績のライブ計算は0で埋められます。")
            return pd.DataFrame()

        try:
            entries = pd.read_csv(ENTRIES_LOG_PATH, low_memory=False)
            results = pd.read_csv(RESULTS_LOG_PATH, low_memory=False)
        except Exception as e:
            if self.debug:
                print(f"[WARN] meeting history読み込み失敗: {e}")
            return pd.DataFrame()

        if entries.empty or results.empty:
            return pd.DataFrame()

        entries["date"] = entries["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
        entries["venue"] = entries["venue"].astype(str).map(normalize_venue_name)
        entries["race_no"] = pd.to_numeric(entries["race_no"], errors="coerce").fillna(0).astype(int)
        # 同一レースが複数回記録されている場合、一番新しいログを使う
        if "logged_at" in entries.columns:
            entries = entries.sort_values("logged_at").drop_duplicates(
                subset=["date", "venue", "race_no"], keep="last"
            )
        else:
            entries = entries.drop_duplicates(subset=["date", "venue", "race_no"], keep="last")

        results = results.copy()
        results["date"] = results["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
        results["venue"] = results["venue"].astype(str).map(normalize_venue_name)
        results["race_no"] = pd.to_numeric(results["race_no"], errors="coerce").fillna(0).astype(int)
        results = results[["date", "venue", "race_no", "actual_combo"]].dropna(subset=["actual_combo"])
        results = results.drop_duplicates(subset=["date", "venue", "race_no"], keep="last")

        merged = entries.merge(results, on=["date", "venue", "race_no"], how="inner")
        if merged.empty:
            return pd.DataFrame()

        combo_parts = merged["actual_combo"].astype(str).str.split("-", expand=True)
        if combo_parts.shape[1] < 3:
            return pd.DataFrame()
        merged["_first_lane"] = pd.to_numeric(combo_parts[0], errors="coerce")
        merged["_second_lane"] = pd.to_numeric(combo_parts[1], errors="coerce")
        merged["_third_lane"] = pd.to_numeric(combo_parts[2], errors="coerce")

        long_rows = []
        for lane in range(1, 7):
            racer_col = f"lane{lane}_racer_no"
            if racer_col not in merged.columns:
                continue
            sub = merged[["date", "venue", "race_no", racer_col, "_first_lane", "_second_lane", "_third_lane"]].copy()
            sub = sub.rename(columns={racer_col: "racer_no"})
            sub["racer_no"] = pd.to_numeric(sub["racer_no"], errors="coerce").fillna(0).astype(int)
            sub["top3"] = (
                (sub["_first_lane"] == lane) | (sub["_second_lane"] == lane) | (sub["_third_lane"] == lane)
            ).astype(int)
            sub["win"] = (sub["_first_lane"] == lane).astype(int)
            sub = sub.drop(columns=["_first_lane", "_second_lane", "_third_lane"])
            long_rows.append(sub)

        if not long_rows:
            return pd.DataFrame()

        long_df = pd.concat(long_rows, ignore_index=True)
        long_df = long_df[long_df["racer_no"] > 0]
        long_df = long_df.sort_values(["venue", "date", "race_no"]).reset_index(drop=True)
        return long_df

    def _reload_meeting_history_if_changed(self) -> None:
        """
        predictions_lane_racer_no.csv / results.csv のどちらかのmtimeが変わったら
        今節成績ライブ計算用のテーブルを再構成する(racer_stats等と同じ軽量mtime方式)。
        """
        try:
            entries_mtime = ENTRIES_LOG_PATH.stat().st_mtime if ENTRIES_LOG_PATH.exists() else 0.0
            results_mtime = RESULTS_LOG_PATH.stat().st_mtime if RESULTS_LOG_PATH.exists() else 0.0
        except Exception:
            return

        if (
            entries_mtime == self._meeting_entries_mtime
            and results_mtime == self._meeting_results_mtime
            and self.meeting_history_long is not None
        ):
            return

        if self.debug:
            print(
                "[RELOAD_MEETING_HISTORY] mtime changed "
                f"(entries {self._meeting_entries_mtime}->{entries_mtime}, "
                f"results {self._meeting_results_mtime}->{results_mtime}), reloading"
            )

        self.meeting_history_long = self._load_meeting_history()
        self._meeting_entries_mtime = entries_mtime
        self._meeting_results_mtime = results_mtime

        self.meeting_raced_dates = {}
        if self.meeting_history_long is not None and not self.meeting_history_long.empty:
            for venue, g in self.meeting_history_long.groupby("venue"):
                self.meeting_raced_dates[venue] = sorted(g["date"].unique().tolist())

    def _compute_meeting_start_date(self, venue_name: str, target_date: str) -> str:
        """
        「節は必ず連続した暦日で開催される」という前提で、target_dateの前日から
        遡り、その会場の既知の開催日(meeting_raced_dates)と連続している限り
        遡り続けて、節の開始日を推定する。連続が途切れた時点、またはそもそも
        前日の開催記録が無ければ、target_date自身が開始日(=節1日目)とみなす。

        raw_txtのK-file「第N日」のような確定情報が無いライブ環境向けの近似。
        """
        raced_dates = set(self.meeting_raced_dates.get(venue_name, []))
        try:
            from datetime import datetime, timedelta
            cur = datetime.strptime(target_date, "%Y%m%d")
        except Exception:
            return target_date

        meeting_start = cur
        while True:
            prev_day = meeting_start - timedelta(days=1)
            prev_str = prev_day.strftime("%Y%m%d")
            if prev_str in raced_dates:
                meeting_start = prev_day
            else:
                break

        return meeting_start.strftime("%Y%m%d")

    def _add_meeting_form_block_live(self, df: pd.DataFrame, venue_name: str) -> pd.DataFrame:
        """
        今節(このシリーズ)これまでの成績特徴量のライブ版。
        scripts/train_binary_catboost_per_venue_with_racer_stats.py の
        _add_meeting_form_block()と同じ列を作るが、値は
        predictions_lane_racer_no.csv + results.csvから動的に計算する。
        avg_stは元データに無いため常にNaN(→0埋め)。

        対象日の前の節内レース(節開始日〜前日の全レース、当日は今回より前の
        race_noのみ)における、各レーンの選手番号のtop3率・勝率・n数を計算する。
        該当データが無ければ(節1日目・ログ未収集等)NaNのままにし、
        _prepare_xで0埋めされる(学習時のNaN扱いと同じ)。
        """
        out = df.copy()
        venue_name = self._normalize_venue_name(venue_name)

        for lane in range(1, 7):
            for suf in MEETING_FORM_STATS_FEATURE_SUFFIXES:
                out[f"lane{lane}_{suf}"] = np.nan

        if self.meeting_history_long is not None and not self.meeting_history_long.empty and len(out) > 0:
            try:
                target_date = str(out["date"].iloc[0])
                target_race_no = int(out["race_no"].iloc[0]) if "race_no" in out.columns else None
            except Exception:
                target_date = None
                target_race_no = None

            if target_date:
                meeting_start = self._compute_meeting_start_date(venue_name, target_date)
                hist = self.meeting_history_long
                hist_v = hist[hist["venue"] == venue_name]
                if not hist_v.empty:
                    mask_prior_days = (hist_v["date"] >= meeting_start) & (hist_v["date"] < target_date)
                    mask_today = pd.Series(False, index=hist_v.index)
                    if target_race_no is not None:
                        mask_today = (hist_v["date"] == target_date) & (hist_v["race_no"] < target_race_no)
                    so_far = hist_v[mask_prior_days | mask_today]

                    if not so_far.empty:
                        agg = so_far.groupby("racer_no").agg(
                            n=("top3", "size"),
                            top3_rate=("top3", "mean"),
                            win_rate=("win", "mean"),
                        )
                        for lane in range(1, 7):
                            racer_col = f"lane{lane}_racer_no"
                            if racer_col not in out.columns:
                                continue
                            racer_nos = pd.to_numeric(out[racer_col], errors="coerce").fillna(0).astype(int)
                            joined = racer_nos.map(
                                lambda rno: agg.loc[rno] if rno in agg.index else None
                            )
                            out[f"lane{lane}_meeting_so_far_n"] = [
                                (r["n"] if r is not None else np.nan) for r in joined
                            ]
                            out[f"lane{lane}_meeting_so_far_top3_rate"] = [
                                (r["top3_rate"] if r is not None else np.nan) for r in joined
                            ]
                            out[f"lane{lane}_meeting_so_far_win_rate"] = [
                                (r["win_rate"] if r is not None else np.nan) for r in joined
                            ]
                            # avg_stは元データに無いため常にNaN(既に上でセット済み)

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

    def _safe_float(self, v: Any, default: float = 0.0) -> float:
        try:
            if v is None or v == "":
                return default
            return float(v)
        except Exception:
            return default

    def _normalize_venue_name(self, venue_name: str) -> str:
        return normalize_venue_name(venue_name)

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

    def _resolve_model_venue(self, venue_name: str) -> str:
        venue_name = self._normalize_venue_name(venue_name)
        if venue_name in self.models:
            return venue_name
        return self.fallback_venue

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

        # 2026-07-22追加: 「展開予想」の要因(scripts/analyze_race_development_patterns.py参照)。
        # そのレース内での展示タイム・STの相対順位(1=最速/最良)。学習側
        # (scripts/train_binary_catboost_per_venue_with_racer_stats.py)と同じロジック。
        # ここでは常に計算しておき、実際に使うかどうかはロード済みモデルの
        # meta.jsonのfeature_cols(_prepare_xがそれだけを選ぶ)に委ねる。
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

        # 2026-07-26追加: 丸亀特化の検証用(ユーザーが丸亀によく行く/思い入れが
        # あることがきっかけ)。丸亀データでOLS(HC1)検証したところ、1号艇
        # (course=1)に限定した勝率は風速が強いほど有意に下がる
        # (win coef=-0.0164, 95%CI[-0.0226,-0.0102], n=7509 /
        #  top3 coef=-0.0085, CI[-0.0132,-0.0037])。
        # 学習側(scripts/train_binary_catboost_per_venue_with_racer_stats.py)
        # の_add_feature_block()と同じロジック。exhibit_rank/st_rankと同様、
        # ここでは常に計算しておき、実際に使うかはロード済みモデルのmeta.jsonの
        # feature_cols(_prepare_xがそれだけを選ぶ)に委ねる。まだROIホールド
        # アウト検証前の特徴量のため、この列を使うモデルは本番投入していない
        # (checked: with_wind_course_interaction_features相当のフラグは
        # まだどのmeta.jsonにも立っていない)。
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

    def _add_racer_stats_block_live(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if self.racer_stats_latest is None or self.racer_stats_latest.empty:
            for lane in range(1, 7):
                for suf in RACER_STATS_FEATURE_SUFFIXES:
                    out[f"lane{lane}_{suf}"] = np.nan
        else:
            for lane in range(1, 7):
                racer_col = f"lane{lane}_racer_no"
                course_col = f"lane{lane}_course"

                racer_nos = pd.to_numeric(out.get(racer_col, 0), errors="coerce").fillna(0).astype(int)
                joined = racer_nos.map(
                    lambda rno: self.racer_stats_latest.loc[rno] if rno in self.racer_stats_latest.index else None
                )

                out[f"lane{lane}_races_prior"] = [
                    (r["races_prior"] if r is not None else np.nan) for r in joined
                ]
                out[f"lane{lane}_win_rate_prior"] = [
                    (r["win_rate_prior"] if r is not None else np.nan) for r in joined
                ]
                out[f"lane{lane}_place_rate_prior"] = [
                    (r["place_rate_prior"] if r is not None else np.nan) for r in joined
                ]
                out[f"lane{lane}_avg_st_prior"] = [
                    (r["avg_st_prior"] if r is not None else np.nan) for r in joined
                ]
                out[f"lane{lane}_recent5_races_prior"] = [
                    (r.get("recent5_races_prior", np.nan) if r is not None else np.nan) for r in joined
                ]
                out[f"lane{lane}_recent5_place_rate_prior"] = [
                    (r.get("recent5_place_rate_prior", np.nan) if r is not None else np.nan) for r in joined
                ]

                course_vals = pd.to_numeric(out.get(course_col, 0), errors="coerce").fillna(0).astype(int)
                place_rate_by_course = []
                avg_st_by_course = []
                for r, c in zip(joined, course_vals):
                    if r is None:
                        place_rate_by_course.append(np.nan)
                        avg_st_by_course.append(np.nan)
                        continue
                    c = int(c) if 1 <= int(c) <= 6 else 0
                    if c == 0:
                        place_rate_by_course.append(np.nan)
                        avg_st_by_course.append(np.nan)
                    else:
                        place_rate_by_course.append(r.get(f"course{c}_place_rate_prior", np.nan))
                        avg_st_by_course.append(r.get(f"course{c}_avg_st_prior", np.nan))

                out[f"lane{lane}_course_place_rate_prior"] = place_rate_by_course
                out[f"lane{lane}_course_avg_st_prior"] = avg_st_by_course

        for pos, pos_name in [
            ("first", "combo_first_lane"),
            ("second", "combo_second_lane"),
            ("third", "combo_third_lane"),
        ]:
            for suf in RACER_STATS_FEATURE_SUFFIXES:
                out[f"{pos}_{suf}"] = np.nan
                for lane in range(1, 7):
                    mask = out[pos_name] == lane
                    out.loc[mask, f"{pos}_{suf}"] = out.loc[mask, f"lane{lane}_{suf}"]

        return out

    def _add_motor_stats_block_live(self, df: pd.DataFrame, venue_name: str) -> pd.DataFrame:
        out = df.copy()
        venue_name = self._normalize_venue_name(venue_name)

        if self.motor_stats_latest is None or self.motor_stats_latest.empty:
            for lane in range(1, 7):
                for suf in MOTOR_STATS_FEATURE_SUFFIXES:
                    out[f"lane{lane}_motor_{suf}"] = np.nan
        else:
            for lane in range(1, 7):
                motor_col = f"lane{lane}_motor"
                motor_nos = pd.to_numeric(out.get(motor_col, 0), errors="coerce").fillna(0).astype(int)
                joined = motor_nos.map(
                    lambda mno: self.motor_stats_latest.loc[(venue_name, mno)]
                    if (venue_name, mno) in self.motor_stats_latest.index else None
                )
                for suf in MOTOR_STATS_FEATURE_SUFFIXES:
                    col = f"motor_{suf}"
                    out[f"lane{lane}_motor_{suf}"] = [
                        (r.get(col, np.nan) if r is not None else np.nan) for r in joined
                    ]

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
        if "combo" not in df120.columns:
            raise ValueError("df120 must have combo column")

        requested_venue = self._normalize_venue_name(venue_name)
        model_venue = self._resolve_model_venue(requested_venue)

        model = self.models[model_venue]
        meta = self.metas[model_venue]
        feature_cols = list(meta.get("feature_cols", []))
        variant = self.model_variant.get(model_venue, "")

        work = self._add_feature_block(df120.copy())
        if variant == "with_racer_stats":
            if self.use_racer_stats:
                self._reload_racer_stats_if_changed()
                self._reload_motor_stats_if_changed()
            work = self._add_racer_stats_block_live(work)
            work = self._add_motor_stats_block_live(work, requested_venue)
        # 2026-07-26追加: 今節成績特徴量。meta.jsonにwith_meeting_form_features=True
        # (=このモデルがmeeting_so_far列で学習されている)場合のみ計算する
        # (旧モデルには不要な列で、計算コスト・reload頻度を無駄に増やさないため)。
        if meta.get("with_meeting_form_features"):
            self._reload_meeting_history_if_changed()
            work = self._add_meeting_form_block_live(work, requested_venue)
        x = self._prepare_x(work, feature_cols)

        raw_scores = model.predict(x, prediction_type="RawFormulaVal")
        raw_scores = [self._safe_float(v, 0.0) for v in raw_scores]

        temperature = self._get_temperature_by_venue(requested_venue)
        probs = self._softmax(raw_scores, temperature=temperature)

        combos = df120["combo"].astype(str).tolist()
        prob_map = {combo: float(p) for combo, p in zip(combos, probs)}

        if self.debug:
            print(
                "[PREDICT]",
                f"requested={requested_venue}",
                f"model={model_venue}",
                f"rows={len(df120)}",
            )

        return prob_map
