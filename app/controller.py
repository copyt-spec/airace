from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from engine.generic_racelist_fetcher import (
    fetch_all_entries_once_by_venue,
    fetch_racelist_by_venue,
)
from engine.odds_fetcher import fetch_odds
from engine.beforeinfo_fetcher_venue import fetch_beforeinfo_venue
from engine.racelist_enricher import enrich_entries_with_racelist
from engine.buy_selector import select_best_bets, should_skip_race
from engine.kelly_allocator import kelly_size_selected_bets
from engine.formation_builder import build_formation_options
from engine.model_loader_catboost_binary import BinaryCatBoostVenueModel
from engine.model_loader_catboost_global_venue_auto import GlobalVenueAutoCatBoostModel
from engine.venue_registry import (
    VENUE_ORDER,
    get_jcd,
    normalize_venue_name,
)


class RaceController:
    _all_entries_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    _race_entries_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    _enriched_entries_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    _beforeinfo_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    _odds_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    _TTL_ALL_ENTRIES = 1800
    _TTL_RACE_ENTRIES = 1800
    _TTL_ENRICHED = 1800
    _TTL_BEFOREINFO = 180
    _TTL_ODDS = 180

    # 2026-07-16追加: Kellyサイジングの前提バンクロール(円)。
    # 実際の資産額はユーザーごとに違うので環境変数で上書き可能にしておく。
    # ROI%自体を上げる施策ではなく、選ばれた買い目に対してエッジの大きさに
    # 応じて賭け金を傾斜配分し、資産の成長効率を上げるためのもの。
    _BANKROLL_YEN = int(os.environ.get("BOAT_BANKROLL_YEN", "100000"))

    def __init__(
        self,
        model_mode: str | None = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.model_mode = self._resolve_model_mode(model_mode)

        self.venue_model = BinaryCatBoostVenueModel(
            model_dir="data/models",
            debug=debug,
            fallback_venue="丸亀",
        )

        # 2026-07-17修正: global_auto用のモデルファイル3点(.cbm/meta.json/
        # venue_stats.csv)がgit管理外だった環境(Renderなど)では
        # GlobalVenueAutoCatBoostModel()がFileNotFoundErrorを送出し、
        # RaceController()のコンストラクタ自体が失敗して「会場ページに入ると
        # 必ずサーバーエラー」になっていた(model_mode="venue"がデフォルトで、
        # global_auto側を一切使わない場合でも、コンストラクタで無条件に
        # 読み込もうとしていたため)。ここで失敗を握りつぶしてNoneにし、
        # 実際にglobal_autoモードが選ばれた時だけ_get_active_modelでエラーに
        # なるようにする(venueモードの利用には影響しない)。
        try:
            self.global_auto_model = GlobalVenueAutoCatBoostModel(
                model_dir="data/models",
                model_name="trifecta_binary_catboost_global_venue_auto",
                debug=debug,
            )
        except Exception as e:
            if self.debug:
                print(f"[WARN] global_auto_model load failed (falling back to venue model only): {e}")
            self.global_auto_model = None

    # =========================
    # model mode
    # =========================
    def _resolve_model_mode(self, model_mode: str | None) -> str:
        mode = str(model_mode or os.environ.get("BOAT_MODEL_MODE", "venue")).strip().lower()
        if mode not in {"venue", "global_auto"}:
            mode = "venue"
        return mode

    def set_model_mode(self, model_mode: str) -> None:
        self.model_mode = self._resolve_model_mode(model_mode)

    def get_model_mode(self) -> str:
        return self.model_mode

    def _get_active_model(self, model_mode: str | None = None):
        mode = self._resolve_model_mode(model_mode or self.model_mode)
        if mode == "global_auto" and self.global_auto_model is not None:
            return self.global_auto_model, "global_auto"
        # global_autoが選ばれたがモデル未配置(ファイル欠損)の場合は、
        # サーバーエラーにせずvenueモデルにフォールバックする。
        return self.venue_model, "venue"

    # =========================
    # cache
    # =========================
    def _cache_get(self, cache: Dict[Any, Dict[str, Any]], key: Any, ttl_sec: int) -> Optional[Any]:
        item = cache.get(key)
        if not item:
            return None

        ts = float(item.get("ts", 0.0))
        if (time.time() - ts) > ttl_sec:
            try:
                del cache[key]
            except Exception:
                pass
            return None

        return item.get("data")

    def _cache_set(self, cache: Dict[Any, Dict[str, Any]], key: Any, data: Any) -> Any:
        cache[key] = {
            "ts": time.time(),
            "data": data,
        }
        return data

    def _trim_old_cache(self) -> None:
        now = time.time()

        def trim(cache: Dict[Any, Dict[str, Any]], ttl: int) -> None:
            dead_keys = []
            for k, v in cache.items():
                ts = float(v.get("ts", 0.0))
                if (now - ts) > (ttl * 2):
                    dead_keys.append(k)
            for k in dead_keys:
                try:
                    del cache[k]
                except Exception:
                    pass

        trim(self._all_entries_cache, self._TTL_ALL_ENTRIES)
        trim(self._race_entries_cache, self._TTL_RACE_ENTRIES)
        trim(self._enriched_entries_cache, self._TTL_ENRICHED)
        trim(self._beforeinfo_cache, self._TTL_BEFOREINFO)
        trim(self._odds_cache, self._TTL_ODDS)

    # =========================
    # utils
    # =========================
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

    def _pick(self, row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        for key in keys:
            if key in row and row.get(key) not in (None, ""):
                return row.get(key)
        return default

    def _normalize_venue_name(self, venue_name: str) -> str:
        return normalize_venue_name(venue_name)

    def _dedupe_by_lane(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lane_map: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            lane = self._safe_int(row.get("lane", 0), 0)
            if lane not in range(1, 7):
                continue
            lane_map[lane] = dict(row)
        return [lane_map[l] for l in sorted(lane_map.keys())]

    def _is_valid_6boats(self, rows: List[Dict[str, Any]]) -> bool:
        lanes = [self._safe_int(x.get("lane", 0), 0) for x in rows]
        return len(rows) == 6 and sorted(lanes) == [1, 2, 3, 4, 5, 6]

    def get_supported_venues(self) -> List[str]:
        return list(VENUE_ORDER)

    # =========================
    # entries
    # =========================
    def get_all_entries(self, venue_name: str, date: str) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        venue_name = self._normalize_venue_name(venue_name)
        key = (venue_name, date)

        cached = self._cache_get(self._all_entries_cache, key, self._TTL_ALL_ENTRIES)
        if cached is not None:
            return cached

        data = fetch_all_entries_once_by_venue(venue_name=venue_name, date=date)
        data = [dict(x) for x in data]
        return self._cache_set(self._all_entries_cache, key, data)

    def get_entries_race(self, venue_name: str, date: str, race_no: int) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        venue_name = self._normalize_venue_name(venue_name)
        key = (venue_name, date, int(race_no))

        cached = self._cache_get(self._race_entries_cache, key, self._TTL_RACE_ENTRIES)
        if cached is not None:
            return cached

        # 2026-07-16修正: 以前はここで必ず get_all_entries()(=当日12レース分を
        # 4並列でまとめて取得)を呼んでから該当レースだけを抜き出していたため、
        # 1レースだけ見たい時でも12レース分のページ取得完了を待つ必要があり、
        # 「情報取得が遅い」という体感の主因になっていた。
        # 既にその日のall_entriesキャッシュが温まっていればそれを使い、
        # 無ければ該当レース1件だけを直接取得する高速パスを先に試す。
        # (取得漏れ・パース失敗時のみ、保険として全レース取得にフォールバック)
        out: List[Dict[str, Any]] = []

        all_entries_cached = self._cache_get(
            self._all_entries_cache, (venue_name, date), self._TTL_ALL_ENTRIES
        )
        if all_entries_cached is not None:
            for row in all_entries_cached:
                try:
                    if int(row.get("race_no", 0)) == int(race_no):
                        out.append(dict(row))
                except Exception:
                    continue

        if not out:
            try:
                rows = fetch_racelist_by_venue(venue_name=venue_name, race_no=race_no, date=date)
                out = [dict(x) for x in rows]
            except Exception:
                out = []

        if not out:
            # 単一レース取得も失敗した場合のみ、保険として当日全レースを取得
            all_entries = self.get_all_entries(venue_name, date)
            for row in all_entries:
                try:
                    if int(row.get("race_no", 0)) == int(race_no):
                        out.append(dict(row))
                except Exception:
                    continue

        out = self._dedupe_by_lane(out)
        out.sort(key=lambda x: int(x.get("lane", 0)))
        return self._cache_set(self._race_entries_cache, key, out)

    # =========================
    # odds
    # =========================
    def _group_odds(self, raw_odds: Dict[str, Any]) -> Dict[str, Any]:
        grouped = {i: {} for i in range(1, 7)}
        numeric_values = []

        for combo, odd in raw_odds.items():
            parts = str(combo).split("-")
            if len(parts) != 3:
                continue

            try:
                first = int(parts[0])
                second = int(parts[1])
                third = int(parts[2])
            except ValueError:
                continue

            grouped[first][(second, third)] = odd

            try:
                numeric_values.append(float(odd))
            except Exception:
                pass

        min_val = min(numeric_values) if numeric_values else 0.0
        max_val = max(numeric_values) if numeric_values else 0.0

        return {"data": grouped, "min": float(min_val), "max": float(max_val)}

    def _flat_odds_map(self, raw_odds: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for combo, odd in raw_odds.items():
            try:
                out[str(combo)] = float(odd)
            except Exception:
                continue
        return out

    def get_odds_only(self, venue_name: str, race_no: int, date: str) -> Dict[str, Any]:
        self._trim_old_cache()
        venue_name = self._normalize_venue_name(venue_name)
        key = (venue_name, date, int(race_no))

        cached = self._cache_get(self._odds_cache, key, self._TTL_ODDS)
        if cached is not None:
            return cached

        venue_code = get_jcd(venue_name)
        raw_odds = fetch_odds(race_no, date, venue_code=venue_code)
        grouped = self._group_odds(raw_odds)
        return self._cache_set(self._odds_cache, key, grouped)

    # =========================
    # beforeinfo
    # =========================
    def get_beforeinfo_only(self, venue_name: str, race_no: int, date: str) -> Dict[str, Any]:
        self._trim_old_cache()
        venue_name = self._normalize_venue_name(venue_name)
        key = (venue_name, date, int(race_no))

        cached = self._cache_get(self._beforeinfo_cache, key, self._TTL_BEFOREINFO)
        if cached is not None:
            return cached

        venue_code = get_jcd(venue_name)
        data = fetch_beforeinfo_venue(race_no, date, venue_code=venue_code)
        return self._cache_set(self._beforeinfo_cache, key, data)

    # =========================
    # enrich
    # =========================
    def enrich_entries(
        self,
        venue_name: str,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        venue_name = self._normalize_venue_name(venue_name)
        key = (venue_name, date, int(race_no))

        cached = self._cache_get(self._enriched_entries_cache, key, self._TTL_ENRICHED)
        if cached is not None:
            return cached

        venue_code = get_jcd(venue_name)
        data = enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=venue_code,
        )
        data = self._dedupe_by_lane([dict(x) for x in data])
        data.sort(key=lambda x: int(x.get("lane", 0)))
        return self._cache_set(self._enriched_entries_cache, key, data)

    # =========================
    # beforeinfo normalize
    # =========================
    def _normalize_beforeinfo(self, beforeinfo: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        lane_map: Dict[int, Dict[str, Any]] = {}

        if not isinstance(beforeinfo, dict):
            return lane_map

        rows = beforeinfo.get("entries") or beforeinfo.get("data") or beforeinfo.get("rows") or []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue

                lane = self._safe_int(
                    self._pick(row, ["lane", "艇番", "艇", "枠", "teiban"], 0),
                    0,
                )
                if lane not in range(1, 7):
                    continue

                lane_map[lane] = {
                    "st": self._safe_float(
                        self._pick(row, ["st", "ST", "start_time", "展示ST"], 0.0),
                        0.0,
                    ),
                    "exhibit": self._safe_float(
                        self._pick(row, ["exhibit", "tenji_time", "展示タイム", "展示"], 0.0),
                        0.0,
                    ),
                    "course": self._safe_int(
                        self._pick(row, ["course", "course_no", "進入"], lane),
                        lane,
                    ),
                }

        if not lane_map:
            for lane in range(1, 7):
                key = str(lane)
                if key not in beforeinfo:
                    continue
                row = beforeinfo.get(key)
                if not isinstance(row, dict):
                    continue

                lane_map[lane] = {
                    "st": self._safe_float(
                        self._pick(row, ["st", "ST", "start_time", "展示ST"], 0.0),
                        0.0,
                    ),
                    "exhibit": self._safe_float(
                        self._pick(row, ["exhibit", "tenji_time", "展示タイム", "展示"], 0.0),
                        0.0,
                    ),
                    "course": self._safe_int(
                        self._pick(row, ["course", "course_no", "進入"], lane),
                        lane,
                    ),
                }

        return lane_map

    def _extract_weather_set(self, beforeinfo: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(beforeinfo, dict):
            return {
                "weather": "",
                "wind_dir": "",
                "wind_speed_mps": 0.0,
                "wave_cm": 0.0,
            }

        return {
            "weather": str(self._pick(beforeinfo, ["weather", "天候"], "")).strip(),
            "wind_dir": str(self._pick(beforeinfo, ["wind_dir", "風向"], "")).strip(),
            "wind_speed_mps": self._safe_float(
                self._pick(beforeinfo, ["wind_speed_mps", "wind_speed", "風速"], 0.0),
                0.0,
            ),
            "wave_cm": self._safe_float(
                self._pick(beforeinfo, ["wave_cm", "wave", "波高"], 0.0),
                0.0,
            ),
        }

    # =========================
    # AI input build
    # =========================
    def _to_ai_entries(
        self,
        entries: List[Dict[str, Any]],
        beforeinfo: Dict[str, Any] | None = None,
        venue_name: str = "",
    ) -> List[Dict[str, Any]]:
        before_lane_map = self._normalize_beforeinfo(beforeinfo or {})
        venue_name = self._normalize_venue_name(venue_name)

        ai_entries: List[Dict[str, Any]] = []

        for row in entries:
            lane = self._safe_int(self._pick(row, ["lane", "艇番", "枠"], 0), 0)
            if lane not in range(1, 7):
                continue

            before_row = before_lane_map.get(lane, {})

            ai_row = {
                "lane": lane,
                "racer_no": self._safe_int(
                    self._pick(row, ["racer_no", "register_no", "登番", "登録番号"], 0),
                    0,
                ),
                "motor": self._safe_int(
                    self._pick(row, ["motor", "motor_no", "モーター", "モーター番号"], 0),
                    0,
                ),
                "boat": self._safe_int(
                    self._pick(row, ["boat", "boat_no", "ボート", "ボート番号"], 0),
                    0,
                ),
                "exhibit": self._safe_float(
                    self._pick(row, ["exhibit", "tenji_time", "展示タイム"], before_row.get("exhibit", 0.0)),
                    before_row.get("exhibit", 0.0),
                ),
                "st": self._safe_float(
                    self._pick(row, ["st", "avg_st", "ST", "平均ST", "start_timing"], before_row.get("st", 0.0)),
                    before_row.get("st", 0.0),
                ),
                "course": self._safe_int(
                    self._pick(row, ["course", "進入", "course_no"], before_row.get("course", lane)),
                    before_row.get("course", lane),
                ),
                "venue": venue_name,
            }

            ai_entries.append(ai_row)

        ai_entries = self._dedupe_by_lane(ai_entries)
        ai_entries.sort(key=lambda x: x["lane"])
        return ai_entries

    def _build_df120_from_ai_entries(
        self,
        ai_entries: List[Dict[str, Any]],
        venue_name: str,
        date: str,
        race_no: int,
        weather_set: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self._is_valid_6boats(ai_entries):
            return []

        lane_map = {int(x["lane"]): x for x in ai_entries}
        rows: List[Dict[str, Any]] = []
        venue_code = get_jcd(venue_name)

        for a in range(1, 7):
            for b in range(1, 7):
                if b == a:
                    continue
                for c in range(1, 7):
                    if c == a or c == b:
                        continue

                    combo = f"{a}-{b}-{c}"

                    row: Dict[str, Any] = {
                        "race_key": f"{venue_name}_{date}_{race_no}",
                        "date": date,
                        "venue": venue_name,
                        "venue_code": venue_code,
                        "race_no": race_no,
                        "combo": combo,
                        "combo_first_lane": a,
                        "combo_second_lane": b,
                        "combo_third_lane": c,
                        "wave_cm": self._safe_float(weather_set.get("wave_cm", 0.0), 0.0),
                        "weather": str(weather_set.get("weather", "") or ""),
                        "wind_dir": str(weather_set.get("wind_dir", "") or ""),
                        "wind_speed_mps": self._safe_float(weather_set.get("wind_speed_mps", 0.0), 0.0),
                    }

                    for lane in range(1, 7):
                        src = lane_map[lane]
                        row[f"lane{lane}_boat"] = self._safe_int(src.get("boat", 0), 0)
                        row[f"lane{lane}_course"] = self._safe_int(src.get("course", lane), lane)
                        row[f"lane{lane}_exhibit"] = self._safe_float(src.get("exhibit", 0.0), 0.0)
                        row[f"lane{lane}_motor"] = self._safe_int(src.get("motor", 0), 0)
                        row[f"lane{lane}_racer_no"] = self._safe_int(src.get("racer_no", 0), 0)
                        row[f"lane{lane}_st"] = self._safe_float(src.get("st", 0.0), 0.0)

                    rows.append(row)

        return rows

    def _prob_map_to_rows(
        self,
        prob_map: Dict[str, float],
        odds_map: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for combo, prob in prob_map.items():
            odds = 0.0
            if odds_map:
                odds = self._safe_float(odds_map.get(combo, 0.0), 0.0)

            rows.append({
                "combo": str(combo),
                "score": float(prob),
                "prob": float(prob),
                "odds": float(odds),
                "ev": float(prob) * float(odds) if odds > 0 else 0.0,
            })

        return rows

    def _select_best_bets(
        self,
        prob_map: Dict[str, float],
        odds_map: Dict[str, float] | None = None,
        *,
        top_n: int = 5,
        venue_name: str = "",
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        戻り値: (best_bets, is_skip_decision)
        is_skip_decision=True は、buy_selector.should_skip_race()による
        「意図的な見送り」(低信頼度レースを買わない戦略)が発動したことを示す。
        2026-07-23追加: この情報をUI側(見送ろ！表示)に伝えるため、単純な
        リストからタプルに変更した。
        """
        ai_preds = self._prob_map_to_rows(prob_map, odds_map=odds_map)

        best_bets = select_best_bets(
            ai_preds,
            top_n=top_n,
            venue=venue_name,
        )

        if not best_bets:
            # best_betsが空でも、それが「意図的な見送り(低信頼度レースを買わない戦略)」
            # によるものなら、素朴なフォールバックで無理に賭けてしまわないようにする。
            if should_skip_race(ai_preds, venue=venue_name):
                return [], True

            fallback = sorted(
                ai_preds,
                key=lambda x: float(x.get("score", 0.0)),
                reverse=True,
            )[:top_n]
            for i, r in enumerate(fallback, start=1):
                r["buy_rank"] = i
                r["buy_score"] = float(r.get("score", 0.0))
                r["is_best_bet"] = True
            return fallback, False

        return best_bets, False

    # =========================
    # predict
    # =========================
    def _predict_bundle(
        self,
        *,
        date: str,
        race_no: int,
        venue_name: str,
        base_entries: List[Dict[str, Any]],
        beforeinfo: Dict[str, Any],
        enriched_entries: List[Dict[str, Any]] | None,
        top_n: int,
        with_odds: bool,
        model_mode: str | None = None,
        raw_odds: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        out = {
            "best_bets": [],
            "prob_map": {},
            "odds_map": {},
            "df120_rows": [],
            "formation_options": [],
            "model_mode": self._resolve_model_mode(model_mode or self.model_mode),
            "is_skip_decision": False,
        }

        if not self._is_valid_6boats(base_entries):
            return out

        use_entries = base_entries
        if enriched_entries and self._is_valid_6boats(enriched_entries):
            use_entries = enriched_entries

        ai_entries = self._to_ai_entries(use_entries, beforeinfo, venue_name=venue_name)
        if not self._is_valid_6boats(ai_entries):
            return out

        weather_set = self._extract_weather_set(beforeinfo)

        df120_rows = self._build_df120_from_ai_entries(
            ai_entries=ai_entries,
            venue_name=venue_name,
            date=date,
            race_no=race_no,
            weather_set=weather_set,
        )
        if not df120_rows or len(df120_rows) != 120:
            return out

        import pandas as pd
        df120 = pd.DataFrame(df120_rows)

        odds_map: Dict[str, float] = {}
        if with_odds:
            try:
                # 呼び出し元(get_ai_prediction_bundle)が並列取得済みのオッズを
                # 渡してくれる場合はそれを使い、二重フェッチしない(2026-07-16改善)。
                # 渡されていない場合(単体呼び出し等)は従来通りここで取得する。
                _raw_odds = raw_odds if raw_odds is not None else fetch_odds(
                    race_no, date, venue_code=get_jcd(venue_name)
                )
                odds_map = self._flat_odds_map(_raw_odds)
            except Exception:
                odds_map = {}

        try:
            model, resolved_mode = self._get_active_model(model_mode)
            prob_map = model.predict_proba(df120, venue_name)
            out["model_mode"] = resolved_mode
        except Exception:
            prob_map = {}

        if not prob_map:
            return out

        best_bets, is_skip_decision = self._select_best_bets(
            prob_map=prob_map,
            odds_map=odds_map,
            top_n=top_n,
            venue_name=venue_name,
        )
        out["is_skip_decision"] = is_skip_decision

        # 2026-07-16追加: 選定自体(buy_selector)はそのままに、賭け金だけを
        # 単点ケリー基準で傾斜配分する。エラー時は賭け金無し(0円)の
        # 表示になるだけで、選定結果自体には影響させない(フェイルセーフ)。
        try:
            best_bets = kelly_size_selected_bets(best_bets, bankroll=self._BANKROLL_YEN)
        except Exception:
            pass

        # 2026-07-17追加: prob_map(120通りの予測確率、再推論不要)から
        # フォーメーション形式(1着/2着/3着候補の絞り込み)の案も一緒に出す。
        # buy_selectorの点買い選定とは独立な、別の見せ方の提案。
        try:
            formation_options = build_formation_options(prob_map, odds_map=odds_map)
        except Exception:
            formation_options = []

        out["best_bets"] = best_bets
        out["prob_map"] = prob_map
        out["odds_map"] = odds_map
        out["df120_rows"] = df120_rows
        out["formation_options"] = formation_options
        return out

    def _fetch_entries_and_enrich(
        self, venue_name: str, date: str, race_no: int
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        try:
            base_entries = self.get_entries_race(venue_name, date, race_no)
        except Exception:
            return [], None

        try:
            enriched_entries = self.enrich_entries(venue_name, base_entries, date, race_no)
            enriched_entries = self._dedupe_by_lane(enriched_entries)
        except Exception:
            enriched_entries = None

        return base_entries, enriched_entries

    def get_ai_prediction_bundle(
        self,
        venue_name: str,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
        model_mode: str | None = None,
    ) -> Dict[str, Any]:
        venue_name = self._normalize_venue_name(venue_name)

        empty_out = {
            "best_bets": [],
            "prob_map": {},
            "odds_map": {},
            "df120_rows": [],
            "formation_options": [],
            "model_mode": self._resolve_model_mode(model_mode or self.model_mode),
        }

        # 出走表+選手成績(同じページを参照するので内部で直列)/ 直前情報 / オッズ は
        # それぞれ別URLへの独立したHTTP取得なので並列化する。
        # 以前はこの3系統を順番に(1つずつ待って)取得しており、
        # 「オッズ取得・レーサー情報取得が遅い」という体感の主因になっていた。
        # (2026-07-16、ユーザー指摘を受けて改善)
        _t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_entries = ex.submit(self._fetch_entries_and_enrich, venue_name, date, race_no)
            fut_beforeinfo = ex.submit(self.get_beforeinfo_only, venue_name, race_no, date)
            fut_odds = ex.submit(fetch_odds, race_no, date, get_jcd(venue_name)) if with_odds else None

            base_entries, enriched_entries = fut_entries.result()

            try:
                beforeinfo = fut_beforeinfo.result()
            except Exception:
                beforeinfo = {}

            raw_odds: Dict[str, Any] = {}
            if fut_odds is not None:
                try:
                    raw_odds = fut_odds.result()
                except Exception:
                    raw_odds = {}

        if self.debug:
            print(f"[TIMING] get_ai_prediction_bundle parallel fetch took {time.monotonic() - _t0:.2f}s")

        if not base_entries:
            return empty_out

        return self._predict_bundle(
            date=date,
            race_no=race_no,
            venue_name=venue_name,
            base_entries=base_entries,
            beforeinfo=beforeinfo,
            enriched_entries=enriched_entries,
            top_n=top_n,
            with_odds=with_odds,
            model_mode=model_mode,
            raw_odds=raw_odds,
        )

    def get_ai_predictions(
        self,
        venue_name: str,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
        model_mode: str | None = None,
    ) -> List[Dict[str, Any]]:
        bundle = self.get_ai_prediction_bundle(
            venue_name=venue_name,
            date=date,
            race_no=race_no,
            top_n=top_n,
            with_odds=with_odds,
            model_mode=model_mode,
        )
        return bundle.get("best_bets", [])

    # =========================
    # legacy convenience wrappers
    # =========================
    def get_ai_prediction_bundle_with_active_model(
        self,
        venue_name: str,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> Dict[str, Any]:
        return self.get_ai_prediction_bundle(
            venue_name=venue_name,
            date=date,
            race_no=race_no,
            top_n=top_n,
            with_odds=with_odds,
            model_mode=self.model_mode,
        )

    def get_ai_prediction_bundle_venue_model(
        self,
        venue_name: str,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> Dict[str, Any]:
        return self.get_ai_prediction_bundle(
            venue_name=venue_name,
            date=date,
            race_no=race_no,
            top_n=top_n,
            with_odds=with_odds,
            model_mode="venue",
        )

    def get_ai_prediction_bundle_global_auto_model(
        self,
        venue_name: str,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> Dict[str, Any]:
        return self.get_ai_prediction_bundle(
            venue_name=venue_name,
            date=date,
            race_no=race_no,
            top_n=top_n,
            with_odds=with_odds,
            model_mode="global_auto",
        )
