from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from engine.marugame_fetcher import fetch_all_entries_once as fetch_all_marugame_entries_once
from engine.toda_fetcher import fetch_all_toda_entries_once, fetch_toda_racelist
from engine.kojima_fetcher import fetch_all_kojima_entries_once, fetch_kojima_racelist
from engine.odds_fetcher import fetch_odds
from engine.beforeinfo_fetcher import fetch_beforeinfo
from engine.beforeinfo_fetcher_venue import fetch_beforeinfo_venue
from engine.racelist_enricher import enrich_entries_with_racelist
from engine.buy_selector import select_best_bets
from engine.model_loader_catboost_binary import BinaryCatBoostVenueModel

JCD_MARUGAME = 15
JCD_TODA = 2
JCD_KOJIMA = 16


class RaceController:
    # =========================
    # 共有キャッシュ
    # =========================
    _all_entries_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    _race_entries_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    _enriched_entries_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    _beforeinfo_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    _odds_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    # TTL秒
    _TTL_ALL_ENTRIES = 1800      # 30分
    _TTL_RACE_ENTRIES = 1800     # 30分
    _TTL_ENRICHED = 1800         # 30分
    _TTL_BEFOREINFO = 180        # 3分
    _TTL_ODDS = 180              # 3分

    def __init__(self) -> None:
        self.binary_model = BinaryCatBoostVenueModel(
            model_dir="data/models",
            debug=False,
        )

    # =========================
    # キャッシュ共通
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

    def _venue_key(self, venue_name: str) -> str:
        s = str(venue_name or "").strip()
        if "丸亀" in s:
            return "丸亀"
        if "戸田" in s:
            return "戸田"
        if "児島" in s:
            return "児島"
        return s

    def _trim_old_cache(self) -> None:
        # 必要最低限の簡易掃除
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
    # 共通小物
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

    def _normalize_grade(self, raw: Any) -> str:
        s = str(raw or "").strip().upper()
        if s in {"A1", "A2", "B1", "B2"}:
            return s
        return ""

    def _normalize_venue_name(self, venue_name: str) -> str:
        v = str(venue_name or "").strip()
        if "丸亀" in v:
            return "丸亀"
        if "戸田" in v:
            return "戸田"
        if "児島" in v:
            return "児島"
        return v

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

    # =========================
    # 一覧用（場ごと全件）
    # =========================
    def get_all_entries(self, date: str) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("丸亀", date)
        cached = self._cache_get(self._all_entries_cache, key, self._TTL_ALL_ENTRIES)
        if cached is not None:
            return cached

        data = fetch_all_marugame_entries_once(date)
        data = [dict(x) for x in data]
        return self._cache_set(self._all_entries_cache, key, data)

    def get_all_entries_toda(self, date: str) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("戸田", date)
        cached = self._cache_get(self._all_entries_cache, key, self._TTL_ALL_ENTRIES)
        if cached is not None:
            return cached

        data = fetch_all_toda_entries_once(date)
        data = [dict(x) for x in data]
        return self._cache_set(self._all_entries_cache, key, data)

    def get_all_entries_kojima(self, date: str) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("児島", date)
        cached = self._cache_get(self._all_entries_cache, key, self._TTL_ALL_ENTRIES)
        if cached is not None:
            return cached

        data = fetch_all_kojima_entries_once(date)
        data = [dict(x) for x in data]
        return self._cache_set(self._all_entries_cache, key, data)

    # =========================
    # 1R詳細用
    # =========================
    def get_entries_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("丸亀", date, int(race_no))
        cached = self._cache_get(self._race_entries_cache, key, self._TTL_RACE_ENTRIES)
        if cached is not None:
            return cached

        all_entries = self.get_all_entries(date)
        out: List[Dict[str, Any]] = []

        for row in all_entries:
            try:
                if int(row.get("race_no", 0)) == int(race_no):
                    out.append(dict(row))
            except Exception:
                continue

        out = self._dedupe_by_lane(out)
        out.sort(key=lambda x: int(x.get("lane", 0)))
        return self._cache_set(self._race_entries_cache, key, out)

    def get_entries_toda_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("戸田", date, int(race_no))
        cached = self._cache_get(self._race_entries_cache, key, self._TTL_RACE_ENTRIES)
        if cached is not None:
            return cached

        # まず全件キャッシュがあればそこから切る
        all_entries = self.get_all_entries_toda(date)
        out = []
        for row in all_entries:
            try:
                if int(row.get("race_no", 0)) == int(race_no):
                    out.append(dict(row))
            except Exception:
                continue

        if not out:
            rows = fetch_toda_racelist(race_no, date)
            out = [dict(x) for x in rows]

        out = self._dedupe_by_lane(out)
        out.sort(key=lambda x: int(x.get("lane", 0)))
        return self._cache_set(self._race_entries_cache, key, out)

    def get_entries_kojima_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("児島", date, int(race_no))
        cached = self._cache_get(self._race_entries_cache, key, self._TTL_RACE_ENTRIES)
        if cached is not None:
            return cached

        all_entries = self.get_all_entries_kojima(date)
        out = []
        for row in all_entries:
            try:
                if int(row.get("race_no", 0)) == int(race_no):
                    out.append(dict(row))
            except Exception:
                continue

        if not out:
            rows = fetch_kojima_racelist(race_no, date)
            out = [dict(x) for x in rows]

        out = self._dedupe_by_lane(out)
        out.sort(key=lambda x: int(x.get("lane", 0)))
        return self._cache_set(self._race_entries_cache, key, out)

    # =========================
    # odds grouped形式
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

    # =========================
    # odds
    # =========================
    def get_odds_only(self, race_no: int, date: str) -> Dict[str, Any]:
        self._trim_old_cache()
        key = ("丸亀", date, int(race_no))
        cached = self._cache_get(self._odds_cache, key, self._TTL_ODDS)
        if cached is not None:
            return cached

        raw_odds = fetch_odds(race_no, date, venue_code=JCD_MARUGAME)
        grouped = self._group_odds(raw_odds)
        return self._cache_set(self._odds_cache, key, grouped)

    def get_odds_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        self._trim_old_cache()
        key = ("戸田", date, int(race_no))
        cached = self._cache_get(self._odds_cache, key, self._TTL_ODDS)
        if cached is not None:
            return cached

        raw_odds = fetch_odds(race_no, date, venue_code=JCD_TODA)
        grouped = self._group_odds(raw_odds)
        return self._cache_set(self._odds_cache, key, grouped)

    def get_odds_only_kojima(self, race_no: int, date: str) -> Dict[str, Any]:
        self._trim_old_cache()
        key = ("児島", date, int(race_no))
        cached = self._cache_get(self._odds_cache, key, self._TTL_ODDS)
        if cached is not None:
            return cached

        raw_odds = fetch_odds(race_no, date, venue_code=JCD_KOJIMA)
        grouped = self._group_odds(raw_odds)
        return self._cache_set(self._odds_cache, key, grouped)

    # =========================
    # beforeinfo
    # =========================
    def get_beforeinfo_only(self, race_no: int, date: str) -> Dict[str, Any]:
        self._trim_old_cache()
        key = ("丸亀", date, int(race_no))
        cached = self._cache_get(self._beforeinfo_cache, key, self._TTL_BEFOREINFO)
        if cached is not None:
            return cached

        data = fetch_beforeinfo(race_no, date)
        return self._cache_set(self._beforeinfo_cache, key, data)

    def get_beforeinfo_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        self._trim_old_cache()
        key = ("戸田", date, int(race_no))
        cached = self._cache_get(self._beforeinfo_cache, key, self._TTL_BEFOREINFO)
        if cached is not None:
            return cached

        data = fetch_beforeinfo_venue(race_no, date, venue_code=JCD_TODA)
        return self._cache_set(self._beforeinfo_cache, key, data)

    def get_beforeinfo_only_kojima(self, race_no: int, date: str) -> Dict[str, Any]:
        self._trim_old_cache()
        key = ("児島", date, int(race_no))
        cached = self._cache_get(self._beforeinfo_cache, key, self._TTL_BEFOREINFO)
        if cached is not None:
            return cached

        data = fetch_beforeinfo_venue(race_no, date, venue_code=JCD_KOJIMA)
        return self._cache_set(self._beforeinfo_cache, key, data)

    # =========================
    # motor/boat enrich
    # =========================
    def enrich_entries_marugame(
        self,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("丸亀", date, int(race_no))
        cached = self._cache_get(self._enriched_entries_cache, key, self._TTL_ENRICHED)
        if cached is not None:
            return cached

        data = enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_MARUGAME,
        )
        data = self._dedupe_by_lane([dict(x) for x in data])
        data.sort(key=lambda x: int(x.get("lane", 0)))
        return self._cache_set(self._enriched_entries_cache, key, data)

    def enrich_entries_toda(
        self,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("戸田", date, int(race_no))
        cached = self._cache_get(self._enriched_entries_cache, key, self._TTL_ENRICHED)
        if cached is not None:
            return cached

        data = enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_TODA,
        )
        data = self._dedupe_by_lane([dict(x) for x in data])
        data.sort(key=lambda x: int(x.get("lane", 0)))
        return self._cache_set(self._enriched_entries_cache, key, data)

    def enrich_entries_kojima(
        self,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        self._trim_old_cache()
        key = ("児島", date, int(race_no))
        cached = self._cache_get(self._enriched_entries_cache, key, self._TTL_ENRICHED)
        if cached is not None:
            return cached

        data = enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_KOJIMA,
        )
        data = self._dedupe_by_lane([dict(x) for x in data])
        data.sort(key=lambda x: int(x.get("lane", 0)))
        return self._cache_set(self._enriched_entries_cache, key, data)

    # =========================
    # beforeinfo 正規化
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
    # AI入力用
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
                        "venue_code": (
                            JCD_MARUGAME if venue_name == "丸亀"
                            else JCD_TODA if venue_name == "戸田"
                            else JCD_KOJIMA
                        ),
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
    ) -> List[Dict[str, Any]]:
        ai_preds = self._prob_map_to_rows(prob_map, odds_map=odds_map)

        best_bets = select_best_bets(
            ai_preds,
            top_n=top_n,
            venue=venue_name,
        )

        if not best_bets:
            fallback = sorted(
                ai_preds,
                key=lambda x: float(x.get("score", 0.0)),
                reverse=True,
            )[:top_n]
            for i, r in enumerate(fallback, start=1):
                r["buy_rank"] = i
                r["buy_score"] = float(r.get("score", 0.0))
                r["is_best_bet"] = True
            return fallback

        return best_bets

    def _predict_bundle(
        self,
        *,
        date: str,
        race_no: int,
        venue_name: str,
        venue_code: int,
        base_entries: List[Dict[str, Any]],
        beforeinfo: Dict[str, Any],
        enriched_entries: List[Dict[str, Any]] | None,
        top_n: int,
        with_odds: bool,
    ) -> Dict[str, Any]:
        out = {
            "best_bets": [],
            "prob_map": {},
            "odds_map": {},
            "df120_rows": [],
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
                raw_odds = fetch_odds(race_no, date, venue_code=venue_code)
                odds_map = self._flat_odds_map(raw_odds)
            except Exception:
                odds_map = {}

        try:
            prob_map = self.binary_model.predict_proba(df120, venue_name)
        except Exception:
            prob_map = {}

        if not prob_map:
            return out

        best_bets = self._select_best_bets(
            prob_map=prob_map,
            odds_map=odds_map,
            top_n=top_n,
            venue_name=venue_name,
        )

        out["best_bets"] = best_bets
        out["prob_map"] = prob_map
        out["odds_map"] = odds_map
        out["df120_rows"] = df120_rows
        return out

    # =========================
    # 旧互換
    # =========================
    def get_ai_predictions_marugame(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> List[Dict[str, Any]]:
        bundle = self.get_ai_prediction_bundle_marugame(date, race_no, top_n=top_n, with_odds=with_odds)
        return bundle.get("best_bets", [])

    def get_ai_predictions_toda(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> List[Dict[str, Any]]:
        bundle = self.get_ai_prediction_bundle_toda(date, race_no, top_n=top_n, with_odds=with_odds)
        return bundle.get("best_bets", [])

    def get_ai_predictions_kojima(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> List[Dict[str, Any]]:
        bundle = self.get_ai_prediction_bundle_kojima(date, race_no, top_n=top_n, with_odds=with_odds)
        return bundle.get("best_bets", [])

    # =========================
    # full bundle
    # =========================
    def get_ai_prediction_bundle_marugame(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> Dict[str, Any]:
        try:
            base_entries = self.get_entries_race(date, race_no)
        except Exception:
            return {"best_bets": [], "prob_map": {}, "odds_map": {}, "df120_rows": []}

        try:
            enriched_entries = self.enrich_entries_marugame(base_entries, date, race_no)
            enriched_entries = self._dedupe_by_lane(enriched_entries)
        except Exception:
            enriched_entries = None

        try:
            beforeinfo = self.get_beforeinfo_only(race_no, date)
        except Exception:
            beforeinfo = {}

        return self._predict_bundle(
            date=date,
            race_no=race_no,
            venue_name="丸亀",
            venue_code=JCD_MARUGAME,
            base_entries=base_entries,
            beforeinfo=beforeinfo,
            enriched_entries=enriched_entries,
            top_n=top_n,
            with_odds=with_odds,
        )

    def get_ai_prediction_bundle_toda(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> Dict[str, Any]:
        try:
            base_entries = self.get_entries_toda_race(date, race_no)
        except Exception:
            return {"best_bets": [], "prob_map": {}, "odds_map": {}, "df120_rows": []}

        try:
            enriched_entries = self.enrich_entries_toda(base_entries, date, race_no)
            enriched_entries = self._dedupe_by_lane(enriched_entries)
        except Exception:
            enriched_entries = None

        try:
            beforeinfo = self.get_beforeinfo_only_toda(race_no, date)
        except Exception:
            beforeinfo = {}

        return self._predict_bundle(
            date=date,
            race_no=race_no,
            venue_name="戸田",
            venue_code=JCD_TODA,
            base_entries=base_entries,
            beforeinfo=beforeinfo,
            enriched_entries=enriched_entries,
            top_n=top_n,
            with_odds=with_odds,
        )

    def get_ai_prediction_bundle_kojima(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> Dict[str, Any]:
        try:
            base_entries = self.get_entries_kojima_race(date, race_no)
        except Exception:
            return {"best_bets": [], "prob_map": {}, "odds_map": {}, "df120_rows": []}

        try:
            enriched_entries = self.enrich_entries_kojima(base_entries, date, race_no)
            enriched_entries = self._dedupe_by_lane(enriched_entries)
        except Exception:
            enriched_entries = None

        try:
            beforeinfo = self.get_beforeinfo_only_kojima(race_no, date)
        except Exception:
            beforeinfo = {}

        return self._predict_bundle(
            date=date,
            race_no=race_no,
            venue_name="児島",
            venue_code=JCD_KOJIMA,
            base_entries=base_entries,
            beforeinfo=beforeinfo,
            enriched_entries=enriched_entries,
            top_n=top_n,
            with_odds=with_odds,
        )
