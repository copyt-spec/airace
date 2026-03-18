from __future__ import annotations

from typing import Any, Dict, List

from engine.marugame_fetcher import fetch_all_entries_once as fetch_all_marugame_entries_once
from engine.toda_fetcher import fetch_all_toda_entries_once, fetch_toda_racelist
from engine.kojima_fetcher import fetch_all_kojima_entries_once, fetch_kojima_racelist
from engine.odds_fetcher import fetch_odds
from engine.beforeinfo_fetcher import fetch_beforeinfo
from engine.beforeinfo_fetcher_venue import fetch_beforeinfo_venue
from engine.racelist_enricher import enrich_entries_with_racelist
from engine.render_ai_predictor import RenderAIPredictor

JCD_MARUGAME = 15
JCD_TODA = 2
JCD_KOJIMA = 16


class RaceController:
    def __init__(self) -> None:
        self.ai_predictor = RenderAIPredictor()

    # ===== 一覧用 =====
    def get_all_entries(self, date: str) -> List[Dict[str, Any]]:
        return fetch_all_marugame_entries_once(date)

    def get_all_entries_toda(self, date: str) -> List[Dict[str, Any]]:
        return fetch_all_toda_entries_once(date)

    def get_all_entries_kojima(self, date: str) -> List[Dict[str, Any]]:
        return fetch_all_kojima_entries_once(date)

    # ===== 共通小物 =====
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

    # ===== 1R詳細用 =====
    def get_entries_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        all_entries = fetch_all_marugame_entries_once(date)
        out: List[Dict[str, Any]] = []

        for row in all_entries:
            try:
                if int(row.get("race_no", 0)) == int(race_no):
                    out.append(dict(row))
            except Exception:
                continue

        out = self._dedupe_by_lane(out)
        out.sort(key=lambda x: int(x.get("lane", 0)))
        return out

    def get_entries_toda_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        rows = fetch_toda_racelist(race_no, date)
        out = [dict(x) for x in rows]
        out = self._dedupe_by_lane(out)
        out.sort(key=lambda x: int(x.get("lane", 0)))
        return out

    def get_entries_kojima_race(self, date: str, race_no: int) -> List[Dict[str, Any]]:
        rows = fetch_kojima_racelist(race_no, date)
        out = [dict(x) for x in rows]
        out = self._dedupe_by_lane(out)
        out.sort(key=lambda x: int(x.get("lane", 0)))
        return out

    # ===== odds grouped形式 =====
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

    # ===== odds =====
    def get_odds_only(self, race_no: int, date: str) -> Dict[str, Any]:
        raw_odds = fetch_odds(race_no, date, venue_code=JCD_MARUGAME)
        return self._group_odds(raw_odds)

    def get_odds_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        raw_odds = fetch_odds(race_no, date, venue_code=JCD_TODA)
        return self._group_odds(raw_odds)

    def get_odds_only_kojima(self, race_no: int, date: str) -> Dict[str, Any]:
        raw_odds = fetch_odds(race_no, date, venue_code=JCD_KOJIMA)
        return self._group_odds(raw_odds)

    # ===== beforeinfo =====
    def get_beforeinfo_only(self, race_no: int, date: str) -> Dict[str, Any]:
        return fetch_beforeinfo(race_no, date)

    def get_beforeinfo_only_toda(self, race_no: int, date: str) -> Dict[str, Any]:
        return fetch_beforeinfo_venue(race_no, date, venue_code=JCD_TODA)

    def get_beforeinfo_only_kojima(self, race_no: int, date: str) -> Dict[str, Any]:
        return fetch_beforeinfo_venue(race_no, date, venue_code=JCD_KOJIMA)

    # ===== motor/boat enrich =====
    def enrich_entries_marugame(
        self,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        return enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_MARUGAME,
        )

    def enrich_entries_toda(
        self,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        return enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_TODA,
        )

    def enrich_entries_kojima(
        self,
        entries: List[Dict[str, Any]],
        date: str,
        race_no: int,
    ) -> List[Dict[str, Any]]:
        return enrich_entries_with_racelist(
            entries,
            date=date,
            race_no=race_no,
            venue_code=JCD_KOJIMA,
        )

    # ===== beforeinfo 正規化 =====
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

    # ===== AI入力用変換 =====
    def _to_ai_entries(
        self,
        entries: List[Dict[str, Any]],
        beforeinfo: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        before_lane_map = self._normalize_beforeinfo(beforeinfo or {})
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
                    self._pick(
                        row,
                        ["exhibit", "tenji_time", "展示タイム"],
                        before_row.get("exhibit", 0.0),
                    ),
                    before_row.get("exhibit", 0.0),
                ),
                "st": self._safe_float(
                    self._pick(
                        row,
                        ["st", "avg_st", "ST", "平均ST", "start_timing"],
                        before_row.get("st", 0.0),
                    ),
                    before_row.get("st", 0.0),
                ),
                "win_rate": self._safe_float(
                    self._pick(row, ["win_rate", "nation_win_rate", "全国勝率"], 0.0),
                    0.0,
                ),
                "place_rate": self._safe_float(
                    self._pick(
                        row,
                        ["place_rate", "nation_place_rate", "two_rate", "全国2連対率", "quinella_rate"],
                        0.0,
                    ),
                    0.0,
                ),
                "age": self._safe_int(self._pick(row, ["age", "年齢"], 0), 0),
                "weight": self._safe_float(self._pick(row, ["weight", "体重"], 0.0), 0.0),
                "grade": self._normalize_grade(self._pick(row, ["grade", "class", "級別"], "")),
                "course": self._safe_int(
                    self._pick(row, ["course", "進入", "course_no"], before_row.get("course", lane)),
                    before_row.get("course", lane),
                ),
            }

            ai_entries.append(ai_row)

        ai_entries = self._dedupe_by_lane(ai_entries)
        ai_entries.sort(key=lambda x: x["lane"])
        return ai_entries

    # ===== 内部共通: AI予想 =====
    def _predict_with_fallback(
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
    ) -> List[Dict[str, Any]]:
        if not self._is_valid_6boats(base_entries):
            return []

        use_entries = base_entries
        if enriched_entries and self._is_valid_6boats(enriched_entries):
            use_entries = enriched_entries

        ai_entries = self._to_ai_entries(use_entries, beforeinfo)
        if not self._is_valid_6boats(ai_entries):
            return []

        weather_set = self._extract_weather_set(beforeinfo)

        odds_map = None
        if with_odds:
            try:
                raw_odds = fetch_odds(race_no, date, venue_code=venue_code)
                odds_map = self._flat_odds_map(raw_odds)
            except Exception:
                odds_map = None

        try:
            return self.ai_predictor.predict_race(
                date=date,
                venue=venue_name,
                race_no=race_no,
                entries=ai_entries,
                weather=weather_set["weather"],
                wind_dir=weather_set["wind_dir"],
                wind_speed_mps=weather_set["wind_speed_mps"],
                wave_cm=weather_set["wave_cm"],
                top_n=top_n,
                odds_map=odds_map,
            )
        except Exception:
            return []

    # ===== AI予想 =====
    def get_ai_predictions_marugame(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            base_entries = self.get_entries_race(date, race_no)
        except Exception:
            return []

        try:
            enriched_entries = self.enrich_entries_marugame(base_entries, date, race_no)
            enriched_entries = self._dedupe_by_lane(enriched_entries)
        except Exception:
            enriched_entries = None

        try:
            beforeinfo = self.get_beforeinfo_only(race_no, date)
        except Exception:
            beforeinfo = {}

        return self._predict_with_fallback(
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

    def get_ai_predictions_toda(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            base_entries = self.get_entries_toda_race(date, race_no)
        except Exception:
            return []

        try:
            enriched_entries = self.enrich_entries_toda(base_entries, date, race_no)
            enriched_entries = self._dedupe_by_lane(enriched_entries)
        except Exception:
            enriched_entries = None

        try:
            beforeinfo = self.get_beforeinfo_only_toda(race_no, date)
        except Exception:
            beforeinfo = {}

        return self._predict_with_fallback(
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

    def get_ai_predictions_kojima(
        self,
        date: str,
        race_no: int,
        top_n: int = 20,
        with_odds: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            base_entries = self.get_entries_kojima_race(date, race_no)
        except Exception:
            return []

        try:
            enriched_entries = self.enrich_entries_kojima(base_entries, date, race_no)
            enriched_entries = self._dedupe_by_lane(enriched_entries)
        except Exception:
            enriched_entries = None

        try:
            beforeinfo = self.get_beforeinfo_only_kojima(race_no, date)
        except Exception:
            beforeinfo = {}

        return self._predict_with_fallback(
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
