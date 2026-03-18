# app/toda_controller.py

from engine.toda_fetcher import fetch_all_toda_entries_once
from engine.odds_fetcher import fetch_odds
from engine.beforeinfo_fetcher import fetch_beforeinfo


class TodaRaceController:

    def get_all_entries(self, date):
        return fetch_all_toda_entries_once(date)

    def get_odds_only(self, race_no, date):
        # ここは “今ある odds_fetcher” を流用（会場コードは odds_fetcher 側が丸亀固定なら後で対応）
        raw_odds = fetch_odds(race_no, date)

        grouped = {i: {} for i in range(1, 7)}
        numeric_values = []

        for combo, odd in raw_odds.items():
            parts = combo.split("-")
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

        min_val = min(numeric_values) if numeric_values else 0
        max_val = max(numeric_values) if numeric_values else 0

        return {"data": grouped, "min": min_val, "max": max_val}

    def get_beforeinfo_only(self, race_no, date):
        return fetch_beforeinfo(race_no, date)
