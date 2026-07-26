# scripts/notify_rank_a_races.py
# -*- coding: utf-8 -*-
"""
発走(締切予定)の約10分前に、AIランクがA(勝負レース)と判定されたレースを
Telegramに通知するウォッチャー。

(2026-07-25追記: 当初LINE Messaging APIで実装したが、無料枠が月200通と
厳しく個人利用にそぐわないため、回数上限が実質無いTelegram Botに変更した。
engine/line_notifier.pyは未使用のまま残しているだけで、このスクリプトからは
呼んでいない。)

ローカルのMacでlaunchdから5分おきに起動する想定
(scripts/launchd/com.boatai.rankAnotify.plist、
 scripts/run_notify_rank_a.sh経由)。

仕組み:
  0. WATCH_START〜WATCH_END(デフォルト07:30〜22:30)の時間帯外なら即終了する
     (この時間帯以外はMacがスリープしていても支障が無いようにするため)
  1. 全24会場について engine.race_schedule_fetcher で今日の発走時刻を取得
     (開催が無い会場は空になるだけで、エラーにはしない)
  2. 発走が「今からWINDOW_MIN_MINUTES〜WINDOW_MAX_MINUTES分後」のレースだけ
     を対象にする(5分おき起動を想定した幅。ずれても取りこぼさないよう
     8〜13分という余裕を持たせている)
  3. 対象レースについて RaceController で予測を作り、app.main._build_race_signal
     と全く同じロジック・関数でrankを判定する(ランク判定ロジックの二重化を
     避けるため、app.mainから直接import する)
  4. rank=="A"ならTelegramに送信する(engine.telegram_notifier、事前準備は
     そちらのdocstring参照)。rankに関わらず、一度判定したレースは
     NOTIFIED_LOGに記録し、二重送信・二重計算をしない

使い方:
  python scripts/notify_rank_a_races.py            (通常運用)
  python scripts/notify_rank_a_races.py --dry-run   (Telegram送信せず判定結果だけ表示)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import (  # noqa: E402
    _build_race_signal,
    _complete_probabilities,
    _complete_odds_map,
    _build_ev_map_from_prob_and_odds,
    _grouped_odds_to_flat_map,
)
from app.controller import RaceController  # noqa: E402
from engine.venue_registry import VENUE_ORDER  # noqa: E402
from engine.race_schedule_fetcher import fetch_post_times  # noqa: E402
from engine.telegram_notifier import send_telegram_message  # noqa: E402

NOTIFIED_LOG = PROJECT_ROOT / "data" / "logs" / "notified_rank_a_races.log"
WINDOW_MIN_MINUTES = 8.0
WINDOW_MAX_MINUTES = 13.0

# 監視する時間帯(この範囲外は起動してもすぐ終了し、発走時刻取得・予測計算を
# 一切行わない)。ボートレースの実開催時間帯(概ね朝〜21時前後)をカバーしつつ、
# それ以外の時間はMacがスリープしていても支障が無いようにするための設定。
WATCH_START = (7, 30)   # (hour, minute)
WATCH_END = (22, 30)

# 通知対象を絞りたい場合はここに会場名を列挙する(空リストなら全24会場)。
# 例: TARGET_VENUES = ["戸田", "江戸川", "平和島"]
TARGET_VENUES: list[str] = []


def _within_watch_hours(now: datetime) -> bool:
    start_minutes = WATCH_START[0] * 60 + WATCH_START[1]
    end_minutes = WATCH_END[0] * 60 + WATCH_END[1]
    now_minutes = now.hour * 60 + now.minute
    return start_minutes <= now_minutes <= end_minutes


def _load_notified() -> Set[Tuple[str, str, int]]:
    if not NOTIFIED_LOG.exists():
        return set()
    keys: Set[Tuple[str, str, int]] = set()
    for line in NOTIFIED_LOG.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) == 3:
            try:
                keys.add((parts[0], parts[1], int(parts[2])))
            except ValueError:
                continue
    return keys


def _mark_notified(date: str, venue: str, race_no: int) -> None:
    NOTIFIED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTIFIED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{date},{venue},{race_no}\n")


def _combo_sort_key(combo: str) -> tuple:
    # "1-2-3" -> (1, 2, 3) の昇順。投票サイトの入力グリッド(1着→2着→3着の順で
    # 数字を選ぶ形式が多い)を上から順になぞるだけで入力できるようにするため、
    # スコア順ではなく組番の数字順に並べ替える。
    parts = str(combo).split("-")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return (99, 99, 99)


def _build_message(venue: str, race_no: int, hhmm: str, minutes_to_start: float, race_signal: dict, best_bets: list) -> str:
    sorted_bets = sorted(best_bets, key=lambda b: _combo_sort_key(b.get("combo", "")))
    n = len(sorted_bets)
    total_yen = n * 100

    # 手入力しやすいよう、組番だけを並べた行(スペース区切り)と、
    # オッズ付きの内訳を両方載せる。前者はそのままコピーして
    # 投票サイトの検索欄等に貼り付けられる想定。
    plain_combos = " ".join(b.get("combo", "?") for b in sorted_bets)
    detail_lines = "\n".join(
        f"  {b.get('combo', '?'):<7} {b.get('odds', '-')}倍"
        for b in sorted_bets
    )

    return (
        f"【Rank A 勝負レース】\n"
        f"{venue} {race_no}R  発走 {hhmm}(あと約{int(round(minutes_to_start))}分)\n"
        f"Top確率 {race_signal['top_prob']}%  差 {race_signal['prob_gap']}pt  期待値 {race_signal['top_ev']}\n"
        f"\n"
        f"買い目 {n}点(投資{total_yen:,}円、1点100円)\n"
        f"{plain_combos}\n"
        f"\n"
        f"内訳:\n{detail_lines}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Telegram送信せず判定結果だけ表示する")
    args = parser.parse_args()

    now = datetime.now()

    if not _within_watch_hours(now):
        print(
            f"監視時間帯({WATCH_START[0]:02d}:{WATCH_START[1]:02d}〜"
            f"{WATCH_END[0]:02d}:{WATCH_END[1]:02d})の外なので何もせず終了します。"
        )
        return

    date = now.strftime("%Y%m%d")
    notified = _load_notified()

    controller = RaceController(model_mode="venue")
    venues = TARGET_VENUES or list(VENUE_ORDER)

    checked = 0
    for venue in venues:
        try:
            post_times = fetch_post_times(venue, date)
        except Exception as e:
            print(f"[SKIP] {venue}: schedule fetch failed -> {e}")
            continue

        for race_no, hhmm in post_times.items():
            try:
                hh, mm = hhmm.split(":")
                post_dt = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            except Exception:
                continue

            minutes_to_start = (post_dt - now).total_seconds() / 60.0
            if not (WINDOW_MIN_MINUTES <= minutes_to_start <= WINDOW_MAX_MINUTES):
                continue

            key = (date, venue, race_no)
            if key in notified:
                continue

            checked += 1
            try:
                bundle = controller.get_ai_prediction_bundle(
                    venue_name=venue, date=date, race_no=race_no,
                    top_n=20, with_odds=True, model_mode="venue",
                ) or {}

                raw_prob_map = bundle.get("prob_map", {}) or {}
                probabilities = _complete_probabilities(raw_prob_map)

                grouped_odds = controller.get_odds_only(venue, race_no=race_no, date=date)
                raw_odds_map = bundle.get("odds_map", {}) or {}
                grouped_flat_odds = _grouped_odds_to_flat_map(grouped_odds)
                merged_odds_map = dict(grouped_flat_odds)
                merged_odds_map.update(raw_odds_map)
                odds_map = _complete_odds_map(merged_odds_map)

                ev_result = _build_ev_map_from_prob_and_odds(probabilities, odds_map)
                best_bets = bundle.get("best_bets", []) or []

                race_signal = _build_race_signal(
                    probabilities=probabilities, ev_result=ev_result, best_bets=best_bets,
                )
            except Exception as e:
                print(f"[ERR] {venue} {race_no}R prediction failed -> {e}")
                continue

            print(
                f"{venue} {race_no}R ({hhmm}発走, あと{minutes_to_start:.1f}分) "
                f"rank={race_signal['rank']}"
            )

            if race_signal["rank"] != "A":
                if not args.dry_run:
                    _mark_notified(date, venue, race_no)
                continue

            text = _build_message(venue, race_no, hhmm, minutes_to_start, race_signal, best_bets)

            if args.dry_run:
                print("--- (dry-run、送信しません) ---")
                print(text)
                continue

            ok = send_telegram_message(text)
            print(f"  -> Telegram送信 {'OK' if ok else 'NG'}")
            _mark_notified(date, venue, race_no)

    if checked == 0:
        print("対象窓(発走8〜13分前)のレースはありませんでした。")


if __name__ == "__main__":
    main()
