# scripts/notify_rank_a_races.py
# -*- coding: utf-8 -*-
"""
発走(締切予定)の約10分前に、AIランク(収益性)または的中信頼度のいずれかが
A以上(A+/A)と判定されたレースをTelegramに通知するウォッチャー。

(2026-08-08追記: 「RANK通知に加えて的中信頼度がAだった場合も連絡してほしい」
という要望を受け、的中信頼度(hit_confidence_tier)がA+/Aのレースも通知対象に
追加した。両基準は[[boat_ai_rank_confidence_roi_inversion]]の通り意図的に別軸
(RANK=収益性、的中信頼度=的中率)で、相関は弱い(RANK A+/Aは全体の約7%、
的中信頼度A+/Aは約38%と頻度も大きく異なる)。ユーザーの希望により両者は
独立したトリガー・独立した通知ログ(NOTIFIED_LOG / NOTIFIED_LOG_CONFIDENCE)
として管理し、1つのレースが両方に該当する場合はメッセージも2通(RANK基準の
ものと的中信頼度基準のもの)を別々に送る。1回の予測計算(race_signal)は
共有し、二重計算はしない。)

(2026-08-03改定: ランク判定がportfolio_ev方式・5段階(D/C/B/A/A+)に切り替わった
のに合わせ、通知対象を旧来の「rank=='A'のみ」から「rank in ('A+','A')」に拡大し、
メッセージ本文にも実際のランク(A+ or A)を明示するようにした。)

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
  3. RANK通知ログ・的中信頼度通知ログのどちらか一方でも未処理なレースだけ
     RaceController で予測を作り、app.main._build_race_signal と全く同じ
     ロジック・関数でrank/hit_confidence_tierを判定する(ロジックの二重化を
     避けるため、app.mainから直接import する)
  4. rankが"A+"か"A"ならRANK基準の通知を送る。hit_confidence_tierが"A+"か
     "A"なら的中信頼度基準の通知を送る(engine.telegram_notifier、事前準備は
     そちらのdocstring参照)。両方に該当すれば2通送る。どちらの基準についても
     判定したレースはそれぞれのNOTIFIED_LOG*に記録し、同じ基準での二重送信を
     しない

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
    _parse_tilt,
)
from app.controller import RaceController  # noqa: E402
from engine.venue_registry import VENUE_ORDER  # noqa: E402
from engine.race_schedule_fetcher import fetch_post_times  # noqa: E402
from engine.telegram_notifier import send_telegram_message  # noqa: E402
from engine.prediction_logger import save_lane_tilt  # noqa: E402

NOTIFIED_LOG = PROJECT_ROOT / "data" / "logs" / "notified_rank_a_races.log"
NOTIFIED_LOG_CONFIDENCE = PROJECT_ROOT / "data" / "logs" / "notified_hit_confidence_a_races.log"
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


def _load_notified(log_path: Path = NOTIFIED_LOG) -> Set[Tuple[str, str, int]]:
    if not log_path.exists():
        return set()
    keys: Set[Tuple[str, str, int]] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) == 3:
            try:
                keys.add((parts[0], parts[1], int(parts[2])))
            except ValueError:
                continue
    return keys


def _mark_notified(date: str, venue: str, race_no: int, log_path: Path = NOTIFIED_LOG) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
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


def _build_message(
    venue: str, race_no: int, hhmm: str, minutes_to_start: float,
    race_signal: dict, best_bets: list, headline: str = "rank",
) -> str:
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

    rank = race_signal.get("rank", "?")
    label = race_signal.get("label", "")
    hit_conf_tier = race_signal.get("hit_confidence_tier", "?")
    hit_conf_label = race_signal.get("hit_confidence_label", "")
    # 2026-08-08: 的中信頼度の表示%はengine.hit_confidence_modelの較正済み
    # スコア(hit_confidence_score)を使う(combined_prob単体より精度が高いと
    # 検証済み)。古いrace_signalにフィールドが無い場合はcombined_probに
    # フォールバックする。
    hit_conf_pct = race_signal.get("hit_confidence_score", race_signal.get("combined_prob", 0.0))

    # headline="rank": RANK(収益性)を見出しにし、的中信頼度は参考情報として本文に。
    # headline="confidence": 的中信頼度を見出しにし、RANKは参考情報として本文に。
    # 両者は意図的に別軸の指標(RANK=収益性、的中信頼度=的中率)であることを
    # 明示するため、どちらが通知理由かをタイトルで区別する。
    if headline == "confidence":
        title = f"【的中信頼度 {hit_conf_tier}{(' ' + hit_conf_label) if hit_conf_label else ''}】"
        metric_line = (
            f"的中信頼度: {hit_conf_label}({hit_conf_pct}%)  "
            f"参考RANK: {rank}{(' ' + label) if label else ''}"
        )
    else:
        title = f"【RANK {rank}{(' ' + label) if label else ''}】"
        metric_line = f"的中信頼度: {hit_conf_label}({hit_conf_pct}%)"

    return (
        f"{title}\n"
        f"{venue} {race_no}R  発走 {hhmm}(あと約{int(round(minutes_to_start))}分)\n"
        f"Top確率 {race_signal['top_prob']}%  差 {race_signal['prob_gap']}pt  買い目期待値 {race_signal['avg_best_ev']}\n"
        f"{metric_line}\n"
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
    notified_rank = _load_notified(NOTIFIED_LOG)
    notified_conf = _load_notified(NOTIFIED_LOG_CONFIDENCE)

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
            need_rank_check = key not in notified_rank
            need_conf_check = key not in notified_conf
            if not need_rank_check and not need_conf_check:
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

                # 2026-08-11追加: fetch_odds/_group_oddsはHTTP自体は200で返るが
                # オッズ表がまだ空(締切直後で未確定/ページ未更新等)のケースで
                # 例外を出さず、全120通り0.0のodds_mapを返すことがある
                # (「的中信頼度A+なのに買い目が全部0.0倍」というオッズ取得失敗の
                # 通知が実際に発生した)。0.0でない件数が極端に少ない場合は
                # まだデータが揃っていないと判断し、このレースは今回スキップして
                # 次の5分後のリトライ(need_rank_check/need_conf_checkが
                # まだTrueのまま=notified_logに書き込まない)に委ねる。
                nonzero_odds = sum(1 for v in odds_map.values() if v and v > 0)
                if nonzero_odds < 100:
                    print(
                        f"[SKIP] {venue} {race_no}R: オッズ取得できず"
                        f"(non-zero={nonzero_odds}/120)。次回リトライします。"
                    )
                    continue

                ev_result = _build_ev_map_from_prob_and_odds(probabilities, odds_map)
                best_bets = bundle.get("best_bets", []) or []

                race_signal = _build_race_signal(
                    probabilities=probabilities, ev_result=ev_result, best_bets=best_bets,
                )
            except Exception as e:
                print(f"[ERR] {venue} {race_no}R prediction failed -> {e}")
                continue

            # 2026-08-10追加: チルト角度ログ(predictions_lane_tilt.csv)がほぼ
            # 溜まっていなかった原因が判明した。daily-prediction-crawlは1日1回
            # 09:02固定で全レースを巡回するが、直前情報(チルト含む)は各レース
            # 発走の直前にならないと公開されないため、朝1回の巡回ではほぼ全レースで
            # 未公開のまま取れずに終わっていた。このスクリプトは発走8〜13分前という
            # 直前情報が確実に出ている時間帯に5分おきで全24会場を回っているため、
            # ここでチルトも合わせて記録することで正しい時間帯にデータを集める。
            try:
                beforeinfo = controller.get_beforeinfo_only(venue, race_no=race_no, date=date)
                if isinstance(beforeinfo, dict):
                    lane_tilt: dict[int, float | None] = {}
                    for lane in range(1, 7):
                        row = beforeinfo.get(lane) or beforeinfo.get(str(lane)) or {}
                        raw_tilt = row.get("tilt") if isinstance(row, dict) else None
                        lane_tilt[lane] = _parse_tilt(raw_tilt)
                    save_lane_tilt(date=date, venue=venue, race_no=race_no, lane_tilt=lane_tilt)
            except Exception as e:
                print(f"[WARN] {venue} {race_no}R lane tilt log save failed -> {e}")

            print(
                f"{venue} {race_no}R ({hhmm}発走, あと{minutes_to_start:.1f}分) "
                f"rank={race_signal['rank']} hit_confidence={race_signal.get('hit_confidence_tier', '?')}"
            )

            # RANK(収益性)基準と的中信頼度基準は独立したトリガー・独立した
            # 通知ログで判定する。両方に該当する場合はメッセージも2通送る。
            if need_rank_check:
                if race_signal["rank"] in ("A+", "A"):
                    text = _build_message(venue, race_no, hhmm, minutes_to_start, race_signal, best_bets, headline="rank")
                    if args.dry_run:
                        print("--- (dry-run、RANK基準は送信しません) ---")
                        print(text)
                    else:
                        ok = send_telegram_message(text)
                        print(f"  -> Telegram送信(RANK基準) {'OK' if ok else 'NG'}")
                if not args.dry_run:
                    _mark_notified(date, venue, race_no, NOTIFIED_LOG)

            if need_conf_check:
                if race_signal.get("hit_confidence_tier") in ("A+", "A"):
                    text = _build_message(venue, race_no, hhmm, minutes_to_start, race_signal, best_bets, headline="confidence")
                    if args.dry_run:
                        print("--- (dry-run、的中信頼度基準は送信しません) ---")
                        print(text)
                    else:
                        ok = send_telegram_message(text)
                        print(f"  -> Telegram送信(的中信頼度基準) {'OK' if ok else 'NG'}")
                if not args.dry_run:
                    _mark_notified(date, venue, race_no, NOTIFIED_LOG_CONFIDENCE)

    if checked == 0:
        print("対象窓(発走8〜13分前)のレースはありませんでした。")


if __name__ == "__main__":
    main()
