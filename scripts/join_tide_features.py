# scripts/join_tide_features.py
"""
潮位特徴量(tide_level_cm・tide_trend_cmph)を学習データに結合するバックフィルスクリプト。

過去レース(raw_txt由来のtrifecta_train_by_venue/{venue}.csv)には発走時刻データが
存在しないため、engine/venue_race_schedule.py の会場別概算発走スケジュールと、
engine/tide_fetcher.py の気象庁公式潮位データを組み合わせて (venue, date, race_no)
ごとに潮位を推定し、既存の会場別分割ファイルに列として追加する。
([[boat_ai_tide_data_research]]参照)

このスクリプトはネットワークアクセス(気象庁サイトへのHTTPリクエスト)が必要なため、
サンドボックス環境では実行できない。ユーザーのローカル環境で実行すること。

事前準備:
  なし(trifecta_train_by_venue/{venue}.csvが既に存在していればそのまま使える)。

使い方:
  python -m scripts.join_tide_features --venue 戸田
  python -m scripts.join_tide_features --all

挙動:
  - engine.tide_fetcher.VENUE_TIDE_STATIONで観測点が未確定(None)の会場
    (桐生・戸田・びわこ・鳴門・大村)は、そのままスキップする(列を追加しない。
    学習側の_add_feature_block()が列欠如を検知して自動的に0埋めするので、
    このスクリプト側で明示的にダミー列を書き込む必要はない)。
  - (venue, date, race_no)のユニークな組み合わせごとに1回だけ潮位を計算し、
    結果を全120行(組み合わせ行)にマージする(重複計算を避けるため)。
  - 気象庁サイトへのHTTPリクエストはengine/tide_fetcher.py内で年単位に
    キャッシュされる(data/tide_cache/{station}_{year}.txt)ため、同じ観測点・年を
    何度も呼び出しても実際のリクエストは年に1回で済む。
  - 各会場ファイルを上書き保存する(バックアップを取りたい場合は事前にコピーすること)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.venue_registry import VENUE_ORDER, normalize_venue_name  # noqa: E402
from engine.venue_race_schedule import approx_post_time_str  # noqa: E402
from engine.tide_fetcher import get_tide_station, get_tide_features  # noqa: E402

SPLIT_BY_VENUE_DIR = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_by_venue"


def _compute_tide_for_venue(df: pd.DataFrame, venue: str) -> pd.DataFrame:
    station = get_tide_station(venue)
    if station is None:
        print(f"[SKIP] venue={venue}: 観測点未確定/対象外のためスキップします")
        return df

    df = df.copy()
    # 既にtide_level_cm/tide_trend_cmph列がある場合(前回実行分など)は落としてから
    # 結合する。残したままmergeすると列名が衝突しtide_level_cm_x/_yに分裂して
    # 直後のdf["tide_level_cm"]がKeyErrorになる。
    df = df.drop(columns=["tide_level_cm", "tide_trend_cmph"], errors="ignore")
    df["date"] = df["date"].astype(str).str.zfill(8)
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)

    unique_keys = df[["date", "race_no"]].drop_duplicates()
    print(f"[{venue}] unique (date, race_no) 組み合わせ数: {len(unique_keys):,}")

    tide_cache: Dict[Tuple[str, int], Tuple[float, float]] = {}
    n_ok = 0
    n_miss = 0

    for i, (date_str, race_no) in enumerate(zip(unique_keys["date"], unique_keys["race_no"])):
        t = approx_post_time_str(venue, race_no)
        if t is None:
            tide_cache[(date_str, race_no)] = (0.0, 0.0)
            n_miss += 1
            continue

        hh, mm = t.split(":")
        level, trend = get_tide_features(venue, date_str, int(hh), int(mm))
        if level is None or trend is None:
            tide_cache[(date_str, race_no)] = (0.0, 0.0)
            n_miss += 1
        else:
            tide_cache[(date_str, race_no)] = (level, trend)
            n_ok += 1

        if (i + 1) % 500 == 0:
            print(f"[{venue}] progress {i + 1:,}/{len(unique_keys):,} (ok={n_ok:,} miss={n_miss:,})")

    print(f"[{venue}] done: ok={n_ok:,} miss={n_miss:,}")

    tide_df = pd.DataFrame(
        [
            {"date": d, "race_no": r, "tide_level_cm": v[0], "tide_trend_cmph": v[1]}
            for (d, r), v in tide_cache.items()
        ]
    )

    df = df.merge(tide_df, on=["date", "race_no"], how="left")
    df["tide_level_cm"] = df["tide_level_cm"].fillna(0.0)
    df["tide_trend_cmph"] = df["tide_trend_cmph"].fillna(0.0)
    return df


def _process_venue(venue: str) -> None:
    venue = normalize_venue_name(venue)
    path = SPLIT_BY_VENUE_DIR / f"{venue}.csv"
    if not path.exists():
        print(f"[WARN] {path} が見つかりません。scripts/split_trifecta_train_by_venue.py を先に実行してください。")
        return

    print(f"Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"rows: {len(df):,}")

    out = _compute_tide_for_venue(df, venue)

    if "tide_level_cm" not in out.columns:
        # スキップされた会場(観測点未確定)はファイルを書き換えない
        return

    out.to_csv(path, index=False)
    print(f"saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="", help="会場名(例: 戸田)")
    parser.add_argument("--all", action="store_true", help="全24会場を処理")
    args = parser.parse_args()

    if args.all:
        venues = list(VENUE_ORDER)
    elif args.venue:
        venues = [args.venue]
    else:
        raise SystemExit("--venue か --all を指定してください")

    for v in venues:
        try:
            _process_venue(v)
        except Exception as e:
            print(f"[ERROR] venue={v}: {e}")


if __name__ == "__main__":
    main()
