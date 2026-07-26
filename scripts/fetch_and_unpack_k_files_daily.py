# scripts/fetch_and_unpack_k_files_daily.py
# -*- coding: utf-8 -*-
"""
data/raw_txt (Kファイル、mbrace由来の日次結果アーカイブ) を自動で最新化する日次バッチ。

これまでdata/raw_txt / data/raw_kは2026-07-15の一回限りの手動バックフィル
(scripts/download_mbrace_k_history.py を手で実行)以降、誰も/何も更新しておらず、
選手成績特徴量(racer_point_in_time_stats.csv, Cowork側のboat-ai-racer-stats-refresh
スケジュールタスクが06:09 JSTに再生成)が古いまま止まる原因になっていた。

ユーザーのMac上でlaunchd等から毎日1回(boat-ai-racer-stats-refreshより前、例: 05:00頃)
実行する想定。このCoworkサンドボックス環境からはmbrace.or.jpに到達できないため、
ここでは実行できない(ユーザーの手元環境で実行・動作確認する必要がある)。

処理内容:
  1. 直近N日分(デフォルト--days-back 5)について、data/raw_txt/K{yymmdd}.TXT が
     まだ無い日付だけを対象に、
       a. https://www1.mbrace.or.jp/od2/K/{yyyymm}/k{yymmdd}.lzh をdata/raw_kにダウンロード
          (data/raw_kに既に同名lzhがあればダウンロードはスキップ)
       b. lhafileでlzhを展開し、中身をdata/raw_txt/K{yymmdd}.TXTとしてそのままのバイト列で保存
  2. まだ公開されていない日付(404)はnot_foundとしてスキップするだけ。次回実行時に
     ローリングウィンドウ(--days-back)の中に入っていれば自動的に再試行される
     ("自己修復"設計。特定の日だけ失敗しても翌日以降のどれかの実行で拾われる)。
  3. data/raw_txt/racer_race_history.csv 等の再生成(選手成績特徴量の再計算)はここでは
     行わない。それはCowork側のboat-ai-racer-stats-refreshスケジュールタスク
     (毎日06:09 JST)が別途担当する想定。この分業を崩さないこと。

使い方の例:
  # 直近5日分のうち未取得の日だけ後追い(通常運用・launchdから毎日呼ぶ想定)
  python scripts/fetch_and_unpack_k_files_daily.py

  # 直近14日分をまとめて後追い(取りこぼし対策・久しぶりに動かす時など)
  python scripts/fetch_and_unpack_k_files_daily.py --days-back 14

  # 特定の期間だけ指定してバックフィル(初回の大きな取りこぼし解消など)
  python scripts/fetch_and_unpack_k_files_daily.py --start 20260715 --end 20260721

  # 既にある.lzh/.TXTも含めて強制的に取り直す(データ破損時の復旧用)
  python scripts/fetch_and_unpack_k_files_daily.py --days-back 5 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_mbrace_k_history import build_url, download_file  # noqa: E402

RAW_K_DIR = PROJECT_ROOT / "data" / "raw_k"
RAW_TXT_DIR = PROJECT_ROOT / "data" / "raw_txt"


def iter_dates(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def extract_txt(lzh_path: Path, out_path: Path) -> tuple[str, str]:
    """
    lzh_path (.lzh) を展開し、中の1ファイルをそのままのバイト列でout_pathに書き出す。
    returns: ("extracted" | "error", detail)
    """
    try:
        import lhafile  # 遅延importにしてスクリプト読み込み時のエラーを避ける
    except ImportError:
        return "error", (
            "lhafile未インストール。`pip install lhafile` (または "
            "`venv/bin/pip install lhafile`) を実行してください。"
        )

    try:
        lh = lhafile.Lhafile(str(lzh_path))
        names = lh.namelist()
        if not names:
            return "error", f"{lzh_path}: アーカイブ内にファイルが見つかりません"
        # 中身のファイル名がそのまま使えるとは限らない(大文字/小文字ゆれ等)ので、
        # 出力先ファイル名は呼び出し側で決めたK{yymmdd}.TXTに固定する。
        data = lh.read(names[0])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp_path, "wb") as f:
            f.write(data)
        tmp_path.replace(out_path)
        return "extracted", str(out_path)
    except Exception as e:  # noqa: BLE001
        return "error", f"{lzh_path}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mbraceからKファイルを取得し、data/raw_txtに展開して配置する日次バッチ。"
    )
    parser.add_argument("--days-back", type=int, default=5, help="今日から何日分遡って確認するか(デフォルト5)")
    parser.add_argument("--start", help="開始日 YYYYMMDD (--days-backより優先)")
    parser.add_argument("--end", help="終了日 YYYYMMDD (省略時は昨日)")
    parser.add_argument("--overwrite", action="store_true", help="既存の.lzh/.TXTも含めて取り直す")
    parser.add_argument("--sleep", type=float, default=0.7, help="ダウンロード間のスリープ秒数")
    args = parser.parse_args()

    if args.start:
        start = parse_ymd(args.start)
    else:
        start = date.today() - timedelta(days=args.days_back)

    if args.end:
        end = parse_ymd(args.end)
    else:
        # 今日のレースはまだ確定していないはずなので、デフォルトは昨日まで
        end = date.today() - timedelta(days=1)

    if end < start:
        raise ValueError("--end must be >= --start")

    RAW_K_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TXT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; BoatAIDownloader/1.0)",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )

    downloaded = 0
    extracted = 0
    skipped_existing = 0
    not_found = 0
    errors = 0

    for d in iter_dates(start, end):
        txt_path = RAW_TXT_DIR / f"K{d.strftime('%y%m%d')}.TXT"

        if txt_path.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"[SKIP]     {d} -> {txt_path.name} (既に存在)")
            continue

        dl_status, dl_detail = download_file(
            session=session,
            d=d,
            out_dir=RAW_K_DIR,
            sleep_sec=args.sleep,
            overwrite=args.overwrite,
        )

        if dl_status == "not_found":
            not_found += 1
            print(f"[404]      {d} -> {build_url(d)} (まだ公開されていない可能性)")
            continue
        elif dl_status == "error":
            errors += 1
            print(f"[DL ERROR] {d} -> {dl_detail}")
            continue
        elif dl_status == "downloaded":
            downloaded += 1
            print(f"[DL OK]    {d} -> {dl_detail}")
        else:  # "exists"
            print(f"[DL SKIP]  {d} -> {dl_detail} (既存.lzhを再利用)")

        lzh_path = RAW_K_DIR / f"k{d.strftime('%y%m%d')}.lzh"
        ex_status, ex_detail = extract_txt(lzh_path, txt_path)
        if ex_status == "extracted":
            extracted += 1
            print(f"[EXTRACT]  {d} -> {ex_detail}")
        else:
            errors += 1
            print(f"[EX ERROR] {d} -> {ex_detail}")

    print("\n===== DONE =====")
    print(f"range        : {start} .. {end}")
    print(f"downloaded   : {downloaded}")
    print(f"extracted    : {extracted}")
    print(f"skip(exists) : {skipped_existing}")
    print(f"not_found    : {not_found}")
    print(f"errors       : {errors}")

    max_date = None
    for p in sorted(RAW_TXT_DIR.glob("K*.TXT")):
        stem = p.stem[1:]  # "K260714" -> "260714"
        if len(stem) == 6 and stem.isdigit():
            if max_date is None or stem > max_date:
                max_date = stem
    print(f"raw_txt max date now: {max_date}")


if __name__ == "__main__":
    main()
