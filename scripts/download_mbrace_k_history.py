# scripts/download_mbrace_k_history.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import requests


BASE_URL = "https://www1.mbrace.or.jp/od2/K/{yyyymm}/k{yymmdd}.lzh"


def iter_dates(start: date, end: date) -> Iterator[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_url(d: date) -> str:
    return BASE_URL.format(
        yyyymm=d.strftime("%Y%m"),
        yymmdd=d.strftime("%y%m%d"),
    )


def download_file(
    session: requests.Session,
    d: date,
    out_dir: Path,
    timeout: tuple[float, float] = (10.0, 30.0),
    sleep_sec: float = 0.5,
    overwrite: bool = False,
) -> tuple[str, str]:
    """
    returns:
      ("downloaded" | "exists" | "not_found" | "error", detail)
    """
    url = build_url(d)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"k{d.strftime('%y%m%d')}.lzh"

    if out_path.exists() and not overwrite:
        return "exists", str(out_path)

    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                time.sleep(sleep_sec)
                return "not_found", url

            r.raise_for_status()

            tmp_path = out_path.with_suffix(".lzh.part")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)

            tmp_path.replace(out_path)
            time.sleep(sleep_sec)
            return "downloaded", str(out_path)

    except requests.RequestException as e:
        time.sleep(sleep_sec)
        return "error", f"{url} :: {e}"


def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical BOAT RACE K files from mbrace.")
    parser.add_argument("--start", required=True, help="start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="end date YYYYMMDD")
    parser.add_argument("--out", default="data/raw_k", help="output directory")
    parser.add_argument("--sleep", type=float, default=0.7, help="sleep seconds between requests")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    start = parse_ymd(args.start)
    end = parse_ymd(args.end)
    out_dir = Path(args.out)

    if end < start:
        raise ValueError("--end must be >= --start")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; BoatAIDownloader/1.0)",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )

    downloaded = 0
    exists = 0
    not_found = 0
    errors = 0

    for d in iter_dates(start, end):
        status, detail = download_file(
            session=session,
            d=d,
            out_dir=out_dir,
            sleep_sec=args.sleep,
            overwrite=args.overwrite,
        )

        if status == "downloaded":
            downloaded += 1
            print(f"[DOWNLOADED] {d} -> {detail}")
        elif status == "exists":
            exists += 1
            print(f"[SKIP]       {d} -> {detail}")
        elif status == "not_found":
            not_found += 1
            print(f"[404]        {d} -> {detail}")
        else:
            errors += 1
            print(f"[ERROR]      {d} -> {detail}")

    print("\n===== DONE =====")
    print(f"downloaded : {downloaded}")
    print(f"exists     : {exists}")
    print(f"not_found  : {not_found}")
    print(f"errors     : {errors}")
    print(f"out_dir    : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
