# scripts/join_meeting_so_far_features.py
# -*- coding: utf-8 -*-
"""
「今節(このシリーズ)これまでの成績」特徴量を、各会場のtrifecta_train_by_venue/
{venue}.csv に列として追加する。

背景: raw_txt K-fileのヘッダーに「第N日」(開催の何日目か)が入っていることを
利用し、venue+date単位で節(meeting_instance_id)を判定 → 節内の同一選手の
それまでの実績(top3率・勝率・平均ST)をリーク無し(expanding、当該レースより前
の節内成績のみ)で計算したもの
(scripts/build_meeting_so_far_all_venues.py 相当、Coworkサンドボックス側で
 全24会場分を計算済み → data/datasets/meeting_so_far_wide_all_venues.csv)。

このスクリプトは、そのwideテーブル(1レース1行、lane1〜6×
{top3_rate, win_rate, avg_st, n} = 24列)を venue+date+race_no で
trifecta_train_by_venue/{venue}.csv に結合するだけ(pandasのみ、catboost不要)。

使い方:
  python scripts/join_meeting_so_far_features.py --all
  python scripts/join_meeting_so_far_features.py --venue 戸田

結合後、train_binary_catboost_per_venue_with_racer_stats.py の
--include_meeting_form_features (デフォルトTrue) で学習に使われる。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "datasets" / "trifecta_train_by_venue"
WIDE_CSV = PROJECT_ROOT / "data" / "datasets" / "meeting_so_far_wide_all_venues.csv"

VENUE_ORDER = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑",
    "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島",
    "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村",
]

MEETING_FORM_COLS = [
    f"lane{i}_meeting_so_far_{stat}"
    for i in range(1, 7)
    for stat in ("top3_rate", "win_rate", "avg_st", "n")
]


def join_for_venue(venue: str, wide: pd.DataFrame, stage: str = "all") -> None:
    """
    2026-07-26追記: 1venue分でも「読込+結合+CSV書き出し」を1プロセスで
    やろうとするとサンドボックスの実行時間上限ぎりぎり(read~10s+write~30s)
    になり、書き込み中に強制終了→元ファイル破損という事故が起きた(戸田)。
    そこで stage="pickle" (読込+結合してpickleに保存するだけ、速い) と
    stage="csv" (pickleを読んでCSVに書き出すだけ) の2段階に分け、
    別々の呼び出しで実行できるようにした。stage="all"は従来通り一括実行
    (十分速いローカル環境で使う想定)。
    """
    src = SPLIT_DIR / f"{venue}.csv"
    pkl_path = src.with_suffix(".csv.merged_pkl")
    tmp_path = src.with_suffix(".csv.tmp_writing")

    if stage in ("all", "pickle"):
        if not src.exists():
            print(f"[SKIP] {venue}: {src} が見つかりません")
            return

        df = pd.read_csv(src, low_memory=False)
        df["date"] = df["date"].astype(str).str.zfill(8)

        # 既に結合済みなら一旦落としてから結合し直す(再実行しても壊れないように)
        df = df.drop(columns=[c for c in MEETING_FORM_COLS if c in df.columns], errors="ignore")

        w = wide[wide["venue"] == venue].copy()
        w["date"] = w["date"].astype(str).str.zfill(8)
        w = w[["date", "race_no"] + MEETING_FORM_COLS]

        merged = df.merge(w, on=["date", "race_no"], how="left")

        if len(merged) != len(df):
            raise RuntimeError(
                f"{venue}: 結合後の行数が変わっています(元{len(df)}行 -> {len(merged)}行)。"
                "leftマージのはずなので本来増減しないはず。安全のため書き込みを中止します。"
            )

        matched_races = merged[["date", "race_no"]].drop_duplicates().merge(
            w[["date", "race_no"]].drop_duplicates(), on=["date", "race_no"], how="inner"
        )
        total_races = merged[["date", "race_no"]].drop_duplicates().shape[0]
        print(f"{venue}: [pickle stage] rows={len(merged)} races={total_races} matched_with_meeting_form={len(matched_races)}")

        if stage == "pickle":
            merged.to_pickle(pkl_path)
            print(f"{venue}: saved intermediate -> {pkl_path}")
            return
    else:
        if not pkl_path.exists():
            print(f"[SKIP] {venue}: {pkl_path} が見つかりません(先にstage=pickleを実行してください)")
            return
        merged = pd.read_pickle(pkl_path)

    # 2026-07-26追記: タイムアウトで書き込み途中のプロセスが強制終了され、
    # 元ファイルが壊れる事故が実際に起きた(戸田.csv)。同じファイルへ直接
    # to_csv()すると壊れた場合に復旧不能になるため、一時ファイルに書き出して
    # から完了後にrenameする(rename自体は一瞬で終わるので、たとえ次の呼び出し
    # がタイムアウトしても「古い正しいファイルのまま」か「新しい正しいファイル
    # に置き換わった後」のどちらかにしかならない)。
    #
    # さらに、この規模のCSV書き出し自体が実行環境のタイムアウトに近い時間
    # かかることが分かったため、行を分割して複数回の呼び出しにまたがって
    # 追記できるようにしてある(state_pathに書き込み済み行数を記録)。
    state_path = src.with_suffix(".csv.write_state")
    chunk_rows = 150_000

    # 2026-07-26追記: 前回呼び出しがチャンク書き込みの途中でタイムアウトkillされた
    # 場合、tmp_pathの末尾行が改行の前で切れて残ることがある。state_pathの数字
    # だけを信用してappendすると、「不完全な行」+「新チャンクの1行目」が改行
    # 無しで連結され、1行に列が2倍出現する壊れたCSVになる事故が実際に起きた
    # (戸田)。対策: state_pathの数字は使わず、tmp_pathの実際の行数を正として
    # 再開する。まずtmp_pathの末尾を「最後に改行で終わっている位置」まで
    # 切り詰め(不完全な最終行を破棄)、そのうえで実際の行数を数えてstart_rowを
    # 決め直す。
    def _count_lines(path: Path) -> int:
        n = 0
        with open(path, "rb") as fh:
            while True:
                buf = fh.read(1024 * 1024 * 16)
                if not buf:
                    break
                n += buf.count(b"\n")
        return n

    start_row = 0
    if tmp_path.exists() and tmp_path.stat().st_size > 0:
        size = tmp_path.stat().st_size
        tail_read = min(size, 5_000_000)
        with open(tmp_path, "rb") as f:
            f.seek(size - tail_read)
            tail = f.read()
        last_nl = tail.rfind(b"\n")
        if last_nl == -1:
            # tail内に改行が1つも無い = 安全に判定できないので破棄して最初から
            tmp_path.unlink()
            start_row = 0
        else:
            truncate_at = (size - tail_read) + last_nl + 1
            if truncate_at < size:
                with open(tmp_path, "r+b") as f:
                    f.truncate(truncate_at)
            line_count = _count_lines(tmp_path)  # ヘッダ込み
            start_row = max(line_count - 1, 0)
    if state_path.exists():
        # 参考情報として残すが、判定には使わない(実ファイルの行数が正)
        pass

    total_rows = len(merged)
    mode = "w" if start_row == 0 else "a"
    header = start_row == 0

    while start_row < total_rows:
        end_row = min(start_row + chunk_rows, total_rows)
        merged.iloc[start_row:end_row].to_csv(
            tmp_path, index=False, encoding="utf-8-sig", mode=mode, header=header,
        )
        mode = "a"
        header = False
        start_row = end_row
        state_path.write_text(str(start_row))
        print(f"{venue}: [csv stage] written {start_row}/{total_rows}")

    tmp_path.replace(src)
    if pkl_path.exists():
        pkl_path.unlink()
    if state_path.exists():
        state_path.unlink()
    print(f"{venue}: [csv stage] DONE rows={len(merged)} -> {src}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--stage", default="all", choices=["all", "pickle", "csv"],
        help="all=読込+結合+CSV書き出しを一括実行(ローカル向け、通常はこれでよい)。"
             "pickle=読込+結合のみ行い中間pickleを保存。csv=保存済みpickleからCSVを書き出す。",
    )
    args = parser.parse_args()

    wide = pd.read_csv(WIDE_CSV, low_memory=False)

    if args.all:
        venues = VENUE_ORDER
    elif args.venue:
        venues = [args.venue]
    else:
        raise SystemExit("--venue か --all を指定してください")

    for v in venues:
        try:
            join_for_venue(v, wide, stage=args.stage)
        except Exception as e:
            print(f"[ERROR] venue={v}: {e}")


if __name__ == "__main__":
    main()
