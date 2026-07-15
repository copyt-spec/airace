from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# data/raw_txt が正(build_dataset_from_txt.pyもここを見ている)。
# リポジトリ直下の raw_txt/ は2026-04-04までの古いスナップショットなので使わない。
RAW_TXT_DIR = PROJECT_ROOT / "data" / "raw_txt"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
OUT_CSV = LOG_DIR / "results.csv"


RE_FILE_DATE = re.compile(r"(\d{4})[^\d]?(\d{1,2})[^\d]?(\d{1,2})")
RE_HEADER_DATE = re.compile(r"第\s*\d+日\s+(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})")
RE_ANY_DATE = re.compile(r"(\d{4})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})")
RE_TRIFECTA_LINE = re.compile(
    r"^\s*３連単\s+([1-6])\-([1-6])\-([1-6])\s+([\d,]+)"
)
RE_PAYOUT_SUMMARY_LINE = re.compile(
    r"^\s*(\d{1,2})R\s+([1-6])\-([1-6])\-([1-6])\s+([\d,]+)"
)
# [払戻金]サマリ行には 3連単 / 3連複 / 2連単 / 2連複 の4種が横並びで入っている。
# 例: " 1R  1-5-4    6950    1-4-5    1770    1-5    3020    1-5    1570"
RE_PAYOUT_SUMMARY_LINE_FULL = re.compile(
    r"^\s*(\d{1,2})R\s+"
    r"([1-6]\-[1-6]\-[1-6])\s+([\d,]+)\s+"
    r"([1-6]\-[1-6]\-[1-6])\s+([\d,]+)\s+"
    r"([1-6]\-[1-6])\s+([\d,]+)\s+"
    r"([1-6]\-[1-6])\s+([\d,]+)"
)

# 1つのKファイルは1日分だが、24KBGN...24KEND のように
# 会場コード(jcd, 2桁)ごとのブロックが複数連結されている。
RE_BLOCK_START = re.compile(r"^\s*(\d{2})KBGN\s*$")
RE_BLOCK_END = re.compile(r"^\s*(\d{2})KEND\s*$")

VENUE_CODE_MAP: Dict[str, str] = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

VENUE_NAMES = list(VENUE_CODE_MAP.values())

_ZEN2HAN = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "－": "-", "ー": "-", "―": "-", "−": "-",
    "　": " ",
})


def _norm_line(s: str) -> str:
    s2 = s.translate(_ZEN2HAN)
    s2 = re.sub(r"[ \t]+", " ", s2)
    return s2.rstrip("\n")


def _normalize_spaces(s: str) -> str:
    return re.sub(r"[ \t　]+", " ", (s or "")).strip()


def _normalize_date(y: str, m: str, d: str) -> str:
    return f"{int(y):04d}{int(m):02d}{int(d):02d}"


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(str(v).replace(",", "").strip())
    except Exception:
        return default


def _read_text_auto(path: Path) -> str:
    encodings = ["cp932", "shift_jis", "utf-8", "utf-8-sig", "euc_jp", "latin1"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception as e:
            last_err = e

    if last_err:
        raise last_err
    raise RuntimeError(f"failed to read: {path}")


def _extract_date_from_text(text: str) -> str:
    m = RE_HEADER_DATE.search(text)
    if m:
        return _normalize_date(m.group(1), m.group(2), m.group(3))

    m2 = RE_ANY_DATE.search(text)
    if m2:
        return _normalize_date(m2.group(1), m2.group(2), m2.group(3))

    return ""


def _extract_date_from_filename(filename: str) -> str:
    m = RE_FILE_DATE.search(filename)
    if m:
        return _normalize_date(m.group(1), m.group(2), m.group(3))
    return ""


def _split_by_kbgn_kend(all_lines: List[str]) -> List[Tuple[Optional[str], List[str]]]:
    """
    KファイルをjcdブロックごとのK(BGN/END)区間に分割する。
    区間外(ヘッダ部分など)は venue_code=None として1ブロック返す。
    """
    sections: List[Tuple[Optional[str], List[str]]] = []

    current_code: Optional[str] = None
    current_lines: List[str] = []
    in_block = False

    header_lines: List[str] = []

    for raw in all_lines:
        ln = _norm_line(raw)

        m_start = RE_BLOCK_START.match(ln)
        if m_start:
            if in_block and current_lines:
                sections.append((current_code, current_lines))
            current_code = m_start.group(1)
            current_lines = []
            in_block = True
            continue

        m_end = RE_BLOCK_END.match(ln)
        if m_end and in_block:
            if current_lines:
                sections.append((current_code, current_lines))
            current_code = None
            current_lines = []
            in_block = False
            continue

        if in_block:
            current_lines.append(ln)
        else:
            header_lines.append(ln)

    if in_block and current_lines:
        sections.append((current_code, current_lines))

    if not sections:
        # ブロックマーカーが見つからない古い形式など: venue名を本文から推定するため
        # 単一ブロックとして返す(呼び出し側で venue をテキストから推定する)
        return [(None, list(map(_norm_line, all_lines)))]

    return sections


def _parse_summary_block(lines: List[str], date: str, venue: str, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int]] = set()

    in_summary = False
    for line in lines:
        s = _normalize_spaces(line)

        if "[払戻金]" in s:
            in_summary = True
            continue

        if in_summary:
            m_full = RE_PAYOUT_SUMMARY_LINE_FULL.match(s)
            if m_full:
                race_no = _safe_int(m_full.group(1), 0)
                combo = m_full.group(2)
                payout = _safe_int(m_full.group(3), 0)
                if 1 <= race_no <= 12 and combo and payout >= 0:
                    key = (date, venue, race_no)
                    if key not in seen:
                        rows.append({
                            "date": date,
                            "venue": venue,
                            "race_no": race_no,
                            "actual_combo": combo,
                            "payout": float(payout),
                            "combo_3puku": m_full.group(4),
                            "payout_3puku": float(_safe_int(m_full.group(5), 0)),
                            "combo_2tan": m_full.group(6),
                            "payout_2tan": float(_safe_int(m_full.group(7), 0)),
                            "combo_2puku": m_full.group(8),
                            "payout_2puku": float(_safe_int(m_full.group(9), 0)),
                            "source": source,
                        })
                        seen.add(key)
                continue

            m = RE_PAYOUT_SUMMARY_LINE.match(s)
            if m:
                race_no = _safe_int(m.group(1), 0)
                combo = f"{m.group(2)}-{m.group(3)}-{m.group(4)}"
                payout = _safe_int(m.group(5), 0)
                if 1 <= race_no <= 12 and combo and payout >= 0:
                    key = (date, venue, race_no)
                    if key not in seen:
                        rows.append({
                            "date": date,
                            "venue": venue,
                            "race_no": race_no,
                            "actual_combo": combo,
                            "payout": float(payout),
                            "source": source,
                        })
                        seen.add(key)
                continue

            if s == "":
                continue
            if re.match(r"^(1[0-2]|[1-9])R\b", s):
                continue

            if "H1800m" in s or "着 艇 登番" in s:
                break

    return rows


def _parse_race_detail_blocks(lines: List[str], date: str, venue: str, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    current_race_no: Optional[int] = None
    race_found: Dict[int, Dict[str, Any]] = {}

    race_line_re = re.compile(r"^\s*(\d{1,2})R\b")

    for line in lines:
        s = _normalize_spaces(line)

        m_race = race_line_re.match(s)
        if m_race:
            rn = _safe_int(m_race.group(1), 0)
            if 1 <= rn <= 12:
                current_race_no = rn

        m_tri = RE_TRIFECTA_LINE.match(s)
        if m_tri and current_race_no is not None:
            combo = f"{m_tri.group(1)}-{m_tri.group(2)}-{m_tri.group(3)}"
            payout = _safe_int(m_tri.group(4), 0)

            race_found[current_race_no] = {
                "date": date,
                "venue": venue,
                "race_no": current_race_no,
                "actual_combo": combo,
                "payout": float(payout),
                "source": source,
            }

    for rn in sorted(race_found.keys()):
        rows.append(race_found[rn])

    return rows


def _dedupe_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for row in rows:
        date = str(row.get("date", "")).strip()
        venue = str(row.get("venue", "")).strip()
        race_no = _safe_int(row.get("race_no", 0), 0)
        combo = str(row.get("actual_combo", "")).strip()
        payout = float(row.get("payout", 0.0) or 0.0)

        if not date or not venue or race_no not in range(1, 13) or not combo:
            continue

        key = (date, venue, race_no)
        existing = best.get(key)

        if existing is None:
            best[key] = dict(row)
            continue

        if payout > float(existing.get("payout", 0.0) or 0.0):
            best[key] = dict(row)

    out = list(best.values())
    out.sort(key=lambda x: (str(x["date"]), str(x["venue"]), int(x["race_no"])))
    return out


def _guess_venue_from_text(lines: List[str], fallback_filename: str = "") -> str:
    full = "\n".join(lines) + "\n" + fallback_filename
    for venue in VENUE_NAMES:
        if venue in full:
            return venue
    return ""


def _parse_one_file(path: Path) -> List[Dict[str, Any]]:
    text = _read_text_auto(path)
    lines = text.splitlines()

    file_date = _extract_date_from_text(text)
    if not file_date:
        file_date = _extract_date_from_filename(path.name)

    source = path.name
    sections = _split_by_kbgn_kend(lines)

    all_rows: List[Dict[str, Any]] = []
    matched_any_block = any(code is not None for code, _ in sections)

    for code, section_lines in sections:
        if code is not None:
            venue = VENUE_CODE_MAP.get(code, "")
            if not venue:
                print(f"[WARN] unknown venue code={code} in {path.name}")
                continue
        else:
            if matched_any_block:
                # ブロックマーカーの外側(ヘッダ等)は結果データを含まない
                continue
            # 古い形式でブロックマーカーが存在しないファイル: 単一会場として推定
            venue = _guess_venue_from_text(section_lines, path.name)
            if not venue:
                print(f"[WARN] venue not found: {path.name}")
                continue

        date = _extract_date_from_text("\n".join(section_lines)) or file_date
        if not date:
            print(f"[WARN] date not found : {path.name} venue={venue}")
            continue

        summary_rows = _parse_summary_block(section_lines, date, venue, source)
        detail_rows = _parse_race_detail_blocks(section_lines, date, venue, source)

        all_rows.extend(_dedupe_results(summary_rows + detail_rows))

    merged = _dedupe_results(all_rows)

    if not merged:
        print(f"[WARN] no result rows parsed: {path.name}")

    return merged


def _collect_txt_files(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw txt dir not found: {raw_dir}")

    files = sorted([p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".k", ".dat", ""}])
    if not files:
        raise RuntimeError(f"no txt-like files found in: {raw_dir}")
    return files


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    files = _collect_txt_files(RAW_TXT_DIR)
    print(f"files found: {len(files)}")

    all_rows: List[Dict[str, Any]] = []

    for i, path in enumerate(files, start=1):
        try:
            rows = _parse_one_file(path)
            all_rows.extend(rows)
            print(f"[{i}/{len(files)}] parsed: {path.name} rows={len(rows)}")
        except Exception as e:
            print(f"[ERROR] failed: {path.name} -> {e}")

    all_rows = _dedupe_results(all_rows)

    if not all_rows:
        raise RuntimeError("no rows parsed from raw_txt")

    df = pd.DataFrame(all_rows)

    df["date"] = df["date"].astype(str).str.replace(".0", "", regex=False).str.zfill(8)
    df["venue"] = df["venue"].astype(str).str.strip()
    df["race_no"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)
    df["actual_combo"] = df["actual_combo"].astype(str).str.strip()
    df["payout"] = pd.to_numeric(df["payout"], errors="coerce").fillna(0.0).astype(float)
    df["source"] = df["source"].astype(str)

    # 3連複/2連単/2連複(サマリ表から拾えた分だけ埋まる。detailブロックのみの行はNaN)
    for combo_col in ["combo_3puku", "combo_2tan", "combo_2puku"]:
        if combo_col not in df.columns:
            df[combo_col] = ""
        df[combo_col] = df[combo_col].astype(str).replace("nan", "")

    for payout_col in ["payout_3puku", "payout_2tan", "payout_2puku"]:
        if payout_col not in df.columns:
            df[payout_col] = pd.NA
        df[payout_col] = pd.to_numeric(df[payout_col], errors="coerce")

    df = df[
        (df["date"] != "") &
        (df["venue"] != "") &
        (df["race_no"].between(1, 12)) &
        (df["actual_combo"] != "")
    ].copy()

    df = df.sort_values(["date", "venue", "race_no"]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("saved   :", OUT_CSV)
    print("rows    :", len(df))
    print("venues  :", df["venue"].nunique())
    print("min date:", df["date"].min())
    print("max date:", df["date"].max())
    print("with 3puku/2tan/2puku:", int((df["combo_3puku"] != "").sum()), "/", len(df))

    venue_counts = df.groupby("venue")["race_no"].count().sort_values(ascending=False)
    print("-" * 80)
    print("rows by venue")
    print(venue_counts.to_string())


if __name__ == "__main__":
    main()
