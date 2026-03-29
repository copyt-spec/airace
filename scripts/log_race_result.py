from __future__ import annotations

import argparse

from engine.prediction_logger import save_race_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument("--venue", required=True, help="丸亀 / 戸田 / 児島 など")
    parser.add_argument("--race", required=True, type=int, help="race no")
    parser.add_argument("--combo", required=True, help="actual trifecta combo ex: 1-2-3")
    parser.add_argument("--payout", required=True, type=float, help="trifecta payout")
    parser.add_argument("--source", default="manual", help="manual / scraped / etc")
    args = parser.parse_args()

    save_race_result(
        date=args.date,
        venue=args.venue,
        race_no=args.race,
        actual_combo=args.combo,
        payout=args.payout,
        source=args.source,
    )
    print("saved result log.")


if __name__ == "__main__":
    main()
