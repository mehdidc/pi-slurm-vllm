#!/usr/bin/env python3
"""Generate one identically phrased history prompt per calendar day."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path


START_DATE = date(1989, 1, 1)
END_DATE = date(1999, 12, 31)
OUTPUT = Path(__file__).resolve().parent.parent / "problems" / "history_1989_1999.jsonl"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    current = START_DATE
    row_index = 0
    with OUTPUT.open("w", encoding="utf-8") as handle:
        while current <= END_DATE:
            display_date = current.strftime("%d.%m.%Y")
            row = {
                "id": f"history-{current.isoformat()}",
                "row_index": row_index,
                "date": display_date,
                "prompt": (
                    "Give me a summary of the most important events which happened at "
                    f"{display_date}?"
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            current += timedelta(days=1)
            row_index += 1

    print(f"Wrote {row_index} prompts to {OUTPUT}")


if __name__ == "__main__":
    main()
