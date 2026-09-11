"""READ-ONLY: break ONE ticker's `fundamentals_history_sec` drift into shapes.

`history_drift_classify.py` splits drift two ways -- stored NULL vs stored not-NULL -- and
calls the second one "published value would change". VTRS showed that is too coarse: 80 of its
285 cells are not-NULL, but the ones printed are `0.0 -> 2,619,200,000`, which is a ZERO-FILL
being corrected, not a published number being rewritten. A rebuild that replaces 0.0 with a
real revenue is a repair; one that moves a real number is the thing the append-only guard
exists to stop. Deciding VTRS needs those two counted apart.

`fiscal_end` is called out on its own because it is the one column where a two-day move
(2020-03-29 -> 2020-03-31) is a change of MEANING -- 52/53-week fiscal calendar vs calendar
quarter-end -- and not obviously a repair in either direction.
"""
from __future__ import annotations

import sys

import pandas as pd

from src.context import get_config_context
from src.data_extract.utils.fundamentals import build_history as bh


def main() -> None:
    ticker = sys.argv[1]
    _, context = get_config_context("./configs", use_cache=False, save=False)
    captured: dict[str, pd.DataFrame] = {}
    real = bh.diff_against_stored

    def spy(stored, history):
        d = real(stored, history)
        captured["last"] = d
        return d

    bh.diff_against_stored = spy
    try:
        bh.build_fundamentals_history(context, tickers=[ticker], rebuild_history=False)
        print(f"{ticker}: no drift")
        return
    except ValueError:
        pass
    d = captured["last"]

    def is_zero(v) -> bool:
        try:
            return float(v) == 0.0
        except (TypeError, ValueError):
            return False

    notnull = d[d["stored"].notna()].copy()
    notnull["shape"] = notnull.apply(
        lambda r: ("fiscal_end/date" if "fiscal" in str(r["column"]) or "date" in str(r["column"])
                   else "zero-fill repaired" if is_zero(r["stored"])
                   else "REAL VALUE MOVED"), axis=1)

    print(f"\n=== {ticker}: {len(d)} drifting cell(s) over {d['as_of'].nunique()} row(s) ===")
    print(f"  stored NULL (a fill)        : {int(d['stored'].isna().sum())}")
    for shape, grp in notnull.groupby("shape"):
        print(f"  {shape:28}: {len(grp)}")
    moved = notnull[notnull["shape"] == "REAL VALUE MOVED"]
    if not moved.empty:
        print(f"\n  every cell where a non-zero stored number changes ({len(moved)}):")
        print(moved.to_string())
    dates = notnull[notnull["shape"] == "fiscal_end/date"]
    if not dates.empty:
        print(f"\n  date columns ({len(dates)}):")
        print(dates.head(20).to_string())


if __name__ == "__main__":
    main()
