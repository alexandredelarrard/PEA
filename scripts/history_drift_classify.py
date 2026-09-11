"""READ-ONLY: classify what `fundamentals-history-sec` would change, per ticker.

The append-only guard refuses any run that would alter a stored cell, and it cannot tell a
NULL being FILLED from a published number being REWRITTEN. A register backfill always produces
the first and must never produce the second, so this answers which one it is before anyone
passes --rebuild-history.

It writes NOTHING: `diff_against_stored` is wrapped to capture the drift frame, and the guard's
own ValueError is allowed to fire, which happens before any save.
"""
from __future__ import annotations

import sys

import pandas as pd

from src.context import get_config_context
from src.data_extract.utils.fundamentals import build_history as bh


def main() -> None:
    tickers = sys.argv[1].split(",")
    _, context = get_config_context("./configs", use_cache=False, save=False)
    captured: dict[str, pd.DataFrame] = {}
    real = bh.diff_against_stored

    def spy(stored, history):
        d = real(stored, history)
        captured["last"] = d
        return d

    bh.diff_against_stored = spy
    print("| ticker | cells | rows | NULL -> value | value -> value | verdict |")
    print("|---|---:|---:|---:|---:|---|")
    for t in tickers:
        captured.pop("last", None)
        try:
            bh.build_fundamentals_history(context, tickers=[t], rebuild_history=False)
            print(f"| {t} | 0 | 0 | 0 | 0 | clean (no drift) |")
            continue
        except ValueError:
            pass
        except Exception as e:                                  # noqa: BLE001
            print(f"| {t} | ? | ? | ? | ? | ERROR {type(e).__name__}: {str(e)[:60]} |")
            continue
        d = captured.get("last")
        if d is None or d.empty:
            print(f"| {t} | ? | ? | ? | ? | raised but no drift captured |")
            continue
        filled = int(d["stored"].isna().sum())
        rewritten = int(d["stored"].notna().sum())
        verdict = ("**NULL-fill only** -- safe to rebuild" if rewritten == 0
                   else f"⚠ **{rewritten} PUBLISHED VALUE(S) WOULD CHANGE** -- inspect")
        print(f"| {t} | {len(d)} | {d['as_of'].nunique()} | {filled} | {rewritten} | {verdict} |")
        if rewritten:
            print(d[d["stored"].notna()].head(12).to_string())


if __name__ == "__main__":
    main()
