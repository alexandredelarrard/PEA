"""RegSHO short-volume fetch: resume window, universe filter, upsert shape.

Resume is on the GLOBAL max date, not per ticker: one RegSHO file covers the whole
market, so once day D is stored every ticker has D. A per-ticker frontier would let
a single lagging symbol (index churn, a renamed ticker) drag the loop back over
thousands of already-held day-files on every run -- the reason `fails_to_deliver`
is a separate table in the first place (schema.py).
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_extract.utils.prices import fetch_short_interest as si


def _regsho(day: str, rows: list[tuple[str, int, int]]) -> str:
    head = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    return head + "".join(f"{day}|{t}|{s}|0|{v}|Q\n" for t, s, v in rows)


def test_resume_day_is_the_day_after_the_global_max(sqlite_store):
    ctx = SimpleNamespace(store=sqlite_store)
    # cold table -> full years_history window
    cold = si._resume_day(ctx, years_history=10)
    assert cold == pd.Timestamp.today().normalize() - pd.DateOffset(years=10)

    sqlite_store.replace("short_interest", pd.DataFrame({
        "ticker": ["AAA", "BBB", "BBB"],
        "date": pd.to_datetime(["2024-05-01", "2024-06-03", "2024-06-04"]),
        "short_volume": [1.0, 2.0, 3.0], "total_volume": [10.0, 20.0, 30.0],
    }))
    # GLOBAL max is 2024-06-04 (BBB's) -- AAA lagging at 05-01 must NOT pull it back
    assert si._resume_day(ctx, years_history=10) == pd.Timestamp("2024-06-05")

    print("\n=== SANITY CHECK: RegSHO resume day ===")
    print(f"  cold table -> {cold.date()} (years_history); stored max 2024-06-04 -> "
          "2024-06-05. A ticker stale at 2024-05-01 does not widen the window. Validated.")


def test_fetch_filters_to_the_universe_and_upserts(sqlite_store, monkeypatch):
    sqlite_store.replace("short_interest", pd.DataFrame({
        "ticker": ["AAA"], "date": pd.to_datetime(["2024-06-03"]),
        "short_volume": [1.0], "total_volume": [10.0],
    }))
    monkeypatch.setattr(si, "record_run", lambda *a, **k: None)
    monkeypatch.setattr(si, "_fetch_day",
                        lambda day: _regsho(day.strftime("%Y%m%d"),
                                            [("AAA", 500, 1000), ("ZZZ", 900, 1800)]))
    # bound the loop: pretend the table is current through the day before "today"
    monkeypatch.setattr(si, "_resume_day",
                        lambda *a, **k: pd.Timestamp.today().normalize())

    out = si.fetch_short_interest(SimpleNamespace(store=sqlite_store),
                                  tickers=["AAA"], pause=0.0)

    assert set(out["ticker"]) == {"AAA"}, "ZZZ leaked past the universe filter"
    stored = sqlite_store.load("short_interest")
    assert set(stored["ticker"]) == {"AAA"}
    assert len(stored) == len(out) + 1                 # prior row kept, new day added

    print("\n=== SANITY CHECK: RegSHO universe filter + upsert ===")
    print(f"  day-file had AAA+ZZZ -> kept {sorted(set(out['ticker']))} only; "
          f"table {len(stored)} rows (1 prior + {len(out)} new). Validated.")


def test_empty_download_still_records_and_returns_the_schema(sqlite_store, monkeypatch):
    """A holiday / all-404 window must not crash: `store.save` warns on the empty
    frame and the caller still gets the declared columns back."""
    monkeypatch.setattr(si, "record_run", lambda *a, **k: None)
    monkeypatch.setattr(si, "_fetch_day", lambda day: None)
    monkeypatch.setattr(si, "_resume_day",
                        lambda *a, **k: pd.Timestamp.today().normalize())

    out = si.fetch_short_interest(SimpleNamespace(store=sqlite_store),
                                  tickers=["AAA"], pause=0.0)

    assert out.empty
    assert list(out.columns) == ["date", "ticker", "short_volume", "total_volume"]
    print("\n=== SANITY CHECK: RegSHO empty window ===")
    print("  every day-file missing -> empty frame with the declared schema, no crash. Validated.")
