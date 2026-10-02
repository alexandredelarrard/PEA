"""Projected / scoped resume reads return the same keys as the full-table read they replaced.

Each test seeds a real SQLite `DataStore`, recomputes the answer from an unprojected
`store.load(table)` the way the old code did, and checks the projected read agrees:
earnings-surprise plan, stored earnings-call quality sets (chunked, scoped), the
`sp500_tickers` roster behind `load_identity`, and the CUSIP map cache.
"""

from __future__ import annotations

import types
from typing import Any

import pandas as pd

from src.constants.constants import CUSIP_TICKER_OVERRIDES, EARNINGS_CALL_SCORED_TAGS
from src.data_extract.utils.behavioral import fetch_earnings_calls as fe
from src.data_extract.utils.behavioral.utils_missing_quarters import stored_call_quarters
from src.data_extract.utils.common.entity_lineage import ROSTER_COLUMNS
from src.data_extract.utils.common.identity import build_identity, load_identity
from src.data_extract.utils.fundamentals import fetch_earnings_surprises as surprises
from src.data_extract.utils.institutionals.fetch_cusip_map import build_cusip_ticker_map, normalize_cusip
from src.data_store.schema import Tables
from src.utils.text_metrics import assess_earnings_call_sections

CONFIG_DIR = "configs"
_WORDS = " ".join(f"word{i}" for i in range(400))


class _Context:
    """A weak-referenceable context (`load_identity` caches per context) over a real store."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
        self.config_dir = CONFIG_DIR
        self.config = types.SimpleNamespace(data_extract=types.SimpleNamespace(redundant_ticks=[], years_history=15))


def _context(store: Any) -> Any:
    return _Context(store)


def _spy_loads(monkeypatch: Any, store: Any, table_name: str) -> list[dict[str, Any]]:
    """Record the keyword arguments of every `store.load` of `table_name`."""
    calls: list[dict[str, Any]] = []
    original = store.load

    def load(table: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(table, "name", table) == table_name:
            calls.append(kwargs)
        return original(table, *args, **kwargs)

    monkeypatch.setattr(store, "load", load)
    return calls


def test_earnings_surprise_plan_is_unchanged_by_the_projected_read(sqlite_store: Any, monkeypatch: Any) -> None:
    today = pd.Timestamp.today().normalize()
    df_stored = pd.DataFrame(
        {
            "ticker": ["FRESH", "FRESH", "STALE", "NOACTUAL", "DUE", "DUE"],
            "earnings_date": [today - pd.Timedelta(days=d) for d in (10, 100, 200, -20, 90, 2)],
            "eps_estimate": [1.0, 1.0, 1.0, 2.0, 1.0, 1.2],
            "eps_actual": [1.1, 1.0, 0.9, None, 1.0, None],
            "surprise_pct": [10.0, 0.0, -10.0, None, 0.0, None],
        }
    )
    df_stored["earnings_date"] = df_stored["earnings_date"].dt.strftime("%Y-%m-%d")
    sqlite_store.save(Tables.earnings_surprises, df_stored)
    tickers = ["FRESH", "STALE", "NOACTUAL", "DUE", "NEW"]

    df_full = sqlite_store.load(Tables.earnings_surprises)
    df_full["earnings_date"] = pd.to_datetime(df_full["earnings_date"], format="%Y-%m-%d")
    expected = surprises._plan_fetch(tickers, df_full, 64, 95)

    plans: list[list[tuple[str, int]]] = []
    plan_fetch = surprises._plan_fetch
    monkeypatch.setattr(surprises, "_plan_fetch", lambda *a: plans.append(plan_fetch(*a)) or plans[-1])
    monkeypatch.setattr(surprises, "_download_one", lambda ticker, limit: None)
    monkeypatch.setattr(surprises, "record_run", lambda *a, **k: None)
    loads = _spy_loads(monkeypatch, sqlite_store, Tables.earnings_surprises.name)
    surprises.fetch_earnings_surprises(_context(sqlite_store), tickers, pause=0.0)

    assert loads[0]["columns"] == ["ticker", "earnings_date", "eps_actual"]
    assert plans == [expected]
    print("\n=== SANITY CHECK: earnings-surprise resume read ===")
    print(f"  projected read of 3 columns -> plan {plans[0]}; full read -> {expected}")
    print("  OK: identical fetch plan from the projected resume read")


def _section_rows() -> pd.DataFrame:
    """30 tickers (two 25-ticker chunks): a valid call, a call valid for 2 tickers in 3 (else an
    empty prepared section), and an unscored tag."""
    prepared, qa = EARNINGS_CALL_SCORED_TAGS[0], EARNINGS_CALL_SCORED_TAGS[1]
    rows = []
    for i in range(30):
        ticker = f"T{i:02d}"
        rows += [
            {"ticker": ticker, "quarter": "2025Q1", "tag": prepared, "as_of": "2025-05-01", "text": _WORDS},
            {"ticker": ticker, "quarter": "2025Q1", "tag": qa, "as_of": "2025-05-01", "text": _WORDS},
            {"ticker": ticker, "quarter": "2025Q2", "tag": prepared, "as_of": "2025-08-01", "text": _WORDS if i % 3 else ""},
            {"ticker": ticker, "quarter": "2025Q2", "tag": qa, "as_of": "2025-08-01", "text": _WORDS},
            {"ticker": ticker, "quarter": "2025Q3", "tag": "participants", "as_of": "2025-11-01", "text": _WORDS},
        ]
    return pd.DataFrame(rows)


def _full_read_quality(store: Any) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """The old algorithm: one unprojected read, scored tags only, gate per (ticker, quarter)."""
    df_all = store.load(Tables.earnings_call_sections)
    df_scored = df_all[df_all["tag"].isin(EARNINGS_CALL_SCORED_TAGS)]
    valid, malformed = set(), set()
    for (ticker, quarter), df_call in df_scored.groupby(["ticker", "quarter"]):
        ok = assess_earnings_call_sections(dict(zip(df_call["tag"].astype(str), df_call["text"], strict=False))).valid
        (valid if ok else malformed).add((str(ticker), str(quarter)))
    return valid, malformed


def test_stored_call_quarters_match_the_full_read(sqlite_store: Any, monkeypatch: Any) -> None:
    sqlite_store.save(Tables.earnings_call_sections, _section_rows())
    expected_valid, expected_malformed = _full_read_quality(sqlite_store)
    tickers = sorted({t for t, _ in expected_valid | expected_malformed})
    loads = _spy_loads(monkeypatch, sqlite_store, Tables.earnings_call_sections.name)

    valid, malformed = stored_call_quarters(_context(sqlite_store), tickers)
    got_valid = {(t, q) for t, qs in valid.items() for q in qs}
    got_malformed = {(t, q) for t, qs in malformed.items() for q in qs}
    every_key = fe._existing_section_keys(_context(sqlite_store), None)
    scoped_key = fe._existing_section_keys(_context(sqlite_store), ["T01", "T03", "ABSENT"])

    assert (got_valid, got_malformed) == (expected_valid, expected_malformed)
    assert every_key == expected_valid
    assert scoped_key == {k for k in expected_valid if k[0] in {"T01", "T03"}}
    assert {tuple(call["columns"]) for call in loads} == {("ticker", "quarter", "tag", "text")}
    assert all(len(call["where"]["ticker"]) <= 25 for call in loads)
    print("\n=== SANITY CHECK: stored earnings-call quality ===")
    print(f"  {len(got_valid)} valid / {len(got_malformed)} malformed calls over {len(tickers)} tickers in {len(loads)} chunked reads")
    print("  OK: chunked, projected and ticker-scoped reads give the full read's sets")


def test_load_identity_with_the_projected_roster_matches_the_full_roster(sqlite_store: Any) -> None:
    sqlite_store.save(
        Tables.entity_lineage,
        pd.DataFrame(
            [
                {"cik": "0000000100", "entity_id": "E0000000100", "source": "register", "confidence": None, "evidence": "t"},
                {"cik": "0000000200", "entity_id": "E0000000100", "source": "register", "confidence": None, "evidence": "t"},
                {"cik": "0000000900", "entity_id": "E0000000900", "source": "roster", "confidence": None, "evidence": "t"},
            ]
        ),
    )
    sqlite_store.save(
        Tables.symbol_tenure,
        pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "issuer_cik": "0000000100",
                    "valid_from": "2006-01-01",
                    "valid_to": None,
                    "n_filings": 40,
                    "source": "form345",
                    "evidence": "",
                },
                {
                    "symbol": "BBB",
                    "issuer_cik": "0000000900",
                    "valid_from": "2010-01-01",
                    "valid_to": None,
                    "n_filings": 9,
                    "source": "form345",
                    "evidence": "",
                },
            ]
        ),
    )
    roster = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "cik": ["100", "900"],
            "name": ["A Co", "B Co"],
            "sector": ["X", "Y"],
            "industry_group": ["X1", "Y1"],
            "sub_industry": ["X11", "Y11"],
        }
    )
    sqlite_store.save(Tables.sp500_tickers, roster)

    projected = load_identity(_context(sqlite_store), CONFIG_DIR, refresh=True)
    full = build_identity(
        lineage=sqlite_store.load(Tables.entity_lineage, project=True),
        tenure=sqlite_store.load(Tables.symbol_tenure, project=True),
        roster=sqlite_store.load(Tables.sp500_tickers),
        d19_allowlist={},
        redundant_symbols=frozenset(),
    )

    assert ROSTER_COLUMNS == ("ticker", "cik")
    assert projected.roster_cik == full.roster_cik == {"AAA": "0000000100", "BBB": "0000000900"}
    assert projected.ticker_by_entity == full.ticker_by_entity
    print("\n=== SANITY CHECK: projected sp500_tickers roster ===")
    print(f"  roster_cik {dict(projected.roster_cik)}; ticker_by_entity {dict(projected.ticker_by_entity)}")
    print("  OK: identity built from ('ticker', 'cik') equals the one built from the full roster")


def test_cusip_map_projected_cache_read_returns_the_full_cache(sqlite_store: Any, monkeypatch: Any) -> None:
    override = next(iter(CUSIP_TICKER_OVERRIDES))
    sqlite_store.save(
        Tables.cusip_ticker_map,
        pd.DataFrame(
            {"cusip": ["037833100", "594918104", "000000001", override], "ticker": ["AAPL", "MSFT", None, CUSIP_TICKER_OVERRIDES[override]]}
        ),
    )
    df_full = sqlite_store.load(Tables.cusip_ticker_map)
    monkeypatch.setattr(
        "src.data_extract.utils.institutionals.fetch_cusip_map._openfigi_request", lambda *a: (_ for _ in ()).throw(AssertionError("network"))
    )
    loads = _spy_loads(monkeypatch, sqlite_store, Tables.cusip_ticker_map.name)

    df_map = build_cusip_ticker_map(_context(sqlite_store), ["37833100", "594918104"])

    expected = {normalize_cusip(c): t for c, t in zip(df_full["cusip"], df_full["ticker"], strict=True) if pd.notna(t)}
    expected |= {normalize_cusip(c): t for c, t in CUSIP_TICKER_OVERRIDES.items()}
    assert loads[0]["columns"] == ["cusip", "ticker"]
    assert dict(zip(df_map["cusip"], df_map["ticker"], strict=True)) == expected
    print("\n=== SANITY CHECK: CUSIP map cache read ===")
    print(f"  {len(df_map)} mapped CUSIPs returned (2 cached + {len(CUSIP_TICKER_OVERRIDES)} overrides), no OpenFIGI call")
    print("  OK: the projected cache read returns the whole mapped cache, as the full read did")
