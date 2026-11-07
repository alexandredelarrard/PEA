"""Lifecycle stage (phase 4d): first price date, the one end date (D-4), status and trading symbol of each universe ticker.

The named companies on the phase-0 fixture store (read-only) with the 4b lineage the parity run wrote: WBD ended
(AC-001), PSKY renamed to SKYD and VMRK continuing (AC-002), AAPL open. Known-truth frames for the end-date dispute rule
(A-3: Sharadar's end and the last stored price more than 5 sessions apart), for the end dates the anchor reads, and for
the closing of an ended company's open rows inside the symbol and CUSIP stamp passes (REQ-006, X57).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.identity.cik_history import BACKLOG_COLUMNS, BLOCKED_COLUMNS, CikHistory
from src.data_extract.utils.identity.cik_history import TABLE_COLUMNS as LINEAGE_COLUMNS
from src.data_extract.utils.identity.cusip_history import TABLE_COLUMNS as CUSIP_COLUMNS
from src.data_extract.utils.identity.cusip_history import finalise_rows
from src.data_extract.utils.identity.lifecycle import (
    LIFE_COLUMNS,
    PRICE_SPAN,
    Lifecycle,
    company_lifecycle,
    ended_dates,
    universe_permatickers,
)
from src.data_extract.utils.identity.symbol_history import symbol_rows
from src.data_store.schema import Tables
from src.data_store.store import DataStore

ROOT = Path(__file__).resolve().parents[3]
RUN = Path("reports") / "validate" / "2026-11-03-identity-security-master" / "_cache"


def _run_cache() -> Path:
    for base in (ROOT, *ROOT.parents):
        cache = base / RUN
        if (cache / "fixture_store" / "identity.sqlite").exists():
            return cache
    pytest.skip("the run's fixture store is not reachable from this checkout")


@pytest.fixture(scope="module")
def fixture_lifecycle() -> tuple[Lifecycle, pd.DataFrame]:
    """The lifecycle compute over the fixture store (read-only) and the 4b lineage; also the Sharadar rows read."""
    cache = _run_cache()
    lineage_path = cache / "parity" / "lineage_4b_new.parquet"
    if not lineage_path.exists():
        pytest.skip(f"no 4b lineage at {lineage_path} (run _scripts/parity_check.py --phase 4b first)")
    path = cache / "fixture_store" / "identity.sqlite"
    store = DataStore(create_engine(f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"))
    universe = store.load(Tables.universe_tickers, columns=["ticker", "permaticker"])
    sharadar = store.load(Tables.sharadar_tickers, columns=list(LIFE_COLUMNS), where={"permaticker": universe_permatickers(universe)})
    prices = store.summarize(Tables.prices, "ticker", dict(PRICE_SPAN), where={"ticker": list(universe["ticker"])})
    return company_lifecycle(universe, sharadar, prices, pd.read_parquet(lineage_path)), sharadar


def _row(lifecycle: Lifecycle, ticker: str) -> pd.Series:
    rows = lifecycle.rows[lifecycle.rows["ticker"].eq(ticker)]
    assert len(rows) == 1, rows
    return rows.iloc[0]


def test_wbd_ends_on_its_last_sharadar_trading_day(fixture_lifecycle: tuple[Lifecycle, pd.DataFrame]) -> None:
    """AC-001: Sharadar delists WBD with lastpricedate 2026-10-05; the stored prices stop 2026-10-06, one session apart, so no item."""
    lifecycle, _ = fixture_lifecycle
    wbd = _row(lifecycle, "WBD")

    assert wbd["ended_on"] == pd.Timestamp("2026-10-05") and wbd["status"] == "ended", wbd
    assert wbd["company_id"] == "E0001437107" and wbd["trading_symbol"] == "WBD", wbd
    assert "WBD" not in set(lifecycle.items["ticker"]), lifecycle.items
    print("\n=== SANITY CHECK: WBD ended (AC-001) ===")
    print(f"  WBD {wbd['company_id']}: ended_on {wbd['ended_on'].date()}, status {wbd['status']}, first price {wbd['first_price_date'].date()}")
    print("  OK: the end is Sharadar's last trading day; the last stored price (2026-10-06) is within 5 sessions")


def test_psky_trades_as_skyd_and_vmrk_continues_neither_ends(fixture_lifecycle: tuple[Lifecycle, pd.DataFrame]) -> None:
    """AC-002 / D-6: PSKY keeps its key, trades as SKYD (renamed), no end; VMRK (ex-EQR) is active under its own key."""
    lifecycle, sharadar = fixture_lifecycle
    psky, vmrk = _row(lifecycle, "PSKY"), _row(lifecycle, "VMRK")

    assert psky["trading_symbol"] == "SKYD" and psky["status"] == "renamed" and pd.isna(psky["ended_on"]), psky
    assert vmrk["trading_symbol"] == "VMRK" and vmrk["status"] == "active" and pd.isna(vmrk["ended_on"]), vmrk
    rows = sharadar[sharadar["ticker"].isin(["PSKY", "SKYD"])][["ticker", "lastpricedate", "isdelisted"]].drop_duplicates()
    print("\n=== SANITY CHECK: PSKY renamed, VMRK continues (AC-002) ===")
    print(f"  Sharadar rows of PSKY's permaticker: {rows.to_dict('records')}")
    print(f"  PSKY -> trading {psky['trading_symbol']}, status {psky['status']}, ended_on {psky['ended_on']}")
    print(f"  VMRK -> trading {vmrk['trading_symbol']}, status {vmrk['status']}, ended_on {vmrk['ended_on']}")
    print("  OK: a ticker change is not an end; the newest Sharadar row of the permaticker (SKYD, listed) decides")


def test_aapl_is_open_from_its_earliest_price_evidence(fixture_lifecycle: tuple[Lifecycle, pd.DataFrame]) -> None:
    """AAPL: no end, active; `first_price_date` is the earlier of Sharadar's (its 1997-12-31 coverage floor) and the first stored price."""
    lifecycle, sharadar = fixture_lifecycle
    aapl = _row(lifecycle, "AAPL")

    assert pd.isna(aapl["ended_on"]) and aapl["status"] == "active" and aapl["trading_symbol"] == "AAPL", aapl
    assert aapl["first_price_date"] == pd.Timestamp("1995-10-09"), aapl
    floor = int(pd.to_datetime(sharadar.drop_duplicates("permaticker")["firstpricedate"]).eq(pd.Timestamp("1997-12-31")).sum())
    counts = lifecycle.rows["status"].value_counts().to_dict()
    print("\n=== SANITY CHECK: AAPL open, first price date ===")
    print(f"  AAPL first_price_date {aapl['first_price_date'].date()} (stored prices) before Sharadar's 1997-12-31 floor")
    print(f"  {floor} universe permaticker(s) carry Sharadar's 1997-12-31 floor; statuses over {len(lifecycle.rows)} tickers: {counts}")
    print(f"  end-date items on the fixture: {len(lifecycle.items)}")
    print("  OK: an open company has no end; the first price is the earliest evidence, never later than a stored price")


def _lineage(tickers: list[str]) -> pd.DataFrame:
    """One roster window per ticker, no symbol rows (trading symbol = the key)."""
    return pd.DataFrame(
        {
            "company_id": [f"E{i:010d}" for i in range(1, len(tickers) + 1)],
            "canonical_ticker": tickers,
            "cik": [f"{i:010d}" for i in range(1, len(tickers) + 1)],
            "role": "cik_window",
            "symbol": "",
            "valid_from": pd.Timestamp("1900-01-01"),
            "valid_to": pd.NaT,
            "status": "curated",
            "sources": "roster",
        }
    )


def _known_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """XXX and YYY delisted on 2026-01-02, ZZZ listed with an old lastpricedate."""
    universe = pd.DataFrame({"ticker": ["XXX", "YYY", "ZZZ"], "permaticker": [1, 2, 3]})
    sharadar = pd.DataFrame(
        {
            "permaticker": [1, 2, 3],
            "ticker": ["XXX", "YYY", "ZZZ"],
            "firstpricedate": ["2010-01-04", "2012-05-01", "2015-01-02"],
            "lastpricedate": ["2026-01-02", "2026-01-02", "2025-06-30"],
            "isdelisted": ["Y", "Y", "N"],
            "lastupdated": ["2026-03-01"] * 3,
        }
    )
    return universe, sharadar


def test_an_end_far_from_the_last_stored_price_is_an_action_item() -> None:
    """A-3 known truth: XXX delisted 2026-01-02 with prices to 2026-02-27 (40 sessions): one item; YYY 1 session apart: none;
    ZZZ listed with an old lastpricedate: open, no item."""
    universe, sharadar = _known_universe()
    prices = pd.DataFrame(
        {
            "ticker": ["XXX", "YYY", "ZZZ"],
            "price_first": ["2009-06-01", "2012-05-01", "2015-01-02"],
            "price_last": ["2026-02-27", "2026-01-05", "2026-03-02"],
        }
    )
    lifecycle = company_lifecycle(universe, sharadar, prices, _lineage(["XXX", "YYY", "ZZZ"]))

    assert list(lifecycle.items["ticker"]) == ["XXX"] and bool(lifecycle.items["action"].iloc[0]), lifecycle.items
    assert list(lifecycle.rows["status"]) == ["ended", "ended", "active"], lifecycle.rows
    assert lifecycle.rows["first_price_date"].tolist() == [pd.Timestamp("2009-06-01"), pd.Timestamp("2012-05-01"), pd.Timestamp("2015-01-02")]
    print("\n=== SANITY CHECK: end-date dispute (A-3) ===")
    print(f"  {lifecycle.items['evidence'].iloc[0]}")
    print("  OK: 40 sessions apart is an action item, 1 session is lag; a listed company is open whatever its lastpricedate")


def test_the_anchor_end_dates_are_the_delisted_companies_last_sharadar_day() -> None:
    """D-4 known truth: the two delisted permatickers end on their lastpricedate; the listed one has no end."""
    universe, sharadar = _known_universe()

    ends = ended_dates(universe, sharadar)

    assert ends == {"XXX": pd.Timestamp("2026-01-02"), "YYY": pd.Timestamp("2026-01-02")}, ends
    print("\n=== SANITY CHECK: anchor end dates (D-4) ===")
    print(f"  {ends}")
    print("  OK: the same rule as the lifecycle's ended_on, read before the stamp passes")


END = pd.Timestamp("2026-10-05")


def _rows(role: str, symbols: tuple[str, str]) -> pd.DataFrame:
    """END and OPN, one open row each of `role`."""
    start = "1900-01-01" if role == "cik_window" else "2006-01-03"
    return pd.DataFrame(
        {
            "company_id": ["E0000000001", "E0000000002"], "canonical_ticker": ["END", "OPN"], "cik": ["0000000001", "0000000002"],
            "role": role, "symbol": list(symbols), "valid_from": pd.to_datetime([start, start]),
            "valid_to": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"), "status": "curated", "sources": "roster", "oracle": "roster",
            "confidence": float("nan"), "n_observations": 5, "evidence": "",
        }
    )  # fmt: skip


def _history(ended_on: dict[str, pd.Timestamp]) -> tuple[CikHistory, pd.DataFrame]:
    """END and OPN, each one open roster window and one open tape symbol interval."""
    history = CikHistory(
        rows=_rows("cik_window", ("", ""))[list(LINEAGE_COLUMNS[:-1])],
        entity_of={"0000000001": "E0000000001", "0000000002": "E0000000002"},
        ticker_by_entity={"E0000000001": "END", "E0000000002": "OPN"},
        window_end={},
        oracle_of={},
        backlog=pd.DataFrame(columns=list(BACKLOG_COLUMNS)),
        blocked=pd.DataFrame(columns=list(BLOCKED_COLUMNS)),
        evidence=pd.DataFrame(),
        activity=pd.DataFrame(),
        ended_on=ended_on,
    )
    return history, _rows("symbol", ("END", "OPN")).assign(sources="form345,sharadar,roster")


def test_an_ended_companys_open_lineage_rows_close_at_its_end_inside_the_stamp_pass() -> None:
    """REQ-006 / X57: END's open window and symbol interval close on the day after its end (valid_to exclusive); a second
    build keeps every stamp; the same rows compared without the end are re-stamped, so closing after the stamp would
    re-stamp END on every run."""
    t1, t2 = pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-02")
    history, holders = _history({"END": END})

    first = symbol_rows(history, holders, None, t1).rows
    second = symbol_rows(history, holders, first, t2).rows
    unclosed = symbol_rows(replace(history, ended_on={}), holders, first, t2).rows

    end_rows, open_rows = first[first["canonical_ticker"].eq("END")], first[first["canonical_ticker"].eq("OPN")]
    assert end_rows["valid_to"].eq(END + pd.Timedelta(days=1)).all() and len(end_rows) == 2, end_rows
    assert open_rows["valid_to"].isna().all(), open_rows
    assert second["scope_changed_at"].eq(t1).all(), second
    assert unclosed.loc[unclosed["canonical_ticker"].eq("END"), "scope_changed_at"].eq(t2).all()
    print("\n=== SANITY CHECK: lineage closed at ended_on (REQ-006) ===")
    print(f"  END rows {end_rows[['role', 'symbol', 'valid_to']].to_dict('records')}; OPN rows open")
    print(f"  second build: stamps {sorted(set(second['scope_changed_at'].astype(str)))}; without the end END is re-stamped {t2.date()}")
    print("  OK: the closing sits inside the stamp pass, so a second run writes nothing")


def test_an_ended_companys_open_cusip_lines_close_at_its_end_inside_the_stamp_pass() -> None:
    """REQ-006 / X57: END's open CUSIP line closes on the day after its end, an already closed line and OPN's line are
    left; a second finalise keeps the stamps."""
    t1, t2 = pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-02")
    rows = []
    lines = (("END", "111111101", None), ("END", "111111200", "2012-01-03"), ("OPN", "222222101", None))
    for company, cusip, end in lines:
        row = dict.fromkeys(c for c in CUSIP_COLUMNS if c != "scope_changed_at")
        row.update(
            security_id=f"C{cusip}", canonical_company=company, source="ftd", source_symbol=company, cusip=cusip, conversion_ratio=1.0,
            lineage_role="canonical_current", valid_from=pd.Timestamp("2009-07-01"), valid_to=pd.Timestamp(end) if end else None, n_observations=3,
        )  # fmt: skip
        rows.append(row)

    first = finalise_rows(rows, None, t1, {"END": END})
    second = finalise_rows(rows, first, t2, {"END": END})

    ends = dict(zip(first["cusip"], first["valid_to"], strict=True))
    assert ends["111111101"] == END + pd.Timedelta(days=1) and ends["111111200"] == pd.Timestamp("2012-01-03") and pd.isna(ends["222222101"]), ends
    assert second["scope_changed_at"].eq(t1).all(), second
    print("\n=== SANITY CHECK: CUSIP lines closed at ended_on (REQ-006) ===")
    print(f"  valid_to by CUSIP: { {c: (None if pd.isna(v) else str(v.date())) for c, v in ends.items()} }")
    print("  OK: only the ended company's open line closes; a second finalise keeps the stamps")
