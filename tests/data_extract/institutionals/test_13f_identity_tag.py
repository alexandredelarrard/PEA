"""AC-009 / REQ-012 and user decisions X2 + the superinvestor-book correction: the two-layer 13F tag.

On the phase-0 fixture store (read-only): the phase-5 `universe_cusip`, the market-wide `cusip_ticker_map` and the
13F holder spellings per (CUSIP, period) of the whole market (`sec13f_cusip_period`, the stored `sec13f_hr` tags).
- a universe company's CUSIP takes its dated line: EQR's 29476L107 -> VMRK, every share class on the company ticker
  (GOOG -> GOOGL, BRK-A -> BRK-B, NWS -> NWSA, FOX -> FOXA);
- any other CUSIP keeps its `cusip_ticker_map` ticker: a superinvestor book keeps every non-universe holding tagged,
  the same count as the old map-only rule;
- the denominator of a combined company is all-class: Sharadar `sharesbas` of GOOGL, FOXA and NWSA is the sum of
  their classes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_aggregate.utils.sec_thirteen.holdings_clean import attach_tickers
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.identity_tables import LINE_COLUMNS, CusipTags, tag_cusips

ROOT = Path(__file__).resolve().parents[3]
RUN = Path("reports") / "validate" / "2026-11-03-identity-security-master" / "_cache"
SECOND_CLASSES = {"02079K107": "GOOGL", "084670108": "BRK-B", "65249B208": "NWSA", "35137L204": "FOXA"}


@pytest.fixture(scope="module")
def fixture() -> tuple[DataStore, CusipTags]:
    for base in (ROOT, *ROOT.parents):
        cache = base / RUN
        published = cache / "parity" / "published_5_runA_universe_cusip.parquet"
        if (cache / "fixture_store" / "identity.sqlite").exists() and published.exists():
            store = DataStore(create_engine(f"sqlite:///file:{(cache / 'fixture_store' / 'identity.sqlite').as_posix()}?mode=ro&uri=true"))
            lines = pd.read_parquet(published)[LINE_COLUMNS]
            return store, CusipTags(lines, store.load(Tables.cusip_ticker_map, columns=["cusip", "ticker"]))
    pytest.skip("the run's fixture store and phase-5 publish are not reachable from this checkout")


def _holders(store: DataStore, where: str) -> pd.DataFrame:
    """Rows of the fixture-only aggregate `sec13f_cusip_period` (no registry entry: read by SQL)."""
    return pd.read_sql(f"select cusip, period, ticker, n_rows from sec13f_cusip_period where {where}", store.engine)


def test_universe_cusips_take_their_dated_company_line(fixture: tuple[DataStore, CusipTags]) -> None:
    store, tags = fixture
    rows = _holders(store, "cusip in ({})".format(",".join(f"'{c}'" for c in ["29476L107", *SECOND_CLASSES])))
    rows = rows.assign(after=tag_cusips(rows, tags.lines, tags.cmap), on_line=rows["cusip"].isin(set(tags.lines["cusip"])))
    recent = rows[pd.to_datetime(rows["period"]) >= "2016-01-01"]

    eqr = recent[recent["cusip"].eq("29476L107")]
    assert set(eqr["after"]) == {"VMRK"} and eqr["on_line"].all(), "EQR's CUSIP is tagged through its universe_cusip line"
    for cusip, company in SECOND_CLASSES.items():
        assert set(recent.loc[recent["cusip"].eq(cusip), "after"].dropna()) == {company}, cusip
    moved = rows.groupby(["cusip", "ticker", "after"], dropna=False)["n_rows"].sum()
    print("\n=== SANITY CHECK: 13F tags of universe CUSIPs (fixture holder rows) ===")
    print(f"  EQR 29476L107: {int(eqr['n_rows'].sum())} holder rows from 2016 all tagged VMRK through universe_cusip")
    for (cusip, before, after), n in moved.items():
        print(f"  {cusip}: {before} -> {after}: {int(n)} rows")
    print("  Validated: second classes land on their company's universe ticker.")


def test_a_superinvestor_book_keeps_its_non_universe_holdings(fixture: tuple[DataStore, CusipTags]) -> None:
    store, tags = fixture
    # one quarter's market-wide holdings as a book: every CUSIP held at 2025-12-31
    book = _holders(store, "period = '2025-12-31'")[["cusip", "period"]]
    book = book[book["cusip"].str.len().eq(9)].drop_duplicates("cusip").reset_index(drop=True)
    known = set(tags.lines["cusip"])

    # the superinvestor path tags with no universe filter (`clean_public_batches(..., universe=None)`)
    tagged = attach_tickers(book, tags)
    old = book.merge(tags.cmap.dropna().drop_duplicates("cusip"), on="cusip", how="left")["ticker"]
    outside = ~book["cusip"].isin(known)
    n_new, n_old = int(tagged.loc[outside, "ticker"].notna().sum()), int(old[outside].notna().sum())

    assert n_new == n_old > 1000
    # AvalonBay (053484101) is VMRK's acquired constituent in universe_cusip, not a tagged line: it keeps its own ticker
    assert set(tagged.loc[book["cusip"].eq("053484101"), "ticker"]) == {"AVB"}
    assert (tagged.loc[outside, "ticker"].fillna("") == old[outside].fillna("")).all()
    print("\n=== SANITY CHECK: superinvestor book at 2025-12-31 (whole-market CUSIPs) ===")
    print(f"  {len(book)} CUSIPs; {int(outside.sum())} outside universe_cusip: {n_new} tagged now vs {n_old} by the old map-only rule")
    print(f"  {int((~outside).sum())} universe CUSIPs tagged by their dated line: {int(tagged.loc[~outside, 'ticker'].notna().sum())}")
    print("  AVB (053484101, an acquired constituent of VMRK in universe_cusip) stays AVB")
    print("  Validated: non-universe holdings keep their market-wide ticker.")


def test_the_combined_company_denominator_covers_every_class(fixture: tuple[DataStore, CusipTags]) -> None:
    store, _ = fixture
    shares = store.load(
        Tables.sharadar_fundamentals,
        columns=["ticker", "dimension", "date", "sharesbas"],
        where={"ticker": ["GOOGL", "FOXA", "NWSA"], "dimension": "ARQ"},
    )
    latest = shares.sort_values("date").groupby("ticker")["sharesbas"].last()
    # known truth (10-Q cover pages, 2026): Alphabet A+B+C ~12.1-12.3 bn; Fox A+B ~0.42-0.46 bn; News Corp A+B ~0.54-0.59 bn
    assert 11.5e9 < latest["GOOGL"] < 13e9 and 0.40e9 < latest["FOXA"] < 0.50e9 and 0.52e9 < latest["NWSA"] < 0.62e9
    print("\n=== SANITY CHECK: all-class share counts of combined companies ===")
    print(f"  Sharadar sharesbas: GOOGL {latest['GOOGL'] / 1e9:.2f} bn, FOXA {latest['FOXA'] / 1e9:.3f} bn, NWSA {latest['NWSA'] / 1e9:.3f} bn")
    print("  Validated: each is the sum of the company's classes, so a combined 13F numerator is over the right base.")
