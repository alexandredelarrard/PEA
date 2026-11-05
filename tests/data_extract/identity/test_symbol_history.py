"""Symbol history (REQ-004, REQ-007, AC-002, AC-003): the Sharadar ticker-change tape attached through the CIK windows,
merged with the filed (Form 3/4/5, `dei`) and manual intervals, and the trading symbol of each frozen universe key.

Known-truth fixtures for the chain and merge rules; the named companies (META, BKNG, DOC, COHR, VMRK, PSKY, HPQ, VST, GOOGL,
TT, WBD) on the phase-0 fixture store with the phase-2 `sharadar_actions` refresh.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.identity.company_anchor import sharadar_ciks
from src.data_extract.utils.identity.symbol_history import TAPE_SOURCE, TICKER_CHANGE_ACTIONS, tape_chains
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.support.lineage_build import cik_history_of, symbol_history_of

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
RUN = Path("reports/validate/2026-11-03-identity-security-master/_cache")
CONFIG_DIR = "./configs"
NO_OWNERS = pd.DataFrame(columns=["issuer_cik", "owner_cik_raw"])


def _actions(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """`sharadar_actions` rows from (date, Sharadar ticker, old symbol, new symbol): one `from` and one `to` row each."""
    out = []
    for day, ticker, old, new in rows:
        out.append({"date": pd.Timestamp(day).date(), "action": "tickerchangefrom", "ticker": ticker, "contraticker": old})
        out.append({"date": pd.Timestamp(day).date(), "action": "tickerchangeto", "ticker": ticker, "contraticker": new})
    return pd.DataFrame(out)


def _tenure(rows: list[tuple[str, str, str, str | None, int]], source: str = "form345") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": s,
                "issuer_cik": c,
                "valid_from": pd.Timestamp(f),
                "valid_to": pd.Timestamp(t) if t else pd.NaT,
                "n_filings": n,
                "source": source,
                "evidence": c,
            }
            for s, c, f, t, n in rows
        ]
    )


def _span(rows: pd.DataFrame, symbol: str, cik: str | None = None) -> list[tuple[str, str | None, str]]:
    """(valid_from, valid_to or None, sources) of a company's rows of `symbol` (on `cik` when given)."""
    mine = rows[rows["role"].eq("symbol") & rows["symbol"].eq(symbol)]
    mine = mine[mine["cik"].eq(cik)] if cik is not None else mine
    return [
        (str(pd.Timestamp(r.valid_from).date()), None if pd.isna(r.valid_to) else str(pd.Timestamp(r.valid_to).date()), r.sources)
        for r in mine.sort_values("valid_from").itertuples(index=False)
    ]


# --------------------------------------------------------------------------- #
# known truth: the chain, the merge, the trading symbol                          #
# --------------------------------------------------------------------------- #
def test_a_ticker_change_chain_is_dated_from_the_first_price_date_to_the_last_change() -> None:
    """Sharadar keys every change by the CURRENT spelling (SKYD carries CBS -> ... -> SKYD); the chain is read per permaticker."""
    universe = pd.DataFrame({"ticker": ["PSKY", "BRK-B"], "permaticker": [7, 9]})
    sharadar = pd.DataFrame(
        {"permaticker": [7, 7, 9], "ticker": ["PSKY", "SKYD", "BRK.B"], "firstpricedate": [pd.Timestamp("1997-12-31").date()] * 2 + [None]}
    )
    actions = _actions([("2019-12-06", "SKYD", "CBS", "VIAC"), ("2025-08-07", "SKYD", "PARA", "PSKY"), ("2022-02-17", "SKYD", "VIAC", "PARA")])
    actions = pd.concat([actions, _actions([("2026-10-06", "SKYD", "PSKY", "SKYD"), ("2020-01-01", "OTHER", "X", "Y")])], ignore_index=True)

    chains = tape_chains(universe, sharadar, actions)

    got = [(r.symbol, str(r.first.date()), None if pd.isna(r.end) else str(r.end.date())) for r in chains.itertuples(index=False)]
    assert got == [
        ("CBS", "1997-12-31", "2019-12-06"),
        ("VIAC", "2019-12-06", "2022-02-17"),
        ("PARA", "2022-02-17", "2025-08-07"),
        ("PSKY", "2025-08-07", "2026-10-06"),
        ("SKYD", "2026-10-06", None),
    ], got
    assert set(chains["ticker"]) == {"PSKY"}, "BRK-B has no change and OTHER is no universe spelling"
    assert set(TICKER_CHANGE_ACTIONS) == {"tickerchangefrom", "tickerchangeto"}
    print("\n=== SANITY CHECK: Sharadar ticker-change chain ===")
    print(f"  PSKY (Sharadar spelling SKYD): {' -> '.join(f'{s} {a}..{b or "open"}' for s, a, b in got)}")
    print("  OK: one dated interval per symbol, first one from the first price date, last one open; other permatickers ignored")


def test_the_tape_extends_filed_intervals_and_dates_a_rename_the_filings_never_closed(tmp_path: Path) -> None:
    """A tape interval merges with the filed one of its (symbol, CIK): bounds are the union; a symbol only the tape knows is
    its own `single_source` row; the trading symbol is the latest open tape symbol, the universe key stays (D-6)."""
    tenure = _tenure([("OLD", "0000000100", "2006-01-05", "2020-05-20", 300), ("KEY", "0000000100", "2020-06-02", None, 40)])
    universe = pd.DataFrame({"ticker": ["KEY"], "permaticker": [1]})
    sharadar = pd.DataFrame({"permaticker": [1], "ticker": ["NEW"], "firstpricedate": [pd.Timestamp("1999-03-01").date()]})
    actions = _actions([("2020-06-01", "NEW", "OLD", "KEY"), ("2026-10-06", "NEW", "KEY", "NEW")])
    history = cik_history_of(tenure, {"KEY": "0000000100"}, NO_OWNERS, CONFIG_DIR)

    build = symbol_history_of(history, {"KEY": "0000000100"}, chains=tape_chains(universe, sharadar, actions))

    rows = build.rows
    assert _span(rows, "OLD") == [("1999-03-01", "2020-06-01", "form345,sharadar")]
    assert _span(rows, "KEY") == [("2020-06-01", None, "form345,roster,sharadar")], "the frozen key stays the roster anchor"
    assert _span(rows, "NEW") == [("2026-10-06", None, TAPE_SOURCE)]
    assert rows.loc[rows["symbol"].eq("NEW"), "status"].tolist() == ["single_source"]
    trading = build.trading.set_index("ticker")
    assert trading.loc["KEY", "trading_symbol"] == "NEW" and trading.loc["KEY", "trading_symbol_from"] == pd.Timestamp("2026-10-06")
    print("\n=== SANITY CHECK: tape merge and trading symbol ===")
    print(f"  OLD {_span(rows, 'OLD')}; KEY {_span(rows, 'KEY')}; NEW {_span(rows, 'NEW')}")
    print("  OK: the tape dates the rename the filings never closed; key KEY trades as NEW from 2026-10-06")


def test_a_tape_interval_is_cut_at_each_cik_window_and_clips_another_companys_derived_holder(tmp_path: Path) -> None:
    """The tape attaches only through the company's CIK windows; it is an anchor, so a derived holder of the same symbol
    in another company is clipped instead of both landing in `conflict`."""
    register = {
        "ABC": {
            "kind": "reorganisation",
            "segments": [{"cik": "100", "valid_to": "2015-01-01", "evidence": "old"}, {"cik": "200", "valid_from": "2015-01-01", "evidence": "new"}],
        }
    }
    sec = tmp_path / "configs" / "sec"
    sec.mkdir(parents=True)
    (sec / "registrant_cutover.json").write_text(json.dumps(register), encoding="utf-8")
    tenure = _tenure(
        [
            ("ABC", "0000000100", "2006-01-05", "2014-12-20", 300),
            ("ABC", "0000000200", "2015-01-05", None, 300),
            ("XYZ", "0000000300", "2016-01-05", "2019-01-01", 50),
            ("XYZ", "0000000100", "2006-01-05", "2014-12-20", 300),
        ]
    )
    universe = pd.DataFrame({"ticker": ["ABC"], "permaticker": [1]})
    sharadar = pd.DataFrame({"permaticker": [1], "ticker": ["ABC"], "firstpricedate": [pd.Timestamp("2000-01-03").date()]})
    actions = _actions([("2017-03-01", "ABC", "XYZ", "ABC")])
    history = cik_history_of(tenure, {"ABC": "0000000200"}, NO_OWNERS, str(tmp_path / "configs"))

    build = symbol_history_of(history, {"ABC": "0000000200"}, chains=tape_chains(universe, sharadar, actions))

    rows = build.rows
    assert _span(rows, "XYZ", "0000000100") == [("2000-01-03", "2015-01-01", "form345,sharadar")]
    assert _span(rows, "XYZ", "0000000200") == [("2015-01-01", "2017-03-01", TAPE_SOURCE)]
    other = build.holders[build.holders["cik"].eq("0000000300")]
    assert other["status"].tolist() == ["single_source"] and other["valid_from"].tolist() == [pd.Timestamp("2017-03-01")]
    assert not rows["status"].eq("conflict").any()
    print("\n=== SANITY CHECK: tape over CIK windows ===")
    print(f"  XYZ on 100 {_span(rows, 'XYZ', '0000000100')}; on 200 {_span(rows, 'XYZ', '0000000200')}")
    print("  OK: one tape interval split at the 2015-01-01 seam; the other company's XYZ clipped to start at the change, no conflict")


# --------------------------------------------------------------------------- #
# real data: the named companies on the fixture store                            #
# --------------------------------------------------------------------------- #
def _run_cache() -> Path:
    for base in (ROOT, *ROOT.parents):
        cache = base / RUN
        if (cache / "fixture_store" / "identity.sqlite").exists():
            return cache
    pytest.skip("the run's fixture store is not reachable from this checkout")


@pytest.fixture(scope="module")
def fixture_build():
    """The CIK history and the symbol history over the fixture store (read-only), with the Sharadar tape."""
    cache = _run_cache()
    owners_path = cache / "parity" / "owner_pairs.parquet"
    if not owners_path.exists():
        pytest.skip(f"no cached owner pairs at {owners_path} (run _scripts/parity_check.py first)")
    store = DataStore(create_engine(f"sqlite:///file:{(cache / 'fixture_store' / 'identity.sqlite').as_posix()}?mode=ro&uri=true"))
    tenure = store.load(Tables.symbol_tenure, project=True, where={"source": ["form345", "manual"]})
    dei = store.load(Tables.symbol_tenure, project=True, where={"source": "dei"})
    universe = store.load(Tables.universe_tickers, columns=["ticker", "cik", "permaticker"])
    sharadar = store.load(Tables.sharadar_tickers, columns=["permaticker", "ticker", "secfilings", "lastupdated", "firstpricedate"])
    actions = store.load(Tables.sharadar_actions, columns=["date", "action", "ticker", "contraticker"], where={"action": list(TICKER_CHANGE_ACTIONS)})
    profile = store.load(
        Tables.sec_cik_profile,
        columns=["cik", "form_25", "form_15", "form_8k12b"],
        where=[{"form_8k12b": store.NOT_NULL}, {"form_25": store.NOT_NULL}, {"form_15": store.NOT_NULL}],
    )
    existing = store.load(Tables.universe_cik_symbol, columns=["canonical_ticker", "company_id"])
    tenure = tenure.assign(valid_from=pd.to_datetime(tenure["valid_from"]), valid_to=pd.to_datetime(tenure["valid_to"]))
    roster = dict(zip(universe["ticker"], universe["cik"].astype(str).str.zfill(10), strict=True))
    history = cik_history_of(
        tenure,
        roster,
        pd.read_parquet(owners_path),
        CONFIG_DIR,
        dei=dei,
        existing=existing,
        profile=profile,
        sharadar_cik=sharadar_ciks(universe, sharadar),
    )
    chains = tape_chains(universe, sharadar, actions)
    return symbol_history_of(history, roster, chains=chains), chains, universe_changes(universe, sharadar, actions)


def universe_changes(universe: pd.DataFrame, sharadar: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """`[ticker, date, old, new, first_price]`: every Sharadar ticker change on a spelling of a universe permaticker, read
    independently of the build (a `tickerchangefrom` and a `tickerchangeto` of one spelling and date), in roster spelling."""
    spellings = universe[["ticker", "permaticker"]].merge(
        sharadar[["permaticker", "ticker", "firstpricedate"]].rename(columns={"ticker": "spelling"}), on="permaticker"
    )
    first = pd.to_datetime(spellings["firstpricedate"]).groupby(spellings["ticker"]).min().rename("first_price")
    frame = actions.assign(date=pd.to_datetime(actions["date"]), contraticker=actions["contraticker"].str.replace(".", "-", regex=False))
    old = frame[frame["action"].eq("tickerchangefrom")].rename(columns={"ticker": "spelling", "contraticker": "old"})
    new = frame[frame["action"].eq("tickerchangeto")].rename(columns={"ticker": "spelling", "contraticker": "new"})
    pairs = old[["spelling", "date", "old"]].merge(new[["spelling", "date", "new"]], on=["spelling", "date"]).merge(spellings, on="spelling")
    pairs = pairs[pairs["old"].ne(pairs["new"])].drop_duplicates(["ticker", "date"]).merge(first, left_on="ticker", right_index=True)
    return pairs[["ticker", "date", "old", "new", "first_price"]].reset_index(drop=True)


def change_coverage(rows: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Each change classified: `exact` (a row of the new symbol covers the date, one of the old symbol the day before),
    `curated` (manual rows date it from a filing instead; `days` = filing vs Sharadar), `pre_listing` (renamed before the
    first price date: no tape span), `pre_2006` (listed only: filings start 2006) or `missing`."""
    symbols = rows[rows["role"].eq("symbol") & rows["canonical_ticker"].notna() & ~rows["status"].isin(["noise", "conflict"])]
    out = []
    for ticker, day, old, new, first_price in changes.itertuples(index=False):
        mine = symbols[symbols["canonical_ticker"].eq(ticker)]
        new_rows, old_rows = mine[mine["symbol"].eq(new)], mine[mine["symbol"].eq(old)]
        covers_new = ((new_rows["valid_from"] <= day) & (new_rows["valid_to"].isna() | new_rows["valid_to"].gt(day))).any()
        covers_old = ((old_rows["valid_from"] < day) & (old_rows["valid_to"].isna() | old_rows["valid_to"].ge(day))).any()
        curated = new_rows["status"].eq("curated").any() and not old_rows.empty
        days = int((new_rows["valid_from"] - day).dt.days.abs().min()) if not new_rows.empty else None
        late = pd.notna(first_price) and day <= first_price and not new_rows.empty
        early = day < pd.Timestamp("2006-01-01")
        kind = "exact" if covers_new and covers_old else "pre_2006" if early else "curated" if curated else "pre_listing" if late else "missing"
        out.append({"ticker": ticker, "date": day, "old": old, "new": new, "coverage": kind, "days": days})
    return pd.DataFrame(out, columns=["ticker", "date", "old", "new", "coverage", "days"])


#: ticker: [(symbol, CIK or None, valid_from, valid_to)] its symbol history must carry (dates as the tape or the manual give them).
NAMED = {
    "META": [("FB", "0001326801", "2012-05-17", "2022-06-09"), ("META", "0001326801", "2022-06-09", None)],
    "BKNG": [("PCLN", "0001075531", "1999-03-31", "2018-02-28"), ("BKNG", "0001075531", "2018-02-28", None)],
    "DOC": [
        ("HCP", "0000765880", "1997-12-31", "2019-11-05"),
        ("PEAK", "0000765880", "2019-11-05", "2024-03-04"),
        ("DOC", "0000765880", "2024-03-04", None),
    ],
    "COHR": [("IIVI", "0000820318", "2006-02-09", "2022-09-08"), ("COHR", "0000820318", "2022-09-08", None)],
    "VMRK": [("EQR", "0000906107", "2006-01-05", "2026-08-18"), ("VMRK", "0000906107", "2026-08-18", None)],
    "PSKY": [
        ("PARA", "0000813828", "2022-02-17", "2025-08-08"),
        ("PSKY", "0002041610", "2025-08-07", None),
        ("SKYD", "0002041610", "2026-10-06", None),
    ],
    "HPQ": [("HWP", "0000047217", "1997-12-31", "2002-05-06"), ("HPQ", "0000047217", "2002-05-06", None)],
    "VST": [("VSTE", "0001692819", "2016-10-05", "2017-05-17"), ("VST", "0001692819", "2017-05-08", None)],
    "GOOGL": [("GOOG", "0001288776", "2004-08-19", "2015-10-02"), ("GOOGL", "0001652044", "2015-10-02", None)],
    "TT": [
        ("IR", "0001160497", "1997-12-31", "2009-07-01"),
        ("IR", "0001466258", "2009-07-09", "2020-03-02"),
        ("TT", "0001466258", "2020-03-02", None),
    ],
    "WBD": [("DISCA", "0001437107", "2008-09-18", "2022-04-11"), ("WBD", "0001437107", "2022-04-11", None)],
}


def test_the_named_companies_carry_their_dated_symbol_history(fixture_build) -> None:
    """Each named rename is a dated symbol row of its company: the tape dates it (FB->META, PCLN->BKNG, HCP->PEAK->DOC,
    HWP->HPQ, VSTE->VST, PSKY->SKYD), the manual where it already does (COHR, VMRK, TT, WBD)."""
    build, _, _ = fixture_build
    rows = build.rows
    print("\n=== SANITY CHECK: named symbol histories on the fixture store ===")
    for ticker, expected in NAMED.items():
        mine = rows[rows["canonical_ticker"].eq(ticker)]
        for symbol, cik, start, end in expected:
            spans = [(a, b) for a, b, _ in _span(mine, symbol, cik)]
            print(f"  {ticker:5s} {symbol:5s} {cik} {spans}")
            assert (start, end) in spans, (ticker, symbol, cik, spans)
    assert not rows.loc[rows["canonical_ticker"].eq("TT") & rows["cik"].eq("0001699150")].shape[0], "today's IR company stays out of TT"
    print("  OK: every named change is a dated symbol row; the reused IR stays with Ingersoll Rand Inc")


def test_psky_keeps_its_key_and_trades_as_skyd_and_wbd_models_disca_without_curation(fixture_build) -> None:
    """AC-002 / D-6: PSKY trades as SKYD from 2026-10-06 under its frozen key; VMRK trades as VMRK. WBD: Sharadar prices DISCA
    from 2005-07-08, but the manual DISCA interval (from 2008-09-18 on 1437107) governs and the one-day DISCA of Discovery
    Holding (1320482, 2008-09-18) is another company's noise: the 2005-2008 gap is curation (phase 8), not modelled away."""
    build, chains, _ = fixture_build
    trading = build.trading.set_index("ticker")
    assert (trading.loc["PSKY", "trading_symbol"], trading.loc["PSKY", "trading_symbol_from"]) == ("SKYD", pd.Timestamp("2026-10-06"))
    assert trading.loc["VMRK", "trading_symbol"] == "VMRK" and trading.loc["AAPL", "trading_symbol"] == "AAPL"
    renamed = trading[trading["trading_symbol"].ne(trading.index)]
    assert list(renamed.index) == ["PSKY"], renamed

    wbd_tape = chains[chains["ticker"].eq("WBD")]
    assert [(r.symbol, str(r.first.date())) for r in wbd_tape.itertuples(index=False)] == [("DISCA", "2005-07-08"), ("WBD", "2022-04-11")]
    holders = build.holders
    discovery = holders[holders["symbol"].eq("DISCA") & holders["cik"].eq("0001320482")]
    assert discovery["status"].tolist() == ["noise"] and discovery["canonical_ticker"].isna().all()
    disca = build.rows[build.rows["symbol"].eq("DISCA") & build.rows["canonical_ticker"].eq("WBD")]
    assert disca["valid_from"].min() == pd.Timestamp("2008-09-18") and set(disca["cik"]) == {"0001437107"}
    print("\n=== SANITY CHECK: PSKY, VMRK, WBD ===")
    print(f"  trading symbols differing from the key: {renamed['trading_symbol'].to_dict()} from {renamed['trading_symbol_from'].dt.date.to_dict()}")
    print(
        f"  WBD tape {[(r.symbol, str(r.first.date())) for r in wbd_tape.itertuples(index=False)]}; DISCA rows start {disca['valid_from'].min().date()}"
    )
    print("  OK: one renamed key (PSKY->SKYD); WBD's 2005-07-08..2008-09-18 DISCA span stays a phase-8 curation item")


def test_every_ticker_change_of_a_universe_company_is_a_symbol_interval(fixture_build) -> None:
    """AC-003: every Sharadar change of a universe company dated from 2006 is a dated symbol interval of its company: the
    tape's own dates or a manual interval citing the filing (its date wins; the gap is listed for curation); a rename before
    the first price date has no tape span; earlier changes are listed with Sharadar as their sole source."""
    build, _, changes = fixture_build

    coverage = change_coverage(build.rows, changes)

    counts = coverage["coverage"].value_counts().to_dict()
    curated = coverage[coverage["coverage"].eq("curated")]
    print("\n=== SANITY CHECK: AC-003 ticker changes ===")
    print(f"  {len(coverage)} changes over {coverage['ticker'].nunique()} universe tickers: {counts}")
    print(
        f"  dated by a manual interval instead (days off Sharadar): {[(r.ticker, f'{r.old}->{r.new}', r.days) for r in curated.itertuples(index=False)]}"
    )
    missing = coverage[coverage["coverage"].eq("missing")]
    assert missing.empty, missing.to_string()
    print("  OK: every change from 2006 is a dated symbol interval; pre-2006 changes are listed from the tape alone")
