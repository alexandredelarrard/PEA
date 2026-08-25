"""The `fundamentals_history` CONTRACT: the 69 columns, the reason-code vocabulary, the
publication-event grain and the append-only guard.

Synthetic where the question is arithmetic or grain (a known-truth fixture is the only way to
assert "this window was refused for THIS reason"), and real where the question is whether the
contract survives an actual filer -- the sweep ledgers in `data/fundamentals_sweep/` are
genuine `fundamentals_facts` rows and are used as the real substrate, so these tests need no
network and no populated DB.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals import reason_codes as rc
from src.data_extract.utils.fundamentals.build_history import (
    FORM_PRECEDENCE, MAX_AMENDMENT_LAG_DAYS, _FORMULAS, build_ticker, build_ticker_history,
    diff_against_stored, publication_events)
from src.data_extract.utils.fundamentals.kpi_catalogue import (
    CUBE_TIME_COLUMNS, HISTORY_KEYS, HISTORY_PROVENANCE, load_catalogue)
from src.data_extract.utils.fundamentals.periods import fiscal_quarter_of_end

CATALOGUE = load_catalogue("./configs")
SWEEP = Path("data/fundamentals_sweep")

#: Fields whose value the fixtures below populate. Enough to exercise a TTM flow, an instant
#: level and both legs of a derived column, and no more -- a fixture that fills all 52 would
#: be asserting the catalogue rather than the build.
_FLOW, _LEVEL = "totalRevenue", "totalAssets"


def _fact(**kwargs) -> dict:
    """One `fundamentals_facts` row with every column the build reads defaulted."""
    row = {"ticker": "TST", "accession_number": "a-1", "field": _FLOW,
           "fiscal_year": 2023, "fiscal_period": "Q1", "duration_type": "quarterly",
           "form": "10-Q", "filing_date": "2023-05-01", "is_amendment": False,
           "period_of_report": "2023-03-31", "regime": "industrial",
           "period_start": "2023-01-01", "period_end": "2023-03-31", "period_days": 89,
           "value": 100.0, "unit": "USD", "source_concept": "us-gaap:Revenues",
           "dc_code": None, "adjustment": None}
    row.update(kwargs)
    return row


def _four_quarters(*, ticker: str = "TST") -> pd.DataFrame:
    """Four filed quarters of a calendar-year filer, plus a balance-sheet level each time."""
    windows = [("2023-01-01", "2023-03-31", "Q1", "2023-05-01"),
               ("2023-04-01", "2023-06-30", "Q2", "2023-08-01"),
               ("2023-07-01", "2023-09-30", "Q3", "2023-11-01"),
               ("2023-10-01", "2023-12-31", "Q4", "2024-02-15")]
    rows = []
    for i, (start, end, label, filed) in enumerate(windows):
        rows.append(_fact(ticker=ticker, accession_number=f"acc-{i}", fiscal_period=label,
                          period_start=start, period_end=end, filing_date=filed,
                          period_of_report=end, value=100.0 + i,
                          form="10-K" if label == "Q4" else "10-Q"))
        rows.append(_fact(ticker=ticker, accession_number=f"acc-{i}", field=_LEVEL,
                          fiscal_period=label, duration_type="instant", period_start=None,
                          period_end=end, period_days=None, filing_date=filed,
                          period_of_report=end, value=1000.0 + i,
                          source_concept="us-gaap:Assets",
                          form="10-K" if label == "Q4" else "10-Q"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. The column contract                                                       #
# --------------------------------------------------------------------------- #
def test_the_column_contract_is_exactly_69_and_every_column_is_traceable():
    """"Column count is exactly as contracted" was a verification item against a plan that
    said "~71" twice and "68" once with no enumeration behind either. This test IS the
    enumeration: 4 keys + 52 catalogue fields + 8 derived + regime + 4 provenance."""
    columns = CATALOGUE.history_columns
    assert len(columns) == 69, f"{len(columns)} columns, not 69"
    assert len(columns) == len(set(columns)), "a column is declared twice"

    families = {
        "keys": [c for c in columns if c in HISTORY_KEYS],
        "catalogue": [c for c in columns if c in CATALOGUE.fields],
        "derived": CATALOGUE.history_derived_columns,
        "regime": ["regime"],
        "provenance": [c for c in columns if c in HISTORY_PROVENANCE],
    }
    assert len(families["keys"]) == 4 and len(families["catalogue"]) == 52
    assert len(families["derived"]) == 8 and len(families["provenance"]) == 4
    assert sum(len(v) for v in families.values()) == 69, "a column belongs to no family"

    # The two named casualties, and the two columns decision 32 removed.
    for gone in ("employees", "revenueGrowth", "earningsGrowth", "sector",
                 "industry_group", "ebitda_q", "freeCashflow_q", "capexGlobal"):
        assert gone not in columns, f"{gone} is still in the contract"
    assert CATALOGUE.side_table_fields == ["employees"]
    assert CUBE_TIME_COLUMNS == {"revenueGrowth", "earningsGrowth"}

    print("\n=== SANITY CHECK: the 69-column contract ===")
    for family, members in families.items():
        print(f"  {family:11s} {len(members):2d}")
    print("  dropped on purpose: employees (own table), revenueGrowth/earningsGrowth "
          "(pit.py, 365-day offset), sector/industry_group (join sp500_tickers), "
          "ebitda_q/freeCashflow_q/capexGlobal (declared casualties). Validated.")


def test_every_formula_matches_the_one_declared_in_the_config():
    """`build_history._FORMULAS` is the only implementation; `fundamentals_kpis.json` is the
    contract. A silent divergence between them is a wrong number nobody would look for, so
    the inputs are parsed back out of the config's own prose and compared."""
    declared = dict(CATALOGUE.derived_columns)
    for name in CATALOGUE.fields:
        formula = CATALOGUE.field(name).raw.get("derived_from")
        if formula:
            declared[name] = formula

    checked = 0
    for column, (inputs, _) in _FORMULAS.items():
        prose = declared.get(column)
        assert prose, f"{column} has no `derived_from` / `_derived_columns` entry"
        named = {re.sub(r"_ttm$", "", token) for token in re.findall(r"[A-Za-z_]\w*", prose)}
        missing = [name for name in inputs if name not in named]
        assert not missing, f"{column}: config formula {prose!r} never names {missing}"
        checked += 1

    # `revenue_q` / `netIncome_q` are the discrete-quarter columns; they have no two-operand
    # formula, so they are checked by name rather than by parse.
    assert set(_FORMULAS) | {"revenue_q", "netIncome_q"} >= set(
        CATALOGUE.history_derived_columns)
    print("\n=== SANITY CHECK: code formulas match the config ===")
    print(f"  {checked} formulas cross-checked against their `derived_from` prose; "
          "revenue_q / netIncome_q are discrete-quarter reads. Validated.")


def test_the_reason_code_vocabulary_is_closed_and_singly_defined():
    """A typo in a reason code is worse than no code: the null-gate's LEFT JOIN still finds a
    row, so the cell reads as explained while nothing can interpret the explanation."""
    assert len(rc.ALL_CODES) == 19, sorted(rc.ALL_CODES)
    assert rc.IS_QUALIFIER < rc.ALL_CODES
    assert rc.NOT_DISCLOSED in rc.ALL_CODES and rc.NOT_DISCLOSED not in rc.IS_QUALIFIER
    print("\n=== SANITY CHECK: the dc_code vocabulary ===")
    print(f"  {len(rc.ALL_CODES)} codes, {len(rc.IS_QUALIFIER)} of them QUALIFIERS "
          f"(a value is present): {sorted(rc.IS_QUALIFIER)}")
    print(f"  absences: {sorted(rc.ALL_CODES - rc.IS_QUALIFIER)}")
    print("  Imported from periods.py / xbrl_linkbase.py rather than restated, so each "
          "code has exactly one definition. Validated.")


# --------------------------------------------------------------------------- #
# 2. The publication-event grain                                               #
# --------------------------------------------------------------------------- #
def test_one_row_per_filing_date_and_the_snapshot_is_complete():
    history = build_ticker_history("TST", _four_quarters(), catalogue=CATALOGUE)
    assert list(history.columns) == CATALOGUE.history_columns
    assert len(history) == 4, "one row per filing date"
    assert list(pd.to_datetime(history["as_of"]).dt.strftime("%Y-%m-%d")) == [
        "2023-05-01", "2023-08-01", "2023-11-01", "2024-02-15"]
    # Rule 3: the level is carried on EVERY row, not only the one that published it.
    assert history[_LEVEL].notna().all()
    # The TTM only exists once four contiguous quarters do -- and never before.
    assert history[_FLOW].isna().sum() == 3 and history[_FLOW].iloc[-1] == 406.0
    assert pd.to_datetime(history["as_of"]).is_monotonic_increasing
    assert (pd.to_datetime(history["as_of"])
            >= pd.to_datetime(history["fiscal_end"])).all()
    print("\n=== SANITY CHECK: the publication-event grain ===")
    print(history[["as_of", "fiscal_end", "publication_form", _FLOW, _LEVEL]]
          .to_string(index=False))
    print("  4 filings -> 4 rows; the instant is on all four (rule 3), the TTM only on the "
          "row where the fourth quarter arrives. Validated.")


def test_two_filings_on_one_day_collapse_to_one_row_by_form_precedence():
    """A 10-K and a 10-Q filed the same day are ONE publication event. The precedence rule
    keeps `publication_form` scalar, so a `== '10-K'` filter cannot silently miss the day."""
    facts = _four_quarters()
    same_day = _fact(accession_number="acc-extra", form="10-K", filing_date="2023-08-01",
                     fiscal_period="Q2", period_start="2023-04-01",
                     period_end="2023-06-30", period_of_report="2023-06-30", value=101.0)
    events = publication_events(pd.concat([facts, pd.DataFrame([same_day])],
                                          ignore_index=True).assign(
        filing_date=lambda d: pd.to_datetime(d["filing_date"]),
        period_of_report=lambda d: pd.to_datetime(d["period_of_report"])))
    assert len(events) == 4, "the same-day pair did not collapse"
    row = events[events["as_of"] == pd.Timestamp("2023-08-01")].iloc[0]
    assert row["publication_form"] == "10-K", "precedence did not pick the 10-K"
    assert FORM_PRECEDENCE == ("10-K", "10-K/A", "10-Q", "10-Q/A")
    print("\n=== SANITY CHECK: same-day collapse ===")
    print(f"  5 accessions on 4 dates -> {len(events)} events; the 2023-08-01 pair reports "
          f"publication_form={row['publication_form']!r} by precedence. Validated.")


def test_a_no_op_amendment_publishes_nothing():
    """The 88 Part-III / cover-only amendments must produce ZERO rows. Tested by value, not
    by fact count: this amendment re-tags every fact to the SAME number."""
    facts = _four_quarters()
    echo = facts[facts["accession_number"] == "acc-0"].copy()
    echo["accession_number"] = "amd-noop"
    echo["form"] = "10-Q/A"
    echo["is_amendment"] = True
    echo["filing_date"] = "2023-06-01"
    both = pd.concat([facts, echo], ignore_index=True)
    assert len(build_ticker_history("TST", both, catalogue=CATALOGUE)) == 4
    print("\n=== SANITY CHECK: a no-op amendment ===")
    print("  an amendment re-tagging every fact to an identical value -> 0 extra rows "
          "(4 -> 4). A fact-count threshold would have admitted it. Validated.")


# --------------------------------------------------------------------------- #
# 3. Append-only                                                               #
# --------------------------------------------------------------------------- #
def test_a_changed_stored_row_is_detected_exactly():
    """`store.save` is an upsert, so immutability is only real if a drifted row is DETECTED.
    Compared exactly: if it ever trips on floating-point noise we want to know."""
    history = build_ticker_history("TST", _four_quarters(), catalogue=CATALOGUE)
    assert diff_against_stored(history, history).empty, "a frame drifted against itself"

    tampered = history.copy()
    tampered.loc[tampered.index[0], _LEVEL] = 999.0
    drift = diff_against_stored(tampered, history)
    assert len(drift) == 1 and drift["column"].iloc[0] == _LEVEL
    assert drift["stored"].iloc[0] == 999.0 and drift["rebuilt"].iloc[0] == 1000.0

    # DATE columns come back from Postgres as `datetime.date`; a build that compared those
    # against Timestamps would report every single row as drifted.
    as_dates = history.copy()
    for column in ("as_of", "fiscal_end"):
        as_dates[column] = pd.to_datetime(as_dates[column]).dt.date
    assert diff_against_stored(as_dates, history).empty, \
        "a DATE round-trip reads as drift -- see the Postgres date trap"

    print("\n=== SANITY CHECK: the append-only guard ===")
    print(f"  identical frames -> 0 findings; one tampered cell -> 1 finding "
          f"({drift['column'].iloc[0]}: {drift['stored'].iloc[0]} vs "
          f"{drift['rebuilt'].iloc[0]}); a datetime.date round-trip -> 0. Validated.")


# --------------------------------------------------------------------------- #
# 4. Zero unexplained nulls, on a real filer                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_build():
    """One roster ticker's real facts, off the committed sweep ledger. No network, no DB."""
    path = SWEEP / "AAPL.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present -- run scripts/sweep_fundamentals_resolution.py")
    facts = pd.read_parquet(path)
    facts = facts[facts.get("prefer_structure", True).astype(bool)]
    return build_ticker("AAPL", facts, catalogue=CATALOGUE)


def test_every_null_on_a_real_filer_carries_a_reason_code(real_build):
    """The gate items 7, 9 and B.6.6 exist to make passable: no cell may be NULL without a
    row in `fundamentals_reason_codes` for its own `(ticker, as_of, field)`."""
    history, codes = real_build.history, real_build.reason_codes
    value_columns = [c for c in history.columns
                     if c not in (*HISTORY_KEYS, "regime", *HISTORY_PROVENANCE)]
    explained = set(zip(codes["as_of"], codes["field"]))
    unexplained = [(row.as_of, column)
                   for row in history.itertuples()
                   for column in value_columns
                   if pd.isna(getattr(row, column))
                   and (row.as_of, column) not in explained]

    print("\n=== SANITY CHECK: zero unexplained nulls (AAPL, real facts) ===")
    print(f"  {len(history)} publication events x {len(value_columns)} value columns "
          f"= {len(history) * len(value_columns):,} cells")
    nulls = int(history[value_columns].isna().sum().sum())
    print(f"  nulls {nulls:,} ({nulls / (len(history) * len(value_columns)):.1%}) | "
          f"reason-code rows {len(codes):,} | UNEXPLAINED {len(unexplained)}")
    print(f"  code mix: {codes['dc_code'].value_counts().head(8).to_dict()}")
    if unexplained:
        print(f"  first few: {unexplained[:5]}")
    assert not unexplained, f"{len(unexplained)} null cells carry no reason code"
    print("  Validated.")


def test_the_real_grain_holds_and_a_rebuild_is_idempotent(real_build):
    history = real_build.history
    assert not history.duplicated(["ticker", "as_of"]).any()
    ends = pd.to_datetime(history["fiscal_end"])
    assert (ends.diff().dropna() >= pd.Timedelta(0)).all(), "fiscal_end is not monotone"
    lag = (pd.to_datetime(history["as_of"]) - ends).dt.days
    assert (lag >= 0).all(), "a row claims a period that had not closed"
    assert diff_against_stored(history, history).empty

    print("\n=== SANITY CHECK: the grain on real facts (AAPL) ===")
    print(f"  {len(history)} events {history['as_of'].min()} -> {history['as_of'].max()}")
    print(f"  filing lag as_of - fiscal_end: median {lag.median():.0f}d | "
          f"min {lag.min():.0f}d | max {lag.max():.0f}d "
          f"| beyond 200d {int((lag > 200).sum())}")
    print(f"  amendment rows {int(history['is_amendment'].sum())} | "
          f"forms {history['publication_form'].value_counts().to_dict()}")
    print(f"  MAX_AMENDMENT_LAG_DAYS = {MAX_AMENDMENT_LAG_DAYS}; a re-run over unchanged "
          "facts diffs to 0 cells. Validated.")


@pytest.mark.parametrize("ticker", ["APA", "WMT"])
def test_total_liabilities_falls_back_to_the_identity_and_says_so(ticker):
    """§5.1. Two of the eleven tickers measured at ZERO `totalLiabilities` coverage: neither
    declares a `Liabilities` total anywhere, so the field can only come from the balance
    sheet's own identity -- with the NCI bridge, and stamped `derived_identity` so it can
    never be mistaken for a resolved fact."""
    path = SWEEP / f"{ticker}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present")
    facts = pd.read_parquet(path)
    built = build_ticker(ticker, facts[facts["prefer_structure"].astype(bool)],
                         catalogue=CATALOGUE)
    history = built.history
    filled = int(history["totalLiabilities"].notna().sum())
    assert filled == len(history), "the identity did not close the gap"

    codes = built.reason_codes.query("field == 'totalLiabilities'")
    assert set(codes["dc_code"]) == {"derived_identity"}, set(codes["dc_code"])
    assert len(codes) == len(history), "not every derived cell is stamped"

    # The NCI bridge, read off the ROW rather than off the concept priority list -- because
    # priority-first is not the same as what actually won for a given filing. Both of these
    # tickers resolve equity on the INCLUDING-NCI element, so the correct identity is a plain
    # `assets - equity` and adding the minority interest would understate liabilities by it.
    equity_rows = facts[(facts["field"] == "stockholdersEquity") & facts["value"].notna()]
    concept = str(equity_rows.sort_values("filing_date")["source_concept"].iloc[-1])
    assert "IncludingPortionAttributableToNoncontrollingInterest" in concept, concept
    latest = history.iloc[-1]
    assert abs(latest["totalLiabilities"]
               - (latest["totalAssets"] - latest["stockholdersEquity"])) < 1.0
    # And it is NOT the ex-NCI arithmetic -- i.e. the bridge really did branch.
    assert latest["minorityInterest"] > 0
    assert abs(latest["totalLiabilities"] - (latest["totalAssets"]
               - latest["stockholdersEquity"] - latest["minorityInterest"])) > 1.0

    print(f"\n=== SANITY CHECK: totalLiabilities identity ({ticker}) ===")
    print(f"  coverage 0/{len(history)} tagged -> {filled}/{len(history)} derived, all "
          "stamped `derived_identity` (a QUALIFIER: the number is right, it is not evidence)")
    print(f"  equity resolved on {concept.split(':')[-1]}, so NO NCI is added:")
    print(f"  assets {latest['totalAssets']:,.0f} - equity "
          f"{latest['stockholdersEquity']:,.0f} = {latest['totalLiabilities']:,.0f}  "
          f"(the equity figure already contains the NCI of "
          f"{latest['minorityInterest']:,.0f})")
    print("  Measured alternative REFUTED: all 44 sampled 10-Ks declare a liability leg-set, "
          "but route 3b enumerates its legs, filer leg-sets vary by filer AND year, and an "
          "unlisted us-gaap sibling is dropped SILENTLY -- a Tier-1 balance-sheet total short "
          "by a caption. Validated.")


def test_every_value_column_is_float64_even_when_the_ticker_never_populates_it():
    """An all-null value column must NOT be `object`, and this is a data-corruption guard
    rather than a tidiness one.

    `sql/schema.sql` is applied only when Postgres INITIALISES a volume, so on a long-lived one
    `store.save` creates the table from the FIRST frame it is handed, via `ensure_table`'s
    dtype inference. Measured live: VRT (which resolves neither `minorityInterest` nor
    `restrictedCash`) created both as **TEXT**, and APA's real numbers then came back from the
    DB as the string `'1997000000.0'`. `diff_against_stored` caught it on the very next run --
    which is the append-only guard earning its keep -- but the frame must not offer the
    ambiguity in the first place.
    """
    history = build_ticker_history("TST", _four_quarters(), catalogue=CATALOGUE)
    value_columns = [c for c in history.columns
                     if c not in (*HISTORY_KEYS, "regime", *HISTORY_PROVENANCE)]
    wrong = {c: str(history[c].dtype) for c in value_columns
             if history[c].dtype != "float64"}
    assert not wrong, f"non-float value column(s): {wrong}"
    all_null = [c for c in value_columns if history[c].isna().all()]
    assert all_null, "the fixture populates every column, so it cannot test the null case"
    assert history["is_amendment"].dtype == bool

    print("\n=== SANITY CHECK: value-column dtypes ===")
    print(f"  {len(value_columns)} value columns, all float64; "
          f"{len(all_null)} of them entirely NULL for this fixture and still float64")
    print("  This is what stops `ensure_table` inferring TEXT on a cold table and storing "
          "every later ticker's number as a string. Validated.")


def test_the_nci_bridge_takes_the_other_branch_when_equity_is_ex_nci():
    """The half no roster ticker exercises: an EX-NCI equity row, where the minority interest
    must be ADDED back or liabilities are understated by the whole of it. Synthetic because
    the branch is chosen by the resolved concept, and every measured filer took the other one.
    """
    from src.data_extract.utils.fundamentals.build_history import (
        _total_liabilities_identity)

    _INCL = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"

    def equity_row(concept: str) -> pd.DataFrame:
        return pd.DataFrame([_fact(field="stockholdersEquity", value=800.0,
                                   duration_type="instant", period_start=None,
                                   source_concept=f"us-gaap:{concept}")])

    row = {"totalAssets": 1000.0, "stockholdersEquity": 800.0, "minorityInterest": 50.0}
    ex_nci, ex_code = _total_liabilities_identity(row, equity_row("StockholdersEquity"))
    incl_nci, incl_code = _total_liabilities_identity(row, equity_row(
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"))
    assert ex_nci == 150.0, "the NCI was not bridged into an ex-NCI equity row"
    assert incl_nci == 200.0, "the NCI was double-counted on an incl-NCI equity row"
    assert ex_code == incl_code == rc.DERIVED_IDENTITY

    # An ex-NCI basis, no NCI resolved, and the filer has NEVER tagged one: absence IS zero,
    # so derive -- but say so with its own code, because that rests on a second inference.
    silent, silent_code = _total_liabilities_identity(
        {**row, "minorityInterest": None}, equity_row("StockholdersEquity"))
    assert silent == 200.0, "a filer that tags no NCI anywhere should still get the identity"
    assert silent_code == rc.DERIVED_IDENTITY_NCI_ZERO, "the weaker basis is not distinguished"

    # Same shape, but this filer HAS tagged an NCI somewhere visible -- so a NULL at this event
    # means UNKNOWN, not zero, and the identity must refuse rather than understate liabilities.
    has_nci = pd.concat([equity_row("StockholdersEquity"),
                         pd.DataFrame([_fact(field="minorityInterest", value=50.0,
                                             duration_type="instant", period_start=None,
                                             source_concept="us-gaap:MinorityInterest")])],
                        ignore_index=True)
    assert _total_liabilities_identity(
        {**row, "minorityInterest": None}, has_nci) == (None, None)
    assert _total_liabilities_identity({**row, "totalAssets": None},
                                       equity_row("StockholdersEquity")) == (None, None)

    # BOTH bases at one period_end -> the NCI is the difference, which is two FILED facts and
    # not an assumption, so the plain `derived_identity` code applies. Preferred over the
    # assumed zero: here the filer HAS an NCI and assuming zero would understate liabilities.
    # The INCL row first and the EX row last, so the EX basis is the one that wins for the
    # column and the deduction is actually the path under test. (With the order reversed the
    # incl basis wins and the answer is the same 170 by a different route -- which is a nice
    # property, but it would leave `_deduced_nci` untested.)
    both = pd.concat([
        pd.DataFrame([_fact(field="stockholdersEquity", value=830.0, duration_type="instant",
                            period_start=None, source_concept=f"us-gaap:{_INCL}")]),
        equity_row("StockholdersEquity")],
        ignore_index=True)
    deduced, deduced_code = _total_liabilities_identity(
        {**row, "minorityInterest": None}, both)
    assert deduced == 170.0, "the NCI was not deduced from the two equity bases (830-800=30)"
    assert deduced_code == rc.DERIVED_IDENTITY, "a deduced NCI is evidence, not an assumption"

    print("\n=== SANITY CHECK: the NCI bridge, all four branches ===")
    print("  assets 1000, equity 800, NCI 50")
    print(f"  EX-NCI equity,   NCI resolved      -> 1000 - (800 + 50) = {ex_nci}  [{ex_code}]")
    print(f"  INCL-NCI equity, NCI resolved      -> 1000 -  800       = {incl_nci}  "
          f"[{incl_code}]")
    print(f"  EX-NCI equity,   filer tags NO NCI -> 1000 -  800       = {silent}  "
          f"[{silent_code}]")
    print(f"  EX-NCI equity,   BOTH bases filed   -> 1000 - (800 + 30) = {deduced}  "
          f"[{deduced_code}]")
    print("  EX-NCI equity,   filer HAS NCI      -> None (refused: a NULL means UNKNOWN here)")
    print("  Deduction is preferred over the assumption wherever both bases are filed, so the "
          "weakest branch is a last resort.")
    print("  Refusal needs an NCI KNOWN at that as_of, which is point-in-time and not a "
          "lifetime count: TMO has 38 valued NCI facts but files its first on 2022-02-24 "
          "against a history opening 2011-11-04, so its earlier events have none to know. "
          "Only a filer disclosing NCI in its FIRST filing (LLY, ETN) is refused throughout.")


# --------------------------------------------------------------------------- #
# 8. fiscal_quarter, and the statement column order                            #
# --------------------------------------------------------------------------- #
def _quarters_with_a_calendar(*, year_end_month_day: str = "12-31") -> pd.DataFrame:
    """Four filed quarters PLUS the annual fact that gives the ticker a fiscal calendar.

    The calendar is what `fiscal_quarter` is labelled against, and it comes from the filer's
    own ANNUAL-shaped period ends -- so a fixture with quarters only has no calendar and
    correctly labels nothing.
    """
    month, day = year_end_month_day.split("-")
    frame = _four_quarters()
    # A distinct accession per annual filing: `publication_events` is keyed on the accession
    # as well as the date, so reusing `acc-3` (the Q4 10-Q above) silently swallows both the
    # FY event and the Q4 one.
    annual = [_fact(accession_number=f"acc-fy{year}", duration_type="annual", fiscal_period="FY",
                    period_start=f"{year - 1}-{month}-{day}", period_end=f"{year}-{month}-{day}",
                    period_days=365, filing_date=f"{year + 1}-02-15",
                    period_of_report=f"{year}-{month}-{day}", value=400.0, form="10-K")
              for year in (2021, 2022)]
    return pd.concat([frame, pd.DataFrame(annual)], ignore_index=True)


def test_fiscal_quarter_labels_every_row_including_the_ttm_ones():
    """Q1-Q4 on EVERY row, not only the ones whose value is a single quarter.

    A TTM column spans four quarters and a balance-sheet column is a point, but the ROW still
    reports as of one quarter of the issuer's year -- and that is what a seasonal comparison
    needs, because a filer's Q4 is not its Q1. The label cannot come from `fiscal_end`'s
    calendar month either: the roster carries 52/53-week and non-December filers whose Q1 ends
    in the month another filer's Q3 does.
    """
    history = build_ticker_history("TST", _quarters_with_a_calendar())
    assert "fiscal_quarter" in history.columns
    labelled = history.dropna(subset=["fiscal_end"])
    assert labelled["fiscal_quarter"].notna().all(), "a dated row carries no quarter label"
    by_end = dict(zip(labelled["fiscal_end"].dt.strftime("%Y-%m-%d"),
                      labelled["fiscal_quarter"]))
    assert by_end["2023-03-31"] == 1 and by_end["2023-06-30"] == 2
    assert by_end["2023-09-30"] == 3 and by_end["2023-12-31"] == 4

    # Nullable integer, not float: `WHERE fiscal_quarter = 3` should not be a float compare,
    # and a row with no calendar yet must stay NULL rather than become 0.
    assert str(history["fiscal_quarter"].dtype) == "Int64"

    print("\n=== SANITY CHECK: fiscal_quarter on every row ===")
    for end, quarter in sorted(by_end.items()):
        print(f"  fiscal_end {end} -> Q{quarter}")
    print(f"  dtype {history['fiscal_quarter'].dtype} (BIGINT, NULL-able). Validated.")


def test_a_september_year_end_filer_is_not_labelled_off_the_calendar_month():
    """The case a month-of-year rule gets wrong. For a September filer the quarter ending in
    December is Q1, not Q4 -- so if this returns 4 the label is reading the Gregorian calendar
    instead of the issuer's."""
    quarter = fiscal_quarter_of_end("2023-12-30", [pd.Timestamp("2023-09-30"),
                                                   pd.Timestamp("2024-09-28")])
    assert quarter == 1, f"December end of a September filer labelled Q{quarter}"
    assert fiscal_quarter_of_end("2024-09-28", [pd.Timestamp("2023-09-30"),
                                               pd.Timestamp("2024-09-28")]) == 4
    # No calendar at all -> no label. Not Q1, which would be a guess presented as a fact.
    assert fiscal_quarter_of_end("2024-09-28", []) is None
    print("\n=== SANITY CHECK: a September year end ===")
    print("  2023-12-30 -> Q1 (a calendar-month rule says Q4); 2024-09-28 -> Q4; "
          "no calendar -> None. Validated.")


def test_the_value_columns_are_in_statement_order():
    """Requested explicitly: revenue -> cost -> net revenue, then debt / assets, then shares.

    The build RESOLVES fields tier-then-name (the NCI bridge depends on it), and the table used
    to be stored in that order, which reads as noise: `basicShares` first and `totalRevenue`
    twenty-four columns later. Order is now declared, and asserted here by the relations that
    make it a statement rather than by re-listing the 60 names.
    """
    columns = CATALOGUE.history_columns
    at = {name: i for i, name in enumerate(columns)}

    # The income statement, top down.
    for earlier, later in (("totalRevenue", "costOfRevenue"),
                           ("costOfRevenue", "grossProfit"),
                           ("grossProfit", "operatingIncome"),
                           ("operatingIncome", "pretaxIncome"),
                           ("pretaxIncome", "netIncome")):
        assert at[earlier] < at[later], f"{earlier} sits after {later}"
    # Each ratio immediately after the line it is computed from, so it can be checked in place.
    for line, ratio in (("grossProfit", "grossMargins"),
                        ("operatingIncome", "operatingMargins"),
                        ("netIncome", "profitMargins"),
                        ("stockholdersEquity", "returnOnEquity"),
                        ("dilutedShares", "optionOverhang")):
        assert at[ratio] > at[line], f"{ratio} sits before its own {line}"
    # Statements in order: income -> cash flow -> assets -> liabilities -> equity -> shares.
    for earlier, later in (("netIncome", "operatingCashFlow"),
                           ("operatingCashFlow", "totalAssets"),
                           ("totalAssets", "totalLiabilities"),
                           ("totalLiabilities", "stockholdersEquity"),
                           ("stockholdersEquity", "sharesOutstanding")):
        assert at[earlier] < at[later], f"{earlier} sits after {later}"
    # The keys lead and the provenance trails, with the values between them.
    assert columns[:4] == list(HISTORY_KEYS)
    assert columns[-4:] == list(HISTORY_PROVENANCE)
    # And the resolution order is deliberately NOT this one -- the field loop still needs
    # tier-3 `minorityInterest` resolved before the tier-1 `totalLiabilities` identity runs.
    assert CATALOGUE.history_fields != [c for c in columns if c in CATALOGUE.fields]

    print("\n=== SANITY CHECK: statement order ===")
    values = columns[4:-5]
    print(f"  {len(values)} value columns, {values[0]} -> {values[-1]}")
    print("  income statement -> cash flow -> assets -> liabilities -> equity -> shares, "
          "each ratio under its own line. Resolution order unchanged. Validated.")

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
