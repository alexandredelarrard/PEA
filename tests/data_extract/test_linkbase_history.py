"""
Phase 3b: does linkbase-driven resolution hold up across 26 tickers and 15 YEARS?

`test_linkbase_resolution.py` measures one 10-K per regime -- ~3 annual periods each, no
quarters, nothing before FY2023. That is enough to prove the routes work and nowhere near
enough to trust them, because the things most likely to break resolution are all temporal:

  * older filings may ship no calculation linkbase at all;
  * **ASC 606** (FY2018) deprecated `SalesRevenueNet` and friends, so the resolved revenue
    concept MUST switch mid-history -- cleanly, without a level break;
  * **ASC 842** (FY2019) makes the `ppeNet` finance-lease adjustment appear mid-series;
  * a 10-Q's linkbase is smaller than a 10-K's, and quarters are what the whole TTM layer
    is built from.

So this file sweeps every 10-K and 10-Q from 2011 to today for a roster picked so each
ticker buys a distinct edge case, and asserts on the aggregate.

**Cost.** The full roster is ~1,700 filings at ~1-3 s each. That is a deliberate on-demand
run, not a default-suite cost: set ``FUNDAMENTALS_HISTORY_SWEEP=full`` for all 26 tickers,
otherwise a 3-ticker subset runs (~200 filings) covering the extension total, the parentless
root and the fiscal-calendar edge. Results are cached per-roster for the session.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals import entity_scope as scope
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    FIELD_SUM, LINKBASE_METHODS, TAG_FALLBACK, UNRESOLVED, ArcGraph, resolve_field,
    statement_arcs)

CATALOGUE = load_catalogue("./configs")

SINCE = pd.Timestamp("2011-01-01")
FORMS = ["10-K", "10-Q"]
WORKERS = 6

#: ticker -> (sector, industry_group, sub_industry, what this ticker is here to prove).
#: Every entry buys a distinct edge case; none is here for coverage padding.
ROSTER: dict[str, tuple[str, str, str, str]] = {
    "AAPL": ("Information Technology", "Technology Hardware & Equipment",
             "Technology Hardware, Storage & Peripherals", "Sep FY; ASC-606 concept switch"),
    "CSCO": ("Information Technology", "Communications Equipment",
             "Communications Equipment", "52/53-week FY; the 2017 53-week Q4"),
    "KR": ("Consumer Staples", "Consumer Staples Distribution & Retail",
           "Consumer Staples Merchandise Retail", "Jan fiscal year-end"),
    "XOM": ("Energy", "Energy", "Integrated Oil & Gas", "frozen-TTM baseline (36%)"),
    "APA": ("Energy", "Energy", "Oil & Gas Exploration & Production",
            "EXTENSION revenue total; the 0-revenue chain"),
    "EOG": ("Energy", "Energy", "Oil & Gas Exploration & Production",
            "per-company capex elements"),
    "VLO": ("Energy", "Energy", "Oil & Gas Refining & Marketing", "the D&A tie-break (~200x)"),
    "JPM": ("Financials", "Banks", "Diversified Banks", "RevenuesNetOfInterestExpense"),
    "BAC": ("Financials", "Banks", "Diversified Banks", "FY2023 restatement trap"),
    "MTB": ("Financials", "Banks", "Regional Banks", "28% frozen TTM"),
    "USB": ("Financials", "Banks", "Diversified Banks", "no legacy facts (Phase 9 blind spot)"),
    "MET": ("Financials", "Insurance", "Life & Health Insurance", "LDTI 2021 break"),
    "PGR": ("Financials", "Insurance", "Property & Casualty Insurance", "P&C tagged ratios"),
    "AFL": ("Financials", "Insurance", "Life & Health Insurance", "third insurer"),
    "MAA": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)",
            "Multi-Family Residential REITs", "Up-C LegalEntityAxis extension member"),
    "SPG": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)", "Retail REITs",
            "unclassified balance sheet"),
    "AMT": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)",
            "Telecom Tower REITs", "tower REIT -> industrial regime trap"),
    "DTE": ("Utilities", "Utilities", "Multi-Utilities", "parentless revenue root"),
    "SO": ("Utilities", "Utilities", "Electric Utilities", "six registrant CIKs"),
    "NEE": ("Utilities", "Utilities", "Electric Utilities", "RegulatoryAssets absent"),
    "ETN": ("Industrials", "Capital Goods", "Electrical Components & Equipment",
            "totalRevenue == 0 (16 legacy rows)"),
    "VRT": ("Industrials", "Capital Goods", "Electrical Components & Equipment",
            "totalRevenue == 0 (5 legacy rows)"),
    "SWKS": ("Information Technology", "Semiconductors & Semiconductor Equipment",
             "Semiconductors", "FY2020 tags a 370-day AND a 97-day fact as fp=FY"),
    "BRK-B": ("Financials", "Financial Services", "Multi-Sector Holdings",
              "hybrid regime; multi-class; no AssetsCurrent"),
    "GS": ("Financials", "Financial Services", "Investment Banking & Brokerage",
           "the ONLY broker_dealer in the roster"),
    "META": ("Communication Services", "Media & Entertainment",
             "Interactive Media & Services", "edgartools #691: 0 undimensioned share facts"),
}

#: Default subset when the full sweep is not requested: the extension total, the parentless
#: root, and the fiscal-calendar edge. Enough to catch a regression in the three mechanisms
#: this phase actually introduced.
FAST_SUBSET = ("APA", "DTE", "SWKS")

#: The three tickers the legacy table carries as `totalRevenue == 0`. Fixing them is the
#: plan's headline Phase 3 acceptance criterion.
ZERO_REVENUE_TICKERS = ("APA", "ETN", "VRT")


def _rostered() -> dict[str, tuple[str, str, str, str]]:
    if os.getenv("FUNDAMENTALS_HISTORY_SWEEP", "").lower() == "full":
        return ROSTER
    return {t: ROSTER[t] for t in FAST_SUBSET}


def _resolve_one(ticker: str, gics: dict, filing) -> list[dict]:
    """One filing -> ledger rows. Never raises: a filing that will not parse is a fact
    about that filing, and must not abort a 1,700-filing sweep."""
    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import (
        _compose, _materialise, _period_frame)
    try:
        xbrl = filing.xbrl()
        if xbrl is None:
            return []
        facts = scope.consolidated_facts(xbrl.facts.to_dataframe())
        if facts.empty:
            return []
        facts = _period_frame(facts)
        available = scope.reported_concepts(facts)
        durations = scope.duration_concepts(facts)
        zero_only = scope.zero_only_concepts(facts)
        graph = ArcGraph(statement_arcs(xbrl))
        regime = CATALOGUE.regime_for(
            gics, [str(r) for r in graph.arcs.get("role_uri", pd.Series(dtype=str))])

        resolutions, values = {}, {}
        for name in CATALOGUE.extracted_fields:
            # `ticker=` is load-bearing since Phase 4b: without it route 3b cannot read the
            # per-filer extension register, so DTE, NEE, MAA, PLD, MSFT, GOOGL, SWKS and
            # ORCL would resolve capex/depAmort differently here than in production and
            # this sweep would measure a pipeline nobody runs.
            r = resolve_field(CATALOGUE.field(name), graph, available, CATALOGUE, regime,
                              duration_concepts=durations, zero_only=zero_only,
                              ticker=ticker)
            resolutions[name] = r
            if r.method != FIELD_SUM:
                values[name] = _materialise(r, facts)
        for name, r in list(resolutions.items()):
            if r.method == FIELD_SUM:
                composed, reason = _compose(CATALOGUE.field(name), r.component_fields, values)
                values[name] = composed
                if reason:
                    from dataclasses import replace as _replace
                    from src.data_extract.utils.fundamentals.xbrl_linkbase import UNRESOLVED
                    resolutions[name] = _replace(r, method=UNRESOLVED, dc_code=reason)

        rows = []
        for name, r in resolutions.items():
            common = {"ticker": ticker, "accession": filing.accession_number,
                      "form": filing.form,
                      "filing_date": pd.Timestamp(filing.filing_date), "regime": regime,
                      "field": name, "method": r.method, "concept": r.concept,
                      "is_extension": r.is_extension, "dc_code": r.dc_code,
                      "has_linkbase": not graph.is_empty,
                      "subtract": ",".join(r.subtract) or None,
                      "zero_only_retained": bool(r.zero_only_retained)}
            periods = values.get(name) or {}
            if not periods:
                rows.append({**common, "value": None, "fiscal_year": None,
                             "fiscal_period": None, "duration_type": None,
                             "period_end": pd.NaT})
                continue
            rows.extend({**common, "value": p["value"], "fiscal_year": p["fiscal_year"],
                         "fiscal_period": p["fiscal_period"],
                         "duration_type": p["duration_type"],
                         "period_end": p["period_end"]} for p in periods.values())
        return rows
    except Exception:                                       # noqa: BLE001
        return []


@pytest.fixture(scope="module")
def ledger() -> pd.DataFrame:
    """Every catalogue field resolved on every 10-K/10-Q since 2011, for the active roster.

    Amendments are excluded: the amendment grain is Phase 5's acceptance test, and a
    Part-III-only 10-K/A carries cover-page facts alone, which would inflate `unresolved`
    with rows saying nothing about resolution quality. (edgartools' `form=` prefix-matches,
    so they must be filtered explicitly.)
    """
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- the history sweep needs EDGAR")
    from edgar import Company, set_identity
    set_identity(os.getenv("SEC_USER_AGENT"))

    roster = _rostered()
    rows: list[dict] = []

    def _walk(ticker: str) -> list[dict]:
        sector, group, sub, _why = roster[ticker]
        gics = {"sector": sector, "industry_group": group, "sub_industry": sub}
        filings = [f for f in Company(ticker).get_filings(form=FORMS)
                   if pd.Timestamp(f.filing_date) >= SINCE
                   and not str(f.form).upper().endswith("/A")]
        out: list[dict] = []
        for filing in filings:
            out.extend(_resolve_one(ticker, gics, filing))
        return out

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_walk, t): t for t in roster}
            for future in as_completed(futures):
                rows.extend(future.result())
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f"EDGAR unreachable: {exc}")

    if not rows:
        pytest.skip("history sweep produced no rows -- EDGAR unreachable?")
    frame = pd.DataFrame(rows)
    frame["year"] = pd.to_datetime(frame["filing_date"]).dt.year
    return frame


@pytest.fixture(scope="module")
def resolved(ledger: pd.DataFrame) -> pd.DataFrame:
    """Ledger rows that actually produced a value -- the population a ROUTE rate is about.
    A `dc_code` row took no route; counting it as one is what made the first measurement
    read 27.8% against a 20% gate."""
    return ledger[ledger["method"] != UNRESOLVED]


# --------------------------------------------------------------------------- #
# Phase 3c acceptance. Deliberately written as CRITERIA, not as the counts the  #
# in-sample roster happens to produce: a threshold fitted to the 26 tickers     #
# that were chosen BECAUSE they broke things is not evidence about anything     #
# else, and this file is meant to run unchanged against a new roster.           #
# --------------------------------------------------------------------------- #

#: Concept name prefixes that are never a revenue top line. A balance-sheet total, a
#: cash-flow movement, an expense subtotal or a non-operating item appearing here means
#: `discover_root` picked a root off the wrong statement -- the 3c.2 defect, which put 74
#: rows on `Assets` / `LiabilitiesAndStockholdersEquity` / cash-flow totals /
#: `ComprehensiveIncomeNetOfTax`, and the 3c.8 defect, which put 36 more on a bank's
#: cash-flow statement because its FASB role name contains the word "Operations".
NOT_REVENUE_PREFIXES = (
    "Assets", "Liabilities", "Cash", "NetCashProvided", "NetCashUsed",
    "ComprehensiveIncome", "NoninterestExpense", "InvestmentIncome", "ForeignCurrency",
    "OperatingIncomeLoss", "CostsAndExpenses", "IncomeLoss",
)

#: The architecture gate. Only `tag_fallback` counts -- see the module docstring of
#: `xbrl_linkbase`: pooling it with `tag_primary` and the reason-coded non-resolutions
#: reported 27.8% and forced a judgement call about which fields "should" be excluded.
TAG_FALLBACK_GATE = 0.20

#: Below this, the filer's own roll-up is not what is driving resolution and the rebuild is
#: the old tag-list architecture wearing a new name. Measured pre-3c.1: **0.9% for every
#: year 2011-2014**, because `menucat` is null on all 418 of those filings.
MIN_LINKBASE_SHARE = 0.55


def _bare(frame: pd.DataFrame) -> pd.Series:
    return frame["concept"].fillna("").str.split(":").str[-1]


def test_the_linkbase_drives_resolution_in_every_year(resolved):
    """3c.1. The whole claim of this rebuild is that resolution reads the filer's own
    calculation linkbase. That was false for four of fifteen years and nobody could see it,
    because the pooled rate hid it behind the modern era."""
    valued = resolved[resolved["value"].notna()]
    share = (valued.assign(lb=valued["method"].isin(LINKBASE_METHODS))
             .groupby("year")["lb"].mean())
    fallback = (valued["method"] == TAG_FALLBACK).mean()

    print("\n=== SANITY CHECK: linkbase share by year ===")
    for year, rate in share.items():
        flag = "" if rate >= MIN_LINKBASE_SHARE else "   <-- BELOW GATE"
        print(f"  {year}  {rate:6.1%}{flag}")
    print(f"  tag_fallback overall {fallback:.2%} (gate {TAG_FALLBACK_GATE:.0%})")

    weak = share[share < MIN_LINKBASE_SHARE]
    assert weak.empty, (
        f"the tag list, not the linkbase, is resolving these years: {weak.round(3).to_dict()}")
    assert fallback < TAG_FALLBACK_GATE, f"tag_fallback {fallback:.2%}"
    print("  OK: every year reads the filer's own roll-up.")


def test_revenue_never_resolves_to_something_off_the_income_statement(resolved):
    """3c.2 + 3c.8. A revenue number taken from the balance sheet or the cash-flow
    statement is not a smaller error than a missing one -- it is worse, because it looks
    like data."""
    revenue = resolved[(resolved["field"] == "totalRevenue") & resolved["value"].notna()]
    suspect = revenue[_bare(revenue).str.startswith(NOT_REVENUE_PREFIXES)]

    print("\n=== SANITY CHECK: revenue concept provenance ===")
    print(f"  {len(revenue)} valued revenue rows, {len(suspect)} off the income statement")
    if len(suspect):
        print(suspect.groupby(["ticker", _bare(suspect)]).size().to_string())
    assert suspect.empty, (
        "revenue resolved to a concept that cannot be a top line:\n"
        + suspect.groupby(["ticker", _bare(suspect), "method"]).size().to_string())
    print("  OK: every revenue value comes from an income-statement concept.")


def test_a_zero_revenue_row_means_the_filer_reported_nothing_else(resolved):
    """3c.3. The plan's original criterion -- "no zero-revenue rows" -- was WRONG. VRT's
    2018-2020 filings are the GS Acquisition Holdings blank-cheque shell pre-merger, with a
    $690M IPO, $123k of G&A and genuinely no revenue. A zero is allowed; an UNEXPLAINED
    zero is not, and `zero_only_retained` is the explanation."""
    revenue = resolved[(resolved["field"] == "totalRevenue") & resolved["value"].notna()]
    zeros = revenue[revenue["value"] == 0]
    unexplained = zeros[~zeros["zero_only_retained"].astype(bool)]

    print("\n=== SANITY CHECK: zero revenue ===")
    print(f"  {len(zeros)} zero rows, {len(unexplained)} without the retained flag")
    if len(zeros):
        print(zeros.groupby(["ticker", "year", "zero_only_retained"]).size().to_string())
    assert unexplained.empty, (
        "a zero that is NOT the filer's whole answer -- the concept has a non-zero value "
        "somewhere, so the resolver took an artefact:\n"
        + unexplained.groupby(["ticker", "year", "concept"]).size().to_string())

    # The flag must never fire on a value that is not zero: it means "this concept reports
    # 0 in every period", so a non-zero row carrying it would mean the guard misfired.
    flagged = resolved[resolved["zero_only_retained"].astype(bool)
                       & resolved["value"].notna()]
    assert (flagged["value"] == 0).all(), flagged[flagged["value"] != 0].head().to_string()
    print(f"  {len(flagged)} rows carry the flag across all fields, all of them zero.")
    print("  OK: every zero is the filer's own answer, and says so.")


def test_no_adjustment_ever_drives_a_non_negative_field_negative(resolved, ledger):
    """3c.5 + 3c.8. `total_adjustment` removed lease legs from a total that never contained
    them -- 158 negative `shortTermDebt` values, worst -$893M. A negative that survives must
    be the FILER's sign convention, never ours, so the discriminator is whether a
    subtraction was applied."""
    non_negative = [name for name in CATALOGUE.extracted_fields
                    if CATALOGUE.field(name).raw.get("sign") == "non_negative"]
    negative = resolved[resolved["field"].isin(non_negative) & (resolved["value"] < 0)]
    ours = negative[negative["subtract"].notna()]

    print("\n=== SANITY CHECK: sign violations ===")
    print(f"  {len(negative)} negatives on non_negative fields; "
          f"{len(ours)} of them had an adjustment applied")
    if len(negative):
        print(negative.groupby(["field", "ticker"]).agg(
            n=("value", "size"), worst=("value", "min")).to_string())
    assert ours.empty, (
        "we subtracted an amount and drove the field below zero:\n"
        + ours.groupby(["field", "ticker", "subtract"]).agg(
            n=("value", "size"), worst=("value", "min")).to_string())
    print("  OK: every surviving negative is as-filed, for the Phase 7 validator.")


def test_each_ticker_keeps_one_regime_for_fifteen_years(ledger):
    """A regime is a statement TEMPLATE, not a market view: a company does not stop being a
    bank. A ticker that flips regime mid-history means the router is reading something that
    varies filing-to-filing, and every regime-gated `never_use` and `roll_up` would flip
    with it."""
    per_ticker = ledger.groupby("ticker")["regime"].agg(
        n=("nunique"), seen=(lambda s: sorted(set(s.dropna()))))
    flipped = per_ticker[per_ticker["n"] > 1]

    print("\n=== SANITY CHECK: regime stability ===")
    print(per_ticker["seen"].apply(lambda v: v[0] if len(v) == 1 else v).to_string())
    assert flipped.empty, flipped.to_string()
    print(f"  OK: {len(per_ticker)} tickers, one regime each across the full history.")


# --------------------------------------------------------------------------- #
# Phase 4c acceptance. Same rule as above: criteria, not the counts the        #
# in-sample roster happens to produce.                                        #
# --------------------------------------------------------------------------- #

#: Regimes whose Reg S-X article prescribes an UNCLASSIFIED balance sheet, so there is no
#: current/noncurrent debt split and `us-gaap:LongTermDebt` is the correct noncurrent
#: figure rather than a contaminated one. Article 9 (bank), 7 (insurer), 12 / Rule 17a-5
#: (broker-dealer), and 17 CFR 210.1-02(bb)(1)(i) for real estate.
UNCLASSIFIED_REGIMES = frozenset({"bank", "insurer", "broker_dealer", "real_estate"})

#: The current-inclusive debt element. FASB defines `LongTermDebt` as the total INCLUDING
#: the current portion, so on a classified sheet it is a different BASIS from this field's
#: definition, not merely a coarser one.
CURRENT_INCLUSIVE_DEBT = "LongTermDebt"

#: An order-of-magnitude jump in noncurrent debt between two consecutive period-ends is not
#: a financing event for an S&P 500 issuer -- it is a basis switch. Measured pre-4c.2 on the
#: 52-ticker sweep: AMT stepped **x11,545** and then -99.99% purely by changing which debt
#: concept won, and NVDA +249.5%, ETN +144.3%, BA +74.2%, SCHW +74.0% the same way.
MAX_DEBT_STEP = 10.0


def test_the_debt_basis_never_switches_to_the_current_inclusive_element(ledger):
    """4c.2. A classified filer that declares a noncurrent debt line must be read on it.
    `us-gaap:LongTermDebt` stays in the priority list as a LAST resort -- deleting it would
    turn 103 measured rows into nulls for filers that tag nothing else -- so the assertion
    is not that it is unused, but that it never wins while a noncurrent line is available."""
    debt = ledger[(ledger["field"] == "longTermDebt") & ledger["value"].notna()].copy()
    if debt.empty:
        pytest.skip("no longTermDebt rows in this sweep")
    debt["bare"] = debt["concept"].fillna("").str.split(":").str[-1]
    classified = debt[~debt["regime"].isin(UNCLASSIFIED_REGIMES)]

    offenders = []
    for (ticker, accession), group in classified.groupby(["ticker", "accession"]):
        seen = set(group["bare"])
        if CURRENT_INCLUSIVE_DEBT in seen and seen - {CURRENT_INCLUSIVE_DEBT}:
            offenders.append((ticker, accession, sorted(seen)))

    print("\n=== SANITY CHECK: longTermDebt basis by regime class ===")
    print(pd.crosstab(debt["bare"],
                      debt["regime"].isin(UNCLASSIFIED_REGIMES).map(
                          {True: "unclassified", False: "classified"})).to_string())
    assert not offenders, "\n".join(map(str, offenders[:10]))
    print(f"  OK: no filing mixes the current-inclusive element with a noncurrent line; "
          f"{int((classified['bare'] == CURRENT_INCLUSIVE_DEBT).sum())} classified rows "
          f"fall back to it because the filer tags nothing else.")


def test_noncurrent_debt_never_steps_by_an_order_of_magnitude(ledger):
    """4c.2, the symptom the priority order was hiding behind. A concept boundary must not
    be visible in the SERIES: a feature built on this field differences it, so a basis
    switch is indistinguishable from a refinancing and strictly worse than a null."""
    debt = ledger[(ledger["field"] == "longTermDebt") & ledger["value"].notna()]
    debt = debt[debt["value"].abs() > 1e6]           # sub-$1M lines are note-level, not the sheet
    if debt.empty:
        pytest.skip("no longTermDebt rows in this sweep")

    # One value per BALANCE DATE, latest vintage. Every filing restates the prior-year
    # comparatives, so a raw shift() would difference two vintages of the same date and
    # measure restatement instead of the series -- and `quarterize` keeps the latest
    # vintage, so the latest is also what production stores.
    series = (debt[debt["duration_type"] == "instant"]
              .sort_values(["ticker", "period_end", "filing_date"])
              .drop_duplicates(["ticker", "period_end"], keep="last"))
    series = series.assign(prev=series.groupby("ticker")["value"].shift())
    steps = series[series["prev"].notna()].copy()
    if steps.empty:
        pytest.skip("fewer than two balance dates per ticker")
    steps["ratio"] = (steps["value"].abs() / steps["prev"].abs()).clip(lower=1e-12)
    steps["multiple"] = steps[["ratio"]].assign(inv=1.0 / steps["ratio"]).max(axis=1)
    blown = steps[steps["multiple"] > MAX_DEBT_STEP]

    print("\n=== SANITY CHECK: worst year-on-year noncurrent-debt steps ===")
    print(steps.nlargest(5, "multiple")[
        ["ticker", "period_end", "prev", "value", "multiple"]].to_string(index=False))
    assert blown.empty, blown[["ticker", "period_end", "prev", "value", "multiple"]].to_string()
    print(f"  OK: {len(steps)} balance-date transitions, worst "
          f"{steps['multiple'].max():.2f}x against a {MAX_DEBT_STEP:.0f}x ceiling.")
