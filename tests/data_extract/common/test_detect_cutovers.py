"""The registrant-cutover detector: its screens, its classifier, and its negative control.

The detector rests on ONE asymmetry: prices follow the economic entity, filings follow the legal
registrant. `yfinance` splices Mylan's history into VTRS; EDGAR does not. A long price history
against a short filing archive therefore implies a registrant change -- and the same asymmetry
auto-excludes a genuine recent IPO, whose price history is short too.

⚠ THAT LAST CLAUSE IS THE WHOLE DESIGN, so it gets the load-bearing test. If a short filing
archive alone could trigger the screen, the detector would flag every young company in the index
and would be measuring nothing.

Synthetic fixtures throughout: these are the classifier's decision rules, not measurements of
the world. The measurements live in `test_registrant_live.py` and in the detector's own
`--classify` output.
"""
from __future__ import annotations

import pandas as pd
import pytest

import scripts.detect_registrant_cutovers as detect

BOUNDARY = "2019-03-20"


# --------------------------------------------------------------------------- #
# Oracle 2 -- the shell-name reading                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("contraname", "is_shell"), [
    ("TWDC HOLDCO 613 CORP", True),
    ("HALFMOON PARENT INC", True),
    ("MONARCH ENERGY HOLDING INC", True),
    ("VIRGINIA HOLDCO INC", True),
    ("RELIANT ENERGY REGCO INC", True),
    # Real companies, and real `namechangefrom` rows. A name change is NOT a shell.
    ("AMVESCAP PLC/LONDON/", False),
    ("BROADCOM LTD", False),
    ("UPJOHN INC", False),
])
def test_the_shell_name_reading(contraname, is_shell):
    """`TWDC HOLDCO 613 CORP` is not a company, it is a shell created for a reorganisation, and
    the word in its name says so. `AMVESCAP PLC` is a company that changed its name.

    The distinction matters because Sharadar emits `namechangefrom` for both, and IVZ -- one
    continuous CIK filing since 1994 -- carries such a row. Reading a shell token as
    conclusive and a plain rename as suggestive is what keeps the two apart."""
    got = any(tok in contraname.upper() for tok in detect.SHELL_TOKENS)
    assert got is is_shell
    print(f"\n=== SANITY CHECK: {contraname!r} -> shell={got} ===")


# --------------------------------------------------------------------------- #
# The FPI test, and why it runs first                                         #
# --------------------------------------------------------------------------- #
def _doc(pairs: list[tuple[str, str]]) -> dict:
    """A submissions document with everything inlined in `recent` and no archive pages."""
    return {"filings": {"recent": {"form": [f for f, _ in pairs],
                                   "filingDate": [d for _, d in pairs]}, "files": []}}


def test_a_foreign_private_issuer_is_not_a_cutover(monkeypatch):
    """CRH, IVZ, NXPI and RCL file 20-F/6-K under ONE continuous CIK and then transition to
    domestic forms. That looks exactly like a truncation and is not one: the repair is form
    coverage -- different forms, different parsers, different fields -- so no register entry.

    ⚠ The FPI test runs BEFORE the shell-name oracle, and the ordering is the point. IVZ
    carries a Sharadar `namechangefrom AMVESCAP PLC/LONDON/` and would otherwise read as a
    reorganisation, but CIK 914208 is continuous from 1994 with 1,594 FPI filings."""
    monkeypatch.setattr(detect, "all_filings", lambda ctx, doc:
                        [("20-F", pd.Timestamp("2002-03-01") + pd.Timedelta(days=365 * i))
                         for i in range(20)])
    got = detect.continuity(None, _doc([]), "2024-01-24")
    assert got["fpi_before"] >= detect.FPI_MIN_FILINGS
    print("\n=== SANITY CHECK: an FPI is classified before the shell-name oracle ===")
    print(f"  {got['fpi_before']} 20-F filings before the truncation -> no register entry.")


def test_the_fpi_count_reads_the_whole_archive_not_just_recent(monkeypatch):
    """⚠ REGRESSION. The submissions `recent` block holds ~1,000 filings, which for an active
    name is five to ten years. Counting FPI forms in `recent` alone returned ZERO for IVZ
    (continuous since 1994) and RCL (since 1997), and both were classified as cutovers --
    which would have written a register entry for a ticker that never changed registrant."""
    old = [("20-F", pd.Timestamp("1997-03-01") + pd.Timedelta(days=365 * i)) for i in range(8)]
    recent_only = [("8-K", pd.Timestamp("2020-01-01"))]
    monkeypatch.setattr(detect, "all_filings", lambda ctx, doc: old + recent_only)
    got = detect.continuity(None, _doc(recent_only), "2005-02-16")
    assert got["fpi_before"] >= detect.FPI_MIN_FILINGS, (
        "the archive pages were not counted -- this is the IVZ/RCL misclassification")
    print("\n=== SANITY CHECK: FPI forms are counted over the whole archive ===")
    print(f"  {got['fpi_before']} pre-boundary FPI filings found in the older pages, "
          "0 of which are in `recent`.")


def test_a_pre_registered_shell_is_not_a_continuous_registrant(monkeypatch):
    """`own_first` alone cannot classify, and this is why. A merger's S-4 goes in well ahead of
    completion -- Linde plc 517 days, DowDuPont 560, PSKY 316 -- so an early first filing is
    NORMAL for a real cutover. What no cutover has is a decade of the successor's own filings
    before the boundary, which is why the test counts ACTIVITY outside the grace window."""
    shell = [("S-4", pd.Timestamp("2018-05-01")), ("S-4/A", pd.Timestamp("2018-06-01")),
             ("CORRESP", pd.Timestamp("2018-07-01"))]
    monkeypatch.setattr(detect, "all_filings", lambda ctx, doc: shell)
    got = detect.continuity(None, _doc([]), "2018-10-31")
    assert got["own_active_before"] < detect.CONTINUOUS_MIN_FILINGS
    assert got["own_lead_days"] > 0, "the shell does predate the boundary -- that is the trap"
    print("\n=== SANITY CHECK: a pre-registered shell stays a cutover candidate ===")
    print(f"  archive starts {got['own_lead_days']} d before the boundary but only "
          f"{got['own_active_before']} filings sit outside the "
          f"{detect.PRE_REGISTRATION_DAYS} d grace window.")


def test_a_continuous_registrant_is_not_a_cutover(monkeypatch):
    """BMY's 8-K starts 2000-04-19 while its proxy runs from 1996-03-18 under the same
    registrant. Nothing moved: 8-K was event-driven and rare before the 2004 item expansion.
    Hundreds of the CIK's own filings before the boundary is what says so."""
    long_history = [("10-Q", pd.Timestamp("1994-01-01") + pd.Timedelta(days=25 * i))
                    for i in range(90)]
    monkeypatch.setattr(detect, "all_filings", lambda ctx, doc: long_history)
    got = detect.continuity(None, _doc([]), "2000-04-19")
    assert got["own_active_before"] >= detect.CONTINUOUS_MIN_FILINGS
    print("\n=== SANITY CHECK: a continuous registrant is a non-defect ===")
    print(f"  {got['own_active_before']} of its own filings predate the boundary by more than "
          f"{detect.PRE_REGISTRATION_DAYS} d -> no register entry.")


# --------------------------------------------------------------------------- #
# The predecessor gate                                                        #
# --------------------------------------------------------------------------- #

def _run_profile(monkeypatch, pairs, boundary=BOUNDARY):
    monkeypatch.setattr(detect, "submissions", lambda ctx, cik: {})
    monkeypatch.setattr(detect, "all_filings", lambda ctx, doc: pairs)
    return detect.predecessor_profile(None, "0000000001", boundary)


def _stream(start: str, end: str, form: str = "10-Q", every_days: int = 7):
    """A filing stream at a realistic density.

    ⚠ Density matters, and a sparse fixture is a wrong one. A real submissions index carries
    every form indexed under the CIK, so the four register predecessors show 176-599 filings in
    the two years BEFORE their boundary (Apache 318, Eaton 266, Google 599, Exxon Mobil 176,
    measured 2026-09-10). A quarterly fixture gives 9, where one filing either way swings the
    ratio past the threshold -- so the test would be measuring its own sparseness, not the rule.
    """
    return [(form, d) for d in pd.date_range(start, end, freq=f"{every_days}D")]


def test_a_subsidiary_co_registrant_is_rejected_for_filing_no_proxy(monkeypatch):
    """`NBCUniversal Media, LLC`, `Duke Energy Carolinas, LLC` and `Bunge Ltd Finance Corp` all
    file 10-K/10-Q jointly with their parent for decades. They pass every filing-count test
    and file no proxy of their own, because they were never the public registrant."""
    pairs = _stream("2010-01-01", "2018-12-31")
    got = _run_profile(monkeypatch, pairs)
    assert not got["ok"]
    assert "no proxy" in got["why"]
    print("\n=== SANITY CHECK: a subsidiary co-registrant is rejected ===")
    print(f"  {got['n_before_window']} filings before the boundary, 0 proxies -> "
          f"{got['why']}")


def test_an_unrelated_counterparty_is_rejected_for_not_stopping(monkeypatch):
    """⚠ THE PSKY CASE. Paramount Skydance ran a hostile tender for Warner Bros. Discovery
    weeks after its own reorganisation, so WBD is co-indexed on its filings. WBD filed its own
    proxies and most of its two-decade history predates 2025, so it passes a
    share-of-whole-archive test and outranks the real predecessor on size. What it did not do
    is stop filing -- measured 206 before / 293 after, a ratio of 1.42."""
    pairs = (_stream("2015-01-01", "2026-06-01")
             + _stream("2025-09-16", "2027-06-01", every_days=5)
             + [("DEF 14A", pd.Timestamp("2018-04-01"))])
    got = _run_profile(monkeypatch, pairs, "2025-09-16")
    assert not got["ok"]
    assert "rate ROSE" in got["why"]
    print("\n=== SANITY CHECK: a counterparty that kept filing is rejected ===")
    print(f"  {got['why']}")


def test_the_apache_shape_is_still_accepted(monkeypatch):
    """APA sets the floor on how strict the rate test can be. Apache Corp kept filing its own
    10-K/10-Q for 3.7 years after APA Corp became the parent, because it retains registered
    public debt -- so the test cannot be "stops dead". It is "the rate at least halves"."""
    # Apache's real shape, measured 2026-09-10: 318 filings in the two years before the
    # boundary, 41 in the two years after, ratio 0.129 -- comfortably inside the 0.5 gate even
    # though it never stopped filing.
    pairs = (_stream("2010-01-01", "2021-02-26", every_days=2)
             + _stream("2021-05-07", "2023-02-01", every_days=18)
             + [("DEF 14A", pd.Timestamp("2020-04-03"))])
    got = _run_profile(monkeypatch, pairs, "2021-03-01")
    assert got["ok"], f"the APA shape must survive the rate test: {got['why']}"
    print("\n=== SANITY CHECK: the APA shape (a predecessor that kept filing) ===")
    print(f"  {got['why']}")


# --------------------------------------------------------------------------- #
# Oracle 4 -- the accounting-acquirer test                                    #
# --------------------------------------------------------------------------- #
def _facts(rows: dict[tuple[str, str], float]) -> dict:
    """A companyfacts document carrying `rows` as FY duration facts."""
    out: dict = {"facts": {"us-gaap": {}}}
    for (tag, end), val in rows.items():
        out["facts"]["us-gaap"].setdefault(tag, {"units": {"USD": []}})
        out["facts"]["us-gaap"][tag]["units"]["USD"].append(
            {"fp": "FY", "start": f"{int(end[:4]) - 1}-01-01", "end": end, "val": val})
    return out


def test_oracle4_picks_the_predecessor_whose_history_the_successor_restated(monkeypatch):
    """WHOSE P&L IS THE PRIOR-YEAR COMPARATIVE COLUMN? When two public companies combine under
    a new holdco BOTH stop filing, so every gate in oracle 3 passes for each of them and
    ranking cannot break the tie. But the successor's first 10-K already answered it: it
    restates ONE predecessor's history as its own comparatives -- the accounting acquirer, the
    successor in substance, and the entity whose price series the ticker continues.

    Measured on the real filings: CI resolved to Cigna Corp (12 of 12 annual facts agree,
    FY2017 revenue $41.806bn) rather than Express Scripts, and ICE to IntercontinentalExchange
    Inc (9 of 9, FY2012 $1.363bn) rather than NYSE Euronext -- overturning oracle 3 in both."""
    succ = _facts({("Revenues", "2017-12-31"): 41_806_000_000.0,
                   ("NetIncomeLoss", "2017-12-31"): 2_237_000_000.0})
    right = _facts({("Revenues", "2017-12-31"): 41_806_000_000.0,
                    ("NetIncomeLoss", "2017-12-31"): 2_237_000_000.0})
    wrong = _facts({("Revenues", "2017-12-31"): 100_064_600_000.0,
                    ("NetIncomeLoss", "2017-12-31"): 4_517_400_000.0})
    docs = {"0000000009": succ, "0000000001": right, "0000000002": wrong}
    monkeypatch.setattr(detect, "companyfacts", lambda ctx, cik: docs[str(cik).zfill(10)])

    cands = [{"name": "wrong co", "cik": "0000000002", "co_indexed_on": 9, "profile": {}},
             {"name": "right co", "cik": "0000000001", "co_indexed_on": 1, "profile": {}}]
    winner, evidence = detect.oracle4_comparative(None, "0000000009", cands, "2018-09-21")

    assert winner["cik"] == "0000000001", "the larger/more co-indexed candidate must not win"
    assert evidence["matched"] == 2
    print("\n=== SANITY CHECK: oracle 4 reads the comparative column ===")
    print(f"  {evidence['why']}")
    print("  OK: co-indexing count and archive size did NOT decide it.")


def test_oracle4_abstains_before_xbrl(monkeypatch):
    """⚠ IT MUST ABSTAIN RATHER THAN GUESS. XBRL does not exist before 2009, so COP (2002),
    DUK (2006) and COR (2001) have no comparative facts to read. Returning `None` sends the
    ticker to the `ambiguous` class for hand adjudication -- picking the wrong one of two
    predecessors attaches another company's accounts to the ticker on every consolidating
    form, which is the failure this register exists to prevent."""
    monkeypatch.setattr(detect, "companyfacts", lambda ctx, cik: {"facts": {"us-gaap": {}}})
    cands = [{"name": "a", "cik": "0000000001", "co_indexed_on": 7, "profile": {}},
             {"name": "b", "cik": "0000000002", "co_indexed_on": 2, "profile": {}}]
    assert detect.oracle4_comparative(None, "0000000009", cands, "2002-10-01") is None
    print("\n=== SANITY CHECK: oracle 4 abstains on a pre-XBRL boundary ===")
    print("  no comparative facts -> None -> `ambiguous`, not a guess. Validated.")


def test_oracle4_abstains_on_a_tie(monkeypatch):
    """Two candidates matching equally means the facts do not separate them. Inventing a
    preference here would be exactly the silent guess the `ambiguous` class exists to stop."""
    same = _facts({("Revenues", "2017-12-31"): 1_000.0})
    monkeypatch.setattr(detect, "companyfacts", lambda ctx, cik: same)
    cands = [{"name": "a", "cik": "0000000001", "co_indexed_on": 1, "profile": {}},
             {"name": "b", "cik": "0000000002", "co_indexed_on": 1, "profile": {}}]
    assert detect.oracle4_comparative(None, "0000000009", cands, "2018-09-21") is None
    print("\n=== SANITY CHECK: oracle 4 abstains on a tie ===")
    print("  equal match counts -> None. Validated.")


# --------------------------------------------------------------------------- #
# ⚠ The negative control                                                      #
# --------------------------------------------------------------------------- #
def test_a_recent_ipo_is_not_flagged():
    """⚠ THE LOAD-BEARING TEST OF THE WHOLE DETECTOR.

    The screens key on price history being LONG while the filing archive is SHORT. A genuine
    recent IPO -- COIN, HOOD, CRWD, DDOG -- has a short archive too, and its price history
    starts at the same time, so the lag is ~0 and it is excluded automatically.

    If a short filing archive alone could trigger the screen, the detector would flag every
    young company in the index and would be measuring nothing at all. The SQL is asserted
    here as the arithmetic it performs, so a future edit that drops the price term fails."""
    ipo_first_price, ipo_first_filing = pd.Timestamp("2021-04-14"), pd.Timestamp("2021-05-13")
    cut_first_price, cut_first_filing = pd.Timestamp("1995-09-01"), pd.Timestamp("2019-03-20")

    ipo_lag = (ipo_first_filing - ipo_first_price).days / 365.25
    cut_lag = (cut_first_filing - cut_first_price).days / 365.25

    assert ipo_lag <= detect.LAG_YEARS, "an IPO must not clear the lag threshold"
    assert cut_lag > detect.LAG_YEARS, "a real cutover must clear it"

    print("\n=== SANITY CHECK: negative control, a recent IPO ===")
    print(f"  IPO      first price 2021-04-14, first filing 2021-05-13 -> lag {ipo_lag:.2f} y "
          f"(threshold {detect.LAG_YEARS}) -> NOT flagged")
    print(f"  cutover  first price 1995-09-01, first filing 2019-03-20 -> lag {cut_lag:.2f} y "
          f"-> flagged")
    print("  OK: it is the price-vs-filing ASYMMETRY that fires, not archive length.")


def test_the_screens_reference_the_price_table():
    """The structural half of the negative control: both screens must actually JOIN prices.
    A screen that stopped doing so would still return plausible-looking tickers."""
    for name, sql in (("tight", detect.SCREEN_TIGHT), ("proxy", detect.SCREEN_PROXY),
                      ("all_late", detect.SCREEN_ALL_LATE)):
        assert "first_px" in sql and "FROM prices" in detect._SCREEN_CTES, name
        assert "365.25" in sql, f"{name} does not compute a lag in years"
    print("\n=== SANITY CHECK: every screen joins the price history ===")
    print("  tight, proxy and all_late each divide a filing-vs-price gap by 365.25.")
