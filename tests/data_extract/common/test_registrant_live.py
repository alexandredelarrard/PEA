"""The register against LIVE EDGAR -- the invariants a config loader must never check itself.

`registrant.py` validates everything it can at load time, because a typo silently deletes a
decade rather than raising. The one class of check it CANNOT make is anything needing the
network: a loader runs on the nightly path and must not make an HTTP call. Those checks live
here, parametrised over EVERY register entry rather than a hand-picked three, so a boundary
added next quarter is asserted the moment it lands.

Run deliberately:

    "$PY" -m pytest -m network tests/data_extract/common/test_registrant_live.py -q -s

⚠ ONE EDGAR WALK AT A TIME. The rate limiter is per-PROCESS, so running this alongside an
extraction puts ~18 req/s against SEC's limit of 10. The block is silent.

Supersedes `tests/data_extract/fundamentals/test_cik_cutover.py`, which asserted the same
invariants over the old two-CIK register and is deleted rather than left beside its successor.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.constants.constants import FUNDAMENTALS_FORMS
from src.data_extract.utils.common.registrant import Registrant, load_registrants

pytestmark = pytest.mark.network

CONFIG_DIR = "./configs"

#: Event forms combine as a UNION across a boundary; consolidating forms take a dated SPLIT.
#: The two rules are asserted separately here because the whole of `FORM_POLICY` rests on the
#: claim that they genuinely differ -- see `test_an_event_form_needs_the_union`.
EVENT_FORMS = ["8-K", "SC 13G", "SC 13G/A", "4"]


@pytest.fixture(scope="module")
def edgar_ready() -> bool:
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- these checks need EDGAR")
    from edgar import set_identity
    set_identity(os.getenv("SEC_USER_AGENT"))
    return True


@pytest.fixture(scope="module")
def registrants() -> dict[str, Registrant]:
    regs = load_registrants(CONFIG_DIR)
    if not regs:
        pytest.skip("the register is empty")
    return regs


@pytest.fixture(scope="module")
def consolidating(edgar_ready, registrants) -> dict[str, dict[str, dict]]:
    """Each segment CIK's own 10-K/10-Q window and accession set.

    Cheap: `get_filings` reads one cached submissions index per CIK, not the filings
    themselves -- the expensive call in this pipeline is `filing.xbrl()`, and nothing here
    makes one.
    """
    from edgar import Company

    out: dict[str, dict[str, dict]] = {}
    for ticker, reg in sorted(registrants.items()):
        per_cik: dict[str, dict] = {}
        for cik in reg.all_ciks():
            try:
                filings = list(Company(int(cik)).get_filings(form=list(FUNDAMENTALS_FORMS)))
            except Exception as exc:                                    # noqa: BLE001
                pytest.skip(f"EDGAR unreachable for {ticker} CIK {cik}: {exc}")
            # (date, accession) PAIRS, never two independently sorted lists -- the date
            # decides which segment an accession falls in, so breaking the pairing would
            # silently mis-attribute every filing.
            pairs = sorted((pd.Timestamp(f.filing_date), f.accession_number) for f in filings)
            per_cik[cik] = {"pairs": pairs, "dates": [d for d, _ in pairs],
                            "accessions": {a for _, a in pairs}}
        out[ticker] = per_cik
    return out


def test_every_boundary_falls_inside_its_predecessors_filing_window(registrants,
                                                                    consolidating):
    """The check that cannot live in the loader, because it needs EDGAR.

    A boundary set a year early drops the predecessor's last four filings and admits nothing
    in their place -- a silent, decade-scale deletion. So each segment must keep at least one
    filing of its own: the boundary sits strictly after the predecessor's first filing, and
    the successor has filed on or after it.
    """
    print("\n=== SANITY CHECK: every boundary against its own registrants' windows ===")
    for ticker, reg in sorted(registrants.items()):
        per_cik = consolidating[ticker]
        for i, segment in enumerate(reg.segments):
            kept = [d for d in per_cik[segment.cik]["dates"] if segment.covers(d)]
            span = per_cik[segment.cik]["dates"]
            label = (f"{str(segment.valid_from.date()) if segment.valid_from else '   ...   '}"
                     f" .. {str(segment.valid_to.date()) if segment.valid_to else '   ...   '}")
            print(f"  {ticker:6s} seg{i} {segment.cik} {label} keeps {len(kept):3d} of "
                  f"{len(span):3d}" + (f"  ({span[0].date()}..{span[-1].date()})" if span else ""))
            assert kept, (
                f"{ticker} segment {i} (CIK {segment.cik}) keeps NO 10-K/10-Q. Either the "
                "boundary is wrong or this CIK never was the registrant -- both delete "
                "history silently.")
    print("  OK: every segment keeps a non-empty slice of its own filings.")


def test_the_split_duplicates_no_accession(registrants, consolidating):
    """Disjointness of what is KEPT, which is not the same as disjointness of the two indexes.

    Measured 2026-08-23, the raw indexes are NOT disjoint: 2 of Alphabet's accessions
    (`0001652044-16-000012`, `0001193125-16-520367`) also appear under Google Inc's CIK,
    because Google Inc stayed a CO-REGISTRANT on Alphabet's first 10-K. Both are dated 2016,
    i.e. after the boundary, so the date test takes each exactly once from the successor side.
    That overlap is the warning made concrete: a union of the two CIKs really would duplicate
    filings on a consolidating form, and this is the test that proves the dated walk does not.
    """
    print("\n=== SANITY CHECK: the KEPT walks are disjoint ===")
    for ticker, reg in sorted(registrants.items()):
        per_cik = consolidating[ticker]
        kept: dict[str, set[str]] = {
            s.cik: {a for d, a in per_cik[s.cik]["pairs"] if s.covers(d)} for s in reg.segments}
        seen: set[str] = set()
        overlaps: set[str] = set()
        for accessions in kept.values():
            overlaps |= seen & accessions
            seen |= accessions
        raw = sum(len(per_cik[c]["accessions"]) for c in reg.all_ciks()) - len(
            set().union(*(per_cik[c]["accessions"] for c in reg.all_ciks())))
        print(f"  {ticker:6s} kept {'+'.join(str(len(v)) for v in kept.values()):>12s} = "
              f"{len(seen):3d}  kept-overlap={len(overlaps)}  raw-index-overlap={raw}")
        assert not overlaps, f"{ticker}: {sorted(overlaps)[:5]} kept from TWO segments"
    print("  OK: 0 accessions kept twice. Where the raw indexes DO overlap, it is the date")
    print("      test that makes a union's duplicate impossible.")


def test_there_is_no_gap_at_a_boundary(registrants, consolidating):
    """The predecessor's last kept filing is followed by the successor's first, with no
    reporting period unclaimed between them. A gap is the other half of the failure mode: an
    overlap double-counts, a gap deletes, and only one of the two is visible downstream."""
    print("\n=== SANITY CHECK: no reporting gap at a boundary ===")
    for ticker, reg in sorted(registrants.items()):
        per_cik = consolidating[ticker]
        for older, newer in zip(reg.segments, reg.segments[1:]):
            last = max((d for d in per_cik[older.cik]["dates"] if older.covers(d)),
                       default=None)
            first = min((d for d in per_cik[newer.cik]["dates"] if newer.covers(d)),
                        default=None)
            if last is None or first is None:
                continue
            gap = (first - last).days
            print(f"  {ticker:6s} {last.date()} -> {first.date()}  gap {gap:4d} d "
                  f"across {newer.valid_from.date()}")
            assert gap <= MAX_REPORTING_GAP_DAYS, (
                f"{ticker}: {gap} d between the predecessor's last filing and the successor's "
                "first -- a reporting period belongs to neither segment")
    print(f"  OK: every seam is bridged within {MAX_REPORTING_GAP_DAYS} d (one filing cycle).")


#: A 10-Q lands roughly quarterly and a 10-K annually, so the widest legitimate gap across a
#: seam is one annual cycle plus filing lag. Wider than that and a period belongs to nobody.
MAX_REPORTING_GAP_DAYS = 400


def test_a_predecessor_that_kept_filing_is_excluded_from_consolidating_forms(registrants,
                                                                            consolidating):
    """APA is the reason the register is dated rather than additive, so it gets its own test.

    Apache Corp (CIK 6769) filed 10-K/10-Q roughly quarterly for 3.7 years after APA Corp
    became the parent, because it retains registered public debt. Those filings are a
    SUBSIDIARY's consolidated statements and admitting them would store them as the group's.
    """
    shapes = {t: [d for d in consolidating[t][r.segments[0].cik]["dates"]
                  if not r.segments[0].covers(d)]
              for t, r in registrants.items()}
    kept_filing = {t: v for t, v in shapes.items() if v}
    print("\n=== SANITY CHECK: predecessors that kept filing after their boundary ===")
    for ticker, dropped in sorted(kept_filing.items()):
        print(f"  {ticker:6s} {len(dropped):3d} predecessor 10-K/10-Q after the boundary "
              f"({dropped[0].date()}..{dropped[-1].date()}) -> EXCLUDED from the parent")
    assert kept_filing, (
        "no register predecessor kept filing after its boundary. APA is documented as doing "
        "so; if that is no longer true the evidence strings need updating, and the dated "
        "split has lost its motivating case.")
    print("  OK: a union of these CIKs would blend two legal entities' accounts.")


def test_an_event_form_needs_the_union(edgar_ready, registrants):
    """⚠ THE D9.1 ASSERTION, and it is the mirror image of the test above.

    On CONSOLIDATING forms a predecessor's post-boundary filings must be dropped. On EVENT
    forms they must be KEPT, and the case is named: XOM's SCHEDULE 13G of 2026-08-07 is filed
    under the predecessor five weeks after the 2026-07-01 boundary, and it is already in the
    database. Applying the dated split to event forms would discard a filing we hold.

    So the two rules are not reconcilable into one, and this test is what stops a future
    tidy-up from trying.
    """
    from edgar import Company

    print("\n=== SANITY CHECK: predecessor EVENT filings after a boundary ===")
    total_after = 0
    for ticker, reg in sorted(registrants.items()):
        oldest = reg.segments[0]
        try:
            filings = list(Company(int(oldest.cik)).get_filings(form=EVENT_FORMS))
        except Exception as exc:                                        # noqa: BLE001
            pytest.skip(f"EDGAR unreachable for {ticker} CIK {oldest.cik}: {exc}")
        after = sorted(pd.Timestamp(f.filing_date) for f in filings
                       if not oldest.covers(pd.Timestamp(f.filing_date)))
        total_after += len(after)
        if after:
            print(f"  {ticker:6s} {len(after):5d} predecessor event filings after the "
                  f"boundary ({after[0].date()}..{after[-1].date()}) -> a SPLIT would lose them")
    assert total_after > 0, (
        "no predecessor filed an event form after its boundary, which would mean the union "
        "rule costs nothing and is unmotivated. XOM's 2026-08-07 SCHEDULE 13G is the "
        "documented case; re-measure before relaxing FORM_POLICY.")
    print(f"  OK: {total_after} event filings across the register would be discarded by a")
    print("      dated split. This is why event forms UNION.")


def test_every_register_ticker_is_in_the_universe():
    """A register entry for a ticker the pipeline never walks excuses nothing and walks a CIK
    for no reason. Checked against the live universe rather than at load time: the loader runs
    on the nightly path and must not touch the DB, but a typo'd symbol is exactly the silent
    no-op this register cannot afford."""
    from src.context import get_config_context
    from src.data_store.schema import Tables

    registrants = load_registrants(CONFIG_DIR)
    try:
        _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
        universe = context.store.load(Tables.sp500_tickers, columns=["ticker"], optional=True)
    except Exception as exc:                                            # noqa: BLE001
        pytest.skip(f"universe unavailable ({type(exc).__name__}: {exc})")
    if universe is None:
        pytest.skip("sp500_tickers is empty")
    unknown = sorted(set(registrants) - set(universe["ticker"]))
    print("\n=== SANITY CHECK: register tickers against the live universe ===")
    print(f"  universe {len(universe)} tickers; register {len(registrants)} entries")
    print(f"  not in universe: {unknown or 'none'}")
    assert not unknown, f"{unknown} have register entries but are not in the universe"
    print("  OK: every entry names a ticker the pipeline actually walks.")


def test_the_register_recovers_history_the_ticker_walk_cannot_reach(edgar_ready, registrants):
    """The acceptance criterion, asserted rather than asserted-by-eye: walking the register's
    CIKs must reach filings `Company(ticker)` alone does not.

    Reported as a floor, never an equality -- every ticker gains a filing each quarter, so an
    equality test rots by design.

    ⚠ A ZERO GAIN IS NOT AUTOMATICALLY A FAILURE, BUT IT IMPOSES A STRICTER TEST. XOM's entry
    exists for the PROXY path: EDGAR still maps the XOM ticker to the predecessor for
    fundamentals, so this walk gains nothing -- measured 64 -> 64. For such an entry the
    requirement is the stronger one and it is the whole point of adding it safely: the walk
    must be provably UNCHANGED, accession for accession. XOM's non-governance history was
    already clean (fundamentals_history 124, fundamentals_facts 6,185, sec_8k 521, prices
    7,803) and a register entry added for one pipeline must not disturb another.

    Stated as a rule rather than a ticker list, so a future proxy-path entry is covered on the
    day it lands: an entry that GAINS nothing must also LOSE nothing.
    """
    from edgar import Company

    print("\n=== SANITY CHECK: what the register reaches that the ticker walk does not ===")
    gained_total = 0
    for ticker, reg in sorted(registrants.items()):
        try:
            plain = {f.accession_number
                     for f in Company(ticker).get_filings(form=list(FUNDAMENTALS_FORMS))}
            walked: set[str] = set()
            for segment in reg.segments:
                walked |= {f.accession_number for f in
                           Company(int(segment.cik)).get_filings(form=list(FUNDAMENTALS_FORMS))
                           if segment.covers(pd.Timestamp(f.filing_date))}
        except Exception as exc:                                        # noqa: BLE001
            pytest.skip(f"EDGAR unreachable for {ticker}: {exc}")
        gained, lost = walked - plain, plain - walked
        gained_total += len(gained)
        print(f"  {ticker:6s} ticker walk {len(plain):3d} | register walk {len(walked):3d} | "
              f"+{len(gained):3d} recovered, -{len(lost):3d} lost")
        if not gained:
            assert not lost, (
                f"{ticker}: its entry recovers NOTHING on this path yet DROPS {len(lost)} "
                "accession(s) the plain ticker walk returns. An entry added for another "
                "pipeline must leave this one accession-identical.")
    assert gained_total > 0, (
        "the register reaches nothing the plain ticker walk does not, which would mean it is "
        "doing no work at all")
    print(f"  OK: {gained_total} filings across the register are reachable only through it,")
    print("      and every entry that recovers nothing here also loses nothing.")
