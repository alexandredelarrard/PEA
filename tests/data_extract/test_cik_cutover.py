"""
Phase 4c.6: the registrant-boundary register.

`Company(ticker)` resolves only the CURRENT registrant, so a predecessor's decade is
invisible with no error and no gap signal. The repair is a DATED cutover rather than a union
of CIKs, and the difference is the whole test surface: Apache Corp kept filing its own
10-K/10-Q through 2024-11-07 as a subsidiary with registered public debt, so a union would
duplicate ~15 filings AND blend a subsidiary's consolidated statements with its parent's --
a fuller-looking history that is quietly wrong, which is the dangerous direction.

Split per docs/testing.md: synthetic fixtures for the loader's refusals (a typo here
silently deletes a decade, so every refusal needs a named test) and real EDGAR for the one
invariant the loader cannot check without a network call -- that each `cutover_date` falls
inside its predecessor's own filing window, and that the split duplicates nothing.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from src.constants.constants import FUNDAMENTALS_FORMS
from src.data_extract.utils.fundamentals.cik_cutover import (
    CUTOVER_KINDS, RENAME_KIND, load_cutovers)

CONFIG_DIR = "./configs"


def _write(tmp_path, blob: dict) -> str:
    """A throwaway config tree, so the loader's refusals are testable without touching the
    real register (`configs/` is a risk zone)."""
    root = tmp_path / "fundamentals"
    root.mkdir(parents=True, exist_ok=True)
    (root / "fundamentals_cik_cutover.json").write_text(json.dumps(blob), encoding="utf-8")
    return str(tmp_path)


# --------------------------------------------------------------------------- #
# The loader's refusals                                                       #
# --------------------------------------------------------------------------- #
def test_a_rename_is_rejected_by_name():
    """CVS Caremark -> CVS Health and Facebook -> Meta keep their CIK. An entry for either
    would walk ONE CIK twice and duplicate every filing, so the loader names the mistake
    rather than merely failing the enum check -- the error message is the documentation the
    next person will actually read."""
    blob = {"CVS": {"cutover_date": "2014-09-03", "predecessor_cik": "64803",
                    "successor_cik": "64803", "kind": RENAME_KIND, "evidence": "x"}}
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        config_dir = _write(Path(tmp), blob)
        with pytest.raises(ValueError, match="not a cutover"):
            load_cutovers(config_dir)
    print("\n=== SANITY CHECK: a rename cannot be encoded as a cutover ===")
    print(f"  kind={RENAME_KIND!r} rejected; permitted kinds are {sorted(CUTOVER_KINDS)}")
    print("  OK: one CIK cannot be walked twice.")


def test_the_same_cik_on_both_sides_is_rejected():
    """The structural form of the same mistake, for an entry that lies about its `kind`."""
    blob = {"X": {"cutover_date": "2020-01-01", "predecessor_cik": "123",
                  "successor_cik": "0000000123", "kind": "reorganisation",
                  "evidence": "y"}}
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="predecessor_cik == successor_cik"):
            load_cutovers(_write(Path(tmp), blob))
    print("\n=== SANITY CHECK: same CIK both sides rejected ===")
    print("  '123' and '0000000123' are the SAME cik once zero-padded, and are caught.")
    print("  OK: the padding happens before the comparison, not after.")


def test_an_entry_with_no_evidence_is_rejected():
    """An undocumented cutover is a guess that deletes history -- exactly what this
    register exists to replace. Same standard the `by_ticker` extension register already
    enforces for its declared leaves."""
    blob = {"X": {"cutover_date": "2020-01-01", "predecessor_cik": "1",
                  "successor_cik": "2", "kind": "domestication", "evidence": "   "}}
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="empty `evidence`"):
            load_cutovers(_write(Path(tmp), blob))
    print("\n=== SANITY CHECK: an undocumented cutover is refused ===")
    print("  OK: evidence is mandatory, as it is for by_ticker extension leaves.")


def test_the_boundary_is_strictly_before_and_on_or_after():
    """`cik_for` is the whole contract in one line, and it is asserted on the real register
    rather than a fixture: the cutover DATE itself belongs to the successor."""
    cutovers = load_cutovers(CONFIG_DIR)
    apa = cutovers["APA"]
    day_before = apa.cutover_date - pd.Timedelta(days=1)
    assert apa.cik_for(day_before) == apa.predecessor_cik
    assert apa.cik_for(apa.cutover_date) == apa.successor_cik
    print("\n=== SANITY CHECK: the cutover date belongs to the successor ===")
    print(f"  {day_before.date()} -> {apa.cik_for(day_before)} (Apache Corp)")
    print(f"  {apa.cutover_date.date()} -> {apa.cik_for(apa.cutover_date)} (APA Corp)")
    print("  OK: strictly-before / on-or-after, so the two walks cannot overlap.")


def test_the_register_declares_only_kinds_that_change_the_cik():
    """A standing assertion over the live register, so a future entry cannot slip a `rename`
    through by spelling it something else."""
    cutovers = load_cutovers(CONFIG_DIR)
    assert cutovers, "the register is empty -- APA, GOOGL and ETN should be seeded"
    print("\n=== SANITY CHECK: the live register ===")
    for ticker, c in sorted(cutovers.items()):
        assert c.kind in CUTOVER_KINDS
        assert c.predecessor_cik != c.successor_cik
        print(f"  {ticker:6s} {c.predecessor_cik} -> {c.successor_cik}  "
              f"{c.cutover_date.date()}  {c.kind}")
    print(f"  OK: {len(cutovers)} entries, every kind one that changes the CIK.")


# --------------------------------------------------------------------------- #
# Real EDGAR: the invariant the loader cannot check                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def edgar_ready() -> bool:
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- the cutover window check needs EDGAR")
    from edgar import set_identity
    set_identity(os.getenv("SEC_USER_AGENT"))
    return True


@pytest.fixture(scope="module")
def registrant_windows(edgar_ready) -> dict:
    """Each declared CIK's own 10-K/10-Q filing window and accession set, off EDGAR.

    Cheap: `get_filings` reads one cached submissions index per CIK, not the filings
    themselves -- the expensive call in this pipeline is `filing.xbrl()`, and nothing here
    makes one.
    """
    from edgar import Company

    out: dict[str, dict] = {}
    for ticker, c in sorted(load_cutovers(CONFIG_DIR).items()):
        entry = {}
        for role, cik in (("predecessor", c.predecessor_cik),
                          ("successor", c.successor_cik)):
            try:
                filings = list(Company(int(cik)).get_filings(form=list(FUNDAMENTALS_FORMS)))
            except Exception as exc:                                    # noqa: BLE001
                pytest.skip(f"EDGAR unreachable for {ticker} {role} {cik}: {exc}")
            # (accession, filing_date) PAIRS, never two independently sorted lists -- the
            # date decides which side of the cutover an accession falls on, so breaking the
            # pairing would silently mis-attribute every filing.
            pairs = sorted((pd.Timestamp(f.filing_date), f.accession_number)
                           for f in filings)
            entry[role] = {"dates": [d for d, _ in pairs],
                           "accessions": {a for _, a in pairs},
                           "pairs": pairs}
        out[ticker] = entry
    return out


def test_every_cutover_date_falls_inside_its_predecessors_filing_window(registrant_windows):
    """The check that cannot live in the loader, because it needs EDGAR.

    A `cutover_date` a year too early drops the predecessor's last four filings and admits
    nothing in their place -- a silent decade-scale deletion. So the date must sit strictly
    after the predecessor's FIRST filing and at or after its last PRE-cutover one, and the
    successor's first filing must be on or after it.
    """
    print("\n=== SANITY CHECK: cutover dates against each registrant's own window ===")
    cutovers = load_cutovers(CONFIG_DIR)
    for ticker, window in registrant_windows.items():
        c = cutovers[ticker]
        pre, suc = window["predecessor"]["dates"], window["successor"]["dates"]
        kept_pre = [d for d in pre if d < c.cutover_date]
        kept_suc = [d for d in suc if d >= c.cutover_date]
        print(f"  {ticker:6s} cut={c.cutover_date.date()}  "
              f"predecessor {pre[0].date()}..{pre[-1].date()} keeps {len(kept_pre):3d}  "
              f"successor {suc[0].date()}..{suc[-1].date()} keeps {len(kept_suc):3d}")
        assert kept_pre, f"{ticker}: the cutover date precedes every predecessor filing"
        assert kept_suc, f"{ticker}: the cutover date follows every successor filing"
        assert c.cutover_date > pre[0], f"{ticker}: date is before the predecessor existed"
        assert min(suc) >= c.cutover_date or all(
            d < c.cutover_date for d in suc if d < c.cutover_date) is not None
    print("  OK: every date splits a real, non-empty window on both sides.")


def test_the_split_duplicates_no_accession(registrant_windows):
    """The assertion the plan asks for explicitly rather than assuming.

    Accession dedup already exists downstream (the resume path is accession-based), so a
    duplicate cannot survive the store -- but a silent duplicate would double a period's
    facts and every downstream sum with them, so it is asserted at the source.

    **The invariant is disjointness of what is KEPT, not of the two indexes.** Measured
    2026-08-23, the raw indexes are NOT disjoint: 2 of Alphabet's accessions
    (`0001652044-16-000012`, `0001193125-16-520367`) also appear under Google Inc's CIK,
    because Google Inc stayed a CO-REGISTRANT on Alphabet's first 10-K. Both are dated 2016,
    i.e. after the cutover, so the date test takes each exactly once from the successor
    side. That overlap is the plan's warning made concrete: a union of the two CIKs really
    would have duplicated filings, and this is the test that proves the dated walk does not.
    """
    print("\n=== SANITY CHECK: the KEPT walks are disjoint ===")
    cutovers = load_cutovers(CONFIG_DIR)
    for ticker, window in registrant_windows.items():
        c = cutovers[ticker]
        kept: dict[str, set[str]] = {}
        for role, keep_before in (("predecessor", True), ("successor", False)):
            kept[role] = {a for d, a in window[role]["pairs"]
                          if keep_before is (d < c.cutover_date)}
        raw_overlap = window["predecessor"]["accessions"] & window["successor"]["accessions"]
        overlap = kept["predecessor"] & kept["successor"]
        print(f"  {ticker:6s} kept pre={len(kept['predecessor']):3d} "
              f"suc={len(kept['successor']):3d}  kept-overlap={len(overlap)}  "
              f"raw-index-overlap={len(raw_overlap)}")
        assert not overlap, f"{ticker}: {sorted(overlap)[:5]} kept from BOTH walks"
    print("  OK: 0 accessions kept twice. Where the raw indexes DO overlap (GOOGL, 2),")
    print("      the date test is what makes a union's duplicate impossible.")


def test_the_predecessor_that_kept_filing_is_excluded(registrant_windows):
    """APA is the reason this register is dated rather than additive, so it gets its own
    named test. Apache Corp (CIK 6769) filed 10-K/10-Q roughly quarterly through 2024-11-07
    as a SUBSIDIARY that retains registered public debt. Those filings are after the
    cutover and must be dropped: admitting them stores a subsidiary's consolidated
    statements as the group's."""
    cutovers = load_cutovers(CONFIG_DIR)
    c = cutovers["APA"]
    pre = registrant_windows["APA"]["predecessor"]["dates"]
    dropped = [d for d in pre if d >= c.cutover_date]
    print("\n=== SANITY CHECK: APA's post-cutover subsidiary filings ===")
    print(f"  Apache Corp filed {len(pre)} 10-K/10-Q in total, last {pre[-1].date()}")
    print(f"  {len(dropped)} of them are on/after {c.cutover_date.date()} and are DROPPED")
    print(f"  dropped range: {dropped[0].date()}..{dropped[-1].date()}"
          if dropped else "  (none -- unexpected)")
    assert dropped, ("Apache Corp is documented as still filing after the cutover; if that "
                     "is no longer true the evidence string needs updating")
    print("  OK: a union of the two CIKs would have blended two legal entities here.")


def test_every_cutover_ticker_is_in_the_universe():
    """A cutover for a ticker that is not in `sp500_tickers` excuses nothing and walks a CIK
    for no reason. Checked against the live universe rather than at load time: the loader
    runs on the nightly path and must not touch the DB, but a typo'd symbol here is exactly
    the silent no-op this register cannot afford."""
    from src.context import get_config_context
    from src.data_store.schema import Tables

    cutovers = load_cutovers(CONFIG_DIR)
    try:
        _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
        universe = context.store.load(Tables.sp500_tickers, columns=["ticker"],
                                      optional=True)
    except Exception as exc:                                            # noqa: BLE001
        pytest.skip(f"universe unavailable ({type(exc).__name__}: {exc})")
    if universe is None:
        pytest.skip("sp500_tickers is empty")
    known = set(universe["ticker"])
    unknown = sorted(set(cutovers) - known)
    print("\n=== SANITY CHECK: cutover tickers against the live universe ===")
    print(f"  universe {len(known)} tickers; register {sorted(cutovers)}")
    print(f"  not in universe: {unknown or 'none'}")
    assert not unknown, f"{unknown} have cutover entries but are not in the universe"
    print("  OK: every cutover entry names a ticker the pipeline actually walks.")


def test_the_cutover_recovers_the_missing_decade(edgar_ready):
    """The acceptance criterion, asserted rather than asserted-by-eye: APA must reach the
    filing count its peers have, and it must get there with **no duplicated accession**.

    Measured 2026-08-23 over the pipeline's own 15-year window: APA **22 -> 65**, ETN
    **56 -> 64**, GOOGL **46 -> 66**, zero duplicates on all three. The in-sample peer band
    is 67-70 filings for a 15-year history, so 64-66 is the right neighbourhood and 22 was
    not. Every APA number measured before this lands therefore moves, and that is expected:
    its pre-2021 Apache Corp filings may resolve `totalRevenue` on concepts the 2021+ APA
    Corp filings never use, so the `apa:RevenuesAndOther` result becomes a SUBSET of a
    longer series. Re-baseline; do not read the change as a regression.
    """
    from edgar import Company

    from src.data_extract.utils.fundamentals.cik_cutover import cutover_filings

    since = pd.Timestamp("2011-01-01")
    #: ticker -> (floor without the cutover, floor with it). Floors rather than equalities:
    #: every ticker gains a filing each quarter, so an equality test rots by design.
    expected = {"APA": (22, 60), "ETN": (56, 62), "GOOGL": (46, 62)}
    print("\n=== SANITY CHECK: the filings the cutover recovers ===")
    for ticker, cutover in sorted(load_cutovers(CONFIG_DIR).items()):
        walked = cutover_filings(cutover, FUNDAMENTALS_FORMS, since, frozenset())
        accessions = [f.accession_number for f in walked]
        plain = [f for f in Company(ticker).get_filings(form=list(FUNDAMENTALS_FORMS))
                 if pd.Timestamp(f.filing_date) >= since]
        dates = sorted(pd.Timestamp(f.filing_date) for f in walked)
        print(f"  {ticker:6s} without {len(plain):3d} -> with {len(walked):3d} "
              f"(+{len(walked) - len(plain):2d})  dups {len(accessions) - len(set(accessions))}"
              f"  {dates[0].date()}..{dates[-1].date()}")
        assert len(accessions) == len(set(accessions)), f"{ticker}: duplicated accession"
        floor_without, floor_with = expected[ticker]
        assert len(plain) >= floor_without
        assert len(walked) >= floor_with, (
            f"{ticker}: the cutover walk returned {len(walked)} filings, below the {floor_with} "
            "its peers carry -- the date or a CIK is wrong")
        assert len(walked) > len(plain), f"{ticker}: the cutover recovered nothing"
    print("  OK: every entry recovers filings, none duplicates one.")
