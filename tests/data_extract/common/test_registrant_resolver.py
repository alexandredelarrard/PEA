"""`resolve_registrant_filings` -- the ONE place that decides which CIKs a ticker's filings
come from, and how they combine.

Before this there were two implementations of one rule, and they looked contradictory:
`edgar_driver.new_filings` unioned the register's CIKs while `cik_cutover.cutover_filings`
split them by date. Both were right, for different form families, and neither said so at the
other's call site -- so the five event fetchers and the fundamentals walk each carried half
the reasoning. `FORM_POLICY` is that reasoning made explicit, and these tests pin it.

⚠ The single most important test here is `test_no_register_entry_is_byte_identical`. ~449 of
the 491 tickers have no entry, so a resolver that changed their behaviour would move far more
data than the register ever could -- and it would move it invisibly, because there is no
before/after to compare a ticker against when every ticker moved.

Offline: a stub `Company`, no network.
"""
from __future__ import annotations

import types

import pandas as pd
import pytest

from src.data_extract.utils.common.registrant import (
    Combine, FORM_POLICY, Registrant, Segment, combine_for, resolve_registrant_filings)

BOUNDARY = pd.Timestamp("2026-07-01")


def _filing(accession: str, filing_date: str):
    return types.SimpleNamespace(accession_number=accession, filing_date=filing_date)


def _patch(monkeypatch, by_key: dict):
    """`Company(x)` -> that registrant's filings. Keys are what the resolver passes: the
    TICKER string for the ticker-resolved lookup, and an `int` CIK per segment."""
    monkeypatch.setattr("edgar.Company",
                        lambda x: types.SimpleNamespace(
                            get_filings=lambda form: by_key.get(x, [])))


def _chain(*pairs) -> dict[str, Registrant]:
    """`_chain(("0000000001", None, "2020-01-01"), ...)` -> `{"T": Registrant}`."""
    segs = tuple(Segment(cik=c,
                         valid_from=pd.Timestamp(f) if f else None,
                         valid_to=pd.Timestamp(t) if t else None,
                         evidence="fixture")
                 for c, f, t in pairs)
    return {"T": Registrant(ticker="T", kind="reorganisation", segments=segs)}


_XOM = {"XOM": Registrant(ticker="XOM", kind="reorganisation", segments=(
    Segment(cik="0000034088", valid_from=None, valid_to=BOUNDARY, evidence="fixture"),
    Segment(cik="0002115436", valid_from=BOUNDARY, valid_to=None, evidence="fixture")))}


# --------------------------------------------------------------------------- #
# The policy table                                                             #
# --------------------------------------------------------------------------- #
def test_an_undeclared_form_raises_and_the_message_names_it():
    """⚠ FAIL CLOSED. A silent default is precisely how this defect class stayed invisible for
    a year: `Company(ticker)` returned one registrant and nothing said so. A new form family
    quietly inheriting the wrong combination rule is the same failure in new clothes."""
    with pytest.raises(ValueError, match="NT 10-K"):
        combine_for(["NT 10-K"])
    print("\n=== SANITY CHECK: an undeclared form raises ===")
    print("  'NT 10-K' has no FORM_POLICY entry -> ValueError naming it. Validated.")


def test_a_mixed_policy_list_raises_at_the_call_site():
    """One call cannot both union and split. The caller is asking one question with two right
    answers, which is a bug where the question is asked, not something to resolve here."""
    with pytest.raises(ValueError, match="mix"):
        combine_for(["8-K", "10-K"])
    print("\n=== SANITY CHECK: a mixed forms list raises ===")
    print("  ['8-K', '10-K'] mixes union and split -> refused. Validated.")


@pytest.mark.parametrize(("forms", "expected"), [
    (["8-K", "8-K/A"], Combine.UNION),
    (["SC 13D", "SCHEDULE 13D/A"], Combine.UNION),
    (["SC 13G", "SC 13G/A", "SCHEDULE 13G", "SCHEDULE 13G/A"], Combine.UNION),
    (["3", "4", "5"], Combine.UNION),
    (["10-K", "10-K/A", "10-Q", "10-Q/A"], Combine.SPLIT),
    (["10-K", "10-Q"], Combine.SPLIT),                       # FILING_TEXT_FORMS
    (["DEF 14A", "DEF 14C", "DEFC14A"], Combine.SPLIT),
])
def test_every_fetched_form_family_has_the_policy_its_pipeline_needs(forms, expected):
    """The repo's actual form lists, each resolving to one policy.

    ⚠ `FILING_TEXT_FORMS` is 10-K/10-Q -- the same forms as fundamentals, and SPLIT for the
    same reason: Item 1A/7 text is a registrant's own narrative, so a subsidiary's MD&A must
    not land in the parent's series.

    ⚠ The proxy is SPLIT, not UNION, deliberately. Two registrants' proxies for one year
    would give the governance panel two boards -- a corrupted grain rather than a duplicated
    row. APA measured 0 predecessor proxies after its boundary, so it costs nothing observed.
    """
    assert combine_for(forms) is expected
    print(f"\n=== SANITY CHECK: {forms[0]}... -> {expected.value} ===")


def test_both_spellings_of_the_renamed_schedules_are_declared():
    """EDGAR renamed 13D/13G at the 2024-12-17 structured-XML mandate and
    `get_filings(form=...)` matches EXACTLY -- 461 filings across 91 tickers were once
    invisible because only one spelling was listed. A policy table that declared one spelling
    and not the other would raise mid-run on the changeover instead."""
    for pair in (("SC 13D", "SCHEDULE 13D"), ("SC 13D/A", "SCHEDULE 13D/A"),
                 ("SC 13G", "SCHEDULE 13G"), ("SC 13G/A", "SCHEDULE 13G/A")):
        assert all(f in FORM_POLICY for f in pair), pair
    print("\n=== SANITY CHECK: both form-string eras are declared ===")
    print("  SC 13D/G and SCHEDULE 13D/G, base and /A, all present. Validated.")


# --------------------------------------------------------------------------- #
# ⚠ The no-entry path                                                          #
# --------------------------------------------------------------------------- #
def test_no_register_entry_is_byte_identical_to_the_plain_ticker_walk(monkeypatch):
    """⚠ THE TEST THAT PROTECTS THE ~449 UNTOUCHED TICKERS.

    Only ~42 of 491 tickers have a register entry. If the resolver changed the other ~449 at
    all, it would move far more data than the register ever could -- and invisibly, because
    with every ticker moving there is no unchanged population to compare against."""
    plain = [_filing("c", "2020-03-01"), _filing("a", "2018-01-01"), _filing("b", "2019-02-01")]
    _patch(monkeypatch, {"AAPL": plain})

    out = resolve_registrant_filings("AAPL", ["8-K"], since=None,
                                     done_accessions=frozenset(), registrants={})

    assert [f.accession_number for f in out] == ["a", "b", "c"]
    assert all(f in plain for f in out), "a filing object was substituted, not just reordered"
    print("\n=== SANITY CHECK: no register entry -> the plain ticker walk ===")
    print("  same 3 filing OBJECTS, sorted oldest-first, nothing added or dropped. Validated.")


# --------------------------------------------------------------------------- #
# UNION                                                                        #
# --------------------------------------------------------------------------- #
def test_union_recovers_the_successors_filings(monkeypatch):
    """The defect this fixes: three real XOM 8-Ks reached no table at all, because the ticker
    resolves to the predecessor and nothing walked the successor."""
    _patch(monkeypatch, {"XOM": [_filing("pred-old", "2026-05-01")],
                         2115436: [_filing("suc-1", "2026-07-07"),
                                   _filing("suc-2", "2026-08-28")]})

    out = resolve_registrant_filings("XOM", ["8-K"], since=None,
                                     done_accessions=frozenset(), registrants=_XOM)

    assert [f.accession_number for f in out] == ["pred-old", "suc-1", "suc-2"]
    print("\n=== SANITY CHECK: the union recovers the successor's filings ===")
    print("  successor-only ['suc-1', 'suc-2'] now reachable. Validated.")


def test_union_keeps_a_predecessor_filing_dated_after_the_boundary(monkeypatch):
    """⚠ THE ANTI-REGRESSION TEST, and the reason event forms UNION where consolidating ones
    SPLIT. XOM's SCHEDULE 13G of 2026-08-07 is filed under the PREDECESSOR five weeks after
    the 2026-07-01 boundary. A dated split would discard a filing already in the database, so
    applying the fundamentals rule to the event pipelines LOSES data."""
    _patch(monkeypatch, {"XOM": [_filing("pred-late", "2026-08-07")],
                         2115436: [_filing("suc-1", "2026-07-07")]})

    out = resolve_registrant_filings("XOM", ["SCHEDULE 13G"], since=None,
                                     done_accessions=frozenset(), registrants=_XOM)
    kept = [f.accession_number for f in out]

    assert "pred-late" in kept, "a dated split would have dropped this"
    assert kept == ["suc-1", "pred-late"]
    print("\n=== SANITY CHECK: a post-boundary predecessor event survives ===")
    print("  'pred-late' (2026-08-07, past a 2026-07-01 boundary) kept. Validated.")


def test_union_takes_a_co_indexed_accession_once(monkeypatch):
    """XOM's 2026-08-03 10-Q carries ONE accession indexed under BOTH CIKs -- which is why
    fundamentals stayed clean while `sec_8k` silently lost filings. It must not arrive
    twice."""
    shared = _filing("0000034088-26-000093", "2026-08-03")
    _patch(monkeypatch, {"XOM": [shared], 34088: [shared], 2115436: [shared]})

    out = resolve_registrant_filings("XOM", ["8-K"], since=None,
                                     done_accessions=frozenset(), registrants=_XOM)

    assert [f.accession_number for f in out] == ["0000034088-26-000093"]
    print("\n=== SANITY CHECK: a co-indexed accession is taken once ===")
    print("  same accession under both registrants -> 1 filing. Validated.")


def test_union_with_a_garbage_register_cik_loses_no_filing(monkeypatch):
    """⚠ THE SAFETY PROPERTY OF ADDING RATHER THAN SUBSTITUTING. Register CIKs are ADDED to
    whatever `Company(ticker)` returns, so a stale, wrong or dead entry can only fail to gain
    -- it can never lose a filing the no-entry path would have found."""
    def _company(x):
        if x == 2115436:
            raise ValueError("no such company")
        return types.SimpleNamespace(
            get_filings=lambda form: [_filing("pred", "2026-05-01")] if x == "XOM" else [])

    monkeypatch.setattr("edgar.Company", _company)

    out = resolve_registrant_filings("XOM", ["8-K"], since=None,
                                     done_accessions=frozenset(), registrants=_XOM)

    assert [f.accession_number for f in out] == ["pred"]
    print("\n=== SANITY CHECK: a dead register CIK costs nothing ===")
    print("  successor unresolvable -> the predecessor's filing still returned. Validated.")


def test_union_prefers_the_ticker_resolved_registrant_as_first_writer(monkeypatch):
    """First-writer-wins with the ticker first, so every filing the pre-register
    implementation already returned keeps the object it returned then."""
    from_ticker = _filing("shared", "2026-05-01")
    from_segment = _filing("shared", "2026-05-01")
    _patch(monkeypatch, {"XOM": [from_ticker], 34088: [from_segment]})

    out = resolve_registrant_filings("XOM", ["8-K"], since=None,
                                     done_accessions=frozenset(), registrants=_XOM)

    assert out[0] is from_ticker, "the segment's copy displaced the ticker-resolved one"
    print("\n=== SANITY CHECK: the ticker-resolved registrant writes first ===")
    print("  provenance of an already-returned filing is unchanged. Validated.")


# --------------------------------------------------------------------------- #
# SPLIT                                                                        #
# --------------------------------------------------------------------------- #
def test_split_is_strictly_before_and_on_or_after(monkeypatch):
    """The boundary date belongs to the SUCCESSOR, which is what makes the two walks disjoint
    by construction rather than by de-duplication."""
    _patch(monkeypatch, {34088: [_filing("pre-eve", "2026-06-30"),
                                 _filing("pre-late", "2026-08-07")],
                         2115436: [_filing("suc-day", "2026-07-01")]})

    out = resolve_registrant_filings("XOM", ["10-Q"], since=None,
                                     done_accessions=frozenset(), registrants=_XOM)

    assert [f.accession_number for f in out] == ["pre-eve", "suc-day"]
    print("\n=== SANITY CHECK: the split at the exact boundary ===")
    print("  2026-06-30 -> predecessor; 2026-07-01 -> successor; the predecessor's")
    print("  2026-08-07 filing is EXCLUDED -- the Apache-subsidiary rule. Validated.")


def test_split_excludes_the_predecessors_post_boundary_consolidating_filings(monkeypatch):
    """APA is the reason the register is dated rather than additive. Apache Corp filed its own
    10-K/10-Q for 3.7 years after APA Corp became the parent because it retains registered
    public debt; admitting those stores a SUBSIDIARY's consolidated statements as the
    group's -- a fuller-looking history that is quietly wrong."""
    _patch(monkeypatch, {34088: [_filing("parent-2020", "2026-01-01"),
                                 _filing("sub-2022", "2026-09-01"),
                                 _filing("sub-2023", "2026-11-01")],
                         2115436: [_filing("suc", "2026-07-15")]})

    out = resolve_registrant_filings("XOM", ["10-K", "10-Q"], since=None,
                                     done_accessions=frozenset(), registrants=_XOM)
    kept = [f.accession_number for f in out]

    assert kept == ["parent-2020", "suc"]
    assert "sub-2022" not in kept and "sub-2023" not in kept
    print("\n=== SANITY CHECK: post-boundary subsidiary filings are excluded ===")
    print("  2 subsidiary 10-Ks dropped; a union would have blended two legal entities.")


def test_split_over_a_five_segment_chain_assigns_each_filing_once(monkeypatch):
    """Chains are the reason for the schema. PSKY is CBS -> Viacom -> ViacomCBS -> Paramount
    Global -> Paramount Skydance: four boundaries, five registrants, and the old two-CIK
    walk gave it one hop."""
    registrants = _chain(("0000000001", None, "2000-01-01"),
                         ("0000000002", "2000-01-01", "2006-01-01"),
                         ("0000000003", "2006-01-01", "2019-12-04"),
                         ("0000000004", "2019-12-04", "2025-08-07"),
                         ("0000000005", "2025-08-07", None))
    _patch(monkeypatch, {
        1: [_filing("s1", "1995-06-01"), _filing("s1-late", "2010-01-01")],
        2: [_filing("s2", "2003-06-01")],
        3: [_filing("s3", "2010-06-01")],
        4: [_filing("s4", "2021-06-01")],
        5: [_filing("s5", "2026-01-01")],
    })

    out = resolve_registrant_filings("T", ["10-K"], since=None,
                                     done_accessions=frozenset(), registrants=registrants)
    kept = [f.accession_number for f in out]

    assert kept == ["s1", "s2", "s3", "s4", "s5"]
    assert "s1-late" not in kept, "segment 1 kept a 2010 filing it does not own"
    assert len(kept) == len(set(kept))
    print("\n=== SANITY CHECK: a 5-segment chain, split ===")
    print(f"  {kept} -- one filing per segment, 0 duplicates, and the oldest registrant's")
    print("  2010 filing correctly belongs to segment 3. Validated.")


# --------------------------------------------------------------------------- #
# since / done_accessions                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("forms", [["8-K"], ["10-K"]])
def test_since_and_done_accessions_apply_under_both_policies(monkeypatch, forms):
    """Applied BEFORE the sort, so a routine incremental run orders a handful of new filings
    rather than a ticker's full multi-decade history. Asserted for BOTH policies because they
    are separate code paths and only one of them used to exist here."""
    _patch(monkeypatch, {"XOM": [_filing("old", "2020-01-01"), _filing("stored", "2026-05-01"),
                                 _filing("new", "2026-06-01")],
                         34088: [_filing("old", "2020-01-01"), _filing("stored", "2026-05-01"),
                                 _filing("new", "2026-06-01")],
                         2115436: []})

    out = resolve_registrant_filings("XOM", forms, since=pd.Timestamp("2026-01-01"),
                                     done_accessions=frozenset({"stored"}), registrants=_XOM)

    assert [f.accession_number for f in out] == ["new"]
    print(f"\n=== SANITY CHECK: since + done_accessions under {combine_for(forms).value} ===")
    print("  'old' before `since`, 'stored' already held -> only 'new'. Validated.")
