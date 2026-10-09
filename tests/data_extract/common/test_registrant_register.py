"""The registrant-boundary register: its schema and its refusals.

A ticker's filings follow the LEGAL registrant while its price history follows the ECONOMIC
entity, so `Company(ticker)` -- which resolves exactly one CIK -- loses everything the
predecessor filed, with no error and no gap signal. The register is what supplies the missing
CIKs, and its failure mode is QUIET DATA LOSS: a `valid_to` set a year early drops the
predecessor's last four filings and admits nothing in their place. Nothing downstream can tell
that from a company that genuinely filed nothing.

So every refusal the loader can make gets a named test here. The refusals ARE the safety
mechanism -- a wrong entry cannot raise on its own.

Split per wiki/guides/testing.md: synthetic fixtures for the schema (this file), real EDGAR for the
one invariant a config loader must never check itself because it needs a network call --
`test_registrant_live.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.common.registrant import (
    CUTOVER_KINDS,
    REGISTRANT_CONFIG_FILENAME,
    REGISTRANT_CONFIG_SUBDIR,
    RENAME_KIND,
    Registrant,
    Segment,
    load_registrants,
)

CONFIG_DIR = "./configs"
FRESH_START = "fresh_start"  # the register spelling of a same-CIK dated start


def _segment_for(reg: Registrant, date: object) -> Segment:
    """The one segment of `reg` covering `date`; fails the test when zero or several do."""
    covering = [segment for segment in reg.segments if segment.covers(date)]
    assert len(covering) == 1, f"{reg.ticker}: {len(covering)} segments cover {date}"
    return covering[0]


def _write(tmp_path: Path, blob: dict) -> str:
    """A throwaway config tree, so the loader's refusals are testable without touching the
    real register (`configs/` is a risk zone)."""
    root = tmp_path / REGISTRANT_CONFIG_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    (root / REGISTRANT_CONFIG_FILENAME).write_text(json.dumps(blob), encoding="utf-8")
    return str(tmp_path)


def _entry(*segments, kind: str = "reorganisation") -> dict:
    return {"kind": kind, "segments": list(segments)}


def _seg(cik: str, *, valid_from: str | None = None, valid_to: str | None = None, evidence: str = "measured") -> dict:
    seg: dict = {"cik": cik, "evidence": evidence}
    if valid_from is not None:
        seg["valid_from"] = valid_from
    if valid_to is not None:
        seg["valid_to"] = valid_to
    return seg


# --------------------------------------------------------------------------- #
# The loader's refusals                                                        #
# --------------------------------------------------------------------------- #
def test_a_rename_is_rejected_by_name(tmp_path):
    """CVS Caremark -> CVS Health and Facebook -> Meta keep their CIK. An entry for either
    would walk ONE CIK twice and duplicate every filing, so the loader names the mistake
    rather than merely failing an enum check -- the error message is the documentation the
    next person will actually read.

    It matters more here than it looks: Sharadar records a `namechangefrom` whether or not
    the CIK moved, so the shell-name evidence that motivates most real entries fits a pure
    rename just as well."""
    blob = {"CVS": _entry(_seg("0000064803", valid_to="2014-09-03"), _seg("0000064803", valid_from="2014-09-03"), kind=RENAME_KIND)}
    with pytest.raises(ValueError, match="not a cutover"):
        load_registrants(_write(tmp_path, blob))
    print("\n=== SANITY CHECK: a rename cannot be encoded as a cutover ===")
    print(f"  kind={RENAME_KIND!r} rejected; permitted kinds are {sorted(CUTOVER_KINDS)}")
    print("  OK: one CIK cannot be walked twice.")


def test_the_same_cik_twice_is_rejected(tmp_path):
    """The structural form of the same mistake, for an entry that lies about its `kind`.
    '64803' and '0000064803' are the SAME CIK once padded, and the padding happens before
    the comparison rather than after."""
    blob = {"X": _entry(_seg("64803", valid_to="2020-01-01"), _seg("0000064803", valid_from="2020-01-01"))}
    with pytest.raises(ValueError, match="repeated CIK"):
        load_registrants(_write(tmp_path, blob))
    print("\n=== SANITY CHECK: the same CIK on both sides is refused ===")
    print("  '64803' and '0000064803' compare equal once zero-padded. Validated.")


def test_a_gap_between_segments_is_rejected(tmp_path):
    """A gap loses every filing inside it, and nothing downstream raises: the ticker simply
    has a hole where a year of filings should be."""
    blob = {"X": _entry(_seg("1", valid_to="2020-01-01"), _seg("2", valid_from="2020-06-01"))}
    with pytest.raises(ValueError, match="CONTIGUOUS"):
        load_registrants(_write(tmp_path, blob))
    print("\n=== SANITY CHECK: a gap between segments is refused ===")
    print("  2020-01-01 -> 2020-06-01 leaves 5 months owned by nobody. Validated.")


def test_an_overlap_between_segments_is_rejected(tmp_path):
    """An overlap double-counts, which on a consolidating form means two legal entities'
    accounts in one series."""
    blob = {"X": _entry(_seg("1", valid_to="2020-06-01"), _seg("2", valid_from="2020-01-01"))}
    with pytest.raises(ValueError, match="CONTIGUOUS"):
        load_registrants(_write(tmp_path, blob))
    print("\n=== SANITY CHECK: an overlap between segments is refused ===")
    print("  5 months claimed by both registrants. Validated.")


def test_a_segment_with_no_evidence_is_rejected(tmp_path):
    """An undocumented cutover is a guess that deletes history -- exactly what this register
    exists to replace. Enforced PER SEGMENT, not per entry: a chain's middle hop is the one a
    hurried edit will leave blank."""
    blob = {"X": _entry(_seg("1", valid_to="2020-01-01"), _seg("2", valid_from="2020-01-01", evidence="   "))}
    with pytest.raises(ValueError, match="empty `evidence`"):
        load_registrants(_write(tmp_path, blob))
    print("\n=== SANITY CHECK: an undocumented segment is refused ===")
    print("  OK: evidence is mandatory on every segment, not just the entry.")


def test_a_single_segment_is_rejected(tmp_path):
    """One segment is not a boundary. An entry with one is a no-op that reads as a fix."""
    blob = {"X": _entry(_seg("1"))}
    with pytest.raises(ValueError, match="at least 2"):
        load_registrants(_write(tmp_path, blob))
    print("\n=== SANITY CHECK: a one-segment entry is refused ===")
    print("  OK: an entry that changes nothing cannot masquerade as a repair.")


def test_the_open_ends_must_stay_open(tmp_path):
    """The oldest segment omits `valid_from` and the newest omits `valid_to`, so every date
    in history lands in exactly one segment and `_segment_for` is total. Closing an end would
    make some dates belong to no registrant at all."""
    closed_old = {"X": _entry(_seg("1", valid_from="1990-01-01", valid_to="2020-01-01"), _seg("2", valid_from="2020-01-01"))}
    with pytest.raises(ValueError, match="must omit `valid_from`"):
        load_registrants(_write(tmp_path, closed_old))
    print("\n=== SANITY CHECK: the chain stays open at both ends ===")
    print("  a `valid_from` on the oldest segment is refused. Validated.")


def test_one_cik_cannot_be_claimed_by_two_tickers(tmp_path):
    """Cross-entry, because it cannot be caught inside a single one.

    Tier C resolves bulk-dataset rows to a ticker through a CIK->ticker dict built from these
    segments. Two tickers claiming one CIK would route one company's insider transactions and
    financial notes to BOTH names -- a duplication that no per-entry check can see."""
    blob = {
        "A": _entry(_seg("1", valid_to="2020-01-01"), _seg("9", valid_from="2020-01-01")),
        "B": _entry(_seg("9", valid_to="2021-01-01"), _seg("3", valid_from="2021-01-01")),
    }
    with pytest.raises(ValueError, match="claimed by both"):
        load_registrants(_write(tmp_path, blob))
    print("\n=== SANITY CHECK: a CIK belongs to one ticker ===")
    print("  CIK 0000000009 claimed by A and B -> refused. Validated.")


# --------------------------------------------------------------------------- #
# The `fresh_start` kind: one CIK that counts from a date                      #
# --------------------------------------------------------------------------- #
def test_a_fresh_start_entry_is_one_dated_segment(tmp_path):
    """A same-CIK fresh start (EXE emerging from bankruptcy) keeps its CIK, so it is one segment
    open at the new end and closed at the old: no seam, and dates before `valid_from` belong to no one."""
    blob = {"X": _entry(_seg("895126", valid_from="2021-02-10"), kind=FRESH_START)}
    reg = load_registrants(_write(tmp_path, blob))["X"]
    only = reg.segments[0]
    assert reg.kind == FRESH_START
    assert len(reg.segments) == 1
    assert only.cik == "0000895126"
    assert only.valid_from == pd.Timestamp("2021-02-10")
    assert only.valid_to is None
    assert reg.boundaries == ()
    assert not only.covers("2021-02-09") and only.covers("2021-02-10")
    print("\n=== SANITY CHECK: a fresh_start entry loads as one dated segment ===")
    print(f"  {reg.ticker}: {only.cik} from {only.valid_from.date()} to open, boundaries {reg.boundaries}")
    print("  OK: 2021-02-09 is outside the window, 2021-02-10 inside.")


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (_entry(_seg("1", valid_to="2021-02-10"), _seg("2", valid_from="2021-02-10"), kind=FRESH_START), "exactly 1"),
        (_entry(_seg("1"), kind=FRESH_START), "requires `valid_from`"),
        (_entry(_seg("1", valid_from="2021-02-10", valid_to="2024-01-01"), kind=FRESH_START), "must omit `valid_to`"),
        (_entry(_seg("1", valid_from="2021-02-10", evidence="  "), kind=FRESH_START), "empty `evidence`"),
        (_entry(_seg("1", valid_from="not-a-date"), kind=FRESH_START), "unparseable date|Unknown datetime string"),
    ],
    ids=["two_segments", "no_valid_from", "valid_to_present", "empty_evidence", "bad_date"],
)
def test_a_malformed_fresh_start_is_rejected(tmp_path, entry, match):
    """Every way a fresh start can be mis-written raises at load: it must be one evidenced segment with a parseable start."""
    with pytest.raises(ValueError, match=match) as raised:
        load_registrants(_write(tmp_path, {"X": entry}))
    print("\n=== SANITY CHECK: a malformed fresh_start is refused ===")
    print(f"  {str(raised.value)[:110]}")
    print("  OK: the loader names the rule broken.")


def test_a_fresh_start_cik_claimed_by_another_entry_is_rejected(tmp_path):
    """The cross-entry CIK uniqueness rule applies to a fresh start like any cutover chain."""
    blob = {
        "A": _entry(_seg("9", valid_from="2021-02-10"), kind=FRESH_START),
        "B": _entry(_seg("9", valid_to="2020-01-01"), _seg("3", valid_from="2020-01-01")),
    }
    with pytest.raises(ValueError, match="claimed by both"):
        load_registrants(_write(tmp_path, blob))
    print("\n=== SANITY CHECK: a fresh_start CIK still belongs to one ticker ===")
    print("  CIK 0000000009 claimed by fresh_start A and cutover B -> refused. Validated.")


# --------------------------------------------------------------------------- #
# The schema's behaviour                                                       #
# --------------------------------------------------------------------------- #
def test_the_boundary_is_strictly_before_and_on_or_after(tmp_path):
    """`_segment_for` is the whole contract in one line: the boundary date belongs to the
    SUCCESSOR. That is what makes two adjacent segments disjoint by construction rather than
    by de-duplication, and it is the same convention the dated split has always used."""
    blob = {"X": _entry(_seg("1", valid_to="2020-01-01"), _seg("2", valid_from="2020-01-01"))}
    reg = load_registrants(_write(tmp_path, blob))["X"]
    boundary = pd.Timestamp("2020-01-01")
    assert _segment_for(reg, boundary - pd.Timedelta(days=1)).cik == "0000000001"
    assert _segment_for(reg, boundary).cik == "0000000002"
    assert _segment_for(reg, boundary - pd.Timedelta(seconds=1)).cik == "0000000001"
    print("\n=== SANITY CHECK: the boundary date belongs to the successor ===")
    print(f"  2019-12-31 -> {_segment_for(reg, boundary - pd.Timedelta(days=1)).cik}")
    print(f"  2020-01-01 -> {_segment_for(reg, boundary).cik}")
    print("  OK: strictly-before / on-or-after, to the second.")


def test_a_five_segment_chain_round_trips(tmp_path):
    """Chains are the reason for the schema change. PSKY is CBS -> Viacom -> ViacomCBS ->
    Paramount Global -> Paramount Skydance: four boundaries, five registrants. The two-CIK
    schema gave such a ticker one hop and left the rest truncated."""
    blob = {
        "CHAIN": _entry(
            _seg("1", valid_to="2000-01-01"),
            _seg("2", valid_from="2000-01-01", valid_to="2006-01-01"),
            _seg("3", valid_from="2006-01-01", valid_to="2019-12-04"),
            _seg("4", valid_from="2019-12-04", valid_to="2025-08-07"),
            _seg("5", valid_from="2025-08-07"),
        )
    }
    reg = load_registrants(_write(tmp_path, blob))["CHAIN"]
    assert reg.all_ciks() == ("0000000001", "0000000002", "0000000003", "0000000004", "0000000005")
    assert len(reg.boundaries) == 4
    assert _segment_for(reg, "1995-06-01").cik == "0000000001"
    assert _segment_for(reg, "2010-06-01").cik == "0000000003"
    assert _segment_for(reg, "2026-01-01").cik == "0000000005"
    print("\n=== SANITY CHECK: a 5-segment chain ===")
    print(f"  ciks {reg.all_ciks()}")
    print(f"  boundaries {[str(b.date()) for b in reg.boundaries]}")
    print("  OK: every date resolves to exactly one of the five.")


def test_every_date_lands_in_exactly_one_segment(tmp_path):
    """`_segment_for` is total AND unambiguous -- the property contiguity plus open ends buys,
    asserted directly rather than inferred from the two rules that produce it."""
    blob = {
        "CHAIN": _entry(
            _seg("1", valid_to="2000-01-01"), _seg("2", valid_from="2000-01-01", valid_to="2010-01-01"), _seg("3", valid_from="2010-01-01")
        )
    }
    reg = load_registrants(_write(tmp_path, blob))["CHAIN"]
    dates = pd.date_range("1990-01-01", "2030-01-01", freq="37D")
    hits = [[s.cik for s in reg.segments if s.covers(d)] for d in dates]
    assert all(len(h) == 1 for h in hits), "a date matched 0 or 2 segments"
    print("\n=== SANITY CHECK: coverage is a partition ===")
    print(f"  {len(dates)} dates from 1990 to 2030, each matched by exactly 1 of 3 segments.")


# --------------------------------------------------------------------------- #
# The live register                                                            #
# --------------------------------------------------------------------------- #
def test_the_live_register_declares_only_kinds_that_change_the_cik():
    """A standing assertion over the real file, so a future entry cannot slip a rename
    through by spelling it something else."""
    registrants = load_registrants(CONFIG_DIR)
    assert registrants, "the register is empty"
    print("\n=== SANITY CHECK: the live register ===")
    for ticker, reg in sorted(registrants.items()):
        assert reg.kind in CUTOVER_KINDS
        assert len(set(reg.all_ciks())) == len(reg.segments)
        assert all(s.evidence.strip() for s in reg.segments)
        print(
            f"  {ticker:6s} {reg.kind:15s} {len(reg.segments)} segments  "
            f"{' -> '.join(reg.all_ciks())}  at "
            f"{', '.join(str(b.date()) for b in reg.boundaries)}"
        )
    n_chains = sum(1 for r in registrants.values() if len(r.segments) > 2)
    print(f"  OK: {len(registrants)} entries, {n_chains} of them chains, every kind one that changes the CIK.")


def test_the_missing_register_file_is_not_an_error(tmp_path):
    """`{}` when absent -- the common case for a fresh checkout or a test config tree, and
    not a condition that should stop a nightly run."""
    assert load_registrants(str(tmp_path)) == {}
    print("\n=== SANITY CHECK: no register file -> no registrants ===")
    print("  OK: an absent file is empty, not an exception.")


def test_the_approved_governance_cutovers_are_in_the_live_register():
    """The accepted governance repairs stay dated, explicit, and reviewable; JCI's left with the traded-security view."""
    registrants = load_registrants(CONFIG_DIR)
    expected = {
        "EVRG": (("0000054507", "0001711269"), "2018-06-04"),
        "PSKY": (("0000813828", "0002041610"), "2025-08-07"),
    }

    for ticker, (ciks, boundary) in expected.items():
        assert registrants[ticker].all_ciks() == ciks
        assert [str(value.date()) for value in registrants[ticker].boundaries] == [boundary]

    print("\n=== SANITY CHECK: approved governance registrant chains ===")
    for ticker, (ciks, boundary) in expected.items():
        print(f"  {ticker}: {' -> '.join(ciks)} at {boundary}")
    print("  OK: the evidence-backed chains are exact and dated.")


def test_the_register_follows_the_traded_security():
    """AC-001: PLD, JCI and DOW have no entry (AMB, Tyco and Dow Inc own their tickers from the sentinel start); DD's
    predecessor is TDCC, whose prices DD carries before 2017-08-31; MRK and TPL keep their revision-4 chains."""
    registrants = load_registrants(CONFIG_DIR)
    raw = json.loads((Path(CONFIG_DIR) / "sec" / "registrant_cutover.json").read_text(encoding="utf-8"))
    kept = {
        "MRK": (("0000064978", "0000310158"), "2009-11-03", "reverse_merger_accounting_predecessor"),
        "TPL": (("0000097517", "0001811074"), "2021-01-11", None),
        "DD": (("0000029915", "0001666700"), "2017-08-31", None),
    }
    absent = ("PLD", "JCI", "DOW")
    by_cik = {cik: ticker for ticker, reg in registrants.items() for cik in reg.all_ciks()}

    print("\n=== SANITY CHECK: register follows the traded security ===")
    for ticker, (ciks, boundary, basis) in kept.items():
        print(f"  {ticker}: {' -> '.join(registrants[ticker].all_ciks())} at {[str(b.date()) for b in registrants[ticker].boundaries]}")
        assert registrants[ticker].all_ciks() == ciks, ticker
        assert [str(value.date()) for value in registrants[ticker].boundaries] == [boundary], ticker
        assert raw[ticker].get("basis") == basis, ticker
    print(
        f"  absent: {[t for t in absent if t not in registrants]}; old ProLogis/old JCI/old DuPont in the register: {[c for c in ('0000899881', '0000053669', '0000030554') if c in by_cik]}"
    )
    assert not set(absent) & set(registrants)
    assert not {"0000899881", "0000053669", "0000030554", "0001045609", "0000833444", "0001751788"} & set(by_cik)
    assert "0001193125-17-274834" in registrants["DD"].segments[0].evidence
    print("  OK: PLD/JCI/DOW absent, DD = TDCC -> DowDuPont at 2017-08-31 citing 8-K12B 0001193125-17-274834.")
