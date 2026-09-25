"""
`identity.py` -- the two-axis resolver, its five load-time raises and the live acceptance
table of every (ticker, issuer CIK) group the 2.3 screen flagged.

The raises are the point of the synthetic half. One of them,
`TwoUniverseTickersOneEntityError`, is the only failure in this design that RELABELS a company's
rows onto another company instead of dropping them, so it is asserted before any other map is
trusted and it is tested on the `DD`/`DOW` shape that makes it a live risk rather than a
theoretical one.

The live half is the acceptance table: all 102 groups, each classified KEEP (a genuine
predecessor whose rows stay), MOVE (rows that belong to a DIFFERENT universe ticker) or DROP
(another company entirely). Every verdict was read by name against the issuer name in the
filing -- Weight Watchers is not Willis Towers Watson, Monster Worldwide is not Monster
Beverage -- so this file pins a reviewed judgement and not merely the code's own output.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.data_extract.utils.common.entity_lineage import TwoUniverseTickersOneEntityError
from src.data_extract.utils.common.identity import (
    AmbiguousSymbolTenureError,
    CikInTwoEntitiesError,
    Identity,
    IdentityError,
    UniverseEntityDisagreementError,
    UnknownUniverseTickerError,
    build_identity,
    load_identity,
    normalise_cik,
    resolve_symbol_rows,
)

CONFIG_DIR = "./configs"


# --------------------------------------------------------------------------- #
# fixtures: synthetic frames with a known truth                                 #
# --------------------------------------------------------------------------- #


def _lineage(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"cik": c, "entity_id": e, "source": s, "confidence": None, "evidence": "test"} for c, e, s in rows])


def _tenure(rows: list[tuple[str, str, object, object, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": s, "issuer_cik": c, "valid_from": f, "valid_to": t, "n_filings": n, "source": "form345", "evidence": ""}
            for s, c, f, t, n in rows
        ]
    )


def _roster(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "cik": c} for t, c in rows])


def _simple() -> Identity:
    """One ticker `AAA` whose entity holds a predecessor, plus an unrelated reuse."""
    return build_identity(
        lineage=_lineage(
            [("0000000100", "E0000000100", "register"), ("0000000200", "E0000000100", "register"), ("0000000900", "E0000000900", "roster")]
        ),
        tenure=_tenure(
            [
                ("AAA", "0000000100", pd.Timestamp("2006-01-01"), pd.Timestamp("2015-01-01"), 40),
                ("AAA", "0000000200", pd.Timestamp("2015-01-01"), None, 900),
                ("BBB", "0000000900", pd.Timestamp("2006-01-01"), None, 50),
            ]
        ),
        roster=_roster([("AAA", "0000000200"), ("BBB", "0000000900")]),
    )


# --------------------------------------------------------------------------- #
# axis A                                                                        #
# --------------------------------------------------------------------------- #


def test_an_unseen_cik_is_its_own_entity_and_does_not_raise():
    """Absence in `entity_lineage` is a VERDICT -- the table stores only non-singleton groups
    and the roster, so a CIK with no row is a company of its own and `owns()` needs exactly
    that answer from it."""
    identity = _simple()
    assert identity.entity_of("0000999999") == "E0000999999"
    assert identity.entity_ticker("0000999999") is None
    # three spellings of one CIK, one entity
    assert identity.entity_of("100") == identity.entity_of("0000000100") == identity.entity_of("0000000100.0") == "E0000000100"

    print("\n=== SANITY CHECK: an unknown CIK ===")
    print(f"  entity_of('0000999999') -> {identity.entity_of('0000999999')} (singleton, no raise)")
    print(f"  '100' / '0000000100' / '0000000100.0' -> {identity.entity_of('100')}")
    print("  OK: silence in the lineage table is an answer, and CIK spelling is normalised")
    print("  -> A 4.3M-row parse cannot abort on a CIK nobody has ever adjudicated.")


def test_entity_ticker_resolves_the_predecessor_and_returns_none_off_universe():
    identity = _simple()
    assert identity.entity_ticker("0000000100") == "AAA"  # predecessor -> today's ticker
    assert identity.entity_ticker("0000000200") == "AAA"
    assert identity.entity_ticker("0000000900") == "BBB"
    assert identity.entity_ticker("0000000042") is None
    assert identity.owns("AAA", "0000000100") is True
    assert identity.owns("BBB", "0000000100") is False

    print("\n=== SANITY CHECK: CIK-first resolution ===")
    print("  predecessor 0000000100 -> AAA; unrelated 0000000042 -> None")
    print("  OK: the row's ticker comes from its OWN CIK, not from the symbol it typed")
    print("  -> This is what admits a predecessor that changed both CIK and symbol.")


def test_owns_raises_for_a_ticker_outside_the_universe():
    identity = _simple()
    with pytest.raises(UnknownUniverseTickerError, match="ZZZ"):
        identity.universe_entity("ZZZ")
    with pytest.raises(UnknownUniverseTickerError):
        identity.owns("ZZZ", "0000000100")

    print("\n=== SANITY CHECK: unknown universe ticker ===")
    print("  universe_entity('ZZZ') raises UnknownUniverseTickerError and names ZZZ")
    print("  OK: 'outside the universe' and 'roster row is broken' stay distinguishable")
    print("  -> That conflation is the known weakness of the cik_to_ticker map.")


def test_two_universe_tickers_in_one_entity_raises_and_names_both():
    """The `DD`/`DOW` shape: two index members descended from DowDuPont, merged by an oracle
    that keyed on the directors they shared through 2017-2019."""
    with pytest.raises(TwoUniverseTickersOneEntityError) as excinfo:
        build_identity(
            lineage=_lineage(
                [
                    ("0001666700", "E0000029915", "owner_overlap"),
                    ("0001751788", "E0000029915", "owner_overlap"),
                    ("0000029915", "E0000029915", "owner_overlap"),
                ]
            ),
            tenure=_tenure([("DD", "0001666700", pd.Timestamp("2019-06-01"), None, 10), ("DOW", "0001751788", pd.Timestamp("2019-04-01"), None, 10)]),
            roster=_roster([("DD", "0001666700"), ("DOW", "0001751788")]),
        )
    message = str(excinfo.value)
    for token in ("DD", "DOW", "0001666700", "0001751788", "owner_overlap"):
        assert token in message

    print("\n=== SANITY CHECK: the one raise that would CORRUPT ===")
    print(f"  {message[:150]}...")
    print("  OK: both tickers, both roster CIKs and the joining source are all named")
    print("  -> Without it a dict would overwrite one ticker and relabel every row it owns.")


def test_a_cik_in_two_entities_raises():
    """Impossible through the table's primary key, so this guards the BUILDER."""
    with pytest.raises(CikInTwoEntitiesError, match="0000000100"):
        build_identity(
            lineage=_lineage([("0000000100", "E0000000100", "register"), ("0000000100", "E0000000200", "owner_overlap")]),
            tenure=_tenure([("AAA", "0000000100", pd.Timestamp("2006-01-01"), None, 5)]),
            roster=_roster([("AAA", "0000000100")]),
        )

    print("\n=== SANITY CHECK: one CIK, one entity ===")
    print("  a duplicated cik with two entity_ids raises CikInTwoEntitiesError naming the CIK")
    print("  OK: the belt-and-braces twin of registrant._check_ciks_unique_across_entries")
    print("  -> It cannot come from Postgres, so if it fires the builder is wrong.")


# --------------------------------------------------------------------------- #
# axis B                                                                        #
# --------------------------------------------------------------------------- #


def test_tenure_is_half_open_on_both_boundaries():
    """`valid_from <= d < valid_to`, exactly `registrant.Segment.covers`, so two adjacent
    tenures are disjoint by construction rather than by de-duplication.

    Two DIFFERENT entities either side of the seam, because that is the only arrangement in
    which an off-by-one day is visible at all -- with one entity both answers agree and the
    boundary could be wrong without any test noticing.
    """
    identity = build_identity(
        lineage=_lineage([("0000000100", "E0000000100", "roster"), ("0000000700", "E0000000700", "roster")]),
        tenure=_tenure(
            [
                ("AAA", "0000000100", pd.Timestamp("2006-01-01"), pd.Timestamp("2015-01-01"), 40),
                ("AAA", "0000000700", pd.Timestamp("2015-01-01"), None, 900),
            ]
        ),
        roster=_roster([("AAA", "0000000700")]),
    )
    assert identity.entity_for("AAA", "2005-12-31") is None  # before valid_from
    assert identity.entity_for("AAA", "2006-01-01") == "E0000000100"  # valid_from INCLUDED
    assert identity.entity_for("AAA", "2014-12-31") == "E0000000100"
    assert identity.entity_for("AAA", "2015-01-01") == "E0000000700"  # valid_to EXCLUDED
    assert identity.entity_for("AAA", "2026-01-01") == "E0000000700"  # open tenure

    print("\n=== SANITY CHECK: half-open tenure boundaries ===")
    print("  2005-12-31 -> None | 2006-01-01 -> predecessor | 2014-12-31 -> predecessor")
    print("  2015-01-01 -> successor (valid_to is excluded) | open end -> successor")
    print("  OK: the successor's valid_from is the predecessor's excluded valid_to")
    print("  -> Same convention as the register, so the two layers cannot disagree by a day.")


def test_postgres_date_round_trip_compares_both_directions():
    """⚠ Postgres `DATE` columns come back as `datetime.date`, not `Timestamp`, and
    `date < Timestamp` raises. A parquet-cached fixture hides this entire bug class."""
    identity = build_identity(
        lineage=_lineage([("0000000100", "E0000000100", "roster")]),
        tenure=_tenure([("AAA", "0000000100", dt.date(2006, 1, 1), dt.date(2015, 1, 1), 5), ("AAA", "0000000200", dt.date(2015, 1, 1), None, 5)]),
        roster=_roster([("AAA", "0000000100")]),
        # two DIFFERENT entities either side of the seam, so an off-by-one day is visible --
        # which necessarily makes the roster CIK disagree with the incumbent, hence the entry
        d19_allowlist={"AAA": "synthetic: the seam is the point of this fixture"},
    )
    assert identity.entity_for("AAA", dt.date(2010, 1, 1)) == "E0000000100"
    assert identity.entity_for("AAA", pd.Timestamp("2010-01-01")) == "E0000000100"
    assert identity.entity_for("AAA", dt.date(2020, 1, 1)) == "E0000000200"
    assert identity.entity_for("AAA", "2020-01-01") == "E0000000200"

    print("\n=== SANITY CHECK: DATE vs Timestamp ===")
    print("  datetime.date boundaries queried with date, Timestamp and str all agree")
    print("  OK: the round-trip trap cannot reach the interval test")
    print("  -> Both sides go through one normaliser instead of comparing raw types.")


def test_a_dateless_lookup_raises_when_the_symbol_changed_hands():
    """⚠ Two CIKs are NOT two holders. `_simple()`'s `AAA` passed through two registrants of
    ONE entity, so a dateless lookup there has exactly one answer and must not raise; the
    raise is for a symbol that genuinely changed COMPANY."""
    one_company = _simple()
    assert one_company.entity_for("AAA", None) == "E0000000100"
    assert one_company.entity_for("BBB", None) == "E0000000900"  # only one holder, ever
    assert one_company.entity_for("ZZZ", None) is None  # nobody, ever

    reused = build_identity(
        lineage=_lineage([("0000000100", "E0000000100", "roster")]),
        tenure=_tenure(
            [
                ("AAA", "0000000100", pd.Timestamp("2006-01-01"), None, 900),
                ("AAA", "0000000700", pd.Timestamp("2006-01-01"), pd.Timestamp("2009-01-01"), 20),
            ]
        ),
        roster=_roster([("AAA", "0000000100")]),
    )
    with pytest.raises(AmbiguousSymbolTenureError, match="AAA"):
        reused.entity_for("AAA", None)

    print("\n=== SANITY CHECK: no silent 'today' ===")
    print("  one entity through two registrants -> resolves; the caller needs no date")
    print("  two entities over the symbol's life -> raises, the caller must say WHEN")
    print("  OK: Zipline's lookup_symbol contract, not last-writer-wins")
    print("  -> A row filed in 2007 would otherwise be handed to the 2026 incumbent.")


def test_ambiguity_is_entity_grain_not_cik_grain():
    """993 adjacent tenure pairs overlap in time; most are two CIKs of ONE entity filing
    under both registrants through a reorganisation. Collapsing to entities first is what
    keeps this raise rare enough to mean something."""
    one_entity = build_identity(
        lineage=_lineage([("0000000100", "E0000000100", "register"), ("0000000200", "E0000000100", "register")]),
        tenure=_tenure([("AAA", "0000000100", pd.Timestamp("2006-01-01"), None, 10), ("AAA", "0000000200", pd.Timestamp("2010-01-01"), None, 10)]),
        roster=_roster([("AAA", "0000000200")]),
    )
    assert one_entity.entity_for("AAA", "2012-01-01") == "E0000000100"  # overlap, no raise

    two_entities = build_identity(
        lineage=_lineage([("0000000100", "E0000000100", "roster")]),
        tenure=_tenure([("AAA", "0000000100", pd.Timestamp("2006-01-01"), None, 10), ("AAA", "0000000700", pd.Timestamp("2010-01-01"), None, 10)]),
        roster=_roster([("AAA", "0000000100")]),
        d19_allowlist={"AAA": "test"},
    )
    with pytest.raises(AmbiguousSymbolTenureError):
        two_entities.entity_for("AAA", "2012-01-01")

    print("\n=== SANITY CHECK: overlap grain ===")
    print("  two CIKs of ONE entity overlapping -> resolves, no raise")
    print("  two DIFFERENT entities overlapping -> AmbiguousSymbolTenureError")
    print("  OK: paperwork overlap is not identity ambiguity")
    print("  -> Measured live: 2,432 symbols ambiguous over all history, 10 at today.")


def test_dominant_entity_is_not_the_latest_observation():
    """A filer's single mistyped ISSUERTRADINGSYMBOL opens a tenure that is both CURRENT and
    WRONG. `COO` and `SPG` each carry one against a thousand real filings."""
    identity = build_identity(
        lineage=_lineage([("0000000100", "E0000000100", "roster")]),
        tenure=_tenure([("AAA", "0000000100", pd.Timestamp("2006-01-01"), None, 1000), ("AAA", "0000000700", pd.Timestamp("2026-01-01"), None, 1)]),
        roster=_roster([("AAA", "0000000100")]),
        d19_allowlist={"AAA": "test"},
    )
    assert identity.dominant_entity("AAA") == "E0000000100"

    print("\n=== SANITY CHECK: whose symbol is this NOW ===")
    print("  1,000 filings (2006-) vs a 1-filing typo opened 2026 -> the 1,000 wins")
    print("  OK: weight of filings, not recency, decides the D19 comparison")
    print("  -> A latest-observation rule hands COO and SPG to a typo.")


def test_symbol_ticker_resolution_covers_rename_reuse_gap_and_universe_scope():
    """The CIK-less source contract: resolve by entity/date, then enforce caller scope."""
    identity = build_identity(
        lineage=_lineage(
            [
                ("0000000100", "E_META", "roster"),
                ("0000000200", "E_FISV", "roster"),
                ("0000000300", "E_TT", "roster"),
                ("0000000400", "E_IR", "roster"),
                ("0000000500", "E_WTW", "roster"),
            ]
        ),
        tenure=_tenure(
            [
                ("FB", "0000000100", "2012-01-01", "2022-06-01", 500),
                ("META", "0000000100", "2022-06-01", None, 500),
                ("FI", "0000000200", "2019-01-01", "2023-06-01", 200),
                ("FISV", "0000000200", "2023-06-01", None, 200),
                ("IR", "0000000300", "2009-01-01", "2020-03-01", 200),
                ("IR", "0000000400", "2020-03-05", None, 200),
                ("TT", "0000000300", "2020-03-01", None, 200),
                ("WTW", "0000000900", "2009-01-01", "2019-04-18", 100),
                ("WTW", "0000000500", "2019-04-18", None, 500),
                ("OVER", "0000000700", "2010-01-01", "2015-01-01", 10),
                ("OVER", "0000000800", "2012-01-01", "2016-01-01", 10),
            ]
        ),
        roster=_roster(
            [
                ("META", "0000000100"),
                ("FISV", "0000000200"),
                ("TT", "0000000300"),
                ("IR", "0000000400"),
                ("WTW", "0000000500"),
            ]
        ),
    )
    universe = frozenset({"META", "FISV", "TT", "IR", "WTW"})

    fb = identity.resolve_symbol_ticker("FB", "2008-01-01", universe)
    fi = identity.resolve_symbol_ticker("FI", "2020-01-01", universe)
    old_ir = identity.resolve_symbol_ticker("IR", "2015-01-01", universe)
    current_ir = identity.resolve_symbol_ticker("IR", "2025-01-01", universe)
    early_wtw = identity.resolve_symbol_ticker("WTW", "2015-01-01", universe)
    overlap = identity.resolve_symbol_ticker("OVER", "2013-01-01", universe)
    gap = identity.resolve_symbol_ticker("IR", "2020-03-03", universe)
    unknown = identity.resolve_symbol_ticker("NEVER", "2020-01-01", universe)
    outside = identity.resolve_symbol_ticker("FI", "2020-01-01", frozenset({"META"}))

    assert fb.ticker is None and fb.verdict == "unknown_gap" and fb.match_kind is None
    assert (fi.ticker, fi.verdict, fi.match_kind) == ("FISV", "mapped_current_ticker", "exact_dated_tenure")
    assert old_ir.ticker == "TT" and old_ir.verdict == "mapped_current_ticker"
    assert current_ir.ticker == "IR" and current_ir.verdict == "exact_dated_tenure"
    assert early_wtw.verdict == "entity_not_in_universe" and not early_wtw.accepted
    assert overlap.verdict == "ambiguous" and gap.verdict == "unknown_gap"
    assert unknown.verdict == "unknown_symbol"
    assert outside.verdict == "entity_not_in_universe" and outside.ticker is None
    assert {"FB", "FI", "FISV", "META", "IR", "TT"} <= identity.candidate_symbols(universe)

    print("\n=== SANITY CHECK: symbol/date -> canonical ticker ===")
    print("  FB before verified tenure -> unknown_gap; FI -> FISV inside verified tenure")
    print("  historical IR -> TT, current IR -> IR; early WTW is outside the universe")
    print("  overlap -> ambiguous; reuse seam gap -> unknown_gap; absent -> unknown_symbol")
    print("  OK: only the caller's universe is returned, and no ambiguity is guessed")


def test_active_manual_tenure_overrides_conflicting_derived_evidence():
    tenure = _tenure(
        [
            ("COO", "0000000100", "2006-01-05", None, 0),
            ("COO", "0000000700", "2024-03-11", None, 1),
            ("TPL", "0000000200", "2007-03-01", "2021-01-11", 0),
            ("TPL", "0000000300", "2021-01-11", None, 0),
        ]
    )
    tenure.loc[[0, 2, 3], "source"] = "manual"
    identity = build_identity(
        lineage=_lineage(
            [
                ("0000000100", "E_COO", "roster"),
                ("0000000200", "E_TPL", "register"),
                ("0000000300", "E_TPL", "roster"),
                ("0000000700", "E_OTHER", "roster"),
            ]
        ),
        tenure=tenure,
        roster=_roster([("COO", "0000000100"), ("TPL", "0000000300")]),
    )

    coo = identity.resolve_symbol_ticker("COO", "2025-01-01", frozenset({"COO", "TPL"}))
    tpl_before = identity.resolve_symbol_ticker("TPL", "2021-01-10", frozenset({"COO", "TPL"}))
    tpl_after = identity.resolve_symbol_ticker("TPL", "2021-01-11", frozenset({"COO", "TPL"}))
    assert (coo.ticker, coo.match_kind) == ("COO", "exact_dated_tenure")
    assert tpl_before.ticker == tpl_after.ticker == "TPL"
    assert tpl_before.entity_id == tpl_after.entity_id == "E_TPL"

    print("\n=== SANITY CHECK: manual source precedence ===")
    print("  COO manual entity wins over a one-filing active derived conflict")
    print("  TPL old/new CIK seam resolves once to the same economic entity on both sides")
    print("  OK: precedence is dated and deterministic; no current-ticker fallback is used")


def test_closed_manual_predecessor_does_not_own_a_reused_symbol_today():
    lineage = _lineage(
        [
            ("0000000100", "E0000000100", "manual"),
            ("0000000200", "E0000000200", "roster"),
        ]
    )
    tenure = pd.DataFrame(
        [
            {
                "symbol": "IR",
                "issuer_cik": "0000000100",
                "valid_from": "2009-07-09",
                "valid_to": "2020-03-02",
                "n_filings": 1000,
                "source": "manual",
            },
            {
                "symbol": "IR",
                "issuer_cik": "0000000200",
                "valid_from": "2020-03-03",
                "valid_to": None,
                "n_filings": 400,
                "source": "derived",
            },
        ]
    )
    identity = build_identity(
        lineage,
        tenure,
        _roster([("IR", "0000000200")]),
    )

    assert identity.dominant_entity("IR") == "E0000000200"
    assert (
        identity.resolve_symbol_ticker(
            "IR",
            "2019-12-31",
            frozenset({"IR"}),
        ).ticker
        is None
    )
    assert (
        identity.resolve_symbol_ticker(
            "IR",
            "2021-01-01",
            frozenset({"IR"}),
        ).ticker
        == "IR"
    )
    print("\n=== SANITY CHECK: closed manual interval versus current reuse ===")
    print("  historical IR stays on its prior entity; the open derived IR tenure owns today")
    print("  OK: manual precedence is active-window precedence, never all-time symbol capture")


def test_symbol_rows_resolve_unique_pairs_once_and_keep_unresolved_evidence():
    identity = _simple()
    source = pd.DataFrame(
        {
            "source_symbol": ["AAA", "AAA", "ZZZ"],
            "date": pd.to_datetime(["2014-01-01", "2014-01-01", "2014-01-01"]),
            "value": [1.0, 2.0, 3.0],
        }
    )
    accepted, unresolved = resolve_symbol_rows(identity, source, frozenset({"AAA", "BBB"}))

    assert accepted["ticker"].tolist() == ["AAA", "AAA"]
    assert accepted["resolution_verdict"].eq("exact_dated_tenure").all()
    assert len(unresolved) == 1 and unresolved.iloc[0]["resolution_verdict"] == "unknown_symbol"

    print("\n=== SANITY CHECK: vector symbol resolution ===")
    print("  duplicate AAA/date rows share one exact verdict; ZZZ remains unresolved evidence")
    print("  OK: accepted and unresolved rows retain their original values and dates")


def test_d19_roster_proxy_is_dated_and_redundant_share_class_is_excluded():
    """A filing symbol may be the sibling class, but its historical issuers still matter."""
    identity = build_identity(
        lineage=_lineage(
            [
                ("0000000100", "E_OLD", "roster"),
                ("0000000200", "E_CURRENT", "roster"),
                ("0000000300", "E_LEN", "roster"),
                ("0000000400", "E_GOOGLE", "roster"),
            ]
        ),
        tenure=_tenure(
            [
                ("FOX", "0000000100", "2013-07-01", "2019-03-21", 300),
                ("FOX", "0000000200", "2019-02-05", None, 300),
                ("LEN, LEN.B", "0000000300", "2006-01-01", None, 800),
                ("GOOG", "0000000400", "2006-01-01", None, 800),
                ("GOOGL", "0000000400", "2017-01-01", None, 800),
            ]
        ),
        roster=_roster([("FOXA", "0000000200"), ("LEN", "0000000300"), ("GOOGL", "0000000400")]),
        d19_allowlist={"FOXA": "files under sibling class", "LEN": "combined filing symbol"},
        redundant_symbols=frozenset({"FOX", "GOOG"}),
    )
    universe = frozenset({"FOXA", "LEN", "GOOGL"})

    old_foxa = identity.resolve_symbol_ticker("FOXA", "2018-01-01", universe)
    overlap_foxa = identity.resolve_symbol_ticker("FOXA", "2019-03-01", universe)
    current_foxa = identity.resolve_symbol_ticker("FOXA", "2020-01-01", universe)
    redundant_fox = identity.resolve_symbol_ticker("FOX", "2020-01-01", universe)
    lennar = identity.resolve_symbol_ticker("LEN", "2018-01-01", universe)
    predecessor_google = identity.resolve_symbol_ticker("GOOG", "2016-01-01", universe)
    redundant_google = identity.resolve_symbol_ticker("GOOG", "2020-01-01", universe)

    assert old_foxa.verdict == "entity_not_in_universe"
    assert overlap_foxa.verdict == "ambiguous"
    assert (current_foxa.ticker, current_foxa.verdict, current_foxa.match_kind) == (
        "FOXA",
        "roster_tenure_proxy",
        "roster_tenure_proxy",
    )
    assert redundant_fox.verdict == "redundant_share_class" and not redundant_fox.accepted
    assert (lennar.ticker, lennar.verdict) == ("LEN", "roster_tenure_proxy")
    assert (predecessor_google.ticker, predecessor_google.verdict) == (
        "GOOGL",
        "mapped_current_ticker",
    )
    assert redundant_google.verdict == "redundant_share_class" and not redundant_google.accepted

    print("\n=== SANITY CHECK: dual-class roster proxy ===")
    print("  FOXA borrows FOX's dated issuer boundary; LEN borrows 'LEN, LEN.B'")
    print("  pre-2019 FOXA stays outside, the overlap stays ambiguous, current FOXA resolves")
    print("  FOX is excluded as a separately traded redundant class, never summed into FOXA")
    print("  GOOG remains GOOGL's predecessor before GOOGL is active, then becomes redundant")


# --------------------------------------------------------------------------- #
# D19 and the empty-table raises                                                #
# --------------------------------------------------------------------------- #


def test_d19_disagreement_raises_and_the_allowlist_suppresses_only_its_own_ticker():
    frames = dict(
        lineage=_lineage([("0000000100", "E0000000100", "roster"), ("0000000900", "E0000000900", "roster")]),
        tenure=_tenure([("AAA", "0000000700", pd.Timestamp("2006-01-01"), None, 500), ("BBB", "0000000800", pd.Timestamp("2006-01-01"), None, 500)]),
        roster=_roster([("AAA", "0000000100"), ("BBB", "0000000900")]),
    )
    with pytest.raises(UniverseEntityDisagreementError) as excinfo:
        build_identity(**frames)
    assert "AAA" in str(excinfo.value) and "BBB" in str(excinfo.value)

    with pytest.raises(UniverseEntityDisagreementError, match="BBB"):
        build_identity(**frames, d19_allowlist={"AAA": "adjudicated in writing"})
    build_identity(**frames, d19_allowlist={"AAA": "reason", "BBB": "reason"})  # both explained

    print("\n=== SANITY CHECK: D19, the XOM check ===")
    print("  roster CIK and symbol_tenure naming different entities -> raises, naming both")
    print("  an allow-list entry suppresses ONE ticker and leaves the other raising")
    print("  OK: the free cross-check between a Wikipedia CIK and the filings themselves")
    print("  -> XOM's shell CIK returned 0 proxies for months; this is what catches that.")


@pytest.mark.parametrize("empty", ["lineage", "tenure", "roster"])
def test_an_empty_table_is_a_loud_raise_not_an_empty_map(empty):
    """`load_registrants` returns {} for a missing file, which is right for an OPTIONAL
    curated layer. These tables are mandatory: an empty one would make `owns()` reject every
    predecessor row in the panel and nothing would say so."""
    frames = dict(
        lineage=_lineage([("0000000100", "E0000000100", "roster")]),
        tenure=_tenure([("AAA", "0000000100", pd.Timestamp("2006-01-01"), None, 5)]),
        roster=_roster([("AAA", "0000000100")]),
    )
    frames[empty] = frames[empty].iloc[0:0]
    with pytest.raises(IdentityError):
        build_identity(**frames)

    print(f"\n=== SANITY CHECK: empty {empty} ===")
    print("  build_identity raises IdentityError rather than returning a silent empty map")
    print("  OK: a mandatory table cannot fail open")
    print("  -> An empty lineage would quarantine every legitimate predecessor in silence.")


def test_load_identity_caches_per_context_and_not_across_them():
    """A module-level cache would hand one test's database to the next."""

    class _Store:
        def __init__(self, frames):
            self._frames = frames

        def load(self, table, **kwargs):
            return self._frames[getattr(table, "name", str(table))]

    class _Ctx:
        def __init__(self, frames):
            self.store = _Store(frames)
            self.config_dir = CONFIG_DIR
            self.config = type("Config", (), {"data_extract": type("Extract", (), {"redundant_ticks": []})()})()

    def frames(cik, ticker):
        return {
            "entity_lineage": _lineage([(cik, f"E{cik}", "roster")]),
            "symbol_tenure": _tenure([(ticker, cik, pd.Timestamp("2006-01-01"), None, 5)]),
            "sp500_tickers": _roster([(ticker, cik)]),
        }

    first, second = _Ctx(frames("0000000100", "AAA")), _Ctx(frames("0000000900", "BBB"))
    a, b = load_identity(first, CONFIG_DIR), load_identity(second, CONFIG_DIR)
    assert load_identity(first, CONFIG_DIR) is a  # same context -> cached instance
    refreshed = load_identity(first, CONFIG_DIR, refresh=True)
    assert refreshed is not a and refreshed.roster_cik == a.roster_cik
    assert a is not b
    assert set(a.roster_cik) == {"AAA"} and set(b.roster_cik) == {"BBB"}

    print("\n=== SANITY CHECK: per-context caching ===")
    print(f"  context 1 -> {sorted(a.roster_cik)}; context 2 -> {sorted(b.roster_cik)}")
    print("  OK: one read per context, and no leakage between two databases")
    print("  -> load_registrants may use @cache only because its key is a config directory.")


def test_normalise_cik_handles_every_spelling_in_the_repo():
    assert normalise_cik("320193") == "0000320193"
    assert normalise_cik("320193.0") == "0000320193"
    assert normalise_cik(" 0000320193 ") == "0000320193"
    assert normalise_cik(320193) == "0000320193"
    assert normalise_cik("not-a-cik") == "not-a-cik"  # passed through, never silently zero

    print("\n=== SANITY CHECK: CIK normalisation ===")
    print("  '320193' / '320193.0' / ' 0000320193 ' / int -> 0000320193")
    print("  OK: three spellings of one CIK cannot become three entities")
    print("  -> The zips, the roster's float round-trip and the register all disagree.")


# --------------------------------------------------------------------------- #
# live acceptance table                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def live():
    from src.context import get_config_context

    try:
        _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
        return load_identity(context, CONFIG_DIR)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable ({type(exc).__name__}: {exc})")


#: The four worked cases of part-2 section 3.2, by CIK, with the entity ids the LIVE tables
#: mint. ⚠ `IR`'s predecessor entity is `E0000836102` and not the plan's `E0001160497`: the
#: register's `TT` entry chains in the older Ingersoll-Rand Company CIK `0000836102`, and D16
#: says in writing that a group gaining an older CIK shifts its id. The VERDICT -- which is
#: what the plan actually asserts -- is unchanged.
WORKED_CASES = [
    ("IR", "0001466258", "E0000836102", "E0001699150", False, "Ingersoll-Rand plc is TT's"),
    ("DD", "0000030554", "E0000030554", "E0000030554", True, "DuPont E I, 3,422 real rows"),
    ("AVGO", "0001317092", "E0001317092", "E0001441634", False, "Avicena Group, a 2006 reuse"),
    ("COR", "0001140859", "E0001140859", "E0001140859", True, "AmerisourceBergen filed as ABC"),
]


@pytest.mark.parametrize("ticker,cik,entity,universe,verdict,why", WORKED_CASES)
def test_the_four_worked_cases(live, ticker, cik, entity, universe, verdict, why):
    assert live.entity_of(cik) == entity
    assert live.universe_entity(ticker) == universe
    assert live.owns(ticker, cik, pd.Timestamp("2020-01-01")) is verdict


def test_the_four_worked_cases_summary(live):
    print("\n=== SANITY CHECK: the four worked cases ===")
    for ticker, cik, entity, universe, verdict, why in WORKED_CASES:
        print(f"  {ticker:5s} {cik} {entity} vs {universe} -> owns={verdict!s:5s} ({why})")
    print("  OK: the ENTITY comparison alone settles all four")
    print("  -> IR is the one a single-axis symbol oracle gets wrong; no IR rule exists here.")


#: Every (ticker, issuer CIK) group the Phase 2.3 screen flagged, with the verdict this
#: resolver must return. Read by name against the issuer name on the filing, not copied from
#: the code's output: `KEEP` is a genuine predecessor or affiliate of today's registrant,
#: `MOVE` means the rows belong to a DIFFERENT universe ticker, `DROP` is another company.
#: `EA` and `AVB` are labelled in the table but are no longer in the 500-ticker roster.
KEEP_GROUPS = {
    ("DD", "0000030554"),
    ("CB", "0000020171"),
    ("JCI", "0000053669"),
    ("MRVL", "0001058057"),
    ("COHR", "0000021510"),
    ("DOW", "0000029915"),
    ("STE", "0000815065"),
    ("PLD", "0000899881"),
    ("AVGO", "0001441634"),
    ("ACN", "0001134538"),
    ("DOC", "0001574540"),
    ("DELL", "0000826083"),
    ("VMC", "0000103973"),
    ("DLR", "0001494877"),
    ("TPL", "0000097517"),
    ("HLT", "0000047580"),
    ("TT", "0000836102"),
    ("MRK", "0000064978"),
    ("GM", "0000040730"),
    ("DUK", "0000030371"),
    ("FERG", "0001832433"),
    ("PCG", "0000075488"),
    ("PSA", "0000318380"),
    ("KMI", "0000054502"),
    ("BLK", "0001060021"),
    ("ORCL", "0000777676"),
    ("OKE", "0000074154"),
    ("TMUS", "0001097609"),
    ("ACN", "0001143908"),
    ("IRM", "0001132694"),
    ("WFC", "0000105598"),
}
MOVE_GROUPS = {("IR", "0001466258"): "TT", ("IR", "0001160497"): "TT", ("TT", "0000749251"): "IT", ("NTRS", "0000049826"): "ITW"}


def test_every_flagged_group_resolves_to_its_reviewed_verdict(live):
    """The acceptance table for the whole plan, on the live tables."""
    flagged = pd.read_csv("reports/planning/active-tasks/2026-09-07-informed-capital/insider_out_of_lineage.csv", dtype=str)
    flagged["rows"] = flagged["rows"].astype(int)
    counts, wrong = {"KEEP": 0, "MOVE": 0, "DROP": 0}, []
    rows = {"KEEP": 0, "MOVE": 0, "DROP": 0}
    for entry in flagged.itertuples():
        key = (entry.ticker, entry.issuer_cik)
        resolved = live.entity_ticker(entry.issuer_cik)
        if key in KEEP_GROUPS:
            expected, kind = entry.ticker, "KEEP"
        elif key in MOVE_GROUPS:
            expected, kind = MOVE_GROUPS[key], "MOVE"
        else:
            expected, kind = None, "DROP"
        counts[kind] += 1
        rows[kind] += entry.rows
        if resolved != expected:
            wrong.append((key, entry.issuer_name, expected, resolved))
    assert not wrong, f"{len(wrong)} group(s) resolved against the reviewed verdict: {wrong}"

    print("\n=== SANITY CHECK: all 102 flagged groups ===")
    print(f"  KEEP {counts['KEEP']:>3} groups {rows['KEEP']:>7,} rows  " "genuine predecessors, retained by entity_lineage")
    print(f"  MOVE {counts['MOVE']:>3} groups {rows['MOVE']:>7,} rows  " "relabelled onto the universe ticker that owns them")
    print(f"  DROP {counts['DROP']:>3} groups {rows['DROP']:>7,} rows  " "another company -- quarantined")
    print("  OK: every one of the 102 matches the verdict read from the issuer name")
    print("  -> A register-only cut would have deleted the 25,635 KEEP rows.")


def test_owns_is_symmetric_with_entity_ticker_on_every_lineage_cik(live):
    """Two spellings of one contract must not drift apart."""
    checked = 0
    for cik in live.entity_by_cik:
        resolved = live.entity_ticker(cik)
        for ticker in live.roster_cik:
            assert live.owns(ticker, cik) is (resolved == ticker)
            checked += 1
            if resolved == ticker:
                break

    print("\n=== SANITY CHECK: owns() == entity_ticker() ===")
    print(f"  {len(live.entity_by_cik)} lineage CIK(s), {checked:,} (ticker, CIK) comparisons")
    print("  OK: the predicate and the resolution core agree everywhere")
    print("  -> The fetcher resolves with entity_ticker; the quarantine reason cites owns().")


def test_only_one_alphabet_ticker_is_investable(live):
    """The roster has one investable class; symbol-volume sources enforce that separately."""
    assert "GOOG" not in live.roster_cik
    assert "GOOGL" in live.roster_cik
    alphabet = live.universe_entity("GOOGL")
    assert live.ticker_by_entity[alphabet] == "GOOGL"
    for pair in (("FOX", "FOXA"), ("NWS", "NWSA")):
        assert pair[0] not in live.roster_cik and pair[1] in live.roster_cik

    print("\n=== SANITY CHECK: dual class stays one investable ticker ===")
    print(f"  GOOGL -> {alphabet}; GOOG absent from the roster; same for FOX/FOXA, NWS/NWSA")
    print("  OK: one entity, one universe ticker, so the two-ticker raise cannot fire on it")
    print("  -> FTD/RegSHO separately exclude the redundant traded class before aggregation.")


def test_live_dual_class_volume_symbols_resolve_without_combining_classes(live):
    universe = frozenset({"FOXA", "NWSA", "GOOGL", "LEN"})

    assert live.resolve_symbol_ticker("FOXA", "2018-01-02", universe).verdict == "entity_not_in_universe"
    assert live.resolve_symbol_ticker("FOXA", "2020-01-02", universe).verdict == "roster_tenure_proxy"
    assert live.resolve_symbol_ticker("NWSA", "2020-01-02", universe).verdict == "roster_tenure_proxy"
    assert live.resolve_symbol_ticker("LEN", "2020-01-02", universe).verdict == "roster_tenure_proxy"
    assert live.resolve_symbol_ticker("GOOG", "2010-01-04", universe).ticker == "GOOGL"
    for redundant in ("FOX", "NWS", "GOOG"):
        assert live.resolve_symbol_ticker(redundant, "2020-01-02", universe).verdict == "redundant_share_class"

    print("\n=== SANITY CHECK: live dual-class symbol-volume policy ===")
    print("  FOXA/NWSA/LEN use dated D19 proxies; pre-restructure FOXA is not imported")
    print("  FOX/NWS/current GOOG are excluded; predecessor GOOG still maps to GOOGL")
    print("  OK: the retained class is recovered without adding its sibling's volume")


def test_no_live_entity_holds_two_universe_tickers(live):
    """`build_identity` raises on this, so reaching the fixture at all proves it -- asserted
    explicitly anyway because it is the invariant the whole design rests on."""
    assert len(live.ticker_by_entity) == len(live.roster_cik) == 500

    print("\n=== SANITY CHECK: one entity per universe ticker ===")
    print(f"  {len(live.roster_cik)} tickers -> {len(live.ticker_by_entity)} entities, " "0 collisions")
    print("  OK: no entity can relabel one universe ticker's rows onto another")
    print("  -> load_identity would have raised before returning if it could.")
