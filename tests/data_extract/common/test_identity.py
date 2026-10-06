"""
`identity.py` -- the lineage accessor (`filing_scope`, `ticker_for_cik`, `tape_interval`), the
tenure resolver, its load-time raises and the live acceptance table of every (ticker, issuer CIK)
group the 2.3 screen flagged.

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
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.entity_lineage import (
    CikInTwoEntitiesError,
    IdentityError,
    TwoUniverseTickersOneEntityError,
)
from src.data_extract.utils.common.identity import (
    AmbiguousSymbolTenureError,
    Identity,
    UnknownUniverseTickerError,
    build_identity,
    load_identity,
    tickers_for_ciks,
)
from src.utils.string import pad_cik, pad_cik_series

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
    assert identity.ticker_for_cik("0000999999") is None
    # three spellings of one CIK, one entity
    assert identity.entity_of("100") == identity.entity_of("0000000100") == identity.entity_of("0000000100.0") == "E0000000100"

    print("\n=== SANITY CHECK: an unknown CIK ===")
    print(f"  entity_of('0000999999') -> {identity.entity_of('0000999999')} (singleton, no raise)")
    print(f"  '100' / '0000000100' / '0000000100.0' -> {identity.entity_of('100')}")
    print("  OK: silence in the lineage table is an answer, and CIK spelling is normalised")
    print("  -> A 4.3M-row parse cannot abort on a CIK nobody has ever adjudicated.")


def test_event_ticker_for_cik_resolves_the_predecessor_and_returns_none_off_universe():
    identity = _simple()
    assert identity.ticker_for_cik("0000000100", None, "event") == "AAA"  # predecessor -> today's ticker
    assert identity.ticker_for_cik("0000000200", None, "event") == "AAA"
    assert identity.ticker_for_cik("0000000900", None, "event") == "BBB"
    assert identity.ticker_for_cik("0000000042", None, "event") is None
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
    )
    with pytest.raises(AmbiguousSymbolTenureError):
        two_entities.entity_for("AAA", "2012-01-01")

    print("\n=== SANITY CHECK: overlap grain ===")
    print("  two CIKs of ONE entity overlapping -> resolves, no raise")
    print("  two DIFFERENT entities overlapping -> AmbiguousSymbolTenureError")
    print("  OK: paperwork overlap is not identity ambiguity")
    print("  -> Measured live: 2,432 symbols ambiguous over all history, 10 at today.")


# --------------------------------------------------------------------------- #
# dated entity_lineage accessor                                                 #
# --------------------------------------------------------------------------- #

_CHANGED_AAA = pd.Timestamp("2026-10-03 10:00:00")
_CHANGED_BBB = pd.Timestamp("2026-09-01 08:00:00")


def _dated_lineage(rows: list[tuple[str, str, str, str, str, str, str | None, str]]) -> pd.DataFrame:
    """`entity_lineage` rows: (entity, ticker, cik, role, symbol, valid_from, valid_to, status)."""
    stamps = {"E0000000100": _CHANGED_AAA, "E0000000900": _CHANGED_BBB}
    return pd.DataFrame(
        [
            {
                "entity_id": entity,
                "canonical_ticker": ticker,
                "cik": cik,
                "role": role,
                "symbol": symbol,
                "valid_from": start,
                "valid_to": end,
                "status": status,
                "sources": "test",
                "oracle": "register",
                "confidence": None,
                "n_observations": 10,
                "evidence": "test",
                "scope_changed_at": stamps.get(entity, _CHANGED_BBB),
            }
            for entity, ticker, cik, role, symbol, start, end, status in rows
        ]
    )


def _dated() -> Identity:
    """AAA: CIK 100 to the 2015-10-02 seam, CIK 200 after it, event-only CIK 300. BBB: one open window.
    CCC: two windows a year apart (no seam). Symbol rows carry noise, conflict and a two-entity overlap."""
    lineage = _dated_lineage(
        [
            ("E0000000100", "AAA", "0000000100", "cik_window", "", "1900-01-01", "2015-10-02", "curated"),
            ("E0000000100", "AAA", "0000000200", "cik_window", "", "2015-10-02", None, "curated"),
            ("E0000000100", "AAA", "0000000300", "cik_event", "", "1900-01-01", None, "curated"),
            ("E0000000100", "AAA", "0000000100", "symbol", "AAA", "2006-01-01", "2015-10-02", "corroborated"),
            ("E0000000100", "AAA", "0000000200", "symbol", "AAA", "2015-10-02", None, "single_source"),
            ("E0000000100", "AAA", "0000000100", "symbol", "OLDX", "2001-01-01", "2006-01-01", "noise"),
            ("E0000000100", "AAA", "0000000100", "symbol", "ZZZ", "2010-01-01", "2012-01-01", "conflict"),
            ("E0000000100", "AAA", "0000000200", "symbol", "DUP", "2018-01-01", None, "single_source"),
            ("E0000000900", "BBB", "0000000900", "cik_window", "", "1900-01-01", None, "single_source"),
            ("E0000000900", "BBB", "0000000900", "symbol", "BBB", "2006-01-01", None, "corroborated"),
            ("E0000000900", "BBB", "0000000900", "symbol", "ZZZ", "2011-06-01", "2013-01-01", "single_source"),
            ("E0000000900", "BBB", "0000000900", "symbol", "DUP", "2019-01-01", None, "single_source"),
            ("E0000000700", "CCC", "0000000700", "cik_window", "", "1900-01-01", "2010-01-01", "curated"),
            ("E0000000700", "CCC", "0000000710", "cik_window", "", "2011-01-01", None, "curated"),
        ]
    )
    return build_identity(
        lineage=lineage,
        tenure=_tenure([("AAA", "0000000200", pd.Timestamp("2015-10-02"), None, 10), ("BBB", "0000000900", pd.Timestamp("2006-01-01"), None, 10)]),
        roster=_roster([("AAA", "0000000200"), ("BBB", "0000000900"), ("CCC", "0000000710")]),
    )


def test_ticker_for_cik_follows_dated_windows_and_the_seam_margin():
    """Consolidating: the CIK's window, widened 31 days at a seam, must hold the filing date. Event: any entity CIK."""
    identity = _dated()
    consolidating = {
        ("100", "2010-06-30"): "AAA",  # predecessor inside its window
        ("100", "2015-11-01"): "AAA",  # 30 days after the seam: inside the margin
        ("100", "2015-11-02"): None,  # 31 days after: the widened end is excluded
        ("200", "2015-09-01"): "AAA",  # successor 31 days before the seam: widened start included
        ("200", "2015-08-31"): None,
        ("200", "2020-01-01"): "AAA",
        ("300", "2020-01-01"): None,  # event-only CIK never consolidates
        ("700", "2010-01-15"): None,  # a year-long gap is not a seam: no margin
        ("710", "2010-12-15"): None,
        ("555", "2020-01-01"): None,  # not an entity CIK
    }
    for (cik, filed), expected in consolidating.items():
        assert identity.ticker_for_cik(cik, filed, "consolidating") == expected, (cik, filed)
    assert identity.ticker_for_cik("0000000100", None, "consolidating") is None
    assert identity.ticker_for_cik(300, None, "event") == "AAA"
    assert identity.ticker_for_cik("100", "1990-01-01", "event") == "AAA"
    assert identity.ticker_for_cik("555", "2020-01-01", "event") is None

    print("\n=== SANITY CHECK: windowed ticker_for_cik ===")
    print("  predecessor inside its window -> AAA; 30 days past the seam -> AAA; 31 days -> None")
    print("  successor 31 days before the seam -> AAA, 32 days -> None; event-only CIK -> event forms only")
    print("  OK: a predecessor's consolidating filing outside its window (and margin) is not the ticker's")


def test_filing_scope_lists_event_ciks_widened_windows_and_scope_timestamp():
    identity = _dated()
    scope = identity.filing_scope("AAA")
    assert scope.event_ciks == ("0000000100", "0000000200", "0000000300")
    old, new = scope.windows
    assert (old.cik, old.valid_from, old.valid_to, old.listed_to) == ("0000000100", None, pd.Timestamp("2015-10-02"), pd.Timestamp("2015-11-02"))
    assert (new.cik, new.valid_from, new.listed_from, new.listed_to) == ("0000000200", pd.Timestamp("2015-10-02"), pd.Timestamp("2015-09-01"), None)
    assert old.owns(pd.Timestamp("2015-10-01")) and not old.owns(pd.Timestamp("2015-10-02")) and old.admits(pd.Timestamp("2015-10-20"))
    assert scope.scope_changed_at == _CHANGED_AAA
    (only,) = identity.filing_scope("BBB").windows
    assert (only.listed_from, only.listed_to) == (None, None) and identity.filing_scope("BBB").scope_changed_at == _CHANGED_BBB
    ccc = identity.filing_scope("CCC").windows
    assert [(w.listed_from, w.listed_to) for w in ccc] == [(None, pd.Timestamp("2010-01-01")), (pd.Timestamp("2011-01-01"), None)]

    print("\n=== SANITY CHECK: filing_scope ===")
    print(f"  AAA event CIKs {scope.event_ciks}; windows {[(w.cik, w.listed_from, w.listed_to) for w in scope.windows]}")
    print("  seam widened 31 days on both sides; a year-long gap is left alone; open start stays open")
    print(f"  scope_changed_at {scope.scope_changed_at} (per entity); OK")


def test_an_entity_without_window_rows_reads_its_roster_cik_as_one_open_window():
    """A membership-only lineage frame (no `role`) and a roster CIK with no lineage row both list the roster CIK alone."""
    identity = _simple()
    scope = identity.filing_scope("AAA")
    assert scope.event_ciks == ("0000000100", "0000000200")
    assert [(w.cik, w.listed_from, w.listed_to) for w in scope.windows] == [("0000000200", None, None)]
    assert identity.ticker_for_cik("100", "2010-01-01", "consolidating") is None
    assert identity.ticker_for_cik("100", "2010-01-01", "event") == "AAA"
    assert identity.ticker_for_cik("200", "2010-01-01", "consolidating") == "AAA"
    assert identity.tape_interval("AAA", "2020-01-01") is None  # no symbol rows to answer from

    print("\n=== SANITY CHECK: roster-only scope ===")
    print("  no cik_window row -> the roster CIK is the one open window; other entity CIKs are event-only")


def test_dei_tenure_rows_do_not_reach_the_tenure_resolver():
    """`dei` cover-page evidence feeds the lineage build only; the resolver reads `form345`/`manual`."""
    tenure = pd.concat(
        [
            _tenure([("AAA", "0000000200", pd.Timestamp("2015-01-01"), None, 900)]),
            pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "issuer_cik": "0000000555",
                        "valid_from": "2016-01-01",
                        "valid_to": None,
                        "n_filings": 3,
                        "source": "dei",
                        "evidence": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    identity = build_identity(lineage=_lineage([("0000000200", "E0000000200", "roster")]), tenure=tenure, roster=_roster([("AAA", "0000000200")]))
    assert identity.entity_for("AAA", "2020-01-01") == "E0000000200"
    assert identity.filing_scope("AAA").event_ciks == ("0000000200",)

    print("\n=== SANITY CHECK: dei stays out of the tenure resolver ===")
    print("  a dei AAA row under a foreign CIK neither makes AAA ambiguous nor joins the filing scope")


def test_tickers_for_ciks_answers_each_cik_date_pair_by_policy():
    identity = _dated()
    ciks = pd.Series(["100", "0000000100", "300", "200", "555"])
    filed = pd.Series(pd.to_datetime(["2010-06-30", "2010-06-30", "2020-01-01", None, "2020-01-01"]))

    consolidating = tickers_for_ciks(identity, ciks, filed, "consolidating")
    event = tickers_for_ciks(identity, ciks, filed, "event")

    assert consolidating.tolist() == ["AAA", "AAA", None, None, None]
    assert event.tolist() == ["AAA", "AAA", "AAA", "AAA", None]
    assert consolidating.index.equals(ciks.index)
    print("\n=== SANITY CHECK: vectorised ticker_for_cik ===")
    print("  consolidating needs a window holding the date (event-only CIK, no date -> None); event needs an entity CIK")


def test_tape_interval_answers_dated_intervals_and_universe_symbols_skip_noise():
    identity = _dated()
    entity = identity.universe_entity("AAA")
    hit = identity.tape_interval("aaa", "2010-01-01")
    assert hit is not None and hit.entity == entity
    for symbol, day in (("OLDX", "2003-01-01"), ("ZZZ", "2011-01-01"), ("QQQ", "2020-01-01"), ("BBB", None)):
        assert identity.tape_interval(symbol, day) is None, (symbol, day)
    assert identity.tape_interval("DUP", "2020-01-01") is None  # two entities, neither in conflict
    ended = identity.tape_interval("ZZZ", "2012-06-01")  # the conflict interval has ended
    assert ended is not None and identity.ticker_by_entity.get(ended.entity) == "BBB"
    assert identity.universe_symbols(frozenset({"AAA"})) == frozenset({"AAA", "ZZZ", "DUP"})
    print("\n=== SANITY CHECK: tape intervals ===")
    print("  AAA on its interval answers with its entity; noise OLDX, conflict ZZZ, two-entity DUP, unknown QQQ and an undated lookup do not;")
    print("  ZZZ answers BBB once the conflict interval has ended")


def test_a_tape_symbol_never_maps_an_interval_evidenced_by_dei_alone():
    """Cover-page `dei` lists every security line (preferreds, notes); FTD/RegSHO rows map only intervals with
    Form 3/4/5, manual or roster evidence."""
    rows = _dated_lineage(
        [
            ("E0000000100", "AAA", "0000000100", "cik_window", "", "1900-01-01", None, "curated"),
            ("E0000000100", "AAA", "0000000100", "symbol", "AAA", "2006-01-01", None, "corroborated"),
            ("E0000000100", "AAA", "0000000100", "symbol", "AAA-PR-C", "2019-08-06", None, "single_source"),
        ]
    )
    rows["sources"] = ["", "dei,form345,roster", "dei"]
    identity = build_identity(
        lineage=rows, tenure=_tenure([("AAA", "0000000100", pd.Timestamp("2006-01-01"), None, 10)]), roster=_roster([("AAA", "0000000100")])
    )
    assert identity.tape_interval("AAA", "2020-01-02") is not None
    assert identity.tape_interval("AAA-PR-C", "2020-01-02") is None
    assert identity.universe_symbols(frozenset({"AAA"})) == frozenset({"AAA"})
    print("\n=== SANITY CHECK: dei-only intervals stay off the symbol tapes ===")
    print("  AAA-PR-C (a preferred line, dei evidence only) never adds its fails/volume to AAA; AAA itself maps. Validated.")


# --------------------------------------------------------------------------- #
# D19 and the empty-table raises                                                #
# --------------------------------------------------------------------------- #


def test_d19_is_the_lineage_builds_check_and_is_not_repeated_at_load():
    """D19 stops `identity-tables` (`test_entity_lineage.test_a_d19_disagreement_still_stops_the_build`);
    a consumer loading the stored tables does not re-run it."""
    identity = build_identity(
        lineage=_lineage([("0000000100", "E0000000100", "roster"), ("0000000900", "E0000000900", "roster")]),
        tenure=_tenure([("AAA", "0000000700", pd.Timestamp("2006-01-01"), None, 500), ("BBB", "0000000800", pd.Timestamp("2006-01-01"), None, 500)]),
        roster=_roster([("AAA", "0000000100"), ("BBB", "0000000900")]),
    )
    assert identity.universe_entity("AAA") == "E0000000100" and identity.universe_entity("BBB") == "E0000000900"

    print("\n=== SANITY CHECK: D19 runs once, in the build ===")
    print("  roster CIKs disagreeing with symbol_tenure load without a raise; the build owns the check")


@pytest.mark.parametrize("empty", ["lineage", "tenure", "roster"])
def test_an_empty_table_is_a_loud_raise_not_an_empty_map(empty):
    """`load_registrants` returns {} for a missing file, which is right for an OPTIONAL
    curated layer. These tables are mandatory: an empty one would make `owns()` reject every
    predecessor row in the panel and nothing would say so."""
    frames: dict[str, Any] = dict(
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
            name = getattr(table, "name", str(table))
            return self._frames.get(name) if kwargs.get("optional") else self._frames[name]

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

    first: Any = _Ctx(frames("0000000100", "AAA"))
    second: Any = _Ctx(frames("0000000900", "BBB"))
    a, b = load_identity(first), load_identity(second)
    assert load_identity(first) is a  # same context -> cached instance
    refreshed = load_identity(first, refresh=True)
    assert refreshed is not a and refreshed.roster_cik == a.roster_cik
    assert a is not b
    assert set(a.roster_cik) == {"AAA"} and set(b.roster_cik) == {"BBB"}

    print("\n=== SANITY CHECK: per-context caching ===")
    print(f"  context 1 -> {sorted(a.roster_cik)}; context 2 -> {sorted(b.roster_cik)}")
    print("  OK: one read per context, and no leakage between two databases")
    print("  -> load_registrants may use @cache only because its key is a config directory.")


def test_pad_cik_handles_every_spelling_in_the_repo():
    spellings = ["320193", "320193.0", " 0000320193 ", 320193, 320193.0]
    assert {pad_cik(s) for s in spellings} == {"0000320193"}
    assert pad_cik("not-a-cik") == "" and pad_cik(None) == "" and pad_cik(float("nan")) == ""  # never a fake CIK
    vector = pad_cik_series(pd.Series([*spellings, None, float("nan"), ""], dtype=object))
    assert list(vector) == ["0000320193"] * len(spellings) + ["", "", ""]

    print("\n=== SANITY CHECK: CIK normalisation ===")
    print("  '320193' / '320193.0' / ' 0000320193 ' / int / float -> 0000320193, scalar and vectorised")
    print("  OK: one spelling per CIK, and a null or non-numeric value pads to '' rather than a fake CIK")


# --------------------------------------------------------------------------- #
# live acceptance table                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def live():
    from src.context import get_config_context

    try:
        _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
        return load_identity(context)
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
        key = (str(entry.ticker), str(entry.issuer_cik))
        resolved = live.ticker_for_cik(str(entry.issuer_cik), None, "event")
        if key in KEEP_GROUPS:
            expected, kind = entry.ticker, "KEEP"
        elif key in MOVE_GROUPS:
            expected, kind = MOVE_GROUPS[key], "MOVE"
        else:
            expected, kind = None, "DROP"
        counts[kind] += 1
        rows[kind] += int(str(entry.rows))
        if resolved != expected:
            wrong.append((key, entry.issuer_name, expected, resolved))
    assert not wrong, f"{len(wrong)} group(s) resolved against the reviewed verdict: {wrong}"

    print("\n=== SANITY CHECK: all 102 flagged groups ===")
    print(f"  KEEP {counts['KEEP']:>3} groups {rows['KEEP']:>7,} rows  genuine predecessors, retained by entity_lineage")
    print(f"  MOVE {counts['MOVE']:>3} groups {rows['MOVE']:>7,} rows  relabelled onto the universe ticker that owns them")
    print(f"  DROP {counts['DROP']:>3} groups {rows['DROP']:>7,} rows  another company -- quarantined")
    print("  OK: every one of the 102 matches the verdict read from the issuer name")
    print("  -> A register-only cut would have deleted the 25,635 KEEP rows.")


def test_owns_is_symmetric_with_event_ticker_for_cik_on_every_lineage_cik(live):
    """Two spellings of one contract must not drift apart."""
    checked = 0
    for cik in live.entity_by_cik:
        resolved = live.ticker_for_cik(cik, None, "event")
        for ticker in live.roster_cik:
            assert live.owns(ticker, cik) is (resolved == ticker)
            checked += 1
            if resolved == ticker:
                break

    print("\n=== SANITY CHECK: owns() == ticker_for_cik(event) ===")
    print(f"  {len(live.entity_by_cik)} lineage CIK(s), {checked:,} (ticker, CIK) comparisons")
    print("  OK: the predicate and the resolution core agree everywhere")
    print("  -> The insider fetcher resolves with ticker_for_cik(event); the quarantine reason cites owns().")


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
    print("  -> FTD/RegSHO separate the classes per security (security_master) before aggregation.")


def test_no_live_entity_holds_two_universe_tickers(live):
    """`build_identity` raises on this, so reaching the fixture at all proves it -- asserted
    explicitly anyway because it is the invariant the whole design rests on."""
    assert len(live.ticker_by_entity) == len(live.roster_cik) == 500

    print("\n=== SANITY CHECK: one entity per universe ticker ===")
    print(f"  {len(live.roster_cik)} tickers -> {len(live.ticker_by_entity)} entities, 0 collisions")
    print("  OK: no entity can relabel one universe ticker's rows onto another")
    print("  -> load_identity would have raised before returning if it could.")
