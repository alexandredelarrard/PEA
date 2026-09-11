"""
test_feature_catalogue.py  (tests/data_aggregate/test_feature_catalogue.py)
--------------------------------------------------------------------------
THE FEATURE CATALOGUE, ASSERTED IN BOTH DIRECTIONS (D9). Every governance characteristic must
have prose saying what it measures, and every entry of prose must describe a characteristic the
part still emits. A column with no entry is a feature nobody can interpret; an entry with no
column is a promise the cube no longer keeps.

⚠ WHAT THIS FILE CAN AND CANNOT SEE, because the guarantee is a CHAIN and only one link runs
without a database:

    catalogue == DECLARED     <- here, no database, runs in CI
    DECLARED  == live columns <- `reports/validate/governance/_scripts/07_catalogue.py`

Together those give "every live column is documented". Split that way deliberately: the link
that can drift on a code change is the first one, and it is the one worth a fast test. Asserting
against live columns here would make the check unrunnable exactly when someone is editing the
field sets on a machine with no built cube.

⚠ WHY THE CATALOGUE IS PER PART. `cube_fundamentals_evidence.py` asserts both directions against
ONE table and exits non-zero on either failure, so 85 governance entries in the dict it reads
would land in its `unused` list and fail every future fundamentals run. Hence
`cube_feature_catalogue.CATALOGUES`, a registry of one dict per part.
"""
from __future__ import annotations

from scripts.cube_feature_catalogue import CATALOGUE, CATALOGUES, split
from scripts.cube_governance_catalogue import GOVERNANCE
from src.data_aggregate.utils.governance import director_comp as dc
from src.data_aggregate.utils.governance import directors as dr
from src.data_aggregate.utils.governance import panel as pn
from src.data_aggregate.utils.governance import pay_features as pay
from src.data_aggregate.utils.governance import provisions_features as pv
from src.data_aggregate.utils.governance import vote_dissent_features as vd

#: ⚠ THE ONE ALLOWED DECLARED-BUT-ABSENT FIELD, named rather than filtered.
#:
#: `poison_pill_added` is in `provisions_features.ALL_FIELDS` and has no column, deliberately:
#: `_MIN_EVENTS` suppresses it because it fires ZERO times on the live archive, and a constant-0
#: column is worse than an absent one because it looks like evidence.
#:
#: Named explicitly, as a one-element set, because a loose filter ("skip anything that looks
#: suppressed") would silently absorb the NEXT genuinely-missing column -- which is the exact
#: failure mode this whole test exists to prevent.
SUPPRESSED: frozenset[str] = frozenset({"poison_pill_added"})


def _declared() -> dict[str, set[str]]:
    """The characteristics the governance family modules declare, by encoding.

    Reads the modules' own field sets rather than a list typed here, so the two cannot drift --
    the same reason `07_catalogue.py` builds it this way.
    """
    legacy = ({n for _, n in pn._LEVEL_FIELDS} | {n for _, n in pn._RAW_DEF14A_FIELDS}
              # the four names `_governance_fields` COMPUTES rather than reads off a source
              # column, so no `(source, name)` pair in the module declares them
              | {"ceo_tenure", "ceo_pay_growth", "ceo_pay_vs_revenue_growth",
                 "control_wedge"})
    raw = (legacy | set(pay.ALL_FIELDS) | set(pv.ALL_FIELDS) | set(dr.ALL_FIELDS)
           | set(dc.ALL_FIELDS) | set(vd.EVENT_FIELDS))
    peers = (set(pn._PEER_LEGACY) | set(pay.PEER_RELATIVE_FIELDS)
             | set(pv.PEER_RELATIVE_FIELDS) | set(dr.PEER_RELATIVE_FIELDS)
             | set(dc.PEER_RELATIVE_FIELDS) | set(vd.PEER_RELATIVE_FIELDS))
    return {"raw": raw, "peers": peers, "hist": set(pn._VS_HIST_LEGACY)}


def test_the_governance_catalogue_matches_the_declared_fields_both_ways():
    """Direction 1: nothing undocumented. Direction 2: nothing stale."""
    d = _declared()
    expected = d["raw"] - SUPPRESSED
    cat = set(GOVERNANCE)

    undocumented = sorted(expected - cat)
    stale = sorted(cat - expected)
    assert not undocumented, f"declared with no catalogue entry: {undocumented}"
    assert not stale, f"catalogue entry with no declared field: {stale}"

    print("\n=== SANITY CHECK: the governance catalogue, both directions ===")
    print(f"  declared raw fields   {len(d['raw'])}")
    print(f"  suppressed (allowed)  {sorted(SUPPRESSED)}")
    print(f"  catalogue entries     {len(cat)}")
    print(f"  peer legs {len(d['peers'])} + self-history {len(d['hist'])} resolve to their raw")
    print(f"  parents, so {len(cat)} entries describe "
          f"{len(d['raw'] - SUPPRESSED) + len(d['peers']) + len(d['hist'])} columns.")
    print("  Before 2026-09-08 this was 0 entries for the whole part.")


def test_every_peer_and_self_history_leg_resolves_to_a_documented_parent():
    """A leg is the same quantity as its parent, so it needs no entry of its own -- but it DOES
    need its parent to exist. This is the direction a `_vs_peers` typo fails."""
    d = _declared()
    for leg_set, suffix in ((d["peers"], "_vs_peers"), (d["hist"], "_vs_hist")):
        for name in sorted(leg_set):
            col = f"f_{name}{suffix}"
            char, got = split(col)
            assert got == suffix, f"{col} did not split to {suffix}"
            assert char == name
            assert char in GOVERNANCE, f"{col} has no documented parent `{name}`"

    print("\n=== SANITY CHECK: every encoded leg has a documented parent ===")
    print(f"  {len(d['peers'])} peer legs and {len(d['hist'])} self-history leg checked")
    print(f"  self-history: {sorted(d['hist'])} -- the only field that earned one")


def test_the_retired_vote_twins_are_absent_from_every_declaration():
    """Phase 4 Part A: `_against_pct` differs from `_dissent` only by abstentions."""
    retired = ("sop_against_pct", "auditor_vote_against_pct")
    for name in retired:
        assert name not in vd.EVENT_FIELDS, f"{name} is still declared as a field"
        assert name not in vd.PEER_RELATIVE_FIELDS, f"{name} still has a peer leg"
        assert name not in vd._SOP_LEVELS and name not in vd._AUDITOR_LEVELS
        assert name not in GOVERNANCE, f"{name} still has a catalogue entry"

    # the surviving twins and their dependents must all still be there
    for name in ("sop_dissent", "auditor_vote_dissent", "sop_dissent_excess_10",
                 "sop_dissent_excess_20", "sop_dissent_gt_10", "sop_dissent_gt_20",
                 "sop_dissent_delta_1y"):
        assert name in vd.EVENT_FIELDS, f"{name} was dropped with its twin"
        assert name in GOVERNANCE

    print("\n=== SANITY CHECK: the two retired vote twins are fully gone ===")
    print(f"  retired: {list(retired)}")
    print("  measured on the live part before removal: f_sop_dissent ~ f_sop_against_pct")
    print("  r = 0.9949 (identical coverage, 6,669 distinct values each); the auditor pair")
    print("  r = 0.9867 with `_dissent` a STRICT superset (254 cells more, 0 the other way).")
    print("  Four columns leave, not three -- both peer legs go with their parents.")
    print("  `sop_dissent` keeps all five of its derived children.")


def test_the_two_catalogues_are_separate_and_key_disjoint():
    """One prose entry must not silently serve two parts."""
    assert CATALOGUES["cube_part_fundamentals"] is CATALOGUE
    assert CATALOGUES["cube_part_governance"] is GOVERNANCE
    shared = sorted(set(CATALOGUE) & set(GOVERNANCE))
    assert not shared, f"a characteristic name is claimed by both parts: {shared}"

    # ⚠ MEASURED 2026-09-08: of the seven `cube%` tables, only these two carry `f_`-prefixed
    # columns at all (fundamentals 249, governance 106). `cube_part_betas`, `_momentum`,
    # `_prices` and `_targets` have ZERO, so they do not use the convention the catalogue is
    # keyed on and are not undocumented in this sense. This assertion is what fails -- loudly,
    # and pointing at the registry -- the day a THIRD `f_` part ships.
    assert set(CATALOGUES) == {"cube_part_fundamentals", "cube_part_governance"}

    print("\n=== SANITY CHECK: the per-part registry ===")
    for part, cat in sorted(CATALOGUES.items()):
        print(f"  {part:<26} {len(cat):>4} entries")
    print("  key-disjoint, so neither part can borrow the other's prose.")
    print("  the other four cube parts carry 0 `f_` columns and are out of scope by naming.")


def test_every_entry_is_a_complete_four_field_record():
    """A blank `why` or `tail` would pass a presence check and document nothing."""
    for name, entry in sorted(GOVERNANCE.items()):
        assert isinstance(entry, tuple) and len(entry) == 4, f"{name}: not a 4-tuple"
        family, what, why, tail = entry
        for label, text in (("family", family), ("what", what), ("why", why), ("tail", tail)):
            assert isinstance(text, str) and text.strip(), f"{name}: empty {label}"
        assert len(what) > 20, f"{name}: `what` is too short to be a description"
        assert len(why) > 40, f"{name}: `why` is too short to justify anything"

    families = sorted({v[0] for v in GOVERNANCE.values()})
    print("\n=== SANITY CHECK: every entry is complete ===")
    print(f"  {len(GOVERNANCE)} entries, {len(families)} families: {families}")
    print("  each carries family + what + why + tail; the length floors are there because a")
    print("  one-word `why` would satisfy a presence check while documenting nothing.")
