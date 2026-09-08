"""
Governance PROVISIONS, board busyness and the auditor block — phase 5's known-truth tests.

Every assertion here is about a rule that is easy to get wrong in a way no coverage number would
reveal:

  * a transition is stamped on the LATER of the two filings, because that is when it became
    public — stamping the earlier one would date a 2022 event to 2021 and leak;
  * a SILENT year is skipped, not read as a change and not allowed to hide one — the tri-state
    rule that `poison_pill` and `majority_voting` exist to honour;
  * a single known observation yields NO event, so a first disclosure is not an adoption;
  * a count is `min_count=1`, so a firm with no provision history is NaN and never a confident 0;
  * a delta whose leg the imputer INVENTED is rejected, because a linearly-filled segment has a
    constant first difference and the delta would be reporting the fill's slope;
  * the auditor flag runs on the CANONICAL firm, so a filer respelling `E&Y` is not a change.

The synthetic archive is built to make each of those visible as a VALUE, not as a count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.provisions_features import (
    ALL_FIELDS,
    DETERIORATION_FLAGS,
    EVENT_FIELDS,
    IMPROVEMENT_FLAGS,
    PEER_RELATIVE_FIELDS,
    RAW_FLAG_FIELDS,
    TRANSITION_FLAGS,
    _transitions,
    provision_fields,
)
from src.data_aggregate.utils.governance.staleness import GOVERNANCE_EVENT_MAX_AGE_DAYS

#: Five annual proxies, 2019..2023. The index runs well past the last one so the 548-day expiry
#: has somewhere to bite (2023-05-01 + 548d = 2024-10-31).
PROXIES = [pd.Timestamp(f"{y}-05-01") for y in range(2019, 2024)]
IDX = pd.bdate_range("2019-01-01", "2025-06-30")

N = np.nan


def _rows() -> list[dict]:
    """One dict per (ticker, proxy). `N` is a proxy that said NOTHING about the field."""
    # classified_board / poison_pill / majority_voting / dual_class / ceo_chair /
    # independent_chair / lead_independent_director
    plan: dict[str, dict[str, list]] = {
        # AAA — the rich case: a provision added, a chair combined, an independent chair lost.
        "AAA": {
            "classified_board":          [0, 0, 1, 1, 1],
            "poison_pill":               [N, N, N, N, N],   # never disclosed -> no pair at all
            "majority_voting":           [1, 1, 1, 1, 1],
            "dual_class_shares":         [0, 0, 0, 0, 0],
            "ceo_is_board_chair":        [0, 0, 0, 1, 1],
            "independent_chair":         [1, 1, 1, 0, 0],
            "lead_independent_director": [1, 1, 1, 1, 1],
            "pct_independent_directors": [0.90, 0.90, 0.80, 0.80, 0.80],
            "avg_other_public_boards":   [1.0, 1.2, 1.5, 1.5, 1.5],
            "auditor_name": ["Ernst & Young LLP", "Ernst and Young, LLP", "E&Y",
                             "KPMG LLP", "KPMG LLP"],
            "auditor_since_year":        [N, N, N, N, N],
        },
        # BBB — THE HOLE. Two silent years in the middle of `classified_board`, and a
        # `poison_pill` pair three years apart.
        "BBB": {
            "classified_board":          [1, N, N, 0, 0],
            "poison_pill":               [N, 1, N, N, 0],
            "majority_voting":           [0, 0, 0, 1, 1],
            "dual_class_shares":         [1, 1, 1, 1, 1],
            "ceo_is_board_chair":        [1, 1, 1, 1, 1],
            "independent_chair":         [0, 0, 0, 0, 0],
            "lead_independent_director": [0, 0, 1, 1, 1],
            "pct_independent_directors": [0.70, 0.72, 0.74, 0.76, 0.78],
            "avg_other_public_boards":   [2.0, 2.0, 2.0, 2.0, 2.0],
            "auditor_name": ["Deloitte & Touche LLP"] * 5,
            "auditor_since_year":        [1990] * 5,
        },
        # CCC — a SINGLE known observation, on the LAST proxy. A first disclosure is not an
        # adoption, so this must produce no event and no count.
        "CCC": {
            "classified_board":          [N, N, N, N, 1],
            "poison_pill":               [N, N, N, N, N],
            "majority_voting":           [N, N, N, N, N],
            "dual_class_shares":         [N, N, N, N, N],
            "ceo_is_board_chair":        [N, N, N, N, N],
            "independent_chair":         [N, N, N, N, N],
            "lead_independent_director": [N, N, N, N, N],
            "pct_independent_directors": [N, N, N, N, 0.60],
            "avg_other_public_boards":   [N, N, N, N, 3.0],
            "auditor_name":              [N, N, N, N, N],
            "auditor_since_year":        [N, N, N, N, N],
        },
        # DDD — everything disclosed and NOTHING changed: the case whose counts must be 0.0,
        # not NaN. "We looked and there was no change" is information.
        "DDD": {
            "classified_board":          [0, 0, 0, 0, 0],
            "poison_pill":               [0, 0, 0, 0, 0],
            "majority_voting":           [1, 1, 1, 1, 1],
            "dual_class_shares":         [0, 0, 0, 0, 0],
            "ceo_is_board_chair":        [0, 0, 0, 0, 0],
            "independent_chair":         [1, 1, 1, 1, 1],
            "lead_independent_director": [1, 1, 1, 1, 1],
            "pct_independent_directors": [0.80, 0.81, 0.82, 0.83, 0.84],
            "avg_other_public_boards":   [1.1, 1.1, 1.1, 1.1, 1.1],
            "auditor_name":              ["Grant Thornton LLP"] * 5,
            "auditor_since_year":        [2005] * 5,
        },
    }
    rows = []
    for tkr, fields in plan.items():
        for i, d in enumerate(PROXIES):
            row = {"ticker": tkr, "as_of": d}
            row.update({k: v[i] for k, v in fields.items()})
            rows.append(row)
    return rows


def _archive(**imputed_cells: list[tuple[str, int]]) -> pd.DataFrame:
    """The synthetic archive. `imputed_cells` marks (ticker, proxy_index) pairs as INTERPOLATED
    on a given column, i.e. what `impute_def14a` would have stamped in `<column>_imputed`."""
    df = pd.DataFrame(_rows())
    for col in ("avg_other_public_boards", "pct_independent_directors"):
        df[f"{col}_imputed"] = 0.0
    for col, cells in imputed_cells.items():
        for tkr, i in cells:
            df.loc[(df["ticker"] == tkr) & (df["as_of"] == PROXIES[i]), f"{col}_imputed"] = 1.0
    return df.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def _at(frame: pd.DataFrame, date: str, ticker: str) -> float:
    """The feature's value on a date, or NaN when the ticker has no column at all."""
    if ticker not in frame.columns:
        return np.nan
    return float(frame.loc[pd.Timestamp(date), ticker])


# --------------------------------------------------------------------------- #
def test_a_transition_is_stamped_on_the_later_filing_and_a_silent_year_is_skipped():
    """The detector's three rules, each asserted by VALUE on a named ticker and date."""
    hist = _archive()
    F, tally = provision_fields(hist, IDX)

    added = F["classified_board_added"]
    # 1. AAA adopts a classified board in the 2021 proxy. The flag is 0 the day BEFORE that
    #    filing and 1 from the filing date on -- stamped on the LATER of the two observations,
    #    which is the only date at which the change was public.
    assert _at(added, "2021-04-30", "AAA") == 0.0
    assert _at(added, "2021-05-03", "AAA") == 1.0
    # ⚠ and NOT on the earlier filing's date, which would date a 2021 disclosure to 2020.
    assert _at(added, "2020-05-04", "AAA") == 0.0

    # 2. THE HOLE. BBB discloses a classified board in 2019, says NOTHING in 2020 or 2021, and
    #    discloses none in 2022. So its only pair is 2019 <-> 2022. The silent years neither
    #    manufacture a transition NOR get a 0.0 -- they are UNKNOWN, because with one
    #    observation behind them and nothing in front there is no comparison to report.
    removed = F["classified_board_removed"]
    assert np.isnan(_at(removed, "2020-05-04", "BBB")), "a silent year invented a comparison"
    assert np.isnan(_at(removed, "2021-05-03", "BBB"))
    # ...and the real change is not hidden either: it is detected ACROSS the gap, on the 2022
    # filing, which is the first date at which "no longer classified" was public.
    assert _at(removed, "2022-05-02", "BBB") == 1.0, "the change across the gap was lost"
    # ⚠ the direction is not doubled up: a 1->0 is a removal and NOT also an addition
    assert _at(added, "2022-05-02", "BBB") == 0.0

    # 3. CCC's only observation of `classified_board` is its LAST proxy. A first disclosure is
    #    not an adoption: no pair, so no event -- and NaN, never 0.0, because "we have never
    #    seen two observations" is not "nothing changed".
    assert np.isnan(_at(added, "2023-05-02", "CCC")), "a first disclosure became an adoption"

    # 4. and the poison-pill pair three years apart is still found, on the later filing
    assert _at(F["poison_pill_removed"], "2023-05-02", "BBB") == 1.0

    # the raw detector, checked directly: one row per adjacent KNOWN pair, not per change
    raw = _transitions(hist, "classified_board")
    assert set(raw.columns) == {"ticker", "as_of", "classified_board__up",
                                "classified_board__down"}
    assert len(raw[raw["ticker"] == "BBB"]) == 2, "BBB has 3 known observations -> 2 pairs"
    assert len(raw[raw["ticker"] == "CCC"]) == 0, "1 known observation -> 0 pairs"

    print("\n=== SANITY CHECK: the transition detector ===")
    print("  AAA classified_board 0->1 in the 2021 proxy: flag 0.0 on 2021-04-30, "
          "1.0 on 2021-05-03,")
    print("  and 0.0 on the EARLIER filing's date -- the event is dated when it became public.")
    print("  BBB: two silent years read NaN, not 0.0 (no comparison exists yet), and the 1->0")
    print("  change is still detected ACROSS them on 2022-05-02 -- skipped, not hidden.")
    print("  Its poison_pill pair spans three years and still fires.")
    print("  CCC: one known observation -> NaN, not an adoption.")
    print(f"  events: classified_board_added={tally['events: classified_board_added']}, "
          f"classified_board_removed={tally['events: classified_board_removed']}")
    print("  CONCLUSION: later-filing stamping, tri-state safety and the no-pair rule all hold.")


def test_the_counts_are_min_count_and_the_net_signs_correctly():
    """A count of zero events is 0.0; NO provision history is NaN. The difference is the test."""
    F, tally = provision_fields(_archive(), IDX)
    det, imp = F["governance_deterioration_count"], F["governance_improvement_count"]
    net = F["net_governance_change"]
    d = "2022-05-02"        # after AAA's chair combination and independent-chair loss

    # 1. DDD disclosed every provision and changed none -> 0.0, which is a FACT, not a gap.
    assert _at(det, d, "DDD") == 0.0
    assert _at(imp, d, "DDD") == 0.0
    assert _at(net, d, "DDD") == 0.0

    # 2. CCC has no adjacent pair of anything -> NaN. ⚠ `sum()` would have said 0.0 here, and
    #    "this board deteriorated on zero counts" is a claim we have no evidence for.
    assert np.isnan(_at(det, d, "CCC")), "a firm with no provision history got a count of 0"
    assert np.isnan(_at(imp, d, "CCC"))

    # 3. AAA on that date: ceo_became_board_chair AND independent_chair_lost both fired at the
    #    2022 proxy -> deterioration 2, improvement 0, net -2.
    assert _at(det, d, "AAA") == 2.0
    assert _at(imp, d, "AAA") == 0.0
    assert _at(net, d, "AAA") == -2.0, "net must be improvement MINUS deterioration"

    # 4. BBB at the 2022 proxy improved twice (classified board removed, majority voting added)
    assert _at(imp, d, "BBB") == 2.0
    assert _at(det, d, "BBB") == 0.0
    assert _at(net, d, "BBB") == 2.0, "higher must mean IMPROVING"

    # 5. the counts INHERIT their components' expiry rather than being aged separately: past
    #    the horizon on the last proxy every component is NaN, so the count must be too
    late = pd.Timestamp("2023-05-01") + pd.Timedelta(days=GOVERNANCE_EVENT_MAX_AGE_DAYS + 30)
    comp = [F[f] for f in DETERIORATION_FLAGS if f in F]
    assert all(np.isnan(float(c.loc[late, "AAA"])) for c in comp)
    assert np.isnan(float(det.loc[late, "AAA"])), "the count outlived its own components"

    print("\n=== SANITY CHECK: the aggregate counts ===")
    print(f"  DDD (all disclosed, nothing changed) -> deterioration 0.0 / improvement 0.0 "
          f"/ net 0.0")
    print("  CCC (no adjacent pair at all)         -> NaN, NOT 0.0")
    print("  AAA on 2022-05-02 (chair combined + independent chair lost) -> det 2, net -2")
    print("  BBB on 2022-05-02 (classified board removed + majority voting added) -> net +2")
    print(f"  members summed: {tally['deterioration count: members summed']} deterioration, "
          f"{tally['improvement count: members summed']} improvement")
    print(f"  past the {GOVERNANCE_EVENT_MAX_AGE_DAYS}d horizon the count is NaN with its "
          f"components.")
    print("  CONCLUSION: min_count=1 semantics and the sign convention both hold. Validated.")


def test_the_independence_drop_fires_at_exactly_ten_points():
    """The boundary, on the boundary. `<=` at exactly -0.10 is the specified behaviour."""
    F, tally = provision_fields(_archive(), IDX)
    delta, drop = F["board_independence_delta_1y"], F["board_independence_drop_10pp"]

    # AAA goes 0.90 -> 0.80 between the 2020 and 2021 proxies: a delta of EXACTLY -0.10.
    assert _at(delta, "2021-05-03", "AAA") == pytest.approx(-0.10)
    # ⚠ THE FLOAT TRAP THIS TEST CAUGHT. `0.80 - 0.90` is -0.09999999999999998, so a bare
    # `<= -0.10` returns False on the textbook ten-point drop and the flag misses the exact case
    # it is named after. This assertion is the reason `_DROP_EPS` exists.
    assert _at(delta, "2021-05-03", "AAA") > -0.10, "the raw difference is NOT <= -0.10"
    assert _at(drop, "2021-05-03", "AAA") == 1.0, "the flag must fire AT the threshold"
    # DDD drifts +1pp a year -> no drop, and 0.0 rather than NaN because we measured it
    assert _at(delta, "2021-05-03", "DDD") == pytest.approx(0.01, abs=1e-6)
    assert _at(drop, "2021-05-03", "DDD") == 0.0
    # and the flag is NaN, never 0.0, where the delta itself is unknown
    assert np.isnan(_at(drop, "2019-05-01", "AAA"))

    print("\n=== SANITY CHECK: board_independence_drop_10pp ===")
    print("  AAA 0.90 -> 0.80 = a delta of exactly -0.1000 -> flag 1.0 (fires AT the threshold)")
    print("  DDD +0.01/yr -> flag 0.0; before the first pair -> NaN, not 0.0")
    print(f"  pairs {tally['pct_independent_directors delta: adjacent pairs']}, "
          f"rejected for an interpolated leg "
          f"{tally['pct_independent_directors delta: rejected (a leg was interpolated)']}")
    print("  CONCLUSION: the boundary is inclusive and the flag preserves the unknown.")


def test_a_delta_across_an_interpolated_leg_is_rejected():
    """⚠ THE PROVENANCE GATE. Two thirds of `board_busyness_delta_1y`'s live pairs had a leg the
    imputer invented, and a linearly-filled segment has a CONSTANT first difference -- so the
    delta would report the fill's slope. Both legs must be disclosed."""
    clean = _archive()
    dirty = _archive(avg_other_public_boards=[("AAA", 2)])     # the 2021 leg is interpolated

    F_clean, t_clean = provision_fields(clean, IDX)
    F_dirty, t_dirty = provision_fields(dirty, IDX)

    clean_d, dirty_d = (F_clean["board_busyness_delta_1y"],
                        F_dirty["board_busyness_delta_1y"])
    # AAA busyness runs 1.0, 1.2, 1.5, 1.5, 1.5. Fully disclosed, all three deltas exist.
    assert _at(clean_d, "2020-05-04", "AAA") == pytest.approx(0.2)
    assert _at(clean_d, "2021-05-03", "AAA") == pytest.approx(0.3)
    assert _at(clean_d, "2022-05-02", "AAA") == pytest.approx(0.0)

    # Marking the 2021 VALUE interpolated rejects the two deltas that touch it: the one landing
    # ON it, and the one landing on the next filing, whose earlier leg it is.
    rejected = "avg_other_public_boards delta: rejected (a leg was interpolated)"
    assert t_clean[rejected] == 0
    assert t_dirty[rejected] == 2, "one interpolated value invalidates the two deltas it touches"

    # ⚠ WHAT A REJECTED PAIR READS AS, and it is not NaN. The rejected filing contributes no
    # row, so `fundamentals_to_daily` forward-fills the last DISCLOSED change -- the standing
    # convention for every fiscal change in this codebase ("the latest known YoY change as of
    # d"). The +0.3 is never computed, which is the point; the reader sees the +0.2 that was
    # actually observed, and the 548-day horizon is what bounds how stale that can get.
    assert _at(dirty_d, "2021-05-03", "AAA") == pytest.approx(0.2)
    assert _at(dirty_d, "2021-05-03", "AAA") != pytest.approx(0.3)
    # and once that last real change ages out (2020-05-04 + 548d = 2021-11-03) it is NaN, rather
    # than sitting on the panel for the remaining four years of the fixture
    assert np.isnan(_at(dirty_d, "2022-05-02", "AAA"))
    # the LEVEL keeps the fill either way -- it is the first difference that is unsound, not the
    # value: a carried board average is a defensible estimate of a standing fact
    assert _at(F_dirty["board_busyness"], "2021-05-03", "AAA") == pytest.approx(1.5)

    print("\n=== SANITY CHECK: the delta provenance gate ===")
    print("  AAA busyness 1.0, 1.2, 1.5, 1.5, 1.5 -> deltas +0.2, +0.3, 0.0 when every leg is")
    print("  disclosed. Marking the 2021 value INTERPOLATED rejects the 2 deltas touching it.")
    print("  The 2021 filing then reads +0.2 -- the last DISCLOSED change, ffilled -- and never")
    print("  the +0.3 the fill's own slope would have produced; by 2022 even that has aged out.")
    print("  The LEVEL stays 1.5: the fill is a fair estimate of a standing fact, its first")
    print("  difference is not.")
    print("  CONCLUSION: a delta is only ever computed between two DISCLOSED legs. Validated.")


def test_the_auditor_flag_runs_on_the_canonical_firm_and_the_two_tenure_bases_stay_apart():
    """51% of apparent auditor changes are spelling drift, and a censored 12 is not a
    disclosed 12."""
    F, tally = provision_fields(_archive(), IDX)
    changed, tenure = F["auditor_changed"], F["auditor_tenure"]
    censored, big4 = F["auditor_tenure_censored"], F["auditor_is_big4"]

    # AAA's raw string changes at EVERY one of its first four proxies
    # (`Ernst & Young LLP` -> `Ernst and Young, LLP` -> `E&Y` -> `KPMG LLP`) but the FIRM
    # changes exactly once. Three of those four are the filer respelling its own auditor.
    assert _at(changed, "2020-05-04", "AAA") == 0.0, "a respelling was read as a change"
    assert _at(changed, "2021-05-03", "AAA") == 0.0, "`E&Y` was read as a different firm"
    assert _at(changed, "2022-05-02", "AAA") == 1.0, "the real EY -> KPMG change was missed"
    assert _at(changed, "2023-05-02", "AAA") == 0.0

    # BBB discloses `auditor_since_year` = 1990 -> the DISCLOSED basis, and the flag says so
    assert _at(censored, "2023-05-02", "BBB") == 0.0
    assert _at(tenure, "2023-05-02", "BBB") == pytest.approx(33.3, abs=0.2)
    # AAA discloses none -> the CENSORED archive run-length, which restarts at the 2022 change
    assert _at(censored, "2023-05-02", "AAA") == 1.0
    assert _at(tenure, "2023-05-02", "AAA") == pytest.approx(1.0, abs=0.05)
    # ⚠ 1.0 against BBB's 33.3 is the whole reason the flag exists: AAA's auditor may well
    # predate the archive, and a censored lower bound must not be read as a tenure.

    # big-4 membership, on the canonical firm rather than the string
    assert _at(big4, "2023-05-02", "BBB") == 1.0        # Deloitte
    assert _at(big4, "2023-05-02", "DDD") == 0.0        # Grant Thornton
    assert np.isnan(_at(big4, "2023-05-02", "CCC"))     # no auditor named -> unknown, not False

    assert tally["auditor: changes detected (canonical firm)"] == 1
    assert tally["auditor: tenure on the DISCLOSED basis"] == 10       # BBB + DDD, 5 proxies each
    assert tally["auditor: tenure on the CENSORED archive basis"] == 5

    print("\n=== SANITY CHECK: the auditor block ===")
    print("  AAA: 4 distinct raw strings, 1 canonical change. The three respellings "
          "(`Ernst and Young, LLP`, `E&Y`) read as 0.0; the EY -> KPMG change reads 1.0.")
    print("  tenure bases stay apart: BBB disclosed 1990 -> 33.3y (flag 0.0); "
          "AAA undisclosed -> 1.0y run-length (flag 1.0).")
    print("  big4: Deloitte 1.0, Grant Thornton 0.0, no auditor named -> NaN (not False).")
    print(f"  {tally['auditor: tenure on the DISCLOSED basis']} filings on the disclosed basis, "
          f"{tally['auditor: tenure on the CENSORED archive basis']} on the censored one.")
    print("  CONCLUSION: normalisation removes the spelling drift and the bases never mix.")


def test_the_encoding_expiry_and_no_interaction_contracts():
    """The declared sets against what the builder actually produces (D16, D21).

    Phase 4 is why this test exists: `pay_up_stock_down` sat in two classification sets while no
    code path produced it, and the two fields that WERE produced sat in neither and silently
    skipped their expiry.
    """
    F, tally = provision_fields(_archive(), IDX)
    built = set(F)

    # 1. no field escapes the declared universe, and every declared name is reachable
    assert built <= ALL_FIELDS, f"undeclared: {sorted(built - ALL_FIELDS)}"
    assert set(TRANSITION_FLAGS) <= ALL_FIELDS
    assert EVENT_FIELDS <= ALL_FIELDS, f"dead event names: {sorted(EVENT_FIELDS - ALL_FIELDS)}"
    assert RAW_FLAG_FIELDS <= ALL_FIELDS, f"dead flag names: {sorted(RAW_FLAG_FIELDS - ALL_FIELDS)}"
    # ⚠ and NOT `RAW_FLAG_FIELDS <= EVENT_FIELDS`, which is the invariant the vote and pay
    # modules hold. Three flags here describe a standing STATE rather than an act, so they are
    # raw in encoding and levels in time; expiring them would delete a fact still true.
    assert RAW_FLAG_FIELDS - EVENT_FIELDS == {"auditor_is_big4", "auditor_tenure_censored",
                                              "ceo_is_board_chair"}
    assert set(DETERIORATION_FLAGS) | set(IMPROVEMENT_FLAGS) == set(TRANSITION_FLAGS)
    assert not (set(DETERIORATION_FLAGS) & set(IMPROVEMENT_FLAGS)), "a flag signed both ways"

    # 2. ⚠ NO INTERACTIONS (D16). GPT §13 lists five products; none is built, and the guard is
    #    on the NAME so a later session cannot quietly reintroduce one.
    assert not [c for c in ALL_FIELDS if "_x_" in c]
    assert not [c for c in built if "_x_" in c]

    # 3. the five LEVELS do not expire; everything else does
    levels = ALL_FIELDS - EVENT_FIELDS
    assert levels == {"board_busyness", "ceo_is_board_chair", "auditor_tenure",
                      "auditor_tenure_censored", "auditor_is_big4"}, sorted(levels)
    # `ceo_is_board_chair` is X05's left leg, shipped as a plain level now the product is gone
    assert "ceo_is_board_chair" in F
    late = pd.Timestamp("2023-05-01") + pd.Timedelta(days=GOVERNANCE_EVENT_MAX_AGE_DAYS + 30)
    assert float(F["ceo_is_board_chair"].loc[late, "AAA"]) == 1.0, "a level was expired"
    assert np.isnan(float(F["ceo_became_board_chair"].loc[late, "AAA"])), "an event survived"

    # 4. NOTHING here gets a peer leg -- measured, not assumed (see PEER_RELATIVE_FIELDS)
    assert PEER_RELATIVE_FIELDS == frozenset()

    # 5. a flag that never fires is not exported, but is still summed into its count
    assert tally["events: dual_class_added"] == 0
    assert "dual_class_added" not in built
    assert tally["NOT exported (0 events, constant column): dual_class_added"] == 1
    assert tally["deterioration count: members summed"] == len(DETERIORATION_FLAGS)

    # 6. optional-source semantics: no archive -> empty, never a raise
    empty, why = provision_fields(None, IDX)
    assert empty == {}
    assert any("no def14a" in k for k in why)

    print("\n=== SANITY CHECK: the encoding, expiry and interaction contracts ===")
    print(f"  {len(ALL_FIELDS)} declared fields = {len(EVENT_FIELDS)} events "
          f"+ {len(levels)} levels; {len(RAW_FLAG_FIELDS)} are raw 1/0 flags.")
    print(f"  {len(TRANSITION_FLAGS)} transition flags = {len(DETERIORATION_FLAGS)} "
          f"deterioration + {len(IMPROVEMENT_FLAGS)} improvement, disjoint and exhaustive.")
    print("  ZERO columns contain `_x_`: GPT §13's five products are not built (D16).")
    print(f"  {len(built)} built on the fixture; a level survives the "
          f"{GOVERNANCE_EVENT_MAX_AGE_DAYS}d horizon and an event does not.")
    print("  0 peer legs, and a never-firing flag is dropped from the panel but kept in "
          "the count.")
    print("  CONCLUSION: what is declared is what is built. Validated.")


# --------------------------------------------------------------------------- #
def test_the_real_data_readout():
    """The live shape of all three families — the plan's required read-out, asserted."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        proxy = ctx.store.load("def14a_llm")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"live def14a_llm not reachable ({e})")
    if proxy is None or proxy.empty:
        pytest.skip("def14a_llm empty")

    from src.data_aggregate.utils.governance.def14a_impute import impute_def14a

    imp, _ = impute_def14a(proxy)
    idx = pd.bdate_range("2011-01-03", "2026-09-04")
    F, tally = provision_fields(imp, idx)

    events = {f: tally.get(f"events: {f}", 0) for f in TRANSITION_FLAGS}
    busy = F["board_busyness"].to_numpy(dtype="float64")
    busy = busy[np.isfinite(busy)]

    # 1. the board average sits in the range the literature reports for an S&P 500 board
    assert 0.5 <= float(np.median(busy)) <= 2.5, f"median board busyness {np.median(busy):.2f}"
    # 2. the thirteen flags: most fire, and the ones that do not are DROPPED rather than shipped
    #    as a constant. The gate is measured at the filing grain, so this reads the tally.
    fired = {f: n for f, n in events.items() if n > 0}
    assert len(fired) >= 10, f"only {len(fired)} of 13 flags ever fire: {events}"
    for flag, n in events.items():
        assert (flag in F) == (n > 0), f"{flag}: {n} events but exported={flag in F}"
    # 3. Coverage. The plan's floor is ≥480 tickers, and it applies to every field whose source
    #    column is well filled. Four are gated on a THIN source and cannot reach it -- the floor
    #    for those is the source's own coverage, pinned here so a regression is still visible:
    #      poison_pill                20.0% of modern proxies -> 172 tickers
    #      majority_voting            72.4%                   -> 422
    #      lead_independent_director  87.3%                   -> 465
    #      avg_other_public_boards    45.8% raw, and the delta needs two DISCLOSED legs -> 427
    thin_floor = {"poison_pill_removed": 150, "majority_voting_added": 410,
                  "majority_voting_removed": 410, "lead_independent_director_added": 455,
                  "lead_independent_director_lost": 455, "board_busyness_delta_1y": 410}
    for name, frame in F.items():
        n = int(frame.notna().any().sum())
        assert n >= thin_floor.get(name, 480), f"{name} covers only {n} tickers"

    print("\n=== SANITY CHECK: phase 5 on the live archive ===")
    print(f"  {len(F)} fields over {imp['ticker'].nunique()} tickers, "
          f"{len(imp):,} proxies.")
    print("  transition events over the whole history (filing grain):")
    for f in sorted(events, key=lambda k: -events[k]):
        mark = "" if events[f] else "   <- 0 events, NOT exported (constant column)"
        print(f"    {f:<34}{events[f]:>6}{mark}")
    print(f"  board_busyness: median {np.median(busy):.2f} other boards per director "
          f"(p10 {np.percentile(busy, 10):.2f}, p90 {np.percentile(busy, 90):.2f}) "
          f"-- inside the ~0.5-2.5 range the measure is expected in.")
    print(f"  auditor: {tally['auditor: changes detected (canonical firm)']} canonical-firm "
          f"changes; tenure {tally['auditor: tenure on the DISCLOSED basis']:,} disclosed vs "
          f"{tally['auditor: tenure on the CENSORED archive basis']:,} censored.")
    print("  coverage (tickers), lowest first:")
    cov = sorted(((int(f.notna().any().sum()), n) for n, f in F.items()))
    for n, name in cov[:5]:
        print(f"    {name:<34}{n:>6}"
              f"{'   <- thin source, floor is its own coverage' if name in thin_floor else ''}")
    print(f"    ...{len(F) - 5} more, all >= {cov[5][0]}")
    print("  CONCLUSION: several flags are rare BY CONSTRUCTION, which is why the components")
    print("  ship separately from the counts; the ones with no events at all are dropped.")
