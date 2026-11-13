"""
Governance PROVISIONS, board busyness and the auditor block — phase 5's known-truth tests.

Every assertion here is about a rule that is easy to get wrong in a way no coverage number would
reveal:

  * a transition is stamped on the LATER of the two filings, because that is when it became
    public — stamping the earlier one would date a 2022 event to 2021 and leak;
  * a SILENT year is skipped, not read as a change and not allowed to hide one — the tri-state
    rule that `majority_voting` exists to honour;
  * a single known observation yields NO event, so a first disclosure is not an adoption;
  * a count is `min_count=1`, so a firm with no provision history is NaN and never a confident 0;
  * a delta whose leg the imputer INVENTED is rejected, because a linearly-filled segment has a
    constant first difference and the delta would be reporting the fill's slope;
  * the auditor flag runs on the CANONICAL firm, so a filer respelling `E&Y` is not a change.

The synthetic archive is built to make each of those visible as a VALUE, not as a count.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.provisions_features import (
    ALL_FIELDS,
    DETERIORATION_FLAGS,
    EVENT_FIELDS,
    EXPORTED_FLAGS,
    IMPROVEMENT_FLAGS,
    PEER_RELATIVE_FIELDS,
    SIGNED_FLAGS,
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
    # classified_board / majority_voting / dual_class / ceo_chair /
    # independent_chair / lead_independent_director
    plan: dict[str, dict[str, list]] = {
        # AAA — the rich case: a provision added, a chair combined, an independent chair lost.
        "AAA": {
            "classified_board": [0, 0, 1, 1, 1],
            "majority_voting": [1, 1, 1, 1, 0],
            "dual_class_shares": [0, 0, 0, 0, 0],
            "ceo_is_board_chair": [0, 0, 0, 1, 1],
            "independent_chair": [1, 1, 1, 0, 0],
            "lead_independent_director": [1, 1, 1, 1, 1],
            "pct_independent_directors": [0.90, 0.90, 0.80, 0.80, 0.80],
            "avg_other_public_boards": [1.0, 1.2, 1.5, 1.5, 1.5],
            "auditor_name": ["Ernst & Young LLP", "Ernst and Young, LLP", "E&Y", "KPMG LLP", "KPMG LLP"],
            "auditor_since_year": [N, N, N, N, 2022],
            "pct_female_directors": [0.10, 0.10, 0.20, 0.20, 0.30],
        },
        # BBB — THE HOLE. Two silent years in the middle of `classified_board`.
        "BBB": {
            "classified_board": [1, N, N, 0, 0],
            "majority_voting": [0, 0, 0, 1, 1],
            "dual_class_shares": [1, 1, 1, 1, 1],
            "ceo_is_board_chair": [1, 1, 1, 1, 1],
            "independent_chair": [0, 0, 0, 0, 0],
            "lead_independent_director": [0, 0, 1, 1, 1],
            "pct_independent_directors": [0.70, 0.72, 0.74, 0.76, 0.78],
            "avg_other_public_boards": [2.0, 2.0, 2.0, 2.0, 2.0],
            "auditor_name": ["Deloitte & Touche LLP"] * 5,
            "auditor_since_year": [1990] * 5,
            "pct_female_directors": [0.20, N, N, N, 0.30],
        },
        # CCC — a SINGLE known observation, on the LAST proxy. A first disclosure is not an
        # adoption, so this must produce no event and no count.
        "CCC": {
            "classified_board": [N, N, N, N, 1],
            "majority_voting": [N, N, N, N, N],
            "dual_class_shares": [N, N, N, N, N],
            "ceo_is_board_chair": [N, N, N, N, N],
            "independent_chair": [N, N, N, N, N],
            "lead_independent_director": [N, N, N, N, N],
            "pct_independent_directors": [N, N, N, N, 0.60],
            "avg_other_public_boards": [N, N, N, N, 3.0],
            "auditor_name": [N, N, N, N, N],
            "auditor_since_year": [N, N, N, N, N],
            "pct_female_directors": [N, N, N, N, N],
        },
        # DDD — everything disclosed and NOTHING changed: the case whose counts must be 0.0,
        # not NaN. "We looked and there was no change" is information.
        "DDD": {
            "classified_board": [0, 0, 0, 0, 0],
            "majority_voting": [1, 1, 1, 1, 1],
            "dual_class_shares": [0, 0, 0, 0, 0],
            "ceo_is_board_chair": [0, 0, 0, 0, 0],
            "independent_chair": [1, 1, 1, 1, 1],
            "lead_independent_director": [1, 1, 1, 1, 1],
            "pct_independent_directors": [0.80, 0.81, 0.82, 0.83, 0.84],
            "avg_other_public_boards": [1.1, 1.1, 1.1, 1.1, 1.1],
            "auditor_name": ["Grant Thornton LLP"] * 5,
            "auditor_since_year": [2005] * 5,
            "pct_female_directors": [0.25] * 5,
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
    value = frame.loc[pd.Timestamp(date), ticker]
    assert not isinstance(value, pd.Series | pd.DataFrame)
    return float(cast(float, value))


# --------------------------------------------------------------------------- #
def test_a_transition_is_stamped_on_the_later_filing_and_a_silent_year_is_skipped():
    """The detector's rules and the signed encoding, each asserted by VALUE on a named ticker and date."""
    hist = _archive()
    f, tally = provision_fields(hist, IDX)

    change = f["classified_board_change"]
    # 1. AAA adopts a classified board in the 2021 proxy. The value is 0 the day BEFORE that
    #    filing and +1 from the filing date on -- stamped on the LATER of the two observations,
    #    which is the only date at which the change was public.
    assert _at(change, "2021-04-30", "AAA") == 0.0
    assert _at(change, "2021-05-03", "AAA") == 1.0
    # and NOT on the earlier filing's date, which would date a 2021 disclosure to 2020.
    assert _at(change, "2020-05-04", "AAA") == 0.0

    # 2. THE HOLE. BBB discloses a classified board in 2019, says NOTHING in 2020 or 2021, and
    #    discloses none in 2022. So its only pair is 2019 <-> 2022. The silent years neither
    #    manufacture a transition NOR get a 0.0 -- they are UNKNOWN.
    assert np.isnan(_at(change, "2020-05-04", "BBB")), "a silent year invented a comparison"
    assert np.isnan(_at(change, "2021-05-03", "BBB"))
    # ...and the real change is not hidden either: it is detected ACROSS the gap, as -1, on the
    # 2022 filing, the first date at which "no longer classified" was public.
    assert _at(change, "2022-05-02", "BBB") == -1.0, "the change across the gap was lost"

    # 3. CCC's only observation of `classified_board` is its LAST proxy. A first disclosure is
    #    not an adoption: no pair, so NaN -- never 0.0.
    assert np.isnan(_at(change, "2023-05-02", "CCC")), "a first disclosure became an adoption"

    # 4. every folded provision ships as ONE signed column: +1 when the value goes 0 -> 1,
    #    -1 when it goes 1 -> 0, 0 when it stays, NaN when never disclosed twice
    majority = f["majority_voting_change"]
    assert _at(majority, "2022-05-02", "BBB") == 1.0, "0 -> 1 must read +1"
    assert _at(majority, "2023-05-02", "AAA") == -1.0, "1 -> 0 must read -1"
    assert _at(majority, "2022-05-02", "AAA") == 0.0, "no change must read 0"
    assert np.isnan(_at(majority, "2023-05-02", "CCC")), "a firm that never disclosed it must stay NaN"
    assert _at(f["lead_independent_director_change"], "2021-05-03", "BBB") == 1.0
    assert _at(f["independent_chair_change"], "2022-05-02", "AAA") == -1.0
    assert _at(f["ceo_board_chair_change"], "2022-05-02", "AAA") == 1.0
    folded = {m for pair in SIGNED_FLAGS.values() for m in pair}
    assert not folded & set(f), "a folded component was shipped as its own column"

    # the raw detector, checked directly: one row per adjacent KNOWN pair, not per change
    raw = _transitions(hist, "classified_board")
    assert raw is not None
    assert set(raw.columns) == {"ticker", "as_of", "classified_board__up", "classified_board__down"}
    assert len(raw[raw["ticker"] == "BBB"]) == 2, "BBB has 3 known observations -> 2 pairs"
    assert len(raw[raw["ticker"] == "CCC"]) == 0, "1 known observation -> 0 pairs"

    print("\n=== SANITY CHECK: the transition detector and the signed columns ===")
    print("  AAA classified_board 0->1 in the 2021 proxy: 0.0 on 2021-04-30, +1.0 on 2021-05-03,")
    print("  and 0.0 on the EARLIER filing's date -- the event is dated when it became public.")
    print("  BBB: two silent years read NaN, and the 1->0 change is still detected ACROSS them as -1")
    print("  on 2022-05-02 -- skipped, not hidden. CCC: one known observation -> NaN, not an adoption.")
    print("  majority_voting_change: BBB 0->1 reads +1, AAA 1->0 reads -1, DDD unchanged reads 0, CCC NaN.")
    print(
        f"  events: classified_board_added={tally['events: classified_board_added']}, "
        f"classified_board_removed={tally['events: classified_board_removed']}"
    )
    print("  CONCLUSION: later-filing stamping, tri-state safety, the no-pair rule and the signed encoding all hold.")


def test_the_counts_are_min_count():
    """A count of zero events is 0.0; NO provision history is NaN. The difference is the test."""
    f, tally = provision_fields(_archive(), IDX)
    det, imp = f["governance_deterioration_count"], f["governance_improvement_count"]
    d = "2022-05-02"  # after AAA's chair combination and independent-chair loss

    # 1. DDD disclosed every provision and changed none -> 0.0, which is a FACT, not a gap.
    assert _at(det, d, "DDD") == 0.0
    assert _at(imp, d, "DDD") == 0.0

    # 2. CCC has no adjacent pair of anything -> NaN. ⚠ `sum()` would have said 0.0 here, and
    #    "this board deteriorated on zero counts" is a claim we have no evidence for.
    assert np.isnan(_at(det, d, "CCC")), "a firm with no provision history got a count of 0"
    assert np.isnan(_at(imp, d, "CCC"))

    # 3. AAA on that date: ceo_became_board_chair AND independent_chair_lost both fired at the
    #    2022 proxy -> deterioration 2, improvement 0, net -2.
    assert _at(det, d, "AAA") == 2.0
    assert _at(imp, d, "AAA") == 0.0

    # 4. BBB at the 2022 proxy improved twice (classified board removed, majority voting added)
    assert _at(imp, d, "BBB") == 2.0
    assert _at(det, d, "BBB") == 0.0

    # 5. the counts INHERIT their components' expiry rather than being aged separately: past
    #    the horizon on the last proxy every component is NaN, so the count must be too
    late = pd.Timestamp("2023-05-01") + pd.Timedelta(days=GOVERNANCE_EVENT_MAX_AGE_DAYS + 30) + pd.offsets.BDay(0)
    comp = [frame for name, frame in f.items() if name in DETERIORATION_FLAGS]
    assert all(np.isnan(float(cast(float, c.loc[late, "AAA"]))) for c in comp)
    assert np.isnan(float(cast(float, det.loc[late, "AAA"]))), "the count outlived its own components"
    assert all(np.isnan(float(cast(float, f[name].loc[late, "AAA"]))) for name in SIGNED_FLAGS), "a signed column outlived its source"

    print("\n=== SANITY CHECK: the aggregate counts ===")
    print("  DDD (all disclosed, nothing changed) -> deterioration 0.0 / improvement 0.0")
    print("  CCC (no adjacent pair at all)         -> NaN, NOT 0.0")
    print("  AAA on 2022-05-02 (chair combined + independent chair lost) -> det 2, imp 0")
    print("  BBB on 2022-05-02 (classified board removed + majority voting added) -> imp 2, det 0")
    print(f"  members summed: {tally['deterioration count: members summed']} deterioration, {tally['improvement count: members summed']} improvement")
    print(f"  past the {GOVERNANCE_EVENT_MAX_AGE_DAYS}d horizon the count is NaN with its components.")
    print("  CONCLUSION: min_count=1 semantics hold. Validated.")


def test_the_independence_delta_is_the_annual_change_in_the_independent_share():
    """AAA goes 0.90 -> 0.80 between the 2020 and 2021 proxies; DDD drifts +1pp a year."""
    f, tally = provision_fields(_archive(), IDX)
    delta = f["board_independence_delta_1y"]

    assert "board_independence_drop_10pp" not in f
    assert _at(delta, "2021-05-03", "AAA") == pytest.approx(-0.10)
    assert _at(delta, "2021-05-03", "DDD") == pytest.approx(0.01, abs=1e-6)
    assert np.isnan(_at(delta, "2019-05-01", "AAA")), "no pair yet -> unknown, not 0.0"

    print("\n=== SANITY CHECK: board_independence_delta_1y ===")
    print("  AAA 0.90 -> 0.80 = -0.10; DDD +0.01/yr = +0.01; before the first pair -> NaN, not 0.0")
    print(
        f"  pairs {tally['pct_independent_directors delta: adjacent pairs']}, "
        f"rejected for an interpolated leg "
        f"{tally['pct_independent_directors delta: rejected (a leg was interpolated)']}"
    )
    print("  CONCLUSION: the delta is the plain annual change and preserves the unknown.")


def test_the_female_share_change_is_measured_against_the_previous_three_proxies():
    """The current share minus the mean share of the previous three proxies."""
    f, _ = provision_fields(_archive(), IDX)
    change = f["female_board_chg_pct"]

    assert _at(change, "2021-05-03", "AAA") == pytest.approx(0.10), "0.20 against the mean of 0.10, 0.10"
    assert _at(change, "2023-05-02", "AAA") == pytest.approx(0.30 - (0.10 + 0.20 + 0.20) / 3), "the previous three proxies are 2020, 2021 and 2022"
    assert _at(change, "2021-05-03", "DDD") == 0.0
    assert np.isnan(_at(change, "2019-05-02", "AAA")), "no earlier proxy -> blank"
    assert np.isnan(_at(change, "2023-05-02", "BBB")), (
        "BBB's only earlier proxy is 1,461 days back, beyond the age cap -> blank, not an old comparison"
    )
    assert np.isnan(_at(change, "2023-05-02", "CCC")), "never disclosed -> blank"
    late = pd.Timestamp("2023-05-01") + pd.Timedelta(days=GOVERNANCE_EVENT_MAX_AGE_DAYS + 30) + pd.offsets.BDay(0)
    assert np.isnan(_at(change, str(late.date()), "AAA")), "an event outlived its horizon"

    print("\n=== SANITY CHECK: female_board_chg_pct ===")
    print("  AAA 2021: 0.20 - mean(0.10, 0.10) = +0.10; 2023: 0.30 - mean(0.10, 0.20, 0.20) = +0.133 (fraction units).")
    print("  BBB: its only earlier proxy is beyond the age cap -> blank. CCC: never disclosed -> blank.")
    print("  CONCLUSION: a difference of shares against whichever of the previous three proxies exist. Validated.")


def test_a_delta_across_an_interpolated_leg_is_rejected():
    """⚠ THE PROVENANCE GATE. Two thirds of `board_busyness_delta_1y`'s live pairs had a leg the
    imputer invented, and a linearly-filled segment has a CONSTANT first difference -- so the
    delta would report the fill's slope. Both legs must be disclosed."""
    clean = _archive()
    dirty = _archive(avg_other_public_boards=[("AAA", 2)])  # the 2021 leg is interpolated

    f_clean, t_clean = provision_fields(clean, IDX)
    f_dirty, t_dirty = provision_fields(dirty, IDX)

    clean_d, dirty_d = (f_clean["board_busyness_delta_1y"], f_dirty["board_busyness_delta_1y"])
    # AAA busyness runs 1.0, 1.2, 1.5, 1.5, 1.5. Fully disclosed, all three deltas exist.
    assert _at(clean_d, "2020-05-04", "AAA") == pytest.approx(0.2)
    assert _at(clean_d, "2021-05-03", "AAA") == pytest.approx(0.3)
    assert _at(clean_d, "2022-05-02", "AAA") == pytest.approx(0.0)

    # Marking the 2021 VALUE interpolated rejects the two deltas that touch it: the one landing
    # ON it, and the one landing on the next filing, whose earlier leg it is.
    rejected = "avg_other_public_boards delta: rejected (a leg was interpolated)"
    assert t_clean[rejected] == 0
    assert t_dirty[rejected] == 2, "one interpolated value invalidates the two deltas it touches"

    # Rejected measurements immediately end the prior change, before its age limit.
    assert np.isnan(_at(dirty_d, "2021-05-03", "AAA"))
    assert np.isnan(_at(dirty_d, "2022-05-02", "AAA"))
    # the LEVEL keeps the fill either way -- it is the first difference that is unsound, not the
    # value: a carried board average is a defensible estimate of a standing fact
    assert _at(f_dirty["board_busyness"], "2021-05-03", "AAA") == pytest.approx(1.5)

    print("\n=== SANITY CHECK: the delta provenance gate ===")
    print("  AAA busyness 1.0, 1.2, 1.5, 1.5, 1.5 -> deltas +0.2, +0.3, 0.0 when every leg is")
    print("  disclosed. Marking the 2021 value INTERPOLATED rejects the 2 deltas touching it.")
    print("  The 2021 filing immediately reads NaN; the prior +0.2 cannot bridge a rejected pair.")
    print("  The LEVEL stays 1.5: the fill is a fair estimate of a standing fact, its first")
    print("  difference is not.")
    print("  CONCLUSION: a delta is only ever computed between two DISCLOSED legs. Validated.")


def test_the_auditor_flag_runs_on_the_canonical_firm_and_tenure_is_disclosed_or_blank():
    """Spelling drift is not a change, a start year fills its whole run, and an undisclosed run has no tenure."""
    f, tally = provision_fields(_archive(), IDX)
    changed, tenure = f["auditor_changed"], f["auditor_tenure"]
    big4 = f["auditor_is_big4"]
    assert "auditor_tenure_censored" not in f

    # AAA's raw string changes at EVERY one of its first four proxies
    # (`Ernst & Young LLP` -> `Ernst and Young, LLP` -> `E&Y` -> `KPMG LLP`) but the FIRM
    # changes exactly once. Three of those four are the filer respelling its own auditor.
    assert _at(changed, "2020-05-04", "AAA") == 0.0, "a respelling was read as a change"
    assert _at(changed, "2021-05-03", "AAA") == 0.0, "`E&Y` was read as a different firm"
    assert _at(changed, "2022-05-02", "AAA") == 1.0, "the real EY -> KPMG change was missed"
    assert _at(changed, "2023-05-02", "AAA") == 0.0

    # BBB discloses `auditor_since_year` = 1990 on every proxy
    assert _at(tenure, "2023-05-02", "BBB") == pytest.approx(33.3, abs=0.2)
    # AAA's KPMG run discloses 2022 only on its LAST proxy; the earlier KPMG proxy takes it too
    assert _at(tenure, "2022-05-02", "AAA") == pytest.approx(121 / 365.25, abs=0.02), "a start year must fill the whole run"
    assert _at(tenure, "2023-05-02", "AAA") == pytest.approx(486 / 365.25, abs=0.02)
    # the EY run never discloses it -> blank, and the previous run's value is never carried in
    assert np.isnan(_at(tenure, "2021-05-03", "AAA")), "an undisclosed run must have no tenure"

    # big-4 membership, on the canonical firm rather than the string
    assert _at(big4, "2023-05-02", "BBB") == 1.0  # Deloitte
    assert _at(big4, "2023-05-02", "DDD") == 0.0  # Grant Thornton
    assert np.isnan(_at(big4, "2023-05-02", "CCC"))  # no auditor named -> unknown, not False

    assert tally["auditor: changes detected (canonical firm)"] == 1
    assert tally["auditor: tenure disclosed"] == 12  # BBB + DDD, 5 proxies each, and AAA's two KPMG proxies
    assert tally["auditor: tenure blank (run never discloses it)"] == 3  # AAA's EY run

    print("\n=== SANITY CHECK: the auditor block ===")
    print(
        "  AAA: 4 distinct raw strings, 1 canonical change. The three respellings "
        "(`Ernst and Young, LLP`, `E&Y`) read as 0.0; the EY -> KPMG change reads 1.0."
    )
    print("  tenure: BBB disclosed 1990 -> 33.3y; AAA KPMG run takes 2022 from its last proxy; the undisclosed EY run is blank.")
    print("  big4: Deloitte 1.0, Grant Thornton 0.0, no auditor named -> NaN (not False).")
    print(f"  {tally['auditor: tenure disclosed']} filings with a tenure, {tally['auditor: tenure blank (run never discloses it)']} blank.")
    print("  CONCLUSION: normalisation removes the spelling drift and tenure is disclosed or blank.")


def test_the_encoding_expiry_and_no_interaction_contracts():
    """The declared sets against what the builder actually produces (D16, D21)."""
    f, tally = provision_fields(_archive(), IDX)
    built = set(f)

    # 1. no field escapes the declared universe, and every declared name is reachable
    assert built <= ALL_FIELDS, f"undeclared: {sorted(built - ALL_FIELDS)}"
    assert EXPORTED_FLAGS <= ALL_FIELDS
    assert set(SIGNED_FLAGS) <= built
    assert EVENT_FIELDS <= ALL_FIELDS, f"dead event names: {sorted(EVENT_FIELDS - ALL_FIELDS)}"
    assert set(DETERIORATION_FLAGS) | set(IMPROVEMENT_FLAGS) == set(TRANSITION_FLAGS)
    assert not (set(DETERIORATION_FLAGS) & set(IMPROVEMENT_FLAGS)), "a flag signed both ways"

    # 2. ⚠ NO INTERACTIONS (D16). GPT §13 lists five products; none is built, and the guard is
    #    on the NAME so a later session cannot quietly reintroduce one.
    assert not [c for c in ALL_FIELDS if "_x_" in c]
    assert not [c for c in built if "_x_" in c]

    # 3. the four LEVELS do not expire; everything else does
    levels = ALL_FIELDS - EVENT_FIELDS
    assert levels == {"board_busyness", "ceo_is_board_chair", "auditor_tenure", "auditor_is_big4"}, sorted(levels)
    # `ceo_is_board_chair` is X05's left leg, shipped as a plain level now the product is gone
    assert "ceo_is_board_chair" in f
    late = pd.Timestamp("2023-05-01") + pd.Timedelta(days=GOVERNANCE_EVENT_MAX_AGE_DAYS + 30) + pd.offsets.BDay(0)
    assert pd.isna(f["ceo_is_board_chair"].loc[late, "AAA"]), "an annual level survived its expiry"
    assert np.isnan(float(cast(float, f["ceo_board_chair_change"].loc[late, "AAA"]))), "an event survived"

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
    print(f"  {len(ALL_FIELDS)} declared fields = {len(EVENT_FIELDS)} events + {len(levels)} levels.")
    print(
        f"  {len(TRANSITION_FLAGS)} transition flags = {len(DETERIORATION_FLAGS)} "
        f"deterioration + {len(IMPROVEMENT_FLAGS)} improvement, disjoint and exhaustive."
    )
    print("  ZERO columns contain `_x_`: GPT §13's five products are not built (D16).")
    print(f"  {len(built)} built on the fixture; both levels and events expire after {GOVERNANCE_EVENT_MAX_AGE_DAYS}d.")
    print("  0 peer legs, and a never-firing flag is dropped from the panel but kept in the count.")
    print("  CONCLUSION: what is declared is what is built. Validated.")


# --------------------------------------------------------------------------- #
def test_the_real_data_readout():
    """The live shape of all three families — the plan's required read-out, asserted."""
    try:
        from src.context import get_config_context

        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        proxy = ctx.store.load("def14a_llm")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"live def14a_llm not reachable ({e})")
    if proxy is None or proxy.empty:
        pytest.skip("def14a_llm empty")

    from src.data_aggregate.utils.governance.def14a_impute import impute_def14a

    imp, _ = impute_def14a(proxy)
    idx = pd.bdate_range("2011-01-03", "2026-09-04")
    fields, tally = provision_fields(imp, idx)

    events = {name: tally.get(f"events: {name}", 0) for name in TRANSITION_FLAGS}
    busy = fields["board_busyness"].to_numpy(dtype="float64")
    busy = busy[np.isfinite(busy)]

    # 1. the board average sits in the range the literature reports for an S&P 500 board
    assert 0.5 <= float(np.median(busy)) <= 2.5, f"median board busyness {np.median(busy):.2f}"
    # 2. the flags: most fire, and the ones that do not are DROPPED rather than shipped as a
    #    constant. The gate is measured at the filing grain, so this reads the tally.
    fired = {f: n for f, n in events.items() if n > 0}
    assert len(fired) >= 9, f"only {len(fired)} of {len(events)} flags ever fire: {events}"
    folded = {m for pair in SIGNED_FLAGS.values() for m in pair}
    for flag, n in events.items():
        if flag not in folded:
            assert (flag in fields) == (n > 0), f"{flag}: {n} events but exported={flag in fields}"
    for name, (plus, minus) in SIGNED_FLAGS.items():
        assert (name in fields) == (events[plus] + events[minus] > 0), f"{name} exported={name in fields}"
    # 3. Coverage. The plan's floor is ≥480 tickers, and it applies to every field whose source
    #    column is well filled. Four are gated on a THIN source and cannot reach it -- the floor
    #    for those is the source's own coverage, pinned here so a regression is still visible:
    #      majority_voting            72.4%                   -> 422
    #      auditor_tenure             disclosed on few proxies, blank otherwise -> 280
    #      lead_independent_director  87.3%                   -> 465
    #      avg_other_public_boards    45.8% raw, and the delta needs two DISCLOSED legs -> 427
    #    ⚠ `board_busyness` CANNOT REACH 480 AND NEVER COULD -- the default floor was misapplied
    #    to it, which is a bug in this test rather than in the data. Measured 2026-09-09:
    #    488 tickers have proxies, and **10 of them carry no `avg_other_public_boards` on any
    #    filing** (ERIE with 30 proxies and 364 director rows, none disclosing other public
    #    directorships; plus nine recent listings -- DDOG, EVRG, AMCR, ABNB, APO, TKO, KKR,
    #    RDDT, BX). So 478 is the arithmetic ceiling. The 1,095-day LEVEL horizon then removes
    #    four more, and each is correct: MGM (last disclosed 1997-03-26), EG (2000-04-13),
    #    CHTR (2003-06-24) and EME (2006-04-26) have their ENTIRE series more than three years
    #    stale on an index that starts 2011-01-03, so every cell they had was forward-filled
    #    fiction. 474 is the true ceiling, floored at 470 for headroom.
    #    Not a re-extraction regression: `avg_other_public_boards` fill is 46.0% (5,676/12,343)
    #    against the 45.8% recorded before it, and child fill is 29.3%, unchanged.
    thin_floor = {
        "majority_voting_change": 410,
        "auditor_tenure": 270,
        "lead_independent_director_change": 455,
        "board_busyness_delta_1y": 410,
        "board_busyness": 470,
    }
    for name, frame in fields.items():
        n = int(frame.notna().any().sum())
        assert n >= thin_floor.get(name, 480), f"{name} covers only {n} tickers"

    print("\n=== SANITY CHECK: phase 5 on the live archive ===")
    print(f"  {len(fields)} fields over {imp['ticker'].nunique()} tickers, {len(imp):,} proxies.")
    print("  transition events over the whole history (filing grain):")
    for flag in sorted(events, key=lambda k: -events[k]):
        mark = "" if events[flag] else "   <- 0 events, NOT exported (constant column)"
        print(f"    {flag:<34}{events[flag]:>6}{mark}")
    print(
        f"  board_busyness: median {np.median(busy):.2f} other boards per director "
        f"(p10 {np.percentile(busy, 10):.2f}, p90 {np.percentile(busy, 90):.2f}) "
        f"-- inside the ~0.5-2.5 range the measure is expected in."
    )
    print(
        f"  auditor: {tally['auditor: changes detected (canonical firm)']} canonical-firm "
        f"changes; tenure {tally['auditor: tenure disclosed']:,} disclosed vs "
        f"{tally['auditor: tenure blank (run never discloses it)']:,} blank."
    )
    print("  coverage (tickers), lowest first:")
    cov = sorted((int(frame.notna().any().sum()), name) for name, frame in fields.items())
    for n, name in cov[:5]:
        print(f"    {name:<34}{n:>6}{'   <- thin source, floor is its own coverage' if name in thin_floor else ''}")
    print(f"    ...{len(fields) - 5} more, all >= {cov[5][0]}")
    print("  CONCLUSION: several flags are rare BY CONSTRUCTION, which is why the components")
    print("  ship separately from the counts; the ones with no events at all are dropped.")
