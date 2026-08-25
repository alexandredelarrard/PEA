"""
test_clusters.py  (tests/validate/fundamentals/)
--------------------------------------------------------------------------------------------
The RANKING, on a synthetic ledger. No DB, no CLI.

Two things are pinned here and they are different in kind:

  * that `(ticker, field)` is the cluster key -- a STRUCTURAL claim. MCD `capex` tripping nine
    checks is one job, and if that ever collapses to nine again the whole loop is back to
    ordering work alphabetically;
  * that the weights rank MCD above VRT -- a POLICY claim, and one that has already been
    retuned once (corroboration was added after volume alone ranked a 2-check cluster above a
    10-check one). It is pinned so the next retune is a deliberate edit with a failing test to
    update, rather than a number that drifts.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.validate.fundamentals.clusters import (
    CLUSTER_WIDE, LIKELY_CHECK, LIKELY_FILER, OPEN, REOPENED, SEVERITY_WEIGHTS,
    TIER_WEIGHTS, WONTFIX, build_clusters, build_families, corroboration, derive_status,
    settled_clusters)
from src.validate.fundamentals.finding import cluster_id


class _Fix:
    """The minimum of a `FixRecord` that `settled_clusters` reads.

    A stand-in rather than the real dataclass: `clusters.py` deliberately knows nothing about
    the ledger layer, and a test that imported `FixRecord` here would quietly assert a
    coupling the module does not have.
    """

    def __init__(self, queued_before: int = 55, queued_after: int = 4) -> None:
        self.layer, self.root_cause = "extraction", "route 1 took a sibling total"
        self.commit_sha, self.test_path = "2fb6ef2", "tests/data_extract/test_x.py"
        self.queued_before, self.queued_after = queued_before, queued_after


def _waiver(check: str, at_decision: int, note: str = "measured at 8.3% vs 3.5% peer median"):
    """One `fundamentals_check_status` row as `status_map` hands it to the cluster build."""
    return {check: {"cluster_id": "c", "check_name": check, "status": WONTFIX,
                    "note": note, "findings_at_decision": at_decision}}


def _rows(*specs) -> pd.DataFrame:
    """A findings frame from `(ticker, field, check, severity, tier, period_key)` tuples."""
    return pd.DataFrame([{
        "run_date": pd.Timestamp("2026-08-24"), "run_id": "run0001",
        "cluster_id": cluster_id(ticker, field), "check_name": check,
        "ticker": ticker, "field": field, "period_key": period,
        "finding_id": f"{ticker}{field}{check}{period}",
        "tier": tier, "severity": severity, "substrate": "facts",
        "edgar_url": f"https://sec.gov/{ticker}", "detail": '{"why": "the stated mechanism"}',
    } for ticker, field, check, severity, tier, period in specs])


#: MCD `capex` as run 2 measured it: 54 findings from NINE checks over six periods. Both
#: numbers matter -- nine checks is the corroboration, 54 findings is what drives the score.
_MCD = _rows(*[("MCD", "capex", f"check_{c}", "high", 2, f"20{19 + q}-06-30")
               for c in range(9) for q in range(6)])

#: VRT: 7 Tier-1 criticals on one field. The case D4 was flagged against.
_VRT = _rows(*[("VRT", "totalAssets", "cross_identity", "critical", 1, f"20{19 + i}-06-30")
               for i in range(7)])


def test_nine_checks_agreeing_is_ONE_cluster_not_nine_issues() -> None:
    """The structural claim. `check_name` is evidence inside a cluster, never a key."""
    clusters = build_clusters(_MCD, roster_size=54)

    print(f"\n{len(_MCD)} findings across {_MCD['check_name'].nunique()} checks "
          f"-> {len(clusters)} cluster(s)")
    print(f"  {clusters[0].ticker} {clusters[0].field}: "
          f"{clusters[0].checks_agreeing} checks agreeing, score {clusters[0].score}, "
          f"periods {clusters[0].period_range}")
    print("  SANITY: nine checks agreeing is one job with nine witnesses. Keying the queue "
          "on the witness made the same fix look like nine separate jobs.")
    assert len(clusters) == 1
    assert clusters[0].findings == 54 and clusters[0].checks_agreeing == 9
    assert clusters[0].period_range == "2019-06-30..2024-06-30"


def test_the_score_multiplies_volume_by_CORROBORATION() -> None:
    """The policy, pinned. Retuning it should FAIL this test, not slip past it."""
    clusters = build_clusters(pd.concat([_MCD, _VRT], ignore_index=True), roster_size=54)
    ranked = {c.ticker: c for c in clusters}
    mcd, vrt = ranked["MCD"], ranked["VRT"]

    expected_mcd = 54 * SEVERITY_WEIGHTS["high"] * TIER_WEIGHTS[2] * corroboration(9)
    expected_vrt = 7 * SEVERITY_WEIGHTS["critical"] * TIER_WEIGHTS[1] * corroboration(1)
    print(f"\nMCD capex       54 x high x T2 x {corroboration(9):.2f} (9 checks) "
          f"= {mcd.score:.0f} (expected {expected_mcd:.0f})")
    print(f"VRT totalAssets  7 x critical x T1 x {corroboration(1):.2f} (1 check)  "
          f"= {vrt.score:.0f} (expected {expected_vrt:.0f})")
    print(f"  ranked first: {clusters[0].ticker} {clusters[0].field}")
    print("  SANITY: the weights are module constants printed in every report, and they are "
          "meant to be retuned once somebody has read a list and disagreed.")
    assert mcd.score == expected_mcd and vrt.score == expected_vrt
    assert clusters[0].ticker == "MCD"


def test_ten_checks_agreeing_outranks_more_findings_from_two() -> None:
    """The retune that produced the corroboration term, pinned to the case that caused it.

    Volume-only scoring put HCA `minorityInterest` (62 findings, TWO checks, 244) above MCD
    `capex` (55 findings, TEN checks, 148) on calibration run 3. One check firing 62 times is
    one opinion repeated; ten checks agreeing is ten arguments for the same conclusion.
    """
    loud = _rows(*[("HCA", "minorityInterest", f"check_{c}", "high", 2, f"20{10 + q}-06-30")
                   for c in range(2) for q in range(31)])          # 62 findings, 2 checks
    corroborated = _rows(*[("MCD", "capex", f"check_{c}", "high", 2, f"20{15 + q}-06-30")
                           for c in range(10) for q in range(5)])  # 50 findings, 10 checks
    clusters = build_clusters(pd.concat([loud, corroborated], ignore_index=True),
                              roster_size=54)
    by_ticker = {c.ticker: c for c in clusters}

    print(f"\nHCA  {by_ticker['HCA'].findings} findings, "
          f"{by_ticker['HCA'].checks_agreeing:2d} checks -> {by_ticker['HCA'].score:.0f}")
    print(f"MCD  {by_ticker['MCD'].findings} findings, "
          f"{by_ticker['MCD'].checks_agreeing:2d} checks -> {by_ticker['MCD'].score:.0f}")
    print(f"  ranked first: {clusters[0].ticker} {clusters[0].field}")
    print("  SANITY: MCD wins on FEWER findings because ten independent checks agree. That is "
          "the strongest prior an agent gets before opening a filing.")
    assert clusters[0].ticker == "MCD"
    assert by_ticker["MCD"].findings < by_ticker["HCA"].findings


def test_corroboration_cannot_rescue_an_all_info_cluster() -> None:
    """0 x anything is 0. Ten checks agreeing that something is benign is still benign."""
    benign = _rows(*[("KO", "totalRevenue", f"check_{c}", "info", 1, f"20{10 + q}-06-30")
                     for c in range(10) for q in range(6)])
    cluster = build_clusters(benign, roster_size=54)[0]

    print(f"\n{cluster.findings} info findings across {cluster.checks_agreeing} checks "
          f"-> score {cluster.score:.0f}, is_work={cluster.is_work}")
    print("  SANITY: the multiplier scales the base score, and an all-info base is 0. A "
          "corroboration BONUS added instead of multiplied would have floated this to the top.")
    assert cluster.score == 0.0 and cluster.is_work is False


def test_an_all_info_cluster_scores_zero_and_cannot_outrank_real_work() -> None:
    """`info` is weighted 0, not small. Otherwise `restatement_ledger` would top the list."""
    noise = _rows(*[("KO", "totalRevenue", "restatement_ledger", "info", 3, f"20{10 + i}-06-30")
                    for i in range(40)])
    clusters = build_clusters(pd.concat([noise, _VRT], ignore_index=True), roster_size=54)

    print(f"\n40 info findings score {clusters[-1].score:.0f}; "
          f"7 tier-1 criticals score {clusters[0].score:.0f}")
    print(f"  ranked first: {clusters[0].ticker}  is_work={clusters[0].is_work}")
    print("  SANITY: an info finding is declared, quantified and expected, and nothing reads "
          "it as work -- so no amount of it can bury a real cluster.")
    assert clusters[0].ticker == "VRT" and clusters[-1].score == 0.0
    assert clusters[-1].is_work is False


@pytest.mark.parametrize("tickers,roster,expected", [
    (20, 54, LIKELY_CHECK),    # 37% of the roster and well over 5 tickers -- both tests pass
    (17, 54, LIKELY_CHECK),    # 31%: just over the line
    (16, 54, LIKELY_FILER),    # 29.6%: just under it. The threshold is real, not decorative
    (12, 54, LIKELY_FILER),    # 12 tickers, but only 22% of the roster
    (4, 6, LIKELY_FILER),      # 67% of the roster, but only 4 tickers
    (5, 54, LIKELY_FILER),     # 5 tickers, but 9% of the roster
    (1, 54, LIKELY_FILER),
])
def test_routing_hint_needs_BOTH_breadth_tests(tickers, roster, expected) -> None:
    """>=5 tickers AND >=30% of the roster. Either alone is a coincidence."""
    rows = _rows(*[(f"T{i}", "incomeTaxExpense", "coverage_field", "high", 2, "2020-06-30")
                   for i in range(tickers)])
    family = build_families(build_clusters(rows, roster_size=roster), roster_size=roster)[0]

    print(f"{family.breadth:22s} share={family.share:.0%}  -> {family.routing_hint}")
    assert family.routing_hint == expected


def test_a_wide_family_reads_as_our_spec_and_a_narrow_one_as_the_filer() -> None:
    """The DQC_0118 prior, made mechanical: 47 of 54 filers do not fail independently."""
    wide = _rows(*[(f"T{i}", "incomeTaxExpense", "coverage_field", "high", 2, "2020-06-30")
                   for i in range(47)])
    families = build_families(build_clusters(pd.concat([wide, _MCD], ignore_index=True),
                                             roster_size=54), roster_size=54)
    by_field = {f.field: f for f in families}

    print(f"\nincomeTaxExpense: {by_field['incomeTaxExpense'].breadth} -> "
          f"{by_field['incomeTaxExpense'].routing_hint}")
    print(f"capex           : {by_field['capex'].breadth} -> {by_field['capex'].routing_hint}")
    print("  SANITY: wide -> challenge the SPEC before opening a filing; narrow -> the filer. "
          "An agent that opens a 10-K on a wide family has spent the hour before it read the "
          "catalogue entry.")
    assert by_field["incomeTaxExpense"].routing_hint == LIKELY_CHECK
    assert by_field["capex"].routing_hint == LIKELY_FILER


def test_a_wontfix_reopens_when_the_cluster_GROWS() -> None:
    """D8: a decision cannot outlive the evidence it was made on."""
    decided = _waiver(CLUSTER_WIDE, 3, "measured at $5.2bn; 3 periods")
    same, _n, _d = derive_status("abc", 3, decided)
    grown, _n2, _d2 = derive_status("abc", 4, decided)
    none_yet, _n3, _d3 = derive_status("abc", 9, None)

    print(f"\n3 findings at decision, 3 now -> {same}")
    print(f"3 findings at decision, 4 now -> {grown}")
    print(f"no decision on file          -> {none_yet}")
    print("  SANITY: a judgement made about 3 findings is not a judgement about 30. The "
          "wontfix expires by itself, so nobody has to remember to revisit it. That is "
          "what replaces the deleted register's staleness report.")
    assert same == WONTFIX and grown == REOPENED and none_yet == OPEN


def test_a_PER_CHECK_waiver_expires_against_its_OWN_population() -> None:
    """The ripple of the widened key, and the reason it is worth having.

    Measured against the CLUSTER total a `peer_ratio` waiver would reopen every time an
    unrelated check fired once more -- which is not what anybody decided.
    """
    waived = _waiver("peer_ratio", 2)
    counts_same = {"peer_ratio": 2, "series_shape": 1}
    counts_other_grew = {"peer_ratio": 2, "series_shape": 40}
    counts_own_grew = {"peer_ratio": 3, "series_shape": 1}

    still, _n, _d = derive_status("c", 3, waived, counts_same)
    unrelated, _n2, _d2 = derive_status("c", 42, waived, counts_other_grew, unwaived_queue=40)
    own, _n3, _d3 = derive_status("c", 4, waived, counts_own_grew, unwaived_queue=1)

    print(f"\npeer_ratio 2 at decision, 2 now, cluster 3   -> {still}")
    print(f"peer_ratio 2 at decision, 2 now, cluster 42  -> {unrelated}")
    print(f"peer_ratio 2 at decision, 3 now              -> {own}")
    print("  SANITY: only its OWN growth reopens it. The middle case is OPEN because 40 "
          "unwaived queue findings are live -- not because the waiver expired.")
    assert still == WONTFIX and unrelated == OPEN and own == REOPENED


def test_a_cluster_absent_from_the_LATEST_comparable_run_is_settled() -> None:
    """The delta. "Fewer rows" is proof only between two runs of one scope."""
    before = pd.concat([_MCD, _VRT], ignore_index=True)
    after = _VRT
    mcd = cluster_id("MCD", "capex")
    closed = settled_clusters(before, after, fixes={mcd: _Fix()})

    print(f"\nbefore: {before['cluster_id'].nunique()} cluster(s); "
          f"after: {after['cluster_id'].nunique()}; settled: {len(closed)}")
    print(f"  closed: {[c.cluster_id for c in closed]} == MCD capex ({mcd}), "
          f"basis={closed[0].basis}")
    print("  SANITY: a cluster missing from a NARROWER run did not close -- it was never "
          "looked at. ledger.comparable_runs is what makes this query sound.")
    assert [c.cluster_id for c in closed] == [mcd]
    assert closed[0].basis == "clean"


# --------------------------------------------------------------------------- #
# the settlement algebra -- seven cases, each blocking a different way to fake #
# a closed defect                                                              #
# --------------------------------------------------------------------------- #

#: MCD `capex` AFTER the 2026-08-25 fix: the 4 findings that survived. Two `peer_ratio` on a
#: documented blind spot, one `series_shape` interior gap, one weighted-zero `info`. The exact
#: residue that a strict set difference calls OPEN forever.
_RESIDUE = _rows(("MCD", "capex", "peer_ratio", "medium", 2, "2019-06-30"),
                 ("MCD", "capex", "peer_ratio", "medium", 2, "2020-06-30"),
                 ("MCD", "capex", "series_shape", "medium", 3, "2011..2026"),
                 ("MCD", "capex", "catalogue_exclusion_cost", "info", 1, ""))

_MCD_ID = cluster_id("MCD", "capex")

#: The two waivers agent B recorded beside the fix, each quantified, each on its own check.
_RESIDUE_WAIVERS = {_MCD_ID: {**_waiver("peer_ratio", 2),
                              **_waiver("series_shape", 1, "1 interior-gap coverage finding")}}


def test_waived_residue_plus_a_qualifying_fix_SETTLES() -> None:
    """The case this whole design exists for. 55 -> 4, all 4 assessed and tolerated."""
    closed = settled_clusters(_MCD, _RESIDUE, waivers=_RESIDUE_WAIVERS,
                              fixes={_MCD_ID: _Fix()})

    print(f"\n{len(_MCD)} findings before -> {len(_RESIDUE)} after")
    print(f"  settled: {[(c.cluster_id, c.basis) for c in closed]}")
    print(f"  findings still on the ledger: {closed[0].findings_after}")
    print("  SANITY: it settles WITHOUT reaching zero, and says on what basis. A strict set "
          "difference calls this OPEN forever, which is how today's fix left no record.")
    assert len(closed) == 1
    assert closed[0].waived_findings == 3
    assert closed[0].waived_checks == ("peer_ratio", "series_shape")
    assert closed[0].findings_after == 4


def test_waived_residue_with_NO_fix_row_does_not_settle() -> None:
    """Decision 1. Waiving every check one at a time is the suppression register, rebuilt."""
    closed = settled_clusters(_MCD, _RESIDUE, waivers=_RESIDUE_WAIVERS, fixes={})
    cluster = build_clusters(_RESIDUE, waivers=_RESIDUE_WAIVERS, roster_size=54)[0]

    print(f"\nsettled with no fix row: {closed}")
    print(f"  the cluster instead reads: {cluster.status} "
          f"({cluster.waiver_summary})")
    print("  SANITY: tolerated, not solved -- and it says so. Without this rule a settlement "
          "can be manufactured by waiving each check in turn, with nobody having fixed "
          "anything.")
    assert closed == []
    assert cluster.status == WONTFIX


def test_a_NO_IMPROVEMENT_fix_row_cannot_settle() -> None:
    """Decision 8. Recordable, warned about on write, and powerless here.

    `settled_clusters` never sees such a row: `Ledger.qualifying_fix` filters it out, which is
    what keeps the rule in one place. This pins the consequence -- an empty `fixes` map.
    """
    closed = settled_clusters(_MCD, _RESIDUE, waivers=_RESIDUE_WAIVERS, fixes={})

    print(f"\nqualifying_fix rejected the 4->4 row, so fixes={{}} -> settled={closed}")
    print("  SANITY: permissive to record, strict to settle. The row stays readable through "
          "`fixes_for`; it simply proves nothing was closed.")
    assert closed == []


def test_an_INFO_only_residue_settles_with_NO_waiver_at_all() -> None:
    """`catalogue_exclusion_cost` needs no paperwork. If it did, the filter would be wrong.

    An `info` finding is declared, quantified and expected, its severity weight is 0, and
    nothing reads it as work -- so it can neither block a settlement nor conceal one.
    """
    info_only = _rows(("MCD", "capex", "catalogue_exclusion_cost", "info", 1, ""))
    closed = settled_clusters(_MCD, info_only, waivers={}, fixes={_MCD_ID: _Fix()})

    print(f"\nresidue: 1 `info` finding, 0 waivers -> settled={[c.cluster_id for c in closed]}"
          f", basis={closed[0].basis}")
    print("  SANITY: no waiver was needed or written. Requiring one for `info` would turn "
          "the benign residue into paperwork and would mean the queue-severity filter is "
          "not doing its job.")
    assert len(closed) == 1 and closed[0].waived_findings == 0
    assert closed[0].basis == "clean"


def test_one_unwaived_HIGH_finding_blocks_a_settlement_whatever_the_fix_says() -> None:
    """Condition 1. A fix row is not a licence to stop looking."""
    residue = pd.concat([_RESIDUE,
                         _rows(("MCD", "capex", "cross_identity", "high", 1, "2026-06-30"))],
                        ignore_index=True)
    closed = settled_clusters(_MCD, residue, waivers=_RESIDUE_WAIVERS,
                              fixes={_MCD_ID: _Fix()})
    cluster = build_clusters(residue, waivers=_RESIDUE_WAIVERS, roster_size=54)[0]

    print(f"\none unwaived `high` added -> settled={closed}, cluster reads {cluster.status}")
    print(f"  waived: {cluster.waived_findings} across {list(cluster.waived_checks)}")
    print("  SANITY: real work is still live, so the cluster is plainly OPEN -- whatever has "
          "been tolerated beside it and whatever the fix row proved.")
    assert closed == [] and cluster.status == OPEN


def test_a_waiver_that_GREW_past_its_decision_un_settles_the_cluster() -> None:
    """Condition 2. A settlement resting on an expired judgement is not a settlement."""
    grown = pd.concat([_RESIDUE,
                       _rows(("MCD", "capex", "peer_ratio", "medium", 2, "2021-06-30"))],
                      ignore_index=True)
    closed = settled_clusters(_MCD, grown, waivers=_RESIDUE_WAIVERS, fixes={_MCD_ID: _Fix()})
    cluster = build_clusters(grown, waivers=_RESIDUE_WAIVERS, roster_size=54)[0]

    print(f"\npeer_ratio was waived at 2 findings; it now has 3")
    print(f"  settled={closed}; cluster reads {cluster.status}")
    print("  SANITY: the judgement expired by itself. Nobody had to remember to revisit it, "
          "and the cluster cannot settle on it in the meantime.")
    assert closed == [] and cluster.status == REOPENED


def test_the_LEDGER_is_untouched_by_every_waiver_and_fix_row() -> None:
    """THE anti-suppression test. Everything else here reverts with `git revert`.

    If waivers or fix rows ever change what `fundamentals_check` contains, a row-count drop
    stops being proof of anything and the whole loop's premise collapses.
    """
    plain = build_clusters(_RESIDUE, roster_size=54)
    waived = build_clusters(_RESIDUE, waivers=_RESIDUE_WAIVERS, roster_size=54)
    settled_clusters(_MCD, _RESIDUE, waivers=_RESIDUE_WAIVERS, fixes={_MCD_ID: _Fix()})

    print(f"\nledger rows in: {len(_RESIDUE)}")
    print(f"  findings counted, no waivers : {plain[0].findings}")
    print(f"  findings counted, waived     : {waived[0].findings}")
    print(f"  check mix unchanged          : {dict(plain[0].checks) == dict(waived[0].checks)}")
    print("  SANITY: identical. A waived finding is still written, still counted and still "
          "fires -- it is applied when the report is RENDERED, never when the row is "
          "written. That is what keeps 'fewer rows than last time' usable as proof.")
    assert plain[0].findings == waived[0].findings == len(_RESIDUE)
    assert dict(plain[0].checks) == dict(waived[0].checks)
    assert waived[0].waived_findings == 3


def test_the_handoff_contract_carries_every_field_agent_B_needs() -> None:
    """6.0's contract. A missing field means B cannot start and must say so."""
    cluster = build_clusters(_MCD, roster_size=54)[0]
    packet = cluster.as_dict()
    required = {"cluster_id", "ticker", "field", "score", "findings", "checks_agreeing",
                "severity_mix", "tier_mix", "period_range", "routing_hint", "family_breadth",
                "edgar_url", "why"}

    print(f"\npacket keys: {sorted(packet)}")
    print(f"  checks_agreeing={packet['checks_agreeing']}")
    print(f"  why={packet['why']!r}")
    print("  SANITY: B parses JSON rather than scraping prose. Markdown is the human "
          "artifact; this is the agent artifact.")
    assert required <= set(packet)
    assert packet["why"] and packet["checks_agreeing"] and packet["edgar_url"]
