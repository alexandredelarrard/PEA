"""
test_planted_violations.py  (tests/validate/fundamentals/)
--------------------------------------------------------------------------------------------
THE BAR the plan sets for every check: plant exactly one violation, and the check fires --
while the CLEAN base stays quiet.

A check that cannot be planted cannot be trusted, and a check that fires on a clean, smooth,
internally consistent filer has a threshold bug rather than a finding. Both halves are asserted
here, per check.

No DB, no CLI, no network: `FundamentalsValidator` is instantiated against a synthetic
`Substrates` built in `conftest.py`. That is what the constructor's signature is for.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.validate.fundamentals.checks import CHECK_REGISTRY
from src.validate.fundamentals.finding import QUEUE_SEVERITIES
from src.validate.fundamentals.validator import FundamentalsValidator
from tests.validate.fundamentals.conftest import TICKER, build_substrates


def _run(substrates, names: list[str] | None = None):
    return FundamentalsValidator(substrates).run(names=names)


def _findings(substrates, check_name: str) -> pd.DataFrame:
    """Just this check's findings, so a plant is scored against its own target."""
    run = _run(substrates, names=[check_name])
    return run.findings


def _plant(catalogue, facts: pd.DataFrame, mutate) -> "object":
    """A copy of the clean facts with ONE mutation applied, rebuilt into full substrates."""
    planted = facts.copy()
    mutate(planted)
    return build_substrates(catalogue, planted)


# --------------------------------------------------------------------------- #
# the clean base                                                               #
# --------------------------------------------------------------------------- #

def test_the_clean_base_produces_no_queue_findings(clean) -> None:
    """A smooth, complete, internally consistent filer earns NOTHING at critical/high/medium.

    The most important test in the file. Every planted-violation test below is meaningless if
    the base is already noisy -- "the check fired" would prove nothing about the plant.
    """
    run = _run(clean)
    queue = run.queue

    print(f"\nclean base: {len(clean.tickers)} tickers x {len(clean.history)} events, "
          f"{len(run.findings)} finding(s) total, {len(queue)} in the QUEUE")
    if not queue.empty:
        print(queue[["check_name", "ticker", "field", "severity", "period_key"]]
              .head(20).to_string(index=False))
    abstained = [o.spec.name for o in run.outcomes if o.abstained]
    print(f"  abstained (examined nothing, NOT a pass): {abstained}")
    print("  SANITY: a filer that did nothing wrong produces an empty work queue. Anything "
          "here is a threshold bug, not a finding.")
    assert queue.empty, f"the clean base fired: {queue['check_name'].unique().tolist()}"


def test_every_registered_check_declares_a_complete_contract() -> None:
    """`CHECK_REGISTRY` round-trip: no check runs on a default it never declared.

    AGENTS.md forbids hand-listing what a registry drives, so the report, the CLI and the
    calibration pass all enumerate this. A check missing a substrate or a ceiling would run
    anyway and be invisible in every one of them.
    """
    for name, spec in CHECK_REGISTRY.items():
        assert spec.name == name
        assert spec.tier in (1, 2, 3, 4)
        assert spec.substrate in ("history", "facts")
        assert spec.grain in ("cell", "row", "series", "ticker")
        assert 0.0 <= spec.expected_fire_rate_ceiling <= 1.0
        assert spec.doc, f"{name} has no docstring first line to show in the registry listing"

    tiers = {}
    for spec in CHECK_REGISTRY.values():
        tiers[spec.tier] = tiers.get(spec.tier, 0) + 1
    print(f"\n{len(CHECK_REGISTRY)} registered checks: "
          + ", ".join(f"tier {t}={n}" for t, n in sorted(tiers.items())))
    print("  SANITY: every check declares name/tier/substrate/severity/grain/ceiling, so "
          "every consumer can enumerate rather than hand-list.")
    assert len(CHECK_REGISTRY) >= 30


# --------------------------------------------------------------------------- #
# tier 1                                                                       #
# --------------------------------------------------------------------------- #

def test_cross_identity_fires_when_the_balance_sheet_does_not_foot(catalogue,
                                                                   clean_facts) -> None:
    """Assets != Liabilities + Equity on one event."""
    def bend(facts):
        mask = ((facts["ticker"] == TICKER) & (facts["field"] == "totalAssets")
                & (facts["period_end"] == facts["period_end"].max()))
        facts.loc[mask, "value"] = facts.loc[mask, "value"] * 1.5

    out = _findings(_plant(catalogue, clean_facts, bend), "cross_identity")
    print(f"\ntotalAssets bent 1.5x on one event -> {len(out)} cross_identity finding(s) "
          f"on {sorted(out['ticker'].unique())}")
    print("  SANITY: a balance sheet that does not foot is PROVABLY wrong -- critical, and "
          "signed arithmetic, so HCA's negative equity would still pass.")
    assert len(out) == 1 and out.iloc[0]["severity"] == "critical"


def test_cross_identity_skips_a_derived_totalLiabilities(catalogue, clean_facts) -> None:
    """A `derived_identity` total is an INPUT, never corroboration (hand-off E-1).

    The identity would be `A - E + E == A`, which passes on any numbers at all -- so testing it
    there is not a weak check, it is no check, and reporting it as a pass would be a false
    claim about 901 rows across 18 tickers.
    """
    substrates = build_substrates(catalogue, clean_facts.copy())
    # bend the row AND mark its totalLiabilities derived
    last = substrates.history["as_of"].max()
    mask = (substrates.history["ticker"] == TICKER) & (substrates.history["as_of"] == last)
    substrates.history.loc[mask, "totalAssets"] *= 1.5
    substrates.codes = pd.concat([substrates.codes, pd.DataFrame([{
        "ticker": TICKER, "as_of": last, "field": "totalLiabilities",
        "dc_code": "derived_identity", "combined_into": None,
        "rejected_value": float("nan")}])], ignore_index=True)

    out = _findings(substrates, "cross_identity")
    print(f"\nsame bent row, but totalLiabilities carries `derived_identity`: "
          f"{len(out)} finding(s)")
    print("  SANITY: skipped, because A - E + E == A is arithmetic and would 'pass' on any "
          "numbers, including wrong ones.")
    assert out.empty


def test_impossible_value_flags_a_negative_top_line_without_nulling_it(catalogue,
                                                                       clean_facts) -> None:
    """APA's shape: a negative revenue. Reported, NEVER nulled."""
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "totalRevenue")
        facts.loc[facts[mask].index[-1], "value"] = -467_000_000.0

    substrates = _plant(catalogue, clean_facts, bend)
    out = _findings(substrates, "impossible_value")
    still_there = substrates.history["totalRevenue"].min()

    print(f"\ntotalRevenue planted at -$467M -> {len(out)} impossible_value finding(s); "
          f"the value is STILL IN THE TABLE at {still_there:,.0f}")
    print("  SANITY: flag-only. Only the four rules in build_history.HARD_GUARDS ever delete "
          "a value -- v2's [-1,1] ratio bound would have nulled HCA's correct negative "
          "debtToEquity, which is the 745-row lesson repeating.")
    assert len(out) == 1 and out.iloc[0]["severity"] == "high"
    assert still_there < 0


def test_unexplained_null_fires_when_a_reason_code_is_missing(catalogue, clean_facts) -> None:
    """A null cell whose reason-code row was removed."""
    substrates = build_substrates(catalogue, clean_facts.copy())
    before = len(_findings(substrates, "unexplained_null"))
    substrates.codes = substrates.codes.iloc[1:].reset_index(drop=True)
    after = _findings(substrates, "unexplained_null")

    print(f"\ndense codes -> {before} unexplained; one code row deleted -> {len(after)}")
    print("  SANITY: THE gate the whole reason-code layer exists to make passable. A null "
          "with no code is a value nobody can account for.")
    assert before == 0 and len(after) == 1


def test_grain_fires_on_a_duplicate_publication_event(catalogue, clean_facts) -> None:
    """Two rows sharing (ticker, as_of) -- the same-day collapse failing."""
    substrates = build_substrates(catalogue, clean_facts.copy())
    duplicate = substrates.history[substrates.history["ticker"] == TICKER].iloc[[0]]
    substrates.history = pd.concat([substrates.history, duplicate], ignore_index=True)

    out = _findings(substrates, "grain")
    print(f"\none publication event duplicated -> {len(out)} grain finding(s), "
          f"severity {out['severity'].unique().tolist()}")
    print("  SANITY: a ZERO-ceiling check. build_history asserts this, so a hit means the "
          "table was written outside the builder.")
    assert len(out) == 1 and out.iloc[0]["severity"] == "critical"


def test_dimensional_scope_fires_on_a_member_scoped_concept(catalogue, clean_facts) -> None:
    """DTE's shape: capex resolved through `dte:DTEElectricMember`."""
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
        facts.loc[facts[mask].index[-1], "source_concept"] = "dte:DTEElectricMember"

    out = _findings(_plant(catalogue, clean_facts, bend), "dimensional_scope")
    print(f"\ncapex resolved through a ...Member concept -> {len(out)} finding(s)")
    print("  SANITY: one subsidiary's number stored as the consolidated group's, 17% low and "
          "entirely plausible. This is the SECOND lock -- entity_scope is the first.")
    assert len(out) == 1 and out.iloc[0]["severity"] == "critical"


def test_coverage_universe_fires_for_a_roster_ticker_with_no_rows(catalogue,
                                                                  clean_facts) -> None:
    """A ticker the run was scoped to that produced nothing at all."""
    substrates = build_substrates(catalogue, clean_facts.copy())
    substrates.tickers = (*substrates.tickers, "MISSING")

    out = _findings(substrates, "coverage_universe")
    print(f"\nroster names MISSING, history has no rows for it -> {len(out)} finding(s)")
    print("  SANITY: '0 findings' and '0 tickers loaded' must never look the same.")
    assert len(out) == 1 and out.iloc[0]["ticker"] == "MISSING"


# --------------------------------------------------------------------------- #
# tier 2                                                                       #
# --------------------------------------------------------------------------- #

def test_trend_break_fires_at_3x_and_is_silent_at_29x(catalogue, clean_facts) -> None:
    """The boundary is exact: 3.1x the TRAILING MEDIAN fires, 2.9x does not.

    Planted against the trailing median, not against the previous value. Those are different
    numbers on any growing series -- the fixture compounds 5% a quarter, so the last point
    already sits ~1.26x its own trailing median, and a "2.9x" plant applied to the VALUE lands
    at 3.65x the reference and fires correctly. Testing the boundary means planting on the
    quantity the check actually compares.
    """
    from src.validate.fundamentals.checks.tier2_series import TREND_WINDOW

    def bend(multiple):
        def mutate(facts):
            mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
            index = facts[mask].index
            window = facts.loc[index[-1 - TREND_WINDOW:-1], "value"].astype(float)
            facts.loc[index[-1], "value"] = float(window.median()) * multiple
        return mutate

    fired = _findings(_plant(catalogue, clean_facts, bend(3.1)), "trend_break")
    quiet = _findings(_plant(catalogue, clean_facts, bend(2.9)), "trend_break")

    print(f"\ncapex at 3.1x its trailing median -> {len(fired)} trend_break finding(s); "
          f"at 2.9x -> {len(quiet)}")
    print("  SANITY: a flat, interpretable 3x rule with an exact boundary. It is NOT "
          "level_outlier: for a lumpy field the MAD is wide enough that a real 3x jump "
          "scores under 3.5 and is missed entirely.")
    assert len(fired) == 1 and quiet.empty


def test_basis_step_fires_when_the_route_changes_with_the_level(catalogue,
                                                                clean_facts) -> None:
    """MCD's shape: a level step at the exact boundary where `resolution_method` changes."""
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
        index = facts[mask].index[-1]
        facts.loc[index, "resolution_method"] = "statement_leaf_sum"
        facts.loc[index, "value"] *= 2.0

    out = _findings(_plant(catalogue, clean_facts, bend), "basis_step")
    print(f"\ncapex changes route AND doubles at one boundary -> {len(out)} finding(s)")
    print("  SANITY: no cross-vintage test can see this -- the filer tags the same narrow "
          "concept consistently within each era. MCD's capex steps 35.6x this way.")
    assert len(out) == 1 and out.iloc[0]["severity"] == "high"


def test_basis_step_is_silent_when_the_route_changes_without_a_step(catalogue,
                                                                    clean_facts) -> None:
    """A route change alone is ordinary. It is the COINCIDENCE that says the basis moved."""
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
        facts.loc[facts[mask].index[-1], "resolution_method"] = "statement_leaf_sum"

    out = _findings(_plant(catalogue, clean_facts, bend), "basis_step")
    print(f"\ncapex changes route with NO level step -> {len(out)} finding(s)")
    print("  SANITY: routes exist because filers differ. Only a route change WITH a step is "
          "a change in what the number means.")
    assert out.empty


def test_series_shape_flags_a_field_that_goes_dark(catalogue, clean_facts) -> None:
    """VLO's shape: capex present until a date, absent after -> `early_stop`, `high`."""
    def bend(facts):
        cutoff = facts["period_end"].max() - pd.Timedelta(days=200)
        drop = ((facts["ticker"] == TICKER) & (facts["field"] == "capex")
                & (facts["period_end"] > cutoff))
        facts.drop(facts[drop].index, inplace=True)

    out = _findings(_plant(catalogue, clean_facts, bend), "series_shape")
    ours = out[out["ticker"] == TICKER]
    print(f"\ncapex dropped from the tail -> {len(ours)} series_shape finding(s) for {TICKER}; "
          f"shape(s) {[__import__('json').loads(d)['shape'] for d in ours['detail']]}")
    print("  SANITY: coverage_field fires per CELL and can never see a shape. This is the "
          "missing dimension -- VLO's capex went dark from 2023-07 in 21 of 63 filings.")
    assert len(ours) == 1
    assert ours.iloc[0]["severity"] == "high"


def test_series_shape_flags_a_late_start(catalogue, clean_facts) -> None:
    """AAPL `totalDebt`'s shape: absent, then present from a date onward, no oracle."""
    def bend(facts):
        cutoff = facts["period_end"].min() + pd.Timedelta(days=200)
        drop = ((facts["ticker"] == TICKER) & (facts["field"] == "totalDebt")
                & (facts["period_end"] < cutoff))
        facts.drop(facts[drop].index, inplace=True)

    out = _findings(_plant(catalogue, clean_facts, bend), "series_shape")
    ours = out[out["ticker"] == TICKER]
    shapes = [__import__("json").loads(d)["shape"] for d in ours["detail"]]
    print(f"\ntotalDebt absent from the head -> {len(ours)} finding(s), shape(s) {shapes}")
    print("  SANITY: `high` because no regime_break explains it and the listing date is NOT "
          "readable from fundamentals_* -- exactly the AAPL totalDebt case, whose answer was "
          "its first bond issue on 30 April 2013.")
    assert shapes == ["late_start"] and ours.iloc[0]["severity"] == "high"


def test_peer_ratio_flags_a_value_in_the_wrong_concept_entirely(catalogue,
                                                                clean_facts) -> None:
    """BRK-B's shape: `totalDebt` resolved to a lease liability -- smooth, and 100x too small."""
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "totalDebt")
        facts.loc[mask, "value"] = facts.loc[mask, "value"] / 100.0

    out = _findings(_plant(catalogue, clean_facts, bend), "peer_ratio")
    ours = out[out["ticker"] == TICKER]
    print(f"\ntotalDebt divided by 100 across the WHOLE series -> {len(ours)} peer_ratio "
          f"finding(s) for {TICKER}")
    print("  SANITY: the series is perfectly smooth, never changes route and never steps, so "
          "every self-comparing check passes. This is the ONLY rule that catches a value "
          "resolved to an entirely wrong concept without a human noticing first.")
    assert len(ours) > 0 and set(ours["severity"]) == {"high"}


def test_peer_ratio_abstains_visibly_below_five_peers(catalogue) -> None:
    """GS's situation: the only filer in its regime. Silence must read as an ABSTENTION."""
    from tests.validate.fundamentals.conftest import make_facts

    lonely = build_substrates(catalogue, make_facts("GS"))
    run = FundamentalsValidator(lonely).run(names=["peer_ratio", "peer_ratio_abstentions"])
    peer = run.findings[run.findings["check_name"] == "peer_ratio"]
    declared = run.findings[run.findings["check_name"] == "peer_ratio_abstentions"]

    print(f"\n1 filer in the regime -> peer_ratio findings={len(peer)}, "
          f"abstentions DECLARED={len(declared)}")
    print("  SANITY: an abstention that is not in the ledger is invisible to anyone reading "
          "the ledger, and 'peer_ratio reported nothing for GS' would read as a pass.")
    assert peer.empty and len(declared) == 1


def test_scale_fires_on_an_order_of_magnitude_jump(catalogue, clean_facts) -> None:
    """ORCL's shape: a full-year figure stamped with a quarterly window."""
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "totalRevenue")
        facts.loc[facts[mask].index[-1], "value"] *= 40.0

    out = _findings(_plant(catalogue, clean_facts, bend), "scale")
    ours = out[out["ticker"] == TICKER]
    print(f"\ntotalRevenue x40 on one quarter -> {len(ours)} scale finding(s)")
    print("  SANITY: at 10x against the field's own median a units error is a likelier "
          "explanation than a business change. ORCL FY2020 Q4: $39,068M vs ~$10,439M.")
    assert len(ours) == 1


# --------------------------------------------------------------------------- #
# tier 3                                                                       #
# --------------------------------------------------------------------------- #

def test_duplicate_fact_fires_when_one_filing_contradicts_itself(catalogue,
                                                                 clean_facts) -> None:
    """ORCL's $7,623M vs $7,600M: one accession, one period, two values."""
    substrates = build_substrates(catalogue, clean_facts.copy())
    row = substrates.facts[(substrates.facts["ticker"] == TICKER)
                           & (substrates.facts["field"] == "totalRevenue")].iloc[[0]].copy()
    row["value"] = row["value"] * 1.003
    substrates.facts = pd.concat([substrates.facts, row], ignore_index=True)

    out = _findings(substrates, "duplicate_fact")
    print(f"\none accession tagging totalRevenue twice -> {len(out)} finding(s)")
    print("  SANITY: whichever value the resolver kept, it kept it by FRAME ORDER. The PK "
          "collapses these silently, and the loser is gone.")
    assert len(out) == 1


def test_cross_vintage_classifies_a_restatement_as_info(catalogue, clean_facts) -> None:
    """Two AS-REPORTED vintages disagreeing is a RESTATEMENT, not a defect. BAC's shape."""
    substrates = build_substrates(catalogue, clean_facts.copy())
    original = substrates.facts[(substrates.facts["ticker"] == TICKER)
                                & (substrates.facts["field"] == "totalRevenue")].iloc[[0]]
    restated = original.copy()
    restated["value"] = restated["value"] * 1.05
    restated["accession_number"] = "TST-RESTATED"
    restated["filing_date"] = restated["filing_date"] + pd.Timedelta(days=400)
    substrates.facts = pd.concat([substrates.facts, restated], ignore_index=True)

    out = _findings(substrates, "cross_vintage")
    print(f"\nsame period re-presented 5% higher, both AS REPORTED -> {len(out)} finding(s), "
          f"severity {out['severity'].unique().tolist()}")
    print("  SANITY: classified `restatement` and reported at `info`. Both numbers are true "
          "and only the FIRST was knowable at the time -- which is the one a point-in-time "
          "model may use. BAC FY2023: 98,581M as filed, 102,769M re-presented.")
    assert len(out) == 1 and out.iloc[0]["severity"] == "info"


def test_cross_vintage_calls_a_derived_disagreement_a_candidate_defect(catalogue,
                                                                       clean_facts) -> None:
    """The other branch: a DERIVED vintage disagreeing puts our arithmetic in play -> `high`."""
    substrates = build_substrates(catalogue, clean_facts.copy())
    original = substrates.facts[(substrates.facts["ticker"] == TICKER)
                                & (substrates.facts["field"] == "totalRevenue")].iloc[[0]]
    restated = original.copy()
    restated["value"] = restated["value"] * 1.05
    restated["accession_number"] = "TST-DERIVED"
    restated["resolution_method"] = "derived_q4"
    restated["filing_date"] = restated["filing_date"] + pd.Timedelta(days=400)
    substrates.facts = pd.concat([substrates.facts, restated], ignore_index=True)

    out = _findings(substrates, "cross_vintage")
    print(f"\nsame disagreement, one vintage DERIVED -> severity "
          f"{out['severity'].unique().tolist()}")
    print("  SANITY: the discriminator needs no external data at all -- a derivation error "
          "leaves a non-as-reported basis, a restatement does not.")
    assert len(out) == 1 and out.iloc[0]["severity"] == "high"


# --------------------------------------------------------------------------- #
# the run's own contracts                                                      #
# --------------------------------------------------------------------------- #

def test_findings_carry_a_self_contained_investigation_packet(catalogue,
                                                              clean_facts) -> None:
    """Decision 47: identity + claim + PROVENANCE + EDGAR URL, on a Tier-2 finding."""
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
        facts.loc[facts[mask].index[-1], "value"] *= 5.0

    out = _findings(_plant(catalogue, clean_facts, bend), "trend_break")
    row = out.iloc[0]
    print(f"\npacket: {row['check_name']} {row['ticker']} {row['field']} "
          f"@{row['period_key']} id={row['finding_id']}\n"
          f"  observed={row['observed']:,.0f} expected={row['expected']:,.0f} "
          f"concept={row['source_concept']} method={row['resolution_method']}\n"
          f"  {row['edgar_url']}")
    print("  SANITY: a Tier-2/3 finding on a DERIVED value has no fact row to join back to, "
          "which is why the payload is fat rather than an identity plus a join.")
    for column in ("finding_id", "source_concept", "resolution_method", "accession_number",
                   "edgar_url", "detail"):
        assert row[column], f"{column} is empty -- the packet is not self-contained"


def test_finding_id_is_stable_across_runs_and_moves_with_its_key(catalogue,
                                                                 clean_facts) -> None:
    """The id survives a re-run and changes when any key component does."""
    from src.validate.fundamentals.finding import finding_id

    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
        facts.loc[facts[mask].index[-1], "value"] *= 5.0

    substrates = _plant(catalogue, clean_facts, bend)
    first = _findings(substrates, "trend_break").iloc[0]["finding_id"]
    second = _findings(substrates, "trend_break").iloc[0]["finding_id"]
    other = finding_id("trend_break", TICKER, "totalRevenue", "2020-03-31")

    print(f"\nrun 1 id={first}  run 2 id={second}  different field id={other}")
    print("  SANITY: hashed from (check, ticker, field, period_key) and NOT from run_date, "
          "severity or observed -- a threshold retune must not resurrect a settled finding, "
          "and a re-measured value must not orphan one.")
    assert first == second and first != other


def test_a_settled_finding_is_subtracted_from_the_queue(catalogue, clean_facts) -> None:
    """The register is what makes the queue SHRINK. `fundamentals_check` only ever grows."""
    from src.validate.fundamentals.check_register import CheckRegister, parse_entry

    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
        facts.loc[facts[mask].index[-1], "value"] *= 5.0

    substrates = _plant(catalogue, clean_facts, bend)
    before = _findings(substrates, "trend_break")
    settled = CheckRegister([parse_entry({
        "finding_id": before.iloc[0]["finding_id"], "check": "trend_break",
        "ticker": TICKER, "field": "capex", "period_key": before.iloc[0]["period_key"],
        "outcome": "accepted",
        "evidence": "planted for the test; verified in accession TST-0011",
        "decided_on": "2026-08-24", "decided_by": "tests"})])
    after = FundamentalsValidator(substrates, register=settled).run(names=["trend_break"])

    print(f"\nbefore settling: {len(before)} finding(s); after: {len(after.findings)}; "
          f"subtracted={after.settled_total}")
    print("  SANITY: settled once, forever. Nothing is ever re-investigated.")
    assert len(before) == 1 and after.findings.empty and after.settled_total == 1


def test_a_config_proposed_fix_does_NOT_close_a_finding(catalogue, clean_facts) -> None:
    """Decision 65: `configs/` is proposed, never applied -- so the data is still wrong."""
    from src.validate.fundamentals.check_register import CheckRegister, parse_entry

    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
        facts.loc[facts[mask].index[-1], "value"] *= 5.0

    substrates = _plant(catalogue, clean_facts, bend)
    before = _findings(substrates, "trend_break")
    proposed = CheckRegister([parse_entry({
        "finding_id": before.iloc[0]["finding_id"], "check": "trend_break",
        "ticker": TICKER, "field": "capex", "period_key": before.iloc[0]["period_key"],
        "outcome": "fixed", "fix_kind": "config_proposed",
        "commit": "abc1234", "regression_test": "tests/validate/x.py::y",
        "regression_swept": False,
        "evidence": "a never_use entry is proposed for the extension leg",
        "decided_on": "2026-08-24", "decided_by": "fundamentals-triage"})])
    after = FundamentalsValidator(substrates, register=proposed).run(names=["trend_break"])

    print(f"\na config_proposed fix is on file -> the finding is STILL OPEN: "
          f"{len(after.findings)} finding(s); "
          f"{len(proposed.open_proposals())} proposal(s) awaiting approval")
    print("  SANITY: the register is the one artifact where a wrong entry is invisible "
          "forever. The withdrawn 'UNH has no premiums' edit was one approval away.")
    assert len(after.findings) == 1 and len(proposed.open_proposals()) == 1


@pytest.mark.parametrize("bad,why", [
    ({"outcome": "fixed", "commit": None},
     "a fix with no commit"),
    ({"outcome": "fixed", "commit": "abc", "regression_test": None},
     "a fix with no regression test -- the 3c.8 failure mode with a resolved label"),
    ({"outcome": "wontfix", "evidence": "not worth it"},
     "a wontfix with no QUANTIFIED cost"),
    ({"outcome": "ignored"},
     "an outcome outside the closed vocabulary"),
])
def test_the_register_schema_refuses_a_suppression(bad, why) -> None:
    """Every rule that keeps `fundamentals_check.json` from becoming a suppression list."""
    from src.validate.fundamentals.check_register import RegisterError, parse_entry

    entry = {"finding_id": "a" * 16, "check": "trend_break", "ticker": "TST",
             "field": "capex", "period_key": "2023-03-31", "outcome": "accepted",
             "fix_kind": "code", "commit": "abc1234",
             "regression_test": "tests/validate/x.py::y",
             "evidence": "read accession 0001-23 and the line is genuinely absent",
             "decided_on": "2026-08-24", "decided_by": "tests"}
    entry.update(bad)
    with pytest.raises(RegisterError):
        parse_entry(entry)
    print(f"  REFUSED: {why}")


def test_the_committed_register_parses(catalogue) -> None:
    """`configs/fundamentals/fundamentals_check.json` is loadable and schema-valid as shipped."""
    from src.validate.fundamentals.check_register import load_register

    register = load_register("./configs")
    print(f"\nconfigs/fundamentals/fundamentals_check.json: {len(register)} settled finding(s), "
          f"{len(register.open_proposals())} open proposal(s), "
          f"{len(register.unswept_fixes())} unswept fix(es)")
    print("  SANITY: empty on purpose -- 5b-core.1 ships the mechanism; the first entries are "
          "written by the agent loop against REAL findings. An entry invented before a "
          "finding exists would be a guess about a measurement.")
    assert len(register) == 0


def test_the_validator_loads_each_substrate_exactly_once(catalogue, clean_facts) -> None:
    """Phase 10's named risk, made structurally impossible rather than merely avoided.

    A check is handed a `Substrates` and has no `store`, so it CANNOT re-read. This asserts the
    count so a future author cannot quietly add one back: ~28M fact rows re-read 35 times is
    the failure this design exists to prevent.
    """
    from types import SimpleNamespace

    from src.validate.fundamentals import substrate as substrate_module

    calls: list[str] = []
    history = substrate_module._normalise(None, ())          # shapes come from the fixture

    def fake_load(table, columns=None, where=None, since=None, optional=False, **kwargs):
        calls.append(str(table))
        return {"fundamentals_facts": clean_facts}.get(str(table), history)

    context = SimpleNamespace(store=SimpleNamespace(load=fake_load))
    substrate_module.load(context, catalogue, ["TST"])

    print(f"\nstore.load calls during one substrate build: {calls}")
    print("  SANITY: four tables, four reads, once. A check never gets a store, so it cannot "
          "re-read -- and 35 checks re-reading fundamentals_facts would be the whole cost.")
    assert len(calls) == 4 and len(set(calls)) == 4


# --------------------------------------------------------------------------- #
# the acceptance corpus: what the FIRST CALIBRATION RUN taught the checks      #
# --------------------------------------------------------------------------- #
# Run 1 (2026-08-24) fired 293 `critical` cross_identity findings at 9.0% on a 2% ceiling.
# Challenged before the data was -- and the check lost both arguments. These pin the two
# corrections so neither can regress, which is the acceptance-corpus rule: a fix that leaves
# no test is a defect waiting to come back.

def test_a_balance_sheet_footing_only_WITH_nci_is_silent(catalogue, clean_facts) -> None:
    """An EX-NCI equity element is not a defect: if the books foot on either basis, they foot.

    Measured: this silences 39 of the 103 balance-sheet findings run 1 produced. We cannot tell
    from a stored row which element `stockholdersEquity` resolved through, and asserting a rule
    we cannot verify is precisely what this check was corrected for.
    """
    substrates = build_substrates(catalogue, clean_facts.copy())
    mask = substrates.history["ticker"] == TICKER
    # shift NCI out of equity: the books now foot only when NCI is added back
    nci = substrates.history.loc[mask, "totalAssets"] * 0.04
    substrates.history.loc[mask, "stockholdersEquity"] -= nci
    substrates.history.loc[mask, "minorityInterest"] = nci

    out = _findings(substrates, "cross_identity")
    print(f"\nequity moved onto an ex-NCI basis with minorityInterest carrying the 4% "
          f"remainder -> {len(out[out['ticker'] == TICKER])} finding(s)")
    print("  SANITY: silent. The identity is tested on BOTH equity bases and passes if either "
          "foots -- 39 of run 1's 103 findings were this shape.")
    assert out[out["ticker"] == TICKER].empty


def test_an_unexplained_balance_sheet_gap_is_high_not_critical(catalogue,
                                                               clean_facts) -> None:
    """A 3%-ish gap no equity basis explains is `high`: MEZZANINE equity could account for it.

    Redeemable NCI and REIT OP units sit BETWEEN liabilities and equity under ASC 480-10-S99,
    and the 69-column contract carries no column for them. `critical` means PROVABLY wrong on
    this ladder, and a gap we cannot decompose is not that. The live population is UNH 1.68%,
    EQIX 1.48%, PGR 1.41%, AMT 3.28%, SPG 1.06%, NVDA 1.15% -- all this shape.
    """
    substrates = build_substrates(catalogue, clean_facts.copy())
    mask = substrates.history["ticker"] == TICKER
    substrates.history.loc[mask, "stockholdersEquity"] *= 0.92   # ~3% of assets goes missing

    out = _findings(substrates, "cross_identity")
    ours = out[(out["ticker"] == TICKER) & (out["field"] == "totalAssets")]
    print(f"\n~3% of assets unaccounted for on EITHER basis -> {len(ours)} finding(s), "
          f"severity {sorted(set(ours['severity']))}")
    print("  SANITY: `high`, not `critical`. The real repair is a catalogue field "
          "(temporaryEquity), which is Phase 9's decision and not the validator's.")
    assert len(ours) > 0 and set(ours["severity"]) == {"high"}


def test_a_shell_scale_gap_stays_critical(catalogue, clean_facts) -> None:
    """ETN's 2012 redomicile holdco and VRT's SPAC: no mezzanine explains a gap that size."""
    substrates = build_substrates(catalogue, clean_facts.copy())
    mask = substrates.history["ticker"] == TICKER
    substrates.history.loc[mask, "totalAssets"] *= 3.0

    out = _findings(substrates, "cross_identity")
    ours = out[(out["ticker"] == TICKER) & (out["field"] == "totalAssets")]
    print(f"\nassets 3x the sum of the other side -> severity {sorted(set(ours['severity']))}")
    print("  SANITY: `critical`. Above 10% of assets no redeemable-NCI or OP-unit balance is "
          "a real capital structure -- ETN's shell scores 172,559x and VRT's SPAC 95%.")
    assert set(ours["severity"]) == {"critical"}


def test_a_gross_profit_basis_difference_is_medium_not_critical(catalogue,
                                                                clean_facts) -> None:
    """`GrossProfit == Revenue - COGS` IS NOT AN ACCOUNTING IDENTITY, and run 1 proved it.

    All 191 failures were industrial, at 15-74%: TMO 50 (33.6%), EQIX 48 (15.0%), CVS 31
    (73.9%), CAT 31 (39.3%), COST 28 (39.7%). Each filer's own us-gaap:GrossProfit tag uses its
    own cost basis -- CVS excludes benefit costs, COST nets membership fees. Both numbers were
    right and the PREMISE was wrong.
    """
    substrates = build_substrates(catalogue, clean_facts.copy())
    mask = substrates.history["ticker"] == TICKER
    substrates.history.loc[mask, "grossProfit"] *= 0.65     # a CVS-shaped basis difference

    out = _findings(substrates, "cross_identity")
    ours = out[(out["ticker"] == TICKER) & (out["field"] == "grossProfit")]
    print(f"\ngrossProfit on a 35%-narrower cost basis -> {len(ours)} finding(s), "
          f"severity {sorted(set(ours['severity']))}")
    print("  SANITY: `medium` -- a candidate, look, do not assume. Still reported, because a "
          "large gap CAN indicate a mis-resolved costOfRevenue.")
    assert len(ours) > 0 and set(ours["severity"]) == {"medium"}


def test_info_findings_do_not_count_toward_a_ceiling(catalogue, clean_facts) -> None:
    """The ceiling asks "is this check burying real findings?" -- `info` cannot bury anything.

    Run 1 reported `series_shape` at 29.1% (1,045 of its 1,632 findings were benign `info` gap
    codes) and `register_cost`, which is info-ONLY, at 825.9%. Both were the METRIC misreading
    the check rather than the check misreading the data.
    """
    substrates = build_substrates(catalogue, clean_facts.copy())
    outcome = next(o for o in FundamentalsValidator(substrates)
                   .run(names=["expected_absent_drift", "peer_ratio_abstentions",
                               "restatement_ledger"]).outcomes
                   if o.spec.name == "peer_ratio_abstentions")

    print(f"\ninfo-only check: {len(outcome.findings)} finding(s), "
          f"{outcome.queued} in the QUEUE, fire rate {outcome.fire_rate:.2%}, "
          f"over ceiling: {outcome.over_ceiling}")
    print("  SANITY: the rate counts QUEUE findings only, so an info-only check can never be "
          "labelled a threshold bug for doing exactly its job.")
    assert outcome.queued == 0 and outcome.fire_rate == 0.0 and not outcome.over_ceiling
