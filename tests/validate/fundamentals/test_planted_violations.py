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

from src.data_extract.utils.fundamentals import reason_codes as rc
from src.validate.fundamentals.checks import CHECK_REGISTRY
from src.validate.fundamentals.checks.tier2_series import (
    _BENIGN_GAP_CODES, _shape_severity, EARLY_STOP, INTERIOR_GAP, LATE_START, SPARSE)
from src.validate.fundamentals.finding import HIGH, INFO, MEDIUM, QUEUE_SEVERITIES
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


def test_a_cross_identity_finding_NAMES_THE_FILING_it_broke_on(catalogue,
                                                                clean_facts) -> None:
    """THE reason Tier 1's value checks moved to `fundamentals_facts`.

    `Finding.edgar_url` is built from `(cik, accession_number)` and `fundamentals_history_sec`
    carries neither, so on the history substrate every one of the run's 1,437 Tier-1 findings
    arrived with a NULL url -- against 100% on Tier 3. An agent handed such a finding cannot
    open the filing that caused it, which is the first move the triage loop requires, so the
    whole tier was unactionable however well it was ranked.
    """
    def bend(facts):
        mask = ((facts["ticker"] == TICKER) & (facts["field"] == "totalAssets")
                & (facts["period_end"] == facts["period_end"].max()))
        facts.loc[mask, "value"] = facts.loc[mask, "value"] * 1.5

    out = _findings(_plant(catalogue, clean_facts, bend), "cross_identity")
    row = out.iloc[0]

    print(f"\naccession {row['accession_number']!r} -> {row['edgar_url']}")
    print("  SANITY: a Tier-1 finding a reviewer can OPEN. On the history substrate this "
          "column was NULL for all 1,437 of them, criticals included.")
    assert row["accession_number"] and str(row["accession_number"]) != "nan"
    assert row["edgar_url"] and "sec.gov" in row["edgar_url"]
    assert row["substrate"] == "facts"


def test_a_derived_identity_reason_code_no_longer_suppresses_anything(catalogue,
                                                                     clean_facts) -> None:
    """The `derived_identity` SKIP is deleted, and this pins why that is safe.

    On history, `totalLiabilities` was computed as `totalAssets - stockholdersEquity` for the
    filers who never tag `us-gaap:Liabilities`, so the identity read `A - E + E == A` --
    arithmetic, which passes on any numbers at all. The check therefore read
    `fundamentals_reason_codes` and skipped those rows.

    `fundamentals_facts` is strictly as-filed and holds no derived total, so the code cannot
    apply: a filer that never tags a `Liabilities` total is simply ABSENT from the identity's
    population rather than silently excused inside it. A stray reason code must NOT reach back
    and suppress a genuine facts-grain break.
    """
    def bend(facts):
        mask = ((facts["ticker"] == TICKER) & (facts["field"] == "totalAssets")
                & (facts["period_end"] == facts["period_end"].max()))
        facts.loc[mask, "value"] = facts.loc[mask, "value"] * 1.5

    substrates = _plant(catalogue, clean_facts, bend)
    substrates.codes = pd.concat([substrates.codes, pd.DataFrame([{
        "ticker": TICKER, "as_of": substrates.history["as_of"].max(),
        "field": "totalLiabilities", "dc_code": "derived_identity",
        "combined_into": None, "rejected_value": float("nan")}])], ignore_index=True)

    out = _findings(substrates, "cross_identity")
    print(f"\na `derived_identity` code on the same (ticker, field) -> {len(out)} finding(s)")
    print("  SANITY: still fires. The code describes a HISTORY cell; the finding is about a "
          "filed statement, and the two are no longer wired together.")
    assert len(out) == 1 and out.iloc[0]["severity"] == "critical"


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


def test_the_run_records_every_finding_including_info(catalogue, clean_facts) -> None:
    """D5: the frame that is WRITTEN carries everything; only the QUEUE view drops `info`.

    The JSON register that used to subtract settled findings before the write is deleted, so
    this is now a property of the run itself rather than of a config. It is the property the
    whole loop rests on: a row-count drop between two runs of one scope has exactly one cause,
    because nothing else can remove a row.

    A `wontfix` is applied when the REPORT is rendered, never when the row is written, so the
    table and the checks always agree about what fired.
    """
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "capex")
        facts.loc[facts[mask].index[-1], "value"] *= 5.0

    run = FundamentalsValidator(_plant(catalogue, clean_facts, bend)).run()
    written, queued = len(run.findings), len(run.queue)
    info = int((run.findings["severity"] == "info").sum())

    print(f"{written} finding(s) written, {queued} in the queue, {info} `info`")
    print(f"  written == queued + info: {written == queued + info}")
    print("  SANITY: D5. Nothing is subtracted on the way in, so the ledger's row count "
          "means one thing. `info` is filtered by the QUEUE VIEW, not by the writer.")
    assert written == queued + info and info > 0
    assert set(run.findings["run_id"]) == {run.run_id}
    assert run.findings["cluster_id"].notna().all()



def test_two_runs_of_one_scope_share_a_run_id_and_a_narrower_scope_does_not(
        catalogue, clean_facts) -> None:
    """`run_id` is the comparability key. Widening the scope MUST change it.

    Without this, re-validating a one-ticker fix against a 54-ticker baseline would report
    ~11,800 findings "closed" and every one of them would read as a triumph.
    """
    from src.validate.fundamentals.scope import RunScope

    wide = RunScope.build(tickers=["AAA", "BBB"], fields=None, tiers=[1, 2, 3])
    same = RunScope.build(tickers=["BBB", "AAA"], fields=None, tiers=[3, 2, 1])
    narrow = RunScope.build(tickers=["AAA"], fields=None, tiers=[1, 2, 3])
    by_field = RunScope.build(tickers=["AAA", "BBB"], fields=["capex"], tiers=[1, 2, 3])
    relabelled = RunScope.build(tickers=["AAA", "BBB"], fields=None, tiers=[1, 2, 3],
                                roster="renamed_overnight")
    day = pd.Timestamp("2026-08-24")

    print(f"\nwide={wide.run_id(day)}  same-scope-other-order={same.run_id(day)}  "
          f"narrow={narrow.run_id(day)}  field-scoped={by_field.run_id(day)}")
    print(f"  same day, next day: {wide.run_id(day)} vs "
          f"{wide.run_id(day + pd.Timedelta(days=1))}")
    print("  SANITY: scope order does not matter, scope CONTENT does, the roster NAME does "
          "not, and the date separates two runs of one scope.")
    assert wide.run_id(day) == same.run_id(day)
    assert wide.scope_hash == relabelled.scope_hash
    assert len({wide.run_id(day), narrow.run_id(day), by_field.run_id(day)}) == 3
    assert wide.run_id(day) != wide.run_id(day + pd.Timedelta(days=1))
    assert wide.scope_hash == wide.scope_hash and narrow.scope_hash != wide.scope_hash


def test_a_duplicated_finding_id_raises_instead_of_silently_upserting(catalogue) -> None:
    """The 536-row gap: 12,462 emitted, 11,926 stored, and nothing said so.

    `finding_id` hashes exactly the `fundamentals_check` PK, so two findings sharing one id
    UPSERT onto each other -- the second overwrites the first and the run reports a number it
    did not write.
    """
    from src.validate.fundamentals.finding import DuplicateFindingError, Finding, findings_frame

    twins = [Finding(check_name="cross_vintage", ticker=TICKER, severity="high",
                     field="capex", period_key="2019-09-28", tier=3, substrate="facts",
                     observed=v, detail={"duration_type": d})
             for v, d in ((1.0, "instant"), (2.0, "quarterly"))]
    singleton = findings_frame(twins[:1], pd.Timestamp("2026-08-24"), "abc123")

    with pytest.raises(DuplicateFindingError) as caught:
        findings_frame(twins, pd.Timestamp("2026-08-24"), "abc123")

    print(f"\none finding writes {len(singleton)} row(s); two sharing an id RAISE:")
    print(f"  {str(caught.value)[:180]}")
    print("  SANITY: the emitted count and the stored count can no longer disagree in "
          "silence -- the run fails loudly and names the offending key.")
    assert twins[0].id == twins[1].id and len(singleton) == 1


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
#
# THEY ALL PLANT INTO `facts` NOW. They used to bend `substrates.history` directly, and when
# `cross_identity` moved substrates three of them failed loudly -- but two went on PASSING
# while asserting nothing at all, because a check that no longer reads history is trivially
# silent about anything planted there. A vacuous green is worse than a red: it is a claim
# nobody re-checks. If a plant here stops firing, bend the FACTS frame.

def test_a_balance_sheet_footing_only_WITH_nci_is_silent(catalogue, clean_facts) -> None:
    """An EX-NCI equity element is not a defect: if the books foot on either basis, they foot.

    Measured: this silences 39 of the 103 balance-sheet findings run 1 produced. We cannot tell
    from a stored row which element `stockholdersEquity` resolved through, and asserting a rule
    we cannot verify is precisely what this check was corrected for.
    """
    def bend(facts):
        # shift 4% of assets out of equity and into a minorityInterest line the fixture does
        # not otherwise carry: the books now foot only when NCI is added back
        assets = facts[(facts["ticker"] == TICKER) & (facts["field"] == "totalAssets")]
        nci_rows = assets.copy()
        nci_rows["field"] = "minorityInterest"
        nci_rows["value"] = nci_rows["value"] * 0.04
        nci_rows["source_concept"] = "us-gaap:MinorityInterest"
        equity = ((facts["ticker"] == TICKER) & (facts["field"] == "stockholdersEquity"))
        shift = dict(zip(assets["period_end"], assets["value"] * 0.04))
        facts.loc[equity, "value"] = (facts.loc[equity, "value"]
                                      - facts.loc[equity, "period_end"].map(shift))
        return pd.concat([facts, nci_rows], ignore_index=True)

    planted = clean_facts.copy()
    planted = bend(planted)
    substrates = build_substrates(catalogue, planted)

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
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "stockholdersEquity")
        facts.loc[mask, "value"] = facts.loc[mask, "value"] * 0.92   # ~3% of assets goes missing

    substrates = _plant(catalogue, clean_facts, bend)

    out = _findings(substrates, "cross_identity")
    ours = out[(out["ticker"] == TICKER) & (out["field"] == "totalAssets")]
    print(f"\n~3% of assets unaccounted for on EITHER basis -> {len(ours)} finding(s), "
          f"severity {sorted(set(ours['severity']))}")
    print("  SANITY: `high`, not `critical`. The real repair is a catalogue field "
          "(temporaryEquity), which is Phase 9's decision and not the validator's.")
    assert len(ours) > 0 and set(ours["severity"]) == {"high"}


def test_a_shell_scale_gap_stays_critical(catalogue, clean_facts) -> None:
    """ETN's 2012 redomicile holdco and VRT's SPAC: no mezzanine explains a gap that size."""
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "totalAssets")
        facts.loc[mask, "value"] = facts.loc[mask, "value"] * 3.0

    substrates = _plant(catalogue, clean_facts, bend)

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
    def bend(facts):
        mask = (facts["ticker"] == TICKER) & (facts["field"] == "grossProfit")
        facts.loc[mask, "value"] = facts.loc[mask, "value"] * 0.65     # a CVS-shaped basis difference

    substrates = _plant(catalogue, clean_facts, bend)

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
    codes) and `catalogue_exclusion_cost`, info-ONLY, at 825.9%. Both were the METRIC misreading
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

# --------------------------------------------------------------------------- #
# series_shape -- the severity ladder, every shape against every benign code   #
# --------------------------------------------------------------------------- #

#: (shape, modal gap code) -> the severity the ladder MUST return, and why.
#:
#: The whole point of enumerating the matrix rather than testing the two bugs: the old ladder
#: tested the CODE before the SHAPE, so the defect was never in one cell -- it was in the
#: order, and any cell could have been the next one to go wrong. Four of these combinations
#: were mislabelled `info` on run 2 and account for all 347 of the reclassified findings.
_SHAPE_LADDER = [
    # LATE START -- the start of a history is exactly what these codes describe. Unchanged.
    (LATE_START, rc.INSUFFICIENT_QUARTERS, INFO, "the TTM warm-up window IS the late start"),
    (LATE_START, rc.NOT_APPLICABLE, INFO, "declared structurally absent for this filer"),
    (LATE_START, rc.NOT_APPLICABLE_FOR_REGIME, INFO, "the regime register declares it absent"),
    (LATE_START, rc.REGIME_BREAK, INFO, "a standard's adoption is real accounting"),
    (LATE_START, rc.NOT_DISCLOSED, HIGH, "a missing tag at the start -- open the filing"),

    # EARLY STOP -- a field CAN cease to apply, but a TTM warm-up cannot end a series.
    (EARLY_STOP, rc.INSUFFICIENT_QUARTERS, HIGH,
     "REGRESSION: 7 findings sat in `info` here and this HIGH branch was unreachable"),
    (EARLY_STOP, rc.NOT_APPLICABLE, INFO, "the field genuinely stopped applying"),
    (EARLY_STOP, rc.REGIME_BREAK, INFO, "a standard retired the caption"),
    (EARLY_STOP, rc.NOT_DISCLOSED, HIGH, "it went dark -- VLO capex, 2023-07"),

    # INTERIOR GAP -- values on BOTH sides contradict every one of the benign rationales.
    (INTERIOR_GAP, rc.INSUFFICIENT_QUARTERS, HIGH,
     "REGRESSION: 268 findings -- a start-of-history rationale on a mid-history hole"),
    (INTERIOR_GAP, rc.REGIME_BREAK, HIGH,
     "REGRESSION: 71 findings -- an adoption is a STEP, and the modal code is measured "
     "over the whole series rather than inside the gap"),
    (INTERIOR_GAP, rc.NOT_APPLICABLE_FOR_REGIME, HIGH,
     "REGRESSION: 1 finding -- absent here, yet reported on both sides of the hole"),
    (INTERIOR_GAP, rc.NOT_APPLICABLE, HIGH, "same contradiction, catalogue-declared"),
    (INTERIOR_GAP, rc.NOT_DISCLOSED, HIGH, "a missing tag -- the 212 that were already right"),
    (INTERIOR_GAP, rc.PERIOD_INTERSECTION_PARTIAL, MEDIUM, "route 3b refused these windows"),

    # SPARSE -- `periodicity` (5b-stats) owns the shape. That deferral is correct and stays.
    (SPARSE, rc.NOT_DISCLOSED, INFO, "an annual-only field is not a series with holes"),
    (SPARSE, rc.INSUFFICIENT_QUARTERS, INFO, "still a periodicity question"),
    (SPARSE, None, INFO, "info, but it must NOT render as a diagnosis -- see below"),
]


@pytest.mark.parametrize("shape,code,expected,why",
                         _SHAPE_LADDER,
                         ids=[f"{s}-{c or 'NULL'}" for s, c, _e, _w in _SHAPE_LADDER])
def test_series_shape_severity_matrix(clean, shape, code, expected, why) -> None:
    """Every shape against every benign code. The ORDER bug is unrepresentable now."""
    severity, reason = _shape_severity(clean, TICKER, "capex", shape, code,
                                       (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-01-01")))
    print(f"{shape:13s} + {str(code):28s} -> {severity:6s}  ({why})")
    assert severity == expected, f"{shape} + {code}: got {severity}, want {expected}"
    assert reason, "an info with no stated reason is indistinguishable from a check that gave up"
    if expected is HIGH and code in _BENIGN_GAP_CODES:
        assert str(code) in reason, ("a HIGH whose payload shows a benign code must say why "
                                     "the code was rejected, or it reads as a false positive")


def test_series_shape_reclassification_counts(clean) -> None:
    """The measured consequence: exactly the 347 run-2 findings move, and nothing else does."""
    moved = [(shape, code) for shape, code, expected, _why in _SHAPE_LADDER
             if code in _BENIGN_GAP_CODES and expected is not INFO]
    run2 = {(INTERIOR_GAP, rc.INSUFFICIENT_QUARTERS): 268,
            (INTERIOR_GAP, rc.REGIME_BREAK): 71,
            (INTERIOR_GAP, rc.NOT_APPLICABLE_FOR_REGIME): 1,
            (EARLY_STOP, rc.INSUFFICIENT_QUARTERS): 7}
    counted = sum(run2.get(pair, 0) for pair in moved)

    print(f"\n{len(moved)} (shape, code) combination(s) leave `info`: {moved}")
    print(f"  run-2 findings they carried: {counted}")
    print("  SANITY: 347 = 340 interior_gap + 7 early_stop, measured off fundamentals_check "
          "before the change. `sparse` keeps its info deferral to `periodicity`.")
    # `moved` is a superset of `run2`: (interior_gap, not_applicable) also stops being benign
    # but happened to carry 0 rows on that roster. A combination the data has not exercised
    # yet still has to be RIGHT, so the matrix above pins it either way.
    assert counted == 347
    assert set(run2) <= set(moved)
    assert (SPARSE, rc.INSUFFICIENT_QUARTERS) not in moved


def test_a_sparse_series_with_NO_code_does_not_render_as_a_diagnosis(clean) -> None:
    """A null code is not `not_disclosed`. 186 of run 2's 686 sparse series carried none."""
    _sev_none, reason_none = _shape_severity(clean, TICKER, "capex", SPARSE, None,
                                             (pd.Timestamp("2015-01-01"),
                                              pd.Timestamp("2016-01-01")))
    _sev_code, reason_code = _shape_severity(clean, TICKER, "capex", SPARSE, rc.NOT_DISCLOSED,
                                             (pd.Timestamp("2015-01-01"),
                                              pd.Timestamp("2016-01-01")))
    print(f"\nno code : {reason_none[:120]}")
    print(f"a code  : {reason_code[:120]}")
    print("  SANITY: an UNEXPLAINED null is a coverage gap in fundamentals_reason_codes, not "
          "a periodicity finding. Both stay `info`; only the rendering differs.")
    assert reason_none != reason_code
    assert "UNEXPLAINED" in reason_none and "UNEXPLAINED" not in reason_code

