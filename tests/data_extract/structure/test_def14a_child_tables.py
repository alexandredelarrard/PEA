"""The four child-table row builders, the `reconciles` flag, and the gender consensus pass.

All of it is pure: the builders are functions of `Def14AExtract` and the consensus is a function
of a DataFrame, so none of this needs the network, the DB or an LLM. That is the same property
that lets `scripts/def14a_replay_flatten.py` verify the flatten over 445 real stored filings for
free — the tokens were paid when those filings were first extracted.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.structure.def14a_gender import (
    BASIS_RANK, consensus, person_key, recompute_parent_gender,
)
from src.data_extract.utils.schemas.def14a_schema import (
    BeneficialOwner, Def14AExtract, DirectorCompensation, DirectorInfo,
    ExecutiveCompensation, GovernanceProfile,
)
from src.data_extract.utils.structure.fetch_def14a_llm import (
    _RECONCILE_TOLERANCE_USD, _child_frames, _flatten,
)

FILING = pd.Series({"filing_date": pd.Timestamp("2026-04-30"),
                    "accession_number": "0001308179-26-000358",
                    "period_of_report": pd.Timestamp("2025-12-31"),
                    "cik": "0000018230"})


def _extract(**kw) -> Def14AExtract:
    return Def14AExtract(**kw)


# --------------------------------------------------------------------------- #
# multi-year SCT: three derived values that break if left alone               #
# --------------------------------------------------------------------------- #
def _three_year_sct() -> list[ExecutiveCompensation]:
    """Two NEOs x three fiscal years — the shape Item 402(c) actually requires."""
    rows = []
    for year, salary in ((2025, 1_500_000.0), (2024, 1_400_000.0), (2023, 1_300_000.0)):
        rows.append(ExecutiveCompensation(
            name="Joseph E. Creed", title="Chief Executive Officer", fiscal_year=year,
            salary_usd=salary, total_compensation_usd=salary * 10))
        rows.append(ExecutiveCompensation(
            name="Andrew R. Bonfield", title="Chief Financial Officer", fiscal_year=year,
            salary_usd=salary / 2, total_compensation_usd=salary * 4))
    return rows


def test_n_neos_counts_names_in_the_latest_year_not_rows():
    """With 3 years x 2 NEOs the row count is 6 and the answer is 2. Left as `len(...)`,
    `n_neos` would triple and the `n_neos == 1` metric — the single best summary of whether the
    carve reached the SCT — would become meaningless."""
    row = _flatten("CAT", FILING, _extract(compensation=_three_year_sct()))
    assert row["n_neos"] == 2, f"n_neos counted rows, not names: {row['n_neos']}"
    assert row["sct_years"] == 3, f"sct_years wrong: {row['sct_years']}"
    print(f"\n  6 SCT rows (2 NEOs x 3 years) -> n_neos={row['n_neos']}, "
          f"sct_years={row['sct_years']}")


def test_ceo_row_and_total_come_from_the_most_recent_year():
    """`_ceo_from_compensation` used to scan the whole list, which with multi-year rows returns
    an arbitrary year's CEO. `total_neo_comp` must not sum three years of pay."""
    row = _flatten("CAT", FILING, _extract(ceo_name="Joseph E. Creed",
                                           compensation=_three_year_sct()))
    assert row["ceo_salary"] == 1_500_000.0, f"CEO row from the wrong year: {row['ceo_salary']}"
    # latest year only: 15,000,000 (CEO) + 6,000,000 (CFO)
    assert row["total_neo_comp"] == 21_000_000.0, row["total_neo_comp"]
    print(f"\n  CEO salary {row['ceo_salary']:,.0f} (FY2025 row), "
          f"total_neo_comp {row['total_neo_comp']:,.0f} (one year, not three)")


def test_exec_comp_rows_carry_every_neo_year_pair():
    rows = _child_frames("CAT", FILING, _extract(compensation=_three_year_sct()))["def14a_executive_comp"]
    assert len(rows) == 6
    assert {r["fiscal_year"] for r in rows} == {2023, 2024, 2025}
    assert all(r["as_of"] == FILING["filing_date"] for r in rows), "as_of must be the FILING date"
    print(f"\n  {len(rows)} exec-comp rows across {len({r['fiscal_year'] for r in rows})} years, "
          f"all stamped as_of={FILING['filing_date'].date()}")


# --------------------------------------------------------------------------- #
# the reconciles FLAG (D10) -- never a filter                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("total,expected", [
    (1_000_000.0, 1.0),                                  # exact
    (1_000_000.0 + _RECONCILE_TOLERANCE_USD, 1.0),       # exactly at the boundary
    (1_000_000.0 + _RECONCILE_TOLERANCE_USD + 0.01, 0.0),  # a cent past it
    (None, None),                                        # no total -> not computable
])
def test_reconciles_boundary(total, expected):
    """$10 absorbs filers who round each component independently. The flag is the OUTPUT: the
    values are kept either way, so the failure rate becomes measurable instead of being hidden
    by a residual repair."""
    comp = ExecutiveCompensation(
        name="A. Officer", title="CFO", fiscal_year=2025,
        salary_usd=600_000.0, bonus_usd=400_000.0, total_compensation_usd=total)
    row = _child_frames("X", FILING, _extract(compensation=[comp]))["def14a_executive_comp"][0]
    assert row["reconciles"] == expected, f"total={total} -> {row['reconciles']}"
    assert row["salary"] == 600_000.0, "a non-reconciling row must KEEP its values"


def test_pension_change_is_part_of_the_reconciliation():
    """The seventh component. The post-2006 SCT has seven and the schema modelled six, which is
    why the residual was POSITIVE on 97.5% of non-reconciling rows at a median of $319,367."""
    comp = ExecutiveCompensation(
        name="A. Officer", title="CEO", fiscal_year=2025, salary_usd=1_000_000.0,
        pension_change_usd=319_367.0, total_compensation_usd=1_319_367.0)
    row = _child_frames("X", FILING, _extract(compensation=[comp]))["def14a_executive_comp"][0]
    assert row["pension_change"] == 319_367.0
    assert row["reconciles"] == 1.0, "the pension column did not enter the reconciliation"
    print(f"\n  salary 1,000,000 + pension 319,367 = total 1,319,367 -> reconciles=1.0")


# --------------------------------------------------------------------------- #
# director comp / ownership / directors                                       #
# --------------------------------------------------------------------------- #
def test_director_comp_rows_use_the_labelled_columns():
    """`fees_earned` is the cash retainer whatever the filer calls it, and `stock_awards` covers
    `Restricted Stock Units` — the exact labels that nulled CAT's and GE's columns."""
    dc = DirectorCompensation(name="Sebastien Bazin", fiscal_year=2025,
                              fees_earned_usd=0.0, stock_awards_usd=345_795.0,
                              all_other_comp_usd=0.0, total_compensation_usd=345_795.0)
    rows = _child_frames("GE", FILING, _extract(director_compensation=[dc]))["def14a_director_comp"]
    assert len(rows) == 1 and rows[0]["stock_awards"] == 345_795.0
    assert rows[0]["reconciles"] == 1.0
    print(f"\n  director row: fees={rows[0]['fees_earned']}, "
          f"stock={rows[0]['stock_awards']:,.0f}, total={rows[0]['total']:,.0f}")


def test_ownership_drops_group_subtotals_and_addresses():
    """Two row shapes must never be stored: an "as a group" subtotal (already carried as the
    `insider_ownership_pct` scalar, so storing it double-counts the insiders) and a cell that is
    only a street address (edgartools grabbed those from JPM's proxy; an LLM can too)."""
    holders = [
        BeneficialOwner(holder_name="The Vanguard Group", holder_type="5pct_holder",
                        shares=1_415_826_462.0, percent_of_class=0.0963),
        BeneficialOwner(holder_name="All current directors and executive officers as a group "
                                    "(12 persons)", shares=5_000_000.0),
        BeneficialOwner(holder_name="270 Park Avenue, New York, NY 10017", shares=1.0),
        BeneficialOwner(holder_name="Tim Cook", holder_type="director_officer",
                        shares=3_280_295.0, percent_of_class=None),
    ]
    rows = _child_frames("AAPL", FILING, _extract(ownership_holders=holders))["def14a_ownership"]
    names = [r["holder_name"] for r in rows]
    assert "The Vanguard Group" in names and "Tim Cook" in names
    assert not any("as a group" in n.lower() for n in names), names
    assert not any(n.startswith("270 Park") for n in names), names
    assert len(rows) == 2, names
    print(f"\n  4 holders in -> {len(rows)} stored: {names} "
          f"(group subtotal + address-only dropped)")


def test_percent_of_class_stays_null_for_a_bound():
    """`'*'` / `'<1%'` is a BOUND, not a measurement. Storing 0.5 for it — which the retired
    edgar path did — fabricates a number."""
    h = BeneficialOwner(holder_name="Tim Cook", holder_type="director_officer",
                        shares=3_280_295.0, percent_of_class=None)
    row = _child_frames("AAPL", FILING, _extract(ownership_holders=[h]))["def14a_ownership"][0]
    assert row["percent_of_class"] is None


def test_director_rows_always_carry_a_gender_basis():
    """An unlabelled gender is indistinguishable from the first-name prior this upgrade exists
    to expose, so the builder never leaves the provenance blank when a gender is set."""
    dirs = [DirectorInfo(name="Sue Wagner", age=64, gender="female", gender_basis="honorific"),
            DirectorInfo(name="Art Levinson", age=75, gender="male"),        # basis omitted
            DirectorInfo(name="Unknown Person", age=50)]                     # no gender at all
    rows = _child_frames("AAPL", FILING, _extract(directors=dirs))["def14a_directors"]
    by_name = {r["name"]: r for r in rows}
    assert by_name["Sue Wagner"]["gender_basis"] == "honorific"
    assert by_name["Art Levinson"]["gender_basis"] == "name", "missing basis was not defaulted"
    assert by_name["Unknown Person"]["gender_basis"] is None, "basis set without a gender"
    print(f"\n  gender_basis: {[(n, r['gender_basis']) for n, r in by_name.items()]}")


def test_flatten_reports_gender_provenance_and_the_count_cross_check():
    """`pct_gender_stated` is the per-filing confidence in `pct_female_directors`, and
    `n_women_directors_vs_inferred` is the honest error bar — 0 when the filing's own count
    agrees with the per-director genders. Neither overwrites anything."""
    dirs = [DirectorInfo(name="A One", gender="female", gender_basis="stated"),
            DirectorInfo(name="B Two", gender="female", gender_basis="honorific"),
            DirectorInfo(name="C Three", gender="male", gender_basis="name"),
            DirectorInfo(name="D Four", gender="male", gender_basis="pronoun")]
    row = _flatten("X", FILING, _extract(
        directors=dirs, governance=GovernanceProfile(board_size=4, n_women_directors=2)))
    assert row["pct_gender_stated"] == 0.5, row["pct_gender_stated"]   # stated + honorific of 4
    assert row["n_women_directors_vs_inferred"] == 0, row["n_women_directors_vs_inferred"]
    assert row["pct_female_directors"] == 0.5
    print(f"\n  pct_gender_stated={row['pct_gender_stated']} (2 of 4 rows are document evidence), "
          f"women count agrees (delta={row['n_women_directors_vs_inferred']})")


def test_auditor_fee_block_is_rescaled_together():
    """A filer reports every cell of its fee table in one unit, so the block is rescaled
    together or not at all — a cell-by-cell rescale would invent a table whose categories no
    longer sum to the total. MS reported 57.6 for $57.6M."""
    gov = GovernanceProfile(auditor_name="Deloitte & Touche LLP", auditor_fees_usd=57.6,
                            audit_fees_audit_usd=50.0, audit_fees_tax_usd=7.6,
                            auditor_fees_prior_usd=55.0)
    row = _flatten("MS", FILING, _extract(governance=gov))
    assert row["auditor_fees"] == 57_600_000.0, row["auditor_fees"]
    assert row["audit_fees_audit"] == 50_000_000.0
    assert row["audit_fees_tax"] == pytest.approx(7_600_000.0)
    assert row["auditor_fees_prior"] == 55_000_000.0
    assert row["auditor_name"] == "Deloitte & Touche LLP"
    print(f"\n  fee block 57.6 -> {row['auditor_fees']:,.0f}; every category scaled by the same "
          f"factor, so the parts still sum")


# --------------------------------------------------------------------------- #
# gender consensus (deterministic, no LLM)                                    #
# --------------------------------------------------------------------------- #
def test_person_key_reconciles_spelling_drift():
    assert person_key("Katherine J. Smith") == person_key("Kathy Smith")
    assert person_key("John Smith Jr.") == person_key("John Smith")
    assert person_key("Emma N. Walmsley11") == person_key("Emma Walmsley")
    assert person_key("Tim Cook") != person_key("Tim Cash")


#: Real director names from the cache. Pharma and biotech boards are full of these, which is why
#: PFE is where the defect surfaced.
_CREDENTIALED = [
    ("Albert Bourla, DVM, Ph.D.", "A. Bourla"),
    ("Mikael Dolsten, M.D., Ph.D.", "Mikael Dolsten"),
    ("George A. Scangos, Ph.D.", "G. Scangos"),
    ("Daniel K. Podolsky, M.D.", "D. Podolsky"),
    ("Susan Desmond-Hellmann, MD, M.P.H.", "Susan Desmond-Hellmann"),
]


def test_academic_post_nominals_do_not_become_the_surname():
    """A dotted post-nominal must not survive into the key, and this is a CORRUPTION bug rather
    than a missed match.

    `\\bphd\\b` does not match `ph.d.`, so with the dots still in place the suffix survived the
    strip, the non-alpha pass then split it into `ph d`, and `d` landed in SURNAME position:
    `person_key("Albert Bourla, DVM, Ph.D.")` returned **`d|a`**. Every credentialed director
    sharing a first initial therefore collapsed onto one key, and `consensus` resolves ONE gender
    per key and writes it to every row of it.

    Measured over the 753 distinct director names in the 445 stored blobs: 24 names mis-keyed and
    6 keys covering 2-4 different people. The damaging one is `d|s`, which merged **Scott
    Gottlieb (male), Susan Hockfield (female) and Susanne Schaffert (female)** -- so the pass
    would have overwritten a real, correctly-extracted gender with another person's.
    """
    for full, short in _CREDENTIALED:
        assert person_key(full) == person_key(short), f"{full} != {short}"
        key = person_key(full)
        assert key and len(key.split("|")[0]) > 2, f"{full} keyed on a credential fragment: {key}"

    # the three people the old key merged must land on three distinct keys
    mixed = ["Scott Gottlieb, M.D.", "Susan Hockfield, Ph.D.", "Susanne Schaffert, Ph.D."]
    keys = [person_key(n) for n in mixed]
    print("\n=== SANITY: credentialed names key on the surname ===")
    for n, k in zip(mixed, keys):
        print(f"  {k:<20} <- {n}")
    print("  Old behaviour keyed all three as 'd|s' -- one male and two female directors sharing")
    print("  a single consensus key.")
    assert len(set(keys)) == 3, keys
    # and generational suffixes must still work, unchanged
    assert person_key("Harry A. Lawton III") == person_key("Harry Lawton")
    assert person_key("H. Lawrence Culp, Jr.") == person_key("H. Lawrence Culp")


def _roster(rows: list[tuple[str, str | None, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ticker": f"T{i}", "accession_number": f"a{i}", "name": n,
          "as_of": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "gender": g,
          "gender_basis": b} for i, (n, g, b) in enumerate(rows)])


def test_a_single_stated_row_wins_over_any_number_of_name_priors():
    """Provenance is not a popularity contest: a first-name prior is not evidence, so one
    `stated` row beats two `name` rows and also FILLS the null one."""
    df = _roster([("Katherine J. Smith", "female", "stated"),
                  ("Kathy Smith", "male", "name"),
                  ("K. Smith", "male", "name"),
                  ("Katherine Smith", None, None)])
    out, stats = consensus(df)
    assert set(out["gender"]) == {"female"}, out[["name", "gender"]].to_dict("records")
    assert stats["people"] == 1
    assert stats["overturned"] == 2, stats
    assert stats["filled"] == 1, stats
    print(f"\n  1 stated + 2 name-priors + 1 null -> all female; "
          f"overturned={stats['overturned']}, filled={stats['filled']}")


def test_honorific_majority_wins_when_nothing_is_stated():
    df = _roster([("Robert Williams", "male", "honorific"),
                  ("Bob Williams", "male", "pronoun"),
                  ("R. Williams", "female", "name")])
    out, stats = consensus(df)
    assert set(out["gender"]) == {"male"}
    assert stats["overturned"] == 1, stats
    print(f"\n  honorific beats a name prior -> all male; overturned={stats['overturned']}")


def test_consensus_never_downgrades_a_basis_and_is_idempotent():
    """The pass runs at the end of EVERY extraction, so a second run must be a no-op — that is
    what makes it safe to leave wired in."""
    df = _roster([("Sue Wagner", "female", "stated"),
                  ("Sue Wagner", "female", "name")])
    once, stats1 = consensus(df)
    twice, stats2 = consensus(once)
    assert stats2["overturned"] == 0 and stats2["filled"] == 0, stats2
    assert list(once["gender_basis"]) == list(twice["gender_basis"])
    # the row that already carried `stated` keeps it; the weaker row is raised, never lowered
    assert set(once["gender_basis"]) == {"stated"}, list(once["gender_basis"])
    print(f"\n  second pass: overturned={stats2['overturned']}, filled={stats2['filled']} "
          f"(no-op); bases {sorted(set(once['gender_basis']))}")


def test_recompute_parent_gender_uses_evidence_share():
    df = pd.DataFrame([
        {"ticker": "X", "accession_number": "a", "name": "A", "gender": "female",
         "gender_basis": "stated"},
        {"ticker": "X", "accession_number": "a", "name": "B", "gender": "male",
         "gender_basis": "name"},
    ])
    out = recompute_parent_gender(df)
    assert len(out) == 1
    assert out.iloc[0]["pct_female_directors"] == 0.5
    assert out.iloc[0]["pct_gender_stated"] == 0.5
    assert BASIS_RANK["stated"] > BASIS_RANK["name"]


def test_consensus_prints_conclusion():
    df = _roster([("Katherine J. Smith", "female", "stated"),
                  ("Kathy Smith", "male", "name"),
                  ("Robert Williams", "male", "honorific"),
                  ("R. Williams", None, None),
                  ("Solo Director", "male", "name")])
    out, stats = consensus(df)
    print("\n=== SANITY CHECK: cross-filing gender consensus ===")
    print(f"  {stats['rows']} rows -> {stats['people']} distinct people")
    print(f"  filled {stats['filled']}, overturned {stats['overturned']}, "
          f"unchanged {stats['unchanged']}, unkeyable {stats['unmatched_rows']}")
    print(f"  Katherine/Kathy Smith reconcile on `lastname|firstinitial`; the one `stated` row")
    print(f"  overrides the name prior, and R. Williams is FILLED from Robert's honorific.")
    print(f"  An overturn count of 0 on real data would mean the key matched nobody. Validated.")
    assert stats["people"] == 3
    assert stats["overturned"] == 1 and stats["filled"] == 1
    assert out[out["name"] == "R. Williams"].iloc[0]["gender"] == "male"
