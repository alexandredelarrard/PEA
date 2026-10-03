"""
def14a_schema.py  (src/data_extract/utils/schemas/def14a_schema.py)
---------------------------------------------------------------------
Pydantic v2 schema for structured extraction of SEC DEF 14A proxy statements.
Deliberately trimmed to the governance / compensation / ownership signals that
carry alpha for a long/short book AND are reliably disclosed in a proxy — the LLM
is constrained to this schema, so a narrow schema = cheaper, more accurate calls.

Design choices for cost / accuracy:
  * board composition and governance provisions are captured as DIRECT scalar fields in
    `governance` (from the compact "governance highlights" summaries) rather than
    reconstructed from long per-person lists — this survives aggressive text trimming.
  * the per-person LISTS are kept and flattened into their own tables downstream, because
    the table-anchored carve now delivers the actual tables to the model: 34,741
    per-executive rows and 87,984 director rows already sit inside `def14a_llm.def14a_json`
    unqueryable, so surfacing them costs no extra tokens.

`n_technology_directors` and `technology_committee` were REMOVED: they were an opinion, not
an extraction. Mean |Δ| of 1.06 directors between consecutive filings of the same company
(only 38.8% unchanged), and wrong by 7x on HUBB 2022, whose own skills matrix states
"Cybersecurity and Technology 78%" of 9 directors.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DirectorInfo(BaseModel):
    name: str = Field(description="Full name of the director or nominee")
    age: int | None = Field(None, description="Age in years")
    tenure_years: float | None = Field(None, description="Years served on the board (derive from 'director since YYYY')")
    is_independent: bool | None = Field(None, description="True if classified as an independent director")
    gender: str | None = Field(
        None,
        description="'male' or 'female'. Resolve IN THIS ORDER: (1) the proxy STATES it "
        "(a board-diversity matrix, a 'women directors' identification), "
        "(2) the HONORIFIC used for that director anywhere in the text "
        "(Mr./Ms./Mrs./Dr. is not gendered), (3) the PRONOUNS used in that "
        "director's biography (he/him/his vs she/her/hers), (4) the first "
        "name -- LAST RESORT ONLY. Null if none of the four applies",
    )
    gender_basis: str | None = Field(
        None, description="Which rule resolved `gender`: 'stated' | 'honorific' | 'pronoun' | 'name'. NEVER leave this null when `gender` is set"
    )
    other_public_company_boards: int | None = Field(
        None,
        description="Number of OTHER public-company boards this director serves on "
        "(over-boarding). Set 0 ONLY when the proxy explicitly shows a count of "
        "zero or states the director serves on no other public boards. Leave "
        "null when other-board service is simply not disclosed for that "
        "director — a null and a 0 are NOT interchangeable here, and mixing "
        "them biases the board average downward",
    )


class ExecutiveCompensation(BaseModel):
    """One Summary Compensation Table row: ONE named executive officer in ONE fiscal year.

    Item 402(c) requires THREE fiscal years, and the table-anchored carve now delivers the
    whole table, so the list carries every (NEO x year) cell the SCT shows -- not just the
    latest year, which is what the schema used to ask for.
    """

    name: str = Field(description="Full name of the named executive officer (NEO)")
    title: str = Field(description="Official title or position")
    fiscal_year: int | None = Field(
        None, description="Fiscal year of THIS row, from the table's Year column. Required to tell the three yearly rows of one executive apart"
    )
    salary_usd: float | None = Field(None, description="Base salary in USD")
    bonus_usd: float | None = Field(None, description="Discretionary cash 'Bonus' column in USD (NOT non-equity incentive)")
    stock_awards_usd: float | None = Field(None, description="Grant-date fair value of stock/RSU awards ('Stock Awards' column), USD")
    option_awards_usd: float | None = Field(None, description="Grant-date fair value of option awards ('Option Awards' column), USD")
    non_equity_incentive_usd: float | None = Field(None, description="'Non-Equity Incentive Plan Compensation' column, USD")
    # The SEVENTH component. The post-2006 SCT has seven and the schema modelled six, which is
    # why the residual against `total` was POSITIVE on 97.5% of non-reconciling rows at a median
    # of $319,367 -- the signature of a missing column, not of misread values.
    pension_change_usd: float | None = Field(
        None, description="'Change in Pension Value and Nonqualified Deferred Compensation Earnings' column, USD"
    )
    all_other_comp_usd: float | None = Field(None, description="'All Other Compensation' column, USD")
    total_compensation_usd: float | None = Field(None, description="'Total' column of the Summary Compensation Table, USD")


class DirectorCompensation(BaseModel):
    """One row of the Director Compensation Table (Item 402(k)).

    Two regulatory facts make this table load-bearing rather than a nice-to-have:
      * 402(k) covers NON-EMPLOYEE directors only, so a nominee's presence here *is* the
        definition of an outside director -- a far stronger signal than the LLM's
        `is_independent` flag (78% fill, 86% concordance). The 8-K vote role map depends on it.
      * 402(k) requires the LAST COMPLETED FISCAL YEAR ONLY, so this list is single-year by
        regulation -- unlike the SCT, which carries three.
    The table exists only from the 2008 proxy season (Reg S-K 2006, FY ending >= 2006-12-15).
    """

    name: str = Field(description="Full name of the non-employee director")
    fiscal_year: int | None = Field(None, description="Fiscal year covered (Item 402(k) shows only the last completed one)")
    fees_earned_usd: float | None = Field(
        None,
        description="The CASH RETAINER column, whatever it is labelled: 'Fees Earned or Paid in Cash', 'Cash Fees', 'Retainer', 'Annual Retainer'",
    )
    stock_awards_usd: float | None = Field(
        None, description="The share-based award column, whatever it is labelled: 'Stock Awards', 'Restricted Stock Units', 'Share Awards'"
    )
    option_awards_usd: float | None = Field(None, description="'Option Awards' column, USD")
    non_equity_incentive_usd: float | None = Field(None, description="'Non-Equity Incentive Plan Compensation' column, USD")
    pension_change_usd: float | None = Field(
        None, description="'Change in Pension Value and Nonqualified Deferred Compensation Earnings' column, USD"
    )
    all_other_comp_usd: float | None = Field(None, description="'All Other Compensation' column, USD")
    total_compensation_usd: float | None = Field(None, description="'Total' column of the Director Compensation Table, USD")


class BeneficialOwner(BaseModel):
    """One row of the beneficial-ownership table (Item 403).

    Knowingly redundant with 13F / SC 13D-G / Forms 3-4-5, which remain the preferred sources;
    these rows land because they are ~100% present and 95-97% already inside the carve, so
    their marginal cost is about zero. Their as-of dates never align with 13F's quarter-end.
    """

    holder_name: str = Field(
        description="Name of the beneficial owner. EXCLUDE subtotal and 'as a group' rows -- "
        "that aggregate is already captured as the `insider_ownership_pct` scalar. "
        "Exclude a row whose 'name' is only a street address"
    )
    holder_type: str | None = Field(
        None, description="'5pct_holder' for an institution or other >=5% owner, 'director_officer' for a named director or executive officer"
    )
    shares: float | None = Field(None, description="Number of shares beneficially owned")
    percent_of_class: float | None = Field(
        None,
        description="Percent of the CLASS of shares outstanding, as a decimal "
        "(0.0963 = 9.63%). NULL for '*' or '<1%' — those are a BOUND, not a "
        "measurement. If the row shows only ONE percent column and it is not "
        "labelled as voting power, put it here",
    )
    percent_of_voting_power: float | None = Field(
        None,
        description="This holder's '% of total voting power' / 'combined voting power', "
        "as a decimal, when the table prints such a column BESIDE the percent "
        "of class. NULL when the table has no voting-power column. Do NOT "
        "copy the percent of class here",
    )


class GovernanceProfile(BaseModel):
    # ---- board composition (from the governance/board-highlights summary) ----
    board_size: int | None = Field(None, description="Total number of directors on the board")
    n_independent_directors: int | None = Field(None, description="Number of independent directors (e.g. '7 of our 8 directors are independent')")
    n_women_directors: int | None = Field(None, description="Number of women / female directors as the proxy itself states it")
    # ---- board leadership & anti-takeover provisions ----
    independent_chair: bool | None = Field(None, description="True if the Board Chair is independent (not the CEO)")
    ceo_is_board_chair: bool | None = Field(None, description="True if the CEO also serves as Chair of the Board (CEO duality)")
    lead_independent_director: bool | None = Field(None, description="True if the company has a Lead Independent Director")
    # these two are STRUCTURALLY always disclosed -> infer FALSE from silence
    classified_board: bool | None = Field(None, description="True if the board is classified/staggered (multi-year terms); else False")
    dual_class_shares: bool | None = Field(None, description="True if there is a dual-class / super-voting share structure; else False")
    # TRI-STATE, unlike the two above: null when the proxy is SILENT. Inferring FALSE from
    # silence made `poison_pill` TRUE in 0.1% of rows (degenerate) and made `majority_voting`
    # flip 21.2% year-over-year on a bylaw that does not change.
    poison_pill: bool | None = Field(
        None,
        description="True if a shareholder rights plan (poison pill) is in place, False if "
        "the proxy states there is none; null if the proxy is SILENT",
    )
    majority_voting_for_directors: bool | None = Field(
        None,
        description="True if directors are elected by majority (vs plurality) voting, False "
        "if the proxy states plurality; null if the proxy is SILENT",
    )
    # ---- pay governance ----
    say_on_pay_support_pct: float | None = Field(
        None,
        description="Most recent say-on-pay approval as a decimal (0.95 = 95% for). A "
        "genuinely failed vote is real signal — report 0.31 if the proxy says "
        "31%, do not assume a low value is an error",
    )
    ceo_pay_ratio: float | None = Field(None, description="CEO-to-median-employee pay ratio (e.g. 250 for 250:1)")
    median_employee_pay_usd: float | None = Field(None, description="Annual total compensation of the median employee, USD")
    # ---- auditor: name, tenure and the fee breakdown ----
    # `auditor_name` was the worst column in the old edgar table at 2.05% fill, while the firm
    # name is present in 98% of documents and already inside 89% of carves.
    auditor_name: str | None = Field(
        None,
        description="Name of the independent registered public accounting firm, e.g. "
        "'PricewaterhouseCoopers LLP'. The FIRM NAME ONLY — not a sentence",
    )
    auditor_since_year: int | None = Field(
        None, description="First year of the auditor's engagement, when the proxy states it ('has served as our auditor since 1934')"
    )
    # Every fee field must be converted to WHOLE USD. This is the fix for the measured 1000x
    # error on 8 of the 10 smallest values: MS reported 57.6 for $57.6M and TSLA 10,919 for
    # $10.9M, both because a "($ in millions)" / "(in thousands)" note was ignored.
    auditor_fees_usd: float | None = Field(
        None,
        description="TOTAL fees paid to the auditor for the CURRENT fiscal year, all "
        "categories, in WHOLE USD. Apply any '(in thousands)' / "
        "'($ in millions)' note from the table header or the sentence before "
        "it. If the table has NO 'Total' row, ADD UP the four categories "
        "(Audit + Audit-Related + Tax + All Other) — do NOT report the "
        "'Audit Fees' line as the total",
    )
    audit_fees_audit_usd: float | None = Field(None, description="'Audit Fees' category, current fiscal year, WHOLE USD")
    audit_fees_audit_related_usd: float | None = Field(None, description="'Audit-Related Fees' category, current fiscal year, WHOLE USD")
    audit_fees_tax_usd: float | None = Field(None, description="'Tax Fees' category, current fiscal year, WHOLE USD")
    audit_fees_other_usd: float | None = Field(None, description="'All Other Fees' category, current fiscal year, WHOLE USD")
    auditor_fees_prior_usd: float | None = Field(
        None,
        description="TOTAL auditor fees for the PRIOR fiscal year, WHOLE USD — the proxy shows two years side by side. Enables a fee-growth signal",
    )
    insider_ownership_pct: float | None = Field(
        None,
        description="Percent of TOTAL shares outstanding (ALL classes combined) held by "
        "ALL directors and executive officers AS A GROUP, as a decimal "
        "(0.03 = 3%); null if shown as '*'/<1%. If the table prints only "
        "PER-CLASS percentage columns (e.g. 'Class A %' and 'Class B %') "
        "and no combined/total column, return NULL — do NOT substitute one "
        "class's percentage. If there is a single percent column not "
        "labelled as voting power, use it here",
    )
    insider_shares: float | None = Field(
        None,
        description="TOTAL NUMBER OF SHARES (not a percent) beneficially owned by all "
        "directors and executive officers AS A GROUP, summed across every "
        "share class if the table separates them. Null if the table shows "
        "no group share count",
    )
    insider_voting_pct: float | None = Field(
        None,
        description="'Percent of TOTAL VOTING POWER' (or 'combined voting power') for ALL "
        "directors and executive officers AS A GROUP, as a decimal, when the "
        "table prints such a column. Null if the table has no voting-power "
        "column. Do NOT copy the ownership percentage here",
    )
    ceo_ownership_pct: float | None = Field(
        None,
        description="Percent of TOTAL shares outstanding (ALL classes combined) "
        "beneficially owned by the CEO, as a decimal; null if '*'/<1%. As "
        "with the group figure: if the table prints only PER-CLASS percent "
        "columns and no combined one, return NULL rather than one class's "
        "percentage. A single unlabelled percent column belongs here",
    )
    ceo_voting_pct: float | None = Field(
        None,
        description="The CEO's '% of total voting power' / 'combined voting power', as a "
        "decimal, when the table prints such a column. Null if the table has "
        "no voting-power column. Do NOT copy the ownership percentage here",
    )
    n_five_percent_holders: int | None = Field(None, description="Number of beneficial owners holding 5% or more of the shares")


class Def14AExtract(BaseModel):
    company_name: str | None = Field(None, description="Legal name of the company")
    fiscal_year: int | None = Field(None, description="Fiscal year covered by this proxy")

    # CEO summary (top-level so it is ALWAYS surfaced even when the CEO also appears
    # in the directors / compensation lists)
    ceo_name: str | None = Field(None, description="Full name of the Chief Executive Officer")
    ceo_age: int | None = Field(None, description="Age of the CEO in years")
    ceo_since_year: int | None = Field(None, description="Year the CEO took the role")
    ceo_is_founder: bool | None = Field(None, description="True if the CEO founded or co-founded the company")
    ceo_is_board_chair: bool | None = Field(None, description="True if the CEO is also Chair of the Board")

    directors: list[DirectorInfo] = Field(default_factory=list, description="All director nominees / current directors")
    compensation: list[ExecutiveCompensation] = Field(
        default_factory=list,
        description="EVERY row of the Summary Compensation Table: one entry per named executive "
        "officer PER FISCAL YEAR shown (the table normally carries three years). Do "
        "not collapse an executive's three years into one entry, and do not return "
        "only the latest year",
    )
    director_compensation: list[DirectorCompensation] = Field(
        default_factory=list,
        description="Every row of the Director Compensation Table (Item 402(k)), one per "
        "non-employee director. Empty for proxies before the 2008 season",
    )
    ownership_holders: list[BeneficialOwner] = Field(
        default_factory=list,
        description="Every row of the beneficial-ownership table(s): the >=5% holders AND the "
        "named directors and executive officers. Exclude subtotal / 'as a group' rows",
    )
    governance: GovernanceProfile | None = Field(
        None,
        description="Board composition, leadership, anti-takeover, say-on-pay, pay-ratio, "
        "auditor name / tenure / fee breakdown and beneficial-ownership summary facts",
    )
