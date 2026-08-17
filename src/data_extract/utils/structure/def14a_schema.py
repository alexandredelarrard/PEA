"""
def14a_schema.py  (src/data_extract/utils/structure/def14a_schema.py)
---------------------------------------------------------------------
Pydantic v2 schema for structured extraction of SEC DEF 14A proxy statements.
Deliberately trimmed to the governance / compensation / ownership signals that
carry alpha for a long/short book AND are reliably disclosed in a proxy — the LLM
is constrained to this schema, so a narrow schema = cheaper, more accurate calls.

Design choices for cost / accuracy:
  * board composition, ownership and governance provisions are captured as DIRECT
    scalar fields in `governance` (from the compact "governance highlights" /
    "beneficial ownership" summaries) rather than reconstructed from long per-person
    lists — this survives aggressive text trimming and is far more reliable.
  * the per-director list is kept (ages / tenure / gender / over-boarding need it)
    but stripped of low-signal fields (committee lists, "director since" year).
  * executive-officer and per-holder ownership lists are dropped (low alpha, verbose).
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DirectorInfo(BaseModel):
    name: str = Field(description="Full name of the director or nominee")
    age: Optional[int] = Field(None, description="Age in years")
    tenure_years: Optional[float] = Field(
        None, description="Years served on the board (derive from 'director since YYYY')")
    is_independent: Optional[bool] = Field(
        None, description="True if classified as an independent director")
    gender: Optional[str] = Field(
        None, description="'male' or 'female' if stated or clearly inferable, else null")
    other_public_company_boards: Optional[int] = Field(
        None, description="Number of OTHER public-company boards this director serves on (over-boarding)")


class ExecutiveCompensation(BaseModel):
    name: str = Field(description="Full name of the named executive officer (NEO)")
    title: str = Field(description="Official title or position")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year of the compensation row")
    salary_usd: Optional[float] = Field(None, description="Base salary in USD")
    bonus_usd: Optional[float] = Field(
        None, description="Discretionary cash 'Bonus' column in USD (NOT non-equity incentive)")
    stock_awards_usd: Optional[float] = Field(
        None, description="Grant-date fair value of stock/RSU awards ('Stock Awards' column), USD")
    option_awards_usd: Optional[float] = Field(
        None, description="Grant-date fair value of option awards ('Option Awards' column), USD")
    non_equity_incentive_usd: Optional[float] = Field(
        None, description="'Non-Equity Incentive Plan Compensation' column, USD")
    all_other_comp_usd: Optional[float] = Field(
        None, description="'All Other Compensation' column, USD")
    total_compensation_usd: Optional[float] = Field(
        None, description="'Total' column of the Summary Compensation Table, USD")


class GovernanceProfile(BaseModel):
    # ---- board composition (from the governance/board-highlights summary) ----
    board_size: Optional[int] = Field(None, description="Total number of directors on the board")
    n_independent_directors: Optional[int] = Field(
        None, description="Number of independent directors (e.g. '7 of our 8 directors are independent')")
    n_women_directors: Optional[int] = Field(
        None, description="Number of women / female directors")
    # ---- board technology / software maturity (AI-adoption governance signal) ----
    n_technology_directors: Optional[int] = Field(
        None, description="Number of directors with material technology / software / IT / "
                          "cybersecurity / digital-transformation expertise, as shown in the board "
                          "SKILLS-AND-QUALIFICATIONS matrix or the director bios; null if not disclosed")
    technology_committee: Optional[bool] = Field(
        None, description="True if the board has a dedicated technology / cybersecurity / digital / "
                          "innovation committee (beyond the usual audit / compensation / nominating "
                          "committees); else False")
    # ---- board leadership & anti-takeover provisions (infer FALSE if not present) ----
    independent_chair: Optional[bool] = Field(
        None, description="True if the Board Chair is independent (not the CEO)")
    ceo_is_board_chair: Optional[bool] = Field(
        None, description="True if the CEO also serves as Chair of the Board (CEO duality)")
    lead_independent_director: Optional[bool] = Field(
        None, description="True if the company has a Lead Independent Director")
    classified_board: Optional[bool] = Field(
        None, description="True if the board is classified/staggered (multi-year terms); else False")
    dual_class_shares: Optional[bool] = Field(
        None, description="True if there is a dual-class / super-voting share structure; else False")
    poison_pill: Optional[bool] = Field(
        None, description="True if a shareholder rights plan (poison pill) is in place; else False")
    majority_voting_for_directors: Optional[bool] = Field(
        None, description="True if directors are elected by majority (vs plurality) voting")
    # ---- pay governance ----
    say_on_pay_support_pct: Optional[float] = Field(
        None, description="Most recent say-on-pay approval as a decimal (0.95 = 95% for)")
    ceo_pay_ratio: Optional[float] = Field(
        None, description="CEO-to-median-employee pay ratio (e.g. 250 for 250:1)")
    median_employee_pay_usd: Optional[float] = Field(
        None, description="Annual total compensation of the median employee, USD")
    auditor_fees_usd: Optional[float] = Field(
        None, description="TOTAL fees paid to the independent auditor for the year (all fee categories), USD")
    # ---- ownership / alignment (from the beneficial-ownership summary) ----
    insider_ownership_pct: Optional[float] = Field(
        None, description="Percent of shares owned by ALL directors and executive officers AS A GROUP, "
                          "as a decimal (0.03 = 3%); null if shown as '*'/<1%")
    ceo_ownership_pct: Optional[float] = Field(
        None, description="Percent of shares beneficially owned by the CEO, as a decimal; null if '*'/<1%")
    n_five_percent_holders: Optional[int] = Field(
        None, description="Number of beneficial owners holding 5% or more of the shares")


class Def14AExtract(BaseModel):
    company_name: Optional[str] = Field(None, description="Legal name of the company")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year covered by this proxy")

    # CEO summary (top-level so it is ALWAYS surfaced even when the CEO also appears
    # in the directors / compensation lists)
    ceo_name: Optional[str] = Field(None, description="Full name of the Chief Executive Officer")
    ceo_age: Optional[int] = Field(None, description="Age of the CEO in years")
    ceo_since_year: Optional[int] = Field(None, description="Year the CEO took the role")
    ceo_is_founder: Optional[bool] = Field(
        None, description="True if the CEO founded or co-founded the company")
    ceo_is_board_chair: Optional[bool] = Field(
        None, description="True if the CEO is also Chair of the Board")

    directors: list[DirectorInfo] = Field(
        default_factory=list, description="All director nominees / current directors")
    compensation: list[ExecutiveCompensation] = Field(
        default_factory=list,
        description="Summary Compensation Table rows for the MOST RECENT fiscal year shown, one per NEO")
    governance: Optional[GovernanceProfile] = Field(
        None, description="Board composition, leadership, anti-takeover, say-on-pay, pay-ratio, "
                          "auditor-fee and beneficial-ownership summary facts")
