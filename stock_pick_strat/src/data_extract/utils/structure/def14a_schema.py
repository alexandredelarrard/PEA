"""
def14a_schema.py  (src/data_extract/utils/def14a_schema.py)
------------------------------------------------------------
Pydantic v2 schema for structured extraction of SEC DEF 14A proxy statements.
Exhaustive coverage of the governance / compensation / ownership signals that
matter for a long/short equity book — the LLM is constrained to this schema.

Groups:
  directors           name, age, tenure, independence, board role, gender,
                      over-boarding (# other public boards), financial expert
  executive_officers  name, age, title, appointment year
  ceo (top level)     name, AGE, title, since year, founder flag, chair duality
  compensation        Summary Compensation Table per NEO (salary, bonus, stock,
                      option, non-equity incentive, pension/deferred, all-other,
                      total)
  share_ownership     shares & % owned by directors / officers / 5%+ holders
  governance          board structure, anti-takeover provisions, say-on-pay
                      support, CEO pay ratio, auditor
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DirectorInfo(BaseModel):
    name: str = Field(description="Full name of the director or nominee")
    age: Optional[int] = Field(None, description="Age in years")
    tenure_years: Optional[float] = Field(
        None, description="Years served on the board (derive from 'director since YYYY')")
    director_since_year: Optional[int] = Field(None, description="Year first became a director")
    is_independent: Optional[bool] = Field(
        None, description="True if classified as an independent director")
    is_board_chair: Optional[bool] = Field(
        None, description="True if this director is Chair/Chairman of the Board")
    is_lead_independent_director: Optional[bool] = Field(
        None, description="True if this director is the Lead Independent Director")
    gender: Optional[str] = Field(
        None, description="'male' or 'female' if stated or clearly inferable, else null")
    other_public_company_boards: Optional[int] = Field(
        None, description="Number of OTHER public-company boards this director serves on (over-boarding)")
    audit_committee_financial_expert: Optional[bool] = Field(
        None, description="True if designated an audit-committee financial expert")
    committees: list[str] = Field(
        default_factory=list,
        description="Board committee memberships (e.g. ['Audit', 'Compensation'])")


class ExecutiveOfficer(BaseModel):
    name: str = Field(description="Full name of the executive officer")
    age: Optional[int] = Field(None, description="Age in years")
    title: str = Field(description="Official title or position")
    officer_since_year: Optional[int] = Field(None, description="Year appointed to an officer role")


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
    pension_and_deferred_usd: Optional[float] = Field(
        None, description="'Change in Pension Value and Nonqualified Deferred Compensation Earnings', USD")
    all_other_comp_usd: Optional[float] = Field(
        None, description="'All Other Compensation' column, USD")
    total_compensation_usd: Optional[float] = Field(
        None, description="'Total' column of the Summary Compensation Table, USD")


class ShareOwnership(BaseModel):
    name: str = Field(description="Full name of the beneficial owner")
    is_director: bool = Field(False, description="True if a director or nominee")
    is_officer: bool = Field(False, description="True if a named executive officer")
    is_five_percent_owner: bool = Field(
        False, description="True if a 5%+ beneficial owner (e.g. Vanguard, BlackRock)")
    shares_owned: Optional[int] = Field(
        None, description="Shares beneficially owned (incl. options/RSUs exercisable within 60 days)")
    percent_owned: Optional[float] = Field(
        None, description="Percent of class owned as a decimal (0.082 = 8.2%); null if '*' / <1%")


class GovernanceProfile(BaseModel):
    board_size: Optional[int] = Field(None, description="Total number of directors on the board")
    independent_chair: Optional[bool] = Field(
        None, description="True if the Board Chair is independent (not the CEO)")
    ceo_is_board_chair: Optional[bool] = Field(
        None, description="True if the CEO also serves as Chair of the Board (CEO duality)")
    lead_independent_director: Optional[bool] = Field(
        None, description="True if the company has a Lead Independent Director")
    classified_board: Optional[bool] = Field(
        None, description="True if the board is classified/staggered (multi-year terms)")
    dual_class_shares: Optional[bool] = Field(
        None, description="True if there is a dual-class / super-voting share structure")
    poison_pill: Optional[bool] = Field(
        None, description="True if a shareholder rights plan (poison pill) is in place")
    majority_voting_for_directors: Optional[bool] = Field(
        None, description="True if directors are elected by majority (vs plurality) voting")
    proxy_access: Optional[bool] = Field(
        None, description="True if shareholders have proxy access")
    say_on_pay_frequency_years: Optional[int] = Field(
        None, description="Say-on-pay advisory vote frequency in years (1 = annual)")
    say_on_pay_support_pct: Optional[float] = Field(
        None, description="Most recent say-on-pay approval as a decimal (0.95 = 95% for)")
    ceo_pay_ratio: Optional[float] = Field(
        None, description="CEO-to-median-employee pay ratio (e.g. 250 for 250:1)")
    median_employee_pay_usd: Optional[float] = Field(
        None, description="Annual total compensation of the median employee, USD")
    auditor_name: Optional[str] = Field(
        None, description="Independent registered public accounting firm (e.g. 'Ernst & Young LLP')")
    auditor_fees_usd: Optional[float] = Field(
        None, description="Total fees paid to the auditor for the year, USD")
    shareholder_proposals_count: Optional[int] = Field(
        None, description="Number of shareholder proposals on the ballot")


class Def14AExtract(BaseModel):
    company_name: Optional[str] = Field(None, description="Legal name of the company")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year covered by this proxy")

    # CEO summary (top-level so it is ALWAYS surfaced even when the CEO also
    # appears in the directors / officers / compensation lists)
    ceo_name: Optional[str] = Field(None, description="Full name of the Chief Executive Officer")
    ceo_age: Optional[int] = Field(None, description="Age of the CEO in years")
    ceo_title: Optional[str] = Field(None, description="Full title of the CEO")
    ceo_since_year: Optional[int] = Field(None, description="Year the CEO took the role")
    ceo_is_founder: Optional[bool] = Field(
        None, description="True if the CEO founded or co-founded the company")
    ceo_is_board_chair: Optional[bool] = Field(
        None, description="True if the CEO is also Chair of the Board")

    directors: list[DirectorInfo] = Field(
        default_factory=list, description="All director nominees / current directors")
    executive_officers: list[ExecutiveOfficer] = Field(
        default_factory=list, description="Named executive officers (may overlap with directors)")
    compensation: list[ExecutiveCompensation] = Field(
        default_factory=list,
        description="Summary Compensation Table rows for the most recent fiscal year shown")
    share_ownership: list[ShareOwnership] = Field(
        default_factory=list,
        description="Directors, officers and 5%+ owners from the Security Ownership table")
    governance: Optional[GovernanceProfile] = Field(
        None, description="Board-structure, anti-takeover, say-on-pay, pay-ratio and auditor facts")
