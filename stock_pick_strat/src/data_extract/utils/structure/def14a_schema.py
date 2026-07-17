"""
def14a_schema.py  (src/data_extract/utils/def14a_schema.py)
------------------------------------------------------------
Pydantic v2 schema for structured extraction of SEC DEF 14A proxy statements.

Fields covered:
  directors        — name, age, board tenure, independence, committee memberships
  executive_officers — name, age, title
  compensation     — Summary Compensation Table: salary, bonus, stock/option awards, total
  share_ownership  — Security Ownership table: shares and % owned by directors/officers
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DirectorInfo(BaseModel):
    name: str = Field(description="Full name of the director or nominee")
    age: Optional[int] = Field(None, description="Age in years")
    tenure_years: Optional[float] = Field(
        None, description="Years served on the board (derive from 'director since YYYY' if present)"
    )
    is_independent: Optional[bool] = Field(
        None, description="True if classified as independent director; False if not independent"
    )
    committees: list[str] = Field(
        default_factory=list,
        description="Board committee memberships (e.g. ['Audit', 'Compensation'])",
    )


class ExecutiveOfficer(BaseModel):
    name: str = Field(description="Full name of the executive officer")
    age: Optional[int] = Field(None, description="Age in years")
    title: str = Field(description="Official title or position")


class ExecutiveCompensation(BaseModel):
    name: str = Field(description="Full name of the named executive officer (NEO)")
    title: str = Field(description="Official title or position")
    fiscal_year: Optional[int] = Field(None, description="Fiscal year of the compensation")
    salary_usd: Optional[float] = Field(None, description="Base salary in USD")
    bonus_usd: Optional[float] = Field(
        None,
        description=(
            "Cash bonus or non-equity incentive plan compensation in USD. "
            "Use the 'Bonus' or 'Non-Equity Incentive Plan Compensation' column."
        ),
    )
    stock_awards_usd: Optional[float] = Field(
        None,
        description="Grant-date fair value of stock/RSU awards in USD ('Stock Awards' column)",
    )
    option_awards_usd: Optional[float] = Field(
        None,
        description="Grant-date fair value of option awards in USD ('Option Awards' column)",
    )
    total_compensation_usd: Optional[float] = Field(
        None,
        description="Total compensation in USD from the Summary Compensation Table ('Total' column)",
    )


class ShareOwnership(BaseModel):
    name: str = Field(description="Full name of the beneficial owner")
    is_director: bool = Field(False, description="True if this person is a director or nominee")
    is_officer: bool = Field(False, description="True if this person is a named executive officer")
    shares_owned: Optional[int] = Field(
        None,
        description=(
            "Number of shares beneficially owned, including shares acquirable "
            "through options or RSUs exercisable within 60 days"
        ),
    )
    percent_owned: Optional[float] = Field(
        None,
        description=(
            "Percentage of outstanding shares owned as a decimal "
            "(e.g. 0.082 for 8.2%). Use null if shown as '*' or 'less than 1%'."
        ),
    )


class Def14AExtract(BaseModel):
    company_name: Optional[str] = Field(None, description="Legal name of the company")
    fiscal_year: Optional[int] = Field(
        None, description="Fiscal year covered by this proxy statement"
    )
    directors: list[DirectorInfo] = Field(
        default_factory=list,
        description="All director nominees for election at the annual meeting",
    )
    executive_officers: list[ExecutiveOfficer] = Field(
        default_factory=list,
        description="Named executive officers listed in the proxy (may overlap with directors)",
    )
    compensation: list[ExecutiveCompensation] = Field(
        default_factory=list,
        description=(
            "Compensation rows from the Summary Compensation Table for named executive officers. "
            "Include all rows for the most recent fiscal year shown."
        ),
    )
    share_ownership: list[ShareOwnership] = Field(
        default_factory=list,
        description=(
            "Share ownership of directors, officers, and 5%+ beneficial owners "
            "from the Security Ownership of Management table."
        ),
    )
