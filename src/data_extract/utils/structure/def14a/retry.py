"""Five-section retry state and narrow attempt reservations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel

from src.data_extract.utils.common.empty_markers import marker_frame
from src.data_extract.utils.schemas.def14a_schema import AuditExtract, DirectorPayExtract, ExecutivePayExtract, GovernanceExtract, OwnershipExtract
from src.data_store.schema import Table, Tables


@dataclass(frozen=True)
class Section:
    """One independently callable domain and the table that owns its attempts."""

    key: str
    schema: type[BaseModel]
    table: Table
    parent_columns: tuple[str, ...]

    @property
    def action(self) -> str:
        return "def14a_" + self.key


SECTIONS = (
    Section(
        "governance",
        GovernanceExtract,
        Tables.def14a_directors,
        parent_columns=(
            "company_name",
            "n_directors",
            "board_size",
            "avg_director_age",
            "avg_board_tenure",
            "pct_independent_directors",
            "pct_female_directors",
            "avg_other_public_boards",
            "ceo_name_proxy",
            "ceo_age",
            "ceo_since_year",
            "ceo_is_founder",
            "ceo_is_board_chair",
            "independent_chair",
            "lead_independent_director",
            "classified_board",
            "dual_class_shares",
            "poison_pill",
            "majority_voting",
            "say_on_pay_support_pct",
            "ceo_pay_ratio",
            "median_employee_pay",
            "pct_gender_stated",
            "n_women_directors_vs_inferred",
        ),
    ),
    Section(
        "executive_pay",
        ExecutivePayExtract,
        Tables.def14a_executive_comp,
        parent_columns=(
            "fiscal_year_extract",
            "ceo_salary",
            "ceo_bonus",
            "ceo_stock_awards",
            "ceo_option_awards",
            "ceo_non_equity_incentive",
            "ceo_all_other_comp",
            "ceo_total_comp",
            "ceo_equity_pay_pct",
            "n_neos",
            "total_neo_comp",
            "sct_years",
        ),
    ),
    Section("director_pay", DirectorPayExtract, Tables.def14a_director_comp, parent_columns=("n_director_comp_rows",)),
    Section(
        "audit",
        AuditExtract,
        Tables.def14a_llm,
        parent_columns=(
            "auditor_name",
            "auditor_since_year",
            "auditor_fees",
            "audit_fees_audit",
            "audit_fees_audit_related",
            "audit_fees_tax",
            "audit_fees_other",
            "auditor_fees_prior",
        ),
    ),
    Section(
        "ownership",
        OwnershipExtract,
        Tables.def14a_ownership,
        parent_columns=(
            "insider_shares",
            "insider_ownership_pct",
            "insider_voting_pct",
            "ceo_ownership_pct",
            "ceo_voting_pct",
            "n_five_percent_holders",
            "n_ownership_rows",
        ),
    ),
)


def reservation_frame(table: Table, stamp: dict[str, object], df_saved: pd.DataFrame | None, retry: int) -> pd.DataFrame:
    """Persist a section attempt without carrying unrelated facts or replacing an archive."""
    # Keep every existing person row on the same attempt and preserve omitted facts.
    if retry not in (1, 2):
        raise ValueError("DEF14A retry must be 1 or 2")
    if df_saved is not None and not df_saved.empty:
        if "retry" in df_saved:
            previous = pd.to_numeric(df_saved["retry"], errors="coerce").max()
            retry = max(retry, int(previous)) if pd.notna(previous) else retry
        columns = [column for column in (*table.pk, "as_of") if column in df_saved]
        return df_saved[columns].copy().assign(retry=retry)

    # Empty child sections carry only marker keys, filing date and the declared CIK.
    if table is not Tables.def14a_llm:
        values = {column: value for column, value in stamp.items() if column in (*table.pk, "as_of", table.marker_cik_col)}
        return marker_frame(table, {**values, "retry": retry}, key_fill=dict(table.marker_key_fill))
    return pd.DataFrame([{**{column: stamp[column] for column in (*table.pk, "as_of")}, "retry": retry}])
