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

    @property
    def action(self) -> str:
        return "def14a_" + self.key


SECTIONS = (
    Section("governance", GovernanceExtract, Tables.def14a_directors),
    Section("executive_pay", ExecutivePayExtract, Tables.def14a_executive_comp),
    Section("director_pay", DirectorPayExtract, Tables.def14a_director_comp),
    Section("audit", AuditExtract, Tables.def14a_llm),
    Section("ownership", OwnershipExtract, Tables.def14a_ownership),
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

    # Empty child sections need a hidden row with all their primary-key fields.
    if table is not Tables.def14a_llm:
        return marker_frame(table, {**stamp, "retry": retry}, key_fill=dict(table.marker_key_fill))
    return pd.DataFrame([{**{column: stamp[column] for column in (*table.pk, "as_of")}, "retry": retry}])
