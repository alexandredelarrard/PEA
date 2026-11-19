"""Configured source coverage starts, distinct from actual source publication eligibility."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Table

_DERIVED_FEATURES = frozenset(
    {
        "ic_inst_breadth_chg",
        "ic_inst_shares_chg",
        "ic_inst_new_buyer_ratio",
        "ic_inst_exit_ratio",
        "ic_inst_cluster_buying",
        "ic_shortvol_high_x_weak_price",
    }
)


def _known_fields(table: Table) -> set[str]:
    fields = set(table.pk) | set(table.read_columns) | set(table.date_type_cols)
    if table.date_col:
        fields.add(table.date_col)
    if table.ticker_col:
        fields.add(table.ticker_col)
    return fields


def _date(value: object, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date, got {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"{label} must be an ISO date, got {value!r}")
    if parsed.tz is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


@dataclass(frozen=True)
class InstitutionalAvailability:
    """Validated institutionals source/field availability loaded from configuration.

    The YAML owns start boundaries. Runtime dependency masks still own whether a particular
    ticker-date has the price, denominator, filing state, or minimum history required to
    construct a signal. Keeping those facts separate prevents a missing calculation from
    being mistaken for a source that did not yet exist.
    """

    tables: Mapping[str, pd.Timestamp]

    @classmethod
    def from_config(cls, config: DictConfig) -> InstitutionalAvailability:
        """Keep extraction starts separate from observed support and publication eligibility."""
        tables = {
            str(name): _date(item.start_date, f"{name}.start_date") for name, item in config.data.tables.items() if item.get("start_date") is not None
        }
        if not tables:
            raise ValueError("data.yml declares no extraction starts")
        return cls(tables=tables)

    def source_date(self, table: Table, field: str | None = None) -> pd.Timestamp:
        """The configured coverage start; field identities remain validated against the schema."""
        rule = self.tables.get(table.name)
        if rule is None:
            raise KeyError(f"No institutionals availability declaration for {table.name}")
        if field is not None and field not in _known_fields(table):
            raise KeyError(f"Unknown field {field!r} for {table.name}")
        return rule

    def derived_date(
        self,
        feature: str,
        dependencies: Iterable[tuple[Table, str | None]] = (),
    ) -> pd.Timestamp:
        """Resolve an explicit derived override or the latest dependency boundary."""
        if feature not in _DERIVED_FEATURES:
            raise KeyError(f"Unknown institutionals derived feature {feature!r}")
        starts = [self.source_date(table, field) for table, field in dependencies]
        if not starts:
            raise KeyError(f"No availability override or dependencies declared for derived feature {feature!r}")
        return max(starts)

    @staticmethod
    def date_mask(
        index: pd.DatetimeIndex,
        columns: pd.Index,
        start: pd.Timestamp,
    ) -> pd.DataFrame:
        """Boolean `(date x ticker)` mask, inclusive of `start`."""
        idx = pd.DatetimeIndex(index).normalize()
        row = np.zeros(len(idx), dtype=bool) if pd.isna(start) else idx >= pd.Timestamp(start).normalize()
        values = np.broadcast_to(row[:, None], (len(idx), len(columns))).copy()
        return pd.DataFrame(values, index=idx, columns=columns, dtype=bool)

    @staticmethod
    def through_mask(
        index: pd.DatetimeIndex,
        columns: pd.Index,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Boolean mask through an inclusive observed source frontier."""
        idx = pd.DatetimeIndex(index).normalize()
        row = np.zeros(len(idx), dtype=bool) if pd.isna(end) else idx <= pd.Timestamp(end).normalize()
        values = np.broadcast_to(row[:, None], (len(idx), len(columns))).copy()
        return pd.DataFrame(values, index=idx, columns=columns, dtype=bool)

    def source_mask(
        self,
        table: Table,
        index: pd.DatetimeIndex,
        columns: pd.Index,
        *,
        field: str | None = None,
        requirements: Sequence[pd.DataFrame] = (),
    ) -> pd.DataFrame:
        mask = self.date_mask(index, columns, self.source_date(table, field))
        return self.combine(mask, *requirements)

    @staticmethod
    def combine(*masks: pd.DataFrame) -> pd.DataFrame:
        """AND aligned boolean requirements without silently reindexing them."""
        if not masks:
            raise ValueError("At least one availability mask is required")
        first = masks[0]
        if not isinstance(first, pd.DataFrame) or not all(pd.api.types.is_bool_dtype(dtype) for dtype in first.dtypes):
            raise TypeError("Availability masks must be boolean DataFrames")
        out = first.copy()
        for mask in masks[1:]:
            if not isinstance(mask, pd.DataFrame):
                raise TypeError("Availability masks must be boolean DataFrames")
            if not mask.index.equals(out.index) or not mask.columns.equals(out.columns):
                raise ValueError("Availability masks must share the same index and columns")
            if not all(pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes):
                raise TypeError("Availability masks must be boolean DataFrames")
            out &= mask
        return out
