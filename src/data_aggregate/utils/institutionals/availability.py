"""
availability.py  (src/data_aggregate/utils/institutionals/availability.py)
--------------------------------------------------------------------------
THE ONE DECLARATION OF WHEN A 13F PERIOD BECOMES PUBLIC.

A 13F-HR reports positions as of the quarter END and is not public until it is filed, which
the statute puts within `period + 45 calendar days`. Turning that into a date the panel can
be stamped on needs two corrections the statute does not make, and both are measured:

  1. THE 45TH DAY IS OFTEN NOT A TRADING DAY -- 13 of 53 quarters (6 Saturdays, 7 Sundays,
     measured 2026-09-15 against `cube_part_prices`' own calendar). The deadline crowd then
     files on the next session, so the date is SNAPPED FORWARD onto the trading calendar.
     Snapping BACKWARDS would be a look-ahead; the next session is the first day the
     information can be acted on, which is the same rule `decay.snap_to_grid` applies to any
     event date that misses a session.
  2. THE DEADLINE ITSELF IS NOT WHEN THE DOLLARS ARRIVE. Filer COUNT completes early and
     SHARES do not -- small filers file early, the mega-filers file at the deadline and
     settle over the following sessions. So the snapped deadline is advanced
     `F13_SETTLE_TRADING_DAYS` trading days, a buffer sized by sweep; its constant carries
     the table.

⚠ SNAP ON THE TRADING CALENDAR, NOT ON `BDay`. A market holiday is a business day and is not
a session, so `BDay` lands the availability date on a day the panel has no row for and the
stamp is then silently carried by the following `reindex`.

⚠ THIS FUNCTION IS IMPORTED, NEVER RESTATED. `validate/institutionals.py` used to re-derive
the snap itself (`grid[np.clip(grid.searchsorted(deadlines))]`), which made the availability
rule two declarations that had to be kept in step by hand -- and the settle buffer existed in
only one of them. The builder stamps with this function and the leakage checks score against
it, so a change to the rule cannot move one without the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS, SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_store.schema import Table, Tables

_DERIVED_FEATURES = frozenset(
    {
        "ic_inst_breadth_chg",
        "ic_inst_shares_chg",
        "ic_inst_new_buyer_ratio",
        "ic_inst_exit_ratio",
        "ic_inst_cluster_buying",
        "ic_inst_flow_to_mcap",
        "ic_super_conviction_chg",
        "ic_super_full_exits",
        "ic_insider_discretionary_sell_mcap_60d",
        "ic_insider_planned_sell_mcap_60d",
        "ic_shortvol_high_x_weak_price",
    }
)


def _registered_tables() -> dict[str, Table]:
    """The schema registry by physical table name."""
    return {value.name: value for value in vars(Tables).values() if isinstance(value, Table)}


def _known_fields(table: Table) -> set[str]:
    fields = set(table.pk) | set(table.read_columns) | set(table.date_type_cols)
    if table.date_col:
        fields.add(table.date_col)
    if table.ticker_col:
        fields.add(table.ticker_col)
    return fields


def _date(value: object, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date, got {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"{label} must be an ISO date, got {value!r}")
    if parsed.tz is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


@dataclass(frozen=True)
class _TableAvailability:
    start: pd.Timestamp
    fields: Mapping[str, pd.Timestamp]


@dataclass(frozen=True)
class InstitutionalAvailability:
    """Validated institutionals source/field availability loaded from configuration.

    The YAML owns start boundaries. Runtime dependency masks still own whether a particular
    ticker-date has the price, denominator, filing state, or minimum history required to
    construct a signal. Keeping those facts separate prevents a missing calculation from
    being mistaken for a source that did not yet exist.
    """

    tables: Mapping[str, _TableAvailability]
    derived_features: Mapping[str, pd.Timestamp]

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> InstitutionalAvailability:
        """Parse either the `data_availability` block or its `institutionals` child."""
        root: object = config
        if isinstance(root, Mapping) and "data_availability" in root:
            root = root["data_availability"]
        if isinstance(root, Mapping) and "institutionals" in root:
            root = root["institutionals"]
        if not isinstance(root, Mapping):
            raise TypeError("data_availability.institutionals must be a mapping")

        registered = _registered_tables()
        tables: dict[str, _TableAvailability] = {}
        derived: dict[str, pd.Timestamp] = {}
        for table_name, raw_rule in root.items():
            name = str(table_name)
            if name == "derived_features":
                if not isinstance(raw_rule, Mapping):
                    raise TypeError("derived_features must be a feature -> date mapping")
                for feature, raw_date in raw_rule.items():
                    feature_name = str(feature)
                    if feature_name not in _DERIVED_FEATURES:
                        raise KeyError(f"Unknown institutionals derived feature {feature_name!r}")
                    derived[feature_name] = _date(raw_date, f"derived_features.{feature_name}")
                continue

            table = registered.get(name)
            if table is None:
                raise KeyError(f"Unknown institutionals availability table {name!r}")
            if not isinstance(raw_rule, Mapping):
                raise TypeError(f"{name} availability must be a field -> date mapping")
            if "__all__" not in raw_rule:
                raise KeyError(f"{name} availability requires an __all__ date")
            start = _date(raw_rule["__all__"], f"{name}.__all__")
            valid_fields = _known_fields(table)
            fields: dict[str, pd.Timestamp] = {}
            for field, raw_date in raw_rule.items():
                field_name = str(field)
                if field_name == "__all__":
                    continue
                if field_name not in valid_fields:
                    raise KeyError(f"Unknown field {field_name!r} for {name}")
                field_start = _date(raw_date, f"{name}.{field_name}")
                if field_start < start:
                    raise ValueError(f"{name}.{field_name} starts {field_start.date()} before " f"the table default {start.date()}")
                fields[field_name] = field_start
            tables[name] = _TableAvailability(start=start, fields=fields)

        if not tables:
            raise ValueError("data_availability.institutionals declares no source tables")
        return cls(tables=tables, derived_features=derived)

    def source_date(self, table: Table, field: str | None = None) -> pd.Timestamp:
        """Resolve a field override or inherit its table start."""
        rule = self.tables.get(table.name)
        if rule is None:
            raise KeyError(f"No institutionals availability declaration for {table.name}")
        if field is not None and field not in _known_fields(table):
            raise KeyError(f"Unknown field {field!r} for {table.name}")
        return rule.fields.get(field, rule.start) if field else rule.start

    def derived_date(
        self,
        feature: str,
        dependencies: Iterable[tuple[Table, str | None]] = (),
    ) -> pd.Timestamp:
        """Resolve an explicit derived override or the latest dependency boundary."""
        if feature not in _DERIVED_FEATURES:
            raise KeyError(f"Unknown institutionals derived feature {feature!r}")
        override = self.derived_features.get(feature)
        starts = [self.source_date(table, field) for table, field in dependencies]
        if override is not None:
            starts.append(override)
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
        row = idx >= pd.Timestamp(start).normalize()
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
        row = idx <= pd.Timestamp(end).normalize()
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

    def derived_mask(
        self,
        feature: str,
        index: pd.DatetimeIndex,
        columns: pd.Index,
        *,
        dependencies: Iterable[tuple[Table, str | None]] = (),
        requirements: Sequence[pd.DataFrame] = (),
    ) -> pd.DataFrame:
        start = self.derived_date(feature, dependencies)
        mask = self.date_mask(index, columns, start)
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

    @staticmethod
    def apply(values: pd.DataFrame, available: pd.DataFrame) -> pd.DataFrame:
        """Mask unavailable cells while preserving zeros inside supported cells."""
        if not values.index.equals(available.index) or not values.columns.equals(available.columns):
            raise ValueError("Signal values and availability mask must be aligned")
        return values.where(available)


def availability_date(
    periods: pd.Series | pd.DatetimeIndex,
    trading_index: pd.DatetimeIndex,
    *,
    settle_trading_days: int = F13_SETTLE_TRADING_DAYS,
) -> pd.Series:
    """The date each 13F `period` becomes public: deadline, snapped forward, plus the settle.

    Returns a `datetime64` Series on `periods`' own index (a `DatetimeIndex` input gets itself
    as the index, so the result is a period -> date lookup).

    ⚠ `NaT` PAST THE END OF THE GRID, NEVER THE LAST SESSION. A period whose availability date
    the calendar has not reached yet is NOT available, and clamping it onto the last trading
    day would publish the newest quarter days-to-weeks early -- the exact look-ahead the snap
    direction is chosen to avoid. `fundamentals_to_daily` reindexes such a stamp away anyway,
    so the honest value costs nothing and the clamped one is a trap.

    ⚠ AND THE FIRST SESSION, WITH NO SETTLE, BEFORE THE START OF THE GRID -- which is the
    opposite case and takes the opposite treatment. `sec13f_hr` carries periods back to
    1987-03-31 while the trading grid starts in 1995, so those deadlines fall off the front.
    They are not "unavailable": they were public years before the panel begins, and the first
    session is the earliest date the panel can carry them. Adding the settle there would say
    the 1987 filing season was still settling on the panel's fourth day, which is how the
    L1/L9 floor came out four sessions later than the panel's own start.

    `settle_trading_days=0` gives the snapped deadline with no buffer. That is the setting the
    ISOLATION TEST uses, not a production option.
    """
    if settle_trading_days < 0:
        raise ValueError(f"settle_trading_days must be >= 0, got {settle_trading_days!r}")
    # A `DatetimeIndex` input gets ITSELF as the index, which is what makes the result a
    # `period -> availability date` lookup a caller can `.map()` with.
    values = to_day(periods if isinstance(periods, pd.Series) else pd.Series(pd.DatetimeIndex(periods), index=pd.DatetimeIndex(periods)))

    grid = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if grid.empty:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    deadline = values + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    snap = grid.searchsorted(deadline.to_numpy(), side="left")
    # The settle applies only where the deadline is IN the grid's span. Before the start it
    # would push a long-settled 1987 season into the panel's first week (see the docstring).
    pos = np.where(deadline.to_numpy() < grid[0].to_datetime64(), snap, snap + settle_trading_days)
    return pd.Series(
        np.where(
            (pos < len(grid)) & deadline.notna().to_numpy(),
            grid.to_numpy()[np.clip(pos, 0, len(grid) - 1)],
            np.datetime64("NaT"),
        ),
        index=values.index,
        dtype="datetime64[ns]",
    )
