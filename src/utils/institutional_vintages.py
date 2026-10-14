"""Shared 13F publication clock and immutable-book resolver for aggregation and validation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS, SEC_13F_FILING_LAG_DAYS, SEC_13F_INFERRED_HOLDINGS_REPORT, SEC_13F_SOURCE_RATIO_BOUNDS


def literal_13f_cusip(value: Any) -> str | None:
    """Keep a source identifier literal; missing check digits are not leading zeros."""
    if pd.isna(value):
        return None
    token = str(value).strip().upper()
    if re.fullmatch(r"[A-Z0-9*@#]{6}-[A-Z0-9*@#]{2}-[0-9]", token):
        token = token.replace("-", "")
    return token or None


def full_13f_report(value: Any) -> bool:
    """Checked and explicitly inferred HR reporting scopes both support normal exits."""
    return str(value) in {"13F HOLDINGS REPORT", SEC_13F_INFERRED_HOLDINGS_REPORT}


def reconcile_13f_source(
    source_rows: int, declared_rows: int | None, source_value: float, declared_value: float | None, *, source_accounted: bool = False
) -> float | None:
    """Validate source totals after any finite power-of-ten cover-to-row alignment. The row count and the
    empty-book rules raise; a non-empty book whose cover VALUE total does not align returns None, because
    that total is a reference only and the caller warns."""
    lower, upper = SEC_13F_SOURCE_RATIO_BOUNDS
    declared_rows = None if pd.isna(declared_rows) else int(declared_rows)
    declared_value = None if pd.isna(declared_value) else float(declared_value)
    if (
        source_rows < 0
        or (declared_rows is None and not source_accounted)
        or (
            declared_rows is not None
            and (declared_rows < 0 or (source_rows != 0 if declared_rows == 0 else not lower <= source_rows / declared_rows <= upper))
        )
    ):
        raise ValueError(
            f"13F source row count does not reconcile: received={source_rows}, declared={declared_rows}; accepted ratio={lower:.0%}-{upper:.0%}"
        )
    if (
        not math.isfinite(source_value)
        or source_value < 0
        or (declared_value is None and not source_accounted)
        or (declared_value is not None and (not math.isfinite(declared_value) or declared_value < 0))
    ):
        raise ValueError(f"13F source VALUE total does not reconcile: rows={source_value:g}, declared={declared_value!r}")
    if declared_value is None:
        return 1.0
    if source_rows == 0:
        # A cover declaring 0 entries with a positive total is an all-exits book; the total stays as filed.
        if source_value != 0 or (declared_value != 0 and (declared_rows != 0 or source_accounted)):
            raise ValueError("13F empty source has a positive declared VALUE total")
        return 1.0
    if math.isclose(source_value, declared_value, rel_tol=0.005, abs_tol=1.0):
        return 1.0
    if source_value > 0 and declared_value > 0:
        exponent = round(math.log10(source_value) - math.log10(declared_value))
        try:
            factor = 10.0**exponent
        except OverflowError:
            return None
        if factor > 0 and math.isfinite(factor):
            aligned = declared_value * factor
            if aligned > 0 and math.isfinite(aligned):
                ratio = source_value / aligned
                if (
                    math.isclose(source_value, aligned, rel_tol=0.005, abs_tol=1.0)
                    or lower <= ratio <= upper
                    or math.isclose(ratio, lower, rel_tol=0.005)
                    or math.isclose(ratio, upper, rel_tol=0.005)
                ):
                    return factor
    return None


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
    values = pd.to_datetime(
        periods if isinstance(periods, pd.Series) else pd.Series(pd.DatetimeIndex(periods), index=pd.DatetimeIndex(periods)), errors="coerce"
    ).dt.normalize()

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


def filing_public_dates(
    manifests: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
) -> pd.Series:
    """Daily features conservatively use the session after filing, never extraction time."""
    grid = pd.DatetimeIndex(trading_index).normalize().drop_duplicates().sort_values()
    filed = pd.to_datetime(manifests["filing_date"]).dt.normalize()
    if "accepted_at" in manifests:
        accepted = pd.to_datetime(manifests["accepted_at"], utc=True)
        local_day = accepted.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        filed = pd.concat([filed, local_day], axis=1).max(axis=1)
    locations = grid.searchsorted(pd.DatetimeIndex(filed), side="right")
    next_session = pd.Series(pd.NaT, index=manifests.index, dtype="datetime64[ns]")
    eligible = locations < len(grid)
    next_session.loc[eligible] = grid[locations[eligible]]
    deadline = availability_date(pd.DatetimeIndex(pd.to_datetime(manifests["period"])), grid, settle_trading_days=settle_days)
    first = pd.Series(deadline.to_numpy(), index=manifests.index)
    # No later amendment is capped by the old 105-day filing band.
    return pd.concat([first, next_session], axis=1).max(axis=1).where(first.notna() & next_session.notna())


def public_books(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    as_of: object,
    trading_index: pd.DatetimeIndex,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve complete public books by (CIK, report quarter), without mutating any vintage."""
    if manifests["accession_number"].duplicated().any():
        raise ValueError("13F manifest accessions must be unique")
    if holdings.duplicated(["cik", "accession_number", "cusip"]).any():
        raise ValueError("13F holding vintages must be unique")
    m = manifests.copy()
    m["period"] = pd.to_datetime(m["period"])
    m["public_at"] = filing_public_dates(m, trading_index, settle_days)
    m = m.loc[m["public_at"].le(pd.Timestamp(as_of)) & m["parse_status"].eq("complete")]
    m["_accepted_order"] = pd.to_datetime(m["accepted_at"], utc=True) if "accepted_at" in m else pd.NaT
    complete_clocks = m.groupby(["cik", "period", "filing_date"])["_accepted_order"].transform(lambda clocks: clocks.notna().all())
    m["_accepted_order"] = m["_accepted_order"].where(complete_clocks)
    m["_amended"] = m.amendment_type.ne("original")
    m = m.sort_values(
        ["public_at", "filing_date", "_accepted_order", "_amended", "amendment_number", "accession_number"], kind="stable", na_position="first"
    )
    frames: list[pd.DataFrame] = []
    states: list[dict[str, Any]] = []
    holdings_by_accession = {accession: rows for accession, rows in holdings.groupby("accession_number", sort=False)}
    amounts = [
        c
        for c in holdings
        if c in {"shares", "value_usd", "call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value"}
        or c.startswith("raw_")
    ]
    for (cik, period), versions in m.groupby(["cik", "period"], sort=True):
        book = holdings.iloc[:0].copy()
        known = False
        restricted = False
        uncertain_absence = False
        full_report = False
        level_public_at = pd.NaT
        resolution_status = "missing_original"
        for version in versions.itertuples():
            incoming = holdings_by_accession.get(version.accession_number, holdings.iloc[:0]).copy()
            if len(incoming) != int(version.parsed_row_count):
                raise ValueError(f"Incomplete stored 13F book: {version.accession_number}")
            if version.amendment_type in {"original", "restatement"}:
                book, known, restricted = incoming, True, bool(version.restricted)
                uncertain_absence = False
                full_report = full_13f_report(version.report_type)
                level_public_at = version.public_at
                resolution_status = "resolved"
            elif version.amendment_type == "additions":
                if known:
                    old = book.set_index("cusip")
                    new = incoming.set_index("cusip")
                    quantities = [column for column in amounts if column in {"shares", "call_shares", "put_shares", "debt_prn"}]
                    raw_values = [column for column in amounts if column.startswith("raw_")]
                    dollar_values = [column for column in amounts if column not in quantities and not column.startswith("raw_")]
                    common = new.reindex(old.index)
                    same_quantities = common[quantities].eq(old[quantities]).fillna(False).all(axis=1)
                    same_raw = common[raw_values].eq(old[raw_values]).fillna(False).all(axis=1)
                    same_dollars = common[dollar_values].eq(old[dollar_values]).fillna(False).all(axis=1)
                    repeats = same_quantities & (same_raw | same_dollars) & old[quantities + raw_values].fillna(0).ne(0).any(axis=1)
                    # ponytail: copied-majority inference recovers resubmitted books; entry-level identity is needed for finer patches.
                    if int(repeats.sum()) > 1 and 2 * int(repeats.sum()) > max(len(old), len(new)):
                        # A reported CUSIP may omit a previously held typed leg; structural zeros do not prove exits.
                        previous = old.reindex(new.index)
                        for quantity, value in (
                            ("shares", "value_usd"),
                            ("call_shares", "call_value"),
                            ("put_shares", "put_value"),
                            ("debt_prn", "debt_value"),
                        ):
                            unsupported_zero = (
                                new.index.isin(old.index) & (previous[quantity].gt(0) | previous[quantity].isna()) & new[quantity].eq(0)
                            )
                            new.loc[unsupported_zero, [quantity, value, "raw_" + value]] = np.nan
                        incoming = new.reset_index().reindex(columns=holdings.columns)
                        omitted = book.loc[~book.cusip.isin(incoming.cusip)].copy()
                        uncertain_absence = not omitted.empty
                        omitted[amounts] = np.nan
                        book = pd.concat([omitted, incoming], ignore_index=True).reindex(columns=holdings.columns)
                        full_report = full_report and full_13f_report(version.report_type)
                        restricted = restricted or bool(version.restricted)
                        level_public_at = version.public_at
                        resolution_status = "inferred_position_patch"
                        continue
                    # NEW HOLDINGS reports extra entries, including extra shares of an existing CUSIP.
                    joined = pd.concat([book, incoming], ignore_index=True)
                    grouped_amounts = joined.groupby("cusip")[amounts]
                    numeric = grouped_amounts.sum(min_count=1).where(grouped_amounts.count().eq(joined.groupby("cusip").size(), axis=0))
                    metadata = joined.drop(columns=amounts).drop_duplicates("cusip", keep="last").set_index("cusip")
                    common_present = joined["position_type"].eq("common").groupby(joined["cusip"]).any()
                    metadata.loc[common_present, "position_type"] = "common"
                    metadata["class_evidence"] = joined.groupby("cusip")["class_evidence"].agg(
                        lambda values: json.dumps(sorted({name for value in values for name in json.loads(value)}))
                    )
                    book = metadata.join(numeric).reset_index().reindex(columns=holdings.columns)
                    # A new-holdings supplement alone cannot prove confidential positions are exhausted.
                    full_report = full_report and full_13f_report(version.report_type)
                    restricted = restricted or bool(version.restricted)
                    level_public_at = version.public_at
                    if resolution_status != "inferred_position_patch":
                        resolution_status = "resolved"
            else:
                raise ValueError(f"Unknown 13F amendment type: {version.amendment_type}")
        latest = versions.iloc[-1]
        states.append(
            {
                "cik": cik,
                "period": period,
                "public_at": level_public_at,
                "last_observed_at": latest.public_at,
                "accession_number": latest.accession_number,
                "book_known": known,
                "zero_eligible": known and not restricted and not uncertain_absence and full_report,
                "comparison_eligible": known and not restricted and full_report,
                "restricted": restricted,
                "resolution_status": resolution_status,
            }
        )
        if known and not book.empty:
            frames.append(book.assign(period=period, cik=cik))
    resolved = pd.concat(frames, ignore_index=True) if frames else holdings.iloc[:0].copy()
    states_frame = pd.DataFrame(
        states,
        columns=[
            "cik",
            "period",
            "public_at",
            "last_observed_at",
            "accession_number",
            "book_known",
            "zero_eligible",
            "comparison_eligible",
            "restricted",
            "resolution_status",
        ],
    )
    return resolved.sort_values(["cik", "period", "cusip"], kind="stable").reset_index(drop=True), states_frame


def iter_public_books(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
    history_quarters: int | None = 1,
    expiry_quarters: int | None = None,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Resolve only changed manager/quarter histories; retain the active dependency quarters."""
    df_manifest = manifests.copy()
    df_manifest["period"] = pd.to_datetime(df_manifest["period"])
    df_manifest["_day"] = filing_public_dates(df_manifest, trading_index, settle_days)
    df_manifest = df_manifest.loc[df_manifest["_day"].notna()]
    groups = df_manifest.groupby(["cik", "period"], sort=False).indices
    positions = holdings.groupby("accession_number", sort=False).indices
    cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]] = {}
    active = pd.NaT
    publications = df_manifest.groupby("_day", sort=True).indices
    days = pd.DatetimeIndex(sorted(publications))
    if expiry_quarters is not None:
        expiry = days + pd.DateOffset(months=3 * expiry_quarters)
        grid = pd.DatetimeIndex(trading_index).normalize().drop_duplicates().sort_values()
        locations = grid.searchsorted(expiry, side="right")
        days = days.union(grid[locations[locations < len(grid)]]).unique().sort_values()
    for day in days:
        events = df_manifest.iloc[publications.get(day, np.array([], dtype=int))]
        floor = (
            (active.to_period("Q") - history_quarters).end_time.normalize() if pd.notna(active) and history_quarters is not None else pd.Timestamp.min
        )
        for key in events[["cik", "period"]].drop_duplicates().itertuples(index=False, name=None):
            if key[1] < floor:
                continue
            versions = df_manifest.iloc[groups[key]]
            indices = np.concatenate([positions.get(a, np.array([], dtype=int)) for a in versions.accession_number])
            cache[key] = public_books(versions.drop(columns="_day"), holdings.iloc[indices], day, trading_index, settle_days)
        usable = [key[1] for key, (_, state) in cache.items() if state.book_known.any()]
        if usable:
            active = max(usable)
            floor = (active.to_period("Q") - history_quarters).end_time.normalize() if history_quarters is not None else pd.Timestamp.min
        cache = {key: value for key, value in cache.items() if key[1] >= floor}
        if cache:
            books = pd.concat([value[0] for value in cache.values()], ignore_index=True)
            states = pd.concat([value[1] for value in cache.values()], ignore_index=True)
            yield pd.Timestamp(day), books, states
