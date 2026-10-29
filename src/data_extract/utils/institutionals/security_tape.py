"""Shared plumbing of the security-grain market tapes (FTD lines, RegSHO short-volume rows).

Both tapes store raw lines stamped from `security_master` (`STAMP_COLUMNS`) and rebuild a ticker-grain table by
summing the canonical and secondary-class lines (`SUMMED_ROLES`). These helpers load stored rows in keyed chunks,
compare stamps, and write only the ticker rows that changed.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS
from src.context import Context
from src.data_extract.utils.common.resume import recently_changed
from src.data_store.schema import Table
from src.data_store.store import NOT_NULL, DataStore, Where
from src.utils.batching import batched_tuples
from src.utils.filer_tables import filing_window
from src.utils.universe import load_universe_tickers

#: The four columns stamped from the security master on every stored tape line.
STAMP_COLUMNS = ["security_id", "ticker", "lineage_role", "security_class"]
#: Roles whose lines are summed into the ticker grain.
SUMMED_ROLES = (*CANONICAL_ROLES, SECONDARY_CLASS)
#: Tickers per scoped load, and keys per targeted load or delete.
TICKER_CHUNK = 50
KEY_CHUNK = 500


def restamp_targets(
    context: Context, table: Table, stamps: Mapping[str, pd.Timestamp], as_of: pd.Timestamp | None, *, every: bool = False
) -> list[str]:
    """The companies whose stored rows of `table` need a re-stamp: those whose change stamp is recent (`every`: all of
    them), plus every company still stamped on a row but with no stamp left (its master rows all vanished)."""
    changed = set(stamps) if every else set(recently_changed(stamps, as_of))
    stored = {str(t) for t in context.store.distinct(table, "ticker") if t is not None and not pd.isna(t)} if context.store.exists(table) else set()
    return sorted(changed | (stored - set(stamps)))


def outside_scope(context: Context, scope: Collection[str], all_tickers: Collection[str] | None = None) -> frozenset[str]:
    """The universe companies outside `scope`: a scoped run neither deletes nor re-stamps their rows (empty when unscoped)."""
    return frozenset(load_universe_tickers(context) if all_tickers is None else all_tickers) - frozenset(scope)


def nullable(values: pd.Series) -> pd.Series:
    """`values` as objects with missing values as None (the store's NULL)."""
    return values.astype(object).where(values.notna(), None)


def load_chunked(context: Context, table: Table, columns: Sequence[str], column: str, values: Sequence[str], size: int) -> list[pd.DataFrame]:
    """The non-empty loads of `table` where `column` is in each chunk of `values`."""
    frames = [context.store.load(table, columns=list(columns), where={column: list(chunk)}, optional=True) for chunk in batched_tuples(values, size)]
    return [frame for frame in frames if frame is not None]


def stamp_changed(old: pd.DataFrame, new: pd.DataFrame, old_suffix: str = "") -> pd.Series:
    """True where any stamp column of `old` (named with `old_suffix`) differs from `new`, row by row; indexed like `new`."""
    differs = np.zeros(len(new), dtype=bool)
    for column in STAMP_COLUMNS:
        differs |= nullable(old[f"{column}{old_suffix}"]).astype(str).to_numpy() != nullable(new[column]).astype(str).to_numpy()
    return pd.Series(differs, index=new.index)


def summed_lines(stamped: pd.DataFrame, tickers: Collection[str] | None) -> pd.DataFrame:
    """The stamped lines summed into the ticker grain: a summed role and a ticker, limited to `tickers` when given."""
    summed = stamped[stamped["lineage_role"].isin(SUMMED_ROLES) & stamped["ticker"].notna()]
    return summed if tickers is None else summed[summed["ticker"].isin(set(tickers))]


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    left = pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")
    right = pd.to_numeric(b, errors="coerce").to_numpy(dtype="float64")
    return np.isclose(left, right, rtol=1e-12, atol=0.0) | (np.isnan(left) & np.isnan(right))


def apply_grain(
    context: Context,
    tickers: Sequence[str],
    grain: pd.DataFrame,
    *,
    dry_run: bool,
    table: Table,
    columns: Sequence[str],
    text_columns: Collection[str] = (),
) -> list[dict]:
    """Write the ticker rows of `tickers` that differ from `grain` (vanished keys deleted); one record per ticker that lost rows."""
    columns = list(columns)
    kept = load_chunked(context, table, columns, "ticker", tickers, TICKER_CHUNK)
    stored = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=columns)
    stored["date"] = pd.to_datetime(stored["date"]).dt.normalize()
    both = stored.merge(grain, on=["ticker", "date"], how="outer", suffixes=("_old", ""), indicator=True)
    same = both["_merge"].eq("both").to_numpy(copy=True)
    for column in (c for c in columns if c not in ("ticker", "date")):
        old, new = both[f"{column}_old"], both[column]
        same &= old.astype(str).eq(new.astype(str)).to_numpy() if column in text_columns else _same(old, new)
    gone = both[both["_merge"].eq("left_only")]
    write = both[both["_merge"].ne("left_only").to_numpy() & ~same][columns]
    records: list[dict] = []
    for ticker, group in gone.groupby("ticker", sort=True):
        first, last = filing_window(group["date"])
        records.append(
            {"table": table.name, "ticker": str(ticker), "cik": "", "first_filed": first, "last_filed": last, "keys": len(group), "rows": len(group)}
        )
        if not dry_run:
            for chunk in batched_tuples(sorted(group["date"]), KEY_CHUNK):
                context.store.delete(table, where={"ticker": str(ticker), "date": chunk})
    if not dry_run and not write.empty:
        context.store.save(table, write)
    return records


def warn_lost_rows(log: logging.Logger, label: str, records: Sequence[dict]) -> None:
    """One WARNING per ticker whose ticker rows vanished on a re-stamp."""
    for record in records:
        log.warning(
            "%s: %s lost %d ticker row(s) %s..%s on re-stamp", label, record["ticker"], record["rows"], record["first_filed"], record["last_filed"]
        )


def validate_tape_frames(frames: dict[Table, pd.DataFrame], scopes: dict[Table, dict[str, object] | None]) -> None:
    """Require matching explicit scopes and unique declared source keys before writing."""
    if set(frames) != set(scopes) or any(scope == {} for scope in scopes.values()):
        raise ValueError("An institutional source write needs one exact source pair and explicit scopes")
    for table, df_rows in frames.items():
        if not set(table.pk).issubset(df_rows) or df_rows.duplicated(list(table.pk)).any():
            raise ValueError(f"{table}: missing or repeated source keys")


def load_tape_cohorts(store: DataStore, queries: dict[Table, Where | None]) -> dict[Table, pd.DataFrame]:
    """Load each complete projected candidate cohort once on the guarded store."""
    cohorts = {}
    for table, where in queries.items():
        columns = store.columns(table)
        df_rows = store.load(table, columns=columns, where=where, markers=True, optional=True)
        cohorts[table] = df_rows if df_rows is not None else pd.DataFrame(columns=columns)
    return cohorts


def scope_tape_rows(table: Table, df_rows: pd.DataFrame, where: dict[str, object] | None) -> pd.DataFrame:
    """Select the already-loaded tape scope with SQL NULL and daily-key semantics."""
    selected = pd.Series(True, index=df_rows.index)
    for column, value in (where or {}).items():
        if column not in df_rows:
            return df_rows.iloc[:0]
        if value is None:
            selected &= df_rows[column].isna()
        elif value is NOT_NULL:
            selected &= df_rows[column].notna()
        else:
            multiple = isinstance(value, list | tuple | set | frozenset | pd.Series | pd.Index | np.ndarray)
            values = pd.Series(list(value) if multiple else [value]).dropna()
            actual = df_rows[column]
            if column in table.date_type_cols:
                actual = pd.to_datetime(actual, errors="coerce").dt.normalize()
                values = pd.to_datetime(values).dt.normalize()
            selected &= actual.isin(values)
    return df_rows.loc[selected]


def tape_key_matches(table: Table, df_incoming: pd.DataFrame, df_actual: pd.DataFrame) -> pd.DataFrame:
    """Match complete source keys, normalizing declared daily keys before the pure merge."""
    if df_incoming.empty or df_actual.empty:
        return pd.DataFrame()
    dates = set(table.pk) & set(table.date_type_cols)
    frames = [
        df_rows.assign(**{column: pd.to_datetime(df_rows[column], errors="coerce").dt.normalize() for column in dates})
        for df_rows in (df_incoming, df_actual)
    ]
    return frames[0].merge(frames[1], on=list(table.pk), suffixes=("", "_stored"))


def assert_tape_owners(df_both: pd.DataFrame, owners: object) -> None:
    """A scoped raw replacement cannot steal another retained issuer's source key."""
    if owners is None or df_both.empty or "ticker_stored" not in df_both:
        return
    values = owners if isinstance(owners, list | tuple | set | frozenset | pd.Series | pd.Index | np.ndarray) else [owners]
    if (df_both["ticker_stored"].notna() & ~df_both["ticker_stored"].isin(values)).any():
        raise ValueError("Scoped institutional source write would mutate another retained issuer")


def assert_tape_source(store: DataStore, actual: dict[Table, pd.DataFrame], expected: dict[Table, pd.DataFrame] | None) -> None:
    """Refuse restamping when the supplied source snapshot changed before commit."""
    for table, df_expected in (expected or {}).items():
        if table not in actual or not store.same_rows(table, actual[table], df_expected):
            raise ValueError("Institutional source cohort changed before atomic restamping")


def commit_tape_source(store: DataStore, frames: dict[Table, pd.DataFrame], scopes: Mapping[Table, Where | None]) -> dict[Table, int]:
    """Publish the prepared tape pair using the caller's active generic transaction."""
    counts = {}
    for table, df_rows in frames.items():
        if scopes[table] is None:
            counts[table] = store.replace(table, df_rows)
        else:
            store.delete(table, cast(Where, scopes[table]))
            counts[table] = store.save(table, df_rows, ignore_all_conflicts=True)
    return counts
