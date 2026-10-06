"""Shared plumbing of the security-grain market tapes (FTD lines, RegSHO short-volume rows).

Both tapes store raw lines stamped from `security_master` (`STAMP_COLUMNS`) and rebuild a ticker-grain table by
summing the canonical and secondary-class lines (`SUMMED_ROLES`). These helpers load stored rows in keyed chunks,
compare stamps, and write only the ticker rows that changed.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from itertools import batched

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS
from src.context import Context
from src.data_store.schema import Table
from src.utils.filer_tables import filing_window
from src.utils.string import normalise_ticker
from src.utils.universe import load_universe_tickers

#: The four columns stamped from the security master on every stored tape line.
STAMP_COLUMNS = ["security_id", "ticker", "lineage_role", "security_class"]
#: Roles whose lines are summed into the ticker grain.
SUMMED_ROLES = (*CANONICAL_ROLES, SECONDARY_CLASS)
#: Tickers per scoped load, and keys per targeted load or delete.
TICKER_CHUNK = 50
KEY_CHUNK = 500


def outside_scope(context: Context, scope: Collection[str]) -> frozenset[str]:
    """The universe companies outside `scope`: a scoped run neither deletes nor re-stamps their rows (empty when unscoped)."""
    return frozenset(normalise_ticker(t) for t in load_universe_tickers(context)) - frozenset(scope)


def nullable(values: pd.Series) -> pd.Series:
    """`values` as objects with missing values as None (the store's NULL)."""
    return values.astype(object).where(values.notna(), None)


def load_chunked(context: Context, table: Table, columns: Sequence[str], column: str, values: Sequence[str], size: int) -> list[pd.DataFrame]:
    """The non-empty loads of `table` where `column` is in each chunk of `values`."""
    frames = [
        context.store.load(table, columns=list(columns), where={column: list(chunk)}, optional=True) for chunk in batched(values, size, strict=False)
    ]
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
            for chunk in batched(sorted(group["date"]), KEY_CHUNK, strict=False):
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
