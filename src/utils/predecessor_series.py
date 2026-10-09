"""The predecessor vendor series of a ticker set: the cited `vendor_series_overrides` of `security_master_manual.json`
plus the `sharadar_tickers` rows whose `secfilings` CIK owns a closed register window in `entity_lineage`.

Shared by the Sharadar fetch and merge, the cube's level factor and the validate price panel.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils.cutover_continuity import PredecessorSeries, predecessor_series, register_windows, vendor_series_overrides
from src.utils.string import normalise_ticker

#: The security master's cited manual config, relative to the config directory.
SECURITY_MANUAL_FILE = Path("sec") / "security_master_manual.json"
_VENDOR_SERIES_KEY = "vendor_series_overrides"
_VENDOR_SERIES_COLUMNS = ("ticker", "vendor_ticker", "cik", "valid_from", "valid_to", "source")


def load_vendor_series(config_dir: str | Path) -> tuple[PredecessorSeries, ...]:
    """The cited `vendor_series_overrides`; `()` when the manual is absent. An entry without a source is refused."""
    path = Path(config_dir) / SECURITY_MANUAL_FILE
    if not path.exists():
        return ()
    rows = list(json.loads(path.read_text(encoding="utf-8")).get(_VENDOR_SERIES_KEY) or [])
    for row in rows:
        if not str(row.get("source") or "").strip():
            raise ValueError(f"{path}: {_VENDOR_SERIES_KEY} entry {row} has no source (URL or accession)")
    frame = pd.DataFrame([{column: row.get(column) for column in _VENDOR_SERIES_COLUMNS} for row in rows], columns=list(_VENDOR_SERIES_COLUMNS))
    return vendor_series_overrides(frame)


def load_predecessor_series(context: Context, tickers: list[str], config_dir: str | Path | None = None) -> tuple[PredecessorSeries, ...]:
    """The predecessor vendor series of `tickers`: the cited overrides (from `config_dir`, else the context's), plus the
    register-derived series. An override wins over a derived series of the same `(ticker, cik)`."""
    names = {normalise_ticker(t) for t in tickers}
    declared = tuple(s for s in load_vendor_series(config_dir or context.config_dir) if s.ticker in names)
    return with_overrides(_register_series(context, tickers), declared)


def with_overrides(derived: Iterable[PredecessorSeries], declared: tuple[PredecessorSeries, ...]) -> tuple[PredecessorSeries, ...]:
    """`derived` without the series a declared override replaces (same `(ticker, cik)`), then `declared`."""
    taken = {(s.ticker, s.cik) for s in declared}
    return tuple(s for s in derived if (s.ticker, s.cik) not in taken) + declared


def _register_series(context: Context, tickers: list[str]) -> tuple[PredecessorSeries, ...]:
    """The register-derived predecessor series; none before the dated lineage exists."""
    if "role" not in context.store.columns(Tables.entity_lineage):
        return ()
    lineage = context.store.load(
        Tables.entity_lineage,
        columns=["canonical_ticker", "cik", "role", "valid_from", "valid_to", "sources"],
        where={"role": "cik_window"},
        optional=True,
    )
    vendor = context.store.load(Tables.sharadar_tickers, columns=["ticker", "secfilings", "lastquarter"], optional=True)
    if lineage is None or vendor is None:
        return ()
    return predecessor_series(vendor, register_windows(lineage), tickers)
