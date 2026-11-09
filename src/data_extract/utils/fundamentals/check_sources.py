"""
check_sources.py  (src/data_extract/utils/fundamentals/check_sources.py)
------------------------------------------------------------------------
The read-only SEC vs Sharadar fundamentals check: a short log that says where to look, not a diagnosis. It reads the two
stored histories, Sharadar's `secfilings` links and the identity lineage, and logs the tickers held by one history only,
the Sharadar CIKs outside their company, the exact `(ticker, as_of)` match share, and the shared fields where enough
matched rows differ. It writes nothing and never raises on a finding.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.fundamentals.kpi_catalogue import HISTORY_STATEMENT_ORDER
from src.data_store.schema import Tables
from src.utils.cutover_continuity import secfilings_cik

logger = logging.getLogger(__name__)

_KEYS = ["ticker", "as_of"]

#: The basis forks designed into the Sharadar field map: each SEC/Sharadar gap on these is expected and explained in
#: `sharadar_field_map.json`. They are still shown, tagged, so they do not drown a gap that is not on this list.
SHARADAR_GAP_EXPECTED_FIELDS: frozenset[str] = frozenset(
    {"stockholdersEquity", "ppeNet", "shortTermDebt", "longTermDebt", "accountsReceivable", "accountsPayable", "cash", "ebitda"}
)


@dataclass(frozen=True)
class SourceCheck:
    """What the check found. `field_gaps`: one row per listed field (`field`, `rows`, `share`, `max_gap`, `top`,
    `expected`), worst first; `top` is `[(ticker, its own share of rows over the gap)]`."""

    in_both: list[str]
    only_sec: list[str]
    only_sharadar: list[str]
    cik_mismatches: list[tuple[str, str, str]]
    no_cik: list[str]
    sec_rows: int
    sharadar_rows: int
    matched_rows: int
    fields_compared: list[str]
    sec_only: list[str]
    field_gaps: pd.DataFrame
    gap_pct: float
    field_share: float


def check_fundamentals_sources(context: Context, tickers: Sequence[str], *, gap_pct: float, field_share: float, top_tickers: int) -> SourceCheck:
    """Read the two histories, the Sharadar links and the identity once each, check them and log the result with its wall time."""
    # the two histories on their shared statement fields, the vendor links, the identity
    started = time.perf_counter()
    scope = {"ticker": list(tickers)}
    shared = [f for f in HISTORY_STATEMENT_ORDER if f in Tables.sharadar_fundamentals_history.read_columns]
    sec = context.store.load(Tables.sec_fundamentals_history, columns=[*_KEYS, *shared], where=scope, optional=True)
    sharadar = context.store.load(Tables.sharadar_fundamentals_history, columns=[*_KEYS, *shared], where=scope, optional=True)
    vendor = context.store.load(Tables.sharadar_tickers, columns=["ticker", "secfilings"], where=scope, optional=True)
    identity = load_identity(context)

    # an absent table is an empty one: the check reports it instead of stopping
    empty = pd.DataFrame(columns=_KEYS)
    check = check_sources(
        empty if sec is None else sec,
        empty if sharadar is None else sharadar,
        pd.DataFrame(columns=["ticker", "secfilings"]) if vendor is None else vendor,
        identity,
        tickers,
        gap_pct=gap_pct,
        field_share=field_share,
        top_tickers=top_tickers,
    )
    logger.info(
        "fundamentals check: %d SEC-only statement field(s) not value-checked %s",
        len(HISTORY_STATEMENT_ORDER) - len(shared),
        [f for f in HISTORY_STATEMENT_ORDER if f not in shared],
    )
    log_source_check(check, seconds=time.perf_counter() - started)
    return check


def check_sources(
    sec: pd.DataFrame,
    sharadar: pd.DataFrame,
    vendor: pd.DataFrame,
    identity: Identity,
    universe: Sequence[str],
    *,
    gap_pct: float,
    field_share: float,
    top_tickers: int,
) -> SourceCheck:
    """The pure check over frames: universe coverage, CIK alignment, and the shared float fields where at least
    `field_share` of the exactly matched rows (both values present) differ by more than `gap_pct` of |Sharadar|."""
    # universe coverage
    members = set(universe)
    sec_tickers = set(sec["ticker"]) & members
    sharadar_tickers = set(sharadar["ticker"]) & members

    # exact-key matches on the fields both frames carry
    fields = [c for c in sharadar.columns if c in sec.columns and c not in _KEYS and pd.api.types.is_numeric_dtype(sharadar[c])]
    sec_only = [c for c in sec.columns if c not in sharadar.columns and c not in _KEYS and pd.api.types.is_numeric_dtype(sec[c])]
    sec_in, sharadar_in = sec[sec["ticker"].isin(members)], sharadar[sharadar["ticker"].isin(members)]
    matched = sharadar_in[[*_KEYS, *fields]].merge(sec_in[[*_KEYS, *fields]], on=_KEYS, suffixes=("", "_sec"))
    mismatches, no_cik = _cik_findings(vendor, identity, sorted(members))

    return SourceCheck(
        in_both=sorted(sec_tickers & sharadar_tickers),
        only_sec=sorted(sec_tickers - sharadar_tickers),
        only_sharadar=sorted(sharadar_tickers - sec_tickers),
        cik_mismatches=mismatches,
        no_cik=no_cik,
        sec_rows=len(sec_in),
        sharadar_rows=len(sharadar_in),
        matched_rows=len(matched),
        fields_compared=fields,
        sec_only=sec_only,
        field_gaps=_field_gaps(matched, fields, gap_pct, field_share, top_tickers),
        gap_pct=gap_pct,
        field_share=field_share,
    )


def log_source_check(check: SourceCheck, seconds: float) -> None:
    """At most about 60 lines: counts at INFO, findings at WARNING, one line per listed field."""
    logger.info(
        "fundamentals check: tickers SEC %d, Sharadar %d, both %d",
        len(check.in_both) + len(check.only_sec),
        len(check.in_both) + len(check.only_sharadar),
        len(check.in_both),
    )
    if check.only_sec:
        logger.warning("fundamentals check: %d ticker(s) only in sec_fundamentals_history: %s", len(check.only_sec), check.only_sec)
    if check.only_sharadar:
        logger.warning("fundamentals check: %d ticker(s) only in sharadar_fundamentals_history: %s", len(check.only_sharadar), check.only_sharadar)
    if check.cik_mismatches:
        pairs = ", ".join(f"{t} sharadar={s} identity={i}" for t, s, i in check.cik_mismatches)
        logger.warning("fundamentals check: %d Sharadar secfilings CIK(s) outside the identity lineage: %s", len(check.cik_mismatches), pairs)
    if check.no_cik:
        logger.warning("fundamentals check: %d ticker(s) with no parseable Sharadar CIK: %s", len(check.no_cik), check.no_cik)

    share = check.matched_rows / check.sharadar_rows if check.sharadar_rows else 0.0
    logger.info(
        "fundamentals check: exact (ticker, as_of) matches %d = %.1f%% of %d Sharadar rows (%d SEC rows); %d shared field(s) compared",
        check.matched_rows,
        100 * share,
        check.sharadar_rows,
        check.sec_rows,
        len(check.fields_compared),
    )
    logger.info(
        "fundamentals check: %d field(s) with >= %.0f%% of matched rows over a %.0f%% gap, worst first",
        len(check.field_gaps),
        100 * check.field_share,
        100 * check.gap_pct,
    )
    for row in check.field_gaps.itertuples(index=False):
        top = ", ".join(f"{t} {s:.0%}" for t, s in row.top)
        tag = " [expected fork]" if row.expected else ""
        logger.warning("  %s: %.1f%% of %d rows, max gap %.1f%%, top %s%s", row.field, 100 * row.share, row.rows, 100 * row.max_gap, top, tag)
    logger.info("fundamentals check: done in %.1f s (read-only)", seconds)


def _cik_findings(vendor: pd.DataFrame, identity: Identity, universe: list[str]) -> tuple[list[tuple[str, str, str]], list[str]]:
    """`[(ticker, Sharadar CIK, identity roster CIK)]` for each vendor CIK its company does not own, and the universe
    tickers with no parseable Sharadar CIK (no vendor row, or a link without one)."""
    links = vendor[vendor["ticker"].isin(universe)]
    ciks = pd.DataFrame({"ticker": links["ticker"].astype(str), "cik": [secfilings_cik(url) for url in links["secfilings"]]})
    ciks = ciks[ciks["cik"] != ""].drop_duplicates()
    mismatches = [
        (t, c, identity.roster_cik.get(t, ""))
        for t, c in zip(ciks["ticker"], ciks["cik"], strict=True)
        if t not in identity.roster_cik or not identity.owns(t, c)
    ]
    return sorted(mismatches), sorted(set(universe) - set(ciks["ticker"]))


def _field_gaps(matched: pd.DataFrame, fields: list[str], gap_pct: float, field_share: float, top_tickers: int) -> pd.DataFrame:
    """One row per field whose share of compared rows over `gap_pct` is at least `field_share`, worst first."""
    rows = []
    for field in fields:
        vendor, sec = matched[field].to_numpy("float64"), matched[f"{field}_sec"].to_numpy("float64")
        both = ~(np.isnan(vendor) | np.isnan(sec))
        if not both.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.where(vendor[both] == sec[both], 0.0, np.abs(sec[both] - vendor[both]) / np.abs(vendor[both]))
        over = gap > gap_pct
        if over.mean() < field_share:
            continue
        by_ticker = pd.Series(over).groupby(matched["ticker"].to_numpy()[both]).mean()
        worst = by_ticker[by_ticker > 0].sort_index().sort_values(ascending=False, kind="mergesort").head(top_tickers)
        top = [(str(t), float(s)) for t, s in worst.items()]
        rows.append({"field": field, "rows": int(both.sum()), "share": float(over.mean()), "max_gap": float(gap.max()), "top": top})
    gaps = pd.DataFrame(rows, columns=["field", "rows", "share", "max_gap", "top"])
    gaps["expected"] = gaps["field"].isin(SHARADAR_GAP_EXPECTED_FIELDS)
    return gaps.sort_values("share", ascending=False, kind="mergesort").reset_index(drop=True)
