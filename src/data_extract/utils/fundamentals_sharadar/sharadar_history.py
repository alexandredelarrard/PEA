"""Build `sharadar_fundamentals_history` from Sharadar's own inputs: SF1 ARQ rows, splits and the register windows.

It reads no SEC table. Inside a register predecessor window (register-derived or a cited vendor series override) the
canonical ARQ rows are replaced by the window owner's vendor series before the TTM build, on the ticker's share basis
through the cited merger exchange ratio. Every value column is cast to float64 before the write.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.constants.constants import (
    SHARADAR_ACTION_SPINOFF,
    SHARADAR_ACTION_SPLIT,
    SHARADAR_COLLAPSE_KEY,
    SHARADAR_COLLAPSE_ORDER,
)
from src.context import Context
from src.data_extract.utils.common.frame_sanitize import pin_dtypes
from src.data_extract.utils.common.security_lines import load_security_manual
from src.data_extract.utils.fundamentals.kpi_catalogue import DEFAULT_CONFIG_DIR
from src.data_extract.utils.fundamentals_sharadar.build_ttm import ARQ, build_ttm
from src.data_extract.utils.fundamentals_sharadar.field_map import (
    FieldMap,
    TranslationReport,
    load_field_map,
    translate,
)
from src.data_store.schema import Tables
from src.utils.cutover_continuity import PredecessorSeries, ShareExchange, apply_predecessor_series, rebase_split_events
from src.utils.predecessor_series import load_level_actions, load_predecessor_series
from src.utils.price.adjustments import ShareBasis, forward_split_factor, load_bugfix, split_events
from src.utils.price.basis import load_share_basis

log = logging.getLogger(__name__)

#: History key: `as_of` is Sharadar's `date` (FILING date), `fiscal_end` its `reportperiod` (period end).
HISTORY_KEYS: tuple[str, ...] = ("ticker", "as_of", "fiscal_end")

#: Vendor -> history names for the two keys that are not already repo-named.
_KEY_FROM_VENDOR: dict[str, str] = {"date": "as_of", "reportperiod": "fiscal_end"}


# --------------------------------------------------------------------------- #
# the column contract                                                          #
# --------------------------------------------------------------------------- #
def history_columns(field_map: FieldMap) -> tuple[str, ...]:
    """The column tuple (keys + field-map outputs).

    Raises unless it equals `Tables.sharadar_fundamentals_history.read_columns` exactly, in order.
    """
    columns = (*HISTORY_KEYS, *field_map.outputs)
    declared = tuple(Tables.sharadar_fundamentals_history.read_columns)
    if columns != declared:
        extra = [c for c in columns if c not in set(declared)]
        gone = [c for c in declared if c not in set(columns)]
        raise RuntimeError(
            f"the Sharadar history column contract disagrees with the registry: {len(columns)} built "
            f"vs {len(declared)} declared; only in the build {extra}; only in schema.py {gone}. "
            f"Fix BOTH -- they are the same decision written twice on purpose."
        )
    return columns


# --------------------------------------------------------------------------- #
# the steps                                                                    #
# --------------------------------------------------------------------------- #
def collapse_same_date(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per `(ticker, as_of)`, keeping the greatest `fiscal_end` (Sharadar ships no form column).

    Returns `(collapsed frame, dropped rows)` so the caller can log the drop.
    """
    ordered = frame.sort_values([*SHARADAR_COLLAPSE_KEY, SHARADAR_COLLAPSE_ORDER])
    duplicated = ordered.duplicated(subset=list(SHARADAR_COLLAPSE_KEY), keep="last")
    return ordered[~duplicated].reset_index(drop=True), ordered[duplicated].copy()


# --------------------------------------------------------------------------- #
# the build                                                                    #
# --------------------------------------------------------------------------- #
def build_frame(
    sharadar_arq: pd.DataFrame,
    actions: pd.DataFrame | None,
    field_map: FieldMap,
    *,
    yf_splits: pd.DataFrame | None = None,
    report: TranslationReport | None = None,
    splits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The whole transform, no I/O: translate -> TTM (+ split de-adjustment) -> same-date collapse -> contract -> cast.

    Split events (`actions` + `yf_splits`, or `splits` when given) de-adjust only the `split_basis` columns
    (`sharesOutstandingPit`); other share columns stay on the vendor's split-adjusted basis.
    """
    columns = history_columns(field_map)
    translated = translate(sharadar_arq, field_map, report=report)

    # split de-adjustment runs after the four-quarter aggregation, inside `build_ttm`
    ttm = build_ttm(translated, field_map, actions=actions, yf_splits=yf_splits, report=report, splits=splits)
    ttm = pin_dtypes(ttm.rename(columns=_KEY_FROM_VENDOR), dates=("as_of", "fiscal_end"))

    collapsed, dropped = collapse_same_date(ttm)
    if not dropped.empty:
        log.warning(
            "same-date collapse: %d row(s) dropped, greatest `fiscal_end` kept "
            "(Sharadar ships no form column, so the SEC 10-K-first rule has no analogue):\n%s",
            len(dropped),
            dropped[["ticker", "as_of", "fiscal_end"]].to_string(),
        )

    missing = [c for c in columns if c not in collapsed.columns]
    if missing:
        raise RuntimeError(f"the Sharadar history frame is missing {len(missing)} contract column(s): {missing}")
    out = collapsed[list(columns)].copy()
    log.info("Sharadar history frame: %d row(s) x %d column(s)", len(out), len(out.columns))
    return _cast(out, columns)


def _cast(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Pin every dtype before the write: an all-None object column would create a TEXT column."""
    out = pin_dtypes(frame, dates=("as_of", "fiscal_end"), floats=[column for column in columns if column not in HISTORY_KEYS])
    out["ticker"] = out["ticker"].astype(str)
    return out


def share_basis_factors(
    series: tuple[PredecessorSeries, ...], exchanges: tuple[ShareExchange, ...], splits: pd.DataFrame
) -> dict[tuple[str, str], tuple[ShareExchange, float] | None]:
    """Per `(ticker, cik)` window: its cited exchange and factor = ratio x the ticker's splits after the seam; None when uncited.

    Only genuine share events (`split_events`) enter the count; a price-only factor Yahoo books as a split (a spinoff)
    stays in the cube's level factor S(d), never in the shares."""
    by_window = {(x.ticker, x.predecessor_cik): x for x in exchanges}
    out: dict[tuple[str, str], tuple[ShareExchange, float] | None] = {}
    for s in series:
        exchange = by_window.get((s.ticker, s.cik))
        if exchange is None:
            out[(s.ticker, s.cik)] = None
            continue
        later = forward_split_factor(pd.Series([s.ticker]), pd.Series([exchange.seam_date]), splits).iloc[0]
        out[(s.ticker, s.cik)] = (exchange, exchange.ratio * float(later))
    return out


def with_predecessor_series(
    context: Context, vendor: pd.DataFrame, names: list[str], splits: pd.DataFrame, *, config_dir: str = DEFAULT_CONFIG_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(vendor, splits)` with each predecessor window's canonical ARQ rows replaced by the window owner's own on the
    ticker's share basis, and the split events rebased so the window's PIT count is the owner's; logged per quarter."""
    series = load_predecessor_series(context, names, config_dir)
    if not series:
        return vendor, splits
    owners = context.store.load(
        Tables.sharadar_fundamentals_facts, project=True, where={"ticker": sorted({s.vendor_ticker for s in series}), "dimension": ARQ}, optional=True
    )
    stored = set(owners["ticker"].astype(str)) if owners is not None else set()
    for s in series:
        if s.vendor_ticker not in stored:
            context.log.warning(
                "Sharadar history: the predecessor vendor series %s of %s (CIK %s) is not stored; rows before %s are kept as fetched",
                s.vendor_ticker,
                s.ticker,
                s.cik,
                s.valid_to.date() if s.valid_to is not None else "-",
            )
    if owners is None or owners.empty:
        return vendor, splits
    exchanges = share_basis_factors(series, load_security_manual(config_dir).exchanges, splits)
    merged, events = apply_predecessor_series(vendor, owners, series, factors={key: hit[1] if hit else None for key, hit in exchanges.items()})
    for (ticker, vendor_ticker, cik), group in events.groupby(["ticker", "vendor_ticker", "cik"], sort=True):
        quarters = {event: part["quarter"].tolist() for event, part in group.groupby("event", sort=True)}
        counts = [value for event in ("replaced", "filled", "dropped") for value in (len(quarters.get(event, [])), quarters.get(event, []))]
        context.log.info(
            "Sharadar history: %s takes %s rows inside the window of CIK %s -- replaced %d %s | filled %d %s | dropped %d %s | owner quarters filed twice or more %s",
            ticker,
            vendor_ticker,
            cik,
            *counts,
            group.loc[group["predecessor_rows"].gt(1), "quarter"].tolist() or "none",
        )
        hit = exchanges.get((str(ticker), str(cik)))
        if hit is None:
            context.log.warning(
                "Sharadar history: no cited exchange ratio for the window of CIK %s in %s (%s rows): its share counts and per-share figures are NULL",
                cik,
                ticker,
                vendor_ticker,
            )
        else:
            context.log.info(
                "Sharadar history: %s rows of CIK %s on %s's share basis -- exchange ratio %s on %s x later splits = factor %.6g",
                vendor_ticker,
                cik,
                ticker,
                hit[0].ratio,
                hit[0].seam_date.date(),
                hit[1],
            )
    applied = set(zip(events["ticker"], events["cik"], strict=True))
    converted = [(s, hit[0]) for s in series if (s.ticker, s.cik) in applied and (hit := exchanges.get((s.ticker, s.cik))) is not None]
    if not converted:
        return merged, splits
    owner_actions = context.store.load(
        Tables.sharadar_actions,
        project=True,
        optional=True,
        where={"ticker": sorted({s.vendor_ticker for s, _ in converted}), "action": [SHARADAR_ACTION_SPLIT]},
    )
    return merged, rebase_split_events(splits, split_events(owner_actions, None), converted)


def vendor_share_basis(
    context: Context,
    vendor: pd.DataFrame,
    names: list[str],
    yf_splits: pd.DataFrame | None,
    splits: pd.DataFrame,
    config_dir: str,
) -> ShareBasis:
    """Resolve on the selected owner actions and the same repaired anchors as prices."""
    bugfix = load_bugfix(config_dir)
    selected = load_level_actions(context, names, config_dir=config_dir)
    return load_share_basis(context, names, bugfix, selected, yf_splits, shares=vendor, base_events=splits)


def build_sharadar_history(context: Context, tickers: list[str], *, full: bool = False, config_dir: str = DEFAULT_CONFIG_DIR) -> None:
    """Build `sharadar_fundamentals_history` from `sharadar_fundamentals_facts` for `tickers`.

    Inputs are read-only. Default upserts; `full=True` first deletes these tickers' rows so vanished keys go too.
    """
    field_map = load_field_map(config_dir)
    names = sorted(set(tickers))

    vendor = context.store.load(Tables.sharadar_fundamentals_facts, project=True, where={"ticker": names, "dimension": ARQ}, optional=True)
    if vendor is None or vendor.empty:
        context.log.warning("Sharadar history: no ARQ rows for %d requested ticker(s) -- run `fundamentals-sharadar` first", len(names))
        return

    # `sharadar_actions` is market-wide: scope to these tickers' splits and spinoffs (spinoffs only name co-dated splits)
    actions = context.store.load(
        Tables.sharadar_actions, project=True, optional=True, where={"ticker": names, "action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF]}
    )
    # second split source; `split_events` unions it with `sharadar_actions` under a corroboration rule
    yf_splits = context.store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": names}, optional=True)
    report = TranslationReport()
    vendor, splits = with_predecessor_series(context, vendor, names, split_events(actions, yf_splits, report=report), config_dir=config_dir)
    basis = vendor_share_basis(context, vendor, names, yf_splits, splits, config_dir)
    splits = basis.vendor

    frame = build_frame(vendor, actions, field_map, yf_splits=yf_splits, report=report, splits=splits)
    if frame.empty:
        context.log.warning("Sharadar history: the transform produced 0 rows")
        return

    if full:
        # one `IN` delete rather than a round-trip per ticker
        deleted = context.store.delete(Tables.sharadar_fundamentals_history, {"ticker": names})
        context.log.warning("Sharadar history: --full deleted %d existing row(s) before the rebuild (scope: %d ticker(s))", deleted, len(names))

    written = context.store.save(Tables.sharadar_fundamentals_history, frame)
    context.log.info(
        "Sharadar history: %d row(s) over %d ticker(s), %s..%s\n%s",
        written,
        frame["ticker"].nunique(),
        frame["as_of"].min().date(),
        frame["as_of"].max().date(),
        report.summary(),
    )
