"""Vendor-series continuity at CIK cutovers: gap detection, classification, recorded exceptions and predecessor series.

Pure functions of frames, shared by the Sharadar continuity test, `validate identity` and the merged-history build.
Quarters are sequenced on the ARQ `calendardate` ordinals; a predecessor window is tested on the filer's own
`reportperiod`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.utils.identity_flags import MARGIN
from src.utils.quarters import quarter_label, quarter_ordinal
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series

MISSING_SEC_FILING = "missing_sec_filing"
INCORRECT_CIK_WINDOW = "incorrect_cik_window"
VENDOR_COVERAGE_GAP = "vendor_coverage_gap"
VENDOR_SERIES_OTHER_COMPANY = "vendor_series_other_company"

#: Half-width, in years, of the window measured around each cutover boundary.
CUTOVER_WINDOW_YEARS = 2

#: The periodic forms whose period of report is a fiscal quarter or year end.
PERIODIC_FORMS = ("10-Q", "10-Q/A", "10-K", "10-K/A", "10-KT", "10-KT/A")

EXCEPTIONS_FILE = Path("sec") / "vendor_coverage_exceptions.json"
FILING_COLUMNS = ("ticker", "cik", "accession_number", "form", "filing_date", "period_of_report")

#: Relative `assets` difference above which a vendor quarter is read as another company's.
OTHER_COMPANY_TOLERANCE = 0.05

#: Vendor share counts, multiplied by the share-basis factor inside a converted predecessor window.
SHARE_COUNT_COLUMNS = ("sharesbas", "shareswa", "shareswadil")
#: Vendor per-share figures, divided by it; totals (`marketcap`, `ev`) and unitless ratios are left alone.
PER_SHARE_COLUMNS = ("eps", "epsdil", "epsusd", "dps", "bvps", "tbvps", "fcfps", "sps", "price")

_SENTINEL = pd.Timestamp("1900-01-01")
_CIK_IN_URL = re.compile(r"CIK=(\d+)", re.IGNORECASE)
_EVENTS = ("replaced", "filled", "dropped")


@dataclass(frozen=True)
class CikWindow:
    """One CIK's declared tenure over a ticker, half-open; `None` is an open end."""

    cik: str
    valid_from: pd.Timestamp | None
    valid_to: pd.Timestamp | None


@dataclass(frozen=True)
class VendorException:
    """One vendor quarter missing at a cutover, explained by the SEC filing that reports it."""

    ticker: str
    quarter: str
    period_end: str
    cik: str
    accession: str
    filed: str
    label: str


@dataclass(frozen=True)
class PredecessorSeries:
    """The vendor ticker carrying a predecessor CIK's own series, and the window it owns for `ticker`."""

    ticker: str
    vendor_ticker: str
    cik: str
    valid_from: pd.Timestamp | None
    valid_to: pd.Timestamp | None


@dataclass(frozen=True)
class ShareExchange:
    """A cited merger exchange ratio: one share of `predecessor_cik` became `ratio` shares of `ticker` on `seam_date`."""

    ticker: str
    predecessor_cik: str
    seam_date: pd.Timestamp
    ratio: float


@dataclass(frozen=True)
class ContinuityReport:
    """Every discontinuity (`ticker, quarter, boundary, class, explained, evidence, cik, accession, label`), the
    recorded quarters present again, the boundaries with no vendor quarter near them, and duplicated records."""

    table: pd.DataFrame
    healed: tuple[VendorException, ...]
    unobserved: tuple[str, ...]
    duplicated: tuple[str, ...]


TABLE_COLUMNS = ("ticker", "quarter", "boundary", "class", "explained", "evidence", "cik", "accession", "label")


# --------------------------------------------------------------------------- exceptions


def load_vendor_exceptions(config_dir: str | Path) -> tuple[VendorException, ...]:
    """The accession-exact vendor exceptions of `configs/sec/vendor_coverage_exceptions.json`; `()` when absent."""
    path = Path(config_dir) / EXCEPTIONS_FILE
    if not path.exists():
        return ()
    rows = json.loads(path.read_text(encoding="utf-8")).get("exceptions") or []
    return tuple(
        VendorException(
            normalise_ticker(row["ticker"]),
            str(row["quarter"]),
            str(row["period_end"]),
            pad_cik(row["cik"]),
            str(row["accession"]),
            str(row["filed"]),
            str(row["label"]),
        )
        for row in rows
    )


# --------------------------------------------------------------------------- windows


def _bound(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    return None if stamp <= _SENTINEL else stamp


def register_windows(lineage: pd.DataFrame) -> dict[str, tuple[CikWindow, ...]]:
    """`{ticker: windows oldest first}` from the `cik_window` lineage rows the register declared."""
    if lineage is None or lineage.empty or not {"role", "sources", "canonical_ticker"} <= set(lineage.columns):
        return {}
    rows = lineage[lineage["role"].eq("cik_window") & lineage["canonical_ticker"].notna()]
    rows = rows[["register" in str(value).split(",") for value in rows["sources"].fillna("")]]
    out: dict[str, tuple[CikWindow, ...]] = {}
    for ticker, group in rows.groupby(rows["canonical_ticker"].map(normalise_ticker), sort=True):
        ordered = group.assign(_from=pd.to_datetime(group["valid_from"])).sort_values("_from", kind="mergesort")
        out[str(ticker)] = tuple(
            CikWindow(pad_cik(cik), _bound(start), _bound(end))
            for cik, start, end in zip(ordered["cik"], ordered["valid_from"], ordered["valid_to"], strict=True)
        )
    return out


def boundaries(windows: Sequence[CikWindow]) -> tuple[pd.Timestamp, ...]:
    """The seam dates of a chain, oldest first."""
    return tuple(sorted(w.valid_from for w in windows if w.valid_from is not None))


def admitted(windows: Sequence[CikWindow], cik: str, filed: pd.Timestamp, margin: pd.Timedelta = MARGIN) -> bool:
    """Whether the seam-widened window of `cik` admits a filing made on `filed`."""
    return any(
        w.cik == cik and (w.valid_from is None or filed >= w.valid_from - margin) and (w.valid_to is None or filed < w.valid_to + margin)
        for w in windows
    )


# --------------------------------------------------------------------------- quarters and gaps


def quarter_of(day: object) -> str:
    """The `2025Q1` label of one date; empty when the date is missing or unparseable."""
    ordinal = quarter_ordinal(pd.Series([day])).iloc[0]
    return "" if pd.isna(ordinal) else quarter_label(int(ordinal))


def missing_quarters(arq: pd.DataFrame) -> list[str]:
    """Quarters missing inside the frame's own observed span of ARQ `calendardate` ordinals."""
    observed = sorted({int(q) for q in quarter_ordinal(arq["calendardate"]).dropna()})
    if not observed:
        return []
    seen = set(observed)
    return [quarter_label(q) for q in range(observed[0], observed[-1] + 1) if q not in seen]


def discontinuities(
    arq: pd.DataFrame, seams: Iterable[pd.Timestamp], window_years: int = CUTOVER_WINDOW_YEARS
) -> tuple[dict[str, pd.Timestamp], list[pd.Timestamp]]:
    """Missing vendor quarters within `window_years` of every seam, each with its seam, plus the seams with no
    vendor quarter near them."""
    dates = pd.to_datetime(arq["calendardate"], errors="coerce")
    offset = pd.DateOffset(years=window_years)
    missing: dict[str, pd.Timestamp] = {}
    unobserved: list[pd.Timestamp] = []
    for seam in seams:
        near = arq[dates.between(seam - offset, seam + offset)]
        if near.empty:
            unobserved.append(seam)
            continue
        for label in missing_quarters(near):
            missing.setdefault(label, seam)
    return dict(sorted(missing.items())), unobserved


def classify(
    windows: Sequence[CikWindow], quarter: str, filings: pd.DataFrame, record: VendorException | None, margin: pd.Timedelta = MARGIN
) -> tuple[str, bool, str]:
    """`(class, explained, evidence)` for one missing vendor quarter.

    The SEC filings stored for the quarter decide the class when there are any; otherwise the exception record
    does. Only a recorded `vendor_coverage_gap` is explained.
    """
    if not filings.empty:
        filed = pd.to_datetime(filings["filing_date"])
        ok_by_row = [admitted(windows, cik, day, margin) for cik, day in zip(filings["cik"], filed, strict=True)]
        cls = VENDOR_COVERAGE_GAP if any(ok_by_row) else INCORRECT_CIK_WINDOW
        evidence = "stored " + "; ".join(
            f"{cik} {accession} {form} filed {day.date()}" + ("" if ok else " OUTSIDE window")
            for cik, accession, form, day, ok in zip(filings["cik"], filings["accession_number"], filings["form"], filed, ok_by_row, strict=True)
        )
    elif record is not None:
        cls = VENDOR_COVERAGE_GAP if admitted(windows, record.cik, pd.Timestamp(record.filed), margin) else INCORRECT_CIK_WINDOW
        evidence = "nothing stored for the period"
    else:
        return MISSING_SEC_FILING, False, "nothing stored for the period and no record"
    if record is None:
        return cls, False, evidence + " | no exception row"
    record_ok = quarter_of(record.period_end) == quarter and admitted(windows, record.cik, pd.Timestamp(record.filed), margin)
    evidence += f" | record {record.cik} {record.accession} filed {record.filed}" + ("" if record_ok else " INCONSISTENT")
    return cls, cls == VENDOR_COVERAGE_GAP and record_ok, evidence


def prepare_filings(facts: pd.DataFrame | None) -> pd.DataFrame:
    """Stored periodic filings, one row per (ticker, accession), with padded CIKs and the period's quarter label."""
    frame = (facts if facts is not None else pd.DataFrame(columns=list(FILING_COLUMNS))).drop_duplicates(["ticker", "accession_number"])
    return frame.assign(cik=pad_cik_series(frame["cik"]), quarter=[quarter_of(day) for day in frame["period_of_report"]])


def assess_continuity(
    arq: pd.DataFrame,
    windows: Mapping[str, Sequence[CikWindow]],
    filings: pd.DataFrame,
    exceptions: Sequence[VendorException],
    window_years: int = CUTOVER_WINDOW_YEARS,
) -> ContinuityReport:
    """Classify every vendor quarter missing near a boundary of each ticker in `windows` that has ARQ rows.

    `arq` is the vendor series as merged (predecessor series applied); `filings` come from `prepare_filings`.
    Records of tickers not measured here are never judged healed.
    """
    stored = set(arq["ticker"].astype(str))
    tickers = sorted(t for t in windows if t in stored and boundaries(windows[t]))
    records = {(r.ticker, r.quarter): r for r in exceptions}
    duplicated = tuple(
        sorted({f"{r.ticker} {r.quarter}" for r in exceptions if sum(1 for o in exceptions if (o.ticker, o.quarter) == (r.ticker, r.quarter)) > 1})
    )
    rows: list[dict[str, object]] = []
    found: set[tuple[str, str]] = set()
    unobserved: list[str] = []
    for ticker in tickers:
        missing, blind = discontinuities(arq[arq["ticker"].astype(str).eq(ticker)], boundaries(windows[ticker]), window_years)
        unobserved += [f"{ticker} {seam.date()}" for seam in blind]
        for quarter, seam in missing.items():
            found.add((ticker, quarter))
            record = records.get((ticker, quarter))
            period = filings[filings["ticker"].astype(str).eq(ticker) & filings["quarter"].eq(quarter)].sort_values("filing_date")
            cls, explained, evidence = classify(windows[ticker], quarter, period, record)
            rows.append(
                {
                    "ticker": ticker,
                    "quarter": quarter,
                    "boundary": str(seam.date()),
                    "class": cls,
                    "explained": explained,
                    "evidence": evidence,
                    "cik": record.cik if record else (str(period["cik"].iloc[0]) if not period.empty else ""),
                    "accession": record.accession if record else (str(period["accession_number"].iloc[0]) if not period.empty else ""),
                    "label": record.label if record else "",
                }
            )
    measured = set(tickers)
    healed = tuple(r for key, r in sorted(records.items()) if r.ticker in measured and key not in found)
    return ContinuityReport(pd.DataFrame(rows, columns=list(TABLE_COLUMNS)), healed, tuple(unobserved), duplicated)


# --------------------------------------------------------------------------- predecessor series


def secfilings_cik(url: object) -> str:
    """The CIK embedded in a Sharadar `secfilings` EDGAR URL; empty when there is none."""
    match = _CIK_IN_URL.search(str(url or ""))
    return pad_cik(match.group(1)) if match else ""


def predecessor_series(
    vendor_tickers: pd.DataFrame, windows: Mapping[str, Sequence[CikWindow]], universe: Iterable[str]
) -> tuple[PredecessorSeries, ...]:
    """The vendor tickers whose `secfilings` CIK owns a closed register window of a universe ticker.

    A vendor ticker that is itself a universe ticker carries a canonical series and is never a predecessor's; of
    several vendor tickers on one CIK, the one with the latest `lastquarter` is taken.
    """
    names = {normalise_ticker(t) for t in universe}
    if vendor_tickers is None or vendor_tickers.empty:
        return ()
    frame = vendor_tickers.assign(
        _cik=[secfilings_cik(url) for url in vendor_tickers["secfilings"]],
        _ticker=vendor_tickers["ticker"].map(normalise_ticker),
        _last=pd.to_datetime(vendor_tickers.get("lastquarter", pd.Series(None, index=vendor_tickers.index, dtype=object)), errors="coerce"),
    )
    frame = frame[frame["_cik"].ne("") & ~frame["_ticker"].isin(names)].sort_values(["_cik", "_last"], ascending=[True, False], kind="mergesort")
    vendor_by_cik = frame.drop_duplicates("_cik").set_index("_cik")["_ticker"].to_dict()
    out = [
        PredecessorSeries(ticker, vendor_by_cik[w.cik], w.cik, w.valid_from, w.valid_to)
        for ticker in sorted(set(windows) & names)
        for w in windows[ticker]
        if w.valid_to is not None and w.cik in vendor_by_cik
    ]
    return tuple(out)


def _inside(frame: pd.DataFrame, series: PredecessorSeries) -> pd.Series:
    period = pd.to_datetime(frame["reportperiod"], errors="coerce")
    mask = period.notna()
    if series.valid_from is not None:
        mask &= period >= series.valid_from
    if series.valid_to is not None:
        mask &= period < series.valid_to
    return mask


EVENT_COLUMNS = ("ticker", "vendor_ticker", "cik", "quarter", "event", "canonical_rows", "predecessor_rows")


def convert_share_basis(frame: pd.DataFrame, factor: float | pd.Series | None) -> pd.DataFrame:
    """`frame` with its share counts multiplied and its per-share figures divided by `factor` (a scalar or one per row).

    `factor=None` (no cited exchange ratio) nulls the whole share block rather than mixing two share bases.
    """
    out = frame.copy()
    counts = [column for column in SHARE_COUNT_COLUMNS if column in out.columns]
    per_share = [column for column in PER_SHARE_COLUMNS if column in out.columns]
    if factor is None:
        out[counts + per_share] = float("nan")
        return out
    out[counts] = out[counts].astype("float64").mul(factor, axis=0)
    out[per_share] = out[per_share].astype("float64").div(factor, axis=0)
    return out


def price_only_events(
    yf_splits: pd.DataFrame | None, vendor_actions: pd.DataFrame | None, windows: Sequence[tuple[PredecessorSeries, pd.Timestamp]]
) -> pd.DataFrame:
    """`(ticker, cik, date, value)`: each window ticker's `prices_splits` events in `[valid_from, seam)` that the
    owner's vendor series never applied, i.e. its `sharadar_actions` show a spinoff and no split that day."""
    empty = pd.DataFrame(
        {
            "ticker": pd.Series(dtype=object),
            "cik": pd.Series(dtype=object),
            "date": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
        }
    )
    if yf_splits is None or yf_splits.empty or vendor_actions is None or vendor_actions.empty:
        return empty
    yf = yf_splits.assign(date=pd.to_datetime(yf_splits["date"]), _ticker=yf_splits["ticker"].map(normalise_ticker))
    acts = vendor_actions.assign(date=pd.to_datetime(vendor_actions["date"]), _vendor=vendor_actions["ticker"].map(normalise_ticker))
    found: list[pd.DataFrame] = []
    for s, seam in windows:
        own = acts[acts["_vendor"].eq(s.vendor_ticker)]
        spun = set(own.loc[own["action"].eq(SHARADAR_ACTION_SPINOFF), "date"]) - set(own.loc[own["action"].eq(SHARADAR_ACTION_SPLIT), "date"])
        start = s.valid_from if s.valid_from is not None else pd.Timestamp.min
        hit = yf[yf["_ticker"].eq(s.ticker) & yf["date"].ge(start) & yf["date"].lt(seam) & yf["date"].isin(spun)]
        found.append(pd.DataFrame({"ticker": s.ticker, "cik": s.cik, "date": hit["date"], "value": hit["ratio"].astype("float64")}))
    frames = [frame for frame in found if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty


def _window_events(events: pd.DataFrame | None, s: PredecessorSeries) -> pd.DataFrame:
    """The `price_only_events` rows of one window."""
    if events is None or events.empty:
        return pd.DataFrame(columns=["ticker", "date", "value"])
    return events.loc[events["ticker"].eq(s.ticker) & events["cik"].eq(s.cik), ["ticker", "date", "value"]].assign(
        date=lambda f: pd.to_datetime(f["date"])
    )


def _later_events(dates: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Per row, the product of `events` dated strictly after its `dates` value (1.0 if none)."""
    factor = pd.Series(1.0, index=dates.index)
    stamps = pd.to_datetime(dates, errors="coerce")
    for day, value in zip(events["date"], events["value"], strict=True):
        factor[stamps < day] *= float(value)
    return factor


def rebase_split_events(
    splits: pd.DataFrame,
    owner_splits: pd.DataFrame,
    windows: Sequence[tuple[PredecessorSeries, ShareExchange]],
    price_only: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The split events with each converted window's pre-seam part replaced by the owner's own splits, its
    `price_only` events and one event of the exchange ratio on the seam, so a count de-adjusted inside the window is
    the owner's as-filed count."""
    out = splits[["ticker", "date", "value"]].assign(date=pd.to_datetime(splits["date"]))
    owners = owner_splits[["ticker", "date", "value"]].assign(
        date=pd.to_datetime(owner_splits["date"]), _vendor=owner_splits["ticker"].map(normalise_ticker)
    )
    added: list[pd.DataFrame] = []
    for s, exchange in windows:
        start = s.valid_from if s.valid_from is not None else pd.Timestamp.min
        before = out["ticker"].eq(s.ticker) & out["date"].ge(start) & out["date"].lt(exchange.seam_date)
        own = owners[owners["_vendor"].eq(s.vendor_ticker) & owners["date"].ge(start) & owners["date"].lt(exchange.seam_date)]
        out = out[~before]
        added.append(own.drop(columns="_vendor").assign(ticker=s.ticker))
        added.append(_window_events(price_only, s))
        added.append(pd.DataFrame({"ticker": [s.ticker], "date": [exchange.seam_date], "value": [float(exchange.ratio)]}))
    frames = [frame for frame in (out, *added) if not frame.empty]
    if not frames:
        return out
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def apply_predecessor_series(
    arq: pd.DataFrame,
    predecessors: pd.DataFrame,
    series: Sequence[PredecessorSeries],
    factors: Mapping[tuple[str, str], float | None] | None = None,
    price_only: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inside each predecessor window, replace the canonical ticker's ARQ rows by the window owner's own rows.

    Rows are tested on `reportperiod`; the owner's rows are relabelled to the canonical ticker and, when `factors`
    is given, put on the canonical share basis by `factors[(ticker, cik)]` times the window's `price_only` events
    dated after the row's `date` (`convert_share_basis`). A window whose
    owner has no stored row inside it is left unchanged. Returns `(arq, events)`, one event per quarter
    (`EVENT_COLUMNS`): `replaced` (both had it), `filled` (only the owner) or `dropped` (only the canonical series).
    """
    out = arq.copy()
    events: list[dict[str, object]] = []
    owner_rows = predecessors.assign(_vendor=predecessors["ticker"].map(normalise_ticker)) if not predecessors.empty else predecessors
    for s in series:
        own = owner_rows[owner_rows["_vendor"].eq(s.vendor_ticker)].drop(columns="_vendor") if not owner_rows.empty else owner_rows
        own = own[_inside(own, s)] if not own.empty else own
        if own.empty:
            continue
        if factors is not None:
            factor = factors.get((s.ticker, s.cik))
            spins = _window_events(price_only, s)
            own = convert_share_basis(own, factor * _later_events(own["date"], spins) if factor is not None and not spins.empty else factor)
        canonical = out["ticker"].astype(str).eq(s.ticker) & _inside(out, s)
        before = quarter_ordinal(out.loc[canonical, "calendardate"]).value_counts()
        after = quarter_ordinal(own["calendardate"]).value_counts()
        for ordinal in sorted(set(before.index) | set(after.index)):
            n_old, n_new = int(before.get(ordinal, 0)), int(after.get(ordinal, 0))
            event = _EVENTS[0] if n_old and n_new else (_EVENTS[1] if n_new else _EVENTS[2])
            events.append(
                {
                    "ticker": s.ticker,
                    "vendor_ticker": s.vendor_ticker,
                    "cik": s.cik,
                    "quarter": quarter_label(int(ordinal)),
                    "event": event,
                    "canonical_rows": n_old,
                    "predecessor_rows": n_new,
                }
            )
        out = pd.concat([out[~canonical], own.assign(ticker=s.ticker)], ignore_index=True)
    return out, pd.DataFrame(events, columns=list(EVENT_COLUMNS))


def _assets_by_quarter(frame: pd.DataFrame, series: PredecessorSeries) -> pd.Series:
    """`assets` per calendar-quarter ordinal inside the series window, from each quarter's earliest filing."""
    frame = frame[_inside(frame, series)]
    keyed = frame.assign(_q=quarter_ordinal(frame["calendardate"]), _d=pd.to_datetime(frame["date"])).sort_values("_d", kind="mergesort")
    return keyed.drop_duplicates("_q").set_index("_q")["assets"].astype("float64")


def other_company_quarters(
    arq: pd.DataFrame, predecessors: pd.DataFrame, series: PredecessorSeries, tolerance: float = OTHER_COMPANY_TOLERANCE
) -> pd.DataFrame:
    """Quarters inside the window where the canonical vendor `assets` differ from the owner's by more than `tolerance`.

    Columns `quarter, canonical_assets, owner_assets`; one row per quarter both series carry (earliest filing).
    """
    canonical = arq[arq["ticker"].astype(str).eq(series.ticker)]
    own = predecessors[predecessors["ticker"].map(normalise_ticker).eq(series.vendor_ticker)]
    left, right = _assets_by_quarter(canonical, series), _assets_by_quarter(own, series)
    both = pd.concat([left.rename("canonical_assets"), right.rename("owner_assets")], axis=1, join="inner").dropna()
    differs = (both["canonical_assets"] - both["owner_assets"]).abs() > tolerance * both["owner_assets"].abs()
    out = both[differs].reset_index(names="_q")
    return out.assign(quarter=[quarter_label(int(q)) for q in out["_q"]])[["quarter", "canonical_assets", "owner_assets"]]
