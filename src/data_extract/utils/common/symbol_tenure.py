"""Ticker identity, axis B: which issuer CIK held symbol X on date d (`symbol_tenure`).

Derived offline from `SUBMISSION.TSV` in the cached Form 345 quarter zips, plus the evidenced
manual config; PK (symbol, issuer_cik, valid_from), manual rows winning a PK collision. Tenures
may overlap: `valid_to` is the observed end of that CIK's own filing window, so the table answers
a membership question, never a single-answer lookup. Not read by `owns()`; it feeds axis-A
candidates, the D19 cross-check and CIK-less symbol/date sources.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.common.incremental import matches_stored
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker, pad_cik_series

logger = logging.getLogger(__name__)

#: SUBMISSION feeds `symbol_tenure`; joined to REPORTINGOWNER on accession it feeds the `entity_lineage` owner sets.
SUBMISSION_MEMBER = "SUBMISSION.TSV"
OWNER_MEMBER = "REPORTINGOWNER.TSV"
_FORM345_READ = {
    SUBMISSION_MEMBER: ZipRead(usecols=frozenset({"ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "ISSUERNAME", "FILING_DATE"}), upper=True),
    OWNER_MEMBER: ZipRead(usecols=frozenset({"ACCESSION_NUMBER", "RPTOWNERCIK"}), upper=True, required=False),
}

#: `ISSUERNAME` is optional and feeds `evidence`; a zip missing any of these is skipped loudly.
_REQUIRED_COLUMNS = frozenset({"ISSUERCIK", "ISSUERTRADINGSYMBOL", "FILING_DATE"})

#: `ISSUERTRADINGSYMBOL` strings that mean "no symbol"; dropped so they never read as a reused ticker.
_NULL_SYMBOLS = frozenset({"", "NONE", "N/A", "NA", "-", "--", "N.A.", "NULL"})

#: Stripped because a quoted symbol would split a real tenure; no other normalisation is applied.
_SYMBOL_NOISE_CHARS = '"'

#: Month names for the `DD-MON-YYYY` filing-date shape (ISO dates are parsed too).
_MONTHS = {month: i + 1 for i, month in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}

#: Fewer cached quarters than this warns (not raises) that the derivation is a partial history.
MIN_EXPECTED_QUARTERS = 80

MANUAL_TENURE_FILE = Path("sec") / "symbol_tenure_manual.json"
MANUAL_TENURE_VERSION = 1


class ManualSymbolTenureError(ValueError):
    """The manual symbol-tenure config is unsafe or cannot be audited."""


@dataclass(frozen=True)
class Form345Scan:
    """One pass over the cached Form 345 quarter zips.

    `tenure_parts` holds each readable quarter's per-(symbol, issuer_cik) aggregate; `owner_pairs`
    the distinct (issuer_cik, owner_cik_raw) reporting-owner pairs over every quarter, the owner CIK
    as filed (padded only for the issuers a caller keeps); `drops` counts rows read, kept and
    dropped per reason.
    """

    cache: Path
    quarters: tuple[pd.Period, ...]
    tenure_parts: tuple[pd.DataFrame, ...]
    owner_pairs: pd.DataFrame
    drops: Counter


def _manual_date(value: object, *, field: str, location: str) -> pd.Timestamp | None:
    """Parse one strict ISO date from the manual config."""
    if value is None and field == "valid_to":
        return None
    if not isinstance(value, str):
        raise ManualSymbolTenureError(f"{location}.{field} must be an ISO YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ManualSymbolTenureError(f"{location}.{field} is not an ISO YYYY-MM-DD date: {value!r}") from exc
    if value != parsed.isoformat():
        raise ManualSymbolTenureError(f"{location}.{field} must use canonical YYYY-MM-DD form: {value!r}")
    return pd.Timestamp(parsed)


def load_manual_symbol_tenure(config_dir: str | Path) -> pd.DataFrame:
    """Load, normalize and validate the evidenced manual ticker-history config; overlapping intervals raise.

    Keeps the ``canonical_ticker`` and ``reason`` audit columns, which are dropped when ``symbol_tenure``
    is materialized; runtime resolution reads only the table.
    """
    path = Path(config_dir) / MANUAL_TENURE_FILE
    if not path.exists():
        raise FileNotFoundError(f"symbol_tenure: manual config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManualSymbolTenureError(f"symbol_tenure: invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != MANUAL_TENURE_VERSION:
        raise ManualSymbolTenureError(f"symbol_tenure: {path} must be an object with version={MANUAL_TENURE_VERSION}")
    tickers = payload.get("tickers")
    if not isinstance(tickers, dict) or not tickers:
        raise ManualSymbolTenureError(f"symbol_tenure: {path}.tickers must be a non-empty object")

    records: list[dict[str, object]] = []
    for raw_ticker, raw_intervals in tickers.items():
        ticker = normalise_ticker(raw_ticker)
        if not ticker:
            raise ManualSymbolTenureError(f"symbol_tenure: {path} contains an empty canonical ticker")
        if not isinstance(raw_intervals, list) or not raw_intervals:
            raise ManualSymbolTenureError(f"symbol_tenure: {ticker} must contain at least one interval")
        records.extend(_parse_manual_interval(raw, ticker, f"tickers.{ticker}[{index}]") for index, raw in enumerate(raw_intervals))

    out = pd.DataFrame.from_records(records)
    identity_columns = ["canonical_ticker", "symbol", "issuer_cik", "valid_from", "valid_to"]
    out = out.drop_duplicates(identity_columns, keep="first").sort_values(["symbol", "valid_from", "issuer_cik"], kind="mergesort", ignore_index=True)
    _check_manual_overlaps(out)
    return out


def _parse_manual_interval(raw: object, ticker: str, location: str) -> dict[str, object]:
    """One validated manual interval as a `symbol_tenure` record plus its audit columns."""
    if not isinstance(raw, dict):
        raise ManualSymbolTenureError(f"{location} must be an object")
    symbol = normalise_ticker(raw.get("symbol", ""))
    if not symbol:
        raise ManualSymbolTenureError(f"{location}.symbol must be non-empty")
    cik = raw.get("issuer_cik")
    if not isinstance(cik, str) or len(cik) != 10 or not cik.isdigit():
        raise ManualSymbolTenureError(f"{location}.issuer_cik must be a zero-padded 10-digit string")
    start = _manual_date(raw.get("valid_from"), field="valid_from", location=location)
    end = _manual_date(raw.get("valid_to"), field="valid_to", location=location)
    if end is not None and start is not None and start >= end:
        raise ManualSymbolTenureError(f"{location} must satisfy valid_from < valid_to")
    evidence = raw.get("evidence")
    if not isinstance(evidence, list) or not evidence or any(not isinstance(item, str) or not item.strip() for item in evidence):
        raise ManualSymbolTenureError(f"{location}.evidence must be a non-empty list of non-empty strings")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ManualSymbolTenureError(f"{location}.reason must be non-empty")
    return {
        "canonical_ticker": ticker,
        "symbol": symbol,
        "issuer_cik": cik,
        "valid_from": start,
        "valid_to": end,
        "n_filings": 0,
        "source": "manual",
        "evidence": " | ".join(item.strip() for item in evidence),
        "reason": reason.strip(),
    }


def _check_manual_overlaps(manual: pd.DataFrame) -> None:
    """Raise on the first manual interval that starts before the previous one of its symbol ends."""
    df_ordered = manual.sort_values(["symbol", "valid_from", "valid_to", "issuer_cik"], kind="mergesort", ignore_index=True)
    df_previous = df_ordered.shift(1)
    same_symbol = df_ordered["symbol"].eq(df_previous["symbol"])
    overlaps = same_symbol & (df_previous["valid_to"].isna() | (df_ordered["valid_from"] < df_previous["valid_to"]))
    if not overlaps.any():
        return
    first = int(overlaps.to_numpy().argmax())
    row, previous = df_ordered.iloc[first], df_previous.iloc[first]
    previous_end = "open" if pd.isna(previous["valid_to"]) else pd.Timestamp(previous["valid_to"]).date()
    row_end = "open" if pd.isna(row["valid_to"]) else pd.Timestamp(row["valid_to"]).date()
    raise ManualSymbolTenureError(
        "symbol_tenure: overlapping manual intervals for "
        f"{row['symbol']}: {previous['canonical_ticker']}/{previous['issuer_cik']} "
        f"[{pd.Timestamp(previous['valid_from']).date()}, {previous_end}) and "
        f"{row['canonical_ticker']}/{row['issuer_cik']} [{pd.Timestamp(row['valid_from']).date()}, {row_end})"
    )


def materialize_symbol_tenure(derived: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Manual plus derived rows; a PK (symbol, issuer_cik, valid_from) collision coalesces with manual first."""
    table_columns = ["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence"]
    out = pd.concat([manual[table_columns], derived[table_columns]], ignore_index=True)
    priority = out["source"].map({"manual": 0, "form345": 1}).fillna(2)
    out = (
        out.assign(_source_priority=priority)
        .sort_values(["symbol", "valid_from", "_source_priority", "issuer_cik"], kind="mergesort")
        .drop(columns="_source_priority")
        .reset_index(drop=True)
    )
    primary_key = ["symbol", "issuer_cik", "valid_from"]
    collides = out.duplicated(primary_key, keep=False)
    out = pd.concat([out[~collides], _coalesce_collisions(out[collides], primary_key)]).sort_index().reset_index(drop=True)
    logger.info(
        "symbol_tenure: materialized %d manual and %d derived row(s) as %d unique table-grain row(s) over %d symbol(s)",
        len(manual),
        len(derived),
        len(out),
        out["symbol"].nunique(),
    )
    return out


def _coalesce_collisions(df_collisions: pd.DataFrame, primary_key: list[str]) -> pd.DataFrame:
    """One row per colliding primary key, kept at its first position: the first row's fields, the
    group's highest `n_filings`, and every non-blank evidence string labelled by its source."""
    if df_collisions.empty:
        return df_collisions
    group = df_collisions.groupby(primary_key, sort=False, dropna=False).ngroup()
    evidence = df_collisions["evidence"].map(str)
    labelled = (df_collisions["source"].map(str) + " evidence: " + evidence).where(evidence.str.strip().ne(""))
    joined = labelled.groupby(group, sort=False).agg(lambda values: " | ".join(dict.fromkeys(values.dropna())))
    n_filings = pd.to_numeric(df_collisions["n_filings"], errors="coerce").groupby(group, sort=False).max()
    df_winners = df_collisions[~df_collisions.duplicated(primary_key)]
    winner_group = group.loc[df_winners.index]
    return df_winners.assign(n_filings=n_filings.loc[winner_group].to_numpy(), evidence=joined.loc[winner_group].to_numpy())


def _parse_filing_dates(raw: pd.Series) -> pd.Series:
    """`FILING_DATE` -> datetime64: `DD-MON-YYYY` decoded explicitly, the rest as ISO (never `format="mixed"`)."""
    text = raw.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    parts = text.str.split("-", n=2, expand=True)
    if parts.shape[1] == 3:
        # Decode only the matching subset; junk rows elsewhere would make a whole-column parse raise.
        is_month_name = parts[1].isin(_MONTHS).fillna(False)
        if is_month_name.any():
            out.loc[is_month_name] = pd.to_datetime(text[is_month_name], format="%d-%b-%Y", errors="coerce")
    todo = out.isna() & text.notna()
    if todo.any():
        out.loc[todo] = pd.to_datetime(text[todo].str.slice(0, 10), format="ISO8601", errors="coerce")
    return out


def _quarter_of(path: Path) -> pd.Period | None:
    """`2026q1.zip` -> that quarterly Period, or None when the name is not a quarter."""
    try:
        return pd.Period(path.stem.upper(), freq="Q")
    except Exception:  # noqa: BLE001 -- an unexpected file name, not a bug
        return None


def _aggregate_submission(raw: pd.DataFrame, name: str, drops: Counter) -> pd.DataFrame | None:
    """Per-(symbol, cik) first/last filing date, count and issuer name for one quarter; None when nothing usable."""
    missing = _REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        logger.warning("symbol_tenure: %s lacks %s -> SKIPPED", name, sorted(missing))
        drops["missing_columns"] += 1
        return None

    df = pd.DataFrame(
        {
            "symbol": (raw["ISSUERTRADINGSYMBOL"].astype("string").str.replace(_SYMBOL_NOISE_CHARS, "", regex=False).str.strip().str.upper()),
            "issuer_cik": raw["ISSUERCIK"],
            "issuer_name": (raw["ISSUERNAME"].astype("string") if "ISSUERNAME" in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string")),
            "filed": _parse_filing_dates(raw["FILING_DATE"]),
        }
    )
    drops["rows_read"] += len(df)
    bad_symbol = df["symbol"].isna() | df["symbol"].isin(_NULL_SYMBOLS)
    bad_cik = df["issuer_cik"].isin(("", "0" * 10))
    bad_date = df["filed"].isna()
    drops["empty_symbol"] += int(bad_symbol.sum())
    drops["empty_cik"] += int((bad_cik & ~bad_symbol).sum())
    drops["unparseable_filing_date"] += int((bad_date & ~bad_symbol & ~bad_cik).sum())
    df = df[~(bad_symbol | bad_cik | bad_date)]
    drops["rows_kept"] += len(df)
    if df.empty:
        return None
    return df.groupby(["symbol", "issuer_cik"], as_index=False, sort=False).agg(
        first_filed=("filed", "min"), last_filed=("filed", "max"), n_filings=("filed", "size"), issuer_name=("issuer_name", "last")
    )


def _owner_pairs(submission: pd.DataFrame, owners: pd.DataFrame) -> pd.DataFrame:
    """Distinct (issuer_cik, owner_cik_raw) pairs of one quarter, joined on the accession number."""
    if owners.empty:
        return pd.DataFrame(columns=["issuer_cik", "owner_cik_raw"])
    df_issuers = pd.DataFrame({"accession": submission["ACCESSION_NUMBER"], "issuer_cik": submission["ISSUERCIK"]})
    df_owners = pd.DataFrame({"accession": owners["ACCESSION_NUMBER"], "owner_cik_raw": owners["RPTOWNERCIK"]}).dropna(subset=["owner_cik_raw"])
    return df_owners.merge(df_issuers, on="accession", how="inner")[["issuer_cik", "owner_cik_raw"]].drop_duplicates(ignore_index=True)


def scan_form345_cache(cache: Path) -> Form345Scan:
    """Read every cached Form 345 quarter zip once for both identity tables.

    A corrupt zip is skipped and kept (`on_corrupt="skip"`); a zip without `SUBMISSION.TSV` is skipped loudly.
    """
    zips = sorted(p for p in cache.glob("*.zip") if _quarter_of(p) is not None)
    if not zips:
        raise FileNotFoundError(
            f"symbol_tenure: no Form 345 quarter zips under {cache}. The derivation is "
            "offline and reads only the cache; run the `insider` extract first."
        )
    if len(zips) < MIN_EXPECTED_QUARTERS:
        logger.warning(
            "symbol_tenure: only %d cached quarter zip(s) under %s (expected >= %d). The "
            "derived table will be a PARTIAL history that LOOKS complete -- every tenure "
            "ending inside an absent quarter is wrong.",
            len(zips),
            cache,
            MIN_EXPECTED_QUARTERS,
        )

    drops: Counter = Counter()
    tenure_parts: list[pd.DataFrame | None] = []
    owner_parts: list[pd.DataFrame] = [_owner_pairs(pd.DataFrame(), pd.DataFrame())]
    for path in zips:
        tables = read_zip_tables(path, _FORM345_READ, on_corrupt="skip", log=logger)
        if tables is None:
            drops["corrupt_zip"] += 1
            continue
        if not tables:
            logger.warning("symbol_tenure: %s has no %s -> SKIPPED", path.name, SUBMISSION_MEMBER)
            drops["no_submission_member"] += 1
            continue
        submission = tables[SUBMISSION_MEMBER]
        if "ISSUERCIK" in submission.columns:
            submission["ISSUERCIK"] = pad_cik_series(submission["ISSUERCIK"])
        tenure_parts.append(_aggregate_submission(submission, path.name, drops))
        owner_parts.append(_owner_pairs(submission, tables[OWNER_MEMBER]))

    return Form345Scan(
        cache=cache,
        quarters=tuple(q for q in (_quarter_of(p) for p in zips) if q is not None),
        tenure_parts=tuple(part for part in tenure_parts if part is not None),
        owner_pairs=pd.concat(owner_parts, ignore_index=True).drop_duplicates(ignore_index=True),
        drops=drops,
    )


def derive_symbol_tenure(scan: Form345Scan) -> pd.DataFrame:
    """(symbol, issuer_cik) -> (valid_from, valid_to, n_filings) from one Form 345 cache scan.

    Deterministic, sorted on (symbol, valid_from, issuer_cik). Half-open: `valid_to` is `last_filed + 1 day`,
    NULL ("no end observed") when `last_filed` falls in the latest cached quarter.
    """
    if not scan.tenure_parts:
        raise ValueError(f"symbol_tenure: every zip under {scan.cache} was unreadable or empty")
    latest_quarter = max(scan.quarters)
    agg = (
        pd.concat(scan.tenure_parts, ignore_index=True)
        .groupby(["symbol", "issuer_cik"], as_index=False, sort=False)
        .agg(valid_from=("first_filed", "min"), last_filed=("last_filed", "max"), n_filings=("n_filings", "sum"), issuer_name=("issuer_name", "last"))
    )

    still_open = agg["last_filed"] >= latest_quarter.start_time
    out = pd.DataFrame(
        {
            "symbol": agg["symbol"].astype(str),
            "issuer_cik": agg["issuer_cik"].astype(str),
            "valid_from": agg["valid_from"],
            "valid_to": (agg["last_filed"] + pd.Timedelta(days=1)).mask(still_open),
            "n_filings": agg["n_filings"].astype("int64"),
            "source": "form345",
            "evidence": agg["issuer_name"].fillna("").astype(str).str.strip(),
        }
    ).sort_values(["symbol", "valid_from", "issuer_cik"], kind="mergesort", ignore_index=True)

    per_symbol = out.groupby("symbol")["issuer_cik"].nunique()
    logger.info(
        "symbol_tenure: %d quarter(s) %s..%s -> %d row(s) over %d symbol(s); %d symbol(s) "
        "had >1 issuer CIK; %d tenure(s) still open. Read %d submission row(s), kept %d; "
        "dropped %s",
        len(scan.quarters),
        min(scan.quarters),
        latest_quarter,
        len(out),
        int(per_symbol.size),
        int((per_symbol > 1).sum()),
        int(out["valid_to"].isna().sum()),
        scan.drops["rows_read"],
        scan.drops["rows_kept"],
        ", ".join(f"{k}={v}" for k, v in sorted(scan.drops.items()) if k not in {"rows_read", "rows_kept"}) or "nothing",
    )
    return out


def changed_tenure_symbols(
    existing: pd.DataFrame,
    derived: pd.DataFrame,
) -> list[str]:
    """Sorted symbols whose CIK membership or observed bounds changed."""
    columns = ["symbol", "issuer_cik", "valid_from", "valid_to"]
    if "source" in existing.columns and "source" in derived.columns:
        columns.append("source")

    def signatures(frame: pd.DataFrame) -> dict[str, tuple[tuple[str, ...], ...]]:
        normal = frame[columns].copy()
        normal["symbol"] = normal["symbol"].astype(str).str.upper().str.strip()
        for column in ("valid_from", "valid_to"):
            normal[column] = pd.to_datetime(normal[column], errors="coerce").astype("string")
        signature_columns = [column for column in columns if column != "symbol"]
        return {
            str(symbol): tuple(sorted(tuple(map(str, row)) for row in group[signature_columns].itertuples(index=False, name=None)))
            for symbol, group in normal.groupby("symbol", sort=False)
        }

    before, after = signatures(existing), signatures(derived)
    return sorted(symbol for symbol in set(before) | set(after) if before.get(symbol) != after.get(symbol))


def build_symbol_tenure(context: Context, scan: Form345Scan, config_dir: str | Path | None = None) -> pd.DataFrame:
    """Derive `symbol_tenure` and replace the table unless unchanged; returns the materialized frame.

    `replace`, never `save`: a full derivation must not leave stale rows behind.
    """
    existing = context.store.load(Tables.symbol_tenure, project=True, optional=True)
    derived = derive_symbol_tenure(scan)
    manual = load_manual_symbol_tenure(config_dir or context.config_dir)
    out = materialize_symbol_tenure(derived, manual)
    context.log.info(
        f"symbol_tenure: validated {len(manual)} manual interval(s) for {manual['canonical_ticker'].nunique()} canonical ticker(s); no manual overlap"
    )
    if existing is None:
        context.log.info(f"symbol_tenure: cold build with {len(out)} row(s) over {out['symbol'].nunique()} symbol(s)")
    else:
        changed = changed_tenure_symbols(existing, out)
        context.log.info(f"symbol_tenure: {len(changed)} changed symbol(s): {', '.join(changed) if changed else 'none'}")
    unchanged = matches_stored(existing, out, Tables.symbol_tenure)
    written = 0 if unchanged else context.store.replace(Tables.symbol_tenure, out)
    if unchanged:
        logger.info("symbol_tenure: unchanged (%d row(s)); replace skipped", len(out))
    else:
        logger.info("symbol_tenure: wrote %d row(s)", written)
    return out
