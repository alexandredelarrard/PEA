"""Ticker identity, axis B: which issuer CIK held symbol X on date d (`symbol_tenure`).

Evidence rows by `source`: `form345` (derived offline from `SUBMISSION.TSV` in the cached Form 345
quarter zips) and `manual` (the evidenced config) are written by the identity build; `dei` (Notes
cover-page symbols) is written one Notes zip period at a time. Each source rewrites only its own partition.
PK (symbol, issuer_cik, valid_from, source, evidence_period); `evidence_period` is '' for `form345` and `manual`. Tenures may overlap: `valid_to` is the observed end of that CIK's own filing
window, so the table answers a membership question, never a single-answer lookup.
"""

from __future__ import annotations

import json
import logging
import re
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

#: Cleaned symbol fields that mean "no symbol"; a field with no letter or digit is one too.
_PLACEHOLDER_SYMBOLS = frozenset({"NONE", "N/A", "NA", "N.A.", "NULL", "NO SYMBOL"})

#: Quote characters removed from a symbol field; brackets read as spaces (`(BBT)`, `NWIN(OB)`).
_SYMBOL_QUOTES = re.compile(r"[\"'`]")
_SYMBOL_BRACKETS = re.compile(r"[()\[\]{}]")

#: EDGAR state-of-incorporation marker typed in front of a symbol (`/DE/CHD`, `DE/TGAL`).
_STATE_MARKER = re.compile(r"^(?:/[A-Z]{2}/|DE/)")

#: A slash before a full symbol separates a list (`BFA/BFB`); before one character (`BRK/B`) or a series code
#: (note `BDX/26A`, preferred `MS/PL`, `BFS/PRD`, warrant `CANO/WS`) it names one security of that issuer.
_SLASH_LIST = re.compile(r"/(?!(?:\d{2}[A-Z]{1,2}|PR?[A-Z]|WS)(?![A-Z0-9]))(?=[A-Z0-9]{2})")

#: Spaces around a slash (`KVA / KVB`).
_SLASH_SPACES = re.compile(r"\s*/\s*")

#: List separators inside one symbol field; whitespace is not one (`ALF A` is one class symbol).
_LIST_SEPARATORS = re.compile(r"[,;:]+")

#: Exchange, venue and qualifier words dropped from a multi-word field (`NYSE: GLW`, `CARR WI`, `HCA INC.`).
_QUALIFIER_WORDS = frozenset({"NYSE", "NASDAQ", "AMEX", "OTC", "OTCBB", "ARCA", "PINK", "PK", "OB", "WI", "US", "INC", "CO", "CORP"})

#: OTC venue suffix on one symbol (`DALRQ.PK`).
_VENUE_SUFFIX = re.compile(r"\.(?:PK|OB)$")

#: A market symbol in roster spelling: letters and digits with at least one letter, `-` between parts.
_SYMBOL_SHAPE = re.compile(r"(?=[A-Z0-9-]*[A-Z])[A-Z0-9]+(?:-[A-Z0-9]+)*")

#: Month names for the `DD-MON-YYYY` filing-date shape (ISO dates are parsed too).
_MONTHS = {month: i + 1 for i, month in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}

#: Fewer cached quarters than this warns (not raises) that the derivation is a partial history.
MIN_EXPECTED_QUARTERS = 80

#: The `symbol_tenure` partitions `build_symbol_tenure` owns; every other source keeps its rows.
BUILD_SOURCES = ("form345", "manual")

#: Column order of the materialized partitions.
TABLE_COLUMNS = ("symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence_period", "evidence")

#: Notes cover-page (`dei:TradingSymbol`) partition, one row set per Notes zip period.
DEI_SOURCE = "dei"

MANUAL_TENURE_FILE = Path("sec") / "symbol_tenure_manual.json"
MANUAL_TENURE_VERSION = 1
#: An SEC accession number, the evidence a rejected tenure must cite.
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")


class ManualSymbolTenureError(ValueError):
    """The manual symbol-tenure config is unsafe or cannot be audited."""


def normalise_market_symbol(value: object) -> str:
    """Use the roster's hyphen spelling for market share-class separators."""
    return str(value).strip().upper().replace(".", "-").replace("/", "-")


def parse_symbol_field(value: object) -> tuple[str, ...] | None:
    """The market symbols one filer-typed symbol field names, in roster spelling.

    None for a missing field or a placeholder ("no symbol"), () for noise. Quotes, brackets, qualifier
    words and venue suffixes are removed; `,` `;` `:` and a slash before a full symbol (not a series code) split a list,
    before the share-class rule (`.` and `/` -> `-`). Words inside one item join as a class (`ALF A` -> `ALF-A`).
    """
    if not isinstance(value, str):
        return None
    text = _SYMBOL_BRACKETS.sub(" ", _SYMBOL_QUOTES.sub("", value)).strip().upper()
    text = _STATE_MARKER.sub("", _SLASH_SPACES.sub("/", " ".join(text.split())))
    if text in _PLACEHOLDER_SYMBOLS or not any(char.isalnum() for char in text):
        return None
    items = [[word.strip("-.") for word in item.split()] for item in _LIST_SEPARATORS.split(_SLASH_LIST.sub(",", text))]
    items = [[word for word in item if word] for item in items]
    if sum(len(item) for item in items) > 1:
        items = [[word for word in item if word not in _QUALIFIER_WORDS] for item in items]
    symbols = [_join_item(item) for item in items if item]
    if not symbols or not all(symbol and _SYMBOL_SHAPE.fullmatch(symbol) for symbol in symbols):
        return ()
    return tuple(dict.fromkeys(str(symbol) for symbol in symbols))


def _join_item(words: list[str]) -> str | None:
    """One list item's words as one symbol: spaced letters join (`N O G`), a base plus short class words
    hyphenate (`HBC PR A`); anything else (`OWL ROCK T`) is None."""
    words = [_VENUE_SUFFIX.sub("", word) for word in words]
    if len(words) == 1:
        return normalise_market_symbol(words[0])
    if all(len(word) == 1 for word in words):
        return "".join(words)
    if len(words[0]) >= 2 and all(len(word) <= 2 for word in words[1:]):
        return normalise_market_symbol("-".join(words))
    return None


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


def load_rejected_symbol_tenure(config_dir: str | Path) -> pd.DataFrame:
    """The manual config's `rejected` derived tenures: `symbol`, `issuer_cik`, `valid_from`, `valid_to`, cited
    `accessions`, `evidence`, `reason`. Empty when the file or the list is absent; an uncited entry raises."""
    columns = ["symbol", "issuer_cik", "valid_from", "valid_to", "accessions", "evidence", "reason"]
    path = Path(config_dir) / MANUAL_TENURE_FILE
    if not path.exists():
        return pd.DataFrame(columns=columns)
    raw_entries = json.loads(path.read_text(encoding="utf-8")).get("rejected", [])
    if not isinstance(raw_entries, list):
        raise ManualSymbolTenureError(f"symbol_tenure: {path}.rejected must be a list")
    records = [_parse_rejection(raw, f"rejected[{index}]") for index, raw in enumerate(raw_entries)]
    return pd.DataFrame.from_records(records, columns=columns)


def _parse_rejection(raw: object, location: str) -> dict[str, object]:
    """One validated rejected tenure: a dated (symbol, issuer CIK) interval and the accessions that mis-tag it."""
    if not isinstance(raw, dict):
        raise ManualSymbolTenureError(f"{location} must be an object")
    record = _parse_manual_interval(
        {key: raw.get(key) for key in ("symbol", "issuer_cik", "valid_from", "valid_to", "evidence", "reason")}, "-", location
    )
    if record["valid_to"] is None:
        raise ManualSymbolTenureError(f"{location}.valid_to must close the rejected interval")
    accessions = raw.get("accessions")
    if not isinstance(accessions, list) or not accessions or any(not isinstance(a, str) or not _ACCESSION.fullmatch(a) for a in accessions):
        raise ManualSymbolTenureError(f"{location}.accessions must be a non-empty list of SEC accession numbers")
    return {key: record[key] for key in ("symbol", "issuer_cik", "valid_from", "valid_to", "evidence", "reason")} | {"accessions": tuple(accessions)}


def reject_derived_tenure(derived: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    """`derived` without its rows that a rejection covers: same (symbol, issuer CIK), interval inside the rejected
    dates. A rejection that matches the pair but no longer covers its interval keeps the row and warns."""
    if rejected.empty or derived.empty:
        return derived
    drop = pd.Series(False, index=derived.index)
    starts, ends = pd.to_datetime(derived["valid_from"]), pd.to_datetime(derived["valid_to"])
    for symbol, cik, start, end, accessions in zip(
        rejected["symbol"],
        rejected["issuer_cik"],
        pd.to_datetime(rejected["valid_from"]),
        pd.to_datetime(rejected["valid_to"]),
        rejected["accessions"],
        strict=True,
    ):
        pair = derived["symbol"].astype(str).eq(symbol) & derived["issuer_cik"].astype(str).eq(cik)
        covered = pair & (starts >= start) & ends.notna() & (ends <= end)
        label = f"{symbol}/{cik} {start.date()}..{end.date()}"
        if (pair & ~covered).any():
            logger.warning("symbol_tenure: the rejection of %s no longer covers the derived interval; kept for review", label)
        if covered.any():
            logger.info("symbol_tenure: rejected %s (%d accession(s) cited)", label, len(accessions))
        drop |= covered
    return derived[~drop].reset_index(drop=True)


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
    """The `form345` and `manual` partitions as table rows: manual ordered first, every row kept."""
    evidence_columns = [column for column in TABLE_COLUMNS if column != "evidence_period"]
    out = pd.concat([manual[evidence_columns], derived[evidence_columns]], ignore_index=True).assign(evidence_period="")
    priority = out["source"].map({"manual": 0, "form345": 1}).fillna(2)
    out = (
        out.assign(_source_priority=priority)
        .sort_values(["symbol", "valid_from", "_source_priority", "issuer_cik"], kind="mergesort")
        .drop(columns="_source_priority")
        .reset_index(drop=True)
    )
    logger.info(
        "symbol_tenure: materialized %d manual and %d derived row(s) over %d symbol(s)",
        len(manual),
        len(derived),
        out["symbol"].nunique(),
    )
    return out[list(TABLE_COLUMNS)]


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

    raw_symbol = raw["ISSUERTRADINGSYMBOL"].astype("string")
    symbols_by_field = {field: parse_symbol_field(field) for field in raw_symbol.dropna().unique()}
    df = pd.DataFrame(
        {
            "symbol": raw_symbol.astype(object).map(symbols_by_field),
            "issuer_cik": raw["ISSUERCIK"],
            "issuer_name": (raw["ISSUERNAME"].astype("string") if "ISSUERNAME" in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string")),
            "filed": _parse_filing_dates(raw["FILING_DATE"]),
        }
    )
    drops["rows_read"] += len(df)
    placeholder = df["symbol"].isna()
    noise = ~placeholder & df["symbol"].map(len, na_action="ignore").eq(0)
    bad_symbol = placeholder | noise
    bad_cik = df["issuer_cik"].isin(("", "0" * 10))
    bad_date = df["filed"].isna()
    drops["empty_symbol"] += int(placeholder.sum())
    drops["noise_symbol"] += int(noise.sum())
    drops["empty_cik"] += int((bad_cik & ~bad_symbol).sum())
    drops["unparseable_filing_date"] += int((bad_date & ~bad_symbol & ~bad_cik).sum())
    df = df[~(bad_symbol | bad_cik | bad_date)]
    drops["rows_kept"] += len(df)
    if df.empty:
        return None
    drops["multi_symbol_rows"] += int(df["symbol"].map(len).gt(1).sum())
    df = df.explode("symbol", ignore_index=True)
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
    """Derive the `form345` and `manual` partitions of `symbol_tenure` (minus the manual config's cited rejections)
    and rewrite them unless unchanged.

    Only those partitions are deleted and saved, so rows of other sources survive; returns the materialized frame.
    """
    existing = context.store.load(Tables.symbol_tenure, project=True, where={"source": list(BUILD_SOURCES)}, optional=True)
    derived = reject_derived_tenure(derive_symbol_tenure(scan), load_rejected_symbol_tenure(config_dir or context.config_dir))
    manual = load_manual_symbol_tenure(config_dir or context.config_dir)
    out = materialize_symbol_tenure(derived, manual)
    context.log.info(
        f"symbol_tenure: validated {len(manual)} manual interval(s) for {manual['canonical_ticker'].nunique()} canonical ticker(s); no manual overlap"
    )
    if existing is None or existing.empty:
        context.log.info(f"symbol_tenure: cold build with {len(out)} row(s) over {out['symbol'].nunique()} symbol(s)")
    else:
        changed = changed_tenure_symbols(existing, out)
        context.log.info(f"symbol_tenure: {len(changed)} changed symbol(s): {', '.join(changed) if changed else 'none'}")
    unchanged = matches_stored(existing, out, Tables.symbol_tenure)
    written = 0
    if not unchanged:
        context.store.delete(Tables.symbol_tenure, where={"source": list(BUILD_SOURCES)})
        written = context.store.save(Tables.symbol_tenure, out)
    if unchanged:
        logger.info("symbol_tenure: unchanged (%d row(s)); write skipped", len(out))
    else:
        logger.info("symbol_tenure: wrote %d %s row(s)", written, "/".join(BUILD_SOURCES))
    return out


def aggregate_dei_symbols(facts: pd.DataFrame, period: str) -> pd.DataFrame:
    """One Notes zip's cover-page symbol facts `[adsh, cik, name, filed, value]` -> its `dei` rows.

    Symbols go through `parse_symbol_field` (placeholders and noise dropped, lists exploded); `n_filings`
    counts distinct accessions per (symbol, issuer_cik); `valid_to` is the last filing + 1 day.
    """
    fields = facts["value"].astype("string")
    symbols_by_field = {field: symbols for field in fields.dropna().unique() if (symbols := parse_symbol_field(field))}
    df = pd.DataFrame(
        {
            "adsh": facts["adsh"],
            "issuer_cik": pad_cik_series(facts["cik"]),
            "name": facts["name"].astype("string"),
            "filed": pd.to_datetime(facts["filed"], format="%Y%m%d", errors="coerce"),
            "symbol": fields.astype(object).map(symbols_by_field),
        }
    )
    df = df[df["symbol"].notna() & df["filed"].notna() & ~df["issuer_cik"].isin(("", "0" * 10))]
    df = df.explode("symbol").drop_duplicates(["symbol", "issuer_cik", "adsh"]).sort_values("filed", kind="mergesort")
    agg = df.groupby(["symbol", "issuer_cik"], as_index=False).agg(
        valid_from=("filed", "min"), last_filed=("filed", "max"), n_filings=("adsh", "size"), evidence=("name", "last")
    )
    out = agg.assign(
        valid_to=agg["last_filed"] + pd.Timedelta(days=1),
        n_filings=agg["n_filings"].astype("int64"),
        source=DEI_SOURCE,
        evidence_period=period,
        evidence=agg["evidence"].fillna("").astype(str).str.strip(),
    )
    return out[list(TABLE_COLUMNS)].sort_values(["symbol", "valid_from", "issuer_cik"], kind="mergesort", ignore_index=True)


def collapse_dei_periods(rows: pd.DataFrame) -> pd.DataFrame:
    """`dei` rows of several Notes periods -> one row per (symbol, issuer_cik).

    A monthly period (`YYYY_MM`) whose quarter (`YYYYqN`) is also present is dropped first, because the
    quarterly zip republishes the same accessions; bounds are then min/max and counts summed.
    """
    df = rows[rows["source"].eq(DEI_SOURCE)]
    periods = df["evidence_period"].astype("string")
    is_month = periods.str.contains("_", regex=False)
    month = pd.to_numeric(periods.str.slice(5).where(is_month), errors="coerce")
    quarter_of_month = periods.str.slice(0, 4) + "q" + ((month - 1) // 3 + 1).astype("Int64").astype("string")
    superseded = is_month & quarter_of_month.isin(set(periods[~is_month])).fillna(False)
    agg = (
        df[~superseded]
        .sort_values("valid_to", kind="mergesort")
        .groupby(["symbol", "issuer_cik"], as_index=False)
        .agg(valid_from=("valid_from", "min"), valid_to=("valid_to", "max"), n_filings=("n_filings", "sum"), evidence=("evidence", "last"))
    )
    return agg.assign(source=DEI_SOURCE)[["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence"]]


def save_dei_period(context: Context, period: str, rows: pd.DataFrame) -> int:
    """Replace the `dei` rows of one Notes period, leaving every other partition; an unchanged period is not rewritten."""
    where: dict[str, object] = {"source": DEI_SOURCE, "evidence_period": period}
    existing = context.store.load(Tables.symbol_tenure, project=True, where=where, optional=True)
    if matches_stored(existing, rows, Tables.symbol_tenure):
        return 0
    if existing is not None and not existing.empty:
        context.store.delete(Tables.symbol_tenure, where=where)
    return context.store.save(Tables.symbol_tenure, rows) if not rows.empty else 0
