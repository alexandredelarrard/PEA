"""Build-only symbol evidence (`symbol_tenure`): which issuer CIK typed symbol X over which dates.

Rows by `source`: `form345` (offline from `SUBMISSION.TSV` in the cached Form 345 quarter zips) and `manual` (the
evidenced config) are written by the identity build; `dei` (Notes cover-page symbols) one Notes zip period at a time
(`evidence.py`). Each source rewrites only its own partition. PK (symbol, issuer_cik, valid_from, source,
evidence_period); `evidence_period` is '' for `form345` and `manual`. Tenures may overlap: `valid_to` is the observed end
of that CIK's own filing window, so the table answers a membership question, never a single-answer lookup.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.common.identity_store import save_identity
from src.data_extract.utils.common.incremental import matches_stored
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker, pad_cik_series

logger = logging.getLogger(__name__)

#: SUBMISSION feeds `symbol_tenure`; joined to REPORTINGOWNER on accession it feeds the `universe_cik_symbol` owner sets.
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
_REJECTION_COLUMNS = ["symbol", "issuer_cik", "valid_from", "valid_to", "accessions", "evidence", "reason"]


class ManualSymbolTenureError(ValueError):
    """The manual symbol-tenure config is unsafe or cannot be audited."""


@dataclass(frozen=True)
class Form345Scan:
    """One pass over the cached Form 345 quarter zips: each readable quarter's per-(symbol, issuer_cik) aggregate, the
    distinct (issuer_cik, owner_cik_raw) reporting-owner pairs over every quarter, and the rows read, kept and dropped."""

    cache: Path
    quarters: tuple[pd.Period, ...]
    tenure_parts: tuple[pd.DataFrame, ...]
    owner_pairs: pd.DataFrame
    drops: Counter


# Symbol fields
def parse_symbol_field(value: object) -> tuple[str, ...] | None:
    """The market symbols one filer-typed symbol field names, in roster spelling; None for no symbol, () for noise.

    Quotes, brackets, qualifier words and venue suffixes go; `,` `;` `:` and a slash before a full symbol split a list.
    """
    # clean the field: quotes and brackets out, upper case, spaces around a slash and the state marker removed
    if not isinstance(value, str):
        return None
    text = _SYMBOL_BRACKETS.sub(" ", _SYMBOL_QUOTES.sub("", value)).strip().upper()
    text = _STATE_MARKER.sub("", _SLASH_SPACES.sub("/", " ".join(text.split())))
    if text in _PLACEHOLDER_SYMBOLS or not any(char.isalnum() for char in text):
        return None

    # split the list into items of words, dropping qualifier words from a multi-word field
    items = [[word.strip("-.") for word in item.split()] for item in _LIST_SEPARATORS.split(_SLASH_LIST.sub(",", text))]
    items = [[word for word in item if word] for item in items]
    if sum(len(item) for item in items) > 1:
        items = [[word for word in item if word not in _QUALIFIER_WORDS] for item in items]

    # each item as one symbol: spaced letters join (`N O G`), a base plus short class words hyphenate (`HBC PR A`)
    symbols: list[str | None] = []
    for words in ([_VENUE_SUFFIX.sub("", word) for word in item] for item in items if item):
        spaced = len(words) > 1 and all(len(word) == 1 for word in words)
        classed = len(words) == 1 or (len(words[0]) >= 2 and all(len(word) <= 2 for word in words[1:]))
        symbols.append("".join(words) if spaced else "-".join(words).replace("/", "-").replace(".", "-") if classed else None)

    # noise unless every item is a market symbol
    if not symbols or not all(symbol and _SYMBOL_SHAPE.fullmatch(symbol) for symbol in symbols):
        return ()
    return tuple(dict.fromkeys(str(symbol) for symbol in symbols))


# Form 3/4/5 scan
def scan_form345_cache(cache: Path) -> Form345Scan:
    """Read every cached Form 345 quarter zip once for both identity tables (a corrupt zip is skipped and kept)."""
    # the cached quarter zips; too few warns that the history is partial
    zips = sorted(p for p in cache.glob("*.zip") if _quarter_of(p) is not None)
    if not zips:
        raise FileNotFoundError(
            f"symbol_tenure: no Form 345 quarter zips under {cache}. The derivation is offline and reads only the cache; run the `insider` extract first."
        )
    if len(zips) < MIN_EXPECTED_QUARTERS:
        logger.warning(
            "symbol_tenure: only %d cached quarter zip(s) under %s (expected >= %d). The derived table will be a PARTIAL history "
            "that LOOKS complete -- every tenure ending inside an absent quarter is wrong.",
            len(zips),
            cache,
            MIN_EXPECTED_QUARTERS,
        )

    # one read per quarter: the issuer-owner pairs, then the per-(symbol, CIK) aggregate
    drops: Counter = Counter()
    tenure_parts: list[pd.DataFrame | None] = []
    owner_parts: list[pd.DataFrame] = [_owner_pairs(pd.DataFrame(), pd.DataFrame())]
    for path in zips:
        tables = read_zip_tables(path, _FORM345_READ, on_corrupt="skip", log=logger)
        skipped = _unreadable_quarter(tables, path.name)
        if skipped is not None:
            drops[skipped] += 1
            continue
        submission = tables[SUBMISSION_MEMBER]
        if "ISSUERCIK" in submission.columns:
            submission["ISSUERCIK"] = pad_cik_series(submission["ISSUERCIK"])
        fields = submission["ISSUERTRADINGSYMBOL"].astype("string").dropna().unique() if "ISSUERTRADINGSYMBOL" in submission else []
        filed = _parse_filing_dates(submission["FILING_DATE"]) if "FILING_DATE" in submission else None
        tenure_parts.append(_aggregate_submission(submission, {field: parse_symbol_field(field) for field in fields}, filed, path.name, drops))
        owner_parts.append(_owner_pairs(submission, tables[OWNER_MEMBER]))

    # the scan
    return Form345Scan(
        cache=cache,
        quarters=tuple(q for q in (_quarter_of(p) for p in zips) if q is not None),
        tenure_parts=tuple(part for part in tenure_parts if part is not None),
        owner_pairs=pd.concat(owner_parts, ignore_index=True).drop_duplicates(ignore_index=True),
        drops=drops,
    )


def _quarter_of(path: Path) -> pd.Period | None:
    """`2026q1.zip` -> that quarterly Period, or None when the name is not a quarter."""
    try:
        return pd.Period(path.stem.upper(), freq="Q")
    except Exception:  # noqa: BLE001 -- an unexpected file name, not a bug
        return None


def _unreadable_quarter(tables: dict[str, pd.DataFrame] | None, name: str) -> str | None:
    """The drop reason of a corrupt zip or one without `SUBMISSION.TSV`, else None."""
    if tables is None:
        return "corrupt_zip"
    if not tables:
        logger.warning("symbol_tenure: %s has no %s -> SKIPPED", name, SUBMISSION_MEMBER)
        return "no_submission_member"
    return None


def _parse_filing_dates(raw: pd.Series) -> pd.Series:
    """`FILING_DATE` -> datetime64: `DD-MON-YYYY` decoded explicitly, the rest as ISO (never `format="mixed"`)."""
    # the month-name subset only, so junk rows elsewhere cannot make a whole-column parse raise
    text = raw.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    parts = text.str.split("-", n=2, expand=True)
    is_month_name = parts[1].isin(_MONTHS).fillna(False) if parts.shape[1] == 3 else pd.Series(False, index=raw.index)
    if is_month_name.any():
        out.loc[is_month_name] = pd.to_datetime(text[is_month_name], format="%d-%b-%Y", errors="coerce")

    # every other field as an ISO date
    todo = out.isna() & text.notna()
    if todo.any():
        out.loc[todo] = pd.to_datetime(text[todo].str.slice(0, 10), format="ISO8601", errors="coerce")
    return out


def _aggregate_submission(
    raw: pd.DataFrame, symbols_by_field: dict[str, tuple[str, ...] | None], filed: pd.Series | None, name: str, drops: Counter
) -> pd.DataFrame | None:
    """Per-(symbol, cik) first/last filing date, count and issuer name of one quarter; None when nothing is usable."""
    # a quarter without the required columns is skipped loudly
    missing = _REQUIRED_COLUMNS - set(raw.columns)
    if missing or filed is None:
        logger.warning("symbol_tenure: %s lacks %s -> SKIPPED", name, sorted(missing))
        drops["missing_columns"] += 1
        return None

    # the parsed symbols, CIK, issuer name and filing date of every row
    names = raw["ISSUERNAME"].astype("string") if "ISSUERNAME" in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string")
    symbols = raw["ISSUERTRADINGSYMBOL"].astype("string").astype(object).map(symbols_by_field)
    df = pd.DataFrame({"symbol": symbols, "issuer_cik": raw["ISSUERCIK"], "issuer_name": names, "filed": filed})

    # drop and count each unusable row by its first reason
    drops["rows_read"] += len(df)
    placeholder = df["symbol"].isna()
    noise = ~placeholder & df["symbol"].map(len, na_action="ignore").eq(0)
    bad_symbol, bad_cik, bad_date = placeholder | noise, df["issuer_cik"].isin(("", "0" * 10)), df["filed"].isna()
    drops["empty_symbol"] += int(placeholder.sum())
    drops["noise_symbol"] += int(noise.sum())
    drops["empty_cik"] += int((bad_cik & ~bad_symbol).sum())
    drops["unparseable_filing_date"] += int((bad_date & ~bad_symbol & ~bad_cik).sum())
    df = df[~(bad_symbol | bad_cik | bad_date)]
    drops["rows_kept"] += len(df)
    if df.empty:
        return None

    # one row per (symbol, CIK), a multi-symbol field exploded
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


def derive_symbol_tenure(scan: Form345Scan) -> pd.DataFrame:
    """(symbol, issuer_cik) -> (valid_from, valid_to, n_filings) from one scan, sorted; `valid_to` is the last filing + 1
    day, NULL ("no end observed") when the last filing falls in the latest cached quarter."""
    # every quarter's aggregate summed per (symbol, CIK)
    if not scan.tenure_parts:
        raise ValueError(f"symbol_tenure: every zip under {scan.cache} was unreadable or empty")
    latest_quarter = max(scan.quarters)
    agg = (
        pd.concat(scan.tenure_parts, ignore_index=True)
        .groupby(["symbol", "issuer_cik"], as_index=False, sort=False)
        .agg(valid_from=("first_filed", "min"), last_filed=("last_filed", "max"), n_filings=("n_filings", "sum"), issuer_name=("issuer_name", "last"))
    )

    # half-open rows, open when still filed in the latest quarter
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

    # one INFO summary of the scan and its drops
    per_symbol = out.groupby("symbol")["issuer_cik"].nunique()
    dropped = ", ".join(f"{k}={v}" for k, v in sorted(scan.drops.items()) if k not in {"rows_read", "rows_kept"}) or "nothing"
    logger.info(
        "symbol_tenure: %d quarter(s) %s..%s -> %d row(s) over %d symbol(s); %d symbol(s) had >1 issuer CIK; %d tenure(s) still open. "
        "Read %d submission row(s), kept %d; dropped %s",
        len(scan.quarters),
        min(scan.quarters),
        latest_quarter,
        len(out),
        int(per_symbol.size),
        int((per_symbol > 1).sum()),
        int(out["valid_to"].isna().sum()),
        scan.drops["rows_read"],
        scan.drops["rows_kept"],
        dropped,
    )
    return out


# Manual config
def load_manual_symbol_tenure(config_dir: str | Path) -> pd.DataFrame:
    """The evidenced manual ticker intervals, normalised and validated (keeping the `canonical_ticker` and `reason` audit
    columns); overlapping intervals raise."""
    # read the config
    path = Path(config_dir) / MANUAL_TENURE_FILE
    if not path.exists():
        raise FileNotFoundError(f"symbol_tenure: manual config not found: {path}")
    tickers = _manual_payload(path).get("tickers")
    if not isinstance(tickers, dict) or not tickers:
        raise ManualSymbolTenureError(f"symbol_tenure: {path}.tickers must be a non-empty object")

    # one validated record per interval, its dates checked apart
    records: list[dict[str, object]] = []
    for raw_ticker, raw_intervals in tickers.items():
        ticker = normalise_ticker(raw_ticker)
        if not ticker or not isinstance(raw_intervals, list) or not raw_intervals:
            raise ManualSymbolTenureError(f"symbol_tenure: {ticker or path} must name a canonical ticker with at least one interval")
        locations = [f"tickers.{ticker}[{index}]" for index in range(len(raw_intervals))]
        records += [_manual_record(raw, ticker, at) | _manual_dates(raw, at) for raw, at in zip(raw_intervals, locations, strict=True)]

    # deduplicated, sorted and free of overlaps
    out = pd.DataFrame.from_records(records)
    identity_columns = ["canonical_ticker", "symbol", "issuer_cik", "valid_from", "valid_to"]
    out = out.drop_duplicates(identity_columns, keep="first").sort_values(["symbol", "valid_from", "issuer_cik"], kind="mergesort", ignore_index=True)
    _check_manual_overlaps(out)
    return out


def load_rejected_symbol_tenure(config_dir: str | Path) -> pd.DataFrame:
    """The manual config's `rejected` derived tenures (symbol, issuer CIK, dates, cited `accessions`, evidence, reason);
    empty when the file or the list is absent; an uncited entry raises."""
    # read the config's `rejected` list
    path = Path(config_dir) / MANUAL_TENURE_FILE
    raw_entries = _manual_payload(path).get("rejected", []) if path.exists() else []
    if not isinstance(raw_entries, list):
        raise ManualSymbolTenureError(f"symbol_tenure: {path}.rejected must be a list")

    # one validated record per rejection: a closed dated interval and the accessions that mis-tag it
    fields = ("symbol", "issuer_cik", "valid_from", "valid_to", "evidence", "reason")
    records = []
    for index, raw in enumerate(raw_entries):
        at = f"rejected[{index}]"
        entry = {key: raw.get(key) for key in fields} if isinstance(raw, dict) else raw
        record = _manual_record(entry, "-", at) | _manual_dates(entry, at)
        records.append({key: record[key] for key in fields} | {"accessions": _rejection_accessions(raw, record, at)})
    return pd.DataFrame.from_records(records, columns=_REJECTION_COLUMNS)


def _manual_payload(path: Path) -> dict:
    """The manual config as JSON, version-checked."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManualSymbolTenureError(f"symbol_tenure: invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != MANUAL_TENURE_VERSION:
        raise ManualSymbolTenureError(f"symbol_tenure: {path} must be an object with version={MANUAL_TENURE_VERSION}")
    return payload


def _manual_record(raw: object, ticker: str, location: str) -> dict[str, object]:
    """One manual interval's symbol, CIK, evidence and reason as `symbol_tenure` columns plus its audit columns."""
    # the entry, its symbol and its padded CIK
    if not isinstance(raw, dict):
        raise ManualSymbolTenureError(f"{location} must be an object")
    symbol = str(raw.get("symbol", "")).strip().upper()
    if not symbol:
        raise ManualSymbolTenureError(f"{location}.symbol must be non-empty")
    cik = raw.get("issuer_cik")
    if not isinstance(cik, str) or len(cik) != 10 or not cik.isdigit():
        raise ManualSymbolTenureError(f"{location}.issuer_cik must be a zero-padded 10-digit string")

    # the evidence list and the reason
    evidence = raw.get("evidence")
    if not isinstance(evidence, list) or not evidence or any(not isinstance(item, str) or not item.strip() for item in evidence):
        raise ManualSymbolTenureError(f"{location}.evidence must be a non-empty list of non-empty strings")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ManualSymbolTenureError(f"{location}.reason must be non-empty")
    joined = " | ".join(item.strip() for item in evidence)
    return {
        "canonical_ticker": ticker,
        "symbol": symbol,
        "issuer_cik": cik,
        "n_filings": 0,
        "source": "manual",
        "evidence": joined,
        "reason": reason.strip(),
    }


def _manual_dates(raw: object, location: str) -> dict[str, pd.Timestamp | None]:
    """The strict ISO `valid_from` and `valid_to` (None: open) of one manual interval, with `valid_from < valid_to`."""
    # each date a canonical YYYY-MM-DD string; only `valid_to` may be null
    out: dict[str, pd.Timestamp | None] = {"valid_from": None, "valid_to": None}
    for field in ("valid_from", "valid_to"):
        value = raw.get(field) if isinstance(raw, dict) else None
        if value is None and field == "valid_to":
            continue
        if not isinstance(value, str):
            raise ManualSymbolTenureError(f"{location}.{field} must be an ISO YYYY-MM-DD string")
        parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
        if pd.isna(parsed) or value != parsed.date().isoformat():
            raise ManualSymbolTenureError(f"{location}.{field} must use canonical YYYY-MM-DD form: {value!r}")
        out[field] = pd.Timestamp(parsed)

    # a closed interval must end after it starts
    if out["valid_to"] is not None and out["valid_from"] is not None and out["valid_from"] >= out["valid_to"]:
        raise ManualSymbolTenureError(f"{location} must satisfy valid_from < valid_to")
    return out


def _rejection_accessions(raw: dict, record: dict[str, object], location: str) -> tuple[str, ...]:
    """The SEC accessions a rejection cites; its interval must be closed."""
    if record["valid_to"] is None:
        raise ManualSymbolTenureError(f"{location}.valid_to must close the rejected interval")
    accessions = raw.get("accessions")
    if not isinstance(accessions, list) or not accessions or any(not isinstance(a, str) or not _ACCESSION.fullmatch(a) for a in accessions):
        raise ManualSymbolTenureError(f"{location}.accessions must be a non-empty list of SEC accession numbers")
    return tuple(accessions)


def _check_manual_overlaps(manual: pd.DataFrame) -> None:
    """Raise on the first manual interval that starts before the previous one of its symbol ends."""
    # each interval against the previous one of its symbol
    df_ordered = manual.sort_values(["symbol", "valid_from", "valid_to", "issuer_cik"], kind="mergesort", ignore_index=True)
    df_previous = df_ordered.shift(1)
    same_symbol = df_ordered["symbol"].eq(df_previous["symbol"])
    overlaps = same_symbol & (df_previous["valid_to"].isna() | (df_ordered["valid_from"] < df_previous["valid_to"]))
    if not overlaps.any():
        return

    # name the first overlap
    first = int(overlaps.to_numpy().argmax())
    row, previous = df_ordered.iloc[first], df_previous.iloc[first]
    previous_end = "open" if pd.isna(previous["valid_to"]) else pd.Timestamp(previous["valid_to"]).date()
    row_end = "open" if pd.isna(row["valid_to"]) else pd.Timestamp(row["valid_to"]).date()
    raise ManualSymbolTenureError(
        f"symbol_tenure: overlapping manual intervals for {row['symbol']}: {previous['canonical_ticker']}/{previous['issuer_cik']} "
        f"[{pd.Timestamp(previous['valid_from']).date()}, {previous_end}) and "
        f"{row['canonical_ticker']}/{row['issuer_cik']} [{pd.Timestamp(row['valid_from']).date()}, {row_end})"
    )


# The form345 / manual partitions
def build_symbol_tenure(context: Context, scan: Form345Scan, manual: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    """Derive the `form345` partition (minus the cited rejections) and the `manual` one, and rewrite both unless unchanged.

    Only those partitions are deleted and saved, so rows of other sources survive; returns the materialized frame.
    """
    # load the stored partitions
    existing = context.store.load(Tables.symbol_tenure, project=True, where={"source": list(BUILD_SOURCES)}, optional=True)

    # compute the two partitions
    out = materialize_symbol_tenure(reject_derived_tenure(derive_symbol_tenure(scan), rejected), manual)
    changed = changed_tenure_symbols(existing, out) if existing is not None and not existing.empty else None

    # log what changed, then save unless unchanged
    context.log.info(
        f"symbol_tenure: validated {len(manual)} manual interval(s) for {manual['canonical_ticker'].nunique()} canonical ticker(s); no manual overlap"
    )
    if changed is None:
        context.log.info(f"symbol_tenure: cold build with {len(out)} row(s) over {out['symbol'].nunique()} symbol(s)")
    else:
        context.log.info(f"symbol_tenure: {len(changed)} changed symbol(s): {', '.join(changed) if changed else 'none'}")
    unchanged = matches_stored(existing, out, Tables.symbol_tenure)
    written = 0 if unchanged else save_identity(context.store, Tables.symbol_tenure, out, {"source": list(BUILD_SOURCES)})
    logger.info(
        "symbol_tenure: unchanged (%d row(s)); write skipped" if unchanged else "symbol_tenure: wrote %d form345/manual row(s)",
        len(out) if unchanged else written,
    )
    return out


def reject_derived_tenure(derived: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    """`derived` without its rows a rejection covers (same symbol and CIK, interval inside the rejected dates); a rejection
    matching the pair but no longer covering its interval keeps the row and warns."""
    # nothing to reject
    if rejected.empty or derived.empty:
        return derived

    # each rejection against the derived rows of its (symbol, CIK)
    drop = pd.Series(False, index=derived.index)
    starts, ends = pd.to_datetime(derived["valid_from"]), pd.to_datetime(derived["valid_to"])
    bounds = zip(
        rejected["symbol"],
        rejected["issuer_cik"],
        pd.to_datetime(rejected["valid_from"]),
        pd.to_datetime(rejected["valid_to"]),
        rejected["accessions"],
        strict=True,
    )
    for symbol, cik, start, end, accessions in bounds:
        pair = derived["symbol"].astype(str).eq(symbol) & derived["issuer_cik"].astype(str).eq(cik)
        covered = pair & (starts >= start) & ends.notna() & (ends <= end)
        label = f"{symbol}/{cik} {start.date()}..{end.date()}"
        if (pair & ~covered).any():
            logger.warning("symbol_tenure: the rejection of %s no longer covers the derived interval; kept for review", label)
        if covered.any():
            logger.info("symbol_tenure: rejected %s (%d accession(s) cited)", label, len(accessions))
        drop |= covered
    return derived[~drop].reset_index(drop=True)


def materialize_symbol_tenure(derived: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """The `form345` and `manual` partitions as table rows: manual ordered first, every row kept."""
    # both partitions, manual before form345 on a tie
    evidence_columns = [column for column in TABLE_COLUMNS if column != "evidence_period"]
    out = pd.concat([manual[evidence_columns], derived[evidence_columns]], ignore_index=True).assign(evidence_period="")
    priority = out["source"].map({"manual": 0, "form345": 1}).fillna(2)
    out = out.assign(_priority=priority).sort_values(["symbol", "valid_from", "_priority", "issuer_cik"], kind="mergesort").drop(columns="_priority")

    # one INFO line
    logger.info("symbol_tenure: materialized %d manual and %d derived row(s) over %d symbol(s)", len(manual), len(derived), out["symbol"].nunique())
    return out.reset_index(drop=True)[list(TABLE_COLUMNS)]


def changed_tenure_symbols(existing: pd.DataFrame, derived: pd.DataFrame) -> list[str]:
    """Sorted symbols whose CIK membership or observed bounds changed."""
    # the compared columns, `source` when both sides carry it
    columns = ["symbol", "issuer_cik", "valid_from", "valid_to"] + (
        ["source"] if "source" in existing.columns and "source" in derived.columns else []
    )

    # one sorted signature per symbol on each side
    signatures = []
    for frame in (existing, derived):
        normal = frame[columns].assign(symbol=frame["symbol"].astype(str).str.strip().str.upper())
        normal = normal.assign(**{column: pd.to_datetime(normal[column], errors="coerce").astype("string") for column in ("valid_from", "valid_to")})
        rows = normal[columns[1:]].astype(str).apply(tuple, axis=1) if not normal.empty else pd.Series(dtype=object)
        signatures.append({str(symbol): tuple(sorted(group)) for symbol, group in rows.groupby(normal["symbol"], sort=False)})

    # the symbols whose signature differs
    before, after = signatures
    return sorted(symbol for symbol in set(before) | set(after) if before.get(symbol) != after.get(symbol))


def load_built_tenure(context: Context) -> pd.DataFrame:
    """The stored `form345` and `manual` partitions (what `build_symbol_tenure` returns), dates parsed; empty when absent."""
    stored = context.store.load(Tables.symbol_tenure, columns=list(TABLE_COLUMNS), where={"source": list(BUILD_SOURCES)}, optional=True)
    out = stored if stored is not None else pd.DataFrame(columns=list(TABLE_COLUMNS))
    return out.assign(valid_from=pd.to_datetime(out["valid_from"]), valid_to=pd.to_datetime(out["valid_to"]))


# The dei partition
def aggregate_dei_symbols(facts: pd.DataFrame, symbols_by_field: dict[str, tuple[str, ...] | None], period: str) -> pd.DataFrame:
    """One Notes zip's cover-page symbol facts `[adsh, cik (padded), name, filed, value]` -> its `dei` rows, each symbol
    field parsed once into `symbols_by_field`; `n_filings` counts distinct accessions; `valid_to` is the last filing + 1 day."""
    # the parsed symbols of every fact; placeholders, noise and unusable CIKs dropped
    parsed = {field: symbols for field, symbols in symbols_by_field.items() if symbols}
    df = pd.DataFrame(
        {
            "adsh": facts["adsh"],
            "issuer_cik": facts["cik"],
            "name": facts["name"].astype("string"),
            "filed": pd.to_datetime(facts["filed"], format="%Y%m%d", errors="coerce"),
            "symbol": facts["value"].astype("string").astype(object).map(parsed),
        }
    )
    df = df[df["symbol"].notna() & df["filed"].notna() & ~df["issuer_cik"].isin(("", "0" * 10))]

    # one row per (symbol, CIK)
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
    """`dei` rows of several Notes periods -> one row per (symbol, issuer_cik); a monthly period (`YYYY_MM`) whose quarter
    (`YYYYqN`) is present is dropped first (the quarterly zip republishes it), then bounds are min/max and counts summed."""
    # drop the monthly periods a quarterly zip supersedes
    df = rows[rows["source"].eq(DEI_SOURCE)]
    periods = df["evidence_period"].astype("string")
    is_month = periods.str.contains("_", regex=False)
    month = pd.to_numeric(periods.str.slice(5).where(is_month), errors="coerce")
    quarter_of_month = periods.str.slice(0, 4) + "q" + ((month - 1) // 3 + 1).astype("Int64").astype("string")
    superseded = is_month & quarter_of_month.isin(set(periods[~is_month])).fillna(False)

    # one row per (symbol, CIK)
    agg = (
        df[~superseded]
        .sort_values("valid_to", kind="mergesort")
        .groupby(["symbol", "issuer_cik"], as_index=False)
        .agg(valid_from=("valid_from", "min"), valid_to=("valid_to", "max"), n_filings=("n_filings", "sum"), evidence=("evidence", "last"))
    )
    return agg.assign(source=DEI_SOURCE)[["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings", "source", "evidence"]]


def captured_dei_periods(context: Context) -> set[str]:
    """The Notes periods whose cover-page symbols are stored in the `dei` partition."""
    return {str(value) for value in context.store.distinct(Tables.symbol_tenure, "evidence_period", where={"source": DEI_SOURCE})}
