"""
symbol_tenure.py (src/data_extract/utils/common/symbol_tenure.py)
--------------------------------------------------------------------------------------------
AXIS B OF TICKER IDENTITY: WHICH ISSUER CIK HELD SYMBOL `X` ON DATE `d`.

A trading symbol is a LEASE, not a name. `IR` was Ingersoll-Rand Co Ltd, then Ingersoll-Rand
plc, then Ingersoll Rand Inc; `COR` was Cortex Pharmaceuticals, then CoreSite Realty, then
Cencora. Any pipeline that resolves a filing to a ticker by matching the filer's own typed
`ISSUERTRADINGSYMBOL` against TODAY's universe therefore imports one company's rows under
another company's name -- measured at 38,910 rows (1.92%) of `insider_transactions`.

⚠ THE SEC PUBLISHES NO HISTORICAL TICKER->CIK DATASET. `company_tickers.json` is a current
snapshot with no dates and `data.sec.gov/submissions` carries `formerNames` but no
`formerTickers` (Meta's CIK 1326801 returns only `META`; the string `FB` appears nowhere).
The fact exists only INSIDE the filings, so this table is DERIVED, not fetched -- from the
`SUBMISSION.TSV` member of the cached Form 345 quarterly zips, which is primary source, free
and entirely offline.

⚠ TENURES OVERLAP. Two issuers legitimately file under the same symbol in the same year --
one is still typing the symbol it lost, or the two sit on different exchanges. `valid_to` is
the OBSERVED end of THAT CIK's own filing window, never the start of the next CIK's, so the
table answers a MEMBERSHIP question ("did this CIK hold X at d") and never a lookup expecting
one answer. Collapsing overlaps onto a single winner would silently rewrite history.

⚠ THIS TABLE IS NOT ON THE `owns()` HOT PATH. The insider screen compares ENTITIES
(`entity_lineage`, axis A) and never reads tenure, which is why the 2006q1 coverage floor
costs nothing. Tenure is the DISCOVERY substrate that produces the candidate list axis A
adjudicates, the third opinion in the roster-CIK cross-check, and the only resolver available
to the tables that carry a symbol and a date but no CIK at all (fails-to-deliver, short
interest).
"""
from __future__ import annotations

import logging
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.run_manifest import record_run
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

#: The one zip member this derivation needs. The other four (REPORTINGOWNER, NONDERIV_TRANS,
#: DERIV_TRANS, FOOTNOTES) are ~20x larger and carry nothing about symbol tenure.
SUBMISSION_MEMBER = "SUBMISSION.TSV"

#: The `SUBMISSION.TSV` columns this derivation reads. `ISSUERNAME` is optional and feeds
#: `evidence`; the other three are required and a zip missing any of them is skipped loudly.
_WANTED_COLUMNS = frozenset({"ISSUERCIK", "ISSUERTRADINGSYMBOL", "ISSUERNAME", "FILING_DATE"})
_REQUIRED_COLUMNS = frozenset({"ISSUERCIK", "ISSUERTRADINGSYMBOL", "FILING_DATE"})

#: `ISSUERTRADINGSYMBOL` strings that mean "no symbol". Filers type all of these, and they are
#: NOT a long tail: dropping them removes 194 (symbol, cik) pairs over 3 pseudo-symbols, each
#: of which would otherwise read as a heavily-reused ticker held by dozens of unrelated CIKs.
#: `NA`, `N/A` and `NULL` never reach here as strings -- pandas' default NA handling has
#: already turned them into NaN -- but they stay listed so the rule is readable in one place.
_NULL_SYMBOLS = frozenset({"", "NONE", "N/A", "NA", "-", "--", "N.A.", "NULL"})

#: A double quote is never part of a ticker, so it is stripped rather than kept as the filer's
#: string. It is not cosmetic: 53 (symbol, cik) pairs arrive quoted, every one of them under
#: the SAME CIK as the unquoted symbol, so leaving them split SHORTENS a real tenure. `WM` is
#: the case that matters -- Washington Mutual's 325 filings of 2006-01..2008-09 are all typed
#: `"WM"`, so without this the table says WM was Washington Mutual's only from 2008-04, and a
#: pre-2008 `WM` filing looks like it belongs to nobody. No other normalisation is applied:
#: `(SPAR)`, `NYSE:DRH` and `AMEX FVE` stay as typed, visible as low-`n_filings` tenures.
_SYMBOL_NOISE_CHARS = '"'

#: `DD-MON-YYYY` is the shape the SEC ships in most quarters; a minority are ISO. Both are
#: parsed, because a silent parse failure here is a silent tenure gap.
_MONTHS = {month: i + 1 for i, month in
           enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}

#: Below this many cached quarters the derivation is a PARTIAL history that looks complete.
#: 2006q1 -> 2026q1 is 81; the guard warns rather than raises so a deliberate subset still runs.
MIN_EXPECTED_QUARTERS = 80


def _parse_filing_dates(raw: pd.Series) -> pd.Series:
    """`FILING_DATE` -> datetime64, accepting BOTH shapes the SEC ships.

    `03-MAR-2006` and `2006-03-03` both appear across the 81 quarters. `pd.to_datetime` with
    `format="mixed"` guesses per element and would read `01-02-2006` as a day-first or a
    month-first date depending on its neighbours, so the `DD-MON-YYYY` shape is decoded
    explicitly and only what it cannot claim falls through to ISO.
    """
    text = raw.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    parts = text.str.split("-", n=2, expand=True)
    if parts.shape[1] == 3:
        # Decoded on the SUBSET, not masked over the whole column: a row that is not this
        # shape has no year to convert, and building the (year, month, day) frame over all
        # rows raises `cannot convert NA to integer` on the first junk date in the quarter.
        is_month_name = parts[1].isin(_MONTHS).fillna(False)
        if is_month_name.any():
            out.loc[is_month_name] = pd.to_datetime(
                text[is_month_name], format="%d-%b-%Y", errors="coerce")
    todo = out.isna() & text.notna()
    if todo.any():
        out.loc[todo] = pd.to_datetime(text[todo].str.slice(0, 10), format="ISO8601",
                                       errors="coerce")
    return out


def _quarter_of(path: Path) -> pd.Period | None:
    """`2026q1.zip` -> that quarterly Period, or None when the name is not a quarter."""
    try:
        return pd.Period(path.stem.upper(), freq="Q")
    except Exception:                      # noqa: BLE001 -- an unexpected file name, not a bug
        return None


def _aggregate_zip(path: Path, drops: Counter) -> pd.DataFrame | None:
    """Per-(symbol, cik) first/last filing date, filing count and issuer name for ONE zip.

    Aggregating inside the loop rather than concatenating 4.3M raw rows keeps the whole
    derivation inside a few hundred MB.
    """
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        logger.warning("symbol_tenure: %s is a corrupt zip -> SKIPPED, so its quarter is "
                       "absent from the derivation", path.name)
        drops["corrupt_zip"] += 1
        return None
    with archive:
        names = {n.upper(): n for n in archive.namelist()}
        if SUBMISSION_MEMBER not in names:
            logger.warning("symbol_tenure: %s has no %s -> SKIPPED", path.name,
                           SUBMISSION_MEMBER)
            drops["no_submission_member"] += 1
            return None
        with archive.open(names[SUBMISSION_MEMBER]) as handle:
            raw = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False,
                              usecols=lambda c: c.upper() in _WANTED_COLUMNS)
    raw.columns = [c.upper() for c in raw.columns]
    missing = _REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        logger.warning("symbol_tenure: %s lacks %s -> SKIPPED", path.name, sorted(missing))
        drops["missing_columns"] += 1
        return None

    df = pd.DataFrame({
        "symbol": (raw["ISSUERTRADINGSYMBOL"].astype("string")
                   .str.replace(_SYMBOL_NOISE_CHARS, "", regex=False).str.strip().str.upper()),
        "issuer_cik": raw["ISSUERCIK"].astype("string").str.strip().str.zfill(10),
        "issuer_name": (raw["ISSUERNAME"].astype("string") if "ISSUERNAME" in raw.columns
                        else pd.Series(pd.NA, index=raw.index, dtype="string")),
        "filed": _parse_filing_dates(raw["FILING_DATE"]),
    })
    drops["rows_read"] += len(df)
    bad_symbol = df["symbol"].isna() | df["symbol"].isin(_NULL_SYMBOLS)
    bad_cik = df["issuer_cik"].isna() | df["issuer_cik"].eq("0" * 10)
    bad_date = df["filed"].isna()
    drops["empty_symbol"] += int(bad_symbol.sum())
    drops["empty_cik"] += int((bad_cik & ~bad_symbol).sum())
    drops["unparseable_filing_date"] += int((bad_date & ~bad_symbol & ~bad_cik).sum())
    df = df[~(bad_symbol | bad_cik | bad_date)]
    drops["rows_kept"] += len(df)
    if df.empty:
        return None
    return (df.groupby(["symbol", "issuer_cik"], as_index=False, sort=False)
              .agg(first_filed=("filed", "min"), last_filed=("filed", "max"),
                   n_filings=("filed", "size"), issuer_name=("issuer_name", "last")))


def derive_symbol_tenure(cache: Path) -> pd.DataFrame:
    """(symbol, issuer_cik) -> (valid_from, valid_to, n_filings) from the cached Form 345 zips.

    Reads `SUBMISSION.TSV` only. Deterministic: the same cache directory always yields the
    same frame, sorted on (symbol, valid_from, issuer_cik).

    BOUNDARY SEMANTICS ARE THE REGISTER'S, EXACTLY (`registrant.Segment.covers`):
    `valid_from <= d < valid_to`, half-open. `valid_to` is therefore `last_filed + 1 day` and
    not `last_filed`, which would put a real filing on the excluded side of its own interval.
    `valid_to` is NULL when `last_filed` falls inside the most recent cached quarter, because
    no end has been OBSERVED; NULL means "still open", never "forever".
    """
    zips = sorted(p for p in cache.glob("*.zip") if _quarter_of(p) is not None)
    if not zips:
        raise FileNotFoundError(
            f"symbol_tenure: no Form 345 quarter zips under {cache}. The derivation is "
            "offline and reads only the cache; run the `insider` extract first.")
    if len(zips) < MIN_EXPECTED_QUARTERS:
        logger.warning(
            "symbol_tenure: only %d cached quarter zip(s) under %s (expected >= %d). The "
            "derived table will be a PARTIAL history that LOOKS complete -- every tenure "
            "ending inside an absent quarter is wrong.", len(zips), cache,
            MIN_EXPECTED_QUARTERS)

    quarters = [q for q in (_quarter_of(p) for p in zips) if q is not None]
    latest_quarter = max(quarters)
    drops: Counter = Counter()
    frames = [frame for frame in (_aggregate_zip(p, drops) for p in zips) if frame is not None]
    if not frames:
        raise ValueError(f"symbol_tenure: every zip under {cache} was unreadable or empty")

    agg = (pd.concat(frames, ignore_index=True)
             .groupby(["symbol", "issuer_cik"], as_index=False, sort=False)
             .agg(valid_from=("first_filed", "min"), last_filed=("last_filed", "max"),
                  n_filings=("n_filings", "sum"), issuer_name=("issuer_name", "last")))

    still_open = agg["last_filed"] >= latest_quarter.start_time
    out = pd.DataFrame({
        "symbol": agg["symbol"].astype(str),
        "issuer_cik": agg["issuer_cik"].astype(str),
        "valid_from": agg["valid_from"],
        "valid_to": (agg["last_filed"] + pd.Timedelta(days=1)).mask(still_open),
        "n_filings": agg["n_filings"].astype("int64"),
        "source": "form345",
        "evidence": agg["issuer_name"].fillna("").astype(str).str.strip(),
    }).sort_values(["symbol", "valid_from", "issuer_cik"], kind="mergesort", ignore_index=True)

    per_symbol = out.groupby("symbol")["issuer_cik"].nunique()
    logger.info(
        "symbol_tenure: %d quarter(s) %s..%s -> %d row(s) over %d symbol(s); %d symbol(s) "
        "had >1 issuer CIK; %d tenure(s) still open. Read %d submission row(s), kept %d; "
        "dropped %s", len(zips), min(quarters), latest_quarter, len(out),
        int(per_symbol.size), int((per_symbol > 1).sum()), int(out["valid_to"].isna().sum()),
        drops["rows_read"], drops["rows_kept"],
        ", ".join(f"{k}={v}" for k, v in sorted(drops.items())
                  if k not in {"rows_read", "rows_kept"}) or "nothing")
    return out


def build_symbol_tenure(context: Context, cache: Path) -> pd.DataFrame:
    """Derive and REPLACE `symbol_tenure`; returns the frame written.

    `replace`, never `save`: the table is a full derivation of the cache, and an upsert would
    leave rows from an earlier, narrower run behind with nothing to tell them from current ones.
    """
    out = derive_symbol_tenure(cache)
    written = context.store.replace(Tables.symbol_tenure, out)
    # `ticker_count=0`: this is a market-wide derivation over every EDGAR symbol, not a
    # per-ticker walk -- the convention `fetch_sharadar_tickers` already uses. Always a full
    # rescan: there is no incremental path, the whole cache is re-read every time.
    record_run(context, Tables.symbol_tenure, 0, written, is_full_rescan=True)
    logger.info("symbol_tenure: wrote %d row(s)", written)
    return out
