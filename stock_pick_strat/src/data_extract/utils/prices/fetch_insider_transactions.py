"""
fetch_insider_transactions.py  (src/data_extract/utils/prices/fetch_insider_transactions.py)
--------------------------------------------------------------------------------------------
Officer / director / 10%-owner transactions from the SEC "Insider Transactions
Data Sets" (free quarterly bulk TSV zips, Forms 3/4/5). Distinct from the 13F
institutional signal: this is the ISSUER's own insiders buying and selling their
stock -- a classic alpha (cluster insider buying tends to precede outperformance).

Each quarter's zip carries (see the FORM_345 readme):
  * SUBMISSION      accession -> issuer CIK / name / TRADING SYMBOL, filing & report
                    dates, DOCUMENT_TYPE (3/4/5, incl. amendments 3A/4A/5A)
  * REPORTINGOWNER  accession -> owner CIK / name, RPTOWNER_RELATIONSHIP (Director /
                    Officer / TenPercentOwner / Other), officer title
  * NONDERIV_TRANS  Table-I transactions: date, code (P/S/A/M/G/F...), shares, price,
                    acquired/disposed (A/D), shares owned after, direct/indirect
  * DERIV_TRANS     Table-II derivative transactions (options etc.), same shape

We flatten NONDERIV + DERIV into one tidy row per transaction (keyed on
accession + table + SK), attach the issuer ticker and the owner's role flags, and
keep our universe.

Incremental (both dimensions the brief asks for):
  * ZIPs are cached under data/sec_bulk_cache/insider_transactions/ and only
    downloaded when missing (a past quarter's zip is final once the quarter ends);
  * a quarter already in the DB is SKIPPED entirely (no download, no parse) UNLESS
    the universe gained tickers, in which case cached zips are re-parsed (no
    re-download) to back-fill the new names. The upsert de-duplicates on the PK.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_INSIDER_TRANSACTIONS_TABLE, SEC_INSIDER_URL_TEMPLATE, SEC_INSIDER_FIRST_YEAR
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir, ensure_zip, quarter_periods,
)
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import (
    load_cik_mapping, bulk_ingested_quarters, load_processed_universe,
    save_processed_universe)

logger = logging.getLogger(__name__)

_OUT_COLS = [
    "accession_number", "security_type", "transaction_sk", "ticker", "issuer_cik",
    "issuer_name", "owner_cik", "owner_name", "is_director", "is_officer",
    "is_ten_pct_owner", "is_other", "officer_title", "document_type",
    "transaction_date", "filing_date", "period_of_report", "security_title",
    "transaction_code", "acquired_disposed", "shares", "price_per_share",
    "value_usd", "shares_owned_after", "direct_indirect", "quarter",
]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column `name` if present, else an all-NA series aligned to df (the insider
    tables are stable, but be defensive across the 2011->today schema history)."""
    return df[name] if name in df.columns else pd.Series(pd.NA, index=df.index)


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_col(df, name), errors="coerce")


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested): the flatten/join has no IO                          #
# --------------------------------------------------------------------------- #
def _transactions(df: pd.DataFrame, sk_col: str, security_type: str) -> pd.DataFrame:
    """One tidy row per (non-)derivative transaction line."""
    if df is None or df.empty or "ACCESSION_NUMBER" not in df.columns:
        return pd.DataFrame()
    shares = _num(df, "TRANS_SHARES")
    price = _num(df, "TRANS_PRICEPERSHARE")
    value = shares * price
    if "TRANS_TOTAL_VALUE" in df.columns:               # derivative table carries it
        value = value.fillna(_num(df, "TRANS_TOTAL_VALUE"))
    return pd.DataFrame({
        "accession_number": _col(df, "ACCESSION_NUMBER"),
        "security_type": security_type,
        "transaction_sk": _col(df, sk_col),
        "security_title": _col(df, "SECURITY_TITLE"),
        "transaction_date": pd.to_datetime(_col(df, "TRANS_DATE"), format="mixed", errors="coerce"),
        "transaction_code": _col(df, "TRANS_CODE"),
        "acquired_disposed": _col(df, "TRANS_ACQUIRED_DISP_CD"),
        "shares": shares,
        "price_per_share": price,
        "value_usd": value,
        "shares_owned_after": _num(df, "SHRS_OWND_FOLWNG_TRANS"),
        "direct_indirect": _col(df, "DIRECT_INDIRECT_OWNERSHIP"),
    })


def _parse_insider(sub: pd.DataFrame, own: pd.DataFrame,
                   nonderiv: pd.DataFrame, deriv: pd.DataFrame) -> pd.DataFrame:
    """SUBMISSION + REPORTINGOWNER + (NON)DERIV_TRANS -> tidy transactions. Pure."""
    if sub is None or sub.empty:
        return pd.DataFrame()
    submission = pd.DataFrame({
        "accession_number": _col(sub, "ACCESSION_NUMBER"),
        "issuer_cik": _col(sub, "ISSUERCIK"),
        "issuer_name": _col(sub, "ISSUERNAME"),
        "ticker": _col(sub, "ISSUERTRADINGSYMBOL").astype("string").str.strip().str.upper(),
        "document_type": _col(sub, "DOCUMENT_TYPE"),
        "filing_date": pd.to_datetime(_col(sub, "FILING_DATE"), format="mixed", errors="coerce"),
        "period_of_report": pd.to_datetime(_col(sub, "PERIOD_OF_REPORT"), format="mixed", errors="coerce"),
    })

    rel = _col(own, "RPTOWNER_RELATIONSHIP").astype("string").str.lower().fillna("")
    owner = pd.DataFrame({
        "accession_number": _col(own, "ACCESSION_NUMBER"),
        "owner_cik": _col(own, "RPTOWNERCIK"),
        "owner_name": _col(own, "RPTOWNERNAME"),
        "officer_title": _col(own, "RPTOWNER_TITLE"),
        "is_director": rel.str.contains("director", na=False).astype(float),
        "is_officer": rel.str.contains("officer", na=False).astype(float),
        "is_ten_pct_owner": (rel.str.contains("ten", na=False)
                             | rel.str.contains("10", na=False)).astype(float),
        "is_other": rel.str.contains("other", na=False).astype(float),
    }).drop_duplicates("accession_number", keep="first")   # 1 owner per filing (a.o.c.)

    trans = pd.concat([_transactions(nonderiv, "NONDERIV_TRANS_SK", "nonderiv"),
                       _transactions(deriv, "DERIV_TRANS_SK", "deriv")], ignore_index=True)
    if trans.empty:
        return pd.DataFrame()
    out = (trans.merge(submission, on="accession_number", how="inner")
                .merge(owner, on="accession_number", how="left"))
    out = _repair_transaction_dates(out)
    # a transaction with no SK can't be keyed (PK) -> drop
    return out.dropna(subset=["transaction_sk"])


def _repair_transaction_dates(df: pd.DataFrame) -> pd.DataFrame:
    """A Form 3/4/5 reports a COMPLETED transaction, so `transaction_date` can never be after
    the `filing_date` that discloses it. That makes the field self-validating, and the live table
    showed 14 rows breaking it in two distinct ways:

      * LOST CENTURY -- `0015-11-23` filed 2015-11-25, `0024-02-01` filed 2024-02-05: a 2-digit
        source year parsed as year 15 / 24 AD. Repaired by lifting the year into the filing's
        century, which is unambiguous because the transaction must precede the filing.
      * FILER TYPO -- `2028-05-24` filed 2024-05-28 (day/year digits transposed), `2031-01-29`
        filed 2021-02-02, `2029-08-12` filed 2019-08-13. Bad at source, with no safe reading, so
        the date is NULLED rather than guessed; the row's amounts still count, only its timing is
        unknown.

    Only 14 of 1.39M rows, but a transaction stamped 2031 poisons any recency-weighted insider
    feature far out of proportion to its count."""
    if df.empty or not {"transaction_date", "filing_date"}.issubset(df.columns):
        return df
    td, fd = df["transaction_date"], df["filing_date"]
    known = td.notna() & fd.notna()

    # lost century: shift the year into the filing's century and keep it only if that lands
    # at or before the filing date (so a genuine old transaction is never rewritten)
    lost = known & (td.dt.year < 1900)
    if lost.any():
        shifted = td.where(~lost).copy()
        for i in df.index[lost]:
            century = (fd[i].year // 100) * 100
            try:
                cand = td[i].replace(year=century + td[i].year % 100)
            except ValueError:                      # 29 Feb in a non-leap target year
                continue
            if cand <= fd[i]:
                shifted[i] = cand
        df = df.assign(transaction_date=shifted)
        td = df["transaction_date"]

    # still after its own filing -> unusable timing, blank it
    impossible = td.notna() & fd.notna() & (td > fd)
    if impossible.any():
        df = df.assign(transaction_date=td.mask(impossible))
    return df


def _filter_universe(df: pd.DataFrame, universe: set[str], cik2tkr: dict) -> pd.DataFrame:
    """Keep issuers in our universe; resolve ticker by trading symbol first, else
    by issuer CIK (zero-padded to the 10-digit form used in sp500_tickers)."""
    if df.empty:
        return df
    df = df.copy()
    df["ticker"] = df["ticker"].where(df["ticker"].isin(universe))
    need = df["ticker"].isna() & df["issuer_cik"].notna()
    if need.any() and cik2tkr:
        df.loc[need, "ticker"] = (df.loc[need, "issuer_cik"].astype("string")
                                  .str.zfill(10).map(cik2tkr))
    return df[df["ticker"].isin(universe)]


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #

def _read_tables(path: Path):
    """SUBMISSION + REPORTINGOWNER + NONDERIV_TRANS + DERIV_TRANS from a cached zip."""
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.upper(): n for n in z.namelist()}

            def rd(key):
                return (pd.read_csv(z.open(names[key]), sep="\t", dtype=str, low_memory=False)
                        if key in names else pd.DataFrame())

            sub = rd("SUBMISSION.TSV")
            if sub.empty:
                return None
            return sub, rd("REPORTINGOWNER.TSV"), rd("NONDERIV_TRANS.TSV"), rd("DERIV_TRANS.TSV")
    except zipfile.BadZipFile:
        logger.warning("insider %s: corrupt zip -> deleting so it re-downloads", path.name)
        path.unlink(missing_ok=True)
        return None


def fetch_insider_transactions(context: Context, tickers: list[str]) -> int:
    """Download (cached) the insider-transactions data sets over `years_history`,
    flatten to transactions, keep the universe, upsert to `insider_transactions`.
    Returns the number of rows upserted."""
    
    store = context.store
    cikmap = load_cik_mapping(context)
    cik2tkr = ({c: str(t).upper() for c, t in zip(cikmap["cik"], cikmap["ticker"])}
               if not cikmap.empty and "ticker" in cikmap.columns else {})
    years_history = context.config.data_extract.years_history + 1
    cache = cache_dir(context, "sec_insider_transactions")

    tickers = {str(t).upper() for t in tickers}          # universe as an uppercased set
    done_q = bulk_ingested_quarters(store, SEC_INSIDER_TRANSACTIONS_TABLE)
    new_tickers = tickers - load_processed_universe(cache, SEC_INSIDER_TRANSACTIONS_TABLE)   # empty once converged
    if new_tickers:
        logger.info("insider: %d new/changed tickers -> re-parsing cached quarters",
                    len(new_tickers))

    saved = 0
    for q in tqdm(quarter_periods(years_history, SEC_INSIDER_FIRST_YEAR), desc="insider data sets"):
        if q in done_q and not new_tickers:
            continue                          # complete quarter already ingested
        path = ensure_zip(cache / f"{q}.zip",
                          SEC_INSIDER_URL_TEMPLATE.format(quarter=q),
                          label=f"insider {q}", log=logger)
        if path is None:
            continue
        tables = _read_tables(path)
        if tables is None:
            continue
        df = _filter_universe(_parse_insider(*tables), tickers, cik2tkr)
        if df.empty:
            continue
        df["quarter"] = q
        saved += store.save(SEC_INSIDER_TRANSACTIONS_TABLE, df[[c for c in _OUT_COLS if c in df.columns]])

    save_processed_universe(cache, SEC_INSIDER_TRANSACTIONS_TABLE, tickers)   # so a converged re-run skips
    logger.info("insider_transactions: upserted %d rows (%d quarters scanned)",
                   saved, len(quarter_periods(years_history, SEC_INSIDER_FIRST_YEAR)))
    record_run(context, SEC_INSIDER_TRANSACTIONS_TABLE, len(tickers), saved)
    return saved
