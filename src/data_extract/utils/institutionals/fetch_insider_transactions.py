"""
fetch_insider_transactions.py (src/data_extract/utils/institutionals/fetch_insider_transactions.py)
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

TODO: get it from sec instead of zips quarterly
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data_store.schema import Tables
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir, ensure_zip, quarter_periods,
)
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import (
    load_cik_mapping, bulk_ingested_quarters, load_processed_universe,
    save_processed_universe, cik_to_ticker)

logger = logging.getLogger(__name__)

_OUT_COLS = [
    "accession_number", "security_type", "transaction_sk", "ticker", "issuer_cik",
    "issuer_name", "owner_cik", "owner_name", "is_director", "is_officer",
    "is_ten_pct_owner", "is_other", "officer_title", "document_type",
    "transaction_date", "filing_date", "period_of_report", "security_title",
    "transaction_code", "acquired_disposed", "shares", "price_per_share",
    "value_usd", "shares_owned_after", "direct_indirect", "quarter",
    # --- enrichment: present in the zips since 2006, previously discarded --- #
    "is_10b5_1", "transaction_form_type", "equity_swap_involved",
    "deemed_execution_date", "nature_of_ownership", "transaction_timeliness",
    # --- derivative block: NULL on nonderiv rows BY CONSTRUCTION --- #
    "exercise_price", "exercise_date", "expiration_date",
    "underlying_security_title", "underlying_shares", "underlying_value",
]

_FOOTNOTE_COLS = ["accession_number", "footnote_id", "footnote_text"]

#: `AFF10B5ONE` is mixed-encoding across (and within) quarters. Measured 2026q1: '0' 42,435,
#: 'false' 11,525, '1' 3,620, 'true' 1,162, NaN 10,517. Anything not in this map -- including
#: the empty string and the whole pre-2023q1 era, where the column does not exist -- stays NaN.
#: Stored as float precisely so NaN survives: a False would assert the insider traded outside a
#: plan, which the source does not say.
_10B5_1_TRUE = {"1", "true", "y", "yes"}
_10B5_1_FALSE = {"0", "false", "n", "no"}

# SEC bulk quarterly structured data sets (free TSV zips; {quarter} = e.g. "2024q1").
# insider = Forms 3/4/5 officer/director transactions; finstmt = primary-statement
# XBRL facts (num/sub) incl. the balance-sheet net pension liability.
SEC_INSIDER_URL_TEMPLATE = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/"
    "{quarter}_form345.zip")
SEC_INSIDER_FIRST_YEAR = 2006     

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column `name` if present, else an all-NA series aligned to df (the insider
    tables are stable, but be defensive across the 2011->today schema history)."""
    return df[name] if name in df.columns else pd.Series(pd.NA, index=df.index)


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_col(df, name), errors="coerce")


def _date(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_datetime(_col(df, name), format="mixed", errors="coerce")


def _normalize_10b5_1(raw: pd.Series) -> pd.Series:
    """`AFF10B5ONE` -> 1.0 / 0.0 / NaN. See `_10B5_1_TRUE` for why the result is a float."""
    text = raw.astype("string").str.strip().str.lower()
    out = pd.Series(float("nan"), index=raw.index, dtype="float64")
    out = out.mask(text.isin(_10B5_1_TRUE), 1.0)
    out = out.mask(text.isin(_10B5_1_FALSE), 0.0)
    return out


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
        "transaction_date": _date(df, "TRANS_DATE"),
        "transaction_code": _col(df, "TRANS_CODE"),
        "acquired_disposed": _col(df, "TRANS_ACQUIRED_DISP_CD"),
        "shares": shares,
        "price_per_share": price,
        "value_usd": value,
        "shares_owned_after": _num(df, "SHRS_OWND_FOLWNG_TRANS"),
        "direct_indirect": _col(df, "DIRECT_INDIRECT_OWNERSHIP"),
        # --- present on BOTH tables since 2006q1 (verified across the 81 cached zips) --- #
        # `transaction_form_type` is 4 or 5: it says which FORM reported the trade, and a
        # Form 5 is the late/exempt annual catch-up, so the same TRANS_CODE means something
        # different depending on it. `deemed_execution_date` is set only when the filer could
        # not know the price on the trade date (broker-executed plans), which is the strongest
        # non-narrative 10b5-1 hint available before 2023.
        "transaction_form_type": _col(df, "TRANS_FORM_TYPE"),
        "equity_swap_involved": _col(df, "EQUITY_SWAP_INVOLVED"),
        "deemed_execution_date": _date(df, "DEEMED_EXECUTION_DATE"),
        "nature_of_ownership": _col(df, "NATURE_OF_OWNERSHIP"),
        "transaction_timeliness": _col(df, "TRANS_TIMELINESS"),
        # --- DERIV_TRANS only; `_col` returns all-NA on the nonderiv table --- #
        # SEC misspells the exercise-date column `EXCERCISE_DATE` in its own data set, in every
        # quarter from 2006q1 to 2026q1. Reading the correct spelling silently yields all-NA.
        "exercise_price": _num(df, "CONV_EXERCISE_PRICE"),
        "exercise_date": _date(df, "EXCERCISE_DATE"),
        "expiration_date": _date(df, "EXPIRATION_DATE"),
        "underlying_security_title": _col(df, "UNDLYNG_SEC_TITLE"),
        "underlying_shares": _num(df, "UNDLYNG_SEC_SHARES"),
        "underlying_value": _num(df, "UNDLYNG_SEC_VALUE"),
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
        "filing_date": _date(sub, "FILING_DATE"),
        "period_of_report": _date(sub, "PERIOD_OF_REPORT"),
        # absent before 2023q1 -- `_col` returns all-NA there, which `_normalize_10b5_1` keeps
        # as NaN rather than turning into a False
        "is_10b5_1": _normalize_10b5_1(_col(sub, "AFF10B5ONE")),
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


def _footnotes(notes: pd.DataFrame, keep_accessions: set[str]) -> pd.DataFrame:
    """FOOTNOTES.tsv -> `insider_footnotes` rows, restricted to accessions we actually kept.

    The raw file is every filer's footnotes (~167k rows a quarter); without the accession filter
    this table would grow to ~13M rows describing companies nothing else in the pipeline reads.
    The (accession, footnote_id) PK holds at source -- 0 duplicate pairs across 2023q1 and
    2026q1 -- so no dedup is applied that could mask a future source change."""
    if notes is None or notes.empty or "ACCESSION_NUMBER" not in notes.columns:
        return pd.DataFrame(columns=_FOOTNOTE_COLS)
    out = pd.DataFrame({
        "accession_number": _col(notes, "ACCESSION_NUMBER"),
        "footnote_id": _col(notes, "FOOTNOTE_ID"),
        "footnote_text": _col(notes, "FOOTNOTE_TXT"),
    })
    out = out[out["accession_number"].isin(keep_accessions)]
    return out.dropna(subset=["accession_number", "footnote_id"])


def _filter_universe(df: pd.DataFrame, universe: set[str], cik2tkr: dict) -> pd.DataFrame:
    """Keep issuers in our universe; resolve ticker by trading SYMBOL first, else by issuer
    CIK (zero-padded to the 10-digit form used in `sp500_tickers`).

    ⚠ SYMBOL-FIRST IS RIGHT, AND IT IS ALSO WHAT HID A REGISTRANT-BOUNDARY BUG FOR A YEAR.
    Most reorganisations keep the trading symbol, so the symbol path kept resolving and
    nothing looked wrong. It fails exactly where the symbol moved TOO -- and there the CIK
    fallback was the only defence, against a map that held one CIK per ticker:

        GOOGL  insider_transactions starts 2015-10-08   (boundary 2015-10-02, predecessor GOOG)
        VTRS   insider_transactions starts 2020-11-16   (predecessor traded MYL)
        APA    insider_transactions starts 2006-01-04   -- saved ONLY because APA never moved

    The repair is in `cik_to_ticker`, which now carries every segment CIK, so the fallback
    consults the whole chain without this function changing shape. Forms 3/4/5 are EVENTS and
    combine as a UNION, so there is deliberately NO date filter here: a predecessor's Form 4
    filed after the boundary is still a real insider transaction in this issuer's security.
    Contrast `notes_*` and `pension_facts`, which are consolidating and take the dated split.
    """
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
    """SUBMISSION + REPORTINGOWNER + NONDERIV_TRANS + DERIV_TRANS + FOOTNOTES from a cached zip."""
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.upper(): n for n in z.namelist()}

            def rd(key):
                return (pd.read_csv(z.open(names[key]), sep="\t", dtype=str, low_memory=False)
                        if key in names else pd.DataFrame())

            sub = rd("SUBMISSION.TSV")
            if sub.empty:
                return None
            return (sub, rd("REPORTINGOWNER.TSV"), rd("NONDERIV_TRANS.TSV"),
                    rd("DERIV_TRANS.TSV"), rd("FOOTNOTES.TSV"))
    except zipfile.BadZipFile:
        logger.warning("insider %s: corrupt zip -> deleting so it re-downloads", path.name)
        path.unlink(missing_ok=True)
        return None


def fetch_insider_transactions(context: Context, tickers: list[str], years_history: int = 15,
                               reparse: bool = False) -> int:
    """Download (cached) the insider-transactions data sets over `years_history`,
    flatten to transactions, keep the universe, upsert to `insider_transactions` and
    `insider_footnotes`. Returns the number of transaction rows upserted.

    `reparse` re-reads every quarter the SOURCE has, even when it is already in the DB. Needed
    whenever the PARSE changes rather than the data -- adding a column to `_OUT_COLS` leaves 20
    years of stored rows with it NULL, and the incremental path would never revisit them. It is
    an explicit flag rather than a faked universe change so the log says what actually happened;
    nothing is re-downloaded either way, since a past quarter's zip is final.

    ⚠ `reparse` also widens the WINDOW, back to `SEC_INSIDER_FIRST_YEAR`. A routine run scans
    `years_history + 1` years, which in 2026 starts at 2010q1 -- but the table holds rows from
    2006-01-03, ingested when that window still reached them. Re-parsing only the window would
    leave those 16 quarters (2006q1-2009q4) carrying the OLD column set for ever, with the new
    fields NULL on the oldest data and populated on the rest. A split like that is worse than
    either state alone, because nothing downstream could tell it from a real coverage cliff."""

    cikmap = load_cik_mapping(context)
    cik2tkr = cik_to_ticker(cikmap)
    cache = cache_dir(context, context.config.local.paths.insider_transactions)

    done_q = bulk_ingested_quarters(context.store, Tables.insider_transactions)
    new_tickers = set(tickers) - load_processed_universe(cache, Tables.insider_transactions)   # empty once converged
    if new_tickers:
        logger.info("insider: %d new/changed tickers -> re-parsing cached quarters",
                    len(new_tickers))
    if reparse:
        logger.info("insider: --reparse -> re-reading every quarter back to %dq1 "
                    "(no re-download; %d already ingested)", SEC_INSIDER_FIRST_YEAR, len(done_q))

    # a reparse must reach every quarter the source has, not just the routine window -- see the
    # docstring. `quarter_periods` clamps to SEC_INSIDER_FIRST_YEAR either way.
    span = (pd.Timestamp.today().year - SEC_INSIDER_FIRST_YEAR + 1) if reparse else years_history + 1
    quarters = quarter_periods(span, SEC_INSIDER_FIRST_YEAR)

    saved = notes_saved = 0
    for q in tqdm(quarters, desc="insider data sets"):
        if q in done_q and not new_tickers and not reparse:
            continue                          # complete quarter already ingested
        path = ensure_zip(context, cache / f"{q}.zip",
                          SEC_INSIDER_URL_TEMPLATE.format(quarter=q),
                          label=f"insider {q}", log=logger)
        if path is None:
            continue
        tables = _read_tables(path)
        if tables is None:
            continue
        sub, own, nonderiv, deriv, notes = tables
        df = _filter_universe(_parse_insider(sub, own, nonderiv, deriv), tickers, cik2tkr)
        if df.empty:
            continue
        df["quarter"] = q
        saved += context.store.save(Tables.insider_transactions, df[[c for c in _OUT_COLS if c in df.columns]])
        # footnotes ride the SAME universe decision -- keyed on the accessions that survived
        foot = _footnotes(notes, set(df["accession_number"].dropna().unique()))
        if not foot.empty:
            notes_saved += context.store.save(Tables.insider_footnotes, foot[_FOOTNOTE_COLS])

    save_processed_universe(cache, Tables.insider_transactions, tickers)   # so a converged re-run skips
    logger.info("insider_transactions: upserted %d rows (+%d footnotes) over %d quarters "
                "(%s -> %s)", saved, notes_saved, len(quarters), quarters[0], quarters[-1])
    record_run(context, Tables.insider_transactions, len(tickers), saved)
    record_run(context, Tables.insider_footnotes, len(tickers), notes_saved)
    return saved
