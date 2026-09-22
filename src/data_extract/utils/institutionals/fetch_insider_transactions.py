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

from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir,
    ensure_zip,
    quarter_periods,
)
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import bulk_ingested_quarters, load_processed_universe, save_processed_universe
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

_OUT_COLS = [
    "accession_number",
    "security_type",
    "transaction_sk",
    "ticker",
    "issuer_cik",
    "issuer_name",
    "owner_cik",
    "owner_name",
    "is_director",
    "is_officer",
    "is_ten_pct_owner",
    "is_other",
    "officer_title",
    "document_type",
    "transaction_date",
    "filing_date",
    "period_of_report",
    "security_title",
    "transaction_code",
    "acquired_disposed",
    "shares",
    "price_per_share",
    "value_usd",
    "shares_owned_after",
    "direct_indirect",
    "quarter",
    # --- enrichment: present in the zips since 2006, previously discarded --- #
    "is_10b5_1",
    "transaction_form_type",
    "equity_swap_involved",
    "deemed_execution_date",
    "nature_of_ownership",
    "transaction_timeliness",
    # --- derivative block: NULL on nonderiv rows BY CONSTRUCTION --- #
    "exercise_price",
    "exercise_date",
    "expiration_date",
    "underlying_security_title",
    "underlying_shares",
    "underlying_value",
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
SEC_INSIDER_URL_TEMPLATE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/" "{quarter}_form345.zip"
SEC_INSIDER_URL_NEW_TEMPLATE = "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/" "{quarter}_form345.zip"
SEC_INSIDER_FIRST_YEAR = 2006
SEC_INSIDER_SWAP_YEAR = 2026


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
    if "TRANS_TOTAL_VALUE" in df.columns:  # derivative table carries it
        value = value.fillna(_num(df, "TRANS_TOTAL_VALUE"))
    return pd.DataFrame(
        {
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
        }
    )


def _parse_insider(sub: pd.DataFrame, own: pd.DataFrame, nonderiv: pd.DataFrame, deriv: pd.DataFrame) -> pd.DataFrame:
    """SUBMISSION + REPORTINGOWNER + (NON)DERIV_TRANS -> tidy transactions. Pure."""
    if sub is None or sub.empty:
        return pd.DataFrame()
    submission = pd.DataFrame(
        {
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
        }
    )

    rel = _col(own, "RPTOWNER_RELATIONSHIP").astype("string").str.lower().fillna("")
    owner = pd.DataFrame(
        {
            "accession_number": _col(own, "ACCESSION_NUMBER"),
            "owner_cik": _col(own, "RPTOWNERCIK"),
            "owner_name": _col(own, "RPTOWNERNAME"),
            "officer_title": _col(own, "RPTOWNER_TITLE"),
            "is_director": rel.str.contains("director", na=False).astype(float),
            "is_officer": rel.str.contains("officer", na=False).astype(float),
            "is_ten_pct_owner": (rel.str.contains("ten", na=False) | rel.str.contains("10", na=False)).astype(float),
            "is_other": rel.str.contains("other", na=False).astype(float),
        }
    ).drop_duplicates("accession_number", keep="first")  # 1 owner per filing (a.o.c.)

    trans = pd.concat([_transactions(nonderiv, "NONDERIV_TRANS_SK", "nonderiv"), _transactions(deriv, "DERIV_TRANS_SK", "deriv")], ignore_index=True)
    if trans.empty:
        return pd.DataFrame()
    out = trans.merge(submission, on="accession_number", how="inner").merge(owner, on="accession_number", how="left")
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
            except ValueError:  # 29 Feb in a non-leap target year
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
    out = pd.DataFrame(
        {
            "accession_number": _col(notes, "ACCESSION_NUMBER"),
            "footnote_id": _col(notes, "FOOTNOTE_ID"),
            "footnote_text": _col(notes, "FOOTNOTE_TXT"),
        }
    )
    out = out[out["accession_number"].isin(keep_accessions)]
    return out.dropna(subset=["accession_number", "footnote_id"])


#: Quarantine rows carry every `insider_transactions` column plus the verdict (see
#: `Tables.insider_transactions_quarantine`).
_VERDICT_COLS = ["reject_reason", "resolved_entity_id", "universe_entity_id", "screened_on"]
_QUARANTINE_COLS = _OUT_COLS + _VERDICT_COLS

#: Columns the stored-row sweep reads over the WHOLE table; the full rows of the rejects are
#: re-read by accession afterwards. 2M rows x 5 narrow columns instead of 2M x 38.
_SWEEP_COLS = ["accession_number", "security_type", "transaction_sk", "ticker", "issuer_cik"]


def _verdicts(df: pd.DataFrame, universe, identity) -> pd.DataFrame:
    """Resolve every row CIK-first and attach the three verdict columns. Pure.

    Returns `df` plus `claimed_ticker` (what the row said), `ticker` (what its CIK resolves
    to), `resolved_entity_id`, `universe_entity_id` and `reject_reason` -- NaN on the rows
    that are kept. Split out of `_filter_universe` because the stored-row sweep
    (`_screen_stored_rows`) needs exactly the same adjudication on rows that came back out of
    the database rather than out of a zip.
    """
    universe = set(universe)
    raw = df["issuer_cik"]
    # ~2k distinct issuers per quarter against ~150k transaction rows: resolve once per CIK.
    uniq = [v for v in pd.unique(raw) if v is not None and not pd.isna(v)]
    to_ticker = {v: identity.entity_ticker(v) for v in uniq}
    to_entity = {v: identity.entity_of(v) for v in uniq}

    claimed = df["ticker"].astype("string")
    resolved = raw.map(to_ticker).astype("string")
    # `universe_entity` RAISES on a ticker absent from the roster, which is the common case
    # here (the claimed string is a filer's free-typed symbol), so it is asked only about
    # tickers the roster actually has.
    known = {t: identity.universe_entity(t) for t in set(claimed.dropna()) & set(identity.roster_cik)}

    out = df.assign(
        claimed_ticker=claimed,
        ticker=resolved,
        resolved_entity_id=raw.map(to_entity).astype("string"),
        universe_entity_id=claimed.map(known).astype("string"),
        screened_on=df["filing_date"] if "filing_date" in df.columns else pd.NaT,
    )

    keep = resolved.isin(universe)
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason[~keep] = "entity_not_in_universe"
    reason[~keep & claimed.isin(universe)] = "entity_mismatch"
    reason[~keep & (raw.isna() | (raw.astype("string").str.strip() == ""))] = "no_issuer_cik"
    return out.assign(reject_reason=reason)


def _filter_universe(df: pd.DataFrame, universe: set[str], identity) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition into (kept, rejected). Resolution is CIK-FIRST: the row's own `issuer_cik`
    names an ENTITY, and the entity names today's universe ticker.

    ⚠ THIS REPLACED A SYMBOL-FIRST SCREEN, AND THE HISTORY IS THE REASON THE MODULE IS
    CAREFUL. The old line accepted whatever `ISSUERTRADINGSYMBOL` the filer typed provided
    that string was in today's universe, with the CIK map only as a fallback. Most
    reorganisations keep the trading symbol, so the symbol path kept resolving and nothing
    looked wrong -- it failed exactly where the symbol moved TOO, and the CIK fallback was
    the only defence:

        GOOGL  insider_transactions started 2015-10-08  (boundary 2015-10-02, predecessor GOOG)
        VTRS   insider_transactions started 2020-11-16  (predecessor traded MYL)
        APA    insider_transactions started 2006-01-04  -- saved ONLY because APA never moved

    What symbol-first could not see at all is a symbol ANOTHER LIVE COMPANY held earlier:
    2,075 Trane Technologies rows under `IR`, 1,207 CoreSite under `COR`, 2,046 Weight
    Watchers under `WTW`. The filer typed a true symbol; it was simply not this company's.

    So the symbol is now a CROSS-CHECK, kept on the rejected rows as `claimed_ticker`, and
    never a resolver. `no_issuer_cik` measured ZERO across all 4,402,307 filings in the 81
    cached quarters, so there is no symbol fallback to build.

    ⚠ STILL NO DATE FILTER, AND THAT IS DELIBERATE. Forms 3/4/5 are EVENTS and combine as a
    UNION (`registrant.FORM_POLICY`): a predecessor's Form 4 filed after a registrant
    boundary is still a real insider transaction in this issuer's security, and a date cut
    here is the named XOM-`SCHEDULE 13G` regression. Contrast `notes_*` and `pension_facts`,
    which are consolidating and take the dated split. `screened_on` is recorded anyway -- a
    verdict without the date it was taken against is not auditable.

    ⚠ THE PARTITION IS NOT EXHAUSTIVE OVER `df`, AND IT MUST NOT BE. `df` is every filer in
    the quarter (~54k filings), so quarantining every non-universe row would store ~50M rows
    about companies nothing in the pipeline reads. `rejected` is scoped to rows that either
    CLAIMED a universe ticker or resolve to a roster company -- the rows the old screen would
    have admitted, which is what makes them evidence. Everything else is dropped as it always
    was. The stored-row sweep in `_screen_stored_rows` is the exhaustive one, because there
    every row is already in the table and must either stay or leave.
    """
    empty = pd.DataFrame(columns=_QUARANTINE_COLS)
    if df.empty:
        return df, empty
    scored = _verdicts(df, universe, identity)
    keep = scored["reject_reason"].isna()
    # scope: claimed a universe ticker, resolves to a roster company, or carries no CIK at all
    in_scope = scored["claimed_ticker"].isin(set(universe)) | scored["ticker"].notna() | (scored["reject_reason"] == "no_issuer_cik")
    return scored[keep], scored[~keep & in_scope]


def _to_quarantine(rejected: pd.DataFrame) -> pd.DataFrame:
    """Rejected rows in `insider_transactions_quarantine` shape.

    `ticker` becomes the ticker the row CLAIMED, not the NULL its CIK resolved to: the claim
    is the whole evidence -- `IR` on a Trane Technologies filing is what the old screen
    believed. The resolved side is preserved as `resolved_entity_id`, which is an entity id
    and so cannot be mistaken for a tradable symbol by a downstream join.
    """
    if rejected is None or rejected.empty:
        return pd.DataFrame(columns=_QUARANTINE_COLS)
    out = rejected.assign(ticker=rejected["claimed_ticker"])
    return out[[c for c in _QUARANTINE_COLS if c in out.columns]]


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #


def _read_tables(path: Path):
    """SUBMISSION + REPORTINGOWNER + NONDERIV_TRANS + DERIV_TRANS + FOOTNOTES from a cached zip."""
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.upper(): n for n in z.namelist()}

            def rd(key):
                return pd.read_csv(z.open(names[key]), sep="\t", dtype=str, low_memory=False) if key in names else pd.DataFrame()

            sub = rd("SUBMISSION.TSV")
            if sub.empty:
                return None
            return (sub, rd("REPORTINGOWNER.TSV"), rd("NONDERIV_TRANS.TSV"), rd("DERIV_TRANS.TSV"), rd("FOOTNOTES.TSV"))
    except zipfile.BadZipFile:
        logger.warning("insider %s: corrupt zip -> deleting so it re-downloads", path.name)
        path.unlink(missing_ok=True)
        return None


def _screen_stored_rows(context: Context, universe, identity: Identity, chunk: int = 2_000) -> tuple[int, int]:
    """Re-adjudicate EVERY STORED ROW against today's universe; quarantine then DELETE the
    rejects. Returns `(quarantined, deleted)`.

    ⚠ THIS EXISTS BECAUSE `store.save` UPSERTS AND SO CAN NEVER REMOVE A ROW. The parse-time
    screen stops a wrong row being written, and that is all it can do: a `--reparse` simply
    declines to re-write the 10,717 out-of-lineage rows already in the table, and they would
    stay there for ever. The RELABELLED rows need none of this -- `ticker` is not in the
    primary key (`accession_number`, `security_type`, `transaction_sk`), so a row whose ticker
    moves `IR` -> `TT` UPDATES in place and the table cannot double -- but a row rejected
    outright has nothing to update it.

    It is also the only thing that reconciles a SHRINKING universe. 8,099 stored rows carry a
    ticker no longer in `load_universe_tickers`: `EA` 6,161 and `AVB` 1,788 (both gone from
    the roster), plus 150 across the five spin-offs now in `INSUFFICIENT_HISTORY_TICKERS`. No
    parse would ever revisit them, because a parse only ever looks at what the zips contain.

    ⚠ UNLIKE THE PARSE SCREEN, THIS PARTITION IS EXHAUSTIVE, and it has to be: every stored
    row either stays or is quarantined, which is what makes the before/after row count close
    arithmetically instead of approximately. One accession's rows all share one `issuer_cik`
    and therefore one verdict, so the DELETE can key on `accession_number` alone without
    touching a row that was kept.
    """
    keys = context.store.load(Tables.insider_transactions, columns=["accession_number", "ticker", "issuer_cik"], optional=True)
    if keys is None or keys.empty:
        return 0, 0
    # One verdict per (accession, claimed ticker, cik): the grain it is actually decided at,
    # so the whole-table pass costs ~1.3M dedup keys rather than 2M full rows.
    scored = _verdicts(keys.drop_duplicates().assign(filing_date=pd.NaT), universe, identity)
    accessions = sorted(scored.loc[scored["reject_reason"].notna(), "accession_number"].dropna().unique())
    if not accessions:
        logger.info("insider: stored-row sweep -- 0 of %d row(s) rejected", len(keys))
        return 0, 0

    quarantined = deleted = 0
    for start in range(0, len(accessions), chunk):
        batch = accessions[start : start + chunk]
        rows = context.store.load(Tables.insider_transactions, where={"accession_number": batch}, optional=True)
        if rows is None or rows.empty:
            continue
        rejected = _verdicts(rows, universe, identity)
        rejected = rejected[rejected["reject_reason"].notna()]
        if rejected.empty:  # re-adjudicated clean on the full row: leave it
            continue
        quarantined += context.store.save(Tables.insider_transactions_quarantine, _to_quarantine(rejected))
        deleted += context.store.delete(Tables.insider_transactions, where={"accession_number": sorted(rejected["accession_number"].unique())})
    logger.info("insider: stored-row sweep -- quarantined %d row(s) over %d accession(s), " "deleted %d", quarantined, len(accessions), deleted)
    return quarantined, deleted


def fetch_insider_transactions(context: Context, tickers: list[str], years_history: int = 15, reparse: bool = False) -> int:
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

    identity = load_identity(context)
    cache = cache_dir(context, context.config.local.paths.insider_transactions)

    done_q = bulk_ingested_quarters(context.store, Tables.insider_transactions)
    new_tickers = set(tickers) - load_processed_universe(cache, Tables.insider_transactions)  # empty once converged
    if new_tickers:
        logger.info("insider: %d new/changed tickers -> re-parsing cached quarters", len(new_tickers))
    if reparse:
        logger.info(
            "insider: --reparse -> re-reading every quarter back to %dq1 " "(no re-download; %d already ingested)",
            SEC_INSIDER_FIRST_YEAR,
            len(done_q),
        )

    # a reparse must reach every quarter the source has, not just the routine window -- see the
    # docstring. `quarter_periods` clamps to SEC_INSIDER_FIRST_YEAR either way.
    span = (pd.Timestamp.today().year - SEC_INSIDER_FIRST_YEAR + 1) if reparse else years_history + 1
    quarters = quarter_periods(span, SEC_INSIDER_FIRST_YEAR)

    saved = notes_saved = quarantined = 0
    for q in tqdm(quarters, desc="insider data sets"):
        if q in done_q and not new_tickers and not reparse:
            continue  # complete quarter already ingested

        if int(q[:4]) >= SEC_INSIDER_SWAP_YEAR:
            url_insider = SEC_INSIDER_URL_NEW_TEMPLATE
        else:
            url_insider = SEC_INSIDER_URL_TEMPLATE

        path = ensure_zip(context, cache / f"{q}.zip", url_insider.format(quarter=q), label=f"insider {q}", log=logger)
        if path is None:
            continue
        tables = _read_tables(path)
        if tables is None:
            continue
        sub, own, nonderiv, deriv, notes = tables
        df, rejected = _filter_universe(_parse_insider(sub, own, nonderiv, deriv), tickers, identity)
        if not rejected.empty:
            quarantined += context.store.save(Tables.insider_transactions_quarantine, _to_quarantine(rejected.assign(quarter=q)))
        if df.empty:
            continue
        df["quarter"] = q
        saved += context.store.save(Tables.insider_transactions, df[[c for c in _OUT_COLS if c in df.columns]])
        # footnotes ride the SAME universe decision -- keyed on the accessions that survived
        foot = _footnotes(notes, set(df["accession_number"].dropna().unique()))
        if not foot.empty:
            notes_saved += context.store.save(Tables.insider_footnotes, foot[_FOOTNOTE_COLS])

    # The upsert above cannot REMOVE anything, so the stored rows are reconciled separately.
    swept, deleted = _screen_stored_rows(context, tickers, identity)
    quarantined += swept
    save_processed_universe(cache, Tables.insider_transactions, tickers)  # so a converged re-run skips
    logger.info(
        "insider_transactions: upserted %d rows (+%d footnotes) over %d quarters " "(%s -> %s); quarantined %d, deleted %d",
        saved,
        notes_saved,
        len(quarters),
        quarters[0],
        quarters[-1],
        quarantined,
        deleted,
    )
    record_run(context, Tables.insider_transactions, len(tickers), saved)
    record_run(context, Tables.insider_footnotes, len(tickers), notes_saved)
    record_run(context, Tables.insider_transactions_quarantine, len(tickers), quarantined)
    return saved
