"""Officer / director / 10%-owner transactions from the SEC quarterly Insider Transactions Data
Sets (bulk TSV zips, Forms 3/4/5).

SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS and DERIV_TRANS are mapped to the canonical string
frame (`insider_common.INSIDER_FIELDS`), typed by `build_insider_frame`, screened CIK-first, and
upserted one row per (accession, table, SK); FOOTNOTES follow the kept accessions.

Zips are cached and downloaded only when missing; a stored quarter is skipped unless the universe
gained tickers or `reparse` is set, in which case cached zips are re-parsed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    SEC_INSIDER_FIRST_YEAR,
    SEC_INSIDER_SWAP_YEAR,
    SEC_INSIDER_URL_NEW_TEMPLATE,
    SEC_INSIDER_URL_TEMPLATE,
)
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    ZipRead,
    cache_dir,
    ensure_zip,
    mark_processed,
    pending_periods,
    quarter_periods,
    read_zip_tables,
)
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.institutionals.insider_common import (
    BULK_DATE_FORMATS,
    INSIDER_COLUMNS,
    INSIDER_FIELDS,
    InsiderField,
    build_insider_frame,
    empty_footnotes,
    filter_footnotes,
    insider_verdicts,
    quarantine_frame,
    screen_insider_rows,
    screened_accessions,
)
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

#: SUBMISSION / REPORTINGOWNER columns `INSIDER_FIELDS` maps; remarks and addresses are never read.
_MEMBER_COLUMNS = {
    scope: frozenset({"ACCESSION_NUMBER", *(field.bulk for field in INSIDER_FIELDS if field.scope == scope and field.bulk)})
    for scope in ("filing", "owner")
}
#: The five Form 345 members, in `_read_tables` order; only SUBMISSION is required.
_ZIP_SPECS = {
    "SUBMISSION.TSV": ZipRead(usecols=_MEMBER_COLUMNS["filing"]),
    "REPORTINGOWNER.TSV": ZipRead(usecols=_MEMBER_COLUMNS["owner"], required=False),
    "NONDERIV_TRANS.TSV": ZipRead(required=False),
    "DERIV_TRANS.TSV": ZipRead(required=False),
    "FOOTNOTES.TSV": ZipRead(required=False),
}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column `name` if present, else an all-NA series aligned to df."""
    return df[name] if name in df.columns else pd.Series(pd.NA, index=df.index)


# --------------------------------------------------------------------------- #
# Pure parse: zip members -> canonical string frame                            #
# --------------------------------------------------------------------------- #
def _bulk_text(df: pd.DataFrame, field: InsiderField) -> pd.Series:
    """One canonical field from its TSV column; a reporting owner without a relationship has no role."""
    column = _col(df, field.bulk or "")
    return column.fillna("") if field.kind == "role" else column


def _member_strings(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """The canonical string columns of one member (`scope`), keyed on `accession_number`."""
    fields = [field for field in INSIDER_FIELDS if field.scope == scope and field.bulk]
    return pd.DataFrame({"accession_number": _col(df, "ACCESSION_NUMBER"), **{field.name: _bulk_text(df, field) for field in fields}})


def _transaction_strings(df: pd.DataFrame | None, sk_col: str, security_type: str) -> pd.DataFrame:
    """One string row per (non-)derivative transaction line."""
    if df is None or df.empty or "ACCESSION_NUMBER" not in df.columns:
        return pd.DataFrame()
    return _member_strings(df, "transaction").assign(security_type=security_type, transaction_sk=_col(df, sk_col))


def extract_bulk_strings(sub: pd.DataFrame, own: pd.DataFrame, nonderiv: pd.DataFrame, deriv: pd.DataFrame) -> pd.DataFrame:
    """SUBMISSION + REPORTINGOWNER (first owner per accession) + (NON)DERIV_TRANS -> canonical
    string rows; rows without a `transaction_sk` cannot be keyed and are dropped."""
    if sub is None or sub.empty:
        return pd.DataFrame()
    df_trans = pd.concat(
        [_transaction_strings(nonderiv, "NONDERIV_TRANS_SK", "nonderiv"), _transaction_strings(deriv, "DERIV_TRANS_SK", "deriv")], ignore_index=True
    )
    if df_trans.empty:
        return pd.DataFrame()
    df_owner = _member_strings(own, "owner").drop_duplicates("accession_number", keep="first")
    df_str = df_trans.merge(_member_strings(sub, "filing"), on="accession_number", how="inner").merge(df_owner, on="accession_number", how="left")
    return df_str.dropna(subset=["transaction_sk"])


def _footnote_strings(notes: pd.DataFrame | None) -> pd.DataFrame:
    """FOOTNOTES.tsv -> `insider_footnotes` columns."""
    if notes is None or notes.empty or "ACCESSION_NUMBER" not in notes.columns:
        return empty_footnotes()
    return pd.DataFrame(
        {"accession_number": _col(notes, "ACCESSION_NUMBER"), "footnote_id": _col(notes, "FOOTNOTE_ID"), "footnote_text": _col(notes, "FOOTNOTE_TXT")}
    )


def _accession_rows(df: pd.DataFrame, accessions: set[str]) -> pd.DataFrame:
    """Rows of one zip member whose `ACCESSION_NUMBER` is in `accessions`."""
    if "ACCESSION_NUMBER" not in df.columns:
        return df
    # `isin` on the Arrow-backed str column is ~15x slower than on its object copy.
    return df[df["ACCESSION_NUMBER"].astype(object).isin(accessions)]


def _parse_quarter(
    tables: tuple[pd.DataFrame, ...], quarter: str, universe: Sequence[str], identity: Identity
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One quarter's members -> (kept transactions, quarantine rows, footnotes of kept accessions).
    Members are first cut to the accessions the screen can keep or quarantine."""
    sub, own, nonderiv, deriv, notes = tables
    accessions = screened_accessions(_member_strings(sub, "filing"), universe, identity)
    df_str = extract_bulk_strings(*(_accession_rows(df, accessions) for df in (sub, own, nonderiv, deriv)))
    df_built = build_insider_frame(df_str, value_rule="shares_x_price_first", numeric_rule="to_numeric", date_formats=BULK_DATE_FORMATS)
    df_kept, df_quarantine = screen_insider_rows(df_built.assign(quarter=quarter), universe, identity)
    if df_kept.empty:
        return df_kept, df_quarantine, empty_footnotes()
    df_notes = filter_footnotes(_footnote_strings(notes), set(df_kept["accession_number"].dropna().unique()))
    return df_kept[[column for column in INSIDER_COLUMNS if column in df_kept.columns]], df_quarantine, df_notes


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #


def _read_tables(path: Path) -> tuple[pd.DataFrame, ...] | None:
    """(SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, DERIV_TRANS, FOOTNOTES) from a cached zip; an
    absent optional member is an empty frame. None when SUBMISSION is absent or empty, or the zip is
    corrupt (deleted for re-download)."""
    tables = read_zip_tables(path, _ZIP_SPECS, on_corrupt="delete", log=logger)
    if not tables or tables["SUBMISSION.TSV"].empty:
        return None
    return tuple(tables.values())


def _rejected_stored_accessions(context: Context, universe: Sequence[str], identity: Identity) -> list[str]:
    """Stored accessions holding a row whose `issuer_cik` (NULL included) resolves outside the
    universe; the reject verdict reads only the CIK, so each distinct CIK is scored once."""
    ciks = context.store.distinct(Tables.insider_transactions, "issuer_cik", dropna=False)
    scored = insider_verdicts(pd.DataFrame({"issuer_cik": pd.Series(ciks, dtype=object), "ticker": pd.NA}), universe, identity)
    rejected = scored.loc[scored["reject_reason"].notna(), "issuer_cik"]
    logger.info("insider: stored-row sweep -- %d of %d issuer CIK(s) rejected", len(rejected), len(ciks))
    if rejected.empty:
        return []
    wheres = [{"issuer_cik": sorted(rejected.dropna())}] + ([{"issuer_cik": None}] if rejected.isna().any() else [])
    return sorted(
        {accession for where in wheres for accession in context.store.distinct(Tables.insider_transactions, "accession_number", where=where)}
    )


def _screen_stored_rows(context: Context, universe: Sequence[str], identity: Identity, chunk: int = 2_000) -> tuple[int, int]:
    """Re-adjudicate stored rows against today's universe; quarantine then delete the rejects
    (the upsert cannot remove rows). Returns `(quarantined, deleted)`.

    Exhaustive over the table, unlike the parse screen. One accession shares one `issuer_cik` and
    so one verdict, which lets the delete key on `accession_number` alone.
    """
    accessions = _rejected_stored_accessions(context, universe, identity)
    if not accessions:
        return 0, 0

    quarantined = deleted = 0
    for start in range(0, len(accessions), chunk):
        batch = accessions[start : start + chunk]
        rows = context.store.load(Tables.insider_transactions, where={"accession_number": batch}, optional=True)
        if rows is None or rows.empty:
            continue
        rejected = insider_verdicts(rows, universe, identity)
        rejected = rejected[rejected["reject_reason"].notna()]
        if rejected.empty:  # re-adjudicated clean on the full row: leave it
            continue
        quarantined += context.store.save(Tables.insider_transactions_quarantine, quarantine_frame(rejected))
        deleted += context.store.delete(Tables.insider_transactions, where={"accession_number": sorted(rejected["accession_number"].unique())})
    logger.info("insider: stored-row sweep -- quarantined %d row(s) over %d accession(s), deleted %d", quarantined, len(accessions), deleted)
    return quarantined, deleted


def fetch_insider_transactions(context: Context, tickers: list[str], years_history: int = 15, reparse: bool = False) -> int:
    """Download (cached) the insider data sets, parse and screen each pending quarter, upsert
    `insider_transactions`, `insider_footnotes` and the quarantine, then sweep stored rows.
    Returns the number of transaction rows upserted.

    `reparse` re-reads every quarter the source has back to `SEC_INSIDER_FIRST_YEAR`, even those
    already stored, so a parse change reaches the oldest rows too.
    """
    identity = load_identity(context)
    cache = cache_dir(context, context.config.local.paths.insider_transactions)
    span = (pd.Timestamp.today().year - SEC_INSIDER_FIRST_YEAR + 1) if reparse else years_history + 1
    quarters = quarter_periods(span, SEC_INSIDER_FIRST_YEAR)
    pending = pending_periods(context, cache, Tables.insider_transactions, quarters, tickers, reparse=reparse, column="quarter")

    saved = notes_saved = quarantined = 0
    for quarter in tqdm(pending, desc="insider data sets"):
        url_template = SEC_INSIDER_URL_NEW_TEMPLATE if int(quarter[:4]) >= SEC_INSIDER_SWAP_YEAR else SEC_INSIDER_URL_TEMPLATE
        path = ensure_zip(context, cache / f"{quarter}.zip", url_template.format(quarter=quarter), label=f"insider {quarter}", log=logger)
        tables = _read_tables(path) if path is not None else None
        if tables is None:
            continue
        df_kept, df_quarantine, df_notes = _parse_quarter(tables, quarter, tickers, identity)
        if not df_quarantine.empty:
            quarantined += context.store.save(Tables.insider_transactions_quarantine, df_quarantine)
        if not df_kept.empty:
            saved += context.store.save(Tables.insider_transactions, df_kept)
        if not df_notes.empty:
            notes_saved += context.store.save(Tables.insider_footnotes, df_notes)

    swept, deleted = _screen_stored_rows(context, tickers, identity)
    quarantined += swept
    mark_processed(cache, Tables.insider_transactions, tickers)
    logger.info(
        "insider_transactions: upserted %d rows (+%d footnotes) over %d quarters (%s -> %s); quarantined %d, deleted %d",
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
