"""Officer / director / 10%-owner transactions from the SEC quarterly Insider Transactions Data
Sets (bulk TSV zips, Forms 3/4/5).

SUBMISSION, (NON)DERIV_TRANS and every REPORTINGOWNER row are mapped to the canonical string frames
(`insider_common.INSIDER_FIELDS`), typed by `build_insider_frame` and screened CIK-first. EDGAR is
authoritative: a filing EDGAR already stored only gets the zip `quarter`; every other filing is
upserted with `source='zip'`, one row per (accession, table, `row_sequence`); FOOTNOTES follow the
kept accessions. Each quarter logs how many of its filings EDGAR missed.

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
from src.data_extract.utils.institutionals.insider_common import (
    BULK_DATE_FORMATS,
    INSIDER_COLUMNS,
    INSIDER_FIELDS,
    INSIDER_KEY,
    OWNER_STRING_COLUMNS,
    accession_batches,
    build_insider_frame,
    empty_footnotes,
    exclusion_rows,
    filter_footnotes,
    insider_verdicts,
    log_exclusions,
    screen_insider_rows,
    screened_accessions,
    top_counts,
)
from src.data_store.schema import Tables
from src.utils.string import normalise_ticker
from src.utils.universe import load_universe_tickers

logger = logging.getLogger(__name__)

#: Fields compared on rows both sources hold; zip numbers carry two decimals, hence the tolerance.
_COMPARED_FIELDS = ("transaction_code", "transaction_date", "shares", "price_per_share", "shares_owned_after", "owner_cik")
_NUMBER_FIELDS = frozenset({"shares", "price_per_share", "shares_owned_after"})
_DATE_FIELDS = frozenset({"transaction_date"})
_NUMBER_TOLERANCE = 0.005

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
def _member_strings(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """The canonical string columns of one member (`scope`), keyed on `accession_number`."""
    fields = [field for field in INSIDER_FIELDS if field.scope == scope and field.bulk]
    return pd.DataFrame({"accession_number": _col(df, "ACCESSION_NUMBER"), **{field.name: _col(df, field.bulk or "") for field in fields}})


def _transaction_strings(df: pd.DataFrame | None, sk_col: str, security_type: str) -> pd.DataFrame:
    """One string row per keyed (non-)derivative transaction line. `row_sequence` is the rank of the
    numeric SEC transaction id inside the accession's table, which is the filing's XML order;
    a line without an id cannot be keyed and is dropped."""
    if df is None or df.empty or "ACCESSION_NUMBER" not in df.columns:
        return pd.DataFrame()
    sk = pd.to_numeric(_col(df, sk_col), errors="coerce")
    df_lines = _member_strings(df, "transaction").assign(security_type=security_type, _sk=sk)[sk.notna()]
    sequence = df_lines.groupby("accession_number")["_sk"].rank(method="first").astype("int64")
    return df_lines.assign(row_sequence=sequence).drop(columns="_sk")


def extract_bulk_strings(sub: pd.DataFrame, own: pd.DataFrame, nonderiv: pd.DataFrame, deriv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SUBMISSION + (NON)DERIV_TRANS -> transaction string rows, and REPORTINGOWNER -> one owner
    string row per reporting owner of those accessions."""
    empty = (pd.DataFrame(), pd.DataFrame(columns=OWNER_STRING_COLUMNS))
    if sub is None or sub.empty:
        return empty
    df_trans = pd.concat(
        [_transaction_strings(nonderiv, "NONDERIV_TRANS_SK", "nonderiv"), _transaction_strings(deriv, "DERIV_TRANS_SK", "deriv")], ignore_index=True
    )
    if df_trans.empty:
        return empty
    df_str = df_trans.merge(_member_strings(sub, "filing"), on="accession_number", how="inner")
    df_owners = _member_strings(own, "owner")
    return df_str, df_owners[df_owners["accession_number"].isin(set(df_str["accession_number"]))].reset_index(drop=True)


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
    # `isin` is much faster on an object copy than on the Arrow-backed str column.
    return df[df["ACCESSION_NUMBER"].astype(object).isin(accessions)]


def _parse_quarter(
    tables: tuple[pd.DataFrame, ...], quarter: str, universe: Sequence[str], identity: Identity, fetched_at: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One quarter's members -> (kept transactions, rejected in-scope rows, footnotes of kept
    accessions). Members are first cut to the accessions the screen can keep or reject in scope;
    rows are stamped `source='zip'`, `quarter` and `fetched_at`."""
    sub, own, nonderiv, deriv, notes = tables
    accessions = screened_accessions(_member_strings(sub, "filing"), universe, identity)
    df_str, df_owners = extract_bulk_strings(*(_accession_rows(df, accessions) for df in (sub, own, nonderiv, deriv)))
    df_built = build_insider_frame(df_str, df_owners, date_formats=BULK_DATE_FORMATS)
    df_kept, df_rejected = screen_insider_rows(df_built.assign(source="zip", quarter=quarter, fetched_at=fetched_at), universe, identity)
    if df_kept.empty:
        return df_kept, df_rejected, empty_footnotes()
    df_notes = filter_footnotes(_footnote_strings(notes), set(df_kept["accession_number"].dropna().unique()))
    return df_kept[[column for column in INSIDER_COLUMNS if column in df_kept.columns]], df_rejected, df_notes


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


def _sweep_universe(context: Context, tickers: Sequence[str]) -> list[str]:
    """The loaded analysis universe when the normalised `tickers` equal it, else `[]` (sweep skipped).
    The sweep adjudicates against that loaded list, never the raw run tickers: on a `--tickers`
    subset every other company's rows would read as rejects."""
    universe = load_universe_tickers(context)
    if universe and {normalise_ticker(ticker) for ticker in tickers} == set(universe):
        return universe
    logger.info("insider: stored-row sweep skipped -- the run's %d ticker(s) are not the %d-ticker universe", len(tickers), len(universe))
    return []


def _screen_stored_rows(context: Context, universe: Sequence[str], identity: Identity) -> int:
    """Re-adjudicate stored rows against today's universe, delete the rejects (the upsert cannot
    remove rows) and log their exclusion summary. Returns the rows deleted.

    Exhaustive over the table, unlike the parse screen. One accession shares one `issuer_cik` and
    so one verdict, which lets the delete key on `accession_number` alone.
    """
    accessions = _rejected_stored_accessions(context, universe, identity)
    excluded: list[pd.DataFrame] = []
    deleted = 0
    for batch in accession_batches(accessions):
        df_rows = context.store.load(
            Tables.insider_transactions,
            columns=["accession_number", "transaction_code", "ticker", "issuer_cik"],
            where={"accession_number": batch},
            optional=True,
        )
        if df_rows is None:
            continue
        df_scored = insider_verdicts(df_rows, universe, identity)
        df_rejected = df_scored[df_scored["reject_reason"].notna()]
        if df_rejected.empty:  # re-adjudicated clean on the full row: leave it
            continue
        excluded.append(exclusion_rows(df_rejected))
        deleted += context.store.delete(Tables.insider_transactions, where={"accession_number": sorted(df_rejected["accession_number"].unique())})
    log_exclusions(logger, "stored-row sweep", excluded)
    return deleted


def _stored_rows(context: Context, accessions: Sequence[str], columns: list[str]) -> pd.DataFrame:
    """`columns` of the stored rows of `accessions`, read in `accession_batches`."""
    frames: list[pd.DataFrame] = []
    for batch in accession_batches(accessions):
        df_batch = context.store.load(Tables.insider_transactions, columns=columns, where={"accession_number": batch}, optional=True)
        if df_batch is not None:
            frames.append(df_batch)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def store_zip_quarter(context: Context, quarter: str, df_kept: pd.DataFrame) -> int:
    """Save one parsed quarter. A filing EDGAR already stored gets only `quarter`: its stored keys
    are re-saved with it, so the merge-upsert changes nothing else and adds no row. Every other
    filing is upserted from the zip. Returns the zip rows saved."""
    if df_kept.empty:
        return 0
    df_stored = _stored_rows(context, sorted(df_kept["accession_number"].unique()), [*INSIDER_KEY, "source"])
    edgar_accessions = set(df_stored.loc[df_stored["source"].eq("edgar"), "accession_number"])
    df_stamp = df_stored.loc[df_stored["accession_number"].isin(edgar_accessions), INSIDER_KEY].assign(quarter=quarter)
    if not df_stamp.empty:
        context.store.save(Tables.insider_transactions, df_stamp)
    df_zip = df_kept[~df_kept["accession_number"].isin(edgar_accessions)]
    return context.store.save(Tables.insider_transactions, df_zip) if not df_zip.empty else 0


def _same(left: pd.Series, right: pd.Series, field: str) -> pd.Series:
    """Per-row agreement of one compared field; two missing values agree."""
    both_missing = left.isna() & right.isna()
    if field in _NUMBER_FIELDS:
        gap = (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).abs()
        return both_missing | gap.le(_NUMBER_TOLERANCE)
    if field in _DATE_FIELDS:
        return both_missing | pd.to_datetime(left, errors="coerce").eq(pd.to_datetime(right, errors="coerce"))
    return both_missing | left.astype("string").eq(right.astype("string")).fillna(False)


def _report_shared_rows(quarter: str, df_zip: pd.DataFrame, df_edgar: pd.DataFrame) -> None:
    """INFO: EDGAR-only filings, and on the filings both sources hold, the rows whose compared
    fields disagree plus the key rows present on one side only."""
    zip_accessions = set(df_zip["accession_number"])
    shared = zip_accessions & set(df_edgar["accession_number"])
    columns = [*INSIDER_KEY, *_COMPARED_FIELDS]
    df_pair = df_zip.loc[df_zip["accession_number"].isin(shared), columns].merge(
        df_edgar.loc[df_edgar["accession_number"].isin(shared), columns], on=INSIDER_KEY, how="outer", suffixes=("_zip", "_edgar"), indicator=True
    )
    df_both = df_pair[df_pair["_merge"].eq("both")]
    agree = pd.Series(True, index=df_both.index)
    for field in _COMPARED_FIELDS:
        agree &= _same(df_both[f"{field}_zip"], df_both[f"{field}_edgar"], field)
    n_mismatched = int((~agree).sum())
    logger.info(
        "insider %s: %d EDGAR-only filing(s); %d / %d shared row(s) mismatched (%.1f%%) on %s; %d key row(s) on one side only",
        quarter,
        len(set(df_edgar["accession_number"]) - zip_accessions),
        n_mismatched,
        len(df_both),
        100.0 * n_mismatched / len(df_both) if len(df_both) else 0.0,
        ", ".join(_COMPARED_FIELDS),
        int(df_pair["_merge"].ne("both").sum()),
    )


def report_zip_quarter(context: Context, quarter: str, df_kept: pd.DataFrame) -> None:
    """Log how many of the quarter's filings EDGAR missed: N = zip filings filed on or after the
    quarter's earliest EDGAR filing date, X = those EDGAR did not store (WARNING), then the shared
    row agreement (INFO). A quarter without EDGAR rows logs one INFO line only."""
    period = pd.Period(quarter.upper(), freq="Q")
    df_edgar = context.store.load(
        Tables.insider_transactions,
        columns=[*INSIDER_KEY, "filing_date", *_COMPARED_FIELDS],
        where={"source": "edgar"},
        since=period.start_time,
        until=period.end_time.normalize(),
        date_col="filing_date",
        optional=True,
    )
    if df_edgar is None:
        logger.info("insider %s: loaded from zip (no EDGAR coverage)", quarter)
        return
    df_zip = df_kept.reindex(columns=[*INSIDER_KEY, "filing_date", "ticker", *_COMPARED_FIELDS])
    first_filed = pd.to_datetime(df_edgar["filing_date"]).min()
    df_window = df_zip[pd.to_datetime(df_zip["filing_date"]).ge(first_filed)].drop_duplicates("accession_number")
    df_missing = df_window[~df_window["accession_number"].isin(set(df_edgar["accession_number"]))]
    logger.warning(
        "insider %s: %d / %d filings missing from EDGAR (%.1f%%), added from zip; top: %s",
        quarter,
        len(df_missing),
        len(df_window),
        100.0 * len(df_missing) / len(df_window) if len(df_window) else 0.0,
        top_counts(df_missing["ticker"], 5) or "none",
    )
    _report_shared_rows(quarter, df_zip, df_edgar)


def fetch_insider_transactions(context: Context, tickers: list[str], years_history: int = 15, reparse: bool = False) -> int:
    """Download (cached) the insider data sets and ingest each pending quarter with
    `store_zip_quarter` and its `report_zip_quarter` lines, save the kept footnotes, log one
    identity-exclusion summary for the run, then, on a full-universe run only, sweep stored rows
    against the loaded universe. Returns the zip rows saved.

    `reparse` re-reads every quarter the source has back to `SEC_INSIDER_FIRST_YEAR`, even those
    already stored, so a parse change reaches the oldest rows too. The run manifest is left to the
    EDGAR ingest, the only writer of the `insider_transactions` completeness entry.
    """
    identity = load_identity(context)
    cache = cache_dir(context, context.config.local.paths.insider_transactions)
    span = (pd.Timestamp.today().year - SEC_INSIDER_FIRST_YEAR + 1) if reparse else years_history + 1
    quarters = quarter_periods(span, SEC_INSIDER_FIRST_YEAR)
    pending = pending_periods(context, cache, Tables.insider_transactions, quarters, tickers, reparse=reparse, column="quarter")

    saved = notes_saved = 0
    excluded: list[pd.DataFrame] = []
    fetched_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for quarter in tqdm(pending, desc="insider data sets"):
        url_template = SEC_INSIDER_URL_NEW_TEMPLATE if int(quarter[:4]) >= SEC_INSIDER_SWAP_YEAR else SEC_INSIDER_URL_TEMPLATE
        path = ensure_zip(context, cache / f"{quarter}.zip", url_template.format(quarter=quarter), label=f"insider {quarter}", log=logger)
        tables = _read_tables(path) if path is not None else None
        if tables is None:
            continue
        df_kept, df_rejected, df_notes = _parse_quarter(tables, quarter, tickers, identity, fetched_at)
        excluded.append(exclusion_rows(df_rejected))
        report_zip_quarter(context, quarter, df_kept)
        saved += store_zip_quarter(context, quarter, df_kept)
        if not df_notes.empty:
            notes_saved += context.store.save(Tables.insider_footnotes, df_notes)
    log_exclusions(logger, f"zip run ({len(pending)} quarter(s))", excluded)

    sweep_universe = _sweep_universe(context, tickers)
    deleted = _screen_stored_rows(context, sweep_universe, identity) if sweep_universe else 0
    mark_processed(cache, Tables.insider_transactions, tickers)
    logger.info(
        "insider_transactions: saved %d zip row(s) (+%d footnotes) over %d pending quarter(s) of %s -> %s; sweep deleted %d",
        saved,
        notes_saved,
        len(pending),
        quarters[0],
        quarters[-1],
        deleted,
    )
    return saved
