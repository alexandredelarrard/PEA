"""Officer / director / 10%-owner transactions from the SEC quarterly Insider Transactions Data
Sets (bulk TSV zips, Forms 3/4/5).

SUBMISSION, (NON)DERIV_TRANS and every REPORTINGOWNER row are mapped to the canonical string frames
(`insider_common.INSIDER_FIELDS`), typed by `build_insider_frame` and screened CIK-first. EDGAR is
authoritative: a filing EDGAR already stored only gets the zip `quarter`; every other filing is
upserted with `source='zip'`, one row per (accession, table, `row_sequence`); FOOTNOTES follow the
kept accessions. Each quarter logs how many of its filings EDGAR missed.

`download_insider_transactions` caches configured quarter units (the identity build reads them), downloading
missing or invalid ZIP containers. The shared archive worklist selects only units absent from the table normally;
Context full/recovery may replay completed units inside the same configured bounds. The identity sync owns lineage
restamping and scoped reconciliation; archive ingestion never follows a parse with a global stored-row sweep.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    SEC_INSIDER_SWAP_YEAR,
    SEC_INSIDER_URL_NEW_TEMPLATE,
    SEC_INSIDER_URL_TEMPLATE,
)
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    ZipRead,
    cache_dir,
    ensure_zip,
    quarter_periods,
    read_zip_tables,
)
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.resume import archive_worklist
from src.data_extract.utils.edgar_extract.insider_common import (
    BULK_DATE_FORMATS,
    INSIDER_COLUMNS,
    INSIDER_FIELDS,
    INSIDER_KEY,
    LINEAGE_COLUMNS,
    OWNER_STRING_COLUMNS,
    build_insider_frame,
    empty_footnotes,
    exclusion_rows,
    filter_footnotes,
    log_exclusions,
    screen_insider_rows,
    screened_accessions,
    stamp_lineage,
    top_counts,
)
from src.data_extract.utils.edgar_extract.insider_store import patch_insider_source, save_insider_source
from src.data_store.schema import Tables
from src.utils.data_policy import window_policy
from src.utils.filer_tables import FilerTable, removal_records
from src.utils.string import pad_cik_series

logger = logging.getLogger(__name__)

#: Fields compared on rows both sources hold; zip numbers carry two decimals, hence the tolerance.
_COMPARED_FIELDS = ("transaction_code", "transaction_date", "shares", "price_per_share", "shares_owned_after", "owner_cik")
_NUMBER_FIELDS = frozenset({"shares", "price_per_share", "shares_owned_after"})
_DATE_FIELDS = frozenset({"transaction_date"})
_NUMBER_TOLERANCE = 0.005
#: Stored columns the lineage re-stamp reads: the key, what the stamp derives from, and the stamp itself.
_RESTAMP_COLUMNS = [
    *INSIDER_KEY,
    "ticker",
    "issuer_cik",
    "owner_cik",
    "document_type",
    "transaction_code",
    "transaction_date",
    "period_of_report",
    "filing_date",
    "original_submission_date",
    "security_title",
    *LINEAGE_COLUMNS,
]
_CO_REGISTRANT_SPEC = FilerTable(Tables.insider_transactions, "issuer_cik", "filing_date", "accession_number")

#: SUBMISSION / REPORTINGOWNER columns `INSIDER_FIELDS` maps; remarks and addresses are never read.
_MEMBER_COLUMNS = {
    scope: frozenset({"ACCESSION_NUMBER", *(field.bulk for field in INSIDER_FIELDS if field.scope == scope and field.bulk)})
    for scope in ("filing", "owner")
}
#: All seven payload members are required before a quarter can support inventory proof.
_ZIP_SPECS = {
    "SUBMISSION.TSV": ZipRead(usecols=_MEMBER_COLUMNS["filing"]),
    "REPORTINGOWNER.TSV": ZipRead(usecols=_MEMBER_COLUMNS["owner"]),
    "NONDERIV_TRANS.TSV": ZipRead(),
    "DERIV_TRANS.TSV": ZipRead(),
    "FOOTNOTES.TSV": ZipRead(),
    "NONDERIV_HOLDING.TSV": ZipRead(),
    "DERIV_HOLDING.TSV": ZipRead(),
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


def _footnote_ids(cells: pd.Series) -> str:
    """The distinct footnote ids of one line's `*_FN` cells, in first-seen order (cells may hold `F1, F2` or `[F1]`)."""
    return ",".join(dict.fromkeys(part.strip() for cell in cells for part in cell.replace("[", "").replace("]", "").split(",") if part.strip()))


def _transaction_strings(df: pd.DataFrame | None, sk_col: str, security_type: str) -> pd.DataFrame:
    """One string row per keyed (non-)derivative transaction line. `row_sequence` is the rank of the
    numeric SEC transaction id inside the accession's table, which is the filing's XML order;
    a missing, invalid or repeated id refuses the quarter."""
    if df is None or df.empty or "ACCESSION_NUMBER" not in df.columns:
        return pd.DataFrame()
    sk = pd.to_numeric(_col(df, sk_col), errors="coerce")
    if sk.isna().any() or sk.le(0).any() or sk.mod(1).ne(0).any() or df.assign(_sk=sk).duplicated(["ACCESSION_NUMBER", "_sk"]).any():
        raise ValueError(f"{security_type}: missing, invalid or repeated source row key")
    frame = df
    if security_type.endswith("_holding"):
        frame = df.rename(columns={"EXERCISE_DATE": "EXCERCISE_DATE", "VALU_OWND_FOLWNG_TRANS": "TRANS_TOTAL_VALUE"})
    note_columns = [column for column in df if isinstance(column, str) and column.endswith("_FN")]
    footnotes = df[note_columns].fillna("").astype(str).agg(_footnote_ids, axis=1) if note_columns else ""
    df_lines = _member_strings(frame, "transaction").assign(security_type=security_type, _sk=sk, footnote_ids=footnotes)
    sequence = df_lines.groupby("accession_number")["_sk"].rank(method="first").astype("int64")
    return df_lines.assign(row_sequence=sequence).drop(columns="_sk")


def extract_bulk_strings(
    sub: pd.DataFrame,
    own: pd.DataFrame,
    nonderiv: pd.DataFrame,
    deriv: pd.DataFrame,
    nonderiv_holding: pd.DataFrame | None = None,
    deriv_holding: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SUBMISSION + (NON)DERIV_TRANS -> transaction string rows, and REPORTINGOWNER -> one owner
    string row per reporting owner of those accessions."""
    empty = (pd.DataFrame(), pd.DataFrame(columns=OWNER_STRING_COLUMNS))
    if sub is None or sub.empty:
        return empty
    df_trans = pd.concat(
        [
            _transaction_strings(nonderiv, "NONDERIV_TRANS_SK", "nonderiv"),
            _transaction_strings(deriv, "DERIV_TRANS_SK", "deriv"),
            _transaction_strings(nonderiv_holding, "NONDERIV_HOLDING_SK", "nonderiv_holding"),
            _transaction_strings(deriv_holding, "DERIV_HOLDING_SK", "deriv_holding"),
        ],
        ignore_index=True,
    )
    if nonderiv_holding is not None and deriv_holding is not None:
        seen = set(df_trans.get("accession_number", pd.Series(dtype=str)))
        empty_accessions = sub.loc[~sub["ACCESSION_NUMBER"].isin(seen), "ACCESSION_NUMBER"]
        markers = pd.DataFrame({"accession_number": empty_accessions, "security_type": "_empty", "row_sequence": 0})
        df_trans = pd.concat([df_trans, markers], ignore_index=True)
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
    sub, own, nonderiv, deriv, notes, *holdings = tables
    accessions = screened_accessions(_member_strings(sub, "filing"), universe, identity)
    df_str, df_owners = extract_bulk_strings(*(_accession_rows(df, accessions) for df in (sub, own, nonderiv, deriv, *holdings)))
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
    """Read a complete source unit; None when invalid. Invalid content is logged and the file kept;
    only a corrupt archive is deleted for the next normal retry."""
    try:
        tables = read_zip_tables(path, _ZIP_SPECS, on_corrupt="delete")
        if not tables:
            raise ValueError("missing, ambiguous or unreadable required source member")
        _validate_bulk_tables(tables)
    except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
        logger.warning("%s: invalid original (%s); skipped, cached file kept; good stored rows retained", path.name, exc)
        return None
    return tuple(tables.values())


def _validate_bulk_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Account for every actual source key and accession before selecting economic rows."""
    submission = tables["SUBMISSION.TSV"]
    required = {"ACCESSION_NUMBER", "ISSUERCIK", "DOCUMENT_TYPE", "FILING_DATE"}
    if not required.issubset(submission) or submission.empty:
        raise ValueError("missing or empty submission inventory")
    accessions = submission["ACCESSION_NUMBER"]
    if accessions.isna().any() or accessions.duplicated().any() or accessions.astype(str).str.strip().eq("").any():
        raise ValueError("invalid or repeated submission accession")
    issuer = submission["ISSUERCIK"].astype("string")
    dates = pd.to_datetime(submission["FILING_DATE"], format="mixed", errors="coerce")
    if (
        not issuer.str.fullmatch(r"\d{1,10}").fillna(False).all()
        or dates.isna().any()
        or not submission["DOCUMENT_TYPE"].isin(["3", "3/A", "4", "4/A", "5", "5/A"]).all()
    ):
        raise ValueError("invalid submission issuer, form or filing date")
    for name, frame in tables.items():
        if "ACCESSION_NUMBER" not in frame or frame["ACCESSION_NUMBER"].isna().any() or not frame["ACCESSION_NUMBER"].isin(accessions).all():
            raise ValueError(f"{name}: missing or unlisted accession")
        _validate_bulk_member(name, frame)
    notes = tables["FOOTNOTES.TSV"]
    available = set(zip(notes["ACCESSION_NUMBER"], notes["FOOTNOTE_ID"], strict=True))
    for name in ("NONDERIV_TRANS.TSV", "DERIV_TRANS.TSV", "NONDERIV_HOLDING.TSV", "DERIV_HOLDING.TSV"):
        _validate_bulk_notes(tables[name], available)


def _validate_bulk_member(name: str, frame: pd.DataFrame) -> None:
    """Source keys are mandatory even in a header-only member; a footnote's text may be empty."""
    if name == "FOOTNOTES.TSV":
        if (
            not {"FOOTNOTE_ID", "FOOTNOTE_TXT"}.issubset(frame)
            or frame["FOOTNOTE_ID"].isna().any()
            or frame.duplicated(["ACCESSION_NUMBER", "FOOTNOTE_ID"]).any()
        ):
            raise ValueError("invalid footnote key")
        return
    if name == "REPORTINGOWNER.TSV":
        if "RPTOWNERCIK" not in frame or not frame["RPTOWNERCIK"].astype("string").str.fullmatch(r"\d{1,10}").fillna(False).all():
            raise ValueError("invalid reporting owner")
        return
    if name == "SUBMISSION.TSV":
        return
    key = name.removesuffix(".TSV") + "_SK"
    if not {key, "SECURITY_TITLE"}.issubset(frame):
        raise ValueError(f"{name}: missing required row fields")
    _transaction_strings(frame, key, name.removesuffix(".TSV").lower())
    _validate_bulk_numbers(frame)


def _validate_bulk_numbers(frame: pd.DataFrame) -> None:
    """Preserve absent operands, but refuse present invalid numbers before typed coercion."""
    columns = {field.bulk for field in INSIDER_FIELDS if field.kind == "number" and field.bulk}
    columns.add("VALU_OWND_FOLWNG_TRANS")
    for column in sorted(columns & set(frame)):
        text = frame[column].dropna().astype(str).str.strip()
        text = text.loc[text.ne("")].str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        if not text.map(lambda value: isfinite(float(str(value)))).all():
            raise ValueError(f"nonfinite bulk numeric field {column}")


def _validate_bulk_notes(frame: pd.DataFrame, available: set[tuple[str, str]]) -> None:
    """A referenced note must exist in the same actual filing payload."""
    for column in (value for value in frame if isinstance(value, str) and value.endswith("_FN")):
        for accession, value in zip(frame["ACCESSION_NUMBER"], frame[column], strict=True):
            references = [] if pd.isna(value) else str(value).replace("[", "").replace("]", "").split(",")
            if any((str(accession), ref.strip()) not in available for ref in references if ref.strip()):
                raise ValueError("unresolved source footnote reference")


def _read_quarter(
    path: Path, quarter: str, universe: Sequence[str], identity: Identity, fetched_at: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    tables = _read_tables(path)
    if tables is None:
        return None
    try:
        return _parse_quarter(tables, quarter, universe, identity, fetched_at)
    except ValueError as exc:
        logger.warning("%s: invalid payload (%s); skipped, cached file kept; good stored rows retained", path.name, exc)
        return None


def store_zip_quarter(context: Context, quarter: str, df_kept: pd.DataFrame, df_notes: pd.DataFrame) -> int:
    """Commit each parsed ZIP accession and its notes together, preserving current EDGAR authority."""
    if df_kept.empty:
        return 0
    counts = save_insider_source(context, df_kept.assign(quarter=quarter), df_notes)
    return counts[Tables.insider_transactions]


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


def zip_urls(quarter: str) -> tuple[str, str]:
    """Both SEC hosting paths for one quarter's zip, the likelier first: the new path from
    `SEC_INSIDER_SWAP_YEAR` on, else the old one. The SEC moves quarters between the two paths, so
    `ensure_zip` falls back to the other on a miss."""
    old, new = (template.format(quarter=quarter) for template in (SEC_INSIDER_URL_TEMPLATE, SEC_INSIDER_URL_NEW_TEMPLATE))
    return (new, old) if int(quarter[:4]) >= SEC_INSIDER_SWAP_YEAR else (old, new)


def download_insider_transactions(context: Context) -> list[str]:
    """Cache every quarterly Form 3/4/5 zip since the table's source start; returns the quarters cached.

    A cached zip is never re-downloaded, and an unpublished quarter is skipped. No parse, no identity.
    """
    cache = cache_dir(context, context.config.local.paths.insider_transactions)
    policy = window_policy(context.config, Tables.insider_transactions, context.run.as_of)
    if policy is None:
        return []
    quarters = quarter_periods(policy.requested)
    cached = [
        quarter
        for quarter in tqdm(quarters, desc="insider data sets (download)")
        if ensure_zip(context, cache / f"{quarter}.zip", zip_urls(quarter), label=f"insider {quarter}") is not None
    ]
    logger.info("insider download: %d of %d quarter zip(s) cached", len(cached), len(quarters))
    return cached


def _keep_keys(parsed: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], keys: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retain authorized tickers and their same-accession footnotes after whole-universe identity screening."""
    df_kept, df_rejected, df_notes = parsed
    wanted = set(keys)
    df_kept = df_kept[df_kept["ticker"].isin(wanted)] if "ticker" in df_kept.columns else df_kept
    df_rejected = df_rejected[df_rejected["claimed_ticker"].isin(wanted)] if "claimed_ticker" in df_rejected.columns else df_rejected
    df_notes = filter_footnotes(df_notes, set(df_kept["accession_number"].dropna().unique())) if not df_kept.empty else empty_footnotes()
    return df_kept, df_rejected, df_notes


def fetch_insider_transactions(context: Context, tickers: list[str]) -> int:
    """Ingest missing quarterly units or explicit bounded replays, preserving daily EDGAR authority."""
    policy = window_policy(context.config, Tables.insider_transactions, context.run.as_of)
    if policy is None:
        return 0
    quarters = quarter_periods(policy.requested)
    if not quarters:
        return 0
    work = archive_worklist(context, (Tables.insider_transactions,), quarters, tickers)
    units = list(work.units())
    if not units:
        return 0
    identity = load_identity(context)
    cache = cache_dir(context, context.config.local.paths.insider_transactions)
    rescanned = set(work.rescan)
    saved = notes_saved = 0
    excluded: list[pd.DataFrame] = []
    fetched_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for quarter, keys in tqdm(units, desc="insider data sets"):
        path = ensure_zip(context, cache / f"{quarter}.zip", zip_urls(quarter), label=f"insider {quarter}")
        parsed = _read_quarter(path, quarter, work.screen, identity, fetched_at) if path is not None else None
        if parsed is None:
            continue
        # Screened against the whole universe, so a row of another universe ticker is never rejected;
        # a rescanned quarter then keeps only its keys' rows.
        df_kept, df_rejected, df_notes = _keep_keys(parsed, keys)
        excluded.append(exclusion_rows(df_rejected))
        if quarter not in rescanned:
            report_zip_quarter(context, quarter, df_kept)
        saved += store_zip_quarter(context, quarter, df_kept, df_notes)
        notes_saved += len(df_notes)
    log_exclusions(f"zip run ({len(units)} quarter(s))", excluded)

    logger.info(
        "insider_transactions: saved %d zip row(s) (+%d footnotes) over %d missing + %d recovery unit(s) of %s -> %s",
        saved,
        notes_saved,
        len(work.pending),
        len(work.rescan),
        quarters[0],
        quarters[-1],
    )
    return saved


def _changed_stamps(df_rows: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """The key and lineage columns of the rows whose `economic_date` or `lineage_role` differs from the stored stamp."""
    old_day = pd.to_datetime(df_rows["economic_date"], errors="coerce")
    new_day = pd.to_datetime(df_new["economic_date"], errors="coerce")
    same_day = (old_day.eq(new_day) | (old_day.isna() & new_day.isna())).fillna(False)
    old_role, new_role = df_rows["lineage_role"].astype("string"), df_new["lineage_role"].astype("string")
    same_role = (old_role.eq(new_role) | (old_role.isna() & new_role.isna())).fillna(False)
    changed = ~(same_day & same_role).astype(bool)
    return df_new.loc[changed, [*INSIDER_KEY, "economic_date", "lineage_role"]]


def restamp_insider_lineage(context: Context, tickers: Sequence[str], *, identity: Identity) -> list[dict]:
    """Repair the requested tickers inside configured filing dates; return co-registrant removals.

    Completed source payloads are reused. Older or wider lineage repairs require an explicit scoped run.
    """
    policy = window_policy(context.config, Tables.insider_transactions, context.run.as_of)
    if not tickers or policy is None or not context.store.exists(Tables.insider_transactions):
        return []
    present = set(context.store.columns(Tables.insider_transactions))
    if "lineage_role" not in present:
        context.log.warning("insider lineage: `insider_transactions` has no lineage columns yet (apply the sql/schema.sql block); re-stamp skipped")
        return []
    targets = sorted(set(tickers))
    columns = [column for column in _RESTAMP_COLUMNS if column in present]
    records: list[dict] = []
    restamped = 0
    for df_rows in context.store.iter_load(
        Tables.insider_transactions,
        columns=columns,
        where={"ticker": targets},
        since=policy.requested.start,
        until=policy.requested.end,
        date_col="filing_date",
    ):
        chunk_records, chunk_restamped = _restamp_rows(context, df_rows.reindex(columns=_RESTAMP_COLUMNS).reset_index(drop=True), identity)
        records += chunk_records
        restamped += chunk_restamped
    context.log.info(
        "insider lineage: re-stamped %d row(s) over %d ticker(s); %d co-registrant row(s) purged",
        restamped,
        len(targets),
        sum(record["rows"] for record in records),
    )
    return records


def _restamp_rows(context: Context, df_rows: pd.DataFrame, identity: Identity) -> tuple[list[dict], int]:
    """Purge one chunk's co-registrant rows and rewrite the stamps that changed; `(removal records, rows re-stamped)`."""
    co_registrant = pad_cik_series(df_rows["issuer_cik"]).isin(identity.co_registrant_ciks)
    df_purge = df_rows[co_registrant]
    records = removal_records(Tables.insider_transactions.name, df_purge, _CO_REGISTRANT_SPEC) if not df_purge.empty else []
    df_kept = df_rows[~co_registrant]
    df_changed = _changed_stamps(df_kept, stamp_lineage(df_kept, identity))
    if not df_changed.empty or not df_purge.empty:
        patch_insider_source(context, df_changed, df_purge[[*Tables.insider_transactions.pk, "ticker", "issuer_cik"]])
    return records, len(df_changed)
