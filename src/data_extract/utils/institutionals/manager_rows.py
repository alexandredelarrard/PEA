"""
manager_rows.py (src/data_extract/utils/institutionals/manager_rows.py)
-----------------------------------------------------------------------
Pure frame logic of the `sec_13f_manager_details` extract: filings (one per 13F-HR(/A) accession, from a data set's cover
pages or from marker rows joined to the EDGAR master index) become one row per (cik, period); rows are merged with the
stored rows so only differences are written. Only what the SEC gives is kept: no category, no parent link, no 13F-NT.
A filing whose period is not a quarter end (a typo in the filing) is placed in the latest quarter end on or before its
filing date and keeps the period it was filed under in `source_period`.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: The columns this extract owns, in table order; the profile columns belong to `data_aggregate`.
EXTRACT_COLUMNS: tuple[str, ...] = (
    "cik",
    "period",
    "manager_name",
    "state_or_country",
    "report_type",
    "sec_file_number",
    "accession_number",
    "first_filed_date",
    "source",
    "source_period",
)
#: `source` of a row whose name comes from the EDGAR master index; any other value is the data-set ZIP name.
INDEX_SOURCE = "edgar_index"
_KEY = ["cik", "period"]
_REPORT_TYPES = {"13F HOLDINGS REPORT": "holdings", "13F COMBINATION REPORT": "combination"}
_FILING_COLUMNS = [
    "cik",
    "period",
    "accession_number",
    "filing_date",
    "original",
    "manager_name",
    "state_or_country",
    "sec_file_number",
    "report_type",
    "source",
]


def zip_manager_filings(df_submissions: pd.DataFrame, df_cover: pd.DataFrame, source: str) -> pd.DataFrame:
    """Filings of one data set: `zip_submissions` rows (13F-HR(/A) only, period from the cover) joined to the cover page's
    name, state, file number and report type; `source` is the ZIP name."""
    covers = df_cover.assign(accession=df_cover["ACCESSION_NUMBER"].astype("string").str.strip()).drop_duplicates("accession").set_index("accession")
    df = df_submissions.join(covers, on="accession", how="left")
    text = df[["FILINGMANAGER_NAME", "FILINGMANAGER_STATEORCOUNTRY", "SECFILENUMBER"]].astype("string").apply(lambda column: column.str.strip())
    text = text.mask(text.isna() | (text == "")).astype(object)
    text = text.where(text.notna(), None)
    return pd.DataFrame(
        {
            "cik": df["cik"],
            "period": df["report_period"],
            "accession_number": df["accession"].astype(object),
            "filing_date": df["filing_date"],
            "original": ~df["is_amendment"].astype(bool),
            "manager_name": text["FILINGMANAGER_NAME"],
            "state_or_country": text["FILINGMANAGER_STATEORCOUNTRY"],
            "sec_file_number": text["SECFILENUMBER"],
            "report_type": df["REPORTTYPE"].astype("string").str.strip().str.upper().map(_REPORT_TYPES).astype(object),
            "source": source,
        }
    )[_FILING_COLUMNS]


def marker_manager_filings(df_markers: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
    """Filings from `sec13f_hr` marker rows (`cik, period, accession_number, filing_date, form`) named by the index rows
    (`cik, company, form, filed, accession`) of the same (cik, accession); the index carries no cover field."""
    names = df_index.rename(columns={"accession": "accession_number"})[["cik", "accession_number", "company", "filed"]]
    df = df_markers.merge(names.drop_duplicates(["cik", "accession_number"]), on=["cik", "accession_number"], how="left")
    name = df["company"].astype("string").str.strip()
    name = name.mask(name.isna() | (name == "")).astype(object)
    name = name.where(name.notna(), None)
    return pd.DataFrame(
        {
            "cik": df["cik"],
            "period": df["period"],
            "accession_number": df["accession_number"],
            "filing_date": df["filing_date"].fillna(df["filed"]),
            "original": df["form"].astype("string").str.strip().str.upper() == "13F-HR",
            "manager_name": name,
            "state_or_country": None,
            "sec_file_number": None,
            "report_type": None,
            "source": INDEX_SOURCE,
        }
    )[_FILING_COLUMNS]


def candidate_rows(df_filings: pd.DataFrame, floor: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """One row per (cik, period) from `df_filings`, plus the `original` flag of its accession.

    A period that is not a quarter end is snapped (`source_period` keeps it) and gives way to a quarter-end filing of its quarter.
    Dropped: a period before `floor`, a filing dated before its period end or after `until`, a nameless filing. ZIP evidence beats the
    index; the accession is the earliest original 13F-HR (else the earliest), `first_filed_date` the earliest filing date of the group."""
    # clean: the filings that can describe a stored row, each in a quarter end
    source_period = pd.to_datetime(df_filings["period"]).dt.normalize()
    filed = pd.to_datetime(df_filings["filing_date"]).dt.normalize()
    latest_quarter_end = filed + pd.Timedelta(days=1) + pd.offsets.QuarterEnd(-1)
    period = source_period.where(source_period.dt.is_quarter_end, latest_quarter_end)
    valid = (period >= floor) & (filed >= period) & (filed <= until) & df_filings["manager_name"].notna() & (df_filings["cik"] != "")
    if not valid.all():
        logger.warning(
            "13F managers: %d filing(s) dropped (period before %s, filed before the period end or after %s, or no name)",
            (~valid).sum(),
            floor.date(),
            until.date(),
        )
    df = df_filings.loc[valid].assign(period=period[valid], source_period=source_period[valid], filing_date=filed[valid])

    # merge: one filing per accession, the ZIP's evidence first
    df = df.assign(indexed=df["source"] == INDEX_SOURCE).sort_values("indexed", kind="stable").drop_duplicates(["cik", "accession_number"])

    # merge: a snapped filing competes only with the snapped filings of its quarter
    snapped = df["source_period"] != df["period"]
    df = df[~snapped | snapped.groupby([df["cik"], df["period"]]).transform("all")]

    # compute: the accession of each (cik, period) and its first filing date over every filing
    df = df.assign(original=df["original"].astype(bool)).sort_values(
        [*_KEY, "original", "filing_date", "accession_number"], ascending=[True, True, False, True, True]
    )
    first = df.groupby(_KEY, as_index=False).agg(first_filed_date=("filing_date", "min"))
    chosen = df.drop_duplicates(_KEY).merge(first, on=_KEY)
    return chosen[[*EXTRACT_COLUMNS, "original"]].reset_index(drop=True)


def rows_to_save(df_rows: pd.DataFrame, df_stored: pd.DataFrame | None) -> pd.DataFrame:
    """The extract columns of the rows that differ from `df_stored` (None: no stored row), ready to save.

    A candidate replaces the stored columns when the row is new, names the same accession, has an earlier first filing, or
    is an original from a ZIP over stored index evidence; index evidence never replaces a stored ZIP row. In every case
    `first_filed_date` is the earlier of the two dates, so it never moves later."""
    columns = list(EXTRACT_COLUMNS)
    if df_stored is None:
        return df_rows[columns].reset_index(drop=True)

    # clean: the stored rows with normalised keys and dates
    stored = df_stored[columns].assign(
        period=pd.to_datetime(df_stored["period"]).dt.normalize(),
        first_filed_date=pd.to_datetime(df_stored["first_filed_date"]).dt.normalize(),
        source_period=pd.to_datetime(df_stored["source_period"]).dt.normalize(),
    )
    data_cols = [c for c in columns if c not in _KEY]
    stored = stored.rename(columns={c: f"{c}_stored" for c in data_cols})

    # merge: each candidate beside its stored row
    joined = df_rows.merge(stored, on=_KEY, how="left")
    known = joined["accession_number_stored"].notna()
    from_zip = joined["source"] != INDEX_SOURCE
    downgrade = known & ~from_zip & (joined["source_stored"] != INDEX_SOURCE)
    replace = ~known | (
        ~downgrade
        & (
            (joined["accession_number"] == joined["accession_number_stored"])
            | (joined["first_filed_date"] < joined["first_filed_date_stored"])
            | (from_zip & joined["original"] & (joined["source_stored"] == INDEX_SOURCE))
        )
    )

    # compute: the merged columns and which rows differ from the stored ones
    merged = joined[_KEY].copy()
    for col in data_cols:
        merged[col] = joined[col].where(replace, joined[f"{col}_stored"])
    merged["first_filed_date"] = joined[["first_filed_date", "first_filed_date_stored"]].min(axis=1)
    same = pd.concat(
        [merged[col].astype("string").fillna("") == joined[f"{col}_stored"].astype("string").fillna("") for col in data_cols], axis=1
    ).all(axis=1)
    return merged.loc[~known | ~same, columns].reset_index(drop=True)
