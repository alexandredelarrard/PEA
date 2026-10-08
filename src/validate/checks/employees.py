"""Employee headcount check: per-ticker coverage of owned annual 10-K dates, status mix, provenance, identity,
basis switches and scope jumps in `fundamentals_employees`.

Owned filings are the cached EDGAR quarterly index (Parquet under `DATA_STORE`, no network) listed through
`entity_lineage` the way the fetcher lists annual forms: a CIK window's rows inside its seam-widened dates, the
roster CIK as one open window for an entity with none. A table still in the pre-component shape
(`ticker, as_of, employees`) is read as totals; the checks that need the new columns are reported as skipped.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from src.context import Context
from src.data_store.schema import Tables
from src.utils.filer_tables import PURGE_TABLES_BY_NAME, ListedWindow
from src.utils.string import normalise_ticker, pad_cik_series
from src.validate.checks.identity import _entity_ciks, _foreign_in_table, _from_old_shape, _lineage, _listed_windows
from src.validate.result import CheckResult, Finding

CHECK = "employees"
#: The fetcher's annual forms; a date with an original form is an original filing date.
ANNUAL_FORMS = ("10-K", "10-K/A", "10-K405")
ORIGINAL_FORMS = frozenset({"10-K", "10-K405"})
#: Mirrors the fetcher's decided statuses and bases (`fundamentals_employees.py`).
DECIDED_STATUSES = ("found", "not_disclosed", "image_only", "incorporated", "ambiguous", "unsupported")
BASES = ("total", "full_part", "full_time_only", "fte", "total_incl_contractors")
#: Stored component column -> its `source_quote` key.
COMPONENTS: Mapping[str, str] = {"employees_total": "total", "employees_full_time": "full_time", "employees_part_time": "part_time"}
PROVENANCE_COLUMNS = ("cik", "accession_number", "form", "source_document", "source_quote")
NEW_SHAPE_COLUMNS = frozenset({*COMPONENTS, "status", "basis", *PROVENANCE_COLUMNS})
#: A year-over-year |change| above this, within one basis, is a scope jump for review.
JUMP_THRESHOLD = 0.5
#: Days between two counted rows for the pair to read as year over year.
JUMP_GAP_DAYS = (180, 550)
_INDEX_SCHEMA = pa.schema([("cik", pa.int64()), ("company", pa.string()), ("form", pa.string()), ("filed", pa.date32()), ("accession", pa.string())])
_YEARS_HISTORY_DEFAULT = 31


@dataclass(frozen=True)
class EmployeesReport:
    """The check result plus its per-ticker and per-row detail frames."""

    result: CheckResult
    per_ticker: pd.DataFrame
    missing: pd.DataFrame
    stale: pd.DataFrame
    jumps: pd.DataFrame
    switches: pd.DataFrame


def index_dir(context: Context) -> Path:
    """The cached EDGAR quarterly index directory."""
    return Path(context.paths["DATA_STORE"]) / str(context.config.local.paths.sec_edgar_index)


def index_entries(directory: Path, ciks: Collection[str], since: pd.Timestamp | None) -> pd.DataFrame:
    """Cached index rows of `ANNUAL_FORMS` for `ciks` filed on or after `since`; `cik` padded, `filed` a Timestamp."""
    columns = ["cik", "company", "form", "filed", "accession"]
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    if not files or not ciks:
        return pd.DataFrame(columns=columns)
    expression = pc.field("cik").isin([int(c) for c in ciks]) & pc.field("form").isin(list(ANNUAL_FORMS))
    if since is not None:
        expression &= pc.field("filed") >= pa.scalar(since.date(), pa.date32())
    frame = ds.dataset([str(path) for path in files], format="parquet", schema=_INDEX_SCHEMA).to_table(filter=expression).to_pandas()
    frame["cik"] = pad_cik_series(frame["cik"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    return frame[columns]


def listing_windows(lineage: pd.DataFrame, roster: pd.DataFrame) -> dict[str, tuple[ListedWindow, ...]]:
    """`{ticker: seam-widened CIK windows}`; a roster ticker with none reads its roster CIK as one open window."""
    windows = dict(_listed_windows(lineage))
    for ticker, cik in zip(roster["ticker"], roster["cik"], strict=True):
        windows.setdefault(normalise_ticker(ticker), ((cik, None, None),))
    return windows


def owned_filings(entries: pd.DataFrame, tickers: Sequence[str], windows: Mapping[str, Sequence[ListedWindow]]) -> pd.DataFrame:
    """Index rows each ticker lists: its windows' CIKs inside their dates, one row per (ticker, accession)."""
    parts = []
    by_cik = dict(tuple(entries.groupby("cik", sort=False))) if not entries.empty else {}
    for ticker in tickers:
        for cik, lo, hi in windows.get(normalise_ticker(ticker), ()):
            rows = by_cik.get(cik)
            if rows is None:
                continue
            inside = pd.Series(True, index=rows.index)
            if lo is not None:
                inside &= rows["filed"] >= lo
            if hi is not None:
                inside &= rows["filed"] < hi
            parts.append(rows[inside].assign(ticker=ticker))
    if not parts:
        return pd.DataFrame(columns=["ticker", "cik", "form", "filed", "accession"])
    owned = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "accession"])
    return owned[["ticker", "cik", "form", "filed", "accession"]].sort_values(["ticker", "filed", "accession"], ignore_index=True)


def _owned_dates(owned: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, filing date): its forms, accessions and whether an original form was filed."""
    if owned.empty:
        return pd.DataFrame(columns=["ticker", "filed", "forms", "accessions", "has_original"])
    return (
        owned.groupby(["ticker", "filed"], sort=True)
        .agg(forms=("form", ",".join), accessions=("accession", ",".join), has_original=("form", lambda s: bool(s.isin(ORIGINAL_FORMS).any())))
        .reset_index()
    )


def _normalised(rows: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """The stored rows in the component shape, and whether the table already has it."""
    out = rows.copy()
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    new_shape = NEW_SHAPE_COLUMNS <= set(out.columns)
    if not new_shape:
        total = out["employees"] if "employees" in out.columns else pd.Series(pd.NA, index=out.index)
        out["employees_total"] = total
        out["status"] = total.notna().map({True: "found", False: None})
        for column in [c for c in NEW_SHAPE_COLUMNS if c not in out.columns]:
            out[column] = None
    for column in COMPONENTS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    return out.sort_values(["ticker", "as_of"], ignore_index=True), new_shape


def _quote_problem(raw: object, components: Sequence[str]) -> str | None:
    """Why a stored `source_quote` does not back the row's components, or None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "source_quote null"
    try:
        quotes = json.loads(str(raw))
    except json.JSONDecodeError:
        return "source_quote not JSON"
    if not isinstance(quotes, dict):
        return "source_quote not a JSON object"
    absent = [key for key in components if not str(quotes.get(key) or "").strip()]
    return f"no quote for {','.join(absent)}" if absent else None


def provenance_problems(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per found row lacking provenance: `ticker`, `as_of`, `problem`."""
    records = []
    found = rows[rows["status"].eq("found")]
    for row in found.to_dict("records"):
        problems = [f"{column} null" for column in PROVENANCE_COLUMNS if column != "source_quote" and pd.isna(row[column])]
        if pd.isna(row["basis"]):
            problems.append("basis null")
        stored = [key for column, key in COMPONENTS.items() if pd.notna(row[column])]
        if (quote := _quote_problem(row["source_quote"], stored)) is not None:
            problems.append(quote)
        records += [{"ticker": row["ticker"], "as_of": row["as_of"], "problem": p} for p in problems]
    return pd.DataFrame(records, columns=["ticker", "as_of", "problem"])


def basis_switches(rows: pd.DataFrame) -> pd.DataFrame:
    """Consecutive counted rows of one ticker whose basis differs."""
    counted = rows[rows["status"].eq("found") & rows["basis"].notna()].sort_values(["ticker", "as_of"])
    prior = counted.groupby("ticker")[["as_of", "basis"]].shift()
    switched = prior["basis"].notna() & prior["basis"].ne(counted["basis"])
    return pd.DataFrame(
        {
            "ticker": counted.loc[switched, "ticker"],
            "from_as_of": prior.loc[switched, "as_of"],
            "to_as_of": counted.loc[switched, "as_of"],
            "from_basis": prior.loc[switched, "basis"],
            "to_basis": counted.loc[switched, "basis"],
        }
    ).reset_index(drop=True)


def scope_jumps(rows: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year `employees_total` changes above `JUMP_THRESHOLD` between consecutive counted rows of one basis."""
    counted = rows[rows["employees_total"].notna()].sort_values(["ticker", "as_of"])
    basis = counted["basis"].fillna("")
    prior = counted.assign(basis=basis).groupby("ticker")[["as_of", "employees_total", "basis"]].shift()
    gap = (counted["as_of"] - prior["as_of"]).dt.days
    change = counted["employees_total"] / prior["employees_total"] - 1
    jump = prior["employees_total"].gt(0) & gap.between(*JUMP_GAP_DAYS) & basis.eq(prior["basis"]) & change.abs().gt(JUMP_THRESHOLD)
    return pd.DataFrame(
        {
            "ticker": counted.loc[jump, "ticker"],
            "from_as_of": prior.loc[jump, "as_of"],
            "to_as_of": counted.loc[jump, "as_of"],
            "from_total": prior.loc[jump, "employees_total"],
            "to_total": counted.loc[jump, "employees_total"],
            "change": change[jump].round(4),
            "basis": basis[jump],
        }
    ).reset_index(drop=True)


def _count_by(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame.groupby("ticker").size().rename(name) if not frame.empty else pd.Series(dtype="int64", name=name)


def _per_ticker(
    scope: Sequence[str], rows: pd.DataFrame, dates: pd.DataFrame, missing: pd.DataFrame, stale: pd.DataFrame, extra: Sequence[pd.Series]
) -> pd.DataFrame:
    status = pd.crosstab(rows["ticker"], rows["status"].fillna("null_status")) if not rows.empty else pd.DataFrame()
    status.columns = [f"status_{c}" for c in status.columns]
    counted = rows[rows[list(COMPONENTS)].notna().any(axis=1)]
    table = pd.DataFrame(index=pd.Index(sorted(scope), name="ticker"))
    table = table.join(
        [
            _count_by(dates, "owned_dates"),
            _count_by(dates[dates["has_original"]], "owned_original_dates"),
            _count_by(rows, "rows"),
            _count_by(counted, "counted_rows"),
            _count_by(missing, "missing_dates"),
            _count_by(missing[missing["has_original"]], "missing_original_dates"),
            _count_by(stale, "stale_rows"),
            *extra,
            status,
        ]
    )
    table = table.fillna(0).astype("int64")
    table["last_counted"] = counted.groupby("ticker")["as_of"].max()
    return table.reset_index()


def _window_start(context: Context) -> pd.Timestamp:
    years = int(getattr(context.config.data_extract, "years_history", _YEARS_HISTORY_DEFAULT))
    return pd.Timestamp.today().normalize() - pd.DateOffset(years=years)


def check_employees(
    context: Context, *, tickers: Sequence[str] | None = None, since: pd.Timestamp | None = None, directory: Path | None = None
) -> EmployeesReport:
    """Coverage, status, provenance, identity, basis switches and scope jumps of `fundamentals_employees` over the roster
    (or `tickers`); `since` defaults to today minus `years_history`, `directory` to the cached index."""
    table = Tables.fundamentals_employees.name
    start: pd.Timestamp = since if since is not None else _window_start(context)
    directory = directory if directory is not None else index_dir(context)
    roster = context.store.load(Tables.sp500_tickers, columns=["ticker", "cik"], optional=True)
    lineage = _lineage(context)
    empty = pd.DataFrame()
    if roster is None or roster.empty or lineage is None:
        return EmployeesReport(CheckResult.abstained(CHECK, table, "sp500_tickers or entity_lineage is empty"), empty, empty, empty, empty, empty)
    roster = roster.assign(cik=pad_cik_series(roster["cik"]))
    wanted = {normalise_ticker(t) for t in tickers} if tickers else None
    roster = roster[roster["ticker"].map(normalise_ticker).isin(wanted)] if wanted is not None else roster
    scope = sorted(roster["ticker"].astype(str))
    if "role" not in lineage.columns:
        lineage = _from_old_shape(lineage, roster)
    stored = context.store.load(Tables.fundamentals_employees, where={"ticker": scope}, optional=True)
    if stored is None:
        return EmployeesReport(CheckResult.abstained(CHECK, table, "fundamentals_employees does not exist"), empty, empty, empty, empty, empty)
    rows, new_shape = _normalised(stored)
    before_window = int(rows["as_of"].lt(start).sum())
    rows = rows[rows["as_of"].ge(start)].reset_index(drop=True)

    windows = listing_windows(lineage, roster)
    ciks = {cik for t in scope for cik, _, _ in windows.get(normalise_ticker(t), ())}
    entries = index_entries(directory, ciks, start)
    owned = owned_filings(entries, scope, windows)
    dates = _owned_dates(owned)
    have = set(zip(rows["ticker"], rows["as_of"], strict=True))
    unrowed = pd.Series([(t, d) not in have for t, d in zip(dates["ticker"], dates["filed"], strict=True)], index=dates.index, dtype=bool)
    missing = dates[unrowed].reset_index(drop=True)
    owned_days = set(zip(dates["ticker"], dates["filed"], strict=True))
    unowned = pd.Series([(t, d) not in owned_days for t, d in zip(rows["ticker"], rows["as_of"], strict=True)], index=rows.index, dtype=bool)
    stale = rows.loc[unowned, ["ticker", "as_of"]].reset_index(drop=True)

    findings: list[Finding] = []
    metrics: dict[str, Any] = {"shape": "component" if new_shape else "old", "rows_before_window": before_window}
    extra: list[pd.Series] = []
    if entries.empty:
        findings.append(Finding.at(5, f"no cached EDGAR index rows under {directory}", "the cached quarterly index", field="coverage"))
    for ticker, group in missing.groupby("ticker"):
        originals = int(group["has_original"].sum())
        findings.append(
            Finding.at(
                6 if originals else 3,
                f"{len(group)} owned annual filing date(s) without a row ({originals} original)",
                "one decided row per owned annual 10-K date (AC-004)",
                field="coverage",
                ticker=str(ticker),
                dates=[d.date().isoformat() for d in group["filed"]],
            )
        )
    for ticker, group in stale.groupby("ticker"):
        findings.append(
            Finding.at(
                7,
                f"{len(group)} row(s) with no owned annual filing on their date",
                "every row on an owned annual 10-K date (AC-005)",
                field="stale_rows",
                ticker=str(ticker),
                dates=[d.date().isoformat() for d in group["as_of"]],
            )
        )

    status = rows["status"].fillna("null_status")
    metrics["status_mix"] = status.value_counts().sort_index().to_dict()
    metrics["status_share"] = status.value_counts(normalize=True).round(4).sort_index().to_dict()
    jumps = scope_jumps(rows)
    switches = basis_switches(rows) if new_shape else pd.DataFrame(columns=["ticker", "from_as_of", "to_as_of", "from_basis", "to_basis"])
    if new_shape:
        findings += _component_findings(context, rows, entries, scope, lineage, metrics, extra)
    else:
        findings.append(
            Finding.at(
                6,
                "fundamentals_employees is in the pre-component shape (ticker, as_of, employees)",
                "the component shape with provenance (REQ-005)",
                field="shape",
                skipped=["status validity", "null components with status found", "provenance", "foreign cik", "as_of vs accession", "basis switches"],
            )
        )
    for ticker, group in switches.groupby("ticker"):
        findings.append(Finding.at(1, f"{len(group)} basis switch(es)", "reported, not failed", field="basis_switch", ticker=str(ticker)))
    for ticker, group in jumps.groupby("ticker"):
        findings.append(
            Finding.at(
                2,
                f"{len(group)} year-over-year change(s) above {JUMP_THRESHOLD:.0%} within one basis",
                "reviewed for a scope error",
                field="scope_jump",
                ticker=str(ticker),
                changes=group["change"].tolist(),
            )
        )
    extra += [_count_by(switches, "basis_switches"), _count_by(jumps, "scope_jumps")]
    per_ticker = _per_ticker(scope, rows, dates, missing, stale, extra)
    metrics |= {
        "owned_dates": len(dates),
        "owned_original_dates": int(dates["has_original"].sum()) if not dates.empty else 0,
        "missing_dates": len(missing),
        "missing_original_dates": int(missing["has_original"].sum()) if not missing.empty else 0,
        "missing_tickers": int(missing["ticker"].nunique()),
        "stale_rows": len(stale),
        "stale_tickers": int(stale["ticker"].nunique()),
        "zero_counted_tickers": sorted(per_ticker.loc[per_ticker["counted_rows"].eq(0), "ticker"]),
        "basis_switches": len(switches),
        "scope_jumps": len(jumps),
        "scope_jump_tickers": int(jumps["ticker"].nunique()),
    }
    scope_info = {
        "rows": len(rows),
        "tickers": len(scope),
        "since": start.strftime("%Y-%m-%d"),
        "index_dir": str(directory),
        "index_last_filed": entries["filed"].max().date().isoformat() if not entries.empty else None,
        "forms": list(ANNUAL_FORMS),
    }
    result = CheckResult.measured(CHECK, table, findings, scope=scope_info, metrics=metrics)
    return EmployeesReport(result, per_ticker, missing, stale, jumps, switches)


def _component_findings(
    context: Context, rows: pd.DataFrame, entries: pd.DataFrame, scope: Sequence[str], lineage: pd.DataFrame, metrics: dict[str, Any], extra: list
) -> list[Finding]:
    """Status validity, provenance, foreign CIKs and the filing date of each accession, on the component shape."""
    findings: list[Finding] = []
    counted = rows[list(COMPONENTS)].notna().any(axis=1)
    found = rows["status"].eq("found")
    checks = {
        "found_without_component": rows[found & ~counted],
        "component_without_found": rows[~found & counted],
        "unknown_status": rows[~rows["status"].isin(DECIDED_STATUSES)],
        "unknown_basis": rows[rows["basis"].notna() & ~rows["basis"].isin(BASES)],
    }
    for field, bad in checks.items():
        metrics[field] = len(bad)
        for ticker, group in bad.groupby("ticker"):
            findings.append(
                Finding.at(
                    7,
                    f"{len(group)} row(s): {field.replace('_', ' ')}",
                    "status found iff a component is stored; known status and basis",
                    field=field,
                    ticker=str(ticker),
                )
            )
    problems = provenance_problems(rows)
    metrics["provenance_incomplete_rows"] = int(problems[["ticker", "as_of"]].drop_duplicates().shape[0])
    metrics["provenance_problems"] = problems["problem"].str.replace(r" for .*", "", regex=True).value_counts().to_dict()
    metrics["found_rows"] = int(found.sum())
    for ticker, group in problems.groupby("ticker"):
        findings.append(
            Finding.at(
                6,
                f"{group['as_of'].nunique()} found row(s) with incomplete provenance",
                "cik, accession, form, source document, basis and a quote per stored component (AC-006)",
                field="provenance",
                ticker=str(ticker),
                problems=group["problem"].value_counts().to_dict(),
            )
        )
    filed = dict(zip(entries["accession"], entries["filed"], strict=True))
    accession_date = rows["accession_number"].map(filed)
    off_date = rows[accession_date.notna() & accession_date.ne(rows["as_of"])]
    metrics["as_of_not_filing_date"] = len(off_date)
    metrics["accession_not_in_index"] = int((rows["accession_number"].notna() & accession_date.isna()).sum())
    for ticker, group in off_date.groupby("ticker"):
        findings.append(
            Finding.at(
                6,
                f"{len(group)} row(s) whose as_of is not the accession's filing date",
                "as_of = SEC filing date (AC-006)",
                field="as_of",
                ticker=str(ticker),
            )
        )
    spec = PURGE_TABLES_BY_NAME[Tables.fundamentals_employees.name]
    removals = pd.DataFrame(_foreign_in_table(context, spec, scope, _entity_ciks(lineage), {}))
    metrics["foreign_rows"] = int(removals["rows"].sum()) if not removals.empty else 0
    for record in removals.to_dict("records"):
        findings.append(
            Finding.at(
                8,
                f"{record['rows']} row(s) filed by CIK {record['cik']} outside the ticker's entity",
                "every row filed by a CIK of the ticker's entity (AC-005)",
                field="foreign_rows",
                ticker=str(record["ticker"]),
            )
        )
    if not removals.empty:
        extra.append(removals.groupby("ticker")["rows"].sum().rename("foreign_rows"))
    return findings
