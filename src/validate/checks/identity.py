"""Identity check: rows filed by a CIK outside the ticker's entity, `entity_lineage` invariants, and the manual-fix flags.

Reads only through `context.store`. The pending removals are the ones `identity-propagate` would purge,
recomputed from the stored rows and the lineage: a filing of any CIK of the entity (margin filings and
siblings included) is own; a null CIK is never judged. Foreign rows and invariant breaches fail the check;
flags needing a manual decision are information.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils.filer_tables import PURGE_TABLES, REMOVAL_COLUMNS, FilerTable, removal_records
from src.utils.identity_flags import FLAG_COLUMNS, MARGIN, cik_activity, identity_flags, log_identity_flags
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series
from src.validate.result import CheckResult, Finding

CHECK = "identity"
SENTINEL = pd.Timestamp("1900-01-01")
_EVIDENCE_COLUMNS = ["symbol", "issuer_cik", "valid_from", "valid_to", "source"]


@dataclass(frozen=True)
class IdentityReport:
    """The check result, the manual-fix flags (`FLAG_COLUMNS`) and the pending removals (`REMOVAL_COLUMNS`)."""

    result: CheckResult
    flags: pd.DataFrame
    removals: pd.DataFrame


def _lineage(context: Context) -> pd.DataFrame | None:
    rows = context.store.load(Tables.entity_lineage, project=True, optional=True)
    if rows is None or rows.empty:
        return None
    out = rows.copy()
    out["cik"] = pad_cik_series(out["cik"])
    out["symbol"] = out["symbol"].fillna("").astype(str)
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"])
    return out


def _entity_ciks(lineage: pd.DataFrame) -> dict[str, frozenset[str]]:
    """`{ticker: every CIK of its entity}` from the CIK rows."""
    cik_rows = lineage[lineage["role"].ne("symbol")]
    ciks_by_entity = cik_rows.groupby("entity_id")["cik"].agg(frozenset).to_dict()
    named = cik_rows[cik_rows["canonical_ticker"].notna()]
    return {normalise_ticker(t): ciks_by_entity[e] for t, e in zip(named["canonical_ticker"], named["entity_id"], strict=True)}


def _foreign_in_table(context: Context, spec: FilerTable, tickers: Sequence[str], ciks: dict[str, frozenset[str]]) -> list[dict]:
    """One removal record per (ticker, filer CIK) of `spec` outside the ticker's entity."""
    records: list[dict] = []
    columns = ["ticker", spec.cik_col, spec.date_col, spec.key_col]
    for ticker in tickers:
        filers = context.store.distinct(spec.table, spec.cik_col, where={"ticker": ticker})
        own = ciks.get(ticker, frozenset())
        foreign = [raw for raw in filers if pad_cik(raw) and pad_cik(raw) not in own]
        if not foreign:
            continue
        rows = context.store.load(spec.table, columns=columns, where={"ticker": ticker, spec.cik_col: foreign})
        if rows is not None and not rows.empty:
            records += removal_records(spec.table.name, rows, spec)
    return records


def pending_removals(context: Context, lineage: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """The rows `identity-propagate` would purge from every filer-CIK table, as `REMOVAL_COLUMNS` records."""
    ciks = _entity_ciks(lineage)
    records: list[dict] = []
    for spec in PURGE_TABLES:
        if context.store.exists(spec.table):
            records += _foreign_in_table(context, spec, tickers, ciks)
    return pd.DataFrame(records, columns=list(REMOVAL_COLUMNS))


def _current_row_finding(lineage: pd.DataFrame, roster: pd.DataFrame) -> Finding | None:
    """AC-012: one current symbol row per roster ticker, on its roster CIK."""
    current = lineage[lineage["role"].eq("symbol") & lineage["valid_to"].isna()]
    counts = current.groupby(["symbol", "cik"]).size().to_dict()
    bad = {t: n for t, c in zip(roster["ticker"], roster["cik"], strict=True) if (n := int(counts.get((t, c), 0))) != 1}
    if not bad:
        return None
    return Finding.at(
        8, f"{len(bad)} roster ticker(s) without exactly one current symbol row", "exactly one (AC-012)", field="current_symbol_rows", tickers=bad
    )


def _invariant_findings(lineage: pd.DataFrame, roster: pd.DataFrame) -> list[Finding]:
    """AC-012, the sentinel open start, one entity per CIK, and no window overlap inside an entity beyond the margin."""
    findings = [f for f in (_current_row_finding(lineage, roster),) if f is not None]
    off = lineage[lineage["valid_from"].isna() | (lineage["valid_from"] < SENTINEL)]
    if not off.empty:
        findings.append(
            Finding.at(
                9,
                f"{len(off)} row(s) with an open start not stored as {SENTINEL.date()}",
                "the sentinel",
                field="sentinel_start",
                ciks=sorted(set(off["cik"])),
            )
        )
    cik_rows = lineage[lineage["role"].ne("symbol")]
    entities = cik_rows.groupby("cik")["entity_id"].nunique()
    shared = sorted(entities[entities.gt(1)].index)
    if shared:
        findings.append(Finding.at(9, f"{len(shared)} CIK(s) in more than one entity", "one entity per CIK", field="one_entity_per_cik", ciks=shared))
    overlaps = _window_overlaps(cik_rows[cik_rows["role"].eq("cik_window")])
    if overlaps:
        findings.append(
            Finding.at(
                7,
                f"{len(overlaps)} CIK window pair(s) overlap beyond the margin",
                f"overlap <= {MARGIN.days} days",
                field="window_overlap",
                pairs=overlaps,
            )
        )
    return findings


def _window_overlaps(windows: pd.DataFrame) -> list[dict[str, Any]]:
    far = pd.Timestamp("2262-01-01")
    out: list[dict[str, Any]] = []
    for entity, group in windows.groupby("entity_id", sort=True):
        records = group.sort_values("valid_from").to_dict("records")
        for i, a in enumerate(records):
            for b in records[i + 1 :]:
                days = (
                    min(a["valid_to"] if pd.notna(a["valid_to"]) else far, b["valid_to"] if pd.notna(b["valid_to"]) else far) - b["valid_from"]
                ).days
                if days > MARGIN.days:
                    out.append({"entity_id": entity, "ciks": [a["cik"], b["cik"]], "overlap_days": days})
    return out


def _flags(context: Context, lineage: pd.DataFrame) -> pd.DataFrame:
    ciks = sorted(set(lineage.loc[lineage["role"].ne("symbol"), "cik"]))
    evidence = context.store.load(
        Tables.symbol_tenure, columns=_EVIDENCE_COLUMNS, where={"issuer_cik": ciks, "source": ["dei", "form345"]}, optional=True
    )
    activity = cik_activity(evidence if evidence is not None else pd.DataFrame(columns=_EVIDENCE_COLUMNS))
    redundant = frozenset(normalise_ticker(t) for t in context.config.data_extract.redundant_ticks)
    return identity_flags(lineage, activity, redundant_symbols=redundant)


def check_identity(context: Context, *, tickers: Sequence[str] | None = None) -> IdentityReport:
    """Foreign rows per filer-CIK table, lineage invariants and the manual-fix flags; logs the flag block."""
    empty = IdentityReport(
        CheckResult.abstained(CHECK, Tables.entity_lineage.name, "entity_lineage is empty: run identity-tables first"),
        pd.DataFrame(columns=list(FLAG_COLUMNS)),
        pd.DataFrame(columns=list(REMOVAL_COLUMNS)),
    )
    lineage = _lineage(context)
    roster = context.store.load(Tables.sp500_tickers, columns=["ticker", "cik"], optional=True)
    if lineage is None or roster is None or roster.empty:
        return empty
    roster = roster.assign(ticker=roster["ticker"].map(normalise_ticker), cik=pad_cik_series(roster["cik"]))
    scope = sorted({normalise_ticker(t) for t in tickers} & set(roster["ticker"])) if tickers else sorted(roster["ticker"])
    removals = pending_removals(context, lineage, scope)
    flags = _flags(context, lineage)
    log_identity_flags(context.log, flags)
    findings = _invariant_findings(lineage, roster[roster["ticker"].isin(scope)])
    findings += [
        Finding.at(
            8,
            f"{r.rows} row(s) / {r.keys} filing(s) in {r.table} filed by CIK {r.cik} ({r.first_filed}..{r.last_filed})",
            "every row filed by a CIK of the ticker's entity (AC-024)",
            field="foreign_rows",
            ticker=str(r.ticker),
            table=r.table,
            cik=r.cik,
        )
        for r in removals.itertuples(index=False)
    ]
    findings += [
        Finding.at(2, f"{f.kind}: {f.evidence}", str(f.suggested_action), field="manual_decision", ticker=str(f.ticker))
        for f in flags[flags["action"].astype(bool)].itertuples()
    ]
    metrics = {
        "foreign_rows": int(removals["rows"].sum()) if not removals.empty else 0,
        "foreign_tickers": sorted(set(removals["ticker"])),
        "foreign_by_table": removals.groupby("table")["rows"].sum().to_dict() if not removals.empty else {},
        "flags_by_kind": flags["kind"].value_counts().sort_index().to_dict(),
        "backlog": int(flags["action"].sum()),
        "symbol_statuses": lineage.loc[lineage["role"].eq("symbol"), "status"].value_counts().sort_index().to_dict(),
    }
    scope_info = {"rows": len(lineage), "tickers": len(scope), "tables": [spec.table.name for spec in PURGE_TABLES]}
    result = CheckResult.measured(CHECK, Tables.entity_lineage.name, findings, scope=scope_info, metrics=metrics)
    return IdentityReport(result, flags, removals)
