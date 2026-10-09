"""Identity check: rows filed by a CIK outside the ticker's entity, `entity_lineage` invariants, the manual-fix flags,
the vendor-series continuity at register cutovers and the traded-security price match (`traded_security`).

Reads only through `context.store`. The pending removals are the ones `identity-propagate` would purge,
recomputed from the stored rows and the lineage: a filing of any CIK of the entity (margin filings and
siblings included) is own, an 8-K / 13D / 13G only inside a seam-widened window of its CIK, a consolidating row
(facts, notes, filing text, proxies) only when its CIK holds a window of the ticker; a null CIK is never judged.
Foreign rows and invariant breaches fail the check; flags needing a manual decision are information.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils import cutover_continuity as cc
from src.utils.cik_windows import widen_seams
from src.utils.filer_tables import (
    PURGE_TABLES,
    REMOVAL_COLUMNS,
    FilerTable,
    ListedWindow,
    judged_cik_mask,
    own_filer_mask,
    removal_records,
    window_owner_mask,
    windowed_filer_mask,
)
from src.utils.identity_flags import FLAG_COLUMNS, KIND_ORDER, MARGIN, MASTER_COLUMNS, cik_activity, identity_flags, log_identity_flags
from src.utils.predecessor_series import load_vendor_series, with_overrides
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series
from src.validate.checks.traded_security import traded_security_flags
from src.validate.result import CheckResult, Finding

CHECK = "identity"
SENTINEL = pd.Timestamp("1900-01-01")
_EVIDENCE_COLUMNS = ["symbol", "issuer_cik", "valid_from", "valid_to", "source"]
_ROLE_WINDOW, _ROLE_EVENT = "cik_window", "cik_event"
_EXCEPTIONS = "configs/sec/vendor_coverage_exceptions.json"
_REGISTER = "configs/sec/registrant_cutover.json"
_VENDOR_COLUMNS = ["ticker", "dimension", "calendardate", "reportperiod", "date", "assets"]


@dataclass(frozen=True)
class IdentityReport:
    """The check result, the manual-fix flags (`FLAG_COLUMNS`) and the pending removals (`REMOVAL_COLUMNS`)."""

    result: CheckResult
    flags: pd.DataFrame
    removals: pd.DataFrame


def _config_dir(context: Context) -> str:
    return str(getattr(context, "config_dir", "./configs"))


def _lineage(context: Context) -> pd.DataFrame | None:
    rows = context.store.load(Tables.entity_lineage, project=True, optional=True)
    if rows is None or rows.empty:
        return None
    out = rows.copy()
    out["cik"] = pad_cik_series(out["cik"])
    if "role" not in out.columns:
        return out
    out["symbol"] = out["symbol"].fillna("").astype(str)
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"])
    return out


def _from_old_shape(lineage: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """A pre-cutover membership table read as the accessor reads it: each CIK an event CIK of its entity,
    each roster CIK its ticker's open window."""
    entity_by_cik = dict(zip(lineage["cik"], lineage["entity_id"].astype(str), strict=True))
    events = pd.DataFrame({"entity_id": lineage["entity_id"].astype(str), "cik": lineage["cik"], "role": _ROLE_EVENT, "canonical_ticker": None})
    windows = pd.DataFrame(
        {
            "entity_id": [entity_by_cik.get(c, f"universe:{t}") for t, c in zip(roster["ticker"], roster["cik"], strict=True)],
            "cik": roster["cik"].to_numpy(),
            "role": _ROLE_WINDOW,
            "canonical_ticker": roster["ticker"].to_numpy(),
        }
    )
    out = pd.concat([windows, events], ignore_index=True)
    return out.assign(symbol="", valid_from=SENTINEL, valid_to=cast(Any, pd.NaT), status="")


def _entity_ciks(lineage: pd.DataFrame) -> dict[str, frozenset[str]]:
    """`{ticker: every CIK of its entity}` from the CIK rows."""
    cik_rows = lineage[lineage["role"].ne("symbol")]
    ciks_by_entity = cik_rows.groupby("entity_id")["cik"].agg(frozenset).to_dict()
    named = cik_rows[cik_rows["canonical_ticker"].notna()]
    return {normalise_ticker(t): ciks_by_entity[e] for t, e in zip(named["canonical_ticker"], named["entity_id"], strict=True)}


def _listed_windows(lineage: pd.DataFrame) -> dict[str, tuple[ListedWindow, ...]]:
    """`{ticker: (cik, listed_from, listed_to) per seam-widened window}` from the `cik_window` rows; a ticker with none is absent."""
    rows = lineage[lineage["role"].eq(_ROLE_WINDOW)]
    named = lineage[lineage["role"].ne("symbol") & lineage["canonical_ticker"].notna()]
    ticker_of = dict(zip(named["entity_id"], named["canonical_ticker"].map(normalise_ticker), strict=True))
    declared: dict[str, list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]]] = {}
    for entity, cik, start, end in zip(
        rows["entity_id"], rows["cik"], pd.to_datetime(rows["valid_from"]), pd.to_datetime(rows["valid_to"]), strict=True
    ):
        if entity in ticker_of:
            declared.setdefault(ticker_of[entity], []).append(
                (pad_cik(cik), None if pd.isna(start) or start <= SENTINEL else start, None if pd.isna(end) else end)
            )
    return {ticker: tuple((cik, lo, hi) for cik, _, _, lo, hi in widen_seams(windows)) for ticker, windows in declared.items()}


def _outside_windows(context: Context, spec: FilerTable, ticker: str, filers: list, windows: dict[str, tuple[ListedWindow, ...]]) -> list[dict]:
    """Removal records of a dated table's rows from the entity's own `filers` that no window of their CIK admits."""
    if not filers or normalise_ticker(ticker) not in windows:
        return []
    rows = context.store.load(
        spec.table, columns=["ticker", spec.cik_col, spec.date_col, spec.key_col], where={"ticker": ticker, spec.cik_col: filers}, optional=True
    )
    if rows is None or rows.empty:
        return []
    outside = rows[~windowed_filer_mask(rows["ticker"], rows[spec.cik_col].map(pad_cik), rows[spec.date_col], windows)]
    return removal_records(spec.table.name, outside, spec) if not outside.empty else []


def _foreign_in_table(
    context: Context, spec: FilerTable, tickers: Sequence[str], ciks: dict[str, frozenset[str]], windows: dict[str, tuple[ListedWindow, ...]]
) -> list[dict]:
    """One removal record per (ticker, filer CIK) of `spec` outside the ticker's entity, outside its windows for a dated
    table, or holding no window of the ticker for a windowed table."""
    records: list[dict] = []
    columns = ["ticker", spec.cik_col, spec.date_col, spec.key_col]
    for ticker in tickers:
        filers = pd.Series(context.store.distinct(spec.table, spec.cik_col, where={"ticker": ticker}), dtype=object)
        padded = filers.map(pad_cik)
        tickers_of = pd.Series(ticker, index=filers.index, dtype=object)
        judged, own = judged_cik_mask(filers), own_filer_mask(tickers_of, padded, ciks)
        if spec.windowed:
            own &= window_owner_mask(tickers_of, padded, windows)
        if spec.dated:
            records += _outside_windows(context, spec, ticker, filers[judged & own].tolist(), windows)
        foreign = filers[judged & ~own].tolist()
        if not foreign:
            continue
        rows = context.store.load(spec.table, columns=columns, where={"ticker": ticker, spec.cik_col: foreign}, optional=True)
        if rows is not None and not rows.empty:
            records += removal_records(spec.table.name, rows, spec)
    return records


def pending_removals(context: Context, lineage: pd.DataFrame, tickers: Sequence[str], *, dated: bool = True) -> pd.DataFrame:
    """The rows `identity-propagate` would purge from every filer-CIK table, as `REMOVAL_COLUMNS` records.

    `dated=False` skips the window rule (a pre-cutover lineage declares no dated windows)."""
    ciks, windows = _entity_ciks(lineage), _listed_windows(lineage) if dated else {}
    records: list[dict] = []
    for spec in PURGE_TABLES:
        if context.store.exists(spec.table):
            records += _foreign_in_table(context, spec, tickers, ciks, windows)
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


def _overlap_days(a: Mapping[Hashable, Any], b: Mapping[Hashable, Any]) -> int:
    """Days two windows share, `b` starting no earlier than `a`; an open end runs to the far future."""
    far = pd.Timestamp("2262-01-01")
    return (min(a["valid_to"] if pd.notna(a["valid_to"]) else far, b["valid_to"] if pd.notna(b["valid_to"]) else far) - b["valid_from"]).days


def _window_overlaps(windows: pd.DataFrame) -> list[dict[str, Any]]:
    """Every pair of one entity's CIK windows overlapping by more than the margin."""
    out: list[dict[str, Any]] = []
    for entity, group in windows.groupby("entity_id", sort=True):
        for a, b in combinations(group.sort_values("valid_from").to_dict("records"), 2):
            if (days := _overlap_days(a, b)) > MARGIN.days:
                out.append({"entity_id": entity, "ciks": [a["cik"], b["cik"]], "overlap_days": days})
    return out


def _flags(context: Context, lineage: pd.DataFrame) -> pd.DataFrame:
    ciks = sorted(set(lineage.loc[lineage["role"].ne("symbol"), "cik"]))
    evidence = context.store.load(
        Tables.symbol_tenure, columns=_EVIDENCE_COLUMNS, where={"issuer_cik": ciks, "source": ["dei", "form345"]}, optional=True
    )
    activity = cik_activity(evidence if evidence is not None else pd.DataFrame(columns=_EVIDENCE_COLUMNS))
    redundant = frozenset(normalise_ticker(t) for t in context.config.data_extract.redundant_ticks)
    master = context.store.load(Tables.security_master, columns=list(MASTER_COLUMNS), optional=True)
    return identity_flags(lineage, activity, redundant_symbols=redundant, master=master)


def _removal_findings(removals: pd.DataFrame) -> list[Finding]:
    return [
        Finding.at(
            8,
            f"{r.rows} row(s) / {r.keys} filing(s) in {r.table} filed by CIK {r.cik} ({r.first_filed}..{r.last_filed})",
            "every row filed by a CIK of the ticker's entity; an 8-K / 13D / 13G inside that CIK's window (AC-024, P35)",
            field="foreign_rows",
            ticker=str(r.ticker),
            table=r.table,
            cik=r.cik,
        )
        for r in removals.itertuples(index=False)
    ]


def _removal_metrics(removals: pd.DataFrame) -> dict[str, Any]:
    return {
        "foreign_rows": int(removals["rows"].sum()) if not removals.empty else 0,
        "foreign_tickers": sorted(set(removals["ticker"])),
        "foreign_by_table": removals.groupby("table")["rows"].sum().to_dict() if not removals.empty else {},
    }


def _vendor_arq(context: Context, tickers: Sequence[str]) -> pd.DataFrame:
    rows = context.store.load(
        Tables.sharadar_fundamentals, columns=_VENDOR_COLUMNS, where={"ticker": list(tickers), "dimension": "ARQ"}, optional=True
    )
    return rows if rows is not None else pd.DataFrame(columns=_VENDOR_COLUMNS)


def _flag(kind: str, action: bool, ticker: str, cik: str, evidence: str, suggested: str, config_file: str) -> dict[str, object]:
    return {
        "kind": kind,
        "action": action,
        "ticker": ticker,
        "ciks": cik,
        "evidence": evidence,
        "suggested_action": suggested,
        "config_file": config_file,
    }


def _continuity_item(row: Mapping[Hashable, Any]) -> dict[str, object]:
    """One flag item per discontinuity: a recorded vendor gap is information, everything else an action."""
    kind, ticker, cik = str(row["class"]), str(row["ticker"]), str(row["cik"])
    record = (
        f"; label '{row['label']}', accession {row['accession']}" if row["label"] else (f"; accession {row['accession']}" if row["accession"] else "")
    )
    evidence = f"{row['quarter']} missing from the vendor series near the boundary {row['boundary']}{record} | {row['evidence']}"
    if row["explained"]:
        return _flag(kind, False, ticker, cik, evidence, "none: recorded vendor coverage exception", _EXCEPTIONS)
    if kind == cc.VENDOR_COVERAGE_GAP:
        return _flag(kind, True, ticker, cik, evidence, "record the quarter's EDGAR accession", _EXCEPTIONS)
    if kind == cc.INCORRECT_CIK_WINDOW:
        return _flag(kind, True, ticker, cik, evidence, "the filer CIK is outside its register window: check the window", _REGISTER)
    return _flag(kind, True, ticker, cik, evidence, "no SEC filing stored or recorded: fetch the quarter's 10-Q/10-K", "")


def _series_items(arq: pd.DataFrame, owners: pd.DataFrame, series: Sequence[cc.PredecessorSeries]) -> list[dict[str, object]]:
    """Canonical vendor quarters inside a predecessor window that carry another company's series, or cannot be checked."""
    items: list[dict[str, object]] = []
    for s in series:
        window = f"{s.cik}'s window ..{s.valid_to.date() if s.valid_to is not None else 'open'}"
        if owners.empty or not owners["ticker"].astype(str).eq(s.vendor_ticker).any():
            period = pd.to_datetime(arq["reportperiod"])
            inside = arq[arq["ticker"].astype(str).eq(s.ticker) & (period < s.valid_to if s.valid_to is not None else period.notna())]
            evidence = f"{s.vendor_ticker} (the window owner's vendor series) is not stored: {len(inside)} canonical vendor row(s) inside {window} are unverified and not replaced"
            items.append(
                _flag(cc.VENDOR_SERIES_OTHER_COMPANY, True, s.ticker, s.cik, evidence, f"fetch {s.vendor_ticker} with fundamentals-sharadar", "")
            )
            continue
        other = cc.other_company_quarters(arq, owners, s)
        if other.empty:
            continue
        sample = ", ".join(f"{r.quarter} {r.canonical_assets:,.0f} vs {r.owner_assets:,.0f}" for r in other.head(3).itertuples(index=False))
        evidence = f"{len(other)} quarter(s) inside {window} carry another company's assets ({other['quarter'].iloc[0]}..{other['quarter'].iloc[-1]}; {sample})"
        items.append(_flag(cc.VENDOR_SERIES_OTHER_COMPANY, False, s.ticker, s.cik, evidence, f"none: replaced by {s.vendor_ticker} at the merge", ""))
    return items


def continuity_flags(context: Context, lineage: pd.DataFrame, scope: Sequence[str], config_dir: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Vendor-series discontinuities near every register or cited-override boundary of the scope's tickers, as flag items, plus metrics.

    The vendor series is read as the merge builds it (predecessor series applied), so a filled quarter heals its record.
    """
    windows = {t: w for t, w in cc.register_windows(lineage).items() if t in set(scope)}
    # a cited vendor-series override (JCI <- TYC) is a boundary too: its CIK before `valid_to`, the same CIK after
    declared = tuple(s for s in load_vendor_series(config_dir) if s.ticker in set(scope) and s.valid_to is not None)
    for s in declared:
        windows.setdefault(s.ticker, (cc.CikWindow(s.cik, s.valid_from, s.valid_to), cc.CikWindow(s.cik, s.valid_to, None)))
    if not windows or not context.store.exists(Tables.sharadar_fundamentals):
        return pd.DataFrame(columns=list(FLAG_COLUMNS)), {}
    vendor_tickers = context.store.load(Tables.sharadar_tickers, columns=["ticker", "secfilings", "lastquarter"], optional=True)
    series = with_overrides(cc.predecessor_series(vendor_tickers if vendor_tickers is not None else pd.DataFrame(), windows, scope), declared)
    arq = _vendor_arq(context, sorted(windows))
    owners = _vendor_arq(context, sorted({s.vendor_ticker for s in series})) if series else pd.DataFrame(columns=_VENDOR_COLUMNS)
    merged, events = cc.apply_predecessor_series(arq, owners, series)
    facts = context.store.load(
        Tables.fundamentals_facts, columns=list(cc.FILING_COLUMNS), where={"ticker": sorted(windows), "form": list(cc.PERIODIC_FORMS)}, optional=True
    )
    report = cc.assess_continuity(merged, windows, cc.prepare_filings(facts), cc.load_vendor_exceptions(config_dir))
    items = [_continuity_item(row) for row in report.table.to_dict("records")]
    items += [
        _flag(
            cc.VENDOR_COVERAGE_GAP,
            True,
            r.ticker,
            r.cik,
            f"{r.quarter} recorded ({r.accession}) but the vendor quarter is present now",
            "delete the healed row",
            _EXCEPTIONS,
        )
        for r in report.healed
    ]
    items += [
        _flag(cc.VENDOR_COVERAGE_GAP, True, key.split()[0], "", f"{key} is recorded twice", "keep one row per (ticker, quarter)", _EXCEPTIONS)
        for key in report.duplicated
    ]
    items += _series_items(arq, owners, series)
    metrics = {
        "continuity_discontinuities": len(report.table),
        "continuity_explained": int(report.table["explained"].sum()) if not report.table.empty else 0,
        "continuity_healed": [f"{r.ticker} {r.quarter}" for r in report.healed],
        "continuity_unobserved": list(report.unobserved),
        "predecessor_vendor_tickers": {s.ticker: s.vendor_ticker for s in series},
        "predecessor_events": events["event"].value_counts().sort_index().to_dict() if not events.empty else {},
    }
    return pd.DataFrame(items, columns=list(FLAG_COLUMNS)), metrics


def _unmigrated_report(context: Context, lineage: pd.DataFrame, scope: list[str]) -> IdentityReport:
    """Foreign rows against a pre-cutover lineage; its invariants and flags need the new shape and are skipped."""
    removals = pending_removals(context, lineage, scope, dated=False)
    findings = [
        Finding.at(
            2,
            "entity_lineage is in the pre-cutover shape: foreign rows only, no invariants or flags",
            "the dated lineage (run the identity cutover steps)",
            field="lineage_not_migrated",
        ),
        *_removal_findings(removals),
    ]
    scope_info = {"rows": len(lineage), "tickers": len(scope), "tables": [spec.table.name for spec in PURGE_TABLES]}
    result = CheckResult.measured(CHECK, Tables.entity_lineage.name, findings, scope=scope_info, metrics=_removal_metrics(removals))
    return IdentityReport(result, pd.DataFrame(columns=list(FLAG_COLUMNS)), removals)


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
    if "role" not in lineage.columns:
        return _unmigrated_report(context, _from_old_shape(lineage, roster), scope)
    removals = pending_removals(context, lineage, scope)
    continuity, continuity_metrics = continuity_flags(context, lineage, scope, _config_dir(context))
    traded, traded_metrics = traded_security_flags(context, scope)
    flags = _flags(context, lineage)
    extra = [part for part in (continuity, traded) if not part.empty]
    if extra:
        flags = pd.concat([flags, *extra], ignore_index=True)
        order = flags["kind"].map({kind: rank for rank, kind in enumerate(KIND_ORDER)})
        flags = flags.assign(_order=order).sort_values(["_order", "ticker"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    log_identity_flags(context.log, flags)
    findings = _invariant_findings(lineage, roster[roster["ticker"].isin(scope)])
    findings += _removal_findings(removals)
    findings += [
        Finding.at(2, f"{f.kind}: {f.evidence}", str(f.suggested_action), field="manual_decision", ticker=str(f.ticker))
        for f in flags[flags["action"].astype(bool)].itertuples()
    ]
    metrics = {
        **_removal_metrics(removals),
        "flags_by_kind": flags["kind"].value_counts().sort_index().to_dict(),
        "backlog": int(flags["action"].sum()),
        "symbol_statuses": lineage.loc[lineage["role"].eq("symbol"), "status"].value_counts().sort_index().to_dict(),
        **continuity_metrics,
        **traded_metrics,
    }
    scope_info = {"rows": len(lineage), "tickers": len(scope), "tables": [spec.table.name for spec in PURGE_TABLES]}
    result = CheckResult.measured(CHECK, Tables.entity_lineage.name, findings, scope=scope_info, metrics=metrics)
    return IdentityReport(result, flags, removals)
