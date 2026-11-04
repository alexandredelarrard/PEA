"""Symbol history of `universe_cik_symbol`: the dated symbol intervals of each company, attached through its CIK history.

`symbol_tenure` holds the raw evidence; the CIK history (`cik_history.py`) says which CIKs are each company and when its
windows close. Each symbol interval gets a `status`: noise, clipped by another company's anchor interval, a reuse
conflict, `corroborated` or `single_source`. The build writes the whole table (CIK rows and symbol rows) unless unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.incremental import matches_stored
from src.data_extract.utils.identity.cik_history import (
    BACKLOG_COLUMNS,
    EVIDENCE_SOURCES,
    FORM345_SOURCE,
    MANUAL_SOURCE,
    ROLE_SYMBOL,
    ROSTER_SOURCE,
    SENTINEL_START,
    TABLE_COLUMNS,
    CikHistory,
    check_one_entity_per_cik,
    entity_by_cik_map,
    entity_or_singleton,
)
from src.data_extract.utils.identity.identity_store import save_identity
from src.data_extract.utils.identity.symbol_tenure import DEI_SOURCE, load_manual_symbol_tenure
from src.data_store.schema import Tables
from src.utils.identity_flags import MASTER_COLUMNS, identity_flags, log_identity_flags
from src.utils.string import pad_cik_series

logger = logging.getLogger(__name__)

#: Two entities holding one symbol closer than this (or overlapping) is a reuse conflict; exchanges allow reuse after 90 days.
REUSE_GAP_DAYS = 90
#: A single-source interval with at most this many filings, next to another entity's interval, is noise.
NOISE_MAX_OBSERVATIONS = 5
#: A `dei` interval whose last filing is this close to the latest `dei` filing is still current.
CURRENT_RECENCY_DAYS = 120


class ManualTenureEntityError(ValueError):
    """A manual ticker interval names a CIK outside its canonical ticker's entity."""


@dataclass(frozen=True)
class LineageBuild:
    """`rows` for `universe_cik_symbol`, plus what the build refused, excluded or measured."""

    rows: pd.DataFrame
    blocked: pd.DataFrame
    backlog: pd.DataFrame
    holders: pd.DataFrame


def validate_manual_tenure_entities(manual: pd.DataFrame, lineage: pd.DataFrame, roster_cik: Mapping[str, str]) -> None:
    """Require each manual CIK to belong to its configured current ticker's entity."""
    entity_by_cik = entity_by_cik_map(lineage)
    errors: list[str] = []
    for row in manual.itertuples(index=False):
        canonical_ticker = str(row.canonical_ticker)
        issuer_cik = str(row.issuer_cik)
        home_cik = roster_cik.get(canonical_ticker)
        if home_cik is None:
            errors.append(f"{canonical_ticker}: absent from the current roster")
            continue
        expected = entity_or_singleton(entity_by_cik, home_cik)
        actual = entity_or_singleton(entity_by_cik, issuer_cik)
        if actual != expected:
            errors.append(f"{canonical_ticker}/{row.symbol}/{issuer_cik}: manual entity {actual}, roster entity {expected}")
    if errors:
        raise ManualTenureEntityError("symbol_tenure_manual contains CIKs outside their canonical current entity: " + "; ".join(errors))


# Symbol rows
def _symbol_holders(
    evidence: pd.DataFrame, entity_of: Mapping[str, str], ticker_by_entity: Mapping[str, str], roster_cik: Mapping[str, str]
) -> pd.DataFrame:
    """Every holder interval of the symbols the universe entities ever typed, classified (other entities included)."""
    universe_ciks = {cik for cik, entity in entity_of.items() if entity in ticker_by_entity}
    scope = set(evidence.loc[evidence["cik"].isin(universe_ciks), "symbol"]) | set(roster_cik)
    in_scope = evidence[evidence["symbol"].isin(scope)]
    dei_rows = evidence[evidence["source"].eq(DEI_SOURCE)]
    current_after = dei_rows["end"].max() - pd.Timedelta(days=CURRENT_RECENCY_DAYS + 1) if not dei_rows.empty else None
    rows = [_merge_holder(group, current_after) for _, group in in_scope.groupby(["symbol", "cik"], sort=True)]
    rows = [row for merged in rows for row in merged]
    rows += _roster_rows(rows, roster_cik)
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_symbol.setdefault(str(row["symbol"]), []).append(row)
    out = [row for symbol in sorted(by_symbol) for row in _classify_symbol(by_symbol[symbol], entity_of)]
    frame = pd.DataFrame(out).drop(columns=["entity", "curated", "roster"])
    frame["company_id"] = [entity_of.get(cik, f"E{cik}") for cik in frame["cik"]]
    frame["canonical_ticker"] = frame["company_id"].map(ticker_by_entity)
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    return frame.sort_values(["symbol", "valid_from", "cik"], kind="mergesort", ignore_index=True)


def _merge_holder(group: pd.DataFrame, current_after: pd.Timestamp | None) -> list[dict[str, Any]]:
    """One (symbol, CIK)'s intervals: manual rows as they are, else `form345` and `dei` merged into one interval."""
    symbol, cik = str(group["symbol"].iloc[0]), str(group["cik"].iloc[0])
    manual = group[group["source"].eq(MANUAL_SOURCE)]
    if not manual.empty:
        return [
            {
                "symbol": symbol,
                "cik": cik,
                "valid_from": row.first,
                "valid_to": None if pd.isna(row.end) else row.end,
                "sources": frozenset({MANUAL_SOURCE}),
                "n_observations": int(group.loc[group["source"].ne(MANUAL_SOURCE), "n"].sum()),
                "evidence": f"manual: {row.evidence}",
                "curated": True,
                "roster": False,
            }
            for row in manual.sort_values("first").itertuples(index=False)
        ]
    derived = group[group["source"].isin(EVIDENCE_SOURCES)]
    if derived.empty:
        return []
    form345 = derived[derived["source"].eq(FORM345_SOURCE)]
    dei = derived[derived["source"].eq(DEI_SOURCE)]
    is_open = bool(form345["end"].isna().any()) or (current_after is not None and not dei.empty and dei["end"].max() > current_after)
    parts = [
        f"{source} {part['first'].min().date()}..{_last_text(part)} n={int(part['n'].sum())}"
        + (f" ({part['evidence'].iloc[-1]})" if part["evidence"].iloc[-1] else "")
        for source, part in ((FORM345_SOURCE, form345), (DEI_SOURCE, dei))
        if not part.empty
    ]
    return [
        {
            "symbol": symbol,
            "cik": cik,
            "valid_from": derived["first"].min(),
            "valid_to": None if is_open else derived["end"].max(),
            "sources": frozenset(derived["source"]),
            "n_observations": int(derived["n"].sum()),
            "evidence": "; ".join(parts),
            "curated": False,
            "roster": False,
        }
    ]


def _last_text(part: pd.DataFrame) -> str:
    """The last observed filing date of an evidence part, or `open`."""
    return "open" if part["end"].isna().any() else str((part["end"].max() - pd.Timedelta(days=1)).date())


def _roster_rows(rows: list[dict[str, Any]], roster_cik: Mapping[str, str]) -> list[dict[str, Any]]:
    """Mark each roster ticker's latest interval on its roster CIK as current; add one when no filing spells the ticker."""
    added: list[dict[str, Any]] = []
    for ticker, cik in sorted(roster_cik.items()):
        own = [row for row in rows if row["symbol"] == ticker and row["cik"] == cik]
        latest = max(own, key=lambda row: row["valid_from"], default=None)
        if latest is not None and not (latest["curated"] and latest["valid_to"] is not None):
            latest["roster"] = True
            latest["sources"] = frozenset(latest["sources"]) | {ROSTER_SOURCE}
            latest["valid_to"] = None
            continue
        starts = [row["valid_from"] for row in rows if row["cik"] == cik]
        start = latest["valid_to"] if latest is not None else min(starts, default=SENTINEL_START)
        added.append(
            {
                "symbol": ticker,
                "cik": cik,
                "valid_from": start,
                "valid_to": None,
                "sources": frozenset({ROSTER_SOURCE}),
                "n_observations": 0,
                "evidence": "roster: the universe names this ticker on this CIK today; no filing spells it this way",
                "curated": False,
                "roster": True,
            }
        )
    return added


def _gap_days(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Days from the end of one half-open interval to the start of the other; negative when they overlap (open end = +inf)."""
    a_after_b = (pd.Timestamp(a["valid_from"]) - pd.Timestamp(b["valid_to"])).days if b["valid_to"] is not None else float("-inf")
    b_after_a = (pd.Timestamp(b["valid_from"]) - pd.Timestamp(a["valid_to"])).days if a["valid_to"] is not None else float("-inf")
    return max(a_after_b, b_after_a)


def _is_weak(row: Mapping[str, Any]) -> bool:
    """A derived single-source interval with at most `NOISE_MAX_OBSERVATIONS` filings."""
    filed = frozenset(row["sources"]) & set(EVIDENCE_SOURCES)
    return not row["curated"] and not row["roster"] and len(filed) == 1 and int(row["n_observations"]) <= NOISE_MAX_OBSERVATIONS


def _classify_symbol(rows: list[dict[str, Any]], entity_of: Mapping[str, str]) -> list[dict[str, Any]]:
    """Status per interval of one symbol: noise, clipping by anchors of other entities, then reuse conflicts.

    Anchors are curated rows and each roster ticker's interval on its roster CIK; a curated row clips a roster
    one, and both clip derived rows of other entities. A weak row is noise when another entity held the symbol
    within `REUSE_GAP_DAYS`, or at any time with a stronger interval. Derived rows of two entities within
    `REUSE_GAP_DAYS` of each other are both `conflict`.
    """
    for row in rows:
        row["entity"] = entity_of.get(str(row["cik"]), f"E{row['cik']}")
        row["status"] = ""
    for row in rows:
        if _is_weak(row) and any(
            other["entity"] != row["entity"] and (not _is_weak(other) or _gap_days(row, other) < REUSE_GAP_DAYS) for other in rows
        ):
            row["status"] = "noise"
    by_start = sorted(rows, key=lambda row: pd.Timestamp(row["valid_from"]))
    curated = [row for row in by_start if row["curated"]]
    roster = [row for row in by_start if row["roster"] and not row["curated"]]
    _clip_by_anchors([row for row in rows if not row["curated"]], curated)
    _clip_by_anchors([row for row in rows if not row["curated"] and not row["roster"]], roster)
    live = [row for row in rows if not row["status"] and not row["curated"] and not row["roster"]]
    for row, other in combinations(live, 2):
        if other["entity"] != row["entity"] and _gap_days(row, other) < REUSE_GAP_DAYS:
            row["status"] = other["status"] = "conflict"
    for row in rows:
        if row["curated"]:
            row["status"] = "curated"
        elif not row["status"]:
            filed = frozenset(row["sources"]) & set(EVIDENCE_SOURCES)
            row["status"] = "corroborated" if len(filed) == len(EVIDENCE_SOURCES) else "single_source"
    return rows


def _clip_by_anchors(targets: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    """Clip each unclassified target by every overlapping anchor of another entity, in anchor order."""
    for row in targets:
        for holder in anchors:
            if row["status"]:
                break
            if holder["entity"] != row["entity"] and _gap_days(row, holder) < 0:
                _clip(row, holder)


def _clip(row: dict[str, Any], holder: Mapping[str, Any]) -> None:
    """Cut `row` to the part outside an anchor interval of another entity; fully inside it, the row is noise."""
    start, end = pd.Timestamp(row["valid_from"]), row["valid_to"]
    h_start, h_end = pd.Timestamp(holder["valid_from"]), holder["valid_to"]
    if h_end is not None and (end is None or pd.Timestamp(end) > pd.Timestamp(h_end)):
        row["valid_from"] = max(start, pd.Timestamp(h_end))
    elif start < h_start:
        row["valid_to"] = h_start
    else:
        row["status"] = "noise"
        row["evidence"] = f"{row['evidence']}; superseded by the {'curated' if holder['curated'] else 'roster'} interval of {holder['cik']}"
        return
    row["evidence"] = f"{row['evidence']}; clipped by the {'curated' if holder['curated'] else 'roster'} interval of {holder['cik']}"


def _close_on_window(symbol_rows: pd.DataFrame, windows: Mapping[str, pd.Timestamp | None]) -> pd.DataFrame:
    """An open symbol row on a CIK whose consolidating window has closed ends at that window (never current)."""
    out = symbol_rows.copy()
    ends = out["cik"].map(lambda cik: windows.get(cik))
    roster_anchored = pd.Series([ROSTER_SOURCE in sources for sources in out["sources"]], index=out.index, dtype=bool)
    close = out["valid_to"].isna() & ends.notna() & ~roster_anchored
    out.loc[close, "valid_to"] = [
        max(pd.Timestamp(end), pd.Timestamp(start) + pd.Timedelta(days=1))
        for end, start in zip(ends[close], out.loc[close, "valid_from"], strict=True)
    ]
    return out


def _conflict_backlog(symbol_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """One backlog row per universe symbol interval left in `conflict`."""
    conflicts = symbol_rows[symbol_rows["status"].eq("conflict")]
    starts = pd.to_datetime(conflicts["valid_from"]).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(conflicts["valid_to"]).dt.strftime("%Y-%m-%d").fillna("open")
    return [
        {
            "kind": "conflict",
            "canonical_ticker": ticker,
            "company_id": entity,
            "cik": cik,
            "symbol": symbol,
            "detail": f"{symbol} held by another entity within {REUSE_GAP_DAYS} days of {start}..{end}",
        }
        for ticker, entity, cik, symbol, start, end in zip(
            conflicts["canonical_ticker"], conflicts["company_id"], conflicts["cik"], conflicts["symbol"], starts, ends, strict=True
        )
    ]


def _finalise(rows: pd.DataFrame, existing: pd.DataFrame | None, stamp: pd.Timestamp) -> pd.DataFrame:
    """Typed, sorted table rows with `scope_changed_at` carried from `existing` for unchanged scopes."""
    out = rows.copy()
    out["symbol"] = out["symbol"].fillna("").astype(str)
    out["sources"] = [value if isinstance(value, str) else ",".join(sorted(value)) for value in out["sources"]]
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"])
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").astype("float64")
    out["n_observations"] = out["n_observations"].astype("int64")
    out["canonical_ticker"] = out["canonical_ticker"].astype(object).where(out["canonical_ticker"].notna(), None)
    out["scope_changed_at"] = _scope_changed_at(out, existing, stamp)
    return out[list(TABLE_COLUMNS)].sort_values(["company_id", "role", "cik", "symbol", "valid_from"], kind="mergesort", ignore_index=True)


def _scope_key(frame: pd.DataFrame) -> pd.Series:
    """Per-row scope key: the canonical ticker, else the entity id."""
    return frame["canonical_ticker"].where(frame["canonical_ticker"].notna(), frame["company_id"]).astype(str)


def _stamp_group(frame: pd.DataFrame) -> pd.Series:
    """Per-row stamp group: `cik` for the CIK rows (EDGAR and bulk scope), `symbol` for the symbol rows (tapes)."""
    return frame["role"].astype(str).eq(ROLE_SYMBOL).map({True: ROLE_SYMBOL, False: "cik"})


def _scope_signatures(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[tuple[str, ...], ...]]:
    """`{(scope key, stamp group): sorted signature}`: `(cik, role, valid_from, valid_to)` over the CIK rows and
    `(symbol, cik, valid_from, valid_to, status)` over the symbol rows a tape can read (not `dei` alone)."""
    bounds = [pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d").fillna("") for column in ("valid_from", "valid_to")]
    sources = frame["sources"].fillna("").astype(str) if "sources" in frame.columns else pd.Series("", index=frame.index)
    status = frame["status"].fillna("").astype(str) if "status" in frame.columns else pd.Series("", index=frame.index)
    symbol = frame["symbol"].fillna("").astype(str) if "symbol" in frame.columns else pd.Series("", index=frame.index)
    out: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    rows = zip(
        _scope_key(frame), _stamp_group(frame), pad_cik_series(frame["cik"]), frame["role"].astype(str), *bounds, symbol, status, sources, strict=True
    )
    for key, group, cik, role, start, end, sym, stat, src in rows:
        if group == ROLE_SYMBOL:
            if src.strip().lower() == DEI_SOURCE:
                continue
            signature: tuple[str, ...] = (sym, cik, start, end, stat)
        else:
            signature = (cik, role, start, end)
        out.setdefault((key, group), []).append(signature)
    return {key: tuple(sorted(values)) for key, values in out.items()}


def _scope_changed_at(rows: pd.DataFrame, existing: pd.DataFrame | None, stamp: pd.Timestamp) -> pd.Series:
    """The stored timestamp of each unchanged (scope, stamp group); `stamp` for every other.

    CIK rows and symbol rows are stamped apart: a symbol-only change moves the tapes, not the EDGAR relist.
    """
    keys = list(zip(_scope_key(rows), _stamp_group(rows), strict=True))
    if existing is None or existing.empty or not {"role", "canonical_ticker", "scope_changed_at", "valid_to"} <= set(existing.columns):
        return pd.Series(stamp, index=rows.index, dtype="datetime64[ns]")
    before, after = _scope_signatures(existing), _scope_signatures(rows)
    stored_stamps = pd.to_datetime(existing["scope_changed_at"])
    stored = stored_stamps.groupby([_scope_key(existing), _stamp_group(existing)]).max().to_dict()
    kept = {key: stored[key] for key in set(keys) if before.get(key, ()) == after.get(key, ()) and pd.notna(stored.get(key))}
    return pd.Series([kept.get(key, stamp) for key in keys], index=rows.index, dtype="datetime64[ns]")


def derive_lineage_rows(
    history: CikHistory, roster_cik: Mapping[str, str], existing: pd.DataFrame | None, built_at: pd.Timestamp | None = None
) -> LineageBuild:
    """The `universe_cik_symbol` rows: the CIK history's rows plus the symbol intervals attached through it.

    Pure: no DB. `existing` (the stored table) carries `scope_changed_at`; `built_at` pins the change timestamp.
    """
    stamp = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s") if built_at is None else pd.Timestamp(built_at)
    holders = _symbol_holders(history.evidence, history.entity_of, history.ticker_by_entity, roster_cik)
    symbol_rows = _close_on_window(holders[holders["canonical_ticker"].notna()], history.window_end).assign(role=ROLE_SYMBOL, confidence=float("nan"))
    symbol_rows["oracle"] = [history.oracle_of.get(cik, "roster") for cik in symbol_rows["cik"]]
    conflicts = pd.DataFrame(_conflict_backlog(symbol_rows), columns=list(BACKLOG_COLUMNS))
    rows = _finalise(pd.concat([history.rows, symbol_rows[list(TABLE_COLUMNS[:-1])]], ignore_index=True), existing, stamp)
    check_one_entity_per_cik(rows)
    backlog = pd.concat([history.backlog, conflicts], ignore_index=True) if not conflicts.empty else history.backlog
    out = LineageBuild(rows=rows, blocked=history.blocked, backlog=backlog, holders=holders)
    _log_symbols(rows, conflicts)
    return out


def _log_symbols(rows: pd.DataFrame, conflicts: pd.DataFrame) -> None:
    """One INFO summary of the symbol rows, and one WARNING listing the symbol conflicts."""
    symbol_rows = rows[rows["role"].eq(ROLE_SYMBOL)]
    logger.info(
        "universe_cik_symbol: %d row(s) over %d entity(ies); roles %s; symbol statuses %s",
        len(rows),
        rows["company_id"].nunique(),
        dict(rows["role"].value_counts().sort_index()),
        dict(symbol_rows["status"].value_counts().sort_index()),
    )
    if not conflicts.empty:
        logger.warning(
            "universe_cik_symbol: symbol curation backlog %d conflict(s) (not applied):\n%s", len(conflicts), conflicts.to_string(index=False)
        )


def build_entity_lineage(
    context: Context, history: CikHistory, config_dir: str | None = None, *, built_at: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Attach the symbol intervals to this run's CIK history and replace `universe_cik_symbol` unless unchanged.

    Logs the items needing a manual decision, reading tape-mix pairs against the stored `universe_cusip`. Returns the rows.
    """
    roster_cik = context.universe.cik_by_ticker
    existing = _stored_rows(context.store.load(Tables.universe_cik_symbol, project=True, optional=True))
    master = context.store.load(Tables.universe_cusip, columns=list(MASTER_COLUMNS), optional=True)
    build = derive_lineage_rows(history, roster_cik, existing, built_at=built_at)
    out = build.rows
    manual_tenure = load_manual_symbol_tenure(config_dir or context.config_dir)
    validate_manual_tenure_entities(manual_tenure, out, roster_cik)
    n_dei = int(history.evidence["source"].eq(DEI_SOURCE).sum())
    context.log.info(
        f"universe_cik_symbol: {len(manual_tenure)} manual symbol interval(s) resolve to "
        f"their {manual_tenure['canonical_ticker'].nunique()} canonical entity(ies); "
        f"dei evidence {n_dei} (symbol, CIK) interval(s); curation backlog {len(build.backlog)} item(s)"
    )
    flags = identity_flags(out, history.activity, redundant_symbols=context.universe.alias_symbols, backlog=build.backlog, master=master)
    log_identity_flags(context.log, flags)
    if existing is None or existing.empty:
        context.log.info(f"universe_cik_symbol: cold build with {len(out)} row(s) over {out['company_id'].nunique()} entity(ies)")
    else:
        _log_changed_assignments(context, existing, out, roster_cik)
    unchanged = matches_stored(existing, out, Tables.universe_cik_symbol)
    written = 0 if unchanged else save_identity(context.store, Tables.universe_cik_symbol, out)
    if unchanged:
        logger.info("universe_cik_symbol: unchanged (%d row(s)); replace skipped", len(out))
    else:
        logger.info("universe_cik_symbol: wrote %d row(s)", written)
    return out


def _stored_rows(existing: pd.DataFrame | None) -> pd.DataFrame | None:
    """The stored table with its date and timestamp columns parsed (a store may return them as text or `date`)."""
    if existing is None:
        return None
    out = existing.copy()
    for column in ("valid_from", "valid_to", "scope_changed_at"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column])
    return out


def load_stored_lineage(context: Context) -> pd.DataFrame | None:
    """The stored `universe_cik_symbol` with its dates parsed, as `build_security_master` reads a lineage just built; None when absent."""
    return _stored_rows(context.store.load(Tables.universe_cik_symbol, project=True, optional=True))


def _log_changed_assignments(context: Context, existing: pd.DataFrame, out: pd.DataFrame, roster_cik: Mapping[str, str]) -> None:
    """Name every CIK whose entity changed and the current tickers those entities hold."""
    old_map = entity_by_cik_map(existing)
    new_map = entity_by_cik_map(out)
    changed = sorted(cik for cik in set(old_map) | set(new_map) if old_map.get(cik) != new_map.get(cik))
    entity_tickers = {entity_or_singleton(new_map, cik): ticker for ticker, cik in roster_cik.items()}
    affected = sorted({entity_tickers[entity] for cik in changed for entity in (old_map.get(cik), new_map.get(cik)) if entity in entity_tickers})
    context.log.info(
        f"universe_cik_symbol: {len(changed)} changed CIK assignment(s): "
        f"{', '.join(changed) if changed else 'none'}; affected current ticker(s): "
        f"{', '.join(affected) if affected else 'none'}"
    )
