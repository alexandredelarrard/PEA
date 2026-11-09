"""Symbol history of `universe_cik_symbol`: the dated symbol intervals of each company, attached through its CIK history.

Evidence: the filed `symbol_tenure` intervals (`form345`, `dei`, `manual`, minus the cited rejections) and the Sharadar
ticker-change tape (`sharadar`), each tape interval cut at the company's CIK windows. One holder interval per (symbol, CIK):
a manual row as curated, else the union of its filed and tape parts. Anchors clip the derived holders of other companies:
curated rows, then each universe ticker's roster interval, then the tape. A weak single-source interval next to another
company is noise; two derived holders within `REUSE_GAP_DAYS` are both `conflict`. The frozen universe key keeps its rows;
its trading symbol is its company's latest open tape symbol (D-6).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity_contract import (
    DEI_SOURCE,
    FORM345_SOURCE,
    MANUAL_SOURCE,
    ROLE_SYMBOL,
    ROLE_WINDOW,
    SENTINEL_START,
    entity_or_singleton,
)
from src.data_extract.utils.identity.cik_history import BACKLOG_COLUMNS, EVIDENCE_SOURCES, ROSTER_SOURCE, TABLE_COLUMNS, CikHistory
from src.data_store.schema import Tables
from src.utils.string import pad_cik_series, yahoo_symbol

logger = logging.getLogger(__name__)

#: The Sharadar ticker-change tape: `tickerchangefrom` names the old symbol, `tickerchangeto` the new one, on one date,
#: both keyed by the permaticker's CURRENT Sharadar spelling.
TAPE_SOURCE = "sharadar"
TICKER_CHANGE_ACTIONS = ("tickerchangefrom", "tickerchangeto")
TAPE_ACTION_COLUMNS = ("date", "action", "ticker", "contraticker")
TAPE_TICKER_COLUMNS = ("permaticker", "ticker", "firstpricedate")
#: Two companies holding one symbol closer than this (or overlapping) is a reuse conflict; exchanges allow reuse after 90 days.
REUSE_GAP_DAYS = 90
#: A single-source interval with at most this many filings, next to another company's interval, is noise.
NOISE_MAX_OBSERVATIONS = 5
#: A `dei` interval whose last filing is this close to the latest `dei` filing is still current.
CURRENT_RECENCY_DAYS = 120

_CHAIN_COLUMNS = ["ticker", "symbol", "first", "end", "evidence"]
_EVIDENCE_COLUMNS = ["symbol", "cik", "source", "first", "end", "n", "evidence"]
_HOLDER_SOURCES = (*EVIDENCE_SOURCES, TAPE_SOURCE)
_FAR = pd.Timestamp("2262-01-01")


class ManualTenureEntityError(ValueError):
    """A manual ticker interval names a CIK outside its canonical ticker's company."""


@dataclass(frozen=True)
class SymbolHistory:
    """`rows` of `universe_cik_symbol` (CIK and symbol rows), the curation backlog and refused merges, every classified
    holder interval (other companies included) and each universe ticker's `trading_symbol` from `trading_symbol_from`."""

    rows: pd.DataFrame
    backlog: pd.DataFrame
    blocked: pd.DataFrame
    holders: pd.DataFrame
    trading: pd.DataFrame


# The Sharadar tape
def tape_chains(universe: pd.DataFrame, sharadar: pd.DataFrame | None, actions: pd.DataFrame | None) -> pd.DataFrame:
    """`[ticker, symbol, first, end, evidence]`: each universe ticker's Sharadar symbols in date order, the first from its
    permaticker's first price date, the last open; a ticker without a change has none."""
    # nothing to chain without the Sharadar tickers and their changes
    empty = pd.DataFrame(columns=_CHAIN_COLUMNS)
    if sharadar is None or actions is None or sharadar.empty or actions.empty:
        return empty

    # every Sharadar spelling of each universe permaticker and its first price date
    spellings = sharadar[list(TAPE_TICKER_COLUMNS)].rename(columns={"ticker": "spelling"})
    perms = universe[["ticker", "permaticker"]].dropna().merge(spellings, on="permaticker", how="inner")
    first = pd.to_datetime(perms["firstpricedate"]).groupby(perms["ticker"]).min()

    # the dated changes on those spellings, in roster spelling (`HUB.B` -> `HUB-B`)
    changes = _paired_changes(actions).merge(perms[["ticker", "spelling"]].drop_duplicates(), on="spelling", how="inner")
    changes = changes.assign(old=changes["old"].map(yahoo_symbol), new=changes["new"].map(yahoo_symbol))
    return _chain_intervals(changes[changes["old"].ne(changes["new"])], first) if not changes.empty else empty


def _paired_changes(actions: pd.DataFrame) -> pd.DataFrame:
    """`[spelling, date, old, new]`: each `tickerchangefrom` paired with the `tickerchangeto` of its spelling and date."""
    frame = actions.assign(date=pd.to_datetime(actions["date"]), contraticker=actions["contraticker"].astype(str))
    old = frame[frame["action"].eq(TICKER_CHANGE_ACTIONS[0])].rename(columns={"contraticker": "old"})[["ticker", "date", "old"]]
    new = frame[frame["action"].eq(TICKER_CHANGE_ACTIONS[1])].rename(columns={"contraticker": "new"})[["ticker", "date", "new"]]
    paired = old.merge(new, on=["ticker", "date"], how="inner").rename(columns={"ticker": "spelling"})
    return paired.drop_duplicates(["spelling", "date"], keep="first")


def _chain_intervals(changes: pd.DataFrame, first: pd.Series) -> pd.DataFrame:
    """One interval per symbol of each ticker's change chain: from the change into it (the first price date for the oldest)
    to the change out of it (open for the newest); an oldest symbol renamed before the first price date has none."""
    rows: list[dict[str, Any]] = []
    for ticker, group in changes.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"], kind="mergesort").groupby("ticker", sort=True):
        dates = list(group["date"])
        labels = [f"{o}->{n} {d.date()}" for o, n, d in zip(group["old"], group["new"], dates, strict=True)]
        start = first.get(ticker)
        symbols = [group["old"].iloc[0], *group["new"]]
        starts = [SENTINEL_START if pd.isna(start) else pd.Timestamp(start), *dates]
        notes = ["; ".join(labels[max(k - 1, 0) : k + 1]) for k in range(len(symbols))]
        rows += [
            {"ticker": ticker, "symbol": s, "first": a, "end": b, "evidence": note}
            for s, a, b, note in zip(symbols, starts, [*dates, pd.NaT], notes, strict=True)
            if pd.isna(b) or a < b
        ]
    return pd.DataFrame(rows, columns=_CHAIN_COLUMNS)


def symbol_evidence(history: CikHistory, chains: pd.DataFrame) -> pd.DataFrame:
    """The CIK history's filed evidence plus each tape interval cut at the CIK windows of its ticker's company."""
    # the dated CIK windows of each universe company
    rows = history.rows
    windows = rows.loc[rows["role"].eq(ROLE_WINDOW) & rows["canonical_ticker"].notna(), ["canonical_ticker", "cik", "valid_from", "valid_to"]]
    tape = chains.merge(windows.rename(columns={"canonical_ticker": "ticker"}), on="ticker", how="inner")

    # each tape interval inside each window it overlaps
    start = pd.concat([pd.to_datetime(tape["first"]), pd.to_datetime(tape["valid_from"])], axis=1).max(axis=1)
    end = pd.concat([pd.to_datetime(tape["end"]).fillna(_FAR), pd.to_datetime(tape["valid_to"]).fillna(_FAR)], axis=1).min(axis=1)
    cut = pd.DataFrame(
        {
            "symbol": tape["symbol"],
            "cik": tape["cik"],
            "source": TAPE_SOURCE,
            "first": start,
            "end": end.mask(end.eq(_FAR)),
            "n": 0,
            "evidence": tape["evidence"],
        }
    )
    cut = cut[cut["first"] < end].astype({"n": "int64"})
    return pd.concat([history.evidence, cut[_EVIDENCE_COLUMNS]], ignore_index=True)


# Holder intervals
def symbol_holders(evidence: pd.DataFrame, history: CikHistory, roster_cik: Mapping[str, str]) -> pd.DataFrame:
    """Every holder interval of the symbols the universe companies ever held (other companies included), classified."""
    # the symbols universe companies held, filed or on the tape, and the roster tickers
    entity_of, ticker_by_entity = history.entity_of, history.ticker_by_entity
    universe_ciks = {cik for cik, entity in entity_of.items() if entity in ticker_by_entity}
    scope = set(evidence.loc[evidence["cik"].isin(universe_ciks), "symbol"]) | set(roster_cik)
    in_scope = evidence[evidence["symbol"].isin(scope)]
    dei_ends = evidence.loc[evidence["source"].eq(DEI_SOURCE), "end"]
    current_after = dei_ends.max() - pd.Timedelta(days=CURRENT_RECENCY_DAYS + 1) if not dei_ends.empty else None

    # one interval per (symbol, CIK), then each roster ticker's current interval
    rows = [row for _, group in in_scope.groupby(["symbol", "cik"], sort=True) for row in _merge_holder(group, current_after)]
    rows += _roster_rows(rows, roster_cik)
    for row in rows:
        row["entity"], row["status"] = entity_of.get(str(row["cik"]), f"E{row['cik']}"), ""

    # each symbol's holders: noise, clipped by the curated, roster and tape anchors of other companies, reuse conflicts
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_symbol.setdefault(str(row["symbol"]), []).append(row)
    for holders in by_symbol.values():
        _mark_noise(holders, _gaps(holders))
        order = sorted(holders, key=lambda row: pd.Timestamp(row["valid_from"]))
        for anchor in ("curated", "roster", "tape"):
            _clip_by_anchors(holders, order, anchor)
        _mark_conflicts(holders, _gaps(holders))
        _final_status(holders)

    # the frame: company and universe ticker of each holder
    out = [row for symbol in sorted(by_symbol) for row in by_symbol[symbol]]
    frame = pd.DataFrame(out).drop(columns=["entity", "curated", "roster", "tape"])
    frame["company_id"] = [entity_of.get(cik, f"E{cik}") for cik in frame["cik"]]
    frame["canonical_ticker"] = frame["company_id"].map(ticker_by_entity)
    frame = frame.assign(valid_from=pd.to_datetime(frame["valid_from"]), valid_to=pd.to_datetime(frame["valid_to"]))
    return frame.sort_values(["symbol", "valid_from", "cik"], kind="mergesort", ignore_index=True)


def _merge_holder(group: pd.DataFrame, current_after: pd.Timestamp | None) -> list[dict[str, Any]]:
    """One (symbol, CIK)'s intervals: manual rows as they are, else the filed and tape parts merged into one interval."""
    # manual rows win as they are
    symbol, cik = str(group["symbol"].iloc[0]), str(group["cik"].iloc[0])
    manual = group[group["source"].eq(MANUAL_SOURCE)]
    observed = int(group.loc[group["source"].ne(MANUAL_SOURCE), "n"].sum())
    curated = [
        {"symbol": symbol, "cik": cik, "valid_from": row.first, "valid_to": None if pd.isna(row.end) else row.end, "sources": frozenset({MANUAL_SOURCE}),
         "n_observations": observed, "evidence": f"manual: {row.evidence}", "curated": True, "roster": False, "tape": False}
        for row in manual.sort_values("first").itertuples(index=False)
    ]  # fmt: skip
    derived = group[group["source"].isin(_HOLDER_SOURCES)]
    if curated or derived.empty:
        return curated

    # the union of the filed and tape parts: open when Forms 3/4/5 or the tape run on, or the cover page is current
    parts = {source: derived[derived["source"].eq(source)] for source in _HOLDER_SOURCES}
    dei = parts[DEI_SOURCE]
    is_open = bool(parts[FORM345_SOURCE]["end"].isna().any() or parts[TAPE_SOURCE]["end"].isna().any())
    is_open = is_open or (current_after is not None and not dei.empty and dei["end"].max() > current_after)
    notes = [
        f"{source} {part['first'].min().date()}..{'open' if part['end'].isna().any() else (part['end'].max() - pd.Timedelta(days=1)).date()} "
        f"n={int(part['n'].sum())}" + (f" ({part['evidence'].iloc[-1]})" if part["evidence"].iloc[-1] else "")
        for source, part in parts.items()
        if not part.empty
    ]
    return [
        {"symbol": symbol, "cik": cik, "valid_from": derived["first"].min(), "valid_to": None if is_open else derived["end"].max(),
         "sources": frozenset(derived["source"]), "n_observations": int(derived["n"].sum()), "evidence": "; ".join(notes),
         "curated": False, "roster": False, "tape": not parts[TAPE_SOURCE].empty}
    ]  # fmt: skip


def _roster_rows(rows: list[dict[str, Any]], roster_cik: Mapping[str, str]) -> list[dict[str, Any]]:
    """Mark each roster ticker's latest interval on its roster CIK as current; add one when no filing spells the ticker."""
    added: list[dict[str, Any]] = []
    for ticker, cik in sorted(roster_cik.items()):
        latest = max((row for row in rows if row["symbol"] == ticker and row["cik"] == cik), key=lambda row: row["valid_from"], default=None)
        if latest is not None and not (latest["curated"] and latest["valid_to"] is not None):
            latest.update(roster=True, sources=frozenset(latest["sources"]) | {ROSTER_SOURCE}, valid_to=None)
            continue
        start = latest["valid_to"] if latest is not None else min((row["valid_from"] for row in rows if row["cik"] == cik), default=SENTINEL_START)
        evidence = "roster: the universe names this ticker on this CIK today; no filing spells it this way"
        added.append({"symbol": ticker, "cik": cik, "valid_from": start, "valid_to": None, "sources": frozenset({ROSTER_SOURCE}), "n_observations": 0,
                      "evidence": evidence, "curated": False, "roster": True, "tape": False})  # fmt: skip
    return added


def _gaps(rows: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    """Days from the end of one half-open interval to the start of the other, per pair of row positions (open end = +inf);
    negative when they overlap."""
    bounds = [(pd.Timestamp(row["valid_from"]), None if row["valid_to"] is None else pd.Timestamp(row["valid_to"])) for row in rows]
    gaps: dict[tuple[int, int], float] = {}
    for (i, (a_from, a_to)), (j, (b_from, b_to)) in combinations(enumerate(bounds), 2):
        a_after_b = (a_from - b_to).days if b_to is not None else float("-inf")
        gaps[(i, j)] = gaps[(j, i)] = max(a_after_b, (b_from - a_to).days if a_to is not None else float("-inf"))
    return gaps


def _mark_noise(rows: list[dict[str, Any]], gaps: Mapping[tuple[int, int], float]) -> None:
    """A weak row (one filed source, at most `NOISE_MAX_OBSERVATIONS` filings, no anchor) is noise when another company
    held the symbol within `REUSE_GAP_DAYS`, or at any time with a stronger interval."""
    filed = set(EVIDENCE_SOURCES)
    weak = [
        not (row["curated"] or row["roster"] or row["tape"])
        and len(frozenset(row["sources"]) & filed) == 1
        and int(row["n_observations"]) <= NOISE_MAX_OBSERVATIONS
        for row in rows
    ]
    for i, row in enumerate(rows):
        others = [j for j, other in enumerate(rows) if other["entity"] != row["entity"]]
        if weak[i] and any(not weak[j] or gaps[(i, j)] < REUSE_GAP_DAYS for j in others):
            row["status"] = "noise"


def _clip_by_anchors(rows: list[dict[str, Any]], order: list[dict[str, Any]], anchor: str) -> None:
    """Clip each unclassified lower-ranked row by every overlapping `anchor` row (curated > roster > tape) of another
    company, in start order; a row fully inside an anchor is noise."""
    rank = ("curated", "roster", "tape")
    higher, ranked = rank[: rank.index(anchor)], rank[: rank.index(anchor) + 1]
    anchors = [row for row in order if row[anchor] and not any(row[kind] for kind in higher)]
    targets = [row for row in rows if not any(row[kind] for kind in ranked)]
    for row, holder in ((row, holder) for row in targets for holder in anchors):
        start, end = pd.Timestamp(row["valid_from"]), None if row["valid_to"] is None else pd.Timestamp(row["valid_to"])
        h_start, h_end = pd.Timestamp(holder["valid_from"]), None if holder["valid_to"] is None else pd.Timestamp(holder["valid_to"])
        overlaps = holder["entity"] != row["entity"] and (end is None or h_start < end) and (h_end is None or start < h_end)
        if row["status"] or not overlaps:
            continue
        keeps_tail = h_end is not None and (end is None or end > h_end)
        keeps_head = not keeps_tail and start < h_start
        row["valid_from"] = max(start, h_end) if keeps_tail and h_end is not None else row["valid_from"]
        row["valid_to"] = h_start if keeps_head else row["valid_to"]
        row["status"] = "" if keeps_tail or keeps_head else "noise"
        row["evidence"] = f"{row['evidence']}; {'clipped' if keeps_tail or keeps_head else 'superseded'} by the {anchor} interval of {holder['cik']}"


def _mark_conflicts(rows: list[dict[str, Any]], gaps: Mapping[tuple[int, int], float]) -> None:
    """Derived rows of two companies within `REUSE_GAP_DAYS` are both `conflict`; so are two overlapping tape rows."""
    live = [not (row["status"] or row["curated"] or row["roster"]) for row in rows]
    for i, j in (pair for pair in combinations(range(len(rows)), 2) if live[pair[0]] and live[pair[1]]):
        a, b = rows[i], rows[j]
        derived = not a["tape"] and not b["tape"] and gaps[(i, j)] < REUSE_GAP_DAYS
        if a["entity"] != b["entity"] and (derived or (a["tape"] and b["tape"] and gaps[(i, j)] < 0)):
            a["status"] = b["status"] = "conflict"


def _final_status(rows: list[dict[str, Any]]) -> None:
    """`curated` for curated rows; else, when unclassified, `corroborated` with both filed sources, `single_source` otherwise."""
    for row in rows:
        filed = frozenset(row["sources"]) & set(EVIDENCE_SOURCES)
        corroborated = "corroborated" if len(filed) == len(EVIDENCE_SOURCES) else "single_source"
        row["status"] = "curated" if row["curated"] else row["status"] or corroborated


# The table rows
def symbol_rows(history: CikHistory, holders: pd.DataFrame, existing: pd.DataFrame | None, built_at: pd.Timestamp | None) -> SymbolHistory:
    """The `universe_cik_symbol` rows: the CIK history's rows plus the universe companies' symbol intervals, an ended
    company's open rows closed at its end, each scope's `scope_changed_at` carried from `existing` while unchanged
    (`built_at`, default now, otherwise)."""
    # the universe companies' holder intervals, closed at their CIK window, with the oracle that joined the CIK
    stamp = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s") if built_at is None else pd.Timestamp(built_at)
    symbols = _close_on_window(holders[holders["canonical_ticker"].notna()], history.window_end).assign(role=ROLE_SYMBOL, confidence=float("nan"))
    symbols["oracle"] = [history.oracle_of.get(cik, "roster") for cik in symbols["cik"]]
    conflicts = _conflict_backlog(symbols)

    # typed rows closed at their company's end (REQ-006, before the stamps), and the scope key of each row, stored and new
    rows = _close_at_end(_typed_rows(pd.concat([history.rows, symbols[list(TABLE_COLUMNS[:-1])]], ignore_index=True)), history.ended_on)
    usable = existing is not None and not existing.empty and {"role", "canonical_ticker", "scope_changed_at", "valid_to"} <= set(existing.columns)
    stored = existing.assign(cik=pad_cik_series(existing["cik"])) if usable and existing is not None else rows.iloc[:0]
    keys, stored_keys = _scope_keys(rows), _scope_keys(stored)

    # each scope keeps its stored stamp while its signature is unchanged
    padded = rows.assign(cik=pad_cik_series(rows["cik"]))
    before, after, stamps = _scope_signatures(stored, stored_keys), _scope_signatures(padded, keys), _stored_stamps(stored, stored_keys)
    kept = [stamps[key] if key in stamps and before.get(key, ()) == after.get(key, ()) else stamp for key in keys]
    rows["scope_changed_at"] = pd.Series(kept, index=rows.index, dtype="datetime64[ns]")
    rows = rows[list(TABLE_COLUMNS)].sort_values(["company_id", "role", "cik", "symbol", "valid_from"], kind="mergesort", ignore_index=True)

    # the build
    backlog = pd.concat([history.backlog, conflicts], ignore_index=True) if not conflicts.empty else history.backlog
    return SymbolHistory(rows=rows, backlog=backlog, blocked=history.blocked, holders=holders, trading=trading_symbols(rows))


def _close_on_window(symbol_rows: pd.DataFrame, windows: Mapping[str, pd.Timestamp | None]) -> pd.DataFrame:
    """An open symbol row on a CIK whose consolidating window has closed ends at that window (never current)."""
    out = symbol_rows.copy()
    ends = out["cik"].map(lambda cik: windows.get(cik))
    roster_anchored = pd.Series([ROSTER_SOURCE in sources for sources in out["sources"]], index=out.index, dtype=bool)
    close = out["valid_to"].isna() & ends.notna() & ~roster_anchored
    closed = [
        max(pd.Timestamp(end), pd.Timestamp(start) + pd.Timedelta(days=1))
        for end, start in zip(ends[close], out.loc[close, "valid_from"], strict=True)
    ]
    out.loc[close, "valid_to"] = closed
    return out


def _close_at_end(rows: pd.DataFrame, ended_on: Mapping[str, pd.Timestamp]) -> pd.DataFrame:
    """Every open CIK window and symbol interval of an ended company closes the day after its `ended_on` (`valid_to` is
    exclusive), never before the row's own start."""
    owners = rows[rows["canonical_ticker"].isin(list(ended_on))].drop_duplicates("canonical_ticker")
    end_of = {str(c): ended_on[t] + pd.Timedelta(days=1) for c, t in zip(owners["company_id"], owners["canonical_ticker"], strict=True)}
    ends = rows["company_id"].astype(str).map(end_of)
    close = rows["valid_to"].isna() & ends.notna() & rows["role"].isin([ROLE_WINDOW, ROLE_SYMBOL])
    out = rows.copy()
    starts = pd.to_datetime(out.loc[close, "valid_from"]) + pd.Timedelta(days=1)
    out.loc[close, "valid_to"] = [max(pd.Timestamp(end), start) for end, start in zip(ends[close], starts, strict=True)]
    return out


def _conflict_backlog(symbol_rows: pd.DataFrame) -> pd.DataFrame:
    """One backlog row per universe symbol interval left in `conflict`."""
    conflicts = symbol_rows[symbol_rows["status"].eq("conflict")]
    starts = pd.to_datetime(conflicts["valid_from"]).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(conflicts["valid_to"]).dt.strftime("%Y-%m-%d").fillna("open")
    rows = [
        {"kind": "conflict", "canonical_ticker": ticker, "company_id": entity, "cik": cik, "symbol": symbol,
         "detail": f"{symbol} held by another entity within {REUSE_GAP_DAYS} days of {start}..{end}"}
        for ticker, entity, cik, symbol, start, end in zip(
            conflicts["canonical_ticker"], conflicts["company_id"], conflicts["cik"], conflicts["symbol"], starts, ends, strict=True
        )
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=list(BACKLOG_COLUMNS))


def _typed_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Table-typed rows: text symbol and sources, parsed dates, float confidence, integer counts, None for no ticker."""
    out = rows.copy()
    out["symbol"] = out["symbol"].fillna("").astype(str)
    out["sources"] = [value if isinstance(value, str) else ",".join(sorted(value)) for value in out["sources"]]
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"])
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").astype("float64")
    out["n_observations"] = out["n_observations"].astype("int64")
    out["canonical_ticker"] = out["canonical_ticker"].astype(object).where(out["canonical_ticker"].notna(), None)
    return out


def _scope_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Per row: (scope key = the canonical ticker, else the company id; stamp group = `symbol` for symbol rows, `cik` else).

    CIK rows and symbol rows are stamped apart: a symbol-only change moves the tapes, not the EDGAR relist.
    """
    scope = frame["canonical_ticker"].where(frame["canonical_ticker"].notna(), frame["company_id"]).astype(str)
    group = frame["role"].astype(str).eq(ROLE_SYMBOL).map({True: ROLE_SYMBOL, False: "cik"})
    return list(zip(scope, group, strict=True))


def _scope_signatures(frame: pd.DataFrame, keys: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[tuple[str, ...], ...]]:
    """`{scope key: sorted signature}`: `(cik, role, valid_from, valid_to)` over the CIK rows and `(symbol, cik, valid_from,
    valid_to, status)` over the symbol rows a tape can read (not `dei` alone); `frame` CIKs padded."""
    # the compared fields of every row, blank where the stored frame lacks a column
    blank = pd.Series("", index=frame.index)
    bounds = [pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d").fillna("") for column in ("valid_from", "valid_to")]
    sources, status, symbol = (frame[c].fillna("").astype(str) if c in frame.columns else blank for c in ("sources", "status", "symbol"))
    fields = zip(keys, frame["cik"].astype(str), frame["role"].astype(str), *bounds, symbol, status, sources, strict=True)

    # one sorted tuple of signatures per key
    out: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    for key, cik, role, start, end, sym, stat, src in fields:
        if key[1] == ROLE_SYMBOL and src.strip().lower() == DEI_SOURCE:
            continue
        out.setdefault(key, []).append((sym, cik, start, end, stat) if key[1] == ROLE_SYMBOL else (cik, role, start, end))
    return {key: tuple(sorted(values)) for key, values in out.items()}


def _stored_stamps(stored: pd.DataFrame, keys: list[tuple[str, str]]) -> dict[tuple[str, str], pd.Timestamp]:
    """The latest stored `scope_changed_at` of each scope key (none for a key without one)."""
    if stored.empty:
        return {}
    stamps = pd.Series(pd.to_datetime(stored["scope_changed_at"]).to_numpy(), index=pd.MultiIndex.from_tuples(keys))
    latest = stamps.groupby(level=[0, 1]).max()
    return {key: value for key, value in latest.items() if pd.notna(value)}


def trading_symbols(rows: pd.DataFrame) -> pd.DataFrame:
    """`[ticker, trading_symbol, trading_symbol_from]`: each universe ticker's latest open tape symbol, else its own key."""
    # the open, readable tape rows of each universe ticker's company
    symbols = rows[rows["role"].eq(ROLE_SYMBOL) & rows["canonical_ticker"].notna()]
    tape = symbols["sources"].str.split(",").map(lambda parts: TAPE_SOURCE in parts)
    current = symbols[tape & symbols["valid_to"].isna() & ~symbols["status"].isin(["noise", "conflict"])]
    latest = current.sort_values(["canonical_ticker", "valid_from", "symbol"], kind="mergesort").groupby("canonical_ticker").tail(1)

    # every universe ticker: its tape symbol, else its key
    tickers = pd.DataFrame({"ticker": pd.Series(sorted(set(symbols["canonical_ticker"])), dtype=object)})
    found = latest.rename(columns={"canonical_ticker": "ticker", "symbol": "trading_symbol", "valid_from": "trading_symbol_from"})
    out = tickers.merge(found[["ticker", "trading_symbol", "trading_symbol_from"]], on="ticker", how="left")
    return out.assign(trading_symbol=out["trading_symbol"].fillna(out["ticker"]))


# Checks and logs
def validate_manual_tenure_entities(manual: pd.DataFrame, entity_by_cik: Mapping[str, str], roster_cik: Mapping[str, str]) -> None:
    """Require each manual CIK to belong to its configured current ticker's company (`entity_by_cik` of the built rows)."""
    # each manual interval's company against its ticker's roster company
    errors: list[str] = []
    for row in manual.itertuples(index=False):
        home_cik = roster_cik.get(str(row.canonical_ticker))
        expected = None if home_cik is None else entity_or_singleton(entity_by_cik, home_cik)
        actual = entity_or_singleton(entity_by_cik, str(row.issuer_cik))
        if actual != expected:
            errors.append(
                f"{row.canonical_ticker}: absent from the current roster"
                if home_cik is None
                else f"{row.canonical_ticker}/{row.symbol}/{row.issuer_cik}: manual entity {actual}, roster entity {expected}"
            )

    # one error naming them all
    if errors:
        raise ManualTenureEntityError("symbol_tenure_manual contains CIKs outside their canonical current entity: " + "; ".join(errors))


def log_symbol_history(context: Context, build: SymbolHistory, manual: pd.DataFrame, n_dei: int, cold: bool) -> None:
    """One INFO summary of the rows and trading symbols; the symbol conflicts in detail at DEBUG (PD-14: each is a
    `conflict` item of the publish review)."""
    # the rows, their roles and the symbol statuses
    rows = build.rows
    statuses = {str(k): int(n) for k, n in rows.loc[rows["role"].eq(ROLE_SYMBOL), "status"].value_counts().sort_index().items()}
    logger.info(
        "universe_cik_symbol: %d row(s) over %d entity(ies); roles %s; symbol statuses %s",
        len(rows),
        rows["company_id"].nunique(),
        {str(k): int(n) for k, n in rows["role"].value_counts().sort_index().items()},
        statuses,
    )

    # the manual intervals, the evidence, the backlog and the keys trading under another symbol (REQ-007)
    renamed = build.trading[build.trading["trading_symbol"].ne(build.trading["ticker"])]
    moves = ", ".join(
        f"{t}->{s} from {pd.Timestamp(d).date()}"
        for t, s, d in zip(renamed["ticker"], renamed["trading_symbol"], renamed["trading_symbol_from"], strict=True)
    )
    context.log.info(
        f"universe_cik_symbol: {len(manual)} manual symbol interval(s) resolve to their {manual['canonical_ticker'].nunique()} canonical entity(ies); "
        f"dei evidence {n_dei} (symbol, CIK) interval(s); curation backlog {len(build.backlog)} item(s); "
        f"{len(renamed)} universe ticker(s) trading under another symbol: {moves or 'none'}"
        + (f"; cold build with {len(rows)} row(s) over {rows['company_id'].nunique()} entity(ies)" if cold else "")
    )

    # the symbol conflicts, never applied
    conflicts = build.backlog[build.backlog["kind"].eq("conflict")] if not build.backlog.empty else build.backlog
    if not conflicts.empty:
        logger.debug(
            "universe_cik_symbol: symbol curation backlog, %d conflict(s) (not applied):\n%s", len(conflicts), conflicts.to_string(index=False)
        )


def log_changed_assignments(context: Context, old_map: Mapping[str, str], new_map: Mapping[str, str], roster_cik: Mapping[str, str]) -> None:
    """Name every CIK whose company changed and the current tickers those companies hold."""
    changed = sorted(cik for cik in set(old_map) | set(new_map) if old_map.get(cik) != new_map.get(cik))
    entity_tickers = {new_map.get(cik, f"E{cik}"): ticker for ticker, cik in roster_cik.items()}
    affected = sorted({entity_tickers[entity] for cik in changed for entity in (old_map.get(cik), new_map.get(cik)) if entity in entity_tickers})
    context.log.info(
        f"universe_cik_symbol: {len(changed)} changed CIK assignment(s): {', '.join(changed) if changed else 'none'}; "
        f"affected current ticker(s): {', '.join(affected) if affected else 'none'}"
    )


def parse_stored_rows(existing: pd.DataFrame | None) -> pd.DataFrame | None:
    """The stored table with its date and timestamp columns parsed (a store may return them as text or `date`)."""
    if existing is None:
        return None
    return existing.assign(
        **{column: pd.to_datetime(existing[column]) for column in ("valid_from", "valid_to", "scope_changed_at") if column in existing.columns}
    )


def load_stored_lineage(context: Context) -> pd.DataFrame | None:
    """The stored `universe_cik_symbol` with its dates parsed, as the CUSIP history reads a lineage just built; None when absent."""
    return parse_stored_rows(context.store.load(Tables.universe_cik_symbol, project=True, optional=True))
