"""Identity items needing a manual decision, classified once for the identity build and the identity validator.

The item stages are pure: `universe_cik_symbol` rows, each CIK's filing activity (`cik_activity`), optionally the
`universe_cusip` lines (which decide tape-mix pairs they already stamp) and, from the build only, its grey-band backlog
and refused merges. `identity_flags` composes them for the validator and the co-registrant reading of `security_lines`.
The build's `run_publish` matches each item to the reviewed `resolutions` of the owning JSONs (REQ-031) and logs a
WARNING block of the unreviewed action items only, everything else in one INFO summary (REQ-032).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.string import pad_cik_series, squash

#: The seam margin: filings this close to a register date are observation lag, not a dispute.
MARGIN = pd.Timedelta(days=31)
BLOCK_TITLE = "IDENTITY ITEMS NEEDING A MANUAL DECISION"
FLAG_COLUMNS = ("kind", "action", "ticker", "ciks", "evidence", "suggested_action", "config_file")
KIND_ORDER = (
    "missing_cutover",
    "register_date_disputed",
    "incorrect_cik_window",
    "missing_sec_filing",
    "end_date_disputed",
    "grey_band",
    "refused_merge",
    "manual_review",
    "conflict",
    "vendor_coverage_gap",
    "vendor_series_other_company",
    "traded_security_mismatch",
    "noise",
    "automatic_cik_window",
    "co_registrant",
)
_REGISTER = "configs/sec/registrant_cutover.json"
_MANUAL = "configs/sec/entity_lineage_manual.json"
_SYMBOLS = "configs/sec/symbol_tenure_manual.json"
_ACTIVITY_SOURCES = ("dei", "form345")
_TAPE_SOURCES = ("form345", "manual")
_FAR = pd.Timestamp("2262-01-01")
#: The stored open start of a window (`common.identity_contract.SENTINEL_START`).
_SENTINEL = pd.Timestamp("1900-01-01")
#: The `universe_cusip` columns the tape-mix rule reads.
MASTER_COLUMNS = ("canonical_company", "source_symbol", "lineage_role", "lineage_reason", "valid_from", "valid_to")
_LINEAGE_COLUMNS = ("company_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources", "oracle", "evidence")
#: `sources` of a `cik_window` row dated by the automatic rule (both filed evidence sources agreeing).
_AUTOMATIC_SOURCES = "dei,form345"
_AUTOMATIC_NOTE = "automatic window: "
Span = tuple[pd.Timestamp, pd.Timestamp]
SpanMap = dict[str, tuple[pd.Timestamp, pd.Timestamp, str]]
#: REQ-031: the `resolutions` list of each owning JSON (under `configs/sec/`); an item is matched on `ITEM_KEY`, verbatim.
RESOLUTIONS_KEY = "resolutions"
RESOLUTION_FILES = ("symbol_tenure_manual.json", "entity_lineage_manual.json", "security_master_manual.json")
ITEM_KEY = ("kind", "ticker", "ciks")
RESOLUTION_COLUMNS = (*ITEM_KEY, "decision", "evidence", "decided_on", "config_file")
#: A cited source: a URL or an SEC accession number.
_CITATION = re.compile(r"https?://\S+|\b\d{10}-\d{2}-\d{6}\b")


class ResolutionError(ValueError):
    """A `resolutions` entry missing a field, citing no URL or accession, or reviewing an item twice."""


@dataclass(frozen=True)
class FlagReview:
    """The items split by review: `unreviewed` action items (the WARNING block), `information` items, `resolved` items
    (with their `decision`) and the `stale` resolutions no item matches any more."""

    unreviewed: pd.DataFrame
    information: pd.DataFrame
    resolved: pd.DataFrame
    stale: pd.DataFrame


@dataclass(frozen=True)
class MixPair:
    """A second tape symbol dated to a ticker while the ticker trades: the merged overlap and its pieces per CIK."""

    ticker: str
    symbol: str
    why: str
    text: str
    by_cik: dict[str, list[Span]]


# Inputs
def cik_activity(evidence: pd.DataFrame) -> pd.DataFrame:
    """`[cik, source, first, last]` per CIK and evidence source; `last` is NaT while a Form 3/4/5 tenure is open.

    `dei` rows are cover-page (10-K/10-Q) symbols, `form345` rows issuer-side Form 3/4/5 symbols (`symbol_tenure` shape).
    """
    rows = evidence[evidence["source"].isin(_ACTIVITY_SOURCES)]
    frame = pd.DataFrame(
        {
            "cik": pad_cik_series(rows["issuer_cik"]),
            "source": rows["source"].astype(str),
            "first": pd.to_datetime(rows["valid_from"]),
            "last": pd.to_datetime(rows["valid_to"]).fillna(_FAR),
        }
    )
    out = frame.groupby(["cik", "source"], as_index=False).agg(first=("first", "min"), last=("last", "max"))
    out["last"] = out["last"].where(out["last"].ne(_FAR))
    return out


def flag_rows(lineage: pd.DataFrame) -> pd.DataFrame:
    """The lineage rows the rules read: padded CIKs, text symbols and sources, parsed dates, company rows only."""
    rows = lineage.reindex(columns=list(_LINEAGE_COLUMNS))
    rows["cik"] = pad_cik_series(rows["cik"])
    rows["symbol"] = rows["symbol"].fillna("").astype(str)
    rows["sources"] = rows["sources"].fillna("").astype(str)
    rows["valid_from"] = pd.to_datetime(rows["valid_from"])
    rows["valid_to"] = pd.to_datetime(rows["valid_to"])
    return rows[rows["canonical_ticker"].notna()]


def master_lines(master: pd.DataFrame | None) -> pd.DataFrame | None:
    """Decided `universe_cusip` lines keyed by squashed symbol, dates parsed; None when there are none.

    An `unclassified` exclusion is the build's default, not a decision, so it covers nothing.
    """
    if master is None or master.empty:
        return None
    out = master.reindex(columns=list(MASTER_COLUMNS))
    out = out[out["lineage_reason"].ne("unclassified")].copy()
    out["canonical_company"] = out["canonical_company"].astype(str)
    out["key"] = out["source_symbol"].map(squash)
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    out["valid_to"] = pd.to_datetime(out["valid_to"])
    return out


# The item stages
def cik_items(rows: pd.DataFrame, activity: pd.DataFrame, backlog: pd.DataFrame | None = None) -> list[dict[str, object]]:
    """Uncurated extra CIKs, disputed register seams, the build's grey-band links and automatically dated windows."""
    # the filing spans of every CIK (all sources, cover pages alone) and the tape symbol rows
    spans, cover = _spans(activity), _spans(activity[activity["source"].eq("dei")])
    tape = _tape(rows[rows["role"].eq("symbol")])

    # one item per uncurated extra CIK (a takeover read off its company's tape rows) and per disputed register seam
    extra = [
        (ticker, other, home, _takeover(tape[tape["company_id"].eq(entity)], ticker, other, home))
        for entity, ticker, other, home in _extra_ciks(rows)
    ]
    items = [_pair_item(ticker, other, home, spans, takeover) for ticker, other, home, takeover in extra]
    items += _register_items(rows, spans, cover)

    # the build-only grey band and the automatic windows
    return [*items, *_backlog_items(backlog), *_automatic_window_items(rows)]


def tape_mix_pairs(rows: pd.DataFrame, redundant: frozenset[str]) -> list[MixPair]:
    """Each second tape symbol overlapping its ticker by more than the margin; a redundant class counts only where the
    ticker trades on cover pages alone (no tape interval of its own)."""
    # the readable tape rows and each (ticker, other symbol) pair, sorted
    symbols = rows[rows["role"].eq("symbol")]
    tape = _tape(symbols)
    pairs = (
        tape.loc[tape["symbol"].ne(tape["canonical_ticker"]), ["canonical_ticker", "symbol"]]
        .drop_duplicates()
        .sort_values(["canonical_ticker", "symbol"])
    )

    # the overlap of each pair per CIK of the other symbol, kept over the margin
    out = []
    for ticker, symbol in zip(pairs["canonical_ticker"].astype(str), pairs["symbol"].astype(str), strict=True):
        other = tape[tape["canonical_ticker"].eq(ticker) & tape["symbol"].eq(symbol)]
        own_tape = tape[tape["canonical_ticker"].eq(ticker) & tape["symbol"].eq(ticker)]
        own_cover = symbols[symbols["canonical_ticker"].eq(ticker) & symbols["symbol"].eq(ticker) & symbols["sources"].eq("dei")]
        by_cik = {
            cik: _subtract(_overlaps(part, own_cover), own_tape) if symbol in redundant else _overlaps(part, own_tape)
            for cik, part in other.groupby("cik", sort=True)
        }
        merged = _merge(span for spans in by_cik.values() for span in spans)
        if _length(merged) > MARGIN:
            out.append(MixPair(ticker, symbol, _mix_why(ticker, symbol, symbol in redundant), _window_text(merged), by_cik))
    return out


def tape_mix_items(pairs: list[MixPair], rows: pd.DataFrame, master: pd.DataFrame | None) -> list[dict[str, object]]:
    """One item per tape-mix pair: an action unless the `universe_cusip` lines already decide it (then information)."""
    # without CUSIP lines every pair is an action item
    if master is None:
        return [_mix_item(pair, None, "") for pair in pairs]

    # per pair: the overlap in the tape era, what the ticker's lines under the symbol leave, and what a CIK window covers
    era, windows, items = master["valid_from"].min(), rows[rows["role"].eq("cik_window")], []
    for pair in pairs:
        clipped = _clip(pair.by_cik, era)
        lines = master[master["canonical_company"].eq(pair.ticker) & master["key"].eq(squash(pair.symbol))]
        uncovered = {cik: _subtract(spans, lines) for cik, spans in clipped.items()}
        rest = _merge(span for spans in uncovered.values() for span in spans)
        own = windows[windows["canonical_ticker"].eq(pair.ticker)]
        windowed = sum(
            (_length(_merge(spans)) - _length(_merge(_subtract(spans, own[own["cik"].eq(cik)]))) for cik, spans in uncovered.items() if spans),
            pd.Timedelta(0),
        )
        in_era = _length(_merge(span for spans in clipped.values() for span in spans))
        left = sorted(cik for cik, spans in uncovered.items() if spans)
        verdict = _master_verdict(era, in_era, _length(rest), windowed, left, _touched_roles(lines, clipped))
        items.append(_mix_item(pair, verdict, _window_text(rest)))
    return items


def symbol_status_items(rows: pd.DataFrame) -> list[dict[str, object]]:
    """One `conflict` and one `noise` line: count, affected tickers, symbols."""
    items: list[dict[str, object]] = []
    symbols = rows[rows["role"].eq("symbol")]
    effects = {
        "conflict": (True, "unresolved on FTD and new short-interest rows: curate the symbol's tenure", _SYMBOLS),
        "noise": (False, "ignored by every resolver; no action unless a listed symbol is among them", ""),
    }
    for status, (action, suggested, config_file) in effects.items():
        part = symbols[symbols["status"].eq(status)]
        if part.empty:
            continue
        pairs = sorted(set(zip(part["canonical_ticker"].astype(str), part["symbol"], strict=True)))
        evidence = f"{len(part)} interval(s) over {part['canonical_ticker'].nunique()} ticker(s): " + ", ".join(f"{s}@{t}" for t, s in pairs)
        tickers, ciks = ",".join(sorted(set(part["canonical_ticker"].astype(str)))), ",".join(sorted(set(part["cik"])))
        items.append(dict(zip(FLAG_COLUMNS, (status, action, tickers, ciks, evidence, suggested, config_file), strict=True)))
    return items


def refused_merge_items(blocked: pd.DataFrame | None) -> list[dict[str, object]]:
    """The build-only kind: one action item per merge refused because it would join two universe tickers."""
    if blocked is None or blocked.empty:
        return []
    suggested = "the refusal is never applied: record the decision (own_entity citing the filing, or a resolution)"
    return [
        dict(
            zip(
                FLAG_COLUMNS,
                ("refused_merge", True, str(a_tickers or b_tickers), f"{a},{b}", _refusal_text(source, a_tickers, b_tickers), suggested, _MANUAL),
                strict=True,
            )
        )
        for a, b, source, a_tickers, b_tickers in zip(
            blocked["cik_a"], blocked["cik_b"], blocked["proposed_by"], blocked["tickers_a"], blocked["tickers_b"], strict=True
        )
    ]


def sorted_flags(items: Iterable[dict[str, object]]) -> pd.DataFrame:
    """The items as a `FLAG_COLUMNS` frame, sorted by kind (`KIND_ORDER`, action kinds first) and ticker."""
    flags = pd.DataFrame(list(items), columns=list(FLAG_COLUMNS))
    order = flags["kind"].map({kind: rank for rank, kind in enumerate(KIND_ORDER)})
    return flags.assign(_order=order).sort_values(["_order", "ticker"], kind="mergesort").drop(columns="_order").reset_index(drop=True)


def identity_flags(
    lineage: pd.DataFrame,
    activity: pd.DataFrame,
    *,
    redundant_symbols: frozenset[str] = frozenset(),
    backlog: pd.DataFrame | None = None,
    master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Every identity item needing a manual decision (`FLAG_COLUMNS`), sorted by kind (action kinds first) and ticker.

    `master`: `universe_cusip` rows (`MASTER_COLUMNS`); without them the tape-mix rule reads the lineage alone.
    """
    rows = flag_rows(lineage)
    mixes = tape_mix_items(tape_mix_pairs(rows, frozenset(redundant_symbols)), rows, master_lines(master))
    return sorted_flags([*cik_items(rows, activity, backlog), *mixes, *symbol_status_items(rows)])


# Resolutions (REQ-031)
def load_resolutions(config_dir: str) -> pd.DataFrame:
    """Every `resolutions` entry of the owning JSONs (`RESOLUTION_COLUMNS`); an entry missing a field or citing no URL or
    accession, or a second entry for one item, is refused."""
    # read each owning JSON once
    entries: list[dict[str, Any]] = []
    for name in RESOLUTION_FILES:
        path = Path(config_dir) / "sec" / name
        blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        entries += [{**entry, "config_file": f"configs/sec/{name}"} for entry in blob.get(RESOLUTIONS_KEY, [])]

    # validate and type them
    return _resolution_frame(entries)


def review_flags(flags: pd.DataFrame, resolutions: pd.DataFrame) -> FlagReview:
    """Each item against the resolutions on `ITEM_KEY`: resolved, an unreviewed action item or information; and the
    resolutions no item matches any more (stale)."""
    # match the items and the resolutions on the verbatim item key
    keys = list(ITEM_KEY)
    items = flags.merge(resolutions[[*keys, "decision"]], on=keys, how="left")
    seen = resolutions.merge(flags[keys].drop_duplicates(), on=keys, how="left", indicator=True)

    # split the items by review
    reviewed, action = items["decision"].notna(), items["action"].astype(bool)
    return FlagReview(
        unreviewed=items[action & ~reviewed].drop(columns="decision").reset_index(drop=True),
        information=items[~action & ~reviewed].drop(columns="decision").reset_index(drop=True),
        resolved=items[reviewed].reset_index(drop=True),
        stale=seen.loc[seen["_merge"].eq("left_only"), list(RESOLUTION_COLUMNS)].reset_index(drop=True),
    )


# Logging
def identity_flag_block(flags: pd.DataFrame) -> str | None:
    """The WARNING block text (action items first, co-registrants left out), or None when there is nothing to show."""
    shown = flags[flags["kind"].ne("co_registrant")].sort_values("action", ascending=False, kind="mergesort")
    if shown.empty:
        return None
    n_action = int(shown["action"].sum())
    lines = [f"{BLOCK_TITLE}: {n_action} action item(s), {len(shown) - n_action} information item(s)"]
    return "\n".join([*lines, *(_item_line(row) for row in shown.to_dict("records"))])


def review_summary(review: FlagReview) -> str:
    """The one INFO line (REQ-032): the unreviewed action items, the information and resolved items by kind, the stale
    resolutions."""
    stale = ", ".join(f"{k} {t} (CIK {c}) in {f}" for k, t, c, f in review.stale[[*ITEM_KEY, "config_file"]].itertuples(index=False))
    return (
        f"identity items: {len(review.unreviewed)} unreviewed action item(s); information {_by_kind(review.information)}; "
        f"resolved {_by_kind(review.resolved)}; {len(review.stale)} stale resolution(s)" + (f": {stale}" if stale else "")
    )


def review_details(review: FlagReview) -> str | None:
    """The DEBUG lines of the information and resolved items (each resolved one with its decision), or None."""
    shown = pd.concat([review.resolved, review.information], ignore_index=True)
    return "\n".join(_item_line(row) for row in shown.to_dict("records")) if not shown.empty else None


# Flat helpers: resolutions and log lines
def _resolution_frame(entries: list[dict[str, Any]]) -> pd.DataFrame:
    """The entries typed (`evidence` joined, `decided_on` parsed); a missing field, no citation or a duplicate item raises."""
    # each entry complete and citing a URL or an accession
    rows = []
    for entry in entries:
        evidence = entry.get("evidence") or []
        cited = [str(text) for text in ([evidence] if isinstance(evidence, str) else evidence)]
        missing = [c for c in ("kind", "ticker", "decision", "decided_on") if not str(entry.get(c) or "").strip()] + (
            [] if "ciks" in entry else ["ciks"]
        )
        if missing or not any(_CITATION.search(text) for text in cited):
            raise ResolutionError(f"{entry.get('config_file')}: resolution {entry} lacks {missing or 'evidence citing a URL or an accession'}")
        typed = {column: str(entry[column]) for column in (*ITEM_KEY, "decision", "config_file")}
        rows.append({**typed, "evidence": "; ".join(cited), "decided_on": pd.Timestamp(str(entry["decided_on"]))})

    # one resolution per item
    frame = pd.DataFrame(rows, columns=list(RESOLUTION_COLUMNS))
    twice = frame[frame.duplicated(list(ITEM_KEY), keep=False)]
    if not twice.empty:
        raise ResolutionError(f"one item reviewed twice: {twice[[*ITEM_KEY, 'config_file']].to_dict('records')}")
    return frame


def _refusal_text(source: str, a_tickers: str, b_tickers: str) -> str:
    """The evidence of one refused merge."""
    return f"{source} would join {a_tickers or 'no universe ticker'} and {b_tickers or 'no universe ticker'} in one company: refused"


def _by_kind(items: pd.DataFrame) -> str:
    """`n (kind n, ...)` with plain integer counts; `0` when empty."""
    counts = ", ".join(f"{kind} {int(n)}" for kind, n in items["kind"].value_counts().sort_index().items())
    return f"{len(items)} ({counts})" if counts else "0"


def _item_line(row: dict[str, Any]) -> str:
    """One item as a log line: level, kind, ticker, CIKs, evidence, the suggested action or the reviewed decision."""
    decision = row.get("decision")
    if isinstance(decision, str):
        return f"  [resolved] {row['kind']} {row['ticker']} (CIK {row['ciks']}): {row['evidence']} | decision: {decision}"
    target = f" -> edit {row['config_file']}" if row["config_file"] else ""
    level = "ACTION" if row["action"] else "info"
    return f"  [{level}] {row['kind']} {row['ticker']} (CIK {row['ciks']}): {row['evidence']} | {row['suggested_action']}{target}"


# Flat helpers: filing spans and tape rows
def _spans(activity: pd.DataFrame) -> SpanMap:
    """`{cik: (first, last or far future, evidence text)}` over every source."""
    out: SpanMap = {}
    for cik, group in activity.groupby("cik", sort=True):
        last = group["last"].fillna(_FAR)
        text = ", ".join(
            f"{'10-K/10-Q cover' if source == 'dei' else 'Form 3/4/5'} {first.date()}..{'open' if pd.isna(end) else end.date()}"
            for source, first, end in zip(group["source"], group["first"], group["last"], strict=True)
        )
        out[str(cik)] = (group["first"].min(), last.max(), text)
    return out


def _tape(symbols: pd.DataFrame) -> pd.DataFrame:
    """Symbol rows the FTD and short-interest tapes read: Form 3/4/5 or manual evidence, neither noise nor conflict."""
    evidenced = pd.Series([bool(set(str(value).split(",")) & set(_TAPE_SOURCES)) for value in symbols["sources"]], index=symbols.index, dtype=bool)
    return symbols[evidenced & ~symbols["status"].isin(("noise", "conflict"))]


# Flat helpers: CIK items
def _extra_ciks(rows: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """`(company, ticker, extra CIK, roster CIK)` for each uncurated extra CIK of a company with a roster-only window."""
    out: list[tuple[str, str, str, str]] = []
    cik_rows = rows[rows["role"].ne("symbol")]
    for entity, group in cik_rows.groupby("company_id", sort=True):
        home_rows = group[group["role"].eq("cik_window") & group["sources"].eq("roster")]
        if home_rows.empty:
            continue
        ticker, home = str(home_rows["canonical_ticker"].iloc[0]), str(home_rows["cik"].iloc[0])
        others = sorted(set(group.loc[group["cik"].ne(home) & ~group["oracle"].isin(("register", "manual")), "cik"]))
        out += [(str(entity), ticker, other, home) for other in others]
    return out


def _takeover(tape: pd.DataFrame, ticker: str, other: str, home: str) -> str | None:
    """The symbol the roster CIK traded under before taking `ticker` over from the concurrent `other` CIK (a merger)."""
    other_rows = tape[tape["cik"].eq(other) & tape["symbol"].eq(ticker)]
    home_rows = tape[tape["cik"].eq(home) & tape["symbol"].eq(ticker)]
    if other_rows.empty or home_rows.empty or other_rows["valid_from"].min() >= home_rows["valid_from"].min():
        return None
    start = home_rows["valid_from"].min()
    renamed = tape[tape["cik"].eq(home) & tape["symbol"].ne(ticker) & tape["valid_to"].notna() & ((tape["valid_to"] - start).abs() <= MARGIN)]
    return None if renamed.empty else f"{renamed['symbol'].iloc[0]} until {renamed['valid_to'].iloc[0].date()}, {ticker} from {start.date()}"


def _pair_item(ticker: str, other: str, home: str, spans: SpanMap, takeover: str | None) -> dict[str, object]:
    """One extra CIK beside a roster-only window: sequential (a missing cutover), a takeover, or a co-registrant."""
    evidence = f"{other}: {spans.get(other, (None, None, 'no filing evidence'))[2]}; {home} (roster): {spans.get(home, (None, None, 'no filing evidence'))[2]}"
    if other not in spans or home not in spans:
        kind, action, suggested, path = "missing_cutover", True, "no filing evidence on one CIK: check for a successor 8-K12B", _REGISTER
    elif min(spans[other][1], spans[home][1]) - max(spans[other][0], spans[home][0]) <= MARGIN:
        kind, action, path = "missing_cutover", True, _REGISTER
        suggested = "the CIKs file one after the other: add a register entry dated from the successor's 8-K12B/8-K12G3"
    elif takeover is not None:
        kind, action, path, evidence = "manual_review", True, _MANUAL, f"{evidence}; {takeover}"
        suggested = (
            f"merger: the roster CIK traded as {takeover}; confirm it is the accounting predecessor "
            f"(a reverse merger needs a manual window giving {other}'s consolidating history to {ticker})"
        )
    else:
        kind, action, suggested, path = "co_registrant", False, "none: event forms on every CIK, 10-K/10-Q on the roster CIK", ""
    return dict(zip(FLAG_COLUMNS, (kind, action, ticker, f"{other},{home}", evidence, suggested, path), strict=True))


def _register_items(rows: pd.DataFrame, spans: SpanMap, cover: SpanMap) -> list[dict[str, object]]:
    """Register seams whose 10-K/10-Q handover (cover-page evidence) sits beyond the margin on the wrong side of the date.

    Form 3/4/5 observations only lag a seam, so they never dispute it; a seam without cover-page evidence on both sides is not judged.
    """
    windows = rows[rows["role"].eq("cik_window") & rows["oracle"].eq("register")].sort_values(["company_id", "valid_from"], kind="mergesort")
    records = windows.to_dict("records")
    seams = [(p, s) for p, s in zip(records, records[1:], strict=False) if p["company_id"] == s["company_id"] and p["valid_to"] == s["valid_from"]]
    items: list[dict[str, object]] = []
    for pred, succ in seams:
        if pred["cik"] not in cover or succ["cik"] not in cover:
            continue
        date, p_last, s_first = succ["valid_from"], cover[pred["cik"]][1], cover[succ["cik"]][0]
        late, early = p_last > date + MARGIN and s_first > p_last, s_first < date - MARGIN and p_last < s_first
        if not (late or early):
            continue
        side = f"the predecessor files alone until {p_last.date()}" if late else f"the successor files alone from {s_first.date()}"
        evidence = f"register seam {date.date()}: {side}; {pred['cik']}: {spans[pred['cik']][2]}; {succ['cik']}: {spans[succ['cik']][2]}"
        values = ("register_date_disputed", True, str(succ["canonical_ticker"]), f"{pred['cik']},{succ['cik']}", evidence)
        items.append(dict(zip(FLAG_COLUMNS, (*values, "check the effective date in the successor's 8-K12B", _REGISTER), strict=True)))
    return items


def _backlog_items(backlog: pd.DataFrame | None) -> list[dict[str, object]]:
    """The build-only kind: grey-band links."""
    if backlog is None or backlog.empty:
        return []
    suggested = "add a manual decision (own_entity or separate) citing the filing"
    return [
        dict(zip(FLAG_COLUMNS, (kind, True, str(ticker or entity), str(cik), str(detail), suggested, _MANUAL), strict=True))
        for kind, ticker, entity, cik, detail in zip(
            backlog["kind"], backlog["canonical_ticker"], backlog["company_id"], backlog["cik"], backlog["detail"], strict=True
        )
        if kind == "grey_band"
    ]


def _automatic_window_items(rows: pd.DataFrame) -> list[dict[str, object]]:
    """One information item per entity whose CIK windows the automatic rule dated, with the chain and its switch evidence."""
    windows = rows[rows["role"].eq("cik_window") & rows["sources"].eq(_AUTOMATIC_SOURCES)]
    items: list[dict[str, object]] = []
    for _, group in windows.sort_values(["company_id", "valid_from"], kind="mergesort").groupby("company_id", sort=True):
        chain = ", ".join(
            f"{cik} from {'open' if start <= _SENTINEL else start.date()}" + ("" if pd.isna(end) else f" to {end.date()}")
            for cik, start, end in zip(group["cik"], group["valid_from"], group["valid_to"], strict=True)
        )
        notes = [str(text).split(_AUTOMATIC_NOTE, 1)[1] for text in group["evidence"].fillna("") if _AUTOMATIC_NOTE in str(text)]
        evidence = f"automatic chain {chain}; " + (notes[0] if notes else "switch evidence not stored")
        suggested = "review: a register entry citing the successor's 8-K12B/8-K12G3 overrides it"
        values = ("automatic_cik_window", False, str(group["canonical_ticker"].iloc[0]), ",".join(group["cik"]), evidence, suggested, _REGISTER)
        items.append(dict(zip(FLAG_COLUMNS, values, strict=True)))
    return items


# Flat helpers: intervals
def _overlaps(left: pd.DataFrame, right: pd.DataFrame) -> list[Span]:
    """Every non-empty intersection of a `left` and a `right` interval (open ends run to the far future)."""
    a = pd.DataFrame({"a_from": left["valid_from"].to_numpy(), "a_to": left["valid_to"].fillna(_FAR).to_numpy()})
    b = pd.DataFrame({"b_from": right["valid_from"].to_numpy(), "b_to": right["valid_to"].fillna(_FAR).to_numpy()})
    pairs = a.merge(b, how="cross")
    starts, ends = pairs[["a_from", "b_from"]].max(axis=1), pairs[["a_to", "b_to"]].min(axis=1)
    return [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in zip(starts, ends, strict=True) if end > start]


def _subtract(spans: list[Span], cut: pd.DataFrame) -> list[Span]:
    """`spans` minus every `cut` interval."""
    for c_from, c_to in zip(cut["valid_from"], cut["valid_to"].fillna(_FAR), strict=True):
        spans = [piece for start, end in spans for piece in ((start, min(end, c_from)), (max(start, c_to), end)) if piece[1] > piece[0]]
    return spans


def _merge(spans: Iterable[Span]) -> list[Span]:
    """`spans` sorted, with overlapping or touching ones joined."""
    merged: list[list[pd.Timestamp]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _length(merged: list[Span]) -> pd.Timedelta:
    """The total length of already merged spans."""
    return sum((end - start for start, end in merged), pd.Timedelta(0))


def _window_text(merged: list[Span]) -> str:
    """Already merged spans as `first..last` text."""
    return ", ".join(f"{start.date()}..{'open' if end >= _FAR else end.date()}" for start, end in merged)


def _clip(by_cik: dict[str, list[Span]], era: pd.Timestamp) -> dict[str, list[Span]]:
    """Each CIK's spans from the tape era on."""
    return {cik: [(max(start, era), end) for start, end in spans if end > max(start, era)] for cik, spans in by_cik.items()}


# Flat helpers: tape-mix items
def _mix_why(ticker: str, symbol: str, redundant: bool) -> str:
    """Why a tape-mix pair is an item."""
    if redundant:
        return f"{symbol} (a redundant class) resolves to {ticker} while {ticker} trades only on cover pages"
    return f"{symbol} resolves to {ticker} while {ticker} trades"


def _touched_roles(lines: pd.DataFrame, clipped: dict[str, list[Span]]) -> list[str]:
    """The sorted roles of the CUSIP lines meeting any clipped overlap."""
    spans = [span for pieces in clipped.values() for span in pieces]
    touched = [
        str(role)
        for role, start, end in zip(lines["lineage_role"], lines["valid_from"], lines["valid_to"].fillna(_FAR), strict=True)
        if any(start < s_end and end > s_start for s_start, s_end in spans)
    ]
    return sorted(set(touched))


def _master_verdict(
    era: pd.Timestamp, in_era: pd.Timedelta, uncovered: pd.Timedelta, windowed: pd.Timedelta, left: list[str], roles: list[str]
) -> tuple[str, str] | None:
    """Why the CUSIP lines leave a pair no decision and the suggested note, else None (an action item).

    In order: the overlap falls before the tape era; the ticker's lines under the symbol cover it; the symbol's CIK holds
    no CIK window over what remains.
    """
    if in_era <= MARGIN:
        return f"before the tape era (universe_cusip starts {era.date()})", "none: no FTD or short-volume line is read before the tape era"
    if uncovered <= MARGIN:
        reason = f"decided by universe_cusip ({', '.join(roles)})"
        return reason, "none: universe_cusip already stamps these lines (edit security_master_manual.json only if a role is wrong)"
    if windowed <= MARGIN:
        reason = f"{', '.join(left)} holds no CIK window at those dates, so no tape row resolves through it"
        return reason, "none: a CIK without a window at those dates does not resolve tape rows"
    return None


def _mix_item(pair: MixPair, verdict: tuple[str, str] | None, uncovered: str) -> dict[str, object]:
    """The item of one tape-mix pair: information when the CUSIP lines decide it, else an action naming what they leave."""
    ciks = ",".join(sorted(pair.by_cik))
    if verdict is not None:
        evidence = f"{pair.why}: {pair.text}; {verdict[0]}"
        return dict(zip(FLAG_COLUMNS, ("manual_review", False, pair.ticker, ciks, evidence, verdict[1], ""), strict=True))
    evidence = f"{pair.why}: {pair.text}; universe_cusip decides per security whether their FTD and short-volume lines are summed"
    evidence += f"; not covered by universe_cusip: {uncovered}" if uncovered else ""
    suggested = f"check the universe_cusip role of {pair.symbol}'s lines (security_master_manual.json) or curate {pair.symbol} (symbol tenure)"
    return dict(zip(FLAG_COLUMNS, ("manual_review", True, pair.ticker, ciks, evidence, suggested, _SYMBOLS), strict=True))
