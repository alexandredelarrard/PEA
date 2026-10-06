"""Identity items needing a manual decision, classified once for the lineage build and the identity validator.

Pure: `entity_lineage` rows, each CIK's filing activity (`cik_activity`) and, from the build only, its
grey-band and rekey backlog. `log_identity_flags` writes one WARNING block (action items first) and one
INFO line for co-registrants.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from src.utils.string import pad_cik_series

#: The seam margin: filings this close to a register date are observation lag, not a dispute.
MARGIN = pd.Timedelta(days=31)
BLOCK_TITLE = "IDENTITY ITEMS NEEDING A MANUAL DECISION"
FLAG_COLUMNS = ("kind", "action", "ticker", "ciks", "evidence", "suggested_action", "config_file")
KIND_ORDER = (
    "missing_cutover",
    "register_date_disputed",
    "incorrect_cik_window",
    "missing_sec_filing",
    "grey_band",
    "rekey",
    "manual_review",
    "conflict",
    "vendor_coverage_gap",
    "vendor_series_other_company",
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
#: The stored open start of a window (`entity_lineage.SENTINEL_START`).
_SENTINEL = pd.Timestamp("1900-01-01")


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


def _spans(activity: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp, str]]:
    """`{cik: (first, last or far future, evidence text)}` over every source."""
    out: dict[str, tuple[pd.Timestamp, pd.Timestamp, str]] = {}
    for cik, group in activity.groupby("cik", sort=True):
        last = group["last"].fillna(_FAR)
        text = ", ".join(
            f"{'10-K/10-Q cover' if source == 'dei' else 'Form 3/4/5'} {first.date()}..{'open' if pd.isna(end) else end.date()}"
            for source, first, end in zip(group["source"], group["first"], group["last"], strict=True)
        )
        out[str(cik)] = (group["first"].min(), last.max(), text)
    return out


def _item(kind: str, action: bool, ticker: str, ciks: Iterable[str], evidence: str, suggested: str, config_file: str) -> dict[str, object]:
    return {
        "kind": kind,
        "action": action,
        "ticker": ticker,
        "ciks": ",".join(ciks),
        "evidence": evidence,
        "suggested_action": suggested,
        "config_file": config_file,
    }


_LINEAGE_COLUMNS = ("entity_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources", "oracle", "evidence")
#: `sources` of a `cik_window` row dated by the automatic rule (both filed evidence sources agreeing).
_AUTOMATIC_SOURCES = "dei,form345"
_AUTOMATIC_NOTE = "automatic window: "


def _prepare(lineage: pd.DataFrame) -> pd.DataFrame:
    rows = lineage.reindex(columns=list(_LINEAGE_COLUMNS))
    rows["cik"] = pad_cik_series(rows["cik"])
    rows["symbol"] = rows["symbol"].fillna("").astype(str)
    rows["sources"] = rows["sources"].fillna("").astype(str)
    rows["valid_from"] = pd.to_datetime(rows["valid_from"])
    rows["valid_to"] = pd.to_datetime(rows["valid_to"])
    return rows[rows["canonical_ticker"].notna()]


def _tape(symbols: pd.DataFrame) -> pd.DataFrame:
    """Symbol rows the FTD and short-interest tapes read: Form 3/4/5 or manual evidence, neither noise nor conflict."""
    evidenced = pd.Series([bool(set(str(value).split(",")) & set(_TAPE_SOURCES)) for value in symbols["sources"]], index=symbols.index, dtype=bool)
    return symbols[evidenced & ~symbols["status"].isin(("noise", "conflict"))]


def _takeover(symbols: pd.DataFrame, ticker: str, other: str, home: str) -> str | None:
    """The symbol the roster CIK traded under before taking `ticker` over from the concurrent `other` CIK (a merger)."""
    tape = _tape(symbols)
    other_rows = tape[tape["cik"].eq(other) & tape["symbol"].eq(ticker)]
    home_rows = tape[tape["cik"].eq(home) & tape["symbol"].eq(ticker)]
    if other_rows.empty or home_rows.empty or other_rows["valid_from"].min() >= home_rows["valid_from"].min():
        return None
    start = home_rows["valid_from"].min()
    renamed = tape[tape["cik"].eq(home) & tape["symbol"].ne(ticker) & tape["valid_to"].notna() & ((tape["valid_to"] - start).abs() <= MARGIN)]
    return None if renamed.empty else f"{renamed['symbol'].iloc[0]} until {renamed['valid_to'].iloc[0].date()}, {ticker} from {start.date()}"


def _multi_cik_items(rows: pd.DataFrame, spans: dict[str, tuple[pd.Timestamp, pd.Timestamp, str]]) -> list[dict[str, object]]:
    """Each uncurated extra CIK beside a roster-only window: sequential, concurrent, or a takeover."""
    items: list[dict[str, object]] = []
    cik_rows = rows[rows["role"].ne("symbol")]
    for entity, group in cik_rows.groupby("entity_id", sort=True):
        home_rows = group[group["role"].eq("cik_window") & group["sources"].eq("roster")]
        if home_rows.empty:
            continue
        ticker, home = str(home_rows["canonical_ticker"].iloc[0]), str(home_rows["cik"].iloc[0])
        symbols = rows[rows["entity_id"].eq(entity) & rows["role"].eq("symbol")]
        for other in sorted(set(group.loc[group["cik"].ne(home) & ~group["oracle"].isin(("register", "manual")), "cik"])):
            items.append(_pair_item(ticker, other, home, spans, symbols))
    return items


def _pair_item(
    ticker: str, other: str, home: str, spans: dict[str, tuple[pd.Timestamp, pd.Timestamp, str]], symbols: pd.DataFrame
) -> dict[str, object]:
    evidence = f"{other}: {spans.get(other, (None, None, 'no filing evidence'))[2]}; {home} (roster): {spans.get(home, (None, None, 'no filing evidence'))[2]}"
    if other not in spans or home not in spans:
        return _item(
            "missing_cutover", True, ticker, (other, home), evidence, "no filing evidence on one CIK: check for a successor 8-K12B", _REGISTER
        )
    overlap = min(spans[other][1], spans[home][1]) - max(spans[other][0], spans[home][0])
    if overlap <= MARGIN:
        suggested = "the CIKs file one after the other: add a register entry dated from the successor's 8-K12B/8-K12G3"
        return _item("missing_cutover", True, ticker, (other, home), evidence, suggested, _REGISTER)
    takeover = _takeover(symbols, ticker, other, home)
    if takeover is not None:
        suggested = (
            f"merger: the roster CIK traded as {takeover}; confirm it is the accounting predecessor "
            f"(a reverse merger needs a manual window giving {other}'s consolidating history to {ticker})"
        )
        return _item("manual_review", True, ticker, (other, home), f"{evidence}; {takeover}", suggested, _MANUAL)
    return _item("co_registrant", False, ticker, (other, home), evidence, "none: event forms on every CIK, 10-K/10-Q on the roster CIK", "")


def _register_items(
    rows: pd.DataFrame, spans: dict[str, tuple[pd.Timestamp, pd.Timestamp, str]], cover: dict[str, tuple[pd.Timestamp, pd.Timestamp, str]]
) -> list[dict[str, object]]:
    """Register seams whose 10-K/10-Q handover (cover-page evidence) sits beyond the margin on the wrong side of the date.

    Form 3/4/5 observations only lag a seam, so they never dispute it; a seam without cover-page evidence on both sides is not judged.
    """
    windows = rows[rows["role"].eq("cik_window") & rows["oracle"].eq("register")].sort_values(["entity_id", "valid_from"], kind="mergesort")
    items: list[dict[str, object]] = []
    for _, group in windows.groupby("entity_id", sort=True):
        records = group.to_dict("records")
        for pred, succ in zip(records, records[1:], strict=False):
            if pred["valid_to"] != succ["valid_from"] or pred["cik"] not in cover or succ["cik"] not in cover:
                continue
            date = succ["valid_from"]
            p_last, s_first = cover[pred["cik"]][1], cover[succ["cik"]][0]
            p_text, s_text = spans[pred["cik"]][2], spans[succ["cik"]][2]
            late = p_last > date + MARGIN and s_first > p_last
            early = s_first < date - MARGIN and p_last < s_first
            if late or early:
                side = f"the predecessor files alone until {p_last.date()}" if late else f"the successor files alone from {s_first.date()}"
                evidence = f"register seam {date.date()}: {side}; {pred['cik']}: {p_text}; {succ['cik']}: {s_text}"
                items.append(
                    _item(
                        "register_date_disputed",
                        True,
                        str(succ["canonical_ticker"]),
                        (pred["cik"], succ["cik"]),
                        evidence,
                        "check the effective date in the successor's 8-K12B",
                        _REGISTER,
                    )
                )
    return items


def _automatic_window_items(rows: pd.DataFrame) -> list[dict[str, object]]:
    """One information item per entity whose CIK windows the automatic rule dated, with the chain and its switch evidence."""
    windows = rows[rows["role"].eq("cik_window") & rows["sources"].eq(_AUTOMATIC_SOURCES)]
    items: list[dict[str, object]] = []
    for _, group in windows.sort_values(["entity_id", "valid_from"], kind="mergesort").groupby("entity_id", sort=True):
        chain = ", ".join(
            f"{cik} from {'open' if start <= _SENTINEL else start.date()}" + ("" if pd.isna(end) else f" to {end.date()}")
            for cik, start, end in zip(group["cik"], group["valid_from"], group["valid_to"], strict=True)
        )
        notes = [str(text).split(_AUTOMATIC_NOTE, 1)[1] for text in group["evidence"].fillna("") if _AUTOMATIC_NOTE in str(text)]
        items.append(
            _item(
                "automatic_cik_window",
                False,
                str(group["canonical_ticker"].iloc[0]),
                list(group["cik"]),
                f"automatic chain {chain}; " + (notes[0] if notes else "switch evidence not stored"),
                "review: a register entry citing the successor's 8-K12B/8-K12G3 overrides it",
                _REGISTER,
            )
        )
    return items


def _overlaps(left: pd.DataFrame, right: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Every non-empty intersection of a `left` and a `right` interval (open ends run to the far future)."""
    spans = []
    for a_from, a_to in zip(left["valid_from"], left["valid_to"].fillna(_FAR), strict=True):
        for b_from, b_to in zip(right["valid_from"], right["valid_to"].fillna(_FAR), strict=True):
            start, end = max(a_from, b_from), min(a_to, b_to)
            if end > start:
                spans.append((start, end))
    return spans


def _subtract(spans: list[tuple[pd.Timestamp, pd.Timestamp]], cut: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """`spans` minus every `cut` interval."""
    for c_from, c_to in zip(cut["valid_from"], cut["valid_to"].fillna(_FAR), strict=True):
        spans = [piece for start, end in spans for piece in ((start, min(end, c_from)), (max(start, c_to), end)) if piece[1] > piece[0]]
    return spans


def _window_text(spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    """The merged spans as `first..last` text."""
    merged: list[list[pd.Timestamp]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return ", ".join(f"{start.date()}..{'open' if end >= _FAR else end.date()}" for start, end in merged)


def _tape_mix_items(rows: pd.DataFrame, redundant: frozenset[str]) -> list[dict[str, object]]:
    """A second tape symbol dated to a ticker while the ticker itself trades: two securities under one company on the tapes.

    `security_master` decides per security (CUSIP, class) whether the second line is summed into the ticker (canonical
    or secondary class) or kept apart. A redundant class counts only where the ticker trades on cover pages alone (no
    tape interval of its own); overlaps of a margin or less (single typed filings) are left out.
    """
    items: list[dict[str, object]] = []
    symbols = rows[rows["role"].eq("symbol")]
    for ticker, group in symbols.groupby("canonical_ticker", sort=True):
        tape = _tape(group)
        own_tape = tape[tape["symbol"].eq(ticker)]
        own_cover = group[group["symbol"].eq(ticker) & group["sources"].eq("dei")]
        for symbol, other in tape[tape["symbol"].ne(ticker)].groupby("symbol", sort=True):
            if symbol in redundant:
                spans = _subtract(_overlaps(other, own_cover), own_tape)
                why = f"{symbol} (a redundant class) resolves to {ticker} while {ticker} trades only on cover pages"
            else:
                spans = _overlaps(other, own_tape)
                why = f"{symbol} resolves to {ticker} while {ticker} trades"
            if sum(((end - start) for start, end in spans), pd.Timedelta(0)) > MARGIN:
                items.append(
                    _item(
                        "manual_review",
                        True,
                        str(ticker),
                        sorted(set(other["cik"])),
                        f"{why}: {_window_text(spans)}; security_master decides per security whether their FTD and short-volume lines are summed",
                        f"check the security_master role of {symbol}'s lines (security_master_manual.json) or curate {symbol} (symbol tenure)",
                        _SYMBOLS,
                    )
                )
    return items


def _symbol_status_items(rows: pd.DataFrame) -> list[dict[str, object]]:
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
        items.append(
            _item(
                status,
                action,
                ",".join(sorted(set(part["canonical_ticker"].astype(str)))),
                sorted(set(part["cik"])),
                evidence,
                suggested,
                config_file,
            )
        )
    return items


def _backlog_items(backlog: pd.DataFrame | None) -> list[dict[str, object]]:
    """The build-only kinds: grey-band links and excluded older-CIK rekeys."""
    if backlog is None or backlog.empty:
        return []
    actions = {
        "grey_band": ("add a manual decision (own_entity or separate) citing the filing", _MANUAL),
        "rekey": ("approve with identity-tables --approve-rekey OLD:NEW, or leave excluded", ""),
    }
    return [
        _item(kind, True, str(ticker or entity), (str(cik),), str(detail), *actions[kind])
        for kind, ticker, entity, cik, detail in zip(
            backlog["kind"], backlog["canonical_ticker"], backlog["entity_id"], backlog["cik"], backlog["detail"], strict=True
        )
        if kind in actions
    ]


def identity_flags(
    lineage: pd.DataFrame, activity: pd.DataFrame, *, redundant_symbols: frozenset[str] = frozenset(), backlog: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Every identity item needing a manual decision (`FLAG_COLUMNS`), sorted by kind (action kinds first) and ticker."""
    rows = _prepare(lineage)
    spans = _spans(activity)
    items = [
        *_multi_cik_items(rows, spans),
        *_register_items(rows, spans, _spans(activity[activity["source"].eq("dei")])),
        *_backlog_items(backlog),
        *_automatic_window_items(rows),
        *_tape_mix_items(rows, frozenset(redundant_symbols)),
        *_symbol_status_items(rows),
    ]
    flags = pd.DataFrame(items, columns=list(FLAG_COLUMNS))
    order = flags["kind"].map({kind: rank for rank, kind in enumerate(KIND_ORDER)})
    return flags.assign(_order=order).sort_values(["_order", "ticker"], kind="mergesort").drop(columns="_order").reset_index(drop=True)


def identity_flag_block(flags: pd.DataFrame) -> str | None:
    """The WARNING block text (action items first, co-registrants left out), or None when there is nothing to show."""
    shown = flags[flags["kind"].ne("co_registrant")].sort_values("action", ascending=False, kind="mergesort")
    if shown.empty:
        return None
    n_action = int(shown["action"].sum())
    lines = [f"{BLOCK_TITLE}: {n_action} action item(s), {len(shown) - n_action} information item(s)"]
    for row in shown.itertuples(index=False):
        target = f" -> edit {row.config_file}" if row.config_file else ""
        lines.append(
            f"  [{'ACTION' if row.action else 'info'}] {row.kind} {row.ticker} (CIK {row.ciks}): {row.evidence} | {row.suggested_action}{target}"
        )
    return "\n".join(lines)


def log_identity_flags(log: logging.Logger, flags: pd.DataFrame) -> None:
    """One WARNING block when anything needs a decision; co-registrants once at INFO."""
    block = identity_flag_block(flags)
    if block is not None:
        log.warning(block)
    co = flags[flags["kind"].eq("co_registrant")]
    if not co.empty:
        log.info(
            "identity: %d co-registrant pair(s), default scope kept (no action): %s",
            len(co),
            ", ".join(f"{t} ({c})" for t, c in zip(co["ticker"], co["ciks"], strict=True)),
        )
