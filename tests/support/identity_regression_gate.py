"""Explained-difference regression gate for the identity cutover (decision P22).

`snapshot <dir>` dumps the before state, read-only and projected, as parquet files. The FTD and FINRA lines are stored
as filed and mapped by the cube, so no market tape is diffed; the shipped `market_tape` hypotheses are reported retired.
`diff <dir>` compares it with the current store, writes one explained-difference CSV per surface plus `gate_hypotheses.csv` and
`gate_summary.txt`, and exits 1 when a row has no reason or a hypothesis of
`configs/sec/expected_lineage_changes.json` does not hold, 2 when the run itself fails.

  python tests/support/identity_regression_gate.py snapshot <dir> [-c ./configs] [-t TICKER ...]
  python tests/support/identity_regression_gate.py diff <dir> [-o <out dir>] [-c ./configs] [-t TICKER ...] [--skip-hypothesis ID ...] [--store-url URL]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.constants.constants import CANONICAL_ROLES, SECONDARY_CLASS  # noqa: E402
from src.data_extract.utils.common.identity import FilingScope, Identity, UnknownUniverseTickerError, load_identity  # noqa: E402
from src.data_extract.utils.common.identity_contract import entity_by_cik_map, entity_or_singleton  # noqa: E402
from src.data_extract.utils.common.registrant import FORM_POLICY, Combine  # noqa: E402
from src.data_extract.utils.fundamentals.build_history import keep_window_owner_filings  # noqa: E402
from src.data_store.schema import Tables  # noqa: E402
from src.utils.filer_tables import PURGE_TABLES, FilerTable  # noqa: E402
from src.utils.predecessor_series import load_predecessor_series  # noqa: E402
from src.utils.string import normalise_ticker, pad_cik, pad_cik_series, yahoo_symbol  # noqa: E402

log = logging.getLogger("identity_regression_gate")

FILING_COLUMNS = [
    "canonical_company",
    "cik",
    "accession",
    "form",
    "accepted_at",
    "period_end",
    "old_decision",
    "new_decision",
    "reason",
    "evidence_accession",
    "table",
    "evidence",
]
INSIDER_COLUMNS = [
    "accession_number",
    "security_type",
    "row_sequence",
    "ticker",
    "issuer_cik",
    "filing_date",
    "old_role",
    "new_role",
    "reason",
    "evidence",
]
MERGED_COLUMNS = ["ticker", "as_of", "fiscal_end", "change", "columns", "reason", "evidence"]
PRICES_COLUMNS = ["ticker", "date", "change", "columns", "reason", "evidence"]
HYPOTHESIS_COLUMNS = ["id", "kind", "table", "tickers", "change", "reason", "expected", "observed", "status"]
OUTPUTS = {
    "filing": "gate_filing_lineage_diff.csv",
    "insider": "gate_insider_diff.csv",
    "merged": "gate_fundamentals_diff.csv",
    "prices": "gate_prices_diff.csv",
}
HISTORY_TABLE = Tables.sec_fundamentals_history.name
HYPOTHESES_FILE = Path("sec") / "expected_lineage_changes.json"
_TABLE_POLICY = {
    Tables.sec_8k.name: Combine.SPLIT,
    Tables.sec_8k_votes.name: Combine.SPLIT,
    Tables.sec_13d.name: Combine.SPLIT,
    Tables.sec_13d_transactions.name: Combine.SPLIT,
    Tables.sec_13g.name: Combine.SPLIT,
    Tables.insider_transactions.name: Combine.UNION,
}
#: Event-filing tables (8-K, 13D/13G by subject company) whose rows belong to a ticker only inside a window of their CIK.
_DATED_TABLES = frozenset(spec.table.name for spec in PURGE_TABLES if spec.dated)
_INSIDER_KEY = ["accession_number", "security_type", "row_sequence"]
_CHUNK = 200_000
#: Every reason a diff row can carry, by surface; a row with none is unexplained.
REASONS = frozenset(
    {
        # filing lineage
        "foreign_filer_purge",
        "acquired_constituent_purge",
        "outside_cik_window",
        "register_window",
        "seam_margin",
        "same_period_rule",
        "symbol_normalisation",
        "new_filing",
        "relisted_own_filing",
        # insider, merged fundamentals, prices
        "acquired_constituent",
        "co_registrant_purge",
        "predecessor_vendor_series",
        "new_period",
        "secondary_class_added",
        "new_data",
        # an insider row following its issuer CIK into the entity that now holds it
        "entity_moved",
    }
)
#: Each hypothesis kind and the gate section whose rows it explains.
HYPOTHESIS_KINDS = {"filing_lineage": "filing", "insider": "insider"}
#: Kinds whose section is gone (the stamped FTD / FINRA tape, P4/P5): their shipped hypotheses are reported, never checked.
RETIRED_KINDS = frozenset({"market_tape"})


def _text(values: pd.Series) -> pd.Series:
    """Strings with nulls as ''."""
    return values.astype(object).where(values.notna(), "").astype(str)


# --------------------------------------------------------------------------- filing lineage


def _consolidating(table: str, forms: pd.Series) -> pd.Series:
    """Per row: whether its form (else its table) combines across a registrant boundary by window (SPLIT)."""
    default = _TABLE_POLICY.get(table, Combine.SPLIT) is Combine.SPLIT
    return forms.map(lambda form: FORM_POLICY[str(form)] is Combine.SPLIT if form in FORM_POLICY else default).astype(bool)


def filing_reason(
    change: str,
    cik: str,
    filed: pd.Timestamp | None,
    consolidating: bool,
    scope: FilingScope | None,
    old_ciks: frozenset[str],
    frontier: pd.Timestamp | None,
    dated: bool = False,
) -> str:
    """The rule explaining one added or removed accession, '' when none does.

    `dated`: an event-filing table, whose removed rows from an event-only CIK or outside the CIK's widened window are named
    as such (P35)."""
    if scope is None:
        return ""
    windows = [w for w in scope.windows if w.cik == cik]
    known = filed is not None and not pd.isna(filed)
    day = cast(pd.Timestamp, filed)
    owned = known and any(w.owns(day) for w in windows)
    admitted = known and any(w.admits(day) for w in windows)
    if change == "removed":
        if cik not in scope.event_ciks:
            return "foreign_filer_purge"
        if dated and not windows:
            return "acquired_constituent_purge"
        if consolidating and known and not admitted:
            return "outside_cik_window" if dated else "register_window"
        return ""
    if cik not in scope.event_ciks:
        return ""
    if consolidating and admitted and not owned:
        return "seam_margin"
    if consolidating and known and not admitted:
        return ""
    if cik not in old_ciks:
        return "register_window"
    if frontier is not None and known and day > frontier:
        return "new_filing"
    return "relisted_own_filing"


def filing_diff(
    table: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    scopes: Mapping[str, FilingScope],
    old_ciks: Mapping[str, frozenset[str]],
    co_registrants: Collection[str] = (),
) -> pd.DataFrame:
    """One row per accession added to or removed from a ticker in a filer-CIK table; an insider or event-filing accession
    of a co-registrant CIK leaves by its own purge."""
    key = ["ticker", "accession"]
    merged = before.merge(after, on=key, how="outer", suffixes=("_b", "_a"), indicator=True)
    merged = merged[merged["_merge"].ne("both")]
    if merged.empty:
        return pd.DataFrame(columns=FILING_COLUMNS)
    removed = merged["_merge"].eq("left_only")
    pick = lambda col: merged[f"{col}_b"].where(removed, merged[f"{col}_a"])  # noqa: E731
    out = pd.DataFrame(
        {
            "table": table,
            "canonical_company": merged["ticker"].astype(str),
            "cik": pad_cik_series(pick("cik")),
            "accession": merged["accession"].astype(str),
            "form": _text(pick("form")),
            "filed": pd.to_datetime(pick("filed"), errors="coerce"),
            "period_end": pd.to_datetime(pick("period_end"), errors="coerce"),
            "change": np.where(removed, "removed", "added"),
        }
    )
    frontier = pd.to_datetime(before["filed"], errors="coerce").max() if not before.empty else None
    out["consolidating"] = _consolidating(table, out["form"])
    # the same accession leaving one spelling of a ticker for its normalised spelling
    moved = out.groupby("accession")["canonical_company"].transform(lambda s: s.map(yahoo_symbol).nunique() == 1 and s.nunique() > 1)
    dated = table in _DATED_TABLES
    co = {pad_cik(c) for c in co_registrants} if dated or table == Tables.insider_transactions.name else set()
    reasons = []
    for row, is_moved in zip(out.itertuples(index=False), moved, strict=True):
        if is_moved:
            reasons.append("symbol_normalisation")
            continue
        if row.change == "removed" and row.cik in co:
            reasons.append("co_registrant_purge")
            continue
        scope = scopes.get(normalise_ticker(row.canonical_company))
        reasons.append(
            filing_reason(
                cast(str, row.change),
                cast(str, row.cik),
                cast(Any, row.filed),
                bool(row.consolidating),
                scope,
                old_ciks.get(normalise_ticker(row.canonical_company), frozenset()),
                frontier,
                dated,
            )
        )
    out["reason"] = reasons
    out["old_decision"] = np.where(out["change"].eq("removed"), "stored", "absent")
    out["new_decision"] = np.where(out["change"].eq("removed"), "absent", "stored")
    out["evidence_accession"] = ""
    out["evidence"] = [_window_text(scopes.get(normalise_ticker(t)), c) for t, c in zip(out["canonical_company"], out["cik"], strict=True)]
    return _filing_frame(out)


def _window_text(scope: FilingScope | None, cik: str) -> str:
    if scope is None:
        return "ticker outside the universe"
    windows = [w for w in scope.windows if w.cik == cik]
    if not windows:
        return f"cik {cik} event-only" if cik in scope.event_ciks else f"cik {cik} not in the entity"
    return "; ".join(
        f"window {w.valid_from.date() if w.valid_from is not None else '-'}..{w.valid_to.date() if w.valid_to is not None else '-'}" for w in windows
    )


def _filing_frame(out: pd.DataFrame) -> pd.DataFrame:
    out = out.assign(
        accepted_at=out["filed"].dt.strftime("%Y-%m-%d").fillna(""), period_end=pd.to_datetime(out["period_end"]).dt.strftime("%Y-%m-%d").fillna("")
    )
    return out[FILING_COLUMNS].reset_index(drop=True)


def history_diff(before: pd.DataFrame, after: pd.DataFrame, scopes: Mapping[str, FilingScope], facts_reasons: pd.DataFrame) -> pd.DataFrame:
    """The filings feeding `sec_fundamentals_history`: before = every stored periodic accession, after = those the
    seam rule keeps (`keep_window_owner_filings`, applied as the history build applies it)."""
    periodic = lambda f: f.loc[f["form"].map(lambda form: FORM_POLICY.get(form) is Combine.SPLIT and not str(form).startswith("DEF")).astype(bool)]  # noqa: E731
    before, after = periodic(before), periodic(after)
    kept_frames = []
    for ticker, facts in after.groupby("ticker", sort=True):
        scope = scopes.get(normalise_ticker(str(ticker)))
        frame = facts.rename(columns={"accession": "accession_number", "filed": "filing_date", "period_end": "period_of_report"})
        ciks = pad_cik_series(frame["cik"])
        if scope is not None and ciks[ciks.ne("")].nunique() > 1:  # a null CIK is no filer, as in the history build
            frame = keep_window_owner_filings(frame, scope.windows)
        kept_frames.append(frame.rename(columns={"accession_number": "accession", "filing_date": "filed", "period_of_report": "period_end"}))
    kept = pd.concat(kept_frames, ignore_index=True) if kept_frames else after.iloc[0:0]
    merged = before.merge(kept, on=["ticker", "accession"], how="outer", suffixes=("_b", "_a"), indicator=True)
    merged = merged[merged["_merge"].ne("both")]
    if merged.empty:
        return pd.DataFrame(columns=FILING_COLUMNS)
    removed = merged["_merge"].eq("left_only")
    pick = lambda col: merged[f"{col}_b"].where(removed, merged[f"{col}_a"])  # noqa: E731
    out = pd.DataFrame(
        {
            "table": HISTORY_TABLE,
            "canonical_company": merged["ticker"].astype(str),
            "cik": pad_cik_series(pick("cik")),
            "accession": merged["accession"].astype(str),
            "form": _text(pick("form")),
            "filed": pd.to_datetime(pick("filed"), errors="coerce"),
            "period_end": pd.to_datetime(pick("period_end"), errors="coerce"),
            "change": np.where(removed, "removed", "added"),
        }
    )
    stored_after = set(zip(after["ticker"], after["accession"], strict=True))
    by_key = {
        (t, a): (r, e)
        for t, a, r, e in zip(
            facts_reasons["canonical_company"], facts_reasons["accession"], facts_reasons["reason"], facts_reasons["evidence"], strict=True
        )
    }
    reasons, evidence, owners, decisions = [], [], [], []
    for row in out.itertuples(index=False):
        if row.change == "added" or (row.canonical_company, row.accession) not in stored_after:
            reason, text = by_key.get((row.canonical_company, row.accession), ("", "no facts-table row"))
            reasons.append(reason)
            evidence.append(text)
            owners.append("")
            decisions.append("kept" if row.change == "added" else "absent")
            continue
        scope = scopes.get(normalise_ticker(row.canonical_company))
        windows = [w for w in scope.windows if w.cik == row.cik] if scope is not None else []
        filed = cast(pd.Timestamp, row.filed)
        admitted = not pd.isna(filed) and any(w.admits(filed) for w in windows)
        decisions.append("set_aside")
        if not admitted:
            reasons.append("register_window")
            evidence.append(_window_text(scope, cast(str, row.cik)))
            owners.append("")
            continue
        holder = next(
            (w.cik for w in (scope.windows if scope else ()) if not pd.isna(row.period_end) and w.owns(cast(pd.Timestamp, row.period_end))), None
        )
        owner_rows = kept[
            (kept["ticker"] == row.canonical_company)
            & (pad_cik_series(kept["cik"]) == holder)
            & (pd.to_datetime(kept["period_end"]) == row.period_end)
        ]
        reasons.append("same_period_rule" if holder is not None and not owner_rows.empty else "")
        evidence.append(f"period owned by cik {holder}")
        owners.append(str(owner_rows["accession"].iloc[0]) if not owner_rows.empty else "")
    out["reason"] = reasons
    out["evidence"] = evidence
    out["evidence_accession"] = owners
    out["old_decision"] = np.where(out["change"].eq("removed"), "kept", "absent")
    out["new_decision"] = decisions
    return _filing_frame(out)


# --------------------------------------------------------------------------- insider, merged fundamentals, prices


def _insider_reason(
    state: str,
    cik: str,
    ticker: str,
    role: str,
    filed: Any,
    ticker_after: str,
    co_registrants: Collection[str],
    own_ciks: Mapping[str, frozenset[str]],
    frontier: Any,
) -> str:
    """The rule explaining one changed insider row, '' when none does."""
    if state == "left_only":
        if cik in co_registrants:
            return "co_registrant_purge"
        return "foreign_filer_purge" if cik not in own_ciks.get(normalise_ticker(ticker), frozenset()) else ""
    if state == "right_only":
        return "new_filing" if frontier is not None and not pd.isna(filed) and filed > frontier else ""
    if ticker != ticker_after and yahoo_symbol(ticker) == yahoo_symbol(ticker_after):
        return "symbol_normalisation"
    if ticker != ticker_after and cik in own_ciks.get(normalise_ticker(ticker_after), ()) and cik not in own_ciks.get(normalise_ticker(ticker), ()):
        return "entity_moved"
    return "acquired_constituent" if role == "acquired_constituent" else ""


def insider_diff(before: pd.DataFrame, after: pd.DataFrame, own_ciks: Mapping[str, frozenset[str]], co_registrants: Collection[str]) -> pd.DataFrame:
    """Rows that leave canonical insider history: a non-canonical role, a purge, or a ticker move."""
    merged = before.merge(after, on=_INSIDER_KEY, how="outer", suffixes=("_b", "_a"), indicator=True)
    frontier = pd.to_datetime(before["filing_date"], errors="coerce").max() if not before.empty else None
    role = _text(merged["lineage_role"]) if "lineage_role" in merged.columns else pd.Series("", index=merged.index)
    gone, new = merged["_merge"].eq("left_only"), merged["_merge"].eq("right_only")
    moved = merged["_merge"].eq("both") & merged["ticker_b"].ne(merged["ticker_a"])
    demoted = merged["_merge"].eq("both") & ~role.isin(CANONICAL_ROLES)
    rows = merged[gone | new | moved | demoted].copy()
    if rows.empty:
        return pd.DataFrame(columns=INSIDER_COLUMNS)
    cik = pad_cik_series(rows["issuer_cik_b"].where(rows["issuer_cik_b"].notna(), rows["issuer_cik_a"]))
    ticker = _text(rows["ticker_b"].where(rows["ticker_b"].notna(), rows["ticker_a"]))
    filed = pd.to_datetime(rows["filing_date_b"].where(rows["filing_date_b"].notna(), rows["filing_date_a"]), errors="coerce")
    new_role = _text(rows["lineage_role"]) if "lineage_role" in rows.columns else pd.Series("", index=rows.index)
    co = {pad_cik(c) for c in co_registrants}
    reasons = [
        _insider_reason(state, c, t, r, f, ta, co, own_ciks, frontier)
        for state, c, t, r, f, ta in zip(rows["_merge"], cik, ticker, new_role, filed, _text(rows["ticker_a"]), strict=True)
    ]
    out = pd.DataFrame(
        {
            "accession_number": rows["accession_number"].astype(str),
            "security_type": _text(rows["security_type"]),
            "row_sequence": rows["row_sequence"],
            "ticker": ticker,
            "issuer_cik": cik,
            "filing_date": filed.dt.strftime("%Y-%m-%d").fillna(""),
            "old_role": np.where(rows["_merge"].eq("right_only"), "absent", "counted"),
            "new_role": np.where(rows["_merge"].eq("left_only"), "absent", new_role),
            "reason": reasons,
            "evidence": _text(rows["economic_date"].astype("string")) if "economic_date" in rows.columns else "",
        }
    )
    return out[INSIDER_COLUMNS].reset_index(drop=True)


@dataclass(frozen=True)
class Window:
    """A predecessor window whose owner series is stored: rows of `ticker` with `fiscal_end` in `[start, end)`."""

    ticker: str
    vendor_ticker: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None

    def holds(self, day: pd.Timestamp) -> bool:
        return not pd.isna(day) and (self.start is None or self.start <= day) and (self.end is None or day < self.end)

    def trails(self, day: pd.Timestamp) -> bool:
        """Within a year after the window: trailing-four-quarter figures still sum replaced quarters."""
        return not pd.isna(day) and self.end is not None and self.end <= day < self.end + pd.DateOffset(years=1)


def _same_cells(a: pd.DataFrame, b: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per column: True where the before and after cells agree (NaN = NaN, numbers to 1e-9 relative)."""
    same = {}
    for column in columns:
        x, y = a[column], b[column]
        if pd.api.types.is_datetime64_any_dtype(x) or pd.api.types.is_datetime64_any_dtype(y):
            x, y = pd.to_datetime(x, errors="coerce").astype("string"), pd.to_datetime(y, errors="coerce").astype("string")
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = xn.notna() | yn.notna()
        close = np.isclose(xn.to_numpy(dtype="float64"), yn.to_numpy(dtype="float64"), rtol=1e-9, atol=0.0)
        text_same = _text(x).to_numpy() == _text(y).to_numpy()
        same[column] = np.where(numeric.to_numpy(), close | (xn.isna() & yn.isna()).to_numpy(), text_same)
    return pd.DataFrame(same, index=a.index)


def _merged_reason(
    change: str,
    ticker: Any,
    stamp: pd.Timestamp,
    fiscal_end: Any,
    windows: Sequence[Window],
    frontier: Mapping[Any, pd.Timestamp],
) -> tuple[str, str]:
    """`(reason, evidence)` explaining one changed Sharadar history row: a predecessor window, a period after the
    snapshot, or the trailing quarters after a window; empty strings when none does."""
    hit = next((w for w in windows if w.ticker == ticker and w.holds(fiscal_end)), None)
    trail = next((w for w in windows if w.ticker == ticker and w.trails(fiscal_end)), None)
    if hit is not None:
        start, end = hit.start.date() if hit.start is not None else "-", hit.end.date() if hit.end is not None else "-"
        return "predecessor_vendor_series", f"{hit.vendor_ticker} window {start}..{end}"
    if change == "added" and ticker in frontier and stamp > frontier[ticker]:
        return "new_period", f"after the snapshot's last as_of {frontier[ticker].date()}"
    if change == "changed" and trail is not None:
        end = trail.end.date() if trail.end is not None else "-"
        return "predecessor_vendor_series", f"trailing quarters after the {trail.vendor_ticker} window ending {end}"
    return "", ""


def merged_diff(before: pd.DataFrame, after: pd.DataFrame, windows: Sequence[Window]) -> pd.DataFrame:
    """Rows of `sharadar_fundamentals_history` that changed, explained by the predecessor replacement or a period after
    the snapshot."""
    key = ["ticker", "as_of"]
    columns = [c for c in before.columns if c in after.columns and c not in key]
    merged = before.merge(after, on=key, how="outer", suffixes=("_b", "_a"), indicator=True)
    same = _same_cells(
        merged[[f"{c}_b" for c in columns]].set_axis(columns, axis=1), merged[[f"{c}_a" for c in columns]].set_axis(columns, axis=1), columns
    )
    changed_cols = [[c for c in columns if not s[c]] for _, s in same.iterrows()]
    frontier = pd.to_datetime(before["as_of"]).groupby(before["ticker"]).max().to_dict() if not before.empty else {}
    out = []
    for (state, ticker, as_of), cols, b_end, a_end in zip(
        merged[["_merge", "ticker", "as_of"]].itertuples(index=False, name=None),
        changed_cols,
        merged.get("fiscal_end_b", pd.Series(pd.NaT, index=merged.index)),
        merged.get("fiscal_end_a", pd.Series(pd.NaT, index=merged.index)),
        strict=True,
    ):
        if state == "both" and not cols:
            continue
        change = {"left_only": "removed", "right_only": "added"}.get(state, "changed")
        fiscal_end = pd.to_datetime(a_end if not pd.isna(a_end) else b_end)
        stamp = pd.Timestamp(as_of)
        reason, evidence = _merged_reason(change, ticker, stamp, fiscal_end, windows, frontier)
        out.append(
            {
                "ticker": ticker,
                "as_of": stamp.strftime("%Y-%m-%d"),
                "fiscal_end": "" if pd.isna(fiscal_end) else fiscal_end.strftime("%Y-%m-%d"),
                "change": change,
                "columns": ",".join(cols[:12]),
                "reason": reason,
                "evidence": evidence,
            }
        )
    return pd.DataFrame(out, columns=MERGED_COLUMNS)


def prices_diff(before: pd.DataFrame, after: pd.DataFrame, secondary_symbols: Collection[str]) -> pd.DataFrame:
    """`prices` may only gain rows: secondary-class symbols, or dates after a ticker's last stored date."""
    key = ["ticker", "date"]
    columns = [c for c in before.columns if c in after.columns and c not in key]
    merged = before.merge(after, on=key, how="outer", suffixes=("_b", "_a"), indicator=True)
    both = merged["_merge"].eq("both")
    same = _same_cells(
        merged[[f"{c}_b" for c in columns]].set_axis(columns, axis=1), merged[[f"{c}_a" for c in columns]].set_axis(columns, axis=1), columns
    )
    differs = ~same.all(axis=1)
    rows = merged[~both | differs]
    known = set(before["ticker"])
    frontier = pd.to_datetime(before["date"]).groupby(before["ticker"]).max().to_dict() if not before.empty else {}
    out = []
    for idx, row in rows.iterrows():
        state, ticker, day = row["_merge"], str(row["ticker"]), pd.Timestamp(row["date"])
        if state == "right_only" and ticker not in known:
            reason = "secondary_class_added" if ticker in secondary_symbols else ""
        elif state == "right_only" and day > frontier.get(ticker, day):
            reason = "new_data"
        else:
            reason = ""
        cols = [c for c in columns if not same.at[cast(Any, idx), c]] if state == "both" else []
        out.append(
            {
                "ticker": ticker,
                "date": day.strftime("%Y-%m-%d"),
                "change": {"left_only": "removed", "right_only": "added"}.get(state, "changed"),
                "columns": ",".join(cols),
                "reason": reason,
                "evidence": "",
            }
        )
    return pd.DataFrame(out, columns=PRICES_COLUMNS)


# --------------------------------------------------------------------------- hypotheses


def check_hypotheses(
    hypotheses: Sequence[Mapping[str, Any]],
    filing: pd.DataFrame,
    skip: Collection[str] = (),
    insider: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Each hypothesis's count of diff rows with its reason; `fail` when it differs from `expected` (null = report only).
    `ciks` narrows filing rows by filer CIK and insider rows by issuer CIK."""
    out = []
    for h in hypotheses:
        tickers = set(h.get("tickers") or ())
        ciks = set(h.get("ciks") or ())
        if h["kind"] == "filing_lineage":
            table = h.get("table", HISTORY_TABLE)
            change = "removed" if h["change"] == "removed" else "added"
            decision = filing["old_decision"].isin(["stored", "kept"]) if change == "removed" else filing["new_decision"].isin(["stored", "kept"])
            rows = filing[filing["table"].eq(table) & filing["canonical_company"].isin(tickers) & decision]
            rows = rows[rows["cik"].isin(ciks)] if ciks else rows
        else:
            table = Tables.insider_transactions.name
            frame = insider if insider is not None else pd.DataFrame(columns=INSIDER_COLUMNS)
            rows = frame[frame["ticker"].isin(tickers)]
            rows = rows[rows["issuer_cik"].isin(ciks)] if ciks else rows
        observed = int(rows["reason"].eq(h["reason"]).sum())
        expected = h.get("expected")
        status = "skipped" if h["id"] in skip else "pass" if expected is None or observed == int(expected) else "fail"
        out.append(
            {
                "id": h["id"],
                "kind": h["kind"],
                "table": table,
                "tickers": ",".join(sorted(tickers)),
                "change": h["change"],
                "reason": h["reason"],
                "expected": expected,
                "observed": observed,
                "status": status,
            }
        )
    return pd.DataFrame(out, columns=HYPOTHESIS_COLUMNS)


def load_hypotheses(config_dir: str | Path) -> list[dict[str, Any]]:
    """The hypotheses of `configs/sec/expected_lineage_changes.json`."""
    return list(json.loads((Path(config_dir) / HYPOTHESES_FILE).read_text(encoding="utf-8"))["hypotheses"])


# --------------------------------------------------------------------------- store reads


def _filer_columns(store: Any, spec: FilerTable) -> dict[str, str]:
    """Stored column -> gate name for one filer table's projection."""
    names = {spec.key_col: "accession", "ticker": "ticker", spec.cik_col: "cik", spec.date_col: "filed"}
    present = set(store.columns(spec.table))
    if "form" in present:
        names["form"] = "form"
    if spec.table.name == Tables.sec_fundamentals_facts.name:
        names["period_of_report"] = "period_end"
    return names


def _stream(
    store: Any, table: Any, columns: Sequence[str], tickers: Sequence[str] | None, ticker_col: str = "ticker", dedupe: bool = False
) -> pd.DataFrame:
    """A projected read, streamed in chunks (and per ticker chunk when scoped)."""
    if not store.exists(table):
        return pd.DataFrame(columns=list(columns))
    wheres: list[dict[str, object] | None] = [None] if not tickers else [{ticker_col: list(tickers[i : i + 50])} for i in range(0, len(tickers), 50)]
    frames = []
    for where in wheres:
        for chunk in store.iter_load(table, columns=list(columns), where=where, chunksize=_CHUNK):
            frames.append(chunk.drop_duplicates() if dedupe else chunk)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))
    return frame.drop_duplicates(ignore_index=True) if dedupe else frame


def read_filer_table(store: Any, spec: FilerTable, tickers: Sequence[str] | None) -> pd.DataFrame:
    """Distinct (ticker, accession, cik, filed[, form, period_end]) of one filer table."""
    names = _filer_columns(store, spec)
    frame = _stream(store, spec.table, list(names), tickers, dedupe=True).rename(columns=names)
    frame["cik"] = pad_cik_series(frame["cik"])
    frame["ticker"] = frame["ticker"].astype(str)
    frame["accession"] = frame["accession"].astype(str)
    for column in ("form", "period_end"):
        if column not in frame.columns:
            frame[column] = None
    return frame.sort_values(["ticker", "accession", "filed"], kind="mergesort").drop_duplicates(["ticker", "accession"], ignore_index=True)


#: Everything `snapshot` dumps: name -> (table, columns or None for all, chunked ticker scope).
_FRAMES: dict[str, tuple[Any, tuple[str, ...] | None]] = {
    "universe_cik_symbol": (Tables.universe_cik_symbol, None),
    "symbol_tenure": (Tables.symbol_tenure, None),
    "roster": (Tables.universe_tickers, ("ticker", "cik")),
    "insider": (Tables.insider_transactions, (*_INSIDER_KEY, "ticker", "issuer_cik", "filing_date")),
    "merged": (Tables.sharadar_fundamentals_history, tuple(Tables.sharadar_fundamentals_history.read_columns)),
    "prices": (Tables.prices, ("ticker", "date", "open", "high", "low", "close_split", "close_total", "volume")),
}
_UNSCOPED = {"universe_cik_symbol", "symbol_tenure", "roster"}


def _read(store: Any, name: str, tickers: Sequence[str] | None) -> pd.DataFrame:
    table, columns = _FRAMES[name]
    if not store.exists(table):
        return pd.DataFrame(columns=list(columns or ()))
    cols = [c for c in (columns or store.columns(table)) if c in set(store.columns(table))]
    return _stream(store, table, cols, None if name in _UNSCOPED else tickers)


def take_snapshot(store: Any, out_dir: Path, tickers: Sequence[str] | None = None) -> dict[str, int]:
    """Dump the before state (read-only, projected) to `out_dir`; returns rows per file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in _FRAMES:
        frame = _read(store, name, tickers)
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        counts[name] = len(frame)
        log.info("snapshot %s: %d row(s)", name, len(frame))
    for spec in PURGE_TABLES:
        frame = read_filer_table(store, spec, tickers)
        frame.to_parquet(out_dir / f"filing_{spec.table.name}.parquet", index=False)
        counts[f"filing_{spec.table.name}"] = len(frame)
        log.info("snapshot filing_%s: %d accession(s)", spec.table.name, len(frame))
    meta = {"taken_at": pd.Timestamp.now().isoformat(timespec="seconds"), "tickers": list(tickers or []), "rows": counts}
    (out_dir / "snapshot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return counts


def _snap(snap_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(snap_dir / f"{name}.parquet")


def _scopes(identity: Identity, tickers: Iterable[str]) -> dict[str, FilingScope]:
    out = {}
    for ticker in sorted({normalise_ticker(t) for t in tickers}):
        try:
            out[ticker] = identity.filing_scope(ticker)
        except UnknownUniverseTickerError:
            continue
    return out


def _roster_cik(roster: pd.DataFrame) -> dict[str, str]:
    """`{ticker: padded CIK}` of a snapshot roster, skipping a row with no CIK."""
    ciks = pad_cik_series(roster["cik"])
    has_cik = ciks.ne("")
    return dict(zip(roster.loc[has_cik, "ticker"].map(normalise_ticker), ciks[has_cik], strict=False))


def _old_ciks(lineage: pd.DataFrame, roster: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Per roster ticker: the CIKs of its entity in the before lineage, roster CIK included."""
    by_cik = entity_by_cik_map(lineage)
    by_entity: dict[str, set[str]] = {}
    for cik, entity in by_cik.items():
        by_entity.setdefault(entity, set()).add(cik)
    out = {}
    for ticker, cik in _roster_cik(roster).items():
        out[ticker] = frozenset(by_entity.get(entity_or_singleton(by_cik, cik), set()) | {cik})
    return out


SECTIONS = ("filing", "insider", "merged", "prices")


def run_diff(
    context: Any,
    snap_dir: Path,
    out_dir: Path,
    *,
    tickers: Sequence[str] | None = None,
    skip: Collection[str] = (),
    sections: Collection[str] = SECTIONS,
) -> int:
    """Compare the snapshot with the current store; write the diff files of `sections` and the summary; 0 when every
    row is explained and no hypothesis of a computed section fails."""
    store, config_dir = context.store, str(context.config_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = [normalise_ticker(t) for t in tickers] if tickers else None
    _check_scope(snap_dir, scope)
    identity = load_identity(context, refresh=True)
    old_lineage, old_roster = _snap(snap_dir, "universe_cik_symbol"), _snap(snap_dir, "roster")
    master = store.load(Tables.universe_cusip, optional=True)
    master = (
        master
        if master is not None
        else pd.DataFrame(
            columns=[
                "security_id",
                "valid_from",
                "valid_to",
                "lineage_role",
                "lineage_reason",
                "conversion_ratio",
                "issuer_cik",
                "exchange",
                "market_symbol",
            ]
        )
    )
    frames: dict[str, pd.DataFrame] = {}
    stats: dict[str, int] = {}
    if "filing" in sections:
        frames["filing"] = _filing_section(store, snap_dir, identity, _old_ciks(old_lineage, old_roster), scope)
    if "insider" in sections:
        frames["insider"] = _insider_section(store, snap_dir, identity, scope)
    if "merged" in sections:
        frames["merged"] = _merged_section(context, snap_dir, scope)
    if "prices" in sections:
        frames["prices"] = _prices_section(store, snap_dir, master, scope)
    shipped = load_hypotheses(config_dir)
    retired = sum(h["kind"] in RETIRED_KINDS for h in shipped)
    listed = [h for h in shipped if HYPOTHESIS_KINDS.get(h["kind"]) in frames]
    outside = {h["id"] for h in listed if scope is not None and not set(h.get("tickers") or ()) <= set(scope)}
    hypotheses = check_hypotheses(
        listed,
        frames.get("filing", pd.DataFrame(columns=FILING_COLUMNS)),
        set(skip) | outside,
        frames.get("insider"),
    )
    for name, frame in frames.items():
        frame.to_csv(out_dir / OUTPUTS[name], index=False)
    hypotheses.to_csv(out_dir / "gate_hypotheses.csv", index=False)
    unexplained = {name: int(frame["reason"].eq("").sum()) for name, frame in frames.items()}
    failed = hypotheses[hypotheses["status"].eq("fail")]
    lines = [
        f"identity regression gate: snapshot {snap_dir}; sections {', '.join(frames)}; scope {'all' if not scope else ', '.join(scope)}",
        *(f"  {k}: {v:,}" for k, v in stats.items()),
    ]
    for name, frame in frames.items():
        counts = frame["reason"].replace("", "UNEXPLAINED").value_counts().to_dict()
        lines.append(f"{OUTPUTS[name]}: {len(frame):,} row(s); by reason {counts}")
    lines.append(f"hypotheses: {hypotheses['status'].value_counts().to_dict()}; {retired} retired market_tape hypothesis(es) not checked")
    lines += [f"  FAIL {r.id}: expected {r.expected}, observed {r.observed} ({r.reason})" for r in failed.itertuples(index=False)]
    verdict = "PASS" if not any(unexplained.values()) and failed.empty else "FAIL"
    lines.append(f"unexplained rows {unexplained}; verdict {verdict}")
    text = "\n".join(lines)
    (out_dir / "gate_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if verdict == "PASS" else 1


def _check_scope(snap_dir: Path, scope: Sequence[str] | None) -> None:
    """Refuse a diff scope the snapshot does not cover (a scoped snapshot holds only its own tickers)."""
    meta_path = snap_dir / "snapshot_meta.json"
    taken = json.loads(meta_path.read_text(encoding="utf-8")).get("tickers") if meta_path.is_file() else None
    if taken and (scope is None or not set(scope) <= {normalise_ticker(t) for t in taken}):
        raise ValueError(f"diff scope {scope or 'all'} is outside the snapshot's {sorted(taken)}")


def _filing_section(
    store: Any, snap_dir: Path, identity: Identity, old_ciks: Mapping[str, frozenset[str]], scope: Sequence[str] | None
) -> pd.DataFrame:
    """Accessions added or removed per filer-CIK table, plus the filings feeding `sec_fundamentals_history`."""
    frames, facts = [], {}
    for spec in PURGE_TABLES:
        before = _snap(snap_dir, f"filing_{spec.table.name}")
        before = before[before["ticker"].isin(scope)] if scope else before
        after = read_filer_table(store, spec, scope)
        diff = filing_diff(
            spec.table.name, before, after, _scopes(identity, set(before["ticker"]) | set(after["ticker"])), old_ciks, identity.co_registrant_ciks
        )
        if spec.table.name == Tables.sec_fundamentals_facts.name:
            facts = {"before": before, "after": after, "diff": diff}
        frames.append(diff)
    tickers = set(facts["before"]["ticker"]) | set(facts["after"]["ticker"])
    frames.append(history_diff(facts["before"], facts["after"], _scopes(identity, tickers), facts["diff"]))
    return pd.concat(frames, ignore_index=True)


def _insider_section(store: Any, snap_dir: Path, identity: Identity, scope: Sequence[str] | None) -> pd.DataFrame:
    """Insider rows that leave canonical history."""
    own = {t: frozenset(s.event_ciks) for t, s in _scopes(identity, identity.roster_cik).items()}
    before = _snap(snap_dir, "insider")
    before = before[before["ticker"].isin(scope)] if scope else before
    present = set(store.columns(Tables.insider_transactions))
    columns = [*_INSIDER_KEY, "ticker", "issuer_cik", "filing_date"] + [c for c in ("lineage_role", "economic_date") if c in present]
    after = _stream(store, Tables.insider_transactions, columns, scope)
    return insider_diff(before, after, own, identity.co_registrant_ciks)


def _merged_section(context: Any, snap_dir: Path, scope: Sequence[str] | None) -> pd.DataFrame:
    store = context.store
    before = _snap(snap_dir, "merged")
    before = before[before["ticker"].isin(scope)] if scope else before
    after = _read(store, "merged", scope)
    if before.empty and after.empty:
        return pd.DataFrame(columns=MERGED_COLUMNS)
    for frame in (before, after):
        for column in ("as_of", "fiscal_end"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
    names = sorted(set(before["ticker"]) | set(after["ticker"]))
    series = load_predecessor_series(context, names)
    stored = (
        set(store.distinct(Tables.sharadar_fundamentals_facts, "ticker", where={"ticker": sorted({s.vendor_ticker for s in series})}))
        if series
        else set()
    )
    windows = [Window(s.ticker, s.vendor_ticker, s.valid_from, s.valid_to) for s in series if s.vendor_ticker in stored]
    return merged_diff(before, after, windows)


def _prices_section(store: Any, snap_dir: Path, master: pd.DataFrame, scope: Sequence[str] | None) -> pd.DataFrame:
    before = _snap(snap_dir, "prices")
    secondary = (
        {yahoo_symbol(str(s)) for s in master.loc[master["lineage_role"].eq(SECONDARY_CLASS), "market_symbol"].dropna()}
        if "market_symbol" in master.columns
        else set()
    )
    if scope:
        before = before[before["ticker"].isin(scope)]
        after = _read(store, "prices", sorted(set(scope) | secondary))
    else:
        after = _read(store, "prices", None)
    for frame in (before, after):
        frame["date"] = pd.to_datetime(frame["date"])
    return prices_diff(before, after, secondary)


# --------------------------------------------------------------------------- CLI


class ReadOnlyStore:
    """The store with every write refused: the gate only reads."""

    WRITES = frozenset({"save", "delete", "replace", "drop", "bulk_seed", "append_tail", "ensure_columns"})

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        if name in self.WRITES:
            raise PermissionError(f"identity regression gate is read-only: store.{name} refused")
        return getattr(self._store, name)


def _context(config_dir: str, store_url: str | None) -> Any:
    from src.context import get_config_context  # noqa: PLC0415  (loads `.env`; only the CLI needs it)

    _, context = get_config_context(config_dir, use_cache=False, save=False)
    if store_url:
        from sqlalchemy import create_engine  # noqa: PLC0415

        from src.data_store.store import DataStore  # noqa: PLC0415

        context.store = DataStore(create_engine(store_url))
    return context


def main(argv: Sequence[str] | None = None, context_factory: Callable[[str, str | None], Any] = _context) -> int:
    """`snapshot <dir>` or `diff <dir>`; returns the exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("command", choices=["snapshot", "diff"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("-c", "--config-dir", default="./configs")
    parser.add_argument("-t", "--tickers", nargs="*", default=None)
    parser.add_argument("-o", "--out", type=Path, default=None, help="diff output directory (default: the snapshot's parent)")
    parser.add_argument("--skip-hypothesis", nargs="*", default=[], help="hypothesis ids reported as skipped")
    parser.add_argument("--store-url", default=None, help="read the 'after' state from this SQLAlchemy URL instead of the configured store")
    parser.add_argument("--sections", nargs="*", default=list(SECTIONS), choices=list(SECTIONS), help="diff surfaces to compute (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    context = context_factory(args.config_dir, args.store_url if args.command == "diff" else None)
    context.store = ReadOnlyStore(context.store)
    tickers = [normalise_ticker(t) for raw in args.tickers for t in raw.split(",") if t.strip()] if args.tickers else None
    try:
        if args.command == "snapshot":
            counts = take_snapshot(context.store, args.directory, tickers)
            print(f"snapshot written to {args.directory}: {counts}")
            return 0
        return run_diff(
            context, args.directory, args.out or args.directory.parent, tickers=tickers, skip=set(args.skip_hypothesis), sections=args.sections
        )
    except Exception:
        log.exception("identity regression gate: %s failed", args.command)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
