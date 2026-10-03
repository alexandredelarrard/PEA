"""
insider_quality.py  (src/data_aggregate/utils/institutionals/insider_quality.py)
-------------------------------------------------------------------
Scope and repair the Form 3/4/5 transaction log before any feature is summed off it.

Scope: non-derivative, common-stock, open-market (P/S) rows with a usable price. Repair: a filed
`price_per_share` more than `PRICE_TOLERANCE` above the split-free consensus (median price of
same-ticker, same-class common-stock filings over a trailing window ending on the filing day) is
replaced by `shares x consensus`; too-low prices are only counted. The clock is `filing_date`, so a
later filing never enters an earlier reference. Corrupt `shares` cannot be separated by size, so
oversized rows are logged, never dropped.
"""

from __future__ import annotations

import logging
import re
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.data_utils import to_day

_log = logging.getLogger(__name__)

#: Open-market purchase/sale codes; every other code is non-discretionary and never netted.
OPEN_MARKET_CODES: tuple[str, ...] = ("P", "S")

#: Codes whose `price_per_share` is a market price and may back the consensus (`F` is withholding at the close).
MARKET_PRICED_CODES: tuple[str, ...] = ("P", "S", "F")

#: Common/ordinary-stock security titles, including common misspellings; preferred and 401(k) units fall through.
COMMON_STOCK_RE = re.compile(
    r"COMMON|COMON|COMMOM|COMM|ORDINARY|REGISTER|CLASS\s+[A-Z]\b|SHARES\s+OF\s+BENEFICIAL|DEPOSITARY|" r"^\s*(?:COM|STOCK|SHARES)\s*$", re.I
)

#: Max ratio of filed price to the trailing consensus before the price is treated as corrupt.
PRICE_TOLERANCE: float = 10.0

#: Share-class token in a security title, so the consensus is built within one class.
_CLASS_RE = re.compile(r"\bCLASS\s+([A-Z])\b|\bSERIES\s+([A-Z])\b", re.I)

#: Trailing publication-time window behind the consensus median.
CONSENSUS_WINDOW: str = "31D"

#: A single transaction above this share of the company is reported, never dropped.
FLAG_PCT_SHARES_OUTSTANDING: float = 0.25

#: Free-text `officer_title` -> role, first match wins (a "Chairman, President & CEO" is a CEO).
CHIEF = r"(?:CHIEF|CHF)"
OFFICER = r"OFF(?:ICER|CR|R)"
OPERATIONS = r"OPER(?:ATION(?:S)?|ATING)?"
ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "CEO",
        rf"\b{CHIEF}\s+EXEC(?:UTIVE)?\s+{OFFICER}\b" rf"|\bC\s*E\s*O\b",
    ),
    ("CFO", rf"\b{CHIEF}\s+FIN(?:ANCIAL)?\s+{OFFICER}\b" rf"|\bPRINCIPAL\s+FIN(?:ANCIAL)?\s+{OFFICER}\b" rf"|\bC\s*F\s*O\b" rf"|\bTREASURER\b"),
    ("COO", rf"\b{CHIEF}\s+{OPERATIONS}\s+{OFFICER}\b" rf"|\bC\s*O\s*O\b"),
    ("CTO_or_technology", rf"\b{CHIEF}\s+TECH(?:NOLOGY)?\s+{OFFICER}\b" rf"|\bC\s*T\s*O\b" rf"|\bTECH(?:NOLOGY)?\b"),
    ("CMO_or_marketing", rf"\b{CHIEF}\s+MARKETING\s+{OFFICER}\b"),
)
OTHER_OFFICER = "other_named_officer"

#: Expected share of titled officer rows falling through to `OTHER_OFFICER` (real named officers outside the map).
ROLE_FALLTHROUGH_MEASURED: float = 0.320

_ROLE_COMPILED = tuple((name, re.compile(pat, re.I)) for name, pat in ROLE_PATTERNS)


def common_stock_mask(security_title: pd.Series) -> pd.Series:
    """True where the security traded is common/ordinary stock; gates both the consensus and the features."""
    return security_title.fillna("").astype(str).str.contains(COMMON_STOCK_RE)


def officer_role(title: object) -> str:
    """Free-text `officer_title` -> the first matching `ROLE_PATTERNS` name, else `OTHER_OFFICER` (also for a blank)."""
    s = "" if title is None else str(title)
    for name, rx in _ROLE_COMPILED:
        if rx.search(s):
            return name
    return OTHER_OFFICER


def security_class(security_title: pd.Series) -> pd.Series:
    """Share-class key for the consensus: `A`, `B`, ... or `COMMON` for an unclassed title."""
    s = security_title.fillna("").astype(str)
    got = s.str.extract(_CLASS_RE)
    return got[0].fillna(got[1]).str.upper().fillna("COMMON")


def consensus_price(txns: pd.DataFrame, *, window: str = CONSENSUS_WINDOW) -> pd.Series:
    """Per-row reference price: trailing-`window` rolling median of per-day medians of common-stock,
    market-priced filed prices for the same ticker and share class, keyed on `filing_date` (the
    publication clock). Aligned to `txns.index`; NaN where the filing day has no reference.
    """
    need = {"ticker", "filing_date", "price_per_share", "transaction_code", "security_title"}
    if txns.empty or not need.issubset(txns.columns):
        return pd.Series(np.nan, index=txns.index, dtype="float64")

    code = txns["transaction_code"].astype(str).str.upper().str.strip()
    pps = pd.to_numeric(txns["price_per_share"], errors="coerce")
    publication_day = to_day(txns["filing_date"])
    usable = code.isin(MARKET_PRICED_CODES) & (pps > 0) & publication_day.notna() & common_stock_mask(txns["security_title"]) & txns["ticker"].notna()

    if not usable.any():
        return pd.Series(np.nan, index=txns.index, dtype="float64")

    klass = security_class(txns["security_title"])
    src = pd.DataFrame(
        {
            "ticker": txns.loc[usable, "ticker"].astype(str),
            "klass": klass[usable],
            "publication_day": publication_day[usable],
            "pps": pps[usable],
        }
    )
    day = src.groupby(["ticker", "klass", "publication_day"], sort=False)["pps"].median().rename("m").reset_index()

    out = []
    for (tkr, cls), g in day.groupby(["ticker", "klass"], sort=False):
        g = g.set_index("publication_day").sort_index()
        med = g["m"].rolling(window).median()
        out.append(pd.DataFrame({"ticker": tkr, "klass": cls, "publication_day": g.index, "consensus": med.to_numpy()}))
    ref = pd.concat(out, ignore_index=True)

    keyed = pd.DataFrame({"ticker": txns["ticker"].astype(str), "klass": klass, "publication_day": publication_day})
    merged = keyed.merge(ref, on=["ticker", "klass", "publication_day"], how="left")
    merged.index = txns.index
    return merged["consensus"]


def clean_transactions(insider: pd.DataFrame, *, price_tolerance: float = PRICE_TOLERANCE) -> tuple[pd.DataFrame, dict]:
    """Scope to priced common-stock open-market trades and repair the mispriced ones.

    Returns `(frame, diagnostics)`. Added columns: `code`, `day` (normalised `filing_date`, the
    point-in-time stamp, never `transaction_date`), `value` (repaired USD), `shares_n`,
    `price_repaired`, `role` and `in_exercise_package` (an `S` sharing accession and transaction
    date with an `M`). Derivative, non-common and unpriced rows are dropped, never zero-filled.
    """

    need = {"ticker", "filing_date", "transaction_code", "shares", "value_usd"}
    diag: dict = {"input_rows": 0 if insider is None else len(insider)}
    if insider is None or insider.empty or not need.issubset(insider.columns):
        return pd.DataFrame(), diag

    t = insider.copy()
    t["ticker"] = t["ticker"].astype(str).str.upper().str.strip()
    t["code"] = t["transaction_code"].astype(str).str.upper().str.strip()
    t["day"] = to_day(t["filing_date"])
    t["shares_n"] = pd.to_numeric(t["shares"], errors="coerce")
    pps = pd.to_numeric(cast(pd.Series, t.get("price_per_share")), errors="coerce")

    # Computed before the scope cut, which removes the derivative `M` leg of the package.
    t["in_exercise_package"] = _exercise_packages(t)

    if "security_type" in t.columns:
        t = t[t["security_type"].astype(str).str.lower().eq("nonderiv")]

    if "security_title" in t.columns:
        t = t[common_stock_mask(t["security_title"])]

    t = t[t["code"].isin(OPEN_MARKET_CODES) & t["day"].notna() & (t["ticker"] != "")]
    t = t.dropna(subset=["shares_n"])
    pps = pps.reindex(t.index)
    diag["scoped_rows"] = len(t)
    if t.empty:
        return pd.DataFrame(), diag

    priced = pps > 0
    diag["dropped_unpriced"] = int((~priced).sum())
    diag["dropped_unpriced_shares"] = float(t.loc[~priced, "shares_n"].sum()) / t["shares_n"].sum()
    diag["unpriced_events"] = t.loc[~priced, ["ticker", "day", "code"]].copy()
    t, pps = t[priced], pps[priced]

    ref = consensus_price(insider).reindex(t.index)
    ratio = pps / ref.where(ref > 0)
    # One-sided repair: a too-low ratio may be a genuine low-priced row under a reused ticker, so it is only counted.
    bad = ratio.notna() & (ratio > price_tolerance)
    low = ratio.notna() & (ratio < 1.0 / price_tolerance)
    raw_value = pd.to_numeric(t["value_usd"], errors="coerce")

    # Repair rather than drop: the trade and its share count are real, only the price is wrong.
    t["value"] = raw_value.where(~bad, t["shares_n"] * ref)
    t["price_repaired"] = bad

    # A row whose price survived but whose `value_usd` is missing is still a real trade.
    t["value"] = t["value"].fillna(t["shares_n"] * pps)

    diag["repaired_rows"] = int(bad.sum())
    diag["underpriced_rows"] = int(low.sum())
    diag["value_before"] = float(raw_value.sum())
    diag["value_after"] = float(t["value"].sum())
    diag["no_consensus_rows"] = int(ratio.isna().sum())

    t["role"] = t["officer_title"].map(officer_role) if "officer_title" in t.columns else OTHER_OFFICER
    for flag in ("is_director", "is_officer", "is_ten_pct_owner"):
        t[flag] = pd.to_numeric(cast(pd.Series, t.get(flag)), errors="coerce")
    t["is_10b5_1"] = pd.to_numeric(cast(pd.Series, t.get("is_10b5_1")), errors="coerce")

    _log.info(
        "insider: %s rows -> %s scoped, %s unpriced dropped, %s overpriced repaired, %s underpriced left as filed ($%.3ftn -> $%.3fbn)",
        diag["input_rows"],
        diag["scoped_rows"],
        diag["dropped_unpriced"],
        diag["repaired_rows"],
        diag["underpriced_rows"],
        diag["value_before"] / 1e12,
        diag["value_after"] / 1e9,
    )
    return t, diag


def _exercise_packages(t: pd.DataFrame) -> pd.Series:
    """True for an `S` row in the same accession and transaction date as an option exercise `M`
    (a mechanical exercise-and-sell, excluded from discretionary-sell features)."""
    if "accession_number" not in t.columns or "transaction_date" not in t.columns:
        return pd.Series(False, index=t.index)
    key = pd.MultiIndex.from_arrays([t["accession_number"].astype(str), pd.to_datetime(t["transaction_date"], errors="coerce")])
    has_m = pd.Series(t["code"].eq("M").to_numpy(), index=key).groupby(level=[0, 1]).any()
    return pd.Series(has_m.reindex(key).to_numpy(), index=t.index).fillna(False) & t["code"].eq("S")


def asof_values(frame: pd.DataFrame | None, tickers: pd.Series, days: pd.Series) -> pd.Series:
    """Value of a wide (date x ticker) `frame` as of each `(ticker, day)`, forward-filled: the last
    row at or before the day, NaN before the first observation (never a look-ahead).
    """
    idx = pd.RangeIndex(len(tickers)) if tickers.index.has_duplicates else tickers.index
    if frame is None or frame.empty:
        return pd.Series(np.nan, index=idx, dtype="float64")
    f = frame.ffill()
    pos = f.index.searchsorted(pd.to_datetime(days).to_numpy(), side="right") - 1
    known = (pos >= 0) & tickers.isin(f.columns).to_numpy()
    out = np.full(len(tickers), np.nan)
    if known.any():
        out[known] = f.to_numpy()[pos[known], f.columns.get_indexer(pd.Index(tickers[known]))]
    return pd.Series(out, index=idx, dtype="float64")


def report_oversized(t: pd.DataFrame, shares_outstanding: pd.DataFrame | None, *, threshold: float = FLAG_PCT_SHARES_OUTSTANDING) -> pd.DataFrame:
    """Transactions above `threshold` of shares outstanding, as a frame to log (never a filter);
    empty when no share count is available.
    """
    if t.empty or shares_outstanding is None or shares_outstanding.empty:
        return pd.DataFrame()
    so = asof_values(shares_outstanding, t["ticker"], t["day"]).to_numpy()
    pct = t["shares_n"].to_numpy() / np.where(so > 0, so, np.nan)
    hit = pct > threshold
    if not hit.any():
        return pd.DataFrame()
    cols = [c for c in ("ticker", "day", "code", "owner_name", "shares_n", "value") if c in t]
    flagged = t.loc[hit, cols].copy()
    flagged["pct_shares_outstanding"] = pct[hit]
    return flagged.sort_values("pct_shares_outstanding", ascending=False)
