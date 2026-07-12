"""
Management, ownership and workforce snapshot per ticker (yfinance).

READ THIS FIRST -- free-data reality (same ceiling as analyst estimates):
yfinance exposes a rich CURRENT snapshot of governance/ownership, but NOT a
historical archive. So, exactly like fetch_analyst_estimates, this script pulls
everything available TODAY and APPENDS it to `management_history.parquet`; a real
point-in-time history accrues going forward as you run it monthly/quarterly.
These are slow-moving structural attributes (founder-led, family ownership,
insider stake), so a single snapshot is already a strong cross-sectional signal
for ranking today's names, and becomes backtestable as history builds up.

NOT available from yfinance (do not fake these):
  * daily hirings / open job postings / layoff plans -> alt-data (job boards,
    news NLP), not in yfinance;
  * CEO tenure / appointment date -> not exposed (would need proxy-statement text);
  * historical employee counts -> only the current `fullTimeEmployees` is given,
    and SEC companyfacts has no clean employee-count concept, so employee-GROWTH
    history is not reconstructable from these free sources.

What IS captured per ticker (snapshot):
  employees, insider %, institution %, #institutions, founder-led / founder-CEO
  flags (parsed from officer titles), family-ownership proxy (parsed from the
  insider roster: repeated surnames / family trusts + high insider stake),
  CEO age & pay, officer count / avg age / total pay, and net insider buying.

Run:
    python -m src.data_extract.fetch_management
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.context import Context

_FAMILY_KEYWORDS = ("family", "trust", "holdings", "foundation")
_FAMILY_INSIDER_MIN = 0.10   # a family firm keeps a meaningful insider stake


def _parse_officers(officers: list | None) -> dict:
    """Founder / CEO / officer aggregates from the companyOfficers list."""
    out = {"ceo_age": np.nan, "ceo_pay": np.nan, "founder_ceo": 0, "founder_present": 0,
           "n_officers": 0, "avg_officer_age": np.nan, "total_officer_pay": np.nan}
    if not officers:
        return out
    ages, pays, ceo = [], [], None
    for o in officers:
        title = (o.get("title") or "").lower()
        if "founder" in title:
            out["founder_present"] = 1
        if ("ceo" in title) or ("chief executive" in title):
            ceo = o
        if o.get("age"):
            ages.append(o["age"])
        if o.get("totalPay"):
            pays.append(o["totalPay"])
    out["n_officers"] = len(officers)
    if ages:
        out["avg_officer_age"] = float(np.mean(ages))
    if pays:
        out["total_officer_pay"] = float(np.sum(pays))
    if ceo is not None:
        out["ceo_age"] = ceo.get("age")
        out["ceo_pay"] = ceo.get("totalPay")
        out["founder_ceo"] = int("founder" in (ceo.get("title") or "").lower())
    return out


def _surname(name: str) -> str:
    parts = str(name).strip().split()
    return parts[0].upper() if parts else ""


def _parse_family(roster: pd.DataFrame | None, held_insiders: float | None) -> dict:
    """Family-ownership proxy from the insider roster: a family firm shows either
    a family trust/holding vehicle or several insiders sharing a surname, AND a
    meaningful insider stake."""
    out = {"family_owned": 0, "family_trust_present": 0,
           "max_surname_repeat": 0, "n_beneficial_owners": 0}
    if roster is None or len(roster) == 0 or "Name" not in roster.columns:
        return out
    names = roster["Name"].astype(str).tolist()
    positions = (roster["Position"].astype(str).tolist()
                 if "Position" in roster.columns else [])

    keyword_hit = any(any(k in n.lower() for k in _FAMILY_KEYWORDS) for n in names)
    out["family_trust_present"] = int(keyword_hit)
    out["n_beneficial_owners"] = sum("beneficial owner" in p.lower() for p in positions)

    surnames = [_surname(n) for n in names
                if not any(k in n.lower() for k in _FAMILY_KEYWORDS)]
    surnames = [s for s in surnames if s]
    repeat = max(Counter(surnames).values()) if surnames else 0
    out["max_surname_repeat"] = int(repeat)

    hi = float(held_insiders) if held_insiders is not None and np.isfinite(held_insiders) else 0.0
    out["family_owned"] = int((repeat >= 2 or keyword_hit) and hi >= _FAMILY_INSIDER_MIN)
    return out


def _parse_insider_net(purchases: pd.DataFrame | None) -> float:
    """Net insider buying over the last 6 months as a fraction (buys - sells)."""
    if purchases is None or len(purchases) == 0:
        return np.nan
    label_col = purchases.columns[0]
    try:
        mask = purchases[label_col].astype(str).str.contains("% Net Shares", case=False, na=False)
        if mask.any():
            v = purchases.loc[mask, "Shares"].iloc[0]
            return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan
    return np.nan


def _safe_attr(ticker_obj, attr):
    try:
        v = getattr(ticker_obj, attr)
        return v if isinstance(v, pd.DataFrame) and not v.empty else None
    except Exception:
        return None


def _snapshot_row(ticker: str, t: yf.Ticker, as_of: str) -> dict:
    info = t.info or {}
    held_ins = info.get("heldPercentInsiders")
    row = {
        "ticker": ticker, "as_of": as_of,
        "fullTimeEmployees": info.get("fullTimeEmployees"),
        "heldPercentInsiders": held_ins,
        "heldPercentInstitutions": info.get("heldPercentInstitutions"),
        "institutionsCount": (info.get("institutionsCount")
                              or (_safe_attr(t, "major_holders") is not None)),
    }
    row.update(_parse_officers(info.get("companyOfficers")))
    row.update(_parse_family(_safe_attr(t, "insider_roster_holders"), held_ins))
    row["net_insider_buying"] = _parse_insider_net(_safe_attr(t, "insider_purchases"))
    return row


def fetch_snapshot(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    as_of = datetime.utcnow().date().isoformat()
    rows = []
    for tkr in tqdm(tickers, desc="Fetching management/ownership snapshot"):
        try:
            rows.append(_snapshot_row(tkr, yf.Ticker(tkr), as_of))
        except Exception as e:  # noqa: BLE001 - per-ticker network/parse issues
            context.log.warning("%s: management snapshot failed (%s)", tkr, e)
        time.sleep(pause)
    return pd.DataFrame(rows)


def append_to_history(context: Context, snapshot: pd.DataFrame) -> pd.DataFrame:
    path = context.paths["MANAGEMENT_HISTORY_PATH"]
    if path.exists():
        hist = pd.concat([pd.read_parquet(path), snapshot], ignore_index=True)
        hist = hist.drop_duplicates(subset=["ticker", "as_of"], keep="last")
    else:
        hist = snapshot
    hist.to_parquet(path, index=False)
    return hist


def fetch_management(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    """Pull today's management/ownership snapshot, save it, and append it to the
    accruing history (point-in-time archive built up over successive runs)."""
    snapshot = fetch_snapshot(context, tickers, pause)
    snapshot.to_parquet(context.paths["MANAGEMENT_PATH"], index=False)
    hist = append_to_history(context, snapshot)
    context.log.info("Management snapshot: %d tickers (founder-led=%d, family-owned=%d); "
                     "history now %d rows across %d dates",
                     len(snapshot), int(snapshot.get("founder_present", pd.Series()).sum()),
                     int(snapshot.get("family_owned", pd.Series()).sum()),
                     len(hist), hist["as_of"].nunique())
    return snapshot
