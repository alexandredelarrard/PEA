"""
insider_quality.py  (src/data_aggregate/utils/institutionals/insider_quality.py)
-------------------------------------------------------------------
Scope and repair the Form 3/4/5 transaction log before any feature is summed off it.

WHY THIS MODULE EXISTS, IN ONE NUMBER. Read raw, `insider_transactions.value_usd` averages
**$447,771,735,138** per Form 4 line. That is not a transaction size, it is one row: the
largest (NCLH 2021-03-09, "Exchangeable Senior Notes due 2026", 414m "shares" at "$1.03e9")
is **49.4% of the whole-table sum on its own**, the top three are **79%**, and dropping the
top five leaves **0.04%**. The median, $16,516, is sane throughout -- only the tail is
broken. After the scope cut and repair below the mean is **$2,339,662** and the median
$114,116.

⚠ THREE POPULATIONS, AND MIXING THEM UNDERSTATES THE DEFECT. Measured 2026-09-10:

    whole table (value_usd non-null)      1,934,681 rows   sum $8.66e17   mean $447.77bn
    non-derivative only                   1,412,708 rows   sum $2.18e16
    market-priced non-deriv P/S/F           875,361 rows   sum $2.14e16
    after scope + repair (features read)    673,746 rows   sum $1,576bn   mean $2.34m

Within the third population -- the only one a price screen can reach -- **210 rows (0.024%)
carry 99.99% of its total**, because their `price_per_share` is wrong, and
`value_usd == shares x price_per_share` on 100% of them, so the multiplication is right and
the PRICE is the corrupt field. ⚠ The three note rows above are NOT among those 210: they
are derivative rows, removed by SCOPE rather than by the price screen. A repair without the
scope cut leaves 79% of the fabricated total in place.

Phase 1b reconciled this table against the SEC's own bulk zips with 0 mismatches, so these
are the FILERS' numbers, not a parse defect of ours; they cannot be fixed upstream and must
be handled here.

The shapes, all measured:

    MSFT  2020-09-01  Nadella        83,572 sh @ $2,261,327   consensus $219.55   x10,299
    NOW   2015-04-01  Slootman       76,637 sh @ $756,231     consensus $79.62    x 9,499
    LLY   2025-11-14  Lilly Endowmt  55,908 sh @ $1,031,414   consensus $1,014.69 x 1,016
    AMD   2017-08-04  Mubadala   40,000,000 sh @ $525,600,000 consensus $13.75    x38m

The first three are decimal slips (74 of the 210 are within 2% of an exact 10^-1, 70 of
10^-2, 13 of 10^-3, and 43 of positive powers); the last is the whole transaction value typed
into the per-share field.

THE SCREEN IS SPLIT-FREE, DELIBERATELY. The obvious check -- compare `price_per_share` to
`close_split` -- fails on every name that ever split, because the filer's price is as-traded
and `close_split` is restated to today's basis. So the reference is the median
`price_per_share` of the OTHER market-priced filings on the same ticker within +/-15 days.
Both sides then carry the same unknown split basis and it cancels. 92.4% of ticker-days have
2+ filings behind that median (p50 = 8); the 7.59% backed by a single row are undetectable by
construction, and only 3 of the 210 hits sit on such a day.

⚠ THE CONSENSUS MUST BE COMMON-STOCK-ONLY. BAC 2016-09-22 has exactly two market-priced
non-derivative rows -- common at $15.61 and PREFERRED at $100.00 par -- and their median is
$57.80, which is not a price BAC ever traded at. Restricting to common stock returns $15.61.
This is why `common_stock_mask` gates the reference and not only the features.

WHAT THIS MODULE DOES NOT FIX, stated because a silent residual is worse than a known one:

  * `shares` can be corrupt too, and no size threshold separates the corrupt from the real.
    The largest genuine single disposal on this universe is TRGP 2010-12-10 at **40.6% of
    shares outstanding** (Warburg Pincus' exit); the largest implausible purchase is BAC
    2016-09-22 "YNOFACE Holdings Inc" at **41.2%** ($65.6bn, owned_after == shares, filer
    declaring neither director nor officer nor 10% owner -- but `is_other = 1`, so the
    relationship boxes do not separate it either). A cap that rejects the second rejects the
    first. Rows above `FLAG_PCT_SHARES_OUTSTANDING` are therefore LOGGED, not dropped.
  * The ticker itself. `insider_transactions.ticker` is resolved SYMBOL-FIRST at extraction,
    so a ticker that a different live company used earlier carries that company's insiders:
    2,046 Weight Watchers rows under `WTW`, 1,207 CoreSite rows under `COR`, 1,111 old
    Constellation Energy rows under `CEG`, and 2,075 Trane rows under `IR` -- whose CIK
    0001466258 is `TT`'s own universe CIK. Against the registrant register, **38,910 rows
    (1.92%), 8,640 of them P/S, across 72 in-universe tickers** sit outside their ticker's lineage. That
    is an extraction defect and is not repaired here; see `_filter_universe` in
    `data_extract/utils/institutionals/fetch_insider_transactions.py`.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

#: Open-market codes. `P` is a purchase and `S` a sale; everything else is non-discretionary
#: and never netted into an insider number -- grants `A` (498,867 rows), option exercises `M`
#: (491,213), tax withholding `F` (203,916), gifts `G` (51,902), other-acquisition `J`
#: (47,379), conversions `C` (27,123), other-disposition `D` (23,132). `P` is only **1.77%**
#: of the table, which is exactly why the taxonomy has to be explicit: net everything and the
#: signal is 94% compensation mechanics.
OPEN_MARKET_CODES: tuple[str, ...] = ("P", "S")

#: Codes whose `price_per_share` is the MARKET price and so may back the consensus. `F` (tax
#: withholding at the vest-day close) is in because it is dense and priced at market; `M`
#: (strike), `A` (0) and `C` (conversion terms) are not prices and are excluded.
MARKET_PRICED_CODES: tuple[str, ...] = ("P", "S", "F")

#: Matches 99.1% of non-derivative `P` titles across 293 distinct spellings. The 324 that fall
#: through are preferred series, trust-preferred securities and 401(k) units -- none of them a
#: common-share signal, and none of them priced like one.
COMMON_STOCK_RE = re.compile(
    r"COMMON|ORDINARY|CLASS\s+[A-Z]\b|SHARES\s+OF\s+BENEFICIAL|DEPOSITARY|"
    r"^\s*(?:COM|STOCK|SHARES)\s*$", re.I)

#: A filed price this far from the ticker's own +/-15-day consensus is not a price. 10x is two
#: decimal places; the tightest real move between a transaction and its neighbours is well
#: inside it -- the p99.99 of the ratio is 4.02 and the p99.9 is 1.49.
PRICE_TOLERANCE: float = 10.0

#: Matches the share-class token in a security title, so the consensus is built WITHIN a
#: class. Erie Indemnity is why: `ERIE` has Class A around $35 and Class B around $32,740,
#: both titled "... Common Stock", and a ticker-wide median put a 5,000-share Class A purchase
#: at $163.7m -- the repair inventing 4,600x the value it was written to remove.
_CLASS_RE = re.compile(r"\bCLASS\s+([A-Z])\b|\bSERIES\s+([A-Z])\b", re.I)

#: Centred calendar window behind the consensus median -- 31 days is +/-15 either side.
CONSENSUS_WINDOW: str = "31D"

#: A single transaction above this share of the company is reported, never dropped -- see the
#: module docstring for why no threshold can separate the real from the fabricated here.
FLAG_PCT_SHARES_OUTSTANDING: float = 0.25

#: `officer_title` is free text and is populated EXACTLY when `is_officer = 1` (measured: 0.0%
#: blank on officer purchase rows, 99.9% blank on the rest), so the role map's denominator is
#: officer rows and the "blank" rate is not a fall-through. Longest concept first: a title
#: reading "Chairman, President & CEO" is a CEO, not a President.
ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("CEO", r"\bCHIEF\s+EXECUTIVE\b|\bCEO\b|\bC\.E\.O\b"),
    ("CFO", r"\bCHIEF\s+FINANCIAL\b|\bCFO\b|\bC\.F\.O\b|\bPRINCIPAL\s+FINANCIAL\s+OFFICER\b|"
            r"\bTREASURER\b"),
    ("COO_or_President", r"\bCHIEF\s+OPERATING\b|\bCOO\b|\bPRESIDENT\b|\bPRES\.\b"),
)
OTHER_OFFICER = "other_named_officer"

#: Measured fall-through to `OTHER_OFFICER`: **32.0% of 10,390 titled officer purchases**. It
#: is reported rather than tuned away because the residue is real -- Executive Chairman (197),
#: Vice Chair (182), Chief Accounting Officer (148), General Counsel (149 across spellings) --
#: roles that are named officers but none of the three the registry asks for. Raising the map
#: to catch them would change what #33-#35 mean, which is a registry edit, not a code change.
ROLE_FALLTHROUGH_MEASURED: float = 0.320

_ROLE_COMPILED = tuple((name, re.compile(pat, re.I)) for name, pat in ROLE_PATTERNS)


def common_stock_mask(security_title: pd.Series) -> pd.Series:
    """True where the security traded is common/ordinary stock.

    A preferred or trust-preferred purchase is not a common-share signal and is not priced
    like one -- letting it into the consensus is what put BAC's reference at $57.80.
    """
    return security_title.fillna("").astype(str).str.contains(COMMON_STOCK_RE)


def officer_role(title: object) -> str:
    """Free-text `officer_title` -> one of CEO / CFO / COO_or_President / other_named_officer.

    Returns `OTHER_OFFICER` for a blank, which is correct rather than lossy: a blank title
    means the filer is not an officer at all (see `ROLE_PATTERNS`), and those rows are
    selected on `is_director` / `is_ten_pct_owner` instead.
    """
    s = "" if title is None else str(title)
    for name, rx in _ROLE_COMPILED:
        if rx.search(s):
            return name
    return OTHER_OFFICER


def security_class(security_title: pd.Series) -> pd.Series:
    """Share-class key for the consensus: `A`, `B`, ... or `COMMON` for an unclassed title.

    Without it the reference price for a dual-class name is the median of two different
    securities, which is a price neither of them ever traded at.
    """
    s = security_title.fillna("").astype(str)
    got = s.str.extract(_CLASS_RE)
    return got[0].fillna(got[1]).str.upper().fillna("COMMON")


def consensus_price(txns: pd.DataFrame, *, window: str = CONSENSUS_WINDOW) -> pd.Series:
    """Per-row reference price: the median filed price for that ticker AND SHARE CLASS
    within +/-`window`.

    Built from common-stock, market-priced rows only, as a per-(ticker, class,
    transaction_date) median first so one busy day cannot outvote a quiet one, then a
    trailing and a leading rolling median whose midpoint is the reference. Returned aligned
    to `txns.index`, NaN where the ticker-day has no reference.
    """
    need = {"ticker", "transaction_date", "price_per_share", "transaction_code",
            "security_title"}
    if txns.empty or not need.issubset(txns.columns):
        return pd.Series(np.nan, index=txns.index, dtype="float64")

    code = txns["transaction_code"].astype(str).str.upper().str.strip()
    pps = pd.to_numeric(txns["price_per_share"], errors="coerce")
    tdate = pd.to_datetime(txns["transaction_date"], errors="coerce")
    usable = (code.isin(MARKET_PRICED_CODES) & (pps > 0) & tdate.notna()
              & common_stock_mask(txns["security_title"])
              & txns["ticker"].notna())
    if not usable.any():
        return pd.Series(np.nan, index=txns.index, dtype="float64")

    klass = security_class(txns["security_title"])
    src = pd.DataFrame({"ticker": txns.loc[usable, "ticker"].astype(str),
                        "klass": klass[usable], "tdate": tdate[usable], "pps": pps[usable]})
    day = (src.groupby(["ticker", "klass", "tdate"], sort=False)["pps"]
           .median().rename("m").reset_index())

    out = []
    for (tkr, cls), g in day.groupby(["ticker", "klass"], sort=False):
        g = g.set_index("tdate").sort_index()
        # CENTRED, so the reference is a median over the days either side rather than a
        # blend of two one-sided medians -- which on a short history is an average of two
        # numbers and lets a single bad row drag the reference a third of the way to itself.
        # Looking forward is legitimate here and only here: the consensus is a DATA-QUALITY
        # reference, never a feature, it never reaches the panel, and the transaction it
        # repairs is stamped on a filing date later than every price behind the median.
        med = g["m"].rolling(window, center=True).median()
        out.append(pd.DataFrame({"ticker": tkr, "klass": cls, "tdate": g.index,
                                 "consensus": med.to_numpy()}))
    ref = pd.concat(out, ignore_index=True)

    keyed = pd.DataFrame({"ticker": txns["ticker"].astype(str), "klass": klass, "tdate": tdate})
    merged = keyed.merge(ref, on=["ticker", "klass", "tdate"], how="left")
    merged.index = txns.index
    return merged["consensus"]


def clean_transactions(insider: pd.DataFrame, *,
                       price_tolerance: float = PRICE_TOLERANCE) -> tuple[pd.DataFrame, dict]:
    """Scope to priced common-stock open-market trades and repair the mispriced ones.

    Returns `(frame, diagnostics)`. The frame carries the source columns plus:

        ``code``          upper-cased `transaction_code`
        ``day``           `filing_date`, normalised -- the POINT-IN-TIME stamp, never
                          `transaction_date`, which is 1-2 business days earlier
        ``value``         repaired transaction value in USD
        ``shares_n``      numeric `shares`
        ``price_repaired`` True where `value` came from `shares x consensus`
        ``role``          CEO / CFO / COO_or_President / other_named_officer
        ``in_exercise_package`` True for an `S` sharing an accession + transaction date with
                          an `M`: an exercise-and-sell is not a discretionary decision to sell

    Three exclusions, each measured in the module docstring or below:

      1. derivative rows (718 of 35,888 `P` rows) -- an option or convertible note is not a
         common-share purchase and its "price" is a strike or a par value;
      2. non-common securities (0.9% of non-derivative `P`);
      3. rows with no usable price (137 non-derivative `P` rows, 0.39%) -- an open-market
         purchase whose price was never reported cannot be valued, and the largest of them is
         4.5bn GOOGL shares at $0.00 from a filer reporting 4.49bn shares owned against a
         ~690m Class A float. Repairing those to `shares x consensus` would mint a $3.6tn
         purchase, so they are dropped on both the value AND the share legs, not zero-filled.
    """
    need = {"ticker", "filing_date", "transaction_code", "shares", "value_usd"}
    diag: dict = {"input_rows": 0 if insider is None else len(insider)}
    if insider is None or insider.empty or not need.issubset(insider.columns):
        return pd.DataFrame(), diag

    t = insider.copy()
    t["ticker"] = t["ticker"].astype(str).str.upper().str.strip()
    t["code"] = t["transaction_code"].astype(str).str.upper().str.strip()
    t["day"] = pd.to_datetime(t["filing_date"], errors="coerce").dt.normalize()
    t["shares_n"] = pd.to_numeric(t["shares"], errors="coerce")
    pps = pd.to_numeric(t.get("price_per_share"), errors="coerce")

    # The exercise-and-sell link is computed BEFORE the scope cut, because the `M` leg of the
    # package is a derivative row and the cut would remove the very evidence of the package.
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
    diag["dropped_unpriced_shares"] = float(t.loc[~priced, "shares_n"].sum())
    t, pps = t[priced], pps[priced]

    ref = consensus_price(insider).reindex(t.index)
    ratio = pps / ref.where(ref > 0)
    # ⚠ THE REPAIR IS ONE-SIDED, AND THAT IS THE WHOLE POINT. Only a price ABOVE the
    # consensus is corrected. Repairing the other direction is what a first version did, and
    # it turned nine genuine ~$25m AXON purchases into $326m ones: `AXON` carries Axovant
    # Sciences rows (the symbol-first extraction defect in the module docstring), whose real
    # ~$1.75 price reads as 0.05x against Axon Enterprise's ~$33 consensus. A too-low price
    # understates a flow; a too-high one invented $182,982,720tn. The asymmetry in the harm
    # is the asymmetry in the rule, and the low side is counted rather than touched.
    bad = ratio.notna() & (ratio > price_tolerance)
    low = ratio.notna() & (ratio < 1.0 / price_tolerance)
    raw_value = pd.to_numeric(t["value_usd"], errors="coerce")
    # `shares x consensus` rather than a drop: the trade happened and its size is filed; only
    # the price is wrong, and the consensus is other filers' own prices for the same days.
    t["value"] = raw_value.where(~bad, t["shares_n"] * ref)
    t["price_repaired"] = bad
    # A row whose price survived but whose `value_usd` is missing is still a real trade.
    t["value"] = t["value"].fillna(t["shares_n"] * pps)

    diag["repaired_rows"] = int(bad.sum())
    diag["underpriced_rows"] = int(low.sum())
    diag["value_before"] = float(raw_value.sum())
    diag["value_after"] = float(t["value"].sum())
    diag["no_consensus_rows"] = int(ratio.isna().sum())

    t["role"] = (t["officer_title"].map(officer_role) if "officer_title" in t.columns
                 else OTHER_OFFICER)
    for flag in ("is_director", "is_officer", "is_ten_pct_owner"):
        t[flag] = pd.to_numeric(t.get(flag), errors="coerce")
    t["is_10b5_1"] = pd.to_numeric(t.get("is_10b5_1"), errors="coerce")

    _log.info("insider: %s rows -> %s scoped, %s unpriced dropped, %s overpriced repaired, "
              "%s underpriced left as filed ($%.3ftn -> $%.3fbn)", diag["input_rows"],
              diag["scoped_rows"], diag["dropped_unpriced"], diag["repaired_rows"],
              diag["underpriced_rows"], diag["value_before"] / 1e12,
              diag["value_after"] / 1e9)
    return t, diag


def _exercise_packages(t: pd.DataFrame) -> pd.Series:
    """True for an `S` row filed in the same accession, on the same transaction date, as an
    option exercise `M`.

    Measured: **225,021 of 642,802 `S` rows (35.0%)** sit in such a package. Selling the
    shares an option just delivered is a mechanical cash-settlement of compensation, not a
    view on the stock, so `ic_insider_discretionary_sell_*` excludes them. It is a large
    enough share of the sell tape that leaving it in would define "discretionary" as mostly
    non-discretionary.
    """
    if "accession_number" not in t.columns or "transaction_date" not in t.columns:
        return pd.Series(False, index=t.index)
    key = pd.MultiIndex.from_arrays(
        [t["accession_number"].astype(str),
         pd.to_datetime(t["transaction_date"], errors="coerce")])
    has_m = pd.Series(t["code"].eq("M").to_numpy(), index=key).groupby(level=[0, 1]).any()
    return pd.Series(key.map(has_m).to_numpy(), index=t.index).fillna(False) & t["code"].eq("S")


def asof_values(frame: pd.DataFrame | None, tickers: pd.Series,
                days: pd.Series) -> pd.Series:
    """Value of a wide (date x ticker) `frame` as of each `(ticker, day)`, forward-filled.

    `searchsorted(side="right") - 1` takes the last row at or BEFORE the day, so a
    transaction filed before the ticker's first observation gets NaN rather than the first
    future value -- the difference between "unknown" and a look-ahead.
    """
    idx = pd.RangeIndex(len(tickers)) if tickers.index.has_duplicates else tickers.index
    if frame is None or frame.empty:
        return pd.Series(np.nan, index=idx, dtype="float64")
    f = frame.ffill()
    pos = f.index.searchsorted(pd.to_datetime(days).to_numpy(), side="right") - 1
    known = (pos >= 0) & tickers.isin(f.columns).to_numpy()
    out = np.full(len(tickers), np.nan)
    if known.any():
        out[known] = f.to_numpy()[pos[known], f.columns.get_indexer(tickers[known])]
    return pd.Series(out, index=idx, dtype="float64")


def report_oversized(t: pd.DataFrame, shares_outstanding: pd.DataFrame | None,
                     *, threshold: float = FLAG_PCT_SHARES_OUTSTANDING) -> pd.DataFrame:
    """Transactions above `threshold` of the company, as a frame to LOG rather than drop.

    Empty when no share count is available. See the module docstring: the largest genuine
    disposal (40.6% of shares outstanding) is bigger than the largest fabricated purchase
    (41.2%), so this is a visibility device, not a filter.
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
