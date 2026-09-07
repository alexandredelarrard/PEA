"""
capital.py  (src/data_aggregate/utils/capital.py)
------------------------------------------------
ONE definition of debt, net debt and invested capital, shared by the row-level KPI layer
(`sector_features`) and the daily wide-frame layer (`fundamental_features`).

There used to be two, and they disagreed:

  * `sector_features.net_debt_to_ebitda` = LTD + STD + operating leases + commercial paper
    - cash - ST investments. `CommercialPaper` is ALSO the fourth candidate inside
    `shortTermDebt`, so it was DOUBLE COUNTED whenever it won there; finance leases and the
    pension deficit were missing.
  * `fundamental_features.net_debt_incl_offbs_to_ebitda` = LTD + STD + both lease legs +
    pension deficit - cash. (That FEATURE was deleted 2026-09-05 -- three of its four
    distinguishing legs have no column on the Sharadar substrate, so it became identical to
    `net_debt_to_ebitda`. The off-balance-sheet DEFINITION lives on here, reachable via
    `net_debt(off_balance_sheet=True)`, and is still the right one for a caller that has
    lease or ARO data.)

And the return metrics used a THIRD basis: invested capital was equity + LTD + STD - cash,
excluding leases entirely, while EV and the leverage ratios treated leases as debt. Under
ASC 842 an operating lease liability IS debt-like, so it belongs on the financing side of
invested capital too (the offsetting ROU asset is the OPERATING-side twin, which is why
`totalAssetsExLease` -- not the ROU asset -- is what the asset-turnover ratios use).

Both call sites pass a `get(field) -> Series | DataFrame` accessor (`sector_features._col`
via `g`, `fundamental_features`'s memoized `daily`), so one implementation serves a
row-level Series and a date x ticker frame: every operation here is NaN-tolerant pandas
arithmetic valid for both shapes.

Layering, from narrowest to widest:
    borrowings(get)        interest-bearing debt only (no leases) -- the `totalDebt` column
    total_debt(get)        borrowings + capitalized leases          <- the leverage default
    net_debt(get)          total_debt - non-operating liquid assets
    net_debt(off_bs=True)  + pension/OPEB deficit + asset-retirement obligations
    liquid_assets(get)     cash + current marketable securities     <- NOT + ST investments,
                                                                       `cash` already holds them
    invested_capital(get)  equity + total_debt - cash               <- the ROIC default

Every site that NETS cash takes `operating_cash`, a set of tickers whose cash is a core
operating asset rather than spare cash (bank reserves, insurance float). One judgement,
applied at all three netting sites, so EV cannot say "not spare cash" while invested
capital says the opposite. `cash_to_debt` and the other cash RATIOS deliberately do not
take it -- a liquidity cushion is a fact about a bank too; only the netting is wrong.

Also here, for the same reason (two layers, one definition, a sign that is easy to get
backwards): `share_repurchases(get)`, which turns Sharadar's NET-ISSUANCE cash-flow line
into the buyback magnitude the payout ratios want.
"""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.pit import FieldGetter

__all__ = ["borrowings", "capitalized_leases", "liquid_assets", "total_debt",
           "net_debt", "invested_capital", "off_balance_sheet_obligations",
           "assets_ex_lease", "share_repurchases", "drop_operating_cash"]

# accessor: field name -> numeric Series (row-level) or date x ticker frame (daily).
# Was the string literal `Getter = "callable"`, which described the protocol in a comment
# instead of expressing it; `FieldGetter` is the real thing (see utils/common/pit.py).
Getter = FieldGetter


def _has(x) -> bool:
    """True when `x` is a non-empty Series/frame holding at least one value. Shape-agnostic:
    `Series.notna().any()` is a scalar while `DataFrame.notna().any()` is a Series."""
    if x is None or x.empty:
        return False
    return bool(x.notna().to_numpy().any())


def _add(*parts) -> pd.Series | pd.DataFrame | None:
    """NaN-tolerant sum: a missing/empty part contributes 0, but the result is NaN where
    EVERY part is NaN (so 'no data' never silently becomes 0).

    The `known` mask is realigned to the running total at each step. Without that, OR-ing
    masks whose COLUMNS differ introduces NaN for the tickers present in one part and not
    the other, and `where` then drops them: a name reporting long-term debt but no
    short-term debt lost its debt entirely, taking it out of EV and invested capital."""
    present = [p for p in parts if p is not None and not p.empty]
    if not present:
        return None
    out = present[0].fillna(0.0)
    known = present[0].notna()
    for p in present[1:]:
        out = out.add(p.fillna(0.0), fill_value=0.0)
        known = (known.reindex_like(out).fillna(False).astype(bool)
                 | p.notna().reindex_like(out).fillna(False).astype(bool))
    return out.where(known)


def drop_operating_cash(frame, tickers: frozenset[str] | None):
    """`frame` with the named tickers' columns set to NaN -- the shape-agnostic way to say
    "this quantity does not mean what the caller wants it to mean for these names".

    NaN rather than 0 because every netting site here already treats a missing liquid
    balance as "net nothing" (`gross.sub(liquid.fillna(0.0))`), so the two agree; a literal
    0 would additionally claim the firm holds no cash, which is a different and false
    statement. A row-level Series has no ticker axis and is returned untouched."""
    if frame is None or not tickers or not isinstance(frame, pd.DataFrame):
        return frame
    hit = [c for c in frame.columns if c in tickers]
    if not hit:
        return frame
    out = frame.copy()
    out[hit] = float("nan")
    return out


def assets_ex_lease(get):
    """Total assets free of the ASC-842 operating-lease ROU asset — the base every
    assets-denominated ratio uses (asset growth, asset turnover, gross profitability,
    accruals, Altman Z, Beneish, acquisition intensity).

    Adopting ASC 842 in FY2019 put the ROU asset on the balance sheet, so `totalAssets`
    steps up once for every lease-heavy filer with no change in the business.

    Resolved in three steps so it works on ANY history vintage: the extractor's precomputed
    column, else derived here as totalAssets - ROU asset, else plain totalAssets. Without the
    fallback, a `fundamentals_history` built before the column existed would silently return
    NaN for every ratio above."""
    precomputed = get("totalAssetsExLease")
    if _has(precomputed):
        return precomputed
    raw = get("totalAssets")
    if not _has(raw):
        return raw
    rou = get("operatingLeaseRouAsset")
    return raw.sub(rou.fillna(0.0), fill_value=0.0) if _has(rou) else raw


def share_repurchases(get) -> pd.Series | pd.DataFrame | None:
    """Gross-of-issuance BUYBACK MAGNITUDE, positive, from Sharadar's `equityIssuanceNet`.

    ⚠ THE SOURCE COLUMN IS NET ISSUANCE, NOT BUYBACKS, and the sign is the whole reason this
    helper exists. `ncfcommon` is "issuance (repurchase) of equity": NEGATIVE when the firm
    bought back more stock than it issued, positive when it raised equity. Measured on the
    live table, 30,741 rows are negative and 15,957 positive; AAPL 2026-07-31 reads
    -$82.2bn, which is a repurchase.

    Three call sites want "how much stock did they buy back", a non-negative magnitude:
    `payout_ratio`'s buyback leg, `buyback_intensity`, and `sbc_to_buyback`. Taking `.abs()`
    would map a $5bn EQUITY RAISE onto "$5bn of buybacks" -- the opposite signal -- so the
    net-issuing side is floored to 0 instead: a firm that issued on net repurchased nothing.

    This is NET of issuance, so it understates gross repurchases for a firm that does both
    in the same year (a serial acquirer paying in stock). It never overstates them, which is
    the direction that would fabricate a payout the firm did not make."""
    net_issuance = get("equityIssuanceNet")
    if not _has(net_issuance):
        return None
    return (-net_issuance).clip(lower=0.0)


def borrowings(get) -> pd.Series | pd.DataFrame | None:
    """Interest-bearing borrowings, EXCLUDING capitalized leases. Prefers the extractor's
    reconciled `totalDebt` (which already resolves the combined ST+LT tag, the two-leg sum
    and the REIT notes-payable fallback, and distinguishes zero debt from unknown debt);
    falls back to long-term + short-term for callers whose history predates it.

    `commercialPaper` is deliberately NOT added: it is one of the `shortTermDebt`
    candidates, so adding it again double counts the same paper."""
    total = get("totalDebt")
    if _has(total):
        return total
    return _add(get("longTermDebt"), get("shortTermDebt"))


def capitalized_leases(get) -> pd.Series | pd.DataFrame | None:
    """Operating + finance lease liabilities. Both are reconstructed by the extractor from
    the combined element, else current + noncurrent, else the pre-2019 capital-lease legs,
    so this covers all three ASC-842 eras."""
    return _add(get("operatingLeaseLiability"), get("financeLeaseLiability"))


def liquid_assets(get, *, operating_cash: frozenset[str] | None = None):
    """NON-OPERATING liquid assets netted against debt: unrestricted cash + current
    marketable securities.

    ⚠ SHORT-TERM INVESTMENTS ARE NOT A SEPARATE LEG, because `cash` ALREADY CONTAINS THEM.
    The field map defines `cash = cashAndEquivalents + coalesce(shortTermInvestments, 0)`
    and maps `shortTermInvestments` straight from `investmentsc`, so adding it again
    subtracted the same money twice from EV, net debt and invested capital. The docstring
    used to claim `cash` was "investment-free (the extractor nets the broader totals
    down)": true of the SEC-era extractor, FALSE on the Sharadar substrate. Magnitude of
    the double count, `investmentsc`/market cap at filing grain: ~0 median in nine sectors
    but 1.30% in Information Technology (p90 13.36%) -- so it understated EV most for
    exactly the cash-rich tech names where the EV yields matter.

    The AFS/HTM investment BOOK (`investmentSecurities`) is excluded on purpose -- for a
    bank or insurer that is the core operating asset, not spare cash.

    `operating_cash` extends that same judgement to `cash` itself for the tickers named:
    "cash and due from banks" is required reserves and interbank float, and insurance cash
    is claims float. Netting it would misstate the capital structure, and the measurement
    says by how much -- median `cashneq` is 10.97% of market cap for Financials, p90
    101.95%, so for the top decile the "cash" exceeds the entire equity value and EV would
    go NEGATIVE. Blanked to NaN rather than 0 so callers cannot distinguish it from "no
    data", which is what the netting sites already handle. Requires the date x ticker
    shape (GICS membership is per ticker); a row-level Series has no ticker axis, so the
    set is ignored there -- no such caller exists today (`sector_features` uses only
    `assets_ex_lease`, `total_debt` and `share_repurchases`, none of which net cash)."""
    liquid = _add(get("cash"), get("marketableSecuritiesCurrent"))
    return drop_operating_cash(liquid, operating_cash)


def off_balance_sheet_obligations(get, pension: pd.DataFrame | None = None):
    """Debt-like obligations outside borrowings and leases: the underfunded pension/OPEB
    deficit and asset-retirement (decommissioning) obligations.

    The deficit arrives ONLY as the caller's already-coalesced `pension` frame, built from
    the bulk SEC data sets. There is no field-getter fallback: this used to read a
    `pensionDeficit` column, but `fundamentals_history` has never carried one on the
    Sharadar-first schema (an `information_schema` match on `%pension%`/`%opeb%`/`%benefit%`
    returns nothing), so that leg contributed exactly zero on every live row."""
    deficit = pension
    if _has(deficit):
        deficit = deficit.clip(lower=0.0)            # underfunding only
    return _add(deficit, get("assetRetirementObligation"))


def total_debt(get, *, include_leases: bool = True):
    """Total debt claim: borrowings plus capitalized leases (the leverage default)."""
    if not include_leases:
        return borrowings(get)
    return _add(borrowings(get), capitalized_leases(get))


def net_debt(get, *, include_leases: bool = True, off_balance_sheet: bool = False,
             pension: pd.DataFrame | None = None,
             operating_cash: frozenset[str] | None = None):
    """Total debt (optionally + off-balance-sheet obligations) minus non-operating liquid
    assets. `off_balance_sheet=True` adds the pension deficit and ARO. `operating_cash`
    names tickers whose cash is not spare cash -- see `liquid_assets`."""
    gross = total_debt(get, include_leases=include_leases)
    if off_balance_sheet:
        gross = _add(gross, off_balance_sheet_obligations(get, pension))
    liquid = liquid_assets(get, operating_cash=operating_cash)
    if gross is None:
        return None
    return gross if liquid is None else gross.sub(liquid.fillna(0.0), fill_value=0.0)


def invested_capital(get, *, include_leases: bool = True,
                     operating_cash: frozenset[str] | None = None):
    """Financing-side invested capital = equity + total debt (incl. leases) - cash.

    Leases are included because they are counted as debt everywhere else (EV, leverage);
    excluding them here understated the capital base of every lease-heavy business
    (retail, restaurants, airlines) and so overstated its ROIC.

    `operating_cash` is honoured for the same reason it is in `liquid_assets`, and it must
    be: deducting restored bank cash here would SHRINK the capital base and inflate the
    ROIC of every bank, which is the same silent re-rating the EV gate exists to prevent.
    One judgement about what a bank's cash is, applied at every site that nets it."""
    equity = get("stockholdersEquity")
    if not _has(equity):
        return None
    ic = _add(equity, total_debt(get, include_leases=include_leases))
    cash = drop_operating_cash(get("cash"), operating_cash)
    if ic is not None and _has(cash):
        ic = ic.sub(cash.fillna(0.0), fill_value=0.0)
    return ic
