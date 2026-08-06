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
    pension deficit - cash.

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
    liquid_assets(get)     cash + short-term investments + current marketable securities
    invested_capital(get)  equity + total_debt - cash               <- the ROIC default
"""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.pit import FieldGetter

__all__ = ["borrowings", "capitalized_leases", "liquid_assets", "total_debt",
           "net_debt", "invested_capital", "off_balance_sheet_obligations",
           "assets_ex_lease"]

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


def liquid_assets(get) -> pd.Series | pd.DataFrame | None:
    """NON-OPERATING liquid assets netted against debt: unrestricted cash + short-term
    investments + current marketable securities. `cash` is already restricted-cash-free and
    investment-free (the extractor nets the broader totals down), so nothing is
    double-counted here. The AFS/HTM investment BOOK (`investmentSecurities`) is excluded on
    purpose -- for a bank or insurer that is the core operating asset, not spare cash."""
    return _add(get("cash"), get("shortTermInvestments"), get("marketableSecuritiesCurrent"))


def off_balance_sheet_obligations(get, pension: pd.DataFrame | None = None):
    """Debt-like obligations outside borrowings and leases: the underfunded pension/OPEB
    deficit and asset-retirement (decommissioning) obligations. `pension` lets a caller
    pass an already-coalesced deficit built from the bulk SEC data sets, which is
    universe-wide and preferred over the single companyfacts tag."""
    deficit = pension if _has(pension) else get("pensionDeficit")
    if _has(deficit):
        deficit = deficit.clip(lower=0.0)            # underfunding only
    return _add(deficit, get("assetRetirementObligation"))


def total_debt(get, *, include_leases: bool = True):
    """Total debt claim: borrowings plus capitalized leases (the leverage default)."""
    if not include_leases:
        return borrowings(get)
    return _add(borrowings(get), capitalized_leases(get))


def net_debt(get, *, include_leases: bool = True, off_balance_sheet: bool = False,
             pension: pd.DataFrame | None = None):
    """Total debt (optionally + off-balance-sheet obligations) minus non-operating liquid
    assets. `off_balance_sheet=True` adds the pension deficit and ARO."""
    gross = total_debt(get, include_leases=include_leases)
    if off_balance_sheet:
        gross = _add(gross, off_balance_sheet_obligations(get, pension))
    liquid = liquid_assets(get)
    if gross is None:
        return None
    return gross if liquid is None else gross.sub(liquid.fillna(0.0), fill_value=0.0)


def invested_capital(get, *, include_leases: bool = True):
    """Financing-side invested capital = equity + total debt (incl. leases) - cash.

    Leases are included because they are counted as debt everywhere else (EV, leverage);
    excluding them here understated the capital base of every lease-heavy business
    (retail, restaurants, airlines) and so overstated its ROIC."""
    equity = get("stockholdersEquity")
    if not _has(equity):
        return None
    ic = _add(equity, total_debt(get, include_leases=include_leases))
    cash = get("cash")
    if ic is not None and _has(cash):
        ic = ic.sub(cash.fillna(0.0), fill_value=0.0)
    return ic
