"""
value_basis.py  (src/data_aggregate/utils/institutionals/value_basis.py)
------------------------------------------------------------------------
Put every 13F filing's reported VALUE onto one basis: dollars.

A 13F reports value in $thousands or in dollars at the FILER's discretion. edgartools infers
the unit per filing and hands `infotable` back in dollars, but it infers WRONG on 4.53% of
filings -- and the error is a clean factor of 1000, in both directions. This module measures
the error against the market and corrects it, per filing, or abstains.

⚠ THE DEFECT IS NOT A TAIL, IT IS THE MAJORITY OF THE DOLLARS. Measured on period 2020-12-31:
61 unit-defective filers carry 90.37% of that quarter's raw `value_usd` total against 5,246
correct filers carrying 9.54%. Any value-weighted statement about that quarter -- concentration,
completeness, flow -- is a statement about those 61 filings until this runs. That is also why
winsorizing was rejected: `value_usd` p99 is $1,482,270,606, a legitimate large holding, so a
p99 clip destroys real data while leaving ~1.1% of the 1000x rows standing.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: The per-filing repair band. A filing whose median implied price sits inside one of these is
#: rescaled by that factor; anything else ABSTAINS (value nulled, flag set), because a factor
#: that is not a clean power of 1000 is not a units error and guessing at it would invent a
#: number. The bands are deliberately WIDE (200-5000, not 900-1100): the median is taken over a
#: manager's whole S&P 500 slice on one quarter-end, so real intra-quarter price dispersion and
#: a few mis-shared rows move it a long way without changing which power of 1000 it is.
#: Measured 2026-09-14 -- the gap between the bands is empty in the data (0.085% of rows sit in
#: 2-200 and 0.046% in 0.005-0.5, and those ABSTAIN).
_REPAIR_BANDS = ((0.5, 1.5, 1.0), (900.0, 1100.0, 1e-3), (0.0009, 0.0011, 1e3))

#: `value_basis_repaired`. Per D2's flag-and-widen shape, a repaired or abstaining row is
#: FLAGGED, never dropped: `shares` is unaffected by a value-unit error, so the share-based
#: features (`ic_inst_holders`, `shares_chg`, `cluster_buying`, `ownership_pct`) keep the filer
#: either way.
KEPT, DIVIDED, MULTIPLIED, ABSTAINED = 0, 1, -1, 9

#: Every VALUE leg of a filing, because the unit is declared once for the whole info table.
#: ⚠ DETECTION READS ONLY `value_usd`, BUT THE CORRECTION APPLIES TO ALL OF THESE. The implied
#: price needs a share count and a market price, which only the common-stock leg has; the
#: $thousands declaration it uncovers governs the option, debt and residual legs of the same
#: filing just as much. Repairing `value_usd` alone would leave `ic_inst_net_options_ratio`
#: comparing a dollar numerator against a thousands denominator.
_VALUE_COLUMNS = ("value_usd", "call_value", "put_value", "debt_value", "other_value")


def _period_close(close_split: pd.DataFrame, periods: pd.Series) -> pd.Series:
    """The last close at or before each period end, as a `(period, ticker)` Series.

    As-of, never exact: a quarter end is 2024-03-31, a Sunday, and an exact lookup on a
    trading calendar returns NaN for every ticker on the most common period ends there are.
    """
    wanted = pd.DatetimeIndex(sorted(pd.Series(periods).dropna().unique()))
    if wanted.empty:
        return pd.Series(dtype=float)
    asof = close_split.reindex(close_split.index.union(wanted)).ffill().reindex(wanted)
    out = asof.stack(future_stack=True)
    out.index = out.index.set_names(["period", "ticker"])
    return out


def repair_value_basis(
    holdings: pd.DataFrame,
    close_split: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rescale reported value per FILING onto dollars, and return the factor register.

    ⚠ THE UNIT ERROR IS PER-FILING, NOT PER-TABLE AND NOT PER-ROW. A 13F reports value in
    $thousands or in dollars at the FILER's discretion, and edgartools' per-filing detection
    fails on 4.53% of filings -- so a table-wide rescale is wrong on 95% of rows and a row-wise
    one has no evidence to work from. The filing is the unit of the decision because it is the
    unit of the declaration.

    ⚠ THE EVIDENCE IS THE FILED PRICE AGAINST THE MARKET PRICE, never an internal identity.
    `value_usd / shares` is the price the filer implies; `close_split` at the period end is the
    price that was. Their ratio is a measurement, and the repair only fires where that
    measurement lands on a clean power of 1000. `close_split` is the LEVEL basis and is the
    right one here -- `close_total` carries dividend reinvestment, which a filer's reported
    position value does not.

    ⚠ IT IS THE MEDIAN OVER THE FILING, NOT ANY SINGLE ROW. One mis-reported share count moves
    one row's implied price arbitrarily far; it cannot move the median of a manager's whole
    S&P 500 slice. This is why a row-wise repair was rejected even though the flag is per row.

    Applies to BOTH holdings tables. Measured 2026-09-14 on `sec13f_manager_holdings`: the
    defect is present there too but one-directional and 14x smaller -- 43 filings / 0.33% of
    rows in the divide-by-1000 band, and the multiply band is EMPTY (not one filing below 0.5,
    against 6,581 on `sec13f_hr`). So `ic_super_conviction` / `ic_super_flow_to_mcap` need this
    as well, just far less of it.

    Returns `(repaired, register)`. `register` is one row per `(cik, period)` with the measured
    ratio, the factor applied, the row count and the value moved -- the target for a later
    extraction-side fix, and the audit trail for this one.
    """

    out = holdings.copy()
    value_columns = [c for c in _VALUE_COLUMNS if c in out.columns]
    if close_split is None or not value_columns or out.empty:
        logger.warning(
            "13F value basis: no close_split or no value column -> repair SKIPPED, " "every row flagged kept and `value_usd` left as filed"
        )
        out["value_basis_repaired"] = KEPT
        return out, _empty_register()

    shares = pd.to_numeric(out["shares"], errors="coerce")
    close = (
        pd.Series(out.set_index(["period", "ticker"]).index.map(_period_close(close_split, out["period"])), index=out.index)
        .replace(0.0, pd.NA)
        .astype(float)
    )

    # `value_usd == shares` is a filer field swap -- the share count landed in the value column,
    # so the implied price is $1.00 and the bands below would read it as a 1000x unit error.
    value = pd.to_numeric(out["value_usd"], errors="coerce")
    swapped = (value == shares) & shares.gt(0) & close.notna()
    if swapped.any():
        before = float(value[swapped].sum())
        value = value.mask(swapped, shares * close)
        out["value_usd"] = value
        logger.warning(
            "13F value basis: %s row(s) filed `value_usd == shares` -> restated at " "the period-end close, $%.3e -> $%.3e",
            f"{int(swapped.sum()):,}",
            before,
            float(value[swapped].sum()),
        )

    # the implied price, on the common-stock leg only -- the one leg with both a share count
    # and a market price to compare against
    implied = value.where(value > 0) / shares.where(shares > 0)  # call and put
    ratio = implied / close

    measured = ratio.groupby([out["cik"], out["period"]]).median().rename("median_ratio")
    factor = pd.Series(pd.NA, index=measured.index, dtype="Float64")
    for low, high, mult in _REPAIR_BANDS:
        factor = factor.mask(measured.between(low, high, inclusive="left"), mult)

    # `close_split` back-adjusts for splits and spinoffs, which a 13F never does, so a ratio on a
    # whole integer (4, 20, 28, 200) is that gap, not a unit error. Strip it, then band the rest:
    # smallest split factor wins, so 4000 reads as $thousands x 4-for-1 and not as a 4000x split.
    for unit in (1e-3, 1.0, 1e3):
        residual = measured * unit
        k = residual.round()
        hit = (factor.isna() & k.between(2, 500) & (residual - k).abs().le(0.01)).fillna(False)
        if hit.any():
            logger.info(
                "13F value basis: %s filing(s) sit on a whole-integer median ratio " "(split/spinoff basis) -> factor %.0e instead of nulled",
                f"{int(hit.sum()):,}",
                unit,
            )
        factor = factor.mask(hit, unit)

    key = pd.MultiIndex.from_arrays([out["cik"], out["period"]])
    row_factor = pd.Series(factor.reindex(key).to_numpy(), index=out.index, dtype="Float64")

    # ⚠ `.fillna(False)` ON EVERY COMPARISON, because an abstaining filing's factor is `pd.NA`
    # and `pd.NA == 1e3` is `pd.NA`, not False -- which `mask` then honoured as a hit. A filing
    # at ratio 50 was flagged MULTIPLIED and had its value scaled by 1000 on no evidence at all,
    # which is the exact failure abstention exists to prevent.
    flag = pd.Series(ABSTAINED, index=out.index, dtype=int)
    for applied, label in ((1.0, KEPT), (1e-3, DIVIDED), (1e3, MULTIPLIED)):
        flag = flag.mask((row_factor == applied).fillna(False), label)
    out["value_basis_repaired"] = flag

    scale = row_factor.astype(float)
    for col in value_columns:
        scaled = pd.to_numeric(out[col], errors="coerce") * scale
        # ⚠ ABSTENTION IS NaN, NOT 0. `clean_holdings` zero-fills the value legs, so a nulled
        # abstention that fell through as 0.0 would read as a REAL zero holding and drag
        # `ic_inst_concentration` down instead of leaving the filing out of the numerator.
        out[col] = scaled.where(flag != ABSTAINED, other=pd.NA)

    return out, _register(out, measured, factor, value, flag)


def _empty_register() -> pd.DataFrame:
    return pd.DataFrame(columns=["cik", "period", "median_ratio", "factor", "rows", "value_before", "value_after"])


def _register(out: pd.DataFrame, measured: pd.Series, factor: pd.Series, value_before: pd.Series, flag: pd.Series) -> pd.DataFrame:
    """One row per filing: what was measured, what was done, and how much value moved."""
    grouped = pd.DataFrame(
        {
            "rows": value_before.groupby([out["cik"], out["period"]]).size(),
            "value_before": value_before.groupby([out["cik"], out["period"]]).sum(min_count=1),
            "value_after": pd.to_numeric(out["value_usd"], errors="coerce").groupby([out["cik"], out["period"]]).sum(min_count=1),
        }
    )
    register = pd.concat([measured, factor.rename("factor"), grouped], axis=1).reset_index()
    register.columns = ["cik", "period", *register.columns[2:]]
    return register


def log_register(register: pd.DataFrame, value_before_total: float, log=logger) -> None:
    """Log the repair summary every build, in the shape `_report_late_filings` uses.

    The SHARE OF TOTAL VALUE MOVED is the number that matters and the reason this is logged
    rather than merely returned: a repair touching 4.5% of rows can move the majority of the
    table's dollars, and a row count alone would report that as a rounding error.
    """
    if register.empty:
        log.info("13F value basis: nothing to repair")
        return
    by = {DIVIDED: 1e-3, MULTIPLIED: 1e3}
    for name, mult in (("divided by 1000", by[DIVIDED]), ("multiplied by 1000", by[MULTIPLIED])):
        hit = register[register["factor"] == mult]
        if not hit.empty:
            moved = float(hit["value_before"].sum())
            log.warning(
                "13F value basis: %s filing(s) %s -- %s row(s), $%.3e of filed value " "(%.2f%% of the table's total)",
                f"{len(hit):,}",
                name,
                f"{int(hit['rows'].sum()):,}",
                moved,
                100.0 * moved / value_before_total if value_before_total else 0.0,
            )
    abstained = register[register["factor"].isna()]
    if not abstained.empty:
        log.warning(
            "13F value basis: %s filing(s) ABSTAINED (median implied price is not a "
            "clean power of 1000 against the market) -- %s row(s), value nulled",
            f"{len(abstained):,}",
            f"{int(abstained['rows'].sum()):,}",
        )
