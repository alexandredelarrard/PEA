"""Validate 13F VALUE units per filing against report-quarter-end as-traded prices.

Source tables remain untouched. One supported scale moves every monetary leg;
uncertain common and option values use tagged underlying-share market estimates.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import SEC_13F_VALUE_FACTORS
from src.data_aggregate.utils.common.level_basis import suffix_factor
from src.data_aggregate.utils.common.pit import fundamentals_to_daily

logger = logging.getLogger(__name__)
KEPT, DIVIDED, MULTIPLIED, ABSTAINED = 0, 1, -1, 9
_VALUE_COLUMNS = ("value_usd", "call_value", "put_value", "debt_value", "other_value")


def as_traded_close(
    close_split: pd.DataFrame | None,
    level_factor: pd.DataFrame | None,
    split_events: pd.DataFrame,
) -> pd.DataFrame | None:
    """Undo the repaired quote's level adjustment and genuine future share splits."""
    if close_split is None or level_factor is None:
        return None
    df_close = close_split * level_factor.reindex_like(close_split)
    factors = suffix_factor(split_events, list(df_close.columns), df_close.index.to_numpy(dtype="datetime64[ns]"))
    for ticker, factor in factors.items():
        df_close[ticker] *= factor
    return df_close.where(np.isfinite(df_close) & df_close.gt(0))


def _period_close(close: pd.DataFrame, periods: pd.Series, max_age_days: int) -> pd.Series:
    """Last observed valuation-day quote, bounded by age and never from the future."""
    wanted = pd.DatetimeIndex(sorted(periods.dropna().unique()))
    df_quotes = close.sort_index().where(np.isfinite(close) & close.gt(0))
    union = df_quotes.index.union(wanted)
    df_asof = df_quotes.reindex(union).ffill().reindex(wanted)
    for ticker in df_quotes:
        observed = pd.Series(df_quotes.index, index=df_quotes.index).where(df_quotes[ticker].notna())
        last = observed.reindex(union).ffill().reindex(wanted)
        df_asof[ticker] = df_asof[ticker].where((pd.Series(wanted, index=wanted) - last).le(pd.Timedelta(days=max_age_days)))
    result = cast(pd.Series, df_asof.stack(future_stack=True))
    result.index = result.index.set_names(["period", "ticker"])
    return result


def repair_value_basis(
    holdings: pd.DataFrame,
    close: pd.DataFrame | None,
    *,
    min_securities: int = 15,
    min_agreement: float = 0.8,
    price_tolerance: float = 0.1,
    max_price_age_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize filing values, reconstruct missing equity inputs and mark exclusions.

    'close' must be report-period as-traded prices. Only raw reported common
    quantities vote for units. Inverse common quantities are marked estimates;
    an unresolved factor leaves their dollar-unit assumption explicit.
    Options use their own underlying quantities; debt needs an instrument quote.
    Source frames remain unchanged and repeat repair is idempotent.
    """
    if min_securities < 1 or not 0.5 < min_agreement <= 1 or not 0 < price_tolerance < 0.5 or max_price_age_days < 0:
        raise ValueError("Invalid 13F value-basis support or price-age policy")
    df_out = holdings.copy()
    if "value_basis_repaired" in df_out:
        return df_out, _empty_register()
    if df_out.empty:
        return df_out, _empty_register()
    if "value_usd" not in df_out:
        df_out["value_usd"] = np.nan
    keys = ["cik", "period"]
    keys += [column for column in ("filing_date", "accession", "accession_number") if column in df_out]
    value_columns = [column for column in _VALUE_COLUMNS if column in df_out]
    value = pd.to_numeric(df_out["value_usd"], errors="coerce").astype("float64")
    shares = pd.to_numeric(df_out["shares"], errors="coerce").astype("float64")
    quote = pd.Series(np.nan, index=df_out.index, dtype=float)
    if close is not None and not close.empty and "ticker" in df_out:
        quotes = _period_close(close, df_out["period"], max_price_age_days)
        quote[:] = quotes.reindex(pd.MultiIndex.from_frame(df_out[["period", "ticker"]])).to_numpy()
    common = df_out["position_type"].astype(str).str.lower().eq("common") if "position_type" in df_out else pd.Series(True, index=df_out.index)
    swapped = common & value.eq(shares) & shares.gt(0)
    ratio = (value / shares.where(shares.gt(0)) / quote).where(value.gt(0) & ~swapped)
    if "position_type" in df_out:
        ratio = ratio.where(df_out["position_type"].astype(str).str.lower().eq("common"))
    ratio = ratio.where(np.isfinite(ratio) & ratio.gt(0))
    security = "cusip" if "cusip" in df_out else "ticker"
    df_evidence = df_out[keys + [security]].assign(_ratio=ratio)
    # One vote per security, irrespective of repeated source rows.
    evidence = df_evidence.groupby(keys + [security], dropna=False)["_ratio"].median()
    grouped = evidence.groupby(level=list(range(len(keys))), dropna=False)
    measured, count = grouped.median(), grouped.count()
    factor = pd.Series(np.nan, index=measured.index, dtype=float)
    agreement = pd.Series(0.0, index=measured.index)
    for candidate in SEC_13F_VALUE_FACTORS:
        support = (evidence * candidate).between(1 - price_tolerance, 1 + price_tolerance)
        fraction = support.groupby(level=list(range(len(keys))), dropna=False).sum() / count.where(count.gt(0))
        match = (measured * candidate).between(1 - price_tolerance, 1 + price_tolerance)
        match &= fraction.ge(min_agreement)
        if candidate != 1.0:
            match &= count.ge(min_securities)
        factor.loc[match] = candidate
        agreement = agreement.where(~match, fraction)
    status = pd.Series("abstained", index=factor.index)
    status.loc[factor.notna()] = "market_checked"
    thin = factor.eq(1.0) & count.lt(min_securities)
    status.loc[thin] = "source_only_thin"
    no_price = count.eq(0)
    factor.loc[no_price] = 1.0
    status.loc[no_price] = "source_only_no_price"
    df_register = (
        pd.concat(
            [
                measured.rename("median_ratio"),
                factor.rename("factor"),
                count.rename("priced_securities"),
                agreement.rename("agreement"),
                status.rename("status"),
            ],
            axis=1,
        )
        .reset_index()
        .assign(price_tolerance=price_tolerance)
    )
    df_rows = df_out[keys].merge(df_register[keys + ["factor", "status"]], on=keys, how="left", validate="many_to_one")
    scale = pd.Series(df_rows["factor"].to_numpy(), index=df_out.index)
    flag = pd.Series(ABSTAINED, index=df_out.index)
    for applied, label in ((1.0, KEPT), (0.001, DIVIDED), (1000.0, MULTIPLIED)):
        flag.loc[scale.eq(applied)] = label
    df_out["value_basis_repaired"] = flag
    df_out["value_basis_status"] = df_rows["status"].to_numpy()
    for column in value_columns:
        original = pd.to_numeric(df_out[column], errors="coerce").astype("float64")
        df_out[column] = (original * scale).mask(original.eq(0), 0.0)
    # Check individual common valuations only after the raw book unit is frozen.
    position_ratio = df_out["value_usd"] / shares.where(shares.gt(0)) / quote
    mismatch = common & np.isfinite(position_ratio) & position_ratio.gt(0)
    mismatch &= ~position_ratio.between(1 - price_tolerance, 1 + price_tolerance)
    df_out["value_basis_price_mismatch"] = mismatch
    _repair_common_quantities(df_out, value, shares, quote, scale, common)
    # 13F options report underlying-share exposure, so each leg uses its own quantity.
    # Debt principal is not an equity quantity and cannot use this stock quote.
    estimates: dict[str, pd.Series] = {}
    for column, quantity_column, label in (
        ("value_usd", "shares", ""),
        ("call_value", "call_shares", "call_"),
        ("put_value", "put_shares", "put_"),
    ):
        estimates[column] = pd.Series(False, index=df_out.index)
        if column not in df_out:
            continue
        raw = value if column == "value_usd" else pd.to_numeric(holdings[column], errors="coerce").astype("float64")
        unverified = df_out["value_basis_status"].eq("source_only_no_price") & (column != "value_usd")
        if quantity_column not in df_out:
            df_out[column] = df_out[column].mask(unverified & raw.ne(0))
            df_out[f"value_basis_{label}reconstructed"] = estimates[column]
            df_out[f"value_basis_{label}value_source"] = np.where(df_out[column].notna(), "filed", "unknown")
            continue
        quantity = pd.to_numeric(df_out[quantity_column], errors="coerce").astype("float64")
        eligible = common if column == "value_usd" else pd.Series(True, index=df_out.index)
        uncertain = unverified & (raw.ne(0) | quantity.gt(0))
        uncertain |= swapped if column == "value_usd" else raw.eq(quantity) & quantity.gt(0)
        if column == "value_usd":
            uncertain |= mismatch
        uncertain |= ~np.isfinite(df_out[column]) | df_out[column].lt(0)
        uncertain |= df_out[column].eq(0) & quantity.gt(0)
        market_value = quantity * quote
        market_value = market_value.where(eligible & quantity.gt(0) & np.isfinite(quantity) & np.isfinite(market_value))
        estimates[column] = eligible & uncertain & market_value.notna()
        # Estimates follow the raw common-holding unit decision and never vote for it.
        df_out[column] = df_out[column].mask(eligible & uncertain, market_value)
        df_out[f"value_basis_{label}reconstructed"] = estimates[column]
        df_out[f"value_basis_{label}value_source"] = np.where(
            estimates[column], "market_estimated", np.where(df_out[column].notna(), "filed", "unknown")
        )
    # Decide whole-row exclusions after unsupported sibling dollars have been nulled.
    sibling = pd.Series(False, index=df_out.index)
    for column in ("call_shares", "put_shares", "debt_prn", "call_value", "put_value", "debt_value", "other_value"):
        if column in df_out:
            exposure = pd.to_numeric(df_out[column], errors="coerce").astype("float64")
            sibling |= np.isfinite(exposure) & exposure.gt(0)
    df_out["value_basis_excluded"] &= ~sibling
    return df_out, _repair_register(df_register, keys, df_out, value, estimates, quote, common)


def _repair_common_quantities(
    df_out: pd.DataFrame,
    value: pd.Series,
    shares: pd.Series,
    quote: pd.Series,
    scale: pd.Series,
    common: pd.Series,
) -> None:
    """Infer absent common quantities once, after raw-only unit evidence is frozen."""
    absent = shares.eq(0) | shares.isna()
    dollars = value * scale.fillna(1.0)
    inferred = dollars / quote
    inverse = common & absent & value.gt(0) & np.isfinite(inferred) & inferred.gt(0)
    excluded = common & ~(np.isfinite(shares) & shares.gt(0)) & quote.isna() & (value.gt(0) | value.isna())
    df_out["value_basis_shares_reconstructed"] = inverse
    df_out["value_basis_inverse_unit_unverified"] = inverse & (scale.isna() | df_out["value_basis_status"].eq("source_only_no_price"))
    df_out["value_basis_common_excluded"] = excluded
    df_out["value_basis_excluded"] = excluded
    if inverse.any() or excluded.any():
        df_out["shares"] = shares.mask(inverse, inferred).mask(excluded)
        df_out["value_usd"] = df_out["value_usd"].mask(inverse, dollars).mask(excluded)
    df_out["value_basis_shares_source"] = np.where(inverse, "value_estimated", np.where(np.isfinite(shares) & ~excluded, "filed", "unknown"))


def _repair_register(
    df_register: pd.DataFrame,
    keys: list[str],
    df_out: pd.DataFrame,
    value: pd.Series,
    estimates: dict[str, pd.Series],
    quote: pd.Series,
    common: pd.Series,
) -> pd.DataFrame:
    """Count repairs and explicit exclusions before consumer-side filtering."""
    estimated = estimates["value_usd"]
    audit = df_out[keys].assign(
        _before=value,
        _after=df_out["value_usd"],
        _estimated=estimated.astype(int),
        _estimated_value=df_out["value_usd"].where(estimated, 0.0),
        _estimated_missing=(estimated & value.isna()).astype(int),
        _estimated_price_mismatch=(estimated & df_out["value_basis_price_mismatch"]).astype(int),
        _estimated_price_mismatch_value=df_out["value_usd"].where(estimated & df_out["value_basis_price_mismatch"], 0.0),
        _estimated_shares=df_out["value_basis_shares_reconstructed"].astype(int),
        _inverse_unverified=df_out["value_basis_inverse_unit_unverified"].astype(int),
        _common_excluded=df_out["value_basis_common_excluded"].astype(int),
        _excluded=df_out["value_basis_excluded"].astype(int),
        _unresolved_price=(common & df_out["shares"].gt(0).fillna(False) & quote.isna() & df_out["value_usd"].isna()).astype(int),
    )
    for leg in ("call", "put"):
        mask = estimates[f"{leg}_value"]
        audit[f"_estimated_{leg}"] = mask.astype(int)
        audit[f"_estimated_{leg}_value"] = df_out[f"{leg}_value"].where(mask, 0.0) if f"{leg}_value" in df_out else 0.0
    totals = audit.groupby(keys, dropna=False)
    df_totals = totals.agg(
        rows=("_before", "size"),
        value_before=("_before", "sum"),
        value_after=("_after", "sum"),
        estimated_rows=("_estimated", "sum"),
        estimated_value=("_estimated_value", "sum"),
        estimated_call_rows=("_estimated_call", "sum"),
        estimated_call_value=("_estimated_call_value", "sum"),
        estimated_put_rows=("_estimated_put", "sum"),
        estimated_put_value=("_estimated_put_value", "sum"),
        estimated_missing_value_rows=("_estimated_missing", "sum"),
        estimated_price_mismatch_rows=("_estimated_price_mismatch", "sum"),
        estimated_price_mismatch_value=("_estimated_price_mismatch_value", "sum"),
        estimated_shares_rows=("_estimated_shares", "sum"),
        inverse_unit_unverified_rows=("_inverse_unverified", "sum"),
        common_excluded_rows=("_common_excluded", "sum"),
        excluded_rows=("_excluded", "sum"),
        unresolved_price_rows=("_unresolved_price", "sum"),
    )
    df_totals.loc[totals["_after"].count().lt(totals.size()), "value_after"] = np.nan
    df_register = df_register.merge(df_totals.reset_index(), on=keys, validate="one_to_one")
    return df_register


def _empty_register() -> pd.DataFrame:
    return pd.DataFrame(columns=["cik", "period", "median_ratio", "factor", "rows", "value_before", "value_after"])


def nullable_value_to_daily(history: pd.DataFrame, field: str, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """An explicit unknown dollar observation replaces an older known dollar value."""
    df_daily = fundamentals_to_daily(history, field, trading_index)
    df_known = history.assign(_value_known=history[field].notna().astype(float))
    known = fundamentals_to_daily(df_known, "_value_known", trading_index)
    return df_daily.where(known.eq(1.0))


def log_register(register: pd.DataFrame, value_before_total: float, log: logging.Logger = logger) -> None:
    """Summarize unit decisions; raw dollar shares are not feature-error rates."""
    if register.empty:
        return
    for name, multiplier in (("divided by 1000", 0.001), ("multiplied by 1000", 1000.0), ("unit unresolved", np.nan)):
        selected = register["factor"].isna() if pd.isna(multiplier) else register["factor"].eq(multiplier)
        df_hit = register.loc[selected]
        if df_hit.empty:
            continue
        moved = float(df_hit["value_before"].sum())
        log.warning(
            "13F value basis: %s filing(s) %s -- %s row(s), %.2f%% of raw pre-correction dollars; quantity estimates tagged",
            f"{len(df_hit):,}",
            name,
            f"{int(df_hit['rows'].sum()):,}",
            100 * moved / value_before_total if value_before_total else 0,
        )
    for leg, prefix in (("common", ""), ("call", "call_"), ("put", "put_")):
        estimated_rows = int(register[f"estimated_{prefix}rows"].sum())
        if estimated_rows:
            log.info(
                "13F value basis: %s %s row(s) market-estimated at the report-period close; %.3e estimated underlying dollars; source tables unchanged",
                f"{estimated_rows:,}",
                leg,
                float(register[f"estimated_{prefix}value"].sum()),
            )
    missing_values = int(register["estimated_missing_value_rows"].sum())
    mismatched_values = int(register["estimated_price_mismatch_rows"].sum())
    inverse_rows = int(register["estimated_shares_rows"].sum())
    unverified_rows = int(register["inverse_unit_unverified_rows"].sum())
    excluded_rows = int(register["excluded_rows"].sum())
    sibling_rows = int(register["common_excluded_rows"].sum()) - excluded_rows
    unresolved_rows = int(register["unresolved_price_rows"].sum())
    if missing_values:
        log.info(
            "13F forward fallback: %s common row(s) with missing filed value -> reported shares x eligible as-traded report-period price",
            f"{missing_values:,}",
        )
    if mismatched_values:
        log.warning(
            "13F position price fallback: %s common row(s) outside +/- %.1f%% of the eligible as-traded report-period price after book scaling -> same forward fallback as missing value; %.3e estimated dollars; reported shares preserved",
            f"{mismatched_values:,}",
            100 * float(register["price_tolerance"].iloc[0]),
            float(register["estimated_price_mismatch_value"].sum()),
        )
    if inverse_rows:
        log.info(
            "13F inverse fallback: %s common row(s) with zero/missing shares -> filed dollars / eligible as-traded report-period price; quantities are estimates",
            f"{inverse_rows:,}",
        )
    if unverified_rows:
        log.warning(
            "13F inverse fallback: %s row(s) unit unverified -> assumes stored value_usd is dollars; no unit factor inferred, quantities flagged",
            f"{unverified_rows:,}",
        )
    if excluded_rows:
        log.warning(
            "13F fallback: %s row(s) excluded before feature aggregation (no usable common quantity and no eligible as-traded report-period price); source rows preserved",
            f"{excluded_rows:,}",
        )
    if sibling_rows:
        log.warning(
            "13F fallback: %s unusable common leg(s) nulled; rows retained for supported option/debt exposures; common totals remain unknown",
            f"{sibling_rows:,}",
        )
    if unresolved_rows:
        log.warning(
            "13F fallback: %s common value(s) still unknown (positive shares but no eligible as-traded report-period price); reported shares retained",
            f"{unresolved_rows:,}",
        )
    for name in ("source_only_thin", "source_only_no_price"):
        df_hit = register.loc[register["status"].eq(name)]
        if not df_hit.empty:
            log.info(
                "13F value basis: %s filing(s) / %s row(s) %s (no supported unit override)",
                f"{len(df_hit):,}",
                f"{int(df_hit['rows'].sum()):,}",
                name,
            )
