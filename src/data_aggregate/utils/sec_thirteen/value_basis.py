"""Clean 13F VALUE per filing and per line against report-quarter-end as-traded prices.

Source tables remain untouched. One power-of-ten unit factor per filing moves every monetary leg; each priced common line is then
checked against the quote (`close_pit`, else a consensus price built from the other filings of the same security and quarter):

- inside the row band the filed line is kept;
- outside it the leg that fails a plausibility test is rebuilt from the other and the quote (value = shares x quote, or
  shares = value / quote), a swapped value/shares line is treated the same way, and a line no rule can support is dropped;
- a dropped or missing line never voids its book: a book is accepted when enough of its lines and of its cover total survive.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.common.quotes import period_close

logger = logging.getLogger(__name__)
KEPT, DIVIDED, MULTIPLIED, ABSTAINED = 0, 1, -1, 9
_VALUE_COLUMNS = ("value_usd", "call_value", "put_value", "debt_value", "other_value")
_MAX_PRICE_AGE_DAYS = 7
#: The largest power-of-ten unit error a filing can carry (a thousand, as in the dollars/thousands change).
_MAX_UNIT_POWER = 3
#: A line whose value equals its share count within a thousand-dollar rounding (and within 0.1% of it) is a swapped column.
_SWAP_ROUNDING = 500.0
_SWAP_RELATIVE = 0.001
#: A position above this share of the issuer's shares outstanding is not a plausible 13F line.
_MAX_OWNED_SHARE = 1.0
#: A book above this many dollars is an error whatever the filer says.
BOOK_CEILING_USD = 12.0e12
LINE_CEILING_USD = 5.0e12
_MCAP_TOLERANCE = 1.1


_MIN_SECURITIES = 5
_MIN_AGREEMENT = 0.8
_ROW_BAND_LOW = 0.35
_ROW_BAND_HIGH = 3.5
MIN_ROW_COVERAGE = 0.8
MIN_CONSENSUS_FILERS = 3


def prepare_public_value_quotes(close: pd.DataFrame | None) -> pd.Series | None:
    """Prepare the existing age-bounded quote lookup once for streamed DATE periods."""
    if close is None or close.empty:
        return None
    first = pd.Timestamp(close.index.min()).normalize()
    last = pd.Timestamp(close.index.max()).normalize() + pd.Timedelta(days=_MAX_PRICE_AGE_DAYS)
    periods = pd.Series(pd.date_range(first, last))
    return period_close(close, periods, _MAX_PRICE_AGE_DAYS)


def repair_value_basis(
    holdings: pd.DataFrame,
    close: pd.DataFrame | None,
    *,
    period_quotes: pd.Series | None = None,
    consensus: pd.Series | None = None,
    limits: Mapping[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unit per filing, then every priced common line, then the book acceptance. Returns the repaired lines and the per-filing register.

    `close` is report-period as-traded prices (`period_quotes` is the same lookup prepared once); `consensus` is a (period, cusip) price for
    securities `close` does not cover; `limits` may carry `mcap` and `shares_out` (period, ticker) series for the plausibility tests.
    Source frames remain unchanged and repeat repair is idempotent.
    """
    df_out = holdings.copy()
    if "value_basis_repaired" in df_out or df_out.empty:
        return df_out, _empty_register()

    # Evidence per line: filed values, quantities and the best available reference price.
    if "value_usd" not in df_out:
        df_out["value_usd"] = np.nan
    keys = ["cik", "period", *(column for column in ("filing_date", "accession", "accession_number") if column in df_out)]
    value = pd.to_numeric(df_out["value_usd"], errors="coerce").astype("float64")
    shares = pd.to_numeric(df_out["shares"], errors="coerce").astype("float64")
    if period_quotes is None and "ticker" in df_out and close is not None and not close.empty:
        period_quotes = period_close(close, df_out["period"], _MAX_PRICE_AGE_DAYS)
    quote = _row_quotes(df_out, period_quotes)
    cons = _row_consensus(df_out, consensus)
    reference = quote.fillna(cons)
    common = df_out["position_type"].astype(str).str.lower().eq("common") if "position_type" in df_out else pd.Series(True, index=df_out.index)
    swapped = _swapped(common, value, shares)
    ratio = (value / shares.where(shares.gt(0)) / reference).where(value.gt(0) & ~swapped & common)
    ratio = ratio.where(np.isfinite(ratio) & ratio.gt(0))

    # One unit factor per filing scales every monetary leg.
    df_register = _book_factors(df_out, keys, ratio, quote.notna())
    _apply_book_factors(df_out, df_register, keys)

    # Every priced common line against its reference, once the book unit is frozen.
    scaled = pd.to_numeric(df_out["value_usd"], errors="coerce").astype("float64")
    cover = _cover_dollars(df_out, scaled)
    _repair_common_lines(df_out, value, scaled, shares, reference, quote, common, cover, limits)

    # Option legs use their own underlying quantities; then book acceptance.
    estimates = _estimate_option_legs(df_out, holdings, reference, common)
    _keep_sibling_exposures(df_out)
    return df_out, _repair_register(df_register, keys, df_out, value, estimates, common, cover, reference)


def _swapped(common: pd.Series, value: pd.Series, shares: pd.Series) -> pd.Series:
    """Common lines whose filed value equals their share count: the filer put one column in the other."""
    return common & shares.gt(0) & (value - shares).abs().le(np.minimum(_SWAP_ROUNDING, _SWAP_RELATIVE * shares))


def _row_quotes(df_out: pd.DataFrame, period_quotes: pd.Series | None) -> pd.Series:
    """The age-bounded report-period close of each row's ticker; NaN without a ticker or quote."""
    quote = pd.Series(np.nan, index=df_out.index, dtype=float)
    if "ticker" not in df_out:
        return quote
    if period_quotes is not None:
        quote[:] = period_quotes.reindex(pd.MultiIndex.from_frame(df_out[["period", "ticker"]])).to_numpy()
    return quote


def _row_consensus(df_out: pd.DataFrame, consensus: pd.Series | None) -> pd.Series:
    """The (period, cusip) consensus price of each row; NaN without one."""
    out = pd.Series(np.nan, index=df_out.index, dtype=float)
    if consensus is None or consensus.empty or "cusip" not in df_out:
        return out
    out[:] = consensus.reindex(pd.MultiIndex.from_frame(df_out[["period", "cusip"]])).to_numpy()
    return out


def _book_factors(df_out: pd.DataFrame, keys: list[str], ratio: pd.Series, priced: pd.Series) -> pd.DataFrame:
    """Per filing: the power-of-ten unit its securities agree on, the support behind it and the status.

    A security's evidence is the median of its lines' value / (shares x reference). The candidate unit is the power of ten that at least
    `min_securities` securities sit on exactly (within a factor 1.6); it is accepted when `min_agreement` of the filing's securities then
    fall inside the row band. With no such candidate the filed unit is tested the same way. No agreement: the unit stays 1, status abstained.
    """
    security = "cusip" if "cusip" in df_out else "ticker"
    frame = df_out[keys + [security]].assign(_log=np.log10(ratio), _priced=priced)
    evidence = frame.dropna(subset=["_log"]).groupby(keys + [security], dropna=False).agg(_log=("_log", "median"), _priced=("_priced", "mean"))
    levels = list(range(len(keys)))
    log_low, log_high = np.log10(_ROW_BAND_LOW), np.log10(_ROW_BAND_HIGH)
    count = evidence["_log"].groupby(level=levels, dropna=False).size()
    priced_share = evidence["_priced"].groupby(level=levels, dropna=False).mean()
    measured = (10.0 ** evidence["_log"]).groupby(level=levels, dropna=False).median()

    # Candidate unit: the modal power of ten among securities lying on one; the smallest |change| wins a tie.
    rounded = np.round(-evidence["_log"])
    tight = ((evidence["_log"] + rounded).abs() <= 0.2) & (rounded.abs() <= _MAX_UNIT_POWER) & rounded.ne(0)
    df_votes = rounded[tight].rename("power").reset_index()
    tally = df_votes.groupby(keys + ["power"], dropna=False).size().rename("n").reset_index()
    tally = tally.assign(_abs=tally["power"].abs()).sort_values(["n", "_abs"], ascending=[False, True])
    best = tally.drop_duplicates(keys).set_index(keys).reindex(count.index)
    power = best["power"].fillna(0.0)

    # Agreement of the securities with the filed unit and with the candidate.
    book_of_security = pd.MultiIndex.from_frame(evidence.index.to_frame(index=False)[keys])

    def agreement(shift: pd.Series) -> pd.Series:
        shifted = evidence["_log"] + shift.reindex(book_of_security).to_numpy()
        return shifted.between(log_low, log_high).groupby(level=levels, dropna=False).sum() / count

    filed_agreement = agreement(pd.Series(0.0, index=count.index))
    candidate_agreement = agreement(power)
    use_candidate = best["n"].ge(_MIN_SECURITIES).fillna(False) & candidate_agreement.ge(_MIN_AGREEMENT) & candidate_agreement.gt(filed_agreement)
    chosen = candidate_agreement.where(use_candidate, filed_agreement)
    accepted = chosen.ge(_MIN_AGREEMENT)

    factor = pd.Series(1.0, index=count.index)
    factor[use_candidate] = 10.0 ** power[use_candidate]
    status = pd.Series("abstained", index=count.index)
    status[accepted] = np.where(priced_share[accepted] >= 0.5, "market_checked", "consensus_checked")
    status[accepted & ~use_candidate & count.lt(_MIN_SECURITIES)] = "source_only_thin"
    columns = {"median_ratio": measured, "factor": factor, "priced_securities": count, "agreement": chosen, "status": status}
    df_register = pd.concat([series.rename(name) for name, series in columns.items()], axis=1)

    # Filings with no usable evidence at all keep their filed unit under an explicit status.
    every = df_out[keys].drop_duplicates().set_index(keys).index
    df_register = df_register.reindex(every)
    df_register["factor"] = df_register["factor"].fillna(1.0)
    df_register["priced_securities"] = df_register["priced_securities"].fillna(0).astype(int)
    df_register["agreement"] = df_register["agreement"].fillna(0.0)
    df_register["status"] = df_register["status"].fillna("source_only_no_price")
    return df_register.reset_index().assign(band_low=_ROW_BAND_LOW, band_high=_ROW_BAND_HIGH)


def _apply_book_factors(df_out: pd.DataFrame, df_register: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Write each row's unit flag and status, scale its monetary legs and return the row scale (never NaN)."""
    df_rows = df_out[keys].merge(df_register[keys + ["factor", "status"]], on=keys, how="left", validate="many_to_one")
    scale = pd.Series(df_rows["factor"].to_numpy(), index=df_out.index).fillna(1.0)
    flag = pd.Series(KEPT, index=df_out.index)
    flag.loc[scale.lt(1.0)] = DIVIDED
    flag.loc[scale.gt(1.0)] = MULTIPLIED
    flag.loc[df_rows["status"].eq("abstained").to_numpy()] = ABSTAINED
    df_out["value_basis_repaired"] = flag
    df_out["value_basis_status"] = df_rows["status"].to_numpy()
    for column in (column for column in _VALUE_COLUMNS if column in df_out):
        original = pd.to_numeric(df_out[column], errors="coerce").astype("float64")
        df_out[column] = (original * scale).mask(original.eq(0), 0.0)
    return scale


def _cover_alignment(df_out: pd.DataFrame, scaled: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Per row: the filing's cover total in the filed unit (aligned to the sum of its raw lines by a power of ten, NaN when it does not
    align) and the book's dollars per filed unit. The declared total's own unit is not reliable in the source, hence the alignment."""
    nan = pd.Series(np.nan, index=df_out.index, dtype=float)
    if "declared_value_total" not in df_out or "raw_value_usd" not in df_out:
        return nan, nan
    book = [column for column in ("cik", "period", "filing_date", "accession_number") if column in df_out]
    by_book = [df_out[column] for column in book]
    raw = pd.to_numeric(df_out["raw_value_usd"], errors="coerce").astype("float64")
    declared = pd.to_numeric(df_out["declared_value_total"], errors="coerce").astype("float64")
    raw_sum = raw.groupby(by_book, dropna=False).transform("sum")
    log_rho = np.log10((declared / raw_sum.where(raw_sum.gt(0))).where(declared.gt(0)))
    power = np.round(log_rho)
    aligned = ((log_rho - power).abs() <= 0.05) & (power.abs() <= 6)
    cover_raw = (declared / 10.0**power).where(aligned)
    per_unit = (scaled / raw.where(raw.gt(0))).replace([np.inf, -np.inf], np.nan)
    return cover_raw, per_unit.groupby(by_book, dropna=False).transform("median")


def _cover_dollars(df_out: pd.DataFrame, scaled: pd.Series) -> pd.Series:
    """The filing's cover total in dollars on the repaired unit; NaN without an aligned declared total."""
    cover_raw, per_unit = _cover_alignment(df_out, scaled)
    return cover_raw * per_unit


def _lookup(df_out: pd.DataFrame, series: pd.Series | None) -> pd.Series:
    """A (period, ticker) limit series on each row; NaN when absent."""
    out = pd.Series(np.nan, index=df_out.index, dtype=float)
    if series is not None and "ticker" in df_out:
        out[:] = series.reindex(pd.MultiIndex.from_frame(df_out[["period", "ticker"]])).to_numpy()
    return out


def _repair_common_lines(
    df_out: pd.DataFrame,
    filed: pd.Series,
    scaled: pd.Series,
    shares: pd.Series,
    reference: pd.Series,
    quote: pd.Series,
    common: pd.Series,
    cover: pd.Series,
    limits: Mapping[str, pd.Series] | None,
) -> None:
    """Check each common line against its reference once the book unit is frozen, and rebuild the failing leg.

    Inside the row band the line is kept. Outside it (or when value equals shares) two repairs compete: value = shares x reference,
    allowed only while it stays under the issuer's market cap and the filing's cover total, and shares = value / reference, allowed
    only while it stays under the issuer's shares outstanding and the value under the cover total. Both allowed: the value is rebuilt
    from the quote. Neither: the line is dropped. A missing value is rebuilt from the quote, or the line stays unknown.
    """
    limits = limits or {}
    mcap, shares_out = _lookup(df_out, limits.get("mcap")), _lookup(df_out, limits.get("shares_out"))
    has_ref = reference.notna() & reference.gt(0)
    valid_qty = shares.gt(0) & np.isfinite(shares)
    valid_value = scaled.gt(0) & np.isfinite(scaled)
    swapped = _swapped(common, filed, shares)
    line_ratio = scaled / (shares.where(valid_qty) * reference)
    outside = common & has_ref & valid_qty & valid_value & ~line_ratio.between(_ROW_BAND_LOW, _ROW_BAND_HIGH)
    outside |= common & has_ref & swapped & valid_qty

    # Candidate repairs.
    value_from_shares = shares * reference
    shares_from_value = scaled / reference
    cap = np.fmin(mcap.fillna(np.inf), cover.fillna(np.inf) * 1.01)
    value_ok = np.isfinite(value_from_shares) & value_from_shares.le(cap)
    shares_ok = (
        np.isfinite(shares_from_value) & shares_from_value.le(shares_out.fillna(np.inf) * _MAX_OWNED_SHARE) & scaled.le(cover.fillna(np.inf) * 1.01)
    )
    take_value = outside & value_ok
    take_shares = outside & ~value_ok & shares_ok
    dropped = outside & ~value_ok & ~shares_ok

    # A missing or unusable value is rebuilt from the quote when the quantity is valid.
    absent_value = common & ~valid_value & has_ref & valid_qty
    absent_value &= ~outside
    no_quantity = common & valid_value & ~valid_qty & has_ref
    out_value, out_shares = scaled.copy(), shares.copy()
    out_value = out_value.mask(take_value | absent_value, value_from_shares).mask(dropped, np.nan)
    out_shares = out_shares.mask(take_shares | no_quantity, shares_from_value).mask(dropped, np.nan)
    # In a book whose unit the priced lines contradict but cannot settle, a line with no reference keeps its shares and loses its value.
    unverified = common & ~has_ref & df_out["value_basis_status"].eq("abstained")
    unusable = common & ~valid_qty & ~has_ref & (scaled.gt(0) | scaled.isna())
    structural_zero = scaled.eq(0) & ~shares.gt(0)
    unknown = (common & ~valid_value & ~(has_ref & valid_qty) & ~structural_zero) | unverified | unusable
    df_out["value_usd"] = out_value.where(~unknown, np.nan)
    df_out["shares"] = out_shares.mask(unusable)
    df_out["value_basis_price_mismatch"] = outside
    df_out["value_basis_value_rebuilt"] = take_value | absent_value
    df_out["value_basis_shares_rebuilt"] = take_shares | no_quantity
    df_out["value_basis_dropped"] = dropped
    df_out["value_basis_missing_value"] = common & ~valid_value
    df_out["value_basis_reference"] = np.where(quote.notna(), "price_pit", np.where(reference.notna(), "consensus", "none"))
    df_out["value_basis_common_excluded"] = dropped | unusable
    df_out["value_basis_excluded"] = dropped | unusable
    # Provenance consumed downstream: how each leg of the line was obtained.
    rebuilt_shares = take_shares | no_quantity
    df_out["value_basis_shares_reconstructed"] = rebuilt_shares
    df_out["value_basis_inverse_unit_unverified"] = no_quantity & df_out["value_basis_status"].isin(["abstained", "source_only_no_price"])
    df_out["value_basis_shares_source"] = np.where(
        rebuilt_shares, "value_estimated", np.where(np.isfinite(df_out["shares"]) & ~dropped, "filed", "unknown")
    )
    df_out["value_basis_reconstructed"] = take_value | absent_value
    df_out["value_basis_value_source"] = np.where(
        take_value | absent_value, "market_estimated", np.where(df_out["value_usd"].notna(), "filed", "unknown")
    )


def _estimate_option_legs(df_out: pd.DataFrame, holdings: pd.DataFrame, reference: pd.Series, common: pd.Series) -> dict[str, pd.Series]:
    """Replace uncertain call/put dollars with tagged quantity x reference estimates (options report underlying-share exposure)."""
    unpriced = df_out["value_basis_status"].isin(["source_only_no_price", "abstained"])
    estimates: dict[str, pd.Series] = {"value_usd": df_out["value_basis_reconstructed"]}
    for column in ("debt_value", "other_value"):
        if column in df_out:
            df_out[column] = df_out[column].mask(unpriced & pd.to_numeric(holdings[column], errors="coerce").ne(0))
    for column, quantity_column, label in (("call_value", "call_shares", "call_"), ("put_value", "put_shares", "put_")):
        estimates[column] = pd.Series(False, index=df_out.index)
        if column not in df_out:
            continue
        raw = pd.to_numeric(holdings[column], errors="coerce").astype("float64")
        if quantity_column in df_out:
            quantity = pd.to_numeric(df_out[quantity_column], errors="coerce").astype("float64")
            uncertain = unpriced & (raw.ne(0) | quantity.gt(0))
            uncertain |= raw.eq(quantity) & quantity.gt(0)
            uncertain = uncertain | ~np.isfinite(df_out[column]) | df_out[column].lt(0) | (df_out[column].eq(0) & quantity.gt(0))
            market_value = (quantity * reference).where(quantity.gt(0) & np.isfinite(quantity) & np.isfinite(quantity * reference))
            df_out[column] = df_out[column].mask(uncertain, market_value)
            estimates[column] = uncertain & market_value.notna()
        else:
            df_out[column] = df_out[column].mask(unpriced & raw.ne(0))
        df_out[f"value_basis_{label}reconstructed"] = estimates[column]
        df_out[f"value_basis_{label}value_source"] = np.where(
            estimates[column], "market_estimated", np.where(df_out[column].notna(), "filed", "unknown")
        )
    return estimates


def _keep_sibling_exposures(df_out: pd.DataFrame) -> None:
    """A row with a supported option/debt/other exposure is retained even when its common leg is excluded."""
    sibling = pd.Series(False, index=df_out.index)
    for column in ("call_shares", "put_shares", "debt_prn", "call_value", "put_value", "debt_value", "other_value"):
        if column in df_out:
            exposure = pd.to_numeric(df_out[column], errors="coerce").astype("float64")
            sibling |= np.isfinite(exposure) & exposure.gt(0)
    df_out["value_basis_excluded"] &= ~sibling


def _repair_register(
    df_register: pd.DataFrame,
    keys: list[str],
    df_out: pd.DataFrame,
    value: pd.Series,
    estimates: dict[str, pd.Series],
    common: pd.Series,
    cover: pd.Series,
    reference: pd.Series,
) -> pd.DataFrame:
    """Count line actions per filing and decide book acceptance: enough lines kept and enough of the cover total retained."""
    kept = common & df_out["value_usd"].notna() & ~df_out["value_basis_excluded"]
    cover_raw, _ = _cover_alignment(df_out, df_out["value_usd"])
    raw_line = pd.to_numeric(df_out["raw_value_usd"], errors="coerce") if "raw_value_usd" in df_out else pd.Series(np.nan, index=df_out.index)
    audit = df_out[keys].assign(
        _kept_raw=raw_line.where(kept, 0.0),
        _cover_raw=cover_raw,
        _before=value,
        _after=df_out["value_usd"],
        _common=common.astype(int),
        _kept=kept.astype(int),
        _kept_value=df_out["value_usd"].where(kept, 0.0),
        _cover=cover,
        _estimated=df_out["value_basis_value_rebuilt"].astype(int),
        _estimated_value=df_out["value_usd"].where(df_out["value_basis_value_rebuilt"], 0.0),
        _estimated_missing=(df_out["value_basis_value_rebuilt"] & df_out["value_basis_missing_value"]).astype(int),
        _price_mismatch=df_out["value_basis_price_mismatch"].astype(int),
        _shares_rebuilt=df_out["value_basis_shares_rebuilt"].astype(int),
        _inverse_unverified=df_out["value_basis_inverse_unit_unverified"].astype(int),
        _common_excluded=df_out["value_basis_common_excluded"].astype(int),
        _excluded=df_out["value_basis_excluded"].astype(int),
        _dropped=df_out["value_basis_dropped"].astype(int),
        _missing=(common & df_out["value_basis_missing_value"]).astype(int),
        _unresolved=(common & df_out["shares"].gt(0).fillna(False) & reference.isna() & df_out["value_usd"].isna()).astype(int),
        _consensus=(common & df_out["value_basis_reference"].eq("consensus")).astype(int),
    )
    for leg in ("call", "put"):
        mask = estimates[f"{leg}_value"]
        audit[f"_estimated_{leg}"] = mask.astype(int)
        audit[f"_estimated_{leg}_value"] = df_out[f"{leg}_value"].where(mask, 0.0) if f"{leg}_value" in df_out else 0.0
    totals = audit.groupby(keys, dropna=False)
    df_totals = totals.agg(
        rows=("_before", "size"),
        common_rows=("_common", "sum"),
        kept_rows=("_kept", "sum"),
        kept_value=("_kept_value", "sum"),
        cover_dollars=("_cover", "max"),
        kept_raw=("_kept_raw", "sum"),
        cover_raw=("_cover_raw", "max"),
        value_before=("_before", "sum"),
        value_after=("_after", "sum"),
        estimated_rows=("_estimated", "sum"),
        estimated_value=("_estimated_value", "sum"),
        estimated_call_rows=("_estimated_call", "sum"),
        estimated_call_value=("_estimated_call_value", "sum"),
        estimated_put_rows=("_estimated_put", "sum"),
        estimated_put_value=("_estimated_put_value", "sum"),
        estimated_missing_value_rows=("_estimated_missing", "sum"),
        estimated_price_mismatch_rows=("_price_mismatch", "sum"),
        estimated_price_mismatch_value=("_estimated_value", "sum"),
        estimated_shares_rows=("_shares_rebuilt", "sum"),
        inverse_unit_unverified_rows=("_inverse_unverified", "sum"),
        common_excluded_rows=("_common_excluded", "sum"),
        excluded_rows=("_excluded", "sum"),
        dropped_rows=("_dropped", "sum"),
        missing_value_rows=("_missing", "sum"),
        unresolved_price_rows=("_unresolved", "sum"),
        consensus_rows=("_consensus", "sum"),
    )
    df_totals["row_coverage"] = df_totals["kept_rows"] / df_totals["common_rows"].where(df_totals["common_rows"].gt(0))
    df_totals["cover_ratio"] = df_totals["kept_raw"] / df_totals["cover_raw"].where(df_totals["cover_raw"].gt(0))
    rows_ok = df_totals["row_coverage"].fillna(1.0).ge(MIN_ROW_COVERAGE)
    cover_ok = df_totals["cover_ratio"].isna() | df_totals["cover_ratio"].ge(MIN_ROW_COVERAGE)
    df_totals["book_ok"] = rows_ok & cover_ok
    df_register = df_register.merge(df_totals.reset_index(), on=keys, validate="one_to_one")
    df_rows = df_out[keys].merge(df_register[[*keys, "book_ok"]], on=keys, how="left", validate="many_to_one")
    df_out["value_basis_book_ok"] = df_rows["book_ok"].fillna(True).to_numpy(dtype=bool)
    return df_register


def _empty_register() -> pd.DataFrame:
    return pd.DataFrame(columns=["cik", "period", "median_ratio", "factor", "rows", "value_before", "value_after"])


def nullable_value_to_daily(history: pd.DataFrame, field: str, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """An explicit unknown dollar observation replaces an older known dollar value."""
    df_values = history[["ticker", "as_of", field]]
    df_daily = fundamentals_to_daily(df_values, field, trading_index)
    df_known = df_values.assign(_value_known=df_values[field].notna().astype(float))
    known = fundamentals_to_daily(df_known, "_value_known", trading_index)
    return df_daily.where(known.eq(1.0))


def consensus_prices(rows: pd.DataFrame, min_filers: int) -> pd.Series:
    """One price per (period, cusip): the median of value / shares over every filer's cleaned common line, vectorized over the whole frame.

    `rows` must already carry the book unit and be free of dropped lines; a price needs at least `min_filers` distinct filers.
    """
    shares = pd.to_numeric(rows["shares"], errors="coerce")
    value = pd.to_numeric(rows["value_usd"], errors="coerce")
    dropped = rows["value_basis_dropped"].fillna(False).astype(bool) if "value_basis_dropped" in rows else pd.Series(False, index=rows.index)
    ok = value.gt(0) & shares.gt(0) & rows["position_type"].astype(str).str.lower().eq("common") & ~dropped
    frame = pd.DataFrame(
        {"period": rows.loc[ok, "period"], "cusip": rows.loc[ok, "cusip"], "cik": rows.loc[ok, "cik"], "price": (value / shares)[ok]}
    )
    grouped = frame.groupby(["period", "cusip"])
    price = grouped["price"].median()
    filers = grouped["cik"].nunique()
    return price[filers.ge(min_filers)].rename("consensus_price")


def unrealistic_lines(df_holdings: pd.DataFrame, limits: Mapping[str, pd.Series] | None) -> pd.Series:
    """The one ceiling rule, applied after repair: a common line is unrealistic above 110% of its issuer's market cap (above the single-line
    ceiling without one), and a book still above the book ceiling loses its largest lines until the rest fits. The caller excludes them."""
    common = df_holdings["position_type"].astype(str).str.lower().eq("common")
    value = pd.to_numeric(df_holdings["value_usd"], errors="coerce")
    cap = np.fmin(_lookup(df_holdings, (limits or {}).get("mcap")) * _MCAP_TOLERANCE, LINE_CEILING_USD)
    bad = common & value.gt(cap)
    keys = [column for column in ("cik", "period", "accession_number") if column in df_holdings]
    kept = pd.DataFrame({"value": value.where(common & ~bad).to_numpy(), **{column: df_holdings[column].to_numpy() for column in keys}}).sort_values(
        "value"
    )
    running = kept.groupby(keys, dropna=False)["value"].cumsum().sort_index().to_numpy()
    return bad | pd.Series(running > BOOK_CEILING_USD, index=df_holdings.index)


def log_register(register: pd.DataFrame, value_before_total: float) -> None:
    """One aggregated view of a cleaning run: percent of filings (books) and of lines per reason, never filing by filing."""
    if register.empty:
        return
    books, lines = len(register), int(register["common_rows"].sum())
    share = lambda mask: 100.0 * float(mask.sum()) / books  # noqa: E731
    changed = register["factor"].ne(1.0)
    logger.info(
        "13F cleaning, %s filings / %s common lines | unit moved %.2f%% of filings (x1000 %.2f%%, /1000 %.2f%%, other power of ten %.2f%%) | "
        "unit unresolved (kept as filed) %.2f%% | no price evidence %.2f%% | thin %.2f%%",
        f"{books:,}",
        f"{lines:,}",
        share(changed),
        share(register["factor"].eq(1000.0)),
        share(register["factor"].eq(0.001)),
        share(changed & ~register["factor"].isin([1000.0, 0.001])),
        share(register["status"].eq("abstained")),
        share(register["status"].eq("source_only_no_price")),
        share(register["status"].eq("source_only_thin")),
    )
    pct = lambda column: 100.0 * float(register[column].sum()) / max(lines, 1)  # noqa: E731
    logger.info(
        "13F cleaning lines | outside the price band %.3f%% | value rebuilt from the quote %.3f%% (of which a missing value %.3f%%) | "
        "shares rebuilt from value %.3f%% | dropped %.3f%% | priced by consensus %.3f%% | still unpriced and unvalued %.3f%%",
        pct("estimated_price_mismatch_rows"),
        pct("estimated_rows"),
        pct("estimated_missing_value_rows"),
        pct("estimated_shares_rows"),
        pct("dropped_rows"),
        pct("consensus_rows"),
        pct("unresolved_price_rows"),
    )
    low = register["row_coverage"].fillna(1.0).lt(0.8)
    off_cover = register["cover_ratio"].notna() & register["cover_ratio"].lt(0.8)
    logger.info(
        "13F cleaning books | rejected %.3f%% (lines kept under 80%%: %.3f%%, cover total retained under 80%%: %.3f%%) | accepted with a dropped line %.3f%% | pre-clean dollars %.3e",
        share(~register["book_ok"]),
        share(low),
        share(off_cover),
        share(register["book_ok"] & register["dropped_rows"].gt(0)),
        value_before_total,
    )
    if "cik" in register:
        managers = register["cik"].nunique()
        by_cik = register.assign(_rejected=~register["book_ok"], _moved=changed, _dropped=register["dropped_rows"].gt(0)).groupby("cik")
        usable = register.loc[register["book_ok"], "cik"].nunique()
        logger.info(
            "13F cleaning managers (%s) | with a rejected book %.2f%% | with a unit moved %.2f%% | with a dropped line %.2f%% | with no accepted book at all %.2f%%",
            f"{managers:,}",
            100.0 * float(by_cik["_rejected"].any().mean()),
            100.0 * float(by_cik["_moved"].any().mean()),
            100.0 * float(by_cik["_dropped"].any().mean()),
            100.0 * (managers - usable) / max(managers, 1),
        )
