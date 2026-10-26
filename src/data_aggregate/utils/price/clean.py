"""Price-stage return and tape-volume corrections."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.data_aggregate.utils.price.frames import get_trading_days, pivot_fields
from src.data_aggregate.utils.price.inputs import PriceInputs
from src.utils.price.repairs import BUGFIX_STEP_TOL, apply_return_seams, apply_split_vintage

logger = logging.getLogger(__name__)

NULL_RET_STEP_TOL = BUGFIX_STEP_TOL
VOLUME_JUMP_TOL = 3.0
VOLUME_JUMP_BARS = 20


def _volume_jump(volume: pd.Series, when: pd.Timestamp) -> float | None:
    """Median volume of the `VOLUME_JUMP_BARS` bars before `when` over that of the bars at or after
    it; None when either side is missing or the later median is not positive."""
    before = volume[volume.index < when].tail(VOLUME_JUMP_BARS).median()
    after = volume[volume.index >= when].head(VOLUME_JUMP_BARS).median()
    if pd.isna(before) or pd.isna(after) or after <= 0:
        return None
    return float(before / after)


def apply_volume_scale(bars: pd.DataFrame, bugfix: Mapping[str, Any]) -> pd.DataFrame:
    """Repair a vendor volume UNIT defect: the `volume` of a registered ticker's bars dated before an
    entry's `before` is multiplied by its `factor`. `bars` is long (`ticker`, `date`, `volume`); the
    input is returned untouched when no entry applies, else a copy.

    Re-verified before it fires: the observed jump across `before` (`_volume_jump`) must sit within
    `VOLUME_JUMP_TOL` of `expect_jump`, else the entry is skipped and logged. A frame with no bar
    before `before` has nothing to repair."""
    entries = bugfix.get("volume_scale") or {}
    if not entries or bars.empty:
        return bars
    out = bars
    dates = pd.to_datetime(bars["date"])
    for ticker, items in entries.items():
        mine = bars["ticker"].astype(str).eq(ticker).to_numpy()
        if not mine.any():
            continue
        for entry in items:
            when = pd.Timestamp(entry["before"])
            target = mine & (dates < when).to_numpy()
            if not target.any():
                continue
            volume = pd.Series(pd.to_numeric(out.loc[mine, "volume"], errors="coerce").to_numpy(), index=dates[mine]).sort_index()
            jump, expect = _volume_jump(volume, when), float(entry["expect_jump"])
            if jump is None or not expect / VOLUME_JUMP_TOL <= jump <= expect * VOLUME_JUMP_TOL:
                logger.info("price bugfix: volume_scale %s before %s SKIPPED -- observed jump %s, expected %.1f", ticker, when.date(), jump, expect)
                continue
            if out is bars:
                out = bars.copy()
                out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("float64")
            out.loc[target, "volume"] = out.loc[target, "volume"] * float(entry["factor"])
            logger.info(
                "price bugfix: volume_scale %s x%s on %d bar(s) before %s (observed jump %.1f)",
                ticker,
                entry["factor"],
                int(target.sum()),
                when.date(),
                jump,
            )
    return out


def apply_null_ret(ret: pd.DataFrame, bugfix: dict) -> int:
    """Null an evidenced but unresolved one-bar return, preserving every price leg.

    The five retained events have no supported economic target. A changed expectation
    is skipped with a diagnostic; rolling consumers independently mask their windows.
    """
    applied = 0
    for ticker, entries in (bugfix.get("null_ret") or {}).items():
        for entry in entries:
            when, expected = pd.Timestamp(entry["date"]), float(entry["expect_ret"])
            if ticker not in ret.columns or when not in ret.index:
                logger.info("price bugfix: %s %s null_ret SKIPPED -- outside this build's window", ticker, when.date())
                continue
            observed = float(cast(float, ret.at[when, ticker]))
            if not np.isfinite(observed):
                logger.info("price bugfix: %s %s null_ret SKIPPED -- already not a number", ticker, when.date())
                continue
            # compared as a one-bar STEP, the same quantity `apply_return_seams` compares
            if abs((1.0 + observed) / (1.0 + expected) - 1.0) > NULL_RET_STEP_TOL:
                logger.info(
                    "price bugfix: %s %s null_ret SKIPPED -- the defect is GONE or CHANGED: "
                    "observed ret %+.5f, register says %+.5f. Re-measure the entry.",
                    ticker,
                    when.date(),
                    observed,
                    expected,
                )
                continue
            ret.at[when, ticker] = np.nan
            applied += 1
            logger.info(
                "price bugfix: %s %s null_ret APPLIED -- a %+.2f%% one-bar 'return' deleted (the price legs are left exactly as published)",
                ticker,
                when.date(),
                observed * 100,
            )
    return applied


def clean_prices(inputs: PriceInputs, bugfix: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Clean volume, align the calendar, then repair quote/total legs before their use."""
    if inputs.bars[["date", "ticker"]].isna().any().any() or inputs.bars.duplicated(["date", "ticker"]).any():
        raise ValueError("Invalid or duplicate raw price keys")
    if inputs.splits is not None:
        ratios = pd.to_numeric(inputs.splits["ratio"], errors="coerce")
        invalid = inputs.splits["ticker"].isna() | pd.to_datetime(inputs.splits["date"], errors="coerce").isna()
        if (invalid | ~np.isfinite(ratios) | ratios.le(0)).any():
            raise ValueError("Invalid Yahoo price-adjustment evidence")
    df_bars = apply_volume_scale(inputs.bars, bugfix)
    wide = pivot_fields(df_bars)
    df_close = wide["close_split"]
    mask = get_trading_days(df_close, inputs.market, MACRO_MARKET_SERIES)
    index = pd.DatetimeIndex(df_close.index[mask.to_numpy()], name="date")
    if index.empty:
        raise ValueError("Invalid empty price trading calendar")
    wide = {field: df_frame.reindex(index) if field == "volume" else df_frame.loc[index] for field, df_frame in wide.items()}
    df_vendor = inputs.vendor
    if df_vendor is not None and {"ticker", "date", "price"}.issubset(df_vendor):
        df_vendor = df_vendor.assign(date=pd.to_datetime(df_vendor["date"]))
    else:
        df_vendor = None
    applied = apply_split_vintage(wide, bugfix, df_vendor)
    applied += apply_return_seams(wide, bugfix)
    logger.info("Price cleaning: %d sessions, %d guarded quote/total repairs", len(index), applied)
    return wide, df_vendor
