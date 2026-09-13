"""
signal_conditioning.py  (src/data_aggregate/utils/institutionals/signal_conditioning.py)
----------------------------------------------------------------------------------------
THE DAILY PRICE-CONDITIONING LAYER (`ic_sig_*`) -- registry section 7, features #68-#82.

THIS IS THE LAYER THAT MAKES THE PANEL MOVE ON A DAY WITH NO FILING, which is the research
report's central architectural point (section 5.2) and the thing the cube otherwise entirely
lacks: a 13F feature is a step function, flat for a quarter, and an insider signal is flat
until the next Form 4. Everything here is a function of (the date of the last disclosure) and
(the price path since), so all 15 features change every trading day while the disclosure
features hold.

    ic_sig_<fam>_age_days        trading days since the family's most recent event
    ic_sig_<fam>_ret_since       close_total(t) / close_total(t_event) - 1
    ic_sig_<fam>_resid_ret_since the same, minus the ticker's SECTOR return over that span
    ic_sig_<fam>_vol_scaled_move the residual move in units of its own span volatility

for `fam` in `super` (elite 13F), `insider` (Form 4 purchase), `act` (Schedule 13D), plus
three insider-only legs built on the transaction price a Form 4 actually carries:

    ic_sig_insider_price_vs_buy         close_split / value-weighted insider buy price - 1
    ic_sig_insider_max_dd_since_buy     max ADVERSE excursion since the last purchase filing
    ic_sig_insider_max_runup_since_buy  max FAVOURABLE excursion since the last purchase filing

NOT applied to `inst` (an all-filer aggregate has no single event to date from) or `bo` (13G
conditioning is subsumed by `act` where a filer escalates). ⚠ **NO 13F COST ANCHOR**: a
quarter-end price is not a manager's cost basis and the report is explicit about it (section
3.1.8), so manufacturing one would be a fabricated number.

THE EVENTS COME FROM THE PANEL BUILDERS, NOT FROM A SECOND READ OF THE SOURCES. Each
`build_*_feature_panel` optionally fills an `event_sink` with its own `(ticker, date)` frame,
because the event dates are only cheap where the source is already open: the insider dates are
the output of a 2M-row scope-and-repair pass, and the elite dates are the output of the
per-manager availability join. Re-deriving either here would double the most expensive read in
the step. `sink.py` states the contract.

THE PRICE BASIS IS NOT A DETAIL, and the registry's "`close_split`" is honoured for exactly one
of the fifteen:

  * every RETURN and EXCURSION uses **`close_total`** -- the price contract reserves the
    split-only series for LEVELS and the dividend-reduced one for returns (`price_frames.py`),
    and a 15-year "return since signal" taken on `close_split` is short by every dividend paid
    since;
  * **`ic_sig_insider_price_vs_buy` uses `close_split`**, because it is a LEVEL comparison
    against a filed price rather than a return, and the filed price is restated onto the same
    split basis first (below).

⚠ THE FILED PRICE MUST BE RESTATED OR THE FEATURE IS A SPLIT DETECTOR. A Form 4 carries the
price AS TRADED; `close_split` is restated to today's basis. Comparing them directly reads
NVDA's 2024 10-for-1 split as a 90% collapse below every pre-split insider purchase. Each
purchase's share count is therefore multiplied by `split_basis.future_split_factor` before the
value-weighted price is taken, so both sides of the ratio are on today's basis and the factor
cancels. The count of restated purchases is logged (the V5 obligation for this family).

TWO WINDOWS, both in TRADING days because the grid is:

  * `COST_ANCHOR_WINDOW` (126 ~ 180 calendar days) bounds the value-weighted buy price, so
    #80 is a statement about purchases an investor could still call recent. ⚠ It is a trading
    -day window where `insider_features` uses a 180-CALENDAR-day one for its own windows; the
    `ic_sig_` family is defined on the grid throughout (like `decay_events`' half-lives), and
    mixing the two units inside one feature is what makes a look-back impossible to audit.
  * `EXCURSION_LOOKBACK` (252) caps #81/#82. An excursion measured since a purchase filed nine
    years ago is a statement about the decade, not about the insider. ⚠ THIS IS THE BINDING
    LOOK-BACK OF THE WHOLE PART and it is registered in `parts.py::binding_lookbacks`, which
    `test_part_registry.py` asserts the warm-up covers -- 160 trading days would silently
    compute a different number on an incremental run than on a rebuild.

    The cap is EXACT, not approximate: the excursion is the running extreme since the anchor
    while the age is inside the cap, and the plain 252-day rolling extreme once it is past --
    each branch is exact in its own regime, and `age <= cap` is precisely where they swap.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.institutionals.decay import days_since_last_true, snap_to_grid
from src.data_aggregate.utils.institutionals.split_basis import future_split_factor

logger = logging.getLogger(__name__)

#: The three families with a well-defined, datable disclosure event.
FAMILIES: tuple[str, ...] = ("super", "insider", "act")

#: Trading days. Caps #81/#82 and sets the part's warm-up (see the module docstring).
EXCURSION_LOOKBACK = 252

#: Trading days (~180 calendar) over which insider purchases back the #80 cost anchor.
COST_ANCHOR_WINDOW = 126

#: Realized-volatility window for #77-#79, in trading days.
VOL_WINDOW = 20

#: Emission (D27, registry section 0.10). Only the three PLAIN return legs take a percentile:
#: a raw return's cross-sectional spread is far wider in 2008 than in 2017, so its scale
#: drifts with the date. The residual and vol-scaled legs have already had their scale removed
#: (a sector residual, a move in units of its own volatility), the two excursions and the cost
#: anchor are bounded ratios, and a day count is a day count in 2013 and in 2026.
EMISSION: dict[str, str] = {
    **{f"ic_sig_{fam}_age_days": "raw" for fam in FAMILIES},
    **{f"ic_sig_{fam}_ret_since": "raw+xs" for fam in FAMILIES},
    **{f"ic_sig_{fam}_resid_ret_since": "raw" for fam in FAMILIES},
    **{f"ic_sig_{fam}_vol_scaled_move": "raw" for fam in FAMILIES},
    "ic_sig_insider_price_vs_buy": "raw",
    "ic_sig_insider_max_dd_since_buy": "raw",
    "ic_sig_insider_max_runup_since_buy": "raw",
}


def _event_mask(events: pd.DataFrame | None, idx: pd.DatetimeIndex,
                columns: pd.Index) -> pd.DataFrame | None:
    """Boolean (date x ticker): did this family have an event on this grid day?

    Every event date is snapped forward onto the grid first -- a filing dated on a Saturday is
    actionable on the Monday, and a plain pivot would drop it.
    """
    if events is None or events.empty:
        return None
    e = events.dropna(subset=["ticker", "date"]).copy()
    if e.empty:
        return None
    e["_grid"] = snap_to_grid(e["date"], idx)
    e = e.dropna(subset=["_grid"])
    if e.empty:
        return None
    wide = e.assign(_f=1.0).pivot_table(index="_grid", columns="ticker", values="_f",
                                        aggfunc="max")
    return wide.reindex(index=idx, columns=columns).notna()


def _anchor(values: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """`values` as they stood on the most recent event day, held forward. NaN before the
    ticker's first event, which is what makes every leg built on it NaN there too."""
    return values.where(mask).ffill()


def _segment_extreme(values: pd.DataFrame, mask: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Running max (or min) of `values` since the last True in `mask`, inclusive.

    A one-pass recursion down the date axis, which is what makes the segment reset exact: an
    event day starts a fresh segment AT that day's value -- even when the previous segment
    carried a more extreme one -- and every later day takes the better (or worse) of the carry
    and today.

    ⚠ `started` is tracked separately rather than inferred from the carry being NaN, because
    `np.fmax` IGNORES NaN: without it the pre-event running extreme of the whole price history
    would leak out as a value, which is precisely the "NaN, never a number, before the first
    event" rule the whole family rests on. A NaN price mid-segment leaves the carry untouched
    (that is also `fmax`, used deliberately here) rather than poisoning the rest of it.
    """
    v = values.to_numpy(dtype="float64", na_value=np.nan)
    m = mask.to_numpy(dtype=bool)
    out = np.full(v.shape, np.nan)
    carry = np.full(v.shape[1], np.nan)
    started = np.zeros(v.shape[1], dtype=bool)
    better = np.fmax if kind == "max" else np.fmin
    for i in range(v.shape[0]):
        started |= m[i]
        carry = np.where(m[i], v[i], better(carry, v[i]))
        out[i] = np.where(started, carry, np.nan)
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def _capped_excursion(close: pd.DataFrame, mask: pd.DataFrame, age: pd.DataFrame,
                      kind: str, lookback: int) -> pd.DataFrame:
    """The extreme of `close` over `[max(t_event, t - lookback), t]`, against the price at the
    START of that same window.

    ⚠ THE CAP MOVES THE ANCHOR TOO, and getting that wrong makes the feature incoherent: the
    trailing 252-day MINIMUM divided by a price from 380 days ago is POSITIVE whenever the
    stock has since doubled, so a "max adverse excursion" comes out favourable. Capping both
    ends keeps it a genuine excursion -- the extreme of a window measured against that
    window's own opening price -- so the run-up is >= 0 and the drawdown <= 0 by construction,
    which is what a test can actually assert.

    EXACT in both regimes: the running extreme since the anchor while the age is inside the
    cap, the plain rolling extreme once it is past, and `age <= lookback` is precisely where
    they swap.
    """
    # ⚠ `age <= lookback` alone is FALSE where the age is NaN, which would hand the capped
    # branch a perfectly good anchor on a ticker that has never had an event -- the one place
    # this family is allowed no value at all. The regimes are gated on `age.notna()` so the
    # never-yet region stays NaN on both sides.
    started = age.notna()
    inside = started & (age <= lookback)
    past = started & ~inside
    since = _segment_extreme(close, mask, kind)
    rolling = (close.rolling(lookback, min_periods=1).max() if kind == "max"
               else close.rolling(lookback, min_periods=1).min())
    extreme = since.where(inside, rolling.where(past))
    anchor = _anchor(close, mask).where(inside, close.shift(lookback - 1).where(past))
    return (extreme / anchor.where(anchor > 0) - 1.0).replace([np.inf, -np.inf], np.nan)


def _family_fields(fam: str, events: pd.DataFrame | None, idx: pd.DatetimeIndex,
                   close_total: pd.DataFrame, sector_ret: pd.DataFrame | None,
                   vol: pd.DataFrame | None) -> tuple[dict[str, pd.DataFrame],
                                                      pd.DataFrame | None]:
    """#68-#79 for one family, and the family's grid event mask (reused by the insider legs)."""
    mask = _event_mask(events, idx, close_total.columns)
    if mask is None or not mask.to_numpy().any():
        logger.warning("`ic_sig_%s_*` skipped: the %s panel supplied no events.", fam, fam)
        return {}, None

    out: dict[str, pd.DataFrame] = {}
    age = days_since_last_true(mask)
    out[f"ic_sig_{fam}_age_days"] = age

    anchor = _anchor(close_total, mask)
    ret_since = (close_total / anchor.where(anchor > 0) - 1.0).replace(
        [np.inf, -np.inf], np.nan)
    out[f"ic_sig_{fam}_ret_since"] = ret_since

    resid = None
    if sector_ret is not None and not sector_ret.empty:
        # The sector's compounded return over the SAME span: a cumulative index divided by
        # its own value on the anchor day. `fillna(0.0)` inside the cumprod treats a missing
        # sector day as flat -- a NaN would otherwise kill the whole forward path, and the
        # ratio form means the level of the index never matters.
        cum = (1.0 + sector_ret.reindex(index=idx, columns=close_total.columns)
               .fillna(0.0)).cumprod()
        base = _anchor(cum, mask)
        span = cum / base.where(base > 0) - 1.0
        resid = (ret_since - span).replace([np.inf, -np.inf], np.nan)
        out[f"ic_sig_{fam}_resid_ret_since"] = resid

    if vol is not None and not vol.empty:
        # The move in units of the volatility of a span that long: daily vol x sqrt(age).
        # Dividing by the 20-day vol alone would make a 200-day move look ten times as
        # significant as a 2-day one of the same size.
        span_vol = vol.mul(np.sqrt(age.clip(lower=1.0)))
        numer = resid if resid is not None else ret_since
        out[f"ic_sig_{fam}_vol_scaled_move"] = (
            numer / span_vol.where(span_vol > 0)).replace([np.inf, -np.inf], np.nan)
    return out, mask


def _insider_price_legs(events: pd.DataFrame | None, mask: pd.DataFrame | None,
                        age: pd.DataFrame | None,
                        idx: pd.DatetimeIndex, close_split: pd.DataFrame | None,
                        close_total: pd.DataFrame, splits: pd.DataFrame | None,
                        excursion_lookback: int) -> dict[str, pd.DataFrame]:
    """#80-#82 -- the three legs that need the transaction price, not just its date."""
    out: dict[str, pd.DataFrame] = {}
    if mask is None:
        return out
    if events is not None and not events.empty and close_split is not None \
            and not close_split.empty and {"value", "shares"}.issubset(events.columns):
        e = events.dropna(subset=["ticker", "date"]).copy()
        e["_grid"] = snap_to_grid(e["date"], idx)
        e = e.dropna(subset=["_grid"])
        e["_value"] = pd.to_numeric(e["value"], errors="coerce")
        e["_shares"] = pd.to_numeric(e["shares"], errors="coerce")
        e = e[(e["_value"] > 0) & (e["_shares"] > 0)]
        if not e.empty:
            factor = future_split_factor(splits, e["ticker"], e["date"])
            n = int((np.abs(factor - 1.0) > 1e-9).sum())
            logger.info("insider cost anchor: %s of %s purchases restated onto today's split "
                        "basis (V5 trigger count); max factor %.1fx",
                        n, len(e), float(np.max(factor)) if len(factor) else 1.0)
            e["_shares_adj"] = e["_shares"] * factor
            val = e.pivot_table(index="_grid", columns="ticker", values="_value",
                                aggfunc="sum").reindex(index=idx,
                                                       columns=close_split.columns)
            sh = e.pivot_table(index="_grid", columns="ticker", values="_shares_adj",
                               aggfunc="sum").reindex(index=idx, columns=close_split.columns)
            num = val.fillna(0.0).rolling(COST_ANCHOR_WINDOW, min_periods=1).sum()
            den = sh.fillna(0.0).rolling(COST_ANCHOR_WINDOW, min_periods=1).sum()
            vw = num / den.where(den > 0)
            out["ic_sig_insider_price_vs_buy"] = (
                close_split / vw.where(vw > 0) - 1.0).replace([np.inf, -np.inf], np.nan)

    if age is not None:
        out["ic_sig_insider_max_runup_since_buy"] = _capped_excursion(
            close_total, mask, age, "max", excursion_lookback)
        out["ic_sig_insider_max_dd_since_buy"] = _capped_excursion(
            close_total, mask, age, "min", excursion_lookback)
    return out


def build_signal_conditioning_panel(
    events: dict[str, pd.DataFrame] | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    close_total: pd.DataFrame | None = None,
    close_split: pd.DataFrame | None = None,
    sector_ret: pd.DataFrame | None = None,
    ret: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    excursion_lookback: int = EXCURSION_LOOKBACK,
) -> pd.DataFrame:
    """Long-format conditioning panel (`f_<name>` per `EMISSION`).

    `events` is the sink the panel builders filled: `{family: [ticker, date, ...]}`. Empty
    when no family supplied events or `close_total` is absent -- every feature here is a
    price path measured from an event date, so neither input has a fallback.
    """
    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if not events or close_total is None or close_total.empty or idx.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    close_total = close_total.reindex(index=idx)
    if close_split is not None and not close_split.empty:
        close_split = close_split.reindex(index=idx, columns=close_total.columns)

    daily_ret = (ret.reindex(index=idx, columns=close_total.columns)
                 if ret is not None and not ret.empty else close_total.pct_change())
    vol = daily_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std()

    fields: dict[str, pd.DataFrame] = {}
    insider_mask = None
    for fam in FAMILIES:
        fam_fields, mask = _family_fields(fam, events.get(fam), idx, close_total,
                                          sector_ret, vol)
        fields.update(fam_fields)
        if fam == "insider":
            insider_mask = mask
    fields.update(_insider_price_legs(
        events.get("insider"), insider_mask, fields.get("ic_sig_insider_age_days"), idx,
        close_split, close_total, splits, excursion_lookback))

    for name in list(fields):
        frame = fields[name]
        if frame is None or frame.empty or not frame.notna().any().any():
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    logger.info("price-conditioning panel: %s of %s declared features built",
                len(fields), len(EMISSION))
    emission = {k: v for k, v in EMISSION.items() if k in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission)
