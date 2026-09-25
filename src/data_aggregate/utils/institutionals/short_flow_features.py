"""
short_flow_features.py  (src/data_aggregate/utils/institutionals/short_flow_features.py)
----------------------------------------------------------------------------------------
SHORT FLOW: FINRA RegSHO daily short-sale VOLUME (`ic_shortvol_*`) and SEC fails-to-deliver
(`ic_ftd_*`). Registry section 6, features #56-#67, plus the `ic_shortvol_market_coverage`
passthrough.

⚠ THIS IS SHORT-SALE VOLUME, NOT SHORT INTEREST, and the module is named for it (it was
`short_interest_features.py`). The distinction is not pedantic: short interest is a POSITION
outstanding at a settlement date, short volume is a FLOW of executions on one day, and the
report's `days_to_cover`, `short_interest_pct_float` and the "high and rising" regime all need
the position series. FINRA publishes those twice monthly and the repo does not fetch them, so
those three features are NOT BUILDABLE here and `ic_shortvol_days_to_cover` -- which the
previous version of this module emitted whenever two columns happened to be present -- is
gone rather than approximated. The DB table keeps its `sec_short_interest` name; renaming a
table is a migration, and the features are what a model reads.

VOLUME-WEIGHTED, NEVER AN AVERAGE OF DAILY RATIOS (registry #56-#58). `mean(short_i/total_i)`
over a window weights a 100k-share session the same as a 20m-share one; the quantity a model
wants is the share of the window's traded volume that was short:

    ic_shortvol_ratio_Nd = SUM(short_volume, N) / SUM(total_volume, N)

THE COVERAGE FLAG IS A MEASUREMENT, NOT A CONSTANT. RegSHO covers off-exchange (ATS and
OTC-reported) executions only, so its `total_volume` is a fraction of the consolidated tape.
Rather than shipping a 1.0 marker that says "remember this is partial",
`ic_shortvol_market_coverage` is `SUM(RegSHO total_volume, 20d) / SUM(tape volume, 20d)` -- the
actual, per-name, per-day share of the market the ratio was computed on, so a model can
condition on it and a reader can see when it moves.

POINT-IN-TIME:
  * RegSHO files are disseminated the NEXT morning -> every `ic_shortvol_*` leg is shifted
    `SHORTVOL_PUB_LAG` trading day. The shift is applied ONCE, to the ratio frames, and every
    derived leg (z-score, acceleration, the two price interactions) is built from the shifted
    frames -- so no leg can forget the lag.
  * SEC FTD files are published well after the settlement period -> `FTD_PUB_LAG` trading days.

⚠ AN ABSENT FTD ROW IS ZERO ONLY ON A DATE THE FILE COVERS. The file lists a security on the
days it had fails, so within a published settlement date a missing ticker means no fails; a
missing DATE means nothing was published and is NaN, never 0. The covered-date count is logged
so "the file has a hole" cannot present as "the market had no fails".

THE TWO PRICE INTERACTIONS (#62/#63) exist because a raw short ratio is ambiguous, which the
report is right about: heavy short flow into a FALLING price confirms selling pressure, while
heavy short flow into a RISING price is absorption or squeeze risk. Shipping them separately
lets the model learn the asymmetry instead of averaging it away. Both are one-sided products
of `max(0, z)` and the signed 20-day return, so each is zero whenever its regime is absent --
a real zero, not a missing value.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.errors import _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.common.xs import self_history_z
from src.data_aggregate.utils.institutionals.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.split_basis import split_adjust_frame
from src.data_store.schema import Tables


def _absent(df: pd.DataFrame | None, need: set[str] | None = None) -> bool:
    """True when `df` cannot be built from: missing, empty, or short a required column.

    The three-part test is the D5 entry contract stated once. `need` is the set the builder
    dereferences unconditionally -- a column it only uses `if present` does NOT belong here,
    or an optional projection turns into an empty panel.
    """
    if df is None or df.empty:
        return True
    return bool(need) and not need.issubset(df.columns)


logger = logging.getLogger(__name__)

#: RegSHO: day-t short volume is public on t+1.
SHORTVOL_PUB_LAG = 1

#: ~2 months of trading days. SEC FTD files are published well after the settlement period, so
#: the signal is lagged to its (conservative) availability date.
FTD_PUB_LAG = 40

#: Trailing self-history window for the two z-scores (#59, #66) and the minimum history before
#: one is emitted. 252 is a year; a half-year floor keeps the first year of a new source from
#: being blank rather than making a z out of 20 days.
Z_WINDOW = 252
Z_MIN_PERIODS = 126

#: #67: how many of the last 30 trading days had `ic_ftd_z252` above `Z_HIGH`. A persistent
#: settlement backlog is a different statement from one bad day.
PERSISTENCE_WINDOW = 30
Z_HIGH = 1.0

#: The three volume-weighted ratio windows (#56-#58). 20d is the one the z-score, the
#: acceleration and both price interactions are built on: 5d is too noisy to z-score and 60d
#: too slow to be the "current" regime.
RATIO_WINDOWS: tuple[int, ...] = (5, 20, 60)
BASE_WINDOW = 20

#: The 20-day price path the two interaction features condition on (#62/#63).
RET_WINDOW = 20

#: Emission (D27, registry section 0.10). The three ratios are the textbook sector-normed rates --
#: an 8% short-volume share means nothing without the sector's norm -- and section 0.10a measured
#: this family's peer legs as the one group with a HEALTHY peer fingerprint (clip 0.05%, NaN
#: 0%), unlike the insider family. The two z-scores are normalized by construction, so ranking
#: them is a second normalization that only loses the tail; the interaction products and the
#: persistence count are bounded or integer-valued; the three fail/turnover rates are skewed
#: dollar-free rates whose cross-sectional spread drifts with the market, so they take `_xs`.
EMISSION: dict[str, str] = {
    "ic_shortvol_ratio_5d": "raw+peers",
    "ic_shortvol_ratio_20d": "raw+peers",
    "ic_shortvol_ratio_60d": "raw+peers",
    "ic_shortvol_ratio_z252": "raw",
    "ic_shortvol_acceleration": "raw",
    "ic_shortvol_turnover_20d": "raw+xs",
    "ic_shortvol_high_x_weak_price": "raw",
    "ic_shortvol_high_x_strong_price": "raw",
    "ic_shortvol_market_coverage": "raw",
    "ic_ftd_pct_so": "raw+xs",
    "ic_ftd_to_adv20": "raw+xs",
    "ic_ftd_z252": "raw",
    "ic_ftd_persistence_30d": "raw",
}


def _min_periods(window: int) -> int:
    """Half the window, floored at 3: enough of the window present to be a window."""
    return max(3, window // 2)


def _pivot(hist: pd.DataFrame, value: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    wide = hist.pivot_table(index="date", columns="ticker", values=value, aggfunc="sum")
    wide.index = pd.to_datetime(wide.index).normalize()
    return wide.reindex(idx)


def _guard_coverage(cov: pd.DataFrame) -> pd.DataFrame:
    """NULL `ic_shortvol_market_coverage` above 1.0 -- a PHYSICAL impossibility, not an outlier.

    Off-exchange volume is a SUBSET of the consolidated tape, so a value above 1.0 means the
    numerator and the denominator are not describing the same security. After the split-basis
    restatement removed the corporate-action cases (2,213 breaching cells -> 179, a 92% cut),
    every survivor is a REUSED TICKER -- the RegSHO file is keyed on symbol, so a symbol that
    belonged to another company in that window carries that company's volume while
    `cube_part_prices` carries today's issuer:

        WTW   135 cells, 2018-08 -> 2019-05, max 3.71   Weight Watchers, before Willis Towers
                                                        Watson took the symbol
        AXON   27 cells, 2019-01 -> 2019-02, max 2.21   the TASR -> AAXN -> AXON rename chain
        GEN    17 cells, 2021-03 -> 2021-04, max 1.58   before Symantec/NortonLifeLock became
                                                        Gen Digital

    This is the same defect class the 2026-09-11 ticker-identity work closed for
    `insider_transactions` (where 2,046 Weight Watchers rows sat under `WTW`) and it is still
    open for `sec_short_interest`. 179 of 953,616 cells is 0.019%, and the guard makes the
    impossibility absent rather than plausible-looking -- D16's principle applied to a value.

    ⚠ ONLY THIS LEG IS GUARDED, and that is a stated limit rather than a claim of completeness.
    A reused-ticker window corrupts every `ic_shortvol_*` leg for that ticker, but coverage is
    the only one with a physical ceiling to detect it by; the rest are ratios that stay in
    range while describing the wrong company.
    """
    over = cov.gt(1.0)
    n = int(over.to_numpy().sum())
    if n:
        bad = [c for c in cov.columns if bool(over[c].any())]
        logger.info(
            "coverage guard: %s of %s non-null `ic_shortvol_market_coverage` cells "
            "above 1.0 -> nulled (off-exchange volume cannot exceed the tape; these "
            "are reused-ticker windows). Tickers: %s",
            f"{n:,}",
            f"{int(cov.notna().to_numpy().sum()):,}",
            ", ".join(sorted(bad)),
        )
    return cov.mask(over)


def _shortvol_fields(
    hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    shares_out: pd.DataFrame | None,
    close_total: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    splits: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """#56-#63 + the coverage measurement. Every leg is shifted by the publication lag."""
    short = _pivot(hist, "short_volume", idx)
    total = _pivot(hist, "total_volume", idx)
    f_dict: dict[str, pd.DataFrame] = {}

    ratios: dict[int, pd.DataFrame] = {}
    for w in RATIO_WINDOWS:
        mp = _min_periods(w)
        num = short.rolling(w, min_periods=mp).sum()
        den = total.rolling(w, min_periods=mp).sum()
        ratio = (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)
        ratios[w] = ratio.shift(SHORTVOL_PUB_LAG)
        f_dict[f"ic_shortvol_ratio_{w}d"] = ratios[w]

    base = ratios[BASE_WINDOW]
    z = self_history_z(base, window=Z_WINDOW, min_periods=Z_MIN_PERIODS)
    f_dict["ic_shortvol_ratio_z252"] = z
    f_dict["ic_shortvol_acceleration"] = ratios[BASE_WINDOW] - ratios[max(RATIO_WINDOWS)]

    if shares_out is not None and not shares_out.empty:
        so = shares_out.reindex(index=idx).reindex(columns=short.columns)
        turn = short.rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).sum()
        f_dict["ic_shortvol_turnover_20d"] = (turn / so.where(so > 0)).replace([np.inf, -np.inf], np.nan).shift(SHORTVOL_PUB_LAG)

    if close_total is not None and not close_total.empty:
        # The 20-day TOTAL return (`close_total`, never `close_split`): this is a return, and
        # the price contract reserves the split-only series for LEVELS.
        ret = close_total.reindex(index=idx).reindex(columns=short.columns).pct_change(RET_WINDOW)
        high = z.clip(lower=0.0)
        f_dict["ic_shortvol_high_x_weak_price"] = high * (-ret).clip(lower=0.0)
        f_dict["ic_shortvol_high_x_strong_price"] = high * ret.clip(lower=0.0)

    if volume is not None and not volume.empty:
        tape = volume.reindex(index=idx).reindex(columns=short.columns)
        num = (total * split_adjust_frame(splits, total)).rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).sum()
        den = tape.rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).sum()
        cov = (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)
        cov = _guard_coverage(cov)
        f_dict["ic_shortvol_market_coverage"] = cov.shift(SHORTVOL_PUB_LAG)
        live = cov.to_numpy(dtype="float64", na_value=np.nan).ravel()
        live = live[np.isfinite(live)]
        if len(live):
            p05, p50, p95 = np.percentile(live, [5, 50, 95])
            logger.info(
                "RegSHO market coverage (off-exchange share of tape volume): " "p50 %.1f%%, p05 %.1f%%, p95 %.1f%% over %s ticker-days",
                100 * p50,
                100 * p05,
                100 * p95,
                len(live),
            )
    return f_dict


def _fails_fields(
    fails_hist: pd.DataFrame, idx: pd.DatetimeIndex, shares_out: pd.DataFrame | None, volume: pd.DataFrame | None, splits: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """#64-#67. Zero-filled ONLY on the dates the FTD file covers -- see the module docstring."""
    fails = _pivot(fails_hist, "fails_quantity", idx)
    covered = pd.DatetimeIndex(to_day(fails_hist["date"]).dropna().unique())
    on_file = pd.Series(idx.isin(covered), index=idx)
    logger.info(
        "FTD file covers %s of %s trading days in the window (%.1f%%); an absent " "ticker on a covered date is 0 fails, an absent date is NaN",
        int(on_file.sum()),
        len(idx),
        100 * float(on_file.mean()),
    )
    # 0 on a covered date (the ticker simply had no fails), NaN on a date nothing was
    # published for. `pd.DataFrame(dict.fromkeys(...))` broadcasts the per-date flag to the
    # ticker axis explicitly rather than relying on a bare ndarray to align.
    covered_wide = pd.DataFrame({c: on_file for c in fails.columns}, index=idx)
    fails = fails.mask(covered_wide & fails.isna(), 0.0)

    f_dict: dict[str, pd.DataFrame] = {}
    pct_so = None
    if shares_out is not None and not shares_out.empty:
        so = shares_out.reindex(index=idx).reindex(columns=fails.columns)
        pct_so = (fails / so.where(so > 0)).replace([np.inf, -np.inf], np.nan)
        f_dict["ic_ftd_pct_so"] = pct_so.shift(FTD_PUB_LAG)
    if volume is not None and not volume.empty:
        adv = volume.reindex(index=idx).reindex(columns=fails.columns).rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).mean()
        # ⚠ SAME BASIS MISMATCH AS `market_coverage`: `fails_quantity` is an as-traded share
        # count and `adv` comes from yfinance `Volume`, which IS retroactively scaled by the
        # split ratio. Restate the fails onto the adjusted basis so the ratio is basis-free.
        fails_adj = fails * split_adjust_frame(splits, fails)
        to_adv = (fails_adj / adv.where(adv > 0)).replace([np.inf, -np.inf], np.nan)
        f_dict["ic_ftd_to_adv20"] = to_adv.shift(FTD_PUB_LAG)
    # The z-score prefers the share-count basis (a fail is a share count, and shares
    # outstanding is the only denominator that makes two names comparable); it falls back to
    # the ADV basis so the family is not lost when fundamentals are absent.
    basis = pct_so if pct_so is not None else (to_adv if volume is not None and not volume.empty else None)
    if basis is not None:
        z = self_history_z(basis, window=Z_WINDOW, min_periods=Z_MIN_PERIODS)
        # A zero standard deviation normally makes a z-score undefined. FTD has one economic
        # exception: if every source-covered observation in the applicable trailing window is
        # present and exactly zero, pressure is known to be neutral. Count the source's covered
        # dates rather than grid rows, because an absent SEC file date is unavailable, not zero.
        covered_basis = pd.DataFrame({c: on_file for c in basis.columns}, index=idx)
        # Denominators can have a legitimate warm-up (ADV20) or a later ticker-specific start
        # (shares outstanding). Applicability begins at that ticker's first computable basis;
        # any hole after that start remains a hole and blocks neutralization.
        applicable = covered_basis & basis.notna().cummax()
        expected = applicable.astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
        observed = basis.notna().astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
        zeros = basis.eq(0.0).astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
        neutral = applicable & basis.eq(0.0) & expected.ge(Z_MIN_PERIODS) & observed.eq(expected) & zeros.eq(expected)
        z = z.mask(z.isna() & neutral, 0.0)
        f_dict["ic_ftd_z252"] = z.shift(FTD_PUB_LAG)
        flag = (z > Z_HIGH).astype("float64").where(z.notna())
        f_dict["ic_ftd_persistence_30d"] = flag.rolling(PERSISTENCE_WINDOW, min_periods=_min_periods(PERSISTENCE_WINDOW)).sum().shift(FTD_PUB_LAG)
    return f_dict


def build_short_flow_feature_panel(
    frames: PriceFrames,
    short_history: pd.DataFrame | None,
    *,
    fails_history: pd.DataFrame | None = None,
    shares_out_history: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    availability: InstitutionalAvailability | None = None,
    sink=None,
) -> pd.DataFrame:
    """Long-format short-flow panel (`f_<name>` per `EMISSION`). Empty if neither source is
    available.

    `volume` (consolidated-tape daily volume) backs ADV20 and the coverage measurement;
    `shares_out_history` (fundamentals carrying `sharesOutstandingPit`) backs the two
    share-count-scaled features; `close_total` backs the two price interactions. Each is
    optional and its absence removes only the features that need it.

    ⚠ `frames` RATHER THAN FOUR UNPACKED FIELDS. `peer_dict`, `trading_index`, `volume` and
    `close_total` were all read off one `PriceFrames` at the call site. Naming the object makes
    the basis un-mistakable: there is one `close_split` and one `close_total` on it, and neither
    can arrive under the other's parameter name.

    ⚠ NO `frames.require(...)`, AND THAT IS MEASURED RATHER THAN FORGOTTEN. Every wide frame
    this builder reads sits behind an explicit `is None` guard, or is handed to a callee that
    documents `None` as a MEANING rather than an error -- `daily_market_cap`'s
    `level_factor=None` IS "S is 1.0 everywhere". `require` would turn each of those graceful
    degrades into a raise, which is exactly what its own docstring warns against.

    The non-frame arguments are KEYWORD-ONLY. A positional slip between two same-typed
    `pd.DataFrame | None` neighbours is a silent wrong-frame bug that reads as a plausible
    call; the keyword form makes it unrepresentable.
    """
    peer_dict = frames.peers
    trading_index = frames.trading_index
    volume = frames.volume
    close_total = frames.close_total
    # D5 entry guard. BOTH legs are optional here and the panel is built from whichever
    # arrived -- RegSHO short volume and fails-to-deliver are separate fetchers on separate
    # clocks -- so the guard is "neither", not "either".
    if _absent(short_history) and _absent(fails_history):
        return _empty_panel()

    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    shares_out = None
    if shares_out_history is not None and not shares_out_history.empty:
        # ⚠ `sharesOutstandingPit`: a fail and a short sale are counts of shares that existed
        # THEN, so the denominator must be the count that existed then -- the vendor-basis
        # column is restated to today's split basis and would read post-split names low.
        shares_out = fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", idx)
        if shares_out.empty or not shares_out.notna().any().any():
            shares_out = None

    fields: dict[str, pd.DataFrame] = {}
    if short_history is not None and not short_history.empty and {"short_volume", "total_volume"}.issubset(short_history.columns):
        fields.update(_shortvol_fields(short_history, idx, shares_out, close_total, volume, splits))
    if fails_history is not None and not fails_history.empty and "fails_quantity" in fails_history.columns:
        fields.update(_fails_fields(fails_history, idx, shares_out, volume, splits))

    for name in list(fields):
        frame = fields[name]
        if frame is None or frame.empty or not frame.notna().any().any():
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    if sink is not None:
        # This family has no ACTOR: FINRA reports the volume, never who traded it. It
        # contributes the confirmed-short-flow leg of the bearish family count and nothing
        # else -- see `cross_source_features` on why there is no bearish actor count.
        name = "ic_shortvol_high_x_weak_price"
        signal_fields = dict(fields)
        signal_masks: dict[str, pd.DataFrame] = {}
        if name in fields:
            columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
            raw = fields[name].reindex(index=idx, columns=columns)
            if close_total is not None and not close_total.empty:
                listed = close_total.reindex(index=idx, columns=columns).notna()
            else:
                listed = pd.DataFrame(True, index=idx, columns=columns)
            mask = (
                availability.derived_mask(
                    name,
                    idx,
                    columns,
                    dependencies=((Tables.short_interest, None),),
                    requirements=(listed, raw.notna()),
                )
                if availability is not None
                else raw.notna()
            )
            signal_masks[name] = mask
            signal_fields[name] = raw.where(mask)
        sink.keep_signals(signal_fields, signal_masks)
    emission = {k: v for k, v in EMISSION.items() if k in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission)
