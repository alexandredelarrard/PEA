"""
Deterministic numeric FINGERPRINT of the AGGREGATION layer (`src/data_aggregate/`).

Companion to `pipeline_fingerprint.py`, which fingerprints extraction + aggregation but
needs the raw SEC `data/sec_bulk_cache/companyfacts_CIK*.json` files. That cache is a
multi-GB download and is absent on most machines, so that guard cannot be replayed --
which left the aggregation refactor with no safety net at all. This module closes that
gap: it is SELF-CONTAINED after its first run.

    * fundamentals come from the DB table `fundamentals_history` ONCE and are then
      frozen to `aggregate_fingerprint_fundamentals.parquet` next to this file, so every
      later run (and every CI machine) replays byte-identical inputs with no DB and no
      SEC cache. Real filings are used deliberately: 237 columns across all 11 GICS
      sectors is what makes the sector gates (banks / REITs / insurance / energy) fire,
      and no synthetic frame reproduces that.
    * everything else (prices, dividends, earnings, proxies, 13F, insider, attention,
      short interest, fails) is a SEEDED synthetic source shaped like the real table.

COVERAGE. `pipeline_fingerprint` hashes 8 aggregation outputs and leaves 9 of the 13
panel builders unguarded, plus `build_peer_relative_panel`, both 13F quarter builders,
the commodity/currency twins and every ratio/standardize helper. Those are exactly the
functions the dedup sweep merges, so they are all fingerprinted here.

TWO KINDS OF KEY, on purpose:
  * `panel.*` / `label.*` -- PUBLIC entry points. Blind to how the work is organised
    internally, sensitive only to what comes out.
  * `prim.*` -- the PRIMITIVES that are about to be deduplicated (momentum, trailing
    vol, forward windows, the 5 cross-sectional standardizers, the ratio helpers, the
    two identical commodity/currency functions, both 13F quarter scaffolds). These are
    pinned BEFORE the merge; afterwards the same key is recomputed through the new
    unified helper and the hash must not move. The key survives the refactor; only the
    import behind it changes.

Regenerate the baseline with
    python -m tests.data_aggregate.aggregate_fingerprint
which writes `aggregate_fingerprint_baseline.json` next to this file.

RULE: the baseline may be regenerated ONLY in a commit that touches no `src/` file, or
in a PR that is exclusively a declared numeric change. Regenerating it alongside a
refactor destroys the very comparison it exists to make.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tests.data_aggregate.pipeline_fingerprint import frame_digest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("aggregate_fingerprint_baseline.json")
FUNDAMENTALS_CACHE = Path(__file__).with_name("aggregate_fingerprint_fundamentals.parquet")

SEED = 20260805
START, END = "2019-01-02", "2026-06-30"
TICKERS_PER_SECTOR = 2          # 11 GICS sectors -> 22 names; >= min_peers(3) by a wide margin
N_MANAGERS = 8                  # 13F filers in the synthetic holdings
FILING_LAG = 45


# --------------------------------------------------------------------------- #
# fixed inputs: fundamentals (DB once -> frozen parquet)                       #
# --------------------------------------------------------------------------- #
def _select_fundamentals() -> pd.DataFrame:
    """`TICKERS_PER_SECTOR` alphabetically-first tickers per GICS sector, so the draw is
    reproducible without an RNG and every sector-gated KPI family has names that pass its
    gate. Read from the DB only when the frozen parquet is absent."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")            # so `python -m ...aggregate_fingerprint` works standalone
    from src.data_store.store import DataStore
    from src.utils.db import get_engine

    fh = DataStore(get_engine()).load("fundamentals_history")
    if fh.empty:
        raise RuntimeError("fundamentals_history is empty -> cannot build the aggregation "
                           "fingerprint (run the extraction step first)")
    picked: list[str] = []
    for sector in sorted(fh["sector"].dropna().astype(str).unique()):
        names = sorted(fh.loc[fh["sector"].astype(str) == sector, "ticker"].unique())
        picked.extend(names[:TICKERS_PER_SECTOR])
    out = fh[fh["ticker"].isin(picked)].copy()
    out["as_of"] = pd.to_datetime(out["as_of"])
    return out.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def fundamentals() -> pd.DataFrame:
    """The frozen fundamentals slice. Written on first use, read verbatim afterwards, so
    the fingerprint is reproducible on a machine with neither the DB nor the SEC cache."""
    if FUNDAMENTALS_CACHE.exists():
        df = pd.read_parquet(FUNDAMENTALS_CACHE)
        df["as_of"] = pd.to_datetime(df["as_of"])
        return df
    df = _select_fundamentals()
    df.to_parquet(FUNDAMENTALS_CACHE, index=False)
    return df


# --------------------------------------------------------------------------- #
# fixed inputs: seeded synthetic sources                                       #
# --------------------------------------------------------------------------- #
def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def synthetic_prices(tickers: list[str], rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Seeded geometric random walks. A flat panel would leave every momentum /
    volatility / valuation-vs-price feature degenerate and the fingerprint blind."""
    idx = pd.bdate_range(START, END)
    steps = rng.normal(0.0004, 0.018, size=(len(idx), len(tickers)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tickers)
    return {
        "close": close,
        "open": close.shift(1).bfill() * (1 + rng.normal(0, 0.002, close.shape)),
        "high": close * (1 + np.abs(rng.normal(0, 0.006, close.shape))),
        "low": close * (1 - np.abs(rng.normal(0, 0.006, close.shape))),
        "volume": pd.DataFrame(rng.lognormal(15, 0.4, close.shape), index=idx, columns=tickers),
    }


def synthetic_dividends(tickers: list[str], idx: pd.DatetimeIndex,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Quarterly ex-dates for two thirds of the names (the rest are true non-payers, which
    must rank as a real 0 yield rather than NaN)."""
    payers = tickers[: max(1, len(tickers) * 2 // 3)]
    ex_dates = pd.date_range(idx[0], idx[-1], freq="QE")
    rows = []
    for t in payers:
        base = float(rng.uniform(0.15, 1.10))
        for k, d in enumerate(ex_dates):
            rows.append({"date": d, "ticker": t,
                         "dividends": round(base * (1.0 + 0.02 * k), 4)})
    return pd.DataFrame(rows)


def synthetic_earnings(tickers: list[str], idx: pd.DatetimeIndex,
                       rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(idx[0], idx[-1], freq="QE")
    rows = []
    for t in tickers:
        level = float(rng.uniform(0.5, 3.0))
        for k, d in enumerate(dates):
            est = level * (1.0 + 0.015 * k)
            act = est * (1.0 + float(rng.normal(0.01, 0.05)))
            rows.append({"ticker": t, "earnings_date": d,
                         "eps_estimate": round(est, 4), "eps_actual": round(act, 4),
                         "surprise_pct": round(100.0 * (act / est - 1.0), 4)})
    return pd.DataFrame(rows)


def synthetic_def14a(tickers: list[str], idx: pd.DatetimeIndex,
                     rng: np.random.Generator) -> pd.DataFrame:
    """One annual proxy per name per year, with the columns `governance_features` reads."""
    years = sorted({d.year for d in idx})
    rows = []
    for t in tickers:
        pay = float(rng.uniform(5e6, 3e7))
        since = int(rng.integers(1998, 2018))
        for k, y in enumerate(years):
            rows.append({
                "ticker": t, "as_of": pd.Timestamp(year=y, month=4, day=15),
                "ceo_total_comp": pay * (1.0 + 0.06 * k),
                "ceo_pay_ratio": float(rng.uniform(50, 400)),
                "ceo_equity_pay_pct": float(rng.uniform(0.3, 0.9)),
                "pct_independent_directors": float(rng.uniform(0.6, 0.95)),
                "pct_female_directors": float(rng.uniform(0.1, 0.5)),
                "board_size": float(rng.integers(7, 15)),
                "avg_board_tenure": float(rng.uniform(3, 14)),
                "say_on_pay_support_pct": float(rng.uniform(60, 99)),
                "insider_ownership_pct": float(rng.uniform(0.001, 0.08)),
                "ceo_is_founder": float(int(rng.integers(0, 2))),
                "ceo_since_year": float(since),
            })
    return pd.DataFrame(rows)


def synthetic_attention(tickers: list[str], idx: pd.DatetimeIndex,
                        rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily Wikipedia pageviews + WEEKLY Google Trends (the weekly->daily bounded ffill
    is a real code path). One ticker is deliberately absent from Trends so the rank-blend
    single-source fallback is exercised."""
    wiki = pd.DataFrame({
        "date": np.repeat(idx.to_numpy(), len(tickers)),
        "ticker": np.tile(np.array(tickers), len(idx)),
        "pageviews": rng.lognormal(7.0, 0.7, len(idx) * len(tickers)).round(0),
    })
    weekly = pd.date_range(idx[0], idx[-1], freq="W-SUN")
    gt_tickers = tickers[:-1]
    trends = pd.DataFrame({
        "date": np.repeat(weekly.to_numpy(), len(gt_tickers)),
        "ticker": np.tile(np.array(gt_tickers), len(weekly)),
        "search_interest": rng.integers(0, 101, len(weekly) * len(gt_tickers)).astype(float),
    })
    return wiki, trends


def synthetic_short_interest(tickers: list[str], idx: pd.DatetimeIndex,
                             rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(idx) * len(tickers)
    total = rng.lognormal(14.0, 0.5, n)
    short = total * rng.uniform(0.15, 0.65, n)
    si = pd.DataFrame({
        "date": np.repeat(idx.to_numpy(), len(tickers)),
        "ticker": np.tile(np.array(tickers), len(idx)),
        "short_volume": short.round(0), "total_volume": total.round(0),
        "short_interest": (total * rng.uniform(0.01, 0.20, n)).round(0),
        "avg_daily_volume": (total * rng.uniform(0.8, 1.2, n)).round(0),
    })
    # fails are SPARSE: a security is listed only on days it actually had fails
    keep = rng.random(n) < 0.15
    ftd = pd.DataFrame({
        "date": np.repeat(idx.to_numpy(), len(tickers))[keep],
        "ticker": np.tile(np.array(tickers), len(idx))[keep],
        "fails_quantity": rng.lognormal(8.0, 1.2, int(keep.sum())).round(0),
    })
    return si, ftd


def synthetic_insider(tickers: list[str], idx: pd.DatetimeIndex,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Forms 3/4/5 rows including the non-discretionary codes the builder must ignore."""
    codes = np.array(["P", "S", "A", "M", "F", "G"])
    n = 40 * len(tickers)
    days = pd.to_datetime(rng.choice(idx.to_numpy(), n))
    return pd.DataFrame({
        "ticker": rng.choice(np.array(tickers), n),
        "filing_date": days,
        "transaction_code": rng.choice(codes, n, p=[0.25, 0.35, 0.15, 0.1, 0.1, 0.05]),
        "value_usd": rng.lognormal(12.0, 1.5, n).round(2),
    })


def synthetic_13f(tickers: list[str], idx: pd.DatetimeIndex,
                  rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Manager-grain 13F: one row per (manager, ticker, quarter). Managers enter and exit
    so new_buyers / exiters / breadth_chg are non-degenerate. Half the CIKs form the
    'superinvestor' roster."""
    periods = pd.date_range(idx[0], idx[-1], freq="QE")
    ciks = [f"{1000000 + 7919 * i:010d}" for i in range(N_MANAGERS)]
    rows = []
    for p in periods:
        for i, cik in enumerate(ciks):
            for t in tickers:
                if rng.random() < 0.25:                  # manager not holding this quarter
                    continue
                shares = float(rng.lognormal(11.0, 0.8))
                rows.append({
                    "cik": cik, "period": p, "ticker": t,
                    "shares": round(shares, 0),
                    "value_usd": round(shares * float(rng.uniform(20, 400)), 2),
                    "call_value": round(shares * float(rng.uniform(0, 12)), 2),
                    "put_value": round(shares * float(rng.uniform(0, 9)), 2),
                    "filing_date": p + pd.Timedelta(days=FILING_LAG - 5 + i),
                })
    holdings = pd.DataFrame(rows)
    roster = {"cik_to_name": {c: f"Manager {k}" for k, c in enumerate(ciks[: N_MANAGERS // 2])}}
    return holdings, roster


def primitive_fixtures(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """The edge cases where the to-be-merged primitives differ from each other: a
    zero-dispersion row, an all-NaN row, a single-name row, an inf, a -100% return, a
    zero / negative / all-NaN denominator and a mismatched column set."""
    idx = pd.bdate_range("2024-01-01", periods=40)
    cols = [f"P{i}" for i in range(12)]
    m = pd.DataFrame(rng.normal(0, 1, (40, 12)), index=idx, columns=cols)
    m.iloc[3] = 7.0                                   # zero cross-sectional dispersion
    m.iloc[7] = np.nan                                # all-NaN row
    m.iloc[11] = np.nan
    m.iloc[11, 0] = 1.5                               # single-name row
    m.iloc[15, 2] = np.inf                            # +inf cell
    m.iloc[16, 3] = -np.inf
    m.iloc[20, 4] = 0.0

    num = pd.DataFrame(rng.normal(5, 2, (40, 12)), index=idx, columns=cols)
    den = pd.DataFrame(rng.normal(1, 3, (40, 12)), index=idx, columns=cols)
    den.iloc[0] = 0.0                                 # zero denominator row
    den.iloc[1] = -1.0                                # negative denominator row
    den.iloc[2] = np.nan
    den_short = den.iloc[:, :8].copy()                # mismatched columns
    den_short.columns = cols[:8]

    ret = pd.DataFrame(rng.normal(0.0005, 0.02, (40, 12)), index=idx, columns=cols)
    ret.iloc[5, 1] = -1.0                             # exactly -100%: log1p edge
    ret.iloc[6, 2] = -1.4                             # below -100%: must be floored
    ret.iloc[9] = np.nan

    price = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, 4)), axis=0)),
                         index=idx, columns=["SPY", "CL=F", "GC=F", "USDEUR=X"])
    return {"matrix": m, "num": num, "den": den, "den_short": den_short,
            "returns": ret, "other_close": price}


# --------------------------------------------------------------------------- #
# the fingerprint                                                              #
# --------------------------------------------------------------------------- #
def compute() -> dict:
    from src.data_aggregate.utils.extras.attention_features import build_combined_attention_panel
    from src.data_aggregate.utils.target.betas import estimate_all_betas
    from src.data_aggregate.utils.assemble.composites import build_composites
    from src.data_aggregate.utils.fundamentals.dividend_features import build_dividend_feature_panel
    from src.data_aggregate.utils.fundamentals.earnings_features import build_earnings_feature_panel
    from src.data_aggregate.utils.fundamentals.employee_features import build_employee_feature_panel
    from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
    from src.data_aggregate.utils.common.prices import (
        forward_compound, forward_cumchange, forward_return, momentum_characteristic,
        trailing_vol,
    )
    from src.data_aggregate.utils.momentum.features import (
        build_feature_panel, compute_raw_features,
    )
    from src.data_aggregate.utils.fundamentals.fundamental_features import build_fundamental_feature_panel
    from src.data_aggregate.utils.extras.governance_features import build_governance_feature_panel
    from src.data_aggregate.utils.extras.insider_features import build_insider_feature_panel
    from src.data_aggregate.utils.extras.institutional_features import (
        _quarter_features, build_institutional_feature_panel,
    )
    from src.data_aggregate.utils.common.frames import ratio, safe_div, sanitize
    from src.data_aggregate.utils.common.panel import build_peer_relative_panel
    from src.data_aggregate.utils.common.xs import (
        winsorize_xs, xs_rank_pct, xs_standardize, xs_z,
    )
    from src.data_aggregate.utils.fundamentals.sector_features import build_sector_feature_panel
    from src.data_aggregate.utils.extras.short_interest_features import (
        build_short_interest_feature_panel,
    )
    from src.data_aggregate.utils.extras.superinvestor_features import (
        _super_quarter_features, _weight_map, build_superinvestor_feature_panel,
    )
    from src.data_aggregate.utils.target.targets import (
        build_targets_multi, cross_sectional_rank, cross_sectional_zscore,
    )

    rng = _rng()
    fund = fundamentals()
    tickers = sorted(fund["ticker"].unique())
    px = synthetic_prices(tickers, rng)
    close, idx = px["close"], px["close"].index
    returns = close.pct_change(fill_method=None)
    sector_ret = returns.rolling(5).mean().bfill()             # deterministic stand-in
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    div = synthetic_dividends(tickers, idx, rng)
    earn = synthetic_earnings(tickers, idx, rng)
    proxies = synthetic_def14a(tickers, idx, rng)
    wiki, trends = synthetic_attention(tickers, idx, rng)
    short_hist, ftd = synthetic_short_interest(tickers, idx, rng)
    insider = synthetic_insider(tickers, idx, rng)
    holdings, roster = synthetic_13f(tickers, idx, rng)
    fx = primitive_fixtures(rng)

    out: dict[str, dict] = {}

    # ---- the frozen input itself: a DB change must fail LOUDLY and by name ---- #
    out["input.fundamentals_slice"] = frame_digest(fund)

    # ---- PUBLIC entry points: every panel builder ---- #
    out["panel.raw_features"] = frame_digest(pd.concat(
        {k: v for k, v in sorted(compute_raw_features(
            close, px["open"], sector_ret, high=px["high"], low=px["low"],
            volume=px["volume"], seasonal_horizons=[30, 60, 90]).items())
         if isinstance(v, pd.DataFrame)}, axis=1))
    out["panel.price"] = frame_digest(build_feature_panel(
        close, px["open"], sector_ret, "rank", px["high"], px["low"], px["volume"],
        [30, 60, 90]))
    fp = build_fundamental_feature_panel(fund, peers, idx, stock_close=close,
                                         earnings_history=earn)
    out["panel.fundamental"] = frame_digest(fp)
    sp = build_sector_feature_panel(fund, peers, idx)
    out["panel.sector"] = frame_digest(sp)
    out["panel.earnings"] = frame_digest(build_earnings_feature_panel(
        earn, peers, idx, stock_close=close))
    out["panel.employee"] = frame_digest(build_employee_feature_panel(
        fund, peers, idx, fundamentals_history=fund))
    out["panel.dividend"] = frame_digest(build_dividend_feature_panel(
        div, peers, idx, stock_close=close, fundamentals_history=fund))
    out["panel.governance"] = frame_digest(build_governance_feature_panel(
        proxies, peers, idx, fundamentals_history=fund))
    out["panel.attention"] = frame_digest(build_combined_attention_panel(
        wiki, trends, peers, idx))
    out["panel.short_interest"] = frame_digest(build_short_interest_feature_panel(
        short_hist, peers, idx, fails_history=ftd, volume=px["volume"]))
    out["panel.institutional"] = frame_digest(build_institutional_feature_panel(
        holdings, peers, idx, shares_out_history=fund, stock_close=close))
    out["panel.superinvestor"] = frame_digest(build_superinvestor_feature_panel(
        holdings, roster, peers, idx, shares_out_history=fund, stock_close=close))
    out["panel.insider"] = frame_digest(build_insider_feature_panel(
        insider, peers, idx, shares_out_history=fund, stock_close=close))

    # ---- composites over the merged panel ---- #
    merged = fp.merge(sp, on=["date", "ticker"], how="outer")
    cfg = yaml.safe_load((ROOT / "configs" / "build_cube.yml").read_text(encoding="utf-8"))
    groups = next(v["composites"]["groups"] for v in cfg.values()
                  if isinstance(v, dict) and "composites" in v)
    comp = build_composites(merged, groups, method="zscore")
    out["panel.composites"] = frame_digest(
        comp[["date", "ticker"] + sorted(c for c in comp.columns if c.startswith("comp_"))])

    # ---- betas + the labels the model actually trains on ---- #
    factor_panel = pd.DataFrame({
        "market": returns.mean(axis=1),
        "momentum": momentum_characteristic(close).mean(axis=1),
    })
    betas = estimate_all_betas(returns, factor_panel)
    out["panel.betas"] = frame_digest(pd.concat(
        {k: v for k, v in betas.items() if isinstance(v, pd.DataFrame)}, axis=1))
    # `min_names` is lowered from the production 20 because this harness runs a
    # 22-name cross-section; it gates which DAYS survive, not how the residual is
    # computed, so the code under test is unaffected.
    # `stock_ret` is REQUIRED: every label is a forward COMPOUNDED total return now rather
    # than a close-to-close price ratio. This fixture is a dividend-free random walk, so
    # `close_split == close_total` here and the two formulations differ only by compounding
    # convention -- which IS a real digest move, and a documented one (see 4e in the plan).
    targets = build_targets_multi(
        close, betas, factor_panel, macro_cols=[],
        horizons=(30, 60, 90), labels=("rank", "zscore"), min_names=5,
        sector_groups={"sector": dict(zip(fund["ticker"], fund["sector"].astype(str)))},
        stock_ret=returns)
    for horizon, by_label in targets.items():
        for label, frame in by_label.items():
            out[f"label.{label}_h{horizon}"] = frame_digest(frame)

    # ---- PRIMITIVES about to be deduplicated (see the module docstring) ---- #
    m, num, den = fx["matrix"], fx["num"], fx["den"]
    ret_fx = fx["returns"]

    out["prim.momentum_characteristic"] = frame_digest(momentum_characteristic(close))
    # the inline copy that `features.mom_12_1` used to carry -- kept as an independent
    # REFERENCE expression so the two can be asserted equal (see
    # test_momentum_dedup_is_provably_identical); the feature now calls the shared helper.
    out["prim.mom_12_1_inline"] = frame_digest(sanitize(close.shift(21) / close.shift(252) - 1.0))
    out["prim.trailing_vol"] = frame_digest(pd.concat({
        "vol_21": sanitize(trailing_vol(returns, 21)),
        "vol_63": sanitize(trailing_vol(returns, 63)),
        "resvol_63": -trailing_vol(returns, 63),
    }, axis=1))
    # A HARD-CODED COPY of `du.daily_returns`, deliberately: it is the independent reference
    # the shared helper is asserted against. In production this is fed `close_total`; the
    # fixture is dividend-free, so the same frame stands in for both bases.
    out["prim.daily_returns"] = frame_digest(close.pct_change(fill_method=None))

    out["prim.forward_windows"] = frame_digest(pd.concat({
        "compound_h20": forward_compound(ret_fx, 20),
        "cumchange_h20": forward_cumchange(ret_fx, 20),
        # `forward_return` is now contract-limited to TOTAL-RETURN INDICES; SPY here stands
        # in for the macro `equity_tr` leg, which is exactly that. The stock labels moved to
        # `forward_compound` and no longer call it.
        "return_h20": forward_return(fx["other_close"].reindex(columns=["SPY"]), 20),
        # the seasonal feature's PARTIAL-window policy (min_periods = round(0.6h)),
        # which differs from the target's full-window policy above
        "compound_h20_partial": forward_compound(ret_fx, 20, min_periods=12),
    }, axis=1))

    # one z + one rank implementation, each call site keeping ITS clip and ITS
    # zero-dispersion policy (see utils/common/xs.py). The two raw `.rank` spellings stay
    # as independent references that xs_rank_pct must reproduce.
    out["prim.xs_standardize"] = frame_digest(pd.concat({
        "factors_xs_z_clip4": xs_z(m, clip=4.0),
        "features_rank": xs_standardize(m, "rank"),
        "features_zscore_clip3": xs_standardize(m, "zscore"),
        "targets_rank_min5": cross_sectional_rank(m, min_names=5),
        "targets_zscore_min5": cross_sectional_zscore(m, min_names=5),
        "winsorize_xs": winsorize_xs(m),
        "rank_pct_plain": m.rank(axis=1, pct=True),
        "rank_pct_average": m.rank(axis=1, pct=True, method="average"),
    }, axis=1))

    out["prim.ratio_helpers"] = frame_digest(pd.concat({
        "ratio": ratio(num, den),
        "ratio_positive_den": ratio(num, den, positive_den=True),
        "ratio_mismatched_cols": ratio(num, fx["den_short"]),
        "clean_ratio": sanitize(num / den),
        "safe": sanitize(num / den),
    }, axis=1))
    out["prim.safe_div"] = frame_digest(pd.DataFrame({
        "plain": safe_div(num["P0"], den["P0"]),
        "positive_den": safe_div(num["P0"], den["P0"], True),
        "none_den": safe_div(num["P0"], None),
    }))

    # `price_column_returns` is GONE. Its whole job was remapping factor name -> price COLUMN
    # ({"oil": "CL=F"}) while the commodity/FX series sat inside the `prices` panel; they now
    # live in `prices_macro` under their factor names, so the remap is the identity and
    # StepCubeTarget._asset_factors just takes the pct_change. Digest the surviving
    # expression, keyed by the factor NAME the panel uses, so the fingerprint still covers the
    # arithmetic that feeds the commodity/currency factors.
    _macro_close = fx["other_close"].rename(
        columns={"SPY": "equity_tr", "CL=F": "oil", "GC=F": "gold", "USDEUR=X": "fx_usdeur"})
    out["prim.macro_factor_returns"] = frame_digest(
        _macro_close[["oil", "gold", "fx_usdeur"]].pct_change())

    out["prim.quarter_features"] = frame_digest(
        _quarter_features(holdings).sort_values(["ticker", "as_of"]).reset_index(drop=True))
    out["prim.super_quarter_features"] = frame_digest(
        _super_quarter_features(holdings, _weight_map(roster))
        .sort_values(["ticker", "as_of"]).reset_index(drop=True))

    out["prim.pit"] = frame_digest(pd.concat({
        "shares_outstanding": fundamentals_to_daily(fund, "sharesOutstanding", idx),
        "total_revenue": fundamentals_to_daily(fund, "totalRevenue", idx),
        "free_cashflow": fundamentals_to_daily(fund, "freeCashflow", idx),
        "net_income": fundamentals_to_daily(fund, "netIncome", idx),
        "market_cap": daily_market_cap(fund, close, level_factor=None),
    }, axis=1))

    # `build_peer_relative_panel` called DIRECTLY, including the two degenerate field
    # shapes its coercion path exists for: an all-NaN field and an object-dtype field
    # carrying a stray Python None.
    objf = num.astype(object).copy()
    objf.iloc[4, 5] = None
    objf.iloc[8, 1] = "n/a"
    out["prim.peer_relative_panel"] = frame_digest(build_peer_relative_panel(
        {"plain": num, "all_nan": num * np.nan, "objecty": objf},
        {c: {p: 1.0 for p in num.columns if p != c} for c in num.columns}))

    out["_meta"] = {"tickers": tickers, "seed": SEED, "start": START, "end": END,
                    "fundamentals_rows": int(len(fund)),
                    "fundamentals_cols": int(len(fund.columns))}
    return out


def main() -> None:
    fp = compute()
    BASELINE.write_text(json.dumps(fp, indent=1, sort_keys=True), encoding="utf-8")
    keys = [k for k in fp if not k.startswith("_")]
    print(f"wrote {BASELINE.name}: {len(keys)} fingerprinted outputs")
    for k in sorted(keys):
        d = fp[k]
        print(f"  {k:34} rows={d['rows']:7d} cols={d['cols']:5d} {d['hash'][:12]}")


if __name__ == "__main__":
    main()
