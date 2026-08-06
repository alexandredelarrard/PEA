"""Shared test fixtures for the data_aggregate suite.

Two flavours of fixture live here, matching the project testing conventions:

* synthetic, known-truth data -> used ONLY to verify the mathematical
  correctness of the beta estimator (you cannot check that a regression
  recovers a loading without knowing the true loading).
* real-data, small-sample -> used for every economic / sanity check, so the
  tests see the real NaNs, delistings and late IPOs that the estimator has to
  survive (this is exactly what surfaced the sector-NaN truncation bug).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `import src...` work no matter where pytest is invoked from.
ROOT = Path(__file__).resolve().parents[1]  # .../stock_pick_strat
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Every LIVE test (Roic, SEC, Motley Fool, yfinance) needs the combined corporate CA bundle
# that main.py builds at startup; without it requests/curl_cffi raise
# CERTIFICATE_VERIFY_FAILED behind the TLS proxy and the test looks like a source-not-covering
# failure. Done at import time, before any fetcher module imports curl_cffi (which freezes its
# bundle at import). Idempotent and offline-safe.
from src.utils.ssl_setup import configure_corporate_ca  # noqa: E402

configure_corporate_ca()

# The DB credentials live in `.env`, which the pipeline loads via `Context._load_env`. Tests
# build the engine directly (no Context), so without this `database_url()` fell back to its
# `pea`/`pea` defaults and every real-data fixture died with
# `FATAL: password authentication failed for user "pea"` — an ERROR that looked like a
# feature bug rather than a missing credential. Loaded by explicit path so it works whatever
# directory pytest is invoked from.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

DATA = ROOT / "data"


def _store():
    """DB-backed data store for the real-data fixtures (DB is the source of
    truth now that the pipeline is DB-only). Skips a fixture when its table is
    empty — mirrors the old 'skip if parquet absent' behaviour.

    An UNREACHABLE database also skips rather than errors: these are integration tests, and
    a machine without the Postgres container should report 'skipped', not a wall of
    connection tracebacks that hides real failures."""
    from src.utils.db import get_engine
    from src.data_store.store import DataStore
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception as exc:                                        # noqa: BLE001
        pytest.skip(f"database unavailable: {type(exc).__name__}")
    return DataStore(engine)

MARKET = "SPY"
OTHER_TICKERS = ["SPY", "^VIX"]
SUBSET_SIZE = 100  # keep the real-data pipeline fast but keep a real cross-section


# --------------------------------------------------------------------------- #
# Real data (small sample)                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def real_frames():
    """Wide close / returns matrices from the real price parquet, subset for
    speed but guaranteed to contain AMD (the ticker under investigation)."""
    from src.data_aggregate.utils.common import data_utils as du

    prices = _store().load("prices")
    if prices.empty:
        pytest.skip("prices table is empty")
    raw = du.prices_long_to_multiindex(prices)
    close = du.extract_field(raw, "Close")
    close = close.loc[close[MARKET].notna()]
    returns = du.daily_returns(close)

    stock_close = close.drop(columns=[c for c in OTHER_TICKERS if c in close.columns])
    stock_ret = returns.drop(columns=[c for c in OTHER_TICKERS if c in returns.columns])

    cols = list(stock_ret.columns)
    keep = (["AMD"] if "AMD" in cols else []) + [c for c in cols if c != "AMD"][:SUBSET_SIZE]
    return {
        "mkt_ret": returns[MARKET],
        "stock_close": stock_close[keep],
        "stock_ret": stock_ret[keep],
        # full close (ALL tickers incl. commodity/FX proxies like CL=F/GC=F/USDEUR=X,
        # which are not in the 100-stock subset) for the commodity/currency factors
        "close_full": close,
    }


@pytest.fixture(scope="session")
def real_pipeline(real_frames):
    """End-to-end real-data aggregate pieces computed once: peers, sector
    returns, factor panel, rolling betas and multi-horizon targets."""
    from src.data_aggregate.utils.target.betas import estimate_all_betas
    from src.data_aggregate.utils.target.targets import build_targets_multi
    from src.data_aggregate.utils.common.prices import price_column_returns
    from src.data_aggregate.utils.target.factors import (
        build_style_factor_returns,
        macro_change_factors,
        assemble_factor_panel,
    )
    from src.data_peers.utils.sector_peers import (
        build_peer_dict,
        compute_sector_returns,
    )

    stock_close = real_frames["stock_close"]
    stock_ret = real_frames["stock_ret"]
    mkt_ret = real_frames["mkt_ret"]
    close_full = real_frames["close_full"]

    store = _store()
    fundamentals = store.load("fundamentals_history")
    fundamentals = None if fundamentals.empty else fundamentals
    macro = store.load("macro")
    macro = None if macro.empty else macro

    peers = build_peer_dict(stock_ret, top_k=20, weighting="corr", min_obs=120)
    sector_ret = compute_sector_returns(stock_ret, peers)

    # mirror step_build_cube.build_factor_panel: market + style + commodity + currency + macro
    style = build_style_factor_returns(stock_close, stock_ret, fundamentals, 63)
    if macro is not None:
        macro_chg = macro_change_factors(macro, stock_close.index)
    else:
        macro_chg = pd.DataFrame(index=stock_close.index)
    commodity_returns = price_column_returns(close_full, {"oil": "CL=F", "gold": "GC=F"})
    currency_returns = price_column_returns(close_full, {"USD/EUR": "USDEUR=X"})
    factor_panel, macro_cols = assemble_factor_panel(
        mkt_ret, style, commodity_returns, currency_returns, macro_chg)

    betas = estimate_all_betas(
        stock_ret, factor_panel, sector_ret,
        window=63, min_obs=40, ridge=5.0, step=5,
    )

    horizons = (5, 20, 60)
    # one label via the multi-label builder (the single-label `build_targets` twin was
    # deleted); unwrap {h: {"rank": df}} -> {h: df} so the fixture's shape is unchanged
    # and its five consumer tests need no edits.
    _multi = build_targets_multi(
        stock_close, stock_ret, peers, betas, factor_panel, macro_cols,
        horizons=horizons, labels=("rank",), min_names=20,
    )
    labels_rank = {h: by_label["rank"] for h, by_label in _multi.items()}

    return {
        "peers": peers,
        "sector_ret": sector_ret,
        "factor_panel": factor_panel,
        "macro_cols": macro_cols,
        "betas": betas,
        "labels_rank": labels_rank,
        "horizons": horizons,
        "stock_close": stock_close,
        "stock_ret": stock_ret,
    }


@pytest.fixture(scope="session")
def fundamental_panel(real_frames):
    """Peer-relative fundamental feature panel on the real (canonical) history,
    plus the inputs needed to sanity-check it against the target."""
    from src.data_aggregate.utils.fundamentals.fundamental_features import build_fundamental_feature_panel
    from src.data_peers.utils.sector_peers import build_peer_dict

    fundamentals = _store().load("fundamentals_history")
    if fundamentals.empty:
        pytest.skip("fundamentals_history table is empty")
    stock_close = real_frames["stock_close"]
    stock_ret = real_frames["stock_ret"]
    peers = build_peer_dict(stock_ret, top_k=20, weighting="corr", min_obs=120)
    panel = build_fundamental_feature_panel(
        fundamentals, peers, stock_close.index, stock_close=stock_close,
    )
    return {"panel": panel, "fundamentals": fundamentals,
            "peers": peers, "stock_close": stock_close}


# --------------------------------------------------------------------------- #
# Synthetic data (known truth) for estimator-correctness tests                 #
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_factor_model():
    """Return (y, shared, sector, true_betas) with KNOWN loadings so we can
    assert the estimator recovers them."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2018-01-01", periods=n)

    market = pd.Series(rng.normal(0.0004, 0.010, n), index=dates, name="market")
    momentum = pd.Series(rng.normal(0.0, 0.006, n), index=dates, name="momentum")
    value = pd.Series(rng.normal(0.0, 0.006, n), index=dates, name="value")
    sector = pd.Series(rng.normal(0.0003, 0.009, n), index=dates, name="sector")

    shared = pd.concat([market, momentum, value], axis=1)

    true_betas = {"market": 1.20, "momentum": 0.50, "value": -0.30, "sector": 0.40}
    idiosyncratic = rng.normal(0.0, 0.004, n)
    y = (
        true_betas["market"] * market
        + true_betas["momentum"] * momentum
        + true_betas["value"] * value
        + true_betas["sector"] * sector
        + idiosyncratic
    )
    y = pd.Series(y, index=dates, name="STOCK")
    return y, shared, sector, true_betas
