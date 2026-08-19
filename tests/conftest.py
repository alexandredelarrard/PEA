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

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.data_store.errors import TableEmptyError  # noqa: E402
from src.data_store.schema import name_of, resolve  # noqa: E402
from src.data_store.store import DataStore  # noqa: E402
from src.utils.db import get_engine  # noqa: E402

DATA = ROOT / "data"


# --------------------------------------------------------------------------- #
# The shared UNIT-level store: a REAL DataStore on in-memory SQLite.           #
# --------------------------------------------------------------------------- #
# This replaces the ~18 hand-rolled `_FakeStore` / `_FakePartStore` duck types scattered
# across tests/. Those fakes each re-implemented a slightly different subset of the store
# API -- only ONE of them accepted `where=` -- so every change to the store surface broke
# them one at a time, and four production modules still carry a whole second "no engine"
# code path that exists PURELY to keep them working (see superinvestor_features.py's
# `getattr(store, "engine", None)` branch).
#
# Deliberately NOT the Postgres fixture below: `_store()` SKIPS when the container is down,
# which silently greens the suite on a machine with no DB. This one never skips, so a unit
# test that depends on store behaviour actually runs everywhere.
#
# SQLite is a genuine dialect for this purpose -- `DataStore` adapts its upsert per dialect
# (`ON CONFLICT DO UPDATE` on both Postgres and SQLite), which is exactly the path under test.
@pytest.fixture
def sqlite_store():
    """A real `DataStore` backed by a fresh in-memory SQLite DB (never skips)."""
    # StaticPool + one shared connection: the default pool hands out a NEW (and therefore
    # EMPTY) in-memory database per checkout, so a table written by `save` would vanish
    # before the next `load` -- the failure mode looks like a broken store, not a fixture bug.
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    store = DataStore(engine)
    yield store
    engine.dispose()


class FakeStore:
    """THE in-memory store double, for the cases `sqlite_store` cannot serve:

      * a vector column (`ticker_embeddings.embedding`, `earning_calls_embedding.embedding`) --
        SQLite's driver refuses to bind a Python list;
      * a test that asserts the ORDER of writes, which a real store does not expose.

    Prefer `sqlite_store`. This exists so those cases share ONE definition that tracks the real
    contract, instead of ten near-identical duck types that each implemented a different subset
    (only one of them accepted `where=`) and broke one at a time whenever the store surface moved.

    Faithful on the parts that matter: keyed by `name_of` so a `Table` and its name are the same
    table, `where` does equality/IN, and a read of an absent-or-empty table RAISES unless
    `optional=True` -- the contract that makes a missing table a visible fault.
    """

    def __init__(self, tables: dict | None = None):
        self.t: dict[str, pd.DataFrame] = {name_of(k): v.copy()
                                           for k, v in (tables or {}).items()}
        self.writes: list[tuple[str, str, pd.DataFrame]] = []   # (op, table, df) in call order

    @staticmethod
    def _filter(df, where):
        for col, val in (where or {}).items():
            df = (df[df[col].isin(list(val))]
                  if isinstance(val, (list, tuple, set, frozenset)) else df[df[col] == val])
        return df

    # -- introspection -- #
    def exists(self, table) -> bool:
        return name_of(table) in self.t

    def columns(self, table) -> list[str]:
        df = self.t.get(name_of(table))
        return [] if df is None else list(df.columns)

    def row_count(self, table) -> int:
        df = self.t.get(name_of(table))
        return 0 if df is None else len(df)

    def distinct(self, table, column, **kw) -> list:
        df = self.t.get(name_of(table))
        return [] if df is None or df.empty else df[column].dropna().unique().tolist()

    def bounds(self, table, column=None):
        df = self.t.get(name_of(table))
        col = column or resolve(table).date_col
        if df is None or df.empty or col not in df.columns:
            return (None, None)
        return (df[col].min(), df[col].max())

    def max_date(self, table, column=None):
        lo, hi = self.bounds(table, column)
        return None if hi is None else pd.Timestamp(hi).normalize()

    def max_date_by(self, table, key_col, date_col=None) -> dict:
        """Per-key latest stored date. The grouped counterpart of `max_date` -- what
        `resume_since` and the macro freshness gate resolve their frontier with, so the double
        needs it or those paths are untestable without a DB. Empty dict when the table or
        either column is absent, matching the real store's "nothing stored yet" contract."""
        df = self.t.get(name_of(table))
        col = date_col or resolve(table).date_col
        if df is None or df.empty or col not in df.columns or key_col not in df.columns:
            return {}
        g = df.dropna(subset=[col]).groupby(key_col)[col].max()
        return {str(k): pd.Timestamp(v).normalize() for k, v in g.items() if pd.notna(v)}

    # -- reads -- #
    def load(self, table, columns=None, limit=None, where=None, *, optional=False, **kw):
        name = name_of(table)
        df = self._filter(self.t.get(name, pd.DataFrame()), where)
        if df.empty:
            if optional:
                return None
            raise TableEmptyError(name, where)
        if columns:
            df = df[list(columns)]
        return (df.head(limit) if limit else df).copy().reset_index(drop=True)

    # -- writes -- #
    def save(self, table, df, pk=None):
        name = name_of(table)
        self.writes.append(("save", name, df.copy()))
        both = pd.concat([self.t.get(name), df], ignore_index=True)
        pk = list(pk or resolve(table).pk)
        keys = [c for c in pk if c in both.columns] or None
        self.t[name] = (both.drop_duplicates(subset=keys, keep="last") if keys else both
                        ).reset_index(drop=True)
        return len(df)

    def replace(self, table, df, chunksize=200_000):
        name = name_of(table)
        self.writes.append(("replace", name, df.copy()))
        self.t[name] = df.copy().reset_index(drop=True)
        return len(df)

    def bulk_seed(self, table, df):
        name = name_of(table)
        self.writes.append(("bulk_seed", name, df.copy()))
        self.t[name] = pd.concat([self.t.get(name), df], ignore_index=True)
        return len(df)

    def delete(self, table, where):
        name = name_of(table)
        df = self.t.get(name)
        if df is None or df.empty:
            return 0
        drop = self._filter(df, where).index
        self.t[name] = df.drop(index=drop).reset_index(drop=True)
        return len(drop)

    def drop(self, table):
        self.t.pop(name_of(table), None)

    def ensure_columns(self, table, df):
        return []

    # -- write assertions -- #
    def saved_frames(self, table=None) -> list[pd.DataFrame]:
        """The frames passed to `save`, in call order (optionally for one table only)."""
        want = None if table is None else name_of(table)
        return [df for op, name, df in self.writes
                if op == "save" and (want is None or name == want)]


def _store():
    """DB-backed data store for the real-data fixtures (DB is the source of
    truth now that the pipeline is DB-only). Skips a fixture when its table is
    empty — mirrors the old 'skip if parquet absent' behaviour.

    An UNREACHABLE database also skips rather than errors: these are integration tests, and
    a machine without the Postgres container should report 'skipped', not a wall of
    connection tracebacks that hides real failures."""
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception as exc:                                        # noqa: BLE001
        pytest.skip(f"database unavailable: {type(exc).__name__}")
    return DataStore(engine)

SUBSET_SIZE = 100  # keep the real-data pipeline fast but keep a real cross-section
# The market / commodity / FX series are NOT in `prices` any more -- they are series in
# `prices_macro`, so the fixtures read them from there instead of filtering them out of the
# equity frame. MARKET / OTHER_TICKERS / FACTOR_PROXY_TICKERS are gone with that split.


# --------------------------------------------------------------------------- #
# Real data (small sample)                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def real_frames():
    """Wide close / returns matrices from real prices, subset for speed but guaranteed to
    contain AMD (the ticker under investigation).

    Reads a SUBSET of TICKERS, never the whole table: `prices` is 1.8M rows / 207MB and the
    pivot below multiplies that. The ticker list is resolved first (a cheap indexed DISTINCT)
    and the same 101 names this fixture has always selected are picked from it, cutting the
    read to ~380k rows. All price COLUMNS are kept -- `prices_long_to_multiindex` needs both
    close and open, and the momentum builders need the full OHLCV.

    TWO tables now: `prices` for the equities and `prices_macro` for the market / commodity /
    FX series. It used to be one read plus three `drop(columns=...)` calls, because those
    series were mixed into `prices`.
    """
    from src.data_aggregate.utils.common import data_utils as du
    from src.constants.constants import MACRO_CUBE_FACTORS, MACRO_MARKET_SERIES
    from src.utils.macro import load_macro_wide

    store = _store()
    all_tickers = sorted(store.distinct("prices", "ticker"))
    if not all_tickers:
        pytest.skip("prices table is empty")
    subset = (["AMD"] if "AMD" in all_tickers else []) + \
             [c for c in all_tickers if c != "AMD"][:SUBSET_SIZE]

    macro = load_macro_wide(store)
    if macro is None or MACRO_MARKET_SERIES not in macro.columns:
        pytest.skip("prices_macro is empty -> no market series for the trading calendar")
    macro = macro.set_index("date").sort_index()

    prices = store.load("prices", where={"ticker": sorted(subset)})
    raw = du.prices_long_to_multiindex(prices)
    close = du.extract_field(raw, "Close")
    # the trading calendar is still the dates the MARKET traded -- sourced from prices_macro
    close = close.loc[macro[MACRO_MARKET_SERIES].reindex(close.index).notna()]
    returns = du.daily_returns(close)

    cols = list(returns.columns)
    keep = (["AMD"] if "AMD" in cols else []) + [c for c in cols if c != "AMD"][:SUBSET_SIZE]
    mkt_level = macro[MACRO_MARKET_SERIES].astype(float)
    mkt_level = mkt_level.reindex(mkt_level.index.union(close.index)).ffill()
    return {
        "mkt_ret": mkt_level.pct_change(fill_method=None).reindex(close.index),
        "stock_close": close[keep],
        "stock_ret": returns[keep],
        # the macro/market series the commodity + currency factors read, wide by series name
        "macro_wide": macro,
        "factor_series": dict(MACRO_CUBE_FACTORS),
    }


@pytest.fixture(scope="session")
def real_pipeline(real_frames):
    """End-to-end real-data aggregate pieces computed once: peers, sector
    returns, factor panel, rolling betas and multi-horizon targets."""
    from src.data_aggregate.utils.target.betas import estimate_all_betas
    from src.data_aggregate.utils.target.targets import build_targets_multi
    from src.data_aggregate.utils.common.gics import load_gics_maps
    from src.data_aggregate.utils.target.factors import (
        build_characteristics,
        characteristic_to_factor_return,
        macro_change_factors,
        assemble_factor_panel,
    )
    from src.data_peers.utils.sector_peers import build_peer_dict

    stock_close = real_frames["stock_close"]
    stock_ret = real_frames["stock_ret"]
    mkt_ret = real_frames["mkt_ret"]
    macro = real_frames["macro_wide"]
    factor_series = real_frames["factor_series"]

    store = _store()
    # optional=True: these builders accept None, and `load` now raises rather than
    # returning an empty frame
    fundamentals = store.load("fundamentals_history", optional=True)

    peers = build_peer_dict(stock_ret, top_k=20, weighting="corr", min_obs=120)
    # mirror StepCubeTarget._gics_groups: GICS sector + industry_group neutralization
    context = type("Ctx", (), {"store": store})()
    sector_groups = load_gics_maps(context)

    # mirror StepCubeTarget._factor_panel: market + style + commodity + currency + macro
    chars = build_characteristics(stock_close, stock_ret, fundamentals, resvol_window=63)
    style_cols = {}
    for name, char in chars.items():
        char.name = name
        style_cols[name] = characteristic_to_factor_return(char, stock_ret)
    style = pd.DataFrame(style_cols)
    macro_chg = macro_change_factors(macro.reset_index(), stock_close.index)

    def _factor_ret(series: str):
        s = macro[series].astype(float)
        s = s.reindex(s.index.union(stock_close.index)).ffill()
        return s.pct_change(fill_method=None).reindex(stock_close.index)

    asset = pd.DataFrame({col: _factor_ret(series) for col, series in factor_series.items()
                          if series in macro.columns}, index=stock_close.index)
    fx_cols = [c for c in asset.columns if factor_series[c].startswith("fx_")]
    commodity_returns = asset.drop(columns=fx_cols)
    currency_returns = asset[fx_cols]
    factor_panel, macro_cols = assemble_factor_panel(
        mkt_ret, style, commodity_returns, currency_returns, macro_chg)

    betas = estimate_all_betas(
        stock_ret, factor_panel,
        window=63, min_obs=40, ridge_alpha=0.08, step=1,
    )

    horizons = (5, 20, 60)
    # one label via the multi-label builder (the single-label `build_targets` twin was
    # deleted); unwrap {h: {"rank": df}} -> {h: df} so the fixture's shape is unchanged
    # and its five consumer tests need no edits.
    _multi = build_targets_multi(
        stock_close, betas, factor_panel, macro_cols,
        horizons=horizons, labels=("rank",), min_names=20,
        sector_groups=sector_groups,
    )
    labels_rank = {h: by_label["rank"] for h, by_label in _multi.items()}

    return {
        "peers": peers,
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

    stock_close = real_frames["stock_close"]
    stock_ret = real_frames["stock_ret"]
    # Scope the fundamentals to the SAME universe as the price frames. Passing the full
    # 495-ticker history against a 101-ticker close frame built a ~1.9M-row x 200-feature
    # panel (with rolling(1260) windows) for tickers the test never looks at.
    tickers = sorted(stock_close.columns)
    fundamentals = _store().load("fundamentals_history",
                                 where={"ticker": tickers}, optional=True)
    if fundamentals is None:
        pytest.skip("fundamentals_history has no rows for the sample universe")
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
