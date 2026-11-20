"""Configured incremental parity and registry projection contracts.

Window arithmetic and bounded/atomic persistence are exercised through the shared resolver,
real writer and actual producer tests; retired hard-coded refresh APIs have no compatibility layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_aggregate import cli as aggregate_cli
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.momentum.features import build_feature_panel
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_store.schema import (
    ALL,
    Tables,
    projection,
    projection_report,
    resolve,
)
from src.data_store.store import DataStore
from src.utils.window import Window, WindowRun
from tests.data_aggregate.test_price_pit_contract import seed_prices
from tests.support.config import fixture_config
from tests.support.universe import universe_of


def _synthetic_prices(
    n_days: int = 2000, n_tickers: int = 8, seed: int = 0
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tk = [f"T{i}" for i in range(n_tickers)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.012, (n_days, n_tickers)), axis=0)), index=dates, columns=tk)
    open_ = close.shift(1).bfill()
    ret = close.pct_change().fillna(0.0)
    sector = pd.DataFrame(np.repeat(ret.mean(axis=1).to_numpy()[:, None], n_tickers, axis=1), index=dates, columns=tk)  # one shared "sector"
    high, low = close * 1.01, close * 0.99
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (n_days, n_tickers)).astype(float), index=dates, columns=tk)
    return dates, close, open_, sector, high, low, volume


def test_windowed_build_reproduces_full_tail() -> None:
    """The equivalence the whole incremental design rests on, over the INCLUSIVE window.

    The compared span is `date >= refresh_from`, not `date > cutoff`: a backward-looking part
    now REWRITES its trailing configured correction Window as well as appending after them, so
    those dates have to be reproduced by the windowed build too -- they are deleted from the
    table before the append and there is no second chance at them.

    This is also what makes `plan_window`'s `warmup + extra_back + refresh` load-bearing: the
    read window starts `warmup` days before `refresh_from`, not before `last`, so the OLDEST
    rewritten date still gets its full look-back."""
    dates, close, open_, sector, high, low, volume = _synthetic_prices()
    sh = [5, 20, 60]

    full = build_feature_panel(
        PriceFrames(
            trading_index=(close).index,
            universe=tuple((close).columns),
            peers={},
            close_total=close,
            open=open_,
            sector_ret=sector,
            high=high,
            low=low,
            volume=volume,
            volume_pit=volume,
            close_pit=close,
        ),
        method="rank",
        seasonal_horizons=sh,
    )

    # simulate an incremental run with the ACTUAL configured price-group warm-up: recompute only
    # [refresh_from - warmup, end]. This must cover the longest daily look-back (the 5-year =
    # 1260-day seasonality feature) -> the test fails if the map value is ever set too low.
    policy = fixture_config().data.tables.cube_part_momentum
    warmup = int(policy.warmup.cube_part_prices)
    cutoff_pos = 1900
    last = dates[cutoff_pos]
    refresh_from = last - pd.Timedelta(days=int(policy.overlap.days))
    start = refresh_from - pd.Timedelta(days=warmup)
    refresh = int(((dates >= refresh_from) & (dates <= last)).sum()) - 1
    win = build_feature_panel(
        PriceFrames(
            trading_index=(close.loc[start:]).index,
            universe=tuple((close.loc[start:]).columns),
            peers={},
            close_total=close.loc[start:],
            open=open_.loc[start:],
            sector_ret=sector.loc[start:],
            high=high.loc[start:],
            low=low.loc[start:],
            volume=volume.loc[start:],
            volume_pit=volume.loc[start:],
            close_pit=close.loc[start:],
        ),
        method="rank",
        seasonal_horizons=sh,
    )

    # the tail the incremental run WRITES (date >= refresh_from) must match the full build
    # bit-for-bit -- both the `refresh` dates it overwrites and everything it appends after.
    f_tail = full[full["date"] >= refresh_from].set_index(["date", "ticker"]).sort_index()
    w_tail = win[win["date"] >= refresh_from].set_index(["date", "ticker"]).sort_index()
    assert not f_tail.empty and f_tail.shape == w_tail.shape
    cols = list(f_tail.columns)
    pd.testing.assert_frame_equal(f_tail[cols], w_tail[cols].reindex(f_tail.index), check_exact=False, atol=1e-9, rtol=0)

    # the rewritten span really does include `last` itself -- the row a strictly-after append
    # could never replace, and the one a truncated extract run left wrong on the live table.
    rewritten = f_tail.index.get_level_values("date").unique()
    assert last in set(rewritten)
    assert (rewritten <= last).sum() == refresh + 1, "refresh_from .. last inclusive"

    n_tail_days = len(rewritten)
    print("\n=== SANITY CHECK: windowed build reproduces the full tail (INCLUSIVE) ===")
    print(
        f"  full={len(full)} rows over {close.shape[0]} days; windowed recompute of the last "
        f"{close.shape[0] - (cutoff_pos - refresh - warmup)} days "
        f"(warmup {warmup} + refresh {refresh})"
    )
    print(
        f"  stored max was {last.date()}; the run REWRITES from {refresh_from.date()} ({refresh + 1} dates through {last.date()}) and appends after"
    )
    print(
        f"  tail (date >= {refresh_from.date()}): {n_tail_days} days x "
        f"{full['ticker'].nunique()} tickers x {len(cols)} feature cols -> IDENTICAL between "
        "full and windowed builds"
    )
    print(
        "  CONCLUSION: backward-looking features + per-day standardization -> the incremental "
        "trailing recompute equals a full rebuild on every date it writes, rewritten dates "
        "included. Validated."
    )


def test_read_projection_covers_builder_needs() -> None:
    """Each tall-table source load is projected to only the columns its builder reads (the
    memory fix for the OOM), and the projection MUST cover every column the builder requires.

    The projection now lives ONCE, on the registry (`Table.read_columns`), and reaches the
    builders through `store.load(project=True)`. This test is the contract that guards against
    a registry edit dropping a needed column -- it fails here rather than silently emptying a
    feature family.
    """
    required = {  # columns each builder actually consumes from the table (the contract)
        Tables.sec13f_hr: {"cik", "period", "ticker", "shares", "value_usd", "call_value", "put_value", "filing_date"},
        Tables.insider_transactions: {
            "ticker",
            "owner_cik",
            "filing_date",
            "transaction_code",
            "shares",
            "price_per_share",
            "value_usd",
            "security_type",
            "shares_owned_after",
        },
        # the institutionals loader builds the ticker short-volume series from the as-filed FINRA lines
        # (`security_series.SHORT_LINE_COLUMNS`); RegSHO carries daily short VOLUME only, no short interest
        Tables.sec_short_volume_security: {"date", "source_symbol", "short_volume", "total_volume"},
        # the institutionals loader builds the ticker FTD series from the as-filed lines (`security_series.LINE_COLUMNS`)
        Tables.sec_fails_to_deliver_security: {"date", "trade_date", "cusip", "source_symbol", "fails_quantity", "fails_value", "period"},
        # ownership_features (13D/13G) counts filings per (ticker, accession_number) on the filing_date stamp.
        Tables.sec_13d: {"ticker", "accession_number", "filing_date"},
        Tables.sec_13g: {"ticker", "accession_number", "filing_date"},
        # the two per-person DEF 14A children, read by StepCubeGovernance. `accession_number`
        # is in the directors set because the per-FILING aggregates key on it (an `as_of` can
        # carry two filings), and every one of the six Item 402(k) components is required
        # because `impute_director_comp` reconstructs a NULL `total` by summing them.
        # ⚠ `def14a_directors.is_independent` is PROJECTED but deliberately not read (D79:
        # `pct_independent_directors` is one of the twelve D3-protected features, so deriving
        # it from the child would move live cells). Left in the projection on purpose -- do not
        # "tidy" it out, it is the substrate for that deferred work.
        Tables.def14a_directors: {"ticker", "accession_number", "as_of", "name", "age", "tenure_years", "other_public_company_boards"},
        Tables.def14a_director_comp: {
            "ticker",
            "as_of",
            "total",
            "fees_earned",
            "stock_awards",
            "option_awards",
            "non_equity_incentive",
            "pension_change",
            "other_compensation",
        },
    }
    for table, need in required.items():
        proj = set(table.read_columns)
        assert proj, f"{table.name}: registry declares no read_columns"
        assert need <= proj, f"{table.name}: projection is MISSING required {need - proj}"

    print()
    print("=== SANITY: registry read_columns cover every builder need ===")
    for table in required:
        print(f"  {table.name:<24} -> {len(table.read_columns)} cols (covers builder needs)")
    print("  def14a_directors keeps `is_independent` projected-but-unread on purpose (D79).")
    print("  sec13f_hr (~21.7M rows) drops the cusip-era bloat; a table declaring no read_columns loads in full. Validated.")


def test_every_projected_table_resolves_to_a_registered_physical_table() -> None:
    """A projection must hang off a REGISTERED table, and the registry key trap is real.

    Registry entries can carry an attribute name that is not their table name -- the one
    that bit was `Tables.short_interest`, whose table was `sec_short_interest`. A projection map
    keyed on the ATTRIBUTE name meant `store.exists()` returned False, `_load_source` returned
    None, and the three `ic_shortvol_*` features were absent from `cube_part_institutionals`
    while 956,640 rows sat unread in the table. Nothing failed: the map, the step and this
    file's other tests all agreed on a name no table has.

    Hanging `read_columns` off the `Table` object makes the whole class unrepresentable --
    there is no key left to get wrong. This test states that, and prints the trap it closes.
    """
    projected = [t for t in ALL if t.read_columns]
    for t in projected:
        assert resolve(t.name) is t, f"{t.name}: not reachable by its own physical name"
        assert set(t.optional_columns) <= set(t.read_columns), (
            f"{t.name}: optional_columns "
            f"{sorted(set(t.optional_columns) - set(t.read_columns))} are not in read_columns "
            f"-- a dead exemption would silently tolerate a real missing column"
        )

    mismatched = {a: t.name for a, t in vars(Tables).items() if hasattr(t, "name") and a != t.name}
    print()
    print("=== SANITY: projections hang off the Table object, not a name key ===")
    print(f"  {len(projected)} tables declare read_columns, all reachable by physical name")
    print(f"  {len(mismatched)} registry entries where attribute != table name -- the trap:")
    for attr, real in sorted(mismatched.items()):
        print(f"      Tables.{attr:22} -> {real}")


def test_projection_tolerates_an_absent_optional_column() -> None:
    """A column the BUILDER treats as optional must not make the READ fail, and a missing
    REQUIRED one must be REPORTED rather than swallowed.

    `read_table` resolves each projected column via `tbl.c[name]`, which raises KeyError for
    an absent one. An older `sec13f_hr` carries no option or accession columns, which its
    projection declares optional; demanding them unconditionally would kill the step.
    """
    full_13f = list(Tables.sec13f_hr.read_columns)
    live_13f = [c for c in full_13f if c not in Tables.sec13f_hr.optional_columns]
    cols, required_missing, optional_missing = projection_report(Tables.sec13f_hr, live_13f)
    assert cols == live_13f, cols
    assert not required_missing
    assert sorted(optional_missing) == sorted(Tables.sec13f_hr.optional_columns)

    # a table with every projected column present is unchanged
    assert projection(Tables.sec13f_hr, full_13f) == full_13f

    # a missing REQUIRED column is REPORTED, not dropped in silence -- that is the half of
    # this that a quiet degrade would hide
    _, req_missing, _ = projection_report(Tables.sec13f_hr, [c for c in full_13f if c != "value_usd"])
    assert req_missing == ["value_usd"], req_missing

    # unknown column list (table shape unreadable) -> project the full declared list
    assert projection(Tables.sec13f_hr, None) == full_13f
    # a table declaring no projection -> None (load in full)
    assert projection(Tables.prices_splits, ["a", "b"]) is None

    print()
    print("=== SANITY CHECK: optional projected columns degrade, required ones report ===")
    print(f"  sec13f_hr live cols {live_13f} -> projection {cols}")
    print(f"  dropped as optional: {sorted(optional_missing)}")
    print(f"  sec13f_hr without `value_usd` -> required_missing={req_missing}")
    print(
        "  CONCLUSION: an optional column missing from the live table is dropped from the "
        "projection instead of raising KeyError and killing the step, while a missing "
        "REQUIRED one is surfaced. Validated."
    )


@pytest.mark.parametrize(("warmup_days", "expected_since"), [(None, None), (30, pd.Timestamp("2025-12-03"))], ids=["full-history", "bounded"])
def test_step_forwards_the_registry_projection_and_configured_bounds(warmup_days: int | None, expected_since: pd.Timestamp | None) -> None:
    calls = []

    class Store:
        def exists(self, table: Any) -> bool:
            return True

        def iter_load(self, table: Any, **kwargs: Any) -> Iterator[pd.DataFrame]:
            calls.append((table, kwargs))
            return iter([pd.DataFrame({"ticker": ["AAPL"]})])

    config = fixture_config()
    config.data.tables[Tables.cube_part_institutionals.name].warmup[Tables.sec_13d.name] = warmup_days
    context = SimpleNamespace(
        store=Store(),
        config=config,
        universe=universe_of(["AAPL"]),
        run=WindowRun("incremental", "projected source contract", pd.Timestamp("2026-11-16")),
    )
    windows = capital_inputs.input_windows(context, Tables.cube_part_institutionals, Window(pd.Timestamp("2026-01-02"), context.run.as_of))
    capital_inputs.load_source(context, Tables.sec_13d, window=windows[Tables.sec_13d])
    assert len(calls) == 1
    table, query = calls[0]
    assert table == Tables.sec_13d and query["project"] and query["columns"] is None
    assert query["where"]["ticker"] == ["AAPL"]
    assert windows[Tables.sec_13d] == (None if expected_since is None else Window(expected_since, context.run.as_of))
    assert query["date_col"] == "filing_date" and query["since"] == expected_since
    assert query["until"] == context.run.as_of
    print(f"SANITY: one projected source stream keeps universe scope and configured bounds: since={expected_since}, until={context.run.as_of}.")


@pytest.mark.parametrize(("warmup_days", "expected_since"), [(None, None), (30, pd.Timestamp("2025-12-03"))], ids=["full-history", "bounded"])
def test_universe_scope_is_pushed_down_to_the_read(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, warmup_days: int | None, expected_since: pd.Timestamp | None
) -> None:
    sqlite_store.save(
        Tables.sec_13d,
        pd.DataFrame(
            {
                "ticker": ["AAPL", "OUTSIDE", "AAPL", "OUTSIDE"],
                "accession_number": ["a-1", "o-1", "a-2", "o-2"],
                "rp_seq": 1,
                "filing_date": pd.to_datetime(["2024-01-02"] * 2 + ["2025-12-05"] * 2),
            }
        ),
    )
    config = fixture_config()
    config.data.tables[Tables.cube_part_institutionals.name].warmup[Tables.sec_13d.name] = warmup_days
    context = SimpleNamespace(
        store=sqlite_store,
        config=config,
        universe=universe_of(["AAPL"]),
        run=WindowRun("incremental", "physical source scope", pd.Timestamp("2026-11-16")),
    )
    calls = []
    original = sqlite_store.iter_load

    def capture(table: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "iter_load", capture)
    windows = capital_inputs.input_windows(context, Tables.cube_part_institutionals, Window(pd.Timestamp("2026-01-02"), context.run.as_of))
    result = capital_inputs.load_source(context, Tables.sec_13d, window=windows[Tables.sec_13d])
    assert result is not None and set(result.ticker) == {"AAPL"}
    assert len(calls) == 1 and calls[0]["where"]["ticker"] == ["AAPL"]
    assert windows[Tables.sec_13d] == (None if expected_since is None else Window(expected_since, context.run.as_of))
    assert calls[0]["since"] == expected_since and calls[0]["until"] == context.run.as_of
    assert calls[0]["date_col"] == "filing_date"
    expected_dates = ["2024-01-02", "2025-12-05"] if expected_since is None else ["2025-12-05"]
    assert sorted(pd.to_datetime(result["filing_date"]).dt.strftime("%Y-%m-%d")) == expected_dates
    print(f"SANITY: one scoped store read retains {expected_dates}; outside symbols and out-of-window filings never enter cleaning.")


def test_institutional_empty_correction_clears_only_configured_tail(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    days = pd.bdate_range("2026-01-02", periods=80)
    config = fixture_config()
    context = SimpleNamespace(
        store=sqlite_store,
        config=config,
        config_dir=tmp_path,
        universe=universe_of(["AAPL"]),
        run=WindowRun("incremental", "known empty institutional correction", days[-1]),
    )
    sqlite_store.replace(Tables.cube_part_prices, pd.DataFrame({"date": days, "ticker": "AAPL", "close_split": 100.0}))
    original = pd.DataFrame({"date": days, "ticker": "AAPL", "f_demo": 1.0})
    sqlite_store.replace(Tables.cube_part_institutionals, original)
    step = StepCubeInstitutionals(context, config)
    empty = original.iloc[:0].copy()
    monkeypatch.setattr(step, "build_panel", lambda window, calendar: empty)
    written = step.run()[Tables.cube_part_institutionals]
    expected_start = days[-1] - pd.Timedelta(days=30)
    assert written.write.start == expected_start
    actual = sqlite_store.load(Tables.cube_part_institutionals)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), original.loc[original.date < expected_start].reset_index(drop=True))
    print("SANITY: the actual institutional writer clears a valid empty30-calendar-day correction and preserves its older physical prefix.")


def test_build_prices_cli_passes_one_root_mode_and_persists_price_keys(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    cutoff = pd.Timestamp("2022-11-30")
    expected_keys = sqlite_store.load(Tables.prices, columns=["date", "ticker"], where={"ticker": ["AAA", "BBB"]}, until=cutoff)
    assert expected_keys is not None
    calls = []

    def load(config_path: str, **kwargs: Any) -> tuple[Any, Any]:
        calls.append(kwargs)
        context.run = WindowRun("full" if kwargs["full"] else "incremental", "real CLI fixture", pd.Timestamp(kwargs["as_of"]))
        return context.config, context

    monkeypatch.setattr(aggregate_cli, "get_config_context", load)
    runner = CliRunner()
    refused = runner.invoke(aggregate_cli.cli, ["build-prices", "-t", "AAA"])
    assert refused.exit_code == 2 and not calls
    result = runner.invoke(aggregate_cli.cli, ["build-prices", "-F", "--as-of", "2022-11-30"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0]["full"]
    assert pd.Timestamp(calls[0]["as_of"]) == cutoff
    assert context.run.mode == "full" and context.run.as_of == cutoff
    actual_keys = sqlite_store.load(Tables.cube_part_prices, columns=["date", "ticker"])
    assert actual_keys is not None and set(actual_keys["ticker"]) == {"AAA", "BBB"}
    pd.testing.assert_frame_equal(
        actual_keys.sort_values(["date", "ticker"]).reset_index(drop=True),
        expected_keys.sort_values(["date", "ticker"]).reset_index(drop=True),
    )
    print(
        "SANITY: the actual CLI fixes mode/as_of once, rejects an undeclared ticker rebuild and persists exactly the expected price keys for both tickers."
    )
