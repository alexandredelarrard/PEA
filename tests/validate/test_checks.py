"""
tests/validate/test_checks.py
--------------------------------------------------------------------------------------------
ONE test per check function in `src/validate/checks/`, and that is the declared ceiling: nine.

Each test plants a defect whose answer is known by construction and asserts that the check
finds THAT defect and nothing else -- because the failure mode this whole package is built
against is a check that returns a clean result having measured nothing. A test that only
asserts "returns a CheckResult" reproduces the bug it is supposed to catch.

Every test prints its conclusion, so `pytest -q -s` reads as a short report rather than a row
of dots. Offline throughout: a real `DataStore` on in-memory SQLite (`sqlite_store`), so the
store facade under test is the production one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.constants.constants import INSUFFICIENT_HISTORY_TICKERS
from src.data_store.schema import Tables
from src.validate.checks import check_bounds, check_coverage, check_grain, check_profile
from src.validate.spec import UndeclaredTableError

#: The part the synthetic frames borrow their grain from: `pk == (date, ticker)`, `date_col ==
#: date`, which is the shape every `cube_part_*` table has. Using a REGISTERED table means the
#: checks resolve the real declared pk rather than one the test invented.
PART = Tables.cube_part_momentum

SESSIONS = pd.bdate_range("2024-01-01", periods=40)


class _Ctx:
    """The two attributes every check touches on a `Context`."""

    def __init__(self, store, config):
        self.store, self.config = store, config


def _config(**defaults):
    """The REAL `configs/validate.yml`, with `defaults` overridden per test.

    Loading the shipped file is deliberate: it makes every one of these tests a check that
    the config still carries every key `TableSpec` requires."""
    config = OmegaConf.create({"validate": OmegaConf.load("configs/validate.yml")["validate"],
                               "data_extract": {"redundant_ticks": []}})
    for key, value in defaults.items():
        config.validate.defaults[key] = value
    return config


def _panel(tickers, sessions=SESSIONS, **columns) -> pd.DataFrame:
    """A (date x ticker) panel with the given feature columns, each a callable of (i, ticker)."""
    rows = []
    for ticker in tickers:
        for i, day in enumerate(sessions):
            row = {"date": day, "ticker": ticker}
            row.update({name: fn(i, ticker) for name, fn in columns.items()})
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- #
# grain                                                                                   #
# --------------------------------------------------------------------------------------- #
def test_grain_finds_a_duplicated_declared_key(sqlite_store, capsys):
    """A second row on one (date, ticker) is a broken grain: score 10, and named."""
    frame = _panel(["AAA", "BBB"], f_x=lambda i, t: float(i))
    doubled = pd.concat([frame, frame.iloc[[7]]], ignore_index=True)
    # `save` upserts on the pk and would silently collapse the duplicate -- the whole point
    # is a table that HAS one, so it goes in as a raw append.
    doubled.to_sql("cube_part_momentum", sqlite_store.engine, index=False)
    context = _Ctx(sqlite_store, _config())

    result = check_grain(context, PART, config=context.config)

    assert result.status == "fail"
    assert result.worst_score == 10
    assert result.metrics["rows"] == len(doubled)
    assert result.metrics["distinct_keys"] == len(frame)
    assert result.metrics["duplicate_rows"] == 1
    example = result.findings[0].evidence["examples"][0]
    assert example["ticker"] == frame.iloc[7]["ticker"] and example["rows"] == 2
    with capsys.disabled():
        print(f"\n  grain: {result.metrics['rows']} rows over "
              f"{result.metrics['distinct_keys']} distinct (date, ticker) keys -> score "
              f"{result.worst_score}, duplicate named as "
              f"{example['ticker']} {pd.Timestamp(example['date']).date()}")


# --------------------------------------------------------------------------------------- #
# coverage                                                                                #
# --------------------------------------------------------------------------------------- #
def test_coverage_files_the_hole_and_not_the_declared_exclusions(sqlite_store, capsys):
    """THE D-08 REGRESSION GUARD, plus the defect coverage is actually for.

    A universe exclusion absent from the table must not be a defect -- it is absent by
    declaration, and filing it is the false positive `momentum/_scripts/08` recorded. A
    traded session missing INSIDE a ticker's own span must be.
    """
    excluded = sorted(INSUFFICIENT_HISTORY_TICKERS)[0]
    roster = ["AAA", "BBB", excluded]
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": roster}))
    sqlite_store.save(Tables.prices, _panel(roster, close=lambda i, t: 10.0 + i))

    # AAA complete; BBB missing three sessions in the middle of its own span.
    panel = _panel(["AAA", "BBB"], f_x=lambda i, t: float(i))
    holed = panel.drop(panel[(panel["ticker"] == "BBB")
                             & panel["date"].isin(SESSIONS[20:23])].index)
    sqlite_store.save(PART, holed)
    context = _Ctx(sqlite_store, _config(universe_expected=2))

    result = check_coverage(context, PART, config=context.config)

    named = {f.ticker for f in result.findings if f.score >= 4}
    assert result.metrics["universe"] == 2, "the exclusion must never enter the universe"
    assert excluded not in result.metrics["absent"] and excluded not in named
    assert named == {"BBB"}, f"expected only BBB, got {named}"
    assert result.status == "fail"
    assert result.metrics["interior"]["min"] == pytest.approx(37 / 40)

    # ... and with the hole filled, the same table passes WITH the exclusion still reported.
    sqlite_store.replace(PART, panel)
    clean = check_coverage(context, PART, config=context.config)
    assert clean.status == "pass"
    assert [f.score for f in clean.findings] == [1, 1], "info only: exclusions and depth"
    info = clean.findings[0].evidence
    with capsys.disabled():
        print(f"\n  coverage: BBB at {result.metrics['interior']['min']:.1%} of its own traded "
              f"sessions -> filed; {excluded} absent and NOT filed "
              f"({len(info['exclusions'])} declared exclusions reported as info); "
              f"hole filled -> {clean.status.upper()}")


# --------------------------------------------------------------------------------------- #
# profile                                                                                 #
# --------------------------------------------------------------------------------------- #
def test_profile_finds_the_dead_constant_and_infinite_legs(sqlite_store, capsys):
    """The 65-of-179 defect: a column that is null everywhere, and its two cousins.

    `f_dead` is all-NULL, which is the case a dtype filter drops -- pandas reads it back as
    `object`, so this test is also the guard on `frame._is_leg`.
    """
    frame = _panel(
        ["AAA", "BBB"],
        f_good=lambda i, t: float(i) + (0.5 if t == "BBB" else 0.0),
        f_dead=lambda i, t: None,
        f_const=lambda i, t: 1.0,
        f_inf=lambda i, t: (np.inf if i == 3 else float(i)),
    )
    sqlite_store.save(PART, frame)
    context = _Ctx(sqlite_store, _config())

    result = check_profile(context, PART, config=context.config, group=2)

    assert result.status == "fail"
    assert result.metrics["dead"] == ["f_dead"], result.metrics["dead"]
    assert result.metrics["constant"] == ["f_const"]
    assert result.metrics["infinite"] == ["f_inf"]
    by_field = {f.field: f.score for f in result.findings}
    assert by_field == {"f_dead": 10, "f_const": 7, "f_inf": 9}
    good = result.metrics["stats"]["f_good"]
    assert good["n_ok"] == len(frame) and good["n_distinct"] == 80
    assert good["min"] == 0.0 and good["max"] == pytest.approx(39.5)
    # `redundancy` reuses this as its centring constant, so it has to be the real mean.
    assert result.metrics["mean"]["f_good"] == pytest.approx(frame["f_good"].mean())
    with capsys.disabled():
        print(f"\n  profile: 4 legs -> dead {result.metrics['dead']}, constant "
              f"{result.metrics['constant']}, infinite {result.metrics['infinite']}; "
              f"f_good n_ok={good['n_ok']} n_distinct={good['n_distinct']} "
              f"p50={good['p50']}")


# --------------------------------------------------------------------------------------- #
# bounds                                                                                  #
# --------------------------------------------------------------------------------------- #
def test_bounds_abstains_undeclared_and_names_the_worst_violation(sqlite_store, capsys):
    """Exit 3 with no declaration; with one, the count AND one (ticker, date, value)."""
    frame = _panel(["AAA", "BBB"],
                   f_pct=lambda i, t: (140.0 if (t == "BBB" and i == 11) else
                                       -3.0 if (t == "AAA" and i == 2) else float(i)))
    sqlite_store.save(PART, frame)
    context = _Ctx(sqlite_store, _config())

    # 1. `cube_part_momentum` declares no bounds -> the check must refuse to answer.
    with pytest.raises(UndeclaredTableError) as undeclared:
        check_bounds(context, PART, config=context.config)
    assert "bounds" in str(undeclared.value)

    # 2. with a declaration, the violations are counted and the worst one is named.
    result = check_bounds(context, PART, config=context.config,
                          bounds={"f_pct": (0.0, 100.0)})
    assert result.status == "fail"
    finding = result.findings[0]
    assert finding.score == 9 and finding.field == "f_pct"
    assert finding.evidence["n_violations"] == 2
    assert finding.evidence["worst"]["ticker"] == "BBB"
    assert finding.evidence["worst"]["value"] == pytest.approx(140.0)
    with capsys.disabled():
        print(f"\n  bounds: undeclared -> ABSTAIN (exit 3); declared [0, 100] -> "
              f"{finding.evidence['n_violations']} violations, worst "
              f"{finding.evidence['worst']['ticker']} "
              f"{pd.Timestamp(finding.evidence['worst']['date']).date()} = "
              f"{finding.evidence['worst']['value']}")
