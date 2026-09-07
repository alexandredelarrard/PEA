"""NO TWO FUNDAMENTALS FEATURES MAY BE THE SAME NUMBER.

The structural gap this closes: nothing in the suite ever compared two cube features to
EACH OTHER. Every other test checks one formula against a hand calculation, so a feature
whose inputs vanished — and which therefore collapsed onto a sibling — stayed green. Three
pairs sat at Pearson r = 1.0000 in production for a whole vintage of the table:

    f_fcf_yield_*          == f_intrinsic_yield_*     `revenueGrowth` was never computed, so
                                                      the DCF grew every firm at a constant
                                                      2.5% and reduced to `fcf x constant`
    f_returnOnEquity_*     == f_sustainable_growth_*  `dividendsPaid` was stored negative, so
                                                      `clip(0,1)` floored every payer's
                                                      payout to 0 and retention to 1
    f_roic_incl_goodwill_* == f_roic_ex_goodwill_*    the bare `goodwill` column is written by
                                                      no producer, so "ex goodwill"
                                                      subtracted nothing

and the sector layer added two more, where a KPI reduced to a GICS-masked copy of a
general-purpose ratio (`implied_cap_rate` and `ebitdax_to_ev` both == `ebitda_to_ev`).

A duplicate is not merely wasteful. It doubles a theme's weight on one underlying quantity
inside a composite, and it hands a tree model two identical splits whose importances then
each read as half the truth.

REAL DATA, deliberately: the defect is a property of what the live table can feed, and a
synthetic frame that hand-builds the missing column reproduces none of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.pit import (
    add_cube_time_growth, fundamentals_to_daily, infer_yoy_periods,
)
from src.data_aggregate.utils.fundamentals.fundamental_features import (
    _FACT_COLS, _FN_PBO_TAG, _FN_PLAN_ASSETS_TAG, _NET_PENSION_TAGS,
    _NOTES_NUM_TABLE, _PENSION_FACTS_TABLE, _derived_fields,
)
from src.data_aggregate.utils.fundamentals.sector_features import (
    SECTOR_KPI_COLS, compute_sector_kpis,
)
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.db import get_engine

#: |r| at or above which two features are the SAME SIGNAL, not two views of one.
#: Deliberately not 1.0: `net_debt_to_ebitdare` reduced to `net_debt_to_ebitda` at 0.9998,
#: which is a duplicate in every way that matters to a model.
CORRELATION_CEILING = 0.999

#: Enough names to span every GICS family the sector gates scope on, so a gated KPI is
#: present in the matrix rather than absent and trivially "not correlated".
_TICKERS = ["JPM", "BAC", "AIG", "PGR", "SPG", "PLD", "XOM", "CVX", "NEE", "DUK",
            "PFE", "MRK", "AAPL", "MSFT", "KO", "PG", "GE", "CAT"]

#: A pair needs at least this many overlapping observations before its correlation is
#: believable; two features that only ever co-occur on a handful of rows can hit 1.0 by
#: coincidence.
_MIN_OVERLAP = 1000


@pytest.fixture(scope="module")
def feature_frames() -> dict[str, pd.DataFrame]:
    """Every daily feature frame both fundamentals builders emit, on the live table.

    Taken BEFORE the `_vs_peers` / `_xs` suffixes: a computational no-op is undiluted at
    this level, while the peer-z and the percentile rank both compress it.

    ⚠ THE PENSION TABLES ARE PART OF PRODUCTION'S INPUT AND MUST BE PASSED. Without them
    `_pension_pool` returns empty and every pension feature silently disappears from the
    matrix — the same unfaithful-fixture defect that was fixed in `test_composites_config`'s
    `real_panel` during Phase 4 and missed here. Passing them raises the comparable-feature
    count from 99 to 103, and it is what let this test do its job: it caught
    `net_debt_incl_offbs_to_ebitda` collapsing onto `net_debt_to_ebitda` after the distress
    block moved to the reconciled `totalDebt`, and that feature was deleted as a result."""
    try:
        store = DataStore(get_engine())
        fh = store.load(Tables.fundamentals_history,
                        where={"ticker": _TICKERS}, optional=True)
    except Exception as exc:                    # pragma: no cover - env without the DB
        pytest.skip(f"{Tables.fundamentals_history} unavailable ({type(exc).__name__})")
    if fh is None or fh.empty:
        pytest.skip(f"{Tables.fundamentals_history} is empty")
    fh["as_of"] = pd.to_datetime(fh["as_of"])
    fh = add_cube_time_growth(fh)

    ref = store.load(Tables.sp500_tickers, columns=["ticker", "sector", "industry_group"])
    fh = fh.merge(ref, on="ticker", how="left")

    px = store.load(Tables.prices, columns=["date", "ticker", "close_split"],
                    where={"ticker": _TICKERS})
    px["date"] = pd.to_datetime(px["date"])
    close = px.pivot_table(index="date", columns="ticker", values="close_split",
                           aggfunc="last").sort_index()
    idx = pd.DatetimeIndex(close.index)

    def _facts(table: str, tags: tuple[str, ...]) -> pd.DataFrame | None:
        got = store.load(table, columns=_FACT_COLS, where={"tag": list(tags)}, optional=True)
        return got.reset_index(drop=True) if got is not None else None

    frames = _derived_fields(
        fund_hist=fh, idx=idx, close=close, yoy_periods=infer_yoy_periods(fh),
        pension_facts=_facts(_PENSION_FACTS_TABLE, _NET_PENSION_TAGS),
        notes_num=_facts(_NOTES_NUM_TABLE, (_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG)))
    kdf = compute_sector_kpis(fh)
    for name in SECTOR_KPI_COLS:
        if name in kdf.columns:
            daily = fundamentals_to_daily(kdf, name, idx)
            if not daily.empty and daily.notna().any().any():
                frames.setdefault(name, daily)
    return frames


def test_no_two_features_are_the_same_signal(feature_frames):
    stacked = {}
    for name, frame in feature_frames.items():
        if frame is None or frame.empty:
            continue
        s = frame.stack()
        # a constant column has no correlation to speak of, and a 2-value state flag is a
        # regime indicator rather than a measurement
        if s.notna().sum() >= _MIN_OVERLAP and s.nunique() > 2:
            stacked[name] = s
    mat = pd.DataFrame(stacked)
    assert mat.shape[1] >= 50, f"only {mat.shape[1]} comparable features -- fixture too thin"

    corr = mat.corr(min_periods=_MIN_OVERLAP).abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0.0)
    corr = pd.DataFrame(corr, index=mat.columns, columns=mat.columns)

    offenders = []
    seen = set()
    for a, b in zip(*np.where(corr.to_numpy() >= CORRELATION_CEILING)):
        pair = tuple(sorted((mat.columns[a], mat.columns[b])))
        if pair in seen:
            continue
        seen.add(pair)
        offenders.append((pair[0], pair[1], float(corr.iloc[a, b])))
    offenders.sort(key=lambda t: -t[2])

    ranked = corr.stack().sort_values(ascending=False)
    top = []
    seen_top = set()
    for (a, b), r in ranked.items():
        key = tuple(sorted((a, b)))
        if key in seen_top:
            continue
        seen_top.add(key)
        top.append((a, b, float(r)))
        if len(top) == 5:
            break

    print("\n=== SANITY CHECK: cross-feature redundancy ===")
    print(f"  {mat.shape[1]} features x {len(mat):,} (date, ticker) cells, "
          f"ceiling |r| >= {CORRELATION_CEILING}")
    print("  most-correlated surviving pairs:")
    for a, b, r in top:
        print(f"    {r:.6f}  {a} <-> {b}")
    if offenders:
        print(f"  ⚠ {len(offenders)} pair(s) AT OR ABOVE the ceiling:")
        for a, b, r in offenders:
            print(f"    {r:.6f}  {a} <-> {b}")
    else:
        print("  no pair reaches the ceiling -> no feature is a computational no-op "
              "of another. Validated.")

    assert not offenders, (
        "these feature pairs carry the same signal; one of each is a computational no-op "
        "(usually a source column that no producer writes, silently read as empty):\n"
        + "\n".join(f"  r={r:.6f}  {a} <-> {b}" for a, b, r in offenders))


def test_the_three_historical_duplicate_pairs_stay_broken(feature_frames):
    """The specific pairs this audit fixed, pinned by name so a regression in any one of the
    three underlying causes (uncomputed growth, an inverted sign, an unwritten column) is
    attributed rather than merely detected."""
    pairs = [
        ("fcf_yield", "intrinsic_yield", "revenueGrowth is computed at cube time"),
        ("returnOnEquity", "sustainable_growth_rate", "dividendsPaid is outflow-positive"),
        ("roic_incl_intangibles", "roic_ex_intangibles",
         "the intangibles deduction is non-empty"),
    ]
    print("\n=== SANITY CHECK: the former r=1.0000 pairs ===")
    for a, b, because in pairs:
        assert a in feature_frames and b in feature_frames, f"{a} / {b} missing"
        x, y = feature_frames[a].align(feature_frames[b], join="inner")
        m = (x.notna() & y.notna()).to_numpy()
        xa, ya = x.to_numpy()[m], y.to_numpy()[m]
        assert len(xa) >= _MIN_OVERLAP, f"{a}/{b}: only {len(xa)} overlapping cells"
        r = float(np.corrcoef(xa, ya)[0, 1])
        print(f"  {a} vs {b}: r={r:+.6f}  n={len(xa):,}  ({because})")
        assert abs(r) < CORRELATION_CEILING, (
            f"{a} and {b} are identical again -- check that {because}")
    print(f"  all {len(pairs)} broken; each was exactly 1.0000 at some point. Validated.")
