"""
test_staleness.py  (tests/data_aggregate/test_staleness.py)
----------------------------------------------------------
The governance EVENT horizon (D21) — `expire_stale` / `GOVERNANCE_EVENT_MAX_AGE_DAYS`.

Four properties, three of them synthetic because they are about the boundary:

  * 17 months old SURVIVES, 19 months old EXPIRES — 548 days sits between them on purpose;
  * a LEGACY-exempt field never expires, whatever its age (the D3 guard in §3.4);
  * a field with two filings expires only once the LATER one has aged out — the age is
    measured against the filing that produced the cell, not the first one ever seen;
  * a NULL-carrying filing does not reset the clock: the age belongs to the last filing that
    actually disclosed a value, which is the one `fundamentals_to_daily` took the cell from.

Then the real-data BITE: what share of each def14a field's non-null daily cells the cap would
null. Nothing in the tree is expired yet (the only event-shaped fields that exist today are
the two legacy-exempt ones), so this is the measurement phases 3-5 need in order to know
whether 548 days is right for the families they add.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.staleness import (
    GOVERNANCE_EVENT_MAX_AGE_DAYS, LEGACY_EXEMPT_FROM_EXPIRY, expire_event_fields, expire_stale,
)


def _daily(history: pd.DataFrame, field: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return fundamentals_to_daily(history, field, idx)


def test_expiry_boundary_and_legacy_exemption():
    """17 months lives, 19 months dies, and a legacy field is immortal."""
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    hist = pd.DataFrame([{"ticker": "AAA", "as_of": "2020-06-01",
                          "sop_dissent": 0.11, "ceo_pay_growth": 0.11}])

    daily = _daily(hist, "sop_dissent", idx)
    capped = expire_stale(daily, hist, "sop_dissent")

    filed = pd.Timestamp("2020-06-01")
    d17 = filed + pd.Timedelta(days=int(30.4 * 17))     # ~517 days -> inside the horizon
    d19 = filed + pd.Timedelta(days=int(30.4 * 19))     # ~577 days -> outside it
    assert (d17 - filed).days <= GOVERNANCE_EVENT_MAX_AGE_DAYS < (d19 - filed).days

    v17 = capped.loc[capped.index.asof(d17), "AAA"]
    v19 = capped.loc[capped.index.asof(d19), "AAA"]
    assert v17 == pytest.approx(0.11), "a 17-month-old event was expired"
    assert pd.isna(v19), "a 19-month-old event survived"

    # the exact boundary: still alive on day 548, gone on day 549
    assert capped.loc[capped.index.asof(filed + pd.Timedelta(days=548)), "AAA"] \
        == pytest.approx(0.11)
    assert pd.isna(capped.loc[capped.index.asof(filed + pd.Timedelta(days=549)), "AAA"])

    # --- the D3 guard: a legacy field is returned untouched, identically ---
    legacy = _daily(hist, "ceo_pay_growth", idx)
    kept = expire_stale(legacy, hist, "ceo_pay_growth")
    assert kept is legacy, "a legacy-exempt field was copied, not passed through"
    assert int(kept.notna().to_numpy().sum()) == int(legacy.notna().to_numpy().sum())
    assert "ceo_pay_growth" in LEGACY_EXEMPT_FROM_EXPIRY
    assert len(LEGACY_EXEMPT_FROM_EXPIRY) == 12, "the twelve legacy features are the exemption"

    print("\n=== SANITY CHECK: governance event expiry (boundary) ===")
    print(f"  horizon = {GOVERNANCE_EVENT_MAX_AGE_DAYS} days")
    print(f"  filed 2020-06-01: +17mo ({(d17 - filed).days}d) = {v17}  (survives)")
    print(f"                    +19mo ({(d19 - filed).days}d) = NaN   (expired)")
    print("  exact boundary: alive on day 548, NaN on day 549.")
    print(f"  the {len(LEGACY_EXEMPT_FROM_EXPIRY)} legacy features are exempt and returned")
    print("  by identity — `ceo_pay_growth` keeps every cell it had (the D3 guard).")


def test_expiry_uses_the_filing_that_produced_the_cell():
    """Two filings: the clock restarts on the later one — and a NULL filing does not restart it."""
    idx = pd.bdate_range("2018-01-01", "2024-12-31")
    hist = pd.DataFrame([
        {"ticker": "AAA", "as_of": "2019-05-01", "sop_dissent": 0.05},
        {"ticker": "AAA", "as_of": "2020-05-01", "sop_dissent": 0.09},
        # a LATER filing that disclosed nothing for this field: it must not reset the clock
        {"ticker": "AAA", "as_of": "2021-05-01", "sop_dissent": None},
        # a different ticker whose only filing is ancient -> fully expired
        {"ticker": "BBB", "as_of": "2018-03-01", "sop_dissent": 0.42},
    ])
    daily = _daily(hist, "sop_dissent", idx)
    capped = expire_stale(daily, hist, "sop_dissent")

    def at(d: str, t: str):
        return capped.loc[capped.index.asof(pd.Timestamp(d)), t]

    # 2020-06-01 is 396d after the 2019 filing but only 31d after the 2020 one -> alive at 0.09
    assert at("2020-06-01", "AAA") == pytest.approx(0.09)
    # 2021-10-01 is 518d after the 2020-05-01 filing that produced the cell -> still alive,
    # even though a 2021-05-01 filing exists: that filing carried no value, so it is not the
    # cell's producer and must not have restarted the clock.
    assert at("2021-10-01", "AAA") == pytest.approx(0.09)
    # 2022-01-01 is 611d after it -> expired
    assert pd.isna(at("2022-01-01", "AAA"))
    # BBB's single 2018 filing is dead everywhere after its horizon
    assert at("2018-06-01", "BBB") == pytest.approx(0.42)
    assert pd.isna(at("2021-01-01", "BBB"))

    frames, stats = expire_event_fields({"sop_dissent": daily}, hist, {"sop_dissent"})
    expired, before = stats["sop_dissent"]

    print("\n=== SANITY CHECK: governance event expiry (provenance) ===")
    print("  AAA filed 2019-05-01 (0.05), 2020-05-01 (0.09), 2021-05-01 (NULL).")
    print("   2020-06-01 -> 0.09 (31d old)   2021-10-01 -> 0.09 (518d old, the NULL filing")
    print("   did NOT restart the clock)     2022-01-01 -> NaN (611d old)")
    print("  BBB's lone 2018 filing is alive at +3mo and dead by 2021.")
    print(f"  expire_event_fields bite: {expired} of {before} non-null cells "
          f"({expired / before:.1%})")
    print("  CONCLUSION: the age is measured against the filing that PRODUCED the cell,")
    print("  which is the only reason a forward-filled frame can be expired at all.")


def test_expiry_bite_on_real_def14a():
    """How hard would 548 days bite each proxy field? The number phases 3-5 need."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load("def14a_llm")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")

    from src.data_aggregate.utils.governance.def14a_impute import impute_def14a
    imp, _ = impute_def14a(raw)
    idx = pd.bdate_range("2011-01-03", "2026-09-04")        # the modern era only

    fields = ["say_on_pay_support_pct", "ceo_total_comp", "avg_other_public_boards",
              "board_size", "pct_independent_directors", "poison_pill", "majority_voting"]
    rows = []
    for f in fields:
        if f not in imp.columns:
            continue
        daily = fundamentals_to_daily(imp, f, idx)
        if daily.empty:
            continue
        before = int(daily.notna().to_numpy().sum())
        # ⚠ `feature=` is passed a name that is deliberately NOT in the exemption, so this
        # measures the HORIZON, not the exemption. Two of these fields (`board_size`,
        # `pct_independent_directors`) share their name with a legacy FEATURE, so calling
        # `expire_stale(daily, imp, f)` bare would return them untouched and this table would
        # report 0.0% — reading as "a board size never goes stale" when it means "exempt".
        capped = expire_stale(daily, imp, f, feature=f"_measure_{f}")
        after = int(capped.notna().to_numpy().sum())
        exempt = f in LEGACY_EXEMPT_FROM_EXPIRY
        rows.append((f, before, before - after,
                     (before - after) / before if before else 0.0, exempt))

    assert rows, "no field produced a daily frame"
    # An expiry can only remove cells, never add them.
    assert all(n >= 0 for _, _, n, _, _ in rows)
    # and the exemption really does short-circuit the ones that carry a legacy name
    for f, _, _, _, exempt in rows:
        if exempt:
            d = fundamentals_to_daily(imp, f, idx)
            assert expire_stale(d, imp, f) is d, f"{f} is exempt but was not passed through"

    print("\n=== SANITY CHECK: what a 548-day horizon would cost each proxy field ===")
    print(f"  daily grid {idx[0].date()}..{idx[-1].date()} ({len(idx)} days), post-impute")
    print(f"  {'field':<28} {'non-null':>10} {'expired':>10} {'share':>8}  exempt?")
    for f, before, n, share, exempt in sorted(rows, key=lambda r: -r[3]):
        print(f"  {f:<28} {before:>10,} {n:>10,} {share:>7.1%}  "
              f"{'EXEMPT (legacy)' if exempt else ''}")
    worst = max(rows, key=lambda r: r[3])
    print(f"  worst: {worst[0]} at {worst[3]:.1%}")
    print("  CONCLUSION: this is the HORIZON's bite, measured with the legacy exemption")
    print("  deliberately bypassed — the two fields marked EXEMPT would report 0.0% otherwise,")
    print("  which would read as 'never stale' rather than 'never expired'. None of these is")
    print("  expired in the tree today; the horizon applies to the vote families phases 3-5")
    print("  add. Read each as the cost of a MISSED FILING CYCLE: >20% means 548 days is too")
    print("  tight for that field and belongs in the report, not in a silent retune.")
