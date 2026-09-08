"""
The STRICT PRIOR-YEAR LAG, and the D44 reconciliation that makes it necessary.

D44, measured in phase 3: `def14a_llm.say_on_pay_support_pct` and `sec_8k_votes`' say-on-pay
tally describe THE SAME MEETING ONE YEAR APART -- the proxy discloses the PRIOR year's vote.
Compared at the same `as_of` the two look ~60 points apart (JPM: 89% vs 31%); shifted back a
year they agree within 2pp on 93.0% of 4,379 overlapping company-years. So the pair is one
quantity at two latencies, not two pieces of evidence, and anything that puts them side by side
is wrong unless the lag is explicit.

⚠ EVERY TEST HERE MUST FAIL IF THE LAG IS DROPPED. A test that merely reports a correlation
passes just as happily on a mis-aligned join -- which is exactly how the 60-point gap survived
until phase 3 went looking. The real-data check is therefore TWO assertions: the lagged
agreement clears the bar, AND the un-lagged comparison does not. Only the second one proves the
lag is doing the work.

The primitive under test is `pay_features.prior_annual_leg`, which is what `_comp_history` uses
for the prior-year package, so this is the shipped rule and not a re-implementation of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.pay_features import prior_annual_leg

_AGREE_WITHIN = 0.02        # 2pp, D44's own tolerance
_AGREE_SHARE = 0.90         # the gate: 90% of overlapping company-years


def _annual(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """`[ticker, as_of, v]`, sorted the way `prior_annual_leg` requires of its caller."""
    df = pd.DataFrame([{"ticker": t, "as_of": pd.Timestamp(d), "v": v} for t, d, v in rows])
    return df.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def test_the_prior_year_leg_is_nan_when_the_prior_year_is_missing():
    """The four cases, asserted BY VALUE rather than by count."""
    hist = _annual([
        # AAA: three consecutive proxies -> two clean pairs
        ("AAA", "2021-05-01", 0.90),
        ("AAA", "2022-05-01", 0.31),
        ("AAA", "2023-05-01", 0.89),
        # BBB: a HOLE. 2022 is missing, so the 2023 row's prior year does not exist.
        ("BBB", "2020-05-01", 0.70),
        ("BBB", "2021-05-01", 0.65),
        ("BBB", "2023-05-01", 0.60),
        # CCC: present in this source only -> its single row has no prior year at all
        ("CCC", "2022-05-01", 0.55),
    ])
    lag = prior_annual_leg(hist, "v")
    at = dict(zip(zip(hist["ticker"], hist["as_of"].dt.year), lag))

    # 1. consecutive years pair up, and carry the PRIOR year's value -- not this year's
    assert at[("AAA", 2022)] == pytest.approx(0.90)
    assert at[("AAA", 2023)] == pytest.approx(0.31)
    # 2. ⚠ THE HOLE. 2021's 0.65 is two years before 2023 and must NOT be borrowed: a plain
    #    `groupby.shift(1)` returns 0.65 here, silently relabelling a 2y lag as a 1y lag.
    assert np.isnan(at[("BBB", 2023)]), "Y-2 was carried forward into a Y-1 slot"
    assert at[("BBB", 2021)] == pytest.approx(0.70)
    # 3. a first observation has no prior year and does not borrow a LATER one
    assert np.isnan(at[("AAA", 2021)])
    assert np.isnan(at[("BBB", 2020)])
    # 4. a ticker with one row keeps its row -- the leg is NaN, the row is not dropped
    assert np.isnan(at[("CCC", 2022)])
    assert len(lag) == len(hist), "the lag must not drop rows"

    # 5. and nothing is zero-filled: NaN is NaN, never 0.0, which would read as "no support"
    assert not (lag.fillna(-1) == 0.0).any()

    print("\n=== SANITY CHECK: the strict prior-year lag ===")
    print(f"  consecutive years pair by VALUE (2023 <- {at[('AAA', 2023)]:.2f}).")
    print("  a MISSING year -> NaN, not Y-2's value: BBB 2023 lag is NaN while 2021 holds 0.65,")
    print("  which is exactly what a bare groupby.shift(1) would have handed back.")
    print("  a first row -> NaN (no back-fill from a later year); a lone row survives.")
    print("  CONCLUSION: the hole stays VISIBLE. Validated.")


def test_the_two_sources_reconcile_only_under_the_lag():
    """D44 as a GATE on live data: the lag clears 90% within 2pp and the un-lagged join fails.

    Both assertions matter. Without the second, this test would pass on a mis-aligned join.
    """
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        proxy = ctx.store.load("def14a_llm")
        votes = ctx.store.load("sec_8k_votes")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"live governance tables not reachable ({e})")
    if proxy is None or proxy.empty or votes is None or votes.empty:
        pytest.skip("def14a_llm or sec_8k_votes empty")

    from src.data_aggregate.utils.governance.vote_dissent_features import _say_on_pay_history

    hist = _say_on_pay_history(votes, {})
    if hist is None or hist.empty:
        pytest.skip("no say-on-pay vote rows")
    # The 8-K is filed days after the meeting, so its filing year IS the meeting year.
    vote = pd.DataFrame({
        "ticker": hist["ticker"],
        "meeting_year": pd.to_datetime(hist["as_of"]).dt.year,
        "vote_support": 1.0 - pd.to_numeric(hist["sop_dissent"], errors="coerce"),
    }).dropna().drop_duplicates(["ticker", "meeting_year"], keep="last")

    p = pd.DataFrame({
        "ticker": proxy["ticker"],
        "proxy_year": pd.to_datetime(proxy["as_of"], errors="coerce").dt.year,
        "proxy_support": pd.to_numeric(proxy["say_on_pay_support_pct"], errors="coerce"),
    }).dropna().drop_duplicates(["ticker", "proxy_year"], keep="last")

    def share_agreeing(shift: int) -> tuple[float, int]:
        """Share of overlapping company-years agreeing within 2pp when the proxy is read as
        describing `proxy_year - shift`. `shift=1` is the D44 claim; `shift=0` is the naive
        same-`as_of` join that looked like a 60-point basis conflict."""
        q = p.assign(meeting_year=p["proxy_year"] - shift)
        m = q.merge(vote, on=["ticker", "meeting_year"], how="inner")
        if m.empty:
            return float("nan"), 0
        gap = (m["proxy_support"] - m["vote_support"]).abs()
        return float((gap <= _AGREE_WITHIN).mean()), int(len(m))

    lagged, n_lag = share_agreeing(1)
    naive, n_naive = share_agreeing(0)

    assert n_lag >= 1000, f"too few overlapping company-years to conclude ({n_lag})"
    assert lagged >= _AGREE_SHARE, (
        f"the lagged join agrees on only {lagged:.1%} of {n_lag} company-years")
    # ⚠ THE ASSERTION THAT MAKES THIS A TEST. If the un-lagged join agreed too, the lag would
    # be decoration and this file would be proving nothing.
    assert naive < _AGREE_SHARE, (
        f"the UN-lagged join also agrees ({naive:.1%}) -- the lag is not doing the work, so "
        f"either D44 is wrong or one of these legs changed basis")

    print("\n=== SANITY CHECK: D44 reconciliation as a gate ===")
    print(f"  proxy(Y) vs meeting(Y-1): {lagged:.1%} of {n_lag:,} company-years agree "
          f"within {_AGREE_WITHIN:.0%}")
    print(f"  proxy(Y) vs meeting(Y)  : {naive:.1%} of {n_naive:,}  <- the naive same-as_of join")
    print("  CONCLUSION: the two sources are ONE quantity at two latencies. Validated.")


def test_the_named_worked_examples():
    """JPM and INTC meeting-2022, the two cases that exposed the lag. Asserted, not printed."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        proxy = ctx.store.load("def14a_llm")
        votes = ctx.store.load("sec_8k_votes")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"live governance tables not reachable ({e})")
    if proxy is None or proxy.empty or votes is None or votes.empty:
        pytest.skip("def14a_llm or sec_8k_votes empty")

    from src.data_aggregate.utils.governance.vote_dissent_features import _say_on_pay_history

    hist = _say_on_pay_history(votes, {})
    if hist is None or hist.empty:
        pytest.skip("no say-on-pay vote rows")
    hist = hist.assign(year=pd.to_datetime(hist["as_of"]).dt.year)
    pr = proxy.assign(year=pd.to_datetime(proxy["as_of"], errors="coerce").dt.year)

    shown = []
    for tkr in ("JPM", "INTC"):
        v = hist[(hist["ticker"] == tkr) & (hist["year"] == 2022)]
        q = pr[(pr["ticker"] == tkr) & (pr["year"] == 2023)]
        if v.empty or q.empty:
            pytest.skip(f"{tkr} 2022 meeting / 2023 proxy absent")
        vote_support = 1.0 - float(pd.to_numeric(v["sop_dissent"], errors="coerce").iloc[-1])
        proxy_support = float(pd.to_numeric(q["say_on_pay_support_pct"],
                                           errors="coerce").iloc[-1])
        gap = abs(vote_support - proxy_support)
        assert gap <= _AGREE_WITHIN, (
            f"{tkr}: 8-K meeting-2022 support {vote_support:.3f} vs proxy-2023 "
            f"{proxy_support:.3f} -- gap {gap:.3f}")
        shown.append((tkr, vote_support, proxy_support, gap))

    print("\n=== SANITY CHECK: the two named worked examples ===")
    for tkr, vs, ps, gap in shown:
        print(f"  {tkr:<5} 8-K meeting-2022 support {vs:.3f} vs proxy-2023 {ps:.3f} "
              f"-> gap {gap:.4f}")
    print("  Both are near-revolts the proxy reports a year later, not a basis conflict.")
    print("  CONCLUSION: named, reproducible, and asserted. Validated.")
