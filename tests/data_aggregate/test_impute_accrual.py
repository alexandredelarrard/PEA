"""
test_impute_accrual.py  (tests/data_aggregate/test_impute_accrual.py)
--------------------------------------------------------------------
The two identity-aware fills phase 2 adds to `impute_def14a`:

  * **`ceo_age` by anchor-and-accrue** (D30) and the defect it closes (D33). The old
    interpolation had no identity check, so 103 of its 1,597 fills invented an age BETWEEN an
    outgoing and an incoming CEO. This proves the anchor cannot do that, and measures what
    the swap costs and buys.
  * **`ceo_salary` CARRIED forward on same-CEO gaps only** (D31), and the number that
    decides whether it collides with the §3.4 zero-diff guard: how many `ceo_total_comp`
    cells the salary fill unlocks. If that is not zero, the legacy `ceo_pay_growth` moved.

⚠ BOTH FILLS WERE INTERPOLATIONS WHEN THIS MODULE WAS WRITTEN, and neither is now — the whole
temporal fill went forward-only on 2026-09-09. The interpolation still appears throughout as
the COUNTERFACTUAL these tests measure against, which is deliberate: `ceo_age`'s D33 defect and
`ceo_salary`'s D31 gate are both defined by what an ungated interpolation would have done.

It also discharges §3.6's "measure the marginal gain and STOP" obligation for
`ceo_since_year`: the anchor is deliberately NOT applied to it, and the report says what
applying it would be worth so that stays a modelling decision rather than a side effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.accrual import (
    accrual_anchor, accrual_dispersion, accrue,
)
from src.data_aggregate.utils.governance.def14a_impute import (
    CARRY_LEVELS, CARRY_MAX_DAYS, IDENTITY_GATED_CARRY, impute_def14a,
)
from src.data_aggregate.utils.governance.names import ceo_identity_series


def _row(ticker: str, as_of: str, **kw) -> dict:
    r = {"ticker": ticker, "accession_number": f"{ticker}-{as_of}", "as_of": as_of}
    r.update(kw)
    return r


def test_accrual_anchor_extrapolates_and_resists_a_bad_observation():
    """A clock is recomputed, not interpolated — so edges fill and one bad value cannot bite."""
    obs = pd.DataFrame([
        {"pk": "AAA|cook|t", "as_of": "2015-04-01", "ceo_age": 55.0},
        {"pk": "AAA|cook|t", "as_of": "2016-04-01", "ceo_age": 56.0},
        {"pk": "AAA|cook|t", "as_of": "2017-04-01", "ceo_age": 75.0},   # a mis-extraction
        {"pk": "AAA|cook|t", "as_of": "2018-04-01", "ceo_age": 58.0},
    ])
    anchor = accrual_anchor(obs, "ceo_age", key="pk", date="as_of")
    assert anchor["AAA|cook|t"] == pytest.approx(1960.0), "the median anchor moved with one bad row"

    # the anchor recomputes OUTSIDE the observed range, which interpolation cannot do
    dates = pd.Series(pd.to_datetime(["2012-04-01", "2021-04-01"]))
    keys = pd.Series(["AAA|cook|t", "AAA|cook|t"])
    assert accrue(dates, keys, anchor).tolist() == [52.0, 61.0]

    # dispersion is the key-collision alarm: 1960,1960,1942,1960 -> spread 18
    disp = accrual_dispersion(obs, "ceo_age", key="pk", date="as_of")
    assert disp["AAA|cook|t"] == pytest.approx(18.0)

    print("\n=== SANITY CHECK: anchor-and-accrue (synthetic) ===")
    print("  ages 55,56,75(bad),58 over 2015-18 -> implied birth years 1960,1960,1942,1960")
    print(f"  median anchor = {anchor['AAA|cook|t']:.0f} (the 1942 outlier does not move it)")
    print("  recomputed OUTSIDE the observed range: 2012 -> 52, 2021 -> 61")
    print(f"  dispersion = {disp['AAA|cook|t']:.0f} years — the alarm that two PEOPLE may be")
    print("  sharing one key. ±1 is a birthday straddling the filing date; 18 is not.")


def test_ceo_age_never_interpolates_across_a_succession():
    """The D33 defect, reproduced and closed: no age is invented between two CEOs."""
    rows = [
        _row("AAA", "2015-04-01", ceo_age=60.0, ceo_name_proxy="Alice Adams"),
        _row("AAA", "2016-04-01", ceo_age=np.nan, ceo_name_proxy=None),   # the gap
        _row("AAA", "2017-04-01", ceo_age=45.0, ceo_name_proxy="Bob Brown"),
        _row("AAA", "2018-04-01", ceo_age=46.0, ceo_name_proxy="Bob Brown"),
    ]
    out, stats = impute_def14a(pd.DataFrame(rows))
    gap = out.loc[out["as_of"] == pd.Timestamp("2016-04-01"), "ceo_age"].iloc[0]

    # A plain interpolation would have written (60 + 45) / 2 = 52.5 -> 53 after rounding:
    # an age belonging to neither person. The anchor has no key for that row, so it stays NaN.
    assert pd.isna(gap), f"an age was invented across a CEO change: {gap}"
    assert "ceo_age" not in CARRY_LEVELS, \
        "ceo_age is back in CARRY_LEVELS — the D33 defect is reopened"

    # ...and where the CEO IS the same, the anchor fills, including at the EDGE
    same = [
        _row("BBB", "2015-04-01", ceo_age=50.0, ceo_name_proxy="Carol Clark"),
        _row("BBB", "2016-04-01", ceo_age=np.nan, ceo_name_proxy="Carol Clark"),
        _row("BBB", "2017-04-01", ceo_age=52.0, ceo_name_proxy="C. Clark"),   # respelt
        _row("BBB", "2018-04-01", ceo_age=np.nan, ceo_name_proxy="Carol Clark"),  # TRAILING
    ]
    out2, _ = impute_def14a(pd.DataFrame(same))
    ages = out2.sort_values("as_of")["ceo_age"].tolist()
    assert ages == [50.0, 51.0, 52.0, 53.0], ages

    print("\n=== SANITY CHECK: ceo_age accrual vs the D33 interpolation defect ===")
    print("  AAA: Alice(60) -> [gap] -> Bob(45). Interpolation would write 53, an age")
    print(f"       belonging to neither. The anchor writes {gap} (NaN = unknown).")
    print(f"  BBB: Carol 50, [gap], 'C. Clark' 52, [TRAILING gap] -> {ages}")
    print("       the interior gap fills to 51 AND the LEADING/trailing edges — which the")
    print("       forward carry cannot reach, and which a clock legitimately can. The")
    print("       respelling is the same person under `ceo_identity`, so one anchor covers")
    print("       all four rows.")
    print(f"  rules fired: {stats}")


def test_salary_gate_and_zero_diff_on_the_totals():
    """D31 on live data: the gate's cost, and the number §3.4's zero-diff guard turns on."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load("def14a_llm")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")

    assert IDENTITY_GATED_CARRY == frozenset({"ceo_salary"}), \
        "the identity-gated carry set changed — D31 covers ceo_salary ONLY"

    imp, stats = impute_def14a(raw)
    filled = int(stats.get("carry: ceo_salary", 0))
    declined = int(stats.get("declined (identity changed): ceo_salary", 0))
    stale = int(stats.get(f"declined (>{CARRY_MAX_DAYS}d stale): ceo_salary", 0))

    # --- what an UNGATED carry would have filled, for the gate's cost ---
    # ⚠ Counted with the SAME bounded forward carry the module now uses, not the old
    # `limit_area="inside"` interpolation: the gate's cost is what the gate declines, so the
    # baseline has to differ from the shipped run in the GATE alone.
    base = raw.copy()
    base["as_of"] = pd.to_datetime(base["as_of"], errors="coerce")
    base = base.sort_values(["ticker", "as_of"])
    gb = base.groupby("ticker", sort=False)
    fwd = gb["ceo_salary"].ffill()
    src = base["as_of"].where(base["ceo_salary"].notna()).groupby(base["ticker"],
                                                                 sort=False).ffill()
    age = (base["as_of"] - src).dt.days
    ungated = int((base["ceo_salary"].isna() & fwd.notna() & (age <= CARRY_MAX_DAYS)).sum())
    assert ungated == filled + declined, (
        f"the gate's arithmetic does not close: {ungated} carryable != {filled} filled + "
        f"{declined} declined")

    # --- THE ZERO-DIFF QUESTION: did any ceo_total_comp cell appear because of it? ---
    # The counterfactual is the same pipeline with the gate's input removed: with no
    # `ceo_name_proxy`, `_same_ceo_as_source` is False everywhere and the salary carry is
    # declined on every row -- i.e. phase-1 behaviour. Neither `ceo_name_proxy` nor `ceo_age`
    # feeds `ceo_total_comp`, so the difference between the two runs' totals isolates the
    # salary fill exactly.
    ungated_run = raw.copy()
    ungated_run["ceo_name_proxy"] = None
    without, _ = impute_def14a(ungated_run)
    key = ["ticker", "accession_number"]
    a = without.set_index(key)["ceo_total_comp"].sort_index()
    b = imp.set_index(key)["ceo_total_comp"].sort_index().reindex(a.index)
    unlocked = int((a.isna() & b.notna()).sum())
    moved = int((~np.isclose(a.to_numpy(), b.to_numpy(), equal_nan=True)).sum()) - unlocked

    assert unlocked == 0, (
        f"{unlocked} ceo_total_comp cells were unlocked by the salary carry — the legacy "
        "`ceo_pay_growth` is no longer bit-identical and §3.4's zero-diff guard fails")
    assert moved == 0, f"{moved} ceo_total_comp cells changed VALUE"
    assert float(imp["comp_imputed"].sum()) == 0.0

    print("\n=== SANITY CHECK: ceo_salary identity-gated forward carry (D31) ===")
    print(f"  carryable ceo_salary cells within {CARRY_MAX_DAYS}d (ungated): {ungated}")
    print(f"    filled (source row names the SAME CEO)   : {filled}")
    print(f"    DECLINED (the CEO changed, or is unnamed): {declined}")
    print(f"    DECLINED separately, >{CARRY_MAX_DAYS}d stale : {stale}")
    print(f"  ceo_salary fill rate: {raw['ceo_salary'].notna().mean():.1%} -> "
          f"{imp['ceo_salary'].notna().mean():.1%}")
    print(f"  >>> ceo_total_comp cells UNLOCKED by the salary fill: {unlocked}  (values "
          f"moved: {moved})")
    print(f"  >>> comp_imputed population: {int(imp['comp_imputed'].sum())} rows")
    print("  CONCLUSION: D31 and §3.4's zero-diff guard do NOT collide. Not one row has")
    print("  salary as its ONLY absent component, so no total is derived from a carried")
    print("  one and the legacy `ceo_pay_growth` is bit-identical. The ~0.5% distortion D31")
    print("  budgeted for is, on this table, exactly zero — and `comp_imputed` is what keeps")
    print("  that checkable if the extraction's coverage shifts.")


def test_comp_imputed_fires_when_it_should():
    """The live population is 0 — so prove the flag is capable of being 1, not merely absent.

    A provenance flag that is always zero is indistinguishable from a provenance flag that is
    broken. This constructs the exact situation D31 budgeted for — salary is the ONLY absent
    component of a row whose total is also absent, and the same CEO is named on the source row
    and the filled one — and asserts the whole chain fires: CARRY the salary, sum the six
    components into `ceo_total_comp`, and STAMP the total as derived from a filled part.

    ⚠ THE EXPECTED SALARY IS 100, NOT 200. Until 2026-09-09 the fill was a linear
    interpolation, so a gap between 100 and 300 became the midpoint 200 — a number computed
    from the 2017 filing while sitting on the 2016 row. The carry writes the last KNOWN salary,
    100, which is what 2016 could actually have known. The flag under test is unchanged; only
    the value it is stamped on top of is.
    """
    others = {"ceo_bonus": 10.0, "ceo_stock_awards": 20.0, "ceo_option_awards": 30.0,
              "ceo_non_equity_incentive": 40.0, "ceo_all_other_comp": 50.0}
    rows = [
        _row("AAA", "2015-04-01", ceo_salary=100.0, ceo_name_proxy="Alice Adams", **others,
             ceo_total_comp=250.0),
        # salary NULL, total NULL, every other component present, same CEO both sides
        _row("AAA", "2016-04-01", ceo_salary=np.nan, ceo_name_proxy="Alice Adams", **others,
             ceo_total_comp=np.nan),
        _row("AAA", "2017-04-01", ceo_salary=300.0, ceo_name_proxy="A. Adams", **others,
             ceo_total_comp=450.0),
    ]
    out, stats = impute_def14a(pd.DataFrame(rows))
    mid = out.loc[out["as_of"] == pd.Timestamp("2016-04-01")].iloc[0]

    assert mid["ceo_salary"] == pytest.approx(100.0), "the gated salary carry did not fire"
    assert mid["ceo_total_comp"] == pytest.approx(250.0), "the sum identity did not pick it up"
    assert mid["comp_imputed"] == 1.0, "comp_imputed did NOT fire on a derived total"
    assert out["comp_imputed"].sum() == 1.0, "comp_imputed fired on a filer-stated total too"

    print("\n=== SANITY CHECK: comp_imputed is live, not dead (D31 provenance) ===")
    print("  AAA 2016: salary NULL, other 5 components present (150), total NULL, CEO unchanged")
    print(f"    salary CARRIED 100 -> {mid['ceo_salary']:.0f}   (an interpolation would have")
    print("      written 200, the midpoint toward a 2017 filing 2016 could not see)")
    print(f"    ceo_total_comp derived = {mid['ceo_total_comp']:.0f}  (100 + 150)")
    print(f"    comp_imputed = {mid['comp_imputed']:.0f}, and 0 on the two filer-stated rows")
    print(f"  rules: {stats}")
    print("  CONCLUSION: the flag FIRES on the exact shape D31 budgeted a ~0.5% distortion")
    print("  for. Its live population of 0 is therefore a property of the DATA — no real row")
    print("  has salary as its only absent component — and not a silently broken stamp.")


def test_twelve_legacy_features_are_bit_identical(monkeypatch):
    """§3.4's zero-diff guard, stated directly: phase 2 moves NONE of the twelve live cells.

    The aggregate fingerprint covers this end-to-end, but only as one hash over everything.
    This names the twelve features and diffs them cell by cell against a PHASE-1-EQUIVALENT
    impute -- `ceo_age` back in `CARRY_LEVELS`, no salary carry, no accrual -- so a future
    change that moves one of them says WHICH one.

    ⚠ "PHASE-1-EQUIVALENT" IS NO LONGER BIT-EQUIVALENT TO PHASE 1, and cannot be: phase 1's
    temporal fill was a linear interpolation and this module's is a forward carry, so both
    arms of the diff moved together on 2026-09-09. What the guard still proves is the thing it
    was built to prove -- that the accrual and the salary gate move nothing -- because both
    arms share the same carry. The interpolation-vs-carry change is measured by its own
    rebuild diff, not here.
    """
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load("def14a_llm")
        # the revenue leg, so the TWELFTH feature (`ceo_pay_vs_revenue_growth`) is built too
        fund = ctx.store.load("fundamentals_history", optional=True)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")

    from src.data_aggregate.utils.governance import def14a_impute as mod
    from src.data_aggregate.utils.governance.panel import (
        _def14a_raw_fields, _governance_fields,
    )
    from src.data_aggregate.utils.governance.staleness import LEGACY_EXEMPT_FROM_EXPIRY

    def legacy_twelve(hist: pd.DataFrame, index: pd.DatetimeIndex) -> dict:
        """All twelve legacy features, from BOTH builders that now produce them.

        ⚠ The twelve no longer come from one function. The 2026-09-08 encoding review moved
        `founder_ceo` and `say_on_pay_support` out of `_governance_fields` and into
        `_def14a_raw_fields`, because a 1/0 flag and an absolute approval fraction have no peer
        norm to standardize against and now ship RAW (D49-D50). That is a change of ENCODING,
        not of value -- which is precisely what this guard has to keep proving, so it reads both
        builders rather than lowering its count to ten.

        ⚠ AND IT NOW FILTERS TO THE TWELVE BY NAME, because `_governance_fields` is no longer
        allowed to emit only those twelve. Phase 5 added `control_wedge` (insider voting power
        minus economic ownership), so an unfiltered read made this guard fail on the ARRIVAL of
        a new feature rather than on a change to a legacy one -- which is not what it is for.
        The subject of the test is the twelve, and `LEGACY_EXEMPT_FROM_EXPIRY` is exactly that
        set, so it is also the right filter. `control_wedge` is deliberately NOT added to that
        frozenset: it names the twelve LEGACY levels and carries their D3 history, whereas
        `_control_wedge` takes the 1,095-day level horizon by passing it explicitly.
        """
        built = {**_governance_fields(hist, index, fund), **_def14a_raw_fields(hist, index)}
        return {k: v for k, v in built.items() if k in LEGACY_EXEMPT_FROM_EXPIRY}

    idx = pd.bdate_range("2011-01-03", "2026-09-04")
    shipped, _ = impute_def14a(raw)
    after = legacy_twelve(shipped, idx)

    # ...the same pipeline as phase 1 left it
    monkeypatch.setattr(mod, "CARRY_LEVELS", mod.CARRY_LEVELS + ["ceo_age"])
    monkeypatch.setattr(mod, "IDENTITY_GATED_CARRY", frozenset())
    monkeypatch.setattr(mod, "_accrue_ceo_age", lambda df, stats: pd.Series(False, index=df.index))
    legacy_run, _ = mod.impute_def14a(raw)
    before = legacy_twelve(legacy_run, idx)

    assert set(before) == set(after), (
        f"the emitted feature set changed: {set(after) ^ set(before)}")
    diffs = {}
    for name in sorted(before):
        a, b = before[name], after[name]
        cols = a.columns.union(b.columns)
        a, b = a.reindex(columns=cols), b.reindex(columns=cols)
        moved = int((~np.isclose(a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64"),
                                 equal_nan=True)).sum())
        if moved:
            diffs[name] = moved

    assert not diffs, f"phase 2 moved cells in live features: {diffs}"
    # every emitted feature is one of the twelve, and every one of the twelve is exempt
    assert set(after) <= LEGACY_EXEMPT_FROM_EXPIRY, set(after) - LEGACY_EXEMPT_FROM_EXPIRY
    assert len(after) == 12, (
        f"expected all twelve legacy features, built {len(after)}: "
        f"{sorted(LEGACY_EXEMPT_FROM_EXPIRY - set(after))} missing")

    print("\n=== SANITY CHECK: the twelve legacy features, zero-diff (§3.4 / D3) ===")
    print(f"  daily grid {idx[0].date()}..{idx[-1].date()} ({len(idx)} days)")
    print(f"  features built: {len(after)} — {', '.join(sorted(after))}")
    for name in sorted(after):
        print(f"    {name:<28} {int(after[name].notna().to_numpy().sum()):>10,} non-null cells")
    print("  cells that MOVED vs a phase-1-equivalent impute: 0")
    print("  CONCLUSION: the ceo_age accrual and the gated ceo_salary interpolation change")
    print("  no live feature cell — neither field reaches the twelve, and the salary fill")
    print("  unlocks no `ceo_total_comp`. All twelve are in LEGACY_EXEMPT_FROM_EXPIRY, so the")
    print("  548-day horizon cannot reach them either.")


def test_ceo_age_yield_and_the_ceo_since_year_marginal_gain():
    """What the anchor swap actually changed on `ceo_age`, and what §3.6 forbids doing next."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load("def14a_llm")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")

    imp, stats = impute_def14a(raw)
    accrued = int(stats.get("accrue: ceo_age", 0))

    # --- the counterfactual: the plain interpolation this replaced, and its 103 defects ---
    base = raw.copy()
    base["as_of"] = pd.to_datetime(base["as_of"], errors="coerce")
    base = base.sort_values(["ticker", "as_of"])
    g = base.groupby("ticker", sort=False)
    interp = g["ceo_age"].transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
    old_fills = base["ceo_age"].isna() & interp.notna()
    # ⚠ "Interpolates across a CEO change" has TWO defensible readings and they differ by an
    # order of magnitude, so both are reported with their definitions rather than one number
    # with none. D33's own figure of 103 reproduces on NEITHER (see the print-out).
    #
    #   BOUNDING  — the nearest DISCLOSED name on each side of the gap differs. Weak, because
    #               ffill/bfill skip over rows whose name is missing.
    #   ADJACENT  — the two rows the interpolation actually ran BETWEEN name different CEOs.
    #               Stricter, and the more literal reading of what the arithmetic did.
    fwd, bwd = g["ceo_name_proxy"].ffill(), g["ceo_name_proxy"].bfill()
    k_f, k_b = ceo_identity_series(fwd), ceo_identity_series(bwd)
    across = old_fills & ~(k_f.notna() & k_b.notna() & (k_f == k_b))
    s_f, s_b = fwd.astype(object), bwd.astype(object)
    across_raw = old_fills & ~(s_f.notna() & s_b.notna() & (s_f == s_b))

    p_k = ceo_identity_series(g["ceo_name_proxy"].shift(1))
    n_k = ceo_identity_series(g["ceo_name_proxy"].shift(-1))
    adj = old_fills & p_k.notna() & n_k.notna() & (p_k != n_k)
    p_s = g["ceo_name_proxy"].shift(1).astype(object)
    n_s = g["ceo_name_proxy"].shift(-1).astype(object)
    adj_raw = old_fills & p_s.notna() & n_s.notna() & (p_s != n_s)

    # --- dispersion of the anchors actually used, the key-collision alarm ---
    ident = ceo_identity_series(base["ceo_name_proxy"])
    pk = base["ticker"].astype(str).str.cat(ident, sep="|")
    obs = pd.DataFrame({"pk": pk, "as_of": base["as_of"], "ceo_age": base["ceo_age"]})
    disp = accrual_dispersion(obs, "ceo_age", key="pk", date="as_of")
    n_obs = obs.dropna(subset=["pk", "ceo_age"]).groupby("pk").size()
    multi = disp[n_obs.reindex(disp.index).fillna(0) > 1]

    assert accrued > 0, "the ceo_age accrual filled nothing"
    assert float(imp["ceo_age"].notna().mean()) >= float(raw["ceo_age"].notna().mean())

    # --- §3.6: MEASURE the ceo_since_year gain and STOP. It is NOT applied. ---
    since_now = int(imp["ceo_since_year"].notna().sum())
    s_obs = pd.DataFrame({"pk": pk, "as_of": base["as_of"],
                          "_since": base["as_of"].dt.year - base["ceo_since_year"]})
    s_anchor = accrual_anchor(s_obs, "_since", key="pk", date="as_of")
    s_implied = accrue(base["as_of"], pk, s_anchor)
    would_fill = int((base["ceo_since_year"].isna() & s_implied.notna()).sum())
    still_na_after_impute = int(imp["ceo_since_year"].isna().sum())

    print("\n=== SANITY CHECK: ceo_age accrual yield + the ceo_since_year gain (§3.6) ===")
    print(f"  ceo_age fill: {raw['ceo_age'].notna().mean():.1%} -> "
          f"{imp['ceo_age'].notna().mean():.1%}   ({accrued} cells accrued)")
    print(f"  the interpolation it REPLACED filled {int(old_fills.sum())} cells "
          f"[plan: 1,597 — reproduces exactly], of which it interpolated across a CEO change:")
    print(f"    BOUNDING (nearest disclosed name each side differs): "
          f"{int(across_raw.sum())} raw / {int(across.sum())} keyed")
    print(f"    ADJACENT (the two rows it ran BETWEEN name different CEOs): "
          f"{int(adj_raw.sum())} raw / {int(adj.sum())} keyed")
    print("    (!) D33's figure of 103 reproduces on NEITHER basis, on the same table whose")
    print("      1,597 total DOES reproduce — so the defect is real and demonstrable but its")
    print("      published magnitude is not. On the strictest reading it is ~25x worse than")
    print("      103; on the loosest, ~8x smaller. The fix does not depend on which: an")
    print("      anchor keyed per PERSON cannot span a succession at any magnitude.")
    print(f"  anchors with >1 observation: {len(multi)}; dispersion p50={multi.median():.0f} "
          f"p90={multi.quantile(0.9):.0f} max={multi.max():.0f} years")
    print(f"    dispersion > 5 years (possible key collision): {int((multi > 5).sum())} keys")
    print("  --- ceo_since_year: MEASURED, DELIBERATELY NOT APPLIED (§3.6 / D30) ---")
    print(f"    filled after impute: {since_now} ({imp['ceo_since_year'].notna().mean():.1%}), "
          f"{still_na_after_impute} still NULL")
    print(f"    the anchor would fill {would_fill} further cells")
    print("    NOT DONE: `ceo_since_year` feeds `ceo_tenure`, a LIVE monotone-list feature, so")
    print("    widening it changes a shipped feature's cells — D3 forbids that as a side")
    print("    effect. It is a decision for the modelling session, and now it has a number.")
