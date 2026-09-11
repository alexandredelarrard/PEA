"""
Point-in-time elite-manager selection
(src/data_aggregate/utils/institutionals/manager_selection.py).

Proves the four properties that make this a SELECTOR rather than a filter applied with
hindsight. Each is a defect the module was written to avoid, not a hypothetical:

  1. NOTHING IN A SCORE COMES FROM A FILING THAT WAS NOT YET PUBLIC. Ranking a manager
     against every other filing for the same PERIOD uses filings made up to 1,311 days
     later; 16.5% of manager-quarters miss the 45-day deadline, so a within-period
     percentile is not reproducible from what a reader had on the day.
  2. THE `top_k` CUT TRACKS THE POOL. The eligible pool grows ~40 -> ~63 managers, so a
     fixed percentile threshold selects 15 managers early and 24 late.
  3. THE POOL IS THE ROSTER AS OF `q`, NOT TODAY'S. Today's roster drops the 19 culled
     managers that have a book and adds managers Dataroma had not yet listed -- and the
     cull is correlated with the selection criterion.
  4. THE ELIGIBILITY FLOOR REMOVES A MANAGER FROM THE RANKING, not just from the result.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.manager_selection import (
    eligibility, elite_weight, manager_concentration_score, selection_diagnostics,
)

_PERIODS = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
                           "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31"])


def _state(n_pos: dict[str, int], periods=_PERIODS, lag_days: dict | None = None,
           n_index: dict | None = None) -> pd.DataFrame:
    """A `manager_quarter_state`-shaped frame: manager `cik` runs `n_pos[cik]` equal-weight
    positions in every period, so `eff_n == n_positions` and `top10_weight == 10 / n_pos`
    and the concentration ordering is exactly the `n_pos` ordering, reversed."""
    lag_days = lag_days or {}
    rows = []
    for cik, n in n_pos.items():
        for p in periods:
            rows.append({
                "cik": cik, "period": p,
                "avail": p + pd.Timedelta(days=lag_days.get(cik, 45)),
                "n_positions": n, "eff_n": float(n),
                "top10_weight": min(10, n) / n,
                "n_index_positions": (n_index or {}).get(cik, n),
            })
    return pd.DataFrame(rows)


def _roster_all(_period) -> set[str]:
    return {f"m{i}" for i in range(20)}


# ------------------------------------------------------------------ 1. no look-ahead ---
def test_a_late_filing_cannot_change_an_earlier_managers_score():
    """m_late files 300 days after period end. Truncating history at a date BEFORE that
    filing must leave every already-public score bit-identical -- the plan's mandated
    truncation test, and the reason the rank runs on the availability grid."""
    st = _state({"m0": 5, "m1": 20, "m2": 40}, lag_days={"m0": 45, "m1": 45, "m2": 300})
    cut = pd.Timestamp("2021-06-30")

    full = manager_concentration_score(st)
    trunc = manager_concentration_score(st[st["avail"] <= cut])

    pre = st[st["avail"] <= cut].set_index(["cik", "period"]).index
    a = full.reindex(pre).sort_index()
    b = trunc.reindex(pre).sort_index()
    moved = ~np.isclose(a["score"].to_numpy(), b["score"].to_numpy(),
                        rtol=1e-12, equal_nan=True)
    print(f"pre-cut rows {len(pre)}, scores that moved when the late filer was removed: "
          f"{int(moved.sum())}")
    assert not moved.any(), a[moved].join(b[moved], rsuffix="_trunc")

    # and the late filer really was late enough to matter
    assert (st.loc[st["cik"] == "m2", "avail"] > cut).any()
    print("=== 1. a filing made after the cut changes NO score that was already public ===")


def test_within_period_ranking_would_have_leaked():
    """The control for the test above: rank inside the PERIOD instead and the same
    truncation moves scores. Proves the property is the grid's doing, not the fixture's."""
    st = _state({"m0": 5, "m1": 20, "m2": 40}, lag_days={"m0": 45, "m1": 45, "m2": 300})
    cut = pd.Timestamp("2021-06-30")

    def within_period(frame: pd.DataFrame) -> pd.Series:
        g = frame.groupby("period")
        s = (g["n_positions"].rank(pct=True, ascending=False)
             + g["eff_n"].rank(pct=True, ascending=False)
             + g["top10_weight"].rank(pct=True, ascending=True)) / 3.0
        return pd.Series(s.to_numpy(),
                         index=pd.MultiIndex.from_frame(frame[["cik", "period"]]))

    pre = st[st["avail"] <= cut].set_index(["cik", "period"]).index
    a = within_period(st).reindex(pre)
    b = within_period(st[st["avail"] <= cut]).reindex(pre)
    moved = int((~np.isclose(a.to_numpy(), b.to_numpy(), equal_nan=True)).sum())
    print(f"=== 2. CONTROL: a within-period rank moves {moved}/{len(pre)} already-public "
          f"scores on the same truncation ===")
    assert moved > 0


# --------------------------------------------------------------------- 2. the top-k ----
def test_top_k_selects_exactly_k_as_the_pool_grows():
    """All managers file on the same day, so the grid has one date per period and the cut is
    unambiguous. The pool grows 4 -> 8; `top_k=3` must stay 3, and a fixed percentile would
    not."""
    early = _state({f"m{i}": 5 * (i + 1) for i in range(4)}, periods=_PERIODS[:4])
    late = _state({f"m{i}": 5 * (i + 1) for i in range(8)}, periods=_PERIODS[4:])
    st = pd.concat([early, late], ignore_index=True)

    scored = manager_concentration_score(st)
    sel = elite_weight(scored, mode="top_k", k=3)
    per_period = sel.groupby(level="period").sum()
    pool = scored["n_public"].groupby(level="period").max()
    print(pd.DataFrame({"pool": pool, "selected": per_period}).to_string())
    assert (per_period == 3).all(), per_period
    assert pool.min() == 4 and pool.max() == 8
    print("=== 3. top_k=3 selects exactly 3 whether the pool is 4 or 8 ===")


def test_the_selected_managers_are_the_concentrated_ones():
    st = _state({"m0": 5, "m1": 10, "m2": 50, "m3": 200})
    sel = elite_weight(manager_concentration_score(st), mode="top_k", k=2)
    chosen = sorted(sel[sel > 0].index.get_level_values("cik").unique())
    print(f"n_positions 5/10/50/200 -> top-2 selected {chosen}")
    assert chosen == ["m0", "m1"]
    print("=== 4. fewest positions / lowest eff-N / highest top-10 weight is what wins ===")


def test_continuous_mode_keeps_the_gradient():
    st = _state({"m0": 5, "m1": 10, "m2": 50, "m3": 200})
    scored = manager_concentration_score(st)
    sel = elite_weight(scored, mode="continuous")
    assert sel.between(0.0, 1.0).all()
    assert sel.nunique() > 2, "continuous mode collapsed to a flag"
    assert np.allclose(sel.to_numpy(), scored["score"].to_numpy(), equal_nan=True)
    print(f"=== 5. continuous mode: {sel.nunique()} distinct weights in "
          f"[{sel.min():.2f}, {sel.max():.2f}] rather than a 0/1 flag ===")


# --------------------------------------------------------- 3. the roster is dynamic -----
def test_a_manager_off_the_roster_at_q_is_not_eligible_at_q():
    st = _state({"m0": 5, "m1": 10, "m2": 50})
    joined = pd.Timestamp("2021-03-31")

    def roster_at(p):
        return {"m0", "m1"} | ({"m2"} if pd.Timestamp(p) >= joined else set())

    ok = eligibility(st, roster_at, min_quarters=0, min_positions=0)
    ok.index = pd.MultiIndex.from_frame(st[["cik", "period"]])
    m2 = ok.xs("m2", level="cik")
    print(f"m2 eligible before {joined.date()}: {bool(m2[m2.index < joined].any())}, "
          f"after: {bool(m2[m2.index >= joined].all())}")
    assert not m2[m2.index < joined].any()
    assert m2[m2.index >= joined].all()
    print("=== 6. roster membership is read AT q, so a manager listed in 2021 does not "
          "count in 2020 ===")


# ------------------------------------------------------- 4. the floor changes ranks -----
def test_the_eligibility_floor_removes_a_manager_from_the_ranking():
    """m_new is the most concentrated manager but has only two quarters of history. It must
    not be selected AND must not occupy a rung: with it excluded from the ranking, the
    remaining managers' scores are the same as if it had never filed."""
    established = _state({"m0": 5, "m1": 10, "m2": 50})
    newcomer = _state({"m9": 2}, periods=_PERIODS[-2:])
    st = pd.concat([established, newcomer], ignore_index=True)

    ok = eligibility(st, _roster_all, min_quarters=4, min_positions=0)
    scored = manager_concentration_score(st, eligible=ok)
    alone = manager_concentration_score(established,
                                        eligible=eligibility(established, _roster_all,
                                                             min_quarters=4,
                                                             min_positions=0))
    assert "m9" not in scored.index.get_level_values("cik")
    shared = scored.index.intersection(alone.index)
    assert len(shared) > 0
    assert np.allclose(scored.loc[shared, "score"], alone.loc[shared, "score"])
    print(f"=== 7. the 2-quarter newcomer is excluded from the RANKING too: "
          f"{len(shared)} shared rows unchanged ===")


def test_the_index_position_floor_bites():
    st = _state({"m0": 5, "m1": 10}, n_index={"m0": 2, "m1": 9})
    ok = eligibility(st, _roster_all, min_quarters=0, min_positions=3)
    ok.index = pd.MultiIndex.from_frame(st[["cik", "period"]])
    print(f"m0 holds 2 universe names -> eligible {bool(ok.xs('m0', level='cik').any())}; "
          f"m1 holds 9 -> eligible {bool(ok.xs('m1', level='cik').all())}")
    assert not ok.xs("m0", level="cik").any()
    assert ok.xs("m1", level="cik").all()
    print("=== 8. a manager with 2 universe names cannot move a consensus basket and is "
          "floored out ===")


# ----------------------------------------------------------------- 5. diagnostics -------
def test_sticky_concentration_gives_a_stable_selected_set():
    st = _state({f"m{i}": 5 * (i + 1) for i in range(8)},
                lag_days={f"m{i}": 45 + i for i in range(8)})
    sel = elite_weight(manager_concentration_score(st), mode="top_k", k=3)
    diag = selection_diagnostics(sel, st)
    settled = diag[diag["n_public"] == 8]
    print(diag.head(12).to_string(index=False))
    print(f"once all 8 managers are public: n_selected "
          f"{settled['n_selected'].min()}-{settled['n_selected'].max()}, "
          f"max churn {settled['churn'].max()}")
    assert (settled["n_selected"] == 3).all()
    assert settled["churn"].max() == 0, "a permanent concentration ordering churned"
    print("=== 9. a fixed concentration ordering produces zero churn and a set of exactly "
          "k once every manager is public ===")


def test_empty_inputs_are_handled():
    empty = pd.DataFrame(columns=["cik", "period", "avail", "n_positions", "eff_n",
                                  "top10_weight"])
    scored = manager_concentration_score(empty)
    assert scored.empty and list(scored.columns) == ["score", "n_public"]
    assert elite_weight(scored).empty
    assert selection_diagnostics(pd.Series(dtype="float64"), empty).empty
    print("=== 10. an empty state yields an empty score, weight and diagnostic, not a "
          "traceback ===")


# ------------------------------------------------- 6. it reaches the panel end-to-end ---
_UNIVERSE = ["HOT", "COLD"]
_CUSIP_MAP = pd.DataFrame({"cusip": ["000000000", "000000001"],
                           "ticker": ["HOT", "COLD"]})
_PEERS = {t: [p for p in _UNIVERSE if p != t] for t in _UNIVERSE}
_INDEX = pd.date_range("2025-11-01", "2026-03-31", freq="B")
_ROSTER = {"0000000001": "Narrow", "0000000002": "Broad"}


def _two_manager_book() -> pd.DataFrame:
    """`Narrow` holds HOT only; `Broad` holds both. So COLD's holder base is `Broad` ALONE,
    which is what makes dropping `Broad` visible -- with both managers holding both names
    the holder SHARE is 1.0 either way and the test would prove nothing."""
    rows = []
    for period, filed in (("2025-09-30", "2025-11-14"), ("2025-12-31", "2026-02-14")):
        rows.append({"cik": "0000000001", "period": period, "filing_date": filed,
                     "cusip": "000000000", "position_type": "common",
                     "shares": 900, "value_usd": 9_000})
        rows += [{"cik": "0000000002", "period": period, "filing_date": filed,
                  "cusip": cusip, "position_type": "common",
                  "shares": 100, "value_usd": 1_000}
                 for cusip in ("000000000", "000000001")]
    return pd.DataFrame(rows)


def _panel(holdings, **kw):
    from src.data_aggregate.utils.institutionals.superinvestor_features import (
        build_superinvestor_feature_panel,
    )
    return build_superinvestor_feature_panel(
        holdings, _ROSTER, _PEERS, _INDEX, cusip_map=_CUSIP_MAP,
        universe=_UNIVERSE, **kw)


def _keys(holdings) -> pd.MultiIndex:
    from src.data_aggregate.utils.institutionals.superinvestor_features import (
        _prepare, attach_tickers, manager_quarter_state,
    )
    st = manager_quarter_state(_prepare(attach_tickers(holdings, _CUSIP_MAP, _UNIVERSE)))
    return pd.MultiIndex.from_frame(st[["cik", "period"]]), st


def test_the_selector_reaches_the_panel_and_zeroes_an_unselected_manager():
    """The whole point of `sel`: a manager the selector drops contributes NOTHING, and a
    callable selector gives the identical answer to the Series it returns."""
    holdings = _two_manager_book()
    keys, st = _keys(holdings)
    drop_broad = pd.Series(np.where(st["cik"] == "0000000001", 1.0, 0.0), index=keys)

    both = _panel(holdings)
    one = _panel(holdings, selection=drop_broad)
    via_call = _panel(holdings, selection=lambda _state: drop_broad)

    col = "f_ic_super_holders"

    def cold(panel) -> pd.Series:
        s = panel[panel["ticker"] == "COLD"][col]
        return s.dropna()

    b, o, c = cold(both), cold(one), cold(via_call)
    print(f"COLD's holder share -- both managers count: {len(b)} live dates, median "
          f"{b.median():.3f}; its only holder deselected: {len(o)} live dates")
    assert len(b) > 0 and b.gt(0).all(), "COLD should have a positive holder share"
    # ⚠ THE NAME LEAVES THE PANEL, it does not go to 0. `_aggregate`'s first-appearance
    # mask keys on the first date a COUNTED manager holds the name, so deselecting COLD's
    # sole holder makes it a name no elite manager has ever held -- a different fact from
    # "they sold out in 2019", and the panel drops the all-NaN ticker entirely.
    assert len(o) == 0, "a deselected sole holder should leave the name unobserved"
    assert len(c) == 0, "the callable selector disagreed with the Series"
    kept = one[one["ticker"] == "HOT"][col].dropna()
    assert len(kept) > 0, "HOT must still be built from the manager that was kept"
    print("=== 11. a deselected manager stops contributing entirely (its sole name goes "
          "back to unobserved), and a callable selector matches its Series ===")


def test_selection_score_is_emitted_only_when_sel_varies():
    """#27 is a constant under the flat default, and a constant is not a feature: it would
    still take a cube column, a fingerprint row and a SHAP slot."""
    holdings = _two_manager_book()
    keys, st = _keys(holdings)
    varied_sel = pd.Series(np.where(st["cik"] == "0000000001", 1.0, 0.25), index=keys)

    flat = _panel(holdings)
    varied = _panel(holdings, selection=varied_sel)
    col = "f_ic_super_selection_score"
    print(f"flat sel -> {col} present: {col in flat.columns}; "
          f"varying sel -> present: {col in varied.columns}")
    assert col not in flat.columns
    assert col in varied.columns
    got = (varied.set_index(["date", "ticker"])[col].dropna()
           .groupby(level="ticker").median().round(4))
    # HOT is held by both (mean sel (1.0 + 0.25) / 2), COLD by the 0.25 manager alone
    print(got.to_string())
    assert got.loc["HOT"] == 0.625 and got.loc["COLD"] == 0.25
    print("=== 12. #27 is emitted only when the selector varies, and it reports the mean "
          "`sel` across each name's OWN holders ===")


# ------------------------------------------------------ 7. the config actually wires up ---
def test_build_cube_yml_declares_the_selection_block_and_parses_it():
    """The live `configs/build_cube.yml` must carry the block the step reads, and `mode`
    must survive YAML.

    ⚠ THIS TEST EXISTS FOR ONE CHARACTER: the quotes around "off". PyYAML is YAML 1.1, where
    a bare `off` is the BOOLEAN False -- so an unquoted `mode: off` reaches the step as
    `False`, `str(False)` is `"False"`, and the selector would fall through to
    `elite_weight(mode="False")` and raise mid-build."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load("configs/build_cube.yml"),
                                 resolve=True)["build_cube"]
    sel = cfg["institutionals"]["superinvestor"]["selection"]
    print(f"superinvestor.selection = {sel}")
    print(f"  stale_quarters = {cfg['institutionals']['superinvestor']['stale_quarters']}")
    assert isinstance(sel["mode"], str), (
        f"`mode` parsed as {type(sel['mode']).__name__} ({sel['mode']!r}) -- quote it")
    assert sel["mode"] in ("off", "top_k", "continuous")
    assert sel["k"] >= 1 and sel["min_quarters"] >= 0 and sel["min_positions"] >= 0
    assert cfg["institutionals"]["superinvestor"]["stale_quarters"] >= 1
    print("=== 13. `configs/build_cube.yml` carries a parseable selection block, and "
          '`mode` is a string rather than YAML 1.1\'s boolean False ===')


def test_every_composite_member_is_a_feature_some_builder_emits():
    """A dead name in `composites` is skipped in silence, which is exactly how the cube
    degraded before: 46 field names and 39 features went stale with 68 tests green. This
    checks the `ic_super_` members against the module's own emission map."""
    from omegaconf import OmegaConf

    from src.data_aggregate.utils.institutionals.superinvestor_features import EMISSION

    cfg = OmegaConf.to_container(OmegaConf.load("configs/build_cube.yml"),
                                 resolve=True)["build_cube"]
    members = []
    for group in cfg["composites"].values():
        if isinstance(group, dict):
            for names in group.values():
                members += [str(n).lstrip("-") for n in names]
    super_members = sorted({m for m in members if m.startswith("f_ic_super_")})

    emitted = set()
    for name, cls in EMISSION.items():
        emitted.add(f"f_{name}")
        if cls.endswith("+xs"):
            emitted.add(f"f_{name}_xs")
    dead = [m for m in super_members if m not in emitted]
    print(f"`ic_super_` composite members: {len(super_members)} -> {super_members}")
    print(f"dead (named in a composite, emitted by nothing): {dead}")
    assert not dead, dead
    print("=== 14. every `ic_super_` composite member is a feature the builder emits ===")


def test_the_step_builds_a_selector_from_config_and_memoises_the_roster():
    """`StepCubeInstitutionals._superinvestor_selector` is the only place config becomes a
    weight, and until now nothing exercised it. Also pins the memo: `roster_as_of` reloads
    the whole roster table per call, so an un-cached lookup is one full read per period."""
    from omegaconf import OmegaConf

    from src.data_aggregate.transformers.step_cube_institutionals import (
        StepCubeInstitutionals,
    )

    reads = {"n": 0}

    class _FakeStore:
        def exists(self, _table):
            return True

    class _FakeContext:
        store = _FakeStore()

    step = StepCubeInstitutionals.__new__(StepCubeInstitutionals)
    step._context = _FakeContext()
    step._log = logging.getLogger(__name__)

    def _configure(mode, **kw):
        step._cfg = OmegaConf.create(
            {"institutionals": {"superinvestor": {
                "selection": {"mode": mode, "k": 3, "min_quarters": 0,
                              "min_positions": 0, **kw}}}})

    # ---- "off", and YAML 1.1's boolean spelling of it, both mean no selector ----
    for spelling in ("off", False, "OFF", "none"):
        _configure(spelling)
        assert step._superinvestor_selector() is None, spelling
    print(f'"off" spellings that correctly yield no selector: '
          f'{["off", False, "OFF", "none"]}')

    # ---- top_k and continuous both produce a working weight ----
    st = _state({f"m{i}": 5 * (i + 1) for i in range(6)})
    import src.data_aggregate.transformers.step_cube_institutionals as mod

    orig_as_of, orig_first = mod.roster_as_of, mod.first_snapshot_date
    try:
        def _counting_as_of(_context, _as_of=None):
            reads["n"] += 1
            return {f"m{i}" for i in range(6)}

        mod.roster_as_of = _counting_as_of
        mod.first_snapshot_date = lambda _context: pd.Timestamp("2019-01-01")

        got = {}
        for mode in ("top_k", "continuous"):
            _configure(mode)
            reads["n"] = 0
            sel = step._superinvestor_selector()(st)
            got[mode] = (sel.nunique(), int(reads["n"]))
        print(f"distinct weights / roster reads: {got}; distinct periods in the state: "
              f"{st['period'].nunique()}")
    finally:
        mod.roster_as_of, mod.first_snapshot_date = orig_as_of, orig_first

    assert got["top_k"][0] == 2, "top_k should be a 0/1 flag"
    assert got["continuous"][0] > 2, "continuous should keep a gradient"
    assert got["top_k"][1] <= st["period"].nunique(), "the roster lookup was not memoised"
    print("=== 15. the step turns config into a weight for both live modes, treats every "
          "spelling of `off` as no-selection, and reads the roster once per period ===")
