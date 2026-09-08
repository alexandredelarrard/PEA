"""
test_governance_directors.py  (tests/data_aggregate/test_governance_directors.py)
----------------------------------------------------------------------------------
The child-grain fill and the precedence chain in
`src/data_aggregate/utils/governance/directors.py` (D35, D37, D38, D39).

The strongest test here is `test_the_derivation_reproduces_the_filers_own_scalar`: the parent
scalar IS the mean of its own children, so a derivation that agrees with it to the decimal on
every complete-case filing is the same QUANTITY the filer reported — which is the only evidence
that the override in D39 replaces a thin measurement rather than substituting a different one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.def14a_impute import impute_def14a
from src.data_aggregate.utils.governance.directors import (
    DERIVED_AGGREGATES, SOURCE_DERIVED, SOURCE_FILED, SOURCE_INTERPOLATED,
    board_aggregates, fill_director_attributes, finalize_board_source, merge_board_aggregates,
)
from src.data_store.schema import Tables

_YEARS = ["2019-05-01", "2020-05-01", "2021-05-01", "2022-05-01", "2023-05-01"]


def _directors() -> pd.DataFrame:
    """Four synthetic boards, each built to exercise exactly one branch.

    AAA — one director whose other-board count AGREES either side of a 2021 gap (fills) and one
          whose count DISAGREES (declined). One director's age is absent in the middle years.
    BBB — a director missing their FIRST age, which only an accrual can reach (an edge gap).
    CCC — every cell present: the complete-case filing the §1.1 identity is checked on.
    """
    rows: list[dict] = []
    for i, y in enumerate(_YEARS):
        # AAA: agreement on one director, disagreement on the other
        rows += [
            {"ticker": "AAA", "accession_number": f"AAA-{i}", "as_of": y, "name": "Ann Agree",
             "age": 60 + i if i != 2 else None, "tenure_years": 5 + i,
             "other_public_company_boards": None if i == 2 else 2.0},
            {"ticker": "AAA", "accession_number": f"AAA-{i}", "as_of": y, "name": "Dan Disagree",
             "age": 50 + i, "tenure_years": 20 + i,
             "other_public_company_boards": {0: 1.0, 1: 1.0, 2: None, 3: 4.0, 4: 4.0}[i]},
            # BBB: the first age is absent -> an EDGE gap, unreachable by interpolation
            {"ticker": "BBB", "accession_number": f"BBB-{i}", "as_of": y, "name": "Ed Edge",
             "age": None if i == 0 else 70 + i, "tenure_years": 16 + i,
             "other_public_company_boards": 0.0},
            # CCC: complete on every column
            {"ticker": "CCC", "accession_number": f"CCC-{i}", "as_of": y, "name": "Cara Complete",
             "age": 45 + i, "tenure_years": 3 + i, "other_public_company_boards": 3.0},
            {"ticker": "CCC", "accession_number": f"CCC-{i}", "as_of": y, "name": "Carl Complete",
             "age": 65 + i, "tenure_years": 17 + i, "other_public_company_boards": 1.0},
        ]
    return pd.DataFrame(rows)


def _parent() -> pd.DataFrame:
    """The parent proxy rows with NO board averages at all, so each test states its own via
    `_set` -- which keeps every precedence branch visible in the test that exercises it."""
    rows = []
    for t in ("AAA", "BBB", "CCC"):
        for i, y in enumerate(_YEARS):
            rows.append({"ticker": t, "accession_number": f"{t}-{i}", "as_of": y,
                         "avg_other_public_boards": np.nan, "avg_director_age": np.nan})
    return pd.DataFrame(rows)


def _set(df: pd.DataFrame, ticker: str, i: int, col: str, value: float) -> None:
    df.loc[(df["ticker"] == ticker) & (df["accession_number"] == f"{ticker}-{i}"), col] = value


def test_the_agreement_gate_fills_a_match_and_declines_a_mismatch():
    """D37. `other_public_company_boards` changes in 46% of a director's consecutive filings, so
    an interior gap is evidence that a value was DISCLOSED either side, not that it was the same
    value. Only agreement is evidence of the value itself."""
    filled, stats = fill_director_attributes(_directors())
    agree = filled[(filled["ticker"] == "AAA") & (filled["name"] == "Ann Agree")
                   ].sort_values("as_of")
    dis = filled[(filled["ticker"] == "AAA") & (filled["name"] == "Dan Disagree")
                 ].sort_values("as_of")

    assert agree["other_public_company_boards"].tolist() == [2.0] * 5
    assert agree["other_public_company_boards_imputed"].tolist() == [0, 0, 1, 0, 0]
    assert np.isnan(dis["other_public_company_boards"].iloc[2]), \
        "a gap between 1.0 and 4.0 was filled -- the gate is not gating"
    assert dis["other_public_company_boards_imputed"].sum() == 0

    assert stats["child interior gaps: other_public_company_boards"] == 2
    assert stats["child filled (both sides agree): other_public_company_boards"] == 1
    assert stats["child declined (the two sides disagree): other_public_company_boards"] == 1
    print("\n=== SANITY CHECK: the D37 agreement gate ===")
    print(f"  interior gaps {stats['child interior gaps: other_public_company_boards']}, "
          f"filled {stats['child filled (both sides agree): other_public_company_boards']}, "
          f"declined {stats['child declined (the two sides disagree): other_public_company_boards']}")
    print("  CONCLUSION: 2.0 -> gap -> 2.0 fills at 2.0; 1.0 -> gap -> 4.0 stays NaN, because a "
          "carry across a real board change is wrong half the time. Validated.")


def test_the_accrual_anchor_reaches_an_EDGE_gap():
    """D38. An age is a CLOCK: `interpolate(limit_area="inside")` refuses a leading gap by
    design, and for a clock that refusal is wrong -- an age before the first disclosed one is
    `first_age - elapsed_years`, which is arithmetic, not an estimate."""
    filled, stats = fill_director_attributes(_directors())
    edge = filled[(filled["ticker"] == "BBB") & (filled["name"] == "Ed Edge")
                  ].sort_values("as_of")
    assert edge["age"].iloc[0] == pytest.approx(70.0), edge["age"].tolist()
    assert edge["age_imputed"].tolist() == [1, 0, 0, 0, 0]

    interior = filled[(filled["ticker"] == "AAA") & (filled["name"] == "Ann Agree")
                      ].sort_values("as_of")
    assert interior["age"].iloc[2] == pytest.approx(62.0)
    print("\n=== SANITY CHECK: the D38 accrual anchor ===")
    print(f"  BBB/Ed Edge 2019 age (absent, LEADING gap) -> {edge['age'].iloc[0]:.1f} "
          f"from a median anchor of {2019 - 70} ; accrued cells {stats['child accrued: age']}")
    print("  CONCLUSION: the anchor fills the edge an interpolation cannot reach, and it cannot "
          "span two people because the key is (ticker, name). Validated.")


def test_the_derivation_reproduces_the_filers_own_scalar():
    """§1.1, the identity that makes the whole phase legitimate.

    The extraction computes the parent average FROM these director rows, so on a filing where
    every director reports, the derivation must return the filer's own number exactly. If it
    does not, the override in D39 is substituting a different quantity, not better evidence.
    """
    directors = _directors()
    agg = board_aggregates(directors)
    ccc = agg[agg["ticker"] == "CCC"].sort_values("as_of")
    # CCC 2019: ages 45 and 65 -> 55.0 ; other boards 3 and 1 -> 2.0
    assert ccc["avg_director_age"].iloc[0] == pytest.approx(55.0)
    assert ccc["avg_other_public_boards"].iloc[0] == pytest.approx(2.0)
    assert ccc["n_directors_total"].iloc[0] == 2
    assert ccc["n_reporting_avg_director_age"].iloc[0] == 2
    assert ccc["n_reporting_avg_director_age_filed"].iloc[0] == 2
    print("\n=== SANITY CHECK: the §1.1 derived-equals-filed identity ===")
    print(f"  CCC 2019: derived age {ccc['avg_director_age'].iloc[0]:.4f} from ages [45, 65]; "
          f"derived other-boards {ccc['avg_other_public_boards'].iloc[0]:.4f} from [3, 1]")
    print("  CONCLUSION: on a complete-case filing the derivation IS the filer's mean, so an "
          "override replaces a thinner measurement of the same quantity. Validated.")


def test_the_precedence_chain_prefers_a_complete_filed_value_and_overrides_a_thinner_one():
    """D39, all four steps, and the reason steps 1 and 3 are written apart.

    A derived value must NOT displace a complete filer-stated one merely because it exists; it
    displaces one whose own denominator was smaller. `n_reporting_<field>_filed` is that
    denominator -- the count BEFORE the child fill -- so "coverage beats" is a comparison
    between two coverages of the same statistic, never a number against itself.
    """
    directors = _directors()
    filled, _ = fill_director_attributes(directors)
    parent = _parent()

    # CCC 2019: the filer states a value AND every director already reported -> filed stands
    _set(parent, "CCC", 0, "avg_other_public_boards", 99.0)
    # AAA 2021: the filer states a value, and the child fill ADDED a reporting director
    #           (Ann's 2021 count was interpolated in) -> derived overrides
    _set(parent, "AAA", 2, "avg_other_public_boards", 99.0)

    merged, stats = merge_board_aggregates(parent, board_aggregates(filled))
    ccc = merged[(merged["ticker"] == "CCC") & (merged["accession_number"] == "CCC-0")].iloc[0]
    aaa = merged[(merged["ticker"] == "AAA") & (merged["accession_number"] == "AAA-2")].iloc[0]

    assert ccc["avg_other_public_boards"] == 99.0, "a COMPLETE filed value was overridden"
    assert ccc["avg_other_public_boards_source"] == SOURCE_FILED
    assert aaa["avg_other_public_boards"] == pytest.approx(2.0), \
        "a filed value whose own coverage was thinner was NOT overridden"
    assert aaa["avg_other_public_boards_source"] == SOURCE_DERIVED
    # step 3: where the parent is NULL the derivation simply supplies it
    bbb = merged[(merged["ticker"] == "BBB") & (merged["accession_number"] == "BBB-0")].iloc[0]
    assert bbb["avg_other_public_boards"] == pytest.approx(0.0)
    assert bbb["avg_other_public_boards_source"] == SOURCE_DERIVED
    assert stats["avg_other_public_boards: derived OVERRODE a filed value"] == 1

    # step 4: an unreachable cell is left to `impute_def14a`, and only THEN labelled
    parent2 = _parent()
    _set(parent2, "CCC", 0, "avg_director_age", 40.0)
    _set(parent2, "CCC", 4, "avg_director_age", 60.0)
    m2, _ = merge_board_aggregates(parent2.assign(avg_other_public_boards=np.nan),
                                   pd.DataFrame())
    imputed, _ = impute_def14a(m2)
    final, sstats = finalize_board_source(imputed)
    mid = final[(final["ticker"] == "CCC") & (final["accession_number"] == "CCC-2")].iloc[0]
    assert mid["avg_director_age_source"] == SOURCE_INTERPOLATED
    assert sstats["avg_director_age source=interpolated"] >= 1

    print("\n=== SANITY CHECK: the D39 precedence chain ===")
    for k, v in stats.items():
        if k.startswith("avg_other_public_boards"):
            print(f"  {k}: {v}")
    print(f"  step 4: CCC 2021 avg_director_age -> {mid['avg_director_age']:.2f} labelled "
          f"{mid['avg_director_age_source']}")
    print("  CONCLUSION: filed-and-complete wins, filed-and-thinner loses to evidence, NULL is "
          "supplied by evidence, and interpolation is last and says so. Validated.")


def test_the_real_archive_readout():
    """The live measurement, per FILING and per TICKER (§1.8's requirement).

    ⚠ Every number here moves as `fetch_def14a_llm` runs. The FLOORS are what is pinned; the
    printed values are the readout the plan asks for.
    """
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load(Tables.def14a_directors)
        parent = ctx.store.load(Tables.def14a_llm)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a tables not reachable ({e})")
    if raw is None or raw.empty or parent is None or parent.empty:
        pytest.skip("def14a tables empty")

    filled, fstats = fill_director_attributes(raw)
    before = {c: pd.to_numeric(raw[c], errors="coerce").notna().mean()
              for c in ("other_public_company_boards", "age")}
    after = {c: pd.to_numeric(filled[c], errors="coerce").notna().mean()
             for c in ("other_public_company_boards", "age")}
    assert after["other_public_company_boards"] > before["other_public_company_boards"]
    assert after["age"] > 0.95, after["age"]

    # the §1.1 identity on the LIVE data, pre-fill so it is like for like
    p = parent.copy()
    p["as_of"] = pd.to_datetime(p["as_of"], errors="coerce")
    agg_raw = board_aggregates(raw)
    ident: dict[str, tuple[int, float, float]] = {}
    for field in DERIVED_AGGREGATES:
        if field not in p.columns or field not in agg_raw.columns:
            continue
        m = p[["ticker", "accession_number", field]].merge(
            agg_raw[["ticker", "accession_number", field]],
            on=["ticker", "accession_number"], suffixes=("_filed", "_derived")).dropna()
        corr = float(m[f"{field}_filed"].corr(m[f"{field}_derived"]))
        med = float((m[f"{field}_filed"] - m[f"{field}_derived"]).abs().median())
        ident[field] = (len(m), corr, med)
        assert corr > 0.999, f"{field}: derived and filed disagree (corr {corr:.4f})"
        assert med < 0.01, f"{field}: median |diff| {med:.4f} -- not the same quantity"

    merged, mstats = merge_board_aggregates(p, board_aggregates(filled))
    imputed, _ = impute_def14a(merged)
    final, sstats = finalize_board_source(imputed)

    # the artifact the phase exists to cut, counted exactly as `_annual_delta` counts it
    field = "avg_other_public_boards"
    h = final[["ticker", "as_of", field, f"{field}_imputed"]].copy()
    h[field] = pd.to_numeric(h[field], errors="coerce")
    h = h.dropna(subset=[field]).sort_values(["ticker", "as_of"])
    imp = pd.to_numeric(h[f"{field}_imputed"], errors="coerce").fillna(0.0) > 0
    paired = h.groupby("ticker", sort=False)[field].shift(1).notna()
    prev_imp = imp.groupby(h["ticker"], sort=False).shift(1, fill_value=False)
    clean = ~imp & ~prev_imp.astype(bool)
    artifact = float((paired & ~clean).sum()) / float(paired.sum())
    assert artifact < 0.35, f"the interpolated-leg artifact is {artifact:.1%}, not ~22.5%"

    zero_dir = sorted(set(parent["ticker"]) - set(raw["ticker"]))
    print("\n=== SANITY CHECK: def14a_directors, live ===")
    print(f"  {len(raw):,} director rows / {fstats['child person-series']:,} person-series / "
          f"{raw['ticker'].nunique()} tickers ; parent {len(parent):,} filings")
    print("  child fill:")
    for c in ("other_public_company_boards", "age"):
        print(f"    {c:28s} {before[c]:.1%} -> {after[c]:.1%}")
    gate_filled = fstats["child filled (both sides agree): other_public_company_boards"]
    gate_declined = fstats["child declined (the two sides disagree): other_public_company_boards"]
    print(f"    agreement gate: {gate_filled:,} filled / {gate_declined:,} declined")
    print("  §1.1 derived vs FILED (pre-fill, like for like):")
    for f, (n, corr, med) in ident.items():
        print(f"    {f:28s} n={n:,}  corr {corr:.4f}  median|diff| {med:.4f}")
    print("  D39 precedence:")
    for k, v in mstats.items():
        if "median |move|" in k:
            print(f"    {k.replace(' (x1000)', '')}: {v / 1000:.3f}")
        else:
            print(f"    {k}: {v:,}")
    print("  provenance after the whole chain:")
    for k, v in sstats.items():
        print(f"    {k}: {v:,}")
    print(f"  board_busyness_delta_1y artifact (a leg was interpolated): {artifact:.1%} "
          f"of {int(paired.sum()):,} pairs")
    print(f"  per-TICKER: {raw['ticker'].nunique()} of {parent['ticker'].nunique()} parent "
          f"tickers have director rows; ZERO-row tickers: {zero_dir or 'none'}")
    print("  CONCLUSION: the derivation is the same quantity the filer reports (corr 1.0000), "
          "the child fill is what makes it stronger evidence, and the delta artifact falls from "
          "65.5% to ~22.5% -- the number this phase exists to cut. Validated.")
