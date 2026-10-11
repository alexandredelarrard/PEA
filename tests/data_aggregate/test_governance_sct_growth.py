"""
Same-filing CEO pay growth from the Summary Compensation Table (clean.py, REQ-009).

Known-truth fixture: the cleaning stage matches the named CEO in the filing's own SCT rows and
writes `log(total_FY / total_FY-1)`. Both pay-growth features prefer it, the CEO-change guard
still nulls a succession, an unmatched CEO falls back to the cross-filing pair, a two-year FY
gap is refused, and a NULL parent total is filled from the same-FY SCT row (kind `identity`).

The historical backfill (accepted lookahead) fills a component from a later SCT's exact value
(kind `backfill`), else bridges at most 2 missing proxies of one CEO stint (kind `interp`), never
the latest row; the total is then recomputed from its six components. Every fill is dated at its
own row with the later evidence date beside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.governance.clean import PAY_BACKFILL_STAGE, GovernanceInputs, clean_governance_inputs
from src.data_aggregate.utils.governance.def14a_impute import (
    CEO_COMP,
    FILL_BACKFILL,
    FILL_IDENTITY,
    FILL_INTERP,
    evidence_date_column,
    fill_kind_column,
)
from src.data_aggregate.utils.governance.panel import _ceo_pay_growth
from src.data_aggregate.utils.governance.pay_features import SCT_GROWTH, _comp_history
from src.data_aggregate.utils.governance.staleness import source_date_column

IDX = pd.bdate_range("2021-01-04", "2023-12-29")


def _bundle() -> GovernanceInputs:
    """Five tickers, one rule each; every first filing is FY2020, every second FY2021.

    AAA  same CEO; SCT FY2021/FY2020 = 12/10 while the cross-filing pair is 15/10
    BBB  CEO change; the new CEO's SCT holds both years -> still nulled by the guard
    CCC  the SCT names someone else -> falls back to the cross-filing pair (9/6)
    DDD  the CEO's SCT rows are FY2021 and FY2019 only -> refused, falls back (8/4)
    EEE  NULL parent total on the second filing -> filled from the FY2021 SCT row
    FFF  the FY2020 SCT total is a sum of components, not filer-stated -> falls back (6/3)
    """
    parent = [
        ("AAA", "2021-04-01", 10e6, "Timothy D. Cook"),
        ("AAA", "2022-04-01", 15e6, "Tim Cook"),
        ("BBB", "2021-04-01", 8e6, "Alice Smith"),
        ("BBB", "2022-04-01", 20e6, "Bob Jones"),
        ("CCC", "2021-04-01", 6e6, "Carol White"),
        ("CCC", "2022-04-01", 9e6, "Carol White"),
        ("DDD", "2021-04-01", 4e6, "Dan Brown"),
        ("DDD", "2022-04-01", 8e6, "Dan Brown"),
        ("EEE", "2021-04-01", 5e6, "Eve Green"),
        ("EEE", "2022-04-01", np.nan, "Eve Green"),
        ("FFF", "2021-04-01", 3e6, "Fay Hill"),
        ("FFF", "2022-04-01", 6e6, "Fay Hill"),
    ]
    def14a = pd.DataFrame(
        [{"ticker": t, "accession_number": f"{t}-{d[:4]}", "as_of": d, "ceo_total_comp": c, "ceo_name_proxy": n} for t, d, c, n in parent]
    )
    child = [
        ("AAA", "Timothy Cook", 2021, 12e6),
        ("AAA", "Timothy Cook", 2020, 10e6),
        ("BBB", "Bob Jones", 2021, 20e6),
        ("BBB", "Bob Jones", 2020, 4e6),
        ("CCC", "Zed Other", 2021, 9e6),
        ("CCC", "Zed Other", 2020, 7e6),
        ("DDD", "Dan Brown", 2021, 8e6),
        ("DDD", "Dan Brown", 2019, 5e6),
        ("EEE", "Eve Green", 2021, 7.5e6),
        ("EEE", "Eve Green", 2020, 5e6),
        ("FFF", "Fay Hill", 2021, 6e6),
        ("FFF", "Fay Hill", 2020, np.nan),
    ]
    exec_comp = pd.DataFrame(
        [
            {"ticker": t, "accession_number": f"{t}-2022", "as_of": "2022-04-01", "name": n, "fiscal_year": fy, "total": tot, "salary": 1e6}
            for t, n, fy, tot in child
        ]
    )
    return GovernanceInputs(def14a=def14a, exec_comp=exec_comp)


def _growth_by_ticker(df: pd.DataFrame, col: str) -> pd.Series:
    second = df.loc[pd.to_datetime(df["as_of"]) == pd.Timestamp("2022-04-01")]
    return pd.Series(second[col].to_numpy(), index=second["ticker"].to_numpy())


def test_same_filing_growth_is_preferred_and_guarded() -> None:
    cleaned, stats = clean_governance_inputs(_bundle(), None)
    d = cleaned.def14a
    assert d is not None
    sct = _growth_by_ticker(d, SCT_GROWTH)
    assert np.isclose(sct["AAA"], np.log(1.2)) and np.isclose(sct["BBB"], np.log(5.0)) and np.isclose(sct["EEE"], np.log(1.5))
    assert np.isnan(sct["CCC"]) and np.isnan(sct["DDD"]) and np.isnan(sct["FFF"])

    hist = _comp_history(d, {})
    assert hist is not None
    g = _growth_by_ticker(hist, "ceo_comp_growth_1y")
    assert np.isclose(g["AAA"], np.log(1.2))  # same-filing pair, not log(1.5)
    assert np.isnan(g["BBB"])  # succession across filings: still nulled
    assert np.isclose(g["CCC"], np.log(1.5))  # unmatched CEO: cross-filing pair
    assert np.isclose(g["DDD"], np.log(2.0))  # FY gap of 2 refused: cross-filing pair
    assert np.isclose(g["EEE"], np.log(1.5))
    assert np.isclose(g["FFF"], np.log(2.0))  # component-sum SCT total refused: cross-filing pair

    daily = _ceo_pay_growth(d, IDX).loc["2022-06-01"]
    assert np.isclose(daily["AAA"], 0.2) and np.isclose(daily["CCC"], 0.5) and np.isclose(daily["DDD"], 1.0)
    assert np.isnan(daily["BBB"])

    second = d.loc[d["accession_number"] == "EEE-2022"].iloc[0]
    assert second["ceo_total_comp"] == 7.5e6 and second[fill_kind_column("ceo_total_comp")] == FILL_IDENTITY
    assert pd.Timestamp(second[source_date_column("ceo_total_comp")]) == pd.Timestamp("2022-04-01")
    assert stats["sct_growth|growth from the filing's own SCT (FY vs FY-1, named CEO)"] == 3
    assert stats["sct_growth|ceo_total_comp filled from the filing's own SCT row"] == 1
    assert stats["sct_growth|growth refused: an SCT total is a sum of components, not filer-stated"] == 1
    print(
        "SANITY: same-filing growth AAA=log(1.2) preferred over log(1.5); BBB succession nulled; "
        "CCC unmatched, DDD FY-gap and FFF component-sum fall back; EEE total filled 7.5M (identity) -> PASS"
    )


def _history_bundle() -> GovernanceInputs:
    """One ticker, CEO Ann Lee 2016-2020 then Bob Ray 2021-2022 (pay in $M, FY = proxy year - 1).

    2016 bonus      NULL; the 2017 SCT discloses Ann's FY2015 bonus 0.25 -> backfill
    2016 total      NULL; all six legs known after the backfill -> identity 4.15
    2018 stock      NULL between 1.0 (2017) and 4.0 (2019), same stint -> interp 2.0
    2017-19 option  three missing proxies -> not bridged
    2021 NEI        first row of Bob's stint (Ann's 0.3 is another stint) -> not bridged
    2022 other      latest row, nothing later -> not filled
    """
    cols = ["year", "name", *CEO_COMP, "ceo_total_comp"]
    nan = np.nan
    parent = pd.DataFrame(
        [
            (2016, "Ann Lee", 1.0, nan, 2.0, 0.5, 0.3, 0.1, nan),
            (2017, "Ann Lee", 1.0, 0.2, 1.0, nan, 0.3, 0.1, nan),
            (2018, "Ann Lee", 1.0, 0.2, nan, nan, 0.3, 0.1, nan),
            (2019, "Ann Lee", 1.0, 0.2, 4.0, nan, 0.3, 0.1, nan),
            (2020, "Ann Lee", 1.0, 0.2, 4.0, 0.8, 0.3, 0.1, 6.4),
            (2021, "Bob Ray", 2.0, 0.0, 5.0, 1.0, nan, 0.2, nan),
            (2022, "Bob Ray", 2.0, 0.0, 5.0, 1.0, 0.9, nan, nan),
        ],
        columns=cols,
    )
    parent[[*CEO_COMP, "ceo_total_comp"]] *= 1e6
    def14a = parent.assign(
        ticker="AAA",
        accession_number=[f"A-{y}" for y in parent["year"]],
        as_of=[f"{y}-04-01" for y in parent["year"]],
        fiscal_year_extract=parent["year"] - 1,
        ceo_name_proxy=parent["name"],
    ).drop(columns=["year", "name"])
    # the 2017 filing: someone else for FY2016 (no same-filing CEO match), Ann's FY2015 bonus only
    exec_comp = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "accession_number": "A-2017",
                "as_of": "2017-04-01",
                "name": "Zed Other",
                "fiscal_year": 2016,
                "salary": 5e5,
                "total": 5e5,
            },
            {
                "ticker": "AAA",
                "accession_number": "A-2017",
                "as_of": "2017-04-01",
                "name": "Ann Lee",
                "fiscal_year": 2015,
                "bonus": 2.5e5,
                "total": nan,
            },
        ]
    )
    return GovernanceInputs(def14a=def14a, exec_comp=exec_comp)


def test_historical_backfill_and_bridge_rules() -> None:
    cleaned, stats = clean_governance_inputs(_history_bundle(), None)
    d = cleaned.def14a
    assert d is not None
    d = d.assign(_y=pd.to_datetime(d["as_of"]).dt.year).set_index("_y")

    def cell(year: int, col: str) -> tuple[float, object, pd.Timestamp, pd.Timestamp]:
        row = d.loc[year]
        return row[col], row[fill_kind_column(col)], pd.Timestamp(row[source_date_column(col)]), pd.Timestamp(row[evidence_date_column(col)])

    value, kind, source, evidence = cell(2016, "ceo_bonus")
    assert value == 2.5e5 and kind == FILL_BACKFILL  # exact later-SCT value
    assert source == pd.Timestamp("2016-04-01") and evidence == pd.Timestamp("2017-04-01")
    value, kind, source, evidence = cell(2016, "ceo_total_comp")
    assert np.isclose(value, 4.15e6) and kind == FILL_IDENTITY  # recomputed after the components
    assert source == pd.Timestamp("2016-04-01") and evidence == pd.Timestamp("2017-04-01")
    value, kind, source, evidence = cell(2018, "ceo_stock_awards")
    assert np.isclose(value, 2.0e6) and kind == FILL_INTERP  # log-linear midpoint of 1 and 4
    assert source == pd.Timestamp("2018-04-01") and evidence == pd.Timestamp("2019-04-01")
    assert d.loc[[2017, 2018, 2019], "ceo_option_awards"].isna().all()  # gap of 3 not bridged
    assert np.isnan(d.loc[2021, "ceo_non_equity_incentive"])  # no bridge across a stint
    assert np.isnan(d.loc[2022, "ceo_all_other_comp"])  # latest row: past only
    assert d.loc[2020, fill_kind_column("ceo_total_comp")] == "filed"
    assert stats[f"{PAY_BACKFILL_STAGE}|backfill (later SCT, same CEO and FY): ceo_bonus"] == 1
    assert stats[f"{PAY_BACKFILL_STAGE}|interp (same stint, <= 2 missing proxies): ceo_stock_awards"] == 1
    assert not any("ceo_option_awards" in k or "ceo_all_other_comp" in k for k in stats if k.startswith(PAY_BACKFILL_STAGE))
    print(
        "SANITY: 2016 bonus backfilled 0.25M (evidence 2017) and total recomputed 4.15M; 2018 stock bridged 2.0M; "
        "3-proxy gap, stint boundary and latest row left NULL -> PASS"
    )


def _relatives_bundle() -> GovernanceInputs:
    """Relatives who share the CEO's `person_key` in one filing (pay in $M, FY = proxy year).

    STZ  CEO Robert Sands; Richard Sands (higher pay) is listed first in the same SCT
    SMCI CEO Charles Liang; "Chiu-Chu (Sara) Liu Liang" keys `liang|c` too
    LLL  CEO James S. Tisch; the SCT prints initials only, "J.M. Tisch" and "J.S. Tisch"
    RRR  CEO "Rob Rex"; "Robert Rex" and "Richard Rex" both have the initial r -> refused
    The 2008 rows have no salary; the 2009 SCT discloses both relatives' FY2008 salary.
    """
    ceo = {"STZ": "Robert Sands", "SMCI": "Charles Liang", "LLL": "James S. Tisch", "RRR": "Rob Rex"}
    relatives = {
        "STZ": [("Richard Sands", 5.0, 3.5, 0.9), ("Robert Sands", 4.0, 3.0, 0.8)],
        "SMCI": [("Chiu-Chu (Sara) Liu Liang", 2.0, 1.0, 0.4), ("Charles Liang", 3.0, 2.4, 0.5)],
        "LLL": [("J.M. Tisch", 9.0, 3.0, 0.9), ("J.S. Tisch", 6.0, 5.0, 0.7)],
        "RRR": [("Robert Rex", 2.0, 1.0, 0.3), ("Richard Rex", 3.0, 2.0, 0.6)],
    }
    parent = [
        {"ticker": t, "accession_number": f"{t}-{y}", "as_of": f"{y}-04-01", "fiscal_year_extract": y, "ceo_name_proxy": n, "ceo_total_comp": np.nan}
        for t, n in ceo.items()
        for y in (2008, 2009)
    ]
    child = [
        {
            "ticker": t,
            "accession_number": f"{t}-2009",
            "as_of": "2009-04-01",
            "name": name,
            "fiscal_year": fy,
            "total": tot * 1e6,
            "salary": sal * 1e6 if fy == 2008 else 1e6,
        }
        for t, rows in relatives.items()
        for name, t1, t0, sal in rows
        for fy, tot in ((2009, t1), (2008, t0))
    ]
    return GovernanceInputs(def14a=pd.DataFrame(parent).assign(ceo_salary=np.nan), exec_comp=pd.DataFrame(child))


def test_relatives_sharing_the_key_are_told_apart_by_first_name() -> None:
    cleaned, stats = clean_governance_inputs(_relatives_bundle(), None)
    d = cleaned.def14a
    assert d is not None
    d = d.assign(_y=pd.to_datetime(d["as_of"]).dt.year).set_index(["ticker", "_y"])
    growth, total, salary = d[SCT_GROWTH], d["ceo_total_comp"], d["ceo_salary"]
    assert np.isclose(growth[("STZ", 2009)], np.log(4.0 / 3.0)) and total[("STZ", 2009)] == 4.0e6, "Robert's own rows, not Richard's"
    assert np.isclose(growth[("SMCI", 2009)], np.log(3.0 / 2.4)) and total[("SMCI", 2009)] == 3.0e6
    assert np.isclose(growth[("LLL", 2009)], np.log(6.0 / 5.0)), "the initials J.S. single out James S."
    assert np.isnan(growth[("RRR", 2009)]) and np.isnan(total[("RRR", 2009)]), "two relatives fit 'Rob': refused"
    # the later-SCT backfill of the 2008 salary reads the named CEO's row too
    assert salary[("STZ", 2008)] == 0.8e6 and salary[("SMCI", 2008)] == 0.5e6 and salary[("LLL", 2008)] == 0.7e6
    assert np.isnan(salary[("RRR", 2008)])
    assert stats["sct_growth|CEO ambiguous in the filing's SCT (relatives share the key)"] == 1
    print(
        f"SANITY: STZ growth {growth[('STZ', 2009)]:.3f} = log(4/3) (Robert), SMCI {growth[('SMCI', 2009)]:.3f} (Charles), "
        f"LLL {growth[('LLL', 2009)]:.3f} (J.S.), RRR refused; 2008 salaries backfilled from the named CEO's rows -> PASS"
    )


def test_bridged_totals_feed_no_cross_filing_growth() -> None:
    """A bridged (`interp`) total is the fill's slope: neither growth that touches it is kept, and
    `comp_imputed` marks it. The 2016 / 2019 pair around it is unaffected (not ~1y apart)."""
    rows = [(2016, 10e6), (2017, np.nan), (2018, 40e6), (2019, 44e6)]
    def14a = pd.DataFrame(
        [
            {"ticker": "BRG", "accession_number": f"B-{y}", "as_of": f"{y}-04-01", "ceo_name_proxy": "Bea Ridge", "ceo_total_comp": tot}
            for y, tot in rows
        ]
    )
    other = pd.DataFrame(
        [{"ticker": "BRG", "accession_number": "B-2016", "as_of": "2016-04-01", "name": "Zed Other", "fiscal_year": 2015, "total": 1e6}]
    )
    cleaned, _ = clean_governance_inputs(GovernanceInputs(def14a=def14a, exec_comp=other), None)
    d = cleaned.def14a
    assert d is not None
    d = d.assign(_y=pd.to_datetime(d["as_of"]).dt.year).set_index("_y")
    assert np.isclose(d.loc[2017, "ceo_total_comp"], 20e6) and d.loc[2017, fill_kind_column("ceo_total_comp")] == FILL_INTERP
    assert d.loc[2017, "comp_imputed"] == 1.0 and d.loc[[2016, 2018, 2019], "comp_imputed"].eq(0.0).all()
    hist = _comp_history(d.reset_index(drop=True), {})
    assert hist is not None
    g = hist.assign(_y=pd.to_datetime(hist["as_of"]).dt.year).set_index("_y")["ceo_comp_growth_1y"]
    assert g[[2017, 2018]].isna().all(), f"growth touching the bridged 2017 total must be NaN: {g.tolist()}"
    assert np.isclose(g[2019], np.log(1.1))
    daily = _ceo_pay_growth(d.reset_index(drop=True), pd.bdate_range("2016-01-04", "2019-12-31"))["BRG"]
    assert daily.loc["2017-06-01":"2019-03-29"].isna().all() and np.isclose(daily.loc["2019-06-03"], 0.1)
    print(
        f"SANITY: bridged 2017 total {d.loc[2017, 'ceo_total_comp'] / 1e6:.0f}M flagged comp_imputed; growth 2017/2018 {g[[2017, 2018]].tolist()}, 2019 {g[2019]:.3f} -> PASS"
    )
