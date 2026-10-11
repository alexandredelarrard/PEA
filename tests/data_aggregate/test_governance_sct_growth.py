"""
Same-filing CEO pay growth from the Summary Compensation Table (clean.py, REQ-009).

Known-truth fixture: the cleaning stage matches the named CEO in the filing's own SCT rows and
writes `log(total_FY / total_FY-1)`. Both pay-growth features prefer it, the CEO-change guard
still nulls a succession, an unmatched CEO falls back to the cross-filing pair, a two-year FY
gap is refused, and a NULL parent total is filled from the same-FY SCT row (kind `identity`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.governance.clean import GovernanceInputs, clean_governance_inputs
from src.data_aggregate.utils.governance.def14a_impute import FILL_IDENTITY, fill_kind_column
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
