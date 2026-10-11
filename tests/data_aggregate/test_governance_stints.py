"""
CEO stints and stint-constant completion (REQ-003/004).

  * `names.ceo_stint_ids`: a stint breaks on a change of CEO identity or of generational suffix;
    an unknown name belongs to no stint and breaks none;
  * `impute_def14a`: `ceo_since_year` = the largest disclosed value <= the stint's first proxy
    year (no value when none qualifies), `ceo_is_founder`
    = the stint mode, `ceo_is_board_chair` carried only within a stint; the filed value stays in
    `<col>_raw`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.governance.def14a_impute import (
    FILL_PERSON,
    evidence_date_column,
    fill_kind_column,
    impute_def14a,
)
from src.data_aggregate.utils.governance.names import ceo_stint_ids
from src.data_aggregate.utils.governance.staleness import source_date_column


def _rows(ticker: str, spec: list[tuple]) -> list[dict]:
    """`spec` items: (year, ceo name, since_year, founder, chair); proxies dated April 1st."""
    out = []
    for year, name, since, founder, chair in spec:
        out.append(
            {
                "ticker": ticker,
                "accession_number": f"{ticker}-{year}",
                "as_of": f"{year}-04-01",
                "ceo_name_proxy": name,
                "ceo_since_year": since,
                "ceo_is_founder": founder,
                "ceo_is_board_chair": chair,
            }
        )
    return out


NA = np.nan
_FIXTURE = pd.DataFrame(
    _rows(
        "WRB",
        [
            (2016, "William R. Berkley", 1967.0, 1.0, 1.0),
            (2017, "William R. Berkley", 1967.0, 1.0, 1.0),
            (2018, "William R. Berkley", NA, 1.0, 1.0),
            (2019, "W. Robert Berkley, Jr.", NA, NA, NA),
            (2020, "W. Robert Berkley, Jr.", 2019.0, 0.0, 0.0),
        ],
    )
    + _rows(
        "SBUX",
        [
            (1998, "Howard Schultz", 1987.0, 1.0, 1.0),
            (2000, "Howard Schultz", 1987.0, 1.0, 1.0),
            (2001, "Orin C. Smith", 2000.0, 0.0, 0.0),
            (2008, "Howard Schultz", 1987.0, 1.0, 1.0),
            (2009, "Howard D. Schultz", 2008.0, NA, 1.0),
            (2018, "Kevin Johnson", 2017.0, 0.0, 0.0),
            (2023, "Howard Schultz", 2022.0, 1.0, 0.0),
            (2024, "Laxman Narasimhan", 2023.0, 0.0, 0.0),
        ],
    )
    + _rows(
        "VLO",
        [
            (2004, "William E. Greehey", 1979.0, 0.0, 1.0),
            (2005, "William E. Greehey", 1979.0, 0.0, 1.0),
            (2006, "William R. Klesse", NA, NA, NA),
            (2007, "William R. Klesse", NA, 0.0, 1.0),
            (2008, "William R. Klesse", 2005.0, 0.0, NA),
            (2009, "William R. Klesse", 1981.0, 0.0, 1.0),
            (2015, "Joseph W. Gorder", 2003.0, 0.0, 1.0),
            (2016, "Joseph W. Gorder", 2014.0, 0.0, 1.0),
            (2017, "Joe Gorder", NA, NA, 1.0),
            (2024, "R. Lane Riggs", 2023.0, 0.0, 0.0),
        ],
    )
    + _rows(
        "AFL",
        [
            (1996, "Daniel P. Amos", 1983.0, 0.0, 0.0),
            (1997, "Daniel P. Amos", 2001.0, 0.0, 0.0),
            (1998, "Daniel P. Amos", 1990.0, 0.0, 0.0),
            (1999, "Daniel P. Amos", NA, 0.0, 0.0),
        ],
    )
    + _rows(
        "CENS",
        [
            (2010, "Alice Archer", NA, NA, 1.0),
            (2011, "Alice Archer", NA, NA, NA),
            (2012, None, NA, NA, NA),
            (2013, "Bob Baker", NA, NA, 1.0),
            (2014, "Bob Baker", 2016.0, NA, NA),
        ],
    )
)


def _out() -> tuple[pd.DataFrame, dict]:
    out, stats = impute_def14a(_FIXTURE.copy())
    out["year"] = out["as_of"].dt.year
    return out.set_index(["ticker", "year"]), stats


def test_stint_ids_split_on_identity_and_suffix() -> None:
    """WRB father/son share `berkley|w` but split on the suffix; SBUX Schultz has 3 stints."""
    df = _FIXTURE.sort_values(["ticker", "as_of"]).reset_index(drop=True)
    stint = ceo_stint_ids(df["ceo_name_proxy"], df["ticker"])
    df["stint"] = stint
    wrb = df.loc[df["ticker"] == "WRB", "stint"]
    assert wrb.nunique() == 2, f"WRB father and son must be two stints, got {wrb.tolist()}"
    sbux = df[df["ticker"] == "SBUX"]
    schultz = sbux.loc[sbux["ceo_name_proxy"].str.contains("Schultz"), "stint"]
    assert schultz.nunique() == 3, f"SBUX Schultz must have 3 stints, got {schultz.tolist()}"
    cens = df[df["ticker"] == "CENS"]
    assert cens["stint"].isna().sum() == 1, "the unknown CEO row must belong to no stint"
    assert cens["stint"].dropna().nunique() == 2
    # an unknown row breaks no stint
    gap = pd.DataFrame({"t": ["X"] * 3, "n": ["Ann Ash", None, "Ann Ash"]})
    g = ceo_stint_ids(gap["n"], gap["t"])
    assert g.iloc[0] == g.iloc[2] and pd.isna(g.iloc[1])
    # stints never span tickers
    assert df.groupby("stint")["ticker"].nunique().max() == 1
    print("\n=== SANITY CHECK: stint identification ===")
    print(f"  WRB stints {wrb.tolist()} | SBUX Schultz stints {sorted(schultz.unique())} | gap row {g.tolist()}")
    print("  CONCLUSION: the suffix separates father and son, a returning CEO opens a new stint, an unknown row joins none.")


def test_since_year_rule_and_no_cross_stint_carry() -> None:
    """Largest disclosed <= first proxy year; never the predecessor's value."""
    out, stats = _out()
    since = out["ceo_since_year"]
    # WRB: the son never inherits the father's 1967
    assert since[("WRB", 2018)] == 1967 and since[("WRB", 2019)] == 2019 and since[("WRB", 2020)] == 2019
    # SBUX: stint 1 -> 1987, stint 2 (2008) -> 2008, stint 3 (2023) -> 2022
    assert [since[("SBUX", y)] for y in (1998, 2000, 2008, 2009, 2023)] == [1987, 1987, 2008, 2008, 2022]
    # VLO: Greehey 1979 never reaches Klesse; Klesse 2005; Gorder 2014; Riggs 2023
    assert [since[("VLO", y)] for y in (2006, 2007, 2008, 2009)] == [2005] * 4
    assert [since[("VLO", y)] for y in (2015, 2016, 2017)] == [2014] * 3
    assert since[("VLO", 2024)] == 2023
    # AFL: 1983 / 2001 / 1990 with first proxy 1996 -> 1990, and the filed values stay in _raw
    assert (since.loc["AFL"] == 1990).all()
    assert out.loc[("AFL", 1997), "ceo_since_year_raw"] == 2001
    assert out.loc[("AFL", 1997), fill_kind_column("ceo_since_year")] == FILL_PERSON
    assert out.loc[("AFL", 1998), fill_kind_column("ceo_since_year")] == "filed"
    # person fills are dated at their own row; the evidence date is the disclosing filing
    vlo06 = out.loc[("VLO", 2006)]
    assert vlo06[source_date_column("ceo_since_year")] == pd.Timestamp("2006-04-01")
    assert vlo06[evidence_date_column("ceo_since_year")] == pd.Timestamp("2008-04-01")
    print("\n=== SANITY CHECK: since_year per stint ===")
    print(f"  VLO {since.loc['VLO'].tolist()} | AFL {since.loc['AFL'].tolist()} | SBUX {since.loc['SBUX'].tolist()}")
    print(f"  stats: { {k: v for k, v in stats.items() if 'since' in k or 'founder' in k or 'chair' in k} }")
    print("  CONCLUSION: one start year per stint, chosen by the largest-disclosed-<=-first-year rule, no predecessor value crosses.")


def test_no_disclosed_start_leaves_the_stint_as_filed() -> None:
    """No censored fallback: a stint without a disclosed start <= its first proxy year stays as filed."""
    out, stats = _out()
    since, kind = out["ceo_since_year"], out[fill_kind_column("ceo_since_year")]
    assert since.loc["CENS"].loc[[2010, 2011, 2012, 2013]].isna().all()
    # Bob's only disclosed value (2016) is after his first proxy year: kept as filed, never copied
    assert since[("CENS", 2014)] == 2016 and kind[("CENS", 2014)] == "filed"
    assert "censored" not in set(out[fill_kind_column("ceo_since_year")].dropna())
    assert stats.get("declined (no disclosed start <= stint's first proxy year): ceo_since_year", 0) >= 4
    print("\n=== SANITY CHECK: no censored fallback ===")
    print(f"  CENS since {since.loc['CENS'].tolist()} kinds {kind.loc['CENS'].tolist()}")
    print("  CONCLUSION: without a qualifying disclosure the start year stays unknown rather than guessed.")


def test_founder_mode_and_chair_stint_gate() -> None:
    out, stats = _out()
    founder, chair = out["ceo_is_founder"], out["ceo_is_board_chair"]
    assert founder[("WRB", 2019)] == 0.0, "the son must not inherit the father's founder flag"
    assert founder[("SBUX", 2009)] == 1.0 and founder[("VLO", 2006)] == 0.0 and founder[("VLO", 2017)] == 0.0
    assert out.loc[("SBUX", 2009), fill_kind_column("ceo_is_founder")] == FILL_PERSON
    # chair: never across a stint boundary, still within a stint
    assert pd.isna(chair[("WRB", 2019)]) and pd.isna(chair[("VLO", 2006)])
    assert chair[("VLO", 2008)] == 1.0
    assert pd.isna(chair[("CENS", 2012)]), "an unknown-CEO row belongs to no stint and takes no chair carry"
    assert stats.get("declined (stint boundary): ceo_is_board_chair", 0) >= 2
    print("\n=== SANITY CHECK: founder mode and chair gate ===")
    print(f"  founder WRB {founder.loc['WRB'].tolist()} | chair VLO {chair.loc['VLO'].tolist()}")
    print("  CONCLUSION: founder is a stint constant, chair is carried only inside the stint.")


def test_completion_is_idempotent_and_keeps_raw() -> None:
    once, _ = impute_def14a(_FIXTURE.copy())
    twice, _ = impute_def14a(once.copy())
    for col in ("ceo_since_year", "ceo_is_founder"):
        pd.testing.assert_series_equal(once[col], twice[col])
        pd.testing.assert_series_equal(once[f"{col}_raw"], twice[f"{col}_raw"])
        filed = _FIXTURE.set_index("accession_number")[col]
        raw = once.set_index("accession_number")[f"{col}_raw"].reindex(filed.index)
        pd.testing.assert_series_equal(raw, filed, check_names=False)
    print("\n=== SANITY CHECK: idempotent completion ===")
    print("  CONCLUSION: a second pass reads the filed values from `_raw` and reproduces the first pass exactly.")
