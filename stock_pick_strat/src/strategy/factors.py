"""
Value/growth factor scoring on top of the fundamentals snapshot.

Approach: cross-sectional z-scores per factor, combined into one composite
score. Missing data is handled by scoring only on available factors per row
(not dropping the whole row), which keeps more of the universe usable.
"""
import numpy as np
import pandas as pd


def zscore(s: pd.Series) -> pd.Series:
    """Standard z-score, ignoring NaNs, robust to zero-variance columns."""
    mean, std = s.mean(skipna=True), s.std(skipna=True)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def compute_factor_scores(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Input: fundamentals snapshot with yfinance-style columns
    (trailingPE, enterpriseValue, ebitda, revenueGrowth,
    researchAndDevelopment, totalRevenue, marketCap, ...).

    Output: same dataframe with added factor + composite score columns.
    Lower PE / EV-EBITDA = better value -> scores are sign-flipped so that,
    consistently, HIGHER composite score = more attractive.
    """
    df = fundamentals.copy()

    # EV/EBITDA (value) — lower is better, so flip sign after z-score
    df["ev_ebitda"] = df["enterpriseValue"] / df["ebitda"].replace(0, np.nan)
    df.loc[df["ev_ebitda"] < 0, "ev_ebitda"] = np.nan  # negative EBITDA -> meaningless multiple

    # PE (value) — lower is better
    df["pe"] = df["trailingPE"]
    df.loc[df["pe"] < 0, "pe"] = np.nan

    # Revenue growth (growth) — higher is better
    df["rev_growth"] = df["revenueGrowth"]

    # R&D intensity (context factor — how much a company reinvests)
    df["rd_intensity"] = df["researchAndDevelopment"].abs() / df["totalRevenue"].replace(0, np.nan)

    z_value_pe = -zscore(df["pe"])
    z_value_ev_ebitda = -zscore(df["ev_ebitda"])
    z_growth = zscore(df["rev_growth"])

    df["value_score"] = pd.concat([z_value_pe, z_value_ev_ebitda], axis=1).mean(axis=1, skipna=True)
    df["growth_score"] = z_growth
    df["composite_score"] = pd.concat(
        [df["value_score"], df["growth_score"]], axis=1
    ).mean(axis=1, skipna=True)

    return df.sort_values("composite_score", ascending=False)


def top_n(scored: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    return scored.dropna(subset=["composite_score"]).head(n)
