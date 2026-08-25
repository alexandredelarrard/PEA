"""
Synthetic substrates for the validator's tests.

`FundamentalsValidator` takes a `Substrates`, never a `Context` -- so every check is exercised
against frames built here, with **no DB, no CLI and no network**. That is the whole point of
the constructor's signature, and it is what makes "plant exactly one violation" a runnable
test rather than an aspiration.

The base fixture is a CLEAN filer: one industrial ticker with a smooth, complete, internally
consistent history. Each test then plants ONE defect into a copy of it and asserts that the
check it targets fires and that nothing else new does. A check that cannot be planted cannot
be trusted, and a check that fires on the clean base has a threshold bug.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.validate.fundamentals.substrate import Substrates

#: A whole clean filer, deliberately small: 12 quarterly publication events on an industrial
#: template. Small enough to reason about by hand, long enough for `series_shape` (min 8) and
#: `level_outlier` (min 8) to run rather than abstain.
EVENTS = 12
TICKER = "TST"
REGIME = "industrial"

#: The peer group `peer_ratio` needs. FIVE is its minimum, so six peers means the check RUNS on
#: the base fixture -- otherwise every peer_ratio test would be silently testing an abstention.
PEERS = ("PR1", "PR2", "PR3", "PR4", "PR5", "PR6")


def _dates(n: int = EVENTS) -> pd.DatetimeIndex:
    """Quarter ends, and the filing dates ~40 days later that become `as_of`."""
    return pd.date_range("2020-03-31", periods=n, freq="QE")


@pytest.fixture(scope="session")
def catalogue():
    return load_catalogue()


def make_facts(ticker: str = TICKER, *, scale: float = 1.0, leverage: float = 0.25,
               n: int = EVENTS) -> pd.DataFrame:
    """One filer's `fundamentals_facts` rows: a smooth, internally consistent quarterly series.

    Two knobs, both load-bearing for the tests that read this fixture:

      * `scale` sizes the filer. Values GROW smoothly rather than being flat, because a flat
        series has zero dispersion and `level_outlier` / `scale` would abstain on it -- so a
        test asserting "the clean base is silent" would prove nothing.
      * `leverage` is `totalDebt / totalAssets`, and peers must DIFFER on it. `peer_ratio`
        MAD-scores the ratio across the peer group, and among k identical peers the modified Z
        of a lone outlier is bounded by `0.6745 * k` -- so a peer group with no dispersion
        caps the achievable score at 3.37 for the 5-peer minimum. A fixture with identical
        peers would be testing that ceiling rather than the check.
    """
    rows = []
    for i, period_end in enumerate(_dates(n)):
        filing_date = period_end + pd.Timedelta(days=40)
        growth = 1.05 ** i
        assets = 8_000_000_000 * growth * scale
        for field, value, duration in (
                ("totalRevenue", 1_000_000_000 * growth * scale, "quarterly"),
                ("costOfRevenue", 600_000_000 * growth * scale, "quarterly"),
                ("grossProfit", 400_000_000 * growth * scale, "quarterly"),
                ("capex", 90_000_000 * growth * scale, "quarterly"),
                ("totalAssets", assets, "instant"),
                ("totalLiabilities", 5_000_000_000 * growth * scale, "instant"),
                ("stockholdersEquity", 3_000_000_000 * growth * scale, "instant"),
                ("totalDebt", assets * leverage, "instant"),
        ):
            rows.append({
                "ticker": ticker, "accession_number": f"{ticker}-{i:04d}",
                "field": field, "duration_type": duration,
                "period_end": period_end,
                "period_start": period_end - pd.Timedelta(days=91),
                "period_days": 91 if duration != "instant" else None,
                "fiscal_year": period_end.year,
                "fiscal_period": f"Q{(i % 4) + 1}",
                "cik": "0000000001", "form": "10-Q", "filing_date": filing_date,
                "is_amendment": False, "period_of_report": period_end, "regime": REGIME,
                "value": value, "resolution_method": "as_reported",
                "source_concept": f"us-gaap:{field}", "roll_up_children": None,
                "root_anchor": None, "role_uri": "http://x/BalanceSheet",
                "is_extension": False, "dc_code": None, "adjustment": None,
            })
    return pd.DataFrame(rows)


def make_history(facts: pd.DataFrame, catalogue) -> pd.DataFrame:
    """A `fundamentals_history`-shaped frame consistent with `facts`.

    Built from the catalogue's own `history_columns`, so it has the real 69-column contract and
    `column_contract` passes on the clean base instead of firing on a hand-written subset.
    """
    columns = catalogue.history_columns
    rows = []
    for ticker, group in facts.groupby("ticker"):
        for i, (period_end, chunk) in enumerate(group.groupby("period_end")):
            as_of = pd.Timestamp(chunk["filing_date"].iloc[0])
            row = {c: None for c in columns}
            row.update({
                "ticker": ticker, "as_of": as_of, "fiscal_end": pd.Timestamp(period_end),
                "fiscal_quarter": (i % 4) + 1, "regime": REGIME,
                "publication_form": "10-Q", "is_amendment": False,
                "amended_fiscal_end": pd.NaT, "amended_fields": None,
            })
            for field, value in zip(chunk["field"], chunk["value"]):
                if field in row:
                    row[field] = float(value)
            rows.append(row)
    frame = pd.DataFrame(rows, columns=list(columns))
    for column in ("as_of", "fiscal_end", "amended_fiscal_end"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["is_amendment"] = frame["is_amendment"].astype(bool)
    return frame


def make_codes(history: pd.DataFrame, catalogue) -> pd.DataFrame:
    """A DENSE reason-code frame: one row per null cell, so `unexplained_null` is 0 on the base.

    Dense and not sparse, exactly like the real table. The zero-unexplained-nulls gate is a
    LEFT JOIN on (ticker, as_of, field), and a sparse fixture would make that gate untestable.
    """
    from src.data_extract.utils.fundamentals.kpi_catalogue import (
        HISTORY_KEYS, HISTORY_PROVENANCE, HISTORY_REGIME)

    skip = {*HISTORY_KEYS, HISTORY_REGIME, *HISTORY_PROVENANCE}
    value_columns = [c for c in history.columns if c not in skip]
    long = history.melt(id_vars=["ticker", "as_of"], value_vars=value_columns,
                        var_name="field", value_name="value")
    nulls = long[long["value"].isna()]
    return pd.DataFrame({
        "ticker": nulls["ticker"], "as_of": nulls["as_of"], "field": nulls["field"],
        "dc_code": "not_disclosed", "combined_into": None, "rejected_value": float("nan"),
    }).reset_index(drop=True)


def build_substrates(catalogue, facts: pd.DataFrame) -> Substrates:
    """A complete, self-consistent `Substrates` from a facts frame. The one constructor tests use."""
    history = make_history(facts, catalogue)
    codes = make_codes(history, catalogue)
    return Substrates(catalogue=catalogue, history=history, codes=codes, facts=facts,
                      employees=pd.DataFrame(),
                      tickers=tuple(sorted(facts["ticker"].unique())))


@pytest.fixture
def clean_facts() -> pd.DataFrame:
    """The base filer plus six same-regime peers -- enough for `peer_ratio` to RUN.

    The peers differ in SIZE and in LEVERAGE. Size alone would not be enough: `peer_ratio`
    scores a ratio, which is scale-free, so six differently-sized peers with identical
    leverage are six identical points to it. See `make_facts`.
    """
    frames = [make_facts(TICKER, leverage=0.25)]
    frames += [make_facts(peer, scale=1.0 + 0.1 * i, leverage=0.20 + 0.02 * i)
               for i, peer in enumerate(PEERS)]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def clean(catalogue, clean_facts) -> Substrates:
    return build_substrates(catalogue, clean_facts)
