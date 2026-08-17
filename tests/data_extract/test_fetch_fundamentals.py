"""Tests for the QUARTERLY SEC fundamentals extraction
(src/data_extract/fetch_fundamentals.py).

Synthetic, known-truth companyfacts are the right tool here: to prove that the
extractor recovers the correct discrete quarter from a year-to-date ladder,
derives Q4 = FY - (Q1+Q2+Q3), sums the right trailing-twelve-months, and never
stamps a value before its filing date, we must know the true inputs. Each test
prints the sanity conclusion it validated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.fetch_fundamentals import (
    _quarterly_flow,
    _extract_concept,
    build_ticker_history,
)
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.common.pit import infer_yoy_periods


# --------------------------------------------------------------------------- #
# Builders for synthetic XBRL facts                                            #
# --------------------------------------------------------------------------- #
def _obs(end, start, val, filed=None, form="10-Q"):
    end = pd.Timestamp(end)
    filed = pd.Timestamp(filed) if filed else end + pd.Timedelta(days=40)
    return {"end": end.date().isoformat(),
            "start": (pd.Timestamp(start).date().isoformat() if start else None),
            "filed": filed.date().isoformat(), "form": form, "fp": "Q1", "val": val}


def _concept_df(rows):
    return _extract_concept({"X": {"units": {"USD": rows}}}, ["X"])


QSTART = ["{y}-01-01", "{y}-04-01", "{y}-07-01", "{y}-10-01"]
QEND = ["{y}-03-31", "{y}-06-30", "{y}-09-30", "{y}-12-31"]


# --------------------------------------------------------------------------- #
# 1. YTD ladder -> discrete quarters                                           #
# --------------------------------------------------------------------------- #
def test_quarterly_flow_decumulates_ytd():
    # cash-flow style: only cumulative year-to-date facts (3M/6M/9M/12M)
    disc = [20.0, 25.0, 30.0, 35.0]                 # true discrete quarters
    ytd = np.cumsum(disc)                            # 20, 45, 75, 110
    rows = [_obs(f"2021-{m}", "2021-01-01", v, form=f)
            for m, v, f in zip(["03-31", "06-30", "09-30", "12-31"], ytd,
                               ["10-Q", "10-Q", "10-Q", "10-K"])]
    out = _quarterly_flow(_concept_df(rows)).sort_values("end")

    assert len(out) == 4
    np.testing.assert_allclose(out["val"].to_numpy(), disc, rtol=1e-9)

    print("\n=== SANITY CHECK: YTD de-cumulation ===")
    print(f"  cumulative {list(ytd)} -> discrete {list(out['val'])} (expected {disc}).")


# --------------------------------------------------------------------------- #
# 2. Discrete Q1-Q3 + FY -> Q4 derived                                         #
# --------------------------------------------------------------------------- #
def test_quarterly_flow_derives_q4():
    # income-statement style: discrete 3-month Q1-Q3 (own starts) + annual FY
    rows = [
        _obs("2021-03-31", "2021-01-01", 100.0),
        _obs("2021-06-30", "2021-04-01", 110.0),
        _obs("2021-09-30", "2021-07-01", 120.0),
        _obs("2021-12-31", "2021-01-01", 460.0, filed="2022-02-20", form="10-K"),  # FY
    ]
    out = _quarterly_flow(_concept_df(rows)).sort_values("end").reset_index(drop=True)

    assert len(out) == 4
    q4 = out.loc[out["end"] == pd.Timestamp("2021-12-31"), "val"].iloc[0]
    assert abs(q4 - (460.0 - 330.0)) < 1e-9         # FY - (Q1+Q2+Q3) = 130
    # Q4 is only public once the 10-K is filed
    q4_filed = out.loc[out["end"] == pd.Timestamp("2021-12-31"), "filed"].iloc[0]
    assert pd.Timestamp(q4_filed) >= pd.Timestamp("2022-02-20")

    print("\n=== SANITY CHECK: Q4 derivation ===")
    print(f"  FY=460, Q1..Q3=330 -> Q4={q4} (expected 130); filed={q4_filed} (>= 10-K date).")


# --------------------------------------------------------------------------- #
# 3. build_ticker_history: quarterly cadence, TTM levels, no look-ahead        #
# --------------------------------------------------------------------------- #
def _make_facts():
    """3 fiscal years of calendar quarters. Income items as discrete 3M + FY;
    cash-flow items as YTD cumulative; equity/shares as instants."""
    rev = {2020: [100, 110, 120, 130], 2021: [140, 150, 160, 170], 2022: [180, 190, 200, 210]}
    ni = {2020: [10, 11, 12, 13], 2021: [14, 15, 16, 17], 2022: [18, 19, 20, 21]}
    da = {y: [5, 5, 5, 5] for y in (2020, 2021, 2022)}
    ocf = {2020: [20, 25, 30, 35], 2021: [22, 27, 32, 37], 2022: [24, 29, 34, 39]}
    capex = {2020: [5, 6, 7, 8], 2021: [6, 7, 8, 9], 2022: [7, 8, 9, 10]}

    def discrete_plus_fy(byyear):
        rows = []
        for y, q in byyear.items():
            for i in range(3):  # discrete Q1-Q3 (own quarter starts)
                rows.append(_obs(QEND[i].format(y=y), QSTART[i].format(y=y), float(q[i])))
            rows.append(_obs(f"{y}-12-31", f"{y}-01-01", float(sum(q)),
                             filed=f"{y+1}-02-20", form="10-K"))  # FY -> Q4 derived
        return rows

    def ytd(byyear):
        rows = []
        for y, q in byyear.items():
            c = np.cumsum(q)
            for i in range(4):
                form = "10-K" if i == 3 else "10-Q"
                filed = f"{y+1}-02-20" if i == 3 else None
                rows.append(_obs(QEND[i].format(y=y), f"{y}-01-01", float(c[i]),
                                 filed=filed, form=form))
        return rows

    def instants(start_val, step):
        rows, v = [], start_val
        for y in (2020, 2021, 2022):
            for i in range(4):
                rows.append(_obs(QEND[i].format(y=y), None, float(v)))
                v += step
        return rows

    usg = {
        "Revenues": {"units": {"USD": discrete_plus_fy(rev)}},
        "NetIncomeLoss": {"units": {"USD": discrete_plus_fy(ni)}},
        "OperatingIncomeLoss": {"units": {"USD": discrete_plus_fy(ni)}},
        "DepreciationDepletionAndAmortization": {"units": {"USD": discrete_plus_fy(da)}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ytd(ocf)}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": ytd(capex)}},
        "StockholdersEquity": {"units": {"USD": instants(200, 5)}},
    }
    # A REALISTIC share count. `apply_plausibility_guards` nulls anything under
    # SHARES_OUTSTANDING_MIN (1e6), which is what catches the 147 sub-million and 166 zero
    # rows the 2026-07 audit found in the live table; an S&P 500 name cannot have fewer
    # (even BRK.A, the extreme, has ~1.4M A-shares). The old 1,000 was below that floor,
    # so this fixture was asserting a magnitude no constituent ever reports.
    dei = {"EntityCommonStockSharesOutstanding":
           {"units": {"shares": instants(1_000_000_000, 10_000_000)}}}
    return {"facts": {"us-gaap": usg, "dei": dei}}


def test_build_ticker_history_quarterly_ttm_and_no_leakage():
    h = build_ticker_history("TEST", _make_facts())
    h["as_of_dt"] = pd.to_datetime(h["as_of"])
    h["fe_dt"] = pd.to_datetime(h["fiscal_end"])

    # cadence: one row per quarter, ~91 days apart
    gap = h["fe_dt"].diff().dt.days.median()
    assert 85 <= gap <= 100, f"not quarterly (median gap {gap}d)"

    # NO LOOK-AHEAD: every value is stamped strictly after its fiscal period ends
    assert (h["as_of_dt"] > h["fe_dt"]).all(), "as_of must post-date fiscal_end"

    row = h.set_index("fiscal_end")
    # TTM revenue at 2020-12-31 = 100+110+120+130 = 460 (== reported FY)
    assert abs(row.loc["2020-12-31", "totalRevenue"] - 460.0) < 1e-6
    # TTM FCF at 2020-12-31 = sum(ocf-capex) = (15+19+23+27) = 84
    assert abs(row.loc["2020-12-31", "freeCashflow"] - 84.0) < 1e-6
    # TTM EBITDA = oi_ttm(46) + da_ttm(20) = 66
    assert abs(row.loc["2020-12-31", "ebitda"] - 66.0) < 1e-6
    # YoY growth at 2021-12-31 = TTM_2021/TTM_2020 - 1 = 620/460 - 1
    assert abs(row.loc["2021-12-31", "revenueGrowth"] - (620.0 / 460.0 - 1)) < 1e-6
    assert row["sharesOutstanding"].notna().any()

    print("\n=== SANITY CHECK: quarterly TTM history + no look-ahead ===")
    print(f"  {len(h)} quarterly rows, median gap {gap:.0f}d; as_of always > fiscal_end.")
    print(f"  TTM rev(2020)=460 (==FY), TTM FCF=84, TTM EBITDA=66, "
          f"YoY rev(2021)={row.loc['2021-12-31','revenueGrowth']:.3f} (=620/460-1). Correct.")


# --------------------------------------------------------------------------- #
# 4. YoY period inference (quarterly -> 4, annual -> 1)                         #
# --------------------------------------------------------------------------- #
def test_infer_yoy_periods():
    q = pd.DataFrame({"ticker": "A",
                      "as_of": pd.date_range("2020-03-31", periods=8, freq="91D").astype(str)})
    a = pd.DataFrame({"ticker": "A",
                      "as_of": pd.date_range("2016-02-01", periods=6, freq="365D").astype(str)})
    assert infer_yoy_periods(q) == 4
    assert infer_yoy_periods(a) == 1
    print("\n=== SANITY CHECK: YoY cadence inference ===")
    print("  ~91d filings -> 4 periods/yr; ~365d filings -> 1. Growth stays true YoY.")


# --------------------------------------------------------------------------- #
# 5. Point-in-time daily mapping never leaks the next filing                   #
# --------------------------------------------------------------------------- #
def test_fundamentals_to_daily_is_point_in_time():
    fund = pd.DataFrame({
        "ticker": ["Z", "Z"],
        "as_of": ["2021-05-10", "2021-08-09"],   # Q1 and Q2 filing dates
        "profitMargins": [0.20, 0.25],
    })
    idx = pd.bdate_range("2021-05-01", "2021-09-01")
    daily = fundamentals_to_daily(fund, "profitMargins", idx)

    assert np.isnan(daily.loc[pd.Timestamp("2021-05-07"), "Z"])   # before Q1 public
    assert abs(daily.loc[pd.Timestamp("2021-08-06"), "Z"] - 0.20) < 1e-9  # still Q1
    assert abs(daily.loc[pd.Timestamp("2021-08-09"), "Z"] - 0.25) < 1e-9  # Q2 on filing day

    print("\n=== SANITY CHECK: point-in-time feature mapping ===")
    print("  Q1 value held until the Q2 filing date, then updates. No look-ahead into "
          "a not-yet-filed quarter.")
