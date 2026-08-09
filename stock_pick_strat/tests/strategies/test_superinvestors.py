"""
Positions-panel loader for the `super_investors` sleeve (src/strategies/utils/superinvestors.py).
Validates each source's point-in-time `as_of` stamping + $ aggregation, and that the final
outer-merge aligns sources on (ticker, as_of) without a many-to-many row blow-up.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.strategies.utils.superinvestors import (
    _aggregate_activist, _aggregate_insiders, _aggregate_shorts, _aggregate_superinvestors,
    merge_positions_panel,
)


def test_aggregate_insiders_filters_to_open_market_and_nets_value():
    df = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA", "AAA"],
        "filing_date": pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-05", "2024-01-05"]),
        "transaction_code": ["P", "S", "P", "A"],          # A = grant, must be ignored
        "shares": [100, 40, 900, 5_000],
        "value_usd": [1_000.0, 400.0, 9_000.0, 50_000.0],
        "shares_owned_after": [1_100, 360, 2_900, 5_000],
        "transaction_sk": [1, 2, 3, 4],
    })
    out = _aggregate_insiders(df).set_index("ticker")
    assert out.loc["AAA", "insider_buy_value"] == 10_000.0     # 1,000 (P) + 9,000 (P) — the A row excluded
    assert out.loc["AAA", "insider_sell_value"] == 400.0
    assert out.loc["AAA", "insider_net_value"] == 9_600.0
    assert out.loc["AAA", "insider_n_transactions"] == 3       # 2 P + 1 S, NOT the grant
    assert out.loc["AAA", "as_of"] == pd.Timestamp("2024-01-05")   # stamped on filing_date

    print("\n=== SANITY CHECK: insider aggregation (open-market only) ===")
    print(f"  buy=${out.loc['AAA','insider_buy_value']:.0f} sell=${out.loc['AAA','insider_sell_value']:.0f} "
          f"net=${out.loc['AAA','insider_net_value']:.0f}; grant (code A) correctly excluded. Validated.")


def test_aggregate_insiders_pct_moved_against_pooled_prior_stake():
    df = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA", "AAA"],
        "filing_date": pd.to_datetime(["2024-01-05"] * 4),
        "transaction_code": ["P", "S", "P", "A"],
        "shares": [100, 40, 900, 5_000],
        "value_usd": [1_000.0, 400.0, 9_000.0, 50_000.0],
        "shares_owned_after": [1_100, 360, 2_900, 5_000],
        "transaction_sk": [1, 2, 3, 4],
    })
    out = _aggregate_insiders(df).set_index("ticker")
    # net move = +1,000 bought - 40 sold = +960, against the pooled PRE-trade stake recovered by
    # undoing that day's net move from the post-trade balances: 4,360 + 40 - 1,000 = 3,400
    assert out.loc["AAA", "insider_buy_shares"] == 1_000 and out.loc["AAA", "insider_sell_shares"] == 40
    assert out.loc["AAA", "insider_pct_moved"] == pytest.approx(960 / 3_400)

    print("\n=== SANITY CHECK: insider net % of pooled prior stake moved ===")
    print(f"  +1,000 bought / -40 sold against a recovered pre-trade stake of 3,400 -> "
          f"pct_moved={out.loc['AAA','insider_pct_moved']:.2%}. Validated.")


def test_aggregate_insiders_pct_moved_skips_unknown_balances():
    df = pd.DataFrame({
        "ticker": ["EEE", "EEE"],
        "filing_date": pd.to_datetime(["2024-03-01", "2024-03-01"]),
        "transaction_code": ["S", "S"],
        "shares": [100, 50],
        "value_usd": [2_000.0, 1_000.0],
        "shares_owned_after": [900, float("nan")],   # 2nd seller's post-sale balance unknown
        "transaction_sk": [10, 11],
    })
    out = _aggregate_insiders(df).set_index("ticker")
    # the unknown post-sale balance contributes nothing to the denominator (NaN is skipped, not
    # read as a 0 balance), so the base is the one known seller's 900 + the day's 150 sold
    assert out.loc["EEE", "insider_sell_shares"] == 150       # both rows still counted in the flow
    assert out.loc["EEE", "insider_pct_moved"] == pytest.approx(-150 / 1_050)

    print("\n=== SANITY CHECK: insider pct_moved with an unknown shares_owned_after ===")
    print(f"  seller with unknown post-sale balance adds nothing to the base (NaN skipped, not "
          f"zero-filled) -> pct_moved={out.loc['EEE','insider_pct_moved']:.2%}; "
          f"insider_sell_shares={out.loc['EEE','insider_sell_shares']:.0f} still counts both. Validated.")


def test_aggregate_activist_nets_buy_sell_value():
    df = pd.DataFrame({
        "ticker": ["BBB", "BBB"],
        "filing_date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
        "transaction_type": ["Buy", "Sell"],
        "quantity": [10_000, 2_000],
        "price_per_share": [20.0, 21.0],
        "trade_seq": [0, 1],
    })
    out = _aggregate_activist(df).set_index("ticker")
    assert out.loc["BBB", "activist_buy_value"] == 200_000.0
    assert out.loc["BBB", "activist_sell_value"] == 42_000.0
    assert out.loc["BBB", "activist_net_value"] == 158_000.0

    print("\n=== SANITY CHECK: activist (13D) aggregation ===")
    print(f"  net=${out.loc['BBB','activist_net_value']:.0f} on filing_date "
          f"{out.loc['BBB','as_of'].date()} (not trade_date). Validated.")


def _superinvestor_lifecycle_frame() -> pd.DataFrame:
    """Two managers ('1', '2') over 3 quarters. Manager 1 holds CCC+ZZZ from Q1 (its own
    first-ever quarter in the data), grows CCC in Q2, then fully exits CCC in Q3 (still filing
    ZZZ that quarter, so the exit is detectable). Manager 2 holds ZZZ from Q1 too, only starts
    CCC in Q2 (a genuine new position, NOT a first-observation artifact since it already has
    ZZZ history), then trims CCC in Q3. Both managers file on the same date each quarter so
    every ticker/quarter lands on one shared `as_of` row."""
    return pd.DataFrame({
        "cik":        ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2"],
        "ticker":     ["ZZZ", "CCC", "ZZZ", "CCC", "ZZZ", "ZZZ", "ZZZ", "CCC", "ZZZ", "CCC"],
        "period":     pd.to_datetime(["2024-03-31", "2024-03-31", "2024-06-30", "2024-06-30",
                                      "2024-09-30", "2024-03-31", "2024-06-30", "2024-06-30",
                                      "2024-09-30", "2024-09-30"]),
        "filing_date": pd.to_datetime(["2024-05-10", "2024-05-10", "2024-08-12", "2024-08-12",
                                       "2024-11-08", "2024-05-10", "2024-08-12", "2024-08-12",
                                       "2024-11-08", "2024-11-08"]),
        "shares":     [300, 1_000, 300, 1_200, 300, 500, 500, 800, 500, 600],
        "value_usd":  [3_000.0, 100_000.0, 3_300.0, 132_000.0, 3_600.0,
                      5_000.0, 5_500.0, 88_000.0, 6_000.0, 66_000.0],
    })


def test_aggregate_superinvestors_reports_first_observation_as_init_not_new():
    out = _aggregate_superinvestors(_superinvestor_lifecycle_frame())
    ccc_q1 = out[(out["ticker"] == "CCC") & (out["as_of"] == "2024-05-10")].iloc[0]
    # manager 1's Q1 CCC row is ITS first-ever filing in the data: the 1,000 shares are a real
    # level and DO move it off 0, so they must count in buy_shares or the conservation identity
    # would break on day one. But they are not an observed purchase (the stake predates our
    # window), so `n_new` stays 0 and the amount is isolated in `init_shares` for netting out.
    assert ccc_q1["superinvestor_shares"] == 1_000 and ccc_q1["superinvestor_n_managers"] == 1
    assert ccc_q1["superinvestor_n_new"] == 0
    assert ccc_q1["superinvestor_buy_shares"] == 1_000
    assert ccc_q1["superinvestor_init_shares"] == 1_000

    print("\n=== SANITY CHECK: superinvestor first filing = init, not a new position ===")
    print(f"  Q1 CCC shares={ccc_q1['superinvestor_shares']:.0f}, buy_shares="
          f"{ccc_q1['superinvestor_buy_shares']:.0f} (identity holds) but n_new="
          f"{ccc_q1['superinvestor_n_new']:.0f} and init_shares="
          f"{ccc_q1['superinvestor_init_shares']:.0f} (warm-up isolated). Validated.")


def test_aggregate_superinvestors_detects_new_position_and_increase():
    out = _aggregate_superinvestors(_superinvestor_lifecycle_frame())
    ccc_q2 = out[(out["ticker"] == "CCC") & (out["as_of"] == "2024-08-12")].iloc[0]
    # manager 1: 1,000 -> 1,200 (a real increase, +20%) ; manager 2: 0 -> 800 (a genuine NEW
    # position, since manager 2 already has ZZZ history -- not a first-observation artifact)
    assert ccc_q2["superinvestor_shares"] == 2_000 and ccc_q2["superinvestor_n_managers"] == 2
    assert ccc_q2["superinvestor_n_new"] == 1 and ccc_q2["superinvestor_n_increased"] == 1
    assert ccc_q2["superinvestor_buy_shares"] == 1_000        # 200 (increase) + 800 (new)

    print("\n=== SANITY CHECK: superinvestor new position vs increase ===")
    print(f"  Q2 CCC: n_new={ccc_q2['superinvestor_n_new']:.0f}, n_increased="
          f"{ccc_q2['superinvestor_n_increased']:.0f}, buy_shares={ccc_q2['superinvestor_buy_shares']:.0f}. "
          f"Validated.")


def test_aggregate_superinvestors_separates_exit_from_decrease():
    out = _aggregate_superinvestors(_superinvestor_lifecycle_frame())
    ccc_q3 = out[(out["ticker"] == "CCC") & (out["as_of"] == "2024-11-08")].iloc[0]
    # manager 1 has no CCC row in Q3 despite still filing ZZZ that quarter -> a full exit
    # (1,200 -> 0, synthesized), NOT a "decrease"; manager 2 trims 800 -> 600 (-25%), a real decrease
    assert ccc_q3["superinvestor_shares"] == 600              # only manager 2 still holds it
    assert ccc_q3["superinvestor_n_managers"] == 1             # the exited manager doesn't count
    assert ccc_q3["superinvestor_n_exited"] == 1 and ccc_q3["superinvestor_n_decreased"] == 1
    assert ccc_q3["superinvestor_sell_shares"] == 1_400        # 1,200 (exit) + 200 (decrease)

    print("\n=== SANITY CHECK: superinvestor exit vs decrease ===")
    print(f"  Q3 CCC: n_exited={ccc_q3['superinvestor_n_exited']:.0f}, n_decreased="
          f"{ccc_q3['superinvestor_n_decreased']:.0f}, sell_shares={ccc_q3['superinvestor_sell_shares']:.0f}. "
          f"Validated.")


def _staggered_frame() -> pd.DataFrame:
    """Manager 1 files each quarter on the 1st; manager 2 files the SAME quarters two weeks
    later. A naive groupby-on-as_of fragments each quarter into two incomplete rows, each
    missing whichever manager didn't file that exact day."""
    return pd.DataFrame({
        "cik":         ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2"],
        "ticker":      ["ZZZ", "CCC", "ZZZ", "CCC", "ZZZ", "ZZZ", "ZZZ", "CCC", "ZZZ", "CCC"],
        "period":      pd.to_datetime(["2024-03-31", "2024-03-31", "2024-06-30", "2024-06-30",
                                       "2024-09-30", "2024-03-31", "2024-06-30", "2024-06-30",
                                       "2024-09-30", "2024-09-30"]),
        "filing_date": pd.to_datetime(["2024-05-01", "2024-05-01", "2024-08-01", "2024-08-01",
                                       "2024-11-01", "2024-05-15", "2024-08-15", "2024-08-15",
                                       "2024-11-15", "2024-11-15"]),
        "shares":      [300, 1_000, 300, 1_200, 300, 500, 500, 800, 500, 600],
        "value_usd":   [3_000.0, 100_000.0, 3_300.0, 132_000.0, 3_600.0,
                       5_000.0, 5_500.0, 88_000.0, 6_000.0, 66_000.0],
    })


def test_aggregate_superinvestors_emits_a_gapless_daily_index():
    """The panel must answer "what did the cohort hold on THIS day" for every business day, not
    only on the ~1% of days somebody filed -- otherwise a strategy reading the merged panel gets
    NaN on almost every date it wants to trade."""
    out = _aggregate_superinvestors(_staggered_frame())
    ccc = out[out["ticker"] == "CCC"].sort_values("as_of").reset_index(drop=True)
    expected_days = pd.bdate_range("2024-05-01", "2024-11-15")
    assert list(ccc["as_of"]) == list(expected_days)
    assert ccc["superinvestor_shares"].notna().all()      # level defined on EVERY day, no holes

    # a day in the middle of a quarter, when nobody filed anything at all: the level still
    # carries (manager 1's 1,200 from 08-01 + manager 2's 800 from 08-15), and the flows are 0
    # (not NaN) because no movement was disclosed that day
    quiet = ccc[ccc["as_of"] == "2024-09-20"].iloc[0]
    assert quiet["superinvestor_shares"] == 2_000 and quiet["superinvestor_buy_shares"] == 0.0
    assert quiet["superinvestor_n_managers"] == 2

    print("\n=== SANITY CHECK: superinvestor panel is a gapless daily grid ===")
    print(f"  CCC spans {len(ccc)} business days {ccc['as_of'].min().date()}..{ccc['as_of'].max().date()} "
          f"with 0 missing levels; quiet day 2024-09-20 (nobody filed) carries shares=2,000 "
          f"across 2 managers, flows=0. Validated.")


def test_aggregate_superinvestors_level_forward_fills_across_staggered_filing_dates():
    """The level must carry each manager's last-known holding forward across the other
    manager's filing dates, and reconcile with the disclosed flow on every consecutive day."""
    out = _aggregate_superinvestors(_staggered_frame())
    ccc = out[out["ticker"] == "CCC"].sort_values("as_of").reset_index(drop=True)

    # the key regression check: on 2024-08-15 (manager 2's filing date), manager 1 hasn't
    # re-filed since 2024-08-01 -- their 1,200 shares must still be carried into the level,
    # not dropped just because they have no row dated exactly 2024-08-15.
    at_0815 = ccc[ccc["as_of"] == "2024-08-15"].iloc[0]
    assert at_0815["superinvestor_shares"] == 2_000       # manager 1's 1,200 (carried) + manager 2's 800
    assert at_0815["superinvestor_n_managers"] == 2
    assert at_0815["superinvestor_buy_shares"] == 800      # only manager 2 actually moved that day

    # conservation invariant on EVERY consecutive pair of days across the whole daily grid
    levels = ccc["superinvestor_shares"].to_numpy()
    buys = ccc["superinvestor_buy_shares"].to_numpy()
    sells = ccc["superinvestor_sell_shares"].to_numpy()
    for i in range(1, len(ccc)):
        assert levels[i] == levels[i - 1] + buys[i] - sells[i], (
            f"conservation broken at {ccc['as_of'][i].date()}: "
            f"{levels[i]} != {levels[i-1]} + {buys[i]} - {sells[i]}")

    print("\n=== SANITY CHECK: superinvestor level survives staggered filing dates ===")
    print(f"  2024-08-15 level=2,000 (manager 1's stale-but-valid 1,200 + manager 2's new 800); "
          f"shares(t)=shares(t-1)+buy(t)-sell(t) holds across all {len(ccc)-1} daily transitions. "
          f"Validated.")


def test_aggregate_superinvestors_ignores_a_superseded_late_filing():
    """REGRESSION (found on live data): 13F filing dates are not monotonic in period. 5 of 67
    roster managers file out of order -- Egerton dumped five quarters at once, years late.
    Accumulating deltas computed in PERIOD order but applied in FILING order walked the level
    through states that never existed, surfacing as 6 impossible NEGATIVE share levels.
    A report disclosing an older period than one already public must be ignored, not replayed."""
    df = pd.DataFrame({
        "cik":         ["1", "1", "1"],
        "ticker":      ["CCC", "CCC", "CCC"],
        "period":      pd.to_datetime(["2024-03-31", "2024-09-30", "2024-06-30"]),
        # Q2 is filed LAST (2024-11-01), long after Q3 was already public on 2024-08-15
        "filing_date": pd.to_datetime(["2024-05-01", "2024-08-15", "2024-11-01"]),
        "shares":      [1_000, 100, 900],
        "value_usd":   [100_000.0, 10_000.0, 90_000.0],
    })
    # `end` extends the daily grid past the last real event -- the superseded Q2 filing is
    # dropped outright, so without it the panel would simply stop at 2024-08-15
    out = (_aggregate_superinvestors(df, end=pd.Timestamp("2024-12-02"))
           .sort_values("as_of").reset_index(drop=True))

    # Q3 (the newest period) is the last word: the stale Q2 filing must not revise it back up
    assert out.loc[out["as_of"] == "2024-08-15", "superinvestor_shares"].iloc[0] == 100
    assert out.loc[out["as_of"] == "2024-11-01", "superinvestor_shares"].iloc[0] == 100
    assert out.loc[out["as_of"] == "2024-12-02", "superinvestor_shares"].iloc[0] == 100
    assert (out["superinvestor_shares"] >= 0).all()        # never a negative stake

    # and the conservation identity still holds everywhere on the daily grid
    lv = out["superinvestor_shares"].to_numpy()
    bs, ss = out["superinvestor_buy_shares"].to_numpy(), out["superinvestor_sell_shares"].to_numpy()
    for i in range(1, len(out)):
        assert lv[i] == lv[i - 1] + bs[i] - ss[i]

    print("\n=== SANITY CHECK: superseded late 13F filing is ignored ===")
    print(f"  Q2 filed 2024-11-01 discloses an OLDER period than the already-public Q3 -> "
          f"level stays 100 (Q3) instead of being revised to 900; no negative levels; "
          f"identity holds across all {len(out)-1} daily transitions. Validated.")


def test_aggregate_shorts_lags_one_business_day_and_computes_ratio():
    df = pd.DataFrame({
        "ticker": ["DDD"], "date": pd.to_datetime(["2024-04-05"]),   # a Friday
        "short_volume": [400.0], "total_volume": [1_000.0],
    })
    out = _aggregate_shorts(df).set_index("ticker")
    assert out.loc["DDD", "as_of"] == pd.Timestamp("2024-04-08")     # next business day (Monday)
    assert out.loc["DDD", "short_ratio"] == 0.4

    print("\n=== SANITY CHECK: short-interest aggregation ===")
    print(f"  date 2024-04-05 (Fri) -> as_of {out.loc['DDD','as_of'].date()} (next business day); "
          f"short_ratio={out.loc['DDD','short_ratio']:.2f}. Validated.")


def test_merge_positions_panel_outer_joins_without_row_blowup():
    insiders = pd.DataFrame({"ticker": ["AAA"], "as_of": pd.to_datetime(["2024-01-05"]),
                             "insider_net_value": [9_600.0]})
    superinv = pd.DataFrame({"ticker": ["BBB"], "as_of": pd.to_datetime(["2024-02-01"]),
                             "superinvestor_shares": [1_500.0]})
    shorts = pd.DataFrame({"ticker": ["AAA"], "as_of": pd.to_datetime(["2024-01-05"]),
                          "short_ratio": [0.1]})

    merged = merge_positions_panel(insiders, superinv, shorts)
    assert len(merged) == 2                                   # AAA/2024-01-05 + BBB/2024-02-01, no blow-up
    aaa = merged.set_index("ticker").loc["AAA"]
    assert aaa["insider_net_value"] == 9_600.0 and aaa["short_ratio"] == 0.1
    bbb = merged.set_index("ticker").loc["BBB"]
    assert bbb["superinvestor_shares"] == 1_500.0
    assert pd.isna(bbb["insider_net_value"])                  # no insider filing that day -> NaN, not 0

    print("\n=== SANITY CHECK: outer-merge positions panel ===")
    print(f"  {len(merged)} rows from 3 non-empty sources on (ticker, as_of); missing sources "
          f"stay NaN (e.g. BBB has no insider_net_value) rather than 0. Validated.")


if __name__ == "__main__":
    test_aggregate_insiders_filters_to_open_market_and_nets_value()
    test_aggregate_insiders_pct_moved_against_pooled_prior_stake()
    test_aggregate_insiders_pct_moved_skips_unknown_balances()
    test_aggregate_activist_nets_buy_sell_value()
    test_aggregate_superinvestors_reports_first_observation_as_init_not_new()
    test_aggregate_superinvestors_detects_new_position_and_increase()
    test_aggregate_superinvestors_separates_exit_from_decrease()
    test_aggregate_superinvestors_emits_a_gapless_daily_index()
    test_aggregate_superinvestors_level_forward_fills_across_staggered_filing_dates()
    test_aggregate_shorts_lags_one_business_day_and_computes_ratio()
    test_merge_positions_panel_outer_joins_without_row_blowup()
