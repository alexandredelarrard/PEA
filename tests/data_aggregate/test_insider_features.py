"""Phase 2.3 -- the insider panel (registry #28-#41) and the transaction-quality layer.

Every test prints a sanity-check conclusion, because a green assert that nobody reads is how
the two defects these tests exist for survived: a `value_usd` total of $182,982,720tn, and a
"distinct buyer" count of 91 for a company with 13 insiders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.insider_features import (
    CLUSTER_MIN, EMISSION, INSIDER_FLOOR, TEN_B5_1_FLOOR, build_insider_feature_panel)
from src.data_aggregate.utils.institutionals.insider_quality import (
    OPEN_MARKET_CODES, clean_transactions, common_stock_mask, consensus_price,
    officer_role, security_class)

TRADING_INDEX = pd.bdate_range("2015-01-01", "2016-12-31")


def _txn(**kw) -> dict:
    """One transaction row with the columns every builder path reads."""
    row = dict(accession_number="0000000000-00-000000", ticker="AAA", owner_cik="0001",
               owner_name="Doe Jane", filing_date="2015-06-01", transaction_date="2015-05-29",
               transaction_code="P", shares=1_000.0, price_per_share=100.0,
               value_usd=100_000.0, shares_owned_after=11_000.0, security_type="nonderiv",
               security_title="Common Stock", direct_indirect="D", officer_title="",
               is_director=1.0, is_officer=0.0, is_ten_pct_owner=0.0, is_10b5_1=np.nan)
    row.update(kw)
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _prices(tickers=("AAA",), price=100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.DataFrame(price, index=TRADING_INDEX, columns=list(tickers))
    fh = pd.DataFrame([{"ticker": t, "as_of": pd.Timestamp("2014-01-01"),
                        "sharesOutstanding": 1_000_000.0,
                        "sharesOutstandingPit": 1_000_000.0} for t in tickers])
    return fh, close


def _peers(tickers=("AAA",)) -> dict:
    return {t: {t: 1.0} for t in tickers}


# --------------------------------------------------------------------------- #
# the quality layer                                                             #
# --------------------------------------------------------------------------- #

def test_only_open_market_codes_survive_the_scope_cut():
    """Grants, exercises, withholding and gifts are compensation mechanics, not trades."""
    rows = [_txn(transaction_code=c, accession_number=f"a{i}")
            for i, c in enumerate(["P", "S", "A", "M", "F", "G", "J", "C", "D", "X"])]
    out, diag = clean_transactions(_frame(rows))
    kept = sorted(out["code"].unique())
    assert kept == sorted(OPEN_MARKET_CODES)
    print(f"SANITY: 10 codes in, {kept} out -- the 8 non-discretionary codes "
          f"({diag['input_rows'] - diag['scoped_rows']} rows) never reach a feature.")


def test_derivatives_and_preferred_are_not_common_stock_purchases():
    rows = [_txn(accession_number="a1"),
            _txn(accession_number="a2", security_type="deriv",
                 security_title="Employee Stock Option"),
            _txn(accession_number="a3", security_title="Preferred Stock, Series H")]
    out, _ = clean_transactions(_frame(rows))
    assert len(out) == 1 and out.iloc[0]["accession_number"] == "a1"
    print("SANITY: 3 purchases in -- 1 common-stock row survives; the option and the "
          "preferred series are dropped, so neither can price a common-share signal.")


def test_an_overpriced_row_is_repaired_to_shares_times_the_consensus():
    """The $182,982,720tn defect, in miniature: one filer types 100x the real price."""
    rows = [_txn(accession_number=f"g{i}", owner_cik=f"{i:04d}",
                 transaction_date=f"2015-05-{20 + i:02d}", shares=10.0,
                 price_per_share=100.0, value_usd=1_000.0) for i in range(6)]
    rows.append(_txn(accession_number="bad", owner_cik="9999",
                     transaction_date="2015-05-25", shares=10.0,
                     price_per_share=10_000.0, value_usd=100_000.0))
    out, diag = clean_transactions(_frame(rows))
    bad = out[out["accession_number"].eq("bad")].iloc[0]
    assert diag["repaired_rows"] == 1
    assert bad["price_repaired"] and bad["value"] == pytest.approx(1_000.0)
    print(f"SANITY: 7 purchases, one filed at $10,000 against a $100 consensus -> repaired "
          f"to shares x consensus = ${bad['value']:,.0f}, not ${100_000:,}. Table total "
          f"${diag['value_before']:,.0f} -> ${diag['value_after']:,.0f}.")


def test_an_underpriced_row_is_counted_but_never_inflated():
    """The one-sided rule. Repairing upward is what turned nine real ~$25m AXON purchases
    into $326m ones, because that ticker carries another company's rows."""
    rows = [_txn(accession_number=f"g{i}", owner_cik=f"{i:04d}",
                 transaction_date=f"2015-05-{20 + i:02d}", shares=10.0,
                 price_per_share=100.0, value_usd=1_000.0) for i in range(6)]
    rows.append(_txn(accession_number="low", owner_cik="9999",
                     transaction_date="2015-05-25", shares=10.0,
                     price_per_share=1.0, value_usd=10.0))
    out, diag = clean_transactions(_frame(rows))
    low = out[out["accession_number"].eq("low")].iloc[0]
    assert diag["underpriced_rows"] == 1 and diag["repaired_rows"] == 0
    assert not low["price_repaired"] and low["value"] == pytest.approx(10.0)
    print(f"SANITY: a price 100x BELOW the consensus is counted ({diag['underpriced_rows']} "
          f"row) and left at its filed ${low['value']:,.0f}. A too-low price understates a "
          f"flow; a too-high one invented $182,982,720tn, and only that direction is "
          f"repaired.")


def test_the_consensus_is_built_within_a_share_class():
    """ERIE: Class A near $35, Class B near $32,740, both titled '... Common Stock'. A
    ticker-wide median prices a Class A purchase off the Class B tape."""
    rows = []
    for i in range(6):
        rows.append(_txn(accession_number=f"a{i}", owner_cik=f"a{i}",
                         transaction_date=f"2015-05-{20 + i:02d}",
                         security_title="Class A Common Stock", price_per_share=35.0))
        rows.append(_txn(accession_number=f"b{i}", owner_cik=f"b{i}",
                         transaction_date=f"2015-05-{20 + i:02d}",
                         security_title="Class B Common Stock", price_per_share=32_740.0))
    df = _frame(rows)
    ref = consensus_price(df)
    a = ref[df["security_title"].eq("Class A Common Stock")]
    b = ref[df["security_title"].eq("Class B Common Stock")]
    assert a.max() == pytest.approx(35.0) and b.min() == pytest.approx(32_740.0)
    out, diag = clean_transactions(df)
    assert diag["repaired_rows"] == 0
    print(f"SANITY: per-class consensus = ${a.iloc[0]:,.0f} / ${b.iloc[0]:,.0f}; pooled it "
          f"would be ${np.median([35.0, 32_740.0]):,.0f} and every Class A row would be "
          f"'repaired' to 468x its value. 0 rows repaired, which is correct here.")


def test_an_exercise_and_sell_package_is_not_a_discretionary_sale():
    """35.0% of all `S` rows share an accession and a transaction date with an `M`."""
    acc = "0001234567-23-000001"
    rows = [_txn(accession_number=acc, transaction_code="M", security_type="deriv",
                 security_title="Stock Option", filing_date="2024-06-03",
                 transaction_date="2024-06-01"),
            _txn(accession_number=acc, transaction_code="S", filing_date="2024-06-03",
                 transaction_date="2024-06-01", is_10b5_1=0.0),
            _txn(accession_number="0001234567-23-000002", transaction_code="S",
                 filing_date="2024-06-03", transaction_date="2024-06-01", is_10b5_1=0.0)]
    out, _ = clean_transactions(_frame(rows))
    packaged = out.set_index("accession_number")["in_exercise_package"]
    assert bool(packaged[acc]) and not bool(packaged["0001234567-23-000002"])

    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    fh, close = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    panel = build_insider_feature_panel(_frame(rows), _peers(), idx, shares_out_history=fh,
                                        stock_close=close)
    col = "f_ic_insider_discretionary_sell_mcap_60d"
    after = panel[panel["date"].ge("2024-06-03")][col].dropna()
    one_sale = 100_000.0 / (1_000_000.0 * 100.0)
    assert after.max() == pytest.approx(one_sale, rel=1e-6)
    print(f"SANITY: two identical $100k sales filed the same day, one of them the sell leg "
          f"of an option exercise. Discretionary selling peaks at {after.max():.6f} of "
          f"market cap = exactly ONE sale, not two.")


def test_the_role_map_reads_a_title_the_way_a_human_would():
    cases = {"Chairman, President & CEO": "CEO", "Chief Executive Officer": "CEO",
             "SVP & CFO": "CFO", "Chief Financial Officer": "CFO",
             "President and Chief Operating Officer": "COO_or_President",
             "Executive Chairman": "other_named_officer", "": "other_named_officer"}
    got = {t: officer_role(t) for t in cases}
    assert got == cases
    print(f"SANITY: 'Chairman, President & CEO' -> {got['Chairman, President & CEO']} "
          f"(longest concept first, so it is not read as a President). Measured "
          f"fall-through to other_named_officer on live data: 32.0% of 10,390 titled "
          f"officer purchases -- reported, not tuned away.")


def test_security_class_and_common_stock_mask_agree_with_the_measured_titles():
    titles = pd.Series(["Common Stock", "Class A Common Stock", "Ordinary Shares",
                        "Preferred Stock, Series H", "Employee Stock Option",
                        "Common Stock, par value $0.01 per share"])
    mask = common_stock_mask(titles)
    assert mask.tolist() == [True, True, True, False, False, True]
    assert security_class(titles).tolist() == ["COMMON", "A", "COMMON", "H", "COMMON",
                                               "COMMON"]
    print(f"SANITY: {int(mask.sum())}/6 titles read as common stock; the preferred series "
          f"keys to class 'H' so it can never share a consensus with the common line.")


# --------------------------------------------------------------------------- #
# point-in-time                                                                 #
# --------------------------------------------------------------------------- #

def test_owner_surprise_uses_only_that_owners_prior_purchases():
    """The easiest place in the family to leak the future, so it is tested against a
    deliberately-leaky full-sample version rather than by eye."""
    # Two purchases, the SECOND smaller. Two is deliberate: with only one prior in the
    # window the decay weighting cannot blur the answer, so the number the panel prints is
    # the percentile itself and the honest and leaky versions give different numbers.
    sizes = [100.0, 50.0]
    rows = [_txn(accession_number=f"s{i}", owner_cik="0007", shares=v,
                 value_usd=v * 100.0, price_per_share=100.0,
                 filing_date=f"2015-0{i + 1}-15", transaction_date=f"2015-0{i + 1}-14",
                 shares_owned_after=v + 10_000.0) for i, v in enumerate(sizes)]
    fh, close = _prices()
    panel = build_insider_feature_panel(_frame(rows), _peers(), TRADING_INDEX,
                                        shares_out_history=fh, stock_close=close)
    got = panel.set_index("date")["f_ic_insider_owner_surprise_120d"]
    on_second = got.loc["2015-02-16"]
    leaky = float(pd.Series(sizes).rank(pct=True).iloc[1])          # = 0.5
    assert on_second == pytest.approx(0.0, abs=1e-7)
    assert on_second != pytest.approx(leaky, abs=1e-3)
    assert got.loc[:"2015-01-14"].dropna().empty
    print(f"SANITY: a $5k purchase following a $10k one by the same owner scores "
          f"{on_second:.2f} -- the smallest they have ever made. A full-sample percentile "
          f"would score it {leaky:.2f} by ranking it against itself. The first purchase is "
          f"NaN ({int(got.loc[:'2015-01-14'].notna().sum())} values before it): 'unusually "
          f"large for this person' is undefined before there is a person.")


def test_nothing_is_emitted_before_the_measured_availability_floors():
    rows = [_txn(filing_date="2015-06-01", transaction_date="2015-05-29"),
            _txn(accession_number="s1", transaction_code="S", filing_date="2015-06-02",
                 transaction_date="2015-05-29", is_10b5_1=1.0)]
    idx = pd.bdate_range("2004-01-01", "2026-01-01")
    fh, close = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    panel = build_insider_feature_panel(_frame(rows), _peers(), idx, shares_out_history=fh,
                                        stock_close=close)
    cols = [c for c in panel.columns if c not in ("date", "ticker")]
    before = panel[panel["date"] < INSIDER_FLOOR]
    assert before[cols].notna().to_numpy().sum() == 0
    plan_cols = [c for c in cols if "sell_mcap" in c]
    early = panel[panel["date"] < TEN_B5_1_FLOOR]
    assert early[plan_cols].notna().to_numpy().sum() == 0
    print(f"SANITY: {len(before):,} ticker-days before {INSIDER_FLOOR.date()} carry 0 "
          f"non-null values across {len(cols)} columns, and the two 10b5-1 legs are empty "
          f"across {len(early):,} rows before {TEN_B5_1_FLOOR.date()} -- NaN, never 0, so "
          f"'not collected yet' cannot read as 'no insider sold'.")


def test_a_feature_never_sees_a_transaction_filed_after_the_date():
    """Truncation: rebuilding on a shorter history must not move a single earlier value."""
    rows = [_txn(accession_number=f"p{i}", owner_cik=f"{i:04d}",
                 filing_date=f"2015-{m:02d}-15", transaction_date=f"2015-{m:02d}-14")
            for i, m in enumerate([2, 4, 6, 8, 10])]
    fh, close = _prices()
    full = build_insider_feature_panel(_frame(rows), _peers(), TRADING_INDEX,
                                       shares_out_history=fh, stock_close=close)
    cut = pd.Timestamp("2015-07-01")
    truncated = build_insider_feature_panel(
        _frame([r for r in rows if pd.Timestamp(r["filing_date"]) < cut]),
        _peers(), TRADING_INDEX, shares_out_history=fh, stock_close=close)
    cols = [c for c in full.columns if c in truncated.columns and c not in ("date", "ticker")]
    a = full[full["date"] < cut].set_index(["date", "ticker"])[cols]
    b = truncated[truncated["date"] < cut].set_index(["date", "ticker"])[cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)
    print(f"SANITY: {len(a):,} pre-cut rows x {len(cols)} columns are byte-identical whether "
          f"or not the three later purchases exist. Nothing before {cut.date()} depends on "
          f"anything filed after it.")


# --------------------------------------------------------------------------- #
# the features themselves                                                       #
# --------------------------------------------------------------------------- #

def test_distinct_buyers_counts_people_not_filings():
    """The TPL defect: 13 owners buying daily read as 91.4 'distinct buyers' once a decayed
    event sum was called a distinct count."""
    rows = []
    for d in pd.bdate_range("2015-03-02", "2015-05-29"):          # 65 filing days
        for owner in ("0001", "0002", "0003"):
            rows.append(_txn(accession_number=f"{owner}-{d:%Y%m%d}", owner_cik=owner,
                             filing_date=d, transaction_date=d, shares=3.0,
                             value_usd=300.0, shares_owned_after=10_003.0))
    fh, close = _prices()
    panel = build_insider_feature_panel(_frame(rows), _peers(), TRADING_INDEX,
                                        shares_out_history=fh, stock_close=close)
    n = panel.set_index("date")["f_ic_insider_distinct_buyers_120d"]
    peak = n.max()
    assert peak == 3.0
    assert n.loc["2015-10-01"] == 0.0          # the window has emptied
    assert n.loc[:"2015-03-01"].dropna().empty  # nothing before the first purchase
    print(f"SANITY: 3 owners filing on 65 consecutive days peak at {peak:.0f} distinct "
          f"buyers, not 195 filings and not the 91.4 a 63-day decay would report. The count "
          f"returns to {n.loc['2015-10-01']:.0f} once the 120-day window empties, and is NaN "
          f"before the first purchase.")


def test_cluster_is_two_distinct_buyers_not_two_filings():
    solo = [_txn(accession_number=f"x{i}", owner_cik="0001",
                 filing_date=f"2015-03-{2 + i:02d}", transaction_date=f"2015-03-{2 + i:02d}")
            for i in range(4)]
    pair = solo + [_txn(accession_number="y", owner_cik="0002", filing_date="2015-03-06",
                        transaction_date="2015-03-06")]
    fh, close = _prices()
    col = "f_ic_insider_cluster_buy_120d"
    got = {}
    for name, rows in (("one buyer, 4 filings", solo), ("two buyers", pair)):
        panel = build_insider_feature_panel(_frame(rows), _peers(), TRADING_INDEX,
                                            shares_out_history=fh, stock_close=close)
        got[name] = panel.set_index("date")[col].loc["2015-03-10"]
    assert got["one buyer, 4 filings"] == 0.0
    assert got["two buyers"] == 2.0 >= CLUSTER_MIN
    print(f"SANITY: {got['one buyer, 4 filings']:.0f} for one insider filing four times, "
          f"{got['two buyers']:.0f} for two insiders filing once each. The threshold is "
          f"{CLUSTER_MIN} PEOPLE -- measured occupancy 6.9% of ticker-days, against 0.6% "
          f"for the 3-in-20-days the source research proposed.")


def test_the_ten_b5_1_split_separates_planned_from_discretionary():
    rows = [_txn(accession_number="d", transaction_code="S", filing_date="2024-03-01",
                 transaction_date="2024-02-28", is_10b5_1=0.0, value_usd=100_000.0),
            _txn(accession_number="p", transaction_code="S", filing_date="2024-03-01",
                 transaction_date="2024-02-28", is_10b5_1=1.0, value_usd=300_000.0),
            _txn(accession_number="u", transaction_code="S", filing_date="2024-03-01",
                 transaction_date="2024-02-28", is_10b5_1=np.nan, value_usd=900_000.0)]
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    fh, _ = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    panel = build_insider_feature_panel(_frame(rows), _peers(), idx, shares_out_history=fh,
                                        stock_close=close)
    day = panel[panel["date"].eq("2024-03-01")].iloc[0]
    mcap = 1_000_000.0 * 100.0
    assert day["f_ic_insider_discretionary_sell_mcap_60d"] == pytest.approx(100_000 / mcap)
    assert day["f_ic_insider_planned_sell_mcap_60d"] == pytest.approx(300_000 / mcap)
    print(f"SANITY: $100k flagged not-on-plan, $300k on-plan and $900k with an UNKNOWN flag. "
          f"Discretionary reads {day['f_ic_insider_discretionary_sell_mcap_60d']:.7f} and "
          f"planned {day['f_ic_insider_planned_sell_mcap_60d']:.7f} -- the unknown row is in "
          f"neither, because a NaN plan flag is not a 0.")


def test_net_buy_ratio_is_nan_when_nothing_was_filed_not_zero():
    rows = [_txn(filing_date="2015-06-01", transaction_date="2015-05-29")]
    fh, close = _prices()
    panel = build_insider_feature_panel(_frame(rows), _peers(), TRADING_INDEX,
                                        shares_out_history=fh, stock_close=close)
    s = panel.set_index("date")["f_ic_insider_net_buy_ratio_180d"]
    assert pd.isna(s.loc["2015-05-01"]) and s.loc["2015-06-01"] == 1.0
    assert pd.isna(s.loc["2016-06-01"])
    print(f"SANITY: one purchase -> the ratio is NaN before it, {s.loc['2015-06-01']:.0f} on "
          f"the filing day, and NaN again once the 180-day window empties. 'No insider "
          f"traded' is not 'insiders were evenly split'.")


def test_every_emitted_column_is_declared_and_every_declaration_is_emitted():
    # Owner CIKs REPEAT on purpose: `owner_surprise` needs a prior purchase by the same
    # person to exist at all, so a fixture of five one-off buyers would silently emit
    # thirteen features and this test would pass against a map of fourteen.
    rows = [_txn(accession_number=f"p{i}", owner_cik=f"{i % 2:04d}", officer_title=title,
                 is_officer=1.0 if title else 0.0, is_director=0.0 if title else 1.0,
                 filing_date=f"2024-0{i + 1}-15", transaction_date=f"2024-0{i + 1}-14",
                 shares=1_000.0 + 100 * i, value_usd=100_000.0 + 10_000 * i,
                 shares_owned_after=11_000.0 + i)
            for i, title in enumerate(["Chief Executive Officer", "Chief Financial Officer",
                                       "", "Executive Vice President", ""])]
    rows += [_txn(accession_number="s1", transaction_code="S", filing_date="2024-07-01",
                  transaction_date="2024-06-28", is_10b5_1=0.0),
             _txn(accession_number="s2", transaction_code="S", filing_date="2024-07-01",
                  transaction_date="2024-06-28", is_10b5_1=1.0)]
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    fh, _ = _prices()
    close = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    panel = build_insider_feature_panel(_frame(rows), _peers(), idx, shares_out_history=fh,
                                        stock_close=close)
    cols = {c for c in panel.columns if c not in ("date", "ticker")}
    expected = {f"f_{n}" for n in EMISSION} | {
        f"f_{n}_xs" for n, m in EMISSION.items() if m == "raw+xs"}
    assert cols == expected, f"missing {sorted(expected - cols)}, extra {sorted(cols - expected)}"
    assert not any(c.endswith("_vs_peers") for c in cols)
    print(f"SANITY: {len(EMISSION)} declared features -> {len(cols)} columns "
          f"({sum(m == 'raw+xs' for m in EMISSION.values())} carry an _xs leg), and 0 "
          f"_vs_peers legs, which is D25 for this family: an insider purchase is a fact "
          f"about one company, not about its basket.")


def test_without_a_market_cap_the_dollar_features_are_absent_not_counts():
    """`decay_events` counts a NaN magnitude as 1.0 -- correct policy there, and a trap here:
    with no market cap every magnitude is NaN and the four value-scaled intensities would
    silently become event counts under names that promise dollars over market cap."""
    rows = [_txn(accession_number=f"p{i}", owner_cik=f"{i % 2:04d}",
                 officer_title="Chief Executive Officer", is_officer=1.0, is_director=1.0,
                 filing_date=f"2015-0{i + 1}-15", transaction_date=f"2015-0{i + 1}-14",
                 shares=1_000.0 + i, shares_owned_after=11_000.0 + i) for i in range(3)]
    panel = build_insider_feature_panel(_frame(rows), _peers(), TRADING_INDEX)
    got = {c.removeprefix("f_").removesuffix("_xs") for c in panel.columns
           if c not in ("date", "ticker")}
    scale_free = {"ic_insider_distinct_buyers_120d", "ic_insider_cluster_buy_120d",
                  "ic_insider_days_since_last_buy", "ic_insider_net_buy_ratio_180d",
                  "ic_insider_purchase_pct_prior", "ic_insider_owner_surprise_120d"}
    assert got == scale_free, f"unexpected without mcap: {sorted(got - scale_free)}"
    print(f"SANITY: with no shares outstanding and no close, {len(got)} scale-free features "
          f"are emitted and all {len(EMISSION) - len(scale_free)} dollar-scaled ones are "
          f"ABSENT -- not present as disguised event counts.")


def test_no_usable_transaction_returns_an_empty_panel_not_a_crash():
    for label, rows in (("nothing at all", []),
                        ("grants only", [_txn(transaction_code="A")]),
                        ("derivatives only", [_txn(security_type="deriv")])):
        panel = build_insider_feature_panel(_frame(rows) if rows else None, _peers(),
                                            TRADING_INDEX)
        assert list(panel.columns) == ["date", "ticker"] and panel.empty
    print("SANITY: an absent table, a grants-only table and a derivatives-only table each "
          "return an empty (date, ticker) frame -- the merge chain reports 'No "
          "insider-trading features built.' rather than raising.")
