"""AC-011: a Form 4/A supersedes its original cell by cell, strictly point in time.

Known-truth fixtures checked day by day, and one mixed fixture checked against the brute-force
as-of reference in `insider_pit_reference` for every date and every insider field.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals import insider_features
from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from src.data_aggregate.utils.institutionals.sink import ConditioningSink
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import compare, frame, reference_panel, row, sentinel_rows

IDX = pd.bdate_range("2024-01-01", "2024-12-31")
TICKERS = ["AAA", "BBB"]
MCAP = 1_000_000.0 * 100.0
COMPLETE = pd.Timestamp("2024-12-31")


def _build(df_rows: pd.DataFrame, sink: ConditioningSink | None = None) -> pd.DataFrame:
    close = pd.DataFrame(100.0, index=IDX, columns=TICKERS)
    fh = pd.DataFrame([{"ticker": t, "as_of": pd.Timestamp("2023-01-01"), "sharesOutstanding": 1e6, "sharesOutstandingPit": 1e6} for t in TICKERS])
    frames = make_frames(IDX, {t: {t: 1.0} for t in TICKERS}, close_split=close)
    return build_insider_feature_panel(frames, df_rows, shares_out_history=fh, complete_through=COMPLETE, sink=sink)


def _series(panel: pd.DataFrame, column: str, ticker: str = "AAA") -> pd.Series:
    return panel[panel["ticker"].eq(ticker)].set_index("date")[column].reindex(IDX)


def _acc(n: int) -> str:
    return f"0000000000-24-{n:06d}"


def mixed_rows() -> list[dict]:
    """Every case on two tickers: partial value-changing 4/A with a new cell, identical 4/A, unlinked 4/A,
    copy pair, coincidence pair, sell-side 4/A, a 4/A filed after the window, a 4/A chain, and an
    unpriced sale priced by its 4/A, plus the `sentinel_rows`. An amendment's rows precede its original's, so the as-of
    re-ranking of an owner's earlier purchases equals the frozen prior set (see
    `test_owner_surprise_prior_set_is_frozen_at_the_visible_day`)."""
    ceo, cfo = "Chief Executive Officer", "Chief Financial Officer"
    o1 = dict(accession_number=_acc(11), owner_cik="0000000003", filing_date="2024-02-01")
    a1 = dict(accession_number=_acc(12), owner_cik="0000000003", filing_date="2024-02-08", document_type="4/A", original_submission_date="2024-02-01")
    o2 = dict(accession_number=_acc(21), owner_cik="0000000004", filing_date="2024-03-01", officer_title=cfo, is_officer=1.0, is_director=0.0)
    o3 = dict(accession_number=_acc(31), owner_cik="0000000002", filing_date="2024-07-01", transaction_code="S", transaction_date="2024-06-28")
    o4 = dict(accession_number=_acc(41), ticker="BBB", owner_cik="0000000010", filing_date="2024-02-15", transaction_date="2024-02-13")
    o5 = dict(accession_number=_acc(51), ticker="BBB", owner_cik="0000000011", filing_date="2024-03-04", transaction_date="2024-03-01")
    o6 = dict(
        accession_number=_acc(61), ticker="BBB", owner_cik="0000000012", filing_date="2024-08-01", transaction_date="2024-07-30", transaction_code="S"
    )
    amend = dict(document_type="4/A")
    return [
        row(accession_number=_acc(1), owner_cik="0000000001", officer_title=ceo, is_officer=1.0, is_director=0.0),
        row(
            accession_number=_acc(2), owner_cik="0000000002", filing_date="2024-01-12", transaction_code="S", shares=900.0, shares_owned_after=4_000.0
        ),
        # partial value-changing 4/A: cell X (2 rows -> 1), new cell Z; cell Y unreported
        row(**a1, transaction_date="2024-01-30", shares=800.0, price_per_share=100.375, shares_owned_after=5_800.0),
        row(**a1, transaction_date="2024-02-02", shares=150.0, price_per_share=103.0, shares_owned_after=5_950.0, row_sequence=2),
        row(**o1, transaction_date="2024-01-30", shares=500.0, price_per_share=100.0, shares_owned_after=5_500.0),
        row(**o1, transaction_date="2024-01-30", shares=300.0, price_per_share=101.0, shares_owned_after=5_800.0, row_sequence=2),
        row(**o1, transaction_date="2024-01-31", shares=200.0, price_per_share=102.0, shares_owned_after=6_000.0, row_sequence=3),
        row(
            accession_number=_acc(13),
            owner_cik="0000000003",
            filing_date="2024-02-05",
            transaction_date="2024-02-02",
            shares=100.0,
            shares_owned_after=6_100.0,
        ),
        # identical linked 4/A
        row(
            **{**o2, **amend, "accession_number": _acc(22), "filing_date": "2024-03-06", "original_submission_date": "2024-03-01"},
            transaction_date="2024-02-28",
            shares=400.0,
            price_per_share=105.0,
            shares_owned_after=9_400.0,
        ),
        row(**o2, transaction_date="2024-02-28", shares=400.0, price_per_share=105.0, shares_owned_after=9_400.0),
        # unlinked 4/A (its original is not stored)
        row(
            accession_number=_acc(23),
            owner_cik="0000000005",
            filing_date="2024-04-02",
            transaction_date="2024-03-28",
            shares=250.0,
            price_per_share=104.0,
            shares_owned_after=2_250.0,
            **amend,
            original_submission_date="2023-12-01",
        ),
        # coincidence pair: equal trade, different holdings -> two trades
        row(
            accession_number=_acc(24),
            owner_cik="0000000006",
            filing_date="2024-05-01",
            transaction_date="2024-04-30",
            shares=100.0,
            price_per_share=106.0,
            shares_owned_after=1_100.0,
        ),
        row(
            accession_number=_acc(25),
            owner_cik="0000000007",
            filing_date="2024-05-02",
            transaction_date="2024-04-30",
            shares=100.0,
            price_per_share=106.0,
            shares_owned_after=2_100.0,
        ),
        # copy pair: one position reported by two owners -> one trade from the first filing
        row(
            accession_number=_acc(26),
            owner_cik="0000000008",
            filing_date="2024-06-03",
            transaction_date="2024-05-31",
            price_per_share=107.0,
            shares_owned_after=50_000.0,
            direct_indirect="I",
        ),
        row(
            accession_number=_acc(27),
            owner_cik="0000000009",
            filing_date="2024-06-07",
            transaction_date="2024-05-31",
            price_per_share=107.0,
            shares_owned_after=50_000.0,
            direct_indirect="I",
        ),
        # sell-side value change
        row(
            **{**o3, **amend, "accession_number": _acc(32), "filing_date": "2024-07-15", "original_submission_date": "2024-07-01"},
            shares=2_000.0,
            price_per_share=110.0,
            shares_owned_after=9_000.0,
        ),
        row(**o3, shares=2_000.0, price_per_share=108.0, shares_owned_after=9_000.0),
        row(
            accession_number=_acc(14),
            owner_cik="0000000001",
            filing_date="2024-10-01",
            transaction_date="2024-09-27",
            shares=2_000.0,
            shares_owned_after=13_000.0,
            officer_title=ceo,
            is_officer=1.0,
            is_director=0.0,
        ),
        # BBB
        row(
            accession_number=_acc(40),
            ticker="BBB",
            owner_cik="0000000020",
            filing_date="2024-01-16",
            transaction_date="2024-01-12",
            shares=500.0,
            price_per_share=50.0,
            shares_owned_after=5_500.0,
        ),
        row(
            accession_number=_acc(39),
            ticker="BBB",
            owner_cik="0000000021",
            filing_date="2024-01-18",
            transaction_date="2024-01-12",
            transaction_code="S",
            shares=700.0,
            price_per_share=50.0,
            shares_owned_after=3_000.0,
            is_10b5_1=1.0,
        ),
        # a 4/A filed after the original left every window
        row(
            **{**o4, **amend, "accession_number": _acc(42), "filing_date": "2024-09-16", "original_submission_date": "2024-02-15"},
            shares=600.0,
            price_per_share=51.0,
            shares_owned_after=6_600.0,
        ),
        row(**o4, shares=600.0, price_per_share=50.0, shares_owned_after=6_600.0),
        # chain: change, identical, change
        row(
            **{**o5, **amend, "accession_number": _acc(52), "filing_date": "2024-03-11", "original_submission_date": "2024-03-04"},
            shares=300.0,
            price_per_share=53.0,
            shares_owned_after=3_300.0,
        ),
        row(
            **{**o5, **amend, "accession_number": _acc(53), "filing_date": "2024-03-18", "original_submission_date": "2024-03-04"},
            shares=300.0,
            price_per_share=53.0,
            shares_owned_after=3_300.0,
        ),
        row(
            **{**o5, **amend, "accession_number": _acc(54), "filing_date": "2024-04-01", "original_submission_date": "2024-03-04"},
            shares=310.0,
            price_per_share=53.0,
            shares_owned_after=3_310.0,
        ),
        row(**o5, shares=300.0, price_per_share=52.0, shares_owned_after=3_300.0),
        # an unpriced sale (unknown window) that its 4/A prices
        row(
            **{**o6, **amend, "accession_number": _acc(62), "filing_date": "2024-08-09", "original_submission_date": "2024-08-01"},
            shares=1_000.0,
            price_per_share=49.0,
            shares_owned_after=4_000.0,
        ),
        row(**o6, shares=1_000.0, price_per_share=None, value_usd=np.nan, shares_owned_after=4_000.0),
        *sentinel_rows(),
    ]


def test_a_value_changing_4a_switches_the_cell_on_its_filing_day_and_keeps_the_original_anchor():
    """RED on base: from the 4/A day both versions are counted, and the 4/A re-anchors the trade."""
    day1, day5 = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-07")
    original = dict(accession_number=_acc(1), filing_date="2024-03-01", transaction_date="2024-02-28", shares=1_000.0, shares_owned_after=11_000.0)
    rows = [
        row(**original, price_per_share=100.0),
        row(
            **{**original, "accession_number": _acc(2), "filing_date": "2024-03-07"},
            price_per_share=110.0,
            document_type="4/A",
            original_submission_date="2024-03-01",
        ),
    ]
    buy = _series(_build(frame(rows)), "f_ic_insider_buy_value_mcap_180d")
    before = buy.loc[day1 : day5 - pd.Timedelta(days=1)]
    after = buy.loc[day5 : day1 + pd.Timedelta(days=179)]
    aged = buy.loc[day1 + pd.Timedelta(days=180) :]

    assert np.allclose(before, 100_000.0 / MCAP, rtol=1e-6, atol=0.0)
    assert np.allclose(after, 110_000.0 / MCAP, rtol=1e-6, atol=0.0), f"days 5+ read {after.max() * MCAP:,.0f}, one trade is $110,000"
    assert aged.eq(0.0).all(), "the corrected trade ages out 180 days after the ORIGINAL filing (D-03)"
    print(
        f"SANITY: $100,000 on {len(before)} sessions before the 4/A, ${after.iloc[0] * MCAP:,.0f} on the {len(after)} sessions from it "
        f"to 180 days after the original, then 0 on {len(aged)} sessions: one trade, switched on the 4/A day, anchored on day 1."
    )


def test_an_identical_4a_adds_nothing_linked_or_unlinked():
    original = row(accession_number=_acc(1), filing_date="2024-03-01", transaction_date="2024-02-28")
    linked = {**original, "accession_number": _acc(2), "filing_date": "2024-03-07", "document_type": "4/A", "original_submission_date": "2024-03-01"}
    unlinked = {**linked, "original_submission_date": "2024-02-01"}
    alone = _build(frame([original]))
    for rows in ([original, linked], [original, unlinked]):
        pd.testing.assert_frame_equal(_build(frame(rows)), alone, check_exact=True)
    _, diag_linked = clean_transactions(frame([original, linked]))
    _, diag_unlinked = clean_transactions(frame([original, unlinked]))
    assert diag_linked["amendment_cells_identical"] == 1 and diag_unlinked["copy_rows_dropped"] == 1
    print(
        "SANITY: an identical 4/A leaves every insider column bit-identical, whether linked (1 identical cell dropped) "
        "or unlinked (1 later copy collapsed)."
    )


def test_an_unlinked_4a_is_a_normal_filing():
    rows = [
        row(accession_number=_acc(1), filing_date="2024-03-01", transaction_date="2024-02-28"),
        row(
            accession_number=_acc(2),
            owner_cik="0000000002",
            filing_date="2024-04-01",
            transaction_date="2024-03-28",
            shares=700.0,
            shares_owned_after=7_700.0,
        ),
    ]
    as_amendment = [rows[0], {**rows[1], "document_type": "4/A", "original_submission_date": "2024-03-01"}]
    pd.testing.assert_frame_equal(_build(frame(as_amendment)), _build(frame(rows)), check_exact=True)
    print("SANITY: a 4/A whose owner shares no stored original on its original-submission day is counted exactly as a plain Form 4.")


def test_a_partial_4a_keeps_the_cells_it_does_not_report():
    cleaned, diag = clean_transactions(frame(mixed_rows()))

    def cell(accession: str, day: str) -> pd.DataFrame:
        return cleaned[cleaned["accession_number"].eq(accession) & cleaned["transaction_date"].eq(pd.Timestamp(day).date())]

    x_orig, y_orig = cell(_acc(11), "2024-01-30"), cell(_acc(11), "2024-01-31")
    x_amend, z_amend = cell(_acc(12), "2024-01-30"), cell(_acc(12), "2024-02-02")
    assert len(x_orig) == 2 and x_orig["visible_until"].eq(pd.Timestamp("2024-02-08")).all()
    assert y_orig["visible_until"].isna().all() and y_orig["anchor"].eq(pd.Timestamp("2024-02-01")).all()
    assert (
        len(x_amend) == 1 and x_amend["anchor"].eq(pd.Timestamp("2024-02-01")).all() and x_amend["visible_from"].eq(pd.Timestamp("2024-02-08")).all()
    )
    assert z_amend["anchor"].eq(pd.Timestamp("2024-02-08")).all() and z_amend["visible_from"].eq(z_amend["anchor"]).all()
    assert diag["amendment_cells_partial"] >= 1 and diag["amendment_cells_new"] == 1
    print(
        f"SANITY: the partial 4/A closes cell X (2 rows) on 2024-02-08 and opens its 1 row anchored on 2024-02-01; cell Y stays open; "
        f"its new cell Z starts on its own day. Mixed fixture: {diag['amendments_linked']} linked, {diag['amendment_cells_superseded']} "
        f"superseded / {diag['amendment_cells_partial']} partial / {diag['amendment_cells_new']} new / {diag['amendment_cells_identical']} identical cells."
    )


def test_an_ambiguous_link_takes_the_original_sharing_most_cells():
    shared = dict(owner_cik="0000000003", filing_date="2024-02-01")
    rows = [
        row(accession_number=_acc(1), **shared, transaction_date="2024-01-29", shares=50.0, shares_owned_after=5_050.0),
        row(accession_number=_acc(2), **shared, transaction_date="2024-01-30", shares=60.0, shares_owned_after=5_060.0),
        row(
            accession_number=_acc(3),
            owner_cik="0000000003",
            filing_date="2024-02-09",
            transaction_date="2024-01-30",
            shares=61.0,
            shares_owned_after=5_061.0,
            document_type="4/A",
            original_submission_date="2024-02-01",
        ),
    ]
    cleaned, diag = clean_transactions(frame(rows))
    closed = cleaned.loc[cleaned["visible_until"].notna(), "accession_number"].tolist()
    assert diag["amendments_ambiguous"] == 1 and closed == [_acc(2)]
    print(f"SANITY: two same-day originals by the same owner; the 4/A links to {closed[0]}, the one sharing its cell (1 ambiguous link counted).")


def test_every_insider_field_matches_the_brute_force_as_of_reference_on_every_day():
    df_rows = frame(mixed_rows())
    new = _build(df_rows)
    ref = reference_panel(df_rows, IDX, _build)
    report = compare(new, ref, IDX, TICKERS)
    bad = {c: r for c, r in report.items() if r["mismatch"]}
    assert not bad, bad
    assert len(report) >= 10 and all(r["non_null"] > 0 for r in report.values())
    worst = max(r["max_abs_diff"] for r in report.values())
    print(
        f"SANITY: {len(report)} insider columns x {len(IDX)} days x {len(TICKERS)} tickers = {sum(r['cells'] for r in report.values()):,} cells "
        f"equal the day-by-day as-of rebuild (rtol 1e-12; max abs diff {worst:.2e}) across a partial 4/A, a new cell, identical and unlinked 4/As, "
        "a copy pair, a coincidence pair, a sell 4/A, a late 4/A, a 4/A chain and an unpriced sale priced by its 4/A."
    )


def _leak_full_sample(d: pd.DataFrame) -> pd.Series:
    """Leaky: rank against every record of the owner, later ones included."""
    df = d[["owner_cik", "value"]].assign(order=np.arange(len(d)))
    pairs = df.merge(df, on="owner_cik", suffixes=("", "_prior"))
    other = pairs["order_prior"] != pairs["order"]
    n = other.groupby(pairs["order"]).sum().to_numpy()
    le = (other & (pairs["value_prior"] <= pairs["value"])).groupby(pairs["order"]).sum().to_numpy()
    return pd.Series(le / np.where(n > 0, n, np.nan), index=d.index)


def _leak_visibility_blind(d: pd.DataFrame) -> pd.Series:
    """Leaky: earlier-anchored records counted before they are filed."""
    df = d[["owner_cik", "value"]].assign(order=np.arange(len(d)))
    pairs = df.merge(df, on="owner_cik", suffixes=("", "_prior"))
    prior = pairs["order_prior"] < pairs["order"]
    n = prior.groupby(pairs["order"]).sum().to_numpy()
    le = (prior & (pairs["value_prior"] <= pairs["value"])).groupby(pairs["order"]).sum().to_numpy()
    return pd.Series(le / np.where(n > 0, n, np.nan), index=d.index)


@pytest.mark.parametrize("leak", [_leak_full_sample, _leak_visibility_blind], ids=["full-sample", "visibility-blind"])
def test_a_leaky_owner_surprise_is_caught_by_the_reference(monkeypatch: pytest.MonkeyPatch, leak):
    df_rows = frame(mixed_rows())
    ref = reference_panel(df_rows, IDX, _build)
    monkeypatch.setattr(insider_features, "_visible_prior_rank", leak)
    report = compare(_build(df_rows), ref, IDX, TICKERS)
    column = "f_ic_insider_owner_surprise_120d"
    assert report[column]["mismatch"] > 0
    print(
        f"SANITY: the {leak.__name__.removeprefix('_leak_')} owner surprise differs from the as-of reference on {report[column]['mismatch']} cells; the honest one on 0."
    )


def test_owner_surprise_prior_set_is_frozen_at_the_visible_day():
    """A purchase is ranked against the owner's records visible on its own visible day; a later 4/A of a prior never re-ranks it."""
    rows = [
        row(accession_number=_acc(1), filing_date="2024-02-01", transaction_date="2024-01-30", shares=500.0, shares_owned_after=5_500.0),
        row(accession_number=_acc(2), filing_date="2024-02-05", transaction_date="2024-02-02", shares=600.0, shares_owned_after=6_100.0),
        row(accession_number=_acc(3), filing_date="2024-02-08", transaction_date="2024-01-30", shares=900.0, shares_owned_after=5_500.0)
        | {"document_type": "4/A", "original_submission_date": "2024-02-01"},
    ]
    surprise = _series(_build(frame(rows)), "f_ic_insider_owner_surprise_120d")
    # the 2024-02-05 purchase ($60k) is above the $50k it saw; the 4/A's $90k never re-ranks it
    assert surprise.loc["2024-02-05"] == pytest.approx(1.0, abs=1e-12)
    assert surprise.loc["2024-03-01"] == pytest.approx(1.0, abs=1e-12)
    print(
        f"SANITY: the second purchase scores {surprise.loc['2024-02-05']:.2f} against the $50k prior it saw on its day, and still "
        f"{surprise.loc['2024-03-01']:.2f} after the prior is restated to $90k: no later record re-ranks it."
    )


def test_a_restatement_that_leaves_a_leg_keeps_the_leg_observed():
    """`has event` comes from the positive leg: a CEO purchase restated as a director's reads 0 on the CEO leg, not NaN."""
    rows = [
        row(
            accession_number=_acc(1), filing_date="2024-03-01", transaction_date="2024-02-28", officer_title="Chief Executive Officer", is_officer=1.0
        ),
        row(accession_number=_acc(2), filing_date="2024-03-07", transaction_date="2024-02-28", officer_title="Director", is_officer=0.0)
        | {"document_type": "4/A", "original_submission_date": "2024-03-01", "shares_owned_after": 11_001.0},
    ]
    ceo = _series(_build(frame(rows)), "f_ic_insider_ceo_buy_mcap_180d")
    assert ceo.loc["2024-03-06"] > 0 and ceo.loc["2024-03-07":].eq(0.0).all() and ceo.loc[:"2024-02-29"].isna().all()
    print(
        f"SANITY: the CEO leg reads {ceo.loc['2024-03-06']:.6f} until the 4/A, then exactly 0 (observed, emptied), and NaN before the first filing."
    )


def test_the_sink_gets_each_trade_once_at_its_first_disclosure():
    sink = ConditioningSink()
    _build(frame(mixed_rows()), sink=sink)
    events = sink.events["insider"]
    aaa = events[events["ticker"].eq("AAA")]
    on_a1_day = aaa[aaa["date"].eq(pd.Timestamp("2024-02-08"))]
    assert len(on_a1_day) == 1 and on_a1_day["shares"].iloc[0] == 150.0  # only the new cell Z
    assert aaa["date"].eq(pd.Timestamp("2024-06-07")).sum() == 0  # the later copy
    print(f"SANITY: {len(events)} sink events; the 4/A restatement of cell X adds none, its new cell Z adds one on 2024-02-08, the later copy none.")
