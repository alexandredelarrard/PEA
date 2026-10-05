"""AC-010: one trade reported in several accessions counts once, from its earliest filing.

Known-truth point-in-time fixtures, and the real 2024q1-2026q2 zip snapshot of the run dir
(skipped when absent) for the copy groups and the dollars they remove.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals import insider_quality as iq
from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.institutionals.insider_quality import OPEN_MARKET_CODES, clean_transactions, common_stock_mask
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import compare, frame, reference_panel, row, sentinel_rows, snapshot_path
from tests.fixtures.insider_zip import fixture_accessions, xml_frame, zip_frame, zip_path

IDX = pd.bdate_range("2024-01-01", "2024-12-31")
MCAP = 1_000_000.0 * 100.0
KEY = ["ticker", "transaction_date", "transaction_code", "shares", "price_per_share", "shares_owned_after"]


def _build(df_rows: pd.DataFrame) -> pd.DataFrame:
    close = pd.DataFrame(100.0, index=IDX, columns=["AAA"])
    fh = pd.DataFrame([{"ticker": "AAA", "as_of": pd.Timestamp("2023-01-01"), "sharesOutstanding": 1e6, "sharesOutstandingPit": 1e6}])
    return build_insider_feature_panel(
        make_frames(IDX, {"AAA": {"AAA": 1.0}}, close_split=close), df_rows, shares_out_history=fh, complete_through=IDX.max()
    )


def _copy_pair() -> list[dict]:
    position = dict(transaction_date="2024-02-28", shares=1_000.0, price_per_share=100.0, shares_owned_after=50_000.0, direct_indirect="I")
    return [
        row(accession_number="0000000000-24-000001", owner_cik="0000000008", filing_date="2024-03-01", **position),
        row(accession_number="0000000000-24-000002", owner_cik="0000000009", filing_date="2024-03-07", **position),
    ]


def test_a_copy_pair_counts_once_from_the_first_filing_on_every_day():
    """RED on base: the second accession doubles the value and the buyer count from its filing day."""
    panel = _build(frame(_copy_pair())).set_index("date")
    value = panel["f_ic_insider_buy_value_mcap_180d"].reindex(IDX)
    buyers = panel["f_ic_insider_distinct_buyers_120d"].reindex(IDX)
    live = value.loc["2024-03-01":"2024-08-27"]
    assert np.allclose(live, 100_000.0 / MCAP, rtol=1e-6, atol=0.0), f"peak {live.max() * MCAP:,.0f}"
    assert buyers.loc["2024-03-01":"2024-06-28"].eq(1.0).all()
    between = value.loc["2024-03-01":"2024-03-06"]
    print(
        f"SANITY: the trade reads ${between.iloc[0] * MCAP:,.0f} between the two filings and ${live.loc['2024-03-07'] * MCAP:,.0f} after the "
        f"second copy, with {buyers.max():.0f} distinct buyer: one trade, from 2024-03-01."
    )


def test_the_copy_pair_matches_the_as_of_reference():
    df_rows = frame(_copy_pair() + sentinel_rows())
    report = compare(_build(df_rows), reference_panel(df_rows, IDX, _build), IDX, ["AAA"])
    assert not any(r["mismatch"] for r in report.values()), report
    print(f"SANITY: {len(report)} insider columns equal the day-by-day as-of rebuild on all {len(IDX)} days for the copy pair.")


def test_rows_kept_apart_by_holding_nan_or_accession():
    shared = dict(transaction_date="2024-02-28", shares=100.0, price_per_share=106.0)
    rows = [
        row(accession_number="0000000000-24-000001", owner_cik="0000000006", filing_date="2024-03-01", shares_owned_after=1_100.0, **shared),
        row(accession_number="0000000000-24-000002", owner_cik="0000000007", filing_date="2024-03-02", shares_owned_after=2_100.0, **shared),
        row(accession_number="0000000000-24-000003", owner_cik="0000000001", filing_date="2024-03-04", shares_owned_after=np.nan, **shared),
        row(accession_number="0000000000-24-000004", owner_cik="0000000002", filing_date="2024-03-05", shares_owned_after=np.nan, **shared),
        row(accession_number="0000000000-24-000005", owner_cik="0000000003", filing_date="2024-03-06", shares_owned_after=7_000.0, **shared),
        row(
            accession_number="0000000000-24-000005",
            owner_cik="0000000003",
            filing_date="2024-03-06",
            shares_owned_after=7_000.0,
            **shared,
            row_sequence=2,
        ),
    ]
    cleaned, diag = clean_transactions(frame(rows))
    assert len(cleaned) == len(rows) and diag["copy_groups"] == 0
    print(
        "SANITY: 6 rows of one size and price stay 6 trades: different post-trade holdings (coincidence), "
        "a NaN holding (never matches) and two equal lots inside one accession."
    )


def test_the_kept_copy_carries_every_copys_owners():
    first, second = _copy_pair()
    second = {**second, "owner_ciks": "0000000009,0000000010", "n_reporting_owners": 2}
    cleaned, diag = clean_transactions(frame([first, second]))
    kept = cleaned.iloc[0]
    assert len(cleaned) == 1 and kept["owner_cik"] == "0000000008"
    assert kept["owner_ciks"] == "0000000008,0000000009,0000000010" and kept["n_reporting_owners"] == 3
    print(
        f"SANITY: the earliest copy keeps primary owner {kept['owner_cik']} and the union owner_ciks {kept['owner_ciks']}; {diag['copy_rows_dropped']} copy dropped."
    )


def _snapshot() -> pd.DataFrame:
    path = snapshot_path()
    if path is None:
        pytest.skip("run-dir snapshot zip_2024q1_2026q2.parquet is absent")
    df = pd.read_parquet(path)
    for column in ("shares", "price_per_share", "value_usd", "shares_owned_after", "is_director", "is_officer", "is_ten_pct_owner", "is_10b5_1"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("filing_date", "transaction_date"):
        df[column] = pd.to_datetime(df[column]).dt.date
    df["owner_ciks"] = df["owner_cik"]
    df["original_submission_date"] = None
    return df.reset_index(drop=True)


def test_real_copy_groups_contribute_once_and_coincidences_survive():
    """RED on base: no collapse exists, so the later copies reach the features with their dollars."""
    df = _snapshot()
    scoped = df[
        df["security_type"].str.lower().eq("nonderiv")
        & common_stock_mask(df["security_title"])
        & df["transaction_code"].str.upper().str.strip().isin(OPEN_MARKET_CODES)
        & df["shares"].notna()
        & df["filing_date"].notna()
    ].copy()
    for column in ("shares", "price_per_share", "shares_owned_after"):
        scoped[column] = scoped[column].round(2)
    keyed = scoped.dropna(subset=KEY)
    n_acc = keyed.groupby(KEY)["accession_number"].transform("nunique")
    grouped = keyed[n_acc > 1].sort_values(["filing_date", "accession_number"])
    keeper = grouped.groupby(KEY)["accession_number"].transform("first")
    later = grouped.index[grouped["accession_number"].ne(keeper)]
    five = KEY[:-1]
    loose = keyed.groupby(five)["accession_number"].transform("nunique") > 1
    coincidence = keyed[loose & ~keyed.index.isin(grouped.index)]

    cleaned, diag = clean_transactions(df)
    survivors = cleaned.index.intersection(later)
    removed = float(df.loc[later, "value_usd"].sum())
    assert survivors.empty, f"{len(survivors)} later copies reach the features"
    assert diag["copy_rows_dropped"] == len(later) and diag["copy_groups"] == grouped.groupby(KEY).ngroups
    assert diag["copy_value_removed"] == pytest.approx(removed, rel=1e-12)
    assert coincidence.index.isin(cleaned.index).all()
    raw = float(scoped["value_usd"].sum())
    print(
        f"SANITY: {len(scoped):,} scoped P/S rows; {diag['copy_groups']:,} copy groups, {len(later):,} later copies dropped carrying "
        f"${removed / 1e9:.2f}bn of ${raw / 1e9:.1f}bn as filed; {coincidence.groupby(five).ngroups:,} coincidence groups "
        f"({len(coincidence):,} rows) all kept."
    )


def _zip_and_edgar_copies(edgar: dict, zipped: dict) -> list[dict]:
    """Joint reporters filing one purchase the same day: one copy stored from EDGAR (values as filed),
    one from the zip (the same values rounded half-up to 2 decimals)."""
    trade = dict(transaction_date="2024-02-28", filing_date="2024-03-01", shares=1_000.0, price_per_share=100.0, shares_owned_after=50_000.0)
    return [
        row(accession_number="0000000000-24-000001", owner_cik="0000000008", source="edgar", **{**trade, **edgar}),
        row(accession_number="0000000000-24-000002", owner_cik="0000000009", source="zip", **{**trade, **zipped}),
    ]


@pytest.mark.parametrize(
    ("edgar", "zipped"),
    [
        ({"shares": 105.085}, {"shares": 105.09}),  # 105.085 is 105.08499... in binary
        ({"price_per_share": 20.125}, {"price_per_share": 20.13}),  # an exact binary half
        ({"shares_owned_after": 483_761.185}, {"shares_owned_after": 483_761.19}),
        ({"shares": 105.084}, {"shares": 105.08}),
    ],
)
def test_a_zip_copy_of_an_edgar_trade_is_one_trade_at_a_half_cent(edgar: dict, zipped: dict):
    """R-02: the copy key rounds as the zip does (half away from zero), so the zip's copy of an EDGAR value collapses."""
    cleaned, diag = clean_transactions(frame(_zip_and_edgar_copies(edgar, zipped)))
    assert diag["copy_groups"] == 1 and len(cleaned) == 1 and cleaned.iloc[0]["source"] == "edgar"
    df_rows = frame(_zip_and_edgar_copies(edgar, zipped) + sentinel_rows())
    report = compare(_build(df_rows), reference_panel(df_rows, IDX, _build), IDX, ["AAA"])
    assert not any(r["mismatch"] for r in report.values()), report
    print(f"SANITY: EDGAR {edgar} and its zip copy {zipped} are one trade (copy_groups 1, the EDGAR copy kept); the as-of reference agrees.")


def test_values_apart_beyond_the_half_stay_two_trades():
    """Only a half rounds up: 105.0849 (zip 105.08) and 105.085 (zip 105.09) stay distinct trades."""
    df_rows = frame(_zip_and_edgar_copies({"shares": 105.0849}, {"shares": 105.09}))
    cleaned, diag = clean_transactions(df_rows)
    assert diag["copy_groups"] == 0 and len(cleaned) == 2
    print("SANITY: EDGAR 105.0849 and zip 105.09 keep two trades: the key rounds 105.0849 to 105.08, as the zip would.")


def _key_values(df: pd.DataFrame) -> pd.DataFrame:
    """The rounded `shares`, `price` and `owned_after` copy-key fields of typed insider rows."""
    df_key = df.assign(
        day=pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]"), code=df["transaction_code"], shares_n=pd.to_numeric(df["shares"])
    )
    return iq._copy_key(df_key, df_key["price_per_share"])[["shares", "price", "owned_after"]]


def test_every_real_edgar_value_keys_like_its_zip_copy():
    """R-02 on the 28 real fixtures: the key of each XML value equals the key of the zip's rounding of it."""
    path = zip_path()
    if path is None:
        pytest.skip("cached 2026q2 insider zip is absent")
    df_xml = pd.concat([xml_frame(accession) for accession in fixture_accessions()], ignore_index=True)
    df_pair = df_xml.merge(zip_frame(path), on=["accession_number", "security_type", "row_sequence"], suffixes=("", "_zip"))
    df_zip = df_pair[[c for c in df_pair.columns if not c.endswith("_zip")]].assign(
        **{c: df_pair[f"{c}_zip"] for c in ("shares", "price_per_share", "shares_owned_after")}
    )
    xml_key, zip_key = _key_values(df_pair), _key_values(df_zip)
    same = (xml_key == zip_key) | (xml_key.isna() & zip_key.isna())
    finer = sum(int((df_pair[c].notna() & df_pair[c].ne(df_pair[c].round(2))).sum()) for c in ("shares", "price_per_share", "shares_owned_after"))
    assert same.all().all(), df_pair.loc[
        ~same.all(axis=1), ["accession_number", "shares", "shares_zip", "shares_owned_after", "shares_owned_after_zip"]
    ]
    print(f"SANITY: {len(df_pair)} real rows x 3 key fields: every EDGAR value keys like its zip copy ({finer} values carry more than 2 decimals).")
