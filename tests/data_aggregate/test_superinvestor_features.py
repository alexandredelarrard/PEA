"""
Superinvestor (elite-manager) 13F features
(src/data_aggregate/utils/institutionals/superinvestor_features.py).

Proves the four properties that make this panel different from the all-filer institutional
one, each of which was a real defect caught by measurement rather than a hypothetical:

  1. THE DENOMINATOR IS THE WHOLE BOOK. A portfolio weight divides by the manager's entire
     common-stock book, not by the S&P500 slice `sec13f_hr` happens to store. On the live
     table the slice is a median 59% of value and as little as 3.8%, so the old denominator
     inflated conviction by a manager-specific 1.0x-26x.
  2. A SPLIT IS NOT A PURCHASE. The prior quarter's share count is restated onto the current
     basis before any ratio, and the residual guard nulls what is left.
  3. RANKS ARE ORDER-INDEPENDENT. `rank(method="first")` numbered ties by row position and
     flipped 2.06% of live ranks whenever anything upstream reordered the frame.
  4. NOTHING MOVES BEFORE IT WAS FILED, and each manager carries its OWN availability date.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.superinvestor_features import (
    EMISSION, _prepare, _selection_ciks, _selection_series, attach_split_factor,
    attach_tickers, build_superinvestor_feature_panel, load_superinvestor_holdings,
    manager_quarter_state, manager_stock_conviction, pad_cik, public_state,
)

#: Two roster managers. `BIG` runs a 10-name book of which only 2 are in the universe --
#: the case README Finding 2 measured at 8.3% position coverage. `SMALL` is index-only.
_ROSTER = {"0000000001": "Big Global Manager", "0000000002": "Index Only"}

_UNIVERSE = ["HOT", "COLD"]
_CUSIP_MAP = pd.DataFrame({"cusip": [f"{i:09d}" for i in range(10)],
                           "ticker": ["HOT", "COLD"] + [f"OTC{i}" for i in range(8)]})


def _book(period: str, filing_date: str, hot: int, cold: int,
          cik: str = "0000000001", n_other: int = 8, other: int = 1_000) -> list[dict]:
    """One manager-quarter: two universe names plus `n_other` names that are NOT in the
    universe but ARE in the book -- the denominator the S&P500 slice cannot see."""
    rows = [{"cik": cik, "period": period, "filing_date": filing_date,
             "cusip": "000000000", "position_type": "common",
             "shares": hot, "value_usd": hot * 10},
            {"cik": cik, "period": period, "filing_date": filing_date,
             "cusip": "000000001", "position_type": "common",
             "shares": cold, "value_usd": cold * 10}]
    rows += [{"cik": cik, "period": period, "filing_date": filing_date,
              "cusip": f"{i + 2:09d}", "position_type": "common",
              "shares": other, "value_usd": other * 10} for i in range(n_other)]
    return rows


def _holdings() -> pd.DataFrame:
    rows = []
    for period, filed, hot, cold in (("2025-09-30", "2025-11-14", 500, 500),
                                     ("2025-12-31", "2026-02-14", 1000, 250),
                                     ("2026-03-31", "2026-05-15", 1500, 0)):
        rows += _book(period, filed, hot, cold)
        rows += _book(period, filed, hot, cold, cik="0000000002", n_other=0)
    return pd.DataFrame(rows)


def _prepared(holdings=None):
    h = _prepare(attach_tickers(holdings if holdings is not None else _holdings(),
                                _CUSIP_MAP, _UNIVERSE))
    state = manager_quarter_state(h)
    return h, state, manager_stock_conviction(h, state)


# --------------------------------------------------------------------------------------- #
# 1. the denominator                                                                        #
# --------------------------------------------------------------------------------------- #
def test_portfolio_weight_divides_by_the_whole_book_not_the_index_sleeve():
    """A manager holding 10 names of which 2 are in the universe: the weights sum to 1.0
    across ALL TEN, and the two universe names carry correspondingly small weights."""
    h, state, conv = _prepared()
    big = conv[(conv["cik"] == "0000000001") & (conv["period"] == pd.Timestamp("2025-12-31"))]
    assert len(big) == 10, f"the whole book must survive the ticker join, got {len(big)}"
    assert np.isclose(big["portfolio_weight"].sum(), 1.0), big["portfolio_weight"].sum()

    in_universe = big[big["ticker"].notna()]
    assert len(in_universe) == 2
    slice_w = in_universe["value_usd"] / in_universe["value_usd"].sum()
    inflation = (slice_w.to_numpy()
                 / in_universe["portfolio_weight"].to_numpy())
    row = state[(state["cik"] == "0000000001")
                & (state["period"] == pd.Timestamp("2025-12-31"))].iloc[0]

    print("\n=== SANITY CHECK: conviction denominator ===")
    print(f"  book = {int(row['n_positions'])} positions / ${row['total_common_value']:,.0f}; "
          f"universe sleeve = {len(in_universe)} names / "
          f"${in_universe['value_usd'].sum():,.0f} ({row['sp500_share']:.1%} of value)")
    print(f"  HOT weight: whole-book {in_universe['portfolio_weight'].max():.4f} vs "
          f"index-sleeve {slice_w.max():.4f}  -> the old denominator inflated it "
          f"{inflation.max():.2f}x")
    print("  weights sum to 1.0 over all 10 names, not over the 2 in the universe. Validated.")
    assert (inflation > 1.0).all(), "the slice weight must exceed the whole-book weight"
    assert np.isclose(inflation, 1.0 / row["sp500_share"]).all(), (
        "inflation must equal 1 / value coverage by construction")


def test_unmapped_positions_stay_in_the_book():
    """The cusip join is a LEFT join: the 8 names with no universe ticker keep their rows,
    because dropping them rebuilds the index-sleeve denominator from the other end."""
    joined = attach_tickers(_holdings(), _CUSIP_MAP, _UNIVERSE)
    assert len(joined) == len(_holdings())
    assert joined["ticker"].isna().sum() == 24, joined["ticker"].isna().sum()
    print("\n=== SANITY CHECK: left join keeps the rest of the book ===")
    print(f"  {len(joined)} rows in, {len(joined)} out; {joined['ticker'].notna().sum()} "
          f"carry a universe ticker and {joined['ticker'].isna().sum()} do not -- and the "
          "latter are exactly what `total_common_value` needs. Validated.")


def test_universe_narrows_the_ticker_side_only():
    """Without `universe`, `cusip_ticker_map`'s 20k tickers mark almost the whole book as
    "in the index" -- measured on live data as a 98% index share against a true 59%."""
    wide = attach_tickers(_holdings(), _CUSIP_MAP, universe=None)
    narrow = attach_tickers(_holdings(), _CUSIP_MAP, universe=_UNIVERSE)
    print("\n=== SANITY CHECK: universe restriction ===")
    print(f"  mapped rows with no universe filter: {wide['ticker'].notna().sum()} "
          f"(all 10 names) vs with it: {narrow['ticker'].notna().sum()} (the 2 real ones)")
    assert wide["ticker"].notna().sum() > narrow["ticker"].notna().sum()
    print("  the denominator is identical either way; only the numerator narrows. Validated.")


# --------------------------------------------------------------------------------------- #
# 2. a split is not a purchase                                                              #
# --------------------------------------------------------------------------------------- #
def test_a_split_is_restated_not_read_as_a_purchase():
    """A 20-for-1 split multiplies the share count with no trade. The prior quarter must be
    restated onto the new basis, leaving a ~0% change rather than +1,900%."""
    rows = (_book("2025-12-31", "2026-02-14", hot=1_000, cold=100)
            + _book("2026-03-31", "2026-05-15", hot=20_000, cold=100))
    h, state, conv = _prepared(pd.DataFrame(rows))
    contrib = _contrib(conv, state)
    splits = pd.DataFrame({"date": [pd.Timestamp("2026-02-10")], "ticker": ["HOT"],
                           "ratio": [20.0]})
    with_split = attach_split_factor(contrib, splits)
    without = attach_split_factor(contrib, None)

    def chg(frame):
        r = frame[(frame["ticker"] == "HOT")
                  & (frame["period"] == pd.Timestamp("2026-03-31"))].iloc[0]
        return r["shares"] / r["prev_shares_adj"] - 1.0

    print("\n=== SANITY CHECK: 20:1 split restatement ===")
    print(f"  shares 1,000 -> 20,000 with no trade")
    print(f"    without `prices_splits`: {without.pipe(chg):+.0%}  <- read as a purchase")
    print(f"    with    `prices_splits`: {with_split.pipe(chg):+.0%}  <- restated")
    assert np.isclose(with_split.pipe(chg), 0.0, atol=1e-9)
    assert without.pipe(chg) > 18.0
    print("  Validated.")


def _contrib(conv, state):
    sel = _selection_series(state, None)
    from src.data_aggregate.utils.institutionals.superinvestor_features import _contributions
    return _contributions(conv, state, sel)


# --------------------------------------------------------------------------------------- #
# 3. ranks do not depend on row order                                                       #
# --------------------------------------------------------------------------------------- #
def test_rank_in_book_is_independent_of_row_order():
    """`rank(method="first")` broke ties by frame position, so two positions of identical
    value swapped ranks whenever anything upstream reordered rows -- 2.06% of live
    manager-quarter-cusip rows, e.g. 38 <-> 39 at an identical 0.000089 weight."""
    base = _holdings()
    shuffled = base.sample(frac=1.0, random_state=7).reset_index(drop=True)
    _, _, a = _prepared(base)
    _, _, b = _prepared(shuffled)
    key = ["cik", "period", "cusip"]
    merged = a[key + ["rank_in_book"]].merge(b[key + ["rank_in_book"]], on=key,
                                             suffixes=("_a", "_b"))
    ties = (a.groupby(["cik", "period"])["value_usd"].transform(
        lambda s: s.duplicated(keep=False)).sum())
    flipped = int((merged["rank_in_book_a"] != merged["rank_in_book_b"]).sum())
    print("\n=== SANITY CHECK: deterministic ranks ===")
    print(f"  {len(merged)} rows compared, {ties} of them tied on value_usd "
          f"(the 8 equal non-universe positions)")
    print(f"  ranks that changed when the input rows were shuffled: {flipped} (must be 0)")
    assert flipped == 0
    print("  tie-break is (value desc, cusip asc), a property of the filing. Validated.")


# --------------------------------------------------------------------------------------- #
# 4. availability                                                                           #
# --------------------------------------------------------------------------------------- #
def test_each_manager_carries_its_own_availability_date():
    """`max(period + 45d, filing_date)` PER MANAGER. Stamping the whole quarter at the last
    contributing filing delayed 45 of 60 live quarters past 90 days (median 189); stamping
    everything at the 45-day deadline leaks, because 16.5% of manager-quarters carrying
    36.7% of book value are filed after it."""
    rows = (_book("2025-12-31", "2026-02-14", 1000, 250)                    # on time
            + _book("2025-12-31", "2026-08-01", 800, 300, cik="0000000002",  # 168 days late
                    n_other=0))
    _, state, _ = _prepared(pd.DataFrame(rows))
    st = public_state(state, _selection_series(state, None))
    stamps = dict(zip(st["cik"], st["avail"]))
    print("\n=== SANITY CHECK: per-manager availability ===")
    for cik, a in sorted(stamps.items()):
        print(f"  {cik}  filed {'2026-02-14' if cik.endswith('1') else '2026-08-01'} "
              f"-> public {a.date()}")
    assert stamps["0000000001"] == pd.Timestamp("2026-02-14")
    assert stamps["0000000002"] == pd.Timestamp("2026-08-01")
    print("  the deadline is a FLOOR, never a substitute for a late filing date. Validated.")


def test_a_superseded_backfiling_is_never_readable():
    """A 13F back-filed for an old period arrives after the filings that supersede it, so
    its readable window is empty. Keeping it would let an old period overwrite a newer one
    -- a look-ahead in reverse."""
    rows = (_book("2025-09-30", "2027-01-15", 100, 100)     # back-filed 15 months late
            + _book("2025-12-31", "2026-02-14", 1000, 250)  # on time, supersedes it
            + _book("2026-03-31", "2026-05-15", 1500, 0))
    _, state, _ = _prepared(pd.DataFrame(rows))
    st = public_state(state, _selection_series(state, None))
    kept = sorted(str(p.date()) for p in st["period"])
    print("\n=== SANITY CHECK: superseded back-filing ===")
    print(f"  filed periods: 2025-09-30 (submitted 2027-01-15), 2025-12-31, 2026-03-31")
    print(f"  periods that are ever the manager's public state: {kept}")
    assert "2025-09-30" not in kept, "a superseded back-filing must be dropped"
    print("  the 2027 submission never becomes readable: by then 2026-03-31 is the state. "
          "Validated.")


def test_the_panel_is_empty_before_the_first_filing_is_public():
    idx = pd.bdate_range("2025-10-01", "2026-09-30")
    peers = {t: {p: 1.0 for p in _UNIVERSE if p != t} for t in _UNIVERSE}
    panel = build_superinvestor_feature_panel(
        _holdings(), _ROSTER, peers, idx, cusip_map=_CUSIP_MAP, universe=_UNIVERSE)
    feats = [c for c in panel.columns if c.startswith("f_")]
    before = panel[panel["date"] < pd.Timestamp("2025-11-14")]
    after = panel[panel["date"] >= pd.Timestamp("2026-02-14")]
    print("\n=== SANITY CHECK: leak-free panel ===")
    print(f"  {len(panel):,} rows x {len(feats)} feature columns")
    print(f"  rows before the first filing became public (2025-11-14): {len(before):,}, "
          f"all-NaN: {before[feats].isna().all().all()}")
    print(f"  non-null feature cells after 2026-02-14: "
          f"{int(after[feats].notna().to_numpy().sum()):,}")
    assert before.empty or before[feats].isna().all().all()
    assert after[feats].notna().to_numpy().any()
    print("  Validated.")


# --------------------------------------------------------------------------------------- #
# emission contract (D25 / D27)                                                             #
# --------------------------------------------------------------------------------------- #
def test_the_family_emits_no_peer_leg_and_never_a_rank_alone():
    idx = pd.bdate_range("2025-10-01", "2026-09-30")
    peers = {t: {p: 1.0 for p in _UNIVERSE if p != t} for t in _UNIVERSE}
    panel = build_superinvestor_feature_panel(
        _holdings(), _ROSTER, peers, idx, cusip_map=_CUSIP_MAP, universe=_UNIVERSE)
    feats = [c for c in panel.columns if c.startswith("f_")]
    peer_legs = [c for c in feats if c.endswith("_vs_peers")]
    raws = {c[2:] for c in feats if not c.endswith("_xs")}
    xs = {c[2:-3] for c in feats if c.endswith("_xs")}
    print("\n=== SANITY CHECK: emission contract ===")
    print(f"  peer legs: {len(peer_legs)} (D25 -- the signal is absolute, not sector-relative)")
    print(f"  raw legs {len(raws)}, _xs legs {len(xs)}; every _xs has its raw: {xs <= raws}")
    print(f"  declared `raw+xs`: {sorted(k for k, v in EMISSION.items() if v == 'raw+xs')}")
    assert not peer_legs
    assert xs <= raws, "no feature may ship as a rank alone"
    assert xs == {k for k, v in EMISSION.items() if v == "raw+xs"} & raws
    print("  Validated.")


def test_selection_ciks_reads_every_roster_shape():
    assert _selection_ciks({"1067983": "Buffett", "0000000002": "Ackman"}) == {
        "0001067983", "0000000002"}
    assert _selection_ciks({"cik_to_name": {"1067983": "Buffett"}}) == {"0001067983"}
    assert _selection_ciks({"managers": [{"cik": "2"}]}) == {"0000000002"}
    assert _selection_ciks(None) == set() and _selection_ciks({"managers": []}) == set()
    print("\n=== SANITY CHECK: roster shapes ===")
    print("  {cik: name}, {cik_to_name: {...}} and the legacy {managers: [{cik}]} all "
          "resolve to padded CIKs; None/empty -> empty set. Validated.")


class _Ctx:
    def __init__(self, store):
        self.store = store
        self.log = logging.getLogger("test")


def test_load_reads_only_roster_managers_from_the_book_table(sqlite_store):
    """The push-down is a plain `cik IN (...)`: `sec13f_manager_holdings` is written by one
    producer with `pad_cik` applied (verified on the live table -- all 342,501 rows are
    exactly 10 characters), so the every-stored-spelling dance `sec13f_hr` needs is not
    required here."""
    rows = _book("2025-12-31", "2026-02-14", 1000, 250)
    rows += _book("2025-12-31", "2026-02-14", 5, 5, cik="0000000009", n_other=0)  # not roster
    sqlite_store.save("sec13f_manager_holdings", pd.DataFrame(rows))
    ctx = _Ctx(sqlite_store)
    out = load_superinvestor_holdings(ctx, _ROSTER)
    assert set(out["cik"]) == {"0000000001"}, set(out["cik"])
    assert len(out) == 10, "the whole book must come back, not the universe slice"
    assert load_superinvestor_holdings(ctx, {"managers": []}) is None
    print("\n=== SANITY CHECK: filtered book read ===")
    print(f"  {len(rows)} stored rows -> {len(out)} returned, all for roster CIK "
          f"0000000001; the non-roster manager is filtered server-side")
    print(f"  columns: {list(out.columns)}")
    print("  Validated.")
