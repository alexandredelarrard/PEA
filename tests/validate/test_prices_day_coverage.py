"""
test_prices_day_coverage.py  (tests/validate/test_prices_day_coverage.py)
------------------------------------------------------------------
Invariant 4 of the price validator: every trading day in the table's span must carry the
whole universe.

Why it had to be a NEW invariant rather than a tweak to an existing one: invariants 1-3 all
score ROWS THAT EXIST, so a missing row is invisible to every one of them. `prices` held 45 of
491 tickers on 2026-08-28 with 491 on both neighbouring sessions, and all three passed that
day -- the 45 rows present were perfectly adjusted. The damage landed downstream, where
`cube_part_momentum` ranked 13 of its 28 features over those 45 names.
"""
import pandas as pd

from src.validate.prices import (
    DAY_COVERAGE_FLOOR, DAY_COVERAGE_WINDOW, PricesReport, invariant_day_coverage)

DATES = pd.bdate_range("2024-01-01", periods=120)


class _Ctx:
    """Context stand-in exposing just the `store.iter_load` the invariant uses."""

    def __init__(self, frame: pd.DataFrame, chunks: int = 3):
        self._frame, self._chunks = frame, chunks
        self.store = self

    def iter_load(self, table, *, columns=None, where=None, chunksize=None):
        # deliberately chunked and NOT date-ordered: the invariant must accumulate per-day
        # counts across chunk boundaries rather than assume one chunk per day.
        shuffled = self._frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
        size = max(1, len(shuffled) // self._chunks)
        for i in range(0, len(shuffled), size):
            yield shuffled.iloc[i:i + size][list(columns)]


def _grid(n_names: int = 100, thin: dict[pd.Timestamp, int] | None = None) -> pd.DataFrame:
    rows = []
    for d in DATES:
        live = (thin or {}).get(d, n_names)
        rows += [{"ticker": f"T{i:03d}", "date": d, "close_split": 10.0}
                 for i in range(live)]
    return pd.DataFrame(rows)


def test_full_coverage_passes():
    res = invariant_day_coverage(_Ctx(_grid()))
    print(f"  {res.summary()}")
    assert res.failed == 0 and not res.detail
    assert res.share == 0.0
    assert res.tickers == 100
    assert "No failures" in PricesReport([res]).to_markdown()


def test_the_live_hole_shape_is_caught():
    """45 of 491, with the full universe on both neighbours."""
    bad = DATES[80]
    res = invariant_day_coverage(_Ctx(_grid(thin={bad: 9})))
    print(f"  {res.summary()}")
    print(f"  detail: {res.detail}")
    assert len(res.detail) == 1
    d = res.detail[0]
    assert d["date"] == str(bad.date())
    assert (d["tickers"], d["expected"], d["missing"]) == (9, 100, 91)
    assert res.failed == 91
    # ticker-days, so `share` is comparable with the other three invariants
    assert 0 < res.share < 0.01


def test_growing_universe_is_not_flagged():
    """The reason the reference is a TRAILING median and not the table's own.

    Measured against the global median on the live table, the floor flagged 2026-08-28 AND 20
    legitimate 2005 dates, where `prices` genuinely carried 383-385 of an eventual 491."""
    thin = {d: min(100, 40 + i) for i, d in enumerate(DATES)}      # 40 -> 100, monotone
    res = invariant_day_coverage(_Ctx(_grid(thin=thin)))
    print(f"  ramp 40 -> 100 names over {len(DATES)} sessions -> "
          f"{len(res.detail)} short day(s)")
    assert not res.detail, f"structural growth flagged: {res.detail}"


def test_a_permanently_smaller_universe_is_not_flagged():
    """A table that simply holds fewer names is not defective; it is judged against itself."""
    res = invariant_day_coverage(_Ctx(_grid(n_names=12)))
    assert not res.detail
    assert res.tickers == 12


def test_floor_binds_where_the_constant_says():
    just_under = int(DAY_COVERAGE_FLOOR * 100) - 1        # 59
    just_over = int(DAY_COVERAGE_FLOOR * 100) + 1         # 61
    lo = invariant_day_coverage(_Ctx(_grid(thin={DATES[80]: just_under})))
    hi = invariant_day_coverage(_Ctx(_grid(thin={DATES[80]: just_over})))
    print(f"  {just_under}/100 -> {len(lo.detail)} short; {just_over}/100 -> "
          f"{len(hi.detail)} short")
    assert len(lo.detail) == 1 and not hi.detail


def test_a_short_run_is_flagged_on_every_day():
    """`shift(1)` on the reference: a run must not drag its own median down day by day, so a
    burst well inside the window is caught in full, not just on its first date."""
    run = {d: 9 for d in DATES[80:85]}
    res = invariant_day_coverage(_Ctx(_grid(thin=run)))
    print(f"  {len(run)} consecutive short sessions -> {len(res.detail)} flagged")
    assert len(res.detail) == len(run)


def test_an_outage_longer_than_half_the_window_stops_being_flagged():
    """A NAMED LIMIT of the trailing reference, asserted so it cannot be discovered by
    surprise. Once short days occupy more than half the reference window the rolling MEDIAN
    becomes the short count, and the outage reads as the universe's new normal.

    Accepted deliberately. The invariant's job is to raise the hole, not to enumerate every
    date in it, and it still fires on the first ~half -- 11 of 21 here, which is many days of
    warning. The alternative (a global median) is strictly worse: measured on the live table
    it flagged 20 legitimate 2005 dates as well, because the universe really did grow from
    383 names to 491. `momentum.features.MIN_XS_POPULATION_FRAC` carries the same trade."""
    run = {d: 9 for d in DATES[80:80 + DAY_COVERAGE_WINDOW]}
    res = invariant_day_coverage(_Ctx(_grid(thin=run)))
    caught = len(res.detail)
    print(f"  {len(run)} consecutive short sessions (= the {DAY_COVERAGE_WINDOW}-day window) "
          f"-> {caught} flagged before the reference follows them down")
    assert 0 < caught < len(run)
    assert caught >= DAY_COVERAGE_WINDOW // 2, "must still catch the onset"


def test_empty_table_is_not_a_failure():
    res = invariant_day_coverage(_Ctx(pd.DataFrame(columns=["ticker", "date"])))
    assert res.rows == 0 and res.failed == 0 and res.share == 0.0


def test_report_renders_the_date_clustered_shape():
    """`InvariantResult` clusters by TICKER for invariants 1-3 and by DATE here, so
    `to_markdown` needs its own branch -- the ticker-shaped branch would print 'No failures.'
    over a real hole, since `failing_tickers` is empty by construction."""
    res = invariant_day_coverage(_Ctx(_grid(thin={DATES[80]: 9})))
    assert res.clustered_by == "date"
    assert not res.failing_tickers, "clusters live in `detail` for this invariant"
    md = PricesReport([res]).to_markdown()
    print("\n".join(md.splitlines()[-4:]))
    assert "| date | tickers | expected | missing | present |" in md
    assert str(DATES[80].date()) in md
    assert "No failures" not in md
    # and it must be visible to the gate's headline number
    assert PricesReport([res]).worst_share() == res.share


def test_detail_is_json_serializable():
    """The DoD report writes this straight to JSON; a numpy scalar would raise there."""
    import json
    res = invariant_day_coverage(_Ctx(_grid(thin={DATES[80]: 9})))
    json.dumps(res.detail)
    for d in res.detail:
        assert all(type(v) in (str, int, float) for v in d.values()), d
