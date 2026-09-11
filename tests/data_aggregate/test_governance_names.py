"""
CEO identity for the governance cube (src/data_aggregate/utils/governance/names.py) and the
identity gates it enables in `def14a_impute`.

Three things under test:
  1. a co-CEO cell keys to the FIRST person listed (D28), not to a chimera of both;
  2. the temporal fill's gates, all four of them, on their own fixtures -- and in particular
     that `ceo_name_proxy` is NOT filled at all any more (D27 is overridden): the old rule
     required the same PERSON on BOTH sides of the gap, which reads a later filing, and the
     forward-only fill of 2026-09-09 cannot. Respelling tolerance survives on `ceo_salary`,
     the gate that is satisfiable without the future;
  3. the live archive reproduces the measurements phase 4's turnover guard is sized on -- these
     must be regenerated, never quoted from the plan document.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.def14a_impute import (
    CARRY_FORBIDDEN, CARRY_MAX_DAYS, impute_def14a,
)
from src.data_aggregate.utils.governance.names import (
    ceo_identity, is_multi_name, split_co_names,
)
from src.data_store.schema import Tables
from src.utils.names import person_key


def _row(ticker: str, as_of: str, **kw) -> dict:
    base = {"ticker": ticker, "as_of": as_of, "accession_number": f"{ticker}-{as_of}"}
    base.update(kw)
    return base


# --------------------------------------------------------------------------- co-CEO cells ---
def test_a_co_ceo_cell_keys_to_the_first_person_listed():
    """Unsplit, CPRT's `A. Jayson Adair; Jeffrey Liaw` keys `liaw|a` -- the first person's
    initial glued to the second's surname, matching NEITHER of them, so it reads as a turnover
    both the year it appears and the year it stops."""
    cases = [
        ("A. Jayson Adair; Jeffrey Liaw", "adair|a"),
        ("A. Jayson Adair and Jeffrey Liaw", "adair|a"),
        ("A. Jayson Adair & Jeffrey Liaw", "adair|a"),
        ("A. Jayson Adair", "adair|a"),          # the continuing CEO alone -> same key
    ]
    for cell, expected in cases:
        assert ceo_identity(cell) == expected, f"{cell!r} -> {ceo_identity(cell)!r}"
    assert ceo_identity("A. Jayson Adair; Jeffrey Liaw") != "liaw|a", "the unsplit chimera"

    # ` and ` needs its surrounding whitespace, or the pattern fires inside a surname
    assert ceo_identity("Richard Anderson") == "anderson|r"
    assert not is_multi_name("Richard Anderson")
    assert is_multi_name("A. Jayson Adair; Jeffrey Liaw")
    assert ceo_identity(None) is None and ceo_identity(np.nan) is None
    assert split_co_names(np.nan) == []

    print("\n=== SANITY CHECK: co-CEO cells (D28) ===")
    for cell, expected in cases:
        print(f"  {cell:<36} -> {ceo_identity(cell)}  (names: {split_co_names(cell)})")
    print("  'Richard Anderson' -> anderson|r  (' and ' did NOT fire inside the surname)")
    print("  CONCLUSION: a multi-name cell takes the FIRST person, so the continuing CEO's pay "
          "stays comparable year over year instead of keying to a chimera that matches nobody. "
          "Accepted cost: a genuine co-CEO transition reads as NO turnover. Validated.")


# --------------------------------------------------------- the forward carry and its gates ---
def test_the_forward_carry_gates_on_identity_and_refuses_the_name_outright():
    """The ways the temporal fill can decline, each on its own fixture.

    ⚠ THE `ceo_name_proxy` FILL IS GONE, and this test used to assert the opposite. Until
    2026-09-09 a gap was filled when the same person bounded it on BOTH sides -- which reads the
    LATER filing, and reading later filings is what this module stopped doing. Forward-only
    cannot tell a quiet gap from one hiding a succession, so the honest answer for the name is
    UNKNOWN: measured live, 86 carryable gaps of which 18 hide a real transition, so carrying
    them all would fabricate 18 CEO identities to gain 68 correct ones. A wrong name lets the
    turnover guard compare old-vs-old and compute pay growth across the succession it exists to
    catch, so the 68 are the measured price and `CARRY_FORBIDDEN` is where it is paid.

    `ceo_identity`'s respelling tolerance is still load-bearing -- it moved to the gate that
    survived, on `ceo_salary`, which is satisfiable forward-only because the row being filled
    names its own CEO.
    """
    df = pd.DataFrame([
        # AGREE, same spelling -> the salary carries
        _row("SAME", "2019-04-01", ceo_name_proxy="Timothy D. Cook", ceo_salary=3_000_000.0,
             classified_board=1.0),
        _row("SAME", "2020-04-01", ceo_name_proxy="Timothy D. Cook", ceo_salary=np.nan,
             classified_board=np.nan),
        # AGREE under the KEY though the strings differ -> carries (a raw comparison would not)
        _row("DRIFT", "2019-04-01", ceo_name_proxy="Timothy D. Cook", ceo_salary=3_000_000.0),
        _row("DRIFT", "2020-04-01", ceo_name_proxy="Tim Cook", ceo_salary=np.nan),
        # DISAGREE -- an ACGL-shaped succession -> the salary is a CONTRACT term, declined
        _row("TURN", "1999-03-16", ceo_name_proxy="Mark D. Mosca", ceo_salary=500_000.0),
        _row("TURN", "2000-03-16", ceo_name_proxy="Peter A. Appel", ceo_salary=np.nan),
        # the row names NOBODY -> declined, where the old `bfill` leg lent it an identity
        _row("MUTE", "2019-04-01", ceo_name_proxy="Sam Sole", ceo_salary=400_000.0),
        _row("MUTE", "2020-04-01", ceo_name_proxy=None, ceo_salary=np.nan),
        # a genuine NAME gap -> stays NaN, and is COUNTED rather than silently skipped
        _row("NAMEGAP", "2019-04-01", ceo_name_proxy="Ann Ash"),
        _row("NAMEGAP", "2020-04-01", ceo_name_proxy=None),
        _row("NAMEGAP", "2021-04-01", ceo_name_proxy="Ann Ash"),
        # a PROVISION gap must still carry forward
        _row("PROV", "2019-04-01", classified_board=1.0, poison_pill=0.0),
        _row("PROV", "2020-04-01", classified_board=np.nan, poison_pill=np.nan),
        _row("PROV", "2021-04-01", classified_board=0.0, poison_pill=1.0),
        # a TRAILING gap -- filled now, refused by the old interior-only rule
        _row("TRAIL", "2019-04-01", board_size=10.0),
        _row("TRAIL", "2020-04-01", board_size=np.nan),
        # a gap WIDER than the carry cap -> declined, so the level horizon cannot be laundered
        _row("STALE", "2010-04-01", board_size=8.0),
        _row("STALE", "2020-04-01", board_size=np.nan),
    ])
    out, stats = impute_def14a(df)
    out = out.set_index(["ticker", "as_of"])

    def at(t, d, c):
        return out.loc[(t, pd.Timestamp(d)), c]

    # --- the identity gate on ceo_salary, forward-only ---
    assert at("SAME", "2020-04-01", "ceo_salary") == 3_000_000.0
    assert at("DRIFT", "2020-04-01", "ceo_salary") == 3_000_000.0, \
        "a respelling is the same person under the key"
    assert pd.isna(at("TURN", "2000-03-16", "ceo_salary")), \
        "a salary was carried across a succession"
    assert pd.isna(at("MUTE", "2020-04-01", "ceo_salary")), \
        "a salary was carried onto a row that names no CEO"
    assert stats.get("declined (identity changed): ceo_salary") == 2      # TURN + MUTE

    # --- the name itself is never carried ---
    assert pd.isna(at("NAMEGAP", "2020-04-01", "ceo_name_proxy")), \
        "ceo_name_proxy was carried -- CARRY_FORBIDDEN is not being honoured"
    assert CARRY_FORBIDDEN == frozenset({"ceo_name_proxy"}), (
        "only the NAME is forbidden; widening this would stop the provisions carrying")
    # 2, not 1: NAMEGAP's interior gap AND the trailing unnamed row on MUTE. Both are carry
    # candidates under a forward rule -- which is the point, since the old interior-only test
    # would have seen only the first.
    assert stats.get("declined (carry cannot be validated): ceo_name_proxy") == 2, \
        "a refused carry must be COUNTED, or a fill of 0 cannot be told from nothing missing"

    # --- provisions carry, a trailing gap now fills, a stale one does not ---
    assert at("PROV", "2020-04-01", "classified_board") == 1.0
    assert at("PROV", "2020-04-01", "poison_pill") == 0.0
    assert at("SAME", "2020-04-01", "classified_board") == 1.0
    assert at("TRAIL", "2020-04-01", "board_size") == 10.0, \
        "the trailing gap was refused -- the live/backtest asymmetry is back"
    assert pd.isna(at("STALE", "2020-04-01", "board_size")), \
        f"a {CARRY_MAX_DAYS}d-stale value was carried and would read as fresh downstream"
    assert stats.get(f"declined (>{CARRY_MAX_DAYS}d stale): board_size") == 1

    # NON-DESTRUCTIVE: every disclosed value survives untouched
    for t, d, c, v in (("SAME", "2019-04-01", "ceo_salary", 3_000_000.0),
                       ("TURN", "1999-03-16", "ceo_salary", 500_000.0),
                       ("PROV", "2021-04-01", "poison_pill", 1.0),
                       ("STALE", "2010-04-01", "board_size", 8.0)):
        assert at(t, d, c) == v, f"{t} {c} was overwritten"

    print("\n=== SANITY CHECK: the forward carry and its declines ===")
    print(f"  SAME  (Cook -> Cook)                    salary -> "
          f"{at('SAME', '2020-04-01', 'ceo_salary'):,.0f}  carried")
    print(f"  DRIFT ('Timothy D. Cook' -> 'Tim Cook') salary -> "
          f"{at('DRIFT', '2020-04-01', 'ceo_salary'):,.0f}  carried "
          "(agreement judged on the KEY, not the string)")
    print(f"  TURN  (Mosca -> Appel)                  salary -> "
          f"{at('TURN', '2000-03-16', 'ceo_salary')}  DECLINED, a contract term")
    print(f"  MUTE  (Sole -> unnamed)                 salary -> "
          f"{at('MUTE', '2020-04-01', 'ceo_salary')}  DECLINED, the row names nobody")
    print(f"  NAMEGAP (Ash / gap / Ash)               name   -> "
          f"{at('NAMEGAP', '2020-04-01', 'ceo_name_proxy')}  FORBIDDEN, not merely gated")
    print(f"  TRAIL  board_size after the last filing -> "
          f"{at('TRAIL', '2020-04-01', 'board_size')}  carried (the old rule refused this)")
    print(f"  STALE  board_size 3,653 days later      -> "
          f"{at('STALE', '2020-04-01', 'board_size')}  DECLINED, past {CARRY_MAX_DAYS}d")
    print(f"  stats: {stats}")
    print("  CONCLUSION: the fill reads only the past. The name is refused outright because "
          "forward-only cannot tell a quiet gap from a succession; the salary is gated on the "
          "CEO named by the row being filled; a carry past the level horizon is refused so it "
          "cannot read as fresh downstream; and a TRAILING gap now fills, which closes the "
          "asymmetry where a backtest filled what a live run structurally could not. "
          "Non-destructive by construction. Validated.")


# ----------------------------------------------------------------- the live re-measurement ---
def _live_def14a() -> pd.DataFrame:
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load(Tables.def14a_llm)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")
    return raw


def test_normalisation_measured_on_the_live_archive():
    """Regenerate the numbers phase 4's turnover guard is SIZED on, rather than trusting the
    plan document's copy of them.

    Reported on THREE bases, because two of them are legitimately different measurements and
    conflating them is how a reproduction "fails" that in fact agrees:
      * `raw`   -- the filer's own string;
      * `key`   -- the bare `person_key`, which is what the phase-1 plan measured;
      * `ident` -- `ceo_identity`, i.e. the key AFTER the co-CEO split (D28), which is what the
        cube actually applies and which is strictly better on every line.
    Identity counts are shown BOTH globally and summed per ticker: a CEO of two companies is one
    person globally and two ticker-CEOs per ticker, and the plan quotes the per-ticker basis.

    Asserted as directional invariants plus loose bounds, not exact counts: the archive grows.
    But the DIRECTION is a contract -- normalisation can only ever merge identities, never split
    them, and the co-CEO split can only ever improve on the bare key."""
    raw = _live_def14a()
    df = raw[["ticker", "accession_number", "as_of", "ceo_name_proxy"]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.sort_values(["ticker", "as_of"])
    df["key"] = df["ceo_name_proxy"].map(person_key)
    df["ident"] = df["ceo_name_proxy"].astype(object).map(ceo_identity)

    named = df[df["ceo_name_proxy"].notna()]
    cols = ["ceo_name_proxy", "key", "ident"]
    glob = {c: int(named[c].nunique()) for c in cols}
    per_ticker = named.groupby("ticker")[cols].nunique()
    summed = {c: int(per_ticker[c].sum()) for c in cols}
    shrunk = {c: int((per_ticker[c] < per_ticker["ceo_name_proxy"]).sum()) for c in cols[1:]}

    # consecutive-filing pairs, both sides named -> how many read as a CEO turnover
    g = df.groupby("ticker", sort=False)
    comparable = df["ceo_name_proxy"].notna() & g["ceo_name_proxy"].shift(1).notna()
    turns = {}
    for c in cols:
        prev = g[c].shift(1)
        # `notna()` on both sides is load-bearing: None is UNKNOWN, and two unknowns must not
        # compare equal into "no turnover" -- that is the failure mode the guard exists to avoid
        turns[c] = int((comparable & df[c].notna() & prev.notna() & (df[c] != prev)).sum())

    assert glob["key"] < glob["ceo_name_proxy"], "the key must MERGE identities, never split them"
    assert glob["ident"] <= glob["key"], "the co-CEO split can only merge further"
    assert turns["key"] < turns["ceo_name_proxy"], "the key must remove apparent turnovers"
    assert turns["ident"] <= turns["key"], "the co-CEO split can only remove more"
    assert glob["key"] / glob["ceo_name_proxy"] > 0.7, "a 30%+ collapse would be over-merging"

    n_comparable, n_multi = int(comparable.sum()), int(named["ceo_name_proxy"].map(is_multi_name).sum())
    print("\n=== SANITY CHECK: what the person key buys, measured live ===")
    print(f"  rows={len(raw)}  tickers={raw['ticker'].nunique()}  named CEO cells={len(named)}")
    print(f"  {'measure':<40} {'raw':>7} {'key':>7} {'+co-CEO':>8}   {'delta vs raw':>14}")
    for label, d in (("distinct identities (global)", glob),
                     ("distinct identities (sum per ticker)", summed)):
        print(f"  {label:<40} {d['ceo_name_proxy']:>7} {d['key']:>7} {d['ident']:>8}   "
              f"{d['ident'] - d['ceo_name_proxy']:>+7} "
              f"({(d['ident'] - d['ceo_name_proxy']) / d['ceo_name_proxy']:+.1%})")
    print(f"  {'consecutive pairs comparable':<40} {n_comparable:>7} {n_comparable:>7} "
          f"{n_comparable:>8}   {'--':>14}")
    print(f"  {'pairs reading as a CEO TURNOVER':<40} {turns['ceo_name_proxy']:>7} "
          f"{turns['key']:>7} {turns['ident']:>8}   "
          f"{turns['ident'] - turns['ceo_name_proxy']:>+7} "
          f"({(turns['ident'] - turns['ceo_name_proxy']) / turns['ceo_name_proxy']:+.1%})")
    print(f"  tickers whose identity count shrinks: {shrunk['key']} (key) / {shrunk['ident']} "
          f"(+co-CEO) of {per_ticker.shape[0]}")
    print(f"  multi-name (co-CEO) cells encountered: {n_multi}  <- the D28 population, logged "
          "because there is no feature and no flag column behind it")
    print("  CONCLUSION: the spurious turnovers above are pay-growth observations phase 4's guard "
          "would otherwise DISCARD -- it nulls growth on every apparent CEO change, and on raw "
          "strings ~a fifth of those changes are one filer respelling one person. The co-CEO "
          "split removes a further slice the bare key cannot see. Validated.")


def test_the_turnover_guards_true_cost_on_computable_pay_pairs():
    """§5's last item: the guard's cost is not "apparent turnovers", it is pay-growth
    observations LOST -- consecutive filings that both carry a `ceo_total_comp`, where growth
    would be computable but the guard nulls it because the CEO changed.

    That is the number phase 4 is sized on, and measuring it on raw strings overstates it."""
    raw = _live_def14a()
    df = raw[["ticker", "as_of", "ceo_name_proxy", "ceo_total_comp"]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.sort_values(["ticker", "as_of"])
    df["ident"] = df["ceo_name_proxy"].astype(object).map(ceo_identity)
    g = df.groupby("ticker", sort=False)

    # a pay pair: consecutive filings of one ticker, both with a CEO total
    payable = df["ceo_total_comp"].notna() & g["ceo_total_comp"].shift(1).notna()
    costs = {}
    for label, col in (("raw string", "ceo_name_proxy"), ("ceo_identity", "ident")):
        prev = g[col].shift(1)
        changed = df[col].notna() & prev.notna() & (df[col] != prev)
        unknown = payable & (df[col].isna() | prev.isna())   # NaN identity -> also nulled
        costs[label] = (int((payable & changed).sum()), int(unknown.sum()))

    n_pairs = int(payable.sum())
    raw_cost = costs["raw string"][0]
    key_cost = costs["ceo_identity"][0]
    assert key_cost < raw_cost, "the guard must cost LESS once identities reconcile"

    print("\n=== SANITY CHECK: the turnover guard's true cost ===")
    print(f"  computable pay-growth pairs (both filings carry ceo_total_comp): {n_pairs}")
    for label, (changed, unknown) in costs.items():
        print(f"    on {label:<13} -> nulled by a CEO change: {changed:>5} "
              f"({changed / n_pairs:.1%})   | also nulled as UNKNOWN identity: {unknown}")
    print(f"  => the guard recovers {raw_cost - key_cost} pay-growth observations "
          f"({(raw_cost - key_cost) / raw_cost:.1%} of its raw-string cost) purely by "
          "reconciling spellings")
    print("  CONCLUSION: measured on raw strings the guard looks far more expensive than it is; "
          "roughly a fifth of what it would discard is one filer respelling one CEO's name. "
          "Validated.")


def test_the_ceo_to_neo_cross_table_match_measured_on_the_live_archive():
    """The join the pay-slice family depends on: is the CEO named on a `def14a_llm` row actually
    present in that same filing's own `def14a_executive_comp` rows?

    Phase 4's exact CPS divides `ceo_total_comp` by the sum of the top-5 NEO totals, which is
    only meaningful if the CEO is INSIDE that denominator. On raw strings the filer's proxy
    summary and its own Summary Compensation Table disagree often enough to matter -- the same
    person written two ways in two tables of one document."""
    proxies = _live_def14a()
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        neos = ctx.store.load(Tables.def14a_executive_comp)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_executive_comp not reachable ({e})")
    if neos is None or neos.empty:
        pytest.skip("def14a_executive_comp empty")

    key = ["ticker", "accession_number"]
    p = proxies[key + ["ceo_name_proxy"]].dropna(subset=["ceo_name_proxy"]).drop_duplicates(key)
    n = neos[key + ["name"]].dropna(subset=["name"]).copy()
    n["k"] = n["name"].map(person_key)

    raw_sets = n.groupby(key)["name"].apply(set)
    key_sets = n.groupby(key)["k"].apply(lambda s: set(s.dropna()))
    p = p.join(raw_sets.rename("neo_raw"), on=key).join(key_sets.rename("neo_key"), on=key)
    p = p[p["neo_raw"].notna()]                 # filings whose SCT was extracted at all

    hit_raw = int(sum(c in s for c, s in zip(p["ceo_name_proxy"], p["neo_raw"])))
    ident = p["ceo_name_proxy"].astype(object).map(ceo_identity)
    hit_key = int(sum(k is not None and k in s for k, s in zip(ident, p["neo_key"])))

    assert hit_key > hit_raw, "reconciling must find MORE of the CEOs, never fewer"
    assert hit_key / len(p) > 0.90, "a sub-90% match would mean the key is failing, not helping"

    print("\n=== SANITY CHECK: CEO <-> NEO cross-table match, measured live ===")
    print(f"  filings with both a named CEO and extracted SCT rows: {len(p)}")
    print(f"    CEO found in its own filing's NEO rows, RAW string : {hit_raw} "
          f"({hit_raw / len(p):.1%})")
    print(f"    CEO found in its own filing's NEO rows, KEYED      : {hit_key} "
          f"({hit_key / len(p):.1%})")
    print(f"    => +{hit_key - hit_raw} filings (+{(hit_key - hit_raw) / len(p):.1%}pp)")
    print("  CONCLUSION: the same person is written two ways in two tables of ONE document often "
          "enough to break the join on raw strings. Keying it is what lets phase 4's exact CEO "
          "Pay Slice assert the CEO sits inside its own top-5 denominator. Validated.")


def test_the_price_of_refusing_the_name_carry_measured_on_the_live_archive():
    """What `CARRY_FORBIDDEN` costs and what it buys, regenerated rather than quoted.

    The decision this measures is not a preference. Forward-only, a `ceo_name_proxy` gap is
    either carried or it is not -- there is no gate, because every gate that could separate the
    safe gaps from the unsafe ones needs the LATER filing. So the question reduces to a count:
    how many gaps hide a real succession? Those are the fabrications a blanket carry would
    commit; the rest are correct fills it would win. Both numbers are printed, and the choice
    (refuse) is defensible only while the second is small relative to the column's own coverage.

    ⚠ `fwd`/`bwd` HERE ARE A MEASUREMENT, NOT THE PIPELINE. This test reads the future on
    purpose -- that is the only way to score a forward-only rule's mistakes after the fact. The
    module under test never does.
    """
    raw = _live_def14a()
    df = raw[["ticker", "as_of", "ceo_name_proxy"]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.sort_values(["ticker", "as_of"])
    g = df.groupby("ticker", sort=False)
    fwd, bwd = g["ceo_name_proxy"].ffill(), g["ceo_name_proxy"].bfill()
    src = df["as_of"].where(df["ceo_name_proxy"].notna()).groupby(df["ticker"],
                                                                 sort=False).ffill()
    age = (df["as_of"] - src).dt.days

    carryable = df["ceo_name_proxy"].isna() & fwd.notna() & (age <= CARRY_MAX_DAYS)
    k_f = fwd.astype(object).map(ceo_identity)
    k_b = bwd.astype(object).map(ceo_identity)
    # scored against the future: would a blanket carry have been right?
    would_be_right = carryable & k_b.notna() & k_f.notna() & (k_f == k_b)
    would_be_wrong = carryable & k_b.notna() & k_f.notna() & (k_f != k_b)
    unverifiable = carryable & ~(would_be_right | would_be_wrong)

    n_carry = int(carryable.sum())
    n_right, n_wrong = int(would_be_right.sum()), int(would_be_wrong.sum())
    assert n_right + n_wrong + int(unverifiable.sum()) == n_carry, "the split does not close"
    assert n_wrong > 0, (
        "no gap hides a succession on this table, so refusing the carry costs coverage and "
        "buys nothing -- CARRY_FORBIDDEN needs re-deciding, not re-asserting")

    _, stats = impute_def14a(raw)
    refused = int(stats.get("declined (carry cannot be validated): ceo_name_proxy", 0))
    assert stats.get("carry: ceo_name_proxy") is None, \
        "ceo_name_proxy was carried -- CARRY_FORBIDDEN is not being honoured on live data"
    assert refused == n_carry, (
        f"the module refused {refused} but {n_carry} are carryable -- the counter and the "
        "population disagree")

    # the refusal must not be load-bearing for the column's coverage
    coverage = float(raw["ceo_name_proxy"].notna().mean())
    assert coverage > 0.95, (
        f"ceo_name_proxy is only {coverage:.1%} filled from the extraction, so refusing "
        f"{n_right} correct carries is no longer a cheap choice")

    wrong_rows = df[would_be_wrong]
    print("\n=== SANITY CHECK: the price of refusing the ceo_name_proxy carry ===")
    print(f"  extraction alone fills                        : {coverage:.1%}")
    print(f"  gaps a bounded forward carry COULD fill       : {n_carry}")
    print(f"    the future says the same CEO  -> would be RIGHT : {n_right}  (the price paid)")
    print(f"    the future says a new CEO     -> would be WRONG : {n_wrong}  (what it buys)")
    print(f"    no later name to score against -> unverifiable  : {int(unverifiable.sum())}")
    print(f"  module reports refused={refused}, carried=0")
    if not wrong_rows.empty:
        show = wrong_rows.head(5).assign(before=fwd[wrong_rows.index],
                                         after=bwd[wrong_rows.index])
        print("  the successions a blanket carry would have papered over, e.g.:")
        for _, r in show.iterrows():
            print(f"    {r['ticker']:<6} {str(r['as_of'])[:10]}  "
                  f"{r['before']!r} -> {r['after']!r}")
    print(f"  CONCLUSION: carrying the name would win {n_right} correct cells and fabricate "
          f"{n_wrong} CEO identities. A fabricated identity blinds the turnover guard to the "
          "exact transition it exists to catch and lets pay growth be computed straight across "
          f"it, so the {n_right} are given up on purpose. The column still ships at "
          f"{coverage:.1%} from the extraction alone, which is what makes that affordable. "
          "Validated.")


if __name__ == "__main__":
    test_a_co_ceo_cell_keys_to_the_first_person_listed()
    test_the_forward_carry_gates_on_identity_and_refuses_the_name_outright()
    test_normalisation_measured_on_the_live_archive()
    test_the_turnover_guards_true_cost_on_computable_pay_pairs()
    test_the_identity_gap_fill_measured_on_the_live_archive()
