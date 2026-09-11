"""
Deterministic numeric FINGERPRINT of the AGGREGATION layer (`src/data_aggregate/`).

Companion to `pipeline_fingerprint.py`, which fingerprints extraction + aggregation but
needs the raw SEC `data/sec_bulk_cache/companyfacts_CIK*.json` files. That cache is a
multi-GB download and is absent on most machines, so that guard cannot be replayed --
which left the aggregation refactor with no safety net at all. This module closes that
gap: it is SELF-CONTAINED after its first run.

    * fundamentals come from the DB table `fundamentals_history` ONCE and are then
      frozen to `aggregate_fingerprint_fundamentals.parquet` next to this file, so every
      later run (and every CI machine) replays byte-identical inputs with no DB and no
      SEC cache. Real filings are used deliberately: 237 columns across all 11 GICS
      sectors is what makes the sector gates (banks / REITs / insurance / energy) fire,
      and no synthetic frame reproduces that.
    * everything else (prices, dividends, earnings, proxies, 13F, insider, attention,
      short interest, fails) is a SEEDED synthetic source shaped like the real table.

COVERAGE. `pipeline_fingerprint` hashes 8 aggregation outputs and leaves 9 of the 13
panel builders unguarded, plus `build_peer_relative_panel`, both 13F quarter builders,
the commodity/currency twins and every ratio/standardize helper. Those are exactly the
functions the dedup sweep merges, so they are all fingerprinted here.

TWO KINDS OF KEY, on purpose:
  * `panel.*` / `label.*` -- PUBLIC entry points. Blind to how the work is organised
    internally, sensitive only to what comes out.
  * `prim.*` -- the PRIMITIVES that are about to be deduplicated (momentum, trailing
    vol, forward windows, the 5 cross-sectional standardizers, the ratio helpers, the
    two identical commodity/currency functions, both 13F quarter scaffolds). These are
    pinned BEFORE the merge; afterwards the same key is recomputed through the new
    unified helper and the hash must not move. The key survives the refactor; only the
    import behind it changes.

Regenerate the baseline with
    python -m tests.data_aggregate.aggregate_fingerprint
which writes `aggregate_fingerprint_baseline.json` next to this file.

RULE: the baseline may be regenerated ONLY in a commit that touches no `src/` file, or
in a PR that is exclusively a declared numeric change. Regenerating it alongside a
refactor destroys the very comparison it exists to make.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tests.data_aggregate.pipeline_fingerprint import frame_digest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("aggregate_fingerprint_baseline.json")
FUNDAMENTALS_CACHE = Path(__file__).with_name("aggregate_fingerprint_fundamentals.parquet")

SEED = 20260805
START, END = "2019-01-02", "2026-06-30"
TICKERS_PER_SECTOR = 2          # 11 GICS sectors -> 22 names; >= min_peers(3) by a wide margin
N_MANAGERS = 8                  # 13F filers in the synthetic holdings
FILING_LAG = 45


# --------------------------------------------------------------------------- #
# fixed inputs: fundamentals (DB once -> frozen parquet)                       #
# --------------------------------------------------------------------------- #
def _select_fundamentals() -> pd.DataFrame:
    """`TICKERS_PER_SECTOR` alphabetically-first tickers per GICS sector, so the draw is
    reproducible without an RNG and every sector-gated KPI family has names that pass its
    gate. Read from the DB only when the frozen parquet is absent.

    THE TWO ENRICHMENTS ARE PART OF THE INPUT, not of the panel builders. `StepCubeFundamentals`
    applies them between the load and the builders, so a slice frozen straight off the table is
    NOT the frame production feeds in:

      * `sector` / `industry_group` are not columns of `fundamentals_history` at all -- they are
        a `sp500_tickers` lookup. Without them `sector_gates.row_gate` fails closed and the whole
        sector-KPI layer is fingerprinted as empty (and the per-sector draw above has nothing to
        iterate over).
      * `revenueGrowth` / `earningsGrowth` are `CUBE_TIME_COLUMNS`: only the cube can compute
        them, because the year-ago leg is found by a 365-DAY as-of match, not a row offset.
    """
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")            # so `python -m ...aggregate_fingerprint` works standalone
    from types import SimpleNamespace

    from src.data_aggregate.utils.common.gics import attach_gics_columns
    from src.data_aggregate.utils.common.pit import add_cube_time_growth
    from src.data_store.store import DataStore
    from src.utils.db import get_engine

    store = DataStore(get_engine())
    fh = store.load("fundamentals_history")
    if fh.empty:
        raise RuntimeError("fundamentals_history is empty -> cannot build the aggregation "
                           "fingerprint (run the extraction step first)")
    # `attach_gics_columns` reads `context.store` and nothing else
    fh = attach_gics_columns(add_cube_time_growth(fh), SimpleNamespace(store=store))
    picked: list[str] = []
    for sector in sorted(fh["sector"].dropna().astype(str).unique()):
        names = sorted(fh.loc[fh["sector"].astype(str) == sector, "ticker"].unique())
        picked.extend(names[:TICKERS_PER_SECTOR])
    out = fh[fh["ticker"].isin(picked)].copy()
    out["as_of"] = pd.to_datetime(out["as_of"])
    return out.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def fundamentals() -> pd.DataFrame:
    """The frozen fundamentals slice. Written on first use, read verbatim afterwards, so
    the fingerprint is reproducible on a machine with neither the DB nor the SEC cache."""
    if FUNDAMENTALS_CACHE.exists():
        df = pd.read_parquet(FUNDAMENTALS_CACHE)
        df["as_of"] = pd.to_datetime(df["as_of"])
        return df
    df = _select_fundamentals()
    df.to_parquet(FUNDAMENTALS_CACHE, index=False)
    return df


# --------------------------------------------------------------------------- #
# fixed inputs: seeded synthetic sources                                       #
# --------------------------------------------------------------------------- #
def rng_for(name: str) -> np.random.Generator:
    """One INDEPENDENT generator per fixture, seeded off the fixture's own name.

    ⚠ THIS REPLACES A SINGLE GENERATOR THREADED THROUGH EVERY `synthetic_*` BUILDER IN ORDER,
    which coupled fixtures that have nothing to do with each other: whoever edited one fixture
    re-phased the draw stream for every fixture built AFTER it, so unrelated outputs moved and
    the diff blamed the production code.

    It is not hypothetical -- it is why this baseline had to be regenerated. Making
    `insider_ownership_pct` conditional in `synthetic_def14a` skipped a draw on the 3 tickers
    with `i % 4 == 0` across 8 years: **measured 888 -> 864 draws**. Six outputs that no
    production change touched -- `prim.safe_div`, `prim.ratio_helpers`, `prim.xs_standardize`,
    `prim.forward_windows`, `prim.macro_factor_returns`, `prim.peer_relative_panel` -- then
    differed from the baseline, and `prim.super_quarter_features` even lost a row (659 -> 658)
    because a `(ticker, quarter)` group lost its last roster holder. Restoring those 24 draws
    put all seven back byte-identical, which is how the coupling was proved rather than assumed.

    Seeding off the NAME (not a positional index) means reordering the calls, or deleting a
    fixture, also changes nothing for the others -- `synthetic_attention` was kept purely as a
    stream spacer under the old scheme and no longer needs to be.
    """
    return np.random.default_rng([SEED, *name.encode("utf-8")])


def synthetic_prices(tickers: list[str], rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Seeded geometric random walks. A flat panel would leave every momentum /
    volatility / valuation-vs-price feature degenerate and the fingerprint blind."""
    idx = pd.bdate_range(START, END)
    steps = rng.normal(0.0004, 0.018, size=(len(idx), len(tickers)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tickers)
    return {
        "close": close,
        "open": close.shift(1).bfill() * (1 + rng.normal(0, 0.002, close.shape)),
        "high": close * (1 + np.abs(rng.normal(0, 0.006, close.shape))),
        "low": close * (1 - np.abs(rng.normal(0, 0.006, close.shape))),
        "volume": pd.DataFrame(rng.lognormal(15, 0.4, close.shape), index=idx, columns=tickers),
    }


def synthetic_dividends(tickers: list[str], idx: pd.DatetimeIndex,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Quarterly ex-dates for two thirds of the names (the rest are true non-payers, which
    must rank as a real 0 yield rather than NaN)."""
    payers = tickers[: max(1, len(tickers) * 2 // 3)]
    ex_dates = pd.date_range(idx[0], idx[-1], freq="QE")
    rows = []
    for t in payers:
        base = float(rng.uniform(0.15, 1.10))
        for k, d in enumerate(ex_dates):
            rows.append({"date": d, "ticker": t,
                         "dividends": round(base * (1.0 + 0.02 * k), 4)})
    return pd.DataFrame(rows)


def synthetic_earnings(tickers: list[str], idx: pd.DatetimeIndex,
                       rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(idx[0], idx[-1], freq="QE")
    rows = []
    for t in tickers:
        level = float(rng.uniform(0.5, 3.0))
        for k, d in enumerate(dates):
            est = level * (1.0 + 0.015 * k)
            act = est * (1.0 + float(rng.normal(0.01, 0.05)))
            rows.append({"ticker": t, "earnings_date": d,
                         "eps_estimate": round(est, 4), "eps_actual": round(act, 4),
                         "surprise_pct": round(100.0 * (act / est - 1.0), 4)})
    return pd.DataFrame(rows)


def synthetic_def14a(tickers: list[str], idx: pd.DatetimeIndex,
                     rng: np.random.Generator) -> pd.DataFrame:
    """One annual proxy per name per year, with the columns `governance_features` reads."""
    years = sorted({d.year for d in idx})
    rows = []
    for i, t in enumerate(tickers):
        pay = float(rng.uniform(5e6, 3e7))
        since = int(rng.integers(1998, 2018))
        # ⚠ `ceo_name_proxy` is REQUIRED for the pay families to exist at all: without it the
        # CEO identity is UNKNOWN on every row, so the turnover guard correctly nulls both
        # `ceo_comp_growth_1y` and `ceo_turnover_flag` and 12 of the 13 pay fields vanish from
        # the digest. Every third ticker changes CEO mid-sample, so the guard's null branch and
        # its pass branch are both exercised rather than only one of them.
        change_at = years[len(years) // 2] if i % 3 == 0 else None
        # ⚠ The PROVISION block is equally required, and for the same reason: with no
        # `classified_board` / `majority_voting` / `auditor_name` there are no adjacent pairs, so
        # all 13 transition flags, both counts and the whole auditor block are absent from the
        # digest. Each provision flips at a ticker-dependent year so both directions of the
        # detector fire somewhere; `majority_voting` is deliberately left NULL on one year per
        # ticker so the tri-state skip path is exercised too, and the auditor changes for one
        # ticker in three so `auditor_changed` and both tenure bases appear.
        flip = years[len(years) // 2 + (i % 3) - 1]
        first_firm, second_firm = ("Ernst & Young LLP", "KPMG LLP") if i % 3 == 0 else \
            ("Deloitte & Touche LLP", "Deloitte & Touche LLP")
        for k, y in enumerate(years):
            who = "Robin Vance" if change_at is not None and y >= change_at else "Alex Mercier"
            late = y >= flip
            rows.append({
                "classified_board": float(late) if i % 2 == 0 else float(not late),
                "dual_class_shares": float(i % 4 == 0),
                "poison_pill": float(late) if i % 5 == 0 else np.nan,
                "majority_voting": np.nan if y == flip else float(not late),
                "ceo_is_board_chair": float(late) if i % 2 else float(not late),
                "independent_chair": float(not late) if i % 3 else float(late),
                "lead_independent_director": float(late) if i % 3 == 1 else 1.0,
                "avg_other_public_boards": float(1.0 + 0.2 * k + 0.3 * (i % 4)),
                "auditor_name": second_firm if late else first_firm,
                "auditor_since_year": float(2004 + i % 5) if i % 2 else np.nan,
                "ticker": t, "as_of": pd.Timestamp(year=y, month=4, day=15),
                "accession_number": f"{t}-{y}",
                "ceo_name_proxy": who,
                "ceo_total_comp": pay * (1.0 + 0.06 * k),
                "ceo_pay_ratio": float(rng.uniform(50, 400)),
                "ceo_equity_pay_pct": float(rng.uniform(0.3, 0.9)),
                "pct_independent_directors": float(rng.uniform(0.6, 0.95)),
                "pct_female_directors": float(rng.uniform(0.1, 0.5)),
                "board_size": float(rng.integers(7, 15)),
                "avg_board_tenure": float(rng.uniform(3, 14)),
                # a FRACTION, like every other share-of-a-whole in this fixture. It was
                # `uniform(60, 99)` -- percent -- and phase 0's `(0, 1)` domain gate then
                # blanked every synthetic cell, so `_def14a_raw_fields` silently dropped
                # `f_say_on_pay_support` from the panel and the baseline lost a whole
                # feature. The live column is 0.034 .. 1.0 (median 0.94).
                "say_on_pay_support_pct": float(rng.uniform(0.60, 0.99)),
                # ⚠ THE OWNERSHIP BLOCK EXERCISES FOUR CODE PATHS AND USED TO EXERCISE NONE.
                # Without `insider_shares` and `insider_voting_pct` this fixture made
                # `panel.economic_ownership`, `panel.repair_ownership_basis` and
                # `panel._control_wedge` all return early on their first guard, so the whole
                # phase-5 ownership rework was absent from the digest -- and regenerating the
                # baseline over that would have frozen `f_control_wedge` as permanently
                # untested. That is exactly what happened to `f_say_on_pay_support` above, and
                # the note there is the reason this one exists.
                #
                # The four paths, by ticker and year:
                #   single-class (i % 4 != 0)  -> a DISCLOSED percentage plus a share count, so
                #       `economic_ownership` must PREFER the disclosed value; the wedge is 0 by
                #       the single-class identity.
                #   dual-class, later years    -> ownership NULL (a per-class table returns
                #       nothing), so the computed leg FILLS the hole and the wedge is positive.
                #   dual-class, first year     -> a per-class-looking 0.92 ABOVE the voting
                #       0.55, so `repair_ownership_basis` blanks it and the computation then
                #       refills it -- which is the only path that proves the two run in the
                #       right ORDER.
                #
                # `insider_shares` is deliberately ~2-8e6: small enough that dividing by ANY
                # real S&P share count (5e7 .. 1.5e10 in the frozen fundamentals slice) lands
                # inside the (0, 1] guard, so the path fires for every ticker without this
                # fixture needing to know each one's share count.
                "insider_ownership_pct": (
                    np.nan if (i % 4 == 0 and k > 0)
                    else 0.92 if i % 4 == 0
                    else float(rng.uniform(0.001, 0.08))),
                "insider_voting_pct": (
                    float(0.55 + 0.01 * (i % 5) + 0.005 * k) if i % 4 == 0 else np.nan),
                "insider_shares": float(2.0e6 + 3.0e5 * i + 1.0e5 * k),
                "ceo_is_founder": float(int(rng.integers(0, 2))),
                "ceo_since_year": float(since),
            })
    return pd.DataFrame(rows)


#: The board this fixture seats. Real names, because `board_turnover` keys on `person_key` and
#: `person_key("Steady 1")` is `steady` -- four placeholder names would collapse to ONE key and
#: a five-seat board would digest as a two-seat one.
_BOARD = ("Ann Alder", "Bob Birch", "Cara Cedar", "Dave Dogwood", "Erin Elm", "Finn Fir",
          "Gina Gum", "Hal Hazel", "Iris Ivy", "Jack Juniper")


def synthetic_directors(proxies: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One row per DIRECTOR per proxy — the child table phase 6 reads.

    ⚠ REQUIRED for six of the ten phase-6 fields to exist at all: with no directors table there
    are no board-quality features, and `board_aggregates` has nothing to derive.

    Each branch of the child fill is exercised by construction rather than by luck:
      * one director's `other_public_company_boards` is NULL on a middle year with the SAME value
        either side -> the D37 agreement gate FILLS it;
      * another's is NULL with DIFFERENT values either side -> the gate DECLINES;
      * one director's `age` is NULL on their FIRST proxy -> only the D38 accrual reaches it;
      * one seat turns over halfway through, so `board_turnover` has a non-zero year AND quiet
        years, which is what makes a measured 0.0 distinguishable from an absence.
    """
    rows = []
    for r in proxies.itertuples(index=False):
        t, y = r.ticker, int(pd.Timestamp(r.as_of).year)
        # 5-7 seats, stable per ticker AND ACROSS PROCESSES. ⚠ `hash(t)` was wrong here
        # and the fingerprint test is what caught it: Python randomizes `hash()` of a str
        # per interpreter (PYTHONHASHSEED), so the seat count -- and with it every
        # board-quality and director-pay value -- was a fresh lottery draw on every run.
        # `crc32` is a fixed function of the bytes, so the fixture is reproducible.
        seats = 5 + (zlib.crc32(t.encode()) % 3)
        mid = int(proxies["as_of"].dt.year.median())
        for s in range(seats):
            # the last seat turns over at `mid`: one departure and one arrival
            name = (_BOARD[(s + 5) % len(_BOARD)] if s == seats - 1 and y >= mid
                    else _BOARD[s % len(_BOARD)])
            age = 52.0 + 3.0 * s + (y - mid)
            boards = float(s % 4)
            if s == 1 and y == mid:                    # agreement -> filled
                boards = np.nan
            elif s == 2:                               # disagreement -> declined
                boards = np.nan if y == mid else float(1 + 3 * (y > mid))
            rows.append({
                "ticker": t, "accession_number": r.accession_number, "as_of": r.as_of,
                "name": name,
                "age": np.nan if (s == 0 and y == int(proxies["as_of"].dt.year.min())) else age,
                "tenure_years": float(2 + 4 * s + (y - mid)),
                "is_independent": float(s > 0),
                "other_public_company_boards": boards,
            })
    return pd.DataFrame(rows)


def synthetic_director_comp(directors: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One Item 402(k) row per director per proxy, for the four director-pay fields.

    ⚠ ONE row per filing carries a NULL `total` with its six components present, so the 402(k)
    identity is exercised; another carries every component NULL, so `min_count=1`'s "stays NULL"
    branch is too. Without both, `impute_director_comp` digests as a pass-through.
    """
    rows = []
    for i, r in enumerate(directors.itertuples(index=False)):
        fees = 90_000.0 + 5_000.0 * (i % 6)
        stock = 160_000.0 + 10_000.0 * (i % 4)
        other = 4_000.0 * (i % 3)
        total = fees + stock + other
        if i % 17 == 0:                                # NULL total, components present
            total = np.nan
        blank = i % 53 == 0                            # no component at all -> stays NULL
        rows.append({
            "ticker": r.ticker, "accession_number": r.accession_number, "as_of": r.as_of,
            "name": r.name,
            "total": np.nan if blank else total,
            "fees_earned": np.nan if blank else fees,
            "stock_awards": np.nan if blank else stock,
            "option_awards": np.nan, "non_equity_incentive": np.nan,
            "pension_change": np.nan,
            "other_compensation": np.nan if blank else other,
        })
    return pd.DataFrame(rows)


def synthetic_exec_comp(proxies: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Five NEOs per filing over TWO fiscal years, mirroring Item 402(c)'s three-year table.

    The second fiscal year is what pins the "latest fiscal year only" rule: summing both would
    inflate the CPS denominator and the digest would move.
    """
    rows = []
    for i, r in enumerate(proxies.itertuples(index=False)):
        fy = int(pd.Timestamp(r.as_of).year) - 1
        ceo = float(r.ceo_total_comp)
        # the CEO plus four deputies at a decaying share, so the slice lands in (0, 1)
        pool = [(r.ceo_name_proxy, ceo)] + [
            (f"Deputy {j} of {i % 7}", ceo * float(share))
            for j, share in enumerate((0.55, 0.42, 0.33, 0.27))]
        for name, total in pool:
            for year, mult in ((fy, 1.0), (fy - 1, 0.85)):
                rows.append({"ticker": r.ticker, "accession_number": f"{r.ticker}-{fy}",
                             "as_of": r.as_of, "name": name, "fiscal_year": year,
                             "total": total * mult, "reconciles": 1.0})
    return pd.DataFrame(rows)


def synthetic_short_interest(tickers: list[str], idx: pd.DatetimeIndex,
                             rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(idx) * len(tickers)
    total = rng.lognormal(14.0, 0.5, n)
    short = total * rng.uniform(0.15, 0.65, n)
    si = pd.DataFrame({
        "date": np.repeat(idx.to_numpy(), len(tickers)),
        "ticker": np.tile(np.array(tickers), len(idx)),
        "short_volume": short.round(0), "total_volume": total.round(0),
        "short_interest": (total * rng.uniform(0.01, 0.20, n)).round(0),
        "avg_daily_volume": (total * rng.uniform(0.8, 1.2, n)).round(0),
    })
    # fails are SPARSE: a security is listed only on days it actually had fails
    keep = rng.random(n) < 0.15
    ftd = pd.DataFrame({
        "date": np.repeat(idx.to_numpy(), len(tickers))[keep],
        "ticker": np.tile(np.array(tickers), len(idx))[keep],
        "fails_quantity": rng.lognormal(8.0, 1.2, int(keep.sum())).round(0),
    })
    return si, ftd


def synthetic_insider(tickers: list[str], idx: pd.DatetimeIndex,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Forms 3/4/5 rows including the non-discretionary codes the builder must ignore.

    ⚠ WIDENED 2026-09-10, AND THE OLD SHAPE WAS PINNING A HOLE. Phase 2.3 rebuilt the insider
    panel on the full Form 4 record -- share class, price, owner identity, post-trade holding,
    the 10b5-1 flag -- and the four-column fixture this used to return produced an **empty**
    panel through the new builder (`panel.insider` 43,010 rows x 10 cols -> 0 x 0). A baseline
    regenerated on that would have frozen the absence of a feature family, which is exactly
    what the elite panel's fixture did in Phase 2.2. Every column below is READ by
    `insider_quality.clean_transactions` or by a feature; none is decoration.

    A handful of rows are deliberately MISPRICED (~1.5%, a 10^2 slip) so the consensus repair
    has something to catch -- a fixture in which the repair never fires cannot detect the
    repair breaking.
    """
    codes = np.array(["P", "S", "A", "M", "F", "G"])
    n = 40 * len(tickers)
    days = pd.to_datetime(rng.choice(idx.to_numpy(), n))
    tkr = rng.choice(np.array(tickers), n)
    shares = rng.lognormal(6.0, 1.0, n).round(0)
    # One price level per ticker so the +/-15-day consensus is well defined and the mispriced
    # rows stand out against it rather than against noise.
    level = {t: float(20.0 + 80.0 * rng.random()) for t in tickers}
    price = np.array([level[t] for t in tkr]) * (1.0 + 0.05 * rng.standard_normal(n))
    price = np.abs(price).round(2)
    slipped = rng.random(n) < 0.015
    filed_price = np.where(slipped, price * 100.0, price)
    owners = np.array([f"{9000000 + 13 * i:010d}" for i in range(12)])
    titles = np.array(["Chief Executive Officer", "Chief Financial Officer",
                       "President and COO", "Executive Vice President", ""])
    title = rng.choice(titles, n, p=[0.15, 0.15, 0.1, 0.1, 0.5])
    return pd.DataFrame({
        "accession_number": [f"{1000000000 + 7 * i:010d}-00-{i % 1000:06d}" for i in range(n)],
        "ticker": tkr,
        "owner_cik": rng.choice(owners, n),
        "owner_name": rng.choice(np.array(["Doe Jane", "Roe Rick", "Poe Pat"]), n),
        "filing_date": days,
        # Two business days before the filing, which is the Form 4 deadline: the stamp must be
        # `filing_date`, and a fixture where the two are equal cannot show that.
        "transaction_date": days - pd.Timedelta(days=2),
        "transaction_code": rng.choice(codes, n, p=[0.25, 0.35, 0.15, 0.1, 0.1, 0.05]),
        "shares": shares,
        "price_per_share": filed_price,
        "value_usd": (shares * filed_price).round(2),
        "shares_owned_after": (shares * (1.0 + 9.0 * rng.random(n))).round(0),
        "security_type": np.where(rng.random(n) < 0.08, "deriv", "nonderiv"),
        "security_title": rng.choice(np.array(["Common Stock", "Class A Common Stock",
                                               "Preferred Stock, Series H"]),
                                     n, p=[0.8, 0.15, 0.05]),
        "direct_indirect": rng.choice(np.array(["D", "I"]), n, p=[0.7, 0.3]),
        "officer_title": title,
        "is_director": (title == "").astype("float64"),
        "is_officer": (title != "").astype("float64"),
        "is_ten_pct_owner": (rng.random(n) < 0.05).astype("float64"),
        # NaN before the 10b5-1 floor, as in the real table: the flag does not exist in the
        # source before 2023, and a 0 there would be a claim the data cannot support.
        "is_10b5_1": np.where(days >= pd.Timestamp("2023-07-01"),
                              (rng.random(n) < 0.6).astype("float64"), np.nan),
    })


def synthetic_13f(tickers: list[str], idx: pd.DatetimeIndex,
                  rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Manager-grain 13F: one row per (manager, ticker, quarter). Managers enter and exit
    so ic_inst_new_buyers / ic_inst_exiters / breadth_chg are non-degenerate. Half the CIKs form the
    'superinvestor' roster."""
    periods = pd.date_range(idx[0], idx[-1], freq="QE")
    ciks = [f"{1000000 + 7919 * i:010d}" for i in range(N_MANAGERS)]
    rows = []
    for p in periods:
        for i, cik in enumerate(ciks):
            for t in tickers:
                if rng.random() < 0.25:                  # manager not holding this quarter
                    continue
                shares = float(rng.lognormal(11.0, 0.8))
                rows.append({
                    "cik": cik, "period": p, "ticker": t,
                    "shares": round(shares, 0),
                    "value_usd": round(shares * float(rng.uniform(20, 400)), 2),
                    "call_value": round(shares * float(rng.uniform(0, 12)), 2),
                    "put_value": round(shares * float(rng.uniform(0, 9)), 2),
                    "filing_date": p + pd.Timedelta(days=FILING_LAG - 5 + i),
                })
    holdings = pd.DataFrame(rows)
    roster = {"cik_to_name": {c: f"Manager {k}" for k, c in enumerate(ciks[: N_MANAGERS // 2])}}
    return holdings, roster


def synthetic_manager_book(holdings: pd.DataFrame, tickers: list[str],
                           rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`sec13f_manager_holdings`: CUSIP grain, and the manager's WHOLE book.

    ⚠ A SEPARATE FIXTURE FROM `synthetic_13f`, MIRRORING A SEPARATE TABLE. The elite panel
    reads the complete book so that `portfolio_weight` divides by the manager's real total;
    the all-filer panel reads `sec13f_hr`, which is universe-filtered at extraction. Feeding
    one frame to both is what the whole Phase 2.2 rebuild exists to stop, and a fixture that
    did it would fingerprint a denominator the production code never sees.

    ⚠ IT ALSO DRAWS FROM ITS OWN RNG STREAM AND LEAVES `holdings` UNTOUCHED, so
    `panel.institutional` stays byte-identical. Off-universe positions are appended, never
    interleaved, for the same reason -- an order-coupled draw is how this fixture previously
    faked drift in seven outputs.
    """
    cusip_map = pd.DataFrame({
        "cusip": [f"{i:08d}C" for i in range(len(tickers))],
        "ticker": sorted(tickers),
    })
    by_ticker = dict(zip(cusip_map["ticker"], cusip_map["cusip"]))
    book = holdings.assign(cusip=holdings["ticker"].map(by_ticker),
                           position_type="common")
    book = book.drop(columns=[c for c in ("ticker", "call_value", "put_value")
                              if c in book.columns])

    # the rest of each manager's book: names outside the analysis universe, which carry a
    # median ~40-50% of real book value and are exactly what the slice denominator missed.
    keys = holdings[["cik", "period", "filing_date"]].drop_duplicates()
    extra = []
    for cik, period, filed in keys.itertuples(index=False):
        for j in range(int(rng.integers(2, 7))):
            shares = float(rng.lognormal(11.0, 0.9))
            extra.append({"cik": cik, "period": period, "filing_date": filed,
                          "cusip": f"OTC{j:05d}X", "position_type": "common",
                          "shares": round(shares, 0),
                          "value_usd": round(shares * float(rng.uniform(20, 400)), 2)})
    book = pd.concat([book, pd.DataFrame(extra)], ignore_index=True)
    return book, cusip_map


def primitive_fixtures(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """The edge cases where the to-be-merged primitives differ from each other: a
    zero-dispersion row, an all-NaN row, a single-name row, an inf, a -100% return, a
    zero / negative / all-NaN denominator and a mismatched column set."""
    idx = pd.bdate_range("2024-01-01", periods=40)
    cols = [f"P{i}" for i in range(12)]
    m = pd.DataFrame(rng.normal(0, 1, (40, 12)), index=idx, columns=cols)
    m.iloc[3] = 7.0                                   # zero cross-sectional dispersion
    m.iloc[7] = np.nan                                # all-NaN row
    m.iloc[11] = np.nan
    m.iloc[11, 0] = 1.5                               # single-name row
    m.iloc[15, 2] = np.inf                            # +inf cell
    m.iloc[16, 3] = -np.inf
    m.iloc[20, 4] = 0.0

    num = pd.DataFrame(rng.normal(5, 2, (40, 12)), index=idx, columns=cols)
    den = pd.DataFrame(rng.normal(1, 3, (40, 12)), index=idx, columns=cols)
    den.iloc[0] = 0.0                                 # zero denominator row
    den.iloc[1] = -1.0                                # negative denominator row
    den.iloc[2] = np.nan
    den_short = den.iloc[:, :8].copy()                # mismatched columns
    den_short.columns = cols[:8]

    ret = pd.DataFrame(rng.normal(0.0005, 0.02, (40, 12)), index=idx, columns=cols)
    ret.iloc[5, 1] = -1.0                             # exactly -100%: log1p edge
    ret.iloc[6, 2] = -1.4                             # below -100%: must be floored
    ret.iloc[9] = np.nan

    price = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, 4)), axis=0)),
                         index=idx, columns=["SPY", "CL=F", "GC=F", "USDEUR=X"])
    return {"matrix": m, "num": num, "den": den, "den_short": den_short,
            "returns": ret, "other_close": price}


# --------------------------------------------------------------------------- #
# the fingerprint                                                              #
# --------------------------------------------------------------------------- #
def compute() -> dict:
    from src.data_aggregate.utils.target.betas import estimate_all_betas
    from src.data_aggregate.utils.assemble.composites import build_composites
    from src.data_aggregate.utils.fundamentals.dividend_features import build_dividend_feature_panel
    from src.data_aggregate.utils.fundamentals.earnings_features import build_earnings_feature_panel
    from src.data_aggregate.utils.fundamentals.employee_features import build_employee_feature_panel
    from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
    from src.data_aggregate.utils.common.prices import (
        forward_compound, forward_cumchange, forward_return, momentum_characteristic,
        trailing_vol,
    )
    from src.data_aggregate.utils.momentum.features import (
        build_feature_panel, compute_raw_features,
    )
    from src.data_aggregate.utils.fundamentals.fundamental_features import build_fundamental_feature_panel
    from src.data_aggregate.utils.governance.director_comp import impute_director_comp
    from src.data_aggregate.utils.governance.directors import fill_director_attributes
    from src.data_aggregate.utils.governance.panel import build_governance_feature_panel
    from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel
    from src.data_aggregate.utils.institutionals.institutional_features import (
        _quarter_features, build_institutional_feature_panel,
    )
    from src.data_aggregate.utils.common.frames import ratio, safe_div, sanitize
    from src.data_aggregate.utils.common.panel import build_peer_relative_panel
    from src.data_aggregate.utils.common.xs import (
        winsorize_xs, xs_rank_pct, xs_standardize, xs_z,
    )
    from src.data_aggregate.utils.fundamentals.sector_features import build_sector_feature_panel
    from src.data_aggregate.utils.institutionals.short_interest_features import (
        build_short_interest_feature_panel,
    )
    from src.data_aggregate.utils.institutionals.superinvestor_features import (
        _prepare, attach_tickers, build_superinvestor_feature_panel,
        manager_quarter_state, manager_stock_conviction,
    )
    from src.data_aggregate.utils.target.targets import (
        build_targets_multi, cross_sectional_rank, cross_sectional_zscore,
    )

    fund = fundamentals()
    tickers = sorted(fund["ticker"].unique())
    px = synthetic_prices(tickers, rng_for("prices"))
    close, idx = px["close"], px["close"].index
    returns = close.pct_change(fill_method=None)
    sector_ret = returns.rolling(5).mean().bfill()             # deterministic stand-in
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    # Each fixture draws from its OWN generator -- see `rng_for`. Editing any one of these can
    # no longer move the numbers produced by any other, which is what made the previous
    # baseline diff unreadable. `synthetic_attention` used to be called here purely to hold the
    # shared stream's position after the attention panel was deleted; with the streams isolated
    # that spacer has no job, so it is gone.
    div = synthetic_dividends(tickers, idx, rng_for("dividends"))
    earn = synthetic_earnings(tickers, idx, rng_for("earnings"))
    proxies = synthetic_def14a(tickers, idx, rng_for("def14a"))
    neo_comp = synthetic_exec_comp(proxies, rng_for("exec_comp"))
    board = synthetic_directors(proxies, rng_for("directors"))
    board_filled, _ = fill_director_attributes(board)
    board_pay, _ = impute_director_comp(synthetic_director_comp(board, rng_for("director_comp")))
    short_hist, ftd = synthetic_short_interest(tickers, idx, rng_for("short_interest"))
    insider = synthetic_insider(tickers, idx, rng_for("insider"))
    holdings, roster = synthetic_13f(tickers, idx, rng_for("13f"))
    book, cusip_map = synthetic_manager_book(holdings, tickers,
                                             rng_for("13f_manager_book"))
    fx = primitive_fixtures(rng_for("primitives"))

    out: dict[str, dict] = {}

    # ---- the frozen input itself: a DB change must fail LOUDLY and by name ---- #
    out["input.fundamentals_slice"] = frame_digest(fund)

    # ---- PUBLIC entry points: every panel builder ---- #
    out["panel.raw_features"] = frame_digest(pd.concat(
        {k: v for k, v in sorted(compute_raw_features(
            close, px["open"], sector_ret, high=px["high"], low=px["low"],
            volume=px["volume"], seasonal_horizons=[30, 60, 90]).items())
         if isinstance(v, pd.DataFrame)}, axis=1))
    out["panel.price"] = frame_digest(build_feature_panel(
        close, px["open"], sector_ret, "rank", px["high"], px["low"], px["volume"],
        [30, 60, 90]))
    # `pension_facts` / `notes_num` are deliberately NOT passed: they are separate bulk SEC
    # tables, and freezing a slice of each would double the fixture for one feature family.
    # The consequence is explicit -- the `pension_*` features are absent from this digest and
    # are guarded by `test_insider_pension_features.py` instead.
    fp = build_fundamental_feature_panel(fund, peers, idx, stock_close=close,
                                         earnings_history=earn)
    out["panel.fundamental"] = frame_digest(fp)
    sp = build_sector_feature_panel(fund, peers, idx)
    out["panel.sector"] = frame_digest(sp)
    out["panel.earnings"] = frame_digest(build_earnings_feature_panel(
        earn, peers, idx, stock_close=close))
    out["panel.employee"] = frame_digest(build_employee_feature_panel(
        fund, peers, idx, fundamentals_history=fund))
    out["panel.dividend"] = frame_digest(build_dividend_feature_panel(
        div, peers, idx, stock_close=close, fundamentals_history=fund))
    # `exec_comp` and `close_total` are what make the three EXECUTIVE-PAY families exist:
    # without the child table there is no CPS denominator, and without a total-return series no
    # pay-vs-performance leg. `close` stands in for `close_total` here, as it does for the
    # dividend panel -- the synthetic series carries no dividends, so the two bases coincide.
    # ⚠ `directors` / `director_comp` are passed CLEANED, as the step passes them: the child
    # fill and the 402(k) identity are the caller's job (`StepCubeGovernance`), so digesting the
    # raw children would freeze a different contract than the one that ships. The board-average
    # REPAIR (D35) is deliberately NOT in this digest -- `merge_board_aggregates` runs in the
    # step, before `impute_def14a`, and is guarded by `test_governance_directors.py` plus the D3
    # twelve-feature guard.
    out["panel.governance"] = frame_digest(build_governance_feature_panel(
        proxies, peers, idx, fundamentals_history=fund,
        exec_comp=neo_comp, directors=board_filled, director_comp=board_pay,
        close_total=close)[0])
    out["panel.short_interest"] = frame_digest(build_short_interest_feature_panel(
        short_hist, peers, idx, fails_history=ftd, volume=px["volume"]))
    out["panel.institutional"] = frame_digest(build_institutional_feature_panel(
        holdings, peers, idx, shares_out_history=fund, stock_close=close))
    # ⚠ `cusip_map` AND `universe` ARE BOTH REQUIRED. Without a cusip map `attach_tickers`
    # nulls every ticker and the builder returns an EMPTY frame -- which is what this
    # fingerprint silently digested between the Phase 2.2 rebuild and 2026-09-10, pinning a
    # hole exactly as the say-on-pay gate once did.
    out["panel.superinvestor"] = frame_digest(build_superinvestor_feature_panel(
        book, roster, peers, idx, shares_out_history=fund, stock_close=close,
        cusip_map=cusip_map, universe=tickers))
    out["panel.insider"] = frame_digest(build_insider_feature_panel(
        insider, peers, idx, shares_out_history=fund, stock_close=close))

    # ---- composites over the merged panel ---- #
    merged = fp.merge(sp, on=["date", "ticker"], how="outer")
    cfg = yaml.safe_load((ROOT / "configs" / "build_cube.yml").read_text(encoding="utf-8"))
    groups = next(v["composites"]["groups"] for v in cfg.values()
                  if isinstance(v, dict) and "composites" in v)
    comp = build_composites(merged, groups, method="zscore")
    out["panel.composites"] = frame_digest(
        comp[["date", "ticker"] + sorted(c for c in comp.columns if c.startswith("comp_"))])

    # ---- betas + the labels the model actually trains on ---- #
    factor_panel = pd.DataFrame({
        "market": returns.mean(axis=1),
        "momentum": momentum_characteristic(close).mean(axis=1),
    })
    betas = estimate_all_betas(returns, factor_panel)
    out["panel.betas"] = frame_digest(pd.concat(
        {k: v for k, v in betas.items() if isinstance(v, pd.DataFrame)}, axis=1))
    # `min_names` is lowered from the production 20 because this harness runs a
    # 22-name cross-section; it gates which DAYS survive, not how the residual is
    # computed, so the code under test is unaffected.
    # `stock_ret` is REQUIRED: every label is a forward COMPOUNDED total return now rather
    # than a close-to-close price ratio. This fixture is a dividend-free random walk, so
    # `close_split == close_total` here and the two formulations differ only by compounding
    # convention -- which IS a real digest move, and a documented one (see 4e in the plan).
    targets = build_targets_multi(
        close, betas, factor_panel, macro_cols=[],
        horizons=(30, 60, 90), labels=("rank", "zscore"), min_names=5,
        sector_groups={"sector": dict(zip(fund["ticker"], fund["sector"].astype(str)))},
        stock_ret=returns)
    for horizon, by_label in targets.items():
        for label, frame in by_label.items():
            out[f"label.{label}_h{horizon}"] = frame_digest(frame)

    # ---- PRIMITIVES about to be deduplicated (see the module docstring) ---- #
    m, num, den = fx["matrix"], fx["num"], fx["den"]
    ret_fx = fx["returns"]

    out["prim.momentum_characteristic"] = frame_digest(momentum_characteristic(close))
    # the inline copy that `features.mom_12_1` used to carry -- kept as an independent
    # REFERENCE expression so the two can be asserted equal (see
    # test_momentum_dedup_is_provably_identical); the feature now calls the shared helper.
    out["prim.mom_12_1_inline"] = frame_digest(sanitize(close.shift(21) / close.shift(252) - 1.0))
    out["prim.trailing_vol"] = frame_digest(pd.concat({
        "vol_21": sanitize(trailing_vol(returns, 21)),
        "vol_63": sanitize(trailing_vol(returns, 63)),
        "resvol_63": -trailing_vol(returns, 63),
    }, axis=1))
    # A HARD-CODED COPY of `du.daily_returns`, deliberately: it is the independent reference
    # the shared helper is asserted against. In production this is fed `close_total`; the
    # fixture is dividend-free, so the same frame stands in for both bases.
    out["prim.daily_returns"] = frame_digest(close.pct_change(fill_method=None))

    out["prim.forward_windows"] = frame_digest(pd.concat({
        "compound_h20": forward_compound(ret_fx, 20),
        "cumchange_h20": forward_cumchange(ret_fx, 20),
        # `forward_return` is now contract-limited to TOTAL-RETURN INDICES; SPY here stands
        # in for the macro `equity_tr` leg, which is exactly that. The stock labels moved to
        # `forward_compound` and no longer call it.
        "return_h20": forward_return(fx["other_close"].reindex(columns=["SPY"]), 20),
        # the seasonal feature's PARTIAL-window policy (min_periods = round(0.6h)),
        # which differs from the target's full-window policy above
        "compound_h20_partial": forward_compound(ret_fx, 20, min_periods=12),
    }, axis=1))

    # one z + one rank implementation, each call site keeping ITS clip and ITS
    # zero-dispersion policy (see utils/common/xs.py). The two raw `.rank` spellings stay
    # as independent references that xs_rank_pct must reproduce.
    out["prim.xs_standardize"] = frame_digest(pd.concat({
        "factors_xs_z_clip4": xs_z(m, clip=4.0),
        "features_rank": xs_standardize(m, "rank"),
        "features_zscore_clip3": xs_standardize(m, "zscore"),
        "targets_rank_min5": cross_sectional_rank(m, min_names=5),
        "targets_zscore_min5": cross_sectional_zscore(m, min_names=5),
        "winsorize_xs": winsorize_xs(m),
        "rank_pct_plain": m.rank(axis=1, pct=True),
        "rank_pct_average": m.rank(axis=1, pct=True, method="average"),
    }, axis=1))

    out["prim.ratio_helpers"] = frame_digest(pd.concat({
        "ratio": ratio(num, den),
        "ratio_positive_den": ratio(num, den, positive_den=True),
        "ratio_mismatched_cols": ratio(num, fx["den_short"]),
        "clean_ratio": sanitize(num / den),
        "safe": sanitize(num / den),
    }, axis=1))
    out["prim.safe_div"] = frame_digest(pd.DataFrame({
        "plain": safe_div(num["P0"], den["P0"]),
        "positive_den": safe_div(num["P0"], den["P0"], True),
        "none_den": safe_div(num["P0"], None),
    }))

    # `price_column_returns` is GONE. Its whole job was remapping factor name -> price COLUMN
    # ({"oil": "CL=F"}) while the commodity/FX series sat inside the `prices` panel; they now
    # live in `prices_macro` under their factor names, so the remap is the identity and
    # StepCubeTarget._asset_factors just takes the pct_change. Digest the surviving
    # expression, keyed by the factor NAME the panel uses, so the fingerprint still covers the
    # arithmetic that feeds the commodity/currency factors.
    _macro_close = fx["other_close"].rename(
        columns={"SPY": "equity_tr", "CL=F": "oil", "GC=F": "gold", "USDEUR=X": "fx_usdeur"})
    out["prim.macro_factor_returns"] = frame_digest(
        _macro_close[["oil", "gold", "fx_usdeur"]].pct_change())

    out["prim.quarter_features"] = frame_digest(
        _quarter_features(holdings).sort_values(["ticker", "as_of"]).reset_index(drop=True))
    # The two INTERMEDIATES the elite panel turns on, replacing `prim.super_quarter_features`
    # (whose function the Phase 2.2 rebuild deleted). `manager_quarter_state` carries the
    # conviction DENOMINATOR and the concentration legs the selector ranks on;
    # `manager_stock_conviction` carries the portfolio weight and the order-independent rank.
    _book_t = _prepare(attach_tickers(book, cusip_map, tickers))
    _state = manager_quarter_state(_book_t)
    out["prim.super_manager_state"] = frame_digest(
        _state.sort_values(["cik", "period"]).reset_index(drop=True))
    out["prim.super_conviction"] = frame_digest(
        manager_stock_conviction(_book_t, _state)
        .sort_values(["cik", "period", "cusip"]).reset_index(drop=True))

    out["prim.pit"] = frame_digest(pd.concat({
        "shares_outstanding": fundamentals_to_daily(fund, "sharesOutstanding", idx),
        "total_revenue": fundamentals_to_daily(fund, "totalRevenue", idx),
        "free_cashflow": fundamentals_to_daily(fund, "freeCashflow", idx),
        "net_income": fundamentals_to_daily(fund, "netIncome", idx),
        "market_cap": daily_market_cap(fund, close, level_factor=None),
    }, axis=1))

    # `build_peer_relative_panel` called DIRECTLY, including the two degenerate field
    # shapes its coercion path exists for: an all-NaN field and an object-dtype field
    # carrying a stray Python None.
    objf = num.astype(object).copy()
    objf.iloc[4, 5] = None
    objf.iloc[8, 1] = "n/a"
    out["prim.peer_relative_panel"] = frame_digest(build_peer_relative_panel(
        {"plain": num, "all_nan": num * np.nan, "objecty": objf},
        {c: {p: 1.0 for p in num.columns if p != c} for c in num.columns}))

    out["_meta"] = {"tickers": tickers, "seed": SEED, "start": START, "end": END,
                    "fundamentals_rows": int(len(fund)),
                    "fundamentals_cols": int(len(fund.columns))}
    return out


def main() -> None:
    fp = compute()
    BASELINE.write_text(json.dumps(fp, indent=1, sort_keys=True), encoding="utf-8")
    keys = [k for k in fp if not k.startswith("_")]
    print(f"wrote {BASELINE.name}: {len(keys)} fingerprinted outputs")
    for k in sorted(keys):
        d = fp[k]
        print(f"  {k:34} rows={d['rows']:7d} cols={d['cols']:5d} {d['hash'][:12]}")


if __name__ == "__main__":
    main()
