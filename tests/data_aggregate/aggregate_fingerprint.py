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
def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


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
                "say_on_pay_support_pct": float(rng.uniform(60, 99)),
                "insider_ownership_pct": float(rng.uniform(0.001, 0.08)),
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


def synthetic_attention(tickers: list[str], idx: pd.DatetimeIndex,
                        rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily Wikipedia pageviews + WEEKLY Google Trends (the weekly->daily bounded ffill
    is a real code path). One ticker is deliberately absent from Trends so the rank-blend
    single-source fallback is exercised."""
    wiki = pd.DataFrame({
        "date": np.repeat(idx.to_numpy(), len(tickers)),
        "ticker": np.tile(np.array(tickers), len(idx)),
        "pageviews": rng.lognormal(7.0, 0.7, len(idx) * len(tickers)).round(0),
    })
    weekly = pd.date_range(idx[0], idx[-1], freq="W-SUN")
    gt_tickers = tickers[:-1]
    trends = pd.DataFrame({
        "date": np.repeat(weekly.to_numpy(), len(gt_tickers)),
        "ticker": np.tile(np.array(gt_tickers), len(weekly)),
        "search_interest": rng.integers(0, 101, len(weekly) * len(gt_tickers)).astype(float),
    })
    return wiki, trends


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
    """Forms 3/4/5 rows including the non-discretionary codes the builder must ignore."""
    codes = np.array(["P", "S", "A", "M", "F", "G"])
    n = 40 * len(tickers)
    days = pd.to_datetime(rng.choice(idx.to_numpy(), n))
    return pd.DataFrame({
        "ticker": rng.choice(np.array(tickers), n),
        "filing_date": days,
        "transaction_code": rng.choice(codes, n, p=[0.25, 0.35, 0.15, 0.1, 0.1, 0.05]),
        "value_usd": rng.lognormal(12.0, 1.5, n).round(2),
    })


def synthetic_13f(tickers: list[str], idx: pd.DatetimeIndex,
                  rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Manager-grain 13F: one row per (manager, ticker, quarter). Managers enter and exit
    so new_buyers / exiters / breadth_chg are non-degenerate. Half the CIKs form the
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
    from src.data_aggregate.utils.extras.attention_features import build_combined_attention_panel
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
    from src.data_aggregate.utils.extras.insider_features import build_insider_feature_panel
    from src.data_aggregate.utils.extras.institutional_features import (
        _quarter_features, build_institutional_feature_panel,
    )
    from src.data_aggregate.utils.common.frames import ratio, safe_div, sanitize
    from src.data_aggregate.utils.common.panel import build_peer_relative_panel
    from src.data_aggregate.utils.common.xs import (
        winsorize_xs, xs_rank_pct, xs_standardize, xs_z,
    )
    from src.data_aggregate.utils.fundamentals.sector_features import build_sector_feature_panel
    from src.data_aggregate.utils.extras.short_interest_features import (
        build_short_interest_feature_panel,
    )
    from src.data_aggregate.utils.extras.superinvestor_features import (
        _super_quarter_features, _weight_map, build_superinvestor_feature_panel,
    )
    from src.data_aggregate.utils.target.targets import (
        build_targets_multi, cross_sectional_rank, cross_sectional_zscore,
    )

    rng = _rng()
    fund = fundamentals()
    tickers = sorted(fund["ticker"].unique())
    px = synthetic_prices(tickers, rng)
    close, idx = px["close"], px["close"].index
    returns = close.pct_change(fill_method=None)
    sector_ret = returns.rolling(5).mean().bfill()             # deterministic stand-in
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    div = synthetic_dividends(tickers, idx, rng)
    earn = synthetic_earnings(tickers, idx, rng)
    proxies = synthetic_def14a(tickers, idx, rng)
    neo_comp = synthetic_exec_comp(proxies, rng)
    board = synthetic_directors(proxies, rng)
    board_filled, _ = fill_director_attributes(board)
    board_pay, _ = impute_director_comp(synthetic_director_comp(board, rng))
    wiki, trends = synthetic_attention(tickers, idx, rng)
    short_hist, ftd = synthetic_short_interest(tickers, idx, rng)
    insider = synthetic_insider(tickers, idx, rng)
    holdings, roster = synthetic_13f(tickers, idx, rng)
    fx = primitive_fixtures(rng)

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
    out["panel.attention"] = frame_digest(build_combined_attention_panel(
        wiki, trends, peers, idx))
    out["panel.short_interest"] = frame_digest(build_short_interest_feature_panel(
        short_hist, peers, idx, fails_history=ftd, volume=px["volume"]))
    out["panel.institutional"] = frame_digest(build_institutional_feature_panel(
        holdings, peers, idx, shares_out_history=fund, stock_close=close))
    out["panel.superinvestor"] = frame_digest(build_superinvestor_feature_panel(
        holdings, roster, peers, idx, shares_out_history=fund, stock_close=close))
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
    out["prim.super_quarter_features"] = frame_digest(
        _super_quarter_features(holdings, _weight_map(roster))
        .sort_values(["ticker", "as_of"]).reset_index(drop=True))

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
