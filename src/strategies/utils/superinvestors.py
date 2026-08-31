"""
superinvestors.py  (src/strategies/utils/superinvestors.py)
-------------------------------------------------------------
Pure aggregation helpers + IO for the `super_investors` strategy sleeve: turn each raw
"who is buying/selling this ticker" source (issuer insiders, 13D activists, elite-manager 13F
holders, RegSHO short volume) into a per (ticker, as_of) panel stamped on ITS OWN
public-disclosure date, then outer-merge them into one wide panel keyed by (ticker, as_of).
`as_of` is always the date the information became publicly knowable, never the underlying
trade/period date, so a signal built on this panel can never look ahead of what a trader could
actually see that day (mirrors the point-in-time convention in
`src/data_aggregate/utils/extras/{insider,institutional,superinvestor}_features.py`, which this
module cannot import -- `strategies` and `data_aggregate` are separate pipeline subfolders).

Every `_aggregate_*` function is a pure DataFrame -> DataFrame transform (unit-tested in
`tests/strategies/test_superinvestors.py`); `load_positions_panel` is the only IO entry point.
"""
from __future__ import annotations

import json
import logging
import re
import pandas as pd
from pathlib import Path

from src.context import Context
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


class SuperinvestorRosterError(RuntimeError):
    """The superinvestors roster JSON exists but cannot be parsed."""

def _aggregate_insiders(df: pd.DataFrame) -> pd.DataFrame:
    """Officer/director/10%-owner Form 3/4/5 transactions -> per (ticker, as_of) net $
    conviction. Restricted to OPEN-MARKET codes P (buy) / S (sell) -- grants, option
    exercises.

    `insider_pct_sold`: shares sold as a % of the pooled insider stake in this ticker BEFORE
    the sale. `shares_owned_after` is each seller's balance AFTER their own row's sale."""

    d = df.copy()

    d["as_of"] = pd.to_datetime(d["filing_date"], errors="coerce")
    d = d.dropna(subset=["ticker", "as_of"])
    code = d["transaction_code"].astype("string").str.upper().str.strip()
    is_buy, is_sell = code == "P", code == "S"
    d = d[is_buy | is_sell]
  
    is_buy, is_sell = is_buy.loc[d.index], is_sell.loc[d.index]
    value = pd.to_numeric(d["value_usd"], errors="coerce").fillna(0.0)
    shares = pd.to_numeric(d["shares"], errors="coerce").fillna(0.0)
    owned_after = pd.to_numeric(d["shares_owned_after"], errors="coerce")

    g = (d.assign(insider_buy_value=value.where(is_buy, 0.0),
                  insider_sell_value=value.where(~is_buy, 0.0),
                  insider_buy_shares=shares.where(is_buy, 0.0),
                  insider_sell_shares=shares.where(~is_buy, 0.0),
                  _owned_after_sell=owned_after)
         .groupby(["ticker", "as_of"], as_index=False)
         .agg(insider_buy_value=("insider_buy_value", "sum"),
              insider_sell_value=("insider_sell_value", "sum"),
              insider_buy_shares=("insider_buy_shares", "sum"),
              insider_sell_shares=("insider_sell_shares", "sum"),
              insider_n_transactions=("transaction_sk", "count"),
              _owned_after_sell=("_owned_after_sell", "sum")))
    g["insider_net_value"] = g["insider_buy_value"] - g["insider_sell_value"]

    sell_denom = (g["_owned_after_sell"] + g["insider_sell_shares"] - g["insider_buy_shares"]).replace(0.0, pd.NA)
    g["insider_pct_moved"] = (-1*g["insider_sell_shares"] + g["insider_buy_shares"]) / sell_denom
    return g


def _aggregate_activist(df: pd.DataFrame) -> pd.DataFrame:
    """SC 13D Item 5(c) 60-day trade log -> per (ticker, as_of) activist net $ conviction.
    `as_of` = filing_date, NEVER `trade_date`: a disclosed trade can be up to 60 days old
    by filing time, so keying on `trade_date` would leak the position ahead of its actual
    public disclosure."""
    cols = ["ticker", "as_of", "activist_buy_value", "activist_sell_value",
            "activist_net_value", "activist_buy_shares", "activist_sell_shares",
            "activist_n_transactions"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d["as_of"] = pd.to_datetime(d["filing_date"], errors="coerce")
    d = d.dropna(subset=["ticker", "as_of"])
    side = d["transaction_type"].astype("string").str.casefold()
    is_buy, is_sell = side == "buy", side == "sell"
    qty = pd.to_numeric(d["quantity"], errors="coerce").fillna(0.0)
    px = pd.to_numeric(d["price_per_share"], errors="coerce").fillna(0.0)
    value = qty * px
    g = (d.assign(activist_buy_value=value.where(is_buy, 0.0),
                  activist_sell_value=value.where(is_sell, 0.0),
                  activist_buy_shares=qty.where(is_buy, 0.0),
                  activist_sell_shares=qty.where(is_sell, 0.0))
         .groupby(["ticker", "as_of"], as_index=False)
         .agg(activist_buy_value=("activist_buy_value", "sum"),
              activist_sell_value=("activist_sell_value", "sum"),
              activist_buy_shares=("activist_buy_shares", "sum"),
              activist_sell_shares=("activist_sell_shares", "sum"),
              activist_n_transactions=("trade_seq", "count")))
    g["activist_net_value"] = g["activist_buy_value"] - g["activist_sell_value"]
    return g[cols]


_SUPER_LEVEL_COLS = ["superinvestor_shares", "superinvestor_value", "superinvestor_n_managers"]
_SUPER_FLOW_COLS = ["superinvestor_buy_shares", "superinvestor_sell_shares",
                    "superinvestor_init_shares", "superinvestor_n_new", "superinvestor_n_exited",
                    "superinvestor_n_increased", "superinvestor_n_decreased"]


def _published_calendar(d: pd.DataFrame) -> pd.DataFrame:
    """Each manager's filing calendar in PUBLICATION order, keeping only reports that ADVANCE
    that manager's period high-water mark.

    13F filing dates are NOT monotonic in period: on the live roster 5 of 67 managers file out
    of order, and one (Egerton, CIK 0001581811) dumped five quarters -- 2020-09-30 through
    2021-09-30 -- on a single day, 2022-08-03, years late. A report disclosing an OLDER period
    than one already public tells an observer nothing new about the manager's current position:
    it is superseded on arrival. Dropping it is what keeps the accumulated level from being
    revised backwards through states that were never real (which surfaced as 6 impossible
    NEGATIVE share levels before this filter existed). Reports sharing one filing date are all
    kept -- their deltas telescope within that day's single `as_of` row."""
    cal = (d[["cik", "period", "filing_date"]].drop_duplicates(["cik", "period"])
           .sort_values(["cik", "filing_date", "period"]))
    return cal[cal["period"] >= cal.groupby("cik")["period"].cummax()]


def _superinvestor_events(d: pd.DataFrame) -> pd.DataFrame:
    """Manager-grain 13F rows -> one row per (cik, ticker, period) on that MANAGER'S OWN filing
    calendar, with a `shares=0` row materialized wherever a previously-held ticker is missing
    from a later 13F that manager did file.

    That materialization is what makes an EXIT visible at all: a 13F lists only non-zero
    holdings, so a manager dropping a name emits no row and a plain per-pair diff would leave
    the position standing forever. Reindexing each pair over the manager's filing calendar (not
    a global one -- managers file on their own dates) also handles RE-ENTRY for free, which the
    previous pairwise scan did not. Rows that are 0 both before and after carry no information
    and are dropped, so a long-exited pair costs nothing.

    Everything is ordered by FILING date, not period, so the deltas are accumulated downstream
    in the same order they became public (see `_published_calendar`)."""
    cal = _published_calendar(d)
    d = d.merge(cal[["cik", "period"]], on=["cik", "period"], how="inner")
    pairs = d[["cik", "ticker"]].drop_duplicates()
    first_period = (d.groupby(["cik", "ticker"], as_index=False)["period"].min()
                    .rename(columns={"period": "_first_period"}))

    full = (pairs.merge(cal, on="cik", how="left")
            .merge(first_period, on=["cik", "ticker"], how="left"))
    full = full[full["period"] >= full["_first_period"]]        # never fabricate pre-history
    full = (full.merge(d[["cik", "ticker", "period", "shares", "value_usd"]],
                       on=["cik", "ticker", "period"], how="left")
            .fillna({"shares": 0.0, "value_usd": 0.0})
            .sort_values(["cik", "ticker", "filing_date", "period"]))   # publication order

    grp = full.groupby(["cik", "ticker"], sort=False)
    prev_shares = grp["shares"].shift(1)
    prev_value = grp["value_usd"].shift(1)
    is_pair_start = prev_shares.isna()
    # a manager's FIRST filing in our extraction window: the stake was accumulated before our
    # data begins, so it is an initialization, not an observed purchase
    is_mgr_start = full["period"] == full.groupby("cik")["period"].transform("min")
    ps, pv = prev_shares.fillna(0.0), prev_value.fillna(0.0)

    full = full.assign(
        _d_shares=full["shares"] - ps,
        _d_value=full["value_usd"] - pv,
        _d_holder=(full["shares"] > 0).astype(float) - (ps > 0).astype(float),
        _is_init=is_pair_start & is_mgr_start,
        _is_new=is_pair_start & ~is_mgr_start,                       # genuinely fresh position
        _is_exit=~is_pair_start & (full["shares"] == 0.0) & (ps > 0),
        _is_increased=~is_pair_start & (full["shares"] - ps > 0),
        _is_decreased=~is_pair_start & (full["shares"] - ps < 0) & (full["shares"] > 0))
    return full[(full["shares"] > 0) | (ps > 0)]                     # drop 0 -> 0 non-events


def _expand_daily(events: pd.DataFrame, end: pd.Timestamp,
                  group_keys: list[str]) -> pd.DataFrame:
    """Per-group event rows -> a DAY-BY-DAY panel: levels forward-filled, flows zero-filled.

    The level is a step function that only moves on a filing, but the panel must answer "what
    was held in this name on THIS day" for every day -- otherwise a strategy reading the merged
    panel sees NaN on the ~99% of days nobody filed. Business-day index, unioned with the actual
    event dates so a filing stamped on a weekend/holiday is never dropped off the grid.

    `group_keys` is `["ticker"]` for the pooled cohort book and `["cik", "ticker"]` when each
    manager is carried as its own portfolio."""
    out = []
    for key, t in events.groupby(group_keys, sort=False):
        t = t.set_index("as_of").sort_index()
        idx = pd.bdate_range(t.index.min(), max(end, t.index.max())).union(t.index)
        r = t.reindex(idx)
        r[_SUPER_LEVEL_COLS] = r[_SUPER_LEVEL_COLS].ffill()
        r[_SUPER_FLOW_COLS] = r[_SUPER_FLOW_COLS].fillna(0.0)
        for name, value in zip(group_keys, key if isinstance(key, tuple) else (key,)):
            r[name] = value
        out.append(r.rename_axis("as_of").reset_index())
    return pd.concat(out, ignore_index=True)


def _aggregate_superinvestors(df: pd.DataFrame, end: pd.Timestamp | None = None,
                              by_cik: bool = False) -> pd.DataFrame:
    """Elite-manager 13F holdings -> a DAILY panel of the aggregate stake and the
    quarter-over-quarter movement behind it, keyed (ticker, as_of) -- or (cik, ticker, as_of)
    when `by_cik` is set, which carries each manager as its own portfolio so a caller can
    replay them one at a time instead of as one pooled book.

    Two managers report the SAME quarter on DIFFERENT filing dates, and between filings nothing
    is republished -- so an event-grain groupby answers "who filed today", not "who holds this
    today", and every day in between is simply missing. Both are fixed here by construction:

      LEVEL (`superinvestor_shares` / `_value` / `_n_managers`) is accumulated from per-manager
      DELTAS and then forward-filled onto a business-day index, so a manager who last filed
      three months ago still counts at their last-known size on every day since. Summing
      telescoping per-manager deltas is exactly equal to forward-filling each manager and then
      summing, but it collapses the manager dimension BEFORE the daily expansion -- peak memory
      is (tickers x days), not (managers x tickers x days).

      FLOWS (`_buy_shares` / `_sell_shares` / the counts) land on the filing date that disclosed
      them and are 0 on every other day.

    Because the level is the running sum of exactly those flows, the panel satisfies
        shares(t) == shares(t-1) + buy_shares(t) - sell_shares(t)
    on EVERY consecutive pair of days, by construction rather than by luck.

    One wrinkle that invariant forces into the open: a manager's first filing in our extraction
    window moves the level off 0, but is not an observed purchase (the stake predates our data).
    It is therefore counted in `buy_shares` -- so the identity holds exactly -- and reported
    separately in `superinvestor_init_shares` so a caller can net out the warm-up. `n_new`
    stays clean: it counts only positions opened by a manager who already had filing history."""

    d = df.copy()

    d["cik"] = d["cik"].map(pad_cik)
    d["period"] = pd.to_datetime(d["period"], errors="coerce").dt.normalize()
    d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce").dt.normalize()
    d["shares"] = pd.to_numeric(d["shares"], errors="coerce").fillna(0.0)
    d["value_usd"] = pd.to_numeric(d["value_usd"], errors="coerce").fillna(0.0)
    d = d.dropna(subset=["ticker", "cik", "period", "filing_date"])
    
    # amendments: keep the last-filed row per (cik, ticker, period)
    d = d.sort_values("filing_date").drop_duplicates(["cik", "ticker", "period"], keep="last")

    ev = _superinvestor_events(d)
    ev["as_of"] = ev["filing_date"]
    delta = ev["_d_shares"]
    ev = ev.assign(_buy=delta.clip(lower=0.0), _sell=(-delta).clip(lower=0.0),
                   _init=delta.where(ev["_is_init"], 0.0).clip(lower=0.0))

    # collapse to the requested grain -> deltas + flows, then accumulate into the level.
    # `by_cik` keeps each manager separate, so the same panel can be replayed as ONE pooled
    # book or as 60-odd single-manager portfolios; the arithmetic below is identical either
    # way, only the grouping key changes.
    group_keys = ["cik", "ticker"] if by_cik else ["ticker"]
    agg = (ev.groupby([*group_keys, "as_of"], as_index=False)
           .agg(_d_shares=("_d_shares", "sum"), _d_value=("_d_value", "sum"),
                _d_holder=("_d_holder", "sum"),
                superinvestor_buy_shares=("_buy", "sum"),
                superinvestor_sell_shares=("_sell", "sum"),
                superinvestor_init_shares=("_init", "sum"),
                superinvestor_n_new=("_is_new", "sum"),
                superinvestor_n_exited=("_is_exit", "sum"),
                superinvestor_n_increased=("_is_increased", "sum"),
                superinvestor_n_decreased=("_is_decreased", "sum"))
           .sort_values([*group_keys, "as_of"]))
    by_group = agg.groupby(group_keys, sort=False)
    agg["superinvestor_shares"] = by_group["_d_shares"].cumsum()
    agg["superinvestor_value"] = by_group["_d_value"].cumsum()
    agg["superinvestor_n_managers"] = by_group["_d_holder"].cumsum()

    keep = [*group_keys, "as_of", *_SUPER_LEVEL_COLS, *_SUPER_FLOW_COLS]
    out = _expand_daily(agg[keep], end or agg["as_of"].max(), group_keys)
    out["superinvestor_net_shares"] = (out["superinvestor_buy_shares"]
                                       - out["superinvestor_sell_shares"])
    return out.sort_values([*group_keys, "as_of"]).reset_index(drop=True)


def _aggregate_shorts(df: pd.DataFrame) -> pd.DataFrame:
    """RegSHO daily short-sale volume -> per (ticker, as_of) short pressure. `as_of = date +
    1 business day`: FINRA disseminates each day's file only the following morning
    (fetch_short_interest.py), so `date` itself is not yet public on `date`."""
    cols = ["ticker", "as_of", "short_volume", "total_volume", "short_ratio"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d["as_of"] = pd.to_datetime(d["date"], errors="coerce") + pd.tseries.offsets.BDay(1)
    d = d.dropna(subset=["ticker", "as_of"])
    g = (d.groupby(["ticker", "as_of"], as_index=False)
         .agg(short_volume=("short_volume", "sum"), total_volume=("total_volume", "sum")))
    g["short_ratio"] = g["short_volume"] / g["total_volume"].replace(0.0, pd.NA)
    return g[cols]


def merge_positions_panel(insiders: pd.DataFrame,
                          superinvestors: pd.DataFrame, shorts: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge the four per-(ticker, as_of) panels into one wide panel keyed by
    (ticker, as_of). A ticker/date with only one source present keeps NaN in the others --
    absence of a filing on a given day is not the same as zero conviction; later signal
    construction decides how to fill it. Each input frame is already unique on
    (ticker, as_of) (grouped upstream), so every merge step is 1:1 -- no row blow-up."""
    panels = [p for p in (insiders, superinvestors, shorts) if not p.empty]
    if not panels:
        return pd.DataFrame(columns=["ticker", "as_of"])
    out = panels[0]
    for p in panels[1:]:
        out = out.merge(p, on=["ticker", "as_of"], how="outer")
    return out.sort_values(["ticker", "as_of"]).reset_index(drop=True)


def _strip_json_line_comments(text: str) -> str:
    """Make the hand-curated roster parseable as JSON. The file is maintained by commenting
    managers in and out, which strict JSON does not survive, so two relaxations are applied:

      * drop lines whose FIRST non-whitespace characters are `//`. Deliberately anchored at the
        start of the line -- a blanket `//` strip would truncate the
        `"https://www.dataroma.com/..."` values the same file carries;
      * drop a trailing comma before `}` / `]`. Commenting out the LAST entry of the map
        necessarily orphans the comma on the line above it, so this is not an edge case but
        the normal result of the editing workflow. A comma inside a quoted name is untouched:
        the pattern only fires on a comma followed by whitespace and a closing bracket.
    """
    no_comments = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("//"))
    return re.sub(r",(\s*[}\]])", r"\1", no_comments)


def _load_superinvestor_roster(context: Context) -> dict[str, str]:
    """`{padded_cik: name}` roster map, read directly from the JSON rather than through
    `data_extract.fetch_superinvestors.load_superinvestors` -- cross-importing between
    `src/` pipeline subfolders is not allowed (same choice as
    `data_aggregate/transformers/step_cube_extras.py::_load_superinvestor_roster`).

    A MISSING file is a legitimate "not built yet" state -> empty roster. A file that exists
    but does not parse is NOT: it silently reduced the whole sleeve to zero managers (the live
    roster had one hand-commented entry, and the swallowed JSONDecodeError made 83 managers
    read as 0 with no error anywhere), so it raises instead."""
    path = context.paths["DATA_STORE"] / Path(context.config.local.paths.superinvestors)
    if not path.exists():
        logger.warning("Superinvestors roster missing at %s -> no elite managers; run "
                       "fetch_superinvestors.build_superinvestors_json.", path)
        return {}
    try:
        roster = json.loads(_strip_json_line_comments(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as e:
        raise SuperinvestorRosterError(f"Superinvestors roster at {path} is unreadable: {e}") from e
    return {pad_cik(k): v for k, v in (roster.get("cik_to_name") or {}).items()}