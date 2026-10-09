"""
anchor_check.py  (src/data_extract/utils/prices/anchor_check.py)
------------------------------------------------------------------
Detects a restated `close_split` basis in `prices` from the data. A split or spinoff restates every earlier
bar, but the incremental fetch only notices an event dated after a ticker's last stored bar. Yahoo's current
`close_split` for the earliest bars of a ticker's first stored calendar year (the anchor) is compared with the
stored one: a stored anchor off by a factor means the ticker's whole history sits on a stale basis.

One `actions=False` download per distinct first year. An empty response (a Yahoo failure, or none of the group
served) is missing data, never evidence: it triggers nothing and those tickers are not checked this run. A
ticker absent from a non-empty response, or with no usable common bars, is re-anchored on the next later
calendar year of its own stored data, one grouped download per candidate year; one still unanchored after
`_MAX_FALLBACK_YEARS` tries is named in a WARNING. When more than `ANCHOR_MAX_TRIGGER_SHARE` of the compared
tickers disagree in one run, none is returned (a market-wide restatement or a Yahoo glitch is suspected).
"""

import logging
from collections.abc import Callable

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)

#: Median `abs(fresh / stored - 1)` over the anchor bars above which a ticker is restated. Float noise between
#: two downloads of the same basis is ~1e-15; the smallest split Yahoo records is 1.0017 (a 0.17% move).
ANCHOR_TOLERANCE = 1e-4
#: Only the earliest common bars of the anchor year are compared, so an event inside that year is still seen.
ANCHOR_MAX_BARS = 20
#: Share of the compared tickers above which a run's anchor triggers are all withheld.
ANCHOR_MAX_TRIGGER_SHARE = 0.10
#: Compared tickers below which the breadth guard does not apply (a `-t` run of a few tickers).
ANCHOR_MIN_GUARDED = 20
#: Fewer common anchor bars than this are judged on their maximum deviation, not their median.
_MIN_COMMON_BARS = 3
#: Later calendar years tried, after the first stored one, to anchor a ticker Yahoo no longer serves there.
_MAX_FALLBACK_YEARS = 4

Download = Callable[[list[str], pd.Timestamp, pd.Timestamp], pd.DataFrame]


def _common_dates(old: pd.Series, new: pd.Series) -> pd.Index:
    """The earliest `ANCHOR_MAX_BARS` dates on which both series hold a positive bar."""
    common = old.dropna()[lambda s: s > 0].index.intersection(new.dropna()[lambda s: s > 0].index)
    return common.sort_values()[:ANCHOR_MAX_BARS]


def anchor_restated(stored: pd.Series, fresh: pd.Series, tol: float) -> bool:
    """Whether the stored `close_split` anchor disagrees with Yahoo's (both Series indexed by date).

    NaN and non-positive bars are dropped on both sides; no common date means no evidence and no trigger."""
    common = _common_dates(stored, fresh)
    if common.empty:
        return False
    deviation = (fresh.loc[common] / stored.loc[common] - 1).abs()
    return bool((deviation.median() if len(common) >= _MIN_COMMON_BARS else deviation.max()) > tol)


def anchor_groups(context: Context, keys: list[str]) -> dict[int, list[str]]:
    """`{first stored calendar year: keys}` of the keys that have a stored `prices` row."""
    stats = context.store.key_stats(Tables.prices, "ticker", where={"ticker": keys})
    stats = stats.dropna(subset=["first"])
    years = stats["first"].dt.year
    return {int(year): sorted(stats.loc[years == year, "key"]) for year in sorted(years.unique())}


def stored_anchor(context: Context, keys: list[str], since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """Stored `[ticker, date, close_split]` of `keys` over `[since, until]`; empty when there is none."""
    rows = context.store.load(
        Tables.prices, columns=["ticker", "date", "close_split"], where={"ticker": keys}, since=since, until=until, optional=True
    )
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["ticker", "date", "close_split"])
    return rows.assign(date=pd.to_datetime(rows["date"]))


def _compare_year(
    context: Context, members: list[str], year: int, until: pd.Timestamp, download: Download
) -> tuple[dict[str, float], list[str], int]:
    """`({restated: median fresh/stored factor}, unanchored, compared)` for the bars of calendar `year`.

    Unanchored members have no stored bar in `year` or no usable common bar in a non-empty response; members of
    an empty response are neither (not checked this run)."""
    since, end = pd.Timestamp(year=year, month=1, day=1), min(pd.Timestamp(year=year, month=12, day=31), until)
    stored = stored_anchor(context, members, since, end)
    present = sorted(set(stored["ticker"]))
    unanchored = [member for member in members if member not in present]
    if not present:
        return {}, unanchored, 0
    fresh = download(present, since, end)
    if fresh.empty:
        logger.warning(
            "Anchor download of year %d returned nothing (a Yahoo failure, or none served); not checked this run: %s", year, ", ".join(present)
        )
        return {}, unanchored, 0
    restated: dict[str, float] = {}
    for ticker, old in stored.groupby("ticker"):
        old_close = old.set_index("date")["close_split"]
        new_close = fresh.loc[fresh["ticker"] == ticker].set_index("date")["close_split"]
        common = _common_dates(old_close, new_close)
        if common.empty:
            unanchored.append(str(ticker))
        elif anchor_restated(old_close, new_close, ANCHOR_TOLERANCE):
            restated[str(ticker)] = float((new_close.loc[common] / old_close.loc[common]).median())
    compared = len(members) - len(unanchored)
    logger.info("Anchor check, year %d: %d ticker(s) compared, %d restated", year, compared, len(restated))
    return restated, sorted(unanchored), compared


def anchor_restated_keys(context: Context, keys: list[str], until: pd.Timestamp, download: Download) -> list[str]:
    """Keys whose anchor `close_split` disagrees with a fresh Yahoo download.

    Each key is anchored on its first stored year; a key Yahoo does not serve there moves to its next later
    year, one grouped download per candidate year, up to `_MAX_FALLBACK_YEARS` later years."""
    pending = {ticker: year for year, members in (anchor_groups(context, keys) if keys else {}).items() for ticker in members}
    factors: dict[str, float] = {}
    compared = 0
    for _ in range(_MAX_FALLBACK_YEARS + 1):
        retry: dict[str, int] = {}
        for year in sorted(set(pending.values())):
            members = sorted(ticker for ticker, tried in pending.items() if tried == year)
            if year > until.year:
                retry.update(dict.fromkeys(members, year))
                continue
            restated, unanchored, checked = _compare_year(context, members, year, until, download)
            factors.update(restated)
            compared += checked
            retry.update(dict.fromkeys(unanchored, year + 1))
        pending = retry
    if pending:
        logger.warning("%d ticker(s) could not be anchor-checked, no served year with common bars: %s", len(pending), ", ".join(sorted(pending)))
    if not factors:
        return []
    listed = ", ".join(f"{ticker} x{factor:.4f}" for ticker, factor in sorted(factors.items()))
    if compared >= ANCHOR_MIN_GUARDED and len(factors) > ANCHOR_MAX_TRIGGER_SHARE * compared:
        logger.error(
            "%d of %d anchor-compared ticker(s) disagree with Yahoo (more than %.0f%%): a market-wide restatement or a Yahoo "
            "glitch is suspected, so none is re-pulled this run. Check them against Yahoo; to force one, run "
            "`price-history -t <TICKER> -F` per ticker. Median fresh/stored factor: %s",
            len(factors),
            compared,
            100 * ANCHOR_MAX_TRIGGER_SHARE,
            listed,
        )
        return []
    logger.warning("%d ticker(s) whose anchor close_split disagrees with Yahoo (median fresh/stored factor): %s", len(factors), listed)
    return sorted(factors)
