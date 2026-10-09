"""
anchor_check.py  (src/data_extract/utils/prices/anchor_check.py)
------------------------------------------------------------------
Detects a restated `close_split` basis in `prices` from the data. A split or spinoff restates every earlier
bar, but the incremental fetch only notices an event dated after a ticker's last stored bar. Yahoo's current
`close_split` for the bars of a ticker's first stored calendar year (the anchor) is compared with the stored
one: a stored anchor off by a factor means the ticker's whole history sits on a stale basis.

One `actions=False` download per distinct first year. A missing or empty response is missing data, never
evidence: it triggers nothing.
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
#: Fewer common anchor bars than this are judged on their maximum deviation, not their median.
_MIN_COMMON_BARS = 3

Download = Callable[[list[str], pd.Timestamp, pd.Timestamp], pd.DataFrame]


def anchor_restated(stored: pd.Series, fresh: pd.Series, tol: float) -> bool:
    """Whether the stored `close_split` anchor disagrees with Yahoo's (both Series indexed by date).

    NaN and non-positive bars are dropped on both sides; no common date means no evidence and no trigger."""
    old, new = stored.dropna(), fresh.dropna()
    common = old[old > 0].index.intersection(new[new > 0].index)
    if common.empty:
        return False
    deviation = (new.loc[common] / old.loc[common] - 1).abs()
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


def anchor_restated_keys(context: Context, keys: list[str], until: pd.Timestamp, download: Download) -> list[str]:
    """Keys whose first-year `close_split` disagrees with a fresh Yahoo download, one download per first year."""
    triggered: list[str] = []
    groups = anchor_groups(context, keys) if keys else {}
    for year, members in groups.items():
        since, end = pd.Timestamp(year=year, month=1, day=1), min(pd.Timestamp(year=year, month=12, day=31), until)
        fresh = download(members, since, end)
        if fresh.empty:
            logger.warning("Anchor download for %d ticker(s) of first year %d returned nothing; not checked", len(members), year)
            continue
        stored = stored_anchor(context, members, since, end)
        hits = [
            str(ticker)
            for ticker, old in stored.groupby("ticker")
            if not (new := fresh[fresh["ticker"] == ticker]).empty
            and anchor_restated(old.set_index("date")["close_split"], new.set_index("date")["close_split"], ANCHOR_TOLERANCE)
        ]
        logger.info("Anchor check, first year %d: %d ticker(s) compared, %d restated", year, len(members), len(hits))
        triggered.extend(hits)
    if triggered:
        logger.warning("%d ticker(s) whose first-year close_split disagrees with Yahoo: %s", len(triggered), ", ".join(sorted(triggered)))
    return sorted(triggered)
