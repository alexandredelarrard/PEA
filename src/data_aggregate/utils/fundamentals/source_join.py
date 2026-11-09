"""Carry three SEC history columns onto the Sharadar spine of the cube's fundamentals input.

The Sharadar history defines the rows; SEC values join on the exact `(ticker, as_of)` key and
enter under a `sec_` prefix, so no bare name `fundamentals_to_daily` resolves changes meaning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The SEC history columns the cube carries (no feature reads them yet).
SEC_CARRIED_COLUMNS: tuple[str, ...] = ("minorityInterest", "operatingLeaseLiability", "financeLeaseLiability")
SEC_PREFIX = "sec_"
_KEYS = ["ticker", "as_of"]


@dataclass(frozen=True, slots=True)
class SecJoin:
    """The joined frame and how many keys matched on each side."""

    frame: pd.DataFrame
    matched: int
    sharadar_unmatched: int
    sec_unmatched: int


def join_sec_columns(sharadar: pd.DataFrame, sec: pd.DataFrame | None) -> SecJoin:
    """Left-join `sec`'s carried columns onto every `sharadar` row on the exact key, prefixed `sec_`."""
    prefixed = [f"{SEC_PREFIX}{column}" for column in SEC_CARRIED_COLUMNS]
    if sec is None:
        frame = sharadar.assign(**{column: np.nan for column in prefixed})
        return SecJoin(frame, 0, len(sharadar), 0)
    right = sec[[*_KEYS, *SEC_CARRIED_COLUMNS]].rename(columns=dict(zip(SEC_CARRIED_COLUMNS, prefixed, strict=True)))
    right = right.assign(ticker=right["ticker"].astype(str), as_of=pd.to_datetime(right["as_of"]))
    spine = sharadar.assign(_ticker=sharadar["ticker"].astype(str), _as_of=pd.to_datetime(sharadar["as_of"]))
    joined = spine.merge(
        right.rename(columns={"ticker": "_ticker", "as_of": "_as_of"}),
        on=["_ticker", "_as_of"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    hit = joined["_merge"].eq("both")
    matched = int(hit.sum())
    frame = joined.drop(columns=["_ticker", "_as_of", "_merge"])
    frame.index = sharadar.index
    return SecJoin(frame, matched, len(sharadar) - matched, len(right) - matched)
