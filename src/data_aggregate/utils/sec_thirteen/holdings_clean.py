"""Shared 13F date normalization and per-batch company tagging and value repair.

`tag_company_lines` tags each filed row and puts a company's share classes in canonical-class shares, one row per filed
row; `sum_company_rows` then merges a company's rows onto its canonical CUSIP per resolved book.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Mapping, Sequence
from contextlib import closing
from typing import Any

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_ROLES
from src.data_aggregate.utils.sec_thirteen.value_basis import log_register, prepare_public_value_quotes, repair_value_basis
from src.utils.identity_tables import TAGGED_ROLES, CusipTags, tag_cusips

logger = logging.getLogger(__name__)


#: Coerced to a midnight `datetime64` on both manifests and positions.
_DATES = ("period", "filing_date")
#: 13F share legs counted in the canonical class (x the dated ratio); every other amount is summed as filed.
SHARE_LEGS = ("shares", "call_shares", "put_shares")
#: 13F amounts of one position row.
AMOUNT_LEGS = (*SHARE_LEGS, "value_usd", "call_value", "put_value", "debt_prn", "debt_value", "other_value")
#: The column `tag_company_lines` adds: the company's canonical CUSIP on the row's period (null off the company lines).
COMPANY_CUSIP = "company_cusip"


def tag_company_lines(book: pd.DataFrame, lines: pd.DataFrame, cmap: pd.DataFrame | None) -> pd.DataFrame:
    """Every row of `book` (13F positions dated by `period`) tagged by `tag_cusips`, one row per filed row. A row a dated
    company line tags has its share legs in canonical-class shares (x the line's dated `conversion_ratio`) and the
    company's canonical CUSIP of the period in `COMPANY_CUSIP`; every other row keeps its legs and a null there."""
    tagged = book.assign(ticker=tag_cusips(book, lines, cmap), **{COMPANY_CUSIP: pd.Series(None, index=book.index, dtype=object)})
    hits = company_lines(tagged, lines)
    for leg in (leg for leg in SHARE_LEGS if leg in tagged):
        tagged.loc[hits.index, leg] = pd.to_numeric(tagged.loc[hits.index, leg], errors="coerce") * hits["ratio"]
    tagged.loc[hits.index, COMPANY_CUSIP] = hits["target"]
    return tagged


def sum_company_rows(tagged: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    """`tag_company_lines` rows with the rows of one company and `key` merged into ONE row on the company's canonical
    CUSIP: metadata of the canonical row first, amounts summed (unknown when one row is unknown). Every other row is kept
    as it is; `COMPANY_CUSIP` is dropped."""
    # compute: the company rows, the canonical CUSIP's row first in each group
    columns = [column for column in tagged.columns if column != COMPANY_CUSIP]
    hit = tagged[COMPANY_CUSIP].notna()
    if not hit.any():
        return tagged[columns]
    first = tagged.loc[hit, COMPANY_CUSIP].ne(tagged.loc[hit, "cusip"])
    rows = tagged.loc[hit].assign(_first=first).sort_values([*key, "_first", "cusip"], kind="stable")

    # merge: one row per company (metadata of its first row, amounts summed, unknown when one row is unknown)
    grouped = rows.groupby(list(key), sort=False)
    merged = grouped.first()
    amounts = [leg for leg in (*AMOUNT_LEGS, *(c for c in rows.columns if c.startswith("raw_"))) if leg in rows]
    merged[amounts] = grouped[amounts].sum(min_count=1).where(grouped[amounts].count().eq(grouped.size(), axis=0))
    merged["cusip"] = merged[COMPANY_CUSIP]
    return pd.concat([tagged.loc[~hit, columns], merged.reset_index()[columns]], ignore_index=True)


def company_lines(tagged: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Per row of `tagged` that a dated company line tags (its index): the line's `ratio` and `target`, the canonical
    CUSIP of the company on the period (canonical current before predecessor, then the latest start; the row's own CUSIP
    when the company has no canonical line that day)."""
    # clean: the rows and the tagged lines with their dates
    rows = pd.DataFrame({"row": tagged.index, "cusip": tagged["cusip"].astype(str).str.strip().str.upper(), "ticker": tagged["ticker"]})
    rows["day"] = pd.to_datetime(tagged["period"]).to_numpy()
    work = lines[lines["lineage_role"].isin(TAGGED_ROLES)].assign(
        start=pd.to_datetime(lines["valid_from"]), end=pd.to_datetime(lines["valid_to"]), cusip=lines["cusip"].astype(str)
    )

    # compute: the line holding each row's date for its own company, then the company's canonical CUSIP that day
    held = rows.merge(work, left_on=["cusip", "ticker"], right_on=["cusip", "canonical_company"])
    held = held[held["start"].isna() | (held["start"] <= held["day"])]
    held = held[held["end"].isna() | (held["day"] < held["end"])].drop_duplicates("row")
    canonical = work[work["lineage_role"].isin(CANONICAL_ROLES)].assign(rank=lambda w: w["lineage_role"].ne(CANONICAL_ROLES[1]))
    pairs = held[["row", "canonical_company", "day"]].merge(canonical[["canonical_company", "cusip", "rank", "start", "end"]], on="canonical_company")
    pairs = pairs[(pairs["start"].isna() | (pairs["start"] <= pairs["day"])) & (pairs["end"].isna() | (pairs["day"] < pairs["end"]))]
    target = pairs.sort_values(["rank", "start"], ascending=[True, False], na_position="last").drop_duplicates("row").set_index("row")["cusip"]

    # post-process: one (ratio, target) per tagged row
    out = held.set_index("row")[["conversion_ratio", "cusip"]].rename(columns={"conversion_ratio": "ratio"})
    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce").fillna(1.0)
    out["target"] = target.reindex(out.index).fillna(out["cusip"])
    return out[["ratio", "target"]]


def clean_filing_dates(df_filings: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize source dates without changing accession grain or missing quantities."""
    for df in (df_filings, df_holdings):
        for column in _DATES:
            df[column] = pd.to_datetime(df[column])
    return df_filings, df_holdings


def clean_public_batches(
    packets: Generator[tuple[pd.DataFrame, pd.DataFrame]],
    tags: CusipTags,
    universe: Sequence[str] | None,
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, Any] | None,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare shared quotes once, then per batch tag each filed row and repair raw accession values before
    reconciliation, retaining unsupported rows. `tag_company_lines` tags every row (the stored ticker is not read),
    one row per stored row so the resolver's count checks hold: a company's share classes in canonical-class shares
    (x the dated ratio, before the value repair) with its canonical CUSIP in `COMPANY_CUSIP`. No ticker where nothing
    tags a row, nor outside `universe` when given; unmapped and off-universe positions stay in the book."""
    period_quotes = prepare_public_value_quotes(valuation_close, value_basis)
    members = None if universe is None else set(universe)
    with closing(packets):
        for df_filings, df_raw in packets:
            # Tags and canonical-class shares, then one vector repair per batch; an empty batch passes through unrepaired.
            df_tagged = tag_company_lines(df_raw, tags.lines, tags.cmap)
            tickers = df_tagged["ticker"].astype(object)
            keep = tickers.notna() if members is None else tickers.isin(members)
            df_holdings, df_register = repair_value_basis(
                df_tagged.assign(ticker=tickers.where(keep, None)), valuation_close, value_basis, period_quotes=period_quotes
            )
            del df_raw, df_tagged

            # Report the unit decisions, then null the excluded amounts; the rows stay in the book.
            log_register(df_register, float(df_holdings.value_usd.sum()))
            excluded = df_holdings.get("value_basis_excluded", pd.Series(False, index=df_holdings.index))
            df_holdings.loc[excluded, ["shares", "value_usd"]] = np.nan
            yield df_filings, df_holdings
            del df_filings, df_holdings
