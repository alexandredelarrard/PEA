"""Company anchor: the frozen `company_id` of each universe ticker and the CIKs its CIK history starts from.

The id is the one stored on the ticker's `universe_tickers` row, else the one its stored `universe_cik_symbol` rows carry. A
ticker with neither is new: the CIK history mints its id (`E` + its CIK current at `MINT_CUTOFF`), anchored from the next build.
Seeds: the roster CIK (U-49: the newest stored CIK window, else the SEC `company_tickers` list, else Sharadar; never the
`universe_tickers.cik` that identity itself publishes) and the CIK of Sharadar's `secfilings` link, read from the newest row of
the ticker's permaticker.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CompanyAnchor:
    """`company_id`: the frozen id of each identified universe ticker; `sharadar_cik`: its Sharadar CIK seed; `ended_on`:
    `roster_cik`: the CIK each ticker's CIK history starts from (U-49); `ended_on`: the end date of each ended company (D-4), which every later stage closes its open rows at (REQ-006)."""

    company_id: dict[str, str]
    sharadar_cik: dict[str, str]
    roster_cik: dict[str, str]
    ended_on: dict[str, pd.Timestamp] = field(default_factory=dict)


def anchor_company_ids(universe_rows: pd.DataFrame | None, lineage_rows: pd.DataFrame | None, tickers: Collection[str]) -> dict[str, str]:
    """`{universe ticker: frozen company_id}` for the tickers already identified; the others are minted by the CIK history."""
    # the ids stored on the published company rows
    anchors: dict[str, str] = {}
    if universe_rows is not None and "company_id" in universe_rows.columns:
        stored = universe_rows[universe_rows["company_id"].notna() & universe_rows["ticker"].isin(list(tickers))]
        anchors.update(zip(stored["ticker"].astype(str), stored["company_id"].astype(str), strict=True))

    # else the id the stored lineage rows of the ticker carry
    if lineage_rows is not None and not lineage_rows.empty:
        owned = lineage_rows[lineage_rows["canonical_ticker"].isin(list(tickers)) & lineage_rows["company_id"].notna()]
        by_ticker = owned.drop_duplicates("canonical_ticker").set_index("canonical_ticker")["company_id"].astype(str)
        anchors.update({str(ticker): entity for ticker, entity in by_ticker.items() if ticker not in anchors})
    return anchors


def sharadar_ciks(universe_rows: pd.DataFrame | None, sharadar: pd.DataFrame | None) -> dict[str, str]:
    """`{universe ticker: 10-digit CIK of its Sharadar secfilings link}`, from the newest row of its permaticker."""
    # the newest Sharadar row of each permaticker
    if universe_rows is None or sharadar is None or sharadar.empty or "permaticker" not in universe_rows.columns:
        return {}
    newest = sharadar.sort_values("lastupdated", kind="mergesort").drop_duplicates("permaticker", keep="last")
    ciks = newest["secfilings"].astype("string").str.extract(r"CIK=(\d+)", expand=False).str.zfill(10)

    # matched to the universe through the permaticker, never the ticker spelling
    by_permaticker = dict(zip(pd.to_numeric(newest["permaticker"], errors="coerce"), ciks, strict=True))
    permatickers = pd.to_numeric(universe_rows["permaticker"], errors="coerce")
    pairs = zip(universe_rows["ticker"].astype(str), permatickers.map(by_permaticker), strict=True)
    return {ticker: str(cik) for ticker, cik in pairs if isinstance(cik, str) and cik}


def roster_ciks(
    tickers: Collection[str], lineage_rows: pd.DataFrame | None, sec_tickers: pd.DataFrame | None, sharadar_cik: Mapping[str, str]
) -> dict[str, str]:
    """`{universe ticker: roster CIK}`: the CIK of its newest stored `cik_window`, else its SEC `company_tickers` CIK (a new
    ticker, a cold start), else its Sharadar `secfilings` CIK (a company the SEC list no longer carries)."""
    # the CIK of each ticker's newest stored window, the SEC list
    stored = newest_window_ciks(lineage_rows) if lineage_rows is not None else {}
    listed: dict[str, str] = {}
    if sec_tickers is not None and not sec_tickers.empty:
        digits = sec_tickers["cik"].astype("string").str.split(".", n=1).str[0].str.zfill(10)
        listed = dict(zip(sec_tickers["ticker"].astype(str), digits, strict=True))

    # in that order, then Sharadar
    pairs = ((ticker, stored.get(ticker) or listed.get(ticker) or sharadar_cik.get(ticker)) for ticker in tickers)
    return {ticker: cik for ticker, cik in pairs if cik}


def newest_window_ciks(lineage_rows: pd.DataFrame) -> dict[str, str]:
    """`{universe ticker: CIK of its newest cik_window}` in `universe_cik_symbol` rows (the current CIK identity publishes)."""
    windows = lineage_rows[lineage_rows["role"].eq("cik_window") & lineage_rows["canonical_ticker"].notna()]
    newest = windows.assign(valid_from=pd.to_datetime(windows["valid_from"])).sort_values("valid_from", kind="mergesort")
    return newest.drop_duplicates("canonical_ticker", keep="last").set_index("canonical_ticker")["cik"].astype(str).to_dict()
