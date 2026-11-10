"""The `universe_cusip` build on in-memory inputs, through the production stage functions.

`derive_cusip_history` is the compute of `StepExtractIdentity.run_cusip_history` (clean -> lines -> facts -> roles -> rows
-> flags) on frames instead of store reads: the FTD lines, the lineage, the roster, the curation, and optionally the SEC
listing and the FINRA days. No DB, no network.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zipfile import ZipFile

import pandas as pd

from src.constants.constants import FTD_ZIP_NAME_TEMPLATE
from src.data_extract.transformers.step_extract_identity import StepExtractIdentity
from src.data_extract.utils.common.security_lines import SecurityManual, tape_intervals
from src.data_extract.utils.identity.company_anchor import CompanyAnchor
from src.data_extract.utils.identity.cusip_history import (
    CusipFlags,
    CusipHistory,
    CusipRoles,
    canonical_segments,
    cusip_facts,
    cusip_flags,
    cusip_issuers,
    cusip_rows,
    extend_lines,
    finalise_rows,
    finra_evidence,
    flag_items,
    ftd_lines,
    line_ends,
    lineage_view,
    prepare_observations,
    ratio_step_flags,
    role_segments,
    row_context,
    sec_listing,
)
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from src.utils.string import pad_cik, run_date


def derive_cusip_history(
    observations: pd.DataFrame,
    lineage: pd.DataFrame,
    roster_cik: Mapping[str, str],
    manual: SecurityManual,
    *,
    built_at: pd.Timestamp,
    sec_tickers: pd.DataFrame | None = None,
    finra_presence: pd.DataFrame | None = None,
    co_registrant_ciks: Collection[str] = (),
    existing: pd.DataFrame | None = None,
    ended_on: Mapping[str, pd.Timestamp] | None = None,
) -> CusipHistory:
    """`run_cusip_history`'s compute on these inputs (`ended_on` = the anchor's end dates, none by default); deterministic
    for fixed inputs and `built_at`."""
    obs = prepare_observations(observations)
    view = lineage_view(lineage, roster_cik, tape_intervals(lineage))
    listing, finra, flags = sec_listing(sec_tickers), finra_evidence(finra_presence), CusipFlags()

    lines, spans = ftd_lines(obs)
    issuers = cusip_issuers(spans, obs, view, manual, flags)
    lines = extend_lines(line_ends(lines, obs, listing, issuers), finra, manual, issuers, view)

    facts = cusip_facts(lines, issuers, view, manual, listing, finra)
    segments, candidates = role_segments(facts, view, frozenset(pad_cik(c) for c in co_registrant_ciks))
    roles = CusipRoles(facts={fact.cusip: fact for fact in facts}, segments=segments)
    canonical_segments(candidates, roles.facts, view, segments, obs["trade_date"].max() + pd.Timedelta(days=1) if not obs.empty else pd.Timestamp.max)

    rows = finalise_rows(cusip_rows(lines, obs, roles, issuers, row_context(view, manual, listing)), existing, built_at, ended_on or {})
    detail, items = flag_items(cusip_flags(flags, roles, lines, obs, rows), ratio_step_flags(obs, rows, manual.ratios))
    return CusipHistory(rows=rows, flags=detail, items=items)


#: The header line of an SEC FTD file.
FTD_HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"


def write_ftd_zips(cache: Path, texts: Mapping[str, str]) -> Path:
    """Each period's FTD file text cached as the SEC ships it (`cnsfails<period>.zip`, one text member); returns `cache`."""
    cache.mkdir(parents=True, exist_ok=True)
    for period, text in texts.items():
        with ZipFile(cache / FTD_ZIP_NAME_TEMPLATE.format(period=period), "w") as archive:
            archive.writestr(f"cnsfails{period}.txt", text)
    return cache


def ftd_texts(observations: pd.DataFrame) -> dict[str, str]:
    """The SEC FTD file text of each period of `observations` (`[date, cusip, source_symbol, description, price,
    fails_quantity, period]`)."""
    texts: dict[str, str] = {}
    for period, group in observations.groupby("period"):
        lines = [
            f"{row.date:%Y%m%d}|{row.cusip}|{row.source_symbol}|{int(row.fails_quantity)}|{row.description}|{row.price}" for row in group.itertuples()
        ]
        texts[str(period)] = FTD_HEADER + "\n".join(lines) + "\n"
    return texts


def run_cusip_stage(context: Any, lineage: pd.DataFrame, ftd_cache: Path, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """`StepExtractIdentity.run_cusip_history` itself over `context`'s store, with every line of the FTD zips cached in
    `ftd_cache` stored first (a superset of the ingest scope) and `lineage` as this run's symbol history."""
    for path in sorted(ftd_cache.glob("cnsfails*.zip")):
        with ZipFile(path) as archive:
            text = archive.read(archive.namelist()[0]).decode("latin-1")
        context.store.save(Tables.sec_fails_to_deliver_security, ftd._parse_ftd_lines(text).assign(period=path.stem.removeprefix("cnsfails")))
    step = cast(Any, object.__new__(StepExtractIdentity))
    step._context, step._config_dir, step._log = context, str(context.config_dir), logging.getLogger("tests.cusip_build")
    step._symbols, step._run_date = SimpleNamespace(rows=lineage), run_date(as_of)
    step._anchor = CompanyAnchor(company_id={}, sharadar_cik={}, roster_cik=dict(context.universe.cik_by_ticker))
    return step.run_cusip_history()
