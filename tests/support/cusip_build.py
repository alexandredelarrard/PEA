"""The `universe_cusip` build on in-memory inputs, through the production stage functions.

`derive_cusip_history` is the compute of `StepExtractIdentity.run_cusip_history` (clean -> lines -> 13F -> facts -> roles
-> rows -> flags) on frames instead of store reads: the FTD lines, the lineage, the roster, the curation, and optionally the
SEC listing, the FINRA days, the 13F holder names, the company names and Sharadar CUSIPs. No DB, no network.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.data_extract.transformers.step_extract_identity import StepExtractIdentity
from src.data_extract.utils.common.security_lines import SecurityManual, load_security_manual, tape_intervals
from src.data_extract.utils.identity import cusip_history as ch
from src.data_extract.utils.identity.cusip_history import (
    CusipFlags,
    CusipHistory,
    CusipRoles,
    back_extend,
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
    line_owners,
    lineage_view,
    prepare_observations,
    ratio_step_flags,
    role_segments,
    row_context,
    sec_listing,
    thirteenf_starts,
)
from src.data_store.schema import Tables
from src.utils.string import pad_cik, run_date

NAME_COLUMNS = ["company", "name", "valid_from", "valid_to", "tokens"]


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
    holdings: pd.DataFrame | None = None,
    names: pd.DataFrame | None = None,
    listed: Mapping[str, frozenset[str]] | None = None,
    as_of: pd.Timestamp | None = None,
) -> CusipHistory:
    """`run_cusip_history`'s compute on these inputs (`holdings` = 13F `[cusip, period, issuer_name, n_rows]`; `names` =
    `company_names` rows); deterministic for fixed inputs and `built_at`."""
    obs = prepare_observations(observations)
    view = lineage_view(lineage, roster_cik, tape_intervals(lineage))
    listing, finra, flags = sec_listing(sec_tickers), finra_evidence(finra_presence), CusipFlags()

    lines, spans = ftd_lines(obs)
    issuers = cusip_issuers(spans, obs, view, manual, flags)
    lines = extend_lines(line_ends(lines, obs, listing, issuers), finra, manual, issuers, view)
    name_rows = names if names is not None else pd.DataFrame(columns=NAME_COLUMNS)
    thirteenf = thirteenf_starts(holdings, line_owners(issuers, view), name_rows, listed or {}, as_of or built_at)
    lines = back_extend(lines, thirteenf.starts)

    facts = cusip_facts(lines, issuers, view, manual, listing, finra)
    segments, candidates = role_segments(facts, view, frozenset(pad_cik(c) for c in co_registrant_ciks))
    roles = CusipRoles(facts={fact.cusip: fact for fact in facts}, segments=segments)
    canonical_segments(candidates, roles.facts, view, segments, obs["trade_date"].max() + pd.Timedelta(days=1) if not obs.empty else pd.Timestamp.max)

    rows = finalise_rows(cusip_rows(lines, obs, roles, issuers, row_context(view, manual, listing)), existing, built_at)
    detail, items = flag_items(cusip_flags(flags, roles, lines, obs, rows), ratio_step_flags(obs, rows, manual.ratios))
    return CusipHistory(rows=rows, flags=detail, items=items, thirteenf=thirteenf)


def capture_ftd(context: Any, cache: Path, years_history: int, *, full: bool, as_of: pd.Timestamp) -> int:
    """The FTD block of `run_capture_evidence` over `context`'s store: scope, pending (and rescanned) lines, save; 0
    before the first `universe_cik_symbol` build."""
    lineage = context.store.load(Tables.universe_cik_symbol, columns=list(ch.FTD_LINEAGE_COLUMNS), optional=True)
    sharadar = context.store.load(Tables.sharadar_tickers, columns=list(ch.SHARADAR_CUSIP_COLUMNS), optional=True)
    stored = context.store.load(Tables.sec_fails_to_deliver_security, columns=list(ch.VOTE_COLUMNS), optional=True)
    cusips = ch.sharadar_cusips(context.universe.frame, sharadar)
    scope = ch.ftd_scope(lineage, cusips, stored, load_security_manual(getattr(context, "config_dir", None)))
    changed = ch.FtdScope() if full or lineage is None else ch.changed_ftd_scope(lineage, cusips, as_of)
    rows, periods = ch.capture_ftd_lines(context, ch.FtdCapture(cache, years_history, full, as_of), scope, changed)
    return ch.save_ftd_lines(context, rows, periods)


def run_cusip_stage(context: Any, lineage: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """`StepExtractIdentity.run_cusip_history` itself over `context`'s store, with `lineage` as this run's symbol history."""
    step = cast(Any, object.__new__(StepExtractIdentity))
    step._context, step._config_dir, step._log = context, str(context.config_dir), logging.getLogger("tests.cusip_build")
    step._symbols, step._run_date = SimpleNamespace(rows=lineage), run_date(as_of)
    return step.run_cusip_history()
