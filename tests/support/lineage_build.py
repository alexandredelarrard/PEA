"""The `universe_cik_symbol` build on in-memory inputs, through the production stages.

`StepExtractIdentity.run_cik_history` runs over a read-only stub store holding the given `dei` rows and SEC profile, with this
run's tenure and Form 3/4/5 owner pairs preset; `derive_lineage_rows` then attaches the symbol intervals. No DB, no network.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.data_extract.transformers.step_extract_identity import StepExtractIdentity
from src.data_extract.utils.identity.cik_history import CikHistory
from src.data_extract.utils.identity.company_anchor import CompanyAnchor, anchor_company_ids
from src.data_extract.utils.identity.entity_lineage import LineageBuild, derive_lineage_rows
from src.data_store.schema import Tables
from src.data_store.store import NOT_NULL


class _ReadOnlyStore:
    """The two tables the CIK history reads, served from frames; any other table is absent."""

    NOT_NULL = NOT_NULL

    def __init__(self, dei: pd.DataFrame | None, profile: pd.DataFrame | None) -> None:
        self._tables = {Tables.symbol_tenure: dei, Tables.sec_cik_profile: profile}

    def load(self, table: Any, **kwargs: Any) -> pd.DataFrame | None:
        frame = self._tables.get(table)
        return None if frame is None or frame.empty else frame.copy()


def cik_history_of(
    tenure: pd.DataFrame,
    roster_cik: Mapping[str, str],
    owner_pairs: pd.DataFrame,
    config_dir: str,
    *,
    dei: pd.DataFrame | None = None,
    existing: pd.DataFrame | None = None,
    profile: pd.DataFrame | None = None,
    sharadar_cik: Mapping[str, str] | None = None,
) -> CikHistory:
    """`run_cik_history` on these inputs; the anchor is read off `existing` (the stored table), as `run_company_anchor` does."""
    stored_dei = None if dei is None else dei.assign(evidence_period=dei.get("evidence_period", ""))
    context = SimpleNamespace(
        store=_ReadOnlyStore(stored_dei, profile),
        log=logging.getLogger("tests.lineage_build"),
        config_dir=config_dir,
        universe=SimpleNamespace(cik_by_ticker=dict(roster_cik), tickers=list(roster_cik), alias_symbols=frozenset()),
    )
    # the stage reads only its context and the inputs preset here, so no cache directory is created
    step = cast(Any, object.__new__(StepExtractIdentity))
    step._context, step._config_dir, step._log = context, config_dir, logging.getLogger("tests.lineage_build")
    step._tenure = tenure
    step._scan = SimpleNamespace(owner_pairs=owner_pairs)
    step._anchor = CompanyAnchor(company_id=anchor_company_ids(None, existing, list(roster_cik)), sharadar_cik=dict(sharadar_cik or {}))
    return step.run_cik_history()


def derive_lineage(
    tenure: pd.DataFrame,
    roster_cik: Mapping[str, str],
    owner_pairs: pd.DataFrame,
    config_dir: str,
    *,
    dei: pd.DataFrame | None = None,
    existing: pd.DataFrame | None = None,
    built_at: pd.Timestamp | None = None,
    profile: pd.DataFrame | None = None,
) -> LineageBuild:
    """The whole `universe_cik_symbol` build: the CIK history, then the symbol intervals attached through it."""
    history = cik_history_of(tenure, roster_cik, owner_pairs, config_dir, dei=dei, existing=existing, profile=profile)
    return derive_lineage_rows(history, roster_cik, existing, built_at=built_at)
