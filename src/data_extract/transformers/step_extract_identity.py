"""Identity step: one dated identity per universe company, built offline from the cached zips and the DB.

`run()` is the flow: capture evidence -> company anchor -> CIK history -> symbol history -> CUSIP history -> lifecycle ->
publish -> propagate. Each `run_*` runs alone from stored inputs (`identity --only <stage>`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.registrant import load_registrants
from src.data_extract.utils.identity.capture_ftd import capture_ftd_cusips
from src.data_extract.utils.identity.cik_history import (
    BACKLOG_COLUMNS,
    CikHistory,
    automatic_chains,
    blocked_frame,
    candidate_ciks,
    check_d19,
    cik_evidence,
    cik_rows,
    cik_windows,
    company_ids,
    curated_membership,
    d19_allowlist,
    evidence_index,
    grey_backlog,
    join_owner_overlaps,
    join_successions,
    join_symbol_handoffs,
    log_cik_history,
    manual_lineage_blob,
    owner_sets,
    parse_manual_lineage,
    sharadar_outside,
    succession_pairs,
)
from src.data_extract.utils.identity.company_anchor import CompanyAnchor, anchor_company_ids, sharadar_ciks
from src.data_extract.utils.identity.entity_lineage import build_entity_lineage, load_stored_lineage
from src.data_extract.utils.identity.evidence import capture_dei_period, pending_dei_periods
from src.data_extract.utils.identity.identity_propagate import PropagationResult, propagate_identity
from src.data_extract.utils.identity.security_master import build_security_master
from src.data_extract.utils.identity.symbol_tenure import (
    DEI_SOURCE,
    Form345Scan,
    build_symbol_tenure,
    collapse_dei_periods,
    load_built_tenure,
    scan_form345_cache,
)
from src.data_store.schema import Tables
from src.utils.identity_flags import cik_activity
from src.utils.step import Step
from src.utils.string import run_date

#: The `sec_cik_profile` columns the CIK history reads, and the rows that carry one of its event forms.
_PROFILE_COLUMNS = ("cik", "form_25", "form_15", "form_8k12b")
_SHARADAR_COLUMNS = ("permaticker", "secfilings", "lastupdated")


@dataclass(frozen=True)
class IdentityOptions:
    """The `identity` command's options: the propagated ticker subset (None: the universe), `full` re-capture and
    re-check, `dry_run` propagation and the run date."""

    tickers: list[str] | None = None
    full: bool = False
    dry_run: bool = False
    as_of: pd.Timestamp | None = None


class StepExtractIdentity(Step):
    def __init__(self, context: Context, config: DictConfig, options: IdentityOptions | None = None) -> None:
        super().__init__(context=context, config=config)
        self._options = options or IdentityOptions()
        paths = config.local.paths
        self._insider_cache = cache_dir(context, paths.insider_transactions)
        self._notes_cache = cache_dir(context, paths.financial_notes)
        self._ftd_cache = cache_dir(context, paths.fails_deliver)
        self._years_history = int(config.data_extract.years_history)
        self._run_date = run_date(self._options.as_of)
        self._scan: Form345Scan | None = None
        self._tenure: pd.DataFrame | None = None
        self._anchor: CompanyAnchor | None = None
        self._cik_history: CikHistory | None = None
        self._lineage: pd.DataFrame | None = None

    def run_capture_evidence(self) -> int:
        """Notes cover-page symbols and FTD lines of the pending cached periods, then the Form 3/4/5 tenure; returns rows captured."""
        # the Notes cover-page symbols of every filer, one cached period at a time (`full`: every period)
        periods = pending_dei_periods(self._context, self._notes_cache, self._options.full)
        written = sum(capture_dei_period(self._context, self._notes_cache, period) for period in tqdm(periods, desc="SEC notes cover-page symbols"))

        # the in-scope FTD lines (a no-op before the first lineage build)
        written += capture_ftd_cusips(self._context, self._ftd_cache, self._years_history, full=self._options.full, as_of=self._run_date)

        # one Form 3/4/5 scan: the `form345` / `manual` tenure here, the owner sets in the CIK history
        self._scan = scan_form345_cache(self._insider_cache)
        self._tenure = build_symbol_tenure(self._context, self._scan, str(self._config_dir))
        return written

    def run_company_anchor(self) -> CompanyAnchor:
        """The frozen `company_id` of each identified universe ticker and its Sharadar CIK seed."""
        # load the published company rows, the stored lineage and the Sharadar tickers
        universe = self._context.store.load(Tables.universe_tickers, project=True, optional=True)
        lineage = self._context.store.load(Tables.universe_cik_symbol, columns=["canonical_ticker", "company_id"], optional=True)
        sharadar = self._context.store.load(Tables.sharadar_tickers, columns=list(_SHARADAR_COLUMNS), optional=True)

        # anchor the stored ids and read the Sharadar seeds
        tickers = self._universe.tickers
        self._anchor = CompanyAnchor(company_id=anchor_company_ids(universe, lineage, tickers), sharadar_cik=sharadar_ciks(universe, sharadar))
        self._log.info(
            f"company anchor: {len(self._anchor.company_id)} of {len(tickers)} universe ticker(s) carry a frozen company_id "
            f"({len(tickers) - len(self._anchor.company_id)} to mint); {len(self._anchor.sharadar_cik)} Sharadar CIK seed(s)"
        )
        return self._anchor

    def run_cik_history(self) -> CikHistory:
        """Which CIKs are each universe company and the dated windows of its consolidating CIKs (saved with the symbol history)."""
        # load the anchor, the tenure and Form 3/4/5 scan of this run (else the stored ones), the dei evidence, the SEC profile, the curation
        anchor = self._anchor if self._anchor is not None else self.run_company_anchor()
        tenure = self._tenure if self._tenure is not None else load_built_tenure(self._context)
        scan = self._scan if self._scan is not None else scan_form345_cache(self._insider_cache)
        dei_rows = self._context.store.load(Tables.symbol_tenure, project=True, where={"source": DEI_SOURCE}, optional=True)
        events = [{column: self._context.store.NOT_NULL} for column in _PROFILE_COLUMNS[1:]]
        profile = self._context.store.load(Tables.sec_cik_profile, columns=list(_PROFILE_COLUMNS), where=events, optional=True)
        registrants = load_registrants(str(self._config_dir))
        blob = manual_lineage_blob(str(self._config_dir))
        roster_cik = self._universe.cik_by_ticker

        # clean: one evidence frame and its index, the candidate CIKs and their owner sets
        dei = collapse_dei_periods(dei_rows) if dei_rows is not None else None
        evidence = cik_evidence(tenure, dei)
        index = evidence_index(evidence)
        candidates, by_ticker = candidate_ciks(tenure, roster_cik, anchor.sharadar_cik)
        owners = owner_sets(scan.owner_pairs, candidates)

        # compute: the oracles in priority order
        membership = curated_membership(candidates, roster_cik, registrants, parse_manual_lineage(blob))
        successions = succession_pairs(profile, by_ticker)
        join_successions(membership, successions)
        join_symbol_handoffs(membership, evidence, index, roster_cik)
        join_owner_overlaps(membership, by_ticker, roster_cik, owners)

        # post-process: frozen ids, then the windows and rows of each company
        entities = company_ids(membership, anchor.company_id, roster_cik)
        windows, multi = cik_windows(
            entities, roster_cik, registrants, automatic_chains(index, entities, roster_cik), successions, membership.curated
        )
        history = CikHistory(
            rows=cik_rows(entities, membership, windows, evidence),
            entity_of=entities.entity_of,
            ticker_by_entity=entities.ticker_by_entity,
            window_end={cik: window[1] for cik, window in windows.items()},
            oracle_of={cik: verdict[0] for cik, verdict in membership.verdicts.items()},
            backlog=pd.DataFrame(grey_backlog(membership, entities) + multi, columns=list(BACKLOG_COLUMNS)),
            blocked=blocked_frame(membership, roster_cik),
            evidence=evidence,
            activity=cik_activity(pd.concat([tenure] if dei is None else [tenure, dei], ignore_index=True)),
        )

        # check: the roster CIK names the company its filings name (D19)
        check_d19(tenure, roster_cik, entities.entity_of, d19_allowlist(blob))
        log_cik_history(history, membership, len(candidates), sharadar_outside(anchor.sharadar_cik, entities, roster_cik))
        self._cik_history = history
        return history

    def run_symbol_history(self) -> pd.DataFrame:
        """Symbol intervals attached through the CIK history; writes `universe_cik_symbol` until `run_publish` does (4d)."""
        history = self._cik_history if self._cik_history is not None else self.run_cik_history()
        self._lineage = build_entity_lineage(self._context, history, str(self._config_dir))
        return self._lineage

    def run_cusip_history(self) -> pd.DataFrame:
        """CUSIP lines of each company (`universe_cusip`), built from this run's lineage or the stored one."""
        lineage = self._lineage if self._lineage is not None else load_stored_lineage(self._context)
        if lineage is None:
            raise ValueError("universe_cusip needs universe_cik_symbol: run `identity --only symbol-history` first")
        return build_security_master(self._context, lineage, str(self._config_dir))

    def run_lifecycle(self) -> None:
        """First price date, end date and status per company: built in phase 4d; nothing to do yet."""
        return None

    def run_publish(self) -> None:
        """Completeness gate and one save of the published tables: phase 4d; the builders still save their own tables."""
        return None

    def run_propagate(self) -> PropagationResult:
        """Carry the lineage changes into the stored rows, under the resolver reloaded once from the rebuilt tables."""
        return propagate_identity(
            self._context,
            self._options.tickers or self._universe.tickers,
            dry_run=self._options.dry_run,
            every_ticker=self._options.full,
            identity=load_identity(self._context, refresh=True),
            as_of=self._run_date,
        )

    def run(self) -> None:
        """The identity flow (REQ-009): each stage by name, in order."""
        self.run_capture_evidence()
        self.run_company_anchor()
        self.run_cik_history()
        self.run_symbol_history()
        self.run_cusip_history()
        self.run_lifecycle()
        self.run_publish()
        self.run_propagate()
