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
from src.data_extract.utils.common.identity import bind_identity, build_identity, identity_inputs
from src.data_extract.utils.common.identity_contract import DEI_SOURCE, FORM345_SOURCE, check_one_entity_per_cik, entity_by_cik_map
from src.data_extract.utils.common.registrant import load_registrants
from src.data_extract.utils.common.security_lines import co_registrant_ciks, load_security_manual, tape_intervals
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
    company_groups,
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
from src.data_extract.utils.identity.company_anchor import CompanyAnchor, anchor_company_ids, newest_window_ciks, roster_ciks, sharadar_ciks
from src.data_extract.utils.identity.cusip_history import (
    EVIDENCE_COLUMNS,
    HOLDING_GROUPS,
    PROFILE_NAME_COLUMNS,
    SHARADAR_CUSIP_COLUMNS,
    CusipFlags,
    CusipHistory,
    CusipRoles,
    back_extend,
    canonical_segments,
    company_names,
    cusip_facts,
    cusip_flags,
    cusip_issuers,
    cusip_rows,
    extend_lines,
    finalise_rows,
    finra_evidence,
    flag_items,
    ftd_lines,
    ftd_scope,
    line_ends,
    line_owners,
    lineage_view,
    log_cusip_history,
    prepare_observations,
    ratio_step_flags,
    read_ftd_evidence,
    role_segments,
    row_context,
    sec_listing,
    sharadar_cusips,
    thirteenf_starts,
)
from src.data_extract.utils.identity.cusip_history import TABLE_COLUMNS as CUSIP_COLUMNS
from src.data_extract.utils.identity.evidence import capture_dei_period, pending_dei_periods
from src.data_extract.utils.identity.identity_propagate import (
    PropagationResult,
    log_propagation,
    owned_scopes,
    purge_rows,
    rebuild_history,
    rebuilt_since,
    recheck_tickers,
    refresh_fails,
    refresh_insider,
    refresh_short_volume,
    reparse_insider,
    reparse_notes,
)
from src.data_extract.utils.identity.lifecycle import (
    LIFE_COLUMNS,
    PRICE_SPAN,
    Lifecycle,
    company_lifecycle,
    ended_dates,
    log_lifecycle,
    universe_permatickers,
)
from src.data_extract.utils.identity.publish import changed_identity, cusip_company_ids, save_unless_unchanged, update_identity_columns
from src.data_extract.utils.identity.symbol_history import (
    TAPE_ACTION_COLUMNS,
    TAPE_TICKER_COLUMNS,
    TICKER_CHANGE_ACTIONS,
    SymbolHistory,
    load_stored_lineage,
    log_changed_assignments,
    log_symbol_history,
    parse_stored_rows,
    symbol_evidence,
    symbol_holders,
    symbol_rows,
    tape_chains,
    validate_manual_tenure_entities,
)
from src.data_extract.utils.identity.symbol_tenure import (
    Form345Scan,
    build_symbol_tenure,
    collapse_dei_periods,
    load_built_tenure,
    load_manual_symbol_tenure,
    load_rejected_symbol_tenure,
    scan_form345_cache,
)
from src.data_store.schema import Tables
from src.utils.filer_tables import PURGE_TABLES, REMOVAL_COLUMNS
from src.utils.identity_flags import (
    FLAG_COLUMNS,
    cik_activity,
    cik_items,
    flag_rows,
    identity_flag_block,
    load_resolutions,
    master_lines,
    refused_merge_items,
    review_details,
    review_flags,
    review_summary,
    sorted_flags,
    symbol_status_items,
    tape_mix_items,
    tape_mix_pairs,
)
from src.utils.step import Step
from src.utils.string import pad_cik_series, run_date

#: The `sec_cik_profile` columns the CIK history reads, and the rows that carry one of its event forms.
_PROFILE_COLUMNS = ("cik", "form_25", "form_15", "form_8k12b")
_SHARADAR_COLUMNS = ("secfilings", *LIFE_COLUMNS)
_ANCHOR_LINEAGE_COLUMNS = ("canonical_ticker", "company_id", "cik", "role", "valid_from")


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
        self._run_date = run_date(self._options.as_of)
        self._ftd_cache = cache_dir(context, paths.fails_deliver)
        self._scan: Form345Scan | None = None
        self._tenure: pd.DataFrame | None = None
        self._anchor: CompanyAnchor | None = None
        self._cik_history: CikHistory | None = None
        self._symbols: SymbolHistory | None = None
        self._cusips: CusipHistory | None = None
        self._lifecycle: Lifecycle | None = None
        self._first_build = False
        #: this run's start (UTC, the lineage stamps' clock): the scopes stamped since then changed in this run (X90)
        self._started = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")

    def run_capture_evidence(self) -> int:
        """Notes cover-page symbols of the pending cached periods, then the Form 3/4/5 tenure; returns rows captured (the FTD
        lines are read by the CUSIP history, never stored here)."""
        # the Notes cover-page symbols of every filer, one cached period at a time (`full`: every period)
        periods = pending_dei_periods(self._context, self._notes_cache, self._options.full)
        written = sum(capture_dei_period(self._context, self._notes_cache, period) for period in tqdm(periods, desc="SEC notes cover-page symbols"))

        # one Form 3/4/5 scan: the `form345` / `manual` tenure (minus the cited rejections) here, the owner sets in the CIK history
        self._scan = scan_form345_cache(self._insider_cache)
        manual, rejected = load_manual_symbol_tenure(str(self._config_dir)), load_rejected_symbol_tenure(str(self._config_dir))
        self._tenure = build_symbol_tenure(self._context, self._scan, manual, rejected)
        return written

    def run_company_anchor(self) -> CompanyAnchor:
        """The frozen `company_id` of each identified universe ticker, its roster CIK (U-49) and Sharadar CIK seeds, and the end
        date of each ended company (D-4: every later stage closes its open rows there, inside its stamp pass)."""
        # load the published company rows, the stored lineage, the SEC ticker list and the Sharadar tickers
        tickers = self._universe.tickers
        universe = self._context.store.load(Tables.universe_tickers, project=True, optional=True)
        lineage = self._context.store.load(Tables.universe_cik_symbol, columns=list(_ANCHOR_LINEAGE_COLUMNS), optional=True)
        sec_tickers = self._context.store.load(Tables.sec_company_tickers, columns=["ticker", "cik"], where={"ticker": tickers}, optional=True)
        sharadar = self._context.store.load(Tables.sharadar_tickers, columns=list(_SHARADAR_COLUMNS), optional=True)

        # anchor the stored ids, read the seeds and end dates
        ids, seeds = anchor_company_ids(universe, lineage, tickers), sharadar_ciks(universe, sharadar)
        roster = roster_ciks(tickers, lineage, sec_tickers, seeds)
        self._anchor = CompanyAnchor(company_id=ids, sharadar_cik=seeds, roster_cik=roster, ended_on=ended_dates(self._universe.frame, sharadar))

        # check every universe ticker has a roster CIK, then log
        unseeded = sorted(set(tickers) - set(roster))
        if unseeded:
            raise ValueError(f"no roster CIK for {unseeded}: not in the stored lineage, {Tables.sec_company_tickers} or Sharadar")
        self._log.info(
            f"company anchor: {len(ids)} of {len(tickers)} universe ticker(s) carry a frozen company_id ({len(tickers) - len(ids)} to mint); "
            f"{len(seeds)} Sharadar CIK seed(s); {len(self._anchor.ended_on)} ended company(ies)"
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
        roster_cik = anchor.roster_cik

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

        # post-process: the windows of each company, then its frozen or minted id and its rows
        groups = company_groups(membership, roster_cik)
        windows, multi = cik_windows(groups, roster_cik, registrants, automatic_chains(index, groups, roster_cik), successions, membership.curated)
        entities = company_ids(groups, anchor.company_id, windows)
        multi = [{**row, "company_id": entities.entity_of[roster_cik[row["canonical_ticker"]]]} for row in multi]
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
            ended_on=anchor.ended_on,
        )

        # check: the roster CIK names the company its filings name (D19)
        check_d19(tenure, roster_cik, entities.entity_of, d19_allowlist(blob))
        log_cik_history(history, membership, len(candidates), sharadar_outside(anchor.sharadar_cik, entities, roster_cik))
        self._cik_history = history
        return history

    def run_symbol_history(self) -> SymbolHistory:
        """Symbol intervals attached through the CIK history, with the Sharadar ticker-change tape (`run_publish` saves
        `universe_cik_symbol`)."""
        # load the CIK history of this run (else from stored inputs), the stored table, the Sharadar tape and the manual tenure
        history = self._cik_history if self._cik_history is not None else self.run_cik_history()
        existing = parse_stored_rows(self._context.store.load(Tables.universe_cik_symbol, project=True, optional=True))
        sharadar = self._context.store.load(Tables.sharadar_tickers, columns=list(TAPE_TICKER_COLUMNS), optional=True)
        actions = self._context.store.load(
            Tables.sharadar_actions, columns=list(TAPE_ACTION_COLUMNS), where={"action": list(TICKER_CHANGE_ACTIONS)}, optional=True
        )
        manual = load_manual_symbol_tenure(str(self._config_dir))
        roster_cik = (self._anchor if self._anchor is not None else self.run_company_anchor()).roster_cik

        # compute: the tape over the CIK windows, every holder interval classified, the rows and the trading symbols
        evidence = symbol_evidence(history, tape_chains(self._universe.frame, sharadar, actions))
        holders = symbol_holders(evidence, history, roster_cik)
        build = symbol_rows(history, holders, existing, None)

        # check: one company per CIK and every manual CIK in its ticker's company; log the build and what moved
        check_one_entity_per_cik(build.rows)
        new_map = entity_by_cik_map(build.rows)
        validate_manual_tenure_entities(manual, new_map, roster_cik)
        cold = existing is None or existing.empty
        log_symbol_history(self._context, build, manual, int(history.evidence["source"].eq(DEI_SOURCE).sum()), cold)
        if existing is not None and not cold:
            log_changed_assignments(self._context, entity_by_cik_map(existing), new_map, roster_cik)
        self._symbols = build
        return build

    def run_cusip_history(self) -> pd.DataFrame:
        """CUSIP lines of each company (`universe_cusip`) from the cached FTD lines in this run's lineage scope (else the
        stored lineage's), back to each CUSIP's first confirmed 13F period (`run_publish` logs its items and saves the table)."""
        # load the lineage, the Sharadar CUSIPs, the curation, the in-scope FTD lines of the zip cache, the SEC listing, the
        # stored table, the filed evidence, FINRA days, names and 13F holders
        lineage = self._symbols.rows if self._symbols is not None else load_stored_lineage(self._context)
        if lineage is None:
            raise ValueError("universe_cusip needs universe_cik_symbol: run `identity --only symbol-history` first")
        sharadar = self._context.store.load(Tables.sharadar_tickers, columns=list(SHARADAR_CUSIP_COLUMNS), optional=True)
        manual = load_security_manual(str(self._config_dir))
        scope = ftd_scope(lineage, sharadar_cusips(self._universe.frame, sharadar), manual)
        observations = read_ftd_evidence(self._context, self._ftd_cache, scope)
        sec_tickers = self._context.store.load(Tables.sec_company_tickers, project=True, optional=True)
        existing = self._context.store.load(Tables.universe_cusip, project=True, optional=True)
        evidence = self._context.store.load(
            Tables.symbol_tenure, columns=list(EVIDENCE_COLUMNS), where={"source": [FORM345_SOURCE, DEI_SOURCE]}, optional=True
        )
        presence = self._context.store.load(Tables.sec_short_volume_security, columns=["source_symbol", "date"], optional=True)
        profile = self._context.store.load(
            Tables.sec_cik_profile, columns=list(PROFILE_NAME_COLUMNS), where={"cik": sorted(set(pad_cik_series(lineage["cik"])))}, optional=True
        )
        cusips = sorted(set(observations["cusip"].astype(str).str.strip().str.upper()))
        holdings = (
            self._context.store.summarize(Tables.sec13f_hr, list(HOLDING_GROUPS), {"n_rows": ("count", None)}, where={"cusip": cusips})
            if cusips
            else None
        )
        anchor = self._anchor if self._anchor is not None else self.run_company_anchor()

        # nothing to build without FTD lines in the cache: the table is left as it is
        if observations.empty:
            self._log.info("universe_cusip: no in-scope FTD line in the zip cache (run `zips-download --only ftd`); table left unchanged")
            return pd.DataFrame(columns=list(CUSIP_COLUMNS))

        # clean: the FTD lines, the lineage view, the SEC listing and the FINRA class symbols and days
        obs = prepare_observations(observations)
        view = lineage_view(lineage, anchor.roster_cik, tape_intervals(lineage))
        listing, finra, flags = sec_listing(sec_tickers), finra_evidence(presence), CusipFlags()

        # compute: the lines and their issuers, their life over FINRA days and back to the first confirmed 13F period (X3, X4)
        lines, spans = ftd_lines(obs)
        issuers = cusip_issuers(spans, obs, view, manual, flags)
        lines = extend_lines(line_ends(lines, obs, listing, issuers), finra, manual, issuers, view)
        names = company_names(view, self._universe.frame, sharadar, profile)
        thirteenf = thirteenf_starts(holdings, line_owners(issuers, view), names, sharadar_cusips(self._universe.frame, sharadar), self._run_date)
        lines = back_extend(lines, thirteenf.starts)

        # compute: each CUSIP's facts and role segments, one canonical line per company and date (co-registrant CIKs left out)
        facts = cusip_facts(lines, issuers, view, manual, listing, finra)
        co_registrants = co_registrant_ciks(lineage, evidence, manual.co_registrants)
        segments, candidates = role_segments(facts, view, co_registrants)
        roles = CusipRoles(facts={fact.cusip: fact for fact in facts}, segments=segments)
        canonical_segments(
            candidates, roles.facts, view, segments, obs["trade_date"].max() + pd.Timedelta(days=1) if not obs.empty else pd.Timestamp.max
        )

        # post-process: the table rows closed at their company's end, each company's stamp kept while unchanged
        built = cusip_rows(lines, obs, roles, issuers, row_context(view, manual, listing))
        rows = finalise_rows(built, existing, pd.Timestamp.now().floor("s"), anchor.ended_on)

        # check: the flags of the build (its WARNING items are logged by `run_publish`) and the INFO summary
        detail, items = flag_items(cusip_flags(flags, roles, lines, obs, rows), ratio_step_flags(obs, rows, manual.ratios))
        log_cusip_history(rows, detail, thirteenf, co_registrants)
        self._cusips = CusipHistory(rows=rows, flags=detail, items=items, thirteenf=thirteenf)
        return rows

    def run_lifecycle(self) -> Lifecycle:
        """First price date, end date (D-4), status and trading symbol of each universe ticker, from this run's lineage
        (else the stored one), the Sharadar price dates and the stored prices."""
        # load the lineage, the Sharadar rows of the universe permatickers and each universe ticker's stored price span
        lineage = self._symbols.rows if self._symbols is not None else load_stored_lineage(self._context)
        where = {"permaticker": universe_permatickers(self._universe.frame)}
        sharadar = self._context.store.load(Tables.sharadar_tickers, columns=list(LIFE_COLUMNS), where=where, optional=True)
        prices = self._context.store.summarize(Tables.prices, "ticker", dict(PRICE_SPAN), where={"ticker": self._universe.tickers})

        # check there is a lineage to read company ids and trading symbols from
        if lineage is None:
            raise ValueError("lifecycle needs universe_cik_symbol: run `identity --only symbol-history` first")

        # compute and log
        self._lifecycle = company_lifecycle(self._universe.frame, sharadar, prices, lineage)
        log_lifecycle(self._lifecycle)
        return self._lifecycle

    def run_publish(self) -> int:
        """The one save (REQ-008): the identity items against their reviewed resolutions (a WARNING block of the unreviewed
        action items only, REQ-032), then `universe_cik_symbol` and `universe_cusip` (with `company_id`) unless unchanged
        and the changed identity columns of `universe_tickers`; returns the rows written."""
        # load this run's builds (each stage runs from stored inputs when publish runs alone), the stored tables, the resolutions
        symbols = self._symbols if self._symbols is not None else self.run_symbol_history()
        history = self._cik_history if self._cik_history is not None else self.run_cik_history()
        lines = self._cusips.rows if self._cusips is not None else self.run_cusip_history()
        cusip_items = self._cusips.items if self._cusips is not None else pd.DataFrame(columns=list(FLAG_COLUMNS))
        lifecycle = self._lifecycle if self._lifecycle is not None else self.run_lifecycle()
        stored_lineage = self._context.store.load(Tables.universe_cik_symbol, project=True, optional=True)
        stored_lines = self._context.store.load(Tables.universe_cusip, project=True, optional=True)
        stored_tickers = self._context.store.load(Tables.universe_tickers, project=True, optional=True)
        resolutions = load_resolutions(str(self._config_dir))

        # compute: the lineage items (refused merges included) with the CUSIP and end-date items, each against its resolution
        rows = flag_rows(symbols.rows)
        mixes = tape_mix_items(tape_mix_pairs(rows, self._universe.alias_symbols), rows, master_lines(stored_lines if lines.empty else lines))
        found = [*cik_items(rows, history.activity, symbols.backlog), *refused_merge_items(symbols.blocked), *mixes, *symbol_status_items(rows)]
        review = review_flags(sorted_flags([*found, *cusip_items.to_dict("records"), *lifecycle.items.to_dict("records")]), resolutions)

        # check: a WARNING block of the unreviewed action items only; everything else in one INFO summary, the items at DEBUG
        block, details = identity_flag_block(review.unreviewed), review_details(review)
        if block is not None:
            self._log.warning(block)
        self._log.info(review_summary(review))
        if details is not None:
            self._log.debug(details)

        # post-process: the CUSIP lines carry their company's id; the identity columns that changed; a first build (PD-13)
        published = cusip_company_ids(lines, symbols.rows)
        self._first_build = stored_lineage is None or stored_lineage.empty
        changed = changed_identity(lifecycle.rows, stored_tickers)

        # save each table unless unchanged
        written = save_unless_unchanged(self._context.store, Tables.universe_cik_symbol, stored_lineage, symbols.rows)
        written += save_unless_unchanged(self._context.store, Tables.universe_cusip, stored_lines, published)
        written += update_identity_columns(self._context.store, changed)
        self._log.info(f"identity publish: {written} row(s) written ({len(changed)} universe ticker(s) with changed identity columns)")
        return written

    def run_propagate(self) -> PropagationResult:
        """Carry the published identity into the stored rows (REQ-014): purge rows filed outside each changed company's
        scope, re-parse, re-stamp the insider, FTD and short-volume rows, rebuild purged histories. A first build or
        `--full` re-checks every ticker (PD-13); `--dry-run` only reports (AC-013)."""
        # load: the resolver rebuilt from the published tables, over each universe ticker's published current CIK (U-49), bound to
        # the universe re-read after the publish (the cached one predates this run's end dates, symbols and CIKs)
        inputs = identity_inputs(self._context)
        universe = self._universe.reread()
        current = newest_window_ciks(inputs.lineage)
        roster_cik = {ticker: current[ticker] for ticker in universe.tickers if ticker in current}
        built = build_identity(inputs.lineage, inputs.tenure, roster_cik, inputs.master, inputs.co_registrants, inputs.mergers)
        identity = bind_identity(self._context, built, universe)
        tickers, dry_run, every = self._options.tickers or universe.tickers, self._options.dry_run, self._options.full or self._first_build

        # compute: the tickers whose scope changed, and what each company owns
        scopes = {ticker: identity.filing_scope(ticker) for ticker in tickers}
        changed = recheck_tickers(scopes, as_of=self._run_date, every=every)
        owned = owned_scopes(identity)

        # contraction: rows filed by a CIK outside the ticker's company or windows (the event-only CIKs on consolidating tables)
        records = [record for spec in PURGE_TABLES for record in purge_rows(self._context, spec, changed, owned, identity.bounds, dry_run)]

        # expansion and re-stamps: the bulk re-parse of the scopes this run changed (X90: every cached zip, so once per change;
        # not on a dry run), then the insider, FTD and short-volume rows
        reparsed = changed if every else rebuilt_since(scopes, self._started)
        notes = {} if dry_run else reparse_notes(self._context, reparsed, scopes, identity)
        insider = {} if dry_run else reparse_insider(self._context, reparsed, scopes, identity, self._run_date)
        records += refresh_insider(self._context, identity, changed, dry_run)
        records += refresh_fails(self._context, tickers, self._run_date, dry_run, every)
        records += refresh_short_volume(self._context, identity, tickers, self._run_date, dry_run, every)

        # post-process: the history of every ticker whose facts were purged; one summary
        removals = pd.DataFrame(records, columns=list(REMOVAL_COLUMNS))
        result = PropagationResult(removals, {**notes, **insider}, rebuild_history(self._context, removals, dry_run))
        log_propagation(self._context, changed, result, dry_run)
        return result

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
