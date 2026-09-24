"""Read-only completed-quarter reconciliation for insider bulk and EDGAR sources."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import load_price_frames
from src.data_aggregate.utils.institutionals.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.insider_features import (
    DEFAULT_DECAY_HALFLIFE,
    build_insider_feature_panel,
)
from src.data_extract.utils.common.edgar_driver import PROGRAMMING_ERRORS
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.parallel_fetch import DEFAULT_WORKERS, run_per_ticker
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.institutionals.fetch_insider_edgar import (
    _filing_frames,
    insider_filings,
)
from src.data_store.schema import Tables
from src.validate.checks.insider_parity import (
    InsiderParityThresholds,
    reconcile_feature_panels,
    reconcile_transactions,
)

_LOG = logging.getLogger(__name__)

_PARITY_COLUMNS = (
    "accession_number",
    "security_type",
    "ticker",
    "issuer_cik",
    "issuer_name",
    "owner_cik",
    "owner_name",
    "is_director",
    "is_officer",
    "is_ten_pct_owner",
    "is_other",
    "officer_title",
    "document_type",
    "transaction_date",
    "filing_date",
    "period_of_report",
    "security_title",
    "transaction_code",
    "acquired_disposed",
    "shares",
    "price_per_share",
    "value_usd",
    "shares_owned_after",
    "direct_indirect",
    "is_10b5_1",
    "transaction_form_type",
    "equity_swap_involved",
    "deemed_execution_date",
    "nature_of_ownership",
    "transaction_timeliness",
    "exercise_price",
    "exercise_date",
    "expiration_date",
    "underlying_security_title",
    "underlying_shares",
    "underlying_value",
)
_SHARES_COLUMNS = (
    "ticker",
    "as_of",
    "sharesOutstanding",
    "sharesOutstandingPit",
)


def _thresholds(config: DictConfig) -> InsiderParityThresholds:
    values = config.validate.insider_parity
    return InsiderParityThresholds(
        accession_coverage=float(values.accession_coverage),
        exact_structure=float(values.exact_structure),
        categorical_agreement=float(values.categorical_agreement),
        share_agreement=float(values.share_agreement),
        economic_within_tolerance=float(values.economic_within_tolerance),
        identity_agreement=float(values.identity_agreement),
        numeric_relative_tolerance=float(values.numeric_relative_tolerance),
        aggregate_value_relative_tolerance=float(values.aggregate_value_relative_tolerance),
        feature_cell_agreement=float(values.feature_cell_agreement),
        edge_rank_correlation=float(values.edge_rank_correlation),
    )


def _quarter(value: str) -> pd.Period:
    try:
        return pd.Period(value.upper(), freq="Q")
    except ValueError as exc:
        raise ValueError(f"invalid quarter {value!r}; expected YYYYQn") from exc


def replay_completed_quarter(
    context: Context,
    quarter: pd.Period,
    bulk: pd.DataFrame,
    *,
    max_workers: int = DEFAULT_WORKERS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replay exactly the transaction-bearing bulk accessions through EDGAR XML."""
    context.ensure_edgar_identity()
    target_accessions = set(bulk["accession_number"].dropna().astype(str))
    tickers = sorted(set(bulk["ticker"].dropna().astype(str)))
    cik_map = load_cik_mapping(context, tickers)
    identity = load_identity(context)
    start = quarter.start_time.normalize()
    end = quarter.end_time.normalize()

    def worker(ticker: str, cik: str) -> dict[str, object]:
        frames: list[pd.DataFrame] = []
        listed_accessions: list[str] = []
        errors: list[dict[str, str]] = []
        try:
            filings = insider_filings(
                ticker,
                cik,
                since=start,
                through=end,
                done_accessions=frozenset(),
            )
        except PROGRAMMING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 -- one issuer must not erase the replay
            return {
                "ticker": ticker,
                "frames": frames,
                "listed_accessions": listed_accessions,
                "errors": [{"accession_number": "", "error": repr(exc)}],
            }

        for filing in filings:
            filing_date = pd.to_datetime(getattr(filing, "filing_date", None), errors="coerce")
            if pd.isna(filing_date) or not start <= filing_date.normalize() <= end:
                continue
            accession = str(getattr(filing, "accession_number", ""))
            if accession not in target_accessions:
                continue
            listed_accessions.append(accession)
            try:
                transactions, _, _ = _filing_frames(
                    filing,
                    universe=tickers,
                    identity=identity,
                    fetched_at=pd.Timestamp.now(tz="UTC").tz_localize(None),
                )
            except PROGRAMMING_ERRORS:
                raise
            except Exception as exc:  # noqa: BLE001 -- retained in the report as evidence
                errors.append({"accession_number": accession, "error": repr(exc)})
                continue
            if not transactions.empty:
                frames.append(transactions)
        return {
            "ticker": ticker,
            "frames": frames,
            "listed_accessions": listed_accessions,
            "errors": errors,
        }

    results = run_per_ticker(
        cik_map,
        worker,
        desc=f"insider parity {quarter}",
        max_workers=max_workers,
    )
    frames = [frame for result in results for frame in result["frames"] if isinstance(frame, pd.DataFrame) and not frame.empty]
    live = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=_PARITY_COLUMNS)
    if not live.empty:
        live = live.drop_duplicates(
            subset=list(Tables.insider_transactions_live.pk),
            keep="last",
        )
    listed = {accession for result in results for accession in result["listed_accessions"]}
    errors = [error for result in results for error in result["errors"]]
    return live, {
        "tickers_scanned": len(cik_map),
        "target_accessions": len(target_accessions),
        "target_accessions_listed": len(listed),
        "listing_coverage": len(listed) / max(len(target_accessions), 1),
        "parse_errors": errors,
    }


def _feature_panels(
    context: Context,
    config: DictConfig,
    quarter: pd.Period,
    bulk: pd.DataFrame,
    live: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = quarter.start_time.normalize()
    end = quarter.end_time.normalize()
    warmup = start - pd.DateOffset(years=1)
    peers = load_peers_or_raise(context, config)
    frames = load_price_frames(
        context.store,
        peers=peers,
        fields=("close_split", "level_factor"),
        since=warmup,
    )
    shares = context.store.load(
        Tables.fundamentals_history,
        columns=_SHARES_COLUMNS,
        where={"ticker": list(frames.universe)},
        since=start - pd.DateOffset(years=3),
        optional=True,
    )
    availability_config = config.get("data_availability")
    availability = InstitutionalAvailability.from_config(availability_config) if availability_config is not None else None
    decay = float(config.build_cube.get("institutionals", {}).get("decay_halflife", {}).get("insider", DEFAULT_DECAY_HALFLIFE))

    def build(source: pd.DataFrame) -> pd.DataFrame:
        panel = build_insider_feature_panel(
            frames,
            source,
            shares_out_history=shares,
            decay_halflife=decay,
            availability=availability,
            complete_through=end,
        )
        dates = pd.to_datetime(panel["date"], errors="coerce")
        return panel.loc[dates.between(start, end)].reset_index(drop=True)

    return build(bulk), build(live)


def run_completed_quarter_reconciliation(
    context: Context,
    config: DictConfig,
    quarter_label: str,
    *,
    max_workers: int = DEFAULT_WORKERS,
    replay_cache: str | Path | None = None,
    refresh_replay: bool = False,
) -> dict[str, object]:
    """Run transaction and feature parity for one completed bulk quarter."""
    quarter = _quarter(quarter_label)
    today_quarter = pd.Timestamp.today().to_period("Q")
    if quarter >= today_quarter:
        raise ValueError(f"{quarter} is not a completed quarter")
    bulk = context.store.load(
        Tables.insider_transactions,
        columns=_PARITY_COLUMNS,
        where={"quarter": str(quarter).lower()},
    )
    cache_path = Path(replay_cache) if replay_cache is not None else None
    metadata_path = cache_path.with_suffix(".json") if cache_path is not None else None
    if cache_path is not None and metadata_path is not None and cache_path.exists() and metadata_path.exists() and not refresh_replay:
        live = pd.read_parquet(cache_path)
        replay = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        live, replay = replay_completed_quarter(
            context,
            quarter,
            bulk,
            max_workers=max_workers,
        )
        if cache_path is not None and metadata_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            live.to_parquet(cache_path, index=False)
            metadata_path.write_text(
                json.dumps(_jsonable(replay), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    thresholds = _thresholds(config)
    transactions = reconcile_transactions(bulk, live, thresholds)
    if live.empty:
        features: dict[str, object] = {
            "passed": False,
            "reason": "EDGAR replay produced no transaction rows",
        }
    else:
        bulk_panel, live_panel = _feature_panels(context, config, quarter, bulk, live)
        features = reconcile_feature_panels(bulk_panel, live_panel, thresholds)
    return {
        "quarter": str(quarter),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "passed": bool(transactions["passed"] and features["passed"]),
        "replay": replay,
        "transactions": transactions,
        "features": features,
        "promotion": {
            "config_key": "source_freshness.insider_bulk_authoritative_through",
            "eligible_value": str(quarter),
            "eligible": bool(transactions["passed"] and features["passed"]),
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp | pd.Period):
        return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def write_reconciliation_report(out: str | Path, result: dict[str, object]) -> tuple[Path, Path]:
    """Write the retained machine-readable gate and its compact human rendering."""
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(result)
    json_path = directory / "insider_parity.json"
    markdown_path = directory / "insider_parity.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    transaction = payload["transactions"]
    features = payload["features"]
    lines = [
        f"# Insider source parity: {payload['quarter']}",
        "",
        f"**Result:** {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        "| Gate | Observed |",
        "|---|---:|",
        f"| Accession coverage | {transaction.get('accession_coverage', 0):.4%} |",
        f"| Exact row structure | {transaction.get('exact_structure', 0):.4%} |",
        f"| Categorical agreement | {transaction.get('categorical_agreement', 0):.4%} |",
        f"| Share agreement | {transaction.get('share_agreement', 0):.4%} |",
        f"| Price/value within tolerance | {transaction.get('economic_within_tolerance', 0):.4%} |",
        f"| Identity agreement | {transaction.get('identity_agreement', 0):.4%} |",
        f"| Aggregate value difference | {transaction.get('aggregate_value_relative_difference', 0):.4%} |",
        f"| Value-weighted row difference | {transaction.get('value_weighted_relative_difference', 0):.4%} |",
        f"| Feature null-mask agreement | {features.get('null_mask_agreement', 0):.4%} |",
        f"| Feature cell agreement | {features.get('cell_agreement', 0):.4%} |",
        f"| Minimum edge rank correlation | {features.get('minimum_edge_rank_correlation', 0):.6f} |",
        "",
        f"Missing accessions: {len(transaction.get('missing_accessions', []))}  ",
        f"Parse errors: {len(payload['replay'].get('parse_errors', []))}",
        "",
        "The ZIP quarter may be promoted only when this report passes. Promotion changes "
        "`source_freshness.insider_bulk_authoritative_through`; it never deletes the retained "
        "daily staging copy.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _LOG.info("wrote insider parity report to %s and %s", json_path, markdown_path)
    return json_path, markdown_path
