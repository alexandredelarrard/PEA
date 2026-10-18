"""Read-only independent reconciliation of a frozen h60 modelling assessment.

Uses artifact predictions and scoped source returns; no modelling scoring/selection helpers,
training, model unpickling, table mutation or active-artifact update.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from math import ceil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

from src.context import Context
from src.data_store.schema import Tables
from src.validate.result import CheckResult, Finding, jsonable

_LOG = logging.getLogger(__name__)


def independent_daily(predictions: pd.DataFrame, model: str) -> pd.Series:
    """Rank correlation via SciPy independently of the modelling metric implementation."""
    output = {}
    for date, group in predictions.groupby("date", sort=True):
        x, y = group[model].to_numpy(dtype=float), group.y.to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 3 and len(np.unique(x[valid])) >= 3 and len(np.unique(y[valid])) >= 3:
            output[date] = float(cast(Any, spearmanr(x[valid], y[valid])).statistic)
    result = pd.Series(output, name="ic", dtype=float).rename_axis("date")
    result.index = pd.DatetimeIndex(result.index).as_unit("ns")
    return result


def independent_paths(prices: pd.DataFrame, calendar: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Prefix products and validity counts over i+1..i+h, independently of rolling-path code."""
    if prices.duplicated(["date", "ticker"]).any() or horizon < 1:
        raise ValueError("Unique price keys and a positive horizon are required")
    results = []
    for ticker, rows in prices.groupby("ticker", sort=False):
        aligned = rows.set_index("date").reindex(calendar)
        p, r = aligned.close_total.to_numpy(dtype=float), aligned.ret.to_numpy(dtype=float)
        good = np.isfinite(r) & (r > -1)
        logs = np.zeros(len(r))
        logs[good] = np.log1p(r[good])
        cumulative = np.r_[0.0, np.cumsum(logs)]
        counts = np.r_[0, np.cumsum(good)]
        compounded = np.expm1(cumulative[horizon + 1 :] - cumulative[1:-horizon])
        complete = counts[horizon + 1 :] - counts[1:-horizon] == horizon
        start, end = p[:-horizon], p[horizon:]
        with np.errstate(divide="ignore", invalid="ignore"):
            endpoint = end / start - 1
        valid = complete & np.isfinite(start) & np.isfinite(end) & (start > 0) & (end > 0) & np.isclose(compounded, endpoint, rtol=1e-5, atol=1e-6)
        results.append(
            pd.DataFrame({"date": calendar[:-horizon], "ticker": ticker, "valid": valid, "realized_return": np.where(valid, compounded, np.nan)})
        )
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-standard JSON {value}")))


def _finding(findings: list[Finding], condition: bool, observed: str, expected: str, **evidence: Any) -> None:
    if not condition:
        findings.append(Finding.at(8, observed, expected, **evidence))


def _configured_h60(block: dict) -> list[str] | None:
    """Resolve a declared h60 list or its default; None permits the RF family fallback."""
    for horizon, value in (block.get("columns_by_horizon") or {}).items():
        try:
            matches = int(horizon) == 60
        except (TypeError, ValueError):
            continue
        if matches:
            if value is not None:
                return list(value)
            break
    return list(block["columns"]) if block.get("columns") is not None else None


def _reconcile_configuration(config: DictConfig, ledger: dict, findings: list[Finding]) -> None:
    """Independently compare effective YAML pools and original controls with the frozen experiment."""
    current = cast(dict, OmegaConf.to_container(config, resolve=True))
    ignored = {"columns", "columns_by_horizon"}
    for family, key in [("lgbm", "lgbm"), ("random_forest", "random_forest"), ("elasticnet", "linear")]:
        block, selected = current.get(key, {}), ledger["selected"][family]
        categoricals = selected["categoricals"]
        columns = _configured_h60(block)
        if columns is None and family == "random_forest":
            columns = _configured_h60(current.get("lgbm", {}))
        columns = columns or []
        registry = current.get("lgbm", {}).get("categoricals") or []
        categories = [] if family == "elasticnet" else [name for name in columns if name in registry]
        for kind, configured, frozen in [("columns", columns, selected["features"]), ("categoricals", categories, categoricals)]:
            _finding(
                findings,
                (configured or []) == frozen,
                f"{family}: configured h60 {kind} differ",
                "Current h60 pool equals the frozen selected pool",
                field=f"{key}.{kind}_h60",
                configured=configured or [],
                frozen=frozen,
            )
        fixed = {name: value for name, value in block.items() if name not in ignored}
        _finding(
            findings,
            fixed == ledger["fixed_parameters"][key],
            f"{family}: fixed non-feature parameters differ",
            "Every family parameter, including monotonic directions, remains frozen",
            field=f"{key}.fixed_parameters",
            configured=fixed,
            frozen=ledger["fixed_parameters"][key],
        )
    controls, model = ledger["controls"], current["model"]
    comparisons = [("seed", current.get("seed"), controls["seed"])]
    for key in ["weight_decay", "label_column", "target_type", "ensemble"]:
        comparisons.append((f"model.{key}", model.get(key), controls["shared"][key]))
    # These CV keys are forced by assessment execution rather than frozen raw YAML.
    forced = {"test_windows", "retain_models", "input_digest"}
    cv = {key: value for key, value in model.get("cv", {}).items() if key not in forced}
    frozen_cv = {key: value for key, value in controls["shared"]["cv"].items() if key not in forced}
    comparisons.append(("model.cv", cv, frozen_cv))
    for key, frozen in controls["assessment"].items():
        comparisons.append((f"model.assessment.{key}", model.get("assessment", {}).get(key), frozen))
    for field, configured, frozen in comparisons:
        _finding(
            findings,
            configured == frozen,
            f"{field}: experiment control differs",
            "Original experiment control equals the frozen ledger",
            field=field,
            configured=configured,
            frozen=frozen,
        )


def _reconcile_metrics(output: Path, evaluation: dict, models: list[str], findings: list[Finding]) -> dict:
    predictions = pd.read_parquet(output / "eval" / "predictions.parquet")
    _finding(findings, not predictions.duplicated(["date", "ticker"]).any(), "Duplicate prediction keys", "Unique date/ticker keys")
    computed = {}
    for period, rows in [("development", predictions.loc[predictions.fold > 0]), ("sealed", predictions.loc[predictions.fold == 0])]:
        stored = pd.read_parquet(output / "eval" / f"daily_{period}.parquet")
        computed[period] = {}
        sessions = pd.DatetimeIndex(rows.date.unique()).as_unit("ns").sort_values()
        for model in models:
            daily = independent_daily(rows, model)
            aligned = daily.reindex(sessions)
            claimed = stored.loc[stored.model == model].set_index("date").ic.sort_index()
            claimed.index = pd.DatetimeIndex(claimed.index).as_unit("ns")
            same = aligned.index.equals(claimed.index) and np.allclose(aligned, claimed, equal_nan=True, rtol=1e-10, atol=1e-12)
            _finding(findings, same, f"{period}/{model}: saved daily IC differs", "Independent finite-paired Spearman series agrees")
            average, std = float(daily.mean()), float(daily.std(ddof=0))
            ratio = average / std * np.sqrt(252 / 60) if std > 0 else np.nan
            expected = evaluation["metrics"][period][model]
            for field, value in [("mean_ic", average), ("ic_ir", ratio)]:
                claim = expected[field]
                agrees = (claim is None and not np.isfinite(value)) or (claim is not None and bool(np.isclose(value, claim, rtol=1e-10, atol=1e-12)))
                _finding(findings, agrees, f"{period}/{model} {field} differs", "Pooled daily arithmetic agrees", measured=value, saved=claim)
            computed[period][model] = {"mean_ic": average, "ic_ir": ratio, "n_days": len(daily)}
    members = predictions[models[:-1]].to_numpy(dtype=float)
    _finding(findings, bool(np.isfinite(members).all()), "Nonfinite member prediction", "Every contemporaneous row scored")
    _finding(
        findings,
        bool(np.allclose(members.mean(axis=1), predictions.ensemble, atol=1e-12)),
        "Ensemble blend differs",
        "Equal average of per-day standardized member scores",
    )
    return computed


def _reconcile_selection(output: Path, ledger: dict, fitted: dict, findings: list[Finding]) -> None:
    cutoff = pd.Timestamp(ledger["outcome_cutoff"])
    _finding(
        findings,
        ledger["frozen"] and fitted["selection_frozen_at"] == ledger["frozen_at"],
        "Frozen selection provenance mismatch",
        "Final fit belongs to the exact frozen selection",
    )
    _finding(
        findings,
        pd.Timestamp(fitted["audit"]["label_end_max"]) <= cutoff,
        "Training outcome exceeds cutoff",
        "Every training outcome known before sealed evaluation",
    )
    for family, state in ledger["families"].items():
        candidates = [row for row in state["candidates"] if row["metrics"]["eligible"]]
        baseline = _read(output / "b0" / "result.json")
        if baseline["metrics"][family]["eligible"]:
            candidates.insert(0, {"id": "b0", "features": baseline["features"][family], "metrics": baseline["metrics"][family]})
        best = sorted(candidates, key=lambda row: (-row["metrics"]["objective"], len(row["features"])))[0]
        _finding(
            findings,
            state["best"]["id"] == best["id"],
            f"{family}: saved winner differs",
            "Highest allowed OOF objective with smaller-pool exact tie",
        )
        for candidate in state["candidates"]:
            _finding(findings, len(candidate["folds"]) == 5, f"{family}/{candidate['id']}: fold count differs", "All five original folds")
            for fold in candidate["folds"]:
                _finding(
                    findings,
                    pd.Timestamp(fold["outer_train_label_end_max"]) < pd.Timestamp(fold["test_start"]),
                    f"{family}/{candidate['id']}: outer outcome overlap",
                    "Training label endpoints precede every test fold",
                )
                if fold["validation_start"]:
                    _finding(
                        findings,
                        pd.Timestamp(fold["inner_train_label_end_max"]) < pd.Timestamp(fold["validation_start"]),
                        f"{family}/{candidate['id']}: early-stop outcome overlap",
                        "Inner training outcomes precede early-stop validation",
                    )
    actual = {str(path.relative_to(output / "final")): hashlib.sha256(path.read_bytes()).hexdigest() for path in (output / "final").rglob("*.pkl")}
    _finding(findings, actual == fitted["artifacts"], "Final model bytes differ", "Frozen fitted artifact identities unchanged")


def _reconcile_cohorts(output: Path, models: list[str], findings: list[Finding]) -> int:
    scores = pd.read_parquet(output / "eval" / "scores.parquet")
    cohorts = pd.read_parquet(output / "eval" / "cohorts.parquet")
    expected = []
    for model in models:
        for date, group in scores.groupby("date", sort=False):
            selected = group.sort_values([model, "ticker"], ascending=[False, True]).head(ceil(len(group) / 10))
            expected.extend((date, ticker, model, 1 / len(selected)) for ticker in selected.ticker)
    independently = (
        pd.DataFrame(expected, columns=["date", "ticker", "model", "weight"]).sort_values(["date", "model", "ticker"]).reset_index(drop=True)
    )
    stored = cohorts[independently.columns].sort_values(["date", "model", "ticker"]).reset_index(drop=True)
    independently["date"] = independently.date.astype("datetime64[ns]")
    stored["date"] = stored.date.astype("datetime64[ns]")
    same = independently[["date", "ticker", "model"]].equals(stored[["date", "ticker", "model"]]) and np.allclose(independently.weight, stored.weight)
    _finding(findings, bool(same), "Selected cohort or weights differ", "Highest predicted ceil(N/10) before attaching outcomes")
    outcomes = pd.read_parquet(output / "eval" / "return_windows.parquet")
    joined = cohorts.merge(outcomes[["date", "ticker", "realized_return", "valid"]], on=["date", "ticker"], how="left", validate="many_to_one")
    joined["valid"] = joined.valid.eq(True) & np.isfinite(joined.realized_return)
    joined["weighted"] = (joined.weight * joined.realized_return).where(joined.valid, 0.0)
    repeated = joined.groupby(["date", "model"]).agg(selected=("ticker", "size"), supported=("valid", "sum"), contribution=("weighted", "sum"))
    repeated["cohort_return"] = repeated.contribution.where(repeated.selected == repeated.supported)
    matched = pd.read_parquet(output / "eval" / "matched_returns.parquet").set_index(["date", "model"]).sort_index()
    repeated = repeated.sort_index()
    _finding(
        findings,
        repeated.index.equals(matched.index) and bool(np.allclose(repeated.cohort_return, matched.cohort_return, equal_nan=True)),
        "Matched stock returns differ",
        "Missing selected outcomes retain fixed missing weights",
    )
    market = pd.read_parquet(output / "eval" / "market_windows.parquet").set_index("date")
    unique_market = bool(not market.index.hasnans and market.index.is_unique and market.ticker.eq("equity_tr").all())
    _finding(findings, unique_market, "Market window keys differ", "Unique nonnull dates for the stored equity_tr benchmark")
    if unique_market:
        compared = matched.reset_index().merge(
            market[["realized_return", "valid"]], left_on="date", right_index=True, how="left", validate="many_to_one", indicator=True
        )
        market_same = bool(
            compared._merge.eq("both").all()
            and compared.market_valid.eq(compared.valid).all()
            and np.allclose(compared.market_return, compared.realized_return.where(compared.valid.eq(True)), equal_nan=True, rtol=1e-10, atol=1e-12)
        )
        _finding(
            findings,
            market_same,
            "Matched SPY returns or validity differ",
            "Every matched benchmark return equals its source-validated date window, masked when invalid",
        )
    _finding(
        findings,
        bool(np.allclose(matched.cohort_return - matched.market_return, matched.excess_return, equal_nan=True)),
        "Stock-minus-SPY arithmetic differs",
        "Identical-window excess return",
    )
    return len(cohorts)


def _live_path_checks(context: Context, output: Path, findings: list[Finding], chunksize: int) -> int:
    scores = pd.read_parquet(output / "eval" / "scores.parquet", columns=["date"])
    start = scores.date.min()
    evaluation = _read(output / "eval" / "result.json")
    stop = pd.Timestamp(evaluation["calendar_end"])
    calendar = pd.DatetimeIndex(context.store.distinct(Tables.cube_part_prices, "date")).normalize().sort_values().unique()
    calendar = calendar[(calendar >= start) & (calendar <= stop)]
    predictions = pd.read_parquet(output / "eval" / "predictions.parquet", columns=["date", "ticker", "fold", "label_end", "y"])
    expected_ends = predictions.date.map(pd.Series(calendar, index=calendar).shift(-60))
    _finding(
        findings,
        bool(expected_ends.notna().all() and expected_ends.eq(predictions.label_end).all()),
        "Prediction maturity endpoints differ",
        "Exactly 60 observed market sessions to every target endpoint",
    )
    ledger = _read(output / "selection.json")
    _finding(
        findings,
        bool((predictions.loc[predictions.fold > 0, "label_end"] <= pd.Timestamp(ledger["outcome_cutoff"])).all()),
        "Development evaluation outcome crosses sealed cutoff",
        "All development labels mature by 2021-12-31",
    )
    _finding(
        findings,
        bool((predictions.loc[predictions.fold == 0, "date"] >= pd.Timestamp(configured_evaluation_start(ledger))).all()),
        "Sealed prediction precedes evaluation start",
        "Separate sealed feature dates",
    )
    labels = pd.concat(
        context.store.iter_load(
            Tables.cube, columns=["date", "ticker", "target_zscore_h60"], since=start, until=scores.date.max(), chunksize=chunksize
        ),
        ignore_index=True,
    )
    attached = predictions.merge(labels, on=["date", "ticker"], how="left", validate="one_to_one", indicator=True)
    _finding(
        findings,
        bool(attached["_merge"].eq("both").all() and np.allclose(attached.y, attached.target_zscore_h60, equal_nan=True, atol=0, rtol=0)),
        "Source target replay differs",
        "Paired labels exactly match the scoped source cube",
    )
    del predictions, labels, attached
    chunks = context.store.iter_load(
        Tables.cube_part_prices, columns=["date", "ticker", "close_total", "ret"], since=start, until=stop, chunksize=chunksize
    )
    prices = pd.concat(chunks, ignore_index=True)
    expected = independent_paths(prices, calendar, 60)
    saved = pd.read_parquet(output / "eval" / "return_windows.parquet")
    compared = expected.merge(saved, on=["date", "ticker"], how="left", suffixes=("_independent", "_saved"), validate="one_to_one")
    same = compared.valid_independent.eq(compared.valid_saved).all() and np.allclose(
        compared.realized_return_independent, compared.realized_return_saved, equal_nan=True, rtol=1e-8, atol=1e-10
    )
    _finding(findings, bool(same), "Source daily-path replay differs", "Every interior persisted daily return required and endpoints reconciled")
    macro = context.store.load(
        Tables.prices_macro, columns=["date", "ticker", "close"], where={"ticker": "equity_tr"}, since=start, until=stop, optional=False
    )
    if macro is None:
        raise ValueError("Market total-return series unavailable")
    macro = macro.set_index("date").reindex(calendar)
    macro["ret"], macro["ticker"] = macro.close.pct_change(fill_method=None), "equity_tr"
    market = independent_paths(macro.rename(columns={"close": "close_total"}).reset_index(names="date"), calendar, 60)
    saved_market = pd.read_parquet(output / "eval" / "market_windows.parquet")
    market_same = not saved_market.duplicated(["date", "ticker"]).any()
    if market_same:
        compared_market = market.merge(saved_market, on=["date", "ticker"], how="left", suffixes=("_independent", "_saved"), validate="one_to_one")
        market_same = compared_market.valid_independent.eq(compared_market.valid_saved).all() and np.allclose(
            compared_market.realized_return_independent, compared_market.realized_return_saved, equal_nan=True, rtol=1e-8, atol=1e-10
        )
    _finding(findings, bool(market_same), "Source SPY path replay differs", "Matching complete market windows on identical dates")
    _LOG.info("Modelling validation replayed %d source windows through context.store", len(expected))
    return len(expected)


def configured_evaluation_start(ledger: dict) -> str:
    return str(ledger["controls"]["assessment"]["evaluation_start"])


def _report_checks(run: Path, ledger: dict, evaluation: dict, findings: list[Finding]) -> None:
    document = (run / "report.html").read_text(encoding="utf-8")
    block = re.search(r'<script id="assessment-data" type="application/json">(.*?)</script>', document, re.S)
    if block is None:
        _finding(findings, False, "Missing embedded report evidence", "Offline report embeds current validated evidence")
        return
    payload = json.loads(block.group(1))
    _finding(
        findings,
        payload["selection"]["selected"] == ledger["selected"] and payload["evaluation"]["metrics"] == evaluation["metrics"],
        "Embedded report metrics or feature pools differ",
        "HTML displays current frozen metrics and pools",
    )
    for family in ["lgbm", "random_forest"]:
        for fold in range(6):
            curves = payload["explanations"][family][str(fold)]
            _finding(
                findings,
                set(curves["features"]) == set(ledger["selected"][family]["features"]),
                f"{family}/{fold}: embedded feature coverage differs",
                "Every retained feature represented in every original fold",
            )
            path = run / "_out" / "eval" / f"{family}_{fold}" / "shap.parquet"
            raw = pd.read_parquet(path)
            feature_sum = raw[ledger["selected"][family]["features"]].sum(axis=1) + raw.base
            _finding(
                findings,
                bool(np.allclose(feature_sum, raw.prediction, atol=1e-7, rtol=1e-6)),
                f"{family}/{fold}: raw SHAP additivity differs",
                "All signed contributions plus base reconstruct raw prediction",
            )
    _finding(
        findings,
        re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://', document) is None,
        "Remote asset dependency in report",
        "All chart code/data embedded; research citation links allowed",
    )


def check_modelling(context: Context, *, out: str | Path, config: DictConfig) -> CheckResult:
    """Validate complete frozen artifacts and source daily paths, with explicit abstention if incomplete."""
    run, findings = Path(out).resolve(), []
    output = run / "_out"
    required = [
        output / "selection.json",
        output / "final" / "result.json",
        output / "eval" / "result.json",
        output / "report_manifest.json",
        run / "report.html",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return CheckResult.abstained("modelling", "ensemble_h60", "Frozen evaluation/report is incomplete", missing=missing)
    ledger, fitted, evaluation = (_read(path) for path in required[:3])
    _reconcile_configuration(config, ledger, findings)
    configuration_agrees = not findings
    models = list(ledger["controls"]["shared"]["ensemble"]) + ["ensemble"]
    _reconcile_selection(output, ledger, fitted, findings)
    metrics = _reconcile_metrics(output, evaluation, models, findings)
    cohorts = _reconcile_cohorts(output, models, findings)
    windows = _live_path_checks(context, output, findings, int(config.model.assessment.read_chunksize)) if configuration_agrees else 0
    report = _read(output / "report_manifest.json")
    digest = hashlib.sha256((run / "report.html").read_bytes()).hexdigest()
    _finding(findings, digest == report["report_sha256"], "HTML bytes differ from manifest", "Current report identity verified")
    for family in ["lgbm", "random_forest"]:
        _finding(
            findings,
            set(report["retained_tree_features"][family]) == set(ledger["selected"][family]["features"]),
            f"{family}: HTML feature coverage differs",
            "All retained variables in feature selector",
        )
    _report_checks(run, ledger, evaluation, findings)
    result = CheckResult.measured(
        "modelling",
        "ensemble_h60",
        findings,
        scope={
            "source": "resolved config, frozen local predictions and scoped context.store returns; independent SciPy and prefix-path implementations",
            "out": str(run),
            "writes_to_application": False,
            "model_fitting": False,
            "configuration_agrees": configuration_agrees,
            "cohort_windows": cohorts,
            "source_windows": windows,
        },
        metrics=metrics,
    )
    (output / "validation.json").write_text(
        json.dumps(
            jsonable({"verdict": "PASS" if result.status == "pass" else "FAIL", "summary": result.summary(), "result": result.to_json()}),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return result
