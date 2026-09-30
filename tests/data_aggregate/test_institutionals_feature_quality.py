from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts import institutionals_feature_quality as quality


def _write_snapshot(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    parquet = pq.ParquetFile(path)
    return {
        "table": "cube_part_institutionals",
        "rows": parquet.metadata.num_rows,
        "columns": parquet.schema_arrow.names,
        "dtypes": {field.name: str(field.type) for field in parquet.schema_arrow},
        "pk": ["date", "ticker"],
        "date_col": "date",
        "first_date": str(pd.to_datetime(frame["date"]).min().date()),
        "last_date": str(pd.to_datetime(frame["date"]).max().date()),
        "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_peer_cache(path: Path) -> dict[str, object]:
    path.write_text(json.dumps({"AAA": {"BBB": 1.0}, "BBB": {"AAA": 1.0}}), encoding="utf-8")
    return quality._peer_metadata(path)


class _MetadataOnlyStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def exists(self, table: object) -> bool:
        self.calls.append("exists")
        return True

    def columns(self, table: object) -> list[str]:
        self.calls.append("columns")
        return ["date", "ticker", "value"]

    def row_count(self, table: object) -> int:
        self.calls.append("row_count")
        return 7

    def bounds(self, table: object, column: str | None = None) -> tuple[str, str]:
        self.calls.append("bounds")
        return "2024-01-02", "2024-01-05"


def test_freeze_baseline_verifies_and_manifests_metadata_only(tmp_path: Path, monkeypatch: object) -> None:
    snapshot = tmp_path / "cache" / "cube_part_institutionals.parquet"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "AAA"],
            "f_ic_demo": [1.0, 2.0],
        }
    )
    metadata = _write_snapshot(snapshot, frame)
    (snapshot.parent / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    store = _MetadataOnlyStore()
    peer_cache = tmp_path / "sector-peers.json"
    _write_peer_cache(peer_cache)
    context = SimpleNamespace(store=store, paths={"SECTOR_PEERS_PATH": peer_cache})
    monkeypatch.setattr(quality, "get_config_context", lambda *_args, **_kwargs: (None, context))

    out = tmp_path / "frozen"
    assert quality.main(["freeze-baseline", "--config", "configs", "--snapshot", str(snapshot), "--as-of", "2024-01-03", "--out", str(out)]) == 0

    frozen = out / "baseline.parquet"
    manifest = json.loads((out / "input-manifest.json").read_text(encoding="utf-8"))
    assert frozen.read_bytes() == snapshot.read_bytes()
    assert manifest["status"] == "pass"
    assert manifest["snapshot"]["sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert manifest["peer_cache"]["sha256"] == hashlib.sha256((out / "peer-baskets.json").read_bytes()).hexdigest()
    assert set(store.calls) <= {"exists", "columns", "row_count", "bounds"}
    assert quality.main(["freeze-baseline", "--config", "configs", "--snapshot", str(snapshot), "--as-of", "2024-01-03", "--out", str(out)]) == 0
    assert [path.name for path in out.glob("*.parquet")] == ["baseline.parquet"]
    print("SANITY: baseline bytes and hash are frozen, while source provenance used DataStore metadata calls only.")


def test_compare_refactor_reports_every_exact_difference(tmp_path: Path) -> None:
    before = tmp_path / "before.parquet"
    after = tmp_path / "after.parquet"
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["AAA", "AAA", "AAA"],
            "null_leg": [1.0, np.nan, 3.0],
            "inf_leg": [1.0, np.inf, 3.0],
            "value_leg": [1.0, 2.0, 3.0],
        }
    )
    base.to_parquet(before, index=False)
    identical_report = tmp_path / "identical.json"
    assert quality.main(["compare-refactor", "--before", str(before), "--after", str(before), "--out", str(identical_report)]) == 0
    assert json.loads(identical_report.read_text(encoding="utf-8"))["status"] == "pass"
    changed = base.assign(
        null_leg=[1.0, 2.0, 3.0],
        inf_leg=[1.0, -np.inf, 3.0],
        value_leg=pd.Series([1.0, 9.0, 3.0], dtype="float32"),
    )[["date", "ticker", "inf_leg", "null_leg", "value_leg"]]
    changed.to_parquet(after, index=False)

    report = tmp_path / "comparison.json"
    assert quality.main(["compare-refactor", "--before", str(before), "--after", str(after), "--out", str(report)]) == 1
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "fail"
    assert result["checks"]["column_order"]["pass"] is False
    assert result["checks"]["dtypes"]["pass"] is False
    assert result["checks"]["nulls"]["pass"] is False
    assert result["checks"]["nonfinite"]["pass"] is False
    assert result["checks"]["values"]["pass"] is False
    print("SANITY: refactor comparison separately exposes order, dtype, null, non-finite, and finite-value drift.")


def test_validate_candidate_is_snapshot_only_and_writes_all_checks(tmp_path: Path, monkeypatch: object) -> None:
    snapshot = tmp_path / "candidate.parquet"
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] * 2)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 4 + ["BBB"] * 4,
            "f_ic_inst_new_buyer_ratio": [0.1, 0.2, 0.3, 0.4, 0.8, 0.7, 0.6, 0.5],
            "f_ic_inst_exit_ratio": [0.9, 0.3, 0.8, 0.2, 0.1, 0.7, 0.4, 0.6],
            "f_ic_inst_concentration": [0.05, 0.2, 0.1, 0.4, 0.3, 0.15, 0.45, 0.25],
            "f_ic_demo": [2.0, 1.0, 4.0, 3.0, 1.0, 4.0, 2.0, 5.0],
        }
    )
    declared_bounds = quality.load_spec(quality.read_config("configs"), quality.TABLE).bounds
    rng = np.random.default_rng(7)
    for column, (lower, upper) in declared_bounds.items():
        if column not in frame:
            width = upper - lower
            frame[column] = lower + width * (0.1 + 0.8 * rng.random(len(frame)))
    metadata = _write_snapshot(snapshot, frame)
    metadata_path = tmp_path / "candidate-meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(
        json.dumps({column: "fixture feature" for column in frame.columns if column not in {"date", "ticker"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(quality, "get_config_context", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live context consulted")))

    out = tmp_path / "candidate-validation"
    assert (
        quality.main(
            [
                "validate-candidate",
                "--config",
                "configs",
                "--snapshot",
                str(snapshot),
                "--metadata",
                str(metadata_path),
                "--catalogue",
                str(catalogue),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    expected = {"grain", "schema", "catalogue", "bounds", "profile", "redundancy", "timeseries", "leakage"}
    summary = json.loads((out / "candidate-validation.json").read_text(encoding="utf-8"))
    assert set(summary["checks"]) == expected
    assert all((out / f"{name}.json").exists() for name in expected)
    assert summary["checks"]["leakage"] == "abstain"
    assert summary["status"] == "pass"

    metadata["snapshot_sha256"] = "stale"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    stale_out = tmp_path / "stale-validation"
    assert (
        quality.main(
            [
                "validate-candidate",
                "--config",
                "configs",
                "--snapshot",
                str(snapshot),
                "--metadata",
                str(metadata_path),
                "--catalogue",
                str(catalogue),
                "--out",
                str(stale_out),
            ]
        )
        == 1
    )
    assert json.loads((stale_out / "schema.json").read_text(encoding="utf-8"))["status"] == "fail"
    print("SANITY: candidate checks consumed only the frozen parquet, its hash-bound metadata, catalogue, and config; leakage abstained honestly.")


def test_validate_candidate_timeseries_fails_on_an_interior_hole(tmp_path: Path, monkeypatch: object) -> None:
    snapshot = tmp_path / "candidate-with-hole.parquet"
    index = np.arange(30)
    bullish = np.linspace(0.1, 0.9, 30)
    bullish[10:16] = np.nan
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=30),
            "ticker": "AAA",
            "f_ic_inst_new_buyer_ratio": bullish,
            "f_ic_inst_exit_ratio": ((index * 7) % 29) / 29,
            "f_ic_inst_concentration": ((index * index + 3) % 31) / 31,
        }
    )
    metadata = _write_snapshot(snapshot, frame)
    metadata_path = tmp_path / "candidate-meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(
        json.dumps(
            {
                "f_ic_inst_new_buyer_ratio": "bounded new-buyer share",
                "f_ic_inst_exit_ratio": "bounded exiting-holder share",
                "f_ic_inst_concentration": "bounded ownership concentration",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(quality, "get_config_context", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live context consulted")))

    out = tmp_path / "candidate-validation"
    assert (
        quality.main(
            [
                "validate-candidate",
                "--config",
                "configs",
                "--snapshot",
                str(snapshot),
                "--metadata",
                str(metadata_path),
                "--catalogue",
                str(catalogue),
                "--out",
                str(out),
            ]
        )
        == 1
    )
    result = json.loads((out / "timeseries.json").read_text(encoding="utf-8"))
    assert result["status"] == "fail"
    assert result["metrics"]["n_holes"] == 1
    print("SANITY: the parquet-only time-series gate files a six-session interior feature hole without consulting live values.")


def _write_manifest(path: Path, snapshot: Path, metadata: dict[str, object], store: object, as_of: str) -> None:
    payload = {
        "status": "pass",
        "table": "cube_part_institutionals",
        "as_of": as_of,
        "snapshot": {
            "frozen": str(snapshot.resolve()),
            "sha256": metadata["snapshot_sha256"],
            "rows": metadata["rows"],
            "columns": metadata["columns"],
            "dtypes": metadata["dtypes"],
            "first_date": metadata["first_date"],
            "last_date": metadata["last_date"],
        },
        "source_tables": [quality._source_metadata(store, table) for table in quality.SOURCE_TABLES],
        "peer_cache": _write_peer_cache(path.with_name("peer-baskets.json")),
    }
    path.write_text(json.dumps(quality.jsonable(payload)), encoding="utf-8")


def test_build_candidate_calls_full_panel_only_and_hashes_cutoff_output(tmp_path: Path, monkeypatch: object) -> None:
    baseline = tmp_path / "baseline.parquet"
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "AAA"],
            "f_ic_inst_holders": [1.0, 2.0],
        }
    )
    baseline_meta = _write_snapshot(baseline, source)
    store = _MetadataOnlyStore()
    manifest = tmp_path / "input-manifest.json"
    _write_manifest(manifest, baseline, baseline_meta, store, "2024-01-03")
    calls: list[bool] = []

    class _Step:
        def __init__(self, context: object, config: object) -> None:
            self.context = context

        def build_panel(self, full: bool = False) -> tuple[pd.DataFrame, object]:
            calls.append(full)
            future = pd.DataFrame({"date": [pd.Timestamp("2024-01-04")], "ticker": ["AAA"], "f_ic_inst_holders": [99.0]})
            return pd.concat([source, future], ignore_index=True), SimpleNamespace(is_full=True)

        def run(self, full: bool = False) -> None:
            raise AssertionError("run/write path must not be called")

    context = SimpleNamespace(store=store, paths={})
    monkeypatch.setattr(quality, "get_config_context", lambda *_args, **_kwargs: (SimpleNamespace(), context))
    monkeypatch.setattr(quality, "StepCubeInstitutionals", _Step)
    candidate = tmp_path / "candidate.parquet"
    out = tmp_path / "build"

    assert (
        quality.main(
            [
                "build-candidate",
                "--config",
                "configs",
                "--manifest",
                str(manifest),
                "--as-of",
                "2024-01-03",
                "--out-cache",
                str(candidate),
                "--out",
                str(out),
                "--compare-to",
                str(baseline),
                "--comparison-out",
                str(out / "comparison.json"),
            ]
        )
        == 0
    )
    metadata = json.loads((out / "candidate-metadata.json").read_text(encoding="utf-8"))
    assert calls == [True]
    assert pd.read_parquet(candidate)["date"].max() == pd.Timestamp("2024-01-03")
    assert metadata["snapshot_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert context.paths["SECTOR_PEERS_PATH"] == tmp_path / "peer-baskets.json"
    assert metadata["peer_cache"]["sha256"] == quality._sha256(context.paths["SECTOR_PEERS_PATH"])
    assert json.loads((out / "comparison.json").read_text(encoding="utf-8"))["status"] == "pass"
    stale = json.loads(manifest.read_text(encoding="utf-8"))
    stale["source_tables"][0]["rows"] += 1
    stale_manifest = tmp_path / "stale-manifest.json"
    stale_manifest.write_text(json.dumps(stale), encoding="utf-8")
    assert (
        quality.main(
            [
                "build-candidate",
                "--config",
                "configs",
                "--manifest",
                str(stale_manifest),
                "--as-of",
                "2024-01-03",
                "--out-cache",
                str(tmp_path / "must-not-build.parquet"),
                "--out",
                str(tmp_path / "must-not-build"),
            ]
        )
        == 2
    )
    assert calls == [True]
    print("SANITY: candidate build used build_panel(full=True), trimmed the future row, wrote no table, and matched the frozen baseline exactly.")


def test_taxonomy_reconciles_baseline_and_limits_peer_diagnostics(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.parquet"
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=8),
            "ticker": ["AAA"] * 8,
            "f_ic_inst_ownership_pct": np.linspace(0.1, 0.8, 8),
            "f_ic_inst_ownership_pct_vs_peers": np.linspace(-1.0, 1.0, 8),
            "f_ic_shortvol_ratio_20d": np.linspace(0.2, 0.5, 8),
            "f_ic_shortvol_ratio_20d_vs_peers": np.linspace(1.0, -1.0, 8),
            "f_ic_inst_holders_xs": np.linspace(0.1, 0.9, 8),
            "f_ic_inst_flow_to_mcap": np.linspace(-0.01, 0.01, 8),
            "f_ic_act_initial_13d": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    metadata = _write_snapshot(baseline, frame)
    manifest = tmp_path / "input-manifest.json"
    manifest.write_text(
        json.dumps(
            quality.jsonable(
                {
                    "table": "cube_part_institutionals",
                    "as_of": "2024-01-11",
                    "snapshot": {"frozen": str(baseline.resolve()), "sha256": metadata["snapshot_sha256"], **metadata},
                    "source_tables": [],
                    "peer_cache": _write_peer_cache(tmp_path / "peer-baskets.json"),
                }
            )
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.parquet"
    frame.drop(columns=["f_ic_inst_holders_xs", "f_ic_inst_flow_to_mcap"]).to_parquet(candidate, index=False)
    out = tmp_path / "taxonomy"

    assert (
        quality.main(
            [
                "taxonomy",
                "--config",
                "configs",
                "--snapshot",
                str(candidate),
                "--manifest",
                str(manifest),
                "--as-of",
                "2024-01-11",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    decisions = pd.read_csv(out / "feature-decisions.csv")
    peers = pd.read_csv(out / "peer-diagnostics.csv")
    schema = json.loads((out / "schema-diff.json").read_text(encoding="utf-8"))
    assert len(decisions) == len(frame.columns) - 2
    assert decisions.set_index("baseline_column").loc["f_ic_inst_holders_xs", "requested_decision"] == "remove_cross_sectional_normalization"
    assert decisions.set_index("baseline_column").loc["f_ic_inst_flow_to_mcap", "requested_decision"] == "remove_characteristic"
    assert decisions.set_index("baseline_column").loc["f_ic_act_initial_13d", "kind"] == "event"
    assert set(peers["characteristic"]) == {"ic_inst_ownership_pct", "ic_shortvol_ratio_20d"}
    assert set(schema["removed"]) == {"f_ic_inst_holders_xs", "f_ic_inst_flow_to_mcap"}
    assert (out / "model-fold-diagnostics.csv").exists()
    print(
        "SANITY: taxonomy reconciled retained and retired baseline features, used explicit event/normalization decisions, and measured only the two provisional peers."
    )


def test_analyze_uses_candidate_row_eligibility_and_reconciles_artifacts(tmp_path: Path, monkeypatch: object) -> None:
    snapshot = tmp_path / "candidate.parquet"
    dates = pd.bdate_range("2024-01-02", periods=10)
    frame = pd.DataFrame(
        {
            "date": np.tile(dates, 2),
            "ticker": ["AAA"] * 10 + ["BBB"] * 10,
            "f_ic_inst_holders": [np.nan, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan] + [np.nan] * 10,
            "f_ic_act_initial_13d": [np.nan, np.nan, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [np.nan] * 10,
        }
    )
    metadata = _write_snapshot(snapshot, frame)
    manifest = tmp_path / "input-manifest.json"
    manifest.write_text(
        json.dumps(
            quality.jsonable(
                {
                    "table": "cube_part_institutionals",
                    "as_of": "2024-01-15",
                    "snapshot": {"frozen": str(snapshot.resolve()), "sha256": metadata["snapshot_sha256"], **metadata},
                    "source_tables": [],
                    "peer_cache": _write_peer_cache(tmp_path / "peer-baskets.json"),
                }
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "analysis"
    monkeypatch.setattr(quality, "get_config_context", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("live context consulted")))

    assert (
        quality.main(
            [
                "analyze",
                "--config",
                "configs",
                "--table",
                "cube_part_institutionals",
                "--snapshot",
                str(snapshot),
                "--manifest",
                str(manifest),
                "--as-of",
                "2024-01-15",
                "--recent-sessions",
                "3",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    coverage = pd.read_csv(out / "coverage.csv")
    holder = coverage[(coverage["feature"] == "f_ic_inst_holders") & (coverage["ticker"] == "AAA")].iloc[0]
    unsupported = coverage[(coverage["feature"] == "f_ic_inst_holders") & (coverage["ticker"] == "BBB")].iloc[0]
    assert (holder["full_numerator"], holder["full_eligible_denominator"]) == (6, 7)
    assert unsupported["full_bucket"] == "no-support"
    assert set(coverage["full_bucket"]) <= {"100%", "70%-<100%", "50%-<70%", "30%-<50%", "<=30%", "no-support"}
    assert pd.read_csv(out / "coverage-full.csv")["reconciled"].all()
    assert pd.read_csv(out / "coverage-recent252.csv")["reconciled"].all()
    assert {"finite_rate", "null_rate", "zero_rate", "p05", "p95", "bound_breaches"}.issubset(pd.read_csv(out / "distributions.csv").columns)
    assert {"pearson", "spearman_sample", "overlap_n", "overlap_ratio", "identical_null_mask_rate", "disposition"}.issubset(
        pd.read_csv(out / "redundancy.csv").columns
    )
    assert {"coverage_ratio", "variance_ratio", "distribution_distance_ks", "disposition"}.issubset(pd.read_csv(out / "drift.csv").columns)
    assert json.loads((out / "leakage.json").read_text(encoding="utf-8"))["status"] == "abstain"
    summary = json.loads((out / "analysis-summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_reconciliation"]["pass"] is True
    print("SANITY: analysis reconciled feature-specific coverage buckets plus robust distribution, drift, and pairwise redundancy artifacts.")
