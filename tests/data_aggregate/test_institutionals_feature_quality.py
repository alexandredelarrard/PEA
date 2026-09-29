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
    monkeypatch.setattr(quality, "get_config_context", lambda *_args, **_kwargs: (None, SimpleNamespace(store=store)))

    out = tmp_path / "frozen"
    assert quality.main(["freeze-baseline", "--config", "configs", "--snapshot", str(snapshot), "--as-of", "2024-01-03", "--out", str(out)]) == 0

    frozen = out / "baseline.parquet"
    manifest = json.loads((out / "input-manifest.json").read_text(encoding="utf-8"))
    assert frozen.read_bytes() == snapshot.read_bytes()
    assert manifest["status"] == "pass"
    assert manifest["snapshot"]["sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
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
            "f_ic_xs_bullish_family_ratio": [0.1, 0.2, 0.3, 0.4, 0.8, 0.7, 0.6, 0.5],
            "f_ic_xs_bearish_family_ratio": [0.9, 0.3, 0.8, 0.2, 0.1, 0.7, 0.4, 0.6],
            "f_ic_xs_conflict_ratio": [0.05, 0.2, 0.1, 0.4, 0.3, 0.15, 0.45, 0.25],
            "f_ic_demo": [2.0, 1.0, 4.0, 3.0, 1.0, 4.0, 2.0, 5.0],
        }
    )
    metadata = _write_snapshot(snapshot, frame)
    metadata_path = tmp_path / "candidate-meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(
        json.dumps(
            {
                "f_ic_xs_bullish_family_ratio": "bounded cross-source vote",
                "f_ic_xs_bearish_family_ratio": "bounded cross-source vote",
                "f_ic_xs_conflict_ratio": "bounded cross-source conflict",
                "f_ic_demo": "fixture feature",
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
            "f_ic_xs_bullish_family_ratio": bullish,
            "f_ic_xs_bearish_family_ratio": ((index * 7) % 29) / 29,
            "f_ic_xs_conflict_ratio": ((index * index + 3) % 31) / 31,
        }
    )
    metadata = _write_snapshot(snapshot, frame)
    metadata_path = tmp_path / "candidate-meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(
        json.dumps(
            {
                "f_ic_xs_bullish_family_ratio": "bounded cross-source vote",
                "f_ic_xs_bearish_family_ratio": "bounded cross-source vote",
                "f_ic_xs_conflict_ratio": "bounded cross-source conflict",
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
