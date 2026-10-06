"""Offline fixtures for the EDGAR document driver: a fake `Context`, a seeded local index, fake filings.

The index cache is written through `edgar_index`'s own parser and writer, so tests read it exactly
as production does. Fake filings stand in for `edgar.Filing` by accession (`patch_index_filings`).
"""

from __future__ import annotations

import types
from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import edgar_index
from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_store.schema import Tables
from src.utils.string import pad_cik
from tests.data_extract.fake_context import extract_config
from tests.fixtures.superinvestor_config import write_roster_config

MASTER_HEADER = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    September 30, 2026
Comments:              webmaster@sec.gov
Anonymous FTP:         ftp://ftp.sec.gov/edgar/
Cloud HTTP:            https://www.sec.gov/Archives/




CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
"""


def master_text(rows: Iterable[tuple[int | str, str, str, str, str]]) -> str:
    """A `master.idx` body for `(cik, company, form, filed, accession)` rows."""
    lines = [f"{int(cik)}|{company}|{form}|{filed}|edgar/data/{int(cik)}/{accession}.txt" for cik, company, form, filed, accession in rows]
    return MASTER_HEADER + "\n".join(lines) + "\n"


def fake_context(tmp_path, store, tickers: list[str], ciks: list[str] | None = None, **data_extract: Any) -> Any:
    """A Context stand-in for the driver: `sp500_tickers` seeded with every column `load_cik_mapping` projects,
    and a `config_dir` holding empty roster overrides (`<config_dir>/superinvestors/overrides.json`)."""
    ciks = ciks or [str(i + 1) for i in range(len(tickers))]
    store.save(
        Tables.sp500_tickers,
        pd.DataFrame({col: [f"{col}-{t}" for t in tickers] for col in CIK_MAPPING_COLS} | {"ticker": tickers, "cik": ciks}),
    )
    warnings: list[str] = []
    errors: list[str] = []
    infos: list[str] = []
    ctx = types.SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=types.SimpleNamespace(
            info=lambda msg, *a: infos.append(msg % a if a else msg),
            warning=lambda msg, *a: warnings.append(msg % a if a else msg),
            error=lambda msg, *a: errors.append(msg % a if a else msg),
        ),
        config=extract_config(data_extract={"years_history": 15, **data_extract}),
        ensure_edgar_identity=lambda: None,
        config_dir=write_roster_config(tmp_path),
    )
    ctx.warnings, ctx.errors, ctx.infos = warnings, errors, infos
    return ctx


def seed_index(ctx: Any, rows: Iterable[tuple[int | str, str, str, str, str]]) -> pd.DataFrame:
    """Write `rows` into the local index cache, one Parquet file per quarter, through the production parser."""
    df = edgar_index.parse_master(master_text(rows).encode("latin-1"))
    directory = edgar_index.index_dir(ctx)
    for path in directory.glob("*.parquet"):
        path.unlink()
    quarters = pd.Series([edgar_index.Quarter.of(pd.Timestamp(str(d))).name for d in df["filed"]], index=df.index)
    for name, part in df.groupby(quarters):
        edgar_index._write_atomic(part.reset_index(drop=True), directory / f"{name}.parquet")
    return df


def patch_index_filings(monkeypatch: pytest.MonkeyPatch, filings: dict[str, Any]) -> None:
    """Make the driver read `filings[accession]` instead of building an `edgar.Filing` from the index row."""
    real = edgar_index.index_filing

    def _filing(cik: object, company: object, form: object, filed: object, accession: object) -> Any:
        return filings.get(str(accession)) or real(cik, company, form, filed, accession)

    monkeypatch.setattr(edgar_index, "index_filing", _filing)


def fake_filing(accession: str, cik: int | str, filed: str, form: str = "8-K", **attrs: Any) -> types.SimpleNamespace:
    """A filing stand-in carrying what `FilingStamp.of` reads, plus `attrs`."""
    return types.SimpleNamespace(accession_number=accession, cik=int(cik), form=form, filing_date=filed, company="Fixture Co", **attrs)


def identity_for(roster: dict[str, str]) -> Identity:
    """A real `Identity` where each roster ticker is its own single-CIK entity owning its symbol."""
    return build_identity(
        lineage=pd.DataFrame([{"cik": pad_cik(cik), "entity_id": f"E{pad_cik(cik)}", "source": "roster"} for cik in roster.values()]),
        tenure=pd.DataFrame(
            [
                {
                    "symbol": t,
                    "issuer_cik": pad_cik(c),
                    "valid_from": pd.Timestamp("2000-01-01"),
                    "valid_to": None,
                    "n_filings": 10,
                    "source": "form345",
                }
                for t, c in roster.items()
            ]
        ),
        roster=pd.DataFrame([{"ticker": t, "cik": pad_cik(c)} for t, c in roster.items()]),
    )
