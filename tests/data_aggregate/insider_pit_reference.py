"""Brute-force point-in-time reference for the insider panel.

`asof_view` rebuilds, with plain row loops, what a builder may know on a date: the rows filed by
then, repeat copies collapsed to their earliest filing, and every linked amendment cell replaced
by its latest restatement, which keeps the first record's filing day. `reference_panel` feeds
each as-of view to the builder as plain rows and keeps only the dates that view covers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions

#: Run-dir parquet snapshot of the stored zip rows, 2024q1-2026q2 (read-only).
SNAPSHOT = Path("reports") / "validate" / "2026-10-03-insider-transactions-merge" / "_cache" / "zip_2024q1_2026q2.parquet"

KEY = ("ticker", "transaction_date", "transaction_code", "shares", "price_per_share", "shares_owned_after")
CELL = ("transaction_date", "transaction_code", "security_type")


def snapshot_path() -> Path | None:
    """The snapshot under the working directory or any parent, else None."""
    cwd = Path.cwd().resolve()
    return next((base / SNAPSHOT for base in (cwd, *cwd.parents) if (base / SNAPSHOT).exists()), None)


def row(**kw: object) -> dict:
    """One in-scope Form 4 purchase row with every column the cleaning and the builder read."""
    base: dict = dict(
        accession_number="0000000000-24-000000",
        ticker="AAA",
        owner_cik="0000000001",
        owner_ciks=None,
        n_reporting_owners=1,
        owner_name="Doe Jane",
        filing_date="2024-01-10",
        transaction_date="2024-01-08",
        transaction_code="P",
        shares=1_000.0,
        price_per_share=100.0,
        value_usd=None,
        shares_owned_after=11_000.0,
        security_type="nonderiv",
        security_title="Common Stock",
        direct_indirect="D",
        officer_title="",
        is_director=1.0,
        is_officer=0.0,
        is_ten_pct_owner=0.0,
        is_10b5_1=0.0,
        document_type="4",
        original_submission_date=None,
        source="edgar",
        row_sequence=1,
    )
    base.update(kw)
    if base["owner_ciks"] is None:
        base["owner_ciks"] = base["owner_cik"]
    if base["value_usd"] is None and base["price_per_share"] is not None:
        base["value_usd"] = float(base["shares"]) * float(base["price_per_share"])  # type: ignore[arg-type]
    return base


def sentinel_rows() -> list[dict]:
    """A buy, a discretionary and a planned sale on `ZZZ`, filed before any grid day and never compared.

    The builder reads an empty leg (no sale filed anywhere yet) as NaN for every ticker; these
    rows keep each leg non-empty in every as-of view, so that property cannot pose as a difference.
    """
    lead = dict(ticker="ZZZ", filing_date="2023-06-01", transaction_date="2023-05-30", owner_cik="0000000999")
    return [
        row(accession_number="0000000000-23-000901", **lead),
        row(accession_number="0000000000-23-000902", **lead, transaction_code="S", shares_owned_after=10_000.0),
        row(accession_number="0000000000-23-000903", **lead, transaction_code="S", shares_owned_after=9_000.0, is_10b5_1=1.0),
    ]


def frame(rows: list[dict]) -> pd.DataFrame:
    """Rows as the store returns them: DATE columns as `datetime.date`."""
    df = pd.DataFrame(rows)
    for column in ("filing_date", "transaction_date", "original_submission_date"):
        df[column] = pd.to_datetime(df[column]).dt.date
    return df


def _round(value: object) -> float:
    """Half away from zero at 2 decimals on the shortest decimal text of the float, as the SEC zip rounds."""
    number = float(value) if value is not None else np.nan  # type: ignore[arg-type]
    if not np.isfinite(number):
        return number
    return float(Decimal(repr(number)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _key(r: pd.Series) -> tuple | None:
    parts = (
        r["ticker"],
        r["transaction_date"],
        r["transaction_code"],
        _round(r["shares"]),
        _round(r["price_per_share"]),
        _round(r["shares_owned_after"]),
    )
    return None if any(pd.isna(p) for p in parts) else parts


def _cell(r: pd.Series) -> tuple:
    return tuple(r[c] for c in CELL)


def _record(r: pd.Series) -> tuple:
    return (_round(r["shares"]), _round(r["price_per_share"]), _round(r["shares_owned_after"]))


def _owners(r: pd.Series) -> set[str]:
    return {o.strip().lstrip("0") for o in str(r["owner_ciks"]).split(",") if o.strip()}


def asof_view(df_rows: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """The plain rows a builder may know on `as_of`, with each kept amendment row re-dated to its anchor."""
    filed = pd.to_datetime(df_rows["filing_date"])
    seen = df_rows[filed <= as_of].copy()
    seen["_day"] = pd.to_datetime(seen["filing_date"])
    by_acc: dict[str, pd.DataFrame] = {str(acc): g for acc, g in seen.groupby("accession_number", sort=False)}

    links: dict[str, str] = {}
    for acc, g in by_acc.items():
        first = g.iloc[0]
        submitted = pd.to_datetime(first["original_submission_date"])
        if not str(first["document_type"]).endswith("/A") or pd.isna(submitted) or submitted > first["_day"]:
            continue
        candidates = []
        for other, o in by_acc.items():
            lead = o.iloc[0]
            if str(lead["document_type"]).endswith("/A") or lead["ticker"] != first["ticker"] or lead["_day"] != submitted:
                continue
            if _owners(first) & _owners(lead):
                shared = len({_cell(r) for _, r in g.iterrows()} & {_cell(r) for _, r in o.iterrows()})
                candidates.append((-shared, other))
        if candidates:
            links[acc] = str(min(candidates)[1])

    groups: dict[tuple, list] = defaultdict(list)
    for i, r in seen[~seen["accession_number"].isin(list(links))].iterrows():
        key = _key(r)
        if key is not None:
            groups[key].append(i)
    drop: set = set()
    for members in groups.values():
        accessions = {str(seen.at[i, "accession_number"]) for i in members}
        if len(accessions) < 2:
            continue
        keeper = min(accessions, key=lambda a: (by_acc[a]["_day"].iloc[0], a))
        drop |= {i for i in members if seen.at[i, "accession_number"] != keeper}

    redate: dict = {}
    for original in set(links.values()):
        chain = [original] + sorted((a for a, o in links.items() if o == original), key=lambda a: (by_acc[a]["_day"].iloc[0], a))
        cells = {_cell(r) for acc in chain for _, r in by_acc[acc].iterrows()}
        for cell in cells:
            reporters = [acc for acc in chain if any(_cell(r) == cell for _, r in by_acc[acc].iterrows())]
            rows_of: dict[str, list] = {acc: [i for i, r in by_acc[acc].iterrows() if _cell(r) == cell] for acc in reporters}
            current = reporters[0]
            for acc in reporters[1:]:
                if Counter(_record(by_acc[acc].loc[i]) for i in rows_of[acc]) != Counter(_record(by_acc[current].loc[i]) for i in rows_of[current]):
                    current = acc
            for acc in reporters:
                if acc != current:
                    drop |= set(rows_of[acc])
            anchor = by_acc[reporters[0]]["_day"].iloc[0]
            redate.update(dict.fromkeys(rows_of[current], anchor))

    view = seen.drop(index=list(drop)).drop(columns="_day")
    for i, anchor in redate.items():
        if i in view.index:
            view.at[i, "filing_date"] = anchor.date()
    view["document_type"] = "4"
    view["original_submission_date"] = None
    return view


def reference_panel(df_rows: pd.DataFrame, idx: pd.DatetimeIndex, build: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Every (date, ticker) of `idx` valued by `build` on the as-of view of that date; one build per filing day."""
    filing_days = sorted(pd.to_datetime(df_rows["filing_date"]).unique())
    epochs = [*filing_days, pd.Timestamp.max]
    parts = []
    for start, stop in zip(epochs[:-1], epochs[1:], strict=True):
        dates = idx[(idx >= start) & (idx < stop)]
        if dates.empty:
            continue
        view = asof_view(df_rows, pd.Timestamp(start))
        _, diag = clean_transactions(view)
        assert diag.get("copy_groups", 0) == 0 and diag.get("amendments_linked", 0) == 0, "the as-of view must already be plain"
        panel = build(view)
        if not panel.empty:
            parts.append(panel[panel["date"].isin(dates)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "ticker"])


def compare(new: pd.DataFrame, ref: pd.DataFrame, idx: pd.DatetimeIndex, tickers: list[str], *, rtol: float = 1e-12) -> dict[str, dict]:
    """Per feature column: cells compared, cells differing beyond `rtol` (plus `rtol` x the column scale), max abs diff."""
    grid = pd.MultiIndex.from_product([idx, tickers], names=["date", "ticker"])
    columns = sorted((set(new.columns) | set(ref.columns)) - {"date", "ticker"})
    a = new.set_index(["date", "ticker"]).reindex(index=grid, columns=columns)
    b = ref.set_index(["date", "ticker"]).reindex(index=grid, columns=columns)
    out: dict[str, dict] = {}
    for column in columns:
        x, y = a[column].to_numpy(dtype="float64"), b[column].to_numpy(dtype="float64")
        scale = np.nanmax(np.abs(y)) if np.isfinite(y).any() else 0.0
        both_nan = np.isnan(x) & np.isnan(y)
        close = np.isclose(x, y, rtol=rtol, atol=rtol * scale) | both_nan
        diff = np.abs(np.where(both_nan, 0.0, x - y))
        out[column] = {
            "cells": int(len(x)),
            "non_null": int(np.isfinite(y).sum()),
            "mismatch": int((~close).sum()),
            "max_abs_diff": float(np.nanmax(diff)),
        }
    return out
