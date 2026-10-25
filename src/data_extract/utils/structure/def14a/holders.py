"""Conservative holder names and same-filing reconciliation for saved proxy rows."""

from __future__ import annotations

import re

import pandas as pd

from src.data_extract.utils.structure.def14a.validate import clean_holder_name


def _holder_name(value: str) -> str:
    """Remove an explicit postal-address tail without dropping entity or person suffixes."""
    name = clean_holder_name(value) or value

    # Comma-separated US postal addresses escape the legacy whitespace-only cleaner.
    parts = re.split(r",\s*(?=\d+\s)", name, maxsplit=1)
    if len(parts) == 2 and re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", parts[1]):
        name = parts[0]
    return name.strip()


def reconcile_holders(df_saved: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Fill missing cells of exact name aliases only at the same filing date and holder type."""
    df_rows = pd.concat([df_saved, df_new], ignore_index=True, sort=False)
    if df_rows.empty:
        return df_rows
    columns = list(df_rows.columns)
    df_rows["as_of"] = pd.to_datetime(df_rows["as_of"])

    # Saved non-null cells precede repair cells; punctuation cannot change an identity.
    df_rows["holder_name"] = df_rows.holder_name.map(_holder_name)
    df_rows["_holder_identity"] = (
        df_rows.holder_name.str.casefold()
        .str.replace("&", " and ")
        .str.replace(r"[\W_]+", " ", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    keys = ["ticker", "accession_number", "as_of", "holder_type", "_holder_identity"]
    return df_rows.groupby(keys, as_index=False, sort=False, dropna=False).first().reindex(columns=columns)


def holders_need_reconciliation(df_saved: pd.DataFrame) -> bool:
    """Find saved aliases or address labels even when all facts and attempts are complete."""
    if df_saved.empty or "holder_name" not in df_saved:
        return False
    df_normalized = reconcile_holders(df_saved, pd.DataFrame())
    return len(df_normalized) != len(df_saved) or not df_normalized.holder_name.reset_index(drop=True).equals(
        df_saved.holder_name.reset_index(drop=True)
    )
