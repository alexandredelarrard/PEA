"""Conservative holder names and same-filing reconciliation for saved proxy rows."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.data_extract.utils.structure.def14a.validate import clean_holder_name


def normalize_name(value: str) -> str:
    """Canonical name identity without parenthetical text, accents or punctuation."""
    text = value
    parentheses = re.compile(r"\([^()]*\)")

    # Remove every complete parenthetical group, including nested groups.
    while parentheses.search(text):
        text = parentheses.sub(" ", text)

    # Preserve name tokens while removing case, accent and punctuation differences.
    text = "".join(character for character in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(character))
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def _holder_name(value: str) -> str:
    """Remove explicit postal tails and terminal letter footnotes, retaining readable names."""
    name = clean_holder_name(value) or value

    # A source footnote may sit directly before whitespace and the street number.
    parts = re.split(r",\s*(?=\d+\s)|(?<=\([a-z]\))\s+(?=\d+\s)", name, maxsplit=1)
    if len(parts) == 2 and re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", parts[1]):
        name = parts[0]
    return re.sub(r"(?:\s*\([A-Za-z]\))+\s*$", "", name).strip()


def reconcile_holders(df_saved: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Fill missing cells of exact name aliases only at the same filing date and holder type."""
    df_rows = pd.concat([df_saved, df_new], ignore_index=True, sort=False)
    if df_rows.empty:
        return df_rows
    columns = list(df_rows.columns)
    df_rows["as_of"] = pd.to_datetime(df_rows["as_of"])

    # Explicit security qualifiers cannot supply issuer facts when an issuer row exists.
    df_rows["_security_qualified"] = df_rows.holder_name.str.contains(r"\([^)]*\b(?:stock|shares|class)\b[^)]*\)", case=False, regex=True)

    # Saved comparable non-null cells precede repair cells under one shared name key.
    df_rows["holder_name"] = df_rows.holder_name.map(_holder_name)
    df_rows["_holder_identity"] = df_rows.holder_name.map(normalize_name)
    keys = ["ticker", "accession_number", "as_of", "holder_type", "_holder_identity"]
    has_issuer = df_rows.groupby(keys, sort=False, dropna=False)["_security_qualified"].transform("min").eq(False)
    df_rows = df_rows.loc[~df_rows._security_qualified | ~has_issuer]
    return df_rows.groupby(keys, as_index=False, sort=False, dropna=False).first().reindex(columns=columns)


def holders_need_reconciliation(df_saved: pd.DataFrame) -> bool:
    """Find saved aliases or address labels even when all facts and attempts are complete."""
    if df_saved.empty or "holder_name" not in df_saved:
        return False
    df_normalized = reconcile_holders(df_saved, pd.DataFrame())
    return len(df_normalized) != len(df_saved) or not df_normalized.holder_name.reset_index(drop=True).equals(
        df_saved.holder_name.reset_index(drop=True)
    )
