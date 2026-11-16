"""Manager category from the cleaned 13F manager name, by the ordered keyword rules of `configs/manager_categories.yml`."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig


def _alternation(patterns: list[str]) -> str:
    """One regex matching any pattern on word boundaries."""
    return r"(?<![A-Z0-9])(?:" + "|".join(patterns) + r")(?![A-Z0-9])"


def clean_names(names: pd.Series, clean: DictConfig) -> pd.Series:
    """Upper-case ASCII names without punctuation, state tags, legal-form tokens or a trailing company word."""
    # Fold accents, upper-case, spell out the ampersand and drop /STATE/ tags.
    text = names.fillna("").astype(str).str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.upper()
    text = text.str.replace("&", " AND ", regex=False).str.replace(r"/[A-Z]{2,5}/?", " ", regex=True)

    # Remove dots and apostrophes inside tokens (L.L.C. reads LLC), then split on any other punctuation.
    text = text.str.replace(r"[.']", "", regex=True).str.replace(r"[^A-Z0-9]+", " ", regex=True)

    # Drop legal-form tokens everywhere, then trailing company words at the end of the name.
    text = text.str.replace(_alternation(list(clean.drop_tokens)), " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    trailing = "|".join(clean.trailing_tokens)
    return text.str.replace(rf"(?:\s(?:{trailing}))+$", "", regex=True).str.strip()


def classify_names(names: pd.Series, rules: DictConfig) -> pd.Series:
    """Per name its category: the first matching rule in config order, else `unknown`."""
    # Classify each distinct name once.
    unique = pd.Series(names.fillna("").astype(str).unique())
    cleaned = clean_names(unique, rules.clean)

    # Walk the rules in order; a name keeps the first rule it matches.
    category = pd.Series("unknown", index=unique.index)
    for rule in rules.rules:
        hit = cleaned.str.contains(_alternation(list(rule.patterns)), regex=True)
        if rule.get("unless"):
            hit &= ~cleaned.str.contains(_alternation(list(rule.unless)), regex=True)
        hit &= category.eq("unknown")
        category[hit] = rule.category

    # Map the distinct results back to every input name.
    out = pd.Series(category.to_numpy(), index=pd.Index(unique.to_numpy()), name="category").reindex(names.fillna("").astype(str).to_numpy())
    out.index = names.index
    return out
