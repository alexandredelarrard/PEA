"""Manager category from the cleaned 13F manager name, by the ordered keyword rules of `configs/manager_categories.yml`, then the CIK overrides."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.utils.string import alternation
from src.utils.superinvestor_roster import superinvestors_config_dir

OVERRIDES_FILENAME = "manager_categories.json"


def clean_names(names: pd.Series, clean: DictConfig) -> pd.Series:
    """Upper-case ASCII names without punctuation, state tags, legal-form tokens or a trailing company word."""
    # Fold accents, upper-case, spell out the ampersand and drop /STATE/ tags.
    text = names.fillna("").astype(str).str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.upper()
    text = text.str.replace("&", " AND ", regex=False).str.replace(r"/[A-Z]{2,5}/?", " ", regex=True)

    # Remove dots and apostrophes inside tokens (L.L.C. reads LLC), then split on any other punctuation.
    text = text.str.replace(r"[.']", "", regex=True).str.replace(r"[^A-Z0-9]+", " ", regex=True)

    # Drop legal-form tokens everywhere, then trailing company words at the end of the name.
    text = text.str.replace(alternation(list(clean.drop_tokens)), " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    trailing = "|".join(clean.trailing_tokens)
    return text.str.replace(rf"(?:\s(?:{trailing}))+$", "", regex=True).str.strip()


def classify_names(names: pd.Series, rules: DictConfig) -> pd.Series:
    """Per name its category: the first matching rule in config order, else `rules.default_category`."""
    # Classify each distinct name once.
    unique = pd.Series(names.fillna("").astype(str).unique())
    cleaned = clean_names(unique, rules.clean)

    # Walk the rules in order; a name keeps the first rule it matches.
    category = pd.Series(rules.default_category, index=unique.index)
    assigned = pd.Series(False, index=unique.index)
    for rule in rules.rules:
        hit = cleaned.str.contains(alternation(list(rule.patterns)), regex=True)
        if rule.get("unless"):
            hit &= ~cleaned.str.contains(alternation(list(rule.unless)), regex=True)
        hit &= ~assigned
        category[hit] = rule.category
        assigned |= hit

    # Map the distinct results back to every input name.
    out = pd.Series(category.to_numpy(), index=pd.Index(unique.to_numpy()), name="category").reindex(names.fillna("").astype(str).to_numpy())
    out.index = names.index
    return out


def load_category_overrides(config_dir: Path | str, rules: DictConfig) -> dict[str, str]:
    """CIK -> category from `<config_dir>/superinvestors/manager_categories.json`; every CIK is ten digits and every category is a legal one."""
    path = superinvestors_config_dir(config_dir) / OVERRIDES_FILENAME
    entries = json.loads(path.read_text(encoding="utf-8"))["overrides"]
    bad_cik = sorted(cik for cik in entries if not re.fullmatch(r"[0-9]{10}", cik))
    legal = set(rules.categories)
    bad_category = sorted(cik for cik, entry in entries.items() if entry["category"] not in legal)
    if bad_cik or bad_category:
        raise ValueError(f"{path}: CIKs not ten digits {bad_cik}; categories outside {list(rules.categories)} for {bad_category}")
    return {cik: entry["category"] for cik, entry in entries.items()}


def apply_overrides(category: pd.Series, ciks: pd.Series, overrides: dict[str, str]) -> pd.Series:
    """`category` with the hand override of each listed CIK in place of the rule result."""
    return category.mask(ciks.isin(list(overrides)), ciks.map(overrides))
