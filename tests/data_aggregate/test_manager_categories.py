"""Manager categories: cleaned-name rules assign one category per 13F manager name.

Known-truth names are real `sec_13f_manager_details.manager_name` values; the expected category is the name's plain meaning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data_aggregate.utils.manager_profile.categories import apply_overrides, classify_names, clean_names, load_category_overrides
from src.utils.config import read_config

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("pension", "insurance", "bank", "broker_dealer", "sovereign", "fund", "other")

KNOWN_TRUTH = [
    # pension
    ("CALIFORNIA STATE TEACHERS RETIREMENT SYSTEM", "pension"),
    ("Maryland State Retirement & Pension System", "pension"),
    ("CANADA PENSION PLAN INVESTMENT BOARD", "pension"),
    ("TRUSTEES OF GENERAL ELECTRIC PENSION TRUST", "pension"),
    ("Retirement Investment Advisors, Inc.", "fund"),
    ("Pensionmark Financial Group, LLC", "fund"),
    # sovereign
    ("NORGES BANK", "sovereign"),
    ("Saudi Central Bank", "sovereign"),
    ("TEMASEK HOLDINGS (PRIVATE) LTD", "sovereign"),
    ("Sovereign Investment Advisors, LLC", "fund"),
    ("CENTRAL BANK & TRUST CO", "bank"),
    # insurance
    ("TEACHERS INSURANCE & ANNUITY ASSOCIATION OF AMERICA", "insurance"),
    ("Penn Mutual Life Insurance Co", "insurance"),
    ("SWISS REINSURANCE CO LTD", "insurance"),
    ("PROTECTIVE LIFE CORP", "insurance"),
    # broker-dealer / prop
    ("BNP PARIBAS SECURITIES CORP", "broker_dealer"),
    ("Virtu Financial BD LLC", "broker_dealer"),
    ("GOLDMAN SACHS GROUP INC", "broker_dealer"),
    ("MORGAN STANLEY", "broker_dealer"),
    # bank
    ("ROCKLAND TRUST CO", "bank"),
    ("BANK OF NEW YORK TRUST COMPANY, N.A.", "bank"),
    ("Bath Savings Trust Co", "bank"),
    ("STATE STREET Corp", "bank"),
    # other (non-profit, family)
    ("George Kaiser Family Foundation", "other"),
    ("University of Illinois Foundation", "other"),
    ("Edgehill Endowment Partners, LLC", "fund"),
    ("Foundation Wealth Management, LLC\\PA", "fund"),
    # fund (asset managers, advisers, hedge funds)
    ("ARES MANAGEMENT LLC", "fund"),
    ("Third Coast Capital Management, L.P.", "fund"),
    ("COGNIOS BETA NEUTRAL LARGE CAP FUND, LP", "fund"),
    ("Kuhn & Co Investment Counsel", "fund"),
    ("Nishkama Capital, LLC", "fund"),
    ("CITADEL ADVISORS LLC", "fund"),
    ("Hennion & Walsh Asset Management", "fund"),
    # no rule matches: the default category
    ("KORNREICH JOHN", "fund"),
    ("GROUP VI 31 LLC", "fund"),
    # keyword additions
    ("Acme Capital GP, LLC", "fund"),
    ("TRUST CO OF CONNECTICUT", "bank"),
    ("Aberdare Gestao de Recursos Ltda", "fund"),
    ("KLP KAPITALFORVALTNING AS", "fund"),
    ("Handelsbanken Fonder AB", "bank"),
    ("Alecta Tjanstepension Omsesidigt", "pension"),
    ("Swedbank AB", "bank"),
    ("COWEN AND COMPANY, LLC", "broker_dealer"),
    # insurance fixes
    ("Assured Investment Management LLC", "fund"),
    ("Shelter Ins Profit Sharing Plan", "fund"),
    ("MetLife Investment Management, LLC", "fund"),
    # words the additions must not catch
    ("Acacia Research Corp", "fund"),
    ("Jay G Goldman", "fund"),
    ("Oppenheimer and Close", "fund"),
]


@pytest.fixture(scope="module")
def rules() -> DictConfig:
    return read_config(str(ROOT / "configs")).manager_categories


def test_clean_names_strips_legal_forms_and_trailing_and_co(rules: DictConfig) -> None:
    """Punctuation, accents, /state/ tags, legal forms and a trailing '& Co' disappear; inner words stay."""
    names = pd.Series(
        [
            "Brown Brothers Harriman & Co.",
            "ABC CAPITAL MANAGEMENT, L.L.C.",
            "MCDONALD INVESTMENTS INC /OH/",
            "Lansforsakringar Fondforvaltning AB (publ)",
            "Zürich Insurance Group Ltd",
            "ROCKLAND TRUST CO",
        ]
    )
    cleaned = clean_names(names, rules.clean)
    print(cleaned.tolist())
    assert cleaned.tolist() == [
        "BROWN BROTHERS HARRIMAN",
        "ABC CAPITAL MANAGEMENT",
        "MCDONALD INVESTMENTS",
        "LANSFORSAKRINGAR FONDFORVALTNING",
        "ZURICH INSURANCE GROUP",
        "ROCKLAND TRUST",
    ]


def test_known_truth_names_get_their_category(rules: DictConfig) -> None:
    """Every known-truth name lands in the expected category, ."""
    names = pd.Series([name for name, _ in KNOWN_TRUTH])
    out = classify_names(names, rules)
    got = dict(zip(names, out, strict=True))
    wrong = {name: (want, got[name]) for name, want in KNOWN_TRUTH if got[name] != want}
    print(f"known-truth names: {len(KNOWN_TRUTH) - len(wrong)}/{len(KNOWN_TRUTH)} correct; wrong={wrong}")
    assert not wrong
    assert set(out) <= {*CATEGORIES}


def test_first_matching_rule_wins_in_config_order(rules: DictConfig) -> None:
    """An insurance name that also reads as a pension plan stays insurance because insurance is listed first."""
    out = classify_names(pd.Series(["Teachers Insurance Annuity Retirement Plan"]), rules)
    print(out.tolist())
    assert out.iloc[0] == "insurance"


def test_rule_config_is_complete(rules: DictConfig) -> None:
    """Every rule carries a one-line description, a legal category and at least one pattern; category values are the spec set."""
    assert rules.default_category in rules.categories and "unknown" not in rules.categories
    names = [rule.name for rule in rules.rules]
    print(f"{len(names)} rules; categories used: {sorted({rule.category for rule in rules.rules})}")
    assert len(set(names)) == len(names)
    for rule in rules.rules:
        assert rule.category in CATEGORIES and rule.category in rules.categories
        assert rule.description and "\n" not in rule.description
        assert len(rule.patterns) >= 1
    print("SANITY: rule config is well formed and the rules classify the known-truth names")


def test_override_file_lists_legal_categories_and_beats_the_name_rules(rules: DictConfig) -> None:
    """Every override CIK is ten digits with a legal category; each reads its mapped category whatever its name says."""
    overrides = load_category_overrides(ROOT / "configs", rules)
    names = pd.DataFrame({"cik": list(overrides), "manager_name": "ZZ PLAIN NAME"})
    by_rule = classify_names(names["manager_name"], rules)

    out = apply_overrides(by_rule, names["cik"], overrides)

    assert len(overrides) >= 90 and set(out) <= set(CATEGORIES)
    assert out.tolist() == list(overrides.values()) and set(by_rule) == {rules.default_category}
    assert overrides["0001067983"] == "insurance", "Berkshire Hathaway is read by CIK, not by its name"
    assert apply_overrides(pd.Series(["fund"]), pd.Series(["0000000001"]), overrides).tolist() == ["fund"], "an unlisted CIK keeps its rule category"
    print(f"{len(overrides)} overrides, by category: {pd.Series(overrides).value_counts().to_dict()}")
    print("SANITY: overrides win over the rules and leave unlisted CIKs alone")


def test_override_loader_rejects_a_bad_cik_or_category(rules: DictConfig, tmp_path: Path) -> None:
    """A malformed CIK or a category outside the legal set stops the load with both lists in the message."""
    folder = tmp_path / "superinvestors"
    folder.mkdir()
    (folder / "manager_categories.json").write_text(
        json.dumps({"overrides": {"123": {"category": "fund"}, "0000000002": {"category": "unknown"}, "0000000003": {"category": "bank"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not ten digits.*123.*outside.*0000000002"):
        load_category_overrides(tmp_path, rules)
    print("SANITY: a bad override file raises a clear error")
