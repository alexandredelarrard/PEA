"""Name rules for the common lines Sharadar does not list: ETF, real estate, foreign and generic US titles.

Known-truth strings are real `sec13f_hr` `title_of_class | issuer_name` values, including the documented false-positive traps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data_aggregate.utils.manager_profile.reference import classify_securities, sharadar_reference
from src.utils.config import read_config

ROOT = Path(__file__).resolve().parents[2]

# (title_of_class, issuer_name, asset, sector_col)
UNLISTED = [
    # ETF rule on the issuer or the title
    ("MSCI EMERG MKT", "ISHARES TR", "etf", "pct_sector_unmapped"),
    ("COM", "SPDR S&P 500 ETF    TRUST", "etf", "pct_sector_unmapped"),
    ("TRUST", "SELECT SECTOR SPDR TR FINL", "etf", "pct_sector_unmapped"),
    ("UNIT SER 1", "POWERSHARES QQQ TRUST", "etf", "pct_sector_unmapped"),
    ("Exchange Traded Product", "SPDR S&P GLOBAL NATURAL RESOURCES ETF", "etf", "pct_sector_unmapped"),
    ("COM", "VANGUARD REAL ESTATE ETF", "etf", "pct_sector_unmapped"),
    # traps that are operating companies: a generic title makes them US equity
    ("COM", "AMERICAN VANGUARD CORP", "equity_us", "pct_sector_unmapped"),
    ("COM", "INVESCO LTD", "equity_us", "pct_sector_unmapped"),
    ("COMMON STOCK", "EATON CORP PLC", "equity_us", "pct_sector_unmapped"),
    ("COM", "S&P GLOBAL INC", "equity_us", "pct_sector_unmapped"),
    ("CS", "Becton Dickinson and Co", "equity_us", "pct_sector_unmapped"),
    ("COM", "UNIVERSAL AMERN CORP NEW", "equity_us", "pct_sector_unmapped"),
    ("COM", "TRUSTCO BK CORP N Y", "equity_us", "pct_sector_unmapped"),
    # bare TRUST, UNIT, ORD, SHS and N V carry no class
    ("TRUST", "BP PRUDHOE BAY ROYALTY TRUST", "other", "pct_sector_unmapped"),
    ("ORD", "AMDOCS LTD", "other", "pct_sector_unmapped"),
    ("SHS", "FREESCALE SEMICONDUCTOR LTD", "other", "pct_sector_unmapped"),
    ("SHS", "ASML HOLDING N V N Y", "other", "pct_sector_unmapped"),
    # real estate words give the sector; the asset class follows the later steps
    ("SH BEN INT", "LIBERTY PPTY TR", "other", "pct_sector_real_estate"),
    ("UNIT LTD PARTN", "BROOKFIELD PPTY PARTNERS L P", "other", "pct_sector_real_estate"),
    ("Real Estate Investment Trust", "SL GREEN REALTY CORP", "other", "pct_sector_real_estate"),
    ("COM", "WEINGARTEN RLTY INVS", "equity_us", "pct_sector_real_estate"),
    # strong foreign set
    ("SPONSORED ADR", "ASTRAZENECA PLC", "equity_foreign", "pct_sector_unmapped"),
    ("COM", "GLAXOSMITHKLINE PLC-SPON ADR", "equity_foreign", "pct_sector_unmapped"),
    ("NAMEN AKT", "DEUTSCHE BANK AG", "equity_foreign", "pct_sector_unmapped"),
    ("SHS NEW", "UBS AG", "equity_foreign", "pct_sector_unmapped"),
    # generic US titles
    ("COM", "WYETH", "equity_us", "pct_sector_unmapped"),
    ("COMMON STOCK USD.01", "GENENTECH INC", "equity_us", "pct_sector_unmapped"),
    ("CL B", "BERKSHIRE HATHAWAY INC", "equity_us", "pct_sector_unmapped"),
    ("CAP STK CL A", "ALPHABET INC", "equity_us", "pct_sector_unmapped"),
    ("COM PAR $0.01", "MONSANTO CO NEW", "equity_us", "pct_sector_unmapped"),
    (None, "ACME CORP", "other", "pct_sector_unmapped"),
]


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    return read_config(str(ROOT / "configs")).data_aggregate.manager_profile


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "permaticker": [1, 2, 3, 4],
            "ticker": ["LISTED", "ASDEBT", "FUNDX", "PREFX"],
            "category": ["Domestic Common Stock", "ADR Common Stock", "ETF", "Domestic Preferred"],
            "sector": ["Technology", "Energy", "", "Financial Services"],
            "lastpricedate": pd.to_datetime(["2024-06-01"] * 4),
        }
    )
    return sharadar_reference(frame)


def _lines(rows: list[tuple[str | None, str, str | None]], kind: str = "common") -> pd.DataFrame:
    """One fixture line per `(title_of_class, issuer_name, ticker)`."""
    return pd.DataFrame(
        {
            "cik": "A",
            "kind": kind,
            "key": [f"K{i}" for i in range(len(rows))],
            "ticker": [ticker for _, _, ticker in rows],
            "issuer_name": [issuer for _, issuer, _ in rows],
            "title_of_class": [title for title, _, _ in rows],
            "value": 1.0,
        }
    )


def test_unlisted_common_lines_follow_the_ordered_name_rules(reference: pd.DataFrame, cfg: DictConfig) -> None:
    """ETF, then real-estate sector, then foreign, then generic US title; every trap string keeps its documented class."""
    lines = _lines([(title, issuer, None) for title, issuer, _, _ in UNLISTED])

    out = classify_securities(lines, reference, cfg)

    got = list(zip(out["asset"], out["sector_col"], strict=True))
    wrong = {(t, i): (want, have) for (t, i, *want), have in zip(UNLISTED, got, strict=True) if tuple(want) != have}
    print(f"{len(UNLISTED) - len(wrong)}/{len(UNLISTED)} known-truth strings classified as expected; wrong={wrong}")
    assert not wrong


def test_etf_rule_runs_before_the_real_estate_rule(reference: pd.DataFrame, cfg: DictConfig) -> None:
    """A REIT ETF is a fund line without a real estate sector; the same words on an operating REIT give the sector."""
    out = classify_securities(_lines([("COM", "ISHARES DJ US REAL ESTATE ETF", None), ("COM", "WEINGARTEN RLTY INVS", None)]), reference, cfg)

    assert out["asset"].tolist() == ["etf", "equity_us"]
    assert out["sector_col"].tolist() == ["pct_sector_unmapped", "pct_sector_real_estate"]
    print("SANITY: the ETF rule wins over the real-estate rule; an operating REIT keeps its sector.")


def test_a_sharadar_listed_line_keeps_its_class_whatever_the_name(reference: pd.DataFrame, cfg: DictConfig) -> None:
    """The name rules apply only to lines Sharadar lacks: a listed ticker with fund words stays equity_us and keeps its sector."""
    out = classify_securities(_lines([("SPONSORED ADR", "ISHARES TR", "LISTED"), ("COM", "ACME", "ASDEBT")]), reference, cfg)

    assert out["asset"].tolist() == ["equity_us", "equity_foreign"]
    assert out["sector_col"].tolist() == ["pct_sector_technology", "pct_sector_energy"]
    print("SANITY: listed tickers keep their Sharadar class and sector whatever their issuer name says.")


def test_a_listed_line_of_a_non_stock_category_is_read_by_the_name_rules(reference: pd.DataFrame, cfg: DictConfig) -> None:
    """A Sharadar ETF or preferred category is not a stock class: its name decides the asset and the real-estate sector."""
    rows: list[tuple[str | None, str, str | None]] = [
        ("MSCI EMERG MKT", "ISHARES TR", "FUNDX"),
        ("COM", "WEINGARTEN RLTY INVS", "FUNDX"),
        ("TRUST", "BP PRUDHOE BAY ROYALTY TRUST", "FUNDX"),
        ("COM", "ISHARES TR", "PREFX"),
    ]
    out = classify_securities(_lines(rows), reference, cfg)

    assert out["asset"].tolist() == ["etf", "equity_us", "other", "equity_us"]
    assert out["sector_col"].tolist() == ["pct_sector_unmapped", "pct_sector_real_estate", "pct_sector_unmapped", "pct_sector_financial_services"]
    print(
        "SANITY: a listed ticker whose Sharadar category is not Domestic, ADR or Canadian falls through to the title and issuer rules; Domestic stays authoritative."
    )


def test_debt_and_option_rows_are_never_name_classified(reference: pd.DataFrame, cfg: DictConfig) -> None:
    """The name rules read common lines only: a debt line titled like an ETF stays a bond."""
    debt = classify_securities(_lines([("ETF", "ISHARES TR", None)], kind="debt"), reference, cfg)
    call = classify_securities(_lines([("COM", "ACME", None)], kind="call"), reference, cfg)

    assert debt["asset"].tolist() == ["bond"] and call["asset"].tolist() == ["options"]
    print("SANITY: bond and option rows are untouched by the title and issuer rules")
