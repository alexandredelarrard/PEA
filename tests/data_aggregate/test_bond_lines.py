"""Share lines that are bond principal: the shared mask, the profile's bond path and the cube cleaning's drop.

Known truth: the mask on hand-built positives (real convertible-note CUSIPs, titles and issuers from the 13F tables) and on the traps the
measurement found (SEDOL-like codes, iShares iBond ETFs, placeholders, ADRs, a filer titling stock lines BOND, preferreds and warrants).
Real data: Lord Abbett 2024-06-30 filed Altair and Veeco convertible principal as shares; the quarter's whole-filer cleaning used to
re-value those lines as stock (cleaned / filed 1.93, equity hhi 0.128).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.data_aggregate.utils.manager_profile.lines import clean_book
from src.data_aggregate.utils.manager_profile.quarter_stats import quarter_stats
from src.data_aggregate.utils.manager_profile.reference import classify_securities
from src.data_aggregate.utils.sec_thirteen.bond_lines import bond_line_mask
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_filings
from src.data_store.schema import Tables
from src.utils.config import read_config
from tests.data_aggregate.test_manager_profile import NO_PREFIX, _frame, _raw, live_prefixes, live_reference, live_store, load_raw

ROOT = Path(__file__).resolve().parents[2]
EMPTY = "_empty"
LORD_ABBETT, PERIOD = "0000728100", "2024-06-30"


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    return read_config(str(ROOT / "configs")).data_aggregate.manager_profile


@pytest.fixture(scope="module")
def bond_cfg() -> DictConfig:
    return read_config(str(ROOT / "configs")).data_aggregate.bond_lines


def _without_bond_lines(cfg: DictConfig) -> DictConfig:
    """`cfg` with a bond mapping that matches nothing: the behaviour before PL-19."""
    return cast(DictConfig, OmegaConf.merge(cfg, {"bond_lines": {"cusip_pattern": r"[^\s\S]", "title_markers": [r"[^\s\S]"]}}))


def _lines(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Positions `(cusip, issuer_name, title_of_class, ticker)`; `_empty` is no ticker."""
    return pd.DataFrame(rows, columns=["cusip", "issuer_name", "title_of_class", "ticker"])


BONDS = [
    ("021369AC7", "Altair Engineering Inc", "CV", EMPTY),  # Lord Abbett 2024Q2
    ("922417AJ9", "Veeco Instruments Inc", "CV", EMPTY),
    ("45826HAB5", "INTEGER HLDGS CORP", "NOTE 2.125% 2/1", EMPTY),
    ("516544AB9", "LANTHEUS HLDGS INC", "NOTE 2.625%12/1", "LNTH"),  # a ticker does not save a letter-issue CUSIP
    ("05592Q7H2", "BNP PARIBAS MEDIUM0  24F", "DUE 10/07/24", EMPTY),
    ("92343EAD4", "VERISIGN INC", "SDCV 3.250% 8/1", EMPTY),
    ("375558100", "GILEAD SCIENCES INC", "BOND", EMPTY),  # digit issue number, no ticker, bond title
]
EQUITIES = [
    ("037833100", "APPLE INC", "COM", "AAPL"),
    ("594918104", "MICROSOFT CORP", "COM", EMPTY),  # digit issue number, no bond word
    ("00B63DG21", "SOME HOLDINGS PLC", "ORD", EMPTY),  # SEDOL-like code
    ("46434VBG4", "ISHARES TR", "IBONDS DEC 2026 TERM CORP ETF", "IBDQ"),  # iBond ETF with a letter issue number
    ("****SOLE", "****SOLE", "COM", EMPTY),  # placeholder garbage
    ("000000AB1", "PLACEHOLDER CO", "COM", EMPTY),  # placeholder issuer code
    ("438128308", "HONDA MOTOR LTD", "ADR ECH CNV IN 3", "HMC"),  # an ADR whose title says CNV
    ("62855J104", "MYLAN NV", "BOND", "MYL"),  # filer 0001166588 titled stock lines BOND
    ("0000LRLCY", "SOME TRUST", "UNITS", EMPTY),
    ("018581108", "ALLIANCE DATA SYSTEMS CORP", "PFD 7% CONV", EMPTY),  # preferred
    ("90184L102", "TWITTER INC", "WT EXP 12/31/25 CONV", EMPTY),  # warrant
]


def test_the_mask_flags_bond_principal_and_spares_the_documented_traps(bond_cfg: DictConfig) -> None:
    flagged = bond_line_mask(_lines([*BONDS, *EQUITIES]), bond_cfg)

    assert flagged.iloc[: len(BONDS)].all(), _lines(BONDS)[~flagged.iloc[: len(BONDS)]]
    assert not flagged.iloc[len(BONDS) :].any(), _lines(EQUITIES)[flagged.iloc[len(BONDS) :].to_numpy()]
    print("\n=== SANITY CHECK: bond_line_mask ===")
    print(f"{len(BONDS)} bond lines flagged (convertible CUSIPs, note titles, a bond title with no ticker); {len(EQUITIES)} equity traps spared.")


def test_the_title_rule_needs_no_ticker_but_the_cusip_rule_does_not(bond_cfg: DictConfig) -> None:
    stock_titled_bond = ("62855J104", "MYLAN NV", "BOND")

    without = bond_line_mask(_lines([(*stock_titled_bond, EMPTY)]), bond_cfg)
    listed = bond_line_mask(_lines([(*stock_titled_bond, "MYL")]), bond_cfg)
    letter_cusip = bond_line_mask(_lines([("021369AC7", "ALTAIR ENGINEERING INC", "COM", "ALTR")]), bond_cfg)

    assert without.all() and not listed.any() and letter_cusip.all()
    print("\n=== SANITY CHECK: ticker guard ===")
    print("A BOND title flags a tickerless line only; a letter issue number on a numeric-issuer CUSIP flags it listed or not.")


def test_a_missing_text_column_reads_empty(bond_cfg: DictConfig) -> None:
    frame = _lines(BONDS[:2]).drop(columns=["issuer_name", "title_of_class"])

    assert bond_line_mask(frame, bond_cfg).all(), "the CUSIP shape alone flags a convertible-note line"
    print("SANITY CHECK: missing text columns")
    print("With no issuer or title column the text reads empty and the CUSIP shape alone still flags both convertible-note lines.")


def _book_with_a_priced_bond(cik: str, scale: float, bond: tuple[float, float]) -> pd.DataFrame:
    """Six equity lines at prices 10..60 (value in `scale` units) plus one convertible-note line `(shares, value)`."""
    lines: list[tuple[str, str, str, str, float, float, dict[str, Any]]] = [
        (f"S{i}", f"CS{i}", f"S{i} INC", "common", 100.0, 10.0 * i * 100.0 * scale, {}) for i in range(1, 7)
    ]
    lines.append((EMPTY, "021369AC7", "ALTAIR ENGINEERING INC", "common", bond[0], bond[1], {}))
    return _raw(cik, "2024-03-31", f"B{cik}", lines)


def test_the_profile_weights_a_bond_line_by_filed_value_and_never_revalues_it(cfg: DictConfig) -> None:
    # Three other filers hold the note at a price of 1e-4 per unit: as stock, filer 1's 16.4m units would be re-valued to about 1.6k.
    books = [_book_with_a_priced_bond("0000000031", 1.0, (16_389_000.0, 23_876_000.0))]
    books += [_book_with_a_priced_bond(f"000000003{n}", 1.0, (1_000_000.0, 100.0)) for n in (2, 3, 4)]
    raw = pd.concat(books)

    off = _without_bond_lines(cfg)
    before = clean_book(raw, off, NO_PREFIX).set_index(["cik", "kind", "key"])
    after = clean_book(raw, cfg, NO_PREFIX).set_index(["cik", "kind", "key"])

    note = ("0000000031", "debt", "021369AC7")
    assert ("0000000031", "common", "021369AC7") in before.index and note not in before.index, "before: the note is a stock line"
    assert note in after.index and ("0000000031", "common", "021369AC7") not in after.index, "after: the note is a debt line"
    assert after.loc[note, "value"] == 23_876_000.0, "the filed value, not shares x consensus price"
    equity = after.xs("common", level="kind")
    assert equity.groupby("cik").size().eq(6).all(), "the equity lines are unchanged"
    print("\n=== SANITY CHECK: profile bond path ===")
    print(
        f"Before: note a common line worth {before.loc[('0000000031', 'common', '021369AC7'), 'value']:,.0f}; "
        f"after: debt line {after.loc[note, 'value']:,.0f} (filed), equity lines untouched."
    )


def test_a_bond_line_takes_only_its_books_unit_factor(cfg: DictConfig) -> None:
    # Filer 1 reports in thousands (x1000 too small); its note value is filed in the same unit.
    books = [_book_with_a_priced_bond(f"000000004{n}", 1.0, (1_000_000.0, 950_000.0)) for n in (2, 3, 4)]
    books.append(_book_with_a_priced_bond("0000000041", 1e-3, (1_000_000.0, 950.0)))

    cleaned = clean_book(pd.concat(books), cfg, NO_PREFIX)

    note = cleaned[cleaned["kind"].eq("debt")].set_index("cik")["value"]
    assert np.allclose(note.to_numpy(), 950_000.0), note.to_dict()
    print("\n=== SANITY CHECK: bond unit factor ===")
    print(f"Note values {note.to_dict()}: the thousands filer's 950 is rescaled by its book factor to 950,000.")


def _cube_packet(bond_cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One accession with a debt row, a flagged share line and two equity lines, run through the cube's `clean_filings`."""
    base = {"cik": "0000000001", "period": "2024-03-31", "filing_date": "2024-05-10", "accession_number": "A1"}
    filings = pd.DataFrame([{**base, "parsed_row_count": 4, "parse_status": "complete"}])
    rows = [
        ("037833100", "APPLE INC", "COM", "AAPL", "common", 10.0, 1_000.0, 0.0),
        ("594918104", "MICROSOFT CORP", "COM", EMPTY, "common", 5.0, 500.0, 0.0),
        ("021369AC7", "ALTAIR ENGINEERING INC", "CV", EMPTY, "common", 16_389_000.0, 23_876_000.0, 0.0),
        ("D1", "A BOND", "NOTE", EMPTY, "debt", 0.0, 0.0, 700.0),
    ]
    holdings = pd.DataFrame(
        [
            {**base, "cusip": c, "issuer_name": n, "title_of_class": t, "ticker": k, "position_type": p, "shares": s, "value_usd": v, "debt_value": d}
            for c, n, t, k, p, s, v, d in rows
        ]
    )
    cleaned_filings, cleaned = clean_filings(filings.copy(), holdings.copy(), bond_cfg)
    return filings, holdings, cleaned_filings, cleaned


def test_the_cube_cleaning_drops_bond_share_lines_like_debt_rows(bond_cfg: DictConfig) -> None:
    filings, holdings, cleaned_filings, cleaned = _cube_packet(bond_cfg)

    assert cleaned["cusip"].tolist() == ["037833100", "594918104"], "the debt row and the note share line leave the book"
    assert cleaned_filings["parsed_row_count"].tolist() == [2], "the stored row count drops with them"
    assert {"issuer_name", "title_of_class"}.isdisjoint(cleaned.columns), "the mask's text columns are dropped"
    assert (cleaned["debt_value"] == 0.0).all()
    print("\n=== SANITY CHECK: cube clean_filings ===")
    print(
        f"{len(holdings)} rows in, {len(cleaned)} out; manifest row count {filings['parsed_row_count'].iloc[0]} -> {cleaned_filings['parsed_row_count'].iloc[0]}."
    )


def _lord_abbett_quarter(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """The whole quarter's lines (the repair's consensus needs every filer), the stored reference and the CUSIP prefix map."""
    store = live_store()
    ciks = _frame(store.load(Tables.sec_13f_manager_details, columns=["cik"], where={"period": PERIOD}))["cik"].tolist()
    raw = pd.concat([load_raw(store, ciks[i : i + 1000], [PERIOD])[PERIOD] for i in range(0, len(ciks), 1000)], ignore_index=True)
    return raw, live_reference(store), live_prefixes(store)


def test_lord_abbett_2024q2_convertible_principal_is_no_longer_stock(cfg: DictConfig) -> None:
    raw, reference, prefixes = _lord_abbett_quarter(cfg)
    off = _without_bond_lines(cfg)
    mine = raw[raw["cik"].eq(LORD_ABBETT)]
    original = mine[mine["form"].eq("13F-HR")]["accession_number"].iloc[0]
    positions = mine[mine["accession_number"].eq(original) & mine["cusip"].ne("_empty")]
    filed = (
        positions["value_usd"].where(positions["position_type"].fillna("common").str.lower().eq("common"), 0.0).sum()
        + positions["debt_value"].fillna(0.0).sum()
    )

    result = {}
    for name, rules in (("before", off), ("after", cfg)):
        lines = classify_securities(clean_book(raw, rules, prefixes), reference, rules)
        own = lines[lines["cik"].eq(LORD_ABBETT)]
        stats = quarter_stats(own, None, None, rules).iloc[0]
        result[name] = {"ratio": own["value"].sum() / filed, "hhi": stats["hhi"], "pct_bond": stats["pct_bond"]}

    assert result["before"]["ratio"] > 1.5 and result["before"]["hhi"] > 0.1, result
    assert abs(result["after"]["ratio"] - 1.0) < 0.02 and result["after"]["hhi"] < 0.02, result
    assert result["after"]["pct_bond"] > 10 * result["before"]["pct_bond"], result
    print("\n=== SANITY CHECK: Lord Abbett 2024-06-30 ===")
    print(pd.DataFrame(result).round(4).to_string())
    print("The two convertible-note lines keep their filed value: the book is back at its filed total and the equity hhi at its true level.")
