"""Tests for the Sharadar phase-3 field map and TTM build
(src/data_extract/utils/fundamentals_sharadar/field_map.py, build_ttm.py).

Split the way the repo's testing rule requires: **parsing math gets synthetic known-truth
fixtures** (the four-quarter window, the NaN propagation, the register's refusals), and **the
basis decisions get real data from Postgres** -- whether `netIncome` is `consolinc` or `netinc`
is not a question a fixture can answer, because the fixture would encode the answer.

Every test prints its conclusion. Several exist specifically to pin a decision that a later
reader will be tempted to "fix": that `ebitda` is top-down, that `debtToEquity` is not the
vendor's `de`, and that the share block is de-adjusted rather than taken as delivered.

!! Nothing here touches `src/validate/` or the `fundamentals_check*` tables (D25), and nothing
writes a table -- the whole phase is a pure transform.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import (
    SHARADAR_CONFIG_SUBDIR, SHARADAR_CORRECTIONS_FILENAME, SHARADAR_SF1_COLUMNS,
    SHARADAR_ZERO_RULES_FILENAME,
)
from src.data_extract.utils.fundamentals.kpi_catalogue import HISTORY_STATEMENT_ORDER
from src.data_extract.utils.fundamentals_sharadar.build_ttm import TTM_SPAN_DAYS, build_ttm
from src.data_extract.utils.fundamentals_sharadar.field_map import (
    DURATION, INSTANT, MEAN, TranslationReport, apply_derived, load_corrections,
    load_field_map, load_zero_rules, split_events, translate,
)
from src.data_store.schema import Tables

CONFIG_DIR = Path("./configs")

#: `|a - b| / max(|a|, |b|)` under which two money figures are the same number. Sharadar
#: rounds to four significant figures, so an exact test against the SEC layer's as-filed value
#: fails on rounding alone -- NVDA's 25,020,000,000 de-adjusts to 2,502,000,000 against a true
#: 2,500,000,000, which is 0.08% and is not a de-adjustment error.
VENDOR_ROUNDING = 0.005


def ttm_coverage(result: pd.DataFrame, field_map) -> pd.DataFrame:
    """Per-column non-null coverage of a built frame. A TEST instrument, not a production one.

    Duration columns are structurally NULL for a ticker's first three quarters -- that is the
    four-discrete-quarter contract, not a defect -- so the count is reported beside the number
    of rows that HAD a whole window, never as a bare percentage of all rows.
    """
    rows = len(result)
    records = [{"column": name, "kind": spec.kind, "basis": spec.basis,
                "n_non_null": int(result[name].notna().sum()), "n_rows": rows,
                "pct_non_null": round(float(result[name].notna().mean()), 4)}
               for name, spec in field_map.outputs.items() if name in result.columns]
    return pd.DataFrame(records).sort_values(["kind", "column"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def field_map():
    return load_field_map(str(CONFIG_DIR))


@pytest.fixture(scope="module")
def context():
    """A real Context (DB + .env), skipping rather than erroring when either is missing."""
    from src.context import get_config_context
    try:
        _, ctx = get_config_context(str(CONFIG_DIR), use_cache=False, save=False)
        with ctx.store.engine.connect():
            pass
    except Exception as exc:                                            # noqa: BLE001
        pytest.skip(f"context/database unavailable ({type(exc).__name__}: {exc})")
    if ctx.store.row_count(Tables.sharadar_fundamentals) == 0:
        pytest.skip(f"{Tables.sharadar_fundamentals} is empty -- run fundamentals-sharadar")
    return ctx


@pytest.fixture(scope="module")
def vendor_arq(context):
    """Every stored ARQ row, as delivered."""
    frame = context.store.load(Tables.sharadar_fundamentals, project=True)
    return frame[frame["dimension"] == "ARQ"].copy()


@pytest.fixture(scope="module")
def actions(context):
    return context.store.load(Tables.sharadar_actions, project=True)


@pytest.fixture(scope="module")
def translated(vendor_arq, field_map):
    """The repo-named DISCRETE-quarter frame, plus the report of what was removed.

    ⚠ Still SPLIT-ADJUSTED. The de-adjustment moved into `build_ttm`, so the share block here
    is on Sharadar's retroactive basis -- assert on the `ttm` fixture, not this one.
    """
    report = TranslationReport()
    frame = translate(vendor_arq, field_map, report=report)
    return frame, report


@pytest.fixture(scope="module")
def ttm(translated, actions, field_map):
    """The TTM frame: de-adjusted, then derived."""
    frame, report = translated
    return build_ttm(frame, field_map, actions=actions, report=report)


def synthetic_vendor(rows: list[dict]) -> pd.DataFrame:
    """A full 112-column vendor frame with only the named cells populated.

    Full rather than minimal on purpose: `translate` refuses a frame missing a mapped column,
    because `fields=` silently drops an unavailable field and a short frame is a projection
    bug rather than an empty column. A fixture that dodged that check would test a code path
    production never takes.
    """
    frame = pd.DataFrame(rows)
    absent = [c for c in SHARADAR_SF1_COLUMNS if c not in frame.columns]
    frame = pd.concat([frame, pd.DataFrame(np.nan, index=frame.index, columns=absent)],
                      axis=1)
    frame["dimension"] = "ARQ"
    return frame[list(SHARADAR_SF1_COLUMNS)]


def four_quarters(ticker: str, values: dict[str, list[float]], *, n: int = 4,
                  start: str = "2023-03-31") -> pd.DataFrame:
    """`n` consecutive quarters for one ticker, with the given per-quarter values."""
    ends = pd.date_range(start, periods=n, freq="QE")
    rows = []
    for index, end in enumerate(ends):
        row = {"ticker": ticker, "calendardate": end, "reportperiod": end,
               "date": end + pd.Timedelta(days=30), "fiscalperiod": f"{end.year}-Q{index + 1}"}
        row.update({name: series[index] for name, series in values.items()})
        rows.append(row)
    return synthetic_vendor(rows)


# --------------------------------------------------------------------------- #
# the contract                                                                 #
# --------------------------------------------------------------------------- #
def test_every_history_column_is_mapped(field_map):
    """All 60 `HISTORY_STATEMENT_ORDER` names resolve to direct / derived / sec / null.

    The loader already refuses a gap; this asserts the resolved kinds and prints the census,
    because the failure mode is not a crash -- the merged table's contract asserts by LIST
    EQUALITY, so an unfilled column would pass it while carrying nothing.
    """
    unmapped = [n for n in HISTORY_STATEMENT_ORDER if n not in field_map.columns]
    kinds = pd.Series([field_map.columns[n].kind for n in HISTORY_STATEMENT_ORDER
                       if n in field_map.columns]).value_counts().to_dict()
    print(f"\ncontract columns : {len(HISTORY_STATEMENT_ORDER)}")
    print(f"unmapped         : {unmapped or 'none'}")
    print(f"by kind          : {kinds}")
    print(f"SEC-owned (D18)  : {len(field_map.sec_owned)} -> {field_map.sec_owned}")
    print(f"permanently NULL : "
          f"{sorted(n for n, s in field_map.columns.items() if s.kind == 'null')}")
    assert not unmapped
    assert len(field_map.sec_owned) == 15, "D18 declares exactly 15 SEC-owned columns"


def test_every_sf1_column_is_accounted_for(field_map):
    """Mapped, extra, excluded or identifier -- but never silently dropped.

    112 columns is the widest table in the schema and D7 keeps all of them in the raw table.
    A column nobody classified is a column nobody decided about.
    """
    from src.constants.constants import SHARADAR_ID_COLUMNS
    # `.source` on BOTH halves. An extra is keyed by the repo name it is EMITTED under, so
    # `set(field_map.extras)` would be a set of camelCase names SF1 has never heard of --
    # every extra would read as unaccounted and every vendor column it covers as phantom.
    accounted = ({s.source for s in field_map.direct.values()}
                 | {s.source for s in field_map.extras.values()}
                 | field_map.excluded | set(SHARADAR_ID_COLUMNS))
    unaccounted = sorted(set(SHARADAR_SF1_COLUMNS) - accounted)
    phantom = sorted(accounted - set(SHARADAR_SF1_COLUMNS))
    print(f"\nSF1 columns   : {len(SHARADAR_SF1_COLUMNS)}")
    print(f"unaccounted   : {unaccounted or 'none'}")
    print(f"named but not delivered by SF1: {phantom or 'none'}")
    assert not unaccounted and not phantom


def test_registers_refuse_an_unapproved_file(tmp_path):
    """Both registers are refused without an `_APPROVED` block.

    The check IS the governance model: a regenerated proposal is byte-identical to a reviewed
    decision, so without it "human-approved" is a sentence in a docstring.
    """
    for filename, loader in ((SHARADAR_ZERO_RULES_FILENAME, load_zero_rules),
                             (SHARADAR_CORRECTIONS_FILENAME, load_corrections)):
        source = CONFIG_DIR / SHARADAR_CONFIG_SUBDIR / filename
        raw = json.loads(source.read_text(encoding="utf-8"))
        raw.pop("_APPROVED")
        target = tmp_path / SHARADAR_CONFIG_SUBDIR / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(RuntimeError, match="_APPROVED"):
            loader(str(tmp_path))
        print(f"\n{filename}: stripped of `_APPROVED` -> REFUSED, as designed")


def test_every_correction_states_its_evidence(field_map):
    """A correction with no stated evidence is indistinguishable from a guess."""
    print()
    for name, by_ticker in field_map.corrections.items():
        for ticker, entry in by_ticker.items():
            print(f"{name}/{ticker:5s} {entry['action']:17s} "
                  f"evidence: {entry['evidence'][:90]}...")
            assert entry["evidence"].strip()
    assert field_map.corrections, "the register is not empty -- phase 2 found three defects"


# --------------------------------------------------------------------------- #
# the basis decisions, on REAL data                                            #
# --------------------------------------------------------------------------- #
def test_capex_sign_is_flipped(vendor_arq, translated):
    """Sharadar's negative `capex` becomes the repo's non-negative one, and the exceptions
    are NULLed rather than flipped into a negative.

    The plan's original `negate: true` is the bug this pins: 13 of 1,346 stored rows carry a
    POSITIVE `capex`, and flipping those writes a negative into a column the SEC catalogue
    declares `non_negative`.
    """
    frame, report = translated
    merged = frame[["ticker", "date", "capex"]].merge(
        vendor_arq[["ticker", "date", "capex", "ncfo", "fcf"]].rename(
            columns={"capex": "capex_vendor"}), on=["ticker", "date"])
    aapl = merged[(merged["ticker"] == "AAPL") & merged["capex"].notna()].iloc[0]
    print(f"\nAAPL {aapl['date']}: vendor capex {aapl['capex_vendor']:,.0f} -> "
          f"repo capex {aapl['capex']:,.0f}")
    assert aapl["capex_vendor"] < 0 < aapl["capex"]
    assert aapl["capex"] == -aapl["capex_vendor"]

    identity = aapl["fcf"] - (aapl["ncfo"] - aapl["capex"])
    print(f"AAPL freeCashflow identity: fcf {aapl['fcf']:,.0f} == ncfo {aapl['ncfo']:,.0f} "
          f"- capex {aapl['capex']:,.0f} ? residual {identity:,.2f}")
    assert abs(identity) <= 1.0

    positive = merged[merged["capex_vendor"] > 0]
    print(f"vendor rows with a POSITIVE capex: {len(positive)} "
          f"({sorted(positive['ticker'].unique())}) -> all NULL in the repo frame")
    print(f"sign-guard NULLs reported: {report.negation_nulled}")
    assert positive["capex"].isna().all()
    assert (frame["capex"].dropna() >= 0).all(), "the repo column is non_negative"


def test_netincome_is_consolinc_not_netinc(vendor_arq, translated):
    """`netIncome` is `consolinc` (incl. NCI), never `netinc`. Measured on JPM."""
    frame, _ = translated
    jpm = frame[frame["ticker"] == "JPM"][["ticker", "date", "netIncome"]].merge(
        vendor_arq[["ticker", "date", "consolinc", "netinc"]], on=["ticker", "date"])
    differs = int((jpm["consolinc"] != jpm["netinc"]).sum())
    print(f"\nJPM dates: {len(jpm)} | netIncome == consolinc on "
          f"{int((jpm['netIncome'] == jpm['consolinc']).sum())} "
          f"| consolinc differs from netinc on {differs}")
    print(jpm.head(6).to_string(index=False))
    assert (jpm["netIncome"] == jpm["consolinc"]).all()
    assert differs > 0, "if the two agreed everywhere the decision would be untestable"


def test_ebitda_is_top_down(ttm):
    """`ebitda` is `operatingIncome + depAmort`, and measurably NOT the bottom-up construction.

    Sharadar's own column is bottom-up (`netinc + taxexp + intexp + depamor`), so it carries
    every non-operating item. The comparison is built HERE, from the same TTM frame, rather
    than against the vendor's `ebitda` column: that column is a QUARTER on the ARQ dimension,
    so comparing it to a trailing twelve measures the period difference (~4x) and hides the
    definitional one this test exists to pin.
    """
    frame = ttm[["ticker", "date", "ebitda", "operatingIncome", "depAmort", "netIncome",
                 "incomeTaxExpense", "interestExpense"]].copy()
    computed = frame.dropna(subset=["ebitda", "operatingIncome", "depAmort"])
    residual = (computed["ebitda"]
                - (computed["operatingIncome"] + computed["depAmort"])).abs().max()
    print(f"\nebitda == operatingIncome + depAmort on {len(computed)} TTM rows; "
          f"max residual {residual:,.2f}")
    assert residual <= 1.0

    gap = frame.dropna().copy()
    gap["bottom_up"] = (gap["netIncome"] + gap["incomeTaxExpense"] + gap["interestExpense"]
                        + gap["depAmort"])
    gap["pct"] = ((gap["ebitda"] - gap["bottom_up"]).abs()
                  / gap["bottom_up"].abs().replace(0, np.nan))
    differing = gap[gap["pct"] > 0.01]
    print(f"top-down vs bottom-up on the SAME trailing twelve, {len(gap)} comparable rows:")
    print(f"  differ by >1% on {len(differing)} rows over {differing['ticker'].nunique()} "
          f"ticker(s); median {gap['pct'].median():.2%}, p90 {gap['pct'].quantile(0.9):.2%}, "
          f"worst {gap['pct'].max():.2%}")
    print(gap.nlargest(3, "pct")[["ticker", "date", "ebitda", "bottom_up", "pct"]]
          .to_string(index=False))
    assert not differing.empty, "top-down and bottom-up must not agree everywhere"
    assert len(differing) / len(gap) > 0.5, "the two bases differ on most rows, not a handful"


def test_debt_to_equity_is_not_vendor_de(context, ttm):
    """The recomputed `debtToEquity` differs from Sharadar's `de`.

    Printed in full because this is the trap most likely to be "helpfully" reverted later:
    `de` is named "Debt to Equity Ratio" and is LIABILITIES/equity, so using it would be wrong
    by the whole non-debt liability stack.

    `de` has to be requested EXPLICITLY: D21 keeps the 35 vendor ratios out of the table's
    `read_columns`, so the default projection cannot even see it. That is the design working,
    and it is why this test loads its own columns instead of reusing `vendor_arq`.
    """
    frame = ttm[["ticker", "date", "debtToEquity", "totalDebt", "stockholdersEquity"]].dropna()
    vendor = context.store.load(
        Tables.sharadar_fundamentals, project=False,
        columns=["ticker", "dimension", "date", "de", "liabilities", "equity"])
    vendor = vendor[vendor["dimension"] == "ARQ"].drop(columns="dimension")
    gap = frame.merge(vendor, on=["ticker", "date"])
    gap = gap[gap["de"].notna() & (gap["de"] != 0)]
    gap["ratio"] = gap["debtToEquity"] / gap["de"]
    print(f"\nrows compared: {len(gap)} | median debtToEquity/de = {gap['ratio'].median():.3f}")
    print(gap[["ticker", "date", "debtToEquity", "de", "totalDebt", "liabilities",
               "equity"]].head(5).to_string(index=False))
    vendor_is_liabilities = ((gap["de"] - gap["liabilities"] / gap["equity"]).abs()
                             / gap["de"].abs() < 0.01)
    print(f"`de` reproduces liabilities/equity on {int(vendor_is_liabilities.sum())} of "
          f"{len(gap)} rows -- it is NOT a debt ratio")
    assert (gap["debtToEquity"] != gap["de"]).mean() > 0.9


def test_epsdiluted_is_derived_not_vendor_epsdil(vendor_arq, ttm):
    """⚠ DEVIATION FROM THE PLAN, and the measurement behind it.

    The plan mapped `epsDiluted` DIRECT from `epsdil`. `epsdil` is on the `netinccmn` basis
    (after preferred dividends) while the repo's `netIncome` is `consolinc`, so a direct map
    would put two different net-income bases in the same row. Derived instead, exactly as the
    SEC path's `_FORMULAS` do.
    """
    vendor = vendor_arq[vendor_arq["shareswadil"] > 0].copy()
    for name, numerator in (("netinccmn", "netinccmn"), ("netinc", "netinc"),
                            ("consolinc", "consolinc")):
        vendor[f"matches_{name}"] = (
            (vendor["epsdil"] - vendor[numerator] / vendor["shareswadil"]).abs() <= 0.005)
    wedge = int((vendor["netinccmn"] != vendor["consolinc"]).sum())
    print(f"\nvendor rows with a diluted count: {len(vendor)}")
    for name in ("netinccmn", "netinc", "consolinc"):
        print(f"  epsdil == {name}/shareswadil on "
              f"{int(vendor[f'matches_{name}'].sum())} rows")
    print(f"  netinccmn != consolinc on {wedge} rows "
          f"({wedge / len(vendor):.1%}) -- the basis wedge the direct map would import")
    assert (vendor["matches_netinccmn"].sum() > vendor["matches_consolinc"].sum()), \
        "epsdil sits closer to netinccmn than to the repo's consolinc basis"

    computed = ttm[["epsDiluted", "netIncome", "dilutedShares"]].dropna()
    residual = (computed["epsDiluted"]
                - computed["netIncome"] / computed["dilutedShares"]).abs().max()
    print(f"repo epsDiluted == netIncome / dilutedShares on {len(computed)} TTM rows; "
          f"max residual {residual:.6f}")
    assert residual <= 1e-6


# --------------------------------------------------------------------------- #
# the split de-adjustment                                                      #
# --------------------------------------------------------------------------- #
def test_share_block_is_deadjusted_against_the_sec_cover_page(context, ttm, translated):
    """NVDA's 2021-2024 rows carry ~25bn shares against the ~2.5bn then outstanding.

    The pin the plan asked for. Checked against `fundamentals_history_sec.sharesOutstanding`,
    which this repo built from the cover-page `dei:EntityCommonStockSharesOutstanding` -- an
    as-filed number, so it is the right authority for "was this point-in-time?".
    """
    # The TTM frame, because that is where the de-adjustment now happens. `sharesOutstanding`
    # is an INSTANT, so its value is identical either way (verified 59/59) -- only the stage
    # it is asserted at moved.
    frame, report = ttm, translated[1]
    sec = context.store.load(
        Tables.fundamentals_history_sec, project=False,
        columns=["ticker", "as_of", "sharesOutstanding"],
        where={"ticker": ["NVDA", "WMT"]})
    sec["as_of"] = pd.to_datetime(sec["as_of"])
    mine = frame[frame["ticker"].isin(["NVDA", "WMT"])][
        ["ticker", "date", "sharesOutstanding"]].copy()
    mine["date"] = pd.to_datetime(mine["date"])
    joined = mine.merge(sec, left_on=["ticker", "date"], right_on=["ticker", "as_of"],
                        suffixes=("_repo", "_sec")).dropna()
    joined["ratio"] = joined["sharesOutstanding_repo"] / joined["sharesOutstanding_sec"]
    print(f"\nsplits applied : {report.splits_applied}")
    print(f"splits rejected: {report.splits_rejected}")
    print(f"cells de-adjusted: {report.split_deadjusted}")
    print(joined.head(8).to_string(index=False))
    print(f"ratio to the SEC cover page over {len(joined)} row(s): "
          f"min {joined['ratio'].min():.5f}, max {joined['ratio'].max():.5f}")
    assert not joined.empty
    assert ((joined["ratio"] - 1.0).abs() <= VENDOR_ROUNDING).all()


def test_a_spinoff_priced_split_row_is_rejected(actions):
    """⚠ HON's `split` = 0.5 is the Honeywell Aerospace spinoff's PRICE adjustment, not a
    share-count event -- and applying it would DOUBLE every HON share count in the history.

    The discriminator is the co-dated `spinoff` row. HON's own cover page confirms the
    verdict: `sharesbas` is unchanged across the date.
    """
    report = TranslationReport()
    kept = split_events(actions, report=report)
    print(f"\nsplit rows in `sharadar_actions`: "
          f"{int((actions['action'] == 'split').sum())}")
    print(f"accepted: {report.splits_applied}")
    print(f"rejected: {report.splits_rejected}")
    assert "HON" not in set(kept["ticker"]), "the spinoff price factor must not be applied"
    assert any(label.startswith("HON") for label in report.splits_rejected)
    assert set(kept["ticker"]) == {"AMZN", "WMT", "NVDA"}


def test_hon_share_count_is_unchanged_across_its_split_row(vendor_arq):
    """The evidence behind the rejection above, measured rather than asserted."""
    hon = vendor_arq[vendor_arq["ticker"] == "HON"].sort_values("date")
    before = hon[pd.to_datetime(hon["date"]) < pd.Timestamp("2026-06-29")].iloc[-1]
    after = hon[pd.to_datetime(hon["date"]) > pd.Timestamp("2026-06-29")].iloc[0]
    step = after["sharesbas"] / before["sharesbas"]
    print(f"\nHON sharesbas {before['date']}: {before['sharesbas']:,.0f}")
    print(f"HON sharesbas {after['date']}: {after['sharesbas']:,.0f}  (step {step:.4f})")
    print("a 1-for-2 reverse split would halve the as-filed count; it did not move")
    assert abs(step - 1.0) < 0.05


# --------------------------------------------------------------------------- #
# the corrections                                                              #
# --------------------------------------------------------------------------- #
def test_interest_expense_corrections_are_applied(vendor_arq, translated):
    """NKE's whole `intexp` series is NULLed (net basis) and MMM's negative quarter with it.

    Neither is reachable by the zero rule -- NKE has 0 zeros in 20 quarters, and MMM's cell is
    negative. The register is what covers them.
    """
    frame, report = translated
    for ticker in ("NKE", "MMM"):
        vendor = vendor_arq[vendor_arq["ticker"] == ticker]
        mapped = frame[frame["ticker"] == ticker]
        print(f"\n{ticker}: vendor intexp non-null {int(vendor['intexp'].notna().sum())} "
              f"(negative {int((vendor['intexp'] < 0).sum())}, "
              f"zero {int((vendor['intexp'] == 0).sum())}) "
              f"-> repo interestExpense non-null {int(mapped['interestExpense'].notna().sum())}")
    print(f"corrections applied: {report.corrected}")
    assert frame.loc[frame["ticker"] == "NKE", "interestExpense"].isna().all()
    assert (frame.loc[frame["ticker"] == "MMM", "interestExpense"].dropna() >= 0).all()
    assert (frame["interestExpense"].dropna() >= 0).all(), "the column is declared non_negative"


def test_the_ebt_minus_ebit_identity_is_a_tautology(vendor_arq):
    """⚠ Records that the plan's proposed EVIDENCE for the NKE correction proves nothing.

    phase-3-field-map.md proposed `ebt - ebit == -intexp on 20/20 rows` as the evidence that
    NKE is on a net basis. Sharadar DEFINES `ebit = ebt + intexp`, so the identity holds for
    every filer on either basis. Measured here so nobody re-derives a decision from it.
    """
    frame = vendor_arq[["ticker", "ebt", "ebit", "intexp"]].dropna()
    holds = ((frame["ebt"] - frame["ebit"]) + frame["intexp"]).abs() <= 1.0
    print(f"\n`ebt - ebit == -intexp` holds on {int(holds.sum())} of {len(frame)} ARQ rows "
          f"across {frame['ticker'].nunique()} tickers")
    print("=> a TAUTOLOGY (Sharadar defines ebit = ebt + intexp), not evidence of any basis")
    assert holds.all()


# --------------------------------------------------------------------------- #
# the TTM contract -- synthetic known truth                                    #
# --------------------------------------------------------------------------- #
def test_ttm_is_four_discrete_quarters(field_map):
    """A duration field is the sum of four discrete quarters; three quarters is NULL.

    Synthetic, because this is parsing math with a known answer. `revenue` is the flow,
    `assets` the instant, and `shareswa` the weighted-average count that must be AVERAGED --
    summing four quarterly averages would report four times the year's.
    """
    values = {"revenue": [100.0, 200.0, 300.0, 400.0],
              "assets": [10.0, 20.0, 30.0, 40.0],
              "shareswa": [8.0, 8.0, 12.0, 12.0]}
    frame = build_ttm(translate(four_quarters("TEST", values), field_map), field_map)
    last, third = frame.iloc[-1], frame.iloc[-2]
    print(f"\nquarters        : {values['revenue']}")
    print(f"totalRevenue TTM: {last['totalRevenue']} (expected 1000.0)")
    print(f"at the 3rd row  : {third['totalRevenue']} (expected NULL -- only 3 quarters)")
    print(f"totalAssets     : {last['totalAssets']} (expected 40.0, the PERIOD END)")
    print(f"basicShares     : {last['basicShares']} (expected 10.0, the four-quarter MEAN)")
    print(f"revenue_q       : {last['revenue_q']} (expected 400.0, the DISCRETE quarter)")
    assert last["totalRevenue"] == 1000.0
    assert pd.isna(third["totalRevenue"])
    assert last["totalAssets"] == 40.0
    assert last["basicShares"] == 10.0
    assert last["revenue_q"] == 400.0


def test_a_gap_in_the_quarters_refuses_the_window(field_map):
    """Four rows that are not four CONSECUTIVE quarters do not make a trailing twelve.

    A window spliced across a missing quarter is a 15-month number wearing a 12-month label,
    and it is invisible afterwards -- the level simply looks high.
    """
    whole = four_quarters("TEST", {"revenue": [100.0, 200.0, 300.0, 400.0]}, n=4)
    gapped = four_quarters("TEST", {"revenue": [100.0, 200.0, 300.0, 400.0, 500.0]},
                           n=5).drop(index=2).reset_index(drop=True)
    built_whole = build_ttm(translate(whole, field_map), field_map)
    built_gapped = build_ttm(translate(gapped, field_map), field_map)
    print(f"\n4 consecutive quarters -> TTM {built_whole['totalRevenue'].iloc[-1]}")
    print(f"4 rows with Q3 missing -> TTM {built_gapped['totalRevenue'].iloc[-1]} "
          f"(expected NULL, not 1200.0)")
    assert built_whole["totalRevenue"].iloc[-1] == 1000.0
    assert pd.isna(built_gapped["totalRevenue"].iloc[-1])


def test_one_ticker_never_borrows_anothers_quarters(field_map):
    """The window is per TICKER. Two issuers sharing a calendar must not splice.

    The classic rolling-window bug, and it is silent: ticker B's first TTM would be three of
    A's quarters plus one of its own, at a plausible level nobody would query.
    """
    values = {"revenue": [100.0, 100.0, 100.0, 100.0]}
    frame = pd.concat([four_quarters("AAA", values),
                       four_quarters("BBB", {"revenue": [7.0, 7.0, 7.0, 7.0]})],
                      ignore_index=True)
    built = build_ttm(translate(frame, field_map), field_map)
    per_ticker = built.groupby("ticker")["totalRevenue"].agg(["count", "max"])
    print(f"\n{per_ticker.to_string()}")
    print("each ticker yields exactly ONE whole window, at its own level (400.0 / 28.0)")
    assert per_ticker.loc["AAA", "max"] == 400.0
    assert per_ticker.loc["BBB", "max"] == 28.0
    assert (per_ticker["count"] == 1).all()


def test_amended_filing_does_not_break_the_window(field_map):
    """An amendment republishing a quarter must not read as a GAP.

    Sharadar's ARQ grain is one row per FILING, so a 10-Q/A arrives as a second row on the same
    `reportperiod` under a later `date`. Un-deduplicated, that repeat makes
    `ordinal - ordinal.shift(3)` equal 2 rather than 3 for three consecutive rows -- the exact
    signature of a missing quarter -- and nulls three trailing twelves that are in fact whole.
    """
    values = {"revenue": [100.0, 200.0, 300.0, 400.0, 500.0]}
    clean = four_quarters("TEST", values, n=5)
    amended = clean.copy()
    repeat = clean.iloc[[1]].copy()
    repeat["date"] = repeat["date"] + pd.Timedelta(days=4)          # the 10-Q/A, 4 days later
    amended = pd.concat([clean, repeat], ignore_index=True)

    built_clean = build_ttm(translate(clean, field_map), field_map)
    built_amended = build_ttm(translate(amended, field_map), field_map)
    print(f"\nquarters              : {values['revenue']}")
    print(f"rows in, clean/amended: {len(clean)} / {len(amended)} (Q2 filed twice)")
    print(f"rows out              : {len(built_clean)} / {len(built_amended)}")
    print(f"TTM at Q4  clean      : {built_clean['totalRevenue'].iloc[3]} (expected 1000.0)")
    print(f"TTM at Q4  amended    : {built_amended['totalRevenue'].iloc[3]} (must MATCH)")
    print(f"whole windows         : {built_clean['totalRevenue'].notna().sum()} / "
          f"{built_amended['totalRevenue'].notna().sum()} (an amendment nulls nothing)")
    assert len(built_amended) == len(clean)
    assert built_amended["totalRevenue"].iloc[3] == 1000.0
    assert (built_amended["totalRevenue"].notna().sum()
            == built_clean["totalRevenue"].notna().sum() == 2)


def test_dedup_keeps_the_earliest_filing(field_map):
    """When an amendment RESTATES the number, the original stands.

    AR* is as-reported and immutable: taking the amendment would file a later restatement under
    an earlier publication date, which is a look-ahead. 97 of the 543 duplicate groups differ in
    value, so this is the choice that actually bites on real data.
    """
    clean = four_quarters("TEST", {"revenue": [100.0, 200.0, 300.0, 400.0]})
    restated = clean.iloc[[1]].copy()
    restated["date"] = restated["date"] + pd.Timedelta(days=4)
    restated["revenue"] = 999.0
    built = build_ttm(translate(pd.concat([clean, restated], ignore_index=True), field_map),
                      field_map)
    print(f"\nQ2 filed 200.0, then RESTATED to 999.0 four days later")
    print(f"revenue_q at Q2 : {built['revenue_q'].iloc[1]} (expected 200.0, the ORIGINAL)")
    print(f"TTM at Q4       : {built['totalRevenue'].iloc[3]} "
          f"(expected 1000.0, not the 1799.0 the restatement would give)")
    assert built["revenue_q"].iloc[1] == 200.0
    assert built["totalRevenue"].iloc[3] == 1000.0


def test_two_real_quarters_on_one_calendardate_both_survive(field_map):
    """Class A: dedup keys on `reportperiod`, so it cannot DELETE a real quarter.

    7 groups over 4 tickers (BBY, GPN, OKE, KR) are two genuine fiscal quarters whose ends
    normalise onto one calendar quarter. Keying the dedup on `calendardate` would look like the
    obvious fix and would silently destroy one of them -- the guard is here rather than in a
    comment.
    """
    frame = four_quarters("TEST", {"revenue": [100.0, 200.0, 300.0, 400.0]})
    collision = frame.iloc[[1]].copy()
    collision["reportperiod"] = collision["reportperiod"] - pd.Timedelta(days=20)
    collision["date"] = collision["date"] - pd.Timedelta(days=20)
    collision["revenue"] = 250.0                       # a DIFFERENT quarter, same normalisation
    built = build_ttm(translate(pd.concat([frame, collision], ignore_index=True), field_map),
                      field_map)
    print(f"\ntwo real quarters sharing calendardate "
          f"{pd.Timestamp(frame['calendardate'].iloc[1]).date()}")
    print(f"rows out       : {len(built)} (expected 5 -- BOTH survive)")
    print(f"revenue_q      : {sorted(built['revenue_q'].dropna().tolist())}")
    print("both 200.0 and 250.0 are present -- keying on calendardate would have dropped one")
    assert len(built) == 5
    assert {200.0, 250.0} <= set(built["revenue_q"].dropna())


def test_span_guard_nulls_a_spliced_window(field_map, caplog):
    """Four CONSECUTIVE quarter labels that do not span a year are still not a trailing twelve.

    The tripwire the 45-day drift cap was replaced with. Contiguity is measured on Sharadar's
    normalised `calendardate`, so a normalisation that mapped a 15-month stretch onto four
    adjacent calendar quarters would pass it -- the span check on the filer's OWN
    `reportperiod` is what refuses. It fires on nothing in today's data, so this synthetic case
    is the only thing that exercises it.
    """
    frame = four_quarters("TEST", {"revenue": [100.0, 200.0, 300.0, 400.0]})
    # Same four calendar quarters, but the filer's own period ends walk ~5 months apart:
    # 4 rows spanning ~15 months, which no trailing twelve can be.
    frame["reportperiod"] = pd.to_datetime(frame["calendardate"]) + \
        pd.to_timedelta([0, 60, 120, 180], unit="D")
    with caplog.at_level(logging.WARNING):
        built = build_ttm(translate(frame, field_map), field_map)
    span = (pd.to_datetime(frame["reportperiod"].iloc[3])
            - pd.to_datetime(frame["reportperiod"].iloc[0])).days
    tripped = [r for r in caplog.records if "do not span" in r.getMessage()]
    print(f"\n4 CONSECUTIVE calendar quarters, reportperiod span {span} days "
          f"(band {TTM_SPAN_DAYS})")
    print(f"TTM at the 4th row : {built['totalRevenue'].iloc[3]} "
          f"(expected NULL, not the 1000.0 contiguity alone would give)")
    print(f"warning logged     : {bool(tripped)} -- the tripwire must not be silent")
    assert pd.isna(built["totalRevenue"].iloc[3])
    assert tripped, "the span refusal was silent"


def test_off_calendar_filers_have_a_ttm_line(context, field_map, actions):
    """AVGO, KR, AZO and COST end their quarters far from a calendar quarter-end -- and must
    still get a trailing twelve.

    Real data, because the defect was entirely about what Sharadar's normalisation does to
    genuine filers and no fixture can encode that. Under the deleted 45-day cap these four lost
    239 ARQ rows between them: AVGO lost ALL 69 and was absent from `fundamentals_history`
    altogether, KR and AZO were 100% NULL revenue, COST 97.6%.
    """
    wanted = ["AVGO", "KR", "AZO", "COST"]
    vendor = context.store.load(Tables.sharadar_fundamentals, project=True,
                                where={"ticker": wanted})
    vendor = vendor[vendor["dimension"] == "ARQ"].copy()
    built = build_ttm(translate(vendor, field_map), field_map, actions=actions)
    whole = built.groupby("ticker")["totalRevenue"].apply(lambda s: int(s.notna().sum()))
    drift = (pd.to_datetime(vendor["calendardate"]) - pd.to_datetime(vendor["reportperiod"])
             ).dt.days.abs().groupby(vendor["ticker"]).max()
    print("\n=== SANITY CHECK: off-calendar filers ===")
    for ticker in wanted:
        print(f"  {ticker:<5} whole TTM rows {whole.get(ticker, 0):>4}  "
              f"max |calendardate - reportperiod| {int(drift.get(ticker, 0)):>3}d "
              f"(the old cap was 45d)")
    print("  -> every one of them clears 60 whole windows; none is empty.")
    for ticker in wanted:
        assert whole.get(ticker, 0) > 60, f"{ticker} has {whole.get(ticker, 0)} whole windows"
    assert drift.max() > 45, ("no filer here drifts past the old cap -- this test would pass "
                              "with the cap still in place and proves nothing")


def test_zero_rules_propagate_into_derived(field_map):
    """A field ruled `"null"` produces NaN through the TTM and into every derived formula --
    never a zero-contaminated result.

    `inventory` is ruled `null`, so a zero-filled quarter must not read as "this company held
    no inventory". `intexp` is ruled `null` too, and the point of nulling BEFORE the sum is
    that one missing quarter nulls the trailing twelve rather than understating it by a quarter.
    """
    values = {"revenue": [100.0, 100.0, 100.0, 100.0],
              "intexp": [5.0, 5.0, 0.0, 5.0],
              "inventory": [7.0, 7.0, 7.0, 0.0],
              "gp": [40.0, 40.0, 40.0, 40.0]}
    frame = build_ttm(translate(four_quarters("TEST", values), field_map), field_map)
    last = frame.iloc[-1]
    print(f"\nintexp quarters {values['intexp']} (one zero, ruled `null`)")
    print(f"  -> interestExpense TTM {last['interestExpense']} "
          f"(expected NULL, NOT the 15.0 a kept zero would give)")
    print(f"inventory quarters {values['inventory']} (period end is the zero)")
    print(f"  -> inventory {last['inventory']} (expected NULL, not 0.0)")
    print(f"grossMargins {last['grossMargins']} (expected 0.4 -- an unaffected column still "
          f"computes)")
    assert pd.isna(last["interestExpense"])
    assert pd.isna(last["inventory"])
    assert last["grossMargins"] == pytest.approx(0.4)


def test_a_zero_denominator_is_null_not_infinity(field_map):
    """`x / 0` survives every plausibility check downstream and then poisons a z-score."""
    frame = pd.DataFrame({"netIncome": [10.0], "totalRevenue": [0.0],
                          "stockholdersEquity": [0.0], "totalDebt": [5.0],
                          "grossProfit": [4.0], "operatingIncome": [3.0],
                          "incomeTaxExpense": [1.0], "pretaxIncome": [0.0],
                          "depAmort": [2.0], "dilutedShares": [0.0], "basicShares": [0.0],
                          # the repo name the extra is EMITTED under, not `cashneq`
                          "cashAndEquivalents": [1.0], "shortTermInvestments": [2.0],
                          "minorityInterest": [np.nan]})
    out = apply_derived(frame, field_map)
    ratios = {n: out[n].iloc[0] for n in ("profitMargins", "returnOnEquity", "grossMargins",
                                          "effectiveTaxRate", "optionOverhang", "epsDiluted")}
    print(f"\nzero-denominator ratios: {ratios}")
    print(f"cash (a SUM, both legs present): {out['cash'].iloc[0]} (expected 3.0)")
    print(f"stockholdersEquityInclNci with a NaN NCI leg: "
          f"{out['stockholdersEquityInclNci'].iloc[0]} (expected NaN, not 0.0)")
    assert all(pd.isna(v) for v in ratios.values())
    assert not np.isinf(out[list(ratios)].to_numpy(dtype="float64")).any()
    assert out["cash"].iloc[0] == 3.0
    assert pd.isna(out["stockholdersEquityInclNci"].iloc[0])


# --------------------------------------------------------------------------- #
# what the phase leaves behind                                                 #
# --------------------------------------------------------------------------- #
def test_coverage_of_the_built_frame(ttm, field_map, translated):
    """The phase's own scoreboard: what the transform produced, and what it removed and why.

    Printed rather than thresholded, apart from the two structural assertions. The duration
    columns are NULL for each ticker's first three quarters BY CONTRACT, so a bare percentage
    would read as a defect.
    """
    _, report = translated
    tickers = ttm["ticker"].nunique()
    coverage = ttm_coverage(ttm, field_map)
    census = coverage["basis"].value_counts(dropna=False).to_dict()
    expected_windows = len(ttm) - 3 * tickers
    print(f"\n{report.summary()}")
    print(f"\nrows out: {len(ttm)} over {tickers} ticker(s)")
    print(f"windows with 4 whole quarters: {expected_windows} "
          f"(= rows - 3 per ticker's cold start)")
    print(f"basis census: {census}")
    print(coverage[coverage["basis"].isin([DURATION, INSTANT, MEAN])]
          .nsmallest(8, "pct_non_null").to_string(index=False))
    print(f"totalRevenue non-null: {int(ttm['totalRevenue'].notna().sum())}")
    print(f"totalAssets  non-null: {int(ttm['totalAssets'].notna().sum())} "
          f"(an instant needs no window)")
    print(f"SEC-owned columns are all NULL until phase 4 merges them: "
          f"{all(ttm[n].isna().all() for n in field_map.sec_owned)}")
    assert int(ttm["totalRevenue"].notna().sum()) == expected_windows
    assert all(ttm[n].isna().all() for n in field_map.sec_owned)
    assert census.get(MEAN) == 2, "only the two weighted-average share counts are averaged"


def test_diluted_share_count_gaps_are_the_vendors(vendor_arq, ttm):
    """⚠ An ABSENCE the plan did not anticipate, made visible rather than left to be found.

    `shareswadil` is missing for 4 of the 30 tickers -- HON entirely, BA/CVX/PG partly -- which
    drags `dilutedShares`, `epsDiluted` and `optionOverhang` below every other column. It is a
    vendor gap, not a transform defect, and phase 4 has to decide whether the SEC layer covers
    it for the overlap tickers (BA and PG are on that roster; HON and CVX are not).
    """
    gaps = (vendor_arq.groupby("ticker")
            .agg(n=("shareswadil", "size"), n_dil=("shareswadil", "count")))
    gaps = gaps[gaps["n_dil"] < gaps["n"]]
    print(f"\ntickers missing a diluted share count: {len(gaps)}")
    print(gaps.to_string())
    print(f"dilutedShares non-null {int(ttm['dilutedShares'].notna().sum())} vs "
          f"basicShares {int(ttm['basicShares'].notna().sum())} of {len(ttm)} rows")
    assert int(ttm["dilutedShares"].notna().sum()) <= int(ttm["basicShares"].notna().sum())


# --------------------------------------------------------------------------- #
# the split de-adjustment runs AFTER the aggregation, and that ORDER is the fix #
# --------------------------------------------------------------------------- #
def test_a_ttm_window_straddling_a_split_stays_on_one_basis(field_map):
    """SYNTHETIC known-truth: the window that straddles a split must be POST-split, whole.

    Synthetic because the property is about an ORDER OF OPERATIONS, and real data can only
    show that the two orders happen to differ -- not what the right answer is. Here it is
    known: Sharadar stores every quarter on today's basis (200), the split is 2-for-1, so a
    row filed BEFORE it is 100 and a row filed AFTER it is 200. Nothing may ever be 125,
    which is what averaging four de-adjusted quarters across the split produces.
    """
    values = {"shareswadil": [200.0] * 8, "shareswa": [200.0] * 8, "sharesbas": [200.0] * 8,
              "consolinc": [10.0] * 8, "revenue": [100.0] * 8}
    frame = four_quarters("TEST", values, n=8)
    actions = pd.DataFrame([{"ticker": "TEST", "date": pd.Timestamp("2024-05-15"),
                             "action": "split", "value": 2.0, "contraticker": "N/A"}])
    built = build_ttm(translate(frame, field_map), field_map, actions=actions)
    got = built[["date", "dilutedShares"]].copy()
    got["filed"] = pd.to_datetime(got["date"]).dt.date
    print("\nsplit 2024-05-15 x2; Sharadar stores 200 on every quarter")
    print(got[["filed", "dilutedShares"]].to_string(index=False))

    before = built[pd.to_datetime(built["date"]) < "2024-05-15"]["dilutedShares"].dropna()
    after = built[pd.to_datetime(built["date"]) >= "2024-05-15"]["dilutedShares"].dropna()
    assert (before == 100.0).all(), f"pre-split rows must be as-filed 100, got {list(before)}"
    assert (after == 200.0).all(), f"post-split rows must be 200, got {list(after)}"
    assert not ((built["dilutedShares"] > 100.0) & (built["dilutedShares"] < 200.0)).any(), \
        "a value between the two bases means the window averaged across the split"
    print("no window mixes the two bases -- de-adjustment ran AFTER the aggregation")


def test_post_split_share_counts_are_not_a_hybrid_basis(ttm):
    """REAL: after a split, the TTM share count must still be a plausible share count.

    The regression this pins cost up to 3.5x. De-adjusting each QUARTER first put pre- and
    post-split numbers in one four-quarter window, so AMZN's 2022-07-29 `dilutedShares` read
    2.93bn against 10.19bn actually outstanding, and `epsDiluted` 3.96 against ~1.14.

    The instrument is `dilutedShares / sharesOutstanding`. A weighted average and a period-end
    count legitimately differ by a few percent; they cannot differ by a factor of three.
    """
    rows = ttm[["ticker", "date", "dilutedShares", "sharesOutstanding", "epsDiluted"]].dropna(
        subset=["dilutedShares", "sharesOutstanding"]).copy()
    rows["ratio"] = rows["dilutedShares"] / rows["sharesOutstanding"]
    worst = rows.reindex(rows["ratio"].sub(1.0).abs().sort_values(ascending=False).index)
    print(f"\ndilutedShares / sharesOutstanding over {len(rows)} row(s): "
          f"min {rows['ratio'].min():.3f}, max {rows['ratio'].max():.3f}")
    print(worst.head(6).to_string(index=False))
    assert (rows["ratio"] > 0.5).all() and (rows["ratio"] < 1.5).all(), (
        "a TTM share count is off its own period-end count by more than 50% -- the window "
        "straddled a split and averaged two bases")
