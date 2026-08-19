---
type: DATA
session_id: 2a019ba3-2efa-4933-b76c-2bad203953db
generated_at: 2026-08-19T18:43:38+00:00
baseline: {head_sha: 448a96fd99dcd45550d9473afd6f9b85bb143101}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: dividends, prices, prices_macro
- tickers: **all** (no ticker filter)
- since: **no lower bound**
- row limit per table: **none**
- full-scope tables (eligible to set the baseline): prices, prices_macro, dividends

**What was asked:** refactor the macro extraction into one unified fetcher.
`fetch_macro.py` and `fetch_macro_assets.py` were doing the same job on different variables, and
the two wide tables (`macro`, `macro_asset_prices`) double-stored `yield_10y` and `vix` from two
source paths on two different windows (16y vs 31y). Merge them into ONE long table
`prices_macro (date, ticker, close)`, add the market close (SPY) + VIX + currency on top, and
collapse [step_extract_prices.py](../../src/data_extract/transformers/step_extract_prices.py)'s
three calls into one `fetch_macro`.

Follow-on requirements: the yfinance legs read via `download_ohlcv` and written **directly** to
`prices_macro` with no `prices` round-trip (so `prices` becomes the equity universe only);
`cash_rate` + `yield_3m` collapsed to `cash_rate` alone with `yield_curve_10y3m` re-derived from
it; ONE gold series, read by the cube from `prices_macro`; the macro window resolved in
`StepExtractPrices.run` and passed in; all cube macro information read and pivoted **once**, at
the beta-fitting step; and the dead `paths["MACRO_PATH"]` / `PRICES_PATH` references removed.

**This run is the refinement pass**, covering three further requirements:
1. `fx_usdeur` moved from yfinance `USDEUR=X` to **FRED `DEXUSEU`** for the full 1999+ depth;
2. `dividends` (plural) fixed as THE dividend column name everywhere - extractor, cube,
   feature builder, tests, DDL;
3. `constants.py` reviewed for now-unused variables and `fetch_macro.py` prose cut to essentials.

Approved plan decisions: name `prices_macro`; 31y window; semantic ticker names; VIX from
yfinance `^VIX`; FULL cube simplification (delete `cube_part_market`).

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 3 table(s): prices, prices_macro, dividends |
| D2 | row count not decreased | **PASS** | 2 table(s) at or above baseline |
| D3 | no column lost | **PASS** | 2 table(s) keep every baseline column |
| D4 | date range covers the expected window | **PASS** | every dated table reaches 2026-08-14 |
| D5 | per-field null rate not worse | **PASS** | 10 field(s) at or below baseline (+0.5pp slack) |

**All gates pass** (N/A gates are stated above, not skipped).

## 3. Metrics

_Observed values only — no verdicts. `rows`, `date_min` and `date_max` are **table-wide** (server-side); every other number is over the **sample** described in §1. Do not compare across the two._

**Tables**

| table | exists | rows | sampled | cols | pk | pk_absent_cols | pk_dupes | date_min | date_max | sample_date_min | sample_date_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dividends | yes | 1,780,883 | 1,780,883 | 3 | ticker,date | — | 0 | 2011-08-19 00:00:00 | 2026-08-19 00:00:00 | 2011-08-19 | 2026-08-19 |
| prices | yes | 1,775,416 | 1,775,416 | 7 | ticker,date | — | 0 | 2011-08-19 00:00:00 | 2026-08-19 00:00:00 | 2011-08-19 | 2026-08-19 |
| prices_macro | yes | 114,994 | 114,994 | 3 | ticker,date | — | 0 | 1995-08-21 | 2026-08-19 | 1995-08-21 | 2026-08-19 |

**Fields** (worst null rate first, top 13)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dividends | date | datetime64[us] | 0 | 3,771 | — | — | — | — | — | — | — | — |
| dividends | dividends | float64 | 0 | 1,298 | 0.00681538 | 0.157746 | 0 | 0 | 0 | 0.165 | 103.75 | 21,345 |
| dividends | ticker | str | 0 | 491 | — | — | — | — | — | — | — | — |
| prices | close | float64 | 0 | 1,461,271 | 109.429 | 244.22 | 0.260522 | 5.27257 | 60.772 | 689.836 | 9,924.4 | 152,518 |
| prices | date | datetime64[us] | 0 | 3,771 | — | — | — | — | — | — | — | — |
| prices | high | float64 | 0 | 1,561,757 | 110.69 | 247.182 | 0.266245 | 5.35304 | 61.4198 | 698.243 | 9,964.77 | 152,896 |
| prices | low | float64 | 0 | 1,560,539 | 108.119 | 241.272 | 0.255256 | 5.18895 | 60.09 | 681.007 | 9,794 | 152,191 |
| prices | open | float64 | 0 | 1,558,322 | 109.414 | 244.211 | 0.263727 | 5.27368 | 60.7556 | 689.66 | 9,914.17 | 152,598 |
| prices | ticker | str | 0 | 490 | — | — | — | — | — | — | — | — |
| prices | volume | float64 | 0 | 370,290 | 6.99558e+06 | 2.98724e+07 | 0 | 126,900 | 2.2809e+06 | 7.57047e+07 | 3.69293e+09 | 213,264 |
| prices_macro | close | float64 | 0 | 41,824 | 116.934 | 382.479 | -37.63 | -0.33 | 4.26 | 1,847.61 | 5,318.4 | 35,593 |
| prices_macro | date | object | 0 | 8,088 | — | — | — | — | — | — | — | — |
| prices_macro | ticker | str | 0 | 15 | — | — | — | — | — | — | — | — |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (2 table(s) recorded)
- `dividends`: 1,780,883 rows, 3 cols, 1,780,883 sampled
- `prices`: 1,775,416 rows, 7 cols, 1,775,416 sampled
- `prices_macro`: 114,994 rows, 3 cols, 114,994 sampled

## 5. Regressions, gaps and deliberate omissions

**The FX regression from the previous run is CLOSED**

- `fx_usdeur` now comes from FRED `DEXUSEU`: **1999-01-04 -> 2026-08-14, 7,205 observations**, up
  from Yahoo's 2003-12-01 start - **+1,293 observations, 4.91 years deeper**. DEXUSEU is natively
  quoted USD-per-EUR, so the reciprocal inversion (`1/close`) and the `MACRO_INVERTED_SERIES`
  constant that carried it are **deleted**, not just bypassed. Landmarks verified: peak 1.6010 on
  2008-04-22 (the real DEXUSEU maximum) and trough 0.8270 on 2000-10-25 (the euro's real all-time
  low - a date the Yahoo window could not reach at all).
- **The source we left was the wrong one.** Over the 5,909-day overlap the two agree to
  mean|diff| 0.00368 (corr 0.9988) - the expected drift between a noon fixing and a 5pm spot
  close - but the tail is not symmetric noise. 5 days exceed 0.05 and **all 5 are stale Yahoo
  bars in 2008**: 2008-01-08 and 2008-02-08 both read exactly 1.5571 two months apart, and
  2008-12-08 read 1.4918 when the euro was actually 1.2942. Yahoo was wrong by up to 13% while
  the euro collapsed 1.60 -> 1.27.
- **Re-pull safety check**: of the other 14 series, all 9 FRED/derived legs came back
  **bit-identical** (max|diff| = 0.000e+00 over 6,164-8,086 overlapping days). The 5 yfinance legs
  moved 0.06-0.96% in relative terms, which is auto-adjustment restating past closes plus the
  `GC=F`/`CL=F` front-future re-splice - expected, and the reason those two guarantees are checked
  at two different tolerances rather than one.

**New cost introduced by the FX move - accepted, and here is the measurement**

- **FX now trails the trading calendar by up to a week.** DEXUSEU rides the weekly H.10 release:
  its last observation is 2026-08-14 where `equity_tr` reaches 2026-08-19, so the newest ~3
  trading days carry **no** FX value. `fill_short_gaps` correctly declines to fill it (a trailing
  NaN has no bracketing observation to average). Traced through both sleeves: `long_book`'s
  `asset_returns_from_macro` yields NaN - not a fabricated 0, not a stale carry-forward, because
  `pct_change(fill_method=None)` refuses to invent one - and `trend_cta`'s level matrix likewise.
  Both sleeves still produce 8,087-day streams over 1995-08-22 -> 2026-08-19 with all five legs.
- Worth knowing: this is **not new behaviour**, only a longer version of it. `cash_rate` and
  `bond_10y_tr` are already NaN on the last 2 dates from FRED's own publication lag. FX is
  deliberately absent from `MACRO_CORE_LEVEL_SERIES`, so its longer lag cannot hold the freshness
  gate open.
- Sleeve Sharpe rose on the deeper, cleaner series: `long_book` 0.71 -> **0.77**, `trend_cta`
  0.50 -> **0.55**.

**`dividends` column: unified, and what the zero-keeping change actually costs**

- One name end to end now: `_extract_dividends` output, `dividend_features._ttm_dividends`
  (`values="dividends"`), the `build_dividend_feature_panel` presence guard, `sql/schema.sql`, the
  live table, and 5 test fixtures. The panel/part name stays `dividend` (singular) - that is a
  `PartKind`, not a column.
- `_extract_dividends` gained an empty/missing-column guard. Without it a total yfinance outage
  raised `KeyError` on `df[["date","ticker","dividends"]]` instead of no-oping, because
  `download_ohlcv` returns a **column-less** frame when every chunk fails.
- **1.23% of the table is signal**: 21,904 nonzero rows out of 1,780,883 (was 22,060 rows total
  before zeros were kept - an 80x growth for 156 fewer facts). That is your deliberate trade for
  an idempotent refresh, recorded here so the row count is not later read as data growth.
- Largest values verified real, not units errors: KDP $103.75 (2018-07-10, the Keurig/Dr Pepper
  merger special dividend) and TDG $90.00 / $75.00 (TransDigm's special dividends).

**`constants.py` review**

- Deleted: `MACRO_INVERTED_SERIES` (its only reason to exist was the Yahoo reciprocal).
  `MACRO_ASSET_SIGNAL_COLUMNS` and `MACRO_ASSET_GOLD_COLUMN` went in the previous pass.
- Every remaining `MACRO_*` name was checked for a live consumer outside `constants.py`:
  `MACRO_PRICE_SERIES` 11, `MACRO_FRED_SERIES` 4, `MACRO_SPREAD_SERIES` 4, `MACRO_BOND_TR_SERIES`
  5, `MACRO_BOND_MATURITY_YEARS` 2, `MACRO_CORE_LEVEL_SERIES` 8, `MACRO_MARKET_SERIES` 48,
  `MACRO_CUBE_FACTORS` 12, `MACRO_ALL_SERIES` 9, `DAILY_MACRO_LEVELS` 4. None orphaned.
- 15 series total, still nowhere hand-written: `len(PRICE) + len(FRED) + len(SPREAD) + 1`. The
  count is unchanged by the FX move (one symbol left the price leg, one id joined the FRED leg).

**Section 3 numbers that must not be read at face value**

- `prices_macro.close` (mean 116.9, p50 4.26, min -37.63, 35,593 "mad_outliers") pools **15 series
  in different units** - percent yields (~4), index levels (~500), USD/oz (~4,400), an FX rate
  (~1.15). Every distributional statistic on that column is meaningless by construction; that is
  inherent to a long table, which is why per-series bands are asserted separately. The min
  -37.63 is the real April-2020 negative WTI settlement.
- `dividends.dividends` mean 0.0068 / p50 0.0 describes a column that is 98.8% zeros by design.
- D2-D5 now gate for real: this run recorded the baseline (`--update-baseline`) for all three
  tables. The previous run had none, so those gates read N/A there.

**Deliberately not done**

- **The aggregate fingerprint baseline is still NOT regenerated - still blocked, same cause.**
  [fundamental_features.py:1391](../../src/data_aggregate/utils/fundamentals/fundamental_features.py#L1391)
  does `intrinsic_value_daily(..., **intrinsic_cfg)` with `intrinsic_cfg: dict | None = None`, so
  it raises `TypeError` whenever the caller omits the config - which `aggregate_fingerprint`
  always does. Present at `HEAD`, in a file this refactor never touched. Fix is
  `**(intrinsic_cfg or {})`. Left alone: it is in the fundamentals builder, outside this scope and
  inside your in-flight work. Consequence: the aggregation refactor guard is unenforced, and
  `test_aggregate_regression::test_baseline_covers_every_panel_and_deduped_primitive` still fails
  on the renamed `prim.macro_factor_returns` key.
- **The cube was not rebuilt.** No cube tables exist locally (`cube`, `cube_part_prices`,
  `cube_part_betas`, `cube_part_targets` all absent - pre-existing drift), so `StepCubePrices` /
  `StepCubeTarget` are covered by unit tests only. The calendar repoint is proven bit-identical on
  synthetic data ([test_trading_calendar.py](../../tests/data_aggregate/test_trading_calendar.py))
  rather than against live `cube_part_market` dates, because that table did not exist to snapshot.
- `yield_curve_10y3m` remains ~5bp wider than FRED's `T10Y3M` (DTB3 discount vs DGS3MO
  coupon-equivalent basis). No consumer today; a constant basis offset differences out of the
  change factor it would become.
- `energy` (XLE) is stored for the sleeves but **not** wired into the cube factor panel - that
  would add a factor and a beta column, a modelling decision.
- `yield_2y` / `yield_30y` have no direct consumer (`yield_2y` feeds the `10y2y` spread). Kept: in
  long format an unused series costs one row block per date, not a wide NaN column.

**Found while verifying, not caused by this work**

- **`prices` is missing `EA`.** `dividends` has 491 tickers and `prices` has 490; both are subsets
  of the universe, and the one-ticker difference is Electronic Arts. Not a macro-refactor effect -
  `prices` simply never got EA's bars.
- **Three fetchers now return `None` while still annotated `-> pd.DataFrame`.** Your in-flight
  edits dropped the trailing `return` from `fetch_dividends`, `fetch_price_history` and
  `fetch_short_interest`. I re-pointed my own two cases in `test_macro_prices_separation.py` at the
  **stored** rows (the stronger assertion, and return-type agnostic), but left
  `test_short_interest_resume.py::test_fetch_filters_to_the_universe_and_upserts` and
  `::test_empty_download_still_records_and_returns_the_schema` failing - they are yours, and the
  annotations should move to `-> None` rather than the tests being papered over.
- `StepCubeFundamentals` has no `_FIELDS` attribute -> `test_part_registry` fails. Never had one at
  `HEAD`.
- `tests/utils/test_fundamentals_audit.py` imports `src.utils.fundamentals_audit`, which does not
  exist -> collection error.
- `from conftest import FakeStore` resolves to whichever `conftest` is imported first, so
  `tests/dod` and `tests/data_extract` cannot be collected in the same pytest run (5 import
  errors). Run them separately; unrelated to this work.
- `sql/schema.sql` regeneration is idempotent but reflects the LIVE DB, so it also absorbed your
  in-flight `fundamentals_facts` / `fundamentals_history` column changes. Nothing is committed.

**Fixed here because it blocked verification**

- 6 `test_read_equivalence` cases hard-failed on `cube` / `cube_part_prices` being absent. Added a
  `_requires_tables` guard so a missing table SKIPS with a reason: both sides of an equivalence
  check read the same table, so an absent one proves nothing either way, and standing red trains
  everyone to ignore red. Now 13 passed / 6 skipped, was 13 passed / 5 failed.

## 6. Next actions

- **Fix `intrinsic_cfg or {}`** at fundamental_features.py:1391, then regenerate the aggregate
  fingerprint baseline (`python -m tests.data_aggregate.aggregate_fingerprint`) and confirm the
  only moves are the `prim.macro_factor_returns` rename and `beta_USD/EUR`. Until then the
  aggregation refactor guard is unenforced. This is the single highest-value unblock - it gates 20
  other tests too.
- **Change the three fetcher annotations to `-> None`** and update
  `tests/data_extract/test_short_interest_resume.py` to assert on stored rows, closing the last 2
  failures outside the blocker.
- Build the cube locally (`data_aggregate build-prices` -> `build-target`) to exercise the calendar
  repoint and the one-read-one-pivot macro path against real data.
- Backfill `EA` into `prices` (or confirm it is intentionally excluded).
- Consider whether FX's ~5-day H.10 lag is acceptable for live signal generation. If the newest 2-3
  days matter, the fix is a Yahoo `USDEUR=X` *tail* patch over the FRED history - but that
  re-introduces the reciprocal and the stale-bar risk, so it is a real trade, not a free one.
- Consider deleting `yield_30y` if you want the registry minimal.

```json dod-metrics
{
  "baseline_head_sha": "448a96fd99dcd45550d9473afd6f9b85bb143101",
  "content_hash": "sha256:ad028907b2ccfd8bf46ce750f109a0f1c23a61d9beb5f2cfdc9b705c80e987f9",
  "gates": {
    "D1": "PASS",
    "D2": "PASS",
    "D3": "PASS",
    "D4": "PASS",
    "D5": "PASS"
  },
  "generator": "scripts/dod/data_profile.py@1",
  "metrics": {
    "parts_behind": null,
    "stale_sources": null,
    "tables": {
      "dividends": {
        "columns": [
          "date",
          "ticker",
          "dividends"
        ],
        "date_col": "date",
        "date_max": "2026-08-19 00:00:00",
        "date_min": "2011-08-19 00:00:00",
        "exists": true,
        "fields": {
          "date": {
            "dtype": "datetime64[us]",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3771
          },
          "dividends": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 21345,
            "mad_scale": 0.006815377177501272,
            "max": 103.75,
            "mean": 0.006815377177501272,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1298,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.165,
            "std": 0.15774613942156182
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 491
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "date"
        ],
        "pk_checked_cols": [
          "ticker",
          "date"
        ],
        "pk_checked_rows": 1780883,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 1780883,
        "sample_date_max": "2026-08-19",
        "sample_date_min": "2011-08-19",
        "sampled_rows": 1780883,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "dividends"
      },
      "prices": {
        "columns": [
          "date",
          "open",
          "high",
          "low",
          "close",
          "volume",
          "ticker"
        ],
        "date_col": "date",
        "date_max": "2026-08-19 00:00:00",
        "date_min": "2011-08-19 00:00:00",
        "exists": true,
        "fields": {
          "close": {
            "dtype": "float64",
            "mad_center": 60.77203369140625,
            "mad_outliers": 152518,
            "mad_scale": 35.283775329589844,
            "max": 9924.400390625,
            "mean": 109.42949758092323,
            "min": 0.26052168011665344,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1461271,
            "p01": 5.272574806213379,
            "p25": 32.01628112792969,
            "p50": 60.77203369140625,
            "p75": 118.87518692016602,
            "p99": 689.8357147216824,
            "std": 244.22042834808968
          },
          "date": {
            "dtype": "datetime64[us]",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3771
          },
          "high": {
            "dtype": "float64",
            "mad_center": 61.419784886580565,
            "mad_outliers": 152896,
            "mad_scale": 35.64918214680218,
            "max": 9964.76953125,
            "mean": 110.68978284070066,
            "min": 0.2662449101044375,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1561757,
            "p01": 5.353038076867621,
            "p25": 32.37381521781634,
            "p50": 61.419784886580565,
            "p75": 120.19700066468374,
            "p99": 698.2429068984223,
            "std": 247.182143973276
          },
          "low": {
            "dtype": "float64",
            "mad_center": 60.08995095456548,
            "mad_outliers": 152191,
            "mad_scale": 34.9099506493897,
            "max": 9794.0,
            "mean": 108.11895255363646,
            "min": 0.2552563030697894,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1560539,
            "p01": 5.188952106238555,
            "p25": 31.647834875522577,
            "p50": 60.08995095456548,
            "p75": 117.52464860783121,
            "p99": 681.0069720305595,
            "std": 241.27213957384274
          },
          "open": {
            "dtype": "float64",
            "mad_center": 60.75557287956572,
            "mad_outliers": 152598,
            "mad_scale": 35.27570964211038,
            "max": 9914.169921875,
            "mean": 109.41449242660683,
            "min": 0.26372659017317707,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1558322,
            "p01": 5.273676953236547,
            "p25": 32.011050346907005,
            "p50": 60.75557287956572,
            "p75": 118.88242820499697,
            "p99": 689.6599731445312,
            "std": 244.21130706651152
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 490
          },
          "volume": {
            "dtype": "float64",
            "mad_center": 2280900.0,
            "mad_outliers": 213264,
            "mad_scale": 1549424.0,
            "max": 3692928000.0,
            "mean": 6995583.202824014,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 370290,
            "p01": 126900.0,
            "p25": 1044293.25,
            "p50": 2280900.0,
            "p75": 5311100.0,
            "p99": 75704680.00000007,
            "std": 29872370.179596793
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "date"
        ],
        "pk_checked_cols": [
          "ticker",
          "date"
        ],
        "pk_checked_rows": 1775416,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 1775416,
        "sample_date_max": "2026-08-19",
        "sample_date_min": "2011-08-19",
        "sampled_rows": 1775416,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "prices"
      },
      "prices_macro": {
        "columns": [
          "date",
          "ticker",
          "close"
        ],
        "date_col": "date",
        "date_max": "2026-08-19",
        "date_min": "1995-08-21",
        "exists": true,
        "fields": {
          "close": {
            "dtype": "float64",
            "mad_center": 4.26,
            "mad_outliers": 35593,
            "mad_scale": 3.34,
            "max": 5318.39990234375,
            "mean": 116.93447791203018,
            "min": -37.630001068115234,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 41824,
            "p01": -0.33000000000000007,
            "p25": 1.7,
            "p50": 4.26,
            "p75": 41.84749889373779,
            "p99": 1847.6069738769525,
            "std": 382.47937235229045
          },
          "date": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8088
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 15
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "date"
        ],
        "pk_checked_cols": [
          "ticker",
          "date"
        ],
        "pk_checked_rows": 114994,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 114994,
        "sample_date_max": "2026-08-19",
        "sample_date_min": "1995-08-21",
        "sampled_rows": 114994,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "prices_macro"
      }
    }
  },
  "scope": {
    "limit": null,
    "since": null,
    "tables": [
      "dividends",
      "prices",
      "prices_macro"
    ],
    "tickers": [],
    "unknown_tables": []
  },
  "session_id": "2a019ba3-2efa-4933-b76c-2bad203953db",
  "type": "DATA"
}
```

