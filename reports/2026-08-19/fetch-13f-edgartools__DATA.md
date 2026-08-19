---
type: DATA
session_id: 5fde04fa-3eac-46c3-9459-88684c475136
generated_at: 2026-08-19T21:06:53+00:00
baseline: {head_sha: 9d02d55a2bddc17624a05957d3d2f52f3be9b255}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: cusip_ticker_map, sec13f_hr
- tickers: **all** (no ticker filter)
- since: **no lower bound**
- row limit per table: **none**
- full-scope tables (eligible to set the baseline): sec13f_hr, cusip_ticker_map

**What was asked:** <!-- TODO(agent): fill this in, in your own words. -->

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 2 table(s): sec13f_hr, cusip_ticker_map |
| D2 | row count not decreased | **N/A** | no full-scope baseline to compare against — this run records one |
| D3 | no column lost | **N/A** | no baseline columns recorded yet |
| D4 | date range covers the expected window | **FAIL** | sec13f_hr: max 2026-03-31 < 2026-05-29 |
| D5 | per-field null rate not worse | **N/A** | no full-scope baseline null rates to compare against |

**1 FAIL** — D4. The work is **NOT done**.

## 3. Metrics

_Observed values only — no verdicts. `rows`, `date_min` and `date_max` are **table-wide** (server-side); every other number is over the **sample** described in §1. Do not compare across the two._

**Tables**

| table | exists | rows | sampled | cols | pk | pk_absent_cols | pk_dupes | date_min | date_max | sample_date_min | sample_date_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cusip_ticker_map | yes | 145,748 | 145,748 | 2 | cusip | — | 0 | — | — | — | — |
| sec13f_hr | yes | 21,659,435 | 21,659,435 | 15 | cik,period,ticker,cusip | — | 0 | 1987-03-31 00:00:00 | 2026-03-31 00:00:00 | 1987-03-31 | 2026-03-31 |

**Fields** (worst null rate first, top 17)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cusip_ticker_map | ticker | str | 86.4 | 19,824 | — | — | — | — | — | — | — | — |
| cusip_ticker_map | cusip | str | 0 | 145,748 | — | — | — | — | — | — | — | — |
| sec13f_hr | call_shares | int64 | 0 | 40,049 | 7,473.59 | 276,602 | 0 | 0 | 0 | 58,400 | 5e+08 | 254,964 |
| sec13f_hr | call_value | float64 | 0 | 220,711 | 6.78108e+06 | 9.51069e+08 | 0 | 0 | 0 | 5.38e+06 | 1.0143e+12 | 77,199 |
| sec13f_hr | cik | str | 0 | 11,779 | — | — | — | — | — | — | — | — |
| sec13f_hr | cusip | str | 0 | 497 | — | — | — | — | — | — | — | — |
| sec13f_hr | debt_prn | int64 | 0 | 8,191 | 160.002 | 244,435 | 0 | 0 | 0 | 0 | 7.94117e+08 | 10,520 |
| sec13f_hr | debt_value | float64 | 0 | 8,573 | 2.14256e+06 | 4.00605e+09 | 0 | 0 | 0 | 0 | 1.63131e+13 | 2,627 |
| sec13f_hr | filing_date | datetime64[us] | 0 | 2,950 | — | — | — | — | — | — | — | — |
| sec13f_hr | other_value | float64 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sec13f_hr | period | datetime64[us] | 0 | 98 | — | — | — | — | — | — | — | — |
| sec13f_hr | put_shares | int64 | 0 | 41,702 | 7,549.68 | 254,200 | 0 | 0 | 0 | 54,700 | 1.96703e+08 | 246,615 |
| sec13f_hr | put_value | float64 | 0 | 221,651 | 1.22715e+07 | 1.87602e+09 | 0 | 0 | 0 | 5.37e+06 | 1.65642e+12 | 53,674 |
| sec13f_hr | quarter | str | 0 | 51 | — | — | — | — | — | — | — | — |
| sec13f_hr | shares | int64 | 0 | 1,916,767 | 488,509 | 6.07184e+06 | 0 | 2 | 12,464 | 8.36477e+06 | 6.617e+09 | 5,366,957 |
| sec13f_hr | ticker | str | 0 | 497 | — | — | — | — | — | — | — | — |
| sec13f_hr | value_usd | float64 | 0 | 4,734,517 | 3.33425e+08 | 1.89005e+10 | 0 | 1 | 1.206e+06 | 1.51272e+09 | 1.59519e+13 | 5,375,396 |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (3 table(s) recorded)
- `cusip_ticker_map`: 145,748 rows, 2 cols, 145,748 sampled
- `sec13f_hr`: 21,659,435 rows, 15 cols, 21,659,435 sampled

## 5. Regressions, gaps and deliberate omissions

<!-- TODO(agent): fill this in, in your own words. -->
- 
<!-- At least one bullet. If genuinely nothing: `- None. Checked: <30+ chars>` -->

## 6. Next actions

<!-- TODO(agent): fill this in, in your own words. -->
- 

```json dod-metrics
{
  "baseline_head_sha": "9d02d55a2bddc17624a05957d3d2f52f3be9b255",
  "content_hash": "sha256:e6188d11b88231536448451066c588fbf31c694242dc6ac549cee02c6478f3a5",
  "gates": {
    "D1": "PASS",
    "D2": "N/A",
    "D3": "N/A",
    "D4": "FAIL",
    "D5": "N/A"
  },
  "generator": "scripts/dod/data_profile.py@1",
  "metrics": {
    "parts_behind": null,
    "stale_sources": null,
    "tables": {
      "cusip_ticker_map": {
        "columns": [
          "cusip",
          "ticker"
        ],
        "date_col": null,
        "date_max": null,
        "date_min": null,
        "exists": true,
        "fields": {
          "cusip": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 145748
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.8639775502922853,
            "nulls": 125923,
            "nunique": 19824
          }
        },
        "kind": "extract",
        "pk": [
          "cusip"
        ],
        "pk_checked_cols": [
          "cusip"
        ],
        "pk_checked_rows": 145748,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 145748,
        "sample_date_max": null,
        "sample_date_min": null,
        "sampled_rows": 145748,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "cusip_ticker_map"
      },
      "sec13f_hr": {
        "columns": [
          "cik",
          "period",
          "filing_date",
          "ticker",
          "cusip",
          "shares",
          "value_usd",
          "call_shares",
          "call_value",
          "put_shares",
          "put_value",
          "debt_prn",
          "debt_value",
          "other_value",
          "quarter"
        ],
        "date_col": "period",
        "date_max": "2026-03-31 00:00:00",
        "date_min": "1987-03-31 00:00:00",
        "exists": true,
        "fields": {
          "call_shares": {
            "dtype": "int64",
            "mad_center": 0.0,
            "mad_outliers": 254964,
            "mad_scale": 7473.593527531997,
            "max": 500000000.0,
            "mean": 7473.593527531997,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 40049,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 58400.0,
            "std": 276601.74488239025
          },
          "call_value": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 77199,
            "mad_scale": 6781080.826653188,
            "max": 1014296465000.0,
            "mean": 6781080.826653188,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 220711,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 5380000.0,
            "std": 951069206.1146668
          },
          "cik": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 11779
          },
          "cusip": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 497
          },
          "debt_prn": {
            "dtype": "int64",
            "mad_center": 0.0,
            "mad_outliers": 10520,
            "mad_scale": 160.0021380982468,
            "max": 794117000.0,
            "mean": 160.0021380982468,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8191,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.0,
            "std": 244434.69228556816
          },
          "debt_value": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 2627,
            "mad_scale": 2142559.660109509,
            "max": 16313100000000.0,
            "mean": 2142559.660109509,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8573,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.0,
            "std": 4006048441.3695707
          },
          "filing_date": {
            "dtype": "datetime64[us]",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2950
          },
          "other_value": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.0,
            "std": 0.0
          },
          "period": {
            "dtype": "datetime64[us]",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 98
          },
          "put_shares": {
            "dtype": "int64",
            "mad_center": 0.0,
            "mad_outliers": 246615,
            "mad_scale": 7549.675037783765,
            "max": 196702900.0,
            "mean": 7549.675037783765,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 41702,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 54700.0,
            "std": 254199.74139719098
          },
          "put_value": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 53674,
            "mad_scale": 12271477.905007632,
            "max": 1656420920000.0,
            "mean": 12271477.905007632,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 221651,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 5370000.0,
            "std": 1876021050.374769
          },
          "quarter": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 51
          },
          "shares": {
            "dtype": "int64",
            "mad_center": 12464.0,
            "mad_outliers": 5366957,
            "mad_scale": 12012.0,
            "max": 6616999998.0,
            "mean": 488508.8457147659,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1916767,
            "p01": 2.0,
            "p25": 2576.0,
            "p50": 12464.0,
            "p75": 73315.0,
            "p99": 8364774.6000000015,
            "std": 6071835.565357076
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 497
          },
          "value_usd": {
            "dtype": "float64",
            "mad_center": 1206000.0,
            "mad_outliers": 5375396,
            "mad_scale": 1181534.0,
            "max": 15951857000000.0,
            "mean": 333425292.98414195,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 4734517,
            "p01": 1.0,
            "p25": 284000.0,
            "p50": 1206000.0,
            "p75": 7220000.0,
            "p99": 1512724720.0000062,
            "std": 18900543122.191788
          }
        },
        "kind": "extract",
        "pk": [
          "cik",
          "period",
          "ticker",
          "cusip"
        ],
        "pk_checked_cols": [
          "cik",
          "period",
          "ticker",
          "cusip"
        ],
        "pk_checked_rows": 21659435,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 21659435,
        "sample_date_max": "2026-03-31",
        "sample_date_min": "1987-03-31",
        "sampled_rows": 21659435,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "sec13f_hr"
      }
    }
  },
  "scope": {
    "limit": null,
    "since": null,
    "tables": [
      "cusip_ticker_map",
      "sec13f_hr"
    ],
    "tickers": [],
    "unknown_tables": []
  },
  "session_id": "5fde04fa-3eac-46c3-9459-88684c475136",
  "type": "DATA"
}
```

