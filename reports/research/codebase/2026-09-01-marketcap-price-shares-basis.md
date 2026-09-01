# Research: the price / sharesOutstanding basis gap behind `marketCap`

**Date**: 2026-09-01
**Research Phase**: 1 of 3 (FIC Workflow)
**Next Phase**: Planning — plan written, `reports/planning/active-tasks/2026-09-01-price-shares-basis-fix/PLAN.md`
**Measured against**: live `pea` Postgres (491 tickers, `prices` 3,263,505 rows 1995-08-30 → 2026-08-28;
`fundamentals_history` joined to `fundamentals_sharadar` ARQ on 51,255 rows / 489 tickers) and
yfinance 1.5.1 live
**Revised 2026-09-01** after verifying yfinance's `auto_adjust=False` semantics: §1 gains the
two-column measurement, §3 the yfinance split-coverage cross-check, §Reconciliation is rewritten
around the cancellation identity (the earlier "Option A vs B" framing was superseded), and the
BKNG / AMCR findings are corrected — both were real missing splits, not vendor defects.

## Research Question

`marketCap = price × sharesOutstanding`. Prices come from yfinance and are back-adjusted
retroactively; Sharadar `sharesOutstanding` is back-filled to reflect the share count that
really existed at the time. Are the two bases consistent across the whole time series for all
stocks, and if not, how can they be reconciled? Goal: leak-free features for a cross-sectional
long/short Sharpe-maximising strategy.

## Summary

**The gut feeling is correct, and the gap is larger than a basis mismatch — it is three
independent defects that compound multiplicatively into the market-cap denominator.**

There is exactly one market-cap formula in the repo, `daily_market_cap` at
[pit.py:73-90](src/data_aggregate/utils/common/pit.py#L73-L90):

```python
shares = fundamentals_to_daily(fundamentals_history, "sharesOutstanding", close.index)
mcap   = close[cols].mul(shares[cols])
```

It multiplies a **fully back-adjusted price** by a **point-in-time (as-filed) share count**.
Those are two different bases, so the product is neither the historical market cap nor a
consistent adjusted-basis one. Measured error, cube mcap ÷ Sharadar's own (correct) `marketcap`:

| year | median error | p05 | min | rows off by >10% |
|---|---|---|---|---|
| 1995 | **0.208** | 0.047 | 0.007 | 189 / 190 |
| 2003 | 0.419 | 0.095 | 0.008 | 1,337 / 1,436 |
| 2013 | 0.722 | 0.271 | 0.023 | 1,424 / 1,749 |
| 2021 | 0.910 | 0.746 | 0.025 | 886 / 1,931 |
| 2026 | 0.997 | 0.980 | 0.517 | 7 / 1,438 |

The error is ~1.0 today and decays monotonically going back in time — the signature of a
cumulative back-adjustment factor. **It decomposes EXACTLY** (median residual 1.0000 in every
year tested) into two multiplicative parts:

```
mcap_error = split_part × dividend_part
  split_part     = sharesOutstanding / sharesbas   (the repo's de-adjustment)
  dividend_part  = prices.close / sharadar.price   (yfinance auto_adjust's dividend leg)
```

| year | median split_part | median dividend_part | median error | residual |
|---|---|---|---|---|
| 1998 | 0.500 | 0.575 | 0.241 | 1.0000 |
| 2003 | 1.000 | 0.618 | 0.419 | 1.0000 |
| 2013 | 1.000 | 0.766 | 0.722 | 1.0000 |
| 2021 | 1.000 | 0.914 | 0.910 | 1.0000 |
| 2026 | 1.000 | 0.997 | 0.997 | 1.0000 |

Both parts are **functions of the future**, so both inject look-ahead into every value feature.
The split part is a clean, strictly-future indicator and it pays:

| group (measured on 48,695 filing rows with a forward 12m price) | n | mean fwd 12m | median |
|---|---|---|---|
| `split_part = 1` (no split after `as_of`) | 37,370 | 15.38% | 11.79% |
| **`split_part < 1` (a split WILL occur after `as_of`)** | **10,812** | **27.81%** | **19.52%** |
| `split_part > 1` (reverse split will occur) | 513 | 9.61% | 5.57% |

`split_part < 1` is *definitionally* "this stock splits later", i.e. strictly future
information, and it is embedded in the mcap denominator of 21.5% of rows. Those rows have mcap
understated by 4× (AAPL pre-2020), 20× (AMZN pre-2022), up to 500× in the tail, so their
`earnings_yield` / `book_yield` / `fcf_yield` are overstated by the same factor — they look like
extreme value stocks, and they earn **+12.4pp/yr**. Any backtest IC or Sharpe on the `value` and
`value_rerating` composites is inflated by this, and it will not survive live.

**The reconciliation is smaller than the diagnosis.** `auto_adjust=False` returns both price series
in one call, and Yahoo's `Close` is split-adjusted-only — the same basis as Sharadar's `price`,
agreeing to the cent. On that basis the future-split factor **cancels exactly** between price and
share count, so market cap becomes correct without needing a split event list at all, and the four
share-change features become correct with no code change. See §Reconciliation.

## Detailed Findings

### 1. The price side: `auto_adjust=True`, hard-coded, split AND dividend adjusted

[fetch_prices.py:155-165](src/data_extract/utils/prices/fetch_prices.py#L155-L165) — the only
equity `yf.download`, with `auto_adjust=True` as a **literal**, not config. yfinance then
replaces OHLC in place with the split- *and* dividend-adjusted series and returns **no
`Adj Close`**. Repo-wide grep for `auto_adjust|adj_close|Adj Close|unadjusted` returns exactly
that one line — nowhere is an unadjusted close or a split factor retained.

`_normalize_prices` ([fetch_prices.py:66-71](src/data_extract/utils/prices/fetch_prices.py#L66-L71))
only lowercases columns and parses `date`. The `prices` table has 7 columns
([sql/schema.sql:20-31](sql/schema.sql#L20-L31), PK `(ticker, date)`) — `date, open, high, low,
close, volume, ticker`. No `adj_close`, no `close_unadj`, no `split_factor`, no basis/vintage
column.

**Proof the dividend leg is the difference, not the split leg.** Sharadar's `price` column is
split-adjusted but NOT dividend-adjusted. Measured on the same dates:

| ticker | date | `prices.close` | `sharadar.price` | ratio | note |
|---|---|---|---|---|---|
| AMZN | 2021-07-30 | 166.38 | 166.38 | **1.0000** | pays no dividend |
| AAPL | 2020-07-31 | 102.80 | 106.26 | 0.9674 | 3.26% of dividends since |

That the non-payer matches to the cent and the payer does not is the clean identification: the
whole `dividend_part` gap is yfinance's dividend back-adjustment.

**Confirmed directly against yfinance 1.5.1 with `auto_adjust=False`**, which returns BOTH series
(`Close`, `Adj Close`, `Dividends`, `Stock Splits`):

| ticker | date | Yahoo `Close` | Sharadar `price` | Yahoo `Adj Close` | stored `prices.close` |
|---|---|---|---|---|---|
| AAPL | 2020-07-31 | **106.26** | **106.26** | 102.795 | 102.80 |
| AAPL | 2019-07-31 | 53.26 | — | 50.916 | 50.92 |
| KO | 2004-02-27 | **24.98** | **24.98** | 12.790 | 12.79 |

So Yahoo's `Close` is **split-adjusted but NOT dividend-adjusted** — the same basis as Sharadar's
`price`, agreeing to the cent on two independent vendors — and `Adj Close` is what the repo
currently stores. The relationship is exactly:

```
Adj Close(d) = Close(d) × D(d)      D(d) = ∏ (1 − div/price) over ex-dates AFTER d
```

`D(d)` → 1.0 at the last row (no future dividends left) and decays going back. Measured on KO:
0.5120 (2004-02-27), 0.6771 (2014-02-27), 0.9868 (2026-02-20), **1.0000 (2026-08-27)**.

⚠ **Neither column is the historical quote.** `Close` is still restated to *today's* share basis:
AAPL 2019-07-31 reads 53.26, but AAPL actually traded at ~213.04 that day (= 53.26 × 4, the
Aug-2020 split). What makes it usable is that Sharadar's `sharesbas` carries the *same* split
restatement — see the cancellation identity in §Reconciliation options.

**Consequence for the L/S sleeve**: `dividend_part(d)` is a monotone function of *future*
dividends, so the value denominators are systematically depressed in proportion to future
dividend policy. Sorted into cross-sectional quintiles by date:

| quintile | n | mean `dividend_part` | mean fwd 12m return |
|---|---|---|---|
| 1 (most depressed) | 12,740 | 0.606 | 13.64% |
| 3 | 9,346 | 0.779 | 17.20% |
| 5 (≈ non-payers) | 7,530 | 0.990 | 27.62% |

Monotone. Read honestly: this spread confounds "is a dividend payer" (a genuine style effect in
this sample) with the bug, so it is not a clean +14pp leak the way the split test is. The
defensible statement is that the bug ties a **40%-average multiplicative distortion of every
value denominator** to dividend policy, which is itself return-relevant — so it contaminates the
value composite with a dividend-policy tilt rather than inflating it cleanly. Note the two
defects push opposite ways (splits inflate value IC, dividends invert it), which is why the
combined error quintiles are U-shaped (Q1 20.6%, Q3 15.5%, Q5 20.3%) and why neither shows up in
an aggregate IC check.

### 2. The shares side: a deliberate, live de-adjustment to point-in-time

`sharesOutstanding` ← Sharadar `sharesbas`, `dimension='ARQ'`, **no unit scaling**
([sharadar_field_map.json:175](configs/sharadar/sharadar_field_map.json#L175)):

```json
"sharesOutstanding": {"kind": "direct", "from": "sharesbas", "split_basis": "count"},
```

Chain: `cast_value_columns` → `apply_zero_rules` (no rule for `sharesbas`) → `apply_corrections`
(no entry) → rename ([field_map.py:617-622](src/data_extract/utils/fundamentals_sharadar/field_map.py#L617-L622))
→ `build_ttm` as `kind: instant`, so period-end, not rolled
([build_ttm.py:240-241](src/data_extract/utils/fundamentals_sharadar/build_ttm.py#L240-L241)) →
**`deadjust_splits`** ([build_ttm.py:254](src/data_extract/utils/fundamentals_sharadar/build_ttm.py#L254))
→ `apply_derived` → `collapse_same_date` → `join_sec_block` (never touches this column).

The de-adjustment
([field_map.py:524-582](src/data_extract/utils/fundamentals_sharadar/field_map.py#L524-L582)):

```python
factor = forward_split_factor(out["ticker"], out["date"], splits)   # ∏ splits strictly AFTER as_of
out.loc[hit, name] = (out.loc[hit, name] / factor[hit] if spec.split_basis == "count"
                      else out.loc[hit, name] * factor[hit])
```

It is **universal, not per-ticker** — a per-column declaration; the factor collapses to 1.0 for
tickers with no forward split. `split_events`
([field_map.py:467-503](src/data_extract/utils/fundamentals_sharadar/field_map.py#L467-L503))
correctly rejects a `split` row co-dated with a `spinoff` (the HON trap). Four columns carry
`split_basis`, and they are the only ones in the whole config:

| merged column | raw | `split_basis` | op |
|---|---|---|---|
| `sharesOutstanding` | `sharesbas` | `count` | ÷ factor |
| `basicShares` | `shareswa` | `count` | ÷ factor |
| `dilutedShares` | `shareswadil` | `count` | ÷ factor |
| `dividendsPerShare` | `dps` | `per_share` | × factor |

`epsDiluted` is derived *after* de-adjustment as `netIncome / dilutedShares`, so it inherits the
as-filed basis. `optionOverhang` is split-invariant (both legs divided by the same factor).
Sharadar's own `eps` / `epsdil` / `marketcap` / `ev` / `sharefactor` are excluded from the merged
table entirely ([sharadar_field_map.json:270-274](configs/sharadar/sharadar_field_map.json#L270-L274)).

`sharesOutstanding` is **never a mix** of Sharadar and SEC: it is `kind: direct`, not in
`field_map.sec_owned`, so the SEC projection at
[merge_history.py:520-524](src/data_extract/utils/fundamentals_sharadar/merge_history.py#L520-L524)
does not even load the column. `fundamentals_history_sec` has its own PIT
`sharesOutstanding` ([sql/schema.sql:181](sql/schema.sql#L181)) which goes unused by the merge.

Measured effect: **10,996 of 51,255 rows (21.5%) across 277 of 489 tickers** carry
`sharesOutstanding < sharesbas`, i.e. were de-adjusted; 601 rows are de-adjusted upward (reverse
splits — GE ×8, C ×10, AIG ×20, TMUS ×2 — which is the *correct* direction, and produces mcap
**overstated** 8-20×, explaining the 23.4 max error).

Worked example, AAPL 2020-07-31 (`sharesbas` 17,102,536,000 ÷ 4.0000 = 4,275,634,000):

```
cube mcap  = 102.80 × 4,275,634,000  =   $439bn
true mcap  = 106.26 × 17,102,536,000 = $1,817bn   (Sharadar marketcap: 1,817,315,475,360)
error      = 0.242  =  0.25 (split) × 0.9674 (dividend)
```

### 3. `sharadar_actions` split coverage is incomplete — so the basis differs ACROSS the cross-section

`sharadar_actions` is fresh (max date 2026-08-25, 510,016 rows, 705 `split` rows post-2020) but
**has holes on major splits**. Verified present vs absent:

| ticker | split | in `sharadar_actions`? |
|---|---|---|
| AAPL 2020-08-31 4:1 | yes | ✅ present |
| AMZN 2022-06-06 20:1 | yes | ✅ present |
| WMT 2024-02-26 3:1 | yes | ✅ present |
| NVDA 2024-06-10 10:1 | yes | ✅ present |
| ANET 2021-11-18 4:1 | yes | ✅ present |
| **GOOGL 2022-07-15 20:1** | yes | ❌ **absent** (only 2014-04-03) |
| **NVDA 2021-07-20 4:1** | yes | ❌ **absent** |
| **TSLA 2022-08-25 3:1** | yes | ❌ **absent** (only 2020) |
| **AVGO 2024-07-15 10:1** | yes | ❌ **absent** |
| **CMG 2024-06-26 50:1** | yes | ❌ **absent** |
| **ANET 2024-12-04 4:1** | yes | ❌ **absent** |
| **MNST 2026 2:1** | yes | ❌ **absent** |

**yfinance has every one of these events** (measured 2026-09-01 via `yf.Ticker(t).splits`), and the
repo already downloads that column on the dividend path and discards it at
[fetch_dividends.py:39](src/data_extract/utils/prices/fetch_dividends.py#L39):

| event | `sharadar_actions` | yfinance |
|---|---|---|
| GOOGL 2022-07-18 ×20 | ❌ | ✅ |
| NVDA 2021-07-20 ×4 | ❌ | ✅ |
| TSLA 2022-08-25 ×3 | ❌ | ✅ |
| AVGO 2024-07-15 ×10 | ❌ | ✅ |
| CMG 2024-06-26 ×50 | ❌ | ✅ |
| ANET 2024-12-04 ×4 | ❌ | ✅ |
| BKNG 2026-04-06 ×25 | ❌ | ✅ |
| MNST 2023-03-28 ×2, 2026-08-11 ×2 | ❌ | ✅ |
| AMCR 2026-01-15 ×0.2 (reverse) | ❌ | ✅ |
| WTW 2016-01-05 ×0.3775 | ✅ | ✅ (both — genuine) |
| SJM 2002-05-30 ×0.945 | ✅ | ❌ (**false positive** — a merger share-issuance factor) |

So the two sources cross-validate: agreement means genuine, and a Sharadar-only non-integer
factor is the false-positive signature.

`deadjust_splits` can only divide by events it can see, so **the same column ends up on
different bases for different tickers**: AAPL is de-adjusted to PIT and wrong against the
adjusted price; GOOGL was never de-adjusted (13,219,426,420 shares on 2022-02-02, the post-split
basis) and is therefore *accidentally consistent* with its adjusted price. **For a
cross-sectional L/S this is the worst case** — the distortion is arbitrary per ticker, so it
corrupts the ranking rather than shifting a level.

Independent validation against the SEC cover page (`fundamentals_history_sec`, 5,553 overlapping
rows / 96 tickers, the only place a PIT truth exists):

- 5,141 agree within ±3%
- 371 rows merged **too high**, 41 too low → **24 of 96 tickers (25%) have a wrong share basis**

| ticker | rows | ratio merged/SEC | reading |
|---|---|---|---|
| AVGO | 25 | 10.0000 | 2024 10:1 missing → not de-adjusted |
| ANET | 42 | 4.0000 | 2024 4:1 missing; 2021 applied → **partially** de-adjusted (vendor 16× → merged 4×) |
| CMG | 56 | 49.70 – 50,000 | 2024 50:1 missing, plus a compounding tail |
| APH | 60 | 2.0000 | one event unapplied |
| AMCR | 26 | 0.2000 | **correct** — a real 1:5 reverse split 2026-01-15 (confirmed in yfinance) |
| CCL | 67 | 0.0012 – 1.4032 | over-de-adjusted, badly |
| **BKNG** | 63 | **25.0000** | ⚠ **CORRECTED 2026-09-01**: a real **25:1 split on 2026-04-06** (in yfinance, absent from `sharadar_actions`). `sharesbas` 1,024,044,900 = SEC 40,961,796 × 25 is the vendor's *correct* adjusted basis. Not a vendor defect — just another missing split event. |
| WTW / SJM | 53 / 27 | 2.6490 / 1.0582 | non-integer, constant — merger/exchange ratios applied as share factors |

ANET is the important shape: a *partial* de-adjustment is worse than none, because the residual
factor is neither the PIT count nor the vendor's adjusted count.

### 4. `prices` has interleaved adjustment vintages, because nothing ever re-pulls history

`resume_since` ([incremental.py:80-89](src/data_extract/utils/common/incremental.py#L80-L89))
returns `max(min(per-ticker MAX(date)), today − years_history)` — one shared start date — and the
write is a plain upsert on `(ticker, date)`
([fetch_prices.py:230-231](src/data_extract/utils/prices/fetch_prices.py#L230-L231)). There is
**no `full` / `force` / `rebuild` flag** on `fetch_price_history`
([fetch_prices.py:204-210](src/data_extract/utils/prices/fetch_prices.py#L204-L210)) and none on
the CLI command ([cli.py:99-107](src/data_extract/cli.py#L99-L107) exposes only `-c` and `-t`,
while eight sibling commands do get `-F/--full`). Nothing in `src/` ever deletes or replaces
`prices` (only `fetch_macro.py:233` replaces `prices_macro`).

So **rows keep whatever adjustment basis was in force the day they were first written**, and a
split after that day leaves the old rows stale. This is live today on MNST, which split ~2026-07-20:

```
2026-07-17   97.50      <- old basis
2026-07-20   47.72      <- new basis
2026-07-23   93.56      <- old basis again
2026-07-31   48.19
2026-08-03   93.55
2026-08-06   47.08
2026-08-07   90.36
2026-08-11   45.53      <- new basis from here
```

Day-to-day returns of ±95% that never happened. `daily_returns` is
`close.pct_change(fill_method=None)` with no jump guard
([data_utils.py:51-53](src/data_aggregate/utils/common/data_utils.py#L51-L53)), so this flows
straight into momentum, volatility, betas and the **labels**.

Scope today is narrow: a spike-and-revert scan (|move| >55% reversing next day) over the whole
table finds **3 days on 1 ticker in 2026 (MNST)**, 1 in 2001, 1 in 1998 — every other large move
checked is genuine (2020-03-09 oil crash: APA/OXY/FANG/TRGP; PCG bankruptcy; CVNA 2022). So the
history is currently clean of split discontinuities, but the mechanism is systemic and will
corrupt every future splitter until a full re-download.

### 5. Blast radius: what depends on the broken denominator

Every feature below goes through `daily_market_cap` or a raw `sharesOutstanding` pivot.

| consumer | features | formula |
|---|---|---|
| [fundamental_features.py:844-849](src/data_aggregate/utils/fundamentals/fundamental_features.py#L844-L849) | `earnings_yield`, `sales_yield`, `book_yield`, `fcf_yield` | `X / mcap` |
| [fundamental_features.py:869-898](src/data_aggregate/utils/fundamentals/fundamental_features.py#L869-L898) | `ev`, `ebitda_to_ev`, `fcf_to_ev` | `ev = close × dilutedShares + debt … − liquid` |
| [fundamental_features.py:913-948](src/data_aggregate/utils/fundamentals/fundamental_features.py#L913-L948) | `altman_z` (0.6 term), `pegy` | `mcap / totalLiabilities`; `mcap / netIncome` |
| [fundamental_features.py:571, 718, 822-829, 966-993](src/data_aggregate/utils/fundamentals/fundamental_features.py#L571) | `core_earnings_yield`, `aro_to_mcap`, `pbo_to_mcap`, `pension_underfunding_to_mcap`, `pension_overhang_leverage`, `ffo_yield`, `implied_cap_rate`, `ebitdax_to_ev` | `X / mcap` or `X / ev` |
| [intrinsic.py:87-97](src/data_aggregate/utils/fundamentals/intrinsic.py#L87-L97) | `intrinsic_yield` | `total / mcap` |
| [dividend_features.py:84-128](src/data_aggregate/utils/fundamentals/dividend_features.py#L84-L128) | `dividend_yield`, `dividend_payout_ratio`, `dividend_coverage`, **`shareholder_yield`** | `buyback_yield = −(shares / shares.shift(252) − 1)` |
| [target/factors.py:81-127](src/data_aggregate/utils/target/factors.py#L81-L127) | `size` and `value` **factor returns** | `size = −log(mcap)`; `value = mean(xs_z(NI/mcap), xs_z(FCF/mcap), xs_z(BV/mcap))` |
| [target/targets.py:207-245](src/data_aggregate/utils/target/targets.py#L207-L245) | **every stored label** — `rank`, `zscore`, `epsilon` at h=30/60/90 | `log(mcap)` is a neutralising exposure; `neutralize_log_mcap: true` at [build_cube.yml:37-47](configs/build_cube.yml#L37-L47) |
| [insider_features.py:104](src/data_aggregate/utils/extras/insider_features.py#L104), [institutional_features.py:145-163](src/data_aggregate/utils/extras/institutional_features.py#L145-L163), [superinvestor_features.py:188-198](src/data_aggregate/utils/extras/superinvestor_features.py#L188-L198) | `insider_net_buy_to_mcap`, `inst_ownership_pct`, `inst_value_to_mcap`, `inst_flow_to_mcap`, `super_value_to_mcap`, `super_flow_to_mcap` | `X / mcap`, `inst_shares / shares` |

Share-count **change** features (four, all reading the de-adjusted counts):
`shareholder_yield`'s buyback leg ([dividend_features.py:126-128](src/data_aggregate/utils/fundamentals/dividend_features.py#L126-L128), 252-**trading-day** shift),
`shares_growth` ([fundamental_features.py:1027-1032](src/data_aggregate/utils/fundamentals/fundamental_features.py#L1027-L1032)),
`diluted_shares_growth` ([:1375-1378](src/data_aggregate/utils/fundamentals/fundamental_features.py#L1375-L1378)),
and one of nine `piotroski_f_score` components ([:1136](src/data_aggregate/utils/fundamentals/fundamental_features.py#L1136)).
**These are the features a wrong split basis destroys outright**: a 4:1 split reads as +300%
issuance / −300% buyback yield on the quarter the de-adjustment factor steps.

Composite groups affected ([build_cube.yml:91-251](configs/build_cube.yml#L91-L251)):
`value` **10/10 members**, `value_rerating` **8/8**, plus `distress`, `pension_risk`,
`shareholder_return` (7 members), `quality` (1 of 9 Piotroski legs), `reit_health`,
`energy_health`, `insider`, `superinvestor`, `institutional`.

### 6. What is *correct* today

- **Returns and the price panel are fine.** Back-adjustment is exactly right for returns — that
  is what it is for. `cube_part_prices` round-trips `prices.close` bit-identically
  ([test_price_part_roundtrip.py:90](tests/data_aggregate/test_price_part_roundtrip.py#L90)).
  The bug is confined to mixing that price with a *count*.
- **Sharadar's own `marketcap`, `price`, `ev` are internally consistent and correct** (AAPL
  2020-07-31: 1,817,315,475,360 = 106.26 × 17,102,536,000). They are excluded from the merged
  table by design.
- `eps_yield`, `fwd_eps_yield`, `forward_earnings_yield`, `dps_growth`, `option_overhang`,
  `inst_ownership_pct` are ratios of two same-basis quantities or per-share/price — they do not
  take the mcap error (though `eps_yield` and `dps_growth` still take the *shares* error via
  `epsDiluted` / `dividendsPerShare`).
- `deadjust_splits`'s spinoff exclusion works as designed; the ordering fix (TTM aggregate
  first, then de-adjust) is documented at [field_map.py:546-557](src/data_extract/utils/fundamentals_sharadar/field_map.py#L546-L557).

## Reconciliation

The invariant to restore: **every quantity that gets multiplied or divided by another must sit on
the SAME adjustment basis.** The naive reading of that ("use a basis that doesn't depend on the
future") turns out to be the wrong requirement — see the cancellation identity below.

### The cancellation identity

Let `F(d)` = product of splits after `d`. The two vendors publish:

```
Sharadar sharesbas(d) = real_shares(d) × F(d)
Yahoo    Close(d)     = raw_price(d)   / F(d)          (auto_adjust=False, verified §1)
```

so their product is

```
Close(d) × sharesbas(d) = raw_price(d) × real_shares(d)   ← the true historical market cap
```

**`F(d)` cancels identically.** Each leg individually depends on future splits; the product does
not. Verified: AAPL 2020-07-31, `106.26 × 17,102,536,000 = 1.8173e12` vs Sharadar's published
`1,817,315,475,360`.

Three consequences, and they reframe the whole fix:

1. **Market cap needs no split event list at all.** The nine missing events in §3 stop mattering
   on the critical path. The current design makes correctness *depend* on that list being
   complete, which it never will be.
2. **Share-change features become correct for free.** On the PIT basis a 4:1 split reads as +300%
   issuance; on the split-adjusted basis both legs carry the same restatement, so
   `shares_adj(t)/shares_adj(t−1)` = 1 through a split. That fixes `shares_growth`,
   `diluted_shares_growth`, `buyback_yield`/`shareholder_yield` and the Piotroski dilution leg
   with **no code change** — they are currently wrong in the opposite direction.
3. **The gate is free.** Sharadar publishes `marketcap` on exactly this basis, so
   `Close × sharesOutstanding == sharadar.marketcap` is checkable on all 51,255 rows.

The point-in-time count is still worth producing, but as a *separate* column consumed only where a
real share count is genuinely required (13F / insider ownership percentages) — never as a feature
denominator.

### What this means concretely

`auto_adjust=False` returns both series in one call, so the enabling change is a flag flip:

| column | Yahoo source | basis | used for |
|---|---|---|---|
| split-adjusted close | `Close` | split-adj, **not** div-adj | **levels** — market cap, EV, dividend yield, anything × a share count |
| total-return close | `Adj Close` | split + div adj | **returns only** — momentum, vol, betas, labels |

Using the wrong one is a real error in either direction: the total-return series understates KO's
2004 market cap by ~49%, and the split-adjusted series gives KO's 2004→2026 return as 3.20× instead
of 6.16×, losing half the return and systematically penalising high-yield names in momentum.
Today's `prices.close` is the total-return series, used for **both** jobs — that is the bug.

Naming is an open decision (see Open Questions 1): the bare `close` matches both vendors'
convention for the split-adjusted series, but redefining it silently changes meaning for existing
readers.

### Prerequisites

1. **Add a `--full` path for prices (§4)** — no way to re-pull history exists today, MNST is
   already corrupted, and split adjustment is retroactive so every future splitter repeats it.
   A one-off full re-download of 3.26M rows is required either way.
2. **Complete the split event list** — needed now only for the PIT column and for validation, not
   for market cap. Source: yfinance's `Stock Splits`, already downloaded and discarded
   ([fetch_dividends.py:39](src/data_extract/utils/prices/fetch_dividends.py#L39)), which covers
   every hole (§3). Cross-validate the two sources: agreement ⇒ genuine; a Sharadar-only
   non-integer factor is the false-positive signature (SJM 0.945). CCL's 0.0012 needs one look.
3. ~~**Settle BKNG.**~~ **RESOLVED 2026-09-01** — a real 25:1 split on 2026-04-06, missing from
   `sharadar_actions` but present in yfinance. Not a vendor defect.
4. **A cross-source invariant test**, which is what would have caught all of this: assert
   `|Close × sharesOutstanding ÷ sharadar.marketcap − 1| < tol` per (ticker, date), plus
   `|Close(d) ÷ sharadar.price(d) − 1| < tol` on filing dates to catch a stale price vintage the
   day it appears. `src/validate/` has **no price validator** and no test anywhere asserts
   adjustment semantics.

**Full implementation plan**: `reports/planning/active-tasks/2026-09-01-price-shares-basis-fix/PLAN.md`.

### Note on the labels

`neutralize_log_mcap: true` ([build_cube.yml:37-47](configs/build_cube.yml#L37-L47)) puts
`log(mcap)` into the projection design for **every stored label** at all three horizons. Fixing
mcap therefore changes `rank`, `zscore` and `epsilon` themselves — so every trained model and
every recorded IC/Sharpe becomes non-comparable to post-fix numbers. Worth an explicit decision
before the rebuild, not after.

## Code References

- [pit.py:73-90](src/data_aggregate/utils/common/pit.py#L73-L90) — `daily_market_cap`, the single formula
- [pit.py:52-70](src/data_aggregate/utils/common/pit.py#L52-L70) — `fundamentals_to_daily`, the PIT pivot
- [fetch_prices.py:161](src/data_extract/utils/prices/fetch_prices.py#L161) — `auto_adjust=True`, the only occurrence in the repo
- [incremental.py:80-89](src/data_extract/utils/common/incremental.py#L80-L89) — `resume_since`, the clamp that pins history
- [field_map.py:524-582](src/data_extract/utils/fundamentals_sharadar/field_map.py#L524-L582) — `deadjust_splits`
- [field_map.py:467-503](src/data_extract/utils/fundamentals_sharadar/field_map.py#L467-L503) — `split_events`, spinoff exclusion
- [field_map.py:506-522](src/data_extract/utils/fundamentals_sharadar/field_map.py#L506-L522) — `forward_split_factor`
- [build_ttm.py:254-255](src/data_extract/utils/fundamentals_sharadar/build_ttm.py#L254-L255) — where de-adjustment is applied
- [merge_history.py:511-524](src/data_extract/utils/fundamentals_sharadar/merge_history.py#L511-L524) — actions load, SEC projection
- [sharadar_field_map.json:173-176](configs/sharadar/sharadar_field_map.json#L173-L176) — the four `split_basis` entries
- [sharadar_field_map.json:73-107](configs/sharadar/sharadar_field_map.json#L73-L107) — the `_SPLIT_ADJUSTMENT` rationale + HON trap
- [sql/schema.sql:20-31](sql/schema.sql#L20-L31) — `prices`, 7 columns, no adjusted/unadjusted pair
- [build_cube.yml:37-47](configs/build_cube.yml#L37-L47) — `neutralize_log_mcap: true`

## Key Data Flows

```
yfinance auto_adjust=True ──> prices(close)  [split+div adjusted, today's basis]
                                   │
                                   ├──> cube_part_prices ──> daily_returns  ✅ correct
                                   │
Sharadar sharesbas [today's split basis, correct]                │
   └─ deadjust_splits (÷ ∏ future splits, from an INCOMPLETE event list)
        └──> fundamentals_history.sharesOutstanding  [PIT-ish, ticker-dependent basis]
                                   │                 │
                                   ▼                 ▼
                        daily_market_cap = close_adj × shares_PIT   ❌ MIXED BASIS
                                   │
      ┌────────────────────────────┼──────────────────────────────┐
      ▼                            ▼                              ▼
 value / value_rerating      size & value factor            log_mcap neutraliser
 (18 composite members)      returns (factors.py)           → rank / zscore / epsilon
                                                              at h = 30/60/90
```

Target flow after reconciliation (one fetch, two columns, three routes):

```
yfinance auto_adjust=False
   ├─ Close      [split-adj]      ──┬──> × sharesbas ──> marketCap  ✅ F(d) cancels
   │                                ├──> × dilutedShares ──> ev
   │                                └──> ÷ dividendsPerShare ──> dividend_yield
   ├─ Adj Close  [split+div adj]  ─────> daily_returns, momentum, vol, betas, LABELS
   └─ Stock Splits ──> prices_splits ──┐
                                       ├─> union with sharadar_actions
Sharadar sharesbas [split-adj]  ───────┴─> sharesOutstandingPit ──> ownership % only
   └──> sharesOutstanding (basis UNCHANGED from vendor, no de-adjustment)
```

## Dependencies

`yfinance` (`auto_adjust`, `actions`), Sharadar SF1 + `actions` endpoints, pandas/numpy.
Internal: `src/data_store/store.py` (upsert on `(ticker, date)`), `src/data_extract/utils/common/incremental.py`.

## Test Coverage

- `tests/data_extract/prices/test_price_prelisting_trim.py` — volume-prefix trimming only
- `tests/data_extract/common/test_resume_since.py` — resume-window contract, uses `"prices"`
- `tests/data_aggregate/test_price_part_roundtrip.py:90` — `cube_part_prices` bit-identical round-trip
- `tests/data_aggregate/test_pit_cache.py` — `PitFrames` memoisation changes no number
- **Zero** occurrences of `auto_adjust` / `adj_close` / `unadjusted` / a split-ratio assertion in `tests/`
- **No price validator** in `src/validate/` ([cli.py:17](src/validate/cli.py#L17) calls a prices validator explicitly future work)

## Related Documentation

- `reports/planning/active-tasks/2026-08-26-sharadar-integration/phase-2-findings.md`
- `reports/research/financial-data/2026-08-26-sharadar-fundamentals.md`
- Memories: `sharadar-db-measured-2026-08-26`, `sharadar-sf1-facts`, `multi-class-share-counts`

## Open Questions for Planning Phase

1. **Column naming** — rename to an explicit pair and drop the bare `close`, so any un-migrated
   reader raises `KeyError` rather than silently changing meaning? Or keep `close` as the
   total-return series and add the split-adjusted one beside it (no reader changes, misleading
   name survives)? The bare `close` matches both vendors' convention for the *split-adjusted*
   series, which argues for redefining it — but only if every reader is migrated in the same
   change. Known readers: [step_cube_prices.py:41,58](src/data_aggregate/transformers/step_cube_prices.py#L41),
   [ls_model.py:94-99](src/strategies/utils/ls_model.py#L94), `scripts/diagnose_missing_dates.py`,
   `tests/conftest.py:227-269`.
2. **Reversing the PIT de-adjustment for the feature columns** — approved on 2026-08-26 as a
   correction. The cancellation identity says PIT is the *wrong* basis for any feature that
   multiplies or divides by a price, and it gets share-change features wrong outright. Move it to
   a separate `sharesOutstandingPit` column for ownership percentages?
3. ~~Which basis~~ — **settled** by the cancellation identity: split-adjusted for levels,
   total-return for returns, PIT only for ownership ratios.
4. ~~Where do the missing split events come from~~ — **settled**: yfinance's `Stock Splits`, already
   downloaded and discarded, covers every hole. Open sub-question: is the union rule
   "agreement ⇒ genuine" strict enough, given SJM's Sharadar-only 0.945?
5. **Does the `prices` full re-download happen now**, and is the split-triggered targeted re-pull
   part of this change or a follow-up?
6. **Label rebuild** — `neutralize_log_mcap` means fixing mcap changes every stored label. Rebuild
   all horizons and re-baseline (invalidating every recorded IC/Sharpe), freeze the labels and fix
   features only, or write new labels alongside the old for one A/B cycle?
7. ~~BKNG's 25× vendor share count~~ — **resolved**, a real missing split. Remaining: SJM's 0.945
   merger factor and CCL's 0.0012, which have no yfinance counterpart.
8. **Which invariant test** becomes the permanent gate, and does it live in `src/validate/` as the
   first prices validator?
