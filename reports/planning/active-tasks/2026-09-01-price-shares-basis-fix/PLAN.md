# Plan: one consistent basis for price, shares and marketCap

**Date**: 2026-09-01
**Phase**: 2 of 3 (FIC) — follows `reports/research/codebase/2026-09-01-marketcap-price-shares-basis.md`
**Status**: **IMPLEMENTED 2026-09-01.** All six phases shipped; see §OUTCOME.
**Next**: the user's manual cube + label rebuild — see §After the fix.

This file is the INDEX. Each phase is a sibling file, independently implementable and
independently verifiable.

| phase | file | goal | depends on |
|---|---|---|---|
| **P0** ✅ | [PHASE-0-baseline.md](PHASE-0-baseline.md) | Record the "before" numbers as a rerunnable script | — |
| **P1** ✅ | [PHASE-1-prices-extract.md](PHASE-1-prices-extract.md) | Two price columns, `--full`, re-download 3.26M rows | P0 |
| **P2** ✅ | [PHASE-2-split-events.md](PHASE-2-split-events.md) | `prices_splits` from yfinance + corroborated union | P0 |
| **P3** ✅ | [PHASE-3-shares-basis.md](PHASE-3-shares-basis.md) | Stop de-adjusting features; add `sharesOutstandingPit` | P2 |
| **P4** ✅ | [PHASE-4-consumers.md](PHASE-4-consumers.md) | Cube schema, rename migration, label routing | P1, P3 |
| **P5** ✅ | [PHASE-5-validator.md](PHASE-5-validator.md) | Prices validator + tests + DoD gate | P4 |

P1 and P2 are independent (different tables, both extract-layer) and should run concurrently —
P1 is the longest wall-clock item.

## Scope boundary

**This task delivers clean price and target CODE, plus a refilled `prices` table (P1).**

The cube rebuild, the label rebuild and the model re-baseline are **the user's, run manually
after the fix lands.** Nothing in P0-P5 rebuilds the cube or retrains anything. See
[§After the fix](#after-the-fix-the-users-manual-rebuild) for the checklist to run then.

### The cube is EMPTY today

There is no cube data and no recorded model metrics. Three consequences, and two of them help:

1. **Nothing in P0-P5 can read the cube, and nothing needs to.** Every measurement and every
   verification below runs against `prices`, `fundamentals_history`, `fundamentals_sharadar`,
   `fundamentals_history_sec`, `sharadar_actions` and `prices_macro`. Where the research says
   "cube mcap" it means *the `daily_market_cap` formula's output*, which P0 recomputes in-script.
2. ✅ **The non-comparability problem disappears.** D6 originally worried that fixing mcap would
   invalidate every recorded IC and Sharpe. There are none to invalidate. The user's first
   rebuild after the fix becomes the baseline, measured on a correct basis from the start.
3. ⚠ **There is no "before" for anything cube-shaped**, so no verification step may be phrased as
   "compare the label / feature to its pre-change value". Every check in P4 and P5 is either an
   absolute value with a known-correct target (AAPL mcap == 1.817e12), an internal identity
   (AMZN `close_split == close_total`), or a cross-vendor agreement.

---

## The governing principle

> **Every quantity that gets multiplied or divided by another must sit on the SAME
> adjustment basis.** For any product involving a share count that basis is the
> SPLIT-ADJUSTED one, because on it the future-split factor cancels exactly.

### The three bases, named

| name | Yahoo source | definition | used for |
|---|---|---|---|
| `close_split` | `Close` (`auto_adjust=False`) | the quote restated **for splits only** — a 4:1 split divides every pre-split day by 4. **No dividend adjustment.** | **levels**: market cap, EV, dividend yield, ATR |
| `close_total` | `Adj Close` | `close_split` further reduced for every dividend paid **after** that date, so the path is buy-and-hold-with-reinvestment | **returns**: `ret`, momentum, vol, betas, **labels** |
| `sharesOutstandingPit` | Sharadar `sharesbas` ÷ F(d) | the count that really existed on the day | **ownership ratios only** (13F, insider) |

Relationship: `close_total(d) = close_split(d) × D(d)`, where `D(d) = ∏(1 − div/price)` over
ex-dates after `d`. KO 2004-02-27: `close_split` 24.98, `close_total` 12.79, `D` = 0.512.

**Verified 2026-09-01 against yfinance 1.5.1**: `Close` is split-adjusted only and matches
Sharadar's `price` to the cent on two independent vendors (AAPL 2020-07-31 both 106.26;
KO 2004-02-27 both 24.98). `Adj Close` is what the repo stores today.

### The cancellation identity — why `close_split`, and why the split list leaves the mcap path

Let `F(d)` = product of splits after `d`.

```
Sharadar sharesbas(d)   = real_shares(d) x F(d)      (vendor back-fills to today's basis)
Yahoo    close_split(d) = raw_price(d)   / F(d)      (vendor restates to today's basis)
--------------------------------------------------------------------------------------
close_split x sharesbas = raw_price x real_shares    <- F(d) cancels IDENTICALLY   OK
close_total x sharesbas = true_mcap x D(d)           <- D(d) does NOT cancel      WRONG
```

Nothing in a share count carries a dividend factor, so `D(d)` survives into the product. That
`D(d)` is the research's `dividend_part`: median **0.618** in 2003 — market cap 38% too low.

**Today** the repo divides `sharesbas` by `F(d)` (`deadjust_splits`), which makes market cap a
*function of the `sharadar_actions` event list*. That list has 9 known holes, so AAPL is
de-adjusted and GOOGL is not — **the same column sits on different bases for different tickers
in the same cross-section**, which is the worst possible failure for a cross-sectional L/S.

**After** we delete that division. `sharesOutstanding = sharesbas` verbatim, `F(d)` cancels
against `close_split`, and **the split event list leaves the market-cap path entirely.** That is
the main reason this design beats patching the list — correctness stops depending on a feed that
will never be complete.

Verified numerically: AAPL 2020-07-31, `106.26 x 17,102,536,000 = 1.8173e12`, against Sharadar's
published `1,817,315,475,360`.

### What this fixes for free

- **Share-change features become correct with no code change.** On the PIT basis a 4:1 split
  reads as +300% issuance; on the split-adjusted basis both legs carry the same restatement, so
  `shares(t)/shares(t-1)` = 1 through a split. Fixes `shares_growth`, `diluted_shares_growth`,
  `buyback_yield`/`shareholder_yield` and the Piotroski dilution leg.
- **`dividend_yield` starts cancelling.** `dividendsPerShare / close_split` — both split-adjusted.
- **A free external gate.** Sharadar publishes `marketcap` on exactly this basis, so
  `close_split x sharesOutstanding == sharadar.marketcap` is checkable on all 51,255 rows.

### Where market cap is computed — unchanged

`marketCap` stays **derived in the cube** by `daily_market_cap`
([pit.py:73-90](src/data_aggregate/utils/common/pit.py#L73-L90)), consumed from
[step_cube_target.py:256](src/data_aggregate/transformers/step_cube_target.py#L256) and
`PitFrames`. Sharadar's own `marketcap` column stays **excluded** from the merged table
([sharadar_field_map.json:270-274](configs/sharadar/sharadar_field_map.json#L270-L274)) and is
read in exactly one place: the P5 validator. **It never becomes a feature.**

---

## Decisions (settled 2026-09-01)

| # | decision | rationale |
|---|---|---|
| **D1** | **Store BOTH price columns**, not one + reconstruction | `auto_adjust=False` returns `Close` and `Adj Close` in one response, written in one upsert, so they cannot drift. Reconstructing total return from `close + dividends` measured +1.29% cumulative error on MO (2007 Kraft / 2008 PMI spinoffs, 173bp on a single day) and would need a dividend panel threaded through 5 `daily_returns` call sites plus the fingerprint harness. |
| **D2** | **Rename both columns**; the bare `close` disappears | `close` changing meaning silently is exactly the bug class this task exists to remove. A missed reader must raise `KeyError`, not quietly compute price returns (MO reads 1.24x over the sample where the truth is 20.2x). |
| **D3** | **`sharesOutstandingPit` only** — no PIT twin for the other three | The only PIT consumers are `inst_ownership_pct` and the insider %-of-shares leg (13F / Form-4 shares are as-filed real counts). `basicShares`/`dilutedShares` feed per-share ratios that cancel; a PIT `dividendsPerShare` has no consumer. |
| **D4** | **`prices_splits` from yfinance, unioned with corroboration** | yfinance covers all 9 `sharadar_actions` holes and carries none of its false positives. Union rule in P2. Feeds `sharesOutstandingPit`, the split-triggered re-pull and the validator — **not** market cap. |
| **D5** | **`--full` path + one-off re-download + split-triggered re-pull** | Split adjustment is retroactive; without the trigger every future splitter re-corrupts the table. MNST is corrupted *today*. |
| **D6** | **The label/cube rebuild is OUT of this task** — the user runs it manually afterwards | The code must be correct and the price table refilled first; rebuilding is a long compute step with no code risk. The cube is empty today, so there is nothing to invalidate and no comparability cost — the user's first rebuild is simply the baseline. |
| **D7** | **Prices validator in `src/validate/`, wired into the DoD gate** | First sibling to the fundamentals validator, as [cli.py:17](src/validate/cli.py#L17) already anticipates. |

### Changed from V1

1. **V1 stored one column and reconstructed total return. V2 stores both** (D1) — V1's own
   measurement showed the reconstruction fails on spinoff names.
2. **V1 missed that `download_ohlcv` is shared with the macro leg.**
   [fetch_macro.py:109](src/data_extract/utils/prices/fetch_macro.py#L109) calls it for `SPY`
   (series name literally `equity_tr`), `XLE`, `^VIX`, `CL=F`, `GC=F`. A bare `auto_adjust=False`
   flip silently converts the market benchmark and the energy factor from total to price return,
   corrupting `beta_market`, `fwd_market` in every label, and the L/S benchmark leg. P1
   parameterises the flag and pins macro to the total-return column.
3. **V1 understated the label hazard.** [targets.py:81](src/data_aggregate/utils/target/targets.py#L81)
   is `forward_return(close, horizon)` = `close.shift(-h)/close - 1`, a literal price ratio. P4
   replaces it with `forward_compound(ret, h)`.
4. **New in V2: the ATR basis trap.** [features.py:67-73](src/data_aggregate/utils/momentum/features.py#L67-L73)
   computes `tr2 = (high - prev_close).abs()`. `high`/`low` come back split-adjusted-only under
   `auto_adjust=False`, so pairing them with `close_total` would be a *new* mixed-basis bug. ATR
   must take `close_split`.
5. **New in V2: `prices_dividends` has no reader.** Declared at
   [schema.py:106](src/data_store/schema.py#L106), consumed nowhere in `src/`. Under D1 it stays
   unread — noted, not fixed here.

---

## Blast radius (research §5)

Corrected by P1+P3 with **no code change**, purely by inheriting a correct `mcap` / `ev`:

`earnings_yield`, `sales_yield`, `book_yield`, `fcf_yield`, `ev`, `ebitda_to_ev`, `fcf_to_ev`,
`altman_z`, `pegy`, `core_earnings_yield`, `aro_to_mcap`, `pbo_to_mcap`,
`pension_underfunding_to_mcap`, `pension_overhang_leverage`, `ffo_yield`, `implied_cap_rate`,
`ebitdax_to_ev`, `intrinsic_yield`, `dividend_yield`, `dividend_payout_ratio`,
`dividend_coverage`, `shareholder_yield`, `shares_growth`, `diluted_shares_growth`, the
Piotroski dilution leg, `insider_net_buy_to_mcap`, `inst_value_to_mcap`, `inst_flow_to_mcap`,
`super_value_to_mcap`, `super_flow_to_mcap`, the `size` and `value` factor returns, and — via
`neutralize_log_mcap` — `rank` / `zscore` / `epsilon` at h=30/60/90.

Composite groups: `value` **10/10 members**, `value_rerating` **8/8**, plus `distress`,
`pension_risk`, `shareholder_return`, `quality` (1 of 9 legs), `reit_health`, `energy_health`,
`insider`, `superinvestor`, `institutional`.

Requiring `sharesOutstandingPit` instead (P4): `inst_ownership_pct`
([institutional_features.py:145](src/data_aggregate/utils/extras/institutional_features.py#L145))
and the insider %-of-shares leg
([insider_features.py:104](src/data_aggregate/utils/extras/insider_features.py#L104)).

## Out of scope

- **The cube rebuild, the label rebuild and the model re-baseline** (D6) — the user's, run
  manually after P5. Checklist below.
- Reconstructing total return from dividends — superseded by D1.
- Giving `prices_dividends` a consumer.
- Survivorship bias (`sharadar_sp500` ingested but unused).
- Multi-class share-count reconciliation — the vendor basis sidesteps it for mcap.
- An `open`/`high`/`low` retention review.

Only **one** table is refilled by this task: `prices` (P1). `fundamentals_history` is rebuilt in
P3 because the shares-basis change is not observable without it and P4/P5 verify against it — but
that is an extract-layer rebuild, not a cube one.

## Risks

| risk | mitigation |
|---|---|
| A missed `close` reader silently computes price returns | D2's rename forces `KeyError`; P5 invariant 1 is the backstop |
| The macro leg flips basis with the equity leg | P1 parameterises `auto_adjust`; success criterion asserts `prices_macro` is bit-identical |
| The 3.26M-row re-download fails partway, leaving mixed vintages | Write to a staging table, swap on success (P1) |
| Sharadar's `marketcap` is itself wrong for some tickers | Invariant 1 tolerance is 1%, and failures are *reported*, not auto-corrected |
| Re-baselining loses the ability to compare to past experiments | P0 archives the pre-fix metrics BEFORE any rebuild; the cube rebuild is the user's and comes after |

## Success criteria (this task, P0-P5)

All measurable **without** a cube rebuild — they read `prices`, `fundamentals_history` and
Sharadar directly.

- [ ] `|close_split x sharesOutstanding / sharadar.marketcap - 1| < 1%` on >99% of 51,255 rows
      (today the median row is off by 15%, and 1995 rows by 5x)
- [ ] `sharesOutstanding == sharesbas` on 100% of rows
- [ ] `sharesOutstandingPit` within ±3% of `fundamentals_history_sec.sharesOutstanding` on the
      96-ticker overlap (today 24 of 96 fail)
- [ ] MNST Jul–Aug 2026 shows no 2x alternation; spike-and-revert count post-2020 is 0
- [ ] No `KeyError: 'close'` anywhere — every reader migrated, full test suite green
- [ ] `prices_macro.equity_tr` is bit-identical before and after — macro must NOT change
- [ ] The three P5 invariants pass, and each has been shown to FIRE on a deliberately
      corrupted ticker

---

## After the fix: the user's manual rebuild

Not part of this task. Recorded here so the handover is complete.

**What to run**: full cube build from empty — `build-prices` → `build-target` → `build-betas` →
the feature parts → merge. Nothing to archive first and no incremental-vs-full question to
settle, because there is nothing there.

**This build IS the baseline.** No pre-fix IC or Sharpe exists, so there is no non-comparability
to manage and no temptation to read a drop as a regression. Record it as run #1 on the corrected
basis and move forward from there.

**The one check that proves the fix reached the labels**, and the reason P0 must run before P1
touches `prices`: recompute the `split_part` cohort split on the rebuilt cube. The research
measured rows where the stock splits *after* the observation date — strictly future information
sitting in the mcap denominator — earning **27.81% forward 12m against 15.38%** for everything
else, +12.4pp/yr on 21.5% of rows. On the corrected basis those two groups must become
**indistinguishable**. If the gap survives, something in P4 is still routing the old basis.

That check is only possible because P0 froze the pre-fix cohort numbers from the extract layer
while the old `prices` still existed. It is the single most valuable thing P0 produces.

Two secondary reads, both against P0's frozen numbers: the `dividend_part` quintile monotonicity
(Q1 13.64% → Q5 27.62%) should collapse to whatever the genuine dividend-payer style effect is —
it will not go to zero, because that spread confounds a real style effect with the bug — and the
combined-error U-shape (Q1 20.6%, Q3 15.5%, Q5 20.3%) should flatten.

When you do have two runs to compare, compare **per composite**, not in aggregate: an aggregate
IC hid both defects in the first place, because they push opposite ways.

---

## OUTCOME (implemented 2026-09-01)

**All six phases shipped.** `prices` is refilled on the two-column basis, `fundamentals_history`
is rebuilt with `sharesOutstanding == sharesbas` on 100% of rows, every consumer is routed, and
`validate prices` gates the cube build. The cube rebuild remains the user's, as scoped (D6).

### What the fix actually bought, measured

`scripts/basis_baseline.py` before vs after (`baseline.json` vs `baseline-after-p3.json`), on
the same 51,255 filing rows:

| year | median mcap error BEFORE | AFTER | rows off >10% BEFORE | AFTER |
|---|---|---|---|---|
| 1995 | 0.2081 | **0.9999** | 189 / 190 | **35** |
| 2003 | 0.4186 | **1.0000** | 1,337 | **165** |
| 2013 | 0.7217 | **1.0000** | 1,425 | **148** |
| 2021 | 0.9105 | **1.0000** | 886 | **65** |
| 2026 | 0.9966 | **1.0000** | 7 | 7 |

Both leaks are gone from the decomposition: `split_part` and `dividend_part` are now **1.0000
in every year** (they were 0.500 / 0.575 in 1998 and 1.000 / 0.618 in 2003). De-adjusted rows
went **11,597 -> 0**; `sharesOutstanding == sharesbas` on **51,255 of 51,255** rows.

Both control digests are **byte-identical** to P0, which is what proves the change touched only
what it should:

| digest | P0 | after P3 |
|---|---|---|
| `macro_equity_tr_digest` | `c794c4b8e6590101` | `c794c4b8e6590101` |
| `option_overhang_digest` | `99f56a4553b6e9c0` | `99f56a4553b6e9c0` |

SEC cover-page agreement on `sharesOutstandingPit`: **5,412 / 5,553 rows** (was 5,141), rows
too-low **41 -> 15**, too-high **371 -> 126**. **Every integer-factor offender is resolved** --
AVGO 10.0, ANET 4.0, CMG 50.0, APH 2.0, BKNG 25.0, AMCR 0.2 all now agree.

### Deviations from the plan, and why

1. **P1 uses `actions=False` on the price path**, not `actions=True`. `auto_adjust=False`
   returns `Close` AND `Adj Close` on its own (verified live: AAPL 2020-07-31 -> 106.26 and
   102.795), so the action columns were never needed for the two-column write -- and omitting
   them keeps `prices` clean OHLCV, an invariant the repo already had a test for. Splits get
   their own fetcher under P2 regardless.

2. **P2's union rule had to be made SYMMETRIC.** The plan's rule was "yfinance only -> keep",
   on the premise that yfinance carries no false positives. **That premise is wrong**:
   yfinance's `Stock Splits` column also carries SPINOFF factors. Trusting it unconditionally
   injected BDX 2022-04-01 x1.025 and 2026-02-10 x1.272, which compound to 1.304 and put
   BDX's whole PIT series 23% off the SEC cover page -- 67 bad rows on a ticker that had none.
   The shape test now applies to BOTH vendors and only CORROBORATION overrides it. The shape
   test itself was widened from "integer or reciprocal of an integer" to "a fraction with
   denominator <= 20", because 3:2, 5:4 and 4:3 splits are real (and GOOGL's genuine 2014
   split is 1.998 in yfinance, kept only because Sharadar corroborates it).

3. **P2's CCL finding is NOT what the research thought.** CCL has exactly ONE split row in
   `sharadar_actions` (1998-06-15 x2) and it is corroborated, so the union rule neither drops
   nor adds anything for it. `merged == sharesbas` exactly on every CCL row -- it was never
   de-adjusted at all. Its 1.15-1.36 ratio against the SEC is **Carnival's dual-listed
   structure** (CCL + CUK; Sharadar counts the combined entity, the SEC cover page counts
   Carnival Corporation only), and the single 0.0012 outlier is an **SEC-side extraction
   defect** at 2021-01-26 (932,485,510,000 shares, ~1000x too large). Not a split problem;
   the rule was correctly left alone rather than widened to make one ticker pass.

4. **P1's MNST verification CANNOT pass, and the reason is upstream.** The plan expected the
   `--full` re-download to clear MNST's 97/47 alternation. It does not: **Yahoo itself serves
   the alternating series**, identically from `yf.download` and `Ticker.history`, on a freshly
   emptied table. Sharadar prices MNST at 45.18 on 2026-08-07 where Yahoo says 90.36 --
   exactly 2x -- so Yahoo has failed to back-adjust MNST for the 2:1 split its OWN splits feed
   reports on 2026-08-11. The re-pull machinery is correct and does fix the general case; this
   specific ticker is a live vendor defect. It is exactly what P5 invariant 2 is for, and the
   validator flags it (6 unexplained jumps, no corroborating split).

5. **P5's invariant 1 does NOT block, and the threshold is measured, not aspirational.** The
   plan targeted >99% pass. The achieved rate is **87.4%**, and the residual is not a repo
   defect: **Yahoo back-adjusts prices for SPINOFFS and Sharadar's `sharesbas` does not**, so
   for ~226 tickers `sharadar.marketcap` is itself internally inconsistent. HON is the proof:
   `sharesbas` is unchanged across its 2026-06-29 spinoff (316,826,560 -> 316,940,010) while
   its `price` drops 428.68 -> 246.27. The cancellation identity therefore removes the split
   and dividend legs but leaves a **spinoff leg** on names with spinoffs.
   **The control settles it**: `sharadar.price x sharesbas / marketcap` is within 1% on
   **99.82%** of rows, so the identity itself is sound and the disagreement is purely about
   which vendor restated which corporate action. The gate blocks on invariant 3 alone
   (measured: 10 unexplained jumps in 3.26M rows = 3e-6, budget 1e-4).

6. **`forward_return` was kept, not deleted.** The plan said to delete it if the label change
   left it callerless. It has a legitimate remaining contract -- a ratio of levels IS a total
   return for a cumulative-return INDEX (`equity_tr`, `bond_10y_tr`) -- so it was kept with
   that contract stated in the signature and docstring, and the stock labels no longer call it.

7. **The cube was NOT empty.** `cube_part_prices` held 3,828,833 rows (the merged `cube` table
   genuinely does not exist). Nothing in P0-P5 read it, so no conclusion changed, but the
   user's rebuild starts from a stale prices part that must be rebuilt with `-F`.

### Fingerprint digests that moved (P4e)

Isolated by fingerprinting HEAD's code and this branch's code against the SAME database:

* **moved by this branch (4)**: `label.zscore_h30/60/90` (the label became
  `forward_compound(stock_ret, h)` instead of a price ratio) and `panel.institutional`
  (`inst_ownership_pct` now divides by `sharesOutstandingPit`).
* **NOT moved by this branch**: `label.rank_h30/60/90`. On the harness's dividend-free random
  walk `forward_compound` is a monotone transform of `forward_return`, so the cross-sectional
  RANK is preserved while the z-score is not -- an internal check that 4c did exactly what it
  claims.
* **already drifted at HEAD (4 + 2 renames)**: `panel.betas`, the three rank labels, and the
  `prim.price_column_returns` -> `prim.macro_factor_returns` rename. `DECLARED_DRIFT` is now
  **empty**, as its own comment always said it must become.

### Test state

`tests/data_aggregate/test_aggregate_regression.py` goes 2 failed -> **4 passed**. The
remaining failures in the suite are **pre-existing at HEAD** and verified so in a worktree:
`test_part_registry` (2 -- momentum declares `open`/`high`/`low`, which `cube_part_prices` has
never stored), the `SimpleNamespace has no attribute config` fixture family in
`tests/data_extract/prices/` (5), and the `daily_market_cap(fund_hist, None)` path in
`test_fundamental_features` (2).

New tests: `tests/data_extract/prices/test_adjustment_basis.py` (7) and
`tests/data_extract/sharadar/test_split_union.py` (22).
