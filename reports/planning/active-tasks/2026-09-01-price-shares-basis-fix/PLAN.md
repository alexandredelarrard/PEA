# Plan: one consistent basis for price, shares and marketCap

**Date**: 2026-09-01
**Phase**: 2 of 3 (FIC) — follows `reports/research/codebase/2026-09-01-marketcap-price-shares-basis.md`
**Status**: **V2, decisions settled.** Ready for `/implement`.
**Next**: `/implement reports/planning/active-tasks/2026-09-01-price-shares-basis-fix/PHASE-0-baseline.md`

This file is the INDEX. Each phase is a sibling file, independently implementable and
independently verifiable.

| phase | file | goal | depends on |
|---|---|---|---|
| **P0** | [PHASE-0-baseline.md](PHASE-0-baseline.md) | Record the "before" numbers as a rerunnable script | — |
| **P1** | [PHASE-1-prices-extract.md](PHASE-1-prices-extract.md) | Two price columns, `--full`, re-download 3.26M rows | P0 |
| **P2** | [PHASE-2-split-events.md](PHASE-2-split-events.md) | `prices_splits` from yfinance + corroborated union | P0 |
| **P3** | [PHASE-3-shares-basis.md](PHASE-3-shares-basis.md) | Stop de-adjusting features; add `sharesOutstandingPit` | P2 |
| **P4** | [PHASE-4-consumers.md](PHASE-4-consumers.md) | Cube schema, rename migration, label routing | P1, P3 |
| **P5** | [PHASE-5-validator.md](PHASE-5-validator.md) | Prices validator + tests + DoD gate | P4 |

P1 and P2 are independent (different tables, both extract-layer) and should run concurrently —
P1 is the longest wall-clock item.

## Scope boundary

**This task delivers clean price and target CODE, plus a refilled `prices` table (P1).**

The cube rebuild, the label rebuild and the model re-baseline are **the user's, run manually
after the fix lands.** Nothing in P0-P5 rebuilds the cube or retrains anything. See
[§After the fix](#after-the-fix-the-users-manual-rebuild) for the checklist to run then.

One consequence to keep in mind while implementing: between the end of P5 and that manual
rebuild, the **cube is stale** — it holds labels and features derived from the old price basis.
Do not treat cube numbers as evidence during P4/P5 verification; verify against `prices`,
`fundamentals_history` and Sharadar directly.

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
| **D6** | **The label/cube rebuild is OUT of this task** — the user runs it manually afterwards | The code must be correct and the price table refilled first; rebuilding is a long compute step with no code risk. `neutralize_log_mcap: true` still means every stored label changes, so the rebuild is *required* before any post-fix metric is meaningful — just not here. |
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

- Reconstructing total return from dividends — superseded by D1.
- Giving `prices_dividends` a consumer.
- Survivorship bias (`sharadar_sp500` ingested but unused).
- Multi-class share-count reconciliation — the vendor basis sidesteps it for mcap.
- An `open`/`high`/`low` retention review.

## Risks

| risk | mitigation |
|---|---|
| A missed `close` reader silently computes price returns | D2's rename forces `KeyError`; P5 invariant 1 is the backstop |
| The macro leg flips basis with the equity leg | P1 parameterises `auto_adjust`; success criterion asserts `prices_macro` is bit-identical |
| The 3.26M-row re-download fails partway, leaving mixed vintages | Write to a staging table, swap on success (P1) |
| Sharadar's `marketcap` is itself wrong for some tickers | Invariant 1 tolerance is 1%, and failures are *reported*, not auto-corrected |
| Re-baselining loses the ability to compare to past experiments | P6 archives the pre-fix metrics before rebuilding |

## Success criteria

- [ ] `|close_split x sharesOutstanding / sharadar.marketcap - 1| < 1%` on >99% of 51,255 rows
      (today the median row is off by 15%, and 1995 rows by 5x)
- [ ] `sharesOutstanding == sharesbas` on 100% of rows
- [ ] `sharesOutstandingPit` within ±3% of `fundamentals_history_sec.sharesOutstanding` on the
      96-ticker overlap (today 24 of 96 fail)
- [ ] MNST Jul–Aug 2026 shows no 2x alternation; spike-and-revert count post-2020 is 0
- [ ] No `KeyError: 'close'` anywhere — every reader migrated
- [ ] The `split_part < 1` cohort's forward-return advantage (27.81% vs 15.38%) disappears
- [ ] `prices_macro.equity_tr` is bit-identical before and after — macro must NOT change
