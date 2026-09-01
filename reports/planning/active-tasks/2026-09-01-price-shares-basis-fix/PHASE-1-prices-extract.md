# Phase 1 — Prices extract: two columns, `--full`, full re-download ⬜

**Parent**: [PLAN.md](PLAN.md) · **Depends on**: P0 · **Blocks**: P4 · **Estimate**: 3-4h code + several hours wall-clock download

Run concurrently with [P2](PHASE-2-split-events.md) — different tables, no shared code path
except `download_ohlcv`, which P1 owns.

## Goal

`prices` carries `close_split` **and** `close_total`, written from one yfinance response; there
is a way to re-pull history; and the whole table is re-downloaded once on the new basis. The
macro leg must come out unchanged.

## ⚠ The trap that must be handled first

`download_ohlcv` is **shared**. [fetch_macro.py:109](src/data_extract/utils/prices/fetch_macro.py#L109)
calls it for `MACRO_PRICE_SERIES` = `SPY` → `equity_tr`, `^VIX` → `vix`, `CL=F` → `oil`,
`GC=F` → `gold`, `XLE` → `energy`.

`SPY` is named `equity_tr` because it *is* the total-return benchmark: it feeds the L/S benchmark
leg ([ls_model.py:99-104](src/strategies/utils/ls_model.py#L99)), `beta_market`, and `fwd_market`
inside every label. `XLE` pays ~3%. A bare `auto_adjust=False` flip silently converts both to
**price** return and corrupts all of that.

**So `auto_adjust` becomes a parameter, and macro pins itself to the total-return column.**

## Changes

### 1. `src/data_extract/utils/prices/fetch_prices.py`

- [ ] `_download_price_chunk`: fix the annotation typo `actions: False` → `actions: bool = False`,
      and add `auto_adjust: bool` as an **explicit, required** argument (no default — a default is
      how this trap gets re-set later).
- [ ] Pass `auto_adjust=False` and `actions=True` from the equity path so the response carries
      `Close`, `Adj Close`, `Dividends` and `Stock Splits` in one call.
- [ ] `download_ohlcv`: thread `auto_adjust` through; keep the current signature order so the
      macro call site is a one-line change.
- [ ] `_normalize_prices` / `_chunk_response_to_frames`: map `Close` → `close_split` and
      `Adj Close` → `close_total`. **Emit neither under the name `close`** (D2). If a response
      lacks `Adj Close`, raise rather than fall back — a silent single-column write is the exact
      failure this phase exists to prevent.
- [ ] `fetch_price_history`: add `full: bool = False`. When true, bypass `resume_since` and use
      the whole `years_history` window.

```python
since = (today - pd.DateOffset(years=years_history)
         if full else resume_since(context, Tables.prices, tickers, years_history))
```

- [ ] Split-triggered re-pull (D5): before computing `since`, read `prices_splits` (P2) for events
      dated after each ticker's last price date; any ticker with one gets the full window
      regardless of `full`. Two windows in one run means either two `download_ohlcv` calls or a
      per-ticker `since` — prefer two calls, it keeps `resume_since` untouched.

### 2. `src/data_extract/utils/prices/fetch_macro.py`

- [ ] `_fetch_price_leg`: pass `auto_adjust=True` explicitly, with a comment naming `SPY` as
      `equity_tr` and stating that the macro legs are consumed as RETURNS only. This keeps the
      macro output byte-identical.
- [ ] It reads `raw[["date","ticker","close"]]` — under `auto_adjust=True` the normaliser must
      still produce a usable column name. Simplest: when `auto_adjust=True` the normaliser emits
      `close_total` only, and macro selects that.

### 3. Schema

- [ ] `sql/schema.sql`: replace `"close"` in the `prices` block with `"close_split"` and
      `"close_total"`, both `DOUBLE PRECISION`. **Splice by hand** — regeneration drops 8
      hand-added indexes (`schema-sql-regeneration-is-lossy` memory). The diff must be purely
      additive apart from the one removed line.
- [ ] `src/data_store/schema.py:105` — `prices` Table. If it needs a `read_columns`, list both.

### 4. CLI

- [ ] [cli.py:99-107](src/data_extract/cli.py#L99-L107): add `-F/--full` to `price-history`,
      matching the eight sibling commands that already have it.

### 5. The one-off re-download

- [ ] Run `python -m src data_extract price-history --full` over the full 491-ticker universe.
- [ ] **Write to a staging table and swap on success.** A partial failure that leaves half the
      table on the old basis and half on the new is strictly worse than today's state, and there
      is no vintage column to tell them apart.
- [ ] `trim_prelisting_bars` still runs before the upsert — a full pull carries the synthetic
      pre-listing prefix that an incremental tail does not.

## Verification

- [ ] `close_split` for AAPL 2020-07-31 == **106.26** (not 102.80), and equals Sharadar `price`
- [ ] `close_split` for KO 2004-02-27 == **24.98**; `close_total` == **12.79**
- [ ] AMZN (non-payer) `close_split == close_total` on every row — the clean identity check
- [ ] `close_total / close_split` is monotone non-decreasing in date for every payer, and → 1.0
      on the last row
- [ ] MNST 2026-07-15 → 2026-08-15: no 97/47 alternation; a single clean split step
- [ ] Spike-and-revert scan post-2020 returns **0** (P0 recorded 3)
- [ ] Row count is within a few hundred of 3,263,505 (delistings and calendar edges aside)
- [ ] **`prices_macro.equity_tr` digest matches P0's `macro_equity_tr_digest` exactly.** If it
      moved, the macro leg flipped basis — stop and fix before P4.
- [ ] `SELECT count(*) FROM prices WHERE close_total IS NULL AND close_split IS NOT NULL` == 0
- [ ] `-F/--full` appears in `price-history --help`

## Risks

| risk | mitigation |
|---|---|
| Download fails partway → mixed vintages | Staging table + atomic swap; never upsert into the live table on a `--full` run |
| yfinance rate-limits a 491-ticker full pull | The existing 3-attempt retry with backoff; run in background and monitor |
| Yahoo has revised old bars since the last pull | Expected and desired — that is the point. Compare row-level deltas and spot-check a handful against Sharadar `price` |
| Macro re-runs on the wrong flag | The digest check above is a hard gate |

## Rollback

Keep the pre-swap `prices` table as `prices_pre_basis_fix` until P6 passes. Revert is a rename
plus reverting the schema splice.
