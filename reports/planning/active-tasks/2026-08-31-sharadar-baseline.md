# Baseline snapshot — Sharadar TTM window defects (Phase 1)

**Taken**: 2026-08-31, from the live `pea` DB, BEFORE any code change.
Re-run the exact SQL below after Phase 4; the after-measurement must be the same query.

## Q1 — table totals

```sql
SELECT count(*) AS total_rows,
       count(*) FILTER (WHERE "totalRevenue" IS NULL) AS null_revenue,
       count(DISTINCT ticker) AS tickers
FROM fundamentals_history;
```

| total_rows | null_revenue | tickers |
|---|---|---|
| 51,608 | 3,593 | 488 |

Matches the plan's expected 51,608 / 3,593.

## Q2 — the named 11 tickers

```sql
SELECT ticker, count(*) AS rows, count("totalRevenue") AS whole_ttm,
       round(100.0*count(*) FILTER (WHERE "totalRevenue" IS NULL)/count(*),1) AS pct_null
FROM fundamentals_history
WHERE ticker IN ('COST','AZO','KR','AVGO','GPN','GOOGL','IBM','KO','AAPL','BBY','OKE')
GROUP BY ticker ORDER BY ticker;
```

| ticker | rows | whole today | pct NULL | plan said | target |
|---|---|---|---|---|---|
| AAPL | 123 | 117 | 4.9 | 117 | 117 (no change) |
| AVGO | **0** | **0** | absent | 0 | ~66 |
| AZO | 69 | 0 | 100.0 | 0 | 113 |
| BBY | 124 | 118 | 4.8 | 118 | 118 (no change) |
| COST | 85 | 2 | 97.6 | 2 | 121 |
| GOOGL | 97 | 83 | 14.4 | 83 | 87 |
| GPN | 111 | **94** | 15.3 | 95 | 98 (+4) |
| IBM | 125 | 119 | 4.8 | 119 | 121 |
| KO | 125 | 119 | 4.8 | 119 | 121 |
| KR | 50 | 0 | 100.0 | 0 | 112 |
| OKE | 125 | 119 | 4.8 | 119 | 119 (no change) |

**AVGO returns no row at all** — it is absent from `fundamentals_history`, exactly as diagnosed.

⚠ **One deviation from the plan**: GPN measures **94** whole rows, not 95. The plan's per-ticker
table said 95 → 99 (+4). The +4 recovery is what is being asserted, so the Phase 4 target moves to
**98**. Every other figure reproduces to the row.

## Q3 — the ARQ input and the duplicate census

```sql
SELECT count(*) AS arq_rows, count(DISTINCT ticker) AS tickers
FROM fundamentals_sharadar WHERE dimension='ARQ';

SELECT count(*) AS dup_groups, sum(n-1) AS extra_rows, count(DISTINCT ticker) AS tickers
FROM (SELECT ticker, calendardate, count(*) n FROM fundamentals_sharadar
      WHERE dimension='ARQ' GROUP BY 1,2 HAVING count(*)>1) d;
```

| metric | value | plan said |
|---|---|---|
| ARQ rows | 51,847 | — |
| ARQ tickers | 489 | — |
| duplicate `(ticker, calendardate)` groups | 543 | 543 |
| extra rows | 599 | 599 |
| tickers affected | 316 | 316 |

## Verdict

The DB has not moved since the plan was written. Every target number in the plan stands, with the
single GPN correction noted above.
