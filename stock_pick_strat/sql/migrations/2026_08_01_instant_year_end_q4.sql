-- Relabel fiscal-year-end BALANCE-SHEET snapshots in `fundamentals_facts` from
-- fiscal_period='FY' to 'Q4'.
--
-- Why: an instant fact has no native fiscal_year/fiscal_period of its own and
-- borrows them from a duration fact in the same filing
-- (fetch_fundamentals_edgar.backfill_fiscal_period_from_filing), so a 10-K's
-- balance sheet inherited that filing's 'FY' label. A fiscal year has no
-- separate "FY balance sheet" -- the year-end snapshot IS the Q4 one -- so every
-- instant field carried a hole at Q4 in an otherwise complete quarter grid.
-- `fundamentals_periods.instant_stock` now emits 'Q4' directly; this migration
-- brings already-extracted rows in line so the table is not half-and-half.
--
-- Scope: `period_start IS NULL` only. That is what separates a genuine instant
-- from the LATEST_DURATION_TAGS fields routed through the same code path
-- (dilutedShares, basicShares, effectiveTaxRate, reportableSegments -- duration
-- facts merely TAKEN point-in-time). For those, 'FY' and 'Q4' are two different
-- measures a 10-K legitimately tags side by side (CBRE fiscal 2011: a
-- 318,454,191 full-year weighted-average basic share count AND a 320,638,316
-- Q4-only one, both dated 2011-12-31), and relabelling would collide them on the
-- primary key.
--
-- RUN ONLY WHEN NO EXTRACTION IS IN FLIGHT -- `fetch_fundamentals_edgartools`
-- writes this table per ticker and would interleave with the UPDATE.
--
-- Usage:
--   docker exec -i pea_db psql -U alexandre -d pea \
--     -f - < sql/migrations/2026_08_01_instant_year_end_q4.sql

BEGIN;

-- Guard: refuse to run if any (ticker, accession, field, fiscal_year) would end
-- up with two instant rows -- i.e. an 'FY' and a 'Q4' both without period_start.
-- Expected to be zero; a non-zero count means an assumption above no longer holds.
DO $$
DECLARE collisions bigint;
BEGIN
    SELECT count(*) INTO collisions FROM (
        SELECT 1 FROM fundamentals_facts
        WHERE duration_type = 'instant'
          AND period_start IS NULL
          AND fiscal_period IN ('FY', 'Q4')
        GROUP BY ticker, accession_number, field, fiscal_year
        HAVING count(*) > 1
    ) x;
    IF collisions > 0 THEN
        RAISE EXCEPTION 'aborting: % (ticker, accession, field, fiscal_year) groups '
                        'already hold both an FY and a Q4 instant row', collisions;
    END IF;
END $$;

UPDATE fundamentals_facts
SET fiscal_period = 'Q4'
WHERE duration_type = 'instant'
  AND fiscal_period = 'FY'
  AND period_start IS NULL;

COMMIT;

-- Verification: instant/'FY' should now be duration-derived rows only (every one
-- of them with a period_start), and instant/'Q4' should be roughly the count of
-- fiscal years x balance-sheet fields x tickers.
SELECT duration_type, fiscal_period, period_start IS NULL AS no_period_start, count(*)
FROM fundamentals_facts
WHERE duration_type = 'instant'
GROUP BY 1, 2, 3
ORDER BY 2, 3;
