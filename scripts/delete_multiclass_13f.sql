-- Delete the class-level sec13f_hr rows of the companies that list a second share class (U-67).
-- The new extraction stores one row per manager x company x accession in canonical-class shares,
-- so these rows must be rebuilt, not reused. Only the quarters of the class window are touched.
-- Windows from security_master (lineage_role = 'secondary_class') as of 2026-11-06.
--
-- Dry run (default): counts only.
--   docker exec -i pea_db psql -U alexandre -d pea < scripts/delete_multiclass_13f.sql
-- Delete: same command with -v apply=1.

\set ON_ERROR_STOP on
\if :{?apply}
\else
\set apply 0
\endif

CREATE TEMP TABLE multiclass_window (company text, ticker text, valid_from date, valid_to date);
INSERT INTO multiclass_window (company, ticker, valid_from, valid_to)
SELECT c.company, t.ticker, c.valid_from, c.valid_to
FROM (VALUES
    ('BF-B', '2009-06-29'::date, NULL::date, ARRAY['BF-B', 'BF-A']),
    ('BRK-B', '2009-06-26', NULL, ARRAY['BRK-B', 'BRK-A']),
    ('CMCSA', '2009-06-26', '2015-12-11', ARRAY['CMCSA', 'CMCSK']),
    ('CMG', '2009-07-02', '2009-12-18', ARRAY['CMG', 'CMG-B']),
    ('FOXA', '2019-03-19', NULL, ARRAY['FOXA', 'FOX']),
    ('GOOGL', '2014-04-01', NULL, ARRAY['GOOGL', 'GOOG']),
    ('HUBB', '2009-07-14', '2015-12-24', ARRAY['HUBB', 'HUBA']),
    ('LEN', '2009-06-26', NULL, ARRAY['LEN', 'LEN-B']),
    ('MKC', '2009-07-17', NULL, ARRAY['MKC', 'MKC-V']),
    ('NWSA', '2013-07-02', NULL, ARRAY['NWSA', 'NWS']),
    ('PSKY', '2009-06-29', '2025-08-07', ARRAY['PSKY', 'CBS', 'VIAC', 'PARA']),
    ('STZ', '2009-07-08', '2022-11-11', ARRAY['STZ', 'STZ-B']),
    ('TAP', '2009-06-26', NULL, ARRAY['TAP', 'TAP-A']),
    ('WBD', '2009-06-26', '2022-04-09', ARRAY['WBD', 'DISCB', 'DISCK'])
) AS c (company, valid_from, valid_to, tickers)
CROSS JOIN LATERAL unnest(c.tickers) AS t (ticker);

-- A 13F period is a quarter end: keep the quarter that holds valid_from, stop at the quarter holding valid_to.
SELECT w.company, h.ticker, count(*) AS rows, count(DISTINCT h.accession_number) AS accessions, min(h.period), max(h.period)
FROM sec13f_hr h
JOIN multiclass_window w ON w.ticker = h.ticker
WHERE h.period >= date_trunc('quarter', w.valid_from) AND (w.valid_to IS NULL OR h.period < date_trunc('quarter', w.valid_to) + interval '3 months')
GROUP BY 1, 2
ORDER BY 1, 2;

\if :apply
BEGIN;
DELETE FROM sec13f_hr h
USING multiclass_window w
WHERE w.ticker = h.ticker
  AND h.period >= date_trunc('quarter', w.valid_from)
  AND (w.valid_to IS NULL OR h.period < date_trunc('quarter', w.valid_to) + interval '3 months');
COMMIT;
\endif
