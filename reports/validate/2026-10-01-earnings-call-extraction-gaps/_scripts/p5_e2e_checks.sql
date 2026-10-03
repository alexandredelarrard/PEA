\echo '=== 1. cache rows for the sample under speaker-clean-v1 (calls scored vs calls stored) ==='
SELECT s.ticker, s.calls_stored, coalesce(f.calls,0) finbert_calls, coalesce(f.sections,0) finbert_sections, coalesce(e.calls,0) embedded_calls, coalesce(e.turns,0) embedded_turns
FROM (SELECT ticker, count(*) calls_stored FROM earnings_call_sections WHERE paragraph=1 AND ticker IN ('ABNB','PLTR','CEG') GROUP BY 1) s
LEFT JOIN (SELECT ticker, count(DISTINCT quarter) calls, count(*) sections FROM earnings_call_sentiment WHERE model='yiyanghkust/finbert-tone:speaker-clean-v1' GROUP BY 1) f USING (ticker)
LEFT JOIN (SELECT ticker, count(DISTINCT quarter) calls, count(*) turns FROM earning_calls_embedding WHERE model='text-embedding-3-small:speaker-clean-v1' GROUP BY 1) e USING (ticker)
ORDER BY 1;
SELECT model, count(DISTINCT (ticker,quarter)) calls, count(DISTINCT ticker) tickers FROM earnings_call_sentiment GROUP BY 1 ORDER BY 1;
\echo '=== 2a. cube_part_text columns ==='
SELECT string_agg(column_name, ', ' ORDER BY ordinal_position) cols, count(*) FILTER (WHERE column_name LIKE 'f_ec_%') n_f_ec, count(*) n_cols FROM information_schema.columns WHERE table_schema='public' AND table_name='cube_part_text';
\echo '=== 2b. cube_part_text rows, tickers, first/last date ==='
SELECT ticker, count(*) rows_stored, min(date)::date first_date, max(date)::date last_date FROM cube_part_text GROUP BY 1 ORDER BY 1;
\echo '=== 2c. non-null share per column over trading sessions strictly after each ticker''s first call (denominator = PRICED cube_part_prices sessions, close_split not null -- CEG has an unpriced 1995-2021 grid, so its 15 calls of the 2008-2011 predecessor issuer are masked by availability; rows with no feature are dropped by design) ==='
WITH fc AS (SELECT ticker, min(as_of)::date d0 FROM earnings_call_sections WHERE paragraph=1 AND ticker IN ('ABNB','PLTR','CEG') GROUP BY 1),
grid AS (SELECT p.ticker, p.date FROM cube_part_prices p JOIN fc ON fc.ticker=p.ticker AND p.date > fc.d0 AND p.close_split IS NOT NULL),
j AS (SELECT g.ticker AS tk, t.* FROM grid g LEFT JOIN cube_part_text t ON t.ticker=g.ticker AND t.date=g.date)
SELECT j.tk AS ticker, count(*) sessions,
 round(avg((f_ec_tone IS NOT NULL)::int),3) tone, round(avg((f_ec_tone_vs_hist IS NOT NULL)::int),3) tone_h,
 round(avg((f_ec_qa_gap IS NOT NULL)::int),3) qa_gap, round(avg((f_ec_qa_gap_vs_hist IS NOT NULL)::int),3) qa_gap_h,
 round(avg((f_ec_uncertainty IS NOT NULL)::int),3) unc, round(avg((f_ec_uncertainty_vs_hist IS NOT NULL)::int),3) unc_h,
 round(avg((f_ec_qa_coherence_mean IS NOT NULL)::int),3) coh, round(avg((f_ec_qa_coherence_mean_vs_hist IS NOT NULL)::int),3) coh_h,
 round(avg((f_ec_tone_delta IS NOT NULL)::int),3) tone_d, round(avg((f_ec_length_delta IS NOT NULL)::int),3) len_d,
 round(avg((f_ec_qa_qq_distance IS NOT NULL)::int),3) qa_qq, round(avg((f_ec_prep_qq_distance IS NOT NULL)::int),3) prep_qq
FROM j GROUP BY 1 ORDER BY 1;
\echo '=== 3. PIT spot check: 3 calls, sessions as_of-3 .. as_of+5 ==='
WITH c AS (SELECT * FROM (VALUES ('ABNB','2025-08-06'::date),('PLTR','2025-11-03'::date),('CEG','2025-05-06'::date),('CEG','2025-08-07'::date)) v(ticker,x)),
cc AS (SELECT s.ticker, s.quarter, s.as_of::date call_date FROM earnings_call_sections s JOIN c ON c.ticker=s.ticker AND s.paragraph=1 AND abs(s.as_of::date - c.x) <= 20)
SELECT cc.ticker, cc.quarter, cc.call_date, p.date::date AS session, CASE WHEN p.date::date=cc.call_date THEN '<- as_of' ELSE '' END flag,
 round(t.f_ec_tone::numeric,4) tone, round(t.f_ec_tone_delta::numeric,4) tone_d, round(t.f_ec_qa_qq_distance::numeric,4) qa_qq
FROM cc JOIN cube_part_prices p ON p.ticker=cc.ticker AND p.date BETWEEN cc.call_date-4 AND cc.call_date+6
LEFT JOIN cube_part_text t ON t.ticker=p.ticker AND t.date=p.date ORDER BY 1,4;
\echo '=== 4. ranges (min/max, count of +-inf / NaN) ==='
SELECT 'tone' c, min(f_ec_tone), max(f_ec_tone), count(*) FILTER (WHERE f_ec_tone IN ('Infinity','-Infinity','NaN')) bad FROM cube_part_text
UNION ALL SELECT 'tone_vs_hist', min(f_ec_tone_vs_hist), max(f_ec_tone_vs_hist), count(*) FILTER (WHERE f_ec_tone_vs_hist IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'qa_gap', min(f_ec_qa_gap), max(f_ec_qa_gap), count(*) FILTER (WHERE f_ec_qa_gap IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'qa_gap_vs_hist', min(f_ec_qa_gap_vs_hist), max(f_ec_qa_gap_vs_hist), count(*) FILTER (WHERE f_ec_qa_gap_vs_hist IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'uncertainty', min(f_ec_uncertainty), max(f_ec_uncertainty), count(*) FILTER (WHERE f_ec_uncertainty IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'uncertainty_vs_hist', min(f_ec_uncertainty_vs_hist), max(f_ec_uncertainty_vs_hist), count(*) FILTER (WHERE f_ec_uncertainty_vs_hist IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'qa_coherence_mean', min(f_ec_qa_coherence_mean), max(f_ec_qa_coherence_mean), count(*) FILTER (WHERE f_ec_qa_coherence_mean IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'qa_coherence_mean_vs_hist', min(f_ec_qa_coherence_mean_vs_hist), max(f_ec_qa_coherence_mean_vs_hist), count(*) FILTER (WHERE f_ec_qa_coherence_mean_vs_hist IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'tone_delta', min(f_ec_tone_delta), max(f_ec_tone_delta), count(*) FILTER (WHERE f_ec_tone_delta IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'length_delta', min(f_ec_length_delta), max(f_ec_length_delta), count(*) FILTER (WHERE f_ec_length_delta IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'qa_qq_distance', min(f_ec_qa_qq_distance), max(f_ec_qa_qq_distance), count(*) FILTER (WHERE f_ec_qa_qq_distance IN ('Infinity','-Infinity','NaN')) FROM cube_part_text
UNION ALL SELECT 'prep_qq_distance', min(f_ec_prep_qq_distance), max(f_ec_prep_qq_distance), count(*) FILTER (WHERE f_ec_prep_qq_distance IN ('Infinity','-Infinity','NaN')) FROM cube_part_text;
