-- P9 read-only DB checks (run: docker exec -i pea_db psql -U alexandre -d pea < p9_db_checks.sql)
\echo '== 1. scope of the EC tables'
select 'sections' t, count(*) rows, count(distinct (ticker,quarter)) calls, count(distinct ticker) tickers, min(as_of), max(as_of) from earnings_call_sections
union all select 'sentiment', count(*), count(distinct (ticker,quarter)), count(distinct ticker), min(as_of)::date, max(as_of)::date from earnings_call_sentiment
union all select 'cube_part_text', count(*), null, count(distinct ticker), min(date)::date, max(date)::date from cube_part_text;
\echo '== 2. grain (expect all 0 except null_tid_calls)'
with c as (select ticker, quarter, min(as_of) a, max(as_of) b, count(*) n, min(paragraph) p0, max(paragraph) p1, count(distinct transcript_id) tids, sum((transcript_id is null)::int) nulltid from earnings_call_sections group by 1,2)
select count(*) calls, sum((a<>b)::int) multi_asof, sum((p0<>1)::int) no_p1, sum((p1<>n)::int) noncontig, sum((tids>1)::int) multi_tid, sum((nulltid>0)::int) null_tid_calls,
 (select count(*) from (select ticker,a from c group by 1,2 having count(*)>1) x) multi_call_per_date from c;
\echo '== 3. AC-001: distinct tickers per calendar quarter of the call date (floor 485)'
select to_char(date_trunc('quarter',as_of),'YYYY"Q"Q') q, count(distinct ticker) from earnings_call_sections where paragraph=1 and as_of>='2024-01-01' group by 1 order by 1;
\echo '== 4. roster tickers with no call (scope ceiling)'
select s.ticker, (select count(distinct quarter) from earnings_call_sections_legacy l where l.ticker=s.ticker) legacy_calls from sp500_tickers s where not exists (select 1 from earnings_call_sections e where e.ticker=s.ticker and e.paragraph=1) order by 1;
\echo '== 5. AC-002: earnings_surprises events since 2026-06-01 with a call within +-3 days (floor 97%)'
with ev as (select e.ticker, e.earnings_date::date d from earnings_surprises e where e.earnings_date>='2026-06-01' and e.earnings_date<=now() and e.eps_actual is not null and e.ticker in (select distinct ticker from earnings_call_sections)),
m as (select ev.*, (select min(abs(c.as_of - ev.d)) from earnings_call_sections c where c.ticker=ev.ticker and c.paragraph=1 and c.as_of between ev.d-10 and ev.d+10) gap from ev)
select count(*) events, sum((gap<=3)::int) within3, round(100.0*sum((gap<=3)::int)/count(*),2) pct, string_agg(case when gap is null or gap>3 then ticker||' '||d end, ', ') misses from m;
\echo '== 6. AC-008 proxy: calls since 2024, new vs legacy sentiment'
with l as (select distinct ticker, quarter, as_of::date a from earnings_call_sentiment_legacy where as_of>='2024-01-01' and as_of<'2026-10-01'),
n as (select ticker, quarter, as_of a from earnings_call_sections where paragraph=1 and as_of>='2024-01-01' and as_of<'2026-10-01')
select (select count(*) from l) legacy_calls, (select count(*) from n) new_calls,
 (select count(*) from l where l.ticker in (select ticker from n) and not exists (select 1 from n where n.ticker=l.ticker and abs(n.a-l.a)<=7)) legacy_unmatched_in_scope,
 (select count(*) from l where l.ticker not in (select ticker from n)) legacy_out_of_scope;
\echo '== 7. F-009: provider fiscal-label jumps on retailers'
select ticker, string_agg(quarter||'@'||as_of, ' ' order by as_of) from earnings_call_sections where paragraph=1 and ticker in ('DG','DLTR','HD','LOW','LULU','TGT','ULTA','WSM') and as_of between '2025-11-01' and '2026-06-30' group by 1;
\echo '== 8. cache rows per model tag'
select 'embedding' t, model, count(*), count(distinct (ticker,quarter)) from earning_calls_embedding group by 1,2 union all select 'sentiment', model, count(*), count(distinct (ticker,quarter)) from earnings_call_sentiment group by 1,2;
\echo '== 9. legacy tables pending deletion'
select relname, pg_size_pretty(pg_total_relation_size(oid)) from pg_class where relname like '%\_legacy' and relkind='r';
