\pset footer off
\echo == empty / whitespace-only content
select count(*) n_rows, sum((content is null)::int) n_null, sum((content is not null and btrim(content, E' \t\r\n\u00a0')='')::int) n_blank,
 round(100.0*sum((content is null or btrim(content, E' \t\r\n\u00a0')='')::int)/count(*),3) pct_empty from earnings_call_sections;
\echo == length quantiles (non-empty rows): chars then words
with t as (select length(content) c, coalesce(array_length(regexp_split_to_array(btrim(content), '\s+'),1),0) w from earnings_call_sections where content is not null and btrim(content)<>'')
select 'chars' k, percentile_disc(array[0.01,0.05,0.25,0.5,0.75,0.95,0.99]) within group (order by c) from t
union all select 'words', percentile_disc(array[0.01,0.05,0.25,0.5,0.75,0.95,0.99]) within group (order by w) from t;
\echo == share <=3 words, <=1 word, <=5, <=10 (non-empty)
with t as (select coalesce(array_length(regexp_split_to_array(btrim(content), '\s+'),1),0) w from earnings_call_sections where content is not null and btrim(content)<>'')
select count(*), round(100.0*avg((w<=1)::int),2) le1, round(100.0*avg((w<=3)::int),2) le3, round(100.0*avg((w<=5)::int),2) le5, round(100.0*avg((w<=10)::int),2) le10 from t;
\echo == top speakers that look like placeholders
select speaker, count(*) from earnings_call_sections where speaker is null or speaker ~* '^(operator|moderator|coordinator|conference operator|ai insights|speaker \d+|unknown|unidentified.*|presentation|participant|analyst|executive|management|corporate participant|company representative|)$' group by 1 order by 2 desc limit 30;
\echo == top 25 raw short contents (<=3 words)
select content, count(*) from earnings_call_sections where coalesce(array_length(regexp_split_to_array(btrim(content), '\s+'),1),0)<=3 group by 1 order by 2 desc limit 25;
\echo == bracketed artefacts over all rows
select sum((content ~* '\[(indiscernible|inaudible|unintelligible|ph|sic|technical difficulty|crosstalk)\]|\((indiscernible|inaudible|unintelligible|ph|sic|crosstalk)\)')::int) n_artefact_rows,
       sum((content ~* 'operator instructions')::int) n_opinstr_rows, sum((content ~ '\[[^\]]{1,40}\]')::int) n_any_bracket_rows from earnings_call_sections;
\echo == calls per year
select extract(year from as_of)::int y, count(distinct (ticker,quarter)) from earnings_call_sections group by 1 order by 1;
