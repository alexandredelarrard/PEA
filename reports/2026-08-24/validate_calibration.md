# fundamentals validation -- 2026-08-24

54 ticker(s) | 12501 open finding(s) | 0 settled and subtracted

severity: critical=293  high=8051  medium=2593  info=1564

*Nothing here gates. The nightly build of `fundamentals_facts` / `fundamentals_history` runs to completion regardless.*

## fire rates

| check | tier | substrate | examined | findings | rate | ceiling | verdict |
|---|---|---|---|---|---|---|---|
| `adjustment_unguarded` | 1 | facts | 0 | 0 | -- | 100.0% | **ABSTAINED** -- nothing to examine, NOT a pass |
| `amendment_ledger` | 1 | facts | 54 | 1 | 1.85% | 100.0% | ok |
| `code_vocabulary` | 1 | history | 76,004 | 0 | 0.00% | 0.0% | ok |
| `column_contract` | 1 | history | 69 | 0 | 0.00% | 0.0% | ok |
| `coverage_field` | 1 | history | 3,240 | 656 | 20.25% | 10.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `coverage_quarters` | 1 | history | 54 | 4 | 7.41% | 2.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `coverage_universe` | 1 | history | 54 | 0 | 0.00% | 0.0% | ok |
| `cross_identity` | 1 | history | 3,267 | 293 | 8.97% | 2.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `dimensional_scope` | 1 | facts | 252,001 | 0 | 0.00% | 0.0% | ok |
| `expected_absent_drift` | 1 | history | 3,240 | 3 | 0.09% | 100.0% | ok |
| `filing_continuity` | 1 | facts | 54 | 0 | 0.00% | 10.0% | ok |
| `filing_lag` | 1 | history | 3,267 | 1 | 0.03% | 1.0% | ok |
| `grain` | 1 | history | 3,267 | 0 | 0.00% | 0.0% | ok |
| `impossible_value` | 1 | history | 196,020 | 9 | 0.00% | 1.0% | ok |
| `pit_leak` | 1 | history | 3,267 | 0 | 0.00% | 0.0% | ok |
| `register_cost` | 1 | history | 54 | 446 | 825.93% | 100.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `register_coverage` | 1 | history | 54 | 54 | 100.00% | 100.0% | ok |
| `same_day_collapse` | 1 | facts | 3,273 | 9 | 0.27% | 100.0% | ok |
| `unexplained_null` | 1 | history | 196,020 | 0 | 0.00% | 0.0% | ok |
| `basis_step` | 2 | facts | 29,661 | 57 | 0.19% | 2.0% | ok |
| `level_outlier` | 2 | facts | 29,661 | 1566 | 5.28% | 5.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `peer_ratio` | 2 | facts | 83,663 | 2018 | 2.41% | 3.0% | ok |
| `peer_ratio_abstentions` | 2 | facts | 8 | 6 | 75.00% | 100.0% | ok |
| `scale` | 2 | facts | 29,661 | 455 | 1.53% | 1.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `series_shape` | 2 | facts | 5,616 | 1632 | 29.06% | 15.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `tag_switch_break` | 2 | facts | 29,661 | 82 | 0.28% | 2.0% | ok |
| `trend_break` | 2 | facts | 29,661 | 1553 | 5.24% | 3.0% | **THRESHOLD BUG** -- above its own declared ceiling |
| `annual_footing` | 3 | facts | 11,805 | 186 | 1.58% | 2.0% | ok |
| `cross_vintage` | 3 | facts | 252,001 | 3003 | 1.19% | 6.0% | ok |
| `derived_vs_asreported` | 3 | facts | 20,910 | 0 | 0.00% | 5.0% | ok |
| `duplicate_fact` | 3 | facts | 252,001 | 0 | 0.00% | 1.0% | ok |
| `holdout_q4` | 3 | facts | 11,805 | 230 | 1.95% | 2.0% | ok |
| `leaf_vs_total` | 3 | facts | 252,001 | 34 | 0.01% | 25.0% | ok |
| `q4_footing` | 3 | facts | 11,805 | 203 | 1.72% | 2.0% | ok |
| `restatement_ledger` | 3 | facts | 1 | 0 | 0.00% | 100.0% | ok |

> **Challenge the check before challenging the data.** 8 check(s) fired above their own declared ceiling: `register_cost` (825.9%), `series_shape` (29.1%), `coverage_field` (20.2%), `cross_identity` (9.0%), `coverage_quarters` (7.4%), `level_outlier` (5.3%), `trend_break` (5.2%), `scale` (1.5%). A check over its ceiling has a threshold bug until proven otherwise, and it buries every real finding under itself.

## the queue -- 10937 open finding(s), worst first

### critical (293)

- **`cross_identity`** ADM grossProfit @ 2017-10-31 `[115cc17bf788e95e]`
    observed=3.593e+09 | expected=3.643e+09 | deviation=-0.0137
    _GrossProfit = Revenue - COGS, on the filer's own tagged lines_
- **`cross_identity`** AMT totalAssets @ 2016-07-28 `[e209060ca2d8e7c5]`
    observed=3.074e+10 | expected=2.966e+10 | deviation=0.0353
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2016-10-27 `[0ce787f945a38199]`
    observed=3.066e+10 | expected=2.956e+10 | deviation=0.0359
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2017-02-27 `[8910491610719e8d]`
    observed=3.088e+10 | expected=2.979e+10 | deviation=0.0353
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2017-04-27 `[3d00d2b539ab8941]`
    observed=3.206e+10 | expected=3.093e+10 | deviation=0.0352
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2017-07-27 `[6da48afde2a30a02]`
    observed=3.214e+10 | expected=3.098e+10 | deviation=0.036
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2017-10-31 `[e568675c48c6a9d2]`
    observed=3.232e+10 | expected=3.117e+10 | deviation=0.0355
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2018-02-28 `[8a3cae81095bbc14]`
    observed=3.321e+10 | expected=3.209e+10 | deviation=0.0339
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2018-05-02 `[501cc4a39c6f13bd]`
    observed=3.437e+10 | expected=3.331e+10 | deviation=0.031
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2018-07-31 `[dafbe06962a28e7a]`
    observed=3.321e+10 | expected=3.22e+10 | deviation=0.0304
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2018-10-30 `[4ac4e7d85832afb2]`
    observed=3.308e+10 | expected=3.212e+10 | deviation=0.0289
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2019-02-27 `[e6682d4c38af3699]`
    observed=3.301e+10 | expected=3.201e+10 | deviation=0.0304
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2019-05-03 `[9e58959001835eb2]`
    observed=3.893e+10 | expected=3.834e+10 | deviation=0.0151
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2019-07-31 `[f75da9545daee508]`
    observed=3.907e+10 | expected=3.848e+10 | deviation=0.0151
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2019-10-31 `[f918fd1a85faf26c]`
    observed=3.931e+10 | expected=3.873e+10 | deviation=0.0146
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2020-02-25 `[fda95e2bc249bcbf]`
    observed=4.28e+10 | expected=4.171e+10 | deviation=0.0256
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2020-04-29 `[d3e94b41fcf8b868]`
    observed=4.079e+10 | expected=4.025e+10 | deviation=0.0133
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2020-07-30 `[c75f8c9809b71710]`
    observed=4.152e+10 | expected=4.097e+10 | deviation=0.0131
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** AMT totalAssets @ 2020-10-29 `[23bda2077bbd70ce]`
    observed=4.146e+10 | expected=4.091e+10 | deviation=0.0134
    _Assets = Liabilities + Equity; a derived totalLiabilities makes this a tautology, so those rows are skipped rather than passed_
- **`cross_identity`** CAT grossProfit @ 2019-02-14 `[eeaf88532b8ff6bc]`
    observed=1.482e+10 | expected=1.772e+10 | deviation=-0.1636
    _GrossProfit = Revenue - COGS, on the filer's own tagged lines_
- **`cross_identity`** CAT grossProfit @ 2019-05-06 `[e94d776e8c2861fd]`
    observed=1.482e+10 | expected=1.79e+10 | deviation=-0.1716
    _GrossProfit = Revenue - COGS, on the filer's own tagged lines_
- **`cross_identity`** CAT grossProfit @ 2019-08-01 `[9f5366ca895c5b5d]`
    observed=1.482e+10 | expected=1.78e+10 | deviation=-0.167
    _GrossProfit = Revenue - COGS, on the filer's own tagged lines_
- **`cross_identity`** CAT grossProfit @ 2019-10-31 `[26cc7013922c45d2]`
    observed=1.482e+10 | expected=1.75e+10 | deviation=-0.1528
    _GrossProfit = Revenue - COGS, on the filer's own tagged lines_
- **`cross_identity`** CAT grossProfit @ 2020-02-19 `[fe7111204dee06ce]`
    observed=1.412e+10 | expected=1.717e+10 | deviation=-0.1773
    _GrossProfit = Revenue - COGS, on the filer's own tagged lines_
- **`cross_identity`** CAT grossProfit @ 2020-05-06 `[6d2a8085cec910e7]`
    observed=1.412e+10 | expected=1.608e+10 | deviation=-0.1214
    _GrossProfit = Revenue - COGS, on the filer's own tagged lines_
*... and 268 more `critical` finding(s).*

### high (8051)

- **`annual_footing`** ADM basicShares @ 2012-06-30 `[26bdd93342e75abd]`
    observed=1.327e+09 | expected=6.65e+08 | deviation=0.9955
    source_concept=us-gaap:WeightedAverageNumberOfSharesOutstandingBasic | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/7084/000000708415000005/0000007084-15-000005-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** ADM dilutedShares @ 2012-06-30 `[48deb6707a2e8cbb]`
    observed=1.329e+09 | expected=6.66e+08 | deviation=0.9955
    source_concept=us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/7084/000000708415000005/0000007084-15-000005-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** AFL incomeTaxExpense @ 2010-12-31 `[594c78cc2f68b7d9]`
    observed=7.81e+08 | expected=1.233e+09 | deviation=-0.3666
    source_concept=us-gaap:IncomeTaxExpenseBenefit | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/4977/000000497713000030/0000004977-13-000030-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** APA netIncome @ 2014-12-31 `[51f58b585e5283b0]`
    observed=-5.06e+09 | expected=-8.019e+09 | deviation=0.369
    source_concept=us-gaap:ProfitLoss | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/1841666/000167337917000004/0001673379-17-000004-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** APA totalRevenue @ 2012-12-31 `[5551af1960d72715]`
    observed=1.708e+10 | expected=1.656e+10 | deviation=0.031
    source_concept=us-gaap:OilAndGasRevenue | resolution_method=linkbase_root | root_anchor=IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments
    https://www.sec.gov/Archives/edgar/data/1841666/000119312515070388/0001193125-15-070388-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** APA totalRevenue @ 2013-12-31 `[9ae93363ecc3c3a7]`
    observed=1.556e+10 | expected=1.444e+10 | deviation=0.0777
    source_concept=us-gaap:OilAndGasRevenue | resolution_method=linkbase_root | root_anchor=IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments
    https://www.sec.gov/Archives/edgar/data/1841666/000119312516481920/0001193125-16-481920-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** APA totalRevenue @ 2014-12-31 `[8973a12472ce82e3]`
    observed=1.307e+10 | expected=1.147e+10 | deviation=0.1393
    source_concept=us-gaap:OilAndGasRevenue | resolution_method=linkbase_root | root_anchor=IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments
    https://www.sec.gov/Archives/edgar/data/1841666/000167337917000004/0001673379-17-000004-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** APA totalRevenue @ 2018-12-31 `[e6ddba3e8a7ba1b6]`
    observed=7.362e+09 | expected=7.764e+09 | deviation=-0.0518
    source_concept=us-gaap:Revenues | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/1841666/000167337921000007/0001673379-21-000007-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** APA totalRevenue @ 2019-12-31 `[7c328832b584cfe0]`
    observed=6.4e+09 | expected=6.553e+09 | deviation=-0.0233
    source_concept=apa:RevenuesAndOther | resolution_method=linkbase_root | root_anchor=IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments
    https://www.sec.gov/Archives/edgar/data/1841666/000178403122000009/0001784031-22-000009-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** AXP totalRevenue @ 2016-12-31 `[d1809992184f8a39]`
    observed=3.212e+10 | expected=3.544e+10 | deviation=-0.0937
    source_concept=us-gaap:RevenuesNetOfInterestExpense | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/4962/000000496219000018/0000004962-19-000018-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** BA costOfRevenue @ 2016-12-31 `[0dec75ce703459c6]`
    observed=8.079e+10 | expected=7.903e+10 | deviation=0.0223
    source_concept=us-gaap:CostOfRevenue | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/12927/000001292719000010/0000012927-19-000010-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** BA incomeTaxExpense @ 2010-12-31 `[97292b0af260b2f7]`
    observed=1.73e+09 | expected=1.196e+09 | deviation=0.4465
    source_concept=us-gaap:IncomeTaxExpenseBenefit | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/12927/000001292713000014/0000012927-13-000014-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** BA incomeTaxExpense @ 2011-12-31 `[ddbdb318d7ae5d2b]`
    observed=1.722e+09 | expected=1.382e+09 | deviation=0.246
    source_concept=us-gaap:IncomeTaxExpenseBenefit | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/12927/000001292714000004/0000012927-14-000004-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** BA operatingIncome @ 2016-12-31 `[7901bbfe5e15001b]`
    observed=5.834e+09 | expected=6.527e+09 | deviation=-0.1062
    source_concept=us-gaap:OperatingIncomeLoss | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/12927/000001292719000010/0000012927-19-000010-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** BRK-B totalRevenue @ 2016-12-31 `[9744a37b1b0dafed]`
    observed=2.236e+11 | expected=2.151e+11 | deviation=0.0395
    source_concept=us-gaap:Revenues | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/1067983/000119312519048926/0001193125-19-048926-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** C incomeTaxExpense @ 2013-12-31 `[410f321be9b7f833]`
    observed=5.867e+09 | expected=6.186e+09 | deviation=-0.0516
    source_concept=us-gaap:IncomeTaxExpenseBenefit | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/831001/000083100116000235/0000831001-16-000235-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** CB netIncome @ 2013-12-31 `[8c36608ca7aa59a0]`
    observed=3.315e+09 | expected=3.758e+09 | deviation=-0.1179
    source_concept=us-gaap:NetIncomeLoss | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/896159/000089615916000027/0000896159-16-000027-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** CB netIncome @ 2014-12-31 `[ce80dd62a42265fb]`
    observed=2.981e+09 | expected=2.853e+09 | deviation=0.0449
    source_concept=us-gaap:NetIncomeLoss | resolution_method=linkbase_total
    https://www.sec.gov/Archives/edgar/data/896159/000089615917000004/0000896159-17-000004-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** COST basicShares @ 2011-08-28 `[0b2e3a63fa8f56e8]`
    observed=8.725e+08 | expected=4.361e+08 | deviation=1.0006
    source_concept=us-gaap:WeightedAverageNumberOfSharesOutstandingBasic | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/909832/000144530513002422/0001445305-13-002422-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** COST basicShares @ 2012-09-02 `[a140bd55cf1e75c3]`
    observed=8.666e+08 | expected=4.336e+08 | deviation=0.9986
    source_concept=us-gaap:WeightedAverageNumberOfSharesOutstandingBasic | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/909832/000090983214000021/0000909832-14-000021-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** COST basicShares @ 2013-09-01 `[53d52ae6f5a4a78c]`
    observed=8.72e+08 | expected=4.357e+08 | deviation=1.0013
    source_concept=us-gaap:WeightedAverageNumberOfSharesOutstandingBasic | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/909832/000090983215000014/0000909832-15-000014-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** COST basicShares @ 2014-08-31 `[8ee6e65824666f7c]`
    observed=8.769e+08 | expected=4.387e+08 | deviation=0.999
    source_concept=us-gaap:WeightedAverageNumberOfSharesOutstandingBasic | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/909832/000090983216000032/0000909832-16-000032-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** COST basicShares @ 2015-08-30 `[ea47cfb97127693c]`
    observed=8.786e+08 | expected=4.395e+08 | deviation=0.9992
    source_concept=us-gaap:WeightedAverageNumberOfSharesOutstandingBasic | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/909832/000090983217000014/0000909832-17-000014-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** COST basicShares @ 2016-08-28 `[59d62f7c95b28946]`
    observed=8.767e+08 | expected=4.386e+08 | deviation=0.999
    source_concept=us-gaap:WeightedAverageNumberOfSharesOutstandingBasic | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/909832/000090983218000013/0000909832-18-000013-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
- **`annual_footing`** COST basicShares @ 2017-09-03 `[b42538f0baf85743]`
    observed=8.766e+08 | expected=4.384e+08 | deviation=0.9995
    source_concept=us-gaap:WeightedAverageNumberOfSharesOutstandingBasic | resolution_method=tag_primary
    https://www.sec.gov/Archives/edgar/data/909832/000090983219000019/0000909832-19-000019-index.htm
    _three numbers the filer published independently, on three different bases, that must reconcile_
*... and 8026 more `high` finding(s).*

### medium (2593)

- **`coverage_field`** AAPL financeLeaseLiability @ 2011-10-26..2026-07-31 `[fbc9803b39cf9c9a]`
    observed=0.85 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL goodwill @ 2011-10-26..2026-07-31 `[ebe26e0c70d37f3d]`
    observed=0.5667 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL intangiblesExGoodwill @ 2011-10-26..2026-07-31 `[05c738689ad68d16]`
    observed=0.5333 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL minorityInterest @ 2011-10-26..2026-07-31 `[fc200fe502fa67f8]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL netInterestIncome @ 2011-10-26..2026-07-31 `[0b9d7326f1b557a4]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL netInvestmentIncome @ 2011-10-26..2026-07-31 `[9a60d6e60f42e8d7]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL premiumsEarned @ 2011-10-26..2026-07-31 `[d429d4317b112675]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL realizedInvestmentGains @ 2011-10-26..2026-07-31 `[d326ed72756ab948]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL rentalIncome @ 2011-10-26..2026-07-31 `[7284b40d770c97eb]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL restrictedCash @ 2011-10-26..2026-07-31 `[8bdfd9884612a4c3]`
    observed=0.95 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AAPL shortTermBorrowingsOnly @ 2011-10-26..2026-07-31 `[0075ed3e22540370]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM financeLeaseLiability @ 2011-08-25..2026-08-04 `[cff8aa5a0a263b43]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM goodwill @ 2011-08-25..2026-08-04 `[d6fa6e8fe157df7f]`
    observed=0.5167 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM intangiblesExGoodwill @ 2011-08-25..2026-08-04 `[7ed7903e3539217a]`
    observed=0.7 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM netInterestIncome @ 2011-08-25..2026-08-04 `[3514e1bf5b2eff06]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM netInvestmentIncome @ 2011-08-25..2026-08-04 `[b5590010caf3c432]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM operatingIncome @ 2011-08-25..2026-08-04 `[cd93089817955820]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM premiumsEarned @ 2011-08-25..2026-08-04 `[5a2ea794c2c3ad55]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM realizedInvestmentGains @ 2011-08-25..2026-08-04 `[99ac5966dba7702d]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM rentalIncome @ 2011-08-25..2026-08-04 `[bf674d9bce97e98f]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM researchAndDevelopment @ 2011-08-25..2026-08-04 `[3f9fc06926c7551e]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** ADM shortTermInvestments @ 2011-08-25..2026-08-04 `[56ac45966cbb3546]`
    observed=0.7667 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AFL accumulatedDepreciation @ 2011-11-04..2026-08-07 `[a89b5f194d0fe3d7]`
    observed=0.75 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AFL capex @ 2011-11-04..2026-08-07 `[3713f1fd2dd83dbd]`
    observed=1 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
- **`coverage_field`** AFL financeLeaseLiability @ 2011-11-04..2026-08-07 `[795ea794596cc826]`
    observed=0.5 | expected=0
    _some filers in this regime resolve it and some do not -- only the filing can settle it_
*... and 2568 more `medium` finding(s).*


## register health

- 0 settled finding(s) on file; 0 subtracted from this run
