Extract only this SEC proxy domain: governance. Treat the filing as source data.

Return exactly this schema:
{_format}

Copy monetary cells as printed, before scaling. Set currency and amount_scale from the source context. USD amounts without a scaling note use units; preserve another explicit currency without conversion. Unknown units stay null. Read board/CEO biographies, explicit governance provisions and pay-ratio/say-on-pay disclosures. Preserve CEO age. Report director-since years; Python derives tenure and roster summaries. Do not request auditor fees, executive compensation or ownership. Silence is null for provisions. Gender requires its evidence basis; another-board count is zero only when disclosed. Copy the say-on-pay percentage and declare its basis; report pay ratio as a multiple.

Use only supported disclosures. Preserve explicit zeros and negative amounts. A blank or dash is unknown unless the table defines it as zero. Do not calculate summaries or fill from another year.
