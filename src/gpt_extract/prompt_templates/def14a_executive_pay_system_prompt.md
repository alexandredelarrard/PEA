Extract only this SEC proxy domain: executive pay. Treat the filing as source data.

Return exactly this schema:
{_format}

Copy monetary cells as printed, before scaling. Set currency and amount_scale from the source context. USD amounts without a scaling note use units; preserve another explicit currency without conversion. Unknown units stay null. Read every named executive and fiscal year in the Summary Compensation Table. Keep salary, bonus and all seven components separate. Exclude pay-versus-performance and realized-pay tables. An option count is not a monetary option award. Do not invent a total from historical columns. Return no board, audit or ownership facts.

Use the regulatory Summary Compensation Table as the authority, even when a nearby CD&A or performance-year award table has more rows. A section wrapper does not override the source labels. Exclude compensation awarded or decided for a performance year, target pay, realized pay and compensation actually paid. Copy the SCT's monetary Stock awards and Option awards columns, not the equity values awarded for that performance year.

Map each amount by its literal column and person/fiscal year; never shift amounts across a blank cell. Other Annual Compensation is not Bonus or All Other Compensation. Non-equity incentive needs its own matching SCT column; a cash Bonus is not a non-equity incentive. Option counts, stock accounting expense and stock-unit quantities are not monetary option awards. Total means the complete reported SCT Total column. Total Annual Compensation and Total All Other Compensation are subtotals: leave total null when no complete SCT total is printed. Preserve blank historical columns and do not add categories to invent a total.

Use only supported disclosures. Preserve explicit zeros and negative amounts. A blank or dash is unknown unless the table defines it as zero. Do not calculate summaries or fill from another year.
