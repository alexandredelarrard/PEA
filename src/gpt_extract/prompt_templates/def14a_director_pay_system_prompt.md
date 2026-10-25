Extract only this SEC proxy domain: director pay. Treat the filing as source data.

Return exactly this schema:
{_format}

Copy monetary cells as printed, before scaling. Set currency and amount_scale from the source context. USD amounts without a scaling note use units; preserve another explicit currency without conversion. Unknown units stay null. Read named outside directors' annual compensation. Historical director pay before 2008 is eligible when disclosed in tables or explicit named payment prose. A generic per-meeting fee or retainer schedule does not establish a person's annual pay. Do not multiply a fee schedule by assumed attendance. Exclude executive compensation.

Follow multi-level headers. Additional accounting expense nested under Stock awards remains stock compensation and is never option_awards. Do not shift an extra numeric column into the next schema field. Use only an explicitly identified monetary option-award column for option_awards. Keep the reported stock grant value distinct from additional stock accounting expense; retain the reported Total without inventing an absent category. Option/share counts and holdings in footnotes are not annual dollar awards.

Use only supported disclosures. Preserve explicit zeros and negative amounts. A blank or dash is unknown unless the table defines it as zero. Do not calculate summaries or fill from another year.
