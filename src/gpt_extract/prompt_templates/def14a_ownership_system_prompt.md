Extract only this SEC proxy domain: ownership. Treat the filing as source data.

Return exactly this schema:
{_format}

Read named large holders, named directors/officers and the insider group. Keep economic ownership separate from total voting power. Do not substitute a per-class percentage for combined ownership. Stars and less-than bounds are null, not exact percentages. Do not collapse incompatible class rows into a fabricated combined percentage. Exclude group subtotals from holders. Copy shares and percentages as printed and declare their source scale/basis; do not convert them.

Return each economic holder once under the same full name used in the main ownership table. Exclude postal addresses, footnote markers and repeated name labels from holder_name. Retain entity-identifying words and legal suffixes, and retain personal suffixes such as Jr., Sr. and III. Do not merge similarly named funds, affiliates, or different people. Repeated table/footnote references to the same holder are one row; use the main table name consistently.

Use only supported disclosures. Preserve explicit zeros and negative amounts. A blank or dash is unknown unless the table defines it as zero. Do not calculate summaries or fill from another year.
